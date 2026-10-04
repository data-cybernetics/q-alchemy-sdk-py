"""Offline regressions for job isolation, allocation limits and transport failures."""
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest
from qiskit import QuantumCircuit
from scipy.sparse import coo_matrix

import q_alchemy.initialize as initialize
import q_alchemy.simulator as simulator
from q_alchemy.qalchemy_backend import QAlchemyBackend
from q_alchemy.qiskit_integration import QAlchemyInitialize
from q_alchemy.quantum_io_contract import State, ExecutionPlan, BasisMeasurement
from q_alchemy.pyarrow_data import recover_sparse_coo_from_arrow


def offline_backend():
    sim = simulator.SparseSimulator(client=object(), tier="standard")
    sim.counts = Mock(side_effect=lambda circuit, **kw: SimpleNamespace(counts={"0": kw["shots"]}))
    sim.sparse_statevector = Mock(return_value=simulator.SparseStatevectorResult(
        2, 1, "hex", "little_endian", ["0x0"], [1+0j], {},
    ))
    return QAlchemyBackend(params=sim), sim


def test_jobs_snapshot_options_without_changing_defaults():
    backend, sim = offline_backend()
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    first = backend.run(qc, shots=10, seed_simulator=1)
    second = backend.run(qc, shots=20, seed_simulator=2)
    backend.set_options(shots=30)
    assert sum(first.result().get_counts().values()) == 10
    assert sum(second.result().get_counts().values()) == 20
    assert [call.kwargs["seed_simulator"] for call in sim.counts.call_args_list] == [1, 2]
    assert backend.options.shots == 30
    assert backend.options.seed_simulator is None


def test_dense_limit_rejects_before_submission_and_on_result():
    backend, sim = offline_backend()
    with pytest.raises(ValueError, match="max_dense_qubits"):
        backend.run(QuantumCircuit(2), save_statevector=True, max_dense_qubits=1)
    sim.sparse_statevector.assert_not_called()
    # A malformed service response must not bypass the local allocation guard.
    with pytest.raises(ValueError, match="max_dense_qubits"):
        backend.run(QuantumCircuit(1), save_statevector=True, max_dense_qubits=1).result()
    result = backend.run(QuantumCircuit(2), save_statevector=True, max_dense_qubits=2).result()
    np.testing.assert_array_equal(result.data(0)["statevector"], [1, 0, 0, 0])
    assert backend.options.save_statevector is False


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_dense_limit_validation(limit):
    sv = simulator.SparseStatevectorResult(1, 1, "hex", "little_endian", ["0x0"], [1], {})
    with pytest.raises(ValueError, match="max_dense_qubits"):
        sv.to_dense(max_dense_qubits=limit)


def test_default_dense_limit_prevents_large_allocation(monkeypatch):
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(simulator.np, "zeros", allocate)
    sv = simulator.SparseStatevectorResult(40, 1, "hex", "little_endian", ["0x0"], [1], {})
    with pytest.raises(ValueError, match="max_dense_qubits"):
        sv.to_dense()
    allocate.assert_not_called()


def mock_simulator_job(monkeypatch):
    job = Mock()
    job.create_and_configure_rapidly.return_value = job
    monkeypatch.setattr(simulator, "find_processing_step", Mock())
    monkeypatch.setattr(simulator, "Job", Mock(return_value=job))
    cleanup = Mock()
    monkeypatch.setattr(simulator, "delete_job_with_data", cleanup)
    return simulator.SparseSimulator(client=object(), tier="standard"), job, cleanup


def test_simulator_preserves_timeout_and_resources(monkeypatch):
    sim, job, cleanup = mock_simulator_job(monkeypatch)
    job.wait_for_state.side_effect = TimeoutError("original timeout")
    cleanup.side_effect = RuntimeError("cleanup error")
    with pytest.raises(TimeoutError, match="original timeout"):
        sim.counts("OPENQASM 2.0;")
    cleanup.assert_not_called()


def test_simulator_preserves_download_failure(monkeypatch):
    sim, job, cleanup = mock_simulator_job(monkeypatch)
    monkeypatch.setattr(sim, "_download_return", Mock(side_effect=IOError("download failed")))
    with pytest.raises(IOError, match="download failed"):
        sim.counts("OPENQASM 2.0;")
    cleanup.assert_not_called()


def test_cleanup_failure_does_not_discard_simulator_result(monkeypatch, caplog):
    sim, job, cleanup = mock_simulator_job(monkeypatch)
    cleanup.side_effect = RuntimeError("cleanup error")
    monkeypatch.setattr(sim, "_download_return", lambda *args: {"num_qubits": 1, "shots": 3, "counts": {"0": 3}})
    assert sim.counts("OPENQASM 2.0;").counts == {"0": 3}
    cleanup.assert_called_once_with(job)
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("value", [True, 1.9, -1, float("nan"), float("inf")])
def test_state_and_plan_reject_invalid_integers(value):
    with pytest.raises(ValueError):
        State.sparse(num_qubits=2, indices=[value], amplitudes=[1])
    with pytest.raises(ValueError):
        State.sparse(num_qubits=value, indices=[0], amplitudes=[1])
    with pytest.raises(ValueError):
        ExecutionPlan(shots=value)
    with pytest.raises(ValueError):
        ExecutionPlan.from_dict({"schema_version": 3, "kind": "execution-plan", "shots": value})
    with pytest.raises(ValueError):
        BasisMeasurement.from_dict({"label": "z", "qubits": [value]})
    with pytest.raises(ValueError):
        State.from_dict({"representation": "sparse", "num_qubits": 2, "indices": [value], "amplitudes": [[1, 0]]})


def test_sparse_wire_indices_keep_full_integer_precision():
    state = State.sparse(num_qubits=np.int64(70), indices=[2**69+1], amplitudes=[1])
    assert State.from_dict(state.to_dict()) == state
    assert ExecutionPlan(shots=np.int64(10)).shots == 10


def test_processing_step_cache_is_client_scoped_and_serializes_lookups(monkeypatch):
    lookup = Mock(side_effect=lambda client, **kw: SimpleNamespace(client=client))
    monkeypatch.setattr(initialize, "from_name", lookup)
    with httpx.Client(base_url="https://example.invalid") as a, httpx.Client(base_url="https://example.invalid") as b:
        first = initialize.find_processing_step(a, "step")
        assert initialize.find_processing_step(b, "step").client is b
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: initialize.find_processing_step(a, "step"), range(8)))
        assert all(step is first for step in results)
        assert lookup.call_count == 2
        a._qalchemy_step_cache.ttl = 0
        assert initialize.find_processing_step(a, "step") is not first
        assert lookup.call_count == 3


def test_parallel_order_limit_serialization_and_client_ownership(monkeypatch):
    second_finished = Event()
    lock = Lock()
    active = peak = 0
    original = initialize._serialize_statevector
    serialize = Mock(wraps=original)
    monkeypatch.setattr(initialize, "_serialize_statevector", serialize)
    payloads = []
    clients = []
    def make_client(opt):
        client = Mock()
        clients.append(client)
        return client
    monkeypatch.setattr(initialize, "create_client", make_client)
    def run(payload, n, opt, client, summary, inline):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            payloads.append(payload)
        try:
            if opt.extra_kwargs["index"] == 0:
                assert second_finished.wait(5)
            else:
                second_finished.set()
            assert base64.b64decode(inline) == payload
            restored = recover_sparse_coo_from_arrow(pq.read_table(io.BytesIO(payload))).toarray()
            np.testing.assert_array_equal(restored, [[1, 0]])
            return str(opt.extra_kwargs["index"])
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(initialize, "_run_prepared_statevector", run)
    options = [initialize.OptParams(extra_kwargs={"index": n}) for n in range(6)]
    assert initialize.q_alchemy_as_qasm_parallel([1, 0], options, max_workers=2) == list(map(str, range(6)))
    assert peak <= 2
    serialize.assert_called_once()
    assert all(payload is payloads[0] for payload in payloads)
    assert len(clients) == 6
    for client in clients:
        client.close.assert_called_once()


def test_parallel_raises_worker_errors_and_does_not_close_supplied_client(monkeypatch):
    client = Mock()
    def fail(*args):
        raise ValueError("synthesis failed")
    monkeypatch.setattr(initialize, "_run_prepared_statevector", fail)
    with pytest.raises(ValueError, match="synthesis failed"):
        initialize.q_alchemy_as_qasm_parallel([1, 0], [{}], client=client)
    client.close.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_parallel_rejects_invalid_worker_limit(limit):
    with pytest.raises(ValueError, match="max_workers"):
        initialize.q_alchemy_as_qasm_parallel([1, 0], [], max_workers=limit)


@pytest.mark.parametrize("hashing", [False, True])
def test_upload_uses_one_root_and_only_queries_when_reusable(monkeypatch, hashing):
    root = Mock()
    root.query_action.execute.return_value.total_entities = 0
    entry = Mock()
    entry.work_data_root_link.navigate.return_value = root
    enter = Mock(return_value=entry)
    monkeypatch.setattr(initialize, "enter_jma", enter)
    monkeypatch.setattr(initialize, "allow_deletion", Mock())
    initialize.upload_statevector(object(), initialize.convert_sparse_coo_to_arrow(coo_matrix([[1, 0]])), initialize.OptParams(assign_data_hash=hashing))
    enter.assert_called_once()
    assert root.query_action.execute.call_count == int(hashing)
    uploaded = root.upload_action.execute.call_args.args[0].binary
    restored = recover_sparse_coo_from_arrow(pq.read_table(io.BytesIO(uploaded))).toarray()
    np.testing.assert_array_equal(restored, [[1, 0]])


@pytest.mark.parametrize("sparse", [False, True])
def test_initialize_owns_amplitudes_without_opening_client(monkeypatch, sparse):
    source = coo_matrix([[1+0j, 0]]) if sparse else np.array([1+0j, 0])
    gate = QAlchemyInitialize(source)
    if sparse:
        source.data[:] = 0
        np.testing.assert_array_equal(gate.params[0].toarray(), [[1, 0]])
    else:
        source[:] = 0
        assert isinstance(gate.params[0], np.ndarray)
        np.testing.assert_array_equal(gate.params[0], [1, 0])
    assert gate._client is None
    # Exercise Qiskit's circuit copy path without synthesizing remotely.
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])
    copied = circuit.copy()
    assert copied.num_qubits == 1


def test_initialize_definition_preserves_amplitudes_and_phase(monkeypatch):
    import q_alchemy.qiskit_integration as integration
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(1)
    qc.ry(0.7, 0)
    qc.rz(0.2, 0)
    phase = 0.31
    target = Statevector(qc).data * np.exp(1j * phase)
    client = Mock()
    monkeypatch.setattr(integration, "create_client", lambda opt: client)
    def synthesize(amplitudes, opt, supplied_client, return_summary):
        np.testing.assert_array_equal(amplitudes, target)
        assert supplied_client is client
        return qasm2.dumps(qc), {"global_phase": phase}
    monkeypatch.setattr(integration, "q_alchemy_as_qasm", synthesize)
    gate = integration.QAlchemyInitialize(target)
    np.testing.assert_allclose(Statevector(gate.definition).data, target, atol=1e-14)
    client.close.assert_called_once()


@pytest.mark.parametrize("research", [None, "research_step"])
def test_shared_payload_follows_inline_and_upload_routes(monkeypatch, research):
    client = Mock()
    upload = Mock(side_effect=lambda *args: object())
    configure = Mock(side_effect=lambda **kw: kw)
    monkeypatch.setattr(initialize, "_upload_statevector_payload", upload)
    monkeypatch.setattr(initialize, "configure_job", configure)
    def finish(job, options, timeout):
        return {"global_phase": 0.0}, str(options.max_fidelity_loss)
    monkeypatch.setattr(initialize, "run_job", finish)
    options = [initialize.OptParams(max_fidelity_loss=loss, use_research_function=research) for loss in (0.0, 0.1)]
    results = initialize.q_alchemy_as_qasm_parallel([1, 0], options, client=client, return_summary=True)
    assert results == [("0.0", {"global_phase": 0.0}), ("0.1", {"global_phase": 0.0})]
    assert configure.call_count == 2
    if research:
        assert upload.call_count == 2
        assert upload.call_args_list[0].args[1] is upload.call_args_list[1].args[1]
        assert configure.call_args_list[0].kwargs["statevector_data"] is not configure.call_args_list[1].kwargs["statevector_data"]
    else:
        upload.assert_not_called()
        for call in configure.call_args_list:
            payload = base64.b64decode(call.kwargs["statevector_data"])
            restored = recover_sparse_coo_from_arrow(pq.read_table(io.BytesIO(payload))).toarray()
            np.testing.assert_array_equal(restored, [[1, 0]])
    client.close.assert_not_called()
