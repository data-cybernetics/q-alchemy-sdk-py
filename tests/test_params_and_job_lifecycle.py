"""Regressions: services copy caller params, and QAlchemyJob reports its lifecycle.

All tests are offline: dummy clients satisfy the API-key guards and no request
is sent.
"""
import threading
import warnings

import numpy as np
import pytest

from q_alchemy import CircuitCompressionConfig, ExecutionPlan
from q_alchemy.feasibility import FeasibilityParams, FeasibilityService
from q_alchemy.initialize import OptParams, populate_opt_params, q_alchemy_as_qasm
from q_alchemy.quantum_io import QuantumIOParams, QuantumIOService
from q_alchemy.quantum_io_compression_contract import (
    CircuitCompressionConfig as QuantumIOCompressionConfig,
)
from q_alchemy.simulator import SimulatorParams, SparseSimulator


# -- OptParams ---------------------------------------------------------------- #
def test_populate_opt_params_copies_and_never_mutates_the_caller():
    original = OptParams(api_key="k", job_tags=["mine"])
    resolved = populate_opt_params(original, max_fidelity_loss=0.1)
    assert resolved is not original
    assert original.max_fidelity_loss == 0.0
    resolved.job_tags.append("other")
    assert original.job_tags == ["mine"]


def test_populate_opt_params_rejects_misspelled_options():
    with pytest.raises(TypeError, match="max_fidelity_los"):
        populate_opt_params(OptParams(api_key="k"), max_fidelity_los=0.1)
    with pytest.raises(TypeError, match="max_fidelity_los"):
        # Raised before a client is created or anything is uploaded.
        q_alchemy_as_qasm(np.array([1.0, 0.0]), api_key="k", max_fidelity_los=0.1)


def test_populate_opt_params_accepts_mappings_and_none():
    assert populate_opt_params({"api_key": "k", "max_fidelity_loss": 0.2}).max_fidelity_loss == 0.2
    assert isinstance(populate_opt_params(None), OptParams)
    with pytest.raises(TypeError):
        populate_opt_params(42)


# -- Simulator / backend ------------------------------------------------------ #
def test_simulator_copies_params_and_rejects_unknown_options():
    params = SimulatorParams(api_key="k", tier="auto")
    simulator = SparseSimulator(params, client=object(), tier="standard")
    assert simulator.params.tier == "standard"
    assert params.tier == "auto"
    with pytest.raises(TypeError, match="api_kye"):
        SparseSimulator(params, client=object(), api_kye="k")


def test_provider_backend_overrides_do_not_leak_between_backends():
    qiskit = pytest.importorskip("qiskit")  # noqa: F841
    from q_alchemy.qalchemy_backend import QAlchemyProvider

    shared = SimulatorParams(api_key="k")
    provider = QAlchemyProvider(params=shared)
    standard = provider.get_backend(tier="standard")
    default = provider.get_backend()
    assert standard.simulator.params.tier == "standard"
    assert default.simulator.params.tier == "auto"
    assert shared.tier == "auto"
    for backend in (standard, default):
        backend.simulator.close()


def test_backend_rejects_options_alongside_an_existing_simulator():
    pytest.importorskip("qiskit")
    from q_alchemy.qalchemy_backend import QAlchemyBackend

    with pytest.raises(TypeError, match="shots"):
        QAlchemyBackend(params=SparseSimulator(client=object()), shots=2048)


# -- Quantum I/O and Feasibility services --------------------------------------- #
@pytest.mark.parametrize("service_cls, params_cls", [
    (QuantumIOService, QuantumIOParams),
    (FeasibilityService, FeasibilityParams),
])
def test_services_copy_params_and_reject_unknown_options(service_cls, params_cls):
    params = params_cls(api_key="k", remove_data=True)
    service = service_cls(params, client=object(), remove_data=False)
    assert service.params.remove_data is False
    assert params.remove_data is True
    with pytest.raises(TypeError, match="from_dict"):
        service_cls(params, client=object(), from_dict=None)  # a method, not an option


# -- PennyLane ---------------------------------------------------------------- #
def test_pennylane_accepts_dict_opt_params_and_never_mutates_or_duplicates_tags():
    pytest.importorskip("pennylane")
    from q_alchemy.pennylane_integration import (
        PENNYLANE_SOURCE_TAG,
        AmplitudeEmbedding,
        QAlchemyStatePreparation,
    )

    state = np.array([1.0, 0.0, 0.0, 0.0])
    op = QAlchemyStatePreparation(state, wires=[0, 1], opt_params={"api_key": "k", "max_fidelity_loss": 0.1})
    assert op.hyperparameters["opt_params"].max_fidelity_loss == 0.1

    shared = OptParams(api_key="k", job_tags=["mine"])
    for _ in range(3):
        op = QAlchemyStatePreparation(state, wires=[0, 1], opt_params=shared)
    embedding = AmplitudeEmbedding(state, wires=[0, 1], opt_params=shared)
    nested = embedding.compute_decomposition(state, wires=[0, 1], **embedding.hyperparameters)[0]
    assert shared.job_tags == ["mine"]
    for operation in (op, embedding, nested):
        assert operation.hyperparameters["opt_params"].job_tags == ["mine", PENNYLANE_SOURCE_TAG]


# -- QAlchemyJob lifecycle ------------------------------------------------------ #
def _job(build_result):
    pytest.importorskip("qiskit")
    from q_alchemy.qalchemy_backend import QAlchemyBackend, QAlchemyJob

    backend = QAlchemyBackend(params=SparseSimulator(client=object()))
    return QAlchemyJob(backend, "job-id", build_result)


def test_job_reports_running_then_done_and_waits():
    from qiskit.providers import JobStatus

    started, release = threading.Event(), threading.Event()

    def build():
        started.set()
        release.wait(5)
        return "result"

    job = _job(build)
    assert job.status() is JobStatus.INITIALIZING
    job.submit()
    assert started.wait(5)
    assert job.status() is JobStatus.RUNNING
    from qiskit.providers.exceptions import JobTimeoutError
    with pytest.raises(JobTimeoutError):
        job.wait_for_final_state(timeout=0.05, wait=0.01)
    with pytest.raises(JobTimeoutError):
        job.result(timeout=0.01)
    release.set()
    job.wait_for_final_state(timeout=5, wait=0.01)
    assert job.status() is JobStatus.DONE
    assert job.result() == "result"


def test_job_failure_is_reported_and_never_resubmitted():
    from qiskit.providers import JobStatus

    calls = []

    def build():
        calls.append(1)
        raise PermissionError("no grant")

    job = _job(build)
    job.submit()
    for _ in range(2):
        with pytest.raises(PermissionError, match="no grant"):
            job.result(timeout=5)
    assert job.status() is JobStatus.ERROR
    assert calls == [1]


def test_job_submits_once():
    from qiskit.providers.exceptions import JobError

    job = _job(lambda: "result")
    with pytest.raises(JobError, match="not been submitted"):
        job.result()
    job.submit()
    with pytest.raises(JobError, match="already been submitted"):
        job.submit()
    assert job.result(timeout=5) == "result"


def test_backend_run_starts_the_job_immediately(monkeypatch):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.providers import JobStatus
    from q_alchemy.qalchemy_backend import QAlchemyBackend

    backend = QAlchemyBackend(params=SparseSimulator(client=object()))
    ran = threading.Event()

    def fake_run_one(circuit, run_options):
        ran.set()
        return {"shots": 1, "success": True, "status": "DONE",
                "header": {"name": circuit.name, "metadata": {}, "n_qubits": 1, "memory_slots": 1},
                "data": {"counts": {"0x0": 1}}}

    monkeypatch.setattr(backend, "_run_one", fake_run_one)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    job = backend.run(qc, shots=1)
    assert ran.wait(5)  # started without anyone calling result()
    assert job.result(timeout=5).get_counts() == {"0": 1}
    assert job.status() is JobStatus.DONE


# -- ExecutionPlan with the Feasibility compression config ---------------------- #
def test_execution_plan_normalizes_the_feasibility_compression_config():
    with pytest.warns(UserWarning, match="enables circuit compression by default"):
        plan = ExecutionPlan(circuit_compression=CircuitCompressionConfig(options={"max_causal_qubits": 4}))
    assert type(plan.circuit_compression) is QuantumIOCompressionConfig
    assert plan.circuit_compression.enabled
    assert ExecutionPlan.from_dict(plan.to_dict()) == plan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        disabled = ExecutionPlan(circuit_compression=CircuitCompressionConfig(enabled=False))
        assert not ExecutionPlan(circuit_compression=QuantumIOCompressionConfig()).circuit_compression.enabled
    assert ExecutionPlan.from_dict(disabled.to_dict()) == disabled


# -- Single-state preparation never flattens a batch ------------------------------ #
@pytest.mark.parametrize("make", [np.asarray, lambda a: __import__("scipy.sparse").sparse.coo_array(a)])
def test_single_state_api_rejects_a_batch_instead_of_flattening_it(make):
    from q_alchemy.initialize import _prepare_statevector

    batch = make(np.eye(4)[:2])  # two 2-qubit states, not one 3-qubit state
    with pytest.raises(ValueError, match=r"shape \(2, 4\).*q_alchemy_as_qasm_parallel_states"):
        _prepare_statevector(batch)
    with pytest.raises(ValueError, match=r"shape \(2, 4\)"):
        # Raised before any request is sent.
        q_alchemy_as_qasm(batch, api_key="k")


@pytest.mark.parametrize("shape", [(4,), (1, 4), (4, 1)])
def test_single_state_api_accepts_row_and_column_vectors(shape):
    from scipy import sparse
    from q_alchemy.initialize import _prepare_statevector

    vector = np.array([1.0, 0.0, 0.0, 0.0]).reshape(shape)
    assert _prepare_statevector(vector)[1] == 2
    if len(shape) == 2:
        assert _prepare_statevector(sparse.coo_array(vector))[1] == 2
