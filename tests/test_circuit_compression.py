"""Offline client tests with mocked PineXQ transport, plus a local contract check.

For real deployed service calls, run test_circuit_compression_live_integration.py.
"""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import q_alchemy.circuit_compression as service_module
from q_alchemy import (
    Circuit, CircuitCompressionExecutionError, CircuitCompressionParams,
    CircuitCompressionReport, CircuitCompressionRequest, CircuitCompressionService,
)


QASM = 'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; x q[0];'
CIRCUIT = {"schema_version": 1, "kind": "quantum-circuit", "format": "qasm3", "qasm": QASM}
METRICS = {"operations": 1, "one_qubit_operations": 1, "two_qubit_operations": 0,
           "cx": 0, "depth": 1, "two_qubit_depth": 0, "counts": {"x": 1}}
REPORT = {
    "schema_version": 1, "kind": "circuit-compression-report", "circuit": CIRCUIT,
    "changed": False, "exact": True, "equivalence": "reachable_subspace",
    "input_semantics": "canonical_zero_state", "options": {},
    "metrics": {"input": METRICS, "baseline": METRICS, "compressed": METRICS},
    "regions": [], "report": None,
}


@pytest.fixture
def platform(monkeypatch):
    state = SimpleNamespace(uploads=[], events=[], jobs=[], payload=copy.deepcopy(REPORT),
                            upload_error_at=None, create_error=False, started_error=False,
                            wait_error=None, deletion_refused=False, input_delete_fail_once=False)

    class Input:
        is_deletable = True

        def __init__(self, name):
            self.name = name
            self.delete_action = SimpleNamespace(execute=self.delete)

        def get_url(self):
            return f"https://example.invalid/work/{self.name}"

        def navigate(self):
            return self

        def delete(self):
            if state.input_delete_fail_once:
                state.input_delete_fail_once = False
                raise RuntimeError("temporary deletion failure")
            state.events.append(self.name)

    def upload(parameters):
        if state.upload_error_at == len(state.uploads):
            raise RuntimeError("upload failed")
        state.uploads.append(parameters)
        return Input(parameters.filename)

    class Job:
        def __init__(self, client):
            self.job_hco = None
            self.wait_calls = []
            self.downloads = 0
            self.created = None
            state.jobs.append(self)

        def create_and_configure_rapidly(self, **kwargs):
            self.created = kwargs
            if state.create_error:
                raise RuntimeError("create failed")
            self.job_hco = object()
            if state.started_error:
                raise RuntimeError("start failed")

        def wait_for_state(self, *args, **kwargs):
            self.wait_calls.append(kwargs)
            if state.wait_error:
                raise state.wait_error

        def get_output_data_slots(self):
            self.downloads += 1
            binary = json.dumps(state.payload).encode()
            output = SimpleNamespace(name="circuit_compression_report.json", size_in_bytes=len(binary),
                                     download_link=SimpleNamespace(download=lambda: binary))
            return [SimpleNamespace(assigned_workdatas=[output])]

        def delete_with_associated(self, **kwargs):
            assert kwargs == dict(delete_subjobs_with_data=True, delete_input_workdata=False,
                                  delete_output_workdata=True)
            if not state.deletion_refused:
                state.events.append("job")
                self.job_hco = None

    root = SimpleNamespace(upload_action=SimpleNamespace(execute=upload))
    state.client = Mock()
    state.discovery = Mock(return_value=SimpleNamespace(self_link=lambda: "https://example.invalid/step"))
    state.pinned = Mock(return_value=SimpleNamespace(self_link=lambda: "https://example.invalid/pinned"))
    monkeypatch.setattr(service_module, "Job", Job)
    monkeypatch.setattr(service_module, "enter_jma", lambda client: SimpleNamespace(work_data_root_link=SimpleNamespace(navigate=lambda: root)))
    monkeypatch.setattr(service_module, "find_processing_step", state.discovery)
    monkeypatch.setattr(service_module, "from_name", state.pinned)
    return state


def test_default_wire_slots_download_cache_and_cleanup(platform):
    with CircuitCompressionService(client=platform.client) as service:
        job = service.compress(Circuit.qasm3(QASM))
        raw = job.raw_job
        report = job.result(timeout=7)
        assert job.result() is report
    platform.client.close.assert_not_called()
    platform.discovery.assert_called_once_with(platform.client, "compress_circuit")
    assert [u.filename for u in platform.uploads] == ["quantum_circuit.json", "compression_request.json"]
    assert json.loads(platform.uploads[0].binary) == CIRCUIT
    assert json.loads(platform.uploads[1].binary) == CircuitCompressionRequest().to_dict()
    assert [(s.index, s.work_data_urls) for s in raw.created["input_data_slots"]] == [
        (0, ["https://example.invalid/work/quantum_circuit.json"]),
        (1, ["https://example.invalid/work/compression_request.json"]),
    ]
    assert raw.wait_calls == [{"polling_interval_s": 0.25, "timeout_s": 7.0}]
    assert raw.downloads == 1
    assert report.input_semantics == "canonical_zero_state"
    assert report.circuit.payload == QASM
    assert platform.events == ["job", "quantum_circuit.json", "compression_request.json"]
    assert job.removed
    with pytest.raises(RuntimeError, match="removed"):
        _ = job.raw_job


def test_operator_options_version_and_preservation(platform):
    params = CircuitCompressionParams(step_version="0.1.1", remove_data=False, job_tags=["test"])
    with CircuitCompressionService(params, client=platform.client) as service:
        job = service.compress(CIRCUIT, CircuitCompressionRequest(options={
            "equivalence": "operator", "collect_report": True, "max_causal_candidates": 1000}))
        job.result()
    assert json.loads(platform.uploads[1].binary)["options"] == {
        "equivalence": "operator", "collect_report": True, "max_causal_candidates": 1000}
    platform.pinned.assert_called_once_with(platform.client, "compress_circuit", version="0.1.1")
    platform.discovery.assert_not_called()
    assert "test" in job.raw_job.created["tags"]
    assert not job.removed and platform.events == []


@pytest.mark.parametrize("error", [TimeoutError("still running"), RuntimeError("trace [/procon/error] bad options")])
def test_failed_wait_preserves_job_then_retries(platform, error):
    job = CircuitCompressionService(client=platform.client).compress(CIRCUIT)
    platform.wait_error = error
    with pytest.raises(CircuitCompressionExecutionError) as captured:
        job.result()
    assert captured.value.original_exception is error
    assert "[/procon/error]" not in str(captured.value)
    assert not job.removed and job.raw_job is platform.jobs[0]
    assert platform.events == []
    platform.wait_error = None
    assert job.result().exact
    assert len(platform.jobs) == 1


def test_invalid_output_preserves_job_for_retry(platform):
    job = CircuitCompressionService(client=platform.client).compress(CIRCUIT)
    platform.payload = {"kind": "wrong-report"}
    with pytest.raises(CircuitCompressionExecutionError):
        job.result()
    assert platform.events == []
    platform.payload = REPORT
    assert job.result().exact


@pytest.mark.parametrize("partial", [False, True])
def test_cleanup_retry_uses_cached_result(platform, partial):
    platform.deletion_refused = not partial
    platform.input_delete_fail_once = partial
    job = CircuitCompressionService(client=platform.client).compress(CIRCUIT)
    first = job.result()
    assert not job.removed
    platform.deletion_refused = False
    assert job.result() is first
    assert job.removed
    assert platform.jobs[0].downloads == 1
    assert platform.events.count("job") == 1


@pytest.mark.parametrize("failure,expected", [
    ("upload", ["quantum_circuit.json"]),
    ("create", ["quantum_circuit.json", "compression_request.json"]),
    ("start", []),
])
def test_submission_failure_cleanup_respects_job_ownership(platform, failure, expected):
    platform.upload_error_at = 1 if failure == "upload" else None
    platform.create_error = failure == "create"
    platform.started_error = failure == "start"
    with pytest.raises(RuntimeError):
        CircuitCompressionService(client=platform.client).compress(CIRCUIT)
    assert platform.events == expected


@pytest.mark.parametrize("bad", [None, "QASM", {}, {**CIRCUIT, "schema_version": True},
                                {**CIRCUIT, "schema_version": 1.0}, {**CIRCUIT, "qasm": ""}])
def test_invalid_input_causes_no_network_calls(platform, bad):
    with pytest.raises((ValueError, TypeError)):
        CircuitCompressionService(client=platform.client).compress(bad)
    platform.discovery.assert_not_called()
    assert platform.uploads == []


def test_nonfinite_options_rejected_before_upload(platform):
    with pytest.raises(ValueError, match="non-finite"):
        CircuitCompressionService(client=platform.client).compress(
            CIRCUIT, CircuitCompressionRequest(options={"synthesis_budget_seconds": float("nan")}))
    assert platform.uploads == []


def test_contract_roundtrips_and_report_summary():
    request = CircuitCompressionRequest(options={"basis_gates": ("u", "cx")})
    assert CircuitCompressionRequest.from_json(request.to_json()) == request
    report = CircuitCompressionReport.from_json(json.dumps(REPORT))
    assert report.to_dict() == REPORT
    assert CircuitCompressionReport.from_json(report.to_json()) == report
    assert "Metrics: input -> baseline -> compressed" in report.format_summary()
    assert "1Q operations: 1 -> 1 -> 1" in report.format_summary()
    assert report.regions == () and report.report is None
    assert report.changed is False and report.exact is True


def test_native_qiskit_phase_survives_upload_and_output(platform):
    qiskit = pytest.importorskip("qiskit")
    import numpy as np
    from qiskit.quantum_info import Operator
    circuit = qiskit.QuantumCircuit(1, global_phase=0.37)
    circuit.h(0)
    original = circuit.copy()
    job = CircuitCompressionService(client=platform.client).compress(circuit)
    encoded = json.loads(platform.uploads[0].binary)
    platform.payload["circuit"] = encoded
    restored = job.result().to_qiskit()
    assert circuit == original
    np.testing.assert_allclose(Operator(restored.to_gate().control()).data,
                               Operator(original.to_gate().control()).data, atol=1e-12, rtol=0)


def test_params_and_client_ownership(monkeypatch, platform):
    params = CircuitCompressionParams(api_key="test-only-key")
    monkeypatch.setattr(service_module, "create_client", lambda params: platform.client)
    with CircuitCompressionService(params, step_version="0.1.1") as service:
        assert service.params.step_version == "0.1.1"
    service.close()
    platform.client.close.assert_called_once()
    assert params.step_version is None
    assert "test-only-key" not in repr(params)
    with pytest.raises(TypeError, match="Unknown"):
        CircuitCompressionService(client=platform.client, typo=True)


def test_portable_contracts_need_no_qiskit_or_private_engine():
    script = '''
import importlib.abc
import json
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "qiskit" or fullname.startswith(("qiskit.", "q_alchemy.circuit_compressor", "q_alchemy.circuit.library")):
            raise ImportError("optional engine blocked by test")
sys.meta_path.insert(0, BlockOptional())
from q_alchemy import Circuit, CircuitCompressionReport, CircuitCompressionRequest, CircuitCompressionService
report = CircuitCompressionReport.from_dict(json.loads(sys.stdin.read()))
assert report.circuit.payload.startswith("OPENQASM")
assert CircuitCompressionRequest().to_dict()["options"] == {}
'''
    subprocess.run([sys.executable, "-c", script], input=json.dumps(REPORT), text=True, check=True)
