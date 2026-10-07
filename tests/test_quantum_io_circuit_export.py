import builtins
import pytest
from q_alchemy import ExperimentReport
from tests.test_quantum_io import _report_payload


def test_old_report_has_no_circuit_and_needs_no_qiskit(monkeypatch):
    original_import = builtins.__import__
    def reject_qiskit(name, *args, **kwargs):
        if name == "qiskit" or name.startswith("qiskit."):
            raise AssertionError("report parsing imported Qiskit")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_qiskit)
    payload = ExperimentReport.from_dict(_report_payload()).to_dict()
    report = ExperimentReport.from_dict(payload)
    assert report.quantum_circuit is None
    payload["report"]["quantum_circuit"] = _circuit_payload()
    assert ExperimentReport.from_dict(payload).to_dict() == payload


def _circuit_payload():
    return {"schema_version": 1, "kind": "quantum-circuit", "role": "logical-experiment-circuit",
            "format": "qasm3", "qasm": 'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; h q[0];'}


@pytest.mark.parametrize("key,value", [("schema_version", True), ("schema_version", 1.0),
    ("schema_version", 2), ("format", "pickle"), ("role", "submitted-isa-circuit"),
    ("kind", "unknown"), ("qasm", ""), ("qasm", None)])
def test_bad_circuit_envelope_rejected_during_report_parsing(key, value):
    payload = ExperimentReport.from_dict(_report_payload()).to_dict()
    circuit = _circuit_payload()
    circuit[key] = value
    payload["report"]["quantum_circuit"] = circuit
    with pytest.raises(ValueError):
        ExperimentReport.from_dict(payload)


def test_fidelity_fields_survive_report_roundtrip():
    payload = ExperimentReport.from_dict(_report_payload()).to_dict()
    payload["report"].update(reference_to_estimated_surrogate_fidelity=.9,
                             reference_to_estimated_surrogate_infidelity=.1)
    assert ExperimentReport.from_dict(payload).to_dict() == payload
