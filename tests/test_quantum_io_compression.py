"""Compression settings and diagnostics remain portable in the public SDK."""
from dataclasses import replace

from q_alchemy.quantum_io_contract import CircuitCompressionConfig, ExecutionPlan, ExperimentReport
from q_alchemy.quantum_io_compression_contract import CircuitCompressionMetrics, CircuitCompressionSummary
from q_alchemy.feasibility_contract import CircuitCompressionConfig as FeasibilityCompressionConfig


def test_qio_and_feasibility_keep_distinct_defaults_and_share_option_contract():
    assert not ExecutionPlan().circuit_compression.enabled
    assert FeasibilityCompressionConfig().enabled
    plan = ExecutionPlan(circuit_compression=FeasibilityCompressionConfig(options={"max_causal_qubits": 4}))
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.circuit_compression.to_dict() == plan.circuit_compression.to_dict()


def test_compression_diagnostics_round_trip():
    cost = CircuitCompressionMetrics(3, 1, 2, 2, 3, 2, {"h": 1, "cx": 2})
    summary = CircuitCompressionSummary(True, True, changed=True, exact=True,
        equivalence="reachable_subspace", input_semantics="canonical_zero_state",
        input_metrics=cost, compressed_metrics=replace(cost, two_qubit_operations=0), accepted_regions=1)
    assert CircuitCompressionSummary.from_dict(summary.to_dict()) == summary
    assert "2Q operations: 2 -> 0" in summary.format_summary()
    # Reuse the SDK's existing report fixture, preserving all unrelated fields.
    from tests.test_quantum_io import _report_payload
    payload = _report_payload()
    payload["report"]["circuit_compression"] = summary.to_dict()
    report = ExperimentReport.from_dict(payload)
    assert ExperimentReport.from_json(report.to_json()).circuit_compression == summary
    assert "CIRCUIT COMPRESSION" in report.format_summary()


def test_sdk_uploads_compression_options_without_algorithmic_policy():
    from unittest.mock import patch
    from tests.test_quantum_io import TestServiceSubmission, _FakePineJob, _bell_experiment
    case = TestServiceSubmission()
    case.setUp()
    plan = ExecutionPlan(circuit_compression=CircuitCompressionConfig(
        enabled=True, options={"max_causal_qubits": 4}))
    with patch("q_alchemy.quantum_io.Job", _FakePineJob):
        case.service.run(_bell_experiment(), plan)
    payload = next(value for filename, value in case.uploads if filename == "execution_plan.json")
    assert payload["circuit_compression"] == {"enabled": True, "options": {"max_causal_qubits": 4}}
