"""Compression settings and diagnostics remain portable in the public SDK."""
from dataclasses import replace
from copy import deepcopy

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


def test_compression_diagnostics_survive_qio_and_feasibility_reports():
    from q_alchemy.feasibility_contract import FeasibilityReport
    from tests.test_feasibility import REPORT
    from tests.test_quantum_io import _report_payload

    diagnostics = {'candidate_regions': 8, 'capacity_skips': 3, 'synthesis_timeouts': 1,
                   'synthesis_budget_skips': 2, 'total_seconds': 1.25,
                   'future_field': {'counts': [1, 2]}}
    summary = CircuitCompressionSummary(True, True, diagnostics=diagnostics,
        options={'max_causal_qubits': 4},
        effective_options={'max_causal_qubits': 4, 'collect_report': True,
                           'equivalence': 'reachable_subspace', 'basis_gates': ['u', 'cx']},
        component_versions={'q-alchemy-circuit-compressor': '0.4.0'})
    qio_payload = _report_payload()
    qio_payload['report']['circuit_compression'] = summary.to_dict()
    qio = ExperimentReport.from_json(ExperimentReport.from_dict(qio_payload).to_json())
    assert qio.circuit_compression == summary
    feasibility_payload = deepcopy(REPORT)
    feasibility_payload['circuit_compression'] = summary.to_dict()
    feasibility_payload['computation_result']['experiment_report'] = qio_payload
    feasibility = FeasibilityReport.from_json(FeasibilityReport.from_dict(feasibility_payload).to_json())
    assert feasibility.circuit_compression['diagnostics'] == diagnostics
    assert feasibility.circuit_compression['effective_options'] == summary.effective_options
    assert feasibility.computation_experiment_report.circuit_compression == summary
    assert 'Capacity skips: 3' in feasibility.format_summary()
    assert 'Synthesis timeouts: 1' in feasibility.format_summary()
    assert 'Synthesis-budget skips: 2' in qio.format_summary()


def test_old_compression_summary_remains_readable_without_new_payloads():
    summary = CircuitCompressionSummary.from_dict({'attempted': False, 'applied': False})
    for field in ('diagnostics', 'effective_options', 'component_versions'):
        assert getattr(summary, field) is None
        assert field not in summary.to_dict()
