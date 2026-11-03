"""Reports from a newer service must stay readable by this SDK version.

A value the service adds later (a new status, recommendation or scope, a new
performance span status) or a null in an optional field must not make
``result()`` fail for a job that succeeded.
"""
from copy import deepcopy

import pytest

from q_alchemy import (
    ExperimentReport,
    FeasibilityReport,
    FeasibilityRequest,
    FeasibilityStatus,
    RecommendedCompute,
    ResultScope,
    SolutionCriteria,
)
from tests.test_feasibility import EXPERIMENT_REPORT, REPORT


def _report(**changes):
    payload = deepcopy(REPORT)
    payload.update(changes)
    return payload


def test_unknown_recommendation_and_status_values_are_preserved():
    report = FeasibilityReport.from_dict(_report(
        recommendation={"compute": "hybrid", "reason": "split the workload"},
        classical={**REPORT["classical"], "status": "partially-feasible"},
    ))
    assert report.recommendation.compute.value == "hybrid"
    assert report.recommendation.compute == "hybrid"
    assert isinstance(report.recommendation.compute, RecommendedCompute)
    assert report.classical_status.value == "partially-feasible"
    assert "hybrid: split the workload" in report.format_summary()


def test_known_values_still_return_the_canonical_members():
    assert RecommendedCompute("classical") is RecommendedCompute.CLASSICAL
    assert FeasibilityStatus("feasible") is FeasibilityStatus.FEASIBLE
    report = FeasibilityReport.from_dict(_report(recommendation={"compute": None, "reason": ""}))
    assert report.recommendation.compute is RecommendedCompute.UNDETERMINED


def test_unknown_result_scope_in_a_report_reads_as_none():
    report = FeasibilityReport.from_dict(_report(result_scope="per-region"))
    assert report.result_scope is None
    assert report.raw["result_scope"] == "per-region"
    # Requests are still validated: the SDK never sends a scope it does not know.
    with pytest.raises(ValueError):
        ResultScope("per-region")
    with pytest.raises(ValueError):
        FeasibilityRequest(SolutionCriteria(()), result_scope="per-region")


def test_result_scope_all_still_requires_computation_results():
    with pytest.raises(ValueError, match="computation_results"):
        FeasibilityReport.from_dict(_report(result_scope="all"))


def test_format_summary_tolerates_nulls_in_optional_fields():
    trace = {**REPORT["execution_trace"], "execution_attempts": [
        {"evidence_request": "qpu-execution", "execution_provenance": None},
        {"evidence_request": "qpu-execution", "execution_provenance": {"submissions": None}},
        {"evidence_request": "qpu-execution", "execution_provenance": {"submissions": [
            None, {"provider": "ibm", "backend": "ibm_x", "execution_id": "job-1", "local_outcome": "done"},
        ]}},
    ]}
    report = FeasibilityReport.from_dict(_report(
        circuit_compression={"enabled": True, "accepted_regions": None},
        execution_trace=trace,
    ))
    summary = report.format_summary()
    assert "Accepted regions: 0" in summary
    assert "ibm / ibm_x: job-1 (done; qpu-execution)" in summary


def test_unreadable_performance_diagnostics_do_not_fail_the_report():
    payload = deepcopy(EXPERIMENT_REPORT)
    payload["report"]["performance"] = {
        "schema_version": 1, "trace_id": "t", "root_span_id": "s",
        "spans": [{"span_id": "s", "parent_span_id": None, "operation": "run",
                   "start_offset_seconds": 0.0, "elapsed_seconds": 1.0, "status": "cancelled"}],
    }
    with pytest.warns(UserWarning, match="performance diagnostics"):
        report = ExperimentReport.from_dict(payload)
    assert report.performance is None


def test_readable_performance_diagnostics_are_still_parsed():
    payload = deepcopy(EXPERIMENT_REPORT)
    payload["report"]["performance"] = {
        "schema_version": 1, "trace_id": "t", "root_span_id": "s",
        "spans": [{"span_id": "s", "parent_span_id": None, "operation": "run",
                   "start_offset_seconds": 0.0, "elapsed_seconds": 1.0, "status": "ok"}],
    }
    report = ExperimentReport.from_dict(payload)
    assert report.performance.spans[0].operation == "run"
