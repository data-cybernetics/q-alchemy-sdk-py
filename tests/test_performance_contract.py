"""Optional timing fields survive current/legacy public SDK contracts."""
from copy import deepcopy
import json

from q_alchemy.performance_contract import PerformanceDiagnostics, PerformanceSpan
from q_alchemy.quantum_io_contract import ExperimentReport
from q_alchemy.feasibility_contract import FeasibilityReport


def fixture():
    from tests.test_feasibility import REPORT
    return deepcopy(REPORT)

def test_public_contract_preserves_timers_in_selected_and_all_results():
    payload = fixture()
    performance = PerformanceDiagnostics("trace-1", "s1", (
        PerformanceSpan("s1", None, "assessment", 0, 10),
        PerformanceSpan("s2", "s1", "evidence.collect", 1, 8,
                        evidence_request_id="evidence-1", result_ids=("result-1",)),
    ))
    payload["performance"] = performance.to_dict()
    embedded = payload["computation_result"]["experiment_report"]
    embedded["report"]["performance"] = performance.to_dict()
    report = FeasibilityReport.from_dict(payload)
    assert report.performance == performance
    assert report.computation_experiment_report.performance == performance
    qio = ExperimentReport.from_json(report.computation_experiment_report.to_json())
    assert qio.performance == performance
    assert report.to_dict() == payload
    payload["result_scope"] = "all"
    payload["computation_results"] = [{"result_id": "result-1", "evidence_request": "classical-execution", "completed": True,
        "selected_by_recommendation": True, "is_primary": True, "experiment_report": embedded}]
    all_report = FeasibilityReport.from_dict(payload)
    assert all_report.performance == performance
    assert all_report.computation_results[0].experiment_report.performance == performance


def test_old_reports_keep_optional_timers_absent():
    payload = fixture()
    report = FeasibilityReport.from_dict(payload)
    assert report.performance is None
    qio = report.computation_experiment_report
    assert qio.performance is None
    assert "performance" not in qio.to_dict()["report"]
    assert report.to_dict() == payload
