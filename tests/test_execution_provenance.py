from copy import deepcopy
import json

from q_alchemy.feasibility_contract import FeasibilityReport
from q_alchemy.quantum_io_contract import ExperimentReport
from tests.test_feasibility_result_reporting import all_report
from tests.test_feasibility import REPORT


def test_receipts_survive_sdk_roundtrip_and_selected_scope():
    payload = all_report(kind="qpu")
    receipt = {"schema_version": 1, "outcome": "completed", "submissions": [
        {"provider": "vendor", "execution_id": "remote-a", "operation": "opaque",
         "usage_metrics": [{"name": "vendor.usage", "value": 12, "unit": "credits"}],
         "extensions": {"vendor": {"future_field": [1, 2]}}},
        {"provider": "vendor", "execution_id": "remote-b"}]}
    embedded = payload["computation_results"][0]["experiment_report"]
    embedded["report"]["execution"]["metadata"] = {"execution_provenance": receipt}
    attempts = [{"evidence_request_id": "evidence-3", "evidence_request": "qpu-execution",
                 "result_id": "result-2", "execution_provenance": receipt}]
    payload["execution_trace"] = {"execution_attempts": attempts}
    report = FeasibilityReport.from_json(json.dumps(payload))
    execution = report.computation_results[0].experiment_report.execution
    assert execution.execution_provenance == receipt
    assert report.execution_attempts == tuple(attempts)
    roundtrip = ExperimentReport.from_json(report.computation_results[0].experiment_report.to_json())
    assert roundtrip.execution.execution_provenance == receipt
    assert report.to_dict() == payload
    payload["result_scope"] = "selected"
    del payload["computation_results"]
    selected = FeasibilityReport.from_dict(payload)
    assert selected.execution_attempts == tuple(attempts)
    assert selected.computation_results is None
    assert "remote-a" in selected.format_summary()


def test_absent_and_empty_receipts_are_distinct():
    payload = deepcopy(REPORT)
    payload.pop("execution_trace", None)
    assert FeasibilityReport.from_dict(payload).execution_attempts is None
    payload["execution_trace"] = {"execution_attempts": []}
    assert FeasibilityReport.from_dict(payload).execution_attempts == ()
