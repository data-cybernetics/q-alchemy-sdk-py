"""Offline parity with serialized Feasibility core reports and current requests."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from q_alchemy.feasibility_contract import (
    EvidenceCollectionConfig, FeasibilityPolicy, FeasibilityReport,
    FeasibilityRequest, SolutionCriteria,
)
from q_alchemy.quantum_io_contract import ExperimentReport, PreparationPreflightSummary
from tests.test_feasibility import EXPERIMENT_REPORT


FIXTURE = json.loads((Path(__file__).parent / "data" / "feasibility_summary.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_summary_matches_serialized_core_report(case):
    report = FeasibilityReport.from_json(json.dumps(case["report"]))
    assert report.to_dict() == case["report"]
    assert report.format_summary() == case["summary"]


@pytest.mark.parametrize("verdict", [True, False, None])
def test_embedded_preflight_preserves_verdict(verdict):
    payload = deepcopy(EXPERIMENT_REPORT)
    data = payload["report"]
    data["preparation_preflight"] = {
        "preparation": data["preparation"], "simulation": None,
        "simulation_status": "unavailable", "target_to_prepared_fidelity": 0.97,
        "preparation_approximation_infidelity": 0.03, "observable_plan": None,
        "generated_at": data["generated_at"], "warnings": ["server diagnostic"],
        "claim_contradicted": verdict,
    }
    # The SDK transports the upstream verdict, even when it could not infer it
    # from the other fields. It does not implement another certification rule.
    report = FeasibilityReport.from_dict({
        "schema_version": 1, "kind": "feasibility-report",
        "computation_result": {"experiment_report": payload},
    }).computation_experiment_report
    assert report.preparation_preflight.claim_contradicted is verdict
    restored = ExperimentReport.from_dict(report.to_dict())
    assert restored.preparation_preflight.claim_contradicted is verdict
    assert restored.preparation_preflight.warnings == ("server diagnostic",)
    del data["preparation_preflight"]["claim_contradicted"]
    assert ExperimentReport.from_dict(payload).preparation_preflight.claim_contradicted is None


@pytest.mark.parametrize("value", [1, 0, "true", [], {}])
def test_preflight_rejects_nonboolean_verdict(value):
    with pytest.raises(ValueError, match="boolean or null"):
        PreparationPreflightSummary.from_dict({"claim_contradicted": value})


@pytest.mark.parametrize("verdict", [False, None, "true", 1])
def test_latest_nontrue_verdict_does_not_warn(verdict):
    case = next(case for case in FIXTURE["cases"] if case["name"] == "reference")
    payload = deepcopy(case["report"])
    records = payload["evidence"]["records"]
    previous = next(record for record in records if record["metric"] == "preparation.claim_contradicted")
    previous["generated_at"] = "2026-09-01T08:00:00Z"
    records.append({**previous, "value": verdict, "generated_at": "2026-09-01T09:00:00Z"})
    assert "WARNING:" not in FeasibilityReport.from_dict(payload).format_summary()


@pytest.mark.parametrize("options", [{"transpile": False}, {}, {
    "transpile": True, "transpile_options": {"optimization_level": 3},
    "run_options": {"seed_simulator": 7},
}])
def test_execution_options_survive_request_roundtrip(options):
    request = FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.1),
        evidence_collection=EvidenceCollectionConfig(execution_options=options),
    )
    restored = FeasibilityRequest.from_json(request.to_json())
    assert restored.evidence_collection.execution_options == options
    assert restored == request


def test_execution_defaults_and_legacy_options():
    assert EvidenceCollectionConfig.from_dict({}).execution_options == {"transpile": True}
    options = {"seed_simulator": 7}
    config = EvidenceCollectionConfig(backend_options={"run_options": options})
    assert config.backend_options == {}
    assert config.execution_options == {"transpile": True, "run_options": options}
    with pytest.raises(ValueError, match="conflicting"):
        EvidenceCollectionConfig(backend_options={"transpile": False})
    with pytest.raises(ValueError, match="require transpile=True"):
        EvidenceCollectionConfig(execution_options={"transpile_options": {"optimization_level": 1}})
    with pytest.raises(ValueError, match="boolean"):
        EvidenceCollectionConfig(execution_options={"transpile": "false"})
    with pytest.raises(ValueError, match="classical-first"):
        FeasibilityPolicy(classical_first=False)
    with pytest.raises(TypeError, match="resource_sweep_scales"):
        EvidenceCollectionConfig(resource_sweep_scales=(1.0, 0.5))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, float("nan"), float("inf"), "5"])
def test_count_controls_are_rejected_before_submission(value):
    with pytest.raises(ValueError, match="positive integer"):
        EvidenceCollectionConfig.from_dict({"shots": value})
    payload = FeasibilityRequest(criteria=SolutionCriteria.common(max_observable_rmse=0.1)).to_dict()
    payload["max_steps"] = value
    with pytest.raises(ValueError, match="positive integer"):
        FeasibilityRequest.from_dict(payload)
