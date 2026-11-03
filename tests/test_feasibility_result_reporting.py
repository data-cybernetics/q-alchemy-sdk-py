from copy import deepcopy
from unittest.mock import patch

import pytest

from q_alchemy import (
    ComputationResult, FeasibilityParams, FeasibilityReport, FeasibilityRequest,
    FeasibilityService, QuantumExperiment, ResultScope, ResultScopeUnavailableError,
    SolutionCriteria, State,
)
from q_alchemy.feasibility import FeasibilityJob
from tests.test_feasibility import EXPERIMENT_REPORT, REPORT, FakeJob, Link, Step


def all_report(*, kind="noisy-simulator", missing=False):
    payload = deepcopy(REPORT)
    measured = deepcopy(EXPERIMENT_REPORT)
    execution = measured["report"]["execution"]
    execution["source_kind"] = kind
    execution["observations"] = None if missing else [
        {"label": "energy", "value": -.67, "stderr": .02, "shots": 1024,
         "metadata": {"uncertainty_source": "provider"}}
    ]
    execution["basis_distributions"] = [
        {"label": "q0", "qubits": [0], "probabilities": {"0": .2, "1": .8}, "shots": 1024}
    ]
    payload.update(result_scope="all", computation_results=[{
        "result_id": "result-2", "evidence_request": "noisy-backend-simulation",
        "mode": measured["report"]["mode"], "source": execution["source"],
        "source_kind": kind, "completed": True,
        "selected_by_recommendation": False, "is_primary": False,
        "experiment_report": measured,
    }])
    return payload


@pytest.mark.parametrize("kind", ["ideal-simulator", "noisy-simulator", "qpu", "preloaded"])
@pytest.mark.parametrize("missing", [False, True])
def test_existing_quantum_io_models_expose_collection_measurements(kind, missing):
    payload = all_report(kind=kind, missing=missing)
    report = FeasibilityReport.from_json(FeasibilityReport.from_dict(payload).to_json())
    assert report.to_dict() == payload
    assert report.result_scope is ResultScope.ALL
    item, = report.require_computation_results()
    assert isinstance(item, ComputationResult)
    assert item.result_id == "result-2" and not item.is_primary
    execution = item.experiment_report.execution
    assert execution.source_kind == kind
    if missing:
        assert execution.observations is None
    else:
        observed = execution.observations.by_label()["energy"]
        assert (observed.value, observed.stderr, observed.shots) == (-.67, .02, 1024)
        assert observed.metadata == {"uncertainty_source": "provider"}
    assert execution.basis_distributions[0].probabilities == {"0": .2, "1": .8}
    assert report.computation_result == REPORT["computation_result"]


def test_unknown_selected_and_empty_all_are_distinct():
    legacy = FeasibilityReport.from_dict(REPORT)
    assert legacy.result_scope is None
    assert legacy.computation_results is None
    with pytest.raises(ResultScopeUnavailableError) as caught:
        legacy.require_computation_results()
    assert caught.value.report is legacy
    selected = FeasibilityReport.from_dict({**REPORT, "result_scope": "selected"})
    assert selected.result_scope is ResultScope.SELECTED
    assert selected.computation_results is None
    empty = FeasibilityReport.from_dict({**REPORT, "result_scope": "all", "computation_results": []})
    assert empty.require_computation_results() == ()


@pytest.mark.parametrize("collection", [None, {}, ""])
def test_malformed_acknowledgement_is_not_accepted_as_empty(collection):
    with pytest.raises(ValueError, match="array"):
        FeasibilityReport.from_dict({**REPORT, "result_scope": "all", "computation_results": collection})


def test_duplicate_report_ids_rejected():
    payload = all_report()
    payload["computation_results"] *= 2
    with pytest.raises(ValueError, match="unique"):
        FeasibilityReport.from_dict(payload).require_computation_results()


@pytest.mark.parametrize("scope", ["all", "selected"])
def test_scope_roundtrip_and_legacy_default(scope):
    request = FeasibilityRequest(SolutionCriteria.common(max_observable_rmse=.1), result_scope=scope)
    assert FeasibilityRequest.from_json(request.to_json()) == request
    legacy = request.to_dict()
    legacy.pop("result_scope")
    assert FeasibilityRequest.from_dict(legacy).result_scope is ResultScope.SELECTED


@pytest.mark.parametrize("scope", [None, True, 1, "", "everything"])
def test_scope_validation(scope):
    with pytest.raises(ValueError):
        FeasibilityRequest(SolutionCriteria.common(max_observable_rmse=.1), result_scope=scope)


def test_all_scope_is_uploaded_and_checked_by_job_without_rerunning():
    service = FeasibilityService(FeasibilityParams(remove_data=False), client=object())
    uploads = {}
    service._upload_json = lambda name, payload, **kwargs: (uploads.update({name: payload}) or Link(name))
    service._step = lambda name: Step()
    request = FeasibilityRequest(SolutionCriteria.common(max_observable_rmse=.1), result_scope="all")
    with patch("q_alchemy.feasibility.Job", FakeJob), patch(
        "q_alchemy.feasibility._download_json_output", return_value=all_report()
    ) as download:
        job = service.analyze(QuantumExperiment(target=State.dense([1, 0])), request)
        result = job.result()
        assert job.result() is result
        download.assert_called_once()
    assert uploads["feasibility_request.json"]["result_scope"] == "all"
    assert len(result.require_computation_results()) == 1


def test_legacy_server_preserves_report_and_job_when_all_scope_was_requested():
    job = FeasibilityJob(FakeJob(), timeout_sec=1, remove_data=True, result_scope=ResultScope.ALL)
    with patch("q_alchemy.feasibility._download_json_output", return_value=REPORT) as download, patch(
        "q_alchemy.feasibility._delete_job_then_inputs"
    ) as cleanup:
        for _ in range(2):
            with pytest.raises(ResultScopeUnavailableError) as caught:
                job.result()
            assert caught.value.report.to_dict() == REPORT
        download.assert_called_once()
        cleanup.assert_not_called()
    assert not job.removed


@pytest.mark.parametrize("kind,label", [
    ("ideal-simulator", "Ideal simulation"),
    ("noisy-simulator", "Noisy simulation"),
    (None, "Quantum simulator execution"),
])
def test_summary_uses_source_kind_without_guessing_noise(kind, label):
    payload = deepcopy(REPORT)
    payload["evidence"] = {"records": [{
        "metric": "quantum.model.execution_completed", "value": True,
        "scope": "quantum-model", "metadata": {"source_kind": kind},
    }]}
    assert f"{label} performed: True" in FeasibilityReport.from_dict(payload).format_summary()
