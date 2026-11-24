"""Regressions: requests fail before upload, failed setups leave nothing behind,
clients are released, and odd inputs are handled consistently. All offline."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from q_alchemy import (
    Criterion,
    FeasibilityExecutionError,
    FeasibilityParams,
    FeasibilityPolicy,
    FeasibilityRequest,
    FeasibilityService,
    QuantumExecutionPolicy,
    QuantumExperiment,
    QuantumIOService,
    Relation,
    ResultScope,
    SolutionCriteria,
    State,
    StateEstimateSummary,
)
from q_alchemy.feasibility import FeasibilityJob
from q_alchemy.initialize import JobSetupError, _prepare_statevector, _state_row, create_job_rapidly
from q_alchemy.quantum_io import _no_job_was_created
from tests.test_feasibility import FakeJob, Link, Step
from tests.test_feasibility_result_reporting import all_report


# -- Feasibility requests are validated before anything is uploaded ------------- #
def test_criterion_and_policy_accept_strings_and_reject_unknown_values():
    criterion = Criterion("quality.observable_rmse", "<=", 0.1)
    assert criterion.relation is Relation.LE
    assert criterion.to_dict()["relation"] == "<="
    with pytest.raises(ValueError):
        Criterion("quality.observable_rmse", "~", 0.1)
    assert FeasibilityPolicy("never").quantum_execution is QuantumExecutionPolicy.NEVER
    assert FeasibilityPolicy("never").to_dict() == {"quantum_execution": "never"}
    with pytest.raises(ValueError):
        FeasibilityPolicy("sometimes")


def _offline_feasibility_service(uploads):
    service = FeasibilityService(FeasibilityParams(remove_data=True), client=object())
    service._upload_json = lambda name, payload, **kwargs: (uploads.append(name) or Link(name))
    service._step = lambda name: Step()
    return service


def test_non_finite_request_values_are_rejected_before_any_upload():
    uploads = []
    service = _offline_feasibility_service(uploads)
    request = FeasibilityRequest(SolutionCriteria.common(max_observable_rmse=.1), metadata={"x": math.nan})
    with pytest.raises(ValueError, match="finite"):
        service.analyze(QuantumExperiment(target=State.dense([1, 0])), request)
    assert uploads == []


def test_strict_json_encoding_rejects_nan():
    from q_alchemy.quantum_io import _encode_json
    with pytest.raises(ValueError, match="feasibility_request.json must contain only finite numbers"):
        _encode_json({"x": math.inf}, "feasibility_request.json")
    assert _encode_json({"b": 1, "a": 2.5}, "x.json") == b'{"a": 2.5, "b": 1}'


# -- Invalid computation results are reported as an execution error ------------- #
def test_invalid_computation_results_raise_execution_error_and_keep_the_job():
    payload = all_report()
    payload["computation_results"] *= 2  # duplicate result IDs
    job = FeasibilityJob(FakeJob(), timeout_sec=1, remove_data=True, result_scope=ResultScope.ALL)
    with patch("q_alchemy.feasibility._download_json_output", return_value=payload), patch(
        "q_alchemy.feasibility._delete_job_then_inputs"
    ) as cleanup:
        with pytest.raises(FeasibilityExecutionError, match="unique") as caught:
            job.result()
        assert isinstance(caught.value.original_exception, ValueError)
        cleanup.assert_not_called()


# -- Job creation never loses a created job -------------------------------------- #
class _CreatedLink:
    def get_url(self):
        return "https://jobs.example/api/jobs/42"


class _PineJob:
    """Mimics pinexq: create (POST) then read back (GET) via _get_by_link."""

    def __init__(self, *, post_fails=False, reads_failing=0):
        self.job_hco = None
        self.post_fails, self.reads_failing = post_fails, reads_failing

    def _get_by_link(self, link):
        if self.reads_failing:
            self.reads_failing -= 1
            raise ConnectionError("read-back failed")
        self.job_hco = object()

    def create_and_configure_rapidly(self, **kwargs):
        if self.post_fails:
            raise ConnectionError("create failed")
        self._get_by_link(_CreatedLink())
        return self


def test_create_job_rapidly_passes_through_success_and_failed_creation():
    job = _PineJob()
    assert create_job_rapidly(job, name="x") is job and job.job_hco is not None
    with pytest.raises(ConnectionError, match="create failed"):
        create_job_rapidly(_PineJob(post_fails=True), name="x")
    assert "_get_by_link" not in vars(job)  # the hook is removed again


def test_create_job_rapidly_retries_a_failed_read_back_once():
    job = _PineJob(reads_failing=1)
    assert create_job_rapidly(job, name="x") is job
    assert job.job_hco is not None


def test_create_job_rapidly_reports_a_created_job_it_cannot_read():
    job = _PineJob(reads_failing=2)
    with pytest.raises(JobSetupError) as caught:
        create_job_rapidly(job, name="x")
    assert caught.value.job_url == "https://jobs.example/api/jobs/42"
    assert not _no_job_was_created(job, caught.value)  # so its inputs are kept
    assert _no_job_was_created(job, ConnectionError("create failed"))


def test_simulator_deletes_its_upload_only_when_no_job_was_created(monkeypatch):
    from q_alchemy import simulator as module
    from q_alchemy.simulator import SimulatorParams, SparseSimulator

    deleted = []
    monkeypatch.setattr(module, "find_processing_step", lambda client, name: SimpleNamespace(self_link=lambda: "step"))
    monkeypatch.setattr(module, "_delete_unused_upload", deleted.append)
    for job_factory, expect_deleted in [
        (lambda client: _PineJob(post_fails=True), True),
        (lambda client: _PineJob(reads_failing=2), False),
    ]:
        deleted.clear()
        monkeypatch.setattr(module, "Job", job_factory)
        simulator = SparseSimulator(SimulatorParams(api_key="k", tier="standard"), client=object())
        upload = SimpleNamespace(get_url=lambda: "https://jobs.example/api/workdata/7")
        simulator._upload = lambda *args: upload
        with pytest.raises((ConnectionError, JobSetupError)):
            simulator.counts("OPENQASM 2.0;\nqreg q[1];", input_form="qasm_file")
        assert deleted == ([upload] if expect_deleted else [])


# -- Clients are released ---------------------------------------------------------- #
def test_backend_close_releases_only_a_client_it_created():
    pytest.importorskip("qiskit")
    from q_alchemy.qalchemy_backend import QAlchemyBackend
    from q_alchemy.simulator import SimulatorParams, SparseSimulator

    with QAlchemyBackend(params=SimulatorParams(api_key="k")) as backend:
        client = backend.simulator.client
    assert client.is_closed
    shared = SparseSimulator(SimulatorParams(api_key="k"))
    QAlchemyBackend(params=shared).close()
    assert not shared.client.is_closed
    shared.close()


def test_provider_does_not_open_a_client_for_an_unknown_backend_name(monkeypatch):
    pytest.importorskip("qiskit")
    from q_alchemy import qalchemy_backend as module

    built = []
    monkeypatch.setattr(module, "SparseSimulator", lambda *a, **k: built.append(1))
    assert module.QAlchemyProvider(params={"api_key": "k"}).backends(name="other") == []
    assert built == []


@pytest.mark.parametrize("service_cls", [QuantumIOService, FeasibilityService])
def test_invalid_credentials_are_rejected_before_a_client_is_opened(monkeypatch, service_cls):
    import q_alchemy.feasibility
    import q_alchemy.quantum_io

    opened = []
    for module in (q_alchemy.quantum_io, q_alchemy.feasibility):
        monkeypatch.setattr(module, "create_client", lambda params: opened.append(params))
    with pytest.raises(ValueError, match="token"):
        service_cls(api_key="k", ibm_credentials={"token": ""})
    assert opened == []


# -- Contract and input details --------------------------------------------------- #
def test_target_infidelity_survives_a_round_trip_without_target_fidelity():
    summary = StateEstimateSummary("qtucker", False, target_infidelity=0.1)
    restored = StateEstimateSummary.from_dict(summary.to_dict())
    assert restored.target_infidelity == 0.1
    assert restored == summary
    assert "target_infidelity" not in StateEstimateSummary("qtucker", False, target_fidelity=0.9).to_dict()


@pytest.mark.parametrize("dtype, expected", [
    (np.float16, np.float32),
    (np.longdouble, np.float64),
    (np.clongdouble, np.complex128),
    (np.complex64, np.complex64),
])
def test_dtype_handling_does_not_depend_on_density(dtype, expected):
    dense = np.full(16, 0.25, dtype=dtype)
    mostly_zero = np.zeros(16, dtype=dtype)
    mostly_zero[0] = 1
    for vector in (dense, mostly_zero):
        _prepare_statevector(vector)  # neither raises
        assert np.dtype(_state_row(vector).dtype) == np.dtype(expected)


@pytest.mark.parametrize("state", [np.array([1.0]), sparse.coo_array(np.array([[1.0]]))])
def test_zero_qubit_inputs_are_rejected_like_zero_qubit_states(state):
    with pytest.raises(ValueError, match="at least one qubit"):
        _prepare_statevector(state)
    with pytest.raises(ValueError, match="at least one qubit"):
        _prepare_statevector(State.dense(np.array([1.0])))
