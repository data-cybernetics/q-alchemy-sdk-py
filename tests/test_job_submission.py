import logging
import re
from unittest.mock import Mock

import httpx
import pytest
from pinexq.client.core.exceptions import (
    ApiException, ClientException, TemporarilyNotAvailableException,
)
from pinexq.client.core.model.error import ProblemDetails

import q_alchemy.initialize as api
import q_alchemy.job_submission as submission
from q_alchemy import JobSubmissionError


@pytest.fixture
def clock(monkeypatch):
    """A fake clock that sleeping advances, so retries run instantly."""
    state = {"now": 1000.0, "sleeps": []}

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(submission.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(submission.time, "sleep", sleep)
    return state


def api_error(status, detail="Could not send message."):
    return ApiException("Error while executing action, unexpected response",
                        ProblemDetails(status=status, detail=detail))


def fake_step():
    step = Mock()
    step.self_link.return_value.get_url.return_value = "https://jobs.example/api/ProcessingSteps/1"
    return step


def failing(*errors, result="job"):
    """A submit callable that raises each error in turn, then returns result."""
    return Mock(side_effect=[*errors, result])


@pytest.mark.parametrize("error", [
    api_error(500),
    api_error(502),
    TemporarilyNotAvailableException("unavailable"),
    ApiException("gateway answered without problem details"),
    api_error(429),
], ids=["500", "502", "503", "no-status", "429"])
def test_transient_failures_are_retried(clock, error):
    submit = failing(error, error)
    assert submission.submit_with_retry(submit, timeout_s=600) == "job"
    assert submit.call_count == 3
    assert len(clock["sleeps"]) == 2


def client_error(cause):
    try:
        raise cause
    except httpx.HTTPError as caught:
        try:
            raise ClientException("Error from httpx while executing action") from caught
        except ClientException as exc:
            return exc


@pytest.mark.parametrize("cause", [
    httpx.ConnectError("connection refused"),
    # 2026-11-18: RabbitMQ could not create the job's queue, so the submission hung.
    httpx.ReadTimeout("timed out"),
    httpx.RemoteProtocolError("server disconnected without sending a response"),
    httpx.ReadError("connection reset"),
], ids=["refused", "read-timeout", "disconnected", "reset"])
def test_connection_failures_are_retried(clock, cause):
    submit = failing(client_error(cause))
    assert submission.submit_with_retry(submit, timeout_s=600) == "job"
    assert submit.call_count == 2


@pytest.mark.parametrize("error", [
    api_error(400, "invalid parameters"),
    api_error(404, "processing step not found"),
], ids=["400", "404"])
def test_client_errors_fail_at_once_with_a_timestamp(clock, error):
    submit = failing(error)
    with pytest.raises(JobSubmissionError) as raised:
        submission.submit_with_retry(submit, timeout_s=600)
    assert submit.call_count == 1 and clock["sleeps"] == []
    message = str(raised.value)
    assert re.search(r"failed at \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ with HTTP 4\d\d\.", message)
    assert error.problem_details.detail in message
    assert raised.value.__cause__ is error
    assert raised.value.attempts == 1


def test_other_client_exceptions_are_not_retried(clock):
    submit = failing(ClientException("Action did not respond with URL"))
    with pytest.raises(JobSubmissionError, match="ClientException: Action did not respond"):
        submission.submit_with_retry(submit, timeout_s=600)
    assert submit.call_count == 1


def test_gives_up_when_the_retry_window_ends(clock, caplog):
    submit = Mock(side_effect=api_error(500))
    with caplog.at_level(logging.WARNING, logger=submission.__name__):
        with pytest.raises(JobSubmissionError) as raised:
            submission.submit_with_retry(submit, timeout_s=300)
    error = raised.value
    assert isinstance(error, ApiException)  # existing handlers keep working
    assert error.status == 500 and error.attempts == submit.call_count > 1
    assert sum(clock["sleeps"]) == pytest.approx(300)
    assert max(clock["sleeps"]) <= 60
    message = str(error)
    assert f"after {error.attempts} attempts since" in message
    assert "submitting again later is safe" in message
    assert "Could not send message." in message
    assert error.first_failed_at <= error.failed_at
    # Each retry is logged with its time, so a long wait is visible while it happens.
    retries = [r.getMessage() for r in caplog.records if "retrying in" in r.getMessage()]
    assert len(retries) == error.attempts - 1
    assert all(re.search(r"failed at \S+Z with HTTP 500", line) for line in retries)


def test_zero_timeout_makes_a_single_attempt(clock):
    submit = Mock(side_effect=api_error(500))
    with pytest.raises(JobSubmissionError):
        submission.submit_with_retry(submit, timeout_s=0)
    assert submit.call_count == 1


def test_no_timeout_keeps_retrying(clock):
    submit = failing(*[api_error(500)] * 50)
    assert submission.submit_with_retry(submit, timeout_s=None) == "job"
    assert submit.call_count == 51


def test_other_exceptions_pass_through_unchanged(clock):
    submit = Mock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError, match="bad input"):
        submission.submit_with_retry(submit, timeout_s=600)
    assert submit.call_count == 1


def test_state_preparation_submission_retries_a_maintenance_500(clock, monkeypatch):
    """The JMA's answer while RabbitMQ was in maintenance (2026-11-18): retried, then submitted."""
    created = Mock(name="created job")
    attempts = []

    class FakeJob:
        def __init__(self, client):
            pass

        def create_and_configure_rapidly(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise api_error(500)
            return created

    step = fake_step()
    monkeypatch.setattr(api, "Job", FakeJob)
    monkeypatch.setattr(api, "find_processing_step", lambda client, name: step)

    job = api.configure_job(Mock(), api.OptParams(api_key="key"), "encoded-state")

    assert job is created
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert attempts[1]["processing_step_url"] is step.self_link()


def test_state_preparation_retry_window_follows_opt_params(clock, monkeypatch):
    class FakeJob:
        def __init__(self, client):
            pass

        def create_and_configure_rapidly(self, **kwargs):
            raise api_error(500)

    monkeypatch.setattr(api, "Job", FakeJob)
    monkeypatch.setattr(api, "find_processing_step", lambda client, name: fake_step())

    options = api.OptParams(api_key="key", job_submission_retry_timeout_sec=0)
    with pytest.raises(JobSubmissionError, match=r"with HTTP 500\."):
        api.configure_job(Mock(), options, "encoded-state")
    assert clock["sleeps"] == []
