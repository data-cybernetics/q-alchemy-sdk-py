"""Retry job submission through short platform outages, and say when it failed.

PineXQ's nightly node maintenance restarts RabbitMQ and the job-management
service. While that happens, ``RapidJobSetup`` can answer HTTP 500 (the job
offer could not be published), hang while RabbitMQ cannot create the job's
queue, or the gateway can refuse connections. On 2026-11-18 this lasted about
13 minutes. Without a retry, the first such failure ends a long-running program.

Retrying is safe: a job whose submission failed this way is either in the error
state or was never offered to a worker, so a retry does not run it twice.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, UTC
from typing import Callable, TypeVar

import httpx
from pinexq.client.core.exceptions import ApiException, ClientException

LOG = logging.getLogger(__name__)

T = TypeVar("T")

_FIRST_DELAY_S = 2.0
_MAX_DELAY_S = 60.0


class JobSubmissionError(ApiException):
    """Submitting a Q-Alchemy job failed, after any retries.

    Subclasses ``ApiException`` so existing handlers still catch it. The message
    states the UTC time of the failure, the HTTP status and the number of
    attempts; the original exception is chained as ``__cause__``.
    """

    def __init__(self, message: str, *, cause: Exception, failed_at: datetime,
                 first_failed_at: datetime, attempts: int):
        super().__init__(message, getattr(cause, "problem_details", None))
        self.status = failure_status(cause)
        self.failed_at = failed_at
        self.first_failed_at = first_failed_at
        self.attempts = attempts


def failure_status(exc: Exception) -> int | None:
    """The HTTP status behind a pinexq-client exception, when it is known."""
    if isinstance(exc, ApiException):
        details = exc.problem_details
        if details is not None and details.status is not None:
            return details.status
        return exc.status  # 429 and 503 have their own classes; otherwise None
    return None


def is_transient(exc: Exception) -> bool:
    """Whether a failed submission is worth repeating.

    Retried: HTTP 5xx and 429, timeouts, and connections that failed or were
    dropped. An error without problem details has no known status; job
    management answers errors with problem details, so such a response comes
    from the gateway in front of it and is retried too. Not retried: other 4xx
    answers, which a repeat would only get again.

    A timed-out submission may have created a job, so a retry can leave an extra
    one. In practice a submission hangs when RabbitMQ cannot create the job's
    queue, and such a job is never offered to a worker: it does not run twice.
    """
    if isinstance(exc, ApiException):
        status = failure_status(exc)
        return status is None or status == 429 or status >= 500
    if isinstance(exc, ClientException):
        return isinstance(exc.__cause__, (httpx.TimeoutException, httpx.NetworkError,
                                          httpx.RemoteProtocolError))
    return False


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(exc: Exception) -> str:
    status = failure_status(exc)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, ApiException):
        return "an error response without a status"
    return f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"


def submit_with_retry(submit: Callable[[], T], *, timeout_s: float | None,
                      what: str = "Q-Alchemy job submission") -> T:
    """Call ``submit`` until it succeeds, a failure is not transient, or time runs out.

    ``timeout_s`` bounds the time spent retrying: ``0`` makes a single attempt
    and ``None`` retries without limit. Delays grow from 2 s to 60 s, with jitter.
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    first_failed_at: datetime | None = None
    attempt = 0
    while True:
        attempt += 1
        try:
            result = submit()
        except (ApiException, ClientException) as exc:
            failed_at = datetime.now(UTC)
            first_failed_at = first_failed_at or failed_at
            remaining = None if deadline is None else deadline - time.monotonic()
            if not is_transient(exc) or (remaining is not None and remaining <= 0):
                raise _final_error(exc, what, failed_at, first_failed_at, attempt) from exc
            delay = min(_MAX_DELAY_S, _FIRST_DELAY_S * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            if remaining is not None:
                delay = min(delay, remaining)
            LOG.warning("%s failed at %s with %s; retrying in %.0f s (attempt %d).",
                        what, _timestamp(failed_at), _describe(exc), delay, attempt)
            time.sleep(delay)
        else:
            if attempt > 1:
                LOG.warning("%s succeeded at %s after %d attempts.",
                            what, _timestamp(datetime.now(UTC)), attempt)
            return result


def _final_error(exc: Exception, what: str, failed_at: datetime,
                 first_failed_at: datetime, attempts: int) -> JobSubmissionError:
    message = f"{what} failed at {_timestamp(failed_at)} with {_describe(exc)}"
    if attempts > 1:
        message += (f", after {attempts} attempts since {_timestamp(first_failed_at)}."
                    " The service was probably unavailable for maintenance; submitting"
                    " again later is safe.")
    else:
        message += "."
    return JobSubmissionError(message, cause=exc, failed_at=failed_at,
                              first_failed_at=first_failed_at, attempts=attempts)
