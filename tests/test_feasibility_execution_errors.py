from q_alchemy import FeasibilityExecutionError, FeasibilityJob


class FailingResultJob:
    def __init__(self, error: BaseException):
        self.error = error

    def wait_for_state(self, *args, **kwargs):
        raise self.error


def test_result_exposes_remote_feasibility_error_without_pine_job_wrapper_noise():
    original = RuntimeError(
        "Job failed'. Error:Exception while processing job <id:test>\n"
        "[/procon/error]\n"
        "requested backend 'ibm_missing' is not accessible"
    )
    job = FeasibilityJob(
        FailingResultJob(original),
        timeout_sec=10,
        remove_data=False,
    )

    try:
        job.result()
    except FeasibilityExecutionError as exc:
        assert str(exc) == (
            "Feasibility execution failed: "
            "requested backend 'ibm_missing' is not accessible"
        )
        assert exc.original_exception is original
        assert exc.__cause__ is None
    else:
        raise AssertionError("FeasibilityExecutionError was not raised")


def test_result_uses_original_exception_message_when_no_remote_marker_exists():
    original = RuntimeError("transport connection reset")
    job = FeasibilityJob(
        FailingResultJob(original),
        timeout_sec=10,
        remove_data=False,
    )

    try:
        job.result()
    except FeasibilityExecutionError as exc:
        assert str(exc) == "Feasibility execution failed: transport connection reset"
        assert exc.original_exception is original
        assert exc.__cause__ is None
    else:
        raise AssertionError("FeasibilityExecutionError was not raised")


def test_result_does_not_translate_process_control_exceptions():
    original = KeyboardInterrupt()
    job = FeasibilityJob(
        FailingResultJob(original),
        timeout_sec=10,
        remove_data=False,
    )

    try:
        job.result()
    except KeyboardInterrupt as exc:
        assert exc is original
    else:
        raise AssertionError("KeyboardInterrupt was unexpectedly translated")
