from unittest.mock import Mock
import numpy as np
import pytest
from scipy.sparse import coo_matrix
import q_alchemy.initialize as api


def test_small_sparse_payload_is_inline_independent_of_qubits():
    payload, qubits = api._prepare_statevector(coo_matrix(([1.], ([0],[0])), shape=(1, 1<<20)))
    assert qubits == 20
    assert api._fits_inline(payload)
    assert not api._fits_inline(b"x" * api.MAX_INLINE_STATE_BYTES)


@pytest.mark.parametrize("failure", ["wait", "download", "invalid"])
def test_recoverable_failures_preserve_job(monkeypatch, failure):
    job = Mock()
    if failure == "wait": job.wait_for_state.side_effect = TimeoutError("waiting")
    extract = Mock(side_effect=OSError("download") if failure == "download" else None,
                   return_value=({}, "") if failure == "invalid" else ({"status":"OK"}, "qasm"))
    monkeypatch.setattr(api, "extract_result", extract)
    cleanup = Mock()
    monkeypatch.setattr(api, "clean_up_job", cleanup)
    with pytest.raises(Exception) as error:
        api.run_job(job, api.OptParams(), 1)
    assert error.value.initialization_job is job
    cleanup.assert_not_called()


def test_success_survives_cleanup_failure_and_retries_cached_result(monkeypatch):
    job = Mock()
    result = ({"status":"OK"}, "qasm")
    extract = Mock(return_value=result)
    cleanup = Mock(side_effect=[OSError("cleanup"), None])
    monkeypatch.setattr(api, "extract_result", extract)
    monkeypatch.setattr(api, "clean_up_job", cleanup)
    assert api.run_job(job, api.OptParams(), 1) == result
    assert api.run_job(job, api.OptParams(), 1) == result
    assert extract.call_count == 1 and job.wait_for_state.call_count == 1
    assert cleanup.call_count == 2


def test_bad_batch_size_is_rejected_before_cleanup(monkeypatch):
    job = Mock()
    monkeypatch.setattr(api, "extract_result", lambda _: ([{"status":"OK"}], ["qasm"]))
    cleanup = Mock()
    monkeypatch.setattr(api, "clean_up_job", cleanup)
    with pytest.raises(ValueError, match="batch size"):
        api.run_job(job, api.OptParams(), 1, expected_states=2)
    cleanup.assert_not_called()
