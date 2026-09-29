"""Public SDK transport selection, tested without private Q-Alchemy packages."""
import base64
import io
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

import q_alchemy.initialize as init
from q_alchemy.pyarrow_data import recover_sparse_coo_from_arrow


def decode(payload):
    if payload.startswith(b'\x93NUMPY'):
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return recover_sparse_coo_from_arrow(pq.read_table(io.BytesIO(payload)))


def dense_state():
    state = np.arange(1, 129) * (1 + 2j)
    return state / np.linalg.norm(state)


@pytest.mark.parametrize('count,expected_sparse', [(0, True), (1, True), (13, True), (14, False), (128, False)])
def test_density_boundary_and_lossless_amplitudes(count, expected_sparse):
    state = np.zeros(128, complex)
    state[:count] = np.arange(1, count + 1) * (1 + 2j)
    if count:
        state[count - 1] = 1e-200j  # must never be thresholded away in transport
    payload, n = init._prepare_statevector(state)
    actual = decode(payload)
    assert n == 7
    assert sparse.issparse(actual) == expected_sparse
    np.testing.assert_array_equal(actual.toarray() if sparse.issparse(actual) else actual, state.reshape(1, -1))


def test_clearly_dense_detection_exits_after_one_block(monkeypatch):
    state = np.ones(4 << 20, np.uint8)
    counter = Mock(wraps=np.count_nonzero)
    monkeypatch.setattr(init.np, 'count_nonzero', counter)
    assert init._maybe_make_sparse_row(state) is None
    counter.assert_called_once()
    assert counter.call_args.args[0].size == 1 << 20


@pytest.mark.parametrize('container', [sparse.coo_matrix, sparse.csr_matrix, sparse.coo_array, sparse.csr_array])
def test_explicit_sparse_input_is_never_density_checked_or_densified(monkeypatch, container):
    state = container(dense_state().reshape(1, -1))
    def forbidden(*args, **kwargs):
        raise AssertionError('Already sparse input must not be scanned or densified')
    monkeypatch.setattr(init, '_maybe_make_sparse_row', forbidden)
    monkeypatch.setattr(container, 'toarray', forbidden)
    payload, _ = init._prepare_statevector(state)
    assert init._payload_type(payload) == 'parquet'
    actual = decode(payload)
    np.testing.assert_array_equal(actual.tocoo().data, state.tocoo().data)


def test_huge_sparse_batch_stays_sparse_without_check(monkeypatch):
    matrix = sparse.coo_matrix(([1, 1j], ([0, 1], [0, (1 << 40) - 1])), shape=(2, 1 << 40))
    monkeypatch.setattr(init, '_maybe_make_sparse_row', Mock(side_effect=AssertionError('density check')))
    groups = init._batch_state_groups(matrix)
    assert groups[0][0] == [0, 1]
    assert sparse.issparse(groups[0][1])
    assert groups[0][1].nnz == 2
    assert groups[0][1].shape == matrix.shape


def test_dense_batches_reuse_input_buffer():
    matrix = np.stack([dense_state(), dense_state()[::-1]])
    groups = init._batch_state_groups(matrix)
    assert len(groups) == 1
    assert groups[0][1] is matrix


@pytest.mark.parametrize('use_sparse', [False, True])
@pytest.mark.parametrize('upload', [False, True])
def test_single_inline_and_upload_routes_carry_correct_format(monkeypatch, use_sparse, upload):
    state = np.eye(1, 128, dtype=complex)[0] if use_sparse else dense_state()
    configure = Mock(return_value=object())
    upload_payload = Mock(return_value=object())
    monkeypatch.setattr(init, 'configure_job', configure)
    monkeypatch.setattr(init, '_upload_statevector_payload', upload_payload)
    monkeypatch.setattr(init, 'run_job', Mock(return_value=({'status': 'OK'}, 'qasm')))
    if upload:
        monkeypatch.setattr(init, 'USE_INLINE_STATE_NUM_QUBITS', 0)
    assert init.q_alchemy_as_qasm(state, client=Mock()) == 'qasm'
    options = configure.call_args.kwargs
    assert options['statevector_type'] == ('parquet' if use_sparse else 'numpy_load')
    if upload:
        payload = upload_payload.call_args.args[1]
    else:
        upload_payload.assert_not_called()
        payload = base64.b64decode(options['statevector_data'])
        _, params = init.create_processing_input(init.OptParams(), options['statevector_data'],
                                                statevector_type=options['statevector_type'])
        assert params['state_vector']['state_vector_type'] == options['statevector_type']
    actual = decode(payload)
    np.testing.assert_array_equal(actual.toarray() if sparse.issparse(actual) else actual, state.reshape(1, -1))


@pytest.mark.parametrize('use_sparse', [False, True])
def test_upload_filename_matches_serialized_data(monkeypatch, use_sparse):
    root = Mock()
    entry = Mock()
    entry.work_data_root_link.navigate.return_value = root
    monkeypatch.setattr(init, 'enter_jma', Mock(return_value=entry))
    monkeypatch.setattr(init, 'allow_deletion', Mock())
    state = np.eye(1, 128)[0] if use_sparse else dense_state()
    payload, _ = init._prepare_statevector(state)
    init._upload_statevector_payload(Mock(), payload, init.OptParams(assign_data_hash=False))
    uploaded = root.upload_action.execute.call_args.args[0]
    assert uploaded.filename.endswith('.parquet' if use_sparse else '.npy')
    assert uploaded.binary == payload


def test_mixed_batches_group_and_restore_original_order(monkeypatch):
    targets = [np.eye(1, 128)[0], dense_state(), sparse.csr_matrix(dense_state()), -dense_state()]
    uploads = []
    def upload(client, data, options):
        uploads.append(data)
        return decode(init._serialize_statevector(data))
    monkeypatch.setattr(init, 'upload_statevector', upload)
    monkeypatch.setattr(init, 'configure_job', lambda **kwargs: kwargs)
    def run(job, options, timeout):
        state = job['statevector_data']
        matrix = state.toarray() if sparse.issparse(state) else state
        labels = []
        for row in matrix:
            if np.count_nonzero(row) == 1:
                labels.append('sparse-input')
            elif sparse.issparse(state):
                labels.append('explicit-sparse')
            else:
                labels.append('dense-positive' if row[0].real > 0 else 'dense-negative')
        summaries = [{'label': label} for label in labels]
        return (summaries[0], labels[0]) if len(labels) == 1 else (summaries, labels)
    monkeypatch.setattr(init, 'run_job', run)
    qasms, summaries = init.q_alchemy_as_qasm_parallel_states(targets, {}, client=Mock(), return_summary=True)
    expected = ['sparse-input', 'dense-positive', 'explicit-sparse', 'dense-negative']
    assert qasms == expected
    assert summaries == [{'label': label} for label in expected]
    assert len(uploads) == 2
    assert sum(isinstance(item, np.ndarray) for item in uploads) == 1
    assert sum(isinstance(item, pa.Table) for item in uploads) == 1


def test_parallel_dense_options_select_and_serialize_only_once(monkeypatch):
    check = Mock(wraps=init._maybe_make_sparse_row)
    serializer = Mock(wraps=init._serialize_statevector)
    monkeypatch.setattr(init, '_maybe_make_sparse_row', check)
    monkeypatch.setattr(init, '_serialize_statevector', serializer)
    def run(payload, n, options, client, summary, inline):
        assert init._payload_type(payload) == 'numpy_load'
        assert base64.b64decode(inline) == payload
        return 'qasm'
    monkeypatch.setattr(init, '_run_prepared_statevector', run)
    assert init.q_alchemy_as_qasm_parallel(dense_state(), [{}, {}, {}], client=Mock()) == ['qasm'] * 3
    check.assert_called_once()
    serializer.assert_called_once()


@pytest.mark.parametrize('states', [[], [np.ones(4), np.ones(8)], [np.ones(4), np.ones(3)]])
def test_invalid_batch_fails_before_any_upload(monkeypatch, states):
    upload = Mock(side_effect=AssertionError('must not upload invalid batch'))
    monkeypatch.setattr(init, 'upload_statevector', upload)
    with pytest.raises(ValueError):
        init.q_alchemy_as_qasm_parallel_states(states, {}, client=Mock())
    upload.assert_not_called()


def test_mixed_group_failure_propagates_without_partial_results(monkeypatch):
    monkeypatch.setattr(init, 'upload_statevector', Mock(return_value=object()))
    monkeypatch.setattr(init, 'configure_job', Mock(return_value=object()))
    monkeypatch.setattr(init, 'run_job', Mock(side_effect=[({'status': 'OK'}, 'qasm'), ValueError('failed second group')]))
    with pytest.raises(ValueError, match='failed second group'):
        init.q_alchemy_as_qasm_parallel_states([dense_state(), np.eye(1, 128)[0]], {}, client=Mock())


def test_short_service_result_is_rejected(monkeypatch):
    monkeypatch.setattr(init, 'upload_statevector', Mock(return_value=object()))
    monkeypatch.setattr(init, 'configure_job', Mock(return_value=object()))
    monkeypatch.setattr(init, 'run_job', Mock(return_value=([{}], ['qasm'])))
    with pytest.raises(ValueError, match='unexpected batch size'):
        init.q_alchemy_as_qasm_parallel_states([dense_state(), dense_state()], {}, client=Mock())
