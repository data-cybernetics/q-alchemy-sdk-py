import json
import numpy as np
import pytest
from q_alchemy.quantum_io_contract import State

DTYPES = ["float32", "float64", "complex64", "complex128"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sparse", [False, True])
def test_state_dtype_roundtrip(dtype, sparse):
    values = np.array([.6, .8j if dtype.startswith("complex") else .8], dtype=dtype)
    state = (State.sparse(num_qubits=72, indices=[0, (1 << 71) + 3], amplitudes=values)
             if sparse else State.dense(values))
    restored = State.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state
    assert restored.dtype == dtype
    np.testing.assert_array_equal(np.asarray(restored.amplitudes, dtype=dtype), values)


def test_legacy_wire_defaults_to_complex128():
    payload = State.dense(np.array([.6, .8], dtype="float32")).to_dict()
    payload.pop("dtype")
    assert State.from_dict(payload).dtype == "complex128"


@pytest.mark.parametrize("dtype", ["float16", "int64", "object", "garbage", 32, []])
def test_invalid_dtype_rejected(dtype):
    with pytest.raises(ValueError, match="dtype"):
        State.dense([1., 0.], dtype=dtype)


def test_explicit_real_dtype_rejects_imaginary_component():
    with pytest.raises(ValueError, match="imaginary"):
        State.sparse(num_qubits=72, indices=[0], amplitudes=[1j], dtype="float64")


def test_cast_overflow_rejected():
    with pytest.raises(ValueError, match="finite"):
        State.dense([1e100, 0.], dtype="float32")


def test_list_dtype_can_be_explicit_and_integral_defaults_to_float64():
    assert State.dense([.6, .8], dtype="float32").dtype == "float32"
    assert State.dense([1, 0]).dtype == "float64"

from io import BytesIO
from scipy import sparse
from q_alchemy import initialize as init
from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow, recover_sparse_coo_from_arrow


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["dense", "auto_sparse", "explicit_sparse", "state_dense", "state_sparse"])
def test_preparation_transport_dtype(dtype, kind):
    values = np.full(128, 1 / np.sqrt(128), dtype=dtype)
    if kind in {"auto_sparse", "explicit_sparse", "state_sparse"}:
        values[:] = 0
        values[0] = 1
    source = values
    if kind == "explicit_sparse": source = sparse.coo_matrix(values.reshape(1, -1))
    if kind == "state_dense": source = State.dense(values)
    if kind == "state_sparse": source = State.sparse(num_qubits=7, indices=[0], amplitudes=values[:1])
    payload, _ = init._prepare_statevector(source)
    if payload.startswith(b"\x93NUMPY"):
        assert np.load(BytesIO(payload)).dtype.name == dtype
    else:
        assert json.loads(payload)["dtype"] == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_parquet_dtype_metadata(dtype):
    values = sparse.coo_matrix(np.array([[.6, .8]], dtype=dtype))
    table = convert_sparse_coo_to_arrow(values)
    assert recover_sparse_coo_from_arrow(table).dtype.name == dtype


def test_mixed_batch_keeps_each_original_dtype():
    vectors = [np.full(128, 1 / np.sqrt(128), dtype=dtype) for dtype in DTYPES]
    groups = init._batch_state_groups(vectors)
    assert len(groups) == 4
    for indices, array in groups:
        assert array.dtype.name == DTYPES[indices[0]]


@pytest.mark.parametrize("values", [1., [[1., 0.]]])
def test_state_rejects_nonvector_amplitudes(values):
    with pytest.raises(ValueError, match="one-dimensional"):
        State.dense(values)


def test_wire_rejects_null_dtype():
    payload = State.dense([1., 0.]).to_dict()
    payload["dtype"] = None
    with pytest.raises(ValueError, match="dtype"):
        State.from_dict(payload)
