import json
import numpy as np

import pyarrow as pa
from scipy.sparse import coo_array, coo_matrix


def convert_sparse_coo_to_arrow(sparse_coo: coo_array | coo_matrix) -> pa.Table:
    """
    Convert a scipy sparse coo_array (complex valued) to a pyarrow Table.

    Parameters:
        sparse_coo (scipy.sparse.coo_array): The input sparse array with complex data.

    Returns:
        pyarrow.Table: A table with columns 'row', 'col', 'real', and 'imag'.
    """
    # Extract the row and column indices from the COO array
    rows = sparse_coo.row
    cols = sparse_coo.col

    # Extract the data and separate the real and imaginary parts
    data = sparse_coo.data
    real_vals = data.real
    imag_vals = data.imag

    # Create a dictionary mapping column names to the corresponding data arrays
    data_dict = {
        'row': rows,
        'col': cols,
        'real': real_vals,
        'imag': imag_vals
    }

    # Convert the dictionary to a PyArrow table
    table: pa.Table = pa.Table.from_pydict(data_dict)

    # Add shape information to the table metadata
    metadata = {
        'shape': json.dumps(sparse_coo.shape),
        'dtype': ('float64' if data.dtype.kind in 'bui' else data.dtype.name)
    }

    return table.replace_schema_metadata(metadata)


def recover_sparse_coo_from_arrow(arrow_table: pa.Table) -> coo_matrix:
    """
    Recover a scipy sparse coo_array (complex valued) from a pyarrow Table.

    Parameters:
        arrow_table (pyarrow.Table): The table with columns 'row', 'col', 'real', 'imag',
                                    and shape information in metadata.

    Returns:
        scipy.sparse.coo_array: The reconstructed sparse COO array with complex data.
    """
    # Convert arrow columns to numpy arrays
    rows = arrow_table.column('row').to_numpy()
    cols = arrow_table.column('col').to_numpy()
    real_vals = arrow_table.column('real').to_numpy()
    imag_vals = arrow_table.column('imag').to_numpy()

    # Reconstruct complex data
    metadata = arrow_table.schema.metadata or {}
    dtype = metadata.get(b'dtype')
    if dtype is None:
        complex_data = real_vals + 1j * imag_vals  # Historical payloads.
    else:
        dtype = dtype.decode('ascii')
        if dtype not in {'float32', 'float64', 'complex64', 'complex128'}:
            raise ValueError('Unsupported state dtype in Parquet metadata')
        if dtype.startswith('float'):
            if np.any(imag_vals != 0):
                raise ValueError('Real state dtype cannot contain imaginary amplitudes')
            complex_data = real_vals.astype(dtype, copy=False)
        else:
            complex_data = np.empty(len(real_vals), dtype=dtype)
            complex_data.real = real_vals
            complex_data.imag = imag_vals
        if not np.all(np.isfinite(complex_data)):
            raise ValueError('State amplitudes must be finite in the requested dtype')

    # Get shape from metadata
    metadata = arrow_table.schema.metadata
    if metadata and b'shape' in metadata:
        shape = tuple(json.loads(metadata[b'shape'].decode('utf-8')))
        if len(shape) == 1:
            shape = (shape[0], 1)
    else:
        # Fallback if metadata is not available
        shape = (int(rows.max() + 1), int(cols.max() + 1))

    # Create and return the sparse COO array with the original shape
    return coo_matrix((complex_data, (rows, cols)), shape=shape)
