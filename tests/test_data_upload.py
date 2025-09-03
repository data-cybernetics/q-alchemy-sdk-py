import io
import unittest

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from sklearn.datasets import fetch_openml

from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow, recover_sparse_coo_from_arrow


class TestDataUpload(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        self.mnist = fetch_openml('mnist_784', version=1, parser="auto")

    def tearDown(self):
        # This method will be called after each test
        pass

    def test_data_parallel_parsing(self):
        zeros: np.ndarray = self.mnist.data[self.mnist.target == "0"].iloc[0:10].to_numpy()
        filler = np.empty((zeros.shape[0], 2 ** 10 - zeros.shape[1]))
        filler.fill(0)

        zeros = np.hstack([zeros, filler])
        zeros = zeros / np.linalg.norm(zeros, axis=1).reshape(-1, 1)

        zeros_sparse = coo_matrix(zeros)

        zeros_table = convert_sparse_coo_to_arrow(zeros_sparse)

        buffer = io.BytesIO()
        pq.write_table(zeros_table, buffer)
        buffer.seek(0)

        table = pq.read_table(buffer)
        zeros_sparse_recovered = recover_sparse_coo_from_arrow(table)

        self.assertLessEqual(np.linalg.norm(zeros_sparse.data - zeros_sparse_recovered.data), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(zeros_sparse.data, zeros_sparse_recovered.data))**2, 1e-13)

    def test_data_single_parsing(self):
        zeros: np.ndarray = self.mnist.data[self.mnist.target == "0"].iloc[0].to_numpy(dtype=np.complex128)
        filler = np.empty(2 ** 10 - zeros.shape[0])
        filler.fill(0)

        zeros = np.hstack([zeros, filler])
        zeros = zeros / np.linalg.norm(zeros)

        zeros_sparse = coo_matrix(zeros).reshape(1, -1)
        zeros_table = convert_sparse_coo_to_arrow(zeros_sparse)

        buffer = io.BytesIO()
        pq.write_table(zeros_table, buffer)
        buffer.seek(0)

        table = pq.read_table(buffer)
        zeros_sparse_recovered = recover_sparse_coo_from_arrow(table)

        self.assertLessEqual(np.linalg.norm(zeros_sparse.data - zeros_sparse_recovered.data), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(zeros_sparse.data, zeros_sparse_recovered.data))**2, 1e-13)



if __name__ == '__main__':
    unittest.main()
