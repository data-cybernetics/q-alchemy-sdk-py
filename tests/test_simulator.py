"""Tests for the hosted sparse simulator client (`q_alchemy.simulator`).

The convention/parsing tests run fully offline. The end-to-end tests require a
live API key (`Q_ALCHEMY_API_KEY`) and the deployed q-alchemy-simulator ProCon;
they skip when no key is present.
"""

import os
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from dotenv import load_dotenv

from q_alchemy.simulator import (
    SparseSimulator,
    SparseStatevectorResult,
    CountsResult,
    TomographyResult,
)
from q_alchemy.pyarrow_data import recover_sparse_coo_from_arrow

load_dotenv("../.env")

A = 1.0 / np.sqrt(2.0)


def _bell_result(index_format: str, index_convention: str, indices: list[str]) -> SparseStatevectorResult:
    return SparseStatevectorResult(
        num_qubits=2,
        nnz=len(indices),
        index_format=index_format,
        index_convention=index_convention,
        indices=indices,
        amplitudes=[complex(A, 0.0)] * len(indices),
        raw={},
    )


class TestSparseFormatConversion(unittest.TestCase):
    """SparseStatevectorResult <-> the SDK's canonical sparse (COO/arrow) format."""

    def test_little_endian_hex_and_bitstring_match_qiskit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        target = Statevector(qc).data
        for result in (
            _bell_result("hex", "little_endian", ["0x0", "0x3"]),
            _bell_result("bitstring", "little_endian", ["00", "11"]),
        ):
            self.assertEqual(result.qiskit_indices(), [0, 3])
            np.testing.assert_allclose(result.to_dense(), target)

    def test_big_endian_indices_are_bit_reversed(self):
        # X on q[0] over 3 qubits is index 1 in Qiskit order, "100" in big-endian.
        result = SparseStatevectorResult(3, 1, "bitstring", "big_endian", ["100"], [complex(1.0, 0.0)], {})
        self.assertEqual(result.qiskit_indices(), [1])
        qc = QuantumCircuit(3)
        qc.x(0)
        np.testing.assert_allclose(result.to_dense(), Statevector(qc).data)

    def test_to_coo_shape_and_arrow_round_trip(self):
        result = _bell_result("hex", "little_endian", ["0x0", "0x3"])
        coo = result.to_coo()
        self.assertEqual(coo.shape, (1, 4))
        self.assertEqual(coo.nnz, 2)
        recovered = recover_sparse_coo_from_arrow(result.to_arrow())
        np.testing.assert_allclose(coo.toarray(), recovered.toarray())


class TestResultParsing(unittest.TestCase):
    def test_counts_result(self):
        r = CountsResult.from_raw({"num_qubits": 2, "shots": 10, "counts": {"00": 5, "11": 5}})
        self.assertEqual(r.counts, {"00": 5, "11": 5})
        self.assertEqual(r.shots, 10)

    def test_tomography_result_state_is_complex_ndarray(self):
        raw = {
            "num_qubits": 2,
            "sparse_statevector": {
                "num_qubits": 2, "nnz": 2, "index_format": "bitstring",
                "index_convention": "little_endian", "indices": ["00", "11"],
                "amplitudes": [[A, 0.0], [A, 0.0]],
            },
            "measurement_indices": None,
            "state": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            "state_is_reduced": False,
            "purity": 1.0,
            "state_fidelity": 1.0,
        }
        t = TomographyResult.from_raw(raw)
        self.assertEqual(t.state.shape, (2, 2))
        self.assertTrue(np.iscomplexobj(t.state))
        self.assertAlmostEqual(t.state_fidelity, 1.0)


class TestInputFormResolution(unittest.TestCase):
    def setUp(self):
        # A client object is enough to bypass the API-key check; no calls are made.
        self.sim = SparseSimulator(client=object())

    def test_auto_small_circuit_is_inline_qasm(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        suffix, params, upload = self.sim._prepare_input(qc, "auto")
        self.assertEqual(suffix, "from_qasm_string")
        self.assertIn("qasm", params)
        self.assertIsNone(upload)

    def test_auto_large_circuit_is_qpy(self):
        sim = SparseSimulator(client=object(), inline_max_qubits=1)
        qc = QuantumCircuit(2)
        qc.h(0)
        suffix, _, upload = sim._prepare_input(qc, "auto")
        self.assertEqual(suffix, "from_qpy_file")
        self.assertEqual(upload[0], "circuit.qpy")

    def test_forced_qasm_file(self):
        qc = QuantumCircuit(1)
        suffix, _, upload = self.sim._prepare_input(qc, "qasm_file")
        self.assertEqual(suffix, "from_qasm_file")
        self.assertEqual(upload[0], "circuit.qasm")


@unittest.skipUnless(os.getenv("Q_ALCHEMY_API_KEY"), "no Q_ALCHEMY_API_KEY: skipping live ProCon tests")
class TestLiveSimulator(unittest.TestCase):
    """End-to-end against the deployed ProCon on jobs.api.q-alchemy.com."""

    @classmethod
    def setUpClass(cls):
        cls.sim = SparseSimulator(job_completion_timeout_sec=180)
        cls.bell = QuantumCircuit(2)
        cls.bell.h(0)
        cls.bell.cx(0, 1)

    def test_counts(self):
        meas = self.bell.copy()
        meas.measure_all()
        r = self.sim.counts(meas, shots=2048, seed_simulator=7)
        self.assertEqual(sum(r.counts.values()), 2048)
        self.assertTrue(set(r.counts) <= {"00", "11"})

    def test_sparse_statevector_matches_qiskit(self):
        r = self.sim.sparse_statevector(self.bell, sparse_index_format="bitstring")
        self.assertEqual(r.nnz, 2)
        np.testing.assert_allclose(r.to_dense(), Statevector(self.bell).data, atol=1e-9)

    def test_tomography_fidelity(self):
        r = self.sim.tomography(self.bell)
        self.assertAlmostEqual(r.state_fidelity, 1.0, places=6)
        self.assertAlmostEqual(r.purity, 1.0, places=6)

    def test_round_trip_into_loader(self):
        from q_alchemy import q_alchemy_as_qasm

        sv = self.sim.sparse_statevector(self.bell)
        qasm = q_alchemy_as_qasm(sv.to_coo().toarray().ravel())
        prep = QuantumCircuit.from_qasm_str(qasm)
        self.assertGreater(state_fidelity(Statevector(prep), Statevector(self.bell)), 0.99)


if __name__ == "__main__":
    unittest.main()
