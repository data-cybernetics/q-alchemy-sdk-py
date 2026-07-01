"""Tests for the Qiskit BackendV2 (QAlchemyBackend).

Offline tests use a dummy client (no network). Live tests need a
Q_ALCHEMY_API_KEY and the deployed simulator ProCon; they skip otherwise.
"""

import os
import unittest

from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit.result import Result

from q_alchemy import QAlchemyBackend, QAlchemyProvider
from q_alchemy.simulator import SparseSimulator
from q_alchemy.qalchemy_backend import _counts_to_hex, QALCHEMY_BASIS_GATES

load_dotenv("../.env")


def _offline_backend(**kwargs) -> QAlchemyBackend:
    # A dummy client satisfies the API-key guard; no calls are made offline.
    return QAlchemyBackend(params=SparseSimulator(client=object()), **kwargs)


class TestBackendConstruction(unittest.TestCase):
    def test_is_backend_v2_with_expected_target(self):
        be = _offline_backend(num_qubits=8)
        self.assertIsInstance(be, BackendV2)
        self.assertEqual(be.num_qubits, 8)
        self.assertIsNone(be.max_circuits)
        ops = set(be.target.operation_names)
        for gate in ("u", "cx", "measure", "reset", "h", "swap", "ccx"):
            self.assertIn(gate, ops)

    def test_default_options(self):
        be = _offline_backend()
        self.assertEqual(be.options.shots, 1024)
        self.assertFalse(be.options.save_sparse_statevector)

    def test_transpile_targets_the_backend_basis(self):
        be = _offline_backend(num_qubits=3)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        tqc = transpile(qc, be)
        assert set(tqc.count_ops()).issubset(set(QALCHEMY_BASIS_GATES))


class TestCountsFormatting(unittest.TestCase):
    def test_counts_to_hex_from_bitstrings_and_hex(self):
        self.assertEqual(_counts_to_hex({"00": 5, "11": 7}), {"0x0": 5, "0x3": 7})
        self.assertEqual(_counts_to_hex({"0x0": 5, "0x3": 7}), {"0x0": 5, "0x3": 7})

    def test_result_get_counts_reformats_to_bitstrings(self):
        result = Result.from_dict({
            "backend_name": "q_alchemy_simulator", "backend_version": "0.1.0",
            "qobj_id": "j", "job_id": "j", "success": True, "status": "COMPLETED",
            "results": [{
                "shots": 12, "success": True, "status": "DONE",
                "header": {"name": "bell", "metadata": {}, "n_qubits": 2, "memory_slots": 2},
                "data": {"counts": {"0x0": 5, "0x3": 7}},
            }],
        })
        self.assertEqual(result.get_counts(), {"00": 5, "11": 7})


class TestInteropNormalization(unittest.TestCase):
    """Fixes that let arbitrary tools (e.g. PennyLane-Qiskit) use the backend."""

    def test_counts_to_memory_expands_to_shots(self):
        from q_alchemy.qalchemy_backend import _counts_to_memory
        memory = _counts_to_memory({"0x0": 2, "0x3": 1})
        self.assertEqual(len(memory), 3)
        self.assertEqual(sorted(memory), ["0x0", "0x0", "0x3"])

    def test_canonicalize_registers(self):
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        from q_alchemy.simulator import _canonicalize_registers

        qc = QuantumCircuit(QuantumRegister(2, "q0"), ClassicalRegister(2, "c0"))
        qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
        canon = _canonicalize_registers(qc)
        self.assertEqual([r.name for r in canon.qregs], ["q"])
        self.assertEqual([r.name for r in canon.cregs], ["c"])
        self.assertEqual(dict(canon.count_ops()), dict(qc.count_ops()))

    def test_canonical_circuit_passes_through_unchanged(self):
        from qiskit import QuantumCircuit
        from q_alchemy.simulator import _canonicalize_registers
        qc = QuantumCircuit(2)
        qc.h(0)
        self.assertIs(_canonicalize_registers(qc), qc)

    def test_backend_is_deepcopyable(self):
        # PennyLane-Qiskit deep-copies the backend; must not choke on the client.
        import copy
        be = _offline_backend()
        self.assertIs(copy.deepcopy(be), be)


class TestProvider(unittest.TestCase):
    def test_provider_backends_offline(self):
        provider = QAlchemyProvider(params=SparseSimulator(client=object()))
        backends = provider.backends()
        self.assertEqual(len(backends), 1)
        self.assertEqual(backends[0].name, "q_alchemy_simulator")
        self.assertEqual(provider.backends(name="nope"), [])


@unittest.skipUnless(os.getenv("Q_ALCHEMY_API_KEY"), "no Q_ALCHEMY_API_KEY: skipping live backend tests")
class TestLiveBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = QAlchemyBackend(num_qubits=8, job_completion_timeout_sec=180)
        cls.bell = QuantumCircuit(2, 2)
        cls.bell.h(0)
        cls.bell.cx(0, 1)
        cls.bell.measure([0, 1], [0, 1])

    def test_run_get_counts(self):
        job = self.backend.run(transpile(self.bell, self.backend), shots=4096, seed_simulator=11)
        counts = job.result().get_counts()
        self.assertEqual(sum(counts.values()), 4096)
        self.assertTrue(set(counts) <= {"00", "11"})

    def test_save_sparse_statevector(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        res = self.backend.run(qc, save_sparse_statevector=True, sparse_index_format="bitstring").result()
        sv = res.data(0)["sparse_statevector"]
        self.assertEqual(set(sv["indices"]), {"00", "11"})


if __name__ == "__main__":
    unittest.main()
