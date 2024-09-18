import unittest

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.parser.qasm_pennylane import from_qasm


class TestPennyLaneIntegration(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    def test_pennylane_vs_qiskit(self):

        with open("data/test.qasm", "r") as f:
            qasm = f.read()

        qc = QuantumCircuit.from_qasm_str(qasm)
        state_qiskit = Statevector(qc).data

        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit_pennylane(q):
            from_qasm(q)
            return qml.state()

        state_pennylane = circuit_pennylane(qasm)

        self.assertLessEqual(np.linalg.norm(state_qiskit - state_pennylane), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(state_qiskit, state_pennylane))**2, 1e-13)


if __name__ == '__main__':
    unittest.main()
