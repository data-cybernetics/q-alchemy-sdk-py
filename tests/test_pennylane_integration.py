import unittest
from dotenv import load_dotenv

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.pennylane_integration import QAlchemyStatePreparation, OptParams

load_dotenv()

class TestPennyLaneIntegration(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    def test_fixed_complex(self):

        with open("data/test.qasm", "r") as f:
            qasm = f.read()

        qc = QuantumCircuit.from_qasm_str(qasm)
        state_qiskit = Statevector(qc).data

        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit_pennylane(state):
            QAlchemyStatePreparation(
                state,
                wires=range(qc.num_qubits),
                opt_params=OptParams(
                    #api_key="<your api key>"
                )
            )
            return qml.state()

        state_pennylane = circuit_pennylane(state_qiskit)

        self.assertLessEqual(1 - abs(np.vdot(state_qiskit, state_pennylane))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_qiskit - state_pennylane), 1e-12) #phase


    def test_rnd_real(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit_pennylane(state):
            QAlchemyStatePreparation(
                state,
                wires=range(n_qubits),
                opt_params=OptParams(
                    #api_key="<your api key>"
                )
            )
            return qml.state()

        state_pennylane = circuit_pennylane(state_vector)

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-12) #phase

    def test_rnd_complex(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        dev = qml.device('default.qubit')

        @qml.qnode(dev)
        def circuit_pennylane(state):
            QAlchemyStatePreparation(
                state,
                wires=range(n_qubits),
                opt_params=OptParams(
                    #api_key="<your api key>"
                )
            )
            return qml.state()

        state_pennylane = circuit_pennylane(state_vector)

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-12) #phase


if __name__ == '__main__':
    unittest.main()
