import unittest
from dotenv import load_dotenv

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.pennylane_integration import QAlchemyStatePreparation, OptParams, batch_initialization
from q_alchemy.qiskit_integration import parallel_initialize

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

    @unittest.expectedFailure #from_qasm3 doesn't support include?
    def test_qasm3(self):

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
                    use_qasm3=True,
                    #api_key="<your api key>"
                )
            )
            return qml.state()

        state_pennylane = circuit_pennylane(state_vector)

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-12) #phase


    def test_batch_complex(self):
        n_qubits = 8
        n_states = 4
        state_vectors = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j] * n_states
        state_vectors = [sv / np.linalg.norm(sv) for sv in state_vectors]

        dev = qml.device('default.qubit')

        ops_list = batch_initialization(
            state_vectors=state_vectors,
            wires=range(n_qubits),
            hyperparameters=OptParams(),
        )

        @qml.qnode(dev)
        def circuit_pennylane(ops):
            for op in ops:
                qml.apply(op)
            return qml.state()

        states_pennylane = [circuit_pennylane(ops) for ops in ops_list]
        for state_vector, state_pennylane in zip(state_vectors, states_pennylane):
            self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-12)  # phase
        fig, ax = qml.draw_mpl(circuit_pennylane)(ops_list[0])
        fig.show()
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()

if __name__ == '__main__':
    unittest.main()
