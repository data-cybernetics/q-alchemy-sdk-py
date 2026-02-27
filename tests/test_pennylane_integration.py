import unittest
from textwrap import dedent

from dotenv import load_dotenv
import math
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.sparse import coo_matrix, coo_array, csr_matrix, vstack

from q_alchemy.pennylane_integration import QAlchemyStatePreparation, OptParams, pennylane_batch_initialize

load_dotenv("../.env-dev")

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

        dev = qml.device('default.qubit', wires=qc.num_qubits)

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
        self.assertLessEqual(np.linalg.norm(state_qiskit - state_pennylane), 1e-10) #not that precise?


    def test_rnd_real(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        dev = qml.device('default.qubit', wires=n_qubits)

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

        dev = qml.device('default.qubit', wires=n_qubits)

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

    def test_fixed_coo(self):

        n_qubits = 4
        coo_data = np.array([1/math.sqrt(3) for i in range(3)])
        coo_rows = np.array([0, 0, 0])
        coo_cols = np.array([0, 1, 11])
        coo_state = csr_matrix((coo_data,(coo_rows, coo_cols)), shape=(1, 2**n_qubits))

        dev = qml.device('default.qubit', wires=n_qubits)

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

        state_pennylane = circuit_pennylane(coo_state)
        state_vector = coo_state.toarray()

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-12) #phase

    @unittest.expectedFailure #from_qasm3 doesn't support include or qubit registers?
    def test_qasm3(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        dev = qml.device('default.qubit', wires=n_qubits)

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

        dev = qml.device('default.qubit', wires=n_qubits)

        circ_list = pennylane_batch_initialize(state_vectors=state_vectors, wires=range(n_qubits), opt_params=OptParams(
        ))

        @qml.qnode(dev)
        def circuit_pennylane(circ):
            circ()
            return qml.state()

        states_pennylane = [circuit_pennylane(circ) for circ in circ_list]
        for state_vector, state_pennylane in zip(state_vectors, states_pennylane):
            self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-11)  # phase. Also a little small?
        fig, ax = qml.draw_mpl(circuit_pennylane)(circ_list[0])
        fig.show()
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()

    def test_batch_coo(self):
        n_qubits = 4
        coo_data = np.array([1/math.sqrt(3) for i in range(3)])
        coo_rows = np.array([0, 0, 0])
        coo_cols = np.array([0, 1, 11])
        coo_states = [coo_matrix((coo_data,(coo_rows, coo_cols)), shape=(1, 2**n_qubits)) for i in range(4)]


        dev = qml.device('default.qubit', wires=n_qubits)

        circ_list = pennylane_batch_initialize(state_vectors=coo_states, wires=range(n_qubits), opt_params=OptParams(
        ))

        @qml.qnode(dev)
        def circuit_pennylane(circ):
            circ()
            return qml.state()

        states_pennylane = [circuit_pennylane(circ) for circ in circ_list]
        for coo_state, state_pennylane in zip(coo_states, states_pennylane):
            state_vector = coo_state.toarray()
            self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-11)  # phase. Also a little small?
        fig, ax = qml.draw_mpl(circuit_pennylane)(circ_list[0])
        fig.show()
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()


    def test_big_coo(self):
        n_qubits = 4
        coo_data = np.array([1/math.sqrt(3) for i in range(3)])
        coo_rows = np.array([0, 0, 0])
        coo_cols = np.array([0, 1, 11])
        coo_states = [coo_matrix((coo_data,(coo_rows, coo_cols)), shape=(1, 2**n_qubits)) for i in range(4)]
        coo_state_stack = vstack(coo_states)


        dev = qml.device('default.qubit', wires=n_qubits)

        circ_list = pennylane_batch_initialize(state_vectors=coo_state_stack, wires=range(n_qubits), opt_params=OptParams(
        ))

        @qml.qnode(dev)
        def circuit_pennylane(circ):
            circ()
            return qml.state()

        states_pennylane = [circuit_pennylane(circ) for circ in circ_list]
        for coo_state, state_pennylane in zip(coo_states, states_pennylane):
            state_vector = coo_state.toarray()
            self.assertLessEqual(1 - abs(np.vdot(state_vector, state_pennylane)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(state_vector - state_pennylane), 1e-11)  # phase. Also a little small?
        fig, ax = qml.draw_mpl(circuit_pennylane)(circ_list[0])
        fig.show()
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()

    def test_qasm3_tiny(self):
        """Uncommenting the include in the QASM program will cause a failure!"""
        prog = '''
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit q0;
        rx(0.5) q0;
        '''
        prog = dedent(prog)
        newprog = '\n'.join([line for line in prog.split('\n') if not line.startswith('include')])
        print(newprog)

        dev = qml.device('default.qubit')
        @qml.qnode(dev)
        def circuit_test():
            qml.from_qasm3(newprog)()
            return qml.state()

        print (circuit_test())

if __name__ == '__main__':
    unittest.main()
