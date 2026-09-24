from pathlib import Path
import unittest
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv
import math
import os
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.sparse import coo_matrix, coo_array, csr_matrix, vstack

from q_alchemy.pennylane_integration import QAlchemyStatePreparation, OptParams, pennylane_batch_initialize

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

requires_api_key = unittest.skipUnless(
    os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"),
    "no Q_ALCHEMY_API_KEY/PINEXQ_API_KEY: skipping live PennyLane integration test",
)


class TestPennyLaneIntegration(unittest.TestCase):

    def test_qasm3_parser_current_support(self):
        qasm3_program = """
        OPENQASM 3.0;

        qubit q0;
        qubit q1;

        h q0;
        cx q0, q1;
        """

        dev = qml.device("default.qubit", wires=2)

        loaded = qml.from_qasm3(
            qasm3_program,
            {
                "q0": 1,
                "q1": 0,
            },
        )

        @qml.qnode(dev)
        def circuit():
            loaded()
            return qml.state()

        state = circuit()

        expected = np.array(
            [
                1 / np.sqrt(2),
                0,
                0,
                1 / np.sqrt(2),
            ],
            dtype=complex,
        )

        assert np.allclose(state, expected, atol=1e-12)

    def test_qasm3_global_phase(self):
        phase = 0.37

        qasm3_program = """
        OPENQASM 3.0;
        qubit q0;
        rx(0.5) q0;
        """

        loaded = qml.from_qasm3(
            qasm3_program,
            {"q0": 0},
        )

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.GlobalPhase(phase)
            loaded()
            return qml.state()

        state = circuit()

        base_state = np.array(
            [
                np.cos(0.25),
                -1j * np.sin(0.25),
            ],
            dtype=complex,
        )

        expected = np.exp(-1j * phase) * base_state

        assert np.allclose(state, expected, atol=1e-12)

    def setUp(self):
        # This method will be called before each test
        pass

    def tearDown(self):
        # This method will be called after each test
        pass

    @requires_api_key
    def test_fixed_complex(self):

        with (Path(__file__).parent / "data" / "test.qasm").open("r") as f:
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


    @requires_api_key
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

    @requires_api_key
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

    @requires_api_key
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
    @requires_api_key
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

    @requires_api_key
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
        plt.close(fig)
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()

    @requires_api_key
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
        plt.close(fig)
        # for ops in ops_list: #too much RAM
        #     fig, ax = qml.draw_mpl(circuit_pennylane)(ops)
        #     fig.show()


    @requires_api_key
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
        plt.close(fig)
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
