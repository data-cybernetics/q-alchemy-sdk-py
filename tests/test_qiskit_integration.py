import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.qiskit_integration import QAlchemyInitialize, OptParams


class TestQiskitIntegration(unittest.TestCase):

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
        state_vector = Statevector(qc).data

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data

        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)


    def test_rnd_real(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data

        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)

    def test_rnd_complex(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data

        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-13)
        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)


if __name__ == '__main__':
    unittest.main()
