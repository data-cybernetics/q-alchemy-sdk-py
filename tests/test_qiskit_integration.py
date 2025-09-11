import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from q_alchemy.qiskit_integration import (
    QAlchemyInitialize,
    OptParams,
    qiskit_batch_initialize
)
from dotenv import load_dotenv

load_dotenv()

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
        state_vector = Statevector(qc)

        opt_params = OptParams(
        #        api_key=os.environ["Q_ALCHEMY_API_KEY"]
        )

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=opt_params
        )

        state_qiskit = Statevector(instr)

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?


    def test_rnd_real(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?

    def test_rnd_complex(self):

        n_qubits = 4
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)
        subtest_params = [True, False]
        for use_qasm3 in subtest_params:
            with self.subTest(use_qasm3=use_qasm3):
                instr = QAlchemyInitialize(
                    params=state_vector,
                    opt_params=OptParams(
                        use_qasm3=use_qasm3,
                        #api_key="<your api key>"
                    )
                )
                circuit_qiskit = instr.definition

                state_qiskit = Statevector(circuit_qiskit).data

                self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
                self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?

    def test_large_complex(self):

        n_qubits = 12
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        instr = QAlchemyInitialize(
            params=state_vector,
            opt_params=OptParams(
                #api_key="<your api key>"
            )
        )
        circuit_qiskit = instr.definition

        state_qiskit = Statevector(circuit_qiskit).data # 16 is too large!

        self.assertLessEqual(1 - abs(np.vdot(state_vector, state_qiskit))**2, 1e-13)
        self.assertLessEqual(np.linalg.norm(state_vector - state_qiskit), 1e-12) # not quite that precise?

    def test_batch_complex(self):
        n_qubits = 4
        n_states = 4
        state_vectors = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j] * n_states
        state_vectors = [sv / np.linalg.norm(sv) for sv in state_vectors]

        gate_list = qiskit_batch_initialize(state_vectors=state_vectors, opt_params=OptParams(
            use_qasm3=True,
            # api_key="<your api key>"
        ))
        qcs = [QuantumCircuit(n_qubits) for gate in gate_list]
        for gate, qc in zip(gate_list, qcs):
            qc.append(gate, range(n_qubits))

        qiskit_states = [Statevector(circuit).data for circuit in qcs]

        for init_state, qiskit_state in zip(state_vectors, qiskit_states):
            self.assertLessEqual(1 - abs(np.vdot(init_state, qiskit_state)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(init_state - qiskit_state), 1e-12)  # not quite that precise?

        for qc in qcs:
            print(qc)


if __name__ == '__main__':
    unittest.main()
