import unittest

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Statevector

from q_alchemy.initialize import OptParams, q_alchemy_as_qasm_parallel_states

from dotenv import load_dotenv

assert load_dotenv("../.env")

class InitializeTestCase(unittest.TestCase):
    def test_batch(self):
        n_qubits = 8

        state_vectors = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j for i in range(4)]
        state_vectors = [sv / np.linalg.norm(sv) for sv in state_vectors]

        qasm_list, summary_list = q_alchemy_as_qasm_parallel_states(
            state_vector=state_vectors,
            opt_params=OptParams(
                # api_key="<your api key>"
            ),
            return_summary = True
        )
        qiskit_circuits = [qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                           for qasm in qasm_list]

        qiskit_states = [Statevector(circuit).data * np.exp(1j*summary["global_phase"])
                         for circuit, summary in zip(qiskit_circuits, summary_list) ]

        for init_state, qiskit_state in zip(state_vectors, qiskit_states):
            self.assertLessEqual(1 - abs(np.vdot(init_state, qiskit_state)) ** 2, 1e-13)
            self.assertLessEqual(np.linalg.norm(init_state-qiskit_state), 1e-12)  # not quite that precise?

if __name__ == '__main__':
    unittest.main()
