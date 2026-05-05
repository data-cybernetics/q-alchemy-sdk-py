import unittest
from cmath import polar

import numpy as np
from qiskit import qasm2, qasm3
from qiskit.quantum_info import Statevector

from qiskit_addon_utils.slicing import slice_by_barriers
from q_alchemy.initialize import OptParams, q_alchemy_as_qasm_parallel_states, InitializationMethods, q_alchemy_as_qasm

from dotenv import load_dotenv

load_dotenv("../.env") # the 'assert' was causing the import to fail during test discovery.

class InitializeTestCase(unittest.TestCase):
    def test_batch(self):
        n_qubits = 8

        subtest_params = [
            (False, InitializationMethods.AUTO),
            (True, InitializationMethods.AUTO),
            (False, InitializationMethods.HIERARCHICAL_TUCKER)
        ]
        for use_qasm3, initialization_method in subtest_params:
            with self.subTest(use_qasm3=use_qasm3, initialization_method=initialization_method):
                state_vectors = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j for i in range(4)]
                state_vectors = [sv / np.linalg.norm(sv) for sv in state_vectors]

                qasm_list, summary_list = q_alchemy_as_qasm_parallel_states(
                    state_vector=state_vectors,
                    opt_params=OptParams(
                        use_qasm3=use_qasm3,
                        initialization_method=initialization_method
                        # api_key="<your api key>"
                    ),
                    return_summary = True
                )
                if use_qasm3:
                    qiskit_circuits = [qasm3.loads(qasm) for qasm in qasm_list]
                else:
                    qiskit_circuits = [qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                                       for qasm in qasm_list]

                qiskit_states = [Statevector(circuit).data * np.exp(1j*summary["global_phase"])
                                 for circuit, summary in zip(qiskit_circuits, summary_list) ]

                for init_state, qiskit_state in zip(state_vectors, qiskit_states):
                    self.assertLessEqual(1 - abs(np.vdot(init_state, qiskit_state) ** 2), 1e-13)
                    self.assertLessEqual(np.linalg.norm(init_state-qiskit_state), 1e-11)  # not quite that precise?
                if initialization_method != InitializationMethods.AUTO:
                    self.assertEqual(summary_list[0]["method"], initialization_method)

    def test_layers(self):
        n_qubits = 8
        state_vector = [np.random.rand(2 ** n_qubits) + np.random.rand(2 ** n_qubits) * 1j]
        state_vector = state_vector / np.linalg.norm(state_vector)

        qasm, summary = q_alchemy_as_qasm(
            state_vector=state_vector,
            opt_params=OptParams(
                initialization_method=InitializationMethods.ITERATIVE_TUCKER,
                extra_kwargs=dict(
                    fallback=False,
                    max_iterations=30,
                    barriers=True,
                    max_stepup=0,
                    factors_size=2
                )
            ),
            return_summary=True
        )
        qc = qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        qc.global_phase = summary['global_phase']

        layers = slice_by_barriers(qc)
        for i, layer in enumerate(layers):
            print(f"layer {i}: {layer.depth()}")

        # back to front
        print("back to front (abs(overlap), phase(overlap)):")
        layered_circuit = layers[-1].copy()
        layered_circuit.global_phase = qc.global_phase
        sv_layer = Statevector(layered_circuit)
        print(f"layer {len(layers) - 1}: {polar(np.vdot(state_vector, sv_layer))}")
        for first_layer in range(len(layers) - 2, -1, -1):
            # fidelity (i.e. abs(overlap)**2) increases with layers, but phase is unpredictable
            layered_circuit.compose(layers[first_layer], front=True, inplace=True)
            sv_layer = Statevector(layered_circuit)
            print(
                f"layers {first_layer}-{len(layers) - 1}: {polar(np.vdot(state_vector, sv_layer))}")

if __name__ == '__main__':
    unittest.main()
