from pathlib import Path
import os
import unittest
from cmath import polar
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit import qasm2, qasm3
from qiskit.quantum_info import Statevector

from qiskit_addon_utils.slicing import slice_by_barriers
from pinexq.client.core.polling import PollingException
from pinexq.client.job_management import Job, ProcessingStep
from q_alchemy.initialize import OptParams, q_alchemy_as_qasm_parallel_states, InitializationMethods, q_alchemy_as_qasm, \
    create_client, from_name

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env") # the 'assert' was causing the import to fail during test discovery.

@unittest.skipUnless(os.getenv("Q_ALCHEMY_API_KEY"), "no Q_ALCHEMY_API_KEY: skipping live state preparation tests")
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

    def test_batch_leaves_no_job_or_upload_behind(self):
        # A batch always uploads its states, whatever the qubit count. With remove_data
        # the job and that upload must both be gone afterwards -- also when the job
        # fails, here on an option AUTO rejects.
        rng = np.random.default_rng()
        states = [rng.normal(size=16) + 1j * rng.normal(size=16) for _ in range(3)]
        states = [s / np.linalg.norm(s) for s in states]

        for failing in (False, True):
            with self.subTest(failing=failing):
                seen = {}
                original = Job.delete_with_associated

                def record(job, **kwargs):
                    job.refresh()
                    seen["urls"] = [job.self_link().get_url()] + [
                        wd.self_link.get_url()
                        for slot in job.job_hco.input_dataslots for wd in slot.selected_workdatas
                    ]
                    return original(job, **kwargs)

                opt_params = OptParams(extra_kwargs={"max_iterations": 3} if failing else {})
                with patch.object(Job, "delete_with_associated", record):
                    if failing:
                        with self.assertRaises(PollingException):
                            q_alchemy_as_qasm_parallel_states(states, opt_params=opt_params)
                    else:
                        q_alchemy_as_qasm_parallel_states(states, opt_params=opt_params)

                self.assertEqual(len(seen["urls"]), 2)  # the job and its one upload
                client = create_client(opt_params)
                for url in seen["urls"]:
                    self.assertEqual(client.get(str(url)).status_code, 404, url)

class FromNameTestCase(unittest.TestCase):
    # Offline: the server query is faked, so no API key is needed.
    def _pick(self, versions, version=None):
        steps = [SimpleNamespace(version=v) for v in versions]
        with patch.object(ProcessingStep, "_query_processing_steps",
                          return_value=SimpleNamespace(processing_steps=steps)),                 patch.object(ProcessingStep, "from_hco", side_effect=lambda hco: hco):
            return from_name(client=SimpleNamespace(base_url="https://example.invalid"),
                             step_name="some_step", version=version)

    def test_picks_newest_version_whatever_the_server_order(self):
        cases = [
            (["0.9.0", "0.10.0", "0.2.0"], "0.10.0"),  # as strings, 0.9.0 would win
            (["0.10.0", "0.9.9", "0.10.1", "0.9.10"], "0.10.1"),
            (["0.10.2", "1.0.0", "0.9.5"], "1.0.0"),
        ]
        for versions, newest in cases:
            with self.subTest(versions=versions):
                self.assertEqual(self._pick(versions).version, newest)

    def test_skips_dev_and_pre_release_versions(self):
        versions = ["0.10.0", "0.11.0.dev3", "0.11.0-dev", "0.11.0-dev.abc123", "0.11.0rc1", "dev", "0.9.0"]
        self.assertEqual(self._pick(versions).version, "0.10.0")

    def test_no_released_version_is_an_error(self):
        with self.assertRaises(NameError):
            self._pick(["0.11.0.dev3", "latest"])

    def test_explicit_version_is_taken_as_is(self):
        self.assertEqual(self._pick(["0.11.0.dev3"], version="0.11.0.dev3").version, "0.11.0.dev3")

if __name__ == '__main__':
    unittest.main()
