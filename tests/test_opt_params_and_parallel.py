"""Options reach the job as given, and parallel results stay with their inputs.

Offline: the jobs are faked, so no API key is needed.
"""
import time
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qasm2

import q_alchemy.initialize as initialize
from q_alchemy.initialize import OptParams, populate_opt_params, q_alchemy_as_qasm_parallel
from q_alchemy.qiskit_integration import qiskit_batch_initialize


class PopulateOptParamsTestCase(unittest.TestCase):

    def test_a_misspelt_option_is_an_error(self):
        """It used to be dropped: max_fidelty_loss=0.1 ran as an exact preparation."""
        with self.assertRaises(TypeError) as caught:
            populate_opt_params(None, max_fidelty_loss=0.1)
        self.assertIn("max_fidelty_loss", str(caught.exception))
        self.assertIn("did you mean 'max_fidelity_loss'", str(caught.exception))

    def test_every_unknown_option_is_named(self):
        with self.assertRaises(TypeError) as caught:
            populate_opt_params(OptParams(), zzz=1, basis_gate=["u"])
        self.assertIn("'zzz'", str(caught.exception))
        self.assertIn("did you mean 'basis_gates'", str(caught.exception))

    def test_overrides_apply(self):
        for base in (None, {"use_qasm3": True}, OptParams(use_qasm3=True)):
            with self.subTest(base=base):
                opt_params = populate_opt_params(base, max_fidelity_loss=0.2)
                self.assertEqual(opt_params.max_fidelity_loss, 0.2)
        self.assertTrue(populate_opt_params({"use_qasm3": True}, max_fidelity_loss=0.2).use_qasm3)

    def test_the_callers_opt_params_are_not_modified(self):
        """Overrides used to be set on the caller's object, so they leaked into later calls reusing it."""
        mine = OptParams()
        populate_opt_params(mine, max_fidelity_loss=0.2)
        self.assertEqual(mine.max_fidelity_loss, 0.0)

    def test_qiskit_batch_passes_only_real_options(self):
        """qiskit_batch_initialize passed num_qubits=, which is not an option and was dropped.
        Now that unknown options raise, every keyword an integration passes must be real."""
        def fake_batch(state_vector, opt_params, client=None, return_summary=False, **kwargs):
            populate_opt_params(opt_params, **kwargs)  # raises on anything unknown
            qasm = qasm2.dumps(QuantumCircuit(1))
            return [qasm] * 2, [{"status": "OK", "global_phase": 0.0}] * 2

        with patch("q_alchemy.qiskit_integration.q_alchemy_as_qasm_parallel_states", side_effect=fake_batch):
            qiskit_batch_initialize([np.array([1, 0], dtype=complex)] * 2)


class ParallelTestCase(unittest.TestCase):
    OPT_PARAMS = [{"max_fidelity_loss": loss} for loss in (0.3, 0.1, 0.2)]

    def test_results_follow_the_order_of_opt_params(self):
        """Results were appended as jobs finished, so the fastest came first."""
        def job(state_vector, opt, client, return_summary):
            time.sleep(opt["max_fidelity_loss"])  # the first job is the slowest
            return opt["max_fidelity_loss"]

        with patch.object(initialize, "q_alchemy_as_qasm", job):
            self.assertEqual(q_alchemy_as_qasm_parallel([1, 0], self.OPT_PARAMS), [0.3, 0.1, 0.2])

    def test_a_failed_job_is_raised_and_names_its_entry(self):
        """A failed job used to only print a thread traceback, leaving the list short."""
        def job(state_vector, opt, client, return_summary):
            if opt["max_fidelity_loss"] != 0.3:
                raise RuntimeError(f"job with {opt['max_fidelity_loss']} failed")
            return "ok"

        with patch.object(initialize, "q_alchemy_as_qasm", job), \
                self.assertRaises(RuntimeError) as caught:
            q_alchemy_as_qasm_parallel([1, 0], self.OPT_PARAMS)
        self.assertEqual(str(caught.exception), "job with 0.1 failed")  # the earliest failed entry
        self.assertIn("opt_params[1]; 2 of 3 jobs failed", caught.exception.__notes__[0])


if __name__ == '__main__':
    unittest.main()
