"""The simulator gets the options it was given.

Offline: a dummy client satisfies the API-key guard, and no job is run.
"""
import unittest
from unittest.mock import patch

from qiskit import QuantumCircuit

from q_alchemy.qalchemy_backend import QAlchemyBackend
from q_alchemy.simulator import SimulatorParams, SparseSimulator, simulate_counts

CLIENT = object()


class SimulatorOptionsTestCase(unittest.TestCase):

    def test_a_misspelt_option_is_an_error(self):
        """It used to be dropped, and the simulator ran with the default."""
        for params, kwargs in [(None, {"job_completion_timeout": 60}),
                               ({"job_completion_timeout": 60}, {})]:
            with self.subTest(params=params, kwargs=kwargs), self.assertRaises(TypeError) as caught:
                SparseSimulator(params, client=CLIENT, **kwargs)
            self.assertIn("did you mean 'job_completion_timeout_sec'", str(caught.exception))

    def test_options_apply(self):
        for params in (None, {"tier": "standard"}, SimulatorParams(tier="standard")):
            with self.subTest(params=params):
                simulator = SparseSimulator(params, client=CLIENT, job_completion_timeout_sec=60)
                self.assertEqual(simulator.params.job_completion_timeout_sec, 60)

    def test_the_callers_params_are_not_modified(self):
        mine = SimulatorParams()
        SparseSimulator(mine, client=CLIENT, job_completion_timeout_sec=60)
        self.assertEqual(mine.job_completion_timeout_sec, 300)

    def test_a_misspelt_run_option_is_an_error(self):
        """shot= fell through to the params, which ignored it, so the run used 1024 shots."""
        with patch.object(SparseSimulator, "counts", autospec=True) as counts, \
                self.assertRaises(TypeError) as caught:
            simulate_counts(QuantumCircuit(1), client=CLIENT, shot=100)
        self.assertIn("did you mean 'shots'", str(caught.exception))
        counts.assert_not_called()

    def test_run_and_params_options_are_both_routed(self):
        with patch.object(SparseSimulator, "counts", autospec=True) as counts, \
                patch("q_alchemy.simulator.SparseSimulator.__init__", return_value=None) as init:
            simulate_counts(QuantumCircuit(1), shots=100, tier="standard")
        self.assertEqual(counts.call_args.kwargs, {"shots": 100})
        self.assertEqual(init.call_args.kwargs, {"tier": "standard"})

    def test_backend_rejects_options_it_cannot_apply(self):
        """With a ready SparseSimulator as params, the options were dropped without a word."""
        with self.assertRaises(TypeError):
            QAlchemyBackend(params=SparseSimulator(client=CLIENT), tier="standard")
        with self.assertRaises(TypeError):
            QAlchemyBackend(params={"tier": "standard"}, client=CLIENT, tierr="standard")


if __name__ == '__main__':
    unittest.main()
