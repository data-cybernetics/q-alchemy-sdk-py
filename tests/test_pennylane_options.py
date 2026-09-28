"""PennyLane operations get the options they were given.

Offline: building an operation does not call the API, and the batch call is faked.
"""
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, qasm2

from q_alchemy.pennylane_integration import (
    AmplitudeEmbedding,
    OptParams,
    QAlchemyStatePreparation,
    SOURCE_TAG,
    pennylane_batch_initialize,
)

STATE = np.array([1, 0, 0, 0], dtype=complex)
WIRES = [0, 1]


def opt_params_of(op) -> OptParams:
    return op.hyperparameters["opt_params"]


class OperationOptionsTestCase(unittest.TestCase):

    def test_a_misspelt_option_is_an_error(self):
        """It used to be dropped: max_fidelty_loss=0.1 ran as an exact preparation."""
        for operation in (AmplitudeEmbedding, QAlchemyStatePreparation):
            with self.subTest(operation=operation.__name__), self.assertRaises(TypeError) as caught:
                operation(STATE, wires=WIRES, max_fidelty_loss=0.1)
            self.assertIn("did you mean 'max_fidelity_loss'", str(caught.exception))

    def test_options_apply_with_or_without_opt_params(self):
        """With opt_params= given, every other keyword used to be ignored."""
        for operation in (AmplitudeEmbedding, QAlchemyStatePreparation):
            for base in ({}, {"opt_params": OptParams(use_qasm3=True)}):
                with self.subTest(operation=operation.__name__, base=base):
                    opt_params = opt_params_of(operation(STATE, wires=WIRES, max_fidelity_loss=0.1, **base))
                    self.assertEqual(opt_params.max_fidelity_loss, 0.1)
                    self.assertEqual(opt_params.use_qasm3, bool(base))

    def test_the_callers_opt_params_are_not_modified(self):
        """job_tags += [...] extended the caller's list, once per operation built from it."""
        mine = OptParams(job_tags=["mine"])
        ops = [QAlchemyStatePreparation(STATE, wires=WIRES, opt_params=mine) for _ in range(3)]
        self.assertEqual(mine.job_tags, ["mine"])
        for op in ops:
            self.assertEqual(opt_params_of(op).job_tags, ["mine", SOURCE_TAG])

    def test_embedding_decomposes_with_its_options(self):
        """The decomposition passes StatePrep's pad_with/normalize/validate_norm down too;
        those are not options, and must not reach the option check."""
        embedding = AmplitudeEmbedding(STATE, wires=WIRES, opt_params=OptParams(max_fidelity_loss=0.1))
        [preparation] = embedding.decomposition()
        self.assertIsInstance(preparation, QAlchemyStatePreparation)
        self.assertEqual(opt_params_of(preparation).max_fidelity_loss, 0.1)
        self.assertEqual(opt_params_of(preparation).job_tags.count(SOURCE_TAG), 1)

    def test_operations_survive_pennylane_rebuilding_them(self):
        # PennyLane rebuilds operators from their hyperparameters (e.g. in map_wires).
        for op in (AmplitudeEmbedding(STATE, wires=WIRES, max_fidelity_loss=0.1),
                   QAlchemyStatePreparation(STATE, wires=WIRES, max_fidelity_loss=0.1)):
            with self.subTest(operation=type(op).__name__):
                rebuilt = qml.map_wires(op, {0: 1, 1: 0})
                self.assertEqual(opt_params_of(rebuilt).max_fidelity_loss, 0.1)
                self.assertEqual(opt_params_of(rebuilt).job_tags, opt_params_of(op).job_tags)


class BatchOptionsTestCase(unittest.TestCase):

    def _submitted_opt_params(self, **hyperparameters) -> OptParams:
        qasm = qasm2.dumps(QuantumCircuit(2))
        with patch("q_alchemy.pennylane_integration.q_alchemy_as_qasm_parallel_states",
                   return_value=([qasm], [{"status": "OK", "global_phase": 0.0}])) as batch, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pennylane_batch_initialize([STATE], wires=WIRES, **hyperparameters)
        return batch.call_args.args[1]

    def test_default_basis_is_kept(self):
        self.assertEqual(self._submitted_opt_params().basis_gates, ["id", "rx", "ry", "rz", "cx"])

    def test_options_apply_with_or_without_opt_params(self):
        """Keywords other than opt_params used to be ignored."""
        self.assertEqual(self._submitted_opt_params(max_fidelity_loss=0.1).max_fidelity_loss, 0.1)
        submitted = self._submitted_opt_params(opt_params=OptParams(use_qasm3=False), max_fidelity_loss=0.1)
        self.assertEqual(submitted.max_fidelity_loss, 0.1)
        self.assertEqual(submitted.basis_gates, ["u", "cx"])  # an explicit OptParams keeps its own

    def test_a_misspelt_option_is_an_error(self):
        with self.assertRaises(TypeError):
            self._submitted_opt_params(max_fidelty_loss=0.1)


if __name__ == '__main__':
    unittest.main()
