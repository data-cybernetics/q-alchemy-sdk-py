"""A batch where some states fail still returns the rest.

Offline: the server's batch output is faked, so no API key is needed. The server
writes null as the circuit of a state it could not prepare, with the reason in
that state's summary; these tests pin down that the SDK turns that into None in
every batch API, rather than handing the placeholder to a QASM parser.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import Gate

from q_alchemy.initialize import OptParams, extract_result
from q_alchemy.pennylane_integration import pennylane_batch_initialize
from q_alchemy.qiskit_integration import qiskit_batch_initialize

FAILED = "Could not build the initialization circuit: failed to diagonalize M2"


def _circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.ry(np.pi / 3, 0)
    return qc


def _ok(global_phase: float = 0.0) -> dict:
    return {"status": "OK", "global_phase": global_phase}


def _failed() -> dict:
    # What the server's ResultSummary(status=str(ex)) serialises to.
    return {"status": FAILED, "global_phase": 0.0}


def _batch_job(summaries: list[dict], circuits: list) -> SimpleNamespace:
    """A finished batch job as extract_result sees it: no inline result, two JSON files."""
    def workdata(name: str, content) -> SimpleNamespace:
        data = json.dumps(content).encode("utf-8")
        return SimpleNamespace(name=name, size_in_bytes=len(data),
                               download_link=SimpleNamespace(download=lambda: data))

    slot = SimpleNamespace(assigned_workdatas=[
        workdata("summaries.json", summaries),
        workdata("qasm_circuit.qasm", circuits),
    ])
    job = SimpleNamespace(get_output_data_slots=lambda: [slot])
    job.refresh = lambda: SimpleNamespace(get_result=lambda: None)
    return job


class ExtractBatchResultTestCase(unittest.TestCase):

    def test_failed_states_come_back_as_none(self):
        qasm = qasm2.dumps(_circuit())
        summaries, circuits = extract_result(
            _batch_job([_ok(), _failed(), _ok()], [qasm, None, qasm]))
        self.assertEqual(circuits, [qasm, None, qasm])
        self.assertEqual(summaries[1]["status"], FAILED)  # the reason is kept

    def test_empty_string_placeholder_is_also_none(self):
        # "" is what the server's batch generator yields for a failed state.
        qasm = qasm2.dumps(_circuit())
        _, circuits = extract_result(_batch_job([_ok(), _failed()], [qasm, ""]))
        self.assertEqual(circuits, [qasm, None])

    def test_a_failed_status_wins_over_a_circuit(self):
        # Never hand back a circuit whose own summary says it failed.
        qasm = qasm2.dumps(_circuit())
        _, circuits = extract_result(_batch_job([_failed()], [qasm]))
        self.assertEqual(circuits, [None])

    def test_the_failure_is_logged_with_its_index(self):
        with self.assertLogs("q_alchemy.initialize", level="WARNING") as logs:
            extract_result(_batch_job([_ok(), _failed()], [qasm2.dumps(_circuit()), None]))
        self.assertIn("state 1", logs.output[0])
        self.assertIn(FAILED, logs.output[0])

    def test_misaligned_lists_are_an_error(self):
        # Pairing summaries with the wrong circuits would be worse than failing.
        with self.assertRaises(ValueError):
            extract_result(_batch_job([_ok(), _ok()], [qasm2.dumps(_circuit())]))


class BatchIntegrationsPassFailuresThroughTestCase(unittest.TestCase):
    """Each integration parses the circuits it gets; a None must not reach the parser."""

    def _fake_batch(self, module: str, qasm: str):
        return patch(f"q_alchemy.{module}.q_alchemy_as_qasm_parallel_states",
                     return_value=([qasm, None, qasm], [_ok(0.5), _failed(), _ok(0.5)]))

    def test_qiskit_batch_keeps_the_other_gates(self):
        states = [np.array([1, 0], dtype=complex)] * 3
        for use_qasm3, qasm in [(False, qasm2.dumps(_circuit())), (True, qasm3.dumps(_circuit()))]:
            with self.subTest(use_qasm3=use_qasm3), self._fake_batch("qiskit_integration", qasm):
                gates = qiskit_batch_initialize(states, opt_params=OptParams(use_qasm3=use_qasm3))
                self.assertIsNone(gates[1])
                for gate, label in [(gates[0], "QAl0"), (gates[2], "QAl2")]:
                    self.assertIsInstance(gate, Gate)
                    self.assertEqual(gate.label, label)  # labels stay with their states

    def test_pennylane_batch_keeps_the_other_circuits(self):
        states = [np.array([1, 0], dtype=complex)] * 3
        for use_qasm3, qasm in [(False, qasm2.dumps(_circuit())), (True, qasm3.dumps(_circuit()))]:
            with self.subTest(use_qasm3=use_qasm3), self._fake_batch("pennylane_integration", qasm):
                circuits = pennylane_batch_initialize(
                    states, wires=[0], opt_params=OptParams(use_qasm3=use_qasm3))
                self.assertIsNone(circuits[1])
                self.assertTrue(callable(circuits[0]) and callable(circuits[2]))


if __name__ == '__main__':
    unittest.main()
