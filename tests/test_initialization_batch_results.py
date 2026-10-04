"""SDK response-shape regressions using mocked transport, no private runtime."""
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import q_alchemy.initialize as initialize
from q_alchemy.qiskit_integration import qiskit_batch_initialize


@pytest.fixture
def single_response(monkeypatch):
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1];'
    summary = {"status": "OK", "global_phase": .37, "fidelity_loss": .2,
               "fidelity_requirement_met": False, "fidelity_loss_source": "initializer_estimate",
               "selection_reason": "best effort: lowest reported fidelity loss"}
    monkeypatch.setattr(initialize, "upload_statevector", Mock(return_value=object()))
    monkeypatch.setattr(initialize, "configure_job", Mock(return_value=object()))
    monkeypatch.setattr(initialize, "create_client", Mock(return_value=Mock()))
    monkeypatch.setattr(initialize, "run_job", Mock(return_value=(summary, qasm)))
    return qasm, summary


@pytest.mark.parametrize("return_summary", [False, True])
def test_singleton_batch_returns_lists_and_retains_provenance(single_response, return_summary):
    qasm, summary = single_response
    result = initialize.q_alchemy_as_qasm_parallel_states(
        [np.array([1, 0])], initialize.OptParams(), client=object(), return_summary=return_summary)
    expected = ([qasm], [summary]) if return_summary else [qasm]
    assert result == expected


def test_singleton_batch_is_composable_through_qiskit_wrapper(single_response):
    state = np.array([np.exp(.37j), 0])
    gates = qiskit_batch_initialize([state])
    assert len(gates) == 1
    circuit = QuantumCircuit(1)
    circuit.append(gates[0], [0])
    np.testing.assert_allclose(Statevector(circuit).data, state, atol=1e-12)


def test_multiple_results_remain_lists(single_response, monkeypatch):
    qasm, summary = single_response
    monkeypatch.setattr(initialize, "run_job", Mock(return_value=([summary, summary], [qasm, qasm])))
    result = initialize.q_alchemy_as_qasm_parallel_states(
        [np.array([1, 0]), np.array([1, 0])], initialize.OptParams(),
        client=object(), return_summary=True)
    assert result == ([qasm, qasm], [summary, summary])
