"""Verify that compression uses the complete chemistry workload, including loading."""

import importlib.util
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector


def test_circuit_matches_chemistry_target_followed_by_evolution(monkeypatch):
    examples = Path(__file__).parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    path = examples / "circuit_compression_quantum_chemistry.py"
    spec = importlib.util.spec_from_file_location("chemistry_compression_example", path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    experiment = example.build_experiment()
    circuit = example.build_circuit()
    assert circuit.num_qubits == 12
    assert len(circuit.data) == 105
    assert circuit.count_ops()["cx"] == 34
    assert sum(len(instruction.qubits) == 1 for instruction in circuit.data) == 71

    # Start from the declared target, independently of the X-gate preparation.
    initial = np.zeros(1 << experiment.target.num_qubits, dtype=complex)
    initial[list(experiment.target.indices)] = experiment.target.amplitudes
    expected = Statevector(initial).evolve(
        experiment.evolution.to_qiskit(), qargs=experiment.evolution_qargs,
    )
    np.testing.assert_allclose(Statevector(circuit).data, expected.data, atol=1e-12, rtol=0)
