"""Compress the full 12-qubit workload from feasibility_quantum_chemistry.py.

Reuses that example's target and evolution: prepare its Hartree--Fock basis state
with six X gates, then apply the correlated active-space evolution. The input has
71 one-qubit gates and 34 CX gates. It starts from |0...0>, so the compressor's
default reachable-subspace semantics apply to the complete circuit.

Install q-alchemy-sdk-py[qiskit] and export Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY).
Only circuit compression runs remotely; no Feasibility assessment or QPU job is
submitted. The example does not simulate the 12-qubit circuit locally.
"""

from __future__ import annotations

from cmath import phase
from math import isclose
import sys

from qiskit import QuantumCircuit

from q_alchemy import (
    CircuitCompressionExecutionError,
    CircuitCompressionRequest,
    CircuitCompressionService,
)

if __package__:
    from .feasibility_quantum_chemistry import build_experiment
else:
    from feasibility_quantum_chemistry import build_experiment


def build_circuit() -> QuantumCircuit:
    """Build preparation + evolution using the existing chemistry definition.

    The target is a single determinant, so basis-state preparation is exactly X
    on each occupied orbital. Fail explicitly if the source example changes to a
    superposition rather than silently preparing a different target.
    """
    experiment = build_experiment()
    target = experiment.target
    if target.indices is None or len(target.indices) != 1:
        raise ValueError("This example requires the chemistry target to be a single basis state")
    if not isclose(abs(target.amplitudes[0]), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The chemistry target must be normalized")
    if experiment.evolution is None:
        raise ValueError("The chemistry example must provide an evolution circuit")

    circuit = QuantumCircuit(target.num_qubits, name="correlated_active_space")
    circuit.global_phase = phase(target.amplitudes[0])
    for qubit in range(target.num_qubits):
        if target.indices[0] & (1 << qubit):
            circuit.x(qubit)
    circuit.compose(
        experiment.evolution.to_qiskit(),
        qubits=experiment.evolution_qargs,
        inplace=True,
    )
    return circuit


def main() -> None:
    circuit = build_circuit()
    print(f"Compressing the {circuit.num_qubits}-qubit chemistry preparation + evolution circuit")
    request = CircuitCompressionRequest(options={"collect_report": True})
    try:
        with CircuitCompressionService() as service:
            report = service.compress(circuit, request).result()
    except CircuitCompressionExecutionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    print(report.format_summary())
    original = report.metrics["input"]["two_qubit_operations"]
    compressed = report.metrics["compressed"]["two_qubit_operations"]
    if original:
        print(f"\n2Q-gate reduction relative to input: {100 * (original - compressed) / original:.1f}%")


if __name__ == "__main__":
    main()
