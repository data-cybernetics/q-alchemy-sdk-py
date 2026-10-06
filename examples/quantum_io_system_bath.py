"""Prepare a system plus a zero-state bath, evolve both, measure the system.

Run with Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY) set in the environment:
    python examples/quantum_io_system_bath.py

Requires the SDK's Qiskit extra. The experiment runs through PineXQ using the
ideal sparse simulator and compresses the full preparation + evolution circuit.
It does not submit a QPU job or reconstruct a state.
"""

from math import sqrt

import numpy as np
from qiskit import QuantumCircuit

from q_alchemy import (
    Circuit,
    ExecutionPlan,
    MeasurementPlan,
    PauliObservable,
    QuantumExperiment,
    QuantumIOService,
    Runtime,
    State,
)

from q_alchemy.quantum_io import CircuitCompressionConfig


def build_experiment() -> QuantumExperiment:
    # A is the system (qubits 0, 1); B is the bath (qubits 2, 3).
    n_a, n_b = 2, 2
    state_a = np.array([sqrt(0.3), 0, 0, sqrt(0.7)], dtype=complex)
    assert state_a.size == 2**n_a and np.isclose(np.vdot(state_a, state_a).real, 1)

    # Bath B starts in |00...0>.
    state_b = np.zeros(2**n_b, dtype=complex)
    state_b[0] = 1.0

    initial_state = State.dense(
        np.kron(state_b, state_a),
        num_qubits=n_a + n_b,
    )

    # Replace these gates with your colleague's evolution on A + B.
    evolution = QuantumCircuit(n_a + n_b)
    evolution.rxx(0.6, 0, 2)  # Couple A[0] to B[0].
    evolution.rxx(0.6, 1, 3)  # Couple A[1] to B[1].
    evolution.rzz(0.4, 0, 1)  # Interaction within A.

    # One observable on A: Z on qubit 0. Pauli strings read right to left:
    # "IZ" acts on A; prefixing identities gives "IIIZ" on the full system.
    observable_a = "IZ"
    observable = PauliObservable.pauli("Z on A[0]", "I" * n_b + observable_a)

    return QuantumExperiment(
        target=initial_state,
        evolution=Circuit.from_qiskit(evolution),
        measurement_plan=MeasurementPlan(observables=(observable,)),
    )


def main() -> None:
    plan = ExecutionPlan(
        preparation_options={"max_fidelity_loss": 0.0, "basis_gates": ["u", "cx"]},
        circuit_compression=CircuitCompressionConfig(enabled=True),
        acquisition=Runtime.qalchemy_sparse(),
    )
    with QuantumIOService() as service:
        report = service.run(build_experiment(), plan).result(timeout=300)

    if report.execution is None or report.execution.observations is None:
        raise RuntimeError("The service returned no observable result")

    observation = report.execution.observations.observations[0]
    print(f"<{observation.label}> = {observation.value:.6f}")

    if report.circuit_compression is not None:
        print()
        print(report.circuit_compression.format_summary())


if __name__ == "__main__":
    main()
