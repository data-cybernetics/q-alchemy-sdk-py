"""Run the simplest Quantum I/O workflow: preparation preflight only."""

from __future__ import annotations

from math import sqrt

from q_alchemy import QuantumExperiment, QuantumIOService, State


def main() -> None:
    amplitude = 1.0 / sqrt(2.0)
    experiment = QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        metadata={"example": "sdk-quantum-io-preflight"},
    )

    report = QuantumIOService().preflight(experiment).result()

    print("mode:", report.mode)
    print("qubits:", report.preparation.num_qubits)
    print("preparation method:", report.preparation.method)
    print("depth:", report.preparation.metrics.depth)
    print("CX gates:", report.preparation.metrics.cx_count)
    print("two-qubit gates:", report.preparation.metrics.two_qubit_count)

    preflight = report.preparation_preflight
    if preflight is not None:
        print("preparation simulation:", preflight.simulation_status)
        print("target-to-prepared fidelity:", preflight.target_to_prepared_fidelity)
        print(
            "preparation approximation infidelity:",
            preflight.preparation_approximation_infidelity,
        )
        if preflight.simulation is not None:
            print("simulator:", preflight.simulation.simulator)
            print("reference quality:", preflight.simulation.reference_quality)
            print("sparse nnz:", preflight.simulation.nnz)

    for warning in report.warnings:
        print("warning:", warning)


if __name__ == "__main__":
    main()
