"""Run a typed Quantum I/O experiment on the Q-Alchemy ideal simulator."""

from __future__ import annotations

from math import sqrt

from q_alchemy import (
    BasisMeasurement,
    MeasurementPlan,
    QuantumExperiment,
    QuantumIOService,
    State,
)


def bell_state_experiment() -> QuantumExperiment:
    amplitude = 1.0 / sqrt(2.0)
    return QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        measurement_plan=MeasurementPlan(
            basis_measurements=(
                BasisMeasurement("q0-q1", (0, 1)),
            ),
        ),
        metadata={"example": "sdk-quantum-io-bell"},
    )


def main() -> None:
    service = QuantumIOService()
    report = service.run(bell_state_experiment(), shots=256).result()

    print("mode:", report.mode)
    print("preparation method:", report.preparation.method)
    print("preparation depth:", report.preparation.metrics.depth)

    if report.execution is not None:
        print("execution source:", report.execution.source)
        print("execution kind:", report.execution.source_kind)
        for distribution in report.execution.basis_distributions:
            print(distribution.label, dict(distribution.probabilities))

    for warning in report.warnings:
        print("warning:", warning)


if __name__ == "__main__":
    main()
