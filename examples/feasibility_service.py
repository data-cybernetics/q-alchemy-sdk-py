"""Run a hosted feasibility assessment and inspect its experiment diagram.

The example is intentionally small and classical-first, so IBM Quantum
credentials are not required when the service can establish that the classical
path satisfies the requested fidelity criterion.

Environment:
    Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY)

The example explicitly requests the logical quantum circuit. Optional diagram
rendering requires the SDK ``visualization`` extra.
"""

from __future__ import annotations

from math import sqrt

from q_alchemy import (
    FeasibilityRequest,
    FeasibilityService,
    QuantumExperiment,
    SolutionCriteria,
    State,
)


def bell_state_experiment() -> QuantumExperiment:
    amplitude = 1.0 / sqrt(2.0)
    return QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        metadata={"example": "sdk-feasibility-bell"},
    )


def main() -> None:
    request = FeasibilityRequest(
        criteria=SolutionCriteria.common(min_final_state_fidelity=0.999),
    )

    with FeasibilityService() as service:
        report = service.analyze(
            bell_state_experiment(),
            request,
            include_quantum_circuits=True,
        ).result()

    print(report.format_summary())

    computation = report.computation_experiment_report
    if computation is not None:
        print("\nQUANTUM I/O REPORT")
        print("------------------")
        print(computation.format_summary())

    try:
        circuit = report.quantum_circuit
    except (RuntimeError, ValueError) as exc:
        print(f"\nQuantum circuit is unavailable: {exc}")
    else:
        if circuit is not None:
            print("\nQUANTUM CIRCUIT")
            print("---------------")
            print(circuit.draw(output="text"))

    try:
        diagram = report.experiment_diagram
    except RuntimeError as exc:
        print(f"\nExperiment diagram is unavailable: {exc}")
    else:
        if diagram is not None:
            print("\nEXPERIMENT DIAGRAM")
            print("------------------")
            print(diagram.draw(output="text"))


if __name__ == "__main__":
    main()
