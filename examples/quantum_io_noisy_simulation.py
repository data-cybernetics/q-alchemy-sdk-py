"""Exercise Quantum I/O end to end without submitting a QPU job.

The example discovers an IBM backend, uses its topology/calibration to construct
an AerSimulator on PineXQ, compares the noisy result with an ideal Q-Alchemy
sparse reference, and exercises observable fitting/held-out verification.
"""

from __future__ import annotations

import os

from qiskit import QuantumCircuit

from q_alchemy import (
    BasisMeasurement,
    Circuit,
    ExperimentReport,
    IBMQuantumCredentials,
    MeasurementPlan,
    PauliObservable,
    QuantumExperiment,
    QuantumIOService,
    State,
    noisy_backend_execution_plan,
)


def experiment() -> QuantumExperiment:
    evolution = QuantumCircuit(2)
    evolution.h(0)
    evolution.cx(0, 1)

    return QuantumExperiment(
        target=State.dense([1.0, 0.0, 0.0, 0.0]),
        evolution=Circuit.from_qiskit(evolution),
        measurement_plan=MeasurementPlan(
            training=(
                PauliObservable.pauli("ZI", "ZI"),
                PauliObservable.pauli("IZ", "IZ"),
            ),
            validation=(
                PauliObservable.pauli("XX", "XX"),
                PauliObservable.pauli("ZZ", "ZZ"),
            ),
            basis_measurements=(
                BasisMeasurement("computational", (0, 1)),
            ),
            observable_plan_metadata={"example": "fit-vs-held-out"},
        ),
        metadata={"example": "sdk-quantum-io-noisy-simulation"},
    )


def print_report(report: ExperimentReport) -> None:
    print("\nEXPERIMENT")
    print("generated at:", report.generated_at)
    print("mode:", report.mode)
    print("target qubits:", report.experiment.target_num_qubits)
    print("evolution present:", report.experiment.evolution_present)
    print("experiment metadata:", dict(report.experiment.metadata))
    plan_summary = report.experiment.measurement_plan
    if plan_summary.observable_plan is not None:
        print("training observables:", plan_summary.observable_plan.training)
        print("validation observables:", plan_summary.observable_plan.validation)
    print(
        "basis measurements:",
        tuple(item.label for item in plan_summary.basis_measurements),
    )

    print("\nPREPARATION")
    print("method:", report.preparation.method)
    print("claimed fidelity loss:", report.preparation.claimed_fidelity_loss)
    print("depth:", report.preparation.metrics.depth)
    print("CX gates:", report.preparation.metrics.cx_count)
    print("two-qubit gates:", report.preparation.metrics.two_qubit_count)
    print("operation counts:", dict(report.preparation.metrics.operation_counts))
    print("preparation metadata:", dict(report.preparation.metadata))

    if report.preparation_preflight is not None:
        preflight = report.preparation_preflight
        print("simulation status:", preflight.simulation_status)
        print("target-to-prepared fidelity:", preflight.target_to_prepared_fidelity)
        if preflight.simulation is not None:
            print("preparation simulator:", preflight.simulation.simulator)
            print("preparation quality:", preflight.simulation.reference_quality)
            print("preparation exact:", preflight.simulation.exact)
            print("preparation nnz:", preflight.simulation.nnz)
            print("preparation simulation metadata:", dict(preflight.simulation.metadata))
        for warning in preflight.warnings:
            print("preflight warning:", warning)

    print("\nCOMPLETE CIRCUIT")
    print("depth:", report.experiment_circuit.metrics.depth)
    print("CX gates:", report.experiment_circuit.metrics.cx_count)
    print("two-qubit gates:", report.experiment_circuit.metrics.two_qubit_count)
    print(
        "operation counts:",
        dict(report.experiment_circuit.metrics.operation_counts),
    )
    print("circuit metadata:", dict(report.experiment_circuit.metadata))

    print("\nIDEAL REFERENCE")
    if report.reference is None:
        print("not requested")
    else:
        print("status:", report.reference.status)
        if report.reference.simulation is not None:
            print("simulator:", report.reference.simulation.simulator)
            print("quality:", report.reference.simulation.reference_quality)
            print("exact:", report.reference.simulation.exact)
            print("nnz:", report.reference.simulation.nnz)
        if report.reference.measurements is not None:
            reference_measurements = report.reference.measurements
            print("measurement source:", reference_measurements.source)
            print("measurement source kind:", reference_measurements.source_kind)
            if reference_measurements.observations is not None:
                for observation in reference_measurements.observations.observations:
                    print("ideal observable:", observation.label, observation.value)
            for distribution in reference_measurements.basis_distributions:
                print(
                    "ideal distribution:",
                    distribution.label,
                    dict(distribution.probabilities),
                )
        for warning in report.reference.warnings:
            print("reference warning:", warning)

    print("\nNOISY EXECUTION")
    if report.execution is None:
        print("not available")
    else:
        print("source:", report.execution.source)
        print("source kind:", report.execution.source_kind)
        if report.execution.observations is not None:
            for observation in report.execution.observations.observations:
                print(
                    "observable:",
                    observation.label,
                    "value=",
                    observation.value,
                    "stderr=",
                    observation.stderr,
                    "shots=",
                    observation.shots,
                )
        for distribution in report.execution.basis_distributions:
            print(
                "distribution:",
                distribution.label,
                dict(distribution.probabilities),
            )
        print("execution metadata:", dict(report.execution.metadata))

    print("\nIDEAL VS NOISY")
    if report.observable_error is not None:
        print("observable count:", report.observable_error.count)
        print("observable RMSE:", report.observable_error.rmse)
        print("observable mean abs error:", report.observable_error.mean_absolute_error)
        print("observable max abs error:", report.observable_error.max_absolute_error)
        print("observable normalized RMSE:", report.observable_error.normalized_rmse)
    for label, metrics in report.distribution_errors.items():
        print(label, "TVD:", metrics.total_variation_distance)
        print(label, "Hellinger distance:", metrics.hellinger_distance)
        print(label, "classical fidelity:", metrics.classical_fidelity)

    print("\nSTATE ESTIMATION")
    if report.estimate is None:
        print("no estimate produced")
    else:
        print("estimator:", report.estimate.estimator)
        print("statevector materialized:", report.estimate.statevector_materialized)
        print("metadata:", dict(report.estimate.metadata))
    if report.held_out_verification_error is not None:
        print("held-out RMSE:", report.held_out_verification_error.rmse)
        print(
            "held-out max abs error:",
            report.held_out_verification_error.max_absolute_error,
        )

    if report.warnings:
        print("\nWARNINGS")
        for warning in report.warnings:
            print("-", warning)


def main() -> None:
    credentials = IBMQuantumCredentials(token=os.environ["IBM_QUANTUM_TOKEN"])
    service = QuantumIOService(ibm_credentials=credentials)

    # Backend discovery reads device metadata only. No QPU execution occurs.
    backends = service.backends(provider="ibm", min_num_qubits=2)
    if not backends:
        raise RuntimeError("IBM Quantum returned no accessible backend with >=2 qubits")

    # Prefer the smallest accessible device model for this demonstration.
    # Aer can truncate inactive qubits, but a smaller backend still keeps the
    # calibration/noise model easier and cheaper to simulate.
    backend = min(
        backends,
        key=lambda item: (item.num_qubits, item.pending_jobs or 0, item.name),
    )
    print("using calibration from:", backend.name)

    plan = noisy_backend_execution_plan(
        provider="ibm",
        backend=backend.name,
        shots=4096,
        ideal_reference=True,
        estimator=True,
    )
    report = service.run(experiment(), execution_plan=plan).result()
    print_report(report)


if __name__ == "__main__":
    main()
