"""Complete SDK example for backend-calibrated noisy Quantum I/O simulation.

The experiment is::

    Bell target -- Q-Alchemy state preparation P -- RZ(pi/3) -- measurements

The SDK discovers an accessible IBM Quantum backend and uses its topology and
calibration to construct a noisy Aer simulation inside the deployed Quantum I/O
service. No QPU job is submitted. In parallel, Q-Alchemy's sparse simulator
provides the ideal reference, and qTucker state estimation is exercised with
training observables plus a held-out validation observable.

This example also shows the portable JSON boundary used by Quantum I/O: the
typed experiment and execution plan are serialized and reconstructed before
submission. IBM credentials are *not* part of either JSON document; the SDK
uploads them separately as Secret PineXQ WorkData.

Required environment variables:

* ``Q_ALCHEMY_API_KEY`` (or ``PINEXQ_API_KEY``)
* ``IBM_QUANTUM_TOKEN``

A local ``.env`` file is loaded when present.
"""

from __future__ import annotations

from dataclasses import replace
from math import pi, sqrt
import os

from dotenv import load_dotenv
from qiskit import QuantumCircuit

from q_alchemy import (
    BasisMeasurement,
    Circuit,
    ExecutionPlan,
    ExperimentReport,
    IBMQuantumCredentials,
    MeasurementPlan,
    PauliObservable,
    QuantumExperiment,
    QuantumIOService,
    Runtime,
    State,
    noisy_backend_execution_plan,
)


def experiment() -> QuantumExperiment:
    """Return the portable WHAT-to-run description for the example."""

    # Make state preparation non-trivial: Q-Alchemy prepares the Bell target,
    # while the evolution is a separate phase rotation applied afterwards.
    evolution = QuantumCircuit(2)
    evolution.rz(pi / 3, 0)

    return QuantumExperiment(
        target=State.dense([1 / sqrt(2), 0.0, 0.0, 1 / sqrt(2)]),
        evolution=Circuit.from_qiskit(evolution),
        measurement_plan=MeasurementPlan(
            training=(
                PauliObservable.pauli("ZI", "ZI"),
                PauliObservable.pauli("IZ", "IZ"),
                PauliObservable.pauli("ZZ", "ZZ"),
            ),
            validation=(PauliObservable.pauli("XX", "XX"),),
            basis_measurements=(BasisMeasurement("q0-q1", (0, 1)),),
            observable_plan_metadata={"example": "fit-vs-held-out"},
        ),
        metadata={"name": "bell-state-preparation-with-phase-evolution"},
    )


def print_report(report: ExperimentReport) -> None:
    """Print the most useful fields from the typed experiment report."""

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
    load_dotenv()

    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError("Set IBM_QUANTUM_TOKEN before running this example")

    credentials = IBMQuantumCredentials(token=token)

    # The service reads Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY) from the
    # environment.  The context manager closes its HTTP connection pool.
    with QuantumIOService(ibm_credentials=credentials) as service:
        # Backend discovery reads device metadata only. No QPU execution occurs.
        backends = service.backends(provider="ibm", min_num_qubits=2)
        if not backends:
            raise RuntimeError(
                "IBM Quantum returned no accessible backend with >=2 qubits"
            )

        # Prefer the smallest accessible device model for this demonstration.
        # Aer can truncate inactive qubits, but a smaller backend keeps the
        # backend-derived topology/calibration model cheaper to simulate.
        backend = min(
            backends,
            key=lambda item: (item.num_qubits, item.pending_jobs or 0, item.name),
        )
        print(
            "using calibration from:",
            backend.name,
            f"({backend.num_qubits} qubits, pending_jobs={backend.pending_jobs})",
        )

        # HOW to run it. The IBM backend is used for topology/calibration only;
        # acquisition itself is a noisy Aer simulation in Quantum I/O.
        plan = noisy_backend_execution_plan(
            provider="ibm",
            backend=backend.name,
            shots=20_000,
            ideal_reference=True,
            estimator=True,
        )
        plan = replace(
            plan,
            reference=Runtime.qalchemy_sparse(
                source="ideal-reference",
                sparse_epsilon=0.0,
                final_sparse_epsilon=0.0,
            ),
            preparation_options={"max_fidelity_loss": 1e-6},
            metadata={
                "purpose": (
                    "backend-calibrated noisy simulation instead of QPU acquisition"
                )
            },
        )

        request = experiment()

        # Demonstrate the service boundary explicitly. These JSON documents are
        # portable and contain no IBM token or Q-Alchemy API key.
        print("\nPORTABLE EXPERIMENT JSON")
        print(request.to_json(indent=2))
        print("\nPORTABLE EXECUTION PLAN JSON")
        print(plan.to_json(indent=2))
        request = QuantumExperiment.from_json(request.to_json())
        plan = ExecutionPlan.from_json(plan.to_json())

        report = service.run(request, execution_plan=plan).result()

    print_report(report)
    print("\nPORTABLE TYPED REPORT JSON")
    print(report.to_json(indent=2))


if __name__ == "__main__":
    main()
