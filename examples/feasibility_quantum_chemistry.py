"""Quantum-chemistry feasibility example with automatic quantum-target selection.

The example runs a small correlated 12-spin-orbital active-space workload through
Q-Alchemy Feasibility.  It deliberately uses ``QuantumExecutionPolicy.COMPARE``
so the configured quantum target is exercised even when the classical reference
is feasible; this makes the backend-selection behavior visible in one runnable
example.

Quantum target selection is controlled by environment variables:

* ``IBM_QUANTUM_TOKEN`` set: use a physical IBM Quantum backend.
* ``IBM_QUANTUM_BACKEND`` also set: use that exact IBM backend.
* no ``IBM_QUANTUM_BACKEND``: ask Feasibility for the least-busy suitable IBM backend.
* no ``IBM_QUANTUM_TOKEN``: use an Aer noisy simulator instead of a physical QPU.

``IBM_QUANTUM_INSTANCE`` is optional and is forwarded when present.  The
Q-Alchemy service credential (normally ``Q_ALCHEMY_API_KEY``) is still required
for the hosted Feasibility call.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

from q_alchemy import (
    Circuit,
    ClassicalResources,
    EvidenceCollectionConfig,
    FeasibilityExecutionError,
    FeasibilityPolicy,
    FeasibilityRequest,
    FeasibilityService,
    IBMQuantumCredentials,
    MeasurementPlan,
    PauliObservable,
    QuantumExecutionPolicy,
    QuantumExperiment,
    SolutionCriteria,
    State,
)


NUM_QUBITS = 12
NUM_ELECTRONS = 6
CLASSICAL_MEMORY_BYTES = 8 * 1024 * 1024
SPARSE_WORKING_BYTES_PER_AMPLITUDE = 96
SPARSE_MAX_NNZ = CLASSICAL_MEMORY_BYTES // SPARSE_WORKING_BYTES_PER_AMPLITUDE
SHOTS = 4096
AER_BACKEND = "aer_simulator"


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def pauli_z(qubit: int) -> str:
    """Return a Qiskit-order Pauli string measuring Z on one qubit."""

    label = ["I"] * NUM_QUBITS
    label[NUM_QUBITS - 1 - qubit] = "Z"
    return "".join(label)


def correlated_active_space_evolution() -> str:
    """Build one small Trotter-like layer for a correlated electronic model."""

    qasm = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{NUM_QUBITS}] q;",
        "",
        "// Particle-number-preserving orbital hopping.",
    ]

    for left in range(0, NUM_QUBITS, 2):
        right = left + 1
        qasm += [
            f"h q[{left}];",
            f"h q[{right}];",
            f"cx q[{left}], q[{right}];",
            f"rz(0.35) q[{right}];",
            f"cx q[{left}], q[{right}];",
            f"h q[{left}];",
            f"h q[{right}];",
            f"rx(pi / 2) q[{left}];",
            f"rx(pi / 2) q[{right}];",
            f"cx q[{left}], q[{right}];",
            f"rz(0.35) q[{right}];",
            f"cx q[{left}], q[{right}];",
            f"rx(-pi / 2) q[{left}];",
            f"rx(-pi / 2) q[{right}];",
        ]

    qasm += ["", "// Density-density interactions between neighboring orbital pairs."]
    for left in range(1, NUM_QUBITS - 1, 2):
        right = left + 1
        qasm += [
            f"cx q[{left}], q[{right}];",
            f"rz(0.12) q[{right}];",
            f"cx q[{left}], q[{right}];",
        ]

    return "\n".join(qasm)


def noisy_simulator_backend_options() -> dict[str, object]:
    """Build a portable Aer noise configuration for the hosted simulator."""

    # Lazy import keeps qiskit-aer an examples-only dependency when IBM hardware
    # is selected.  The NoiseModel object itself is not sent over the wire: its
    # dictionary representation is wrapped in Quantum I/O's portable envelope.
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.002, 1),
        ["u", "rz", "sx", "x"],
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(0.02, 2),
        ["cx"],
    )
    noise_model.add_all_qubit_readout_error(
        ReadoutError([[0.985, 0.015], [0.02, 0.98]])
    )
    return {
        "noise_model": {
            "format": "qiskit-aer-noise-model-v1",
            "data": noise_model.to_dict(),
        },
        "transpile_options": {"optimization_level": 1},
        # This experiment requests Pauli expectation values, so Quantum I/O
        # executes Qiskit's Estimator. Estimator seeds belong here rather than
        # in backend_run_options/run_options, which configure count execution.
        "estimator_options": {"seed_simulator": 20260904},
    }


def select_quantum_target(
    env: Mapping[str, str] = os.environ,
    *,
    simulator_backend_options: Mapping[str, object] | None = None,
) -> tuple[EvidenceCollectionConfig, IBMQuantumCredentials | None, str]:
    """Resolve IBM hardware vs. noisy Aer from the example environment."""

    ibm_token = _nonempty(env.get("IBM_QUANTUM_TOKEN"))
    ibm_backend = _nonempty(env.get("IBM_QUANTUM_BACKEND"))
    ibm_instance = _nonempty(env.get("IBM_QUANTUM_INSTANCE"))

    if ibm_token is not None:
        config = EvidenceCollectionConfig(
            provider="ibm",
            backend=ibm_backend,
            least_busy=ibm_backend is None,
            shots=SHOTS,
            sparse_config={
                "max_nnz": SPARSE_MAX_NNZ,
                "sparse_epsilon": 0.0,
                "final_sparse_epsilon": 0.0,
            },
        )
        credentials = IBMQuantumCredentials(
            token=ibm_token,
            instance=ibm_instance,
            channel="ibm_quantum_platform",
        )
        description = (
            f"IBM physical QPU: {ibm_backend}"
            if ibm_backend is not None
            else "IBM physical QPU: least-busy suitable backend"
        )
        return config, credentials, description

    backend_options = (
        dict(simulator_backend_options)
        if simulator_backend_options is not None
        else noisy_simulator_backend_options()
    )
    config = EvidenceCollectionConfig(
        provider="aer",
        backend=AER_BACKEND,
        backend_options=backend_options,
        shots=SHOTS,
        sparse_config={
            "max_nnz": SPARSE_MAX_NNZ,
            "sparse_epsilon": 0.0,
            "final_sparse_epsilon": 0.0,
        },
    )
    return config, None, "Aer noisy simulator: aer_simulator"


def build_experiment() -> QuantumExperiment:
    """Create the illustrative correlated active-space chemistry workload."""

    hartree_fock_index = sum(1 << qubit for qubit in range(0, NUM_QUBITS, 2))
    target_state = State.sparse(
        num_qubits=NUM_QUBITS,
        indices=[hartree_fock_index],
        amplitudes=[1.0],
        metadata={
            "model": "illustrative 12-spin-orbital correlated active space",
            "electrons": NUM_ELECTRONS,
        },
    )
    observable_qubits = tuple(range(NUM_QUBITS))
    measurement_plan = MeasurementPlan(
        observables=tuple(
            PauliObservable.pauli(f"occupation-{qubit}", pauli_z(qubit))
            for qubit in observable_qubits
        ),
        observable_plan_metadata={
            "example": "correlated-active-space-orbital-occupations",
            "observable_qubits": list(observable_qubits),
            "purpose": "direct-observable-diagnostics",
        },
    )
    return QuantumExperiment(
        target=target_state,
        evolution=Circuit.qasm3(correlated_active_space_evolution()),
        measurement_plan=measurement_plan,
        metadata={"name": "sdk-correlated-active-space-feasibility"},
    )


def build_request(evidence_collection: EvidenceCollectionConfig) -> FeasibilityRequest:
    """Create a request that exercises the selected quantum target."""

    return FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.15),
        policy=FeasibilityPolicy(
            # This is an example of target selection, so execute the selected
            # quantum target even when the 12-qubit classical reference succeeds.
            quantum_execution=QuantumExecutionPolicy.COMPARE,
        ),
        classical_resources=ClassicalResources(
            available_memory_bytes=CLASSICAL_MEMORY_BYTES,
            cpu_cores=2,
            metadata={"purpose": "reproducible SDK quantum-chemistry example"},
        ),
        evidence_collection=evidence_collection,
    )


def main() -> None:
    evidence_collection, credentials, target_description = select_quantum_target()
    print(f"Quantum target: {target_description}")

    try:
        report = FeasibilityService().analyze(
            build_experiment(),
            build_request(evidence_collection),
            credentials=credentials,
        ).result()
    except FeasibilityExecutionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    print()
    print(report.format_summary())

    quantum = report.quantum
    backend = quantum.get("backend")
    if backend:
        print(f"Selected backend: {backend}")
    quality = quantum.get("quality_status")
    if quality:
        print(f"Quantum quality: {quality}")

    if report.warnings:
        print("\nWARNINGS")
        print("--------")
        for warning in report.warnings:
            print(f"- {warning}")
if __name__ == "__main__":
    main()
