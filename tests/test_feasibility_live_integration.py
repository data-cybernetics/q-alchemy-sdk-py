"""Opt-in live SDK -> PineXQ -> feasibility integration checks.

These tests submit real PineXQ jobs.  They are intentionally disabled during
ordinary test runs and require both an API key and
``Q_ALCHEMY_RUN_LIVE_FEASIBILITY=1``.
"""

from __future__ import annotations

import os
from math import sqrt

import pytest

from q_alchemy import (
    CircuitCompressionConfig,
    ClassicalResources,
    EvidenceCollectionConfig,
    FeasibilityParams,
    FeasibilityPolicy,
    FeasibilityRequest,
    FeasibilityService,
    MeasurementPlan,
    PauliObservable,
    QuantumExecutionPolicy,
    QuantumExperiment,
    SolutionCriteria,
    State,
)


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_API_KEY_PRESENT = bool(
    os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY")
)
_LIVE_ENABLED = _enabled(os.getenv("Q_ALCHEMY_RUN_LIVE_FEASIBILITY"))

pytestmark = pytest.mark.skipif(
    not (_API_KEY_PRESENT and _LIVE_ENABLED),
    reason=(
        "live feasibility integration requires Q_ALCHEMY_API_KEY/PINEXQ_API_KEY "
        "and Q_ALCHEMY_RUN_LIVE_FEASIBILITY=1"
    ),
)


def _bell_experiment(name: str) -> QuantumExperiment:
    amplitude = 1.0 / sqrt(2.0)
    return QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        metadata={"name": name, "integration": "sdk-pinexq-feasibility"},
    )


def _request(*, compression_enabled: bool | None = None) -> FeasibilityRequest:
    if compression_enabled is None:
        # Exercise the SDK/core default rather than spelling it out here.
        evidence = EvidenceCollectionConfig()
    else:
        evidence = EvidenceCollectionConfig(
            circuit_compression=CircuitCompressionConfig(
                enabled=compression_enabled,
            )
        )
    return FeasibilityRequest(
        criteria=SolutionCriteria.common(min_final_state_fidelity=0.999),
        evidence_collection=evidence,
    )


def _service() -> FeasibilityService:
    version = os.getenv("Q_ALCHEMY_FEASIBILITY_STEP_VERSION") or None
    return FeasibilityService(FeasibilityParams(step_version=version))


def _experiment_summary(report) -> dict:
    trace = report.execution_trace or {}
    value = trace.get("experiment_summary")
    assert isinstance(value, dict)
    return value


def test_live_feasibility_preserves_explicit_compression_disabled():
    """The typed SDK setting must survive WorkData/PineXQ transport unchanged."""

    with _service() as service:
        report = service.analyze(
            _bell_experiment("sdk-live-compression-disabled"),
            _request(compression_enabled=False),
        ).result()

    assert report.final is True
    assert report.recommendation.compute.value == "classical"

    # Disabled compression has no compression session/summary in the core report.
    assert report.circuit_compression is None
    experiment_summary = _experiment_summary(report)
    assert experiment_summary["circuit_compression_requested"] is False
    assert experiment_summary["circuit_compression_applied"] is False
    assert "CIRCUIT COMPRESSION" not in report.format_summary()


def test_live_feasibility_uses_default_compression():
    """The service default must run the real compressor through the deployed stack."""

    with _service() as service:
        report = service.analyze(
            _bell_experiment("sdk-live-compression-default"),
            _request(),
        ).result()

    assert report.final is True
    assert report.recommendation.compute.value == "classical"

    compression = report.circuit_compression
    assert compression is not None
    assert compression["attempted"] is True
    assert compression["applied"] is True
    assert compression["exact"] is True
    assert compression["equivalence"] == "reachable_subspace"
    assert compression["input_semantics"] == "canonical_zero_state"

    experiment_summary = _experiment_summary(report)
    assert experiment_summary["circuit_compression_requested"] is True
    assert experiment_summary["circuit_compression_applied"] is True
    assert "CIRCUIT COMPRESSION" in report.format_summary()


def _attempted_requests(report) -> set[str]:
    trace = report.execution_trace or {}
    values = trace.get("attempted_requests", ())
    assert isinstance(values, (list, tuple))
    return {str(value) for value in values}


def _bell_observable_experiment(name: str) -> QuantumExperiment:
    amplitude = 1.0 / sqrt(2.0)
    return QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        measurement_plan=MeasurementPlan(
            observables=(PauliObservable.pauli("bell-zz", "ZZ"),),
        ),
        metadata={"name": name, "integration": "sdk-pinexq-feasibility"},
    )


def _ghz4_experiment(name: str) -> QuantumExperiment:
    amplitude = 1.0 / sqrt(2.0)
    return QuantumExperiment(
        target=State.dense([amplitude] + [0.0] * 14 + [amplitude]),
        metadata={"name": name, "integration": "sdk-pinexq-feasibility"},
    )


def test_live_when_needed_short_circuits_estimator_only_criteria():
    """Estimator-only criteria must not force quantum execution under WHEN_NEEDED."""

    request = FeasibilityRequest(
        criteria=SolutionCriteria.common(
            min_reference_to_estimated_fidelity=0.90,
        ),
        policy=FeasibilityPolicy(
            quantum_execution=QuantumExecutionPolicy.WHEN_NEEDED,
        ),
        evidence_collection=EvidenceCollectionConfig(
            provider="aer",
            backend="aer_simulator",
            circuit_compression=CircuitCompressionConfig(enabled=False),
        ),
    )

    with _service() as service:
        report = service.analyze(
            _bell_experiment("sdk-live-when-needed-estimator-short-circuit"),
            request,
        ).result()

    assert report.final is True
    assert report.classical["status"] == "feasible"
    assert report.quantum["status"] == "not-needed"
    assert report.quantum["assessment_basis"] == "classical-short-circuit"
    assert report.recommendation.compute.value == "classical"

    trace = report.execution_trace or {}
    assert trace.get("policy", {}).get("quantum_execution") == "when-needed"
    attempted = _attempted_requests(report)
    assert "classical-execution" in attempted
    assert "noisy-backend-simulation" not in attempted
    assert "qpu-execution" not in attempted


def test_live_compare_runs_aer_after_classical_success():
    """COMPARE must retain the classical result and still execute the quantum path."""

    request = FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.15),
        policy=FeasibilityPolicy(
            quantum_execution=QuantumExecutionPolicy.COMPARE,
        ),
        evidence_collection=EvidenceCollectionConfig(
            provider="aer",
            backend="aer_simulator",
            shots=1024,
            circuit_compression=CircuitCompressionConfig(enabled=False),
        ),
    )

    with _service() as service:
        report = service.analyze(
            _bell_observable_experiment("sdk-live-compare-aer"),
            request,
        ).result()

    assert report.final is True
    assert report.classical["status"] == "feasible"
    assert report.quantum["status"] == "feasible"
    assert report.quantum["assessment_basis"] == "simulated-quantum"
    # COMPARE guarantees that the configured quantum target is executed even
    # after classical success. It does not guarantee that the resulting
    # quality evidence is conclusive: for example, a metric compared against
    # a qualified/approximate reference is intentionally reported as
    # ``inconclusive`` by feasibility.
    assert report.recommendation.compute.value == "classical"

    trace = report.execution_trace or {}
    assert trace.get("policy", {}).get("quantum_execution") == "compare"
    attempted = _attempted_requests(report)
    assert "classical-execution" in attempted
    assert "noisy-backend-simulation" in attempted
    assert "qpu-execution" not in attempted


def test_live_qtucker_observable_config_reaches_state_estimation():
    """The SDK observable-generation config must reach the deployed QTucker path."""

    request = FeasibilityRequest(
        # Held-out RMSE forces estimator validation. The threshold is deliberately
        # permissive: this test validates transport/orchestration, not estimator
        # convergence quality.
        criteria=SolutionCriteria.common(max_held_out_rmse=2.0),
        policy=FeasibilityPolicy(
            quantum_execution=QuantumExecutionPolicy.WHEN_NEEDED,
        ),
        classical_resources=ClassicalResources(available_memory_bytes=1),
        evidence_collection=EvidenceCollectionConfig(
            provider="aer",
            backend="aer_simulator",
            shots=512,
            # GHZ has two nonzero amplitudes. max_nnz=1 makes the sparse
            # classical reference resource-limited; the one-byte classical
            # budget also excludes the dense fallback, so WHEN_NEEDED must
            # escalate to the simulator-backed quantum path.
            sparse_config={
                "max_nnz": 1,
                "sparse_epsilon": 0.0,
                "final_sparse_epsilon": 0.0,
            },
            qtucker_config={
                "n_qubits": 4,
                "blocks": [[0, 1], [2, 3]],
                "ranks": [2, 2],
                "n_restarts": 1,
                "fit_config": {
                    "epochs": 25,
                    "learning_rate": 0.05,
                    "verbose": False,
                    "seed": 7,
                },
                "materialize_statevector_max_qubits": 4,
            },
            qtucker_observable_config={
                "within_block_mode": "all_paulis",
                "cross_block_mode": None,
                "extra_edges": [[0, 2], [1, 3]],
                "paulis": "XYZ",
                "validation_edges": [[0, 3], [1, 2]],
                "validation_paulis": "XYZ",
            },
            circuit_compression=CircuitCompressionConfig(enabled=False),
        ),
    )

    with _service() as service:
        report = service.analyze(
            _ghz4_experiment("sdk-live-qtucker-observable-config"),
            request,
        ).result()

    assert report.final is True
    assert report.classical["status"] == "infeasible"
    assert report.classical["infeasibility_kind"] == "resource"
    assert report.computation_result is not None
    assert report.computation_result["compute"] == "quantum"
    assert report.computation_result["evidence_request"] == "noisy-backend-simulation"

    experiment_report = report.computation_experiment_report
    assert experiment_report is not None
    observable_plan = experiment_report.experiment.measurement_plan.observable_plan
    assert observable_plan is not None

    metadata = observable_plan.metadata
    assert metadata["training_generator"].endswith(
        "make_default_reconstruction_observables"
    )
    assert metadata["validation_generator"] == "explicit-validation-edges"
    assert metadata["generated_training_observables"] == len(observable_plan.training)
    assert metadata["generated_validation_observables"] == len(
        observable_plan.validation
    )
    assert metadata["generated_validation_observables"] == 18

    attempted = _attempted_requests(report)
    assert "classical-execution" in attempted
    assert "noisy-backend-simulation" in attempted
