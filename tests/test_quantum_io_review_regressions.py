import pytest
from q_alchemy import (
    State,
    QuantumExperiment,
    MeasurementPlan,
    PauliObservable,
    ExecutionPlan,
    Runtime,
)
from q_alchemy.quantum_io import (
    local_simulator_execution_plan,
    quantum_backend_execution_plan,
    noisy_backend_execution_plan,
    _quantum_backend_provider,
)
from q_alchemy.quantum_io_compression_contract import (
    CircuitCompressionMetrics,
    CircuitCompressionSummary,
)


@pytest.mark.parametrize("shots", [True, 3.9, 0, -1, float("nan")])
@pytest.mark.parametrize(
    "helper,kwargs",
    [
        (local_simulator_execution_plan, {}),
        (quantum_backend_execution_plan, {"provider": "ibm", "backend": "test"}),
        (noisy_backend_execution_plan, {"provider": "ibm", "backend": "test"}),
    ],
)
def test_helpers_do_not_coerce_invalid_shots(helper, kwargs, shots):
    with pytest.raises(ValueError):
        helper(shots=shots, **kwargs)


@pytest.mark.parametrize(
    "amplitudes", [[0, 0], [complex("nan"), 0], [complex("inf"), 0]]
)
def test_invalid_state_rejected_before_transport(amplitudes):
    with pytest.raises(ValueError):
        State.dense(amplitudes).to_dict()


@pytest.mark.parametrize("version", [True, 3.9, "3"])
def test_schema_version_must_be_integer(version):
    data = QuantumExperiment(State.dense([1, 0])).to_dict()
    data["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        QuantumExperiment.from_dict(data)


def test_held_out_operators_cannot_be_renamed_training_data():
    with pytest.raises(ValueError, match="distinct operators"):
        MeasurementPlan(
            training=(PauliObservable.pauli("train", "X"),),
            validation=(PauliObservable.pauli("test", "X"),),
        )


def test_aer_alias_does_not_require_provider_credentials():
    plan = ExecutionPlan(
        acquisition=Runtime.resource(
            "noisy-backend-simulator", provider="qiskit-aer", backend="aer_simulator"
        )
    )
    assert _quantum_backend_provider(plan) is None


def test_compression_baseline_is_additive_and_used_for_display():
    def metrics(n):
        return CircuitCompressionMetrics(n, 0, n, n, n, n)

    summary = CircuitCompressionSummary(
        True,
        True,
        input_metrics=metrics(4),
        baseline_metrics=metrics(7),
        compressed_metrics=metrics(5),
    )
    assert "2Q operations: 7 -> 5" in summary.format_summary()
    assert CircuitCompressionSummary.from_dict(summary.to_dict()) == summary
    legacy = summary.to_dict()
    legacy.pop("baseline_metrics")
    assert (
        "2Q operations: 4 -> 5"
        in CircuitCompressionSummary.from_dict(legacy).format_summary()
    )
