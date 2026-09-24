from __future__ import annotations

import importlib.util
from pathlib import Path

from q_alchemy import EvidenceCollectionConfig, FeasibilityRequest, SolutionCriteria


EXAMPLE = Path(__file__).parents[1] / "examples" / "feasibility_quantum_chemistry.py"
spec = importlib.util.spec_from_file_location("feasibility_quantum_chemistry_example", EXAMPLE)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


NOISY_OPTIONS = {
    "noise_model": {
        "format": "qiskit-aer-noise-model-v1",
        "data": {"errors": []},
    },
    "run_options": {"seed_simulator": 7},
}


def test_simulator_backend_options_round_trip():
    value = FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.1),
        evidence_collection=EvidenceCollectionConfig(
            provider="aer",
            backend="aer_simulator",
            backend_options=NOISY_OPTIONS,
        ),
    )

    payload = value.to_dict()
    assert payload["evidence_collection"]["backend_options"] == {"noise_model": NOISY_OPTIONS["noise_model"]}
    assert payload["evidence_collection"]["execution_options"] == {"transpile": True, "run_options": NOISY_OPTIONS["run_options"]}
    assert FeasibilityRequest.from_dict(payload) == value
    assert FeasibilityRequest.from_json(value.to_json()) == value


def test_explicit_ibm_backend_wins_when_token_is_present():
    config, credentials, description = example.select_quantum_target(
        {
            "IBM_QUANTUM_TOKEN": "secret",
            "IBM_QUANTUM_BACKEND": "ibm_test_backend",
            "IBM_QUANTUM_INSTANCE": "instance-test",
        },
        simulator_backend_options=NOISY_OPTIONS,
    )

    assert config.provider == "ibm"
    assert config.backend == "ibm_test_backend"
    assert config.least_busy is False
    assert credentials is not None
    assert credentials.instance == "instance-test"
    assert "ibm_test_backend" in description


def test_ibm_token_without_backend_requests_least_busy_device():
    config, credentials, description = example.select_quantum_target(
        {"IBM_QUANTUM_TOKEN": "secret"},
        simulator_backend_options=NOISY_OPTIONS,
    )

    assert config.provider == "ibm"
    assert config.backend is None
    assert config.least_busy is True
    assert credentials is not None
    assert "least-busy" in description


def test_missing_ibm_token_selects_noisy_aer_simulator():
    config, credentials, description = example.select_quantum_target(
        {"IBM_QUANTUM_BACKEND": "ignored-without-token"},
        simulator_backend_options=NOISY_OPTIONS,
    )

    assert config.provider == "aer"
    assert config.backend == "aer_simulator"
    assert config.least_busy is False
    assert config.backend_options == {"noise_model": NOISY_OPTIONS["noise_model"]}
    assert config.execution_options == {"transpile": True, "run_options": NOISY_OPTIONS["run_options"]}
    assert credentials is None
    assert "noisy simulator" in description


def test_example_keeps_diagnostic_observables_out_of_estimator_training():
    experiment = example.build_experiment()
    plan = experiment.measurement_plan

    assert len(plan.observables) == example.NUM_QUBITS
    assert plan.training == ()
    assert plan.validation == ()
    assert {item.label for item in plan.observables} == {
        f"occupation-{qubit}" for qubit in range(example.NUM_QUBITS)
    }
    assert plan.observable_plan_metadata["purpose"] == "direct-observable-diagnostics"
