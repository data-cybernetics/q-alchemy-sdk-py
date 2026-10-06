from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from q_alchemy import ExecutionPlan, noisy_backend_execution_plan


@pytest.mark.parametrize("options", [
    {"transpile": True, "transpile_options": {"optimization_level": 1}},
    {"transpile": False},
    {},
])
def test_noisy_helper_preserves_execution_options_through_json(options):
    plan = noisy_backend_execution_plan(
        provider="ibm", backend="ibm_test", execution_options=options,
    )
    restored = ExecutionPlan.from_json(plan.to_json())
    assert restored.acquisition.config["execution_options"] == options
    assert restored.acquisition.config["provider"] == "ibm"
    assert restored.acquisition.config["backend"] == "ibm_test"
    assert plan.acquisition.config["execution_options"] is not options


@pytest.mark.parametrize("options", [True, [], "transpile"])
def test_noisy_helper_rejects_non_mapping_execution_options(options):
    with pytest.raises(ValueError, match="execution_options must be a mapping"):
        noisy_backend_execution_plan(provider="ibm", backend="ibm_test", execution_options=options)


def test_noisy_example_requests_device_compilation_before_submission(monkeypatch, capsys):
    """Execute main through its real JSON roundtrip, intercepting the network boundary."""
    path = Path(__file__).parents[1] / "examples" / "quantum_io_noisy_simulation.py"
    spec = importlib.util.spec_from_file_location("quantum_io_noisy_simulation_example", path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    monkeypatch.setattr(example, "load_dotenv", lambda: None)
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "test-token")
    submitted = []
    report = SimpleNamespace(to_json=lambda **kwargs: '{}')

    class Service:
        def __init__(self, **kwargs):
            assert kwargs["ibm_credentials"].token == "test-token"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def backends(self, **kwargs):
            return [SimpleNamespace(name="ibm_test", num_qubits=120, pending_jobs=2)]

        def run(self, request, *, execution_plan):
            submitted.append((request, execution_plan))
            return SimpleNamespace(result=lambda: report)

    monkeypatch.setattr(example, "QuantumIOService", Service)
    monkeypatch.setattr(example, "print_report", lambda value: None)
    example.main()
    assert len(submitted) == 1
    request, plan = submitted[0]
    assert request.target.num_qubits == 2
    assert request.measurement_plan.training and request.measurement_plan.validation
    assert request.measurement_plan.basis_measurements
    assert plan.acquisition.config["execution_options"] == {"transpile": True}
    assert plan.acquisition.config["backend"] == "ibm_test"
    assert plan.estimator.kind == "qtucker"
    assert plan.reference.kind == "q-alchemy-sparse"
    output = capsys.readouterr().out
    assert '"transpile": true' in output
    assert "test-token" not in output
