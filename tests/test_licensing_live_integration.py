"""Opt-in deployed licensing checks. Never enabled by an API key alone."""
from contextlib import contextmanager
import os

import pytest
from pinexq.client.core.polling import PollingException
from pinexq.client.job_management.model import JobStates

import q_alchemy.initialize as preparation
from q_alchemy import (
    CircuitCompressionConfig, EvidenceCollectionConfig, FeasibilityExecutionError,
    FeasibilityParams, FeasibilityPolicy, FeasibilityRequest, FeasibilityService,
    QuantumExecutionPolicy, QuantumExperiment, QuantumIOParams, QuantumIOService,
    SolutionCriteria, State,
)


pytestmark = pytest.mark.skipif(
    os.getenv("QALCHEMY_RUN_LIVE_LICENSING") != "1",
    reason="real jobs require QALCHEMY_RUN_LIVE_LICENSING=1",
)
ADAPTERS = ("preparation", "quantum_io", "feasibility")
# Public commercial contract fixtures, not SDK-side admission rules.
BOUNDARIES = {"free": (49_152, 16, 3072),
              "enterprise": (20_971_520, 5120, 4096)}


def _sparse_state(n, k):
    return State.sparse(num_qubits=n, indices=list(range(k)),
                        amplitudes=[k**-.5] * k)


@pytest.fixture(scope="module")
def live_configuration():
    # Resolve every requirement before creating any jobs. Keep credentials out
    # of parametrization IDs and assertion messages.
    options = preparation.OptParams()
    if not options.api_key:
        pytest.fail("Live licensing tests require Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY)")
    versions = {
        adapter: os.getenv(f"QALCHEMY_LICENSING_{adapter.upper()}_VERSION", "").strip() or None
        for adapter in ADAPTERS
    }
    with preparation.create_client(options) as client:
        response = client.get("/api/EntryPoint")
        response.raise_for_status()
        info_link = next(link["href"] for link in response.json()["links"]
                         if "Info" in (link.get("rel") or []))
        response = client.get(info_link)
        response.raise_for_status()
        current_user = next(entity for entity in response.json()["entities"]
                            if "CurrentUser" in (entity.get("rel") or []))
        grants = current_user["properties"]["UserGrants"]
    if "plan:enterprise" in grants:
        plan = "enterprise"
    elif "plan:free" in grants:
        plan = "free"
    else:
        pytest.fail("The API key must belong to a Free or Enterprise subscription")
    return options, versions, BOUNDARIES[plan]


def _delete_terminal_job(job):
    # A timeout is not cancellation. Preserve a running job for diagnosis.
    if job.get_state() in (JobStates.completed, JobStates.error, JobStates.canceled):
        preparation.delete_job_with_data(job)


@contextmanager
def _submission(adapter, state, configuration, monkeypatch):
    connection, versions, _ = configuration
    version = versions[adapter]
    if adapter == "preparation":
        # Keep the SDK's latest-release lookup unless an override is supplied.
        # Preparation has no public step_version option, so pin in this test.
        if version is not None:
            monkeypatch.setattr(preparation, "find_processing_step",
                                lambda client, processing_name: preparation.from_name(
                                    client, processing_name, version=version))
        jobs = []
        configure = preparation.configure_job

        def record_job(**kwargs):
            job = configure(**kwargs)
            jobs.append(job)
            return job

        monkeypatch.setattr(preparation, "configure_job", record_job)
        options = preparation.OptParams(
            api_key=connection.api_key, host=connection.host, remove_data=False,
            initialization_method=preparation.InitializationMethods.ITERATIVE_TUCKER,
            job_completion_timeout_sec=180,
        )
        with preparation.create_client(options) as client:
            try:
                yield lambda: preparation.q_alchemy_as_qasm(
                    state, options, client=client, return_summary=True)
            finally:
                for job in jobs:
                    _delete_terminal_job(job)
    else:
        params_type, service_type = {
            "quantum_io": (QuantumIOParams, QuantumIOService),
            "feasibility": (FeasibilityParams, FeasibilityService),
        }[adapter]
        params = params_type(api_key=connection.api_key, host=connection.host, step_version=version,
                             remove_data=True, job_completion_timeout_sec=180)
        with service_type(params) as service:
            experiment = QuantumExperiment(target=state)
            if adapter == "quantum_io":
                handle = service.preflight(experiment, verify_preparation=False)
            else:
                request = FeasibilityRequest(
                    criteria=SolutionCriteria.common(min_final_state_fidelity=.999),
                    policy=FeasibilityPolicy(quantum_execution=QuantumExecutionPolicy.NEVER),
                    evidence_collection=EvidenceCollectionConfig(
                        circuit_compression=CircuitCompressionConfig(enabled=False)),
                )
                handle = service.analyze(experiment, request)
            try:
                yield handle.result
            finally:
                if not handle.removed:
                    _delete_terminal_job(handle.raw_job)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_live_wide_basis_state_is_admitted(live_configuration, adapter, monkeypatch):
    # Cheap product state: validates removal of width-only admission without
    # synthesizing a large state at the Enterprise commercial ceiling.
    state = State.sparse(num_qubits=72, indices=[0], amplitudes=[1.])
    with _submission(adapter, state, live_configuration, monkeypatch) as result:
        report = result()
    if adapter == "preparation":
        qasm, summary = report
        assert summary["status"].startswith("OK") and qasm.strip()
    elif adapter == "quantum_io":
        assert report.mode == "preflight" and report.execution is None
    else:
        assert report.final is True


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_live_over_limit_reports_subscription_failure(
        live_configuration, adapter, monkeypatch):
    # Rejection must precede packing, synthesis, or simulation. The unusually
    # wide Enterprise fixture keeps the uploaded payload small.
    _, _, (limit, n, k) = live_configuration
    error_type = FeasibilityExecutionError if adapter == "feasibility" else PollingException
    with _submission(adapter, _sparse_state(n, k + 1), live_configuration, monkeypatch) as result:
        with pytest.raises(error_type) as captured:
            result()
    message = str(captured.value)
    assert "state complexity" in message
    assert str(limit) in message and str(n * (k + 1)) in message
