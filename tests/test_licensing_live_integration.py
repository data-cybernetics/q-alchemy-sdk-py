"""Opt-in deployed licensing checks. Never enabled by an API key alone."""
from contextlib import contextmanager
import os

import pytest
from pinexq.client.core.polling import PollingException
from pinexq.client.job_management.model import JobStates

import q_alchemy.initialize as preparation
from q_alchemy import (
    Circuit, CircuitCompressionExecutionError, CircuitCompressionParams,
    CircuitCompressionRequest, CircuitCompressionService,
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
        for adapter in ADAPTERS + ('simulator', 'circuit_compressor')
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


def _circuit_workload(score, *, measured=False):
    """Cheap zero-state circuit; score counts input before cancellations."""
    from qiskit import QuantumCircuit
    block = QuantumCircuit(2)
    for _ in range(50):
        block.cx(0, 1)
    gate = block.to_gate()
    circuit = QuantumCircuit(2, 1 if measured else 0)
    score -= int(measured)
    for _ in range(score // 100):
        circuit.append(gate, [0, 1])
    for _ in range(score % 100):
        circuit.z(0)
    if measured:
        circuit.measure(0, 0)
    return circuit


@contextmanager
def _circuit_submission(adapter, circuit, configuration, monkeypatch, *, form=None, capability='counts'):
    connection, versions, _ = configuration
    version = versions[adapter]
    if adapter == 'simulator':
        import q_alchemy.simulator as simulator
        if version is not None:
            monkeypatch.setattr(simulator, 'find_processing_step',
                                lambda client, name: preparation.from_name(client, name, version=version))
        jobs = []
        configure = simulator.Job.create_and_configure_rapidly
        def record_job(self, *args, **kwargs):
            job = configure(self, *args, **kwargs)
            jobs.append(job)
            return job
        monkeypatch.setattr(simulator.Job, 'create_and_configure_rapidly', record_job)
        params = simulator.SimulatorParams(api_key=connection.api_key, host=connection.host,
                                           remove_data=False, tier='standard',
                                           job_completion_timeout_sec=300)
        with simulator.SparseSimulator(params) as service:
            try:
                yield lambda: getattr(service, capability)(circuit, input_form=form)
            finally:
                for job in jobs:
                    _delete_terminal_job(job)
        return
    params_type, service_type = {
        'circuit_compressor': (CircuitCompressionParams, CircuitCompressionService),
        'quantum_io': (QuantumIOParams, QuantumIOService),
        'feasibility': (FeasibilityParams, FeasibilityService),
    }[adapter]
    params = params_type(api_key=connection.api_key, host=connection.host,
                         step_version=version, remove_data=False, job_completion_timeout_sec=300)
    with service_type(params) as service:
        if adapter == 'circuit_compressor':
            handle = service.compress(circuit, CircuitCompressionRequest(options={'optimization_level': 3}))
        else:
            experiment = QuantumExperiment(
                target=State.sparse(num_qubits=2, indices=[0], amplitudes=[1.]),
                evolution=Circuit.from_qiskit(circuit))
            if adapter == 'quantum_io':
                handle = service.preflight(experiment, verify_preparation=False)
            else:
                request = FeasibilityRequest(
                    criteria=SolutionCriteria.common(min_final_state_fidelity=.999),
                    policy=FeasibilityPolicy(quantum_execution=QuantumExecutionPolicy.NEVER),
                    evidence_collection=EvidenceCollectionConfig(
                        circuit_compression=CircuitCompressionConfig(enabled=False)))
                handle = service.analyze(experiment, request)
        try:
            yield handle.result
        finally:
            if not handle.removed:
                _delete_terminal_job(handle.raw_job)


CIRCUIT_CASES = [('simulator', form, capability)
                 for form in ('qasm_string', 'qasm_file', 'qpy')
                 for capability in ('counts', 'sparse_statevector', 'tomography')]
CIRCUIT_CASES += [(adapter, None, None)
                  for adapter in ('circuit_compressor', 'quantum_io', 'feasibility')]


@pytest.mark.parametrize('adapter,form,capability', CIRCUIT_CASES)
@pytest.mark.parametrize('extra', [0, 1], ids=['exact-limit', 'one-over'])
def test_live_circuit_admission_success_and_failure(
        live_configuration, monkeypatch, adapter, form, capability, extra):
    # The account's authenticated plan selects the public contract fixture.
    state_limit = live_configuration[2][0]
    enterprise = state_limit == BOUNDARIES['enterprise'][0]
    limit = ((100_000 if enterprise else 10_000) if adapter == 'circuit_compressor'
             else (25_000 if enterprise else 2_500))
    circuit = _circuit_workload(limit + extra, measured=capability == 'counts')
    error_type = {'feasibility': FeasibilityExecutionError,
                  'circuit_compressor': CircuitCompressionExecutionError}.get(adapter, PollingException)
    with _circuit_submission(adapter, circuit, live_configuration, monkeypatch,
                             form=form, capability=capability) as result:
        if extra:
            with pytest.raises(error_type) as captured:
                result()
            message = str(captured.value)
            assert 'circuit complexity' in message
            assert str(limit) in message and str(limit + 1) in message
        else:
            report = result()
            complexity = (report.circuit_complexity if adapter == 'quantum_io'
                          else report.raw['circuit_complexity'])
            assert complexity['score'] == complexity['limit'] == limit
            assert complexity['policy'] == 'circuit-score-v1'


@pytest.mark.parametrize('max_nnz,expected_nnz', [(0, 8), (2, 2), (2_000_000, 8)])
def test_live_simulator_preserves_amplitude_cap(
        live_configuration, monkeypatch, max_nnz, expected_nnz):
    import q_alchemy.simulator as simulator
    from qiskit import QuantumCircuit
    connection, versions, _ = live_configuration
    if versions['simulator'] is not None:
        monkeypatch.setattr(simulator, 'find_processing_step',
                            lambda client, name: preparation.from_name(
                                client, name, version=versions['simulator']))
    circuit = QuantumCircuit(3)
    circuit.h(range(3))
    with simulator.SparseSimulator(api_key=connection.api_key, host=connection.host,
                                   tier='standard', remove_data=True) as service:
        result = service.sparse_statevector(circuit, max_nnz=max_nnz, input_form='qpy')
    metadata = result.raw['simulation_metadata']
    assert result.nnz == expected_nnz
    assert metadata['max_nnz'] == max_nnz
    assert metadata['max_nnz_mode'] == ('unlimited' if max_nnz == 0 else 'explicit')
    assert bool(metadata['amplitude_capping_occurred']) == (max_nnz == 2)
    assert result.raw['circuit_complexity']['score'] == 3
