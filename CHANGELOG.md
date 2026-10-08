# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each release on GitHub has fuller notes.

## [Unreleased]

Adds three hosted services (Quantum I/O, feasibility analysis and circuit compression)
and moves state preparation to the current Q-Alchemy service. **Contains breaking
changes**; see *Removed* and *Changed*. SDK 0.3.1 keeps working against the current
state-preparation service, so you can upgrade when convenient.

### Added

- `QuantumIOService`. Runs a `QuantumExperiment` (prepare a state, evolve it, measure)
  on an ideal simulator, a noisy backend simulator or IBM quantum hardware, and returns
  a typed `ExperimentReport`. Also includes a preflight check, compact sparse
  reconstruction, optional compression of the experiment circuit, and
  `include_quantum_circuits=True` to return the logical circuit as
  `report.quantum_circuit`. Hardware runs need `IBM_QUANTUM_TOKEN`; without it, noisy
  runs use Aer.
- `FeasibilityService`. Sends a `QuantumExperiment` and a `FeasibilityRequest` and
  returns a `FeasibilityReport`: whether the workload is better run classically or on a
  quantum target, with the evidence and a recommendation.
- `CircuitCompressionService`. Compresses a Qiskit circuit, an SDK `Circuit` or a
  `quantum-circuit` envelope. Returns the compressed circuit with metrics, the regions
  that were accepted, and a statement of what "exact" means for the result.
- `step_version` on each new service, to pin a deployed step version.
- A `visualization` extra, which installs `q-alchemy-visualization`.
- Recoverable initialization failures. If waiting, downloading or validating fails in a
  way that can be retried, the job and its data are kept. The exception carries
  `initialization_job` (and `initialization_job_url`), so you can retry with `run_job`.

### Changed

- **Breaking:** under `AUTO`, `basis_gates` on `OptParams` now also decides how
  candidate circuits are compared, not only which gate set the returned circuit uses.
  Don't also set it in `extra_kwargs`.
- **Breaking:** the default `AUTO` cost function is now `"two_qubit_then_depth"` (it was
  `"cx_then_depth"`). Pass `extra_kwargs={"cost_function": "cx_then_depth"}` to keep the
  old ranking.
- **Breaking:** higher minimum versions: `numpy>=2.0`, `pydantic>=2.13.4`,
  `qiskit>=2.3`, `pennylane>=0.45.1`, `pennylane-qiskit>=0.45.0`, and
  `pinexq-client>=2.1,<3` (PineXQ JobManagement API 10).
- States are sent inline when the serialized payload is at most 1 MiB, and uploaded
  otherwise. The cut-off used to be 16 qubits. Sparse states are sent sparse.
- Initialization results and compression reports are checked before cleanup. A cleanup
  error no longer discards a successful result.
- API keys are hidden from the `repr()` of parameter objects.
- Step lookups are cached per client instead of process-wide.

### Removed

- **Breaking:** `InitializationMethods.SWAP_PIVOT` and
  `InitializationMethods.BAA_LOW_RANK`.
- **Breaking:** the `OptParams` fields `isometry_scheme` and `unitary_scheme`. They were
  never sent to the service.
- The filter that silenced pinexq-client's version-mismatch warning, and the
  `Q_ALCHEMY_API_VERSION_WARNING` variable that turned the warning back on.

### Fixed

- `from_name` picks the newest released step version. It used to discard its own sort
  and take whichever version the API listed first (#62).

## [0.3.1] - 2026-09-23

### Added

- README section *Advanced options*: the `OptParams` fields, and the `extra_kwargs` keys
  each initialization method accepts (#58).

### Fixed

- With `remove_data=True` (the default), jobs, their outputs and their uploads are
  deleted, including for batch jobs, research functions, simulator QPY uploads and
  failed jobs (#59). A cleanup error no longer hides the reason a job failed.
- Docstrings no longer document options that are never sent.

## [0.3.0] - 2026-08-03

### Added

- `SparseSimulator`, a client for the hosted sparse state-vector simulator: `counts()`,
  `sparse_statevector()` and `tomography()`, taking a `QuantumCircuit` or an
  OpenQASM 2 string.
- `QAlchemyBackend` (a Qiskit `BackendV2`) and `QAlchemyProvider`. The backend also works
  as a PennyLane device through `qml.device("qiskit.remote", backend=...)`.
- Simulator tiers. `tier="standard"` is always available. `tier="enterprise"` without
  the plan raises `PermissionError` before any job is created.

### Changed

- pinexq-client's version-mismatch warning is silenced. Set
  `Q_ALCHEMY_API_VERSION_WARNING=1` to see it again.
- `numpy` and `scipy` are declared as runtime dependencies.

### Fixed

- A processing step that isn't registered now raises a clear error naming the endpoint,
  instead of an `AttributeError`.

## [0.2.27] - 2026-08-02

### Changed

- First release published through GitHub Actions with PyPI trusted publishing.
- Python 3.11 to 3.14 supported.
- Integration tests check the requested fidelity budget instead of near-exact output
  (#53).
- Reworked example notebooks (#50). The README uses `uv` for setup.

[Unreleased]: https://github.com/data-cybernetics/q-alchemy-sdk-py/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/data-cybernetics/q-alchemy-sdk-py/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/data-cybernetics/q-alchemy-sdk-py/compare/v0.2.27...v0.3.0
[0.2.27]: https://github.com/data-cybernetics/q-alchemy-sdk-py/releases/tag/v0.2.27
