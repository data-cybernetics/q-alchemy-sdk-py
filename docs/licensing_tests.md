# Live state-complexity licensing tests

The commercial score is `logical_num_qubits * canonical_nnz`, with inclusive
limits of **49,152** for Free and **20,971,520** for Enterprise. Exact zeros do
not count. These tests cover Preparation, Quantum I/O and Feasibility when an
explicit target state is supplied. Circuit licensing for Simulator, Compressor
and evolution circuits is outside this suite.

The SDK tests licensing through deployed service calls using its own public
contracts. No adapter repositories, core packages or second Python environment
are required. Commercial admission remains the responsibility of the
authenticated adapters.

## Configuration

`tests/test_licensing_live_integration.py` submits real jobs only when
`QALCHEMY_RUN_LIVE_LICENSING=1`. An API key alone never enables this module.
Use the SDK's existing credentials. Version overrides are optional:

- `Q_ALCHEMY_API_KEY`: your account's API key; `PINEXQ_API_KEY` is the usual fallback.
- `QALCHEMY_LICENSING_PREPARATION_VERSION`: optional preparation step version override.
- `QALCHEMY_LICENSING_QUANTUM_IO_VERSION`: optional Quantum I/O step version override.
- `QALCHEMY_LICENSING_FEASIBILITY_VERSION`: optional Feasibility step version override.
- `Q_ALCHEMY_HOST`: optional host; defaults to `jobs.api.q-alchemy.com`.

By default, each service uses the SDK's normal lookup for the latest released
ProCon version visible to your account. Set an override to test a specific
deployment; unset or empty version settings use the default. Preparation's
lookup is pinned only when its override is supplied, because its public options
have no version field.

Before submitting jobs, the suite reads the account's grants through the
authenticated platform Info endpoint and selects the matching Free or Enterprise
allowance. Enterprise takes precedence when both grants exist. An account
without either grant fails setup.

```bash
QALCHEMY_RUN_LIVE_LICENSING=1 python -m pytest -q tests/test_licensing_live_integration.py
```

The six deployment cases exercise each adapter under the supplied account: successful
admission of a cheap 72-qubit basis state, and rejection above the account's
allowance with the calculated score in the error. Quantum I/O uses preflight
without simulator verification. Feasibility disables compression and quantum
execution. No hardware execution is requested.

To verify both subscription plans, run the suite once with a Free account's key
and once with an Enterprise account's key, using `Q_ALCHEMY_API_KEY` each time.

The Free rejection fixture uses 16 qubits and 3,073 nonzeros (score 49,168).
The Enterprise rejection fixture uses 5,120 qubits and 4,097 nonzeros (score
20,976,640), keeping the upload small. These over-limit requests must be rejected
before packing, synthesis or simulation; the Enterprise fixture is not a claim
that computation supports that width. The suite does not execute large states
at the admitted commercial ceilings or replace the adapters' own boundary tests.

Completed and failed test jobs and their associated data are cleaned up. A job
still running after a timeout is preserved for diagnosis. A technical failure
does not count as a successful licensing rejection.
