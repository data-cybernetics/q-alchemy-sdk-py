# Live workload licensing tests

The commercial score is `logical_num_qubits * canonical_nnz`, with inclusive
limits of **49,152** for Free and **20,971,520** for Enterprise. Exact zeros do
not count. These tests cover Preparation, Quantum I/O and Feasibility when an
explicit target state is supplied. Circuit checks also cover all three Simulator
capabilities and input forms, standalone Compressor, and supplied evolution in
Quantum I/O and Feasibility.

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
- `QALCHEMY_LICENSING_SIMULATOR_VERSION`: optional Simulator step version override.
- `QALCHEMY_LICENSING_CIRCUIT_COMPRESSOR_VERSION`: optional Compressor step version override.
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

The six state deployment cases exercise each state-oriented adapter under the supplied account: successful
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


## Circuit cases

| Adapter / supplied circuit | Free | Enterprise |
|---|---:|---:|
| Simulator, Quantum I/O evolution, Feasibility evolution | 2,500 | 25,000 |
| Circuit Compressor | 10,000 | 100,000 |

Native gates weigh their arity (1/2/3). The fixed catalog is `u`, `p`, `rx`,
`ry`, `rz`, `x`, `y`, `z`, `h`, `cx`, `cy`, `cz`, `rzz`, `swap`, `ccx`, `cswap`.
Other atomic gates weigh `4 ** (arity - 1)`. Explicit custom bodies count recursively
on every invocation. Measurement/reset weigh one; barriers/delays/global phase
weigh zero. Known native open controls add two X gates per open control.
For example, 1,000 H + 500 CX + 100 CCX scores 2,300.

Count the submitted wire representation before optimization. RZZ is native in
QPY (score 2); QASM export can replace it with CX-RZ-CX (score 5). Static gate
bodies and broadcasts are countable. Expand control flow, aliases/slices and
powers before submission. Controlled custom QASM gates require an explicit
controlled body, so the importer cannot synthesize hidden work before admission.
QPY must contain exactly one bound static circuit.

There are 24 circuit cases: exact-limit success and one-over rejection for each
of nine Simulator capability/transport combinations and three other adapters.
Simulator uses the standard preset, including for Enterprise accounts, exercising
plan-based allowances independently of resource-tier selection. The adapter's
own tests also cover XLarge authorization. Fixtures use cancelling gates on two
qubits, keeping execution cheap while testing admission before optimization.
Three additional live cases check that `max_nnz=0` retains the complete small
state, a positive cap of 2 truncates it, and a request of 2,000,000 reaches the
core unchanged. Together with the state cases, this module collects **33 live
tests**.

Successful results expose score/limit in `result.raw['circuit_complexity']` for
Simulator, Compressor and Feasibility, and `report.circuit_complexity` for
Quantum I/O. Simulator also returns `result.raw['simulation_metadata']` with
effective amplitude limits and approximation statistics. `max_nnz=0` means no
amplitude-count cap, allowing growth using available process/container memory.
Positive values are used unchanged, without an adapter-imposed amplitude ceiling.
These are server reports, not SDK admission calculations.

State and supplied-circuit limits are independent. Generated preparation,
estimation and compression work uses technical budgets; no extra commercial
shot or orchestration-step limit is introduced. Calibration targets of one/ten
minutes are not completion guarantees. Deploy and validate adapter releases
before opting into this suite; older deployments can fail the new expectations.
