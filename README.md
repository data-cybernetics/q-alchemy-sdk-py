# Q-Alchemy Python SDK

This is the Python SDK for using the data cybernetics [Q-Alchemy](https://www.q-alchemy.com) API.
It provides state preparation, hosted sparse simulation, and provider-neutral Quantum I/O
execution through PineXQ. The state-preparation API helps quantum computing researchers put
classical data into a quantum computer (the loading/encoding problem, sometimes described as a
form of QRAM).

Under the hood, Q-Alchemy runs on [PineXQ](https://pinexq.net), the hypermedia (Siren) API platform of
[data cybernetics](https://www.data-cybernetics.com). You do not need to know anything about it to use this
SDK; if you want to work with the API directly, see the [PineXQ documentation](https://pinexq.net/docs/).

## Installation

The SDK is published on PyPI, so you can install it with pip (or poetry, uv, pdm, ...):

```bash
pip install q-alchemy-sdk-py
# or
uv add q-alchemy-sdk-py
# or
pdm add q-alchemy-sdk-py
```

If you want to use the qiskit-integration, please use
```bash
pip install q-alchemy-sdk-py[qiskit]
```

And if you want the PennyLane-integration, please use
```bash
pip install q-alchemy-sdk-py[pennylane]
```

If you would like to run our examples, please use
```bash
pip install q-alchemy-sdk-py[examples]
```

We use [uv](https://docs.astral.sh/uv/) and have tested this all with Python 3.11 through 3.14. So the way to install 
it after cloning is simply

```bash
uv sync --locked
```

Again, for qiskit- or PennyLane-integrations, please add the groups
```bash
uv sync --locked --extra qiskit --extra pennylane
```

And for running our examples,
```bash
uv sync --locked --extra examples
```

## Usage

There are examples under the `/examples` folder, but for those that are eager to find out, here it is.
First, you will want to get an API key from the [Q-Alchemy Portal](https://portal.q-alchemy.com/). You 
need to sign up for this, sorry, but this is necessary. Once you have the API key (free of charge of course)
you can test it!

### Direct Example

```python
import numpy as np
import os
from sklearn.datasets import fetch_openml

from q_alchemy.initialize import q_alchemy_as_qasm

mnist = fetch_openml('mnist_784', version=1, parser="auto")

zero: np.ndarray = mnist.data[mnist.target == "0"].iloc[0].to_numpy()
filler = np.empty(2 ** 10 - zero.shape[0])
filler.fill(0)

zero = np.hstack([zero, filler])
zero = zero / np.linalg.norm(zero)

qasm, summary = q_alchemy_as_qasm(zero, max_fidelity_loss=0.2, 
    api_key=os.environ["Q_ALCHEMY_API_KEY"], return_summary=True)
print(summary)
```

### Qiskit Example

```python
import numpy as np
from sklearn.datasets import fetch_openml
import os

from q_alchemy.qiskit_integration import QAlchemyInitialize, OptParams

mnist = fetch_openml('mnist_784', version=1, parser="auto")

zero: np.ndarray = mnist.data[mnist.target == "0"].iloc[0].to_numpy()
filler = np.empty(2 ** 10 - zero.shape[0])
filler.fill(0)

zero = np.hstack([zero, filler])
zero = zero / np.linalg.norm(zero)

instr = QAlchemyInitialize(
    params=zero.tolist(),
    opt_params=OptParams(
        max_fidelity_loss=0.1,
        basis_gates=["id", "rx", "ry", "rz", "cx"],
        api_key=os.environ["Q_ALCHEMY_API_KEY"]
    )
)
instr.definition.draw(fold=-1)
```

### PennyLane Example

```python
import numpy as np
import pennylane as qml
from sklearn.datasets import fetch_openml
import os

from q_alchemy.pennylane_integration import QAlchemyStatePreparation, OptParams

mnist = fetch_openml('mnist_784', version=1, parser="auto")

zero: np.ndarray = mnist.data[mnist.target == "0"].iloc[0].to_numpy()
filler = np.empty(2 ** 10 - zero.shape[0])
filler.fill(0)

zero = np.hstack([zero, filler])
zero = zero / np.linalg.norm(zero)

dev = qml.device('lightning.qubit', wires=10)

@qml.qnode(dev)
def circuit(state=None):
    QAlchemyStatePreparation(
        state,
        wires=range(10),
        opt_params=OptParams(
            max_fidelity_loss=0.1,
            basis_gates=["id", "rx", "ry", "rz", "cx"],
            api_key=os.environ["Q_ALCHEMY_API_KEY"]
        )
    )
    return qml.state()

print(qml.draw(circuit, level="device", max_length=100)(zero.tolist()))
```

### Broadcasting with PennyLane

PennyLane provides native support for *broadcasting*, which allows quantum nodes to process batches of inputs efficiently. This is particularly useful in machine learning applications where inputs often come in batches. When broadcasting is used in conjunction with Q-Alchemy, each state in the batch is individually prepared using Q-Alchemy's circuit synthesis capabilities.

> ⚠️ **Note:** For simulators or backends that support native state initialization using the `StatePrep` gate—such as `default.qubit`, and `lightning.qubit`—the state vector is injected directly without any decomposition into quantum gates. In this case, Q-Alchemy is not used. This behavior is ideal for rapid prototyping and testing. Switching to a hardware backend (or one without native state prep) will automatically invoke Q-Alchemy for state preparation.

#### Broadcasting Example

```python
import numpy as np
import pennylane as qml
import os
import torch

from q_alchemy.pennylane_integration import AmplitudeEmbedding, OptParams
from sklearn.datasets import make_moons

# Sample data
X, _ = make_moons(n_samples=5, noise=0.1)
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize each row for amplitude embedding

# Create PennyLane device
dev = qml.device("qiskit.aer", wires=1)

@qml.qnode(dev, interface="torch")
def circuit(x):
    AmplitudeEmbedding(
        x,
        wires=[0],
        opt_params=OptParams(
            max_fidelity_loss=0.0,
            api_key=os.environ["Q_ALCHEMY_API_KEY"]
        )
    )
    return qml.expval(qml.PauliZ(0))

# Run the circuit on a batch of inputs
X_tensor = torch.tensor(X, dtype=torch.float64)
print(qml.draw(circuit, level="device", max_length=100)(X_tensor))
```

This example demonstrates how batched data can be processed using broadcasting with `AmplitudeEmbedding`, and how Q-Alchemy is triggered on simulators like `qiskit.aer`. When moving to real hardware or gate-based backends that lack `StatePrep` gate, Q-Alchemy will transparently handle the state preparation.

### Advanced options

Every entry point (`q_alchemy_as_qasm`, `QAlchemyInitialize`,
`QAlchemyStatePreparation`, `AmplitudeEmbedding`) takes its settings as an
`OptParams` object, or as a plain dict with the same keys. `q_alchemy_as_qasm`
also accepts them as keyword arguments. The state itself can be a list, a numpy
array, a scipy sparse array or, for Qiskit, a `Statevector`.

The fields you are most likely to touch:

| Field | Default | Meaning |
|---|---|---|
| `max_fidelity_loss` | `0.0` | How much fidelity you are willing to give up for a shallower circuit. `0.0` asks for an exact preparation. |
| `basis_gates` | `["u", "cx"]` | Gate set the returned circuit is transpiled to. |
| `api_key` | `$Q_ALCHEMY_API_KEY` | Your Q-Alchemy API key. Keep it safe! |
| `initialization_method` | `InitializationMethods.AUTO` | Which algorithm builds the circuit (see below). |
| `extra_kwargs` | `{}` | Method-specific options, as a dict (see below). |
| `use_qasm3` | `False` | Experimental: return OpenQASM 3 instead of OpenQASM 2. |
| `remove_data` | `True` | Delete the job and its uploaded data once the result is fetched. |
| `job_completion_timeout_sec` | `300` | How long to wait for the job before giving up. |

`q_alchemy_as_qasm_parallel(state, option_sets, max_workers=4)` compares option
sets concurrently. Results follow the order of `option_sets`; a failed worker
raises an exception instead of returning an incomplete list. Input serialization
is shared, and `max_workers` bounds concurrent jobs. Already-running jobs finish
their normal cleanup if another worker fails. A supplied HTTP client remains
caller-owned. Processing-step lookups are cached per client for five minutes;
create a new client when changing accounts.

`QAlchemyInitialize` owns a copy of the input amplitudes, so later changes to the
caller's array cannot change the pending initialization. Dense inputs remain
NumPy arrays internally. Upload hashing is controlled by `assign_data_hash`;
the instruction no longer computes the unused `param_hash` attribute.

`InitializationMethods` lives in `q_alchemy.initialize`:

- `AUTO` (default) runs several Tucker candidates (iterative and hierarchical,
  as resources allow) at your fidelity budget and keeps the cheapest realized
  circuit. Trivial inputs such as single-qubit and single-basis states take an
  exact fast path.
- `ITERATIVE_TUCKER` and `HIERARCHICAL_TUCKER` pin one Tucker variant.
- `SWAP_PIVOT` is suited to very sparse states.
- `BAA_LOW_RANK` uses the BAA low-rank initializer; it is limited to 12 qubits.

#### Method-specific options (`extra_kwargs`)

`extra_kwargs` is passed through to the chosen method. Pass a dict, and the SDK
serializes it for you:

```python
from q_alchemy.initialize import OptParams, InitializationMethods

opt_params = OptParams(
    max_fidelity_loss=0.05,
    initialization_method=InitializationMethods.ITERATIVE_TUCKER,
    extra_kwargs={"max_iterations": 10, "factors_size": 4},
)
```

> ⚠️ **Options are validated per method.** A key that the chosen method does
> not accept fails the job with `Could not build the initialization circuit: ...`
> instead of being silently ignored. In particular, the iterative-Tucker keys
> below are **not** valid under `AUTO`. Pin the method to use them.

| Method | Accepted `extra_kwargs` keys |
|---|---|
| `AUTO` | `cost_function` (`"cx_then_depth"` default, `"depth_then_cx"`, `"two_qubit_then_depth"`, `"cx+depth"`, `"cx"`, `"depth"`), `basis_gates` (gate set used to *compare* candidates, default `["u", "cx"]`), `transpile_optimization_level` (1), `seed_transpiler` (0), `dominant_basis_fast_path` (`True`), `fidelity_tolerance`, `geometric_entanglement`, `check_normalization` (`True`) |
| `ITERATIVE_TUCKER` | `max_iterations` (≤ 0 picks one from the qubit count), `factors_size` (0 = automatic), `max_stepup` (0), `fallback` (`True`), `perturbation` (`None`), `geometric_entanglement` (0.0), `check_normalization` (`True`), `barriers` (`False`; debugging only, hurts transpilation) |
| `HIERARCHICAL_TUCKER` | `geometric_entanglement`, `check_normalization` |
| `SWAP_PIVOT` | `aux` |
| `BAA_LOW_RANK` | `strategy` (`"greedy"`), `use_low_rank` (`True`), `max_combination_size`, `iso_scheme`, `unitary_scheme` |

Two interactions worth knowing:

- **Set the fidelity with `max_fidelity_loss` on `OptParams`, not in
  `extra_kwargs`.** Every method accepts `max_fidelity_loss` in `extra_kwargs`
  too, but if you put it there it silently overrides the top-level value.
- **Under `AUTO`, `basis_gates` on `OptParams` only affects the final
  transpilation.** Candidates are compared on `u`/`cx` cost unless you also pass
  `extra_kwargs={"basis_gates": [...]}`.

### Running experiments with Quantum I/O

State widths, sparse indices, qubit selections and shot counts require integers;
booleans and fractional values are rejected locally. Sparse indices in JSON
remain decimal strings so large basis indices retain their full precision.

Quantum I/O uses typed Python objects throughout the public SDK. Users construct
`State`, `Circuit`, `MeasurementPlan`, `QuantumExperiment`, `Runtime`, and
`ExecutionPlan` objects and receive a typed `ExperimentReport`. The SDK handles the
schema-3 JSON serialization used by PineXQ internally; application code does not need
to assemble contract dictionaries or encode complex amplitudes manually.
`ExperimentReport.format_summary()` uses the same human-readable section layout and
number formatting as the Quantum I/O runtime report, so local/service output remains
consistent at the client boundary.

Quantum I/O removes its SDK-created PineXQ Job and input/output WorkData by default
once `result()` has successfully returned a report. Provider credentials are uploaded
separately as Secret WorkData before they are attached to a Job and never become part of
`QuantumExperiment` or `ExecutionPlan`. Set `QuantumIOParams(remove_data=False)` when
you explicitly want to preserve the complete PineXQ execution lineage.

> **Execution-lineage cleanup happens after successful result retrieval.** If the job
> fails, times out, or returns a report the SDK cannot parse, the Job and its WorkData are
> left in place so the failure can be diagnosed and `result()` retried. If submission
> fails before PineXQ creates a Job, unreferenced WorkData created by that attempt is
> removed when `remove_data=True`. If cleanup itself fails after the report was cached,
> another `result()` call retries the remaining cleanup. After a successful automatic
> cleanup the PineXQ resources are gone: `QuantumIOJob.raw_job` then raises, and
> `QuantumIOJob.removed` is `True`. Use `remove_data=False` if you need to inspect the Job
> afterwards.

#### Preflight

Preflight is the simplest Quantum I/O workflow. It prepares the target state and, by
default, independently verifies the preparation with Q-Alchemy's sparse simulator. It
does not run a reference experiment, noisy simulation, or QPU acquisition.

```python
from math import sqrt

from q_alchemy import QuantumExperiment, QuantumIOService, State

a = 1 / sqrt(2)
experiment = QuantumExperiment(
    target=State.dense([a, 0, 0, a]),
)

report = QuantumIOService().preflight(experiment).result()

print(report.mode)  # preflight
print(report.preparation.method)
print(report.preparation.metrics.cx_count)
print(report.preparation_preflight.target_to_prepared_fidelity)
```

See `examples/quantum_io_preflight.py` for the complete preflight example.

#### Ideal simulation

With no explicit execution plan, `run()` uses the deployed Q-Alchemy sparse simulator
as an ideal acquisition source:

```python
from math import sqrt

from q_alchemy import (
    BasisMeasurement,
    MeasurementPlan,
    QuantumExperiment,
    QuantumIOService,
    State,
)

a = 1 / sqrt(2)
experiment = QuantumExperiment(
    target=State.dense([a, 0, 0, a]),
    measurement_plan=MeasurementPlan(
        basis_measurements=(BasisMeasurement("q0-q1", (0, 1)),),
    ),
)

report = QuantumIOService().run(experiment, shots=256).result()
print(report.mode)                       # ideal-simulation
print(report.execution.source_kind)      # ideal-simulator
print(report.execution.basis_distributions[0].probabilities)
```

See `examples/quantum_io_service.py` for a runnable version.

#### Noisy backend simulation

A noisy simulation can use the topology and calibration of a real IBM backend without
submitting a QPU job. The PineXQ runtime constructs `AerSimulator.from_backend(...)`
and reports `source_kind="noisy-simulator"`. An ideal sparse reference can be run in
the same experiment so the report includes observable and distribution error metrics.

```python
import os
from qiskit import QuantumCircuit

from q_alchemy import (
    BasisMeasurement,
    Circuit,
    IBMQuantumCredentials,
    MeasurementPlan,
    PauliObservable,
    QuantumExperiment,
    QuantumIOService,
    State,
    noisy_backend_execution_plan,
)

credentials = IBMQuantumCredentials(token=os.environ["IBM_QUANTUM_TOKEN"])
service = QuantumIOService(ibm_credentials=credentials)
backends = service.backends(provider="ibm", min_num_qubits=2)
backend = min(
    backends,
    key=lambda item: (item.num_qubits, item.pending_jobs or 0, item.name),
)

evolution = QuantumCircuit(2)
evolution.h(0)
evolution.cx(0, 1)

experiment = QuantumExperiment(
    target=State.dense([1, 0, 0, 0]),
    evolution=Circuit.from_qiskit(evolution),
    measurement_plan=MeasurementPlan(
        training=(
            PauliObservable.pauli("ZI", "ZI"),
            PauliObservable.pauli("IZ", "IZ"),
        ),
        validation=(
            PauliObservable.pauli("XX", "XX"),
            PauliObservable.pauli("ZZ", "ZZ"),
        ),
        basis_measurements=(BasisMeasurement("computational", (0, 1)),),
    ),
)

plan = noisy_backend_execution_plan(
    provider="ibm",
    backend=backend.name,
    shots=4096,
    ideal_reference=True,
    estimator=True,
)
report = service.run(experiment, execution_plan=plan).result()

print(report.mode)                       # noisy-simulation
print(report.execution.source_kind)      # noisy-simulator
print(report.observable_error)
print(report.distribution_errors)
print(report.held_out_verification_error)
```

No QPU execution occurs in this workflow. IBM credentials are used only to discover the
backend and read the device information needed to construct the calibrated simulator.
`examples/quantum_io_noisy_simulation.py` demonstrates the full typed report, including
preparation diagnostics, complete-circuit metrics, ideal reference, noisy observations,
basis distributions, error metrics, state-estimation output, held-out verification, and
warnings.

#### Quantum hardware

When hardware is needed, backend discovery and execution follow the familiar
service/backend pattern. The experiment object is exactly the same one used for
simulation:

```python
backend = service.backends(provider="ibm", min_num_qubits=2)[0]
report = backend.run(experiment, shots=1024).result()
print(report.mode)                       # qpu
print(report.execution.source_kind)      # qpu
```

For advanced workflows, construct a typed `ExecutionPlan` explicitly. `Runtime.resource`
selects PineXQ runtime resources without exposing credentials in the experiment contract.

### Choosing classical or quantum execution with feasibility

The feasibility API is a separate hosted service. The SDK is only its remote client: it
uploads a `QuantumExperiment` and a portable `FeasibilityRequest`, submits one PineXQ
`assess_feasibility` Job, and returns the final `FeasibilityReport`. The classical-first
decision loop, execution of the configured quantum target (a simulator or a physical
QPU), and evidence collection happen on the service side rather than
in the SDK process.

```python
from q_alchemy import (
    FeasibilityRequest,
    FeasibilityService,
    QuantumExperiment,
    SolutionCriteria,
    State,
)

experiment = QuantumExperiment(target=State.dense([1, 0]))
request = FeasibilityRequest(
    criteria=SolutionCriteria.common(max_observable_rmse=0.01),
)

report = FeasibilityService().analyze(experiment, request).result()
print(report.format_summary())
```

`FeasibilityReport.format_summary()` mirrors the canonical human-readable formatter in
`q-alchemy-feasibility`, including classical and quantum status, resource/quality
criteria, backend/model evidence, recommendation, and next-evidence requests.
The SDK formatter is aligned with Feasibility 0.6.48, the PineXQ adapter 0.2.9, and
Quantum I/O 0.10.3. Serialized core reports and their canonical summaries are kept
as regression fixtures so later core changes can be checked without installing
the server runtime in the SDK environment.

Classical `Status` describes resource feasibility; `Quality` is a separate assessment.
Quality based on `quantum-io:exact-reference` is explicitly labeled as relative to
the simulated circuit reference, not necessarily the intended target state.
When Quantum I/O reports a contradicted preparation estimate, the summary displays
its warning. The raw evidence and `report.warnings` remain available as well.

The embedded computation's `preparation_preflight.claim_contradicted` is tri-state:
`True` means a certified comparison found more preparation loss than the initializer
estimated; `False` means the comparison passed; `None` (including absent legacy fields)
means no certified comparison is available. `preparation.claimed_fidelity_loss` is
the initializer's estimate, `target_to_prepared_fidelity` is the measured fidelity,
and `preparation_approximation_infidelity` is the measured loss. Quantum I/O owns
the exactness and comparison-tolerance checks. The SDK preserves these fields;
it does not simulate circuits, recompute the verdict, or infer target-relative
energy error from a fidelity discrepancy.

Execution settings use `EvidenceCollectionConfig(execution_options=...)`, separate
from `backend_options`. The default is `{"transpile": True}`. For example:

```python
from q_alchemy import EvidenceCollectionConfig

evidence = EvidenceCollectionConfig(
    provider="aer",
    execution_options={
        "transpile": True,
        "transpile_options": {"optimization_level": 3},
        "estimator_options": {"seed_simulator": 7},
    },
)
```

Use `execution_options={"transpile": False}` only for circuits already compatible
with the selected backend. Legacy execution keys inside `backend_options` are
moved into `execution_options`; conflicting values are rejected locally.
Use `estimator_options` for Pauli-observable execution. `backend_run_options`
(and its legacy alias `run_options`) applies to basis-measurement/count execution.
Retired resource-sweep and attribution controls are no longer constructor
parameters. Their keys in old request JSON are ignored, matching the service.
`classical_first=False` is rejected locally; use `quantum_execution=COMPARE`
when both classical and quantum paths are wanted.

To check formatter parity after a core upgrade, run
`tests/generate_feasibility_fixtures.py --check` using a Python environment with
`q-alchemy-feasibility` installed and without the SDK on its import path. If the
core intentionally changes, rerun without `--check`, review the fixture diff,
and run `pytest tests/test_feasibility_contract_alignment.py` in the SDK environment.
The fixture generator only analyzes synthetic evidence; it submits no PineXQ jobs.

Full-circuit compression is enabled by default by the feasibility service. The typed SDK
contract exposes the same control when callers need to disable it or pass compressor
options:

```python
from q_alchemy import CircuitCompressionConfig, EvidenceCollectionConfig

evidence = EvidenceCollectionConfig(
    circuit_compression=CircuitCompressionConfig(
        enabled=True,
        options={"optimization_level": 2},
    )
)
```

When a criterion requires QTucker state-estimation evidence and the experiment does not
supply an explicit estimator training plan, the service can generate the reconstruction
family automatically. `qtucker_config` controls the fitted Tucker model, while
`qtucker_observable_config` controls observable generation and independent validation:

```python
evidence = EvidenceCollectionConfig(
    qtucker_config={
        "blocks": [[0, 1], [2, 3]],
        "ranks": [2, 2],
    },
    qtucker_observable_config={
        "within_block_mode": "all_paulis",
        "cross_block_mode": None,
        "extra_edges": [[0, 2]],
        "validation_edges": [[1, 3]],
        "validation_paulis": "XYZ",
    },
)
```

Every observable returned by QTucker's native reconstruction-observable generator remains
in the training set. Validation is generated separately and is never carved out of that
training family. Direct application diagnostics belong in `MeasurementPlan.observables`;
`training` and `validation` should be used only for an explicit estimator plan.

The default `QuantumExecutionPolicy.WHEN_NEEDED` stops once the classical path is already
sufficient. Estimator-only criteria do not force quantum execution merely so they can be
evaluated. Use `QuantumExecutionPolicy.COMPARE` when the quantum path should run even when
classical execution is feasible. The SDK only serializes this policy; the feasibility
service owns the routing decision.

For an opt-in end-to-end check of the SDK -> PineXQ -> feasibility transport, the
repository includes a live integration test that submits two small Bell-state jobs: one
with compression explicitly disabled and one using the default enabled compression path.
Ordinary test runs skip it. Run it explicitly with:

```bash
Q_ALCHEMY_RUN_LIVE_FEASIBILITY=1 \
Q_ALCHEMY_API_KEY=... \
pytest -v tests/test_feasibility_live_integration.py
```

Set `Q_ALCHEMY_FEASIBILITY_STEP_VERSION` as well when a specific published ProcessingStep
version should be tested; otherwise the SDK selects the latest available version. The live
test does not require IBM credentials because the two-qubit workload is intentionally
classical-first.

The SDK preserves the complete portable report returned by the service.
`report.computation_result` is the feasibility computation-result envelope: it records
which compute path ran, execution mode/source, whether the requested criteria were met,
and whether the result was selected by the recommendation. If feasibility ran a
classical, simulated, or QPU computation, `report.computation_experiment_report`
deserializes the embedded Quantum I/O `ExperimentReport`; the SDK does not rerun that
computation. The nested report uses the same formatter as a directly submitted Quantum
I/O experiment:

```python
computation = report.computation_experiment_report
if computation is not None:
    print(computation.format_summary())
```

Quantum circuits are not returned by default because generated state-preparation circuits can
be large. Request the logical experiment circuit (`P + U`) explicitly on the SDK call:

```python
report = FeasibilityService().analyze(
    experiment,
    request,
    include_quantum_circuits=True,
).result()

circuit = report.quantum_circuit
if circuit is not None:
    print(circuit.draw(output="text"))
```

The service returns the selected logical circuit as portable QASM 3. The SDK reconstructs
the native Qiskit circuit and delegates parsing/rendering to Qiskit; it does not implement a
second circuit parser or drawer. Internal measurement and backend-transpiled circuits
are not returned.

Feasibility core also returns the renderer-neutral experiment graph that describes which
steps ran, were skipped, were unavailable, or were not needed. The SDK never reconstructs
that graph from statuses. With the optional `visualization` extra installed,
`report.experiment_diagram` deserializes the service representation into the shared
`q-alchemy-visualization` `ExperimentDiagram` type:

```python
diagram = report.experiment_diagram
if diagram is not None:
    print(diagram.draw(output="text"))
```

The SDK similarly exposes a requested logical circuit as a native Qiskit
`QuantumCircuit` through `report.quantum_circuit`. Raw diagram and circuit payloads are
transport details retained by `report.to_dict()` rather than separate user-facing APIs.

The visualization package is optional so the public SDK remains installable without the
private Q-Alchemy package feed. Graphical Matplotlib output additionally requires the
`mpl` extra of `q-alchemy-visualization`.

IBM credentials are optional at submission time because the server may prove that the
problem is classically feasible and stop before any quantum-backend evidence is needed.
When credentials are supplied, the SDK transports them as separate Secret WorkData; they
are never serialized into the experiment or feasibility request.

Noisy simulation is not an automatic pre-QPU stage. When simulation is the desired
quantum target, select a simulator explicitly in `EvidenceCollectionConfig`, for example
`provider="aer"`, `backend="aer_simulator"`, optionally with portable `backend_options`.
When IBM is selected, feasibility proceeds from static target checks to physical-QPU
execution when the policy requires quantum evidence.

Use `QuantumIOService` when you explicitly want to execute a particular Quantum I/O plan.
Use `FeasibilityService` when the question is which compute path is sufficient under the
stated solution criteria.

A complete hosted example is available in `examples/feasibility_service.py`. It submits a
small Bell-state workload, prints the canonical feasibility summary and embedded Quantum I/O
report, draws the returned logical experiment circuit with Qiskit when available, and renders
the server-provided experiment diagram when the optional visualization extra is installed.

For a notebook-oriented introduction, see
[`examples/feasibility_h2_dynamics.ipynb`](examples/feasibility_h2_dynamics.ipynb). It uses
the published two-qubit parity-reduced H2 electronic Hamiltonian at 0.735 Angstrom in STO-3G,
starts from the Hartree-Fock molecular reference state, applies one first-order Trotter step,
and runs feasibility in `COMPARE` mode against an Aer target. The final cells display the
canonical `format_summary()`, the server-provided experiment diagram, and the returned logical
`P + U` circuit.

A quantum-chemistry example is available in
[`examples/feasibility_quantum_chemistry.py`](examples/feasibility_quantum_chemistry.py).
It uses an illustrative 12-spin-orbital correlated active-space workload and selects the
quantum target from the environment:

- with `IBM_QUANTUM_TOKEN`, it uses a physical IBM Quantum device;
- with both `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_BACKEND`, it uses that backend;
- with an IBM token but no backend name, it requests the least-busy suitable backend;
- without an IBM token, it uses `provider="aer"`, `backend="aer_simulator"` with an
  illustrative portable noise model.

`IBM_QUANTUM_INSTANCE` is optional and is forwarded when present. The example uses
`QuantumExecutionPolicy.COMPARE` deliberately so the selected quantum target is actually
executed even if the small classical reference is feasible. Normal feasibility requests can
keep the default `when-needed` policy. Run it with:

```bash
# Required for the hosted Q-Alchemy service.
export Q_ALCHEMY_API_KEY=...

# Optional: physical IBM execution. Omit IBM_QUANTUM_TOKEN for noisy Aer.
export IBM_QUANTUM_TOKEN=...
export IBM_QUANTUM_BACKEND=ibm_example   # optional; otherwise least busy
export IBM_QUANTUM_INSTANCE=...          # optional

 python examples/feasibility_quantum_chemistry.py
 ```

### Verifying preparation circuits with the sparse simulator

Simulator jobs and their WorkData are preserved on execution, timeout or result
download failure for diagnosis. After a successful download, `remove_data=True`
requests cleanup; a cleanup failure logs a warning and still returns the result.

Q-Alchemy also hosts a **sparse state-vector simulator** so you can verify that a
preparation circuit really produces your target state. The typical loop is
**prepare → simulate → verify**:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from q_alchemy import q_alchemy_as_qasm, SparseSimulator

qasm = q_alchemy_as_qasm(target, max_fidelity_loss=0.0)   # prepare
prep = QuantumCircuit.from_qasm_str(qasm)

sim = SparseSimulator()                                    # verify
sv = sim.sparse_statevector(prep)
print("fidelity:", state_fidelity(Statevector(sv.to_dense()), Statevector(target)))
```

`SparseSimulator` also exposes `.counts(...)` and `.tomography(...)`, and it
auto-selects your resource tier from your plan (Standard/Medium for everyone,
XLarge for enterprise).

> ⚠️ **The free plan is strongly limited** — state preparation is capped (currently
> ~12 qubits, batches up to 100) and runs on the Medium simulator tier. Enterprise
> plans get larger circuits and the XLarge tier. See the limits table in the guide.

📖 **Full guide:** [docs/initialize-and-verify.md](docs/initialize-and-verify.md) ·
🧪 **Runnable notebook:** [examples/simulator_vs_initializer.ipynb](examples/simulator_vs_initializer.ipynb)

### Using the simulator as a Qiskit backend

The hosted simulator is also exposed through Qiskit's standard `BackendV2`
interface — the same way IBM backends are used — so it drops straight into the
Qiskit ecosystem (transpiler, `Sampler`, etc.):

```python
from qiskit import QuantumCircuit, transpile
from q_alchemy import QAlchemyBackend

backend = QAlchemyBackend()                         # reads Q_ALCHEMY_API_KEY from env

qc = QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])

job = backend.run(transpile(qc, backend), shots=4096)
print(job.result().get_counts())                    # {'00': ~2048, '11': ~2048}
```

`backend.run(...)` accepts Aer-style options (`shots`, `seed_simulator`,
`save_sparse_statevector`, `save_statevector`, `sparse_index_format`, ...), and
there's an IBM-style `QAlchemyProvider().get_backend()` for discovery. The
resource tier (Medium vs the enterprise XLarge) is selected automatically from
your plan — see [the guide](docs/initialize-and-verify.md).

#### `save_statevector` vs `save_sparse_statevector`

Both ask for the state the circuit prepares, and both are served by the same
remote call — the difference is what you get back, and how big it is.

`save_statevector` is the option you already know from Aer. It hands you the
familiar dense `2**n` vector, so `Statevector`, `state_fidelity` and friends all
work unchanged:

```python
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)                                # no measurement: this is a state export

result = backend.run(qc, save_statevector=True).result()
print(result.data(0)["statevector"])
# [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]   -> length 2**n
```

Dense exports default to `max_dense_qubits=26` (up to 1 GiB of amplitudes).
Larger requests fail before submission. `SparseStatevectorResult.to_dense()`
enforces the same default; pass a larger `max_dense_qubits` explicitly only when
sufficient memory is available. Backend `run()` options apply only to that job;
use `backend.set_options()` to change defaults for future jobs.

`save_sparse_statevector` is Q-Alchemy's own, and it is the one that scales. It
returns **only the amplitudes the circuit actually populates**, so nothing of
size `2**n` is ever built:

```python
result = backend.run(qc, save_sparse_statevector=True).result()
print(result.data(0)["sparse_statevector"])
# {'format': 'sparse_statevector_v1', 'num_qubits': 2, 'nnz': 2,
#  'index_format': 'hex', 'index_convention': 'little_endian',
#  'indices': ['0x0', '0x3'],
#  'amplitudes': [[0.7071067811865476, 0.0], [0.7071067811865476, 0.0]]}
```

Note the amplitudes are `[real, imag]` pairs: this entry is the simulator's own
JSON payload, passed through untouched. If you would rather have parsed
`complex` values — plus `to_coo()`, `to_arrow()` and `amplitudes_dict()` — go
through the client instead of the backend, which returns a typed
`SparseStatevectorResult`:

```python
from q_alchemy import SparseSimulator

sv = SparseSimulator().sparse_statevector(qc)
print(sv.amplitudes)          # [(0.7071067811865476+0j), (0.7071067811865476+0j)]
print(sv.amplitudes_dict())   # {'0x0': (0.707...+0j), '0x3': (0.707...+0j)}
```

`indices` and `amplitudes` line up element by element, and `nnz` is how many
were stored — two here, not four. That gap is the whole point, and it widens
fast: a 40-qubit state-preparation circuit populating a thousand basis states
returns a thousand amplitudes, while the dense form would need `2**40` complex
numbers, or roughly 17 TB. **A dense export is impossible in that regime; a
sparse one is a few hundred kilobytes.** Standard Qiskit backends offer no
equivalent.

That only helps if the state *is* sparse, and nothing forces it to be. By
default the simulator keeps every amplitude above `1e-10`, so the result is
exact, but a circuit that populates most of its basis states needs as much
memory as the dense form and can exhaust the simulator. Pass `max_nnz=N` (to
`backend.run` or `SparseSimulator.sparse_statevector`) to cap it: after every
gate only the `N` largest amplitudes are kept and renormalised. The result is
then approximate, and nothing in it marks that it was truncated.

Ask for both together and you pay for one simulation:

```python
result = backend.run(qc, save_statevector=True, save_sparse_statevector=True).result()
dense = result.data(0)["statevector"]               # 2**n numpy array
sparse = result.data(0)["sparse_statevector"]       # nnz entries
```

Use the dense form for small circuits you want to compare against Qiskit
directly; use the sparse form whenever `2**n` would not fit — which is exactly
the regime this simulator exists for. Reach for `sparse_index_format`
(`hex`/`bitstring`) and `sparse_index_convention`
(`little_endian`/`big_endian`) to control how the indices are written, and see
[the guide](docs/initialize-and-verify.md) for feeding a sparse result straight
back into the loader via `to_coo()`.

#### From PennyLane

Because it's a standard Qiskit backend, you can use it as a PennyLane device via
the [PennyLane–Qiskit plugin](https://github.com/PennyLaneAI/pennylane-qiskit)
(whose original version was written by this SDK's author, Carsten Blank):

```python
import pennylane as qml
from q_alchemy import QAlchemyBackend

dev = qml.device("qiskit.remote", wires=2, backend=QAlchemyBackend(), shots=4096)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts()
```

## Contributions

We welcome contributions - simply fork the repository of this plugin, and then make a pull request 
containing your contribution. All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements!

## Authors

Carsten Blank

## License

The q-alchemy-sdk-py is free and open source, released under the Apache License, Version 2.0.
