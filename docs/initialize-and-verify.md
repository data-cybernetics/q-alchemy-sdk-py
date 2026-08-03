# Initialize & verify: state preparation and the sparse simulator

Q-Alchemy has two complementary halves, and they're meant to be used together:

1. **Initialize** — turn a classical state vector into a quantum **preparation
   circuit** (the loading / encoding problem). This is `q_alchemy_as_qasm` and
   the Qiskit/PennyLane integrations.
2. **Verify** — run that circuit on the hosted **sparse state-vector simulator**
   (`q_alchemy.simulator.SparseSimulator`) and check that the state it produces
   is the one you asked for.

The workflow is **prepare → simulate → verify**, entirely against hosted
Q-Alchemy services. A runnable, end-to-end version of everything below is in
[`examples/simulator_vs_initializer.ipynb`](../examples/simulator_vs_initializer.ipynb).

> **Prerequisites:** an API key from the [Q-Alchemy Portal](https://portal.q-alchemy.com/)
> in `Q_ALCHEMY_API_KEY`, and the `qiskit` extra:
> `pip install "q-alchemy-sdk-py[qiskit]"`.

---

## 1. Initialize: prepare a circuit for a target state

`q_alchemy_as_qasm` returns an OpenQASM 2 circuit that prepares your state.
`max_fidelity_loss=0.0` asks for an exact preparation; a larger value trades
fidelity for fewer gates.

```python
import os
import numpy as np
from qiskit import QuantumCircuit

from q_alchemy import q_alchemy_as_qasm

# A target state (here: a sparse 5-qubit state).
rng = np.random.default_rng(7)
target = np.zeros(2**5, dtype=complex)
support = rng.choice(2**5, size=5, replace=False)
target[support] = rng.normal(size=5) + 1j * rng.normal(size=5)
target /= np.linalg.norm(target)

qasm = q_alchemy_as_qasm(target, max_fidelity_loss=0.0, basis_gates=["u", "cx"])
prep = QuantumCircuit.from_qasm_str(qasm)   # the preparation circuit
```

---

## 2. Verify: run the circuit on the sparse simulator

`SparseSimulator` submits the circuit to the hosted simulator and reads back the
result. It picks the circuit transport automatically (inline OpenQASM for small
circuits, a QPY upload for large ones).

```python
from qiskit.quantum_info import Statevector, state_fidelity
from q_alchemy import SparseSimulator

sim = SparseSimulator()                     # reads Q_ALCHEMY_API_KEY from the env

# (a) Exact sparse state-vector — only the non-zero amplitudes.
sv = sim.sparse_statevector(prep, sparse_index_format="bitstring")
print(sv.nnz, "non-zero amplitudes")

# Did we prepare the state we asked for, and does the simulator agree with Qiskit?
print("vs target:", state_fidelity(Statevector(sv.to_dense()), Statevector(target)))
print("vs qiskit:", state_fidelity(Statevector(sv.to_dense()), Statevector(prep)))

# (b) Measurement counts (Born-rule sampling).
measured = prep.copy(); measured.measure_all()
print(sim.counts(measured, shots=8192).counts)

# (c) Tomography: density matrix + fidelity against the dense reference.
tomo = sim.tomography(prep)
print("purity:", tomo.purity, "fidelity:", tomo.state_fidelity)
```

For a sparse target the simulator reproduces the prepared state **exactly**
(fidelity ≈ 1). Because `SparseStatevectorResult` converts back into the SDK's
sparse format, you can even feed a simulated state straight back into the
initializer:

```python
qasm_again = q_alchemy_as_qasm(sv.to_coo())   # round-trip: simulate -> prepare
```

---

## Plans & limits — the free tier is strongly limited

State preparation runs on Q-Alchemy's servers, so usage is bounded by your plan
(read from your account's grants). **The free tier is intentionally limited** —
it's meant for trying things out, not production-scale workloads:

| | Free plan | Enterprise plan |
|---|---|---|
| Max qubits for **state preparation** | **up to 12 qubits** | larger (currently up to ~16; in active expansion) |
| Batch size (multi-state preparation) | **up to 100 states** | higher |
| Simulator resource tier | **Standard (Medium)** | **XLarge** (auto-selected) |

If you exceed a free-tier limit you'll get a clear error asking you to upgrade
(`support@q-alchemy.com`). These numbers reflect the current prototype phase and
may change — check the [portal](https://portal.q-alchemy.com/) for what your key
allows.

### Simulator tiers are automatic

`SparseSimulator` reads your plan from the API and routes to the right functions:

- **Free / standard** users run on the **Medium** preset.
- **Enterprise** users are automatically routed to the **XLarge** functions —
  no configuration needed.

You can inspect or override this:

```python
sim = SparseSimulator()         # tier="auto" (default)
sim.tier                        # -> "standard" or "enterprise"
sim.is_enterprise()             # -> bool, from your account grants

SparseSimulator(tier="standard")    # force the Medium tier — always allowed
SparseSimulator(tier="enterprise")  # force XLarge — needs the enterprise plan
```

The two directions are deliberately not symmetric:

- **Down is always allowed.** An enterprise caller can select the standard
  functions whenever XLarge is not worth it — small circuits, cheaper runs,
  or reproducing what a free-tier user sees.
- **Up requires the plan.** `tier="enterprise"` without `plan:enterprise` raises
  a `PermissionError` at lookup, before any job is created. The ProCon's grant
  check would refuse it in any case; failing client-side turns a wasted round
  trip and an opaque server-side error into an immediate, explanatory one.

An unrecognized tier raises `ValueError` rather than quietly falling back to
standard.

---

## Use it as a Qiskit backend

Besides the `SparseSimulator` client, the simulator is exposed as a Qiskit
`BackendV2` (`QAlchemyBackend`), so it works like any IBM-style backend and,
through the PennyLane–Qiskit plugin, as a PennyLane device. See the
["Using the simulator as a Qiskit backend"](../README.md#using-the-simulator-as-a-qiskit-backend)
section of the README.

## See also

- [`examples/simulator_vs_initializer.ipynb`](../examples/simulator_vs_initializer.ipynb)
  — the full prepare → simulate → verify notebook with plots.
- The [Hypermedia-Test-UI](https://hypermedia-ui-demo.q-alchemy.com/hui?apiPath=https%3A%2F%2Fjobs.api.q-alchemy.com%2Fapi%2FEntryPoint)
  to explore the API directly.
