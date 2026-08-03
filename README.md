# Q-Alchemy Python SDK

This is the Python-SDK for using the data cybernetics [Q-Alchemy](https://www.q-alchemy.com) 
API which helps quantum computing researchers to put classical data into the quantum computer.
This is all also called: the loading problem, encoding problem, or quantum state preparation.
Some people also call it a form of QRAM, or quantum random-access memory.

This SDK builds upon the Hypermedia-Siren API of [data cybernetics](https://www.data-cybernetics.com)
which uses a document-first approach added with actions. The standardized way makes the API programmatically
accessible, which can be explored by the [Hypermedia-Test-UI](https://hypermedia-ui-demo.q-alchemy.com/hui?apiPath=https%3A%2F%2Fjobs.api.q-alchemy.com%2Fapi%2FEntryPoint)

The SDK builds upon this, so that any software developer planning to integrate with the API and
experience the API through the UI and the SDK in a very similar fashion. Also, any GUI around this
has similar characteristics.

## Installation

The SDK is published on PyPI, so you can install it with pip (or poetry, uv, ...):

```bash
pip install q-alchemy-sdk-py
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

We use [uv](https://docs.astral.sh/uv/) and have tested this all with Python 3.11 or higher (but less than 4!). So the way to install 
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

### Verifying preparation circuits with the sparse simulator

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

### Developer UI

You can play around with this as you please and check out the [Hypermedia-Test-UI](https://hypermedia-ui-demo.q-alchemy.com/hui?apiPath=https%3A%2F%2Fjobs.api.q-alchemy.com%2Fapi%2FEntryPoint)
for more info!

## Contributions

We welcome contributions - simply fork the repository of this plugin, and then make a pull request 
containing your contribution. All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements!

## Authors

Carsten Blank

## License

The q-alchemy-sdk-py is free and open source, released under the Apache License, Version 2.0.