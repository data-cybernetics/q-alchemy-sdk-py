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

We have decided not to go through pypi, but you can install this through pip or poetry nonetheless

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

We use [python-pdm](https://pdm-project.org/) and have tested this all with Python 3.11 or higher (but less than 4!). So the way to install 
it after cloning is simply

```bash
pdm install
```

Again, for qiskit- or PennyLane-integrations, please add the groups
```bash
pdm install -G qiskit -G pennylane
```
Or whatever combination you need. Currently, the PennyLane-integration is dependent on the qiskit-integration... what a 
fallacy! We will -- of course -- fix this soon!

## Usage

There are examples under the `/examples` folder, but for those that are eager to find out, here it is.
First, you will want to get an API key from the [Q-Alchemy Portal](https://portal.q-alchemy.com/). You 
need to sign up for this, sorry, but this is necessary. Once you have the API key (free of charge of course)
you can test it!

### Direct Example

```python
import numpy as np
from sklearn.datasets import fetch_openml

from q_alchemy.initialize import q_alchemy_as_qasm

mnist = fetch_openml('mnist_784', version=1, parser="auto")

zero: np.ndarray = mnist.data[mnist.target == "0"].iloc[0].to_numpy()
filler = np.empty(2 ** 10 - zero.shape[0])
filler.fill(0)

zero = np.hstack([zero, filler])
zero = zero / np.linalg.norm(zero)

qasm, summary = q_alchemy_as_qasm(zero, max_fidelity_loss=0.2, api_key="<your api key>", return_summary=True)
print(summary)
```

### Qiskit Example

```python
import numpy as np
from sklearn.datasets import fetch_openml

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
        api_key="<your api key>"
    )
)
instr.definition.draw(fold=-1)
```

### PennyLane Example

```python
import numpy as np
import pennylane as qml
from sklearn.datasets import fetch_openml

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
            api_key="<your api key>"
        )
    )
    return qml.state()

print(qml.draw(circuit, level="device", max_length=100)(zero.tolist()))
```

### Broadcasting with PennyLane

PennyLane provides native support for *broadcasting*, which allows quantum nodes to process batches of inputs efficiently. This is particularly useful in machine learning applications where inputs often come in batches. When broadcasting is used in conjunction with Q-Alchemy, each state in the batch is individually prepared using Q-Alchemy's circuit synthesis capabilities.

> ⚠️ **Note:** For simulators or backends that support native state initialization using the `StatePrep` gate—such as `default.qubit`, and `lightning.qubit`—the state vector is injected directly without any decomposition into quantum gates. In this case, Q-Alchemy is not used. This behavior is ideal for rapid prototyping and testing. Switching to a hardware backend (or one without native state prep) will automatically invoke Q-Alchemy for state preparation.

#### Broadcasting Example with `qiskit.aer`

```python
import numpy as np
import pennylane as qml
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
            api_key="<your api key>"
        )
    )
    return qml.expval(qml.PauliZ(0))

# Run the circuit on a batch of inputs
X_tensor = torch.tensor(X, dtype=torch.float64)
print(qml.draw(circuit, level="device", max_length=100)(X_tensor))
```

This example demonstrates how batched data can be processed using broadcasting with `AmplitudeEmbedding`, and how Q-Alchemy is triggered on simulators like `qiskit.aer`. When moving to real hardware or gate-based backends that lack `StatePrep` gate, Q-Alchemy will transparently handle the state preparation.

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