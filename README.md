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

We use [python-pdm](https://pdm-project.org/) and have tested this all with Python 3.11 or higher (but less than 4!). So the way to install 
it after cloning is simply

```bash
pdm install
```

## Usage

There are examples under the `/examples` folder, but for those that are eager to find out, here it is.
First, you will want to get an API key from the [Q-Alchemy Portal](https://portal.q-alchemy.com/). You 
need to sign up for this, sorry, but this is necessary. Once you have the API key (free of charge of course)
you can test it!

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

### PennyLane

Will come soon!

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