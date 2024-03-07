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

and even anaconda can use pip installations, so you should be fine. The dependencies that this 
SDK has are currently the following:

```toml
numpy = "*"
pyarrow = "*"
requests = "*"
pydantic = "<2"
python-dateutil = "*"
retry = "*"
qiskit_aer = "*"
qclib = "*"
````

We use poetry and have tested this all with Python 3.9 or higher (but less than 4!). So the way to install 
it after cloning is simply

```bash
poetry install
```

## Usage

There are examples under the `/examples` folder, but for those that are eager to find out, here it is.
First, you will want to get an API key from the [Q-Alchemy Portal](https://portal.q-alchemy.com/). You 
need to sign up for this, sorry, but this is necessary. Once you have the API key (free of charge of course)
you instantiate a client:

```python
from q_alchemy import Client
client = Client(api_key="<get your own at https://portal.q-alchemy.com>")
```

The client offers access to a job-root element:

```python
root = client.get_jobs_root()
```

with which one can create a new job like this:

```python
# Let us create a new and empty job, but rename it right away!
job = root.create_job()
job.rename("Your first Test-Job!")

# First, let us configure the job with the job's config
# for that we summon up the config resource!
config = job.get_config()
```

When we want to create a new quantum state, we can generate one with the help of
random circuits, this little "helper" is also part of the SDK and can be achieved 
this way:

```python
from q_alchemy.random_circuits import get_vector

qb = 12
state, entanglement, circuit_depth = get_vector(0.5, 0.8, qb, 0.5, 'geometric')
```

here we create a quantum state with *geometric* entanglement between 0.5 and 0.8, with
12 qubits and some extra parameter for the generation. 

After that, we can upload the state vector:

```python
# Upload the state vector now:
state_vector = job.get_state_vector()
state_vector.upload_vector(state)
```

and then we can configure the q-alchemy job such we use an allowable fidelity loss
of `0.21`

```python
# Set the fidelity loss and some helpful tags to find the job again
fid_loss = 0.21
config.set_config(fid_loss, [f"{qb}qb", str(fid_loss)])
```

Finally, we simply start the job and wait for the result, and then print out the 
qiskit circuit:

```python
# Start the Job
job.schedule()

# Wait for the result
import time
time.sleep(60)

# Now get the best result and plot it as given
qc = job.get_result().get_best_node().to_circuit()
qc.draw(fold=-1)
```

You can use the transpile function of qiskit to get the IMB-Q basis gates:

```python
import qiskit

qiskit.transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3).draw(output="text", fold=-1)
```


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