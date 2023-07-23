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
pip install git+https://github.com/data-cybernetics/q-alchemy-sdk-py@main
```

and even anaconda can use pip installations, so you should be fine. The dependencies that this 
SDK has are currently the following:

```toml
numpy = "*"
pyarrow = "*"
requests = "*"
pydantic = "*"
python-dateutil = "*"
retry = "*"
qclib = { git = "https://github.com/qclib/qclib.git", branch = "master" }
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
job.rename("Carsten's Test-Job")

# First, let us configure the job with the job's config
# for that we summon up the config resource!
config = job.get_config()

# Create a job config with the fluent syntax:
config.create_config() \
    .with_use_low_rank(True) \
    .with_max_fidelity_loss(0.1) \
    .with_strategy(Strategy.GREEDY) \
    .with_tags("Test Job", "Q-Alchemy") \
    .upload()

#Check out the job's config as it has been configured!
job_config = config.job_config()

# Now prepare to load a state vector. We support numpy arrays natively,
# but under the hood pyarrow with parquet is used:
vector = np.load("../tests/data/test_baa_state.12.1.npy")

# Upload the state vector now:
state_vector = job.get_state_vector()
state_vector.upload_vector(vector)

# Check out, what the state actually is that you just uploaded:
downloaded_vector = state_vector.get_vector()

# Start the Job
job.schedule()
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