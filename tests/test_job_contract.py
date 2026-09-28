"""The job the SDK submits, pinned to the server's published names.

Offline. Every other test that builds a job runs against the live API, and all of
them skip in the keyless PR check. So nothing in CI noticed if a step name,
parameter name or method string drifted from what the server resolves by name.

The expected values are the server's (qalchemy_procon: the step functions in
qautotucker_step.py, InitializeOptions/InitializeMethods in qautotucker_models.py,
StateVectorInput/DataFileType in qtensor_models.py). Change them here only
together with a server release.
"""
import json
import unittest
from types import SimpleNamespace

from q_alchemy.initialize import InitializationMethods, OptParams, create_processing_input

OPTIONS = OptParams(
    max_fidelity_loss=0.1,
    basis_gates=["rz", "sx", "cx"],
    initialization_method=InitializationMethods.HIERARCHICAL_TUCKER,
    use_qasm3=True,
    extra_kwargs={"max_rank": 4},
)

# What the server's build_initialization_circuit* steps take, besides the state.
COMMON = {
    "min_fidelity": 0.9,
    "basis_gates": ["rz", "sx", "cx"],
    "options": {
        "method": "hierarchical_tucker",
        "use_qasm3": True,
        "opt_params": '{"max_rank": 4}',
    },
}

# statevector_data is a WorkDataLink for uploaded states; only "not a str" matters here.
UPLOADED = SimpleNamespace()


def submitted(name_and_parameters):
    """The job as sent: configure_job serialises the parameters with json.dumps."""
    name, parameters = name_and_parameters
    return name, json.loads(json.dumps(parameters))


class JobContractTestCase(unittest.TestCase):

    def test_inline_state(self):
        name, parameters = submitted(create_processing_input(OPTIONS, "c3RhdGU="))
        self.assertEqual(name, "build_initialization_circuit_inline")
        self.assertEqual(parameters, COMMON | {
            "state_vector": {"state_vector_base64": "c3RhdGU=", "state_vector_type": "parquet"},
        })

    def test_uploaded_state(self):
        # The state goes in through the input dataslot, not the parameters.
        self.assertEqual(submitted(create_processing_input(OPTIONS, UPLOADED)),
                         ("build_initialization_circuit", COMMON))

    def test_batch(self):
        self.assertEqual(submitted(create_processing_input(OPTIONS, UPLOADED, num_states=3)),
                         ("build_initialization_circuits", COMMON))

    def test_every_method_is_sent_as_the_server_spells_it(self):
        server_methods = {"auto", "hierarchical_tucker", "iterative_tucker", "swap_pivot", "baa_low_rank"}
        self.assertEqual({str(m) for m in InitializationMethods}, server_methods)
        for method in InitializationMethods:
            with self.subTest(method=method):
                _, parameters = submitted(create_processing_input(
                    OptParams(initialization_method=method), UPLOADED))
                self.assertEqual(parameters["options"]["method"], str(method))

    def test_defaults(self):
        _, parameters = submitted(create_processing_input(OptParams(), UPLOADED))
        self.assertEqual(parameters, {
            "min_fidelity": 1.0,
            "basis_gates": ["u", "cx"],
            "options": {"method": "auto", "use_qasm3": False, "opt_params": "{}"},
        })


if __name__ == '__main__':
    unittest.main()
