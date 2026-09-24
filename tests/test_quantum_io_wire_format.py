"""Frozen wire-format fixtures for the Quantum I/O schema-3 contract.

`src/q_alchemy/quantum_io_contract.py` is a hand-maintained copy of
`q-alchemy-quantum-io`'s `contract.py`. Nothing at import time detects drift
between the two, so these golden payloads pin the exact JSON the SDK produces
and consumes. A diff here means the wire format changed and the deployed
ProcessingStep must be checked before release.
"""

from __future__ import annotations

import unittest

from q_alchemy import (
    SCHEMA_VERSION,
    BasisMeasurement,
    ExecutionPlan,
    ExperimentReport,
    MeasurementPlan,
    QuantumExperiment,
    Runtime,
    State,
    LOCAL_SIMULATOR_RESOURCE,
    NOISY_BACKEND_SIMULATOR_RESOURCE,
    QUANTUM_BACKEND_RESOURCE,
)
from q_alchemy.quantum_io import (
    EXECUTION_PLAN_INPUT_ALIAS,
    EXPERIMENT_INPUT_ALIAS,
    EXPERIMENT_REPORT_OUTPUT_ALIAS,
    IBM_CREDENTIALS_INPUT_ALIAS,
    LIST_QUANTUM_BACKENDS_STEP,
    QUANTUM_BACKENDS_OUTPUT_ALIAS,
    RUN_QUANTUM_EXPERIMENT_STEP,
    local_simulator_execution_plan,
    quantum_backend_execution_plan,
)


class TestFrozenIdentifiers(unittest.TestCase):
    """Names shared with q-alchemy-quantum-io and the deployed ProcessingSteps."""

    def test_schema_version(self):
        self.assertEqual(SCHEMA_VERSION, 3)

    def test_processing_step_names(self):
        self.assertEqual(RUN_QUANTUM_EXPERIMENT_STEP, "run_quantum_experiment")
        self.assertEqual(LIST_QUANTUM_BACKENDS_STEP, "list_quantum_backends")

    def test_workdata_aliases(self):
        self.assertEqual(EXPERIMENT_INPUT_ALIAS, "quantum_experiment.json")
        self.assertEqual(EXECUTION_PLAN_INPUT_ALIAS, "execution_plan.json")
        self.assertEqual(IBM_CREDENTIALS_INPUT_ALIAS, "ibm_credentials.json")
        self.assertEqual(EXPERIMENT_REPORT_OUTPUT_ALIAS, "experiment_report.json")
        self.assertEqual(QUANTUM_BACKENDS_OUTPUT_ALIAS, "quantum_backends.json")

    def test_resource_names(self):
        self.assertEqual(LOCAL_SIMULATOR_RESOURCE, "local-simulator")
        self.assertEqual(QUANTUM_BACKEND_RESOURCE, "quantum-backend")
        self.assertEqual(NOISY_BACKEND_SIMULATOR_RESOURCE, "noisy-backend-simulator")


class TestFrozenPayloads(unittest.TestCase):
    def test_experiment_payload(self):
        amplitude = 2.0**-0.5
        experiment = QuantumExperiment(
            target=State.dense([amplitude, 0.0, 0.0, amplitude]),
            measurement_plan=MeasurementPlan(
                basis_measurements=(BasisMeasurement("q0-q1", (0, 1)),),
            ),
        )
        payload = experiment.to_dict()
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["kind"], "quantum-experiment")
        self.assertEqual(
            sorted(payload),
            ["evolution", "evolution_qargs", "kind", "measurement_plan",
             "metadata", "schema_version", "target"],
        )
        self.assertEqual(payload["target"]["representation"], "dense")
        self.assertEqual(payload["target"]["num_qubits"], 2)
        self.assertEqual(
            sorted(payload["measurement_plan"]),
            ["basis_measurements", "metadata", "observable_plan_metadata",
             "training", "validation"],
        )
        self.assertEqual(QuantumExperiment.from_dict(payload), experiment)

    def test_execution_plan_payload(self):
        payload = local_simulator_execution_plan(shots=256).to_dict()
        self.assertEqual(payload["kind"], "execution-plan")
        self.assertEqual(
            sorted(payload),
            ["acquisition", "estimator", "kind", "metadata", "preparation_options",
             "preparation_simulator", "reference", "schema_version", "shots"],
        )
        self.assertEqual(
            payload["acquisition"],
            {"kind": "resource", "resource": "local-simulator", "config": {}},
        )
        self.assertEqual(payload["shots"], 256)

    def test_backend_plan_config_keys(self):
        """The host runtime consumes these keys before core resolution."""
        named = quantum_backend_execution_plan(provider="ibm", backend="ibm_x").to_dict()
        self.assertEqual(
            named["acquisition"]["config"], {"provider": "ibm", "backend": "ibm_x"}
        )
        busy = quantum_backend_execution_plan(provider="ibm", least_busy=True).to_dict()
        self.assertEqual(
            busy["acquisition"]["config"], {"provider": "ibm", "least_busy": True}
        )

    def test_report_is_parsed_from_the_documented_shape(self):
        report = ExperimentReport.from_dict(
            {
                "schema_version": 3,
                "kind": "experiment-report",
                "report": {
                    "generated_at": "2026-08-27T00:00:00Z",
                    "mode": "ideal-simulation",
                    "experiment": {
                        "target_num_qubits": 2,
                        "evolution_present": False,
                        "measurement_plan": {
                            "observable_plan": None,
                            "basis_measurements": [],
                            "metadata": {},
                        },
                        "metadata": {},
                    },
                    "preparation": {
                        "num_qubits": 2, "method": "test",
                        "claimed_fidelity_loss": 0.0, "found": True,
                        "metrics": {}, "metadata": {},
                    },
                    "experiment_circuit": {
                        "num_qubits": 2, "evolution_present": False,
                        "evolution_qargs": None, "metrics": {}, "metadata": {},
                    },
                    "preparation_preflight": None,
                    "reference": None,
                    "execution": {
                        "source": "q-alchemy-sparse",
                        "source_kind": "ideal-simulator",
                        "observations": None,
                        "basis_distributions": [],
                        "metadata": {},
                    },
                    "observable_error": None,
                    "distribution_errors": {},
                    "estimate": None,
                    "held_out_verification_error": None,
                    "warnings": [],
                },
            }
        )
        self.assertEqual(report.mode, "ideal-simulation")
        self.assertEqual(report.execution.source_kind, "ideal-simulator")

    def test_unsupported_schema_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            ExecutionPlan.from_dict(
                {"schema_version": 4, "kind": "execution-plan", "shots": 1}
            )

    def test_obsolete_runtime_kinds_are_rejected(self):
        for kind in ("resource-simulator", "resource-acquisition"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "obsolete"):
                    Runtime.from_dict({"kind": kind, "resource": "x", "config": {}})


if __name__ == "__main__":
    unittest.main()
