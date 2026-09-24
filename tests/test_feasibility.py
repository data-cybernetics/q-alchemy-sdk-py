import sys
from types import ModuleType
from unittest.mock import patch

from q_alchemy import (
    CircuitCompressionConfig,
    Criterion,
    EvidenceCollectionConfig,
    FeasibilityParams,
    FeasibilityPolicy,
    FeasibilityReport,
    FeasibilityRequest,
    FeasibilityService,
    IBMQuantumCredentials,
    QuantumExecutionPolicy,
    QuantumExperiment,
    SolutionCriteria,
    State,
)


class Link:
    def __init__(self, name):
        self.name = name

    def get_url(self):
        return f"https://work/{self.name}"


class Step:
    def self_link(self):
        return "https://step/assess"


class FakeJob:
    last = None

    def __init__(self, client=None):
        self.client = client
        self.job_hco = object()
        self.created = None
        FakeJob.last = self

    def create_and_configure_rapidly(self, **kwargs):
        self.created = kwargs

    def wait_for_state(self, *args, **kwargs):
        return None


EXPERIMENT_REPORT = {
    "schema_version": 3,
    "kind": "experiment-report",
    "report": {
        "generated_at": "2026-09-02T00:00:00Z",
        "mode": "ideal-simulation",
        "experiment": {
            "target_num_qubits": 1,
            "evolution_present": False,
            "measurement_plan": {
                "observable_plan": None,
                "basis_measurements": [],
                "metadata": {},
            },
            "metadata": {},
        },
        "preparation": {
            "num_qubits": 1,
            "method": "test",
            "claimed_fidelity_loss": 0.0,
            "found": True,
            "metrics": {},
            "metadata": {},
        },
        "experiment_circuit": {
            "num_qubits": 1,
            "evolution_present": False,
            "evolution_qargs": None,
            "metrics": {},
            "metadata": {},
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


QUANTUM_CIRCUIT = {
    "schema_version": 1,
    "kind": "quantum-circuit",
    "role": "logical-experiment-circuit",
    "format": "qasm3",
    "evidence_request": "classical-execution",
    "mode": "reference-only",
    "source": "q-alchemy-simulator:SparseAerBackend",
    "qasm": (
        "OPENQASM 3.0;\n"
        'include "stdgates.inc";\n'
        "qubit[2] q;\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
    ),
}

EXPERIMENT_DIAGRAM = {
    "schema_version": 1,
    "kind": "experiment-diagram",
    "title": "Feasibility execution",
    "nodes": [
        {
            "id": "classical",
            "title": "Classical execution",
            "details": [],
            "status": "run",
        },
        {
            "id": "quantum",
            "title": "Quantum execution",
            "details": [],
            "status": "not-needed",
        },
    ],
    "edges": [{"source": "classical", "target": "quantum", "label": None}],
}

REPORT = {
    "schema_version": 1,
    "kind": "feasibility-report",
    "criteria": {"criteria": []},
    "classical": {"status": "feasible"},
    "quantum": {"status": "not-needed"},
    "recommendation": {"compute": "classical", "reason": "classical is enough"},
    "computation_result": {
        "compute": "classical",
        "evidence_request": "classical-execution",
        "mode": "ideal-simulation",
        "source": "quantum-io:classical",
        "meets_requested_criteria": True,
        "quality_status": None,
        "infeasibility_kind": None,
        "inconclusive_kind": None,
        "selected_by_recommendation": True,
        "experiment_report": EXPERIMENT_REPORT,
    },
    "evidence": {"records": []},
    "execution_trace": {
        "attempted_requests": ["classical-execution"],
        "unavailable_requests": [],
        "policy": {
            "classical_first": True,
            "quantum_execution": "when-needed",
            "dense_memory_utilization_fraction": 0.8,
        },
        "experiment_summary": {"num_qubits": 1},
    },
    "quantum_circuit": QUANTUM_CIRCUIT,
    "experiment_diagram": EXPERIMENT_DIAGRAM,
    "next_evidence": [],
    "warnings": [],
    "final": True,
}


def request():
    return FeasibilityRequest(
        criteria=SolutionCriteria.of(Criterion.at_most("quality.observable_rmse", 0.01)),
        evidence_collection=EvidenceCollectionConfig(provider="ibm"),
    )


def test_request_round_trip():
    value = request()
    assert FeasibilityRequest.from_dict(value.to_dict()) == value
    assert FeasibilityRequest.from_json(value.to_json()) == value


def test_simulator_backend_options_round_trip():
    envelope = {
        "format": "qiskit-aer-noise-model-v1",
        "data": {"errors": []},
    }
    value = FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.1),
        evidence_collection=EvidenceCollectionConfig(
            provider="aer",
            backend="aer_simulator",
            backend_options={
                "noise_model": envelope,
                "run_options": {"seed_simulator": 7},
            },
        ),
    )

    payload = value.to_dict()
    assert payload["evidence_collection"]["backend_options"] == {"noise_model": envelope}
    assert payload["evidence_collection"]["execution_options"] == {
        "transpile": True, "run_options": {"seed_simulator": 7},
    }
    assert FeasibilityRequest.from_dict(payload) == value
    assert FeasibilityRequest.from_json(value.to_json()) == value


def test_qtucker_observable_config_round_trip_and_validation():
    config = EvidenceCollectionConfig(
        qtucker_config={"blocks": [[0, 1], [2, 3]], "ranks": [2, 2]},
        qtucker_observable_config={
            "within_block_mode": "all_paulis",
            "cross_block_mode": None,
            "extra_edges": [[0, 2]],
            "validation_edges": [[1, 3]],
            "validation_paulis": "XYZ",
        },
    )

    payload = config.to_dict()
    assert payload["qtucker_observable_config"] == {
        "within_block_mode": "all_paulis",
        "cross_block_mode": None,
        "extra_edges": [[0, 2]],
        "validation_edges": [[1, 3]],
        "validation_paulis": "XYZ",
    }
    assert EvidenceCollectionConfig.from_dict(payload) == config

    try:
        EvidenceCollectionConfig(
            qtucker_observable_config=[("validation_edges", [[0, 1]])]
        )
    except ValueError as exc:
        assert "qtucker_observable_config must be a mapping" in str(exc)
    else:
        raise AssertionError("non-mapping qtucker_observable_config was accepted")


def test_circuit_compression_config_round_trip_and_preserves_disabled_setting():
    value = FeasibilityRequest(
        criteria=SolutionCriteria.common(max_observable_rmse=0.1),
        evidence_collection=EvidenceCollectionConfig(
            provider="ibm",
            circuit_compression=CircuitCompressionConfig(
                enabled=False,
                options={"optimization_level": 2},
            ),
        ),
    )

    payload = value.to_dict()
    assert payload["evidence_collection"]["circuit_compression"] == {
        "enabled": False,
        "options": {"optimization_level": 2},
    }
    assert FeasibilityRequest.from_dict(payload) == value
    assert FeasibilityRequest.from_json(value.to_json()) == value


def test_circuit_compression_config_enforces_feasibility_owned_options():
    assert CircuitCompressionConfig(
        options={"equivalence": "reachable_subspace", "collect_report": True}
    ).options == {}

    for options in (
        {"equivalence": "full_unitary"},
        {"collect_report": False},
    ):
        try:
            CircuitCompressionConfig(options=options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid compression options were accepted: {options!r}")


def test_request_rejects_nonfinite_numeric_controls_locally():
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            Criterion.at_most("quality.observable_rmse", value)
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError(f"non-finite criterion threshold was accepted: {value!r}")

        for options in ({"transpile_options": {"invalid": value}}, {"run_options": {"invalid": value}}):
            try:
                EvidenceCollectionConfig(execution_options={"transpile": True, **options})
            except ValueError as exc:
                assert "JSON-serializable" in str(exc)
            else:
                raise AssertionError(f"non-finite execution option was accepted: {value!r}")


def test_policy_matches_current_feasibility_wire_contract():
    policy = FeasibilityPolicy()
    payload = policy.to_dict()

    assert "allow_noisy_simulation" not in payload
    assert FeasibilityPolicy.from_dict(payload) == policy

    # Payloads persisted before q-alchemy-feasibility 0.6.31 may still carry the
    # removed field. It is ignored for backward compatibility and not re-emitted.
    old_payload = dict(payload)
    old_payload["allow_noisy_simulation"] = False
    restored = FeasibilityPolicy.from_dict(old_payload)
    assert restored == policy
    assert "allow_noisy_simulation" not in restored.to_dict()


def test_quantum_execution_policy_modes_round_trip_without_sdk_interpretation():
    for mode in QuantumExecutionPolicy:
        policy = FeasibilityPolicy(quantum_execution=mode)
        payload = policy.to_dict()
        assert payload["quantum_execution"] == mode.value
        assert FeasibilityPolicy.from_dict(payload) == policy


def test_report_preserves_computation_trace_and_server_diagram():
    report = FeasibilityReport.from_dict(REPORT)

    assert report.execution_trace == REPORT["execution_trace"]
    assert report.computation_result == REPORT["computation_result"]
    assert report.to_dict()["quantum_circuit"] == QUANTUM_CIRCUIT
    assert report.to_dict()["experiment_diagram"] == EXPERIMENT_DIAGRAM

    experiment_report = report.computation_experiment_report
    assert experiment_report is not None
    assert experiment_report.mode == "ideal-simulation"
    assert experiment_report.execution.source_kind == "ideal-simulator"


def test_report_delegates_diagram_deserialization_and_native_rendering():
    calls = []

    class FakeDiagram:
        @classmethod
        def from_dict(cls, payload):
            calls.append(("from_dict", payload))
            return cls()

        def draw(self, *, output, show_title):
            calls.append(("draw", output, show_title))
            return "rendered"

    module = ModuleType("q_alchemy.visualization")
    module.ExperimentDiagram = FakeDiagram

    report = FeasibilityReport.from_dict(REPORT)
    with patch.dict(sys.modules, {"q_alchemy.visualization": module}):
        diagram = report.experiment_diagram
        assert isinstance(diagram, FakeDiagram)
        assert diagram.draw(output="text", show_title=True) == "rendered"

    assert calls == [
        ("from_dict", EXPERIMENT_DIAGRAM),
        ("draw", "text", True),
    ]


def test_report_reconstructs_quantum_circuit_and_uses_qiskit_rendering():
    calls = []

    class FakeCircuit:
        def draw(self, *, output, **kwargs):
            calls.append(("draw", output, kwargs))
            return "circuit-rendered"

    class FakeQasm3:
        @staticmethod
        def loads(payload):
            calls.append(("loads", payload))
            return FakeCircuit()

    qiskit = ModuleType("qiskit")
    qiskit.qasm3 = FakeQasm3

    report = FeasibilityReport.from_dict(REPORT)
    with patch.dict(sys.modules, {"qiskit": qiskit}):
        circuit = report.quantum_circuit
        assert isinstance(circuit, FakeCircuit)
        assert circuit.draw(output="text", fold=120) == "circuit-rendered"

    assert calls == [
        ("loads", QUANTUM_CIRCUIT["qasm"]),
        ("draw", "text", {"fold": 120}),
    ]


def test_legacy_sweep_controls_are_ignored_in_saved_json():
    config = EvidenceCollectionConfig.from_dict({
        "resource_sweep_scales": [1.0, 0.5],
        "attribution_noise_scales": [0.5],
        "attribution_shot_multipliers": [2.0],
        "attribution_dimensions": ["shots"],
    })
    assert config == EvidenceCollectionConfig()
    assert not any("sweep" in key or "attribution" in key for key in config.to_dict())


def test_service_submits_one_remote_feasibility_job_and_secret_credentials():
    service = FeasibilityService(
        FeasibilityParams(remove_data=False),
        ibm_credentials=IBMQuantumCredentials(
            token="secret",
            instance="crn:test-instance",
            channel="ibm_quantum_platform",
        ),
        client=object(),
    )
    uploads = []
    payloads = {}

    def upload(name, payload, *, secret=False):
        uploads.append((name, secret))
        payloads[name] = payload
        return Link(name)

    service._upload_json = upload
    service._step = lambda name: Step()
    experiment = QuantumExperiment(target=State.dense([1, 0]))

    with patch("q_alchemy.feasibility.Job", FakeJob), patch(
        "q_alchemy.feasibility._download_json_output", return_value=REPORT
    ):
        job = service.analyze(
            experiment,
            request(),
            include_quantum_circuits=True,
        )
        result = job.result()

    assert result.final is True
    assert result.recommendation.compute.value == "classical"
    assert result.classical_status.value == "feasible"
    assert result.quantum_status.value == "not-needed"
    assert uploads == [
        ("quantum_experiment.json", False),
        ("feasibility_request.json", False),
        ("ibm_credentials.json", True),
    ]
    assert payloads["feasibility_request.json"]["service_options"] == {
        "include_quantum_circuits": True,
    }
    assert payloads["ibm_credentials.json"] == {
        "token": "secret",
        "instance": "crn:test-instance",
        "channel": "ibm_quantum_platform",
    }
    assert "parameters" not in FakeJob.last.created
    assert len(FakeJob.last.created["input_data_slots"]) == 3


def test_service_disables_quantum_circuit_return_by_default():
    service = FeasibilityService(FeasibilityParams(remove_data=False), client=object())
    payloads = {}

    def upload(name, payload, *, secret=False):
        payloads[name] = payload
        return Link(name)

    service._upload_json = upload
    service._step = lambda name: Step()
    with patch("q_alchemy.feasibility.Job", FakeJob):
        service.analyze(QuantumExperiment(target=State.dense([1, 0])), request())

    assert payloads["feasibility_request.json"]["service_options"] == {
        "include_quantum_circuits": False,
    }


def test_credentials_are_optional_for_classical_short_circuit_request():
    service = FeasibilityService(FeasibilityParams(remove_data=False), client=object())
    uploads = []
    service._upload_json = lambda name, payload, secret=False: (uploads.append((name, secret)) or Link(name))
    service._step = lambda name: Step()
    with patch("q_alchemy.feasibility.Job", FakeJob):
        service.analyze(QuantumExperiment(target=State.dense([1, 0])), request())
    assert uploads == [
        ("quantum_experiment.json", False),
        ("feasibility_request.json", False),
    ]


def test_report_format_summary_matches_feasibility_core_classical_style():
    payload = {
        "schema_version": 1,
        "kind": "feasibility-report",
        "criteria": {"criteria": []},
        "classical": {
            "status": "feasible",
            "quality_status": "satisfied",
            "criteria": {
                "complete": True,
                "satisfied": True,
                "violated": False,
                "results": [
                    {
                        "criterion": {
                            "metric": "quality.final_state_fidelity",
                            "relation": ">=",
                            "threshold": 0.999,
                            "label": None,
                            "required": True,
                        },
                        "status": "satisfied",
                        "evidence": {
                            "metric": "quality.final_state_fidelity",
                            "value": 1.0,
                            "kind": "simulated",
                            "source": "q-alchemy-simulator:SparseAerBackend",
                            "scope": "classical",
                            "unit": None,
                            "uncertainty": None,
                            "generated_at": None,
                            "lower_bound": False,
                            "metadata": {},
                        },
                    }
                ],
            },
            "available_resources": {
                "total_memory_bytes": 32 * 1024**3,
                "available_memory_bytes": 24 * 1024**3,
                "cpu_cores": 16,
                "gpu_memory_bytes": None,
                "accelerator": None,
                "metadata": {},
            },
            "selected_method": "q-alchemy-simulator:SparseAerBackend",
            "required_resources": None,
            "resource_gap": None,
            "infeasibility_kind": None,
            "notes": [],
        },
        "quantum": {
            "status": "not-needed",
            "quality_status": "not-assessed",
            "criteria": {
                "complete": False,
                "satisfied": False,
                "violated": False,
                "results": [
                    {
                        "criterion": {
                            "metric": "quality.final_state_fidelity",
                            "relation": ">=",
                            "threshold": 0.999,
                            "label": None,
                            "required": True,
                        },
                        "status": "missing",
                        "evidence": None,
                    }
                ],
            },
            "model_criteria": None,
            "available_resources": None,
            "backend": None,
            "execution_performed": False,
            "assessment_basis": None,
            "required_resources": None,
            "resource_gap": None,
            "resource_attribution": None,
            "infeasibility_kind": None,
            "inconclusive_kind": None,
            "notes": [],
        },
        "recommendation": {
            "compute": "classical",
            "reason": (
                "A quantum computer is not required for this workload under the "
                "stated solution criteria and available classical resources."
            ),
        },
        "computation_result": None,
        "evidence": {"records": []},
        "execution_trace": None,
        "quantum_circuit": QUANTUM_CIRCUIT,
        "experiment_diagram": EXPERIMENT_DIAGRAM,
        "next_evidence": [],
        "warnings": [],
        "final": True,
    }

    assert FeasibilityReport.from_dict(payload).format_summary() == (
        "COMPUTE FEASIBILITY\n"
        "-------------------\n"
        "\n"
        "CLASSICAL\n"
        "  Status: feasible\n"
        "  Quality: satisfied\n"
        "  Method: q-alchemy-simulator:SparseAerBackend\n"
        "  Resource criteria:\n"
        "    Classical RAM limit: 24 GiB\n"
        "  Quality criteria:\n"
        "    Final-state fidelity: 1 >= 0.999 [SATISFIED]\n"
        "\n"
        "QUANTUM\n"
        "  Status: not-needed\n"
        "  Quality: not-assessed\n"
        "  Backend: not selected\n"
        "  Assessment basis: not available\n"
        "  QPU execution performed: False\n"
        "\n"
        "RECOMMENDATION\n"
        "  classical: A quantum computer is not required for this workload under "
        "the stated solution criteria and available classical resources."
    )


def test_report_format_summary_matches_feasibility_core_resource_and_model_style():
    criterion = {
        "metric": "quality.observable_rmse",
        "relation": "<=",
        "threshold": 0.01,
        "label": None,
        "required": True,
    }
    payload = {
        "schema_version": 1,
        "kind": "feasibility-report",
        "criteria": {"criteria": [criterion]},
        "classical": {
            "status": "infeasible",
            "quality_status": "not-assessed",
            "criteria": {
                "complete": False,
                "satisfied": False,
                "violated": False,
                "results": [
                    {"criterion": criterion, "status": "missing", "evidence": None}
                ],
            },
            "available_resources": {
                "total_memory_bytes": 16 * 1024**2,
                "available_memory_bytes": 16 * 1024**2,
                "cpu_cores": 2,
                "gpu_memory_bytes": None,
                "accelerator": None,
                "metadata": {},
            },
            "selected_method": "q-alchemy-sparse",
            "required_resources": {
                "memory_bytes": 32 * 1024**2,
                "memory_is_lower_bound": True,
                "cpu_cores": None,
                "gpu_memory_bytes": None,
                "expected_runtime_sec": None,
                "method": "q-alchemy-sparse",
                "metadata": {},
            },
            "resource_gap": None,
            "infeasibility_kind": "resource",
            "notes": ["Sparse simulation exceeded the configured memory allocation."],
        },
        "quantum": {
            "status": "infeasible",
            "quality_status": "violated",
            "criteria": {
                "complete": False,
                "satisfied": False,
                "violated": False,
                "results": [
                    {"criterion": criterion, "status": "missing", "evidence": None}
                ],
            },
            "model_criteria": {
                "complete": True,
                "satisfied": False,
                "violated": True,
                "results": [
                    {
                        "criterion": criterion,
                        "status": "violated",
                        "evidence": {
                            "metric": "quality.observable_rmse",
                            "value": 0.04,
                            "kind": "simulated",
                            "source": "calibrated-model",
                            "scope": "quantum-model",
                            "unit": None,
                            "uncertainty": None,
                            "generated_at": "2026-09-02T12:00:00Z",
                            "lower_bound": False,
                            "metadata": {},
                        },
                    }
                ],
            },
            "available_resources": {
                "backend": "ibm_test",
                "qubits": 127,
                "operational": True,
                "pending_jobs": 0,
                "median_t1_sec": 0.0001,
                "median_t2_sec": 0.00008,
                "median_one_qubit_error": 0.001,
                "median_two_qubit_error": 0.008,
                "median_readout_error": 0.02,
                "coupling_edge_count": None,
                "basis_gates": [],
                "metadata": {},
            },
            "backend": "ibm_test",
            "execution_performed": False,
            "assessment_basis": "calibrated-model",
            "required_resources": {
                "qubits": 40,
                "max_one_qubit_error": None,
                "max_two_qubit_error": 0.004,
                "min_t1_sec": None,
                "min_t2_sec": None,
                "max_readout_error": None,
                "required_shots": None,
                "max_balanced_quantum_noise_scale": None,
                "metadata": {},
            },
            "resource_gap": {
                "additional_qubits": None,
                "max_one_qubit_error": None,
                "max_two_qubit_error": 0.004,
                "min_t1_sec": None,
                "min_t2_sec": None,
                "max_readout_error": None,
                "additional_shots": None,
                "required_balanced_quantum_noise_improvement_factor": 2.0,
                "balanced_quantum_noise_improvement_is_lower_bound": False,
                "metadata": {},
            },
            "resource_attribution": {
                "complete": True,
                "dominant_dimension": "two-qubit-gate-error",
                "single_resource_remedies": ["two-qubit-gate-error"],
                "items": [],
                "notes": [],
            },
            "infeasibility_kind": "quality",
            "inconclusive_kind": None,
            "notes": [],
        },
        "recommendation": {
            "compute": "none-available",
            "reason": "Neither tested compute path satisfies the requested criteria.",
        },
        "computation_result": None,
        "evidence": {
            "records": [
                {
                    "metric": "classical.approximation.amplitude_capping_occurred",
                    "value": True,
                    "kind": "simulated",
                    "source": "q-alchemy-sparse",
                    "scope": "classical",
                    "unit": None,
                    "uncertainty": None,
                    "generated_at": "2026-09-02T11:00:00Z",
                    "lower_bound": False,
                    "metadata": {},
                },
                {
                    "metric": "quantum.model.execution_completed",
                    "value": True,
                    "kind": "simulated",
                    "source": "calibrated-model",
                    "scope": "quantum-model",
                    "unit": None,
                    "uncertainty": None,
                    "generated_at": "2026-09-02T12:00:00Z",
                    "lower_bound": False,
                    "metadata": {},
                },
            ]
        },
        "execution_trace": None,
        "quantum_circuit": QUANTUM_CIRCUIT,
        "experiment_diagram": EXPERIMENT_DIAGRAM,
        "next_evidence": [
            {
                "kind": "quantum-resource-attribution-sweep",
                "reason": "Quantify the remaining device resource gap.",
                "parameters": {},
            }
        ],
        "warnings": [],
        "final": False,
    }

    assert FeasibilityReport.from_dict(payload).format_summary() == (
        "COMPUTE FEASIBILITY\n"
        "-------------------\n"
        "\n"
        "CLASSICAL\n"
        "  Status: resource infeasible\n"
        "  Reason: Sparse simulation exceeded the configured memory allocation.\n"
        "  Quality: not-assessed\n"
        "  Method: q-alchemy-sparse\n"
        "  Resource criteria:\n"
        "    Classical RAM limit: 16 MiB\n"
        "    Sparse simulation memory: > 32 MiB required (lower bound); 16 MiB available [VIOLATED]\n"
        "\n"
        "QUANTUM\n"
        "  Status: quality infeasible\n"
        "  Quality: violated\n"
        "  Reason: Backend-calibrated noisy simulation violates the required Observable RMSE criterion.\n"
        "  Quality reason: Backend-calibrated noisy simulation violates the required Observable RMSE criterion.\n"
        "  Backend: ibm_test\n"
        "  Assessment basis: calibrated-model\n"
        "  Noisy simulation performed: True\n"
        "  QPU execution performed: False\n"
        "  Median 2Q error: 0.008\n"
        "  Median T2: 8e-05 s\n"
        "  Resource criteria:\n"
        "    Qubit capacity: 127 available >= 40 required [SATISFIED]\n"
        "    Backend operational: yes [SATISFIED]\n"
        "    Median 2Q error: 0.008 <= 0.004 [VIOLATED]\n"
        "  Quality criteria (calibrated model):\n"
        "    Observable RMSE: 0.04 <= 0.01 [VIOLATED]\n"
        "\n"
        "RECOMMENDATION\n"
        "  none-available: Neither tested compute path satisfies the requested criteria.\n"
        "\n"
        "NEXT EVIDENCE\n"
        "  - quantum-resource-attribution-sweep: Quantify the remaining device resource gap."
    )


def test_report_exposes_and_formats_circuit_compression_summary():
    payload = dict(REPORT)
    payload["classical"] = {
        **REPORT["classical"],
        "quality_status": "satisfied",
        "selected_method": "q-alchemy-sparse",
    }
    payload["circuit_compression"] = {
        "attempted": True,
        "applied": True,
        "changed": True,
        "circuit_used": "compressor-output",
        "exact": True,
        "equivalence": "reachable_subspace",
        "input_semantics": "canonical_zero_state",
        "input_metrics": {
            "operations": 12,
            "one_qubit_operations": 5,
            "two_qubit_operations": 7,
            "cx": 7,
            "depth": 10,
            "two_qubit_depth": 6,
            "counts": {"cx": 7},
        },
        "compressed_metrics": {
            "operations": 8,
            "one_qubit_operations": 4,
            "two_qubit_operations": 4,
            "cx": 4,
            "depth": 7,
            "two_qubit_depth": 4,
            "counts": {"cx": 4},
        },
        "accepted_regions": 2,
        "reason": "compressor output selected; circuit changed",
        "options": {},
    }

    report = FeasibilityReport.from_dict(payload)
    assert report.circuit_compression == payload["circuit_compression"]
    summary = report.format_summary()
    assert "CIRCUIT COMPRESSION" in summary
    assert "Changed: True" in summary
    assert "Circuit used: compressor-output" in summary
    assert "1Q operations: 5 -> 4" in summary
    assert "2Q operations: 7 -> 4" in summary
    assert "Depth: 10 -> 7" in summary
    assert "Accepted regions: 2" in summary
    assert "CLASSICAL\n  Status: feasible\n  Quality: satisfied" in summary


def test_summary_latest_evidence_compares_timestamp_instants_not_strings():
    payload = dict(REPORT)
    payload["classical"] = {
        **REPORT["classical"],
        "quality_status": "satisfied",
    }
    payload["quantum"] = {
        **REPORT["quantum"],
        "quality_status": "not-assessed",
        "assessment_basis": "calibrated-model",
    }
    payload["evidence"] = {
        "records": [
            {
                "metric": "quantum.model.execution_completed",
                "value": True,
                "scope": "quantum-model",
                "generated_at": "2026-09-01T10:00:00+02:00",
            },
            {
                "metric": "quantum.model.execution_completed",
                "value": False,
                "scope": "quantum-model",
                "generated_at": "2026-09-01T09:30:00+00:00",
            },
        ]
    }

    # 09:30 UTC is later than 10:00 +02:00 (08:00 UTC). The older lexical
    # ordering would incorrectly select the first record and print this line.
    assert "Noisy simulation performed: True" not in (
        FeasibilityReport.from_dict(payload).format_summary()
    )



def test_report_format_summary_matches_feasibility_core_simulated_quantum_style():
    criterion = {
        "metric": "quality.observable_rmse",
        "relation": "<=",
        "threshold": 0.01,
        "label": None,
        "required": True,
    }
    payload = {
        "schema_version": 1,
        "kind": "feasibility-report",
        "criteria": {"criteria": [criterion]},
        "classical": {
            "status": "infeasible",
            "quality_status": "not-assessed",
            "criteria": {
                "complete": False,
                "satisfied": False,
                "violated": False,
                "results": [
                    {"criterion": criterion, "status": "missing", "evidence": None}
                ],
            },
            "available_resources": {
                "total_memory_bytes": None,
                "available_memory_bytes": None,
                "cpu_cores": None,
                "gpu_memory_bytes": None,
                "accelerator": None,
                "metadata": {},
            },
            "selected_method": None,
            "required_resources": None,
            "resource_gap": None,
            "infeasibility_kind": "resource",
            "notes": ["Classical path unavailable."],
        },
        "quantum": {
            "status": "infeasible",
            "quality_status": "violated",
            "criteria": {
                "complete": False,
                "satisfied": False,
                "violated": False,
                "results": [
                    {"criterion": criterion, "status": "missing", "evidence": None}
                ],
            },
            "model_criteria": {
                "complete": True,
                "satisfied": False,
                "violated": True,
                "results": [
                    {
                        "criterion": criterion,
                        "status": "violated",
                        "evidence": {
                            "metric": "quality.observable_rmse",
                            "value": 0.02,
                            "kind": "simulated",
                            "source": "aer_simulator",
                            "scope": "quantum-model",
                            "unit": None,
                            "uncertainty": None,
                            "generated_at": "2026-09-16T12:00:00Z",
                            "lower_bound": False,
                            "metadata": {},
                        },
                    }
                ],
            },
            "available_resources": None,
            "backend": "aer_simulator",
            "execution_performed": False,
            "assessment_basis": "simulated-quantum",
            "required_resources": None,
            "resource_gap": None,
            "resource_attribution": None,
            "infeasibility_kind": "quality",
            "inconclusive_kind": None,
            "notes": [],
        },
        "recommendation": {
            "compute": "none-available",
            "reason": "Neither tested compute path satisfies the requested criteria.",
        },
        "evidence": {"records": []},
        "next_evidence": [],
        "warnings": [],
        "final": True,
    }

    assert FeasibilityReport.from_dict(payload).format_summary() == (
        "COMPUTE FEASIBILITY\n"
        "-------------------\n"
        "\n"
        "CLASSICAL\n"
        "  Status: resource infeasible\n"
        "  Reason: Classical path unavailable.\n"
        "  Quality: not-assessed\n"
        "  Method: not selected\n"
        "  Resource criteria:\n"
        "    Classical RAM limit: not available\n"
        "\n"
        "QUANTUM\n"
        "  Status: quality infeasible\n"
        "  Quality: violated\n"
        "  Reason: Configured quantum simulator violates the required Observable RMSE criterion.\n"
        "  Quality reason: Configured quantum simulator violates the required Observable RMSE criterion.\n"
        "  Backend: aer_simulator\n"
        "  Assessment basis: simulated-quantum\n"
        "  QPU execution performed: False\n"
        "  Quality criteria (quantum simulator):\n"
        "    Observable RMSE: 0.02 <= 0.01 [VIOLATED]\n"
        "\n"
        "RECOMMENDATION\n"
        "  none-available: Neither tested compute path satisfies the requested criteria."
    )


def test_report_summary_matches_core_classical_resource_telemetry_and_memory_bound():
    payload = dict(REPORT)
    payload["classical"] = {
        "status": "feasible",
        "quality_status": "not-assessed",
        "criteria": None,
        "available_resources": {
            "total_memory_bytes": 32 * 1024**3,
            "available_memory_bytes": 24 * 1024**3,
            "cpu_cores": 16,
            "gpu_memory_bytes": None,
            "accelerator": None,
            "metadata": {},
        },
        "selected_method": "q-alchemy-sparse",
        "required_resources": {
            "memory_bytes": 8 * 1024**3,
            "memory_is_lower_bound": True,
            "cpu_cores": None,
            "gpu_memory_bytes": None,
            "expected_runtime_sec": None,
            "method": "q-alchemy-sparse",
            "metadata": {},
        },
        "resource_gap": None,
        "infeasibility_kind": None,
        "notes": [],
    }
    payload["quantum"] = {
        **REPORT["quantum"],
        "quality_status": "not-assessed",
    }
    payload["evidence"] = {
        "records": [
            {
                "metric": "classical.resources.simulator_budget_bytes",
                "value": 20 * 1024**3,
                "source": "q-alchemy-sparse",
                "scope": "classical",
                "generated_at": "2026-09-16T12:00:00Z",
            },
            {
                "metric": "classical.resources.estimated_peak_memory_bytes",
                "value": 12 * 1024**3,
                "source": "q-alchemy-sparse",
                "scope": "classical",
                "generated_at": "2026-09-16T12:00:00Z",
            },
            {
                "metric": "classical.resources.simulator_budget_bytes",
                "value": 1 * 1024**3,
                "source": "preparation-preflight",
                "scope": "classical",
                "generated_at": "2026-09-16T13:00:00Z",
            },
        ]
    }

    summary = FeasibilityReport.from_dict(payload).format_summary()
    assert "  Resource criteria:\n" in summary
    assert "    Classical RAM limit: 24 GiB\n" in summary
    assert "    Detected simulator host budget: 20 GiB\n" in summary
    assert "    Sparse memory estimate at amplitude limit: 12 GiB\n" in summary
    assert (
        "    Classical memory requirement: 8 GiB (lower bound); "
        "<= 24 GiB available [INCONCLUSIVE]\n"
    ) in summary
    assert "Detected simulator host budget: 1 GiB" not in summary


def test_report_summary_matches_core_quantum_resource_criterion_order():
    payload = dict(REPORT)
    payload["classical"] = {
        **REPORT["classical"],
        "quality_status": "not-assessed",
        "available_resources": {
            "available_memory_bytes": None,
            "cpu_cores": None,
            "gpu_memory_bytes": None,
        },
    }
    payload["quantum"] = {
        "status": "infeasible",
        "quality_status": "not-assessed",
        "criteria": None,
        "model_criteria": None,
        "available_resources": {
            "backend": "ibm_test",
            "qubits": 127,
            "operational": True,
            "median_one_qubit_error": 0.001,
            "median_two_qubit_error": 0.008,
            "median_t1_sec": 0.0001,
            "median_t2_sec": 0.00008,
            "median_readout_error": 0.02,
        },
        "backend": "ibm_test",
        "execution_performed": False,
        "assessment_basis": "static-hard-constraint",
        "required_resources": {
            "qubits": 40,
            "max_one_qubit_error": 0.002,
            "max_two_qubit_error": 0.004,
            "min_t1_sec": 0.00009,
            "min_t2_sec": 0.00007,
            "max_readout_error": 0.01,
        },
        "resource_gap": None,
        "resource_attribution": None,
        "infeasibility_kind": "resource",
        "inconclusive_kind": None,
        "notes": ["Backend fails required hardware constraints."],
    }

    summary = FeasibilityReport.from_dict(payload).format_summary()
    labels = [
        "Qubit capacity:",
        "Backend operational:",
        "Median 1Q error:",
        "Median 2Q error:",
        "Median T1:",
        "Median T2:",
        "Median readout error:",
    ]
    quantum_section = summary.split("\nQUANTUM\n", 1)[1].split("\nRECOMMENDATION\n", 1)[0]
    resource_section = quantum_section.split("  Resource criteria:\n", 1)[1]
    positions = [resource_section.index(label) for label in labels]
    assert positions == sorted(positions)

def test_report_summary_surfaces_quantum_execution_unavailable_reason():
    payload = {
        "schema_version": 1,
        "kind": "feasibility-report",
        "criteria": {"criteria": []},
        "classical": {"status": "feasible", "selected_method": "q-alchemy-sparse"},
        "quantum": {
            "status": "unknown",
            "quality_status": "not-assessed",
            "backend": "ibm_test",
            "execution_performed": False,
            "assessment_basis": "execution-unavailable",
            "inconclusive_kind": "execution",
            "notes": [
                "IBM Quantum error 1370: The allocation for this instance's plan on the account is not active. Solution: Submit the workload to a different plan"
            ],
        },
        "recommendation": {
            "compute": "classical",
            "reason": "Classical computing satisfies the stated solution criteria.",
        },
        "evidence": {"records": []},
        "next_evidence": [],
        "warnings": [],
        "final": True,
    }

    summary = FeasibilityReport.from_dict(payload).format_summary()
    assert "Status: execution inconclusive" in summary
    assert "Reason: IBM Quantum error 1370:" in summary
    assert "Assessment basis: execution-unavailable" in summary
    assert "QPU execution performed: False" in summary


def test_report_summary_surfaces_unavailable_quantum_target_inside_quantum_path():
    payload = {
        "schema_version": 1,
        "kind": "feasibility-report",
        "criteria": {"criteria": []},
        "classical": {
            "status": "feasible",
            "selected_method": "quantum-io:reference-simulation",
        },
        "quantum": {
            "status": "unknown",
            "quality_status": "not-assessed",
            "backend": "xxxxxx",
            "execution_performed": False,
            "assessment_basis": "execution-unavailable",
            "inconclusive_kind": "execution",
            "notes": ["requested backend 'xxxxxx' is not accessible"],
        },
        "recommendation": {
            "compute": "classical",
            "reason": (
                "Classical computing satisfies the stated solution criteria. "
                "The requested quantum comparison could not be completed because "
                "the configured quantum execution target is unavailable."
            ),
        },
        "evidence": {"records": []},
        "next_evidence": [],
        "warnings": ["requested backend 'xxxxxx' is not accessible"],
        "final": True,
    }

    summary = FeasibilityReport.from_dict(payload).format_summary()
    assert "Status: execution inconclusive" in summary
    assert "Quality: not-assessed" in summary
    assert "Reason: requested backend 'xxxxxx' is not accessible" in summary
    assert "Backend: xxxxxx" in summary
    assert "Assessment basis: execution-unavailable" in summary
    assert "QPU execution performed: False" in summary
