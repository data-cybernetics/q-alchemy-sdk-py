"""Offline and opt-in live tests for the hosted Quantum I/O SDK client."""

from __future__ import annotations

import json
import os
import time
import unittest
import warnings

import httpx
from unittest.mock import patch

from q_alchemy import (
    BasisMeasurement,
    ExecutionPlan,
    ExperimentReport,
    IBMQuantumCredentials,
    MeasurementPlan,
    QuantumExperiment,
    QuantumIOParams,
    QuantumIOService,
    Runtime,
    State,
    noisy_backend_execution_plan,
)
from q_alchemy.initialize import _version_sort_key
from q_alchemy.quantum_io import (
    _download_json_output,
    EXPERIMENT_INPUT_ALIAS,
    EXECUTION_PLAN_INPUT_ALIAS,
    IBM_CREDENTIALS_INPUT_ALIAS,
    EXPERIMENT_REPORT_OUTPUT_ALIAS,
    NOISY_BACKEND_SIMULATOR_RESOURCE,
    QuantumBackend,
    local_simulator_execution_plan,
    quantum_backend_execution_plan,
)


def _bell_experiment() -> QuantumExperiment:
    amplitude = 2.0**-0.5
    return QuantumExperiment(
        target=State.dense([amplitude, 0.0, 0.0, amplitude]),
        measurement_plan=MeasurementPlan(
            basis_measurements=(BasisMeasurement("q0-q1", (0, 1)),),
        ),
        metadata={"test": "sdk-quantum-io-live"},
    )


def _report_payload(
    *,
    mode: str = "ideal-simulation",
    source_kind: str | None = "ideal-simulator",
) -> dict:
    execution = None
    if source_kind is not None:
        execution = {
            "source": "test-source",
            "source_kind": source_kind,
            "observations": None,
            "basis_distributions": [],
            "metadata": {},
        }
    return {
        "schema_version": 3,
        "kind": "experiment-report",
        "report": {
            "generated_at": "2026-08-27T00:00:00Z",
            "mode": mode,
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
                "num_qubits": 2,
                "method": "test",
                "claimed_fidelity_loss": 0.0,
                "found": True,
                "metrics": {},
                "metadata": {},
            },
            "experiment_circuit": {
                "num_qubits": 2,
                "evolution_present": False,
                "evolution_qargs": None,
                "metrics": {},
                "metadata": {},
            },
            "preparation_preflight": None,
            "reference": None,
            "execution": execution,
            "observable_error": None,
            "distribution_errors": {},
            "estimate": None,
            "held_out_verification_error": None,
            "warnings": ["warning"],
        },
    }


class _DeleteAction:
    def __init__(self, owner):
        self.owner = owner

    def execute(self):
        if not self.owner.deletable:
            raise RuntimeError("action is not available")
        self.owner.events.append(f"workdata:{self.owner.name}")
        self.owner.deleted = True


class _AllowDeletionAction:
    def __init__(self, owner):
        self.owner = owner

    def execute(self):
        self.owner.events.append(f"allow-deletion:{self.owner.name}")
        self.owner.deletable = True


class _WorkDataHco:
    def __init__(self, owner):
        self.is_deletable = owner.deletable
        self.allow_deletion_action = _AllowDeletionAction(owner)
        self.delete_action = _DeleteAction(owner)


class _WorkData:
    def __init__(self, url: str, events=None, name=None, *, deletable=False):
        self._url = url
        self.events = events
        self.name = name
        self.deletable = deletable
        self.deleted = False

    def get_url(self):
        return self._url

    def navigate(self):
        if self.events is None or self.name is None:
            raise AssertionError("this fake WorkData is not configured for deletion")
        return _WorkDataHco(self)


class _OutputSlot:
    def __init__(self, *work_datas):
        self.assigned_workdatas = list(work_datas)


class _DownloadLink:
    def __init__(self, payload: bytes):
        self._payload = payload

    def download(self) -> bytes:
        return self._payload


class _OutputWorkData:
    """Output WorkData with the attributes _download_json_output actually reads."""

    def __init__(self, name, payload: bytes):
        self.name = name
        self.size_in_bytes = len(payload)
        self.download_link = _DownloadLink(payload)


class _Step:
    def self_link(self):
        return "https://jobs.example/processing-step"


class _FakePineJob:
    last: "_FakePineJob | None" = None
    cleanup_events = None

    #: When False the fake reproduces pinexq's best-effort behaviour for a Job
    #: whose Delete action is unavailable: warn, keep ``job_hco``, do not raise.
    deletion_allowed = True

    def __init__(self, client=None):
        self.client = client
        self.created = None
        self.job_hco = object()  # pinexq clears this only on real deletion
        _FakePineJob.last = self

    def create_and_configure_rapidly(self, **kwargs):
        self.created = kwargs
        return self

    def wait_for_state(self, *args, **kwargs):
        self.wait_args = args
        self.wait_kwargs = kwargs
        return self

    def get_output_data_slots(self):
        output = _WorkData(
            "https://workdata.example/experiment_report.json",
            self.cleanup_events,
            EXPERIMENT_REPORT_OUTPUT_ALIAS,
            deletable=True,
        )
        return [_OutputSlot(output)]

    def refresh(self):
        return self

    def delete_with_associated(
        self,
        *,
        delete_subjobs_with_data,
        delete_input_workdata,
        delete_output_workdata,
    ):
        self.delete_with_associated_kwargs = {
            "delete_subjobs_with_data": delete_subjobs_with_data,
            "delete_input_workdata": delete_input_workdata,
            "delete_output_workdata": delete_output_workdata,
        }
        if not self.deletion_allowed:
            warnings.warn("Could not delete job: https://jobs.example/job/1")
            if self.cleanup_events is not None:
                self.cleanup_events.append("job-delete-refused")
            return None
        self.deleted = True
        self.job_hco = None
        if self.cleanup_events is not None:
            self.cleanup_events.append("job")
        return self


class _ServiceRecorder:
    def __init__(self):
        self.calls = []

    def run(self, experiment, execution_plan=None, *, credentials=None, **kwargs):
        self.calls.append((experiment, execution_plan, credentials, kwargs))
        return "job"


def _slot_dump(slot):
    if hasattr(slot, "model_dump"):
        return slot.model_dump(by_alias=True)
    if hasattr(slot, "dict"):
        return slot.dict(by_alias=True)
    return vars(slot)


class TestProcessingStepVersionSelection(unittest.TestCase):
    def test_version_key_uses_numeric_order(self):
        versions = ["0.2.0", "0.10.0", "0.9.0", "0.10.0rc1"]
        self.assertEqual(max(versions, key=_version_sort_key), "0.10.0")


class TestTypedContract(unittest.TestCase):
    def test_experiment_roundtrip_hides_wire_details_from_constructor(self):
        experiment = _bell_experiment()
        payload = experiment.to_dict()
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["kind"], "quantum-experiment")
        self.assertEqual(payload["target"]["representation"], "dense")
        self.assertEqual(QuantumExperiment.from_dict(payload), experiment)

    def test_experiment_report_is_fully_typed(self):
        result = ExperimentReport.from_dict(_report_payload())
        self.assertEqual(result.mode, "ideal-simulation")
        self.assertIsNotNone(result.execution)
        self.assertEqual(result.execution.source_kind, "ideal-simulator")
        self.assertEqual(result.preparation.metrics.cx_count, None)
        self.assertEqual(result.warnings, ("warning",))


class TestExecutionPlans(unittest.TestCase):
    def test_local_simulator_plan(self):
        plan = local_simulator_execution_plan(shots=256)
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(plan.shots, 256)
        self.assertEqual(plan.acquisition.kind, "resource")
        self.assertEqual(plan.acquisition.resource_name, "local-simulator")
        self.assertEqual(dict(plan.acquisition.config), {})

    def test_quantum_backend_plan(self):
        plan = quantum_backend_execution_plan(
            provider="ibm",
            backend="ibm_test",
            shots=4096,
        )
        self.assertEqual(plan.shots, 4096)
        self.assertEqual(plan.acquisition.resource_name, "quantum-backend")
        self.assertEqual(
            dict(plan.acquisition.config),
            {"provider": "ibm", "backend": "ibm_test"},
        )

    def test_quantum_backend_plan_least_busy(self):
        plan = quantum_backend_execution_plan(provider="IBM", least_busy=True)
        self.assertEqual(
            dict(plan.acquisition.config),
            {"provider": "ibm", "least_busy": True},
        )

    def test_noisy_backend_plan_adds_ideal_reference_and_optional_estimator(self):
        plan = noisy_backend_execution_plan(
            provider="ibm",
            backend="ibm_test",
            shots=2048,
            ideal_reference=True,
            estimator=True,
        )
        self.assertEqual(plan.acquisition.resource_name, NOISY_BACKEND_SIMULATOR_RESOURCE)
        self.assertEqual(plan.reference.kind, "q-alchemy-sparse")
        self.assertEqual(plan.estimator.kind, "qtucker")
        self.assertEqual(plan.shots, 2048)

    def test_backend_plan_requires_one_strategy(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            quantum_backend_execution_plan(provider="ibm")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            noisy_backend_execution_plan(
                provider="ibm", backend="ibm_test", least_busy=True
            )


class TestCredentials(unittest.TestCase):
    def test_quantum_io_params_repr_redacts_api_key(self):
        params = QuantumIOParams(
            api_key="qa-live-super-secret",
            added_headers={"authorization": "Bearer another-secret"},
        )
        text = repr(params)
        self.assertNotIn("qa-live-super-secret", text)
        self.assertNotIn("another-secret", text)
        # Redaction must not make the values unreadable.
        self.assertEqual(params.api_key, "qa-live-super-secret")
        self.assertIn("host=", text)

    def test_ibm_credentials_repr_redacts_token(self):
        credentials = IBMQuantumCredentials(
            token="super-secret",
            instance="instance",
            channel="ibm_quantum_platform",
        )
        self.assertNotIn("super-secret", repr(credentials))
        self.assertEqual(
            credentials.to_dict(),
            {
                "token": "super-secret",
                "instance": "instance",
                "channel": "ibm_quantum_platform",
            },
        )


class TestBackend(unittest.TestCase):
    def test_discovered_backend_runs_through_bound_service(self):
        service = _ServiceRecorder()
        credentials = IBMQuantumCredentials(token="secret")
        backend = QuantumBackend(
            provider="ibm",
            name="ibm_test",
            num_qubits=127,
            operational=True,
            pending_jobs=1,
            _service=service,
            _credentials=credentials,
        )
        experiment = _bell_experiment()

        self.assertEqual(backend.run(experiment, shots=512), "job")
        _, plan, used_credentials, _ = service.calls[0]
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(plan.shots, 512)
        self.assertEqual(plan.acquisition.config["backend"], "ibm_test")
        self.assertIs(used_credentials, credentials)


class TestClientLifecycle(unittest.TestCase):
    def test_service_closes_the_client_it_created(self):
        service = QuantumIOService(QuantumIOParams(api_key="k"))
        client = service.client
        self.assertFalse(client.is_closed)
        service.close()
        self.assertTrue(client.is_closed)

    def test_service_leaves_a_caller_supplied_client_open(self):
        borrowed = httpx.Client()
        try:
            service = QuantumIOService(QuantumIOParams(api_key="k"), client=borrowed)
            service.close()
            self.assertFalse(borrowed.is_closed)
        finally:
            borrowed.close()

    def test_service_is_a_context_manager_and_close_is_idempotent(self):
        with QuantumIOService(QuantumIOParams(api_key="k")) as service:
            client = service.client
        self.assertTrue(client.is_closed)
        service.close()  # must not raise


class TestServiceSubmission(unittest.TestCase):
    def setUp(self):
        self.service = QuantumIOService(QuantumIOParams(), client=object())
        self.uploads = []
        self.cleanup_events = []
        _FakePineJob.cleanup_events = self.cleanup_events

        def upload(filename, payload):
            self.uploads.append((filename, dict(payload)))
            return _WorkData(
                f"https://workdata.example/{filename}",
                self.cleanup_events,
                filename,
            )

        self.service._upload_json = upload
        self.service._step = lambda name: _Step()

    def test_run_requires_typed_experiment(self):
        with self.assertRaisesRegex(TypeError, "QuantumExperiment"):
            self.service.run({"kind": "quantum-experiment"})
        self.assertEqual(self.uploads, [])

    def test_default_run_uses_local_simulator_and_no_credentials(self):
        with patch("q_alchemy.quantum_io.Job", _FakePineJob):
            job = self.service.run(_bell_experiment(), shots=256)

        self.assertIsNotNone(job)
        self.assertEqual(
            [name for name, _ in self.uploads],
            [EXPERIMENT_INPUT_ALIAS, EXECUTION_PLAN_INPUT_ALIAS],
        )
        experiment_payload = self.uploads[0][1]
        self.assertEqual(experiment_payload["kind"], "quantum-experiment")
        plan = self.uploads[1][1]
        self.assertEqual(plan["schema_version"], 3)
        self.assertEqual(plan["acquisition"]["resource"], "local-simulator")
        self.assertEqual(plan["shots"], 256)

        created = _FakePineJob.last.created
        slots = [_slot_dump(slot) for slot in created["input_data_slots"]]
        self.assertEqual([slot.get("Index", slot.get("index")) for slot in slots], [0, 1])
        self.assertNotIn("parameters", created)

    def test_preflight_uses_preparation_simulator_without_acquisition(self):
        with patch("q_alchemy.quantum_io.Job", _FakePineJob):
            self.service.preflight(_bell_experiment())
        plan = self.uploads[1][1]
        self.assertEqual(plan["preparation_simulator"]["kind"], "q-alchemy-sparse")
        self.assertIsNone(plan["reference"])
        self.assertIsNone(plan["acquisition"])
        self.assertEqual(plan["shots"], 1)

    def test_quantum_io_preserves_job_and_workdata_when_remove_data_false(self):
        self.service.params.remove_data = False
        self.assertFalse(self.service.params.remove_data)
        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ):
            result = self.service.run(_bell_experiment(), shots=256).result()

        self.assertEqual(result.mode, "ideal-simulation")
        self.assertEqual(self.cleanup_events, [])
        self.assertFalse(hasattr(_FakePineJob.last, "deleted"))

    def test_default_result_cleanup_deletes_job_before_sdk_owned_inputs(self):
        self.assertTrue(self.service.params.remove_data)
        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ):
            result = self.service.run(_bell_experiment(), shots=256).result()

        self.assertEqual(result.mode, "ideal-simulation")
        self.assertTrue(_FakePineJob.last.deleted)
        self.assertEqual(
            _FakePineJob.last.delete_with_associated_kwargs,
            {
                "delete_subjobs_with_data": True,
                "delete_input_workdata": False,
                "delete_output_workdata": True,
            },
        )
        self.assertEqual(
            self.cleanup_events,
            [
                "job",
                f"allow-deletion:{EXPERIMENT_INPUT_ALIAS}",
                f"workdata:{EXPERIMENT_INPUT_ALIAS}",
                f"allow-deletion:{EXECUTION_PLAN_INPUT_ALIAS}",
                f"workdata:{EXECUTION_PLAN_INPUT_ALIAS}",
            ],
        )

    def test_cleanup_preserves_inputs_if_job_deletion_raises(self):
        self.service.params.remove_data = True

        def fail_delete(_job, **_kwargs):
            raise RuntimeError("delete failed")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ), patch.object(_FakePineJob, "delete_with_associated", fail_delete):
            self.service.run(_bell_experiment(), shots=256).result()

        self.assertEqual(self.cleanup_events, [])

    def test_cleanup_preserves_inputs_when_pinexq_only_warns(self):
        """pinexq's delete_with_associated warns instead of raising when the
        Job cannot be deleted. Input WorkData must still be preserved."""
        self.service.params.remove_data = True

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ), patch.object(_FakePineJob, "deletion_allowed", False), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.service.run(_bell_experiment(), shots=256).result()

        self.assertEqual(self.cleanup_events, ["job-delete-refused"])

    def test_failed_job_preserves_lineage_for_diagnosis(self):
        self.service.params.remove_data = True

        def fail_wait(_self, *args, **kwargs):
            raise RuntimeError("Job failed'. Error: processing step crashed")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch.object(
            _FakePineJob, "wait_for_state", fail_wait
        ):
            job = self.service.run(_bell_experiment(), shots=256)
            with self.assertRaisesRegex(RuntimeError, "processing step crashed"):
                job.result()

        self.assertEqual(self.cleanup_events, [])
        self.assertFalse(_FakePineJob.last.deleted if hasattr(_FakePineJob.last, "deleted") else False)

    def test_timeout_preserves_running_job_and_allows_retry(self):
        self.service.params.remove_data = True
        calls = {"n": 0}

        def flaky_wait(_self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("Job did not reach state: 'completed'")
            return _self

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch.object(
            _FakePineJob, "wait_for_state", flaky_wait
        ), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ):
            job = self.service.run(_bell_experiment(), shots=256)
            with self.assertRaises(TimeoutError):
                job.result()
            # Nothing was deleted, so a longer retry can still succeed.
            self.assertEqual(self.cleanup_events, [])
            report = job.result(timeout=5)

        self.assertEqual(report.mode, "ideal-simulation")
        self.assertIn("job", self.cleanup_events)

    def test_report_parsing_failure_preserves_output_workdata(self):
        self.service.params.remove_data = True

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value={"schema_version": 99, "kind": "experiment-report", "report": {}},
        ):
            job = self.service.run(_bell_experiment(), shots=256)
            with self.assertRaises(ValueError):
                job.result()

        self.assertEqual(self.cleanup_events, [])

    def test_raw_job_rejects_use_after_automatic_cleanup(self):
        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ):
            job = self.service.run(_bell_experiment(), shots=256)
            self.assertIsNotNone(job.raw_job)
            job.result()

        self.assertTrue(job.removed)
        with self.assertRaisesRegex(RuntimeError, "remove_data=False"):
            job.raw_job

    def test_repeated_result_reuses_cache_and_cleans_up_once(self):
        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(),
        ):
            job = self.service.run(_bell_experiment(), shots=256)
            first = job.result()
            second = job.result()

        self.assertIs(first, second)
        self.assertEqual(self.cleanup_events.count("job"), 1)

    def test_ibm_result_cleanup_deletes_credentials_after_job(self):
        self.service.params.remove_data = True
        plan = quantum_backend_execution_plan(
            provider="ibm", backend="ibm_test", shots=100
        )
        credentials = IBMQuantumCredentials(token="secret")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=_report_payload(mode="qpu", source_kind="qpu"),
        ):
            self.service.run(
                _bell_experiment(),
                plan,
                credentials=credentials,
            ).result()

        self.assertEqual(
            self.cleanup_events,
            [
                "job",
                f"allow-deletion:{EXPERIMENT_INPUT_ALIAS}",
                f"workdata:{EXPERIMENT_INPUT_ALIAS}",
                f"allow-deletion:{EXECUTION_PLAN_INPUT_ALIAS}",
                f"workdata:{EXECUTION_PLAN_INPUT_ALIAS}",
                f"allow-deletion:{IBM_CREDENTIALS_INPUT_ALIAS}",
                f"workdata:{IBM_CREDENTIALS_INPUT_ALIAS}",
            ],
        )

    def test_ibm_run_uploads_credentials_in_third_dataslot(self):
        plan = quantum_backend_execution_plan(
            provider="ibm", backend="ibm_test", shots=100
        )
        credentials = IBMQuantumCredentials(token="secret")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob):
            self.service.run(_bell_experiment(), plan, credentials=credentials)

        self.assertEqual(
            [name for name, _ in self.uploads],
            [
                EXPERIMENT_INPUT_ALIAS,
                EXECUTION_PLAN_INPUT_ALIAS,
                IBM_CREDENTIALS_INPUT_ALIAS,
            ],
        )
        self.assertEqual(self.uploads[-1][1], {"token": "secret"})
        created = _FakePineJob.last.created
        slots = [_slot_dump(slot) for slot in created["input_data_slots"]]
        self.assertEqual([slot.get("Index", slot.get("index")) for slot in slots], [0, 1, 2])
        self.assertNotIn("parameters", created)

    def test_noisy_backend_run_uses_provider_credentials(self):
        plan = noisy_backend_execution_plan(
            provider="ibm", backend="ibm_test", shots=100
        )
        credentials = IBMQuantumCredentials(token="secret")
        with patch("q_alchemy.quantum_io.Job", _FakePineJob):
            self.service.run(_bell_experiment(), plan, credentials=credentials)
        self.assertEqual(self.uploads[-1], (IBM_CREDENTIALS_INPUT_ALIAS, {"token": "secret"}))

    def test_ibm_run_rejects_missing_credentials_before_job_creation(self):
        plan = quantum_backend_execution_plan(
            provider="ibm", backend="ibm_test", shots=100
        )
        with self.assertRaisesRegex(ValueError, "IBMQuantumCredentials"):
            self.service.run(_bell_experiment(), plan)

    def test_backend_resource_requires_provider_before_upload(self):
        plan = ExecutionPlan(
            acquisition=Runtime.resource("quantum-backend", backend="ibm_test"),
            shots=100,
        )
        with self.assertRaisesRegex(ValueError, "config.provider"):
            self.service.run(_bell_experiment(), plan)
        self.assertEqual(self.uploads, [])

    def test_backend_discovery_calls_list_step_and_binds_backend(self):
        credentials = IBMQuantumCredentials(token="secret")
        self.service._ibm_credentials = credentials
        backend_payload = {
            "provider": "ibm",
            "backends": [
                {
                    "name": "ibm_test",
                    "num_qubits": 127,
                    "operational": True,
                    "pending_jobs": 3,
                }
            ],
        }

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=backend_payload,
        ):
            backends = self.service.backends(provider="ibm", min_num_qubits=10)

        self.assertEqual(len(backends), 1)
        backend = backends[0]
        self.assertEqual(backend.provider, "ibm")
        self.assertEqual(backend.name, "ibm_test")
        self.assertEqual(backend.num_qubits, 127)
        self.assertEqual(backend.pending_jobs, 3)
        self.assertIs(backend._service, self.service)
        self.assertIs(backend._credentials, credentials)

        self.assertEqual(self.uploads, [(IBM_CREDENTIALS_INPUT_ALIAS, {"token": "secret"})])
        created = _FakePineJob.last.created
        parameters = json.loads(created["parameters"])
        self.assertEqual(parameters["provider"], "ibm")
        self.assertEqual(parameters["min_num_qubits"], 10)
        self.assertTrue(parameters["operational_only"])
        slots = [_slot_dump(slot) for slot in created["input_data_slots"]]
        self.assertEqual([slot.get("Index", slot.get("index")) for slot in slots], [0])

    def test_backend_discovery_failure_preserves_job_and_credentials(self):
        self.service.params.remove_data = True
        self.service._ibm_credentials = IBMQuantumCredentials(token="secret")

        def fail_wait(_self, *args, **kwargs):
            raise RuntimeError("Job failed'. Error: provider unreachable")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch.object(
            _FakePineJob, "wait_for_state", fail_wait
        ):
            with self.assertRaisesRegex(RuntimeError, "provider unreachable"):
                self.service.backends(provider="ibm")

        # The credential WorkData must survive so the failure can be diagnosed.
        self.assertEqual(self.cleanup_events, [])

    def test_backend_discovery_timeout_preserves_running_job(self):
        self.service.params.remove_data = True
        self.service._ibm_credentials = IBMQuantumCredentials(token="secret")

        def timeout_wait(_self, *args, **kwargs):
            raise TimeoutError("Job did not reach state: 'completed'")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch.object(
            _FakePineJob, "wait_for_state", timeout_wait
        ):
            with self.assertRaises(TimeoutError):
                self.service.backends(provider="ibm")

        # A discovery job that is still running still needs its credentials.
        self.assertEqual(self.cleanup_events, [])

    def test_backend_discovery_parse_failure_preserves_credentials(self):
        self.service.params.remove_data = True
        self.service._ibm_credentials = IBMQuantumCredentials(token="secret")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value={"provider": "ibm", "backends": [{"name": "x"}]},  # no num_qubits
        ):
            with self.assertRaises(KeyError):
                self.service.backends(provider="ibm")

        self.assertEqual(self.cleanup_events, [])

    def test_backend_discovery_preserves_everything_when_remove_data_false(self):
        self.service.params.remove_data = False
        self.service._ibm_credentials = IBMQuantumCredentials(token="secret")

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value={"provider": "ibm", "backends": []},
        ):
            self.service.backends(provider="ibm")

        self.assertEqual(self.cleanup_events, [])

    def test_backend_discovery_cleanup_deletes_job_before_credentials(self):
        self.service.params.remove_data = True
        credentials = IBMQuantumCredentials(token="secret")
        self.service._ibm_credentials = credentials
        backend_payload = {"provider": "ibm", "backends": []}

        with patch("q_alchemy.quantum_io.Job", _FakePineJob), patch(
            "q_alchemy.quantum_io._download_json_output",
            return_value=backend_payload,
        ):
            self.service.backends(provider="ibm")

        self.assertEqual(
            self.cleanup_events,
            [
                "job",
                f"allow-deletion:{IBM_CREDENTIALS_INPUT_ALIAS}",
                f"workdata:{IBM_CREDENTIALS_INPUT_ALIAS}",
            ],
        )


class TestDownloadJsonOutput(unittest.TestCase):
    """_download_json_output is stubbed out everywhere else; exercise it here."""

    @staticmethod
    def _job(*work_datas):
        class Job:
            def get_output_data_slots(self):
                return [_OutputSlot(*work_datas)]

        return Job()

    def test_reads_the_named_output(self):
        payload = json.dumps(_report_payload()).encode("utf-8")
        job = self._job(_OutputWorkData(EXPERIMENT_REPORT_OUTPUT_ALIAS, payload))
        result = _download_json_output(job, EXPERIMENT_REPORT_OUTPUT_ALIAS)
        self.assertEqual(result["kind"], "experiment-report")

    def test_missing_output_is_reported_by_name(self):
        job = self._job(_OutputWorkData("something_else.json", b"{}"))
        with self.assertRaisesRegex(IOError, EXPERIMENT_REPORT_OUTPUT_ALIAS):
            _download_json_output(job, EXPERIMENT_REPORT_OUTPUT_ALIAS)

    def test_empty_output_is_rejected_without_downloading(self):
        job = self._job(_OutputWorkData(EXPERIMENT_REPORT_OUTPUT_ALIAS, b""))
        with self.assertRaisesRegex(IOError, "empty"):
            _download_json_output(job, EXPERIMENT_REPORT_OUTPUT_ALIAS)

    def test_non_object_json_is_rejected(self):
        job = self._job(_OutputWorkData(EXPERIMENT_REPORT_OUTPUT_ALIAS, b"[1, 2, 3]"))
        with self.assertRaisesRegex(IOError, "not a JSON object"):
            _download_json_output(job, EXPERIMENT_REPORT_OUTPUT_ALIAS)

    def test_duplicate_outputs_are_ambiguous_rather_than_first_wins(self):
        job = self._job(
            _OutputWorkData(EXPERIMENT_REPORT_OUTPUT_ALIAS, b'{"a": 1}'),
            _OutputWorkData(EXPERIMENT_REPORT_OUTPUT_ALIAS, b'{"a": 2}'),
        )
        with self.assertRaisesRegex(IOError, "ambiguous"):
            _download_json_output(job, EXPERIMENT_REPORT_OUTPUT_ALIAS)


class TestArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.service = QuantumIOService(QuantumIOParams(), client=object())

    def test_shots_conflicting_with_an_execution_plan_is_rejected(self):
        plan = local_simulator_execution_plan(shots=64)
        with self.assertRaisesRegex(ValueError, "conflicts with execution_plan.shots"):
            self.service.run(_bell_experiment(), plan, shots=8192)

    def test_shots_matching_the_plan_is_accepted(self):
        plan = local_simulator_execution_plan(shots=64)
        with patch("q_alchemy.quantum_io.Job", _FakePineJob):
            self.service._upload_json = lambda f, p: _WorkData(f"https://wd/{f}")
            self.service._step = lambda name: _Step()
            self.service.run(_bell_experiment(), plan, shots=64)

    def test_unbound_backend_cannot_run(self):
        backend = QuantumBackend(provider="ibm", name="ibm_test", num_qubits=5)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            backend.run(_bell_experiment())

    def test_credentials_reject_an_empty_token(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            IBMQuantumCredentials(token="   ")

    def test_credentials_reject_an_unknown_channel(self):
        with self.assertRaisesRegex(ValueError, "channel"):
            IBMQuantumCredentials(token="t", channel="nope")

    def test_backend_discovery_rejects_unsupported_provider(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            self.service.backends(provider="rigetti")

    def test_backend_discovery_rejects_negative_min_num_qubits(self):
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            self.service.backends(provider="ibm", min_num_qubits=-1)


@unittest.skipUnless(
    os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"),
    "no Q_ALCHEMY_API_KEY/PINEXQ_API_KEY: skipping live Quantum I/O test",
)
class TestLiveQuantumIO(unittest.TestCase):
    def test_preflight(self):
        report = QuantumIOService(job_completion_timeout_sec=180).preflight(
            _bell_experiment()
        ).result()
        self.assertEqual(report.mode, "preflight")
        self.assertIsNone(report.execution)
        self.assertIsNotNone(report.preparation_preflight)

    def test_local_bell_experiment(self):
        report = QuantumIOService(job_completion_timeout_sec=180).run(
            _bell_experiment(),
            shots=256,
        ).result()
        self.assertEqual(report.mode, "ideal-simulation")
        execution = report.execution
        self.assertIsNotNone(execution)
        self.assertEqual(execution.source_kind, "ideal-simulator")
        distribution = execution.basis_distributions[0]
        self.assertEqual(distribution.label, "q0-q1")
        self.assertEqual(distribution.shots, 256)
        self.assertTrue(set(distribution.probabilities) <= {"00", "11"})
        self.assertAlmostEqual(sum(distribution.probabilities.values()), 1.0)

    @unittest.skipUnless(
        os.getenv("IBM_QUANTUM_TOKEN"),
        "no IBM_QUANTUM_TOKEN: skipping live IBM backend discovery test",
    )
    def test_ibm_backend_discovery(self):
        credentials = IBMQuantumCredentials(token=os.environ["IBM_QUANTUM_TOKEN"])
        service = QuantumIOService(
            ibm_credentials=credentials,
            job_completion_timeout_sec=180,
        )
        backends = service.backends(
            provider="ibm",
            min_num_qubits=1,
            operational_only=False,
        )

        self.assertTrue(backends, "IBM Quantum returned no accessible hardware backends")
        names = [backend.name for backend in backends]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        for backend in backends:
            with self.subTest(backend=backend.name):
                self.assertEqual(backend.provider, "ibm")
                self.assertTrue(backend.name)
                self.assertGreaterEqual(backend.num_qubits, 1)
                self.assertIn(backend.operational, {True, False, None})
                self.assertTrue(backend.pending_jobs is None or backend.pending_jobs >= 0)

        self.assertNotIn(os.environ["IBM_QUANTUM_TOKEN"], repr(backends))

@unittest.skipUnless(
    (os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"))
    and os.getenv("Q_ALCHEMY_RUN_DESTRUCTIVE_TESTS") == "1",
    "set Q_ALCHEMY_RUN_DESTRUCTIVE_TESTS=1 with an API key to run destructive cleanup tests",
)
class TestLiveQuantumIOCleanup(unittest.TestCase):
    def test_remove_data_true_deletes_input_workdata(self):
        service = QuantumIOService(
            QuantumIOParams(
                remove_data=True,
                job_completion_timeout_sec=180,
            )
        )

        job = service.run(
            _bell_experiment(),
            shots=64,
        )

        # Capture the SDK-created input WorkData links before result() performs
        # cleanup. Accessing this private field is intentional in this
        # integration regression test.
        input_links = [
            owned.link
            for owned in job._input_workdata
        ]

        self.assertGreaterEqual(len(input_links), 2)

        # They must exist before cleanup.
        for link in input_links:
            self.assertIsNotNone(link.navigate())

        report = job.result()

        self.assertEqual(report.mode, "ideal-simulation")

        # PineXQ deletion may not become visible instantaneously, so allow
        # a short interval for server-side propagation.
        for link in input_links:
            deleted = False

            for _ in range(20):
                try:
                    link.navigate()
                except Exception:
                    deleted = True
                    break

                time.sleep(0.25)

            self.assertTrue(
                deleted,
                f"Input WorkData still exists after remove_data=True: "
                f"{link.get_url()}",
            )


if __name__ == "__main__":
    unittest.main()
