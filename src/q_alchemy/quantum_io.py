"""Qiskit-inspired client for the hosted Q-Alchemy Quantum I/O service.

Users construct experiments and execution plans with typed SDK classes. JSON is
used only as the private PineXQ WorkData wire representation. The public API
mirrors q-alchemy-quantum-io schema 3 without depending on that runtime package.

Typical usage::

    experiment = QuantumExperiment(target=State.dense([1, 0]))
    report = QuantumIOService().preflight(experiment).result()

Provider credentials are uploaded as dedicated WorkData and never placed in the
experiment or execution plan. Quantum I/O removes SDK-created Jobs and WorkData
by default after retrieving results. Set ``remove_data=False`` to preserve the
PineXQ execution lineage.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping

import httpx
from pinexq.client.core.hco.upload_action_hco import UploadParameters
from pinexq.client.job_management import Job, enter_jma
from pinexq.client.job_management.hcos import WorkDataLink
from pinexq.client.job_management.model import InputDataSlotParameter, JobStates

from q_alchemy.initialize import create_client, find_processing_step, from_name
from q_alchemy.quantum_io_contract import (
    ExecutionPlan,
    ExperimentReport,
    QuantumExperiment,
    Runtime,
)

RUN_QUANTUM_EXPERIMENT_STEP = "run_quantum_experiment"
LIST_QUANTUM_BACKENDS_STEP = "list_quantum_backends"

EXPERIMENT_INPUT_ALIAS = "quantum_experiment.json"
EXECUTION_PLAN_INPUT_ALIAS = "execution_plan.json"
IBM_CREDENTIALS_INPUT_ALIAS = "ibm_credentials.json"
EXPERIMENT_REPORT_OUTPUT_ALIAS = "experiment_report.json"
QUANTUM_BACKENDS_OUTPUT_ALIAS = "quantum_backends.json"

LOCAL_SIMULATOR_RESOURCE = "local-simulator"
QUANTUM_BACKEND_RESOURCE = "quantum-backend"
NOISY_BACKEND_SIMULATOR_RESOURCE = "noisy-backend-simulator"

LOG = logging.getLogger(__name__)

# DataSlot order is part of the deployed Quantum I/O ProcessingStep contract.
_RUN_EXPERIMENT_SLOT = 0
_RUN_PLAN_SLOT = 1
_RUN_IBM_CREDENTIALS_SLOT = 2
_LIST_IBM_CREDENTIALS_SLOT = 0


@dataclass(frozen=True)
class _OwnedWorkData:
    """WorkData created by this SDK call and safe to remove after its Job."""

    name: str
    link: WorkDataLink
    sensitive: bool = False


@dataclass
class QuantumIOParams:
    """Connection and job options for :class:`QuantumIOService`.

    Attribute names intentionally mirror the rest of the SDK.  ``api_key``
    defaults to ``Q_ALCHEMY_API_KEY`` and falls back to ``PINEXQ_API_KEY``.
    ``step_version`` can pin a deployed Quantum I/O ProcessingStep version;
    when omitted, the SDK resolves the visible service version in the same way
    as the other Q-Alchemy clients.
    """

    api_key: str | None = field(
        default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"),
        repr=False,
    )
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = "https"
    added_headers: dict[str, str] = field(default_factory=dict, repr=False)
    job_completion_timeout_sec: int | None = 300
    job_tags: list[str] = field(default_factory=list)
    remove_data: bool = True
    step_version: str | None = None

    @classmethod
    def from_dict(cls, env: Mapping[str, Any]) -> "QuantumIOParams":
        """Build params from a mapping, ignoring keys this class does not define.

        Unknown keys are dropped rather than rejected, so a settings dict shared
        with the older SDK APIs still works. They are warned about, because a
        silently discarded key is indistinguishable from a misspelled option.
        """

        names = {item.name for item in fields(cls)}
        unknown = sorted(key for key in env if key not in names)
        if unknown:
            warnings.warn(
                "QuantumIOParams ignored unknown option(s): "
                + ", ".join(repr(key) for key in unknown),
                stacklevel=2,
            )
        return cls(**{key: value for key, value in env.items() if key in names})


@dataclass(frozen=True, repr=False)
class IBMQuantumCredentials:
    """IBM Quantum BYOC credentials used by the deployed Quantum I/O service.

    The token is deliberately redacted from ``repr``.  The object is serialized
    only into the dedicated ``ibm_credentials.json`` WorkData input required by
    IBM-backed calls.
    """

    token: str
    instance: str | None = None
    channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("IBM Quantum token must be a non-empty string")
        if self.channel is not None and self.channel not in {"ibm_quantum_platform", "ibm_cloud"}:
            raise ValueError(
                "IBM Quantum channel must be 'ibm_quantum_platform' or 'ibm_cloud'"
            )

    def __repr__(self) -> str:
        return (
            "IBMQuantumCredentials(token='***', "
            f"instance={self.instance!r}, channel={self.channel!r})"
        )

    def to_dict(self) -> dict[str, str]:
        payload = {"token": self.token}
        if self.instance is not None:
            payload["instance"] = self.instance
        if self.channel is not None:
            payload["channel"] = self.channel
        return payload


@dataclass(frozen=True)
class QuantumBackend:
    """Provider-neutral backend information returned by ``list_quantum_backends``.

    Instances returned by :meth:`QuantumIOService.backends` are bound to the
    originating service, so ``backend.run(...)`` mirrors the familiar Qiskit
    backend workflow while still using the provider-neutral Quantum I/O
    ProcessingStep underneath.
    """

    provider: str
    name: str
    num_qubits: int
    operational: bool | None = None
    pending_jobs: int | None = None
    _service: "QuantumIOService | None" = field(default=None, repr=False, compare=False)
    _credentials: IBMQuantumCredentials | Mapping[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_raw(
        cls,
        provider: str,
        raw: Mapping[str, Any],
        *,
        service: "QuantumIOService | None" = None,
        credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
    ) -> "QuantumBackend":
        return cls(
            provider=provider,
            name=str(raw["name"]),
            num_qubits=int(raw["num_qubits"]),
            operational=(
                bool(raw["operational"]) if raw.get("operational") is not None else None
            ),
            pending_jobs=(
                int(raw["pending_jobs"]) if raw.get("pending_jobs") is not None else None
            ),
            _service=service,
            _credentials=credentials,
        )

    def run(
        self,
        experiment: QuantumExperiment,
        *,
        shots: int = 1024,
    ) -> "QuantumIOJob":
        """Run ``experiment`` on this QPU through ``run_quantum_experiment``.

        The acquisition plan is generated for this concrete provider/backend
        pair. Advanced workflows that need custom reference, simulation, or
        estimation stages should use ``QuantumIOService.run(..., execution_plan=...)``.
        """

        if self._service is None:
            raise RuntimeError("This QuantumBackend is not bound to a QuantumIOService")
        execution_plan = quantum_backend_execution_plan(
            provider=self.provider,
            backend=self.name,
            shots=shots,
        )
        return self._service.run(
            experiment,
            execution_plan=execution_plan,
            credentials=self._credentials,
        )


class QuantumIOJob:
    """Handle for a submitted ``run_quantum_experiment`` PineXQ job.

    Execution starts when the object is created by :meth:`QuantumIOService.run`.
    ``result()`` waits for completion, downloads ``experiment_report.json`` and
    caches the typed :class:`ExperimentReport` locally.
    """

    def __init__(
        self,
        job: Job,
        *,
        timeout_sec: int | None,
        remove_data: bool,
        input_workdata: list[_OwnedWorkData] | None = None,
    ) -> None:
        self._job = job
        self._timeout_sec = timeout_sec
        self._remove_data = remove_data
        self._input_workdata = list(input_workdata or [])
        self._result: ExperimentReport | None = None
        self._cleaned = False

    @property
    def removed(self) -> bool:
        """True once automatic cleanup has deleted this job's PineXQ resources."""
        return self._cleaned

    @property
    def raw_job(self) -> Job:
        """Underlying ``pinexq-client`` Job for advanced inspection.

        Only available while the PineXQ resources still exist. With the default
        ``remove_data=True`` they are deleted as soon as :meth:`result` returns.
        """
        if self._cleaned:
            raise RuntimeError(
                "The PineXQ Job for this Quantum I/O call has been removed. "
                "Use QuantumIOParams(remove_data=False) to keep the Job and its "
                "WorkData for inspection."
            )
        return self._job

    def result(self, timeout: float | None = None) -> ExperimentReport:
        if self._result is not None:
            return self._result

        timeout_sec = (
            float(timeout)
            if timeout is not None
            else (
                float(self._timeout_sec)
                if self._timeout_sec is not None
                else float(24 * 60 * 60)
            )
        )
        try:
            self._job.wait_for_state(
                JobStates.completed,
                polling_interval_s=0.25,
                timeout_s=timeout_sec,
            )
            raw = _download_json_output(self._job, EXPERIMENT_REPORT_OUTPUT_ALIAS)
            self._result = ExperimentReport.from_dict(raw)
        except BaseException:
            # Timeout, job failure, download failure and report-parsing failure
            # all leave the PineXQ lineage in place. Deleting it here would
            # destroy a still-running Job (timeout), the only record of why a
            # Job failed, or the only copy of an unparsable report.
            LOG.warning(
                "Quantum I/O result retrieval failed; the PineXQ Job and its "
                "WorkData were preserved for diagnosis and retry"
            )
            raise
        if self._remove_data and not self._cleaned:
            self._cleanup()
        return self._result

    def _cleanup(self) -> None:
        try:
            _delete_job_then_inputs(self._job, self._input_workdata)
        finally:
            self._cleaned = True


class QuantumIOService:
    """Qiskit-inspired entry point for the hosted Q-Alchemy Quantum I/O service."""

    def __init__(
        self,
        params: QuantumIOParams | Mapping[str, Any] | None = None,
        *,
        ibm_credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None:
        if params is None:
            resolved = QuantumIOParams()
        elif isinstance(params, QuantumIOParams):
            resolved = params
        else:
            resolved = QuantumIOParams.from_dict(params)

        for name, value in kwargs.items():
            if hasattr(resolved, name):
                setattr(resolved, name, value)
            else:
                raise TypeError(f"Unknown QuantumIOService option {name!r}")

        owns_client = client is None
        if client is None:
            if not resolved.api_key:
                raise ValueError(
                    "A Q-Alchemy API key is required. Set Q_ALCHEMY_API_KEY or "
                    "PINEXQ_API_KEY, or pass api_key=..."
                )
            client = create_client(resolved)  # QuantumIOParams is structurally compatible.

        self.params = resolved
        self.client = client
        self._owns_client = owns_client
        self._ibm_credentials = _coerce_ibm_credentials(ibm_credentials)

    def close(self) -> None:
        """Release the HTTP connection pool created by this service.

        A client passed in by the caller is left open: the service does not own
        it. Calling this more than once is safe.
        """

        if self._owns_client and self.client is not None:
            self.client.close()
            self._owns_client = False

    def __enter__(self) -> "QuantumIOService":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def run(
        self,
        experiment: QuantumExperiment,
        execution_plan: ExecutionPlan | None = None,
        *,
        shots: int = 1024,
        credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
    ) -> QuantumIOJob:
        """Submit a typed quantum experiment and return a :class:`QuantumIOJob`.

        With no explicit execution plan, the service uses Q-Alchemy's local
        sparse simulator. Advanced workflows pass a typed :class:`ExecutionPlan`.
        Serialization to schema-3 JSON is entirely internal to the SDK.
        """

        if not isinstance(experiment, QuantumExperiment):
            raise TypeError("experiment must be a QuantumExperiment")
        if execution_plan is None:
            plan = local_simulator_execution_plan(shots=shots)
        elif isinstance(execution_plan, ExecutionPlan):
            plan = execution_plan
        else:
            raise TypeError("execution_plan must be an ExecutionPlan")

        experiment_payload = experiment.to_dict()
        plan_payload = plan.to_dict()

        provider = _quantum_backend_provider(plan)
        ibm_credentials = None
        if provider is not None:
            if provider not in {"ibm", "ibm-quantum"}:
                raise ValueError(
                    f"SDK credential handling for provider {provider!r} is not implemented"
                )
            ibm_credentials = _coerce_ibm_credentials(credentials) or self._ibm_credentials
            if ibm_credentials is None:
                raise ValueError(
                    "IBM-backed Quantum I/O execution requires IBMQuantumCredentials"
                )

        step = self._step(RUN_QUANTUM_EXPERIMENT_STEP)
        experiment_wd = self._upload_json(EXPERIMENT_INPUT_ALIAS, experiment_payload)
        plan_wd = self._upload_json(EXECUTION_PLAN_INPUT_ALIAS, plan_payload)
        owned_inputs = [
            _OwnedWorkData(EXPERIMENT_INPUT_ALIAS, experiment_wd),
            _OwnedWorkData(EXECUTION_PLAN_INPUT_ALIAS, plan_wd),
        ]
        input_slots = [
            InputDataSlotParameter(
                Index=_RUN_EXPERIMENT_SLOT,
                WorkDataUrls=[str(experiment_wd.get_url())],
            ),
            InputDataSlotParameter(
                Index=_RUN_PLAN_SLOT,
                WorkDataUrls=[str(plan_wd.get_url())],
            ),
        ]
        if ibm_credentials is not None:
            credentials_wd = self._upload_json(
                IBM_CREDENTIALS_INPUT_ALIAS,
                ibm_credentials.to_dict(),
            )
            owned_inputs.append(
                _OwnedWorkData(
                    IBM_CREDENTIALS_INPUT_ALIAS,
                    credentials_wd,
                    sensitive=True,
                )
            )
            input_slots.append(
                InputDataSlotParameter(
                    Index=_RUN_IBM_CREDENTIALS_SLOT,
                    WorkDataUrls=[str(credentials_wd.get_url())],
                )
            )

        job = Job(client=self.client).create_and_configure_rapidly(
            name=f"Quantum I/O experiment ({datetime.now():%Y-%m-%d %H:%M:%S})",
            tags=["SDK", "QuantumIO", "Experiment"] + list(self.params.job_tags),
            processing_step_url=step.self_link(),
            start=True,
            allow_output_data_deletion=True,
            input_data_slots=input_slots,
        )
        return QuantumIOJob(
            job,
            timeout_sec=self.params.job_completion_timeout_sec,
            remove_data=self.params.remove_data,
            input_workdata=owned_inputs,
        )

    def preflight(
        self,
        experiment: QuantumExperiment,
        *,
        verify_preparation: bool = True,
    ) -> QuantumIOJob:
        """Run the lightest Quantum I/O workflow without reference or acquisition.

        By default the generated preparation circuit is independently checked
        with Q-Alchemy's sparse simulator. Set ``verify_preparation=False`` for
        synthesis/reporting only.
        """

        plan = ExecutionPlan(
            preparation_simulator=(Runtime.qalchemy_sparse() if verify_preparation else None),
            shots=1,
        )
        return self.run(experiment, execution_plan=plan)

    def backends(
        self,
        provider: str = "ibm",
        *,
        credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
        min_num_qubits: int = 0,
        operational_only: bool = True,
    ) -> list[QuantumBackend]:
        """Return QPUs visible through ``list_quantum_backends``.

        The returned backend objects are bound to this service and may be used
        directly with ``backend.run(experiment, shots=...)``.
        """

        provider_id = provider.strip().lower()
        if min_num_qubits < 0:
            raise ValueError("min_num_qubits must not be negative")
        if provider_id not in {"ibm", "ibm-quantum"}:
            raise ValueError(
                f"SDK credential handling for provider {provider_id!r} is not implemented"
            )
        resolved_credentials = _coerce_ibm_credentials(credentials) or self._ibm_credentials
        if resolved_credentials is None:
            raise ValueError("IBM backend discovery requires IBMQuantumCredentials")

        step = self._step(LIST_QUANTUM_BACKENDS_STEP)
        credentials_wd = self._upload_json(
            IBM_CREDENTIALS_INPUT_ALIAS,
            resolved_credentials.to_dict(),
        )
        input_slots = [
            InputDataSlotParameter(
                Index=_LIST_IBM_CREDENTIALS_SLOT,
                WorkDataUrls=[str(credentials_wd.get_url())],
            )
        ]
        job = Job(client=self.client).create_and_configure_rapidly(
            name=f"List quantum backends ({datetime.now():%Y-%m-%d %H:%M:%S})",
            tags=["SDK", "QuantumIO", "Backends"] + list(self.params.job_tags),
            processing_step_url=step.self_link(),
            start=True,
            parameters=json.dumps(
                {
                    "provider": provider_id,
                    "min_num_qubits": int(min_num_qubits),
                    "operational_only": bool(operational_only),
                }
            ),
            allow_output_data_deletion=True,
            input_data_slots=input_slots,
        )

        timeout = (
            self.params.job_completion_timeout_sec
            if self.params.job_completion_timeout_sec is not None
            else 24 * 60 * 60
        )
        try:
            job.wait_for_state(
                JobStates.completed,
                polling_interval_s=0.25,
                timeout_s=timeout,
            )
            payload = _download_json_output(job, QUANTUM_BACKENDS_OUTPUT_ALIAS)
            response_provider = str(payload.get("provider") or provider_id)
            discovered = [
                QuantumBackend.from_raw(
                    response_provider,
                    raw,
                    service=self,
                    credentials=resolved_credentials,
                )
                for raw in payload.get("backends", [])
            ]
        except BaseException:
            # A discovery Job that timed out is still running and still needs
            # its credential WorkData; a failed one is the only diagnosis we
            # have. Neither may be deleted here.
            LOG.warning(
                "Quantum I/O backend discovery failed; the PineXQ Job and its "
                "credential WorkData were preserved for diagnosis"
            )
            raise
        if self.params.remove_data:
            _delete_job_then_inputs(
                job,
                [
                    _OwnedWorkData(
                        IBM_CREDENTIALS_INPUT_ALIAS,
                        credentials_wd,
                        sensitive=True,
                    )
                ],
            )
        return discovered

    def backend(
        self,
        name: str,
        *,
        provider: str = "ibm",
        credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
    ) -> QuantumBackend:
        """Return one named QPU, analogous to a Qiskit service ``backend()`` lookup."""

        matches = self.backends(
            provider=provider,
            credentials=credentials,
            operational_only=False,
        )
        for backend in matches:
            if backend.name == name:
                return backend
        raise LookupError(f"No {provider!r} quantum backend named {name!r} is available")

    def _step(self, name: str):
        if self.params.step_version is not None:
            return from_name(self.client, name, version=self.params.step_version)
        return find_processing_step(self.client, name)

    def _upload_json(self, filename: str, payload: Mapping[str, Any]) -> WorkDataLink:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        root = enter_jma(self.client).work_data_root_link.navigate()
        return root.upload_action.execute(
            UploadParameters(
                filename=filename,
                binary=encoded,
                mediatype="application/json",
                json=None,
            )
        )



def local_simulator_execution_plan(*, shots: int = 1024) -> ExecutionPlan:
    """Return an execution plan for Q-Alchemy's local sparse simulator."""

    return ExecutionPlan(
        acquisition=Runtime.resource(LOCAL_SIMULATOR_RESOURCE),
        shots=int(shots),
    )


def quantum_backend_execution_plan(
    *,
    provider: str,
    backend: str | None = None,
    least_busy: bool = False,
    shots: int = 1024,
) -> ExecutionPlan:
    """Return an execution plan for a live provider-neutral QPU."""

    provider_id, config = _backend_resource_config(
        provider=provider, backend=backend, least_busy=least_busy
    )
    return ExecutionPlan(
        acquisition=Runtime.resource(QUANTUM_BACKEND_RESOURCE, provider=provider_id, **config),
        shots=int(shots),
    )


def noisy_backend_execution_plan(
    *,
    provider: str,
    backend: str | None = None,
    least_busy: bool = False,
    shots: int = 4096,
    ideal_reference: bool = True,
    estimator: bool = False,
) -> ExecutionPlan:
    """Return a plan for a simulator calibrated from a real provider backend.

    The real backend is queried for topology/calibration only; no QPU job is
    submitted. ``ideal_reference=True`` also runs Q-Alchemy's sparse reference
    so the report can compare ideal and noisy outputs.
    """

    provider_id, config = _backend_resource_config(
        provider=provider, backend=backend, least_busy=least_busy
    )
    return ExecutionPlan(
        reference=(Runtime.qalchemy_sparse(source="ideal-reference") if ideal_reference else None),
        acquisition=Runtime.resource(
            NOISY_BACKEND_SIMULATOR_RESOURCE,
            provider=provider_id,
            **config,
        ),
        estimator=(Runtime.qtucker() if estimator else None),
        shots=int(shots),
    )


def _backend_resource_config(
    *,
    provider: str,
    backend: str | None,
    least_busy: bool,
) -> tuple[str, dict[str, Any]]:
    provider_id = provider.strip().lower()
    backend_name = backend.strip() if isinstance(backend, str) and backend.strip() else None
    if not provider_id:
        raise ValueError("provider must be a non-empty string")
    if bool(backend_name) == bool(least_busy):
        raise ValueError("select exactly one backend strategy: backend or least_busy=True")
    config: dict[str, Any] = {}
    if backend_name is not None:
        config["backend"] = backend_name
    else:
        config["least_busy"] = True
    return provider_id, config


def _quantum_backend_provider(plan: ExecutionPlan) -> str | None:
    acquisition = plan.acquisition
    if acquisition is None or acquisition.kind != "resource":
        return None
    if acquisition.resource_name not in {
        QUANTUM_BACKEND_RESOURCE,
        NOISY_BACKEND_SIMULATOR_RESOURCE,
    }:
        return None
    config = acquisition.config
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(
            f"{acquisition.resource_name} execution requires acquisition.config.provider"
        )
    return provider.strip().lower()


def _coerce_ibm_credentials(
    credentials: IBMQuantumCredentials | Mapping[str, Any] | None,
) -> IBMQuantumCredentials | None:
    if credentials is None:
        return None
    if isinstance(credentials, IBMQuantumCredentials):
        return credentials
    if not isinstance(credentials, Mapping):
        raise TypeError("IBM credentials must be IBMQuantumCredentials or a mapping")
    return IBMQuantumCredentials(
        token=str(credentials.get("token") or ""),
        instance=(str(credentials["instance"]) if credentials.get("instance") is not None else None),
        channel=(str(credentials["channel"]) if credentials.get("channel") is not None else None),
    )



def _delete_job_then_inputs(
    job: Job,
    input_workdata: list[_OwnedWorkData],
) -> None:
    """Delete SDK-owned Quantum I/O data in PineXQ lineage order.

    PineXQ requires Job outputs to be deleted before the Job itself. The
    documented ``Job.delete_with_associated`` operation handles that ordering
    while leaving inputs intact. Once the Job is gone, each SDK-created input
    can be allowed for deletion and then deleted.
    """

    try:
        job.delete_with_associated(
            delete_subjobs_with_data=True,
            delete_input_workdata=False,
            delete_output_workdata=True,
        )
    except Exception as exc:
        # Do not attempt input deletion while the Job may still reference it.
        LOG.warning(
            "Could not remove Quantum I/O output data/job; preserving input WorkData: %s",
            exc,
        )
        return

    if getattr(job, "job_hco", None) is not None:
        # ``delete_with_associated`` is best effort: when PineXQ does not offer
        # the Delete action (the Job is still running, or deletion is refused)
        # pinexq-client only emits a warning. It clears ``job_hco`` exclusively
        # on a real deletion, so a surviving hco means the Job -- and its
        # reference to these inputs -- is still there.
        LOG.warning(
            "Quantum I/O job was not deleted by PineXQ; preserving input WorkData"
        )
        return

    for work_data in input_workdata:
        _allow_then_delete_owned_workdata(work_data)


def _allow_then_delete_owned_workdata(work_data: _OwnedWorkData) -> None:
    """Best-effort deletion of one SDK-created input WorkData object.

    ``pinexq-client`` 1.10 exposes the required HCO operations as
    ``allow_deletion_action`` and ``delete_action``. Re-navigate after
    AllowDeletion because the Siren actions available on the WorkData change
    with its server-side state.
    """

    try:
        hco = work_data.link.navigate()
        if hco.is_deletable is not True:
            hco.allow_deletion_action.execute()
            hco = work_data.link.navigate()
        hco.delete_action.execute()
    except Exception as exc:
        log = LOG.error if work_data.sensitive else LOG.warning
        log(
            "Could not remove %sQuantum I/O input WorkData %r: %s",
            "sensitive " if work_data.sensitive else "",
            work_data.name,
            exc,
        )


def _download_json_output(job: Job, output_name: str) -> dict[str, Any]:
    matches = [
        work_data
        for slot in job.get_output_data_slots()
        for work_data in slot.assigned_workdatas
        if work_data.name == output_name
    ]
    if not matches:
        raise IOError(f"Quantum I/O job produced no {output_name!r} output")
    work_data = matches[0]
    if work_data.size_in_bytes == 0:
        raise IOError(f"Quantum I/O job returned an empty {output_name!r}")
    payload = json.loads(work_data.download_link.download().decode("utf-8"))
    if not isinstance(payload, dict):
        raise IOError(f"Quantum I/O output {output_name!r} is not a JSON object")
    return payload
