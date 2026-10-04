"""Remote client for the hosted Q-Alchemy feasibility service."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Mapping

import httpx
from pinexq.client.core.hco.upload_action_hco import UploadParameters
from pinexq.client.job_management import Job, enter_jma
from pinexq.client.job_management.hcos import WorkDataLink
from pinexq.client.job_management.model import InputDataSlotParameter, JobStates

from q_alchemy.feasibility_contract import FeasibilityReport, FeasibilityRequest
from q_alchemy.initialize import create_client, find_processing_step, from_name
from q_alchemy.quantum_io import (
    IBMQuantumCredentials,
    _OwnedWorkData,
    _coerce_ibm_credentials,
    _delete_job_then_inputs,
    _download_json_output,
    _mark_workdata_secret,
    _delete_owned_inputs,
)
from q_alchemy.quantum_io_contract import QuantumExperiment


ASSESS_FEASIBILITY_STEP = "assess_feasibility"
EXPERIMENT_INPUT_ALIAS = "quantum_experiment.json"
FEASIBILITY_REQUEST_INPUT_ALIAS = "feasibility_request.json"
IBM_CREDENTIALS_INPUT_ALIAS = "ibm_credentials.json"
FEASIBILITY_REPORT_OUTPUT_ALIAS = "feasibility_report.json"

_EXPERIMENT_SLOT = 0
_REQUEST_SLOT = 1
_IBM_CREDENTIALS_SLOT = 2

LOG = logging.getLogger(__name__)


class FeasibilityExecutionError(RuntimeError):
    """The hosted feasibility execution did not complete successfully."""

    def __init__(self, message: str, *, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


def _remote_feasibility_error_message(exc: Exception) -> str:
    """Extract the user-visible failure detail returned by PineXQ."""

    message = str(exc).strip()
    marker = "[/procon/error]"
    if marker in message:
        remote_message = message.rsplit(marker, 1)[-1].strip()
        if remote_message:
            return remote_message
    return message or type(exc).__name__


@dataclass
class FeasibilityParams:
    api_key: str | None = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"), repr=False)
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = "https"
    added_headers: dict[str, str] = field(default_factory=dict, repr=False)
    job_completion_timeout_sec: int | None = 300
    job_tags: list[str] = field(default_factory=list)
    remove_data: bool = True
    step_version: str | None = None

    @classmethod
    def from_dict(cls, env: Mapping[str, Any]) -> "FeasibilityParams":
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in env.items() if key in names})


class FeasibilityJob:
    def __init__(self, job: Job, *, timeout_sec: int | None, remove_data: bool, input_workdata: list[_OwnedWorkData] | None = None) -> None:
        self._job = job
        self._timeout_sec = timeout_sec
        self._remove_data = remove_data
        self._input_workdata = list(input_workdata or [])
        self._result: FeasibilityReport | None = None
        self._job_removed = False
        self._cleanup_complete = False

    @property
    def removed(self) -> bool:
        return self._cleanup_complete

    @property
    def raw_job(self) -> Job:
        if self._job_removed:
            raise RuntimeError("The PineXQ Job for this feasibility call has been removed")
        return self._job

    def result(self, timeout: float | None = None) -> FeasibilityReport:
        if self._result is None:
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
                self._result = FeasibilityReport.from_dict(
                    _download_json_output(self._job, FEASIBILITY_REPORT_OUTPUT_ALIAS)
                )
            except Exception as exc:
                LOG.warning(
                    "Feasibility result retrieval failed; the PineXQ Job and its "
                    "WorkData were preserved for diagnosis and retry"
                )
                detail = _remote_feasibility_error_message(exc)
                raise FeasibilityExecutionError(
                    f"Feasibility execution failed: {detail}",
                    original_exception=exc,
                ) from None
        if self._remove_data and not self._cleanup_complete:
            self._job_removed, self._input_workdata = _delete_job_then_inputs(
                self._job,
                self._input_workdata,
                job_already_removed=self._job_removed,
            )
            self._cleanup_complete = self._job_removed and not self._input_workdata
        return self._result


class FeasibilityService:
    """Remote facade for the server-side feasibility doctor."""

    def __init__(
        self,
        params: FeasibilityParams | Mapping[str, Any] | None = None,
        *,
        ibm_credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> None:
        if params is None:
            resolved = FeasibilityParams()
        elif isinstance(params, FeasibilityParams):
            resolved = params
        else:
            resolved = FeasibilityParams.from_dict(params)
        for name, value in kwargs.items():
            if not hasattr(resolved, name):
                raise TypeError(f"Unknown FeasibilityService option {name!r}")
            setattr(resolved, name, value)
        owns_client = client is None
        if client is None:
            if not resolved.api_key:
                raise ValueError("A Q-Alchemy API key is required")
            client = create_client(resolved)
        self.params = resolved
        self.client = client
        self._owns_client = owns_client
        self._ibm_credentials = _coerce_ibm_credentials(ibm_credentials)

    def close(self) -> None:
        if self._owns_client and self.client is not None:
            self.client.close()
            self._owns_client = False

    def __enter__(self) -> "FeasibilityService":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def analyze(
        self,
        experiment: QuantumExperiment,
        request: FeasibilityRequest,
        *,
        credentials: IBMQuantumCredentials | Mapping[str, Any] | None = None,
        include_quantum_circuits: bool = False,
    ) -> FeasibilityJob:
        if not isinstance(experiment, QuantumExperiment):
            raise TypeError("experiment must be a QuantumExperiment")
        if not isinstance(request, FeasibilityRequest):
            raise TypeError("request must be a FeasibilityRequest")
        if not isinstance(include_quantum_circuits, bool):
            raise TypeError("include_quantum_circuits must be a bool")

        step = self._step(ASSESS_FEASIBILITY_STEP)
        owned_inputs: list[_OwnedWorkData] = []
        job: Job | None = None
        resolved_credentials = _coerce_ibm_credentials(credentials) or self._ibm_credentials
        try:
            experiment_wd = self._upload_json(EXPERIMENT_INPUT_ALIAS, experiment.to_dict())
            owned_inputs.append(_OwnedWorkData(EXPERIMENT_INPUT_ALIAS, experiment_wd))
            request_payload = request.to_dict()
            request_payload["service_options"] = {
                "include_quantum_circuits": include_quantum_circuits,
            }
            request_wd = self._upload_json(FEASIBILITY_REQUEST_INPUT_ALIAS, request_payload)
            owned_inputs.append(_OwnedWorkData(FEASIBILITY_REQUEST_INPUT_ALIAS, request_wd))
            slots = [
                InputDataSlotParameter(Index=_EXPERIMENT_SLOT, WorkDataUrls=[str(experiment_wd.get_url())]),
                InputDataSlotParameter(Index=_REQUEST_SLOT, WorkDataUrls=[str(request_wd.get_url())]),
            ]
            if resolved_credentials is not None:
                credentials_wd = self._upload_json(IBM_CREDENTIALS_INPUT_ALIAS, resolved_credentials.to_dict(), secret=True)
                owned_inputs.append(_OwnedWorkData(IBM_CREDENTIALS_INPUT_ALIAS, credentials_wd, sensitive=True))
                slots.append(InputDataSlotParameter(Index=_IBM_CREDENTIALS_SLOT, WorkDataUrls=[str(credentials_wd.get_url())]))

            job = Job(client=self.client)
            job.create_and_configure_rapidly(
                name=f"Feasibility assessment ({datetime.now():%Y-%m-%d %H:%M:%S})",
                tags=["SDK", "Feasibility"] + list(self.params.job_tags),
                processing_step_url=step.self_link(),
                start=True,
                allow_output_data_deletion=True,
                input_data_slots=slots,
            )
        except BaseException:
            if self.params.remove_data and (job is None or getattr(job, "job_hco", None) is None):
                _delete_owned_inputs(owned_inputs)
            raise
        assert job is not None
        return FeasibilityJob(
            job,
            timeout_sec=self.params.job_completion_timeout_sec,
            remove_data=self.params.remove_data,
            input_workdata=owned_inputs,
        )

    def _step(self, name: str):
        if self.params.step_version is not None:
            return from_name(self.client, name, version=self.params.step_version)
        return find_processing_step(self.client, name)

    def _upload_json(self, filename: str, payload: Mapping[str, Any], *, secret: bool = False) -> WorkDataLink:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        root = enter_jma(self.client).work_data_root_link.navigate()
        link = root.upload_action.execute(UploadParameters(filename=filename, binary=encoded, mediatype="application/json", json=None))
        if secret:
            _mark_workdata_secret(_OwnedWorkData(filename, link, sensitive=True))
        return link
