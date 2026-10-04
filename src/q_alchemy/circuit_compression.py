"""Remote circuit compression through the compress_circuit PineXQ step.

No compression engine or simulator is installed by this client. Call
``CircuitCompressionService().compress(circuit).result()`` to retrieve a report.
The default input semantics assume |0...0>; pass an operator-mode request for
arbitrary-input subroutines.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from typing import Any, Mapping

import httpx
from pinexq.client.core.hco.upload_action_hco import UploadParameters
from pinexq.client.job_management import Job, enter_jma
from pinexq.client.job_management.model import InputDataSlotParameter, JobStates

from .circuit_compression_contract import CircuitCompressionReport, CircuitCompressionRequest, _circuit_envelope
from .initialize import create_client, find_processing_step, from_name
from .quantum_io import _OwnedWorkData, _delete_job_then_inputs, _delete_owned_inputs, _download_json_output

COMPRESS_CIRCUIT_STEP = "compress_circuit"
CIRCUIT_INPUT_ALIAS = "quantum_circuit.json"
COMPRESSION_REQUEST_INPUT_ALIAS = "compression_request.json"
COMPRESSION_REPORT_OUTPUT_ALIAS = "circuit_compression_report.json"
LOG = logging.getLogger(__name__)


@dataclass
class CircuitCompressionParams:
    """Connection options. The wait timeout does not cancel a remote job.

    remove_data=True removes SDK-owned Jobs and WorkData after successful result
    retrieval. Failures preserve them for diagnosis and retry. step_version pins
    a deployed ProcessingStep; None uses the SDK's normal version discovery.
    """

    api_key: str | None = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"), repr=False)
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = "https"
    added_headers: dict[str, str] = field(default_factory=dict, repr=False)
    job_completion_timeout_sec: int | None = 300
    job_tags: list[str] = field(default_factory=list)
    remove_data: bool = True
    step_version: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CircuitCompressionParams":
        names = {item.name for item in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise TypeError(f"Unknown CircuitCompressionParams options: {sorted(unknown)}")
        return cls(**values)


class CircuitCompressionExecutionError(RuntimeError):
    """Remote execution or result retrieval failed; the job remains available."""

    def __init__(self, message: str, *, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class CircuitCompressionJob:
    def __init__(self, job: Job, *, timeout_sec: int | None, remove_data: bool,
                 input_workdata: list[_OwnedWorkData] | None = None):
        self._job = job
        self._timeout_sec = timeout_sec
        self._remove_data = remove_data
        self._input_workdata = list(input_workdata or [])
        self._result: CircuitCompressionReport | None = None
        self._job_removed = False
        self._cleanup_complete = False

    @property
    def raw_job(self) -> Job:
        if self._job_removed:
            raise RuntimeError("The PineXQ Job for this compression call has been removed")
        return self._job

    @property
    def removed(self) -> bool:
        return self._cleanup_complete

    def result(self, timeout: float | None = None) -> CircuitCompressionReport:
        """Wait, retrieve once, and clean up only after a valid report is cached.

        A timeout bounds this wait, not server execution. Retry result() using the
        same job handle after a timeout or a transient download failure.
        """
        if self._result is None:
            timeout_sec = timeout if timeout is not None else self._timeout_sec
            try:
                self._job.wait_for_state(JobStates.completed, polling_interval_s=0.25,
                                         timeout_s=float(timeout_sec) if timeout_sec is not None else 86400.0)
                self._result = CircuitCompressionReport.from_dict(
                    _download_json_output(self._job, COMPRESSION_REPORT_OUTPUT_ALIAS))
            except Exception as exc:
                LOG.warning("Circuit compression result retrieval failed; the PineXQ Job and WorkData were preserved for diagnosis and retry")
                detail = str(exc).rsplit("[/procon/error]", 1)[-1].strip() or type(exc).__name__
                raise CircuitCompressionExecutionError(
                    f"Circuit compression failed: {detail}", original_exception=exc) from None
        if self._remove_data and not self._cleanup_complete:
            self._job_removed, self._input_workdata = _delete_job_then_inputs(
                self._job, self._input_workdata, job_already_removed=self._job_removed)
            self._cleanup_complete = self._job_removed and not self._input_workdata
        return self._result


class CircuitCompressionService:
    """Submit standalone circuit compression to PineXQ.

    Accepts the SDK Circuit, a native Qiskit QuantumCircuit, or the schema-1
    quantum-circuit envelope returned by Feasibility. All computation stays on
    the service. The optional injected HTTP client remains owned by its caller.
    """

    def __init__(self, params: CircuitCompressionParams | Mapping[str, Any] | None = None,
                 *, client: httpx.Client | None = None, **kwargs: Any):
        if params is None:
            resolved = CircuitCompressionParams()
        elif isinstance(params, CircuitCompressionParams):
            resolved = replace(params)
        elif isinstance(params, Mapping):
            resolved = CircuitCompressionParams.from_dict(params)
        else:
            raise TypeError("params must be CircuitCompressionParams or a mapping")
        names = {item.name for item in fields(resolved)}
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f"Unknown CircuitCompressionService option {name!r}")
            setattr(resolved, name, value)
        owns_client = client is None
        if client is None:
            if not resolved.api_key:
                raise ValueError("A Q-Alchemy API key is required")
            client = create_client(resolved)
        self.params, self.client, self._owns_client = resolved, client, owns_client

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
            self._owns_client = False

    def __enter__(self) -> "CircuitCompressionService":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def compress(self, circuit: Any, request: CircuitCompressionRequest | None = None) -> CircuitCompressionJob:
        """Compress with zero-input semantics by default.

        For a subroutine use
        CircuitCompressionRequest(options={"equivalence": "operator"}).
        """
        if request is None:
            request = CircuitCompressionRequest()
        if not isinstance(request, CircuitCompressionRequest):
            raise TypeError("request must be a CircuitCompressionRequest")
        # Validate and serialize both inputs before creating any remote data.
        payloads = [(CIRCUIT_INPUT_ALIAS, _circuit_envelope(circuit)),
                    (COMPRESSION_REQUEST_INPUT_ALIAS, request.to_dict())]
        encoded = [(name, json.dumps(payload, allow_nan=False).encode("utf-8")) for name, payload in payloads]
        step = (from_name(self.client, COMPRESS_CIRCUIT_STEP, version=self.params.step_version)
                if self.params.step_version is not None else find_processing_step(self.client, COMPRESS_CIRCUIT_STEP))
        owned_inputs: list[_OwnedWorkData] = []
        job = None
        try:
            root = enter_jma(self.client).work_data_root_link.navigate()
            slots = []
            for index, (name, binary) in enumerate(encoded):
                link = root.upload_action.execute(UploadParameters(
                    filename=name, binary=binary, mediatype="application/json", json=None))
                owned_inputs.append(_OwnedWorkData(name, link))
                slots.append(InputDataSlotParameter(Index=index, WorkDataUrls=[str(link.get_url())]))
            job = Job(client=self.client)
            job.create_and_configure_rapidly(
                name=f"Circuit compression ({datetime.now():%Y-%m-%d %H:%M:%S})",
                tags=["SDK", "CircuitCompression"] + list(self.params.job_tags),
                processing_step_url=step.self_link(), start=True, allow_output_data_deletion=True,
                input_data_slots=slots)
        except BaseException:
            if self.params.remove_data and (job is None or getattr(job, "job_hco", None) is None):
                _delete_owned_inputs(owned_inputs)
            raise
        return CircuitCompressionJob(job, timeout_sec=self.params.job_completion_timeout_sec,
                                     remove_data=self.params.remove_data, input_workdata=owned_inputs)


__all__ = ["CircuitCompressionParams", "CircuitCompressionRequest", "CircuitCompressionReport",
           "CircuitCompressionService", "CircuitCompressionJob", "CircuitCompressionExecutionError"]
