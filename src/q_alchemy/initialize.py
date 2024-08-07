import hashlib
import inspect
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict

import httpx
import numpy as np
from httpx import HTTPTransport
from pinexq_client.core import MediaTypes
from pinexq_client.core.hco.upload_action_hco import UploadParameters
from pinexq_client.job_management import enter_jma, Job
from pinexq_client.job_management.model import WorkDataQueryParameters, WorkDataFilterParameter, \
    SetTagsWorkDataParameters, JobStates


@dataclass
class OptParams:
    remove_data: bool = field(default=True)
    max_fidelity_loss: float = field(default=0.0)
    job_tags: List[str] = field(default_factory=list)
    api_key: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY"))
    host: str = field(default="jobs.api.q-alchemy.com")
    schema: str = field(default="https")
    added_headers: Dict[str, str] = field(default_factory=dict)
    isometry_scheme: str = field(default="ccd")
    unitary_scheme: str = field(default="qsd")
    job_completion_timeout_sec: int = field(default=300)
    basis_gates: List[str] = field(default_factory=lambda: ["rx", "ry", "rz", "cx"])
    image_size: Tuple[int, int] = field(default=(-1, -1))
    with_debug_data: bool = field(default=False)
    assign_data_hash: bool = field(default=True)

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


def create_client(opt_params: OptParams):
    headers = {"x-api-key": opt_params.api_key}
    headers.update(opt_params.added_headers)

    client = httpx.Client(
        base_url=f"{opt_params.schema}://{opt_params.host}",
        headers=headers,
        timeout=httpx.Timeout(
            timeout=opt_params.job_completion_timeout_sec + 10.0,
            connect=10.0
        ),
        transport=HTTPTransport(retries=3)
    )
    return client


def hash_state_vector(state_vector: List[complex] | np.ndarray, opt_params: OptParams):
    if opt_params.assign_data_hash:
        param_hash = hashlib.md5(np.asarray(state_vector).tobytes()).hexdigest()
    else:
        param_hash = datetime.utcnow().timestamp()
    return param_hash


def q_alchemy_as_qasm(state_vector: List[complex] | np.ndarray, opt_params: dict | OptParams | None = None, client: httpx.Client | None = None, return_summary=False,  **kwargs) -> str:
    if opt_params is None:
        opt_params = OptParams()
    elif isinstance(opt_params, OptParams):
        opt_params = opt_params
    else:
        opt_params = OptParams(**opt_params)

    for attr in kwargs:
        if hasattr(opt_params, attr):
            setattr(opt_params, attr, kwargs[attr])

    client = client if client is not None else create_client(opt_params)
    param_hash = hash_state_vector(state_vector, opt_params)

    sequence_wd_tags = [
        f"Hash={param_hash}",
        "Source=Qiskit-Integration",
        f"ImageSize={opt_params.image_size}"
    ]
    sequence_wd_tags += opt_params.job_tags
    wd_root = enter_jma(client).work_data_root_link.navigate()

    existing_wd_query = wd_root.query_action.execute(WorkDataQueryParameters(
        filter=WorkDataFilterParameter(tags_by_and=sequence_wd_tags)
    ))

    if existing_wd_query.total_entities == 0:
        wd_root = enter_jma(client).work_data_root_link.navigate()
        wd_link = wd_root.upload_action.execute(UploadParameters(
            filename=f"{param_hash}.bin",
            binary=np.asarray(state_vector, dtype=np.complex128).tobytes(),
            mediatype=MediaTypes.OCTET_STREAM,
        ))
        wd_link.navigate().edit_tags_action.execute(
            SetTagsWorkDataParameters(tags=sequence_wd_tags)
        )
    else:
        wd_link = existing_wd_query.workdatas[0].self_link

    job_timeout = opt_params.job_completion_timeout_sec * 1000
    processing_name = "convert_circuit_layers_qasm_only"
    job_parameters = dict(
        min_fidelity=1.0 - opt_params.max_fidelity_loss,
        basis_gates=opt_params.basis_gates,
    )
    if all(i > 0 for i in opt_params.image_size) or opt_params.with_debug_data:
        processing_name = "convert_circuit_layers"
        job_parameters.update(dict(
            image_shape_x=opt_params.image_size[0],
            image_shape_y=opt_params.image_size[1]
        ))
    job = (
        Job(client)
        .create(name=f'Execute Transformation ({datetime.now()})')
        .select_processing(function_name=processing_name)
        .configure_parameters(**job_parameters)
        .assign_input_dataslot(0, work_data_link=wd_link)
        .allow_output_data_deletion()
        .start()
        .wait_for_state(JobStates.completed, timeout_ms=job_timeout)
    )
    result_summary: dict = job.get_result()
    inner_job = job._job
    if result_summary["status"].startswith("OK"):
        qasm_wd = \
        [wd for s in inner_job.output_dataslots for wd in s.assigned_workdatas if wd.name == "qasm_circuit.qasm"][0]
        if qasm_wd.size_in_bytes > 0:
            qasm: str = qasm_wd.download_link.download().decode("utf-8")
        else:
            raise IOError("Q-Alchemy API call failed for unknown reasons.")
    else:
        raise IOError(f"Q-Alchemy API call failed. Reason: {result_summary['status']}.")
    # Clean-up now.
    if opt_params.remove_data and inner_job is not None:
        for od in inner_job.output_dataslots:
            for wd in od.assigned_workdatas:
                delete_action = wd.delete_action
                if delete_action is not None:
                    delete_action.execute()
        job.refresh().delete()

    if return_summary:
        return qasm, result_summary
    else:
        return qasm
