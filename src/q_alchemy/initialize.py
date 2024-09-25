import hashlib
import inspect
import os
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from typing import List, Tuple, Dict

import httpx
import numpy as np
from httpx import HTTPTransport
from pinexq_client.core import MediaTypes
from pinexq_client.core.hco.upload_action_hco import UploadParameters
from pinexq_client.job_management import enter_jma, Job
from pinexq_client.job_management.hcos import WorkDataLink
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
    job_completion_timeout_sec: int | None = field(default=300)
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
            timeout=opt_params.job_completion_timeout_sec + 10.0
            if opt_params.job_completion_timeout_sec is not None
            else None,
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


def upload_statevector(client: httpx.Client, state_vector: np.ndarray, opt_params: OptParams) -> WorkDataLink:
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

    return wd_link


def populate_opt_params(opt_params: dict | OptParams | None = None, **kwargs) -> OptParams:
    if opt_params is None:
        opt_params = OptParams()
    elif isinstance(opt_params, OptParams):
        opt_params = opt_params
    else:
        opt_params = OptParams(**opt_params)

    for attr in kwargs:
        if hasattr(opt_params, attr):
            setattr(opt_params, attr, kwargs[attr])
    return opt_params


def configure_job(client: httpx.Client, statevector_link: WorkDataLink, opt_params: OptParams) -> Job:
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
        .assign_input_dataslot(0, work_data_link=statevector_link)
        .allow_output_data_deletion()
    )
    return job


def extract_result(job: Job):
    result_summary: dict = job.refresh().get_result()
    inner_job = job._job
    if result_summary["status"].startswith("OK"):
        qasm_wd = [
            wd for s in inner_job.output_dataslots
            for wd in s.assigned_workdatas if wd.name == "qasm_circuit.qasm"
        ][0]
        if qasm_wd.size_in_bytes > 0:
            qasm: str = qasm_wd.download_link.download().decode("utf-8")
        else:
            raise IOError("Q-Alchemy API call failed for unknown reasons.")
    else:
        raise IOError(f"Q-Alchemy API call failed. Reason: {result_summary['status']}.")
    return result_summary, qasm


def clean_up_job(job: Job, opt_params: OptParams) -> None:
    # Clean-up now.
    inner_job = job._job
    if opt_params.remove_data and inner_job is not None:
        for od in inner_job.output_dataslots:
            for wd in od.assigned_workdatas:
                delete_action = wd.delete_action
                if delete_action is not None:
                    delete_action.execute()
        job.refresh().delete()


def q_alchemy_as_qasm(state_vector: List[complex] | np.ndarray, opt_params: dict | OptParams | None = None,
                      client: httpx.Client | None = None, return_summary=False,  **kwargs) -> str | Tuple[str, dict]:

    opt_params: OptParams = populate_opt_params(opt_params, **kwargs)
    client = client if client is not None else create_client(opt_params)
    statevector_link = upload_statevector(client, state_vector, opt_params)

    job_timeout = (
        opt_params.job_completion_timeout_sec * 1000
        if opt_params.job_completion_timeout_sec is not None
        else 24 * 60 * 60 * 1000
    )
    job = (
       configure_job(client, statevector_link, opt_params)
        .start()
        .wait_for_state(JobStates.completed, timeout_ms=job_timeout)
    )
    result_summary, qasm = extract_result(job)
    clean_up_job(job, opt_params)

    if return_summary:
        return qasm, result_summary
    else:
        return qasm


def q_alchemy_as_qasm_parallel(state_vector: List[complex] | np.ndarray, opt_params: List[dict | OptParams], client: httpx.Client | None = None, return_summary=False):
    from threading import Thread
    from tqdm import tqdm

    threads = []
    result = []
    for opt in opt_params:
        def func(_opt):
            sp_qasm = q_alchemy_as_qasm(state_vector, _opt, client, return_summary)
            result.append(sp_qasm)

        job = Thread(target=func, args=(opt,))
        job.start()
        sleep(0.05)  # be easy on the API
        threads.append(job)

    # print(f"Waiting for {len(threads)} jobs to finish.")
    for x in tqdm(threads):
        x.join()

    return result


def q_alchemy_as_qasm_parallel_states(state_vector: List[List[complex] | np.ndarray], opt_params: dict | OptParams, client: httpx.Client | None = None, return_summary=False):
    from threading import Thread
    from tqdm import tqdm

    threads = []
    result = []
    for vec in state_vector:
        def func(_vec):
            sp_qasm = q_alchemy_as_qasm(_vec, opt_params, client, return_summary)
            result.append(sp_qasm)

        job = Thread(target=func, args=(vec,))
        job.start()
        sleep(0.05)  # be easy on the API
        threads.append(job)

    # print(f"Waiting for {len(threads)} jobs to finish.")
    for x in tqdm(threads):
        x.join()

    return result
