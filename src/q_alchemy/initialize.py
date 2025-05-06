import hashlib
import inspect
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from time import sleep
from typing import List, Tuple, Dict

from threading import Thread, Lock
from tqdm import tqdm

import httpx
import numpy as np
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq
from httpx import HTTPTransport
from pinexq_client.core import MediaTypes
from pinexq_client.core.hco.upload_action_hco import UploadParameters
from pinexq_client.job_management import enter_jma, Job, ProcessingStep
from pinexq_client.job_management.hcos import WorkDataLink
from pinexq_client.job_management.model import WorkDataQueryParameters, WorkDataFilterParameter, \
    SetTagsWorkDataParameters, JobStates

from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow


@dataclass
class OptParams:
    remove_data: bool = field(default=True)
    max_fidelity_loss: float = field(default=0.0)
    job_tags: List[str] = field(default_factory=list)
    api_key: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY"))
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = field(default="https")
    added_headers: Dict[str, str] = field(default_factory=dict)
    isometry_scheme: str = field(default="ccd")
    unitary_scheme: str = field(default="qsd")
    job_completion_timeout_sec: int | None = field(default=300)
    basis_gates: List[str] = field(default_factory=lambda: ["u", "cx"])
    image_size: Tuple[int, int] = field(default=(-1, -1))
    with_debug_data: bool = field(default=False)
    assign_data_hash: bool = field(default=True)
    use_research_function: str | None = field(default=None)
    extra_kwargs: dict = field(default_factory=dict)

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


def hash_state_vector(buffer: io.BytesIO, opt_params: OptParams):
    if opt_params.assign_data_hash:
        param_hash = hashlib.md5(buffer.read()).hexdigest()
        buffer.seek(0)
    else:
        param_hash = datetime.now(UTC).timestamp()
    return param_hash


def upload_statevector(client: httpx.Client, state_vector: pa.Table, opt_params: OptParams) -> WorkDataLink:
    # Convert to buffer to get hash and later possibly upload
    buffer = io.BytesIO()
    pq.write_table(state_vector, buffer)
    buffer.seek(0)
    param_hash = hash_state_vector(buffer, opt_params)

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
            filename=f"{param_hash}.parquet",
            binary=buffer.read(),
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


def create_processing_input(opt_params: OptParams) -> tuple[str, dict[str, float | list[str]]]:
    processing_name = "convert_circuit_layers_qasm_only"
    job_parameters = dict(
        min_fidelity=1.0 - opt_params.max_fidelity_loss,
        basis_gates=opt_params.basis_gates,
    )
    if opt_params.use_research_function is None and all(i > 0 for i in opt_params.image_size) or opt_params.with_debug_data:
        processing_name = "convert_circuit_layers"
        job_parameters.update(dict(
            image_shape_x=opt_params.image_size[0],
            image_shape_y=opt_params.image_size[1]
        ))
    elif opt_params.use_research_function is not None:
        processing_name = opt_params.use_research_function
        if processing_name == "baa_tucker_initialize":
            job_parameters.update(dict(
                options={
                    "isometry_scheme": opt_params.isometry_scheme,
                    "unitary_scheme": opt_params.unitary_scheme,
                    "strategy": opt_params.extra_kwargs.get("strategy", "BruteForce"),
                    "max_combination_size": opt_params.extra_kwargs.get("max_combination_size", 0),
                    "use_low_rank": opt_params.extra_kwargs.get("use_low_rank", False),
                    "initializer": opt_params.extra_kwargs.get("initializer", "BruteForceTuckerInitialize")
                }
            ))
        elif processing_name == "pivot_initialize":
            job_parameters.update(dict(
                options={
                    "aux": opt_params.extra_kwargs.get("aux", False),
                    "initializer": opt_params.extra_kwargs.get("initializer", "BaaTuckerInitialize")
                }
            ))
        elif processing_name == "brute_force_tucker_initialize":
            job_parameters.update(dict(
                options={
                    "max_blocks": opt_params.extra_kwargs.get("max_blocks", 3)
                }
            ))

    return processing_name, job_parameters


def find_processing_step(client, processing_name):
    from pinexq_client.job_management.model import ProcessingStepQueryParameters
    from pinexq_client.job_management.model import ProcessingStepFilterParameter
    from pinexq_client.job_management.model import FunctionNameMatchTypes

    query_param = ProcessingStepQueryParameters(
        filter=ProcessingStepFilterParameter(
            function_name=processing_name,
            function_name_match_type=FunctionNameMatchTypes.match_exact
        )
    )
    processing_step_root = enter_jma(client).processing_step_root_link.navigate()
    query_result = processing_step_root.query_action.execute(query_param)
    if len(query_result.processing_steps) < 1:
        raise NameError(
            f"There is a misconfiguration. Please contact customer support. "
            f"We are very sorry! Reason: no step known for function name: {processing_name}"
        )
    sorted(query_result.processing_steps, key=lambda x: x.version, reverse=True)
    step = ProcessingStep.from_hco(query_result.processing_steps[0])

    return step


def configure_job(client: httpx.Client, statevector_link: WorkDataLink, opt_params: OptParams) -> Job:
    processing_name, job_parameters = create_processing_input(opt_params)
    step = find_processing_step(client, processing_name)
    job = (
        Job(client)
        .create(name=f'Execute Transformation ({datetime.now()})')
        .select_processing(processing_step_instance=step)
        .configure_parameters(**job_parameters)
        .assign_input_dataslot(0, work_data_link=statevector_link)
        .allow_output_data_deletion()
    )
    return job


def extract_result(job: Job):
    result_summary: dict = job.refresh().get_result()
    if result_summary["status"].startswith("OK"):
        qasm_wd = [
            wd for s in job.get_output_data_slots()
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
    if opt_params.remove_data:
        for od in job.get_output_data_slots():
            for wd in od.assigned_workdatas:
                delete_action = wd.delete_action
                if delete_action is not None:
                    delete_action.execute()
        job.refresh().delete()


def q_alchemy_as_qasm(
        state_vector: List[complex] | np.ndarray | sparse.coo_array | sparse.coo_matrix,
        opt_params: dict | OptParams | None = None,
        client: httpx.Client | None = None,
        return_summary=False,
        **kwargs
) -> str | Tuple[str, dict]:

    opt_params: OptParams = populate_opt_params(opt_params, **kwargs)
    client = client if client is not None else create_client(opt_params)

    # The state vector need to be converted to a (1, 2**n) sparse (COO) matrix
    if isinstance(state_vector, sparse.coo_array):
        data_matrix: sparse.coo_matrix = sparse.coo_matrix(state_vector.reshape(1, -1)).reshape(1, -1)
    else:
        data_matrix: sparse.coo_matrix = sparse.coo_matrix(state_vector).reshape(1, -1)
    data_matrix_pyarrow: pa.Table = convert_sparse_coo_to_arrow(data_matrix)
    statevector_link = upload_statevector(client, data_matrix_pyarrow, opt_params)

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


def q_alchemy_as_qasm_parallel_states(
        state_vector: List[List[complex] | np.ndarray | sparse.coo_array | sparse.coo_matrix],
        opt_params: dict | OptParams,
        client: httpx.Client | None = None,
        return_summary=False,
        **kwargs
) -> List[str | Tuple[str, dict]]:

    opt_params: OptParams = populate_opt_params(opt_params, **kwargs)
    client = client if client is not None else create_client(opt_params)

    result = []
    result_lock = Lock()
    threads = []

    def run_single(vec, opt_params, client):
        try:
            qasm_or_pair = q_alchemy_as_qasm(
                vec,
                opt_params=opt_params,
                client=client,
                return_summary=return_summary,
                **kwargs
            )
            with result_lock:
                result.append(qasm_or_pair)
        except Exception as e:
            print(f"Error processing vector: {e}")

    for vec in state_vector:
        t = Thread(target=run_single, args=(vec, opt_params, client,))
        t.start()
        threads.append(t)
        sleep(0.2)  # Slight delay to avoid spamming requests

    for t in tqdm(threads, desc="Running Jobs", unit="jobs"):
        t.join()

    return result
