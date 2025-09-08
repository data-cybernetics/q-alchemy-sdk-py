import time
import base64
import json
import hashlib
import inspect
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from time import sleep
from typing import List, Tuple, Dict, Optional

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
    SetTagsWorkDataParameters, JobStates, RapidJobSetupParameters, InputDataSlotParameter

from q_alchemy.utils import is_power_of_two
from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow

# 1MB state vectors (16 bytes/amplitude * 2**16 amplitudes = 1048576 bytes)
USE_INLINE_STATE_NUM_QUBITS = 16


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


def encode_statevector(state_vector: pa.Table) -> str:
    buffer = io.BytesIO()
    pq.write_table(state_vector, buffer)
    buffer.seek(0)
    return base64.encodebytes(buffer.read()).decode("utf-8").replace("\n", "")


def upload_statevector(client: httpx.Client, state_vector: pa.Table, opt_params: OptParams) -> WorkDataLink:
    # Convert to buffer to get hash and later possibly upload
    buffer = io.BytesIO()
    pq.write_table(state_vector, buffer)
    buffer.seek(0)
    param_hash = hash_state_vector(buffer, opt_params)

    sequence_wd_tags = [
        f"Hash={param_hash}",
        "Source=Qiskit-Integration"
    ]
    sequence_wd_tags += opt_params.job_tags
    wd_root = enter_jma(client).work_data_root_link.navigate()

    existing_wd_query = wd_root.query_action.execute(WorkDataQueryParameters(
        Filter=WorkDataFilterParameter(
            TagsByAnd=sequence_wd_tags,
            NameContains=None,
            ShowHidden=None,
            MediaTypeContains=None,
            TagsByOr=None,
            IsKind=None,
            CreatedBefore=None,
            CreatedAfter=None,
            IsDeletable=None,
            IsUsed=None,
            ProducerProcessingStepUrl=None,
        ),
        SortBy=None,
        IncludeRemainingTags=None,
        Pagination=None,
    ))

    if existing_wd_query.total_entities == 0:
        wd_root = enter_jma(client).work_data_root_link.navigate()
        wd_link = wd_root.upload_action.execute(UploadParameters(
            filename=f"{param_hash}.parquet",
            binary=buffer.read(),
            mediatype=MediaTypes.OCTET_STREAM,
            json=None,
        ))
        wd_link.navigate().edit_tags_action.execute(
            SetTagsWorkDataParameters(Tags=sequence_wd_tags)
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


def create_processing_input(opt_params: OptParams, statevector_data: WorkDataLink | str) -> tuple[str, dict[str, float | list[str]]]:
    processing_name = "build_initialization_circuit"
    job_parameters: Dict[str, str | float | int | bool | dict] = {
        "min_fidelity": 1.0 - opt_params.max_fidelity_loss,
        "basis_gates": opt_params.basis_gates,
    }
    if isinstance(statevector_data, str):
        processing_name = "build_initialization_circuit_inline"
        job_parameters.update({
            "state_vector": {
               "state_vector_base64":statevector_data,
               "state_vector_type":"parquet"
           }
        })
    elif opt_params.use_research_function is not None:
        processing_name = opt_params.use_research_function

    return processing_name, job_parameters

class TimeAwareCache:
    """
    A simple time-based (TTL) in-memory cache.

    Stores key-value pairs with associated timestamps. Items expire
    after a specified time-to-live (TTL), ensuring they are automatically
    invalidated and removed on access if outdated.
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize the cache with a given TTL.

        Args: ttl_seconds (int): Time-to-live for each cache entry in seconds.
        """
        self._store = {}  # Internal storage for (timestamp, value) tuples
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[object]:
        """
        Retrieve a value from the cache if it hasn't expired.

        Args: key (str): The key to look up.

        Returns: The cached value if present and valid; otherwise, None.
        """
        item = self._store.get(key)
        if item:
            timestamp, value = item
            if time.time() - timestamp < self.ttl:
                return value

            # Entry has expired; remove it
            del self._store[key]

        return None

    def set(self, key: str, value: object):
        """
        Store a value in the cache with the current timestamp.

        Args: key (str): The key under which to store the value.
              value (object): The value to cache.
        """
        self._store[key] = (time.time(), value)

step_cache = TimeAwareCache(ttl_seconds=300)

def from_name(
    client: httpx.Client,
    step_name: str,
    version: str = None
) -> ProcessingStep:
    """Create a ProcessingStep object from an existing name.

    Args:
        client: Create a ProcessingStep object from an existing name.
        step_name: Name of the registered processing step.
        version: Version of the ProcessingStep to be created

    Returns:
        The newly created processing step as `ProcessingStep` object
    """

    # Attempt to find the processing step
    query_result = ProcessingStep._query_processing_steps(client, step_name, version)

    # Check if at least one result is found
    if len(query_result.processing_steps) == 0:
        # Attempt to suggest alternative steps if exact match not found
        suggested_steps = ProcessingStep._processing_steps_by_name(client, step_name)
        raise NameError(
            f"No processing step with the name {step_name} and version {version} registered. "
            f"Suggestions: {suggested_steps}"
        )

    sorted(query_result.processing_steps, key=lambda x: x.version, reverse=True)
    processing_step_hco = query_result.processing_steps[0]

    return ProcessingStep.from_hco(processing_step_hco)

def find_processing_step(client, processing_name):
    step_key = str(client.base_url) + '/' + processing_name
    step = step_cache.get(step_key)

    if step is None:
        step = from_name(client=client, step_name=processing_name, version=None)
        step_cache.set(step_key, step)

    return step

def configure_job(
    client: httpx.Client,
    opt_params: OptParams,
    statevector_data: WorkDataLink | str
) -> Job:
    processing_name, job_parameters = create_processing_input(opt_params, statevector_data)
    step = find_processing_step(client, processing_name)

    if isinstance(statevector_data, WorkDataLink):
        job_parameters = RapidJobSetupParameters(
            Name=f'Execute Transformation ({datetime.now()})',
            parameters=json.dumps(job_parameters),
            ProcessingStepUrl=str(step.self_link().get_url()),
            Tags=["SDK", "WorkDataLink"],
            AllowOutputDataDeletion=True,
            Start=True,
            InputDataSlots=[
                InputDataSlotParameter(
                    Index=0,
                    WorkDataUrls=[str(statevector_data.get_url())]
                )
            ]
        )
    else:
        job_parameters = RapidJobSetupParameters(
            Name=f'Execute Transformation ({datetime.now()})',
            parameters=json.dumps(job_parameters),
            ProcessingStepUrl=str(step.self_link().get_url()),
            Tags=["SDK", "InLine"],
            AllowOutputDataDeletion=True,
            Start=True
        )

    # create_and_... does not unpack job_parameters (for later pinexqq-client)
    job = Job(client=client).create_and_configure_rapidly(
        name=job_parameters.name,
        tags=job_parameters.tags,
        processing_step_url=step.self_link(), #needs the ProcessingStepLink itself
        start=job_parameters.start,
        parameters=job_parameters.parameters,
        allow_output_data_slots=job_parameters.allow_output_data_deletion, #misleading keyword name, also does not match
        input_data_slots=job_parameters.input_data_slots,
    )
    return job

def extract_result(job: Job):
    # the inline job returns [str, dict], while the dataslot job returns dict only...
    res = job.refresh().get_result()
    if isinstance(res, list):
        qasm = res[0]
        result_summary = res[1]
        if result_summary["status"].startswith("OK"):
            return result_summary, qasm
        else:
            raise IOError("Q-Alchemy API call failed for unknown reasons.")
    elif isinstance(res, dict):
        result_summary = res
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
    else:
        raise IOError("Unknown return value.")


def clean_up_job(job: Job, opt_params: OptParams, num_qubits: int) -> None:
    # Clean-up now.
    if opt_params.remove_data:
        job.delete_with_associated(
            delete_subjobs_with_data=True,
            delete_input_workdata=num_qubits > USE_INLINE_STATE_NUM_QUBITS,
            delete_output_workdata=True,
        )


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

    # Now we decide if we use inline state-vectors
    # (saves hussle and resources) or if we use the
    # work-data approach:
    # currently, all states <= 16 qubits are going inline.
    num_qubits = np.log2(data_matrix.shape[1])
    if not is_power_of_two(data_matrix):
        raise ValueError(
            f"The state vector is not a power of two. "
            f"The length of the state vector is {data_matrix.shape[1]}."
        )
    if num_qubits > USE_INLINE_STATE_NUM_QUBITS or opt_params.use_research_function is not None:
        statevector_data = upload_statevector(client, data_matrix_pyarrow, opt_params)
    else:
        statevector_data = encode_statevector(data_matrix_pyarrow)

    job_timeout = (
        opt_params.job_completion_timeout_sec * 1000
        if opt_params.job_completion_timeout_sec is not None
        else 24 * 60 * 60 * 1000
    )

    job = configure_job(
        client=client,
        opt_params=opt_params,
        statevector_data=statevector_data
    )

    job.wait_for_state(
        state=JobStates.completed,
        polling_interval_ms=250,
        timeout_ms=job_timeout
    )

    result_summary, qasm = extract_result(job)
    clean_up_job(job, opt_params, num_qubits)

    if return_summary:
        return qasm, result_summary

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
