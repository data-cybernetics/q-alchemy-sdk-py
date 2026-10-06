import warnings
import logging
import time
import base64
import json
import hashlib
import inspect
import io
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import StrEnum
from typing import List, Tuple, Dict, Optional

from threading import Lock

import httpx
import numpy as np
from packaging.version import Version, InvalidVersion
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq
from httpx import HTTPTransport
from pinexq.client.core import MediaTypes
from pinexq.client.core.hco.upload_action_hco import UploadParameters
from pinexq.client.job_management import enter_jma, Job, ProcessingStep
from pinexq.client.job_management.hcos import WorkDataLink
from pinexq.client.job_management.model import WorkDataQueryParameters, WorkDataFilterParameter, \
    SetTagsWorkDataParameters, JobStates, RapidJobSetupParameters, InputDataSlotParameter

from q_alchemy.utils import is_power_of_two, nonnegative_integer
from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow

# 1MB state vectors (16 bytes/amplitude * 2**16 amplitudes = 1048576 bytes)
USE_INLINE_STATE_NUM_QUBITS = 16

LOG = logging.getLogger(__name__)

class InitializationMethods(StrEnum):
    AUTO = "auto"
    HIERARCHICAL_TUCKER = "hierarchical_tucker"
    ITERATIVE_TUCKER = "iterative_tucker"

@dataclass
class OptParams:
    remove_data: bool = field(default=True)
    max_fidelity_loss: float = field(default=0.0)
    job_tags: List[str] = field(default_factory=list)
    api_key: str | None = field(
        default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"),
        repr=False,
    )
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = field(default="https")
    added_headers: Dict[str, str] = field(default_factory=dict, repr=False)
    job_completion_timeout_sec: int | None = field(default=300)
    basis_gates: List[str] = field(default_factory=lambda: ["u", "cx"])
    assign_data_hash: bool = field(default=True)
    use_research_function: str | None = field(default=None)
    use_qasm3: bool = field(default=False)
    initialization_method: InitializationMethods = field(default=InitializationMethods.AUTO)
    extra_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


def create_client(opt_params: OptParams):
    if not opt_params.api_key:
        raise ValueError(
            "A Q-Alchemy API key is required. Set Q_ALCHEMY_API_KEY or "
            "PINEXQ_API_KEY, or pass api_key=... in OptParams."
        )

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
    warnings.warn("hash_state_vector is deprecated; the SDK manages upload identities internally", DeprecationWarning, stacklevel=2)
    if opt_params.assign_data_hash:
        param_hash = hashlib.md5(buffer.read()).hexdigest()
        buffer.seek(0)
    else:
        param_hash = datetime.now(UTC).timestamp()
    return param_hash


def encode_statevector(state_vector: pa.Table | np.ndarray) -> str:
    warnings.warn("encode_statevector is deprecated; use the SDK state upload API", DeprecationWarning, stacklevel=2)
    return base64.b64encode(_serialize_statevector(state_vector)).decode("ascii")


def _serialize_statevector(state_vector: pa.Table | np.ndarray) -> bytes:
    buffer = io.BytesIO()
    if isinstance(state_vector, np.ndarray):
        np.save(buffer, state_vector, allow_pickle=False)
    else:
        pq.write_table(state_vector, buffer)
    return buffer.getvalue()


def _payload_type(payload: bytes) -> str:
    """Identify the two formats produced by our serializer."""
    return "numpy_load" if payload.startswith(b"\x93NUMPY") else "parquet"


def allow_deletion(wd_link: WorkDataLink) -> None:
    """Mark WorkData deletable, unless it already is.

    Do it right after the upload: the platform stops offering AllowDeletion
    while a job uses the WorkData, and the upload's creator is then the only one
    to send it -- which matters because AllowDeletion is not idempotent. Marking
    only permits deletion; nothing is deleted unless remove_data asks for it.
    """
    try:
        wd = wd_link.navigate()
        if not wd.is_deletable and wd.allow_deletion_action.is_available():
            wd.allow_deletion_action.execute()
    except Exception:
        # A concurrent call sharing the WorkData can race on AllowDeletion.
        LOG.warning("Could not mark WorkData %s deletable.", wd_link.get_url(), exc_info=True)


def upload_statevector(client: httpx.Client, state_vector: pa.Table | np.ndarray, opt_params: OptParams) -> WorkDataLink:
    return _upload_statevector_payload(client, _serialize_statevector(state_vector), opt_params)


def _upload_statevector_payload(client: httpx.Client, payload: bytes, opt_params: OptParams) -> WorkDataLink:
    param_hash = (
        hashlib.md5(payload).hexdigest()
        if opt_params.assign_data_hash else datetime.now(UTC).timestamp()
    )

    sequence_wd_tags = [
        f"Hash={param_hash}",
        "Source=Qiskit-Integration"
    ]
    sequence_wd_tags += opt_params.job_tags
    wd_root = enter_jma(client).work_data_root_link.navigate()

    existing_wd_query = None
    if opt_params.assign_data_hash:
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

    if existing_wd_query is None or existing_wd_query.total_entities == 0:
        wd_link = wd_root.upload_action.execute(UploadParameters(
            filename=f"{param_hash}.npy" if _payload_type(payload) == "numpy_load" else f"{param_hash}.parquet",
            binary=payload,
            mediatype=MediaTypes.OCTET_STREAM,
            json=None,
        ))
        wd_link.navigate().edit_tags_action.execute(
            SetTagsWorkDataParameters(Tags=sequence_wd_tags)
        )
        allow_deletion(wd_link)
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


def create_processing_input(opt_params: OptParams, statevector_data: WorkDataLink | str,
                            num_states: int = 1, statevector_type: str = "parquet") -> tuple[str, dict[str, float | list[str]]]:
    processing_name = "build_initialization_circuit"
    job_parameters: Dict[str, str | float | int | bool | dict] = {
        "min_fidelity": 1.0 - opt_params.max_fidelity_loss,
        "basis_gates": opt_params.basis_gates,
        "options": {
            "method": opt_params.initialization_method,
            "use_qasm3": opt_params.use_qasm3,
            "opt_params": json.dumps(opt_params.extra_kwargs)
        }
    }

    if isinstance(statevector_data, str):
        processing_name = "build_initialization_circuit_inline"
        job_parameters.update({
            "state_vector": {
               "state_vector_base64":statevector_data,
               "state_vector_type":statevector_type
           }
        })
    elif num_states > 1:
        processing_name = "build_initialization_circuits"
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
        self.lock = Lock()

    def get(self, key: str) -> Optional[object]:
        """
        Retrieve a value from the cache if it hasn't expired.

        Args: key (str): The key to look up.

        Returns: The cached value if present and valid; otherwise, None.
        """
        item = self._store.get(key)
        if item:
            timestamp, value = item
            if time.monotonic() - timestamp < self.ttl:
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
        self._store[key] = (time.monotonic(), value)

_step_cache_lock = Lock()

def _release_version(version: str) -> Optional[Version]:
    """The version as a `Version` if it is a final release, else None (dev, pre-release or unparseable)."""
    try:
        parsed = Version(version)
    except (InvalidVersion, TypeError):
        return None
    return None if parsed.is_prerelease else parsed

def from_name(
    client: httpx.Client,
    step_name: str,
    version: str = None
) -> ProcessingStep:
    """Create a ProcessingStep object from an existing name.

    Args:
        client: Create a ProcessingStep object from an existing name.
        step_name: Name of the registered processing step.
        version: Version of the ProcessingStep to be created. If None, the newest
            released version is used; dev and pre-release versions are skipped.

    Returns:
        The newly created processing step as `ProcessingStep` object
    """

    # Attempt to find the processing step
    query_result = ProcessingStep._query_processing_steps(client, step_name, version)

    # Check if at least one result is found
    if len(query_result.processing_steps) == 0:
        # NOTE: no suggestion lookup here — the private helper this used to
        # call (ProcessingStep._processing_steps_by_name) no longer exists in
        # pinexq-client >= 9.7, which turned this error path into a confusing
        # AttributeError that masked the actual problem.
        version_part = f" (version {version})" if version else ""
        raise NameError(
            f"No processing step named '{step_name}'{version_part} is registered "
            f"or visible to this account on {client.base_url}. If the step was "
            f"deployed recently, it may not have been set public yet."
        )

    if version is not None:
        processing_step_hco = query_result.processing_steps[0]
    else:
        # The SDK is the public interface: never hand out dev or pre-release steps.
        releases = [(v, step) for step in query_result.processing_steps
                    if (v := _release_version(step.version)) is not None]
        if not releases:
            raise NameError(
                f"Processing step '{step_name}' has no released version on {client.base_url}, "
                f"only: {', '.join(str(step.version) for step in query_result.processing_steps)}."
            )
        # Newest first. Compare as versions, not strings: as text "0.10.0" sorts below "0.9.0".
        newest_first = sorted(releases, key=lambda pair: pair[0], reverse=True)
        processing_step_hco = newest_first[0][1]

    return ProcessingStep.from_hco(processing_step_hco)

def find_processing_step(client, processing_name):
    # ProcessingStep objects retain their HTTP client. Keep the cache on that
    # client so accounts cannot share lookups and a global cache cannot retain
    # closed clients. Callers should use a new client when changing accounts.
    with _step_cache_lock:
        cache = getattr(client, "_qalchemy_step_cache", None)
        if cache is None:
            cache = TimeAwareCache(ttl_seconds=300)
            client._qalchemy_step_cache = cache
    step_key = str(client.base_url) + '/' + processing_name
    with cache.lock:
        step = cache.get(step_key)
        if step is None:
            step = from_name(client=client, step_name=processing_name, version=None)
            cache.set(step_key, step)
        return step

def configure_job(
    client: httpx.Client,
    opt_params: OptParams,
    statevector_data: WorkDataLink | str,
    num_states: int = 1,
    statevector_type: str = "parquet",
) -> Job:
    processing_name, inner_job_parameters = create_processing_input(
        opt_params, statevector_data, num_states, statevector_type
    )
    step = find_processing_step(client, processing_name)

    # job_parameters
    if isinstance(statevector_data, WorkDataLink):
        job_parameters = RapidJobSetupParameters(
            Name=f'Execute Transformation ({datetime.now()})',
            Parameters=json.dumps(inner_job_parameters),
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
            Parameters=json.dumps(inner_job_parameters),
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
        allow_output_data_deletion=job_parameters.allow_output_data_deletion, #misleading keyword name, also does not match
        input_data_slots=job_parameters.input_data_slots,
    )
    return job

def extract_result(job: Job):
    # the inline job returns [str, dict], while the dataslot job returns dict only...
    # and the batch job returns None!
    res = job.refresh().get_result()
    match res: # should we figure out the cases in some other way?
        case None:
            result_wd = [
                    wd for s in job.get_output_data_slots()
                    for wd in s.assigned_workdatas if wd.name == "summaries.json"
                ][0]
            if result_wd.size_in_bytes > 0:
                result_str: str = result_wd.download_link.download().decode("utf-8")
                result_list = json.loads(result_str)
            else:
                raise IOError("Q-Alchemy API call failed for unknown reasons.")
            qasm_wd = [
                    wd for s in job.get_output_data_slots()
                    for wd in s.assigned_workdatas if wd.name == "qasm_circuit.qasm" #should probably be .json. oops.
                ][0]
            if qasm_wd.size_in_bytes > 0:
                qasm_str: str = qasm_wd.download_link.download().decode("utf-8")
                qasm_list = json.loads(qasm_str)
            else:
                raise IOError("Q-Alchemy API call failed for unknown reasons.")
            return result_list, qasm_list
        case [qasm, result_summary]:
            if result_summary["status"].startswith("OK"):
                return result_summary, qasm
            else:
                raise IOError("Q-Alchemy API call failed for unknown reasons.")
        case dict():
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
        case _:
            raise IOError("Unknown return value.")


def delete_job_with_data(job: Job) -> None:
    """Delete the job, its output WorkData and its uploaded input WorkData.

    Data lineage fixes the order: output WorkData must go before the job that
    produced it, and input WorkData only once no job uses it. So
    delete_with_associated handles the outputs and the job, and the inputs go
    afterwards -- here rather than in delete_with_associated, which would warn
    about every input another job still uses (state uploads are shared by hash);
    those are left for that job's clean-up.

    Uploads are marked deletable when they are made. An input uploaded before
    that was done can only be marked now, once the job no longer uses it.
    """
    job.refresh()
    inputs = [wd.self_link for slot in job.job_hco.input_dataslots for wd in slot.selected_workdatas]
    job.delete_with_associated(
        delete_subjobs_with_data=True,
        delete_input_workdata=False,
        delete_output_workdata=True,
    )
    for link in inputs:
        allow_deletion(link)
        try:
            wd = link.navigate()
            if wd.delete_action.is_available():
                wd.delete_action.execute()
            else:
                LOG.info("Input WorkData %s is still in use; leaving it.", link.get_url())
        except Exception:
            # A concurrent call sharing the upload can delete it first.
            LOG.warning("Could not delete input WorkData %s.", link.get_url(), exc_info=True)


def clean_up_job(job: Job, opt_params: OptParams) -> None:
    if opt_params.remove_data:
        delete_job_with_data(job)


def run_job(job: Job, opt_params: OptParams, timeout_s: float):
    """Wait for the job, extract its result, and clean up whether or not it succeeded.

    A failed job (e.g. an option the chosen method rejects) must not leak the job
    and its uploaded state when remove_data is set. On that path a clean-up error
    is only logged, so it cannot mask the reason the job failed.
    """
    try:
        job.wait_for_state(
            state=JobStates.completed,
            polling_interval_s=0.25,
            timeout_s=timeout_s
        )
        result = extract_result(job)
    except BaseException:
        try:
            clean_up_job(job, opt_params)
        except Exception:
            LOG.warning("Could not clean up the failed Q-Alchemy job.", exc_info=True)
        raise
    clean_up_job(job, opt_params)
    return result


def q_alchemy_as_qasm(
        state_vector: List[complex] | np.ndarray | sparse.sparray,
        opt_params: dict | OptParams | None = None,
        client: httpx.Client | None = None,
        return_summary=False,
        **kwargs
) -> str | Tuple[str, dict]:
    """Prepare a state through the service.

    With ``return_summary=True``, AUTO's ``fidelity_requirement_met`` reports
    QTucker's decision independently of execution ``status``. False means best
    effort; missing/None means unavailable (older services or pinned methods).
    ``fidelity_loss`` is an estimate, not a simulated verification.
    """
    opt_params: OptParams = populate_opt_params(opt_params, **kwargs)
    owns_client = client is None
    client = client if client is not None else create_client(opt_params)
    try:
        return _q_alchemy_as_qasm(state_vector, opt_params, client, return_summary)
    finally:
        # A caller-supplied client belongs to the caller and is left open.
        if owns_client:
            client.close()


def _q_alchemy_as_qasm(
        state_vector: List[complex] | np.ndarray | sparse.sparray,
        opt_params: OptParams,
        client: httpx.Client,
        return_summary: bool,
) -> str | Tuple[str, dict]:
    payload, num_qubits = _prepare_statevector(state_vector)
    return _run_prepared_statevector(payload, num_qubits, opt_params, client, return_summary)


def _maybe_make_sparse_row(vector: np.ndarray):
    """Use QTucker's 10% density/early-exit rule, with lossless transport.

    Mirrors ``SparseMixin._maybe_make_sparse_triplet`` / its backend density
    check (QTucker 0.2.15), but uses eps_mass=0: no nonzero amplitude is dropped
    in the SDK. Compilation/approximation policy stays with QTucker. Scanning
    uses bounded blocks and avoids constructing COO indices for dense inputs.
    """
    limit = (vector.size + 9) // 10  # ceil(0.1 * size), without float rounding
    nnz = 0
    for start in range(0, vector.size, 1 << 20):
        nnz += np.count_nonzero(vector[start:start + (1 << 20)])
        if nnz > limit:
            return None
    return sparse.coo_matrix(vector.reshape(1, -1))


def _state_row(state_vector):
    """Validate width and choose a transport representation once per state."""
    if sparse.issparse(state_vector):
        row = sparse.coo_matrix(state_vector.reshape(1, -1))
    else:
        vector = np.asarray(state_vector).reshape(-1)
        if vector.dtype.kind not in "buifc":
            raise ValueError("State amplitudes must be numeric")
        row = vector.reshape(1, -1)
    if not is_power_of_two(row):
        raise ValueError(
            f"The state vector is not a power of two. "
            f"The length of the state vector is {row.shape[1]}."
        )
    if sparse.issparse(row):
        return row  # Caller already chose sparse storage: never scan or densify.
    sparse_row = _maybe_make_sparse_row(vector)
    return row if sparse_row is None else sparse_row


def _transport_data(matrix):
    return convert_sparse_coo_to_arrow(matrix.tocoo()) if sparse.issparse(matrix) else matrix


def _prepare_statevector(state_vector) -> tuple[bytes, int]:
    """Validate/select/serialize once, even when option sets share a state."""
    matrix = _state_row(state_vector)
    return _serialize_statevector(_transport_data(matrix)), matrix.shape[1].bit_length() - 1


def _run_prepared_statevector(
    payload: bytes,
    num_qubits: int,
    opt_params: OptParams,
    client: httpx.Client,
    return_summary: bool,
    inline_payload: str | None = None,
) -> str | Tuple[str, dict]:
    if num_qubits > USE_INLINE_STATE_NUM_QUBITS or opt_params.use_research_function is not None:
        statevector_data = _upload_statevector_payload(client, payload, opt_params)
    else:
        statevector_data = inline_payload if inline_payload is not None else base64.b64encode(payload).decode("ascii")

    job_timeout = (
        opt_params.job_completion_timeout_sec
        if opt_params.job_completion_timeout_sec is not None
        else 24 * 60 * 60
    )

    job = configure_job(
        client=client,
        opt_params=opt_params,
        statevector_data=statevector_data,
        statevector_type=_payload_type(payload),
    )

    result_summary, qasm = run_job(job, opt_params, job_timeout)

    if return_summary:
        return qasm, result_summary

    return qasm


def q_alchemy_as_qasm_parallel(state_vector: List[complex] | np.ndarray | sparse.sparray,
                                opt_params: List[dict | OptParams], client: httpx.Client | None = None,
                                return_summary=False, *, max_workers: int = 4):
    """Run option sets concurrently, preserving order and propagating failures.

    At most ``max_workers`` jobs run concurrently. Serialization is shared, but
    each job retains its own upload/cleanup lifecycle. A supplied client remains
    caller-owned; otherwise each worker opens and closes its own client.
    """
    if nonnegative_integer(max_workers, "max_workers") == 0:
        raise ValueError("max_workers must be positive")
    options = [deepcopy(populate_opt_params(opt)) for opt in opt_params]
    if not options:
        return []
    payload, num_qubits = _prepare_statevector(state_vector)
    inline_payload = (
        base64.b64encode(payload).decode("ascii")
        if num_qubits <= USE_INLINE_STATE_NUM_QUBITS
        and any(opt.use_research_function is None for opt in options)
        else None
    )

    def run(opt):
        worker_client = client if client is not None else create_client(opt)
        try:
            return _run_prepared_statevector(
                payload, num_qubits, opt, worker_client, return_summary, inline_payload
            )
        finally:
            if client is None:
                worker_client.close()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(options))) as executor:
        futures = [executor.submit(run, opt) for opt in options]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # Running jobs finish their normal cleanup; queued work need not run.
            for future in futures:
                future.cancel()
            raise


def q_alchemy_as_qasm_parallel_states(
        state_vector: List[List[complex] | np.ndarray | sparse.sparray] | sparse.sparray,
        opt_params: dict | OptParams,
        client: httpx.Client | None = None,
        return_summary=False,
        **kwargs
) -> list[str] | tuple[list[str], list[dict]]:
    """Run QAlchemy on a set of states.

    Note that the circuit's global phase is included both in the return summary and in the QASM;
    in the latter, if the circuit is a QASM2, the gphase is included as a comment.

    Results are lists even for a one-state batch. With ``return_summary=True``,
    AUTO summaries expose ``fidelity_requirement_met`` independently of execution
    ``status``; fidelity loss is an initializer estimate, not a simulation.
    """

    opt_params: OptParams = populate_opt_params(opt_params, **kwargs)
    owns_client = client is None
    client = client if client is not None else create_client(opt_params)
    try:
        return _q_alchemy_as_qasm_parallel_states(
            state_vector, opt_params, client, return_summary
        )
    finally:
        # A caller-supplied client belongs to the caller and is left open.
        if owns_client:
            client.close()


def _q_alchemy_as_qasm_parallel_states(
        state_vector: List[List[complex] | np.ndarray | sparse.sparray] | sparse.sparray,
        opt_params: OptParams,
        client: httpx.Client,
        return_summary: bool,
) -> list[str] | tuple[list[str], list[dict]]:
    # A file has one representation. Split mixed batches into at most two
    # groups, then restore the caller's order. Sparse rows never become dense.
    groups = _batch_state_groups(state_vector)
    qasm_results = [None] * sum(len(indices) for indices, _ in groups)
    summary_results = [None] * len(qasm_results)
    for indices, matrix in groups:
        qasms, summaries = _run_state_batch(matrix, opt_params, client)
        for index, qasm, summary in zip(indices, qasms, summaries):
            qasm_results[index] = qasm
            summary_results[index] = summary
    return (qasm_results, summary_results) if return_summary else qasm_results


def _batch_state_groups(state_vector):
    """Select every row before any upload; preserve sparse batches wholesale."""
    if sparse.issparse(state_vector):
        if state_vector.ndim != 2 or state_vector.shape[0] == 0:
            raise ValueError("Expected a nonempty batch of state rows")
        if not is_power_of_two(state_vector):
            raise ValueError("The state vector is not a power of two")
        return [(list(range(state_vector.shape[0])), state_vector.tocoo())]

    rows = list(state_vector)
    if not rows:
        raise ValueError("Expected a nonempty batch of state rows")
    groups = {False: ([], []), True: ([], [])}
    width = None
    for index, state in enumerate(rows):
        row = _state_row(state)
        if width is not None and row.shape[1] != width:
            raise ValueError("All states in a batch must have the same width")
        width = row.shape[1]
        indices, selected = groups[sparse.issparse(row)]
        indices.append(index)
        selected.append(row)

    result = []
    for is_sparse, (indices, selected) in groups.items():
        if not indices:
            continue
        if is_sparse:
            matrix = sparse.vstack(selected, format="coo")
        elif isinstance(state_vector, np.ndarray) and state_vector.ndim == 2 and len(indices) == len(rows):
            matrix = state_vector  # Do not copy an already dense batch.
        else:
            matrix = np.concatenate(selected, axis=0)
        result.append((indices, matrix))
    return result


def _run_state_batch(matrix, opt_params, client):
    num_states = matrix.shape[0]
    statevector_data = upload_statevector(client, _transport_data(matrix), opt_params)

    job_timeout = (
        opt_params.job_completion_timeout_sec
        if opt_params.job_completion_timeout_sec is not None
        else 24 * 60 * 60
    )

    job = configure_job(
        client=client,
        opt_params=opt_params,
        statevector_data=statevector_data,
        num_states=num_states
    )

    result_summary_list, qasm_list = run_job(job, opt_params, job_timeout)

    # One uploaded state uses the single-state service, whose response is a
    # string and dict. Preserve this batch API's list contract for every size.
    if num_states == 1 and isinstance(qasm_list, str) and isinstance(result_summary_list, dict):
        qasm_list = [qasm_list]
        result_summary_list = [result_summary_list]

    if (not isinstance(qasm_list, list) or not isinstance(result_summary_list, list)
            or len(qasm_list) != num_states or len(result_summary_list) != num_states):
        raise ValueError("State-preparation service returned an unexpected batch size or format")
    return qasm_list, result_summary_list
