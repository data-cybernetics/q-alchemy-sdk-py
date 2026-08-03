"""Remote sparse state-vector simulation via the Q-Alchemy ProCon.

The Q-Alchemy simulator (``q-alchemy-simulator``) is deployed as a pinexq ProCon
that exposes nine ProcessingSteps — three capabilities, each in three circuit
input forms:

============================  ===================  ===================  ================
capability                    QASM file            inline QASM string   QPY file
============================  ===================  ===================  ================
``counts``                    counts_from_qasm_file     counts_from_qasm_string     counts_from_qpy_file
``sparse_statevector``        sparse_statevector_from_qasm_file  ...  sparse_statevector_from_qpy_file
``tomography``                tomography_from_qasm_file ...        tomography_from_qpy_file
============================  ===================  ===================  ================

This module wraps those functions behind a small, pythonic client so SDK users
can run a Qiskit circuit (or a raw OpenQASM 2 string) on the hosted simulator
without touching the job-management API directly::

    from qiskit import QuantumCircuit
    from q_alchemy.simulator import SparseSimulator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    sim = SparseSimulator()                     # reads Q_ALCHEMY_API_KEY from env
    print(sim.counts(qc, shots=2048).counts)    # {'00': ..., '11': ...}
    print(sim.sparse_statevector(qc).amplitudes_dict())
    print(sim.tomography(qc).state_fidelity)    # ~1.0 vs the dense reference

The circuit input form is chosen automatically (small circuits go inline as
OpenQASM 2, larger ones are uploaded as QPY WorkData); override it with
``input_form=`` when needed.

:class:`SparseStatevectorResult` converts into the SDK's canonical sparse state
format (a scipy COO matrix / pyarrow parquet, see :mod:`q_alchemy.pyarrow_data`),
so a simulated state round-trips straight back into the loader::

    sv = sim.sparse_statevector(qc)
    qasm = q_alchemy_as_qasm(sv.to_coo())
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Literal, Sequence

import numpy as np
import httpx

from pinexq.client.core import MediaTypes
from pinexq.client.core.hco.upload_action_hco import UploadParameters
from pinexq.client.job_management import enter_jma, Job
from pinexq.client.job_management.hcos import WorkDataLink
from pinexq.client.job_management.model import InputDataSlotParameter, JobStates

# Reuse the SDK's existing job-management plumbing so simulator jobs behave
# exactly like the rest of the SDK (auth, retries, step lookup + caching).
from q_alchemy.initialize import create_client, find_processing_step

Capability = Literal["counts", "sparse_statevector", "tomography"]
InputForm = Literal["auto", "qasm_string", "qasm_file", "qpy"]
Tier = Literal["auto", "standard", "enterprise"]
Circuit = Any  # qiskit.QuantumCircuit (imported lazily) or an OpenQASM 2 string

# Grant (from the caller's UserGrants) that unlocks the XLarge enterprise tier.
ENTERPRISE_GRANT = "plan:enterprise"

# Return-WorkData alias produced by each capability (matches the ProCon).
_OUTPUT_ALIAS: dict[str, str] = {
    "counts": "counts.json",
    "sparse_statevector": "sparse_statevector.json",
    "tomography": "tomography.json",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimulatorParams:
    """Connection and job options for :class:`SparseSimulator`.

    The attribute names mirror :class:`q_alchemy.initialize.OptParams` where they
    overlap, so the same client/credentials work across the SDK.
    """

    api_key: str = field(
        default_factory=lambda: os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY")
    )
    host: str = field(default_factory=lambda: os.getenv("Q_ALCHEMY_HOST", "jobs.api.q-alchemy.com"))
    schema: str = field(default="https")
    added_headers: dict[str, str] = field(default_factory=dict)
    job_completion_timeout_sec: int | None = field(default=300)
    job_tags: list[str] = field(default_factory=list)
    remove_data: bool = field(default=True)
    #: Resource tier. "auto" picks the enterprise (XLarge) functions when the
    #: caller's plan allows it, else the standard (Medium) ones. "standard" is
    #: always available — an enterprise caller may opt down to Medium — while
    #: "enterprise" is rejected up front without the plan.
    tier: Tier = field(default="auto")
    #: Circuits with at most this many qubits default to the inline QASM form.
    inline_max_qubits: int = field(default=24)
    #: Raw QASM strings at most this long default to the inline form.
    inline_max_chars: int = field(default=200_000)

    @classmethod
    def from_dict(cls, env: dict) -> "SimulatorParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in env.items() if k in names})


def _populate_params(params: "SimulatorParams | dict | None", **kwargs) -> SimulatorParams:
    if params is None:
        params = SimulatorParams()
    elif isinstance(params, dict):
        params = SimulatorParams.from_dict(params)
    for attr, value in kwargs.items():
        if hasattr(params, attr):
            setattr(params, attr, value)
    return params


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _to_complex(pair: Sequence[float]) -> complex:
    return complex(pair[0], pair[1])


@dataclass
class CountsResult:
    """Measurement counts sampled from the ideal distribution."""

    counts: dict[str, int]
    num_qubits: int
    shots: int
    raw: dict

    @classmethod
    def from_raw(cls, raw: dict) -> "CountsResult":
        return cls(counts=raw["counts"], num_qubits=raw["num_qubits"], shots=raw["shots"], raw=raw)


@dataclass
class SparseStatevectorResult:
    """Exact sparse state-vector: only non-zero/significant amplitudes."""

    num_qubits: int
    nnz: int
    index_format: str
    index_convention: str
    indices: list[str]
    amplitudes: list[complex]
    raw: dict

    @classmethod
    def from_raw(cls, raw: dict) -> "SparseStatevectorResult":
        return cls(
            num_qubits=raw["num_qubits"],
            nnz=raw["nnz"],
            index_format=raw["index_format"],
            index_convention=raw["index_convention"],
            indices=list(raw["indices"]),
            amplitudes=[_to_complex(a) for a in raw["amplitudes"]],
            raw=raw,
        )

    def amplitudes_dict(self) -> dict[str, complex]:
        """Map each basis (string) index to its complex amplitude."""
        return dict(zip(self.indices, self.amplitudes))

    # -- interop with the SDK's sparse format -------------------------------- #
    #
    # The rest of the SDK speaks one sparse state representation: a scipy COO
    # matrix (1 x 2**n, complex) that round-trips through pyarrow parquet via
    # ``q_alchemy.pyarrow_data`` and is consumed by ``q_alchemy_as_qasm``. The
    # methods below convert this result into that representation so a simulated
    # state can be fed straight back into the loader:
    #
    #     sv = sim.sparse_statevector(qc)
    #     qasm = q_alchemy_as_qasm(sv.to_coo())
    #
    _BIG_ENDIAN = {"big_endian", "regular", "q0_msb", "msb"}

    def qiskit_indices(self) -> list[int]:
        """Basis indices as integers in Qiskit's little-endian column order.

        Parses the hex/bitstring indices and, for the big-endian convention
        (q[0] as most-significant bit), reverses the bits so the result indexes
        the standard ``2**n`` statevector the same way Qiskit does.
        """
        base = 16 if self.index_format == "hex" else 2
        big_endian = self.index_convention.lower() in self._BIG_ENDIAN
        out = []
        for token in self.indices:
            value = int(token, base)
            if big_endian:
                value = int(format(value, f"0{self.num_qubits}b")[::-1], 2)
            out.append(value)
        return out

    def to_coo(self):
        """Return a scipy ``coo_matrix`` of shape ``(1, 2**num_qubits)`` (complex).

        This is the SDK's canonical sparse state format (see
        :mod:`q_alchemy.pyarrow_data`); it can be passed directly to
        :func:`q_alchemy.q_alchemy_as_qasm`.
        """
        from scipy.sparse import coo_matrix

        cols = np.asarray(self.qiskit_indices(), dtype=np.int64)
        rows = np.zeros(len(cols), dtype=np.int64)
        data = np.asarray(self.amplitudes, dtype=complex)
        return coo_matrix((data, (rows, cols)), shape=(1, 2 ** self.num_qubits))

    def to_arrow(self):
        """Return the pyarrow parquet table for this state (the upload format)."""
        from q_alchemy.pyarrow_data import convert_sparse_coo_to_arrow

        return convert_sparse_coo_to_arrow(self.to_coo())

    def to_dense(self) -> np.ndarray:
        """Materialize the full dense statevector (``2**num_qubits`` complex)."""
        vector = np.zeros(2 ** self.num_qubits, dtype=complex)
        vector[self.qiskit_indices()] = self.amplitudes
        return vector


@dataclass
class TomographyResult:
    """Exact simulator-state analysis (not measurement-based tomography).

    ``state`` is the reconstructed density matrix (full, or the reduced
    subsystem selected by ``measurement_indices``) as a complex ``ndarray``, when
    one was materialized. ``state_fidelity`` is the fidelity against Qiskit's
    dense ``Statevector`` reference, reported for the small full-state case.
    """

    num_qubits: int
    sparse_statevector: SparseStatevectorResult
    measurement_indices: list[int] | None
    state: np.ndarray | None
    state_is_reduced: bool | None
    purity: float | None
    state_fidelity: float | None
    raw: dict

    @classmethod
    def from_raw(cls, raw: dict) -> "TomographyResult":
        state = None
        if raw.get("state") is not None:
            state = np.array(
                [[_to_complex(entry) for entry in row] for row in raw["state"]],
                dtype=complex,
            )
        return cls(
            num_qubits=raw["num_qubits"],
            sparse_statevector=SparseStatevectorResult.from_raw(raw["sparse_statevector"]),
            measurement_indices=raw.get("measurement_indices"),
            state=state,
            state_is_reduced=raw.get("state_is_reduced"),
            purity=raw.get("purity"),
            state_fidelity=raw.get("state_fidelity"),
            raw=raw,
        )


# --------------------------------------------------------------------------- #
# Circuit serialization helpers
# --------------------------------------------------------------------------- #
def _canonicalize_registers(circuit: Circuit) -> Circuit:
    """Rebuild the circuit with single ``q``/``c`` registers.

    The simulator's QASM runtime only recognizes a quantum register named ``q``
    (and classical ``c``). Tools that name registers differently — e.g. the
    PennyLane-Qiskit plugin emits ``q0``/``c0`` — otherwise fail server-side.
    Operations and measurement targets are preserved by global index.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if len(circuit.qregs) == 1 and circuit.qregs[0].name == "q" and (
        not circuit.cregs or (len(circuit.cregs) == 1 and circuit.cregs[0].name == "c")
    ):
        return circuit  # already canonical

    n, m = circuit.num_qubits, circuit.num_clbits
    registers = []
    if n:
        registers.append(QuantumRegister(n, "q"))
    if m:
        registers.append(ClassicalRegister(m, "c"))
    normalized = QuantumCircuit(*registers, name=circuit.name)
    normalized.compose(circuit, qubits=range(n), clbits=range(m), inplace=True)
    return normalized


def _to_qasm2(circuit: Circuit) -> str:
    from qiskit import qasm2

    return qasm2.dumps(circuit)


def _to_qpy_bytes(circuit: Circuit) -> bytes:
    from qiskit import qpy

    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class SparseSimulator:
    """Pythonic client for the hosted Q-Alchemy sparse state-vector simulator.

    A single instance holds one HTTP client and can run any number of circuits.
    """

    def __init__(
        self,
        params: SimulatorParams | dict | None = None,
        client: httpx.Client | None = None,
        **kwargs,
    ):
        self.params = _populate_params(params, **kwargs)
        if client is None and not self.params.api_key:
            raise ValueError(
                "No API key. Set Q_ALCHEMY_API_KEY (or pass api_key=...) before "
                "running the hosted simulator."
            )
        # create_client only reads api_key/added_headers/schema/host/timeout, all
        # of which SimulatorParams provides.
        self.client = client if client is not None else create_client(self.params)
        self._grants: list[str] | None = None  # cached UserGrants
        self._tier: str | None = None          # cached resolved tier

    # -- plan / tier --------------------------------------------------------- #
    def user_grants(self) -> list[str]:
        """The caller's grants, read once from the hypermedia API and cached.

        Navigates the entry point to ``Info`` and returns the embedded
        ``CurrentUser.UserGrants`` (e.g. ``["plan:enterprise", ...]``). Returns
        ``[]`` if the API doesn't expose it, so detection degrades to standard.
        """
        if self._grants is None:
            self._grants = self._fetch_user_grants()
        return self._grants

    def is_enterprise(self) -> bool:
        """Whether the caller's plan unlocks the enterprise (XLarge) tier."""
        return ENTERPRISE_GRANT in self.user_grants()

    @property
    def tier(self) -> str:
        """Resolved tier, decided here — before any step lookup or job.

        ``"auto"`` follows the caller's plan. Either tier may also be forced,
        and the two directions are not symmetric:

        * ``"standard"`` is always allowed. An enterprise caller opting down to
          the Medium functions is a normal thing to want — smaller circuits do
          not need XLarge, and the standard tier is the cheaper path.
        * ``"enterprise"`` requires the plan. Without it the ProCon's grant
          check would refuse the job anyway, so refusing here turns a wasted
          round trip and an opaque server-side failure into an immediate,
          explanatory error.
        """
        if self._tier is None:
            requested = (self.params.tier or "auto").lower()
            if requested not in ("auto", "standard", "enterprise"):
                raise ValueError(
                    f"Unknown tier {self.params.tier!r}. Use 'auto' (follow your "
                    f"plan), 'standard' (Medium), or 'enterprise' (XLarge). "
                    f"An unrecognized value used to fall through to the standard "
                    f"tier silently."
                )
            if requested == "auto":
                requested = "enterprise" if self.is_enterprise() else "standard"
            elif requested == "enterprise" and not self.is_enterprise():
                raise PermissionError(
                    "tier='enterprise' needs the Q-Alchemy enterprise plan; this "
                    f"key's grants are {self.user_grants()}. Use tier='standard' "
                    "(or the default tier='auto'), or contact support@q-alchemy.com "
                    "to upgrade."
                )
            self._tier = requested
        return self._tier

    def _fetch_user_grants(self) -> list[str]:
        try:
            entry = self.client.get("/api/EntryPoint").json()
            info_href = next(
                (l["href"] for l in entry.get("links", []) if "Info" in (l.get("rel") or [])),
                None,
            )
            if not info_href:
                return []
            info = self.client.get(info_href).json()
            for entity in info.get("entities", []):
                if "CurrentUser" in (entity.get("rel") or []):
                    return list((entity.get("properties") or {}).get("UserGrants", []))
        except Exception:
            return []
        return []

    # -- public capabilities ------------------------------------------------- #
    def counts(
        self,
        circuit: Circuit,
        *,
        shots: int = 1024,
        seed_simulator: int | None = None,
        optimization_level: int = 1,
        input_form: InputForm = "auto",
    ) -> CountsResult:
        """Sample terminal measurement counts from ``circuit``."""
        raw = self._run(
            "counts",
            circuit,
            input_form,
            {"shots": shots, "seed_simulator": seed_simulator, "optimization_level": optimization_level},
        )
        return CountsResult.from_raw(raw)

    def sparse_statevector(
        self,
        circuit: Circuit,
        *,
        sparse_index_format: Literal["hex", "bitstring"] = "hex",
        sparse_index_convention: str = "little_endian",
        max_nnz: int = 0,
        input_form: InputForm = "auto",
    ) -> SparseStatevectorResult:
        """Export the exact sparse state-vector produced by ``circuit``."""
        raw = self._run(
            "sparse_statevector",
            circuit,
            input_form,
            {
                "sparse_index_format": sparse_index_format,
                "sparse_index_convention": sparse_index_convention,
                "max_nnz": max_nnz,
            },
        )
        return SparseStatevectorResult.from_raw(raw)

    def tomography(
        self,
        circuit: Circuit,
        *,
        measurement_indices: Sequence[int] | None = None,
        materialize_state: bool = True,
        max_dense_qubits: int = 12,
        input_form: InputForm = "auto",
    ) -> TomographyResult:
        """Reconstruct and score the simulator state of ``circuit``."""
        raw = self._run(
            "tomography",
            circuit,
            input_form,
            {
                "measurement_indices": list(measurement_indices) if measurement_indices is not None else None,
                "materialize_state": materialize_state,
                "max_dense_qubits": max_dense_qubits,
            },
        )
        return TomographyResult.from_raw(raw)

    # -- internals ----------------------------------------------------------- #
    def _run(self, capability: Capability, circuit: Circuit, input_form: InputForm, parameters: dict) -> dict:
        suffix, extra_parameters, upload = self._prepare_input(circuit, input_form)
        function_name = f"{capability}_{suffix}"
        if self.tier == "enterprise":
            # XLarge functions; the server's grant check enforces eligibility.
            function_name += "_enterprise"
        step = find_processing_step(self.client, function_name)

        input_data_slots = None
        if upload is not None:
            filename, payload, mediatype = upload
            work_data = self._upload(filename, payload, mediatype)
            input_data_slots = [
                InputDataSlotParameter(Index=0, WorkDataUrls=[str(work_data.get_url())])
            ]

        job = Job(self.client).create_and_configure_rapidly(
            name=f"Simulator {function_name} ({datetime.now():%Y-%m-%d %H:%M:%S})",
            tags=["SDK", "Simulator", capability] + list(self.params.job_tags),
            processing_step_url=step.self_link(),
            start=True,
            parameters=json.dumps({**parameters, **extra_parameters}),
            allow_output_data_deletion=True,
            input_data_slots=input_data_slots,
        )

        timeout = (
            self.params.job_completion_timeout_sec
            if self.params.job_completion_timeout_sec is not None
            else 24 * 60 * 60
        )
        try:
            job.wait_for_state(JobStates.completed, polling_interval_s=0.25, timeout_s=timeout)
            return self._download_return(job, _OUTPUT_ALIAS[capability])
        finally:
            if self.params.remove_data:
                # Leave input workdata alone: an uploaded circuit is a pinexq
                # "Client Upload", protected by data-lineage and not deletable
                # here (attempting it only warns). Only the job + its outputs go.
                job.delete_with_associated(
                    delete_subjobs_with_data=True,
                    delete_input_workdata=False,
                    delete_output_workdata=True,
                )

    def _prepare_input(
        self, circuit: Circuit, input_form: InputForm
    ) -> tuple[str, dict, tuple[str, bytes, str] | None]:
        """Resolve (function suffix, extra JSON parameters, optional upload)."""
        is_str = isinstance(circuit, str)
        if not is_str:
            # Normalize register names so any Qiskit circuit runs on the
            # simulator's QASM runtime (which requires a `q`/`c` register).
            circuit = _canonicalize_registers(circuit)
        auto = input_form == "auto"
        if auto:
            if is_str:
                input_form = "qasm_string" if len(circuit) <= self.params.inline_max_chars else "qasm_file"
            else:
                input_form = "qasm_string" if circuit.num_qubits <= self.params.inline_max_qubits else "qpy"

        if input_form == "qasm_string":
            if is_str:
                return "from_qasm_string", {"qasm": circuit}, None
            try:
                return "from_qasm_string", {"qasm": _to_qasm2(circuit)}, None
            except Exception:
                # Custom gates etc. may not serialize to OpenQASM 2; QPY round-trips
                # any Qiskit circuit, so fall back to it when the form was inferred.
                if auto:
                    return self._qpy_input(circuit)
                raise

        if input_form == "qasm_file":
            qasm = circuit if is_str else _to_qasm2(circuit)
            return "from_qasm_file", {}, ("circuit.qasm", qasm.encode("utf-8"), MediaTypes.TEXT)

        if input_form == "qpy":
            if is_str:
                raise ValueError("input_form='qpy' requires a QuantumCircuit, not a QASM string.")
            return self._qpy_input(circuit)

        raise ValueError(f"Unknown input_form {input_form!r}.")

    @staticmethod
    def _qpy_input(circuit: Circuit) -> tuple[str, dict, tuple[str, bytes, str]]:
        return "from_qpy_file", {}, ("circuit.qpy", _to_qpy_bytes(circuit), MediaTypes.OCTET_STREAM)

    def _upload(self, filename: str, payload: bytes, mediatype: str) -> WorkDataLink:
        work_data_root = enter_jma(self.client).work_data_root_link.navigate()
        return work_data_root.upload_action.execute(
            UploadParameters(filename=filename, binary=payload, mediatype=mediatype, json=None)
        )

    @staticmethod
    def _download_return(job: Job, output_name: str) -> dict:
        matches = [
            wd
            for slot in job.get_output_data_slots()
            for wd in slot.assigned_workdatas
            if wd.name == output_name
        ]
        if not matches:
            raise IOError(f"Simulator job produced no '{output_name}' output.")
        work_data = matches[0]
        if work_data.size_in_bytes == 0:
            raise IOError(f"Simulator job returned an empty '{output_name}'.")
        return json.loads(work_data.download_link.download().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Module-level convenience functions
# --------------------------------------------------------------------------- #
def simulate_counts(circuit: Circuit, params: SimulatorParams | dict | None = None, **kwargs) -> CountsResult:
    """One-shot :meth:`SparseSimulator.counts` (see it for keyword options)."""
    run_kwargs = _split_run_kwargs(kwargs, SparseSimulator.counts)
    return SparseSimulator(params, **kwargs).counts(circuit, **run_kwargs)


def simulate_sparse_statevector(
    circuit: Circuit, params: SimulatorParams | dict | None = None, **kwargs
) -> SparseStatevectorResult:
    """One-shot :meth:`SparseSimulator.sparse_statevector`."""
    run_kwargs = _split_run_kwargs(kwargs, SparseSimulator.sparse_statevector)
    return SparseSimulator(params, **kwargs).sparse_statevector(circuit, **run_kwargs)


def simulate_tomography(
    circuit: Circuit, params: SimulatorParams | dict | None = None, **kwargs
) -> TomographyResult:
    """One-shot :meth:`SparseSimulator.tomography`."""
    run_kwargs = _split_run_kwargs(kwargs, SparseSimulator.tomography)
    return SparseSimulator(params, **kwargs).tomography(circuit, **run_kwargs)


def _split_run_kwargs(kwargs: dict, method) -> dict:
    """Pull per-run keyword args out of ``kwargs``, leaving SimulatorParams overrides.

    Lets the convenience functions accept both connection options (api_key,
    job_tags, ...) and run options (shots, input_form, ...) in one call.
    """
    import inspect

    run_names = set(inspect.signature(method).parameters) - {"self", "circuit"}
    run_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in run_names}
    return run_kwargs
