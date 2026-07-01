"""Qiskit ``BackendV2`` for the hosted Q-Alchemy sparse simulator.

This exposes the remote simulator through Qiskit's standard backend interface —
the same way IBM backends are used — so it drops into the Qiskit (and, via the
Qiskit plugin, PennyLane) ecosystem:

    from qiskit import QuantumCircuit, transpile
    from q_alchemy import QAlchemyBackend

    backend = QAlchemyBackend()               # reads Q_ALCHEMY_API_KEY from env
    qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    job = backend.run(transpile(qc, backend), shots=2048)
    print(job.result().get_counts())

Execution is delegated to :class:`q_alchemy.simulator.SparseSimulator`, which
submits the circuit to the deployed ProCon and (transparently) picks the user's
resource tier. The backend just adapts circuits in and a Qiskit ``Result`` out,
mirroring the local ``SparseAerBackend`` so the two are interchangeable.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Sequence

from qiskit.circuit import Parameter, QuantumCircuit
try:
    from qiskit.circuit import Reset
except ImportError:  # pragma: no cover
    from qiskit.circuit.library import Reset
from qiskit.circuit.library import (
    CCXGate,
    CSwapGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    Measure,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SwapGate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.result import Result
from qiskit.transpiler import Target

from q_alchemy.simulator import SparseSimulator, SimulatorParams

# The gate set the simulator natively supports (matches SparseAerBackend).
QALCHEMY_BASIS_GATES = [
    "u", "p", "rx", "ry", "rz", "x", "y", "z", "h",
    "cx", "cy", "cz", "swap", "ccx", "cswap", "measure", "reset",
]


def _counts_to_hex(counts: dict[str, int]) -> dict[str, int]:
    """Normalize count keys to hex so ``Result.get_counts()`` reformats them.

    The hosted simulator returns already-formatted keys (bit strings such as
    ``"01"``, or hex such as ``"0x1"``). Qiskit's ``Result`` stores counts in hex
    and re-formats to bit strings using the header's ``memory_slots``, so we map
    every key to hex here.
    """
    out: dict[str, int] = {}
    for key, value in counts.items():
        token = key.replace(" ", "")
        integer = int(token, 16) if token.lower().startswith("0x") else int(token, 2)
        hex_key = hex(integer)
        out[hex_key] = out.get(hex_key, 0) + int(value)
    return out


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return any(instruction.operation.name == "measure" for instruction in circuit.data)


class QAlchemyJob(JobV1):
    """Synchronous job that executes on first ``result()`` and caches it."""

    def __init__(self, backend: "QAlchemyBackend", job_id: str, build_result):
        super().__init__(backend, job_id)
        self._build_result = build_result
        self._result: Result | None = None

    def submit(self) -> None:  # pragma: no cover - Qiskit signature
        return None

    def result(self, timeout: float | None = None) -> Result:  # noqa: ARG002
        if self._result is None:
            self._result = self._build_result()
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE if self._result is not None else JobStatus.INITIALIZING

    def cancel(self) -> bool:
        return False


class QAlchemyBackend(BackendV2):
    """Aer-style Qiskit backend backed by the *hosted* Q-Alchemy simulator."""

    backend_version = "0.1.0"

    def __init__(
        self,
        num_qubits: int = 128,
        *,
        params: SimulatorParams | SparseSimulator | dict | None = None,
        name: str = "q_alchemy_simulator",
        provider=None,
        description: str | None = None,
        **sim_kwargs,
    ):
        # Only genuine BackendV2 fields go to the base class; everything else
        # (api_key, host, tier, job_completion_timeout_sec, ...) configures the
        # hosted-simulator client.
        super().__init__(
            provider=provider,
            name=name,
            description=description or "Q-Alchemy hosted sparse simulator",
            backend_version=self.backend_version,
        )
        self._num_qubits = int(num_qubits)
        self._target = self._build_target(self._num_qubits)
        # Reuse a passed-in client, or build one from params + kwargs.
        if isinstance(params, SparseSimulator):
            self._simulator = params
        else:
            self._simulator = SparseSimulator(params, **sim_kwargs)

    @property
    def simulator(self) -> SparseSimulator:
        """The underlying hosted-simulator client (exposes ``.tier`` etc.)."""
        return self._simulator

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            seed_simulator=None,
            memory=False,
            max_nnz=0,
            save_statevector=False,
            save_sparse_statevector=False,
            sparse_index_format="hex",
            sparse_index_convention="little_endian",
            max_dense_qubits=26,
            optimization_level=1,
        )

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int | None:
        return None

    @staticmethod
    def _build_target(num_qubits: int) -> Target:
        theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
        target = Target(num_qubits=num_qubits, description="Q-Alchemy hosted sparse simulator")
        for instruction in (
            UGate(theta, phi, lam), PhaseGate(theta),
            RXGate(theta), RYGate(theta), RZGate(theta),
            XGate(), YGate(), ZGate(), HGate(),
            CXGate(), CYGate(), CZGate(), SwapGate(), CCXGate(), CSwapGate(),
            Measure(), Reset(),
        ):
            target.add_instruction(instruction)
        return target

    def run(self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options) -> QAlchemyJob:
        run_options = self.options
        run_options.update_options(**options)

        circuits = list(run_input) if isinstance(run_input, (list, tuple)) else [run_input]
        if not circuits:
            raise ValueError("No circuits were supplied.")
        for circuit in circuits:
            if circuit.num_qubits > self._num_qubits:
                raise ValueError(
                    f"Circuit uses {circuit.num_qubits} qubits, backend limit is {self._num_qubits}."
                )

        job_id = str(uuid.uuid4())

        def build_result() -> Result:
            experiment_results = [self._run_one(c, run_options) for c in circuits]
            return Result.from_dict(
                {
                    "backend_name": self.name,
                    "backend_version": self.backend_version,
                    "qobj_id": job_id,
                    "job_id": job_id,
                    "success": True,
                    "status": "COMPLETED",
                    "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "results": experiment_results,
                }
            )

        return QAlchemyJob(self, job_id, build_result)

    def _run_one(self, circuit: QuantumCircuit, run_options) -> dict:
        shots = int(run_options.shots)
        data: dict[str, object] = {}

        if _has_measurements(circuit):
            counts = self._simulator.counts(
                circuit,
                shots=shots,
                seed_simulator=run_options.seed_simulator,
                optimization_level=int(run_options.optimization_level),
            ).counts
            data["counts"] = _counts_to_hex(counts)

        # Statevector exports share one remote call (state capabilities strip
        # any terminal measurements server-side).
        want_sparse = bool(getattr(run_options, "save_sparse_statevector", False))
        want_dense = bool(getattr(run_options, "save_statevector", False))
        if want_sparse or want_dense:
            sv = self._simulator.sparse_statevector(
                circuit,
                sparse_index_format=str(getattr(run_options, "sparse_index_format", "hex")),
                sparse_index_convention=str(getattr(run_options, "sparse_index_convention", "little_endian")),
                max_nnz=int(getattr(run_options, "max_nnz", 0)),
            )
            if want_sparse:
                data["sparse_statevector"] = sv.raw
            if want_dense:
                data["statevector"] = sv.to_dense()

        return {
            "shots": shots,
            "success": True,
            "status": "DONE",
            "seed_simulator": run_options.seed_simulator,
            "header": {
                "name": circuit.name,
                "metadata": circuit.metadata or {},
                "n_qubits": circuit.num_qubits,
                "memory_slots": circuit.num_clbits,
            },
            "data": data,
            "metadata": {"method": "sparse-remote", "tier": self._simulator.tier},
        }


class QAlchemyProvider:
    """Minimal provider so backends are discoverable IBM-style.

    >>> from q_alchemy import QAlchemyProvider
    >>> backend = QAlchemyProvider().get_backend()
    """

    def __init__(self, params: SimulatorParams | dict | None = None, **kwargs):
        self._params = params
        self._kwargs = kwargs

    def get_backend(self, name: str = "q_alchemy_simulator", **kwargs) -> QAlchemyBackend:
        return QAlchemyBackend(params=self._params, name=name, **{**self._kwargs, **kwargs})

    def backends(self, name: str | None = None) -> list[QAlchemyBackend]:
        backend = self.get_backend()
        if name is not None and name != backend.name:
            return []
        return [backend]
