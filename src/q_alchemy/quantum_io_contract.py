"""SDK-native mirror of the Q-Alchemy Quantum I/O schema-3 contract.

These classes intentionally mirror q-alchemy-quantum-io without importing that
private runtime package. JSON serialization remains an internal transport detail.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 3

_SENSITIVE_KEYS = {
    "access_key",
    "access_token",
    "api_key",
    "api_token",
    "auth_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
}


def _complex_to_data(value: complex) -> list[float]:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise ValueError("portable complex values must be finite")
    return [float(number.real), float(number.imag)]


def _complex_from_data(value: Any) -> complex:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("complex values must be encoded as [real, imag]")
    return complex(float(value[0]), float(value[1]))


def _json_value(value: Any, *, path: str = "value") -> Any:
    """Convert supported values to deterministic JSON-compatible data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite complex value")
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            output[key] = _json_value(item, path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_value(item, path=f"{path}[]") for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item(), path=path)
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_value(tolist(), path=path)
        except Exception:
            pass
    raise TypeError(f"{path} contains unsupported non-portable value {type(value).__name__}")


def _validated_metadata(metadata: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    return dict(_json_value(dict(metadata), path=path))


def _check_no_secrets(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower().replace("-", "_")
            sensitive = (
                key in _SENSITIVE_KEYS
                or key.endswith(("_password", "_secret", "_token", "_api_key", "_private_key"))
            )
            if sensitive:
                raise ValueError(
                    f"{path}.{raw_key} looks credential-sensitive; provider credentials "
                    "must be supplied through service runtime bindings/secrets, not the portable contract"
                )
            _check_no_secrets(item, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_no_secrets(item, path=f"{path}[{index}]")


def _require_schema(data: Mapping[str, Any], *, kind: str) -> None:
    if data.get("kind") != kind:
        raise ValueError(f"payload is not a {kind} contract")
    if "schema_version" not in data:
        raise ValueError(f"{kind} payload is missing schema_version")
    version = int(data["schema_version"])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {kind} schema_version={version}; supported={SCHEMA_VERSION}"
        )


class PortablePauliSum:
    """Minimal backend-neutral Pauli-sum object implementing ``to_list()``."""

    def __init__(self, terms: Sequence[tuple[str, complex]]) -> None:
        self._terms = tuple((str(label), complex(coeff)) for label, coeff in terms)

    def to_list(self) -> list[tuple[str, complex]]:
        return list(self._terms)


@dataclass(frozen=True)
class State:
    """Portable target state.

    Use :meth:`dense` or :meth:`sparse`; users never need to supply the wire
    discriminator manually.
    """

    num_qubits: int
    amplitudes: tuple[complex, ...]
    indices: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        if self.indices is None:
            if len(self.amplitudes) != 1 << self.num_qubits:
                raise ValueError("dense amplitude count does not match num_qubits")
        else:
            if not self.amplitudes:
                raise ValueError("sparse state must contain at least one amplitude")
            if len(self.indices) != len(self.amplitudes):
                raise ValueError("sparse indices/amplitudes length mismatch")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("sparse state indices must be unique")
            limit = 1 << self.num_qubits
            if any(index < 0 or index >= limit for index in self.indices):
                raise ValueError("sparse state index is outside the state dimension")
        _validated_metadata(self.metadata, path="state.metadata")

    @property
    def representation(self) -> str:
        return "sparse" if self.indices is not None else "dense"

    @classmethod
    def dense(
        cls,
        amplitudes: Sequence[complex],
        *,
        num_qubits: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "State":
        values = tuple(complex(value) for value in amplitudes)
        if num_qubits is None:
            size = len(values)
            if size == 0 or size & (size - 1):
                raise ValueError("dense state length must be a non-zero power of two")
            num_qubits = size.bit_length() - 1
        return cls(int(num_qubits), values, None, dict(metadata or {}))

    @classmethod
    def sparse(
        cls,
        *,
        num_qubits: int,
        indices: Sequence[int],
        amplitudes: Sequence[complex],
        metadata: Mapping[str, Any] | None = None,
    ) -> "State":
        return cls(
            int(num_qubits),
            tuple(complex(value) for value in amplitudes),
            tuple(int(index) for index in indices),
            dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "representation": self.representation,
            "num_qubits": self.num_qubits,
            "amplitudes": [_complex_to_data(value) for value in self.amplitudes],
            "metadata": _validated_metadata(self.metadata, path="state.metadata"),
        }
        if self.indices is not None:
            data["indices"] = [str(index) for index in self.indices]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "State":
        representation = str(data["representation"])
        amplitudes = tuple(_complex_from_data(value) for value in data["amplitudes"])
        metadata = dict(data.get("metadata", {}))
        if representation == "dense":
            if "indices" in data and data.get("indices") not in (None, [], ()):
                raise ValueError("dense state must not contain sparse indices")
            return cls.dense(
                amplitudes,
                num_qubits=int(data["num_qubits"]),
                metadata=metadata,
            )
        if representation == "sparse":
            return cls.sparse(
                num_qubits=int(data["num_qubits"]),
                indices=tuple(int(value) for value in data.get("indices", ())),
                amplitudes=amplitudes,
                metadata=metadata,
            )
        raise ValueError("state representation must be 'dense' or 'sparse'")




@dataclass(frozen=True)
class Circuit:
    """Portable OpenQASM 3 circuit.

    Use :meth:`qasm3` or :meth:`from_qiskit`; ``format`` remains only a wire
    discriminator for future circuit encodings.
    """

    payload: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.payload.strip():
            raise ValueError("circuit payload must not be empty")
        _validated_metadata(self.metadata, path="circuit.metadata")

    @property
    def format(self) -> str:
        return "qasm3"

    @classmethod
    def qasm3(
        cls,
        payload: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Circuit":
        return cls(str(payload), dict(metadata or {}))

    @classmethod
    def from_qiskit(
        cls,
        circuit: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Circuit":
        try:
            from qiskit.qasm3 import dumps
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Circuit.from_qiskit requires qiskit>=2.3") from exc
        return cls.qasm3(dumps(circuit), metadata=metadata)

    def to_qiskit(self) -> Any:
        try:
            from qiskit.qasm3 import loads
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenQASM 3 import requires qiskit and qiskit-qasm3-import; "
                "install the qasm3 optional dependency"
            ) from exc
        try:
            return loads(self.payload)
        except ImportError as exc:  # pragma: no cover - optional importer dependency
            raise RuntimeError(
                "OpenQASM 3 import requires qiskit-qasm3-import; "
                "install the qasm3 optional dependency"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "payload": self.payload,
            "metadata": _validated_metadata(self.metadata, path="circuit.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Circuit":
        if data.get("format") != "qasm3":
            raise ValueError("Circuit currently supports only format='qasm3'")
        return cls.qasm3(
            str(data["payload"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class PauliObservable:
    label: str
    terms: tuple[tuple[str, complex], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("observable label must not be empty")
        if not self.terms:
            raise ValueError("portable observable must contain at least one Pauli term")
        widths = {len(pauli) for pauli, _ in self.terms}
        if 0 in widths or len(widths) != 1 or any(
            any(char not in "IXYZ" for char in pauli) for pauli, _ in self.terms
        ):
            raise ValueError("portable observables require equal-width I/X/Y/Z Pauli strings")
        _validated_metadata(self.metadata, path=f"observable[{self.label}].metadata")

    @classmethod
    def pauli(
        cls,
        label: str,
        pauli: str,
        *,
        coefficient: complex = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PauliObservable":
        return cls(
            str(label),
            ((str(pauli), complex(coefficient)),),
            dict(metadata or {}),
        )



    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "terms": [
                {"pauli": pauli, "coefficient": _complex_to_data(coefficient)}
                for pauli, coefficient in self.terms
            ],
            "metadata": _validated_metadata(
                self.metadata, path=f"observable[{self.label}].metadata"
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PauliObservable":
        return cls(
            str(data["label"]),
            tuple(
                (str(term["pauli"]), _complex_from_data(term["coefficient"]))
                for term in data["terms"]
            ),
            dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class BasisMeasurement:
    label: str
    qubits: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("basis-measurement label must not be empty")
        if not self.qubits:
            raise ValueError("basis measurement must contain at least one qubit")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("basis-measurement qubits must be unique")
        if any(qubit < 0 for qubit in self.qubits):
            raise ValueError("basis-measurement qubits must be non-negative")
        _validated_metadata(self.metadata, path=f"basis_measurement[{self.label}].metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "qubits": list(self.qubits),
            "metadata": _validated_metadata(
                self.metadata, path=f"basis_measurement[{self.label}].metadata"
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BasisMeasurement":
        return cls(
            str(data["label"]),
            tuple(int(qubit) for qubit in data["qubits"]),
            dict(data.get("metadata", {})),
        )




@dataclass(frozen=True)
class MeasurementPlan:
    training: tuple[PauliObservable, ...] = ()
    validation: tuple[PauliObservable, ...] = ()
    basis_measurements: tuple[BasisMeasurement, ...] = ()
    observable_plan_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.training) != bool(self.validation):
            raise ValueError(
                "portable observable plans require both fitting and held-out observable sets"
            )
        labels = [item.label for item in self.training + self.validation]
        labels.extend(item.label for item in self.basis_measurements)
        if len(set(labels)) != len(labels):
            raise ValueError("measurement labels must be unique across the portable plan")
        _validated_metadata(
            self.observable_plan_metadata, path="measurement_plan.observable_plan_metadata"
        )
        _validated_metadata(self.metadata, path="measurement_plan.metadata")



    def to_dict(self) -> dict[str, Any]:
        return {
            "training": [item.to_dict() for item in self.training],
            "validation": [item.to_dict() for item in self.validation],
            "observable_plan_metadata": _validated_metadata(
                self.observable_plan_metadata,
                path="measurement_plan.observable_plan_metadata",
            ),
            "basis_measurements": [item.to_dict() for item in self.basis_measurements],
            "metadata": _validated_metadata(self.metadata, path="measurement_plan.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementPlan":
        return cls(
            training=tuple(PauliObservable.from_dict(item) for item in data.get("training", ())),
            validation=tuple(
                PauliObservable.from_dict(item) for item in data.get("validation", ())
            ),
            basis_measurements=tuple(
                BasisMeasurement.from_dict(item) for item in data.get("basis_measurements", ())
            ),
            observable_plan_metadata=dict(data.get("observable_plan_metadata", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class QuantumExperiment:
    """Portable quantum experiment; schema details are serialization-only."""

    target: State
    evolution: Circuit | None = None
    evolution_qargs: tuple[int, ...] | None = None
    measurement_plan: MeasurementPlan = field(default_factory=MeasurementPlan)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evolution is None and self.evolution_qargs is not None:
            raise ValueError("evolution_qargs requires an evolution circuit")
        if self.evolution_qargs is not None:
            if not self.evolution_qargs:
                raise ValueError("evolution_qargs must not be empty")
            if len(set(self.evolution_qargs)) != len(self.evolution_qargs):
                raise ValueError("evolution_qargs must be unique")
            if any(
                qubit < 0 or qubit >= self.target.num_qubits
                for qubit in self.evolution_qargs
            ):
                raise ValueError("evolution_qargs contains an out-of-range qubit")
        for observable in self.measurement_plan.training + self.measurement_plan.validation:
            if any(len(pauli) != self.target.num_qubits for pauli, _ in observable.terms):
                raise ValueError(
                    f"observable {observable.label!r} width does not match target num_qubits"
                )
        for measurement in self.measurement_plan.basis_measurements:
            if any(qubit >= self.target.num_qubits for qubit in measurement.qubits):
                raise ValueError(
                    f"basis measurement {measurement.label!r} contains an out-of-range qubit"
                )
        _validated_metadata(self.metadata, path="experiment.metadata")



    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "quantum-experiment",
            "target": self.target.to_dict(),
            "evolution": self.evolution.to_dict() if self.evolution is not None else None,
            "evolution_qargs": (
                list(self.evolution_qargs) if self.evolution_qargs is not None else None
            ),
            "measurement_plan": self.measurement_plan.to_dict(),
            "metadata": _validated_metadata(self.metadata, path="experiment.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantumExperiment":
        _require_schema(data, kind="quantum-experiment")
        evolution_data = data.get("evolution")
        qargs = data.get("evolution_qargs")
        return cls(
            target=State.from_dict(data["target"]),
            evolution=(Circuit.from_dict(evolution_data) if evolution_data is not None else None),
            evolution_qargs=(tuple(int(value) for value in qargs) if qargs is not None else None),
            measurement_plan=MeasurementPlan.from_dict(data.get("measurement_plan", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "QuantumExperiment":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class Runtime:
    """Credential-free declaration of an execution component.

    Prefer the named constructors so callers do not need to know contract kind
    strings. A ``resource`` declaration is role-neutral: its containing
    :class:`ExecutionPlan` field determines whether the resolved object must act
    as a simulator, reference, acquisition source, or estimator.
    """

    kind: str
    resource_name: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("runtime kind must not be empty")
        if self.kind == "resource":
            if not self.resource_name:
                raise ValueError("resource runtime requires a non-empty resource name")
        elif self.resource_name is not None and not self.resource_name:
            raise ValueError("runtime resource name must not be empty")
        _check_no_secrets(self.config)
        _validated_metadata(self.config, path=f"runtime[{self.kind}].config")

    @classmethod
    def resource(
        cls,
        name: str,
        *,
        config: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> "Runtime":
        merged = dict(config or {})
        overlap = set(merged) & set(options)
        if overlap:
            raise ValueError(
                "duplicate resource config keys: " + ", ".join(sorted(overlap))
            )
        merged.update(options)
        return cls("resource", str(name), merged)

    @classmethod
    def qalchemy_sparse(cls, **config: Any) -> "Runtime":
        return cls("q-alchemy-sparse", config=config)

    @classmethod
    def qiskit_statevector(cls) -> "Runtime":
        return cls("qiskit-statevector")

    @classmethod
    def preloaded(
        cls,
        *,
        result: Mapping[str, Any] | None = None,
        resource: str | None = None,
    ) -> "Runtime":
        if (result is None) == (resource is None):
            raise ValueError("preloaded runtime requires exactly one of result or resource")
        return cls(
            "preloaded",
            resource_name=resource,
            config={"result": dict(result)} if result is not None else {},
        )

    @classmethod
    def qtucker(
        cls,
        *,
        model: Mapping[str, Any] | None = None,
        **config: Any,
    ) -> "Runtime":
        payload = dict(config)
        if model is not None:
            payload["model_spec"] = dict(model)
        return cls("qtucker", config=payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resource": self.resource_name,
            "config": _validated_metadata(self.config, path=f"runtime[{self.kind}].config"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Runtime":
        if "source_kind" in data:
            raise ValueError(
                "runtime source_kind is not part of an execution request; "
                "execution provenance belongs to the resolved result"
            )
        kind = str(data["kind"])
        if kind in {
            "resource-simulator",
            "resource-reference",
            "resource-acquisition",
            "resource-estimator",
        }:
            raise ValueError(
                f"runtime kind {kind!r} is obsolete; use kind='resource' and let the "
                "ExecutionPlan field determine the resource role"
            )
        return cls(
            kind=kind,
            resource_name=(
                str(data["resource"]) if data.get("resource") is not None else None
            ),
            config=dict(data.get("config", {})),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    """Portable execution choices independent of live objects and secrets."""

    preparation_simulator: Runtime | None = None
    reference: Runtime | None = None
    acquisition: Runtime | None = None
    estimator: Runtime | None = None
    preparation_options: Mapping[str, Any] = field(default_factory=dict)
    shots: int = 4096
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        _check_no_secrets(self.preparation_options, path="preparation_options")
        _validated_metadata(self.preparation_options, path="preparation_options")
        _validated_metadata(self.metadata, path="execution_plan.metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "execution-plan",
            "preparation_simulator": (
                self.preparation_simulator.to_dict()
                if self.preparation_simulator is not None
                else None
            ),
            "reference": self.reference.to_dict() if self.reference is not None else None,
            "acquisition": self.acquisition.to_dict() if self.acquisition is not None else None,
            "estimator": self.estimator.to_dict() if self.estimator is not None else None,
            "preparation_options": _validated_metadata(
                self.preparation_options, path="preparation_options"
            ),
            "shots": self.shots,
            "metadata": _validated_metadata(self.metadata, path="execution_plan.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        _require_schema(data, kind="execution-plan")
        return cls(
            preparation_simulator=(
                Runtime.from_dict(data["preparation_simulator"])
                if data.get("preparation_simulator") is not None
                else None
            ),
            reference=(
                Runtime.from_dict(data["reference"])
                if data.get("reference") is not None
                else None
            ),
            acquisition=(
                Runtime.from_dict(data["acquisition"])
                if data.get("acquisition") is not None
                else None
            ),
            estimator=(
                Runtime.from_dict(data["estimator"])
                if data.get("estimator") is not None
                else None
            ),
            preparation_options=dict(data.get("preparation_options", {})),
            shots=int(data.get("shots", 4096)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ExecutionPlan":
        return cls.from_dict(json.loads(payload))


# Typed service result hierarchy -------------------------------------------------


@dataclass(frozen=True)
class CircuitMetrics:
    depth: int | None = None
    size: int | None = None
    cx_count: int | None = None
    two_qubit_count: int | None = None
    operation_counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitMetrics":
        return cls(
            depth=(int(data["depth"]) if data.get("depth") is not None else None),
            size=(int(data["size"]) if data.get("size") is not None else None),
            cx_count=(int(data["cx_count"]) if data.get("cx_count") is not None else None),
            two_qubit_count=(
                int(data["two_qubit_count"])
                if data.get("two_qubit_count") is not None
                else None
            ),
            operation_counts={
                str(name): int(count)
                for name, count in dict(data.get("operation_counts", {})).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "size": self.size,
            "cx_count": self.cx_count,
            "two_qubit_count": self.two_qubit_count,
            "operation_counts": dict(self.operation_counts),
        }


@dataclass(frozen=True)
class PreparationSummary:
    num_qubits: int
    method: str
    claimed_fidelity_loss: float | None = None
    found: bool | None = None
    metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreparationSummary":
        return cls(
            num_qubits=int(data["num_qubits"]),
            method=str(data["method"]),
            claimed_fidelity_loss=(
                float(data["claimed_fidelity_loss"])
                if data.get("claimed_fidelity_loss") is not None
                else None
            ),
            found=(bool(data["found"]) if data.get("found") is not None else None),
            metrics=CircuitMetrics.from_dict(data.get("metrics", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "method": self.method,
            "claimed_fidelity_loss": self.claimed_fidelity_loss,
            "found": self.found,
            "metrics": self.metrics.to_dict(),
            "metadata": _validated_metadata(self.metadata, path="preparation.metadata"),
        }


@dataclass(frozen=True)
class SimulationSummary:
    simulator: str
    statevector_materialized: bool
    sparse_state_available: bool
    nnz: int | None = None
    exact: bool | None = None
    reference_quality: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationSummary":
        return cls(
            simulator=str(data["simulator"]),
            statevector_materialized=bool(data.get("statevector_materialized", False)),
            sparse_state_available=bool(data.get("sparse_state_available", False)),
            nnz=(int(data["nnz"]) if data.get("nnz") is not None else None),
            exact=(bool(data["exact"]) if data.get("exact") is not None else None),
            reference_quality=str(data.get("reference_quality", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator": self.simulator,
            "statevector_materialized": self.statevector_materialized,
            "sparse_state_available": self.sparse_state_available,
            "nnz": self.nnz,
            "exact": self.exact,
            "reference_quality": self.reference_quality,
            "metadata": _validated_metadata(self.metadata, path="simulation.metadata"),
        }


@dataclass(frozen=True)
class ObservablePlanSummary:
    training: tuple[str, ...]
    validation: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservablePlanSummary":
        return cls(
            tuple(str(value) for value in data.get("training", ())),
            tuple(str(value) for value in data.get("validation", ())),
            dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": list(self.training),
            "validation": list(self.validation),
            "metadata": _validated_metadata(self.metadata, path="observable_plan.metadata"),
        }


@dataclass(frozen=True)
class MeasurementPlanSummary:
    observable_plan: ObservablePlanSummary | None = None
    basis_measurements: tuple[BasisMeasurement, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementPlanSummary":
        observable = data.get("observable_plan")
        return cls(
            observable_plan=(
                ObservablePlanSummary.from_dict(observable)
                if isinstance(observable, Mapping)
                else None
            ),
            basis_measurements=tuple(
                BasisMeasurement.from_dict(item)
                for item in data.get("basis_measurements", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observable_plan": (
                self.observable_plan.to_dict() if self.observable_plan is not None else None
            ),
            "basis_measurements": [item.to_dict() for item in self.basis_measurements],
            "metadata": _validated_metadata(self.metadata, path="measurement_plan_summary.metadata"),
        }


@dataclass(frozen=True)
class ExperimentSummary:
    target_num_qubits: int | None
    evolution_present: bool
    measurement_plan: MeasurementPlanSummary
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSummary":
        return cls(
            target_num_qubits=(
                int(data["target_num_qubits"])
                if data.get("target_num_qubits") is not None
                else None
            ),
            evolution_present=bool(data.get("evolution_present", False)),
            measurement_plan=MeasurementPlanSummary.from_dict(
                data.get("measurement_plan", {})
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_num_qubits": self.target_num_qubits,
            "evolution_present": self.evolution_present,
            "measurement_plan": self.measurement_plan.to_dict(),
            "metadata": _validated_metadata(self.metadata, path="experiment_summary.metadata"),
        }


@dataclass(frozen=True)
class ExperimentCircuitSummary:
    num_qubits: int
    evolution_present: bool
    evolution_qargs: tuple[int, ...] | None
    metrics: CircuitMetrics
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentCircuitSummary":
        qargs = data.get("evolution_qargs")
        return cls(
            num_qubits=int(data["num_qubits"]),
            evolution_present=bool(data.get("evolution_present", False)),
            evolution_qargs=(
                tuple(int(value) for value in qargs) if qargs is not None else None
            ),
            metrics=CircuitMetrics.from_dict(data.get("metrics", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "evolution_present": self.evolution_present,
            "evolution_qargs": (
                list(self.evolution_qargs) if self.evolution_qargs is not None else None
            ),
            "metrics": self.metrics.to_dict(),
            "metadata": _validated_metadata(self.metadata, path="experiment_circuit.metadata"),
        }


@dataclass(frozen=True)
class Observation:
    label: str
    value: float
    stderr: float | None = None
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        return cls(
            label=str(data["label"]),
            value=float(data["value"]),
            stderr=(float(data["stderr"]) if data.get("stderr") is not None else None),
            shots=(int(data["shots"]) if data.get("shots") is not None else None),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "stderr": self.stderr,
            "shots": self.shots,
            "metadata": _validated_metadata(self.metadata, path=f"observation[{self.label}].metadata"),
        }


@dataclass(frozen=True)
class ObservationSet:
    observations: tuple[Observation, ...]
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = [item.label for item in self.observations]
        if len(set(labels)) != len(labels):
            raise ValueError("observation labels must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationSet":
        return cls(
            tuple(Observation.from_dict(item) for item in data.get("observations", ())),
            source=str(data["source"]),
            metadata=dict(data.get("metadata", {})),
        )

    def by_label(self) -> dict[str, Observation]:
        return {item.label: item for item in self.observations}

    def select(self, labels: Sequence[str], *, source: str | None = None) -> "ObservationSet":
        indexed = self.by_label()
        missing = [label for label in labels if label not in indexed]
        if missing:
            raise KeyError("missing observations: " + ", ".join(missing))
        return ObservationSet(
            tuple(indexed[label] for label in labels),
            source=source or self.source,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observations": [item.to_dict() for item in self.observations],
            "metadata": _validated_metadata(self.metadata, path="observations.metadata"),
        }


@dataclass(frozen=True)
class BasisDistribution:
    label: str
    qubits: tuple[int, ...]
    probabilities: Mapping[str, float]
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BasisDistribution":
        return cls(
            label=str(data["label"]),
            qubits=tuple(int(qubit) for qubit in data["qubits"]),
            probabilities={
                str(outcome): float(probability)
                for outcome, probability in dict(data.get("probabilities", {})).items()
            },
            shots=(int(data["shots"]) if data.get("shots") is not None else None),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "qubits": list(self.qubits),
            "probabilities": {
                str(outcome): float(probability)
                for outcome, probability in self.probabilities.items()
            },
            "shots": self.shots,
            "metadata": _validated_metadata(self.metadata, path=f"distribution[{self.label}].metadata"),
        }

    def summary(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class ExecutionResult:
    source: str
    source_kind: str
    observations: ObservationSet | None = None
    basis_distributions: tuple[BasisDistribution, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionResult":
        observations = data.get("observations")
        # Runtime MeasurementResult.summary() historically represented observations
        # as a flat list. The service contract normalizes both forms to one typed set.
        observation_set = None
        if isinstance(observations, Mapping):
            observation_set = ObservationSet.from_dict(observations)
        elif isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            observation_set = ObservationSet(
                tuple(Observation.from_dict(item) for item in observations),
                source=str(data.get("source", "unknown")),
            )
        return cls(
            source=str(data["source"]),
            source_kind=str(data["source_kind"]),
            observations=observation_set,
            basis_distributions=tuple(
                BasisDistribution.from_dict(item)
                for item in data.get("basis_distributions", ())
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "observations": (
                self.observations.to_dict() if self.observations is not None else None
            ),
            "basis_distributions": [item.to_dict() for item in self.basis_distributions],
            "metadata": _validated_metadata(self.metadata, path="execution.metadata"),
        }




@dataclass(frozen=True)
class PreparationPreflightSummary:
    preparation: PreparationSummary
    simulation: SimulationSummary | None
    simulation_status: str
    target_to_prepared_fidelity: float | None
    preparation_approximation_infidelity: float | None
    observable_plan: ObservablePlanSummary | None
    generated_at: str
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreparationPreflightSummary":
        simulation = data.get("simulation")
        observable_plan = data.get("observable_plan")
        return cls(
            preparation=PreparationSummary.from_dict(data["preparation"]),
            simulation=(
                SimulationSummary.from_dict(simulation)
                if isinstance(simulation, Mapping)
                else None
            ),
            simulation_status=str(data["simulation_status"]),
            target_to_prepared_fidelity=(
                float(data["target_to_prepared_fidelity"])
                if data.get("target_to_prepared_fidelity") is not None
                else None
            ),
            preparation_approximation_infidelity=(
                float(data["preparation_approximation_infidelity"])
                if data.get("preparation_approximation_infidelity") is not None
                else None
            ),
            observable_plan=(
                ObservablePlanSummary.from_dict(observable_plan)
                if isinstance(observable_plan, Mapping)
                else None
            ),
            generated_at=str(data["generated_at"]),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preparation": self.preparation.to_dict(),
            "simulation": self.simulation.to_dict() if self.simulation is not None else None,
            "simulation_status": self.simulation_status,
            "target_to_prepared_fidelity": self.target_to_prepared_fidelity,
            "preparation_approximation_infidelity": self.preparation_approximation_infidelity,
            "observable_plan": (
                self.observable_plan.to_dict() if self.observable_plan is not None else None
            ),
            "generated_at": self.generated_at,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ReferenceSummary:
    simulation: SimulationSummary | None
    measurements: ExecutionResult | None
    status: str = "available"
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceSummary":
        simulation = data.get("simulation")
        measurements = data.get("measurements")
        return cls(
            simulation=(
                SimulationSummary.from_dict(simulation)
                if isinstance(simulation, Mapping)
                else None
            ),
            measurements=(
                ExecutionResult.from_dict(measurements)
                if isinstance(measurements, Mapping)
                else None
            ),
            status=str(data.get("status", "available")),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "simulation": self.simulation.to_dict() if self.simulation is not None else None,
            "measurements": (
                self.measurements.to_dict() if self.measurements is not None else None
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ErrorMetrics:
    count: int
    rmse: float
    mean_absolute_error: float
    max_absolute_error: float
    normalized_rmse: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ErrorMetrics":
        return cls(
            count=int(data["count"]),
            rmse=float(data["rmse"]),
            mean_absolute_error=float(data["mean_absolute_error"]),
            max_absolute_error=float(data["max_absolute_error"]),
            normalized_rmse=(
                float(data["normalized_rmse"])
                if data.get("normalized_rmse") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "rmse": self.rmse,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "normalized_rmse": self.normalized_rmse,
        }


@dataclass(frozen=True)
class DistributionMetrics:
    outcome_count: int
    total_variation_distance: float
    hellinger_distance: float
    classical_fidelity: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistributionMetrics":
        return cls(
            outcome_count=int(data["outcome_count"]),
            total_variation_distance=float(data["total_variation_distance"]),
            hellinger_distance=float(data["hellinger_distance"]),
            classical_fidelity=float(data["classical_fidelity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_count": self.outcome_count,
            "total_variation_distance": self.total_variation_distance,
            "hellinger_distance": self.hellinger_distance,
            "classical_fidelity": self.classical_fidelity,
        }


@dataclass(frozen=True)
class StateEstimateSummary:
    estimator: str
    statevector_materialized: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateEstimateSummary":
        return cls(
            estimator=str(data["estimator"]),
            statevector_materialized=bool(data.get("statevector_materialized", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "statevector_materialized": self.statevector_materialized,
            "metadata": _validated_metadata(self.metadata, path="estimate.metadata"),
        }


@dataclass(frozen=True)
class ExperimentReport:
    """Typed, service-safe experiment report.

    The wire representation retains the stable ``kind``/``schema_version``
    envelope, but users interact with typed fields instead of nested dictionaries.
    """

    generated_at: str
    mode: str
    experiment: ExperimentSummary
    preparation: PreparationSummary
    experiment_circuit: ExperimentCircuitSummary
    preparation_preflight: PreparationPreflightSummary | None
    reference: ReferenceSummary | None
    execution: ExecutionResult | None
    observable_error: ErrorMetrics | None = None
    distribution_errors: Mapping[str, DistributionMetrics] = field(default_factory=dict)
    estimate: StateEstimateSummary | None = None
    held_out_verification_error: ErrorMetrics | None = None
    warnings: tuple[str, ...] = ()


    @classmethod
    def _from_report_data(cls, data: Mapping[str, Any]) -> "ExperimentReport":
        preflight = data.get("preparation_preflight")
        reference = data.get("reference")
        execution = data.get("execution")
        observable_error = data.get("observable_error")
        estimate = data.get("estimate")
        held_out = data.get("held_out_verification_error")
        return cls(
            generated_at=str(data["generated_at"]),
            mode=str(data["mode"]),
            experiment=ExperimentSummary.from_dict(data["experiment"]),
            preparation=PreparationSummary.from_dict(data["preparation"]),
            experiment_circuit=ExperimentCircuitSummary.from_dict(data["experiment_circuit"]),
            preparation_preflight=(
                PreparationPreflightSummary.from_dict(preflight)
                if isinstance(preflight, Mapping)
                else None
            ),
            reference=(
                ReferenceSummary.from_dict(reference)
                if isinstance(reference, Mapping)
                else None
            ),
            execution=(
                ExecutionResult.from_dict(execution)
                if isinstance(execution, Mapping)
                else None
            ),
            observable_error=(
                ErrorMetrics.from_dict(observable_error)
                if isinstance(observable_error, Mapping)
                else None
            ),
            distribution_errors={
                str(label): DistributionMetrics.from_dict(metrics)
                for label, metrics in dict(data.get("distribution_errors", {})).items()
            },
            estimate=(
                StateEstimateSummary.from_dict(estimate)
                if isinstance(estimate, Mapping)
                else None
            ),
            held_out_verification_error=(
                ErrorMetrics.from_dict(held_out)
                if isinstance(held_out, Mapping)
                else None
            ),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
        )

    def _report_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "mode": self.mode,
            "experiment": self.experiment.to_dict(),
            "preparation": self.preparation.to_dict(),
            "experiment_circuit": self.experiment_circuit.to_dict(),
            "preparation_preflight": (
                self.preparation_preflight.to_dict()
                if self.preparation_preflight is not None
                else None
            ),
            "reference": self.reference.to_dict() if self.reference is not None else None,
            "execution": self.execution.to_dict() if self.execution is not None else None,
            "observable_error": (
                self.observable_error.to_dict() if self.observable_error is not None else None
            ),
            "distribution_errors": {
                label: metrics.to_dict() for label, metrics in self.distribution_errors.items()
            },
            "estimate": self.estimate.to_dict() if self.estimate is not None else None,
            "held_out_verification_error": (
                self.held_out_verification_error.to_dict()
                if self.held_out_verification_error is not None
                else None
            ),
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "experiment-report",
            "report": _json_value(self._report_dict(), path="report"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentReport":
        _require_schema(data, kind="experiment-report")
        report = data.get("report")
        if not isinstance(report, Mapping):
            raise ValueError("experiment-report payload has no report object")
        return cls._from_report_data(report)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentReport":
        return cls.from_dict(json.loads(payload))
