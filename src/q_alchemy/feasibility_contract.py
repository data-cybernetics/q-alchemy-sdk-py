"""Dependency-free portable contract for the hosted Q-Alchemy feasibility service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .quantum_io_contract import ExperimentReport
    from q_alchemy.visualization import ExperimentDiagram
    from qiskit import QuantumCircuit


class Relation(str, Enum):
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"


@dataclass(frozen=True)
class Criterion:
    metric: str
    relation: Relation
    threshold: float
    label: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("criterion metric must not be empty")
        if not isfinite(self.threshold):
            raise ValueError("criterion threshold must be finite")

    @classmethod
    def at_most(cls, metric: str, threshold: float, *, label: str | None = None) -> "Criterion":
        return cls(metric, Relation.LE, float(threshold), label=label)

    @classmethod
    def at_least(cls, metric: str, threshold: float, *, label: str | None = None) -> "Criterion":
        return cls(metric, Relation.GE, float(threshold), label=label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "relation": self.relation.value,
            "threshold": self.threshold,
            "label": self.label,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Criterion":
        return cls(
            metric=str(data["metric"]),
            relation=Relation(str(data["relation"])),
            threshold=float(data["threshold"]),
            label=(str(data["label"]) if data.get("label") is not None else None),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class SolutionCriteria:
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("at least one solution criterion is required")
        metrics = [item.metric for item in self.criteria]
        if len(set(metrics)) != len(metrics):
            raise ValueError("solution criteria must use unique metric names")

    @classmethod
    def of(cls, *criteria: Criterion) -> "SolutionCriteria":
        return cls(tuple(criteria))

    @classmethod
    def common(
        cls,
        *,
        max_observable_rmse: float | None = None,
        max_distribution_tvd: float | None = None,
        min_distribution_fidelity: float | None = None,
        min_target_to_prepared_fidelity: float | None = None,
        max_held_out_rmse: float | None = None,
        min_final_state_fidelity: float | None = None,
        min_reference_to_estimated_fidelity: float | None = None,
    ) -> "SolutionCriteria":
        names = {
            "max_observable_rmse": ("quality.observable_rmse", Criterion.at_most),
            "max_distribution_tvd": ("quality.distribution_tvd_max", Criterion.at_most),
            "min_distribution_fidelity": ("quality.distribution_classical_fidelity_min", Criterion.at_least),
            "min_target_to_prepared_fidelity": ("quality.target_to_prepared_fidelity", Criterion.at_least),
            "max_held_out_rmse": ("quality.held_out_rmse", Criterion.at_most),
            "min_final_state_fidelity": ("quality.final_state_fidelity", Criterion.at_least),
            "min_reference_to_estimated_fidelity": ("quality.reference_to_estimated_fidelity", Criterion.at_least),
        }
        values = locals()
        items = []
        for argument, (metric, constructor) in names.items():
            value = values[argument]
            if value is not None:
                items.append(constructor(metric, value))
        return cls(tuple(items))

    def to_dict(self) -> dict[str, Any]:
        return {"criteria": [item.to_dict() for item in self.criteria]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolutionCriteria":
        return cls(tuple(Criterion.from_dict(item) for item in data.get("criteria", ())))


class QuantumExecutionPolicy(str, Enum):
    WHEN_NEEDED = "when-needed"
    COMPARE = "compare"
    NEVER = "never"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NOT_NEEDED = "not-needed"
    UNKNOWN = "unknown"


class RecommendedCompute(str, Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    NONE_AVAILABLE = "none-available"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class FeasibilityPolicy:
    classical_first: bool = True
    quantum_execution: QuantumExecutionPolicy = QuantumExecutionPolicy.WHEN_NEEDED
    dense_memory_utilization_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.classical_first is not True:
            raise ValueError(
                "classical_first=False is not supported; feasibility is always "
                "classical-first. Use quantum_execution=COMPARE to run both paths."
            )
        if not 0 < self.dense_memory_utilization_fraction <= 1:
            raise ValueError("dense_memory_utilization_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "classical_first": self.classical_first,
            "quantum_execution": self.quantum_execution.value,
            "dense_memory_utilization_fraction": self.dense_memory_utilization_fraction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeasibilityPolicy":
        return cls(
            classical_first=bool(data.get("classical_first", True)),
            quantum_execution=QuantumExecutionPolicy(
                str(data.get("quantum_execution", QuantumExecutionPolicy.WHEN_NEEDED.value))
            ),
            dense_memory_utilization_fraction=float(
                data.get("dense_memory_utilization_fraction", 0.80)
            ),
        )


@dataclass(frozen=True)
class ClassicalResources:
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    cpu_cores: int | None = None
    gpu_memory_bytes: int | None = None
    accelerator: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "cpu_cores": self.cpu_cores,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "accelerator": self.accelerator,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassicalResources":
        return cls(
            total_memory_bytes=_int_or_none(data.get("total_memory_bytes")),
            available_memory_bytes=_int_or_none(data.get("available_memory_bytes")),
            cpu_cores=_int_or_none(data.get("cpu_cores")),
            gpu_memory_bytes=_int_or_none(data.get("gpu_memory_bytes")),
            accelerator=(str(data["accelerator"]) if data.get("accelerator") is not None else None),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CircuitCompressionConfig:
    """Exact compression settings for the complete logical experiment circuit."""

    enabled: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("circuit_compression.enabled must be a bool")
        if not isinstance(self.options, Mapping):
            raise ValueError("circuit_compression.options must be a mapping")
        normalized = dict(self.options)
        if "equivalence" in normalized:
            if normalized["equivalence"] != "reachable_subspace":
                raise ValueError(
                    "feasibility full-circuit compression requires "
                    "equivalence='reachable_subspace'"
                )
            normalized.pop("equivalence")
        if "collect_report" in normalized:
            if normalized["collect_report"] is not True:
                raise ValueError(
                    "feasibility owns collect_report and requires it to be true"
                )
            normalized.pop("collect_report")
        object.__setattr__(self, "options", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionConfig":
        if not isinstance(data, Mapping):
            raise ValueError("circuit_compression must be an object")
        return cls(
            enabled=data.get("enabled", True),
            options=data.get("options", {}),
        )


@dataclass(frozen=True)
class EvidenceCollectionConfig:
    """Portable acquisition controls for Feasibility >= 0.6.48.

    Execution controls are separate from backend configuration. Retired sweep
    keys in saved JSON are ignored, as they are by the service.
    """
    provider: str = "ibm"
    backend: str | None = None
    least_busy: bool = False
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    shots: int = 4096
    preparation_options: Mapping[str, Any] = field(default_factory=dict)
    sparse_config: Mapping[str, Any] = field(default_factory=lambda: {"sparse_epsilon": 0.0, "final_sparse_epsilon": 0.0})
    qtucker_config: Mapping[str, Any] = field(default_factory=dict)
    circuit_compression: CircuitCompressionConfig = field(
        default_factory=CircuitCompressionConfig
    )
    # Controls automatic QTucker reconstruction-observable generation when the
    # feasibility service needs state-estimation evidence and the experiment did
    # not supply an explicit estimator training plan.
    qtucker_observable_config: Mapping[str, Any] = field(default_factory=dict)
    execution_options: Mapping[str, Any] = field(
        default_factory=lambda: {"transpile": True}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.circuit_compression, CircuitCompressionConfig):
            raise ValueError("circuit_compression must be a CircuitCompressionConfig")
        if not isinstance(self.qtucker_observable_config, Mapping):
            raise ValueError("qtucker_observable_config must be a mapping")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend.strip()
        ):
            raise ValueError("backend must be a non-empty string when provided")
        if self.backend is not None and self.least_busy:
            raise ValueError("select at most one backend strategy: backend or least_busy")
        if not isinstance(self.execution_options, Mapping):
            raise ValueError("execution_options must be a mapping")
        backend_options = dict(self.backend_options)
        execution_options = dict(self.execution_options)
        # Normalize the legacy request form exactly as the core does. No
        # backend objects are constructed by this lightweight client.
        for key in ("transpile", "transpile_options", "estimator_options",
                    "backend_run_options", "run_options"):
            if key not in backend_options:
                continue
            value = backend_options.pop(key)
            if key in execution_options and execution_options[key] != value:
                raise ValueError(
                    f"conflicting Quantum I/O execution option {key!r} in "
                    "backend_options and execution_options"
                )
            execution_options[key] = value
        transpile = execution_options.get("transpile", False)
        if not isinstance(transpile, bool):
            raise ValueError("execution_options.transpile must be a boolean")
        if execution_options.get("transpile_options") and not transpile:
            raise ValueError("execution_options.transpile_options require transpile=True")
        try:
            json.dumps(backend_options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "backend_options must be JSON-serializable; encode Qiskit Aer "
                "NoiseModel values using the qiskit-aer-noise-model-v1 envelope"
            ) from exc
        try:
            json.dumps(execution_options, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution_options must be JSON-serializable") from exc
        object.__setattr__(self, "backend_options", backend_options)
        object.__setattr__(self, "execution_options", execution_options)
        if isinstance(self.shots, bool) or not isinstance(self.shots, Integral) or self.shots <= 0:
            raise ValueError("shots must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "backend": self.backend,
            "least_busy": self.least_busy,
            "backend_options": dict(self.backend_options),
            "shots": self.shots,
            "preparation_options": dict(self.preparation_options),
            "sparse_config": dict(self.sparse_config),
            "qtucker_config": dict(self.qtucker_config),
            "qtucker_observable_config": dict(self.qtucker_observable_config),
            "execution_options": dict(self.execution_options),
            "circuit_compression": self.circuit_compression.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceCollectionConfig":
        return cls(
            provider=str(data.get("provider", "ibm")),
            backend=(str(data["backend"]) if data.get("backend") is not None else None),
            least_busy=bool(data.get("least_busy", False)),
            backend_options=dict(data.get("backend_options", {})),
            shots=data.get("shots", 4096),
            preparation_options=dict(data.get("preparation_options", {})),
            sparse_config=dict(data.get("sparse_config", {"sparse_epsilon": 0.0, "final_sparse_epsilon": 0.0})),
            qtucker_config=dict(data.get("qtucker_config", {})),
            circuit_compression=CircuitCompressionConfig.from_dict(
                data.get("circuit_compression", {})
            ),
            qtucker_observable_config=dict(data.get("qtucker_observable_config", {})),
            execution_options=dict(data.get("execution_options", {"transpile": True})),
        )


@dataclass(frozen=True)
class FeasibilityRequest:
    criteria: SolutionCriteria
    policy: FeasibilityPolicy = field(default_factory=FeasibilityPolicy)
    evidence_collection: EvidenceCollectionConfig = field(default_factory=EvidenceCollectionConfig)
    classical_resources: ClassicalResources | None = None
    max_steps: int = 16
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported feasibility request schema_version={self.schema_version}"
            )
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, Integral) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "feasibility-request",
            "criteria": self.criteria.to_dict(),
            "policy": self.policy.to_dict(),
            "evidence_collection": self.evidence_collection.to_dict(),
            "classical_resources": self.classical_resources.to_dict() if self.classical_resources else None,
            "max_steps": self.max_steps,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeasibilityRequest":
        if int(data.get("schema_version", 0)) != 1 or data.get("kind") != "feasibility-request":
            raise ValueError("unsupported feasibility request contract")
        resources = data.get("classical_resources")
        return cls(
            criteria=SolutionCriteria.from_dict(data["criteria"]),
            policy=FeasibilityPolicy.from_dict(data.get("policy", {})),
            evidence_collection=EvidenceCollectionConfig.from_dict(data.get("evidence_collection", {})),
            classical_resources=ClassicalResources.from_dict(resources) if resources is not None else None,
            max_steps=data.get("max_steps", 16),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "FeasibilityRequest":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("feasibility request JSON must contain an object")
        return cls.from_dict(data)



_FEASIBILITY_CRITERION_LABELS = {
    "quality.observable_rmse": "Observable RMSE",
    "quality.distribution_tvd_max": "Distribution TVD",
    "quality.distribution_classical_fidelity_min": "Distribution fidelity",
    "quality.target_to_prepared_fidelity": "Target-to-prepared fidelity",
    "quality.held_out_rmse": "Held-out RMSE",
    "quality.final_state_fidelity": "Final-state fidelity",
    "quality.reference_to_estimated_fidelity": "Reference-to-estimated fidelity",
}

_CLASSICAL_AMPLITUDE_CAPPING_OCCURRED = (
    "classical.approximation.amplitude_capping_occurred"
)
_CLASSICAL_SIMULATOR_BUDGET_BYTES = "classical.resources.simulator_budget_bytes"
_CLASSICAL_ESTIMATED_PEAK_MEMORY_BYTES = (
    "classical.resources.estimated_peak_memory_bytes"
)
_QPU_NOISY_SIMULATION_COMPLETED = "quantum.model.execution_completed"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _criterion_label(metric: str, explicit_label: Any) -> str:
    if explicit_label:
        return str(explicit_label)
    return _FEASIBILITY_CRITERION_LABELS.get(metric, metric)


def _format_criterion_number(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_criteria_lines(
    evaluation: Any,
    *,
    heading: str = "Quality criteria",
    include_all_missing: bool = False,
    qualify_reference: bool = False,
) -> list[str]:
    evaluation_data = _mapping(evaluation)
    results = _sequence_of_mappings(evaluation_data.get("results"))
    if not results:
        return []

    # Match q-alchemy-feasibility: suppress an all-missing table when a branch
    # was decided entirely from static/resource evidence. Once one criterion has
    # evidence, keep missing companion criteria visible.
    if not include_all_missing and not any(isinstance(result.get("evidence"), Mapping) for result in results):
        return []

    lines = [f"  {heading}:"]
    for result in results:
        criterion = _mapping(result.get("criterion"))
        metric = str(criterion.get("metric", ""))
        label = _criterion_label(metric, criterion.get("label"))
        relation = str(criterion.get("relation", ""))
        threshold = _format_criterion_number(criterion.get("threshold"))
        status = str(result.get("status", "missing"))
        evidence = result.get("evidence")

        if not isinstance(evidence, Mapping) or status == "missing":
            lines.append(
                f"    {label}: not available "
                f"(required {relation} {threshold}) "
                f"[{status.upper()}]"
            )
            continue

        numeric = _numeric_value(evidence.get("value"))
        rendered_value = (
            str(evidence.get("value"))
            if numeric is None
            else _format_criterion_number(numeric)
        )
        if evidence.get("uncertainty") is not None:
            rendered_value += (
                f" ± {_format_criterion_number(evidence.get('uncertainty'))}"
            )
        if bool(evidence.get("lower_bound", False)):
            rendered_value += " (lower bound)"

        unit = f" {evidence.get('unit')}" if evidence.get("unit") else ""
        status_text = status.upper()
        if status == "inconclusive":
            metadata = _mapping(evidence.get("metadata"))
            qualification = str(metadata.get("qualification") or "qualified evidence")
            status_text = f"INCONCLUSIVE: {qualification.replace('-', ' ')}"
        lines.append(
            f"    {label}: {rendered_value} {relation} "
            f"{threshold}{unit} [{status_text}]"
            + (" (relative to the simulated circuit reference)"
               if qualify_reference and evidence.get("source") == "quantum-io:exact-reference"
               else "")
        )
    return lines


def _format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "not available"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.3g} {unit}"


def _violated_criterion_labels(evaluation: Any) -> tuple[str, ...]:
    results = _sequence_of_mappings(_mapping(evaluation).get("results"))
    labels: list[str] = []
    for result in results:
        if str(result.get("status")) != "violated":
            continue
        criterion = _mapping(result.get("criterion"))
        labels.append(
            _criterion_label(
                str(criterion.get("metric", "")),
                criterion.get("label"),
            )
        )
    return tuple(labels)


def _quality_failure_reason(evaluation: Any, *, source: str) -> str | None:
    labels = _violated_criterion_labels(evaluation)
    if not labels:
        return None
    if len(labels) == 1:
        return f"{source} violates the required {labels[0]} criterion."
    return (
        f"{source} violates {len(labels)} required quality criteria: "
        + ", ".join(labels)
        + "."
    )


def _format_feasibility_status(assessment: Mapping[str, Any]) -> str:
    status = str(assessment.get("status", "unknown"))
    infeasibility_kind = assessment.get("infeasibility_kind")
    inconclusive_kind = assessment.get("inconclusive_kind")
    if status == "infeasible" and infeasibility_kind is not None:
        return f"{infeasibility_kind} infeasible"
    if status == "unknown" and inconclusive_kind is not None:
        return f"{inconclusive_kind} inconclusive"
    return status


def _notes(assessment: Mapping[str, Any]) -> tuple[str, ...]:
    value = assessment.get("notes")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _classical_infeasibility_reason(
    assessment: Mapping[str, Any],
) -> str | None:
    if str(assessment.get("status")) != "infeasible":
        return None
    notes = _notes(assessment)
    if notes:
        return notes[0]
    return _quality_failure_reason(
        assessment.get("criteria"),
        source="Available classical evidence",
    )


def _quantum_infeasibility_reason(
    assessment: Mapping[str, Any],
) -> str | None:
    status = str(assessment.get("status"))
    basis = str(assessment.get("assessment_basis") or "")
    if (
        status == "unknown"
        and str(assessment.get("inconclusive_kind") or "") == "execution"
        and basis == "execution-unavailable"
    ):
        notes = _notes(assessment)
        return notes[-1] if notes else "The configured quantum target is unavailable."
    if status != "infeasible":
        return None

    if basis.startswith("measured-qpu"):
        reason = _quality_failure_reason(
            assessment.get("criteria"),
            source="Measured QPU evidence",
        )
        if reason is not None:
            return reason
    if basis.startswith("calibrated-model") or basis == "simulated-quantum":
        reason = _quality_failure_reason(
            assessment.get("model_criteria"),
            source=(
                "Configured quantum simulator"
                if basis == "simulated-quantum"
                else "Backend-calibrated noisy simulation"
            ),
        )
        if reason is not None:
            return reason

    notes = _notes(assessment)
    if basis == "static-hard-constraint":
        for note in notes:
            if "requires" in note or "non-operational" in note:
                return note
    for note in reversed(notes):
        lowered = note.lower()
        if "violat" in lowered or "fails" in lowered or "infeasible" in lowered:
            return note
    return notes[0] if notes else None


def _quantum_quality_reason(assessment: Mapping[str, Any]) -> str | None:
    quality_status = str(assessment.get("quality_status", "not-assessed"))
    basis = str(assessment.get("assessment_basis") or "")
    if quality_status == "violated":
        if basis.startswith("measured-qpu"):
            return _quality_failure_reason(
                assessment.get("criteria"),
                source="Measured QPU evidence",
            )
        if basis.startswith("calibrated-model") or basis == "simulated-quantum":
            return _quality_failure_reason(
                assessment.get("model_criteria"),
                source=(
                    "Configured quantum simulator"
                    if basis == "simulated-quantum"
                    else "Backend-calibrated noisy simulation"
                ),
            )

    if quality_status != "inconclusive":
        return None

    evaluation = (
        assessment.get("model_criteria")
        if basis.startswith("calibrated-model") or basis == "simulated-quantum"
        else assessment.get("criteria")
    )
    for result in _sequence_of_mappings(_mapping(evaluation).get("results")):
        if str(result.get("status")) != "inconclusive":
            continue
        evidence = result.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        reason = _mapping(evidence.get("metadata")).get("qualification_reason")
        if reason:
            return str(reason)
    missing = [
        result for result in _sequence_of_mappings(_mapping(evaluation).get("results"))
        if _mapping(result.get("criterion")).get("required", True)
        and result.get("status") == "missing"
    ]
    if missing:
        labels = [
            _criterion_label(str(_mapping(result.get("criterion")).get("metric", "")),
                             _mapping(result.get("criterion")).get("label"))
            for result in missing
        ]
        subject = (
            "The configured quantum simulator"
            if basis.startswith("calibrated-model") or basis == "simulated-quantum"
            else "The QPU execution"
        )
        message = (
            f"{subject} completed, but the required {labels[0]} metric is unavailable"
            if len(labels) == 1 else
            f"{subject} completed, but {len(labels)} required quality metrics are "
            f"unavailable: {', '.join(labels)}"
        )
        metrics = {_mapping(result.get("criterion")).get("metric") for result in missing}
        if metrics == {"quality.observable_rmse"}:
            return message + " because no independent reference observable values were available."
        return message + "."
    for note in reversed(_notes(assessment)):
        lowered = note.lower()
        if "cannot conclusively" in lowered or "did not produce every metric" in lowered:
            return note
    return None


def _latest_evidence(
    evidence: Mapping[str, Any],
    metric: str,
    *,
    scopes: tuple[str, ...],
) -> Mapping[str, Any] | None:
    records = [
        record
        for record in _sequence_of_mappings(evidence.get("records"))
        if str(record.get("metric")) == metric
        and str(record.get("scope", "general")) in scopes
    ]
    if not records:
        return None
    # Compare instants, not ISO strings: offsets/precision may differ. Missing,
    # malformed, or naive timestamps use report insertion order, matching core.
    instants: list[datetime] = []
    for record in records:
        try:
            instant = datetime.fromisoformat(str(record.get("generated_at")))
            if instant.utcoffset() is None:
                return records[-1]
            instants.append(instant.astimezone(timezone.utc))
        except (ValueError, TypeError, OverflowError):
            return records[-1]
    return records[max(range(len(records)), key=lambda i: (instants[i], i))]


def _preparation_claim_warning(evidence: Mapping[str, Any]) -> str | None:
    """Render Quantum I/O's verdict; never infer it or simulate in the SDK.

    True means the verified loss contradicted the estimate, False means the
    comparison passed, and None/missing means no certified comparison. This
    diagnostic does not change resource feasibility or measure target energy.
    """
    record = _latest_evidence(evidence, "preparation.claim_contradicted", scopes=("classical",))
    if record is None or record.get("value") is not True:
        return None
    metadata = _mapping(record.get("metadata"))

    def number(key: str) -> str:
        value = _numeric_value(metadata.get(key))
        return f"{value:.12g}" if value is not None and isfinite(value) else "unavailable"

    return (
        "Preparation fidelity discrepancy: Quantum I/O reports that measured "
        f"loss {number('preparation_approximation_infidelity')} exceeds the "
        f"initializer's estimated loss {number('claimed_fidelity_loss')} "
        f"(target-to-prepared fidelity {number('target_to_prepared_fidelity')}). "
        "The simulated circuit reference includes the preparation circuit; "
        "reference-relative quality does not establish accuracy against the intended target."
    )


def _format_classical_resource_criteria_lines(
    assessment: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    # Keep this formatter aligned with q-alchemy-feasibility 0.6.48. The SDK
    # intentionally keeps a dependency-free copy so formatting stays local.
    criteria: list[str] = []
    available_resources = _mapping(assessment.get("available_resources"))
    available = available_resources.get("available_memory_bytes")
    requirement_value = assessment.get("required_resources")
    requirement = _mapping(requirement_value) if requirement_value is not None else None
    criteria.append(f"Classical RAM limit: {_format_bytes(available)}")

    selected_method = assessment.get("selected_method")
    telemetry_records = [
        record
        for record in _sequence_of_mappings(evidence.get("records"))
        if str(record.get("scope", "general")) == "classical"
        and (selected_method is None or record.get("source") == selected_method)
    ]
    telemetry = {"records": telemetry_records}
    for metric, label in (
        (
            _CLASSICAL_SIMULATOR_BUDGET_BYTES,
            "Detected simulator host budget",
        ),
        (
            _CLASSICAL_ESTIMATED_PEAK_MEMORY_BYTES,
            "Sparse memory estimate at amplitude limit",
        ),
    ):
        record = _latest_evidence(telemetry, metric, scopes=("classical",))
        value = _numeric_value(record.get("value")) if record is not None else None
        if value is not None and value >= 0:
            criteria.append(f"{label}: {_format_bytes(int(value))}")

    capping = _latest_evidence(
        evidence,
        _CLASSICAL_AMPLITUDE_CAPPING_OCCURRED,
        scopes=("classical",),
    )
    if capping is not None and capping.get("value") is True:
        if (
            requirement is not None
            and requirement.get("memory_bytes") is not None
            and bool(requirement.get("memory_is_lower_bound", False))
        ):
            lower_bound = int(requirement["memory_bytes"])
            if available is not None:
                criteria.append(
                    "Sparse simulation memory: > "
                    f"{_format_bytes(lower_bound)} required (lower bound); "
                    f"{_format_bytes(available)} available [VIOLATED]"
                )
            else:
                criteria.append(
                    "Sparse simulation memory: > "
                    f"{_format_bytes(lower_bound)} required (lower bound) [VIOLATED]"
                )
        elif available is not None:
            criteria.append(
                f"Sparse simulation memory: > {_format_bytes(available)} required "
                f"(lower bound); {_format_bytes(available)} available [VIOLATED]"
            )
        else:
            criteria.append(
                "Sparse simulation memory: configured allocation exceeded [VIOLATED]"
            )
    elif requirement is not None and requirement.get("memory_bytes") is not None:
        needed = int(requirement["memory_bytes"])
        is_lower_bound = bool(requirement.get("memory_is_lower_bound", False))
        bound = " (lower bound)" if is_lower_bound else ""
        if available is None:
            comparison = "available RAM unknown [INCONCLUSIVE]"
        elif needed > int(available):
            comparison = f"> {_format_bytes(available)} available [VIOLATED]"
        elif is_lower_bound:
            comparison = f"<= {_format_bytes(available)} available [INCONCLUSIVE]"
        else:
            comparison = f"<= {_format_bytes(available)} available [SATISFIED]"
        criteria.append(
            f"Classical memory requirement: {_format_bytes(needed)}{bound}; {comparison}"
        )

    if requirement is not None:
        available_cpu = available_resources.get("cpu_cores")
        required_cpu = requirement.get("cpu_cores")
        if required_cpu is not None and available_cpu is not None:
            status = "SATISFIED" if int(available_cpu) >= int(required_cpu) else "VIOLATED"
            relation = ">=" if int(available_cpu) >= int(required_cpu) else "<"
            criteria.append(
                f"CPU cores: {available_cpu} available {relation} "
                f"{required_cpu} required [{status}]"
            )

        available_gpu = available_resources.get("gpu_memory_bytes")
        required_gpu = requirement.get("gpu_memory_bytes")
        if required_gpu is not None and available_gpu is not None:
            status = "SATISFIED" if int(available_gpu) >= int(required_gpu) else "VIOLATED"
            relation = ">=" if int(available_gpu) >= int(required_gpu) else "<"
            criteria.append(
                f"GPU memory: {_format_bytes(available_gpu)} available {relation} "
                f"{_format_bytes(required_gpu)} required [{status}]"
            )

    resource_gap_value = assessment.get("resource_gap")
    resource_gap = _mapping(resource_gap_value) if resource_gap_value is not None else None
    if resource_gap is not None:
        additional_cpu = resource_gap.get("additional_cpu_cores")
        if additional_cpu is not None and int(additional_cpu) > 0:
            criteria.append(
                f"CPU capacity: {additional_cpu} additional cores required [VIOLATED]"
            )
        additional_gpu = resource_gap.get("additional_gpu_memory_bytes")
        if additional_gpu is not None and int(additional_gpu) > 0:
            criteria.append(
                "GPU memory capacity: "
                f"{_format_bytes(additional_gpu)} additional required [VIOLATED]"
            )

    return ["  Resource criteria:"] + [f"    {item}" for item in criteria]

def _format_quantum_resource_criteria_lines(
    assessment: Mapping[str, Any],
) -> list[str]:
    criteria: list[str] = []
    resources_value = assessment.get("available_resources")
    requirement_value = assessment.get("required_resources")
    resources = _mapping(resources_value) if resources_value is not None else None
    requirement = _mapping(requirement_value) if requirement_value is not None else None

    if resources is not None and requirement is not None:
        available_qubits = resources.get("qubits")
        required_qubits = requirement.get("qubits")
        if required_qubits is not None and available_qubits is not None:
            status = "SATISFIED" if int(available_qubits) >= int(required_qubits) else "VIOLATED"
            relation = ">=" if int(available_qubits) >= int(required_qubits) else "<"
            criteria.append(
                f"Qubit capacity: {available_qubits} available {relation} "
                f"{required_qubits} required [{status}]"
            )

        operational = resources.get("operational")
        if operational is not None:
            status = "SATISFIED" if bool(operational) else "VIOLATED"
            criteria.append(
                f"Backend operational: {'yes' if bool(operational) else 'no'} [{status}]"
            )

        for required_key, available_key, label in (
            ("max_one_qubit_error", "median_one_qubit_error", "Median 1Q error"),
            ("max_two_qubit_error", "median_two_qubit_error", "Median 2Q error"),
        ):
            required_value = requirement.get(required_key)
            available_value = resources.get(available_key)
            if required_value is None or available_value is None:
                continue
            satisfied = float(available_value) <= float(required_value)
            criteria.append(
                f"{label}: {float(available_value):.4g} <= "
                f"{float(required_value):.4g} "
                f"[{'SATISFIED' if satisfied else 'VIOLATED'}]"
            )

        for required_key, available_key, label in (
            ("min_t1_sec", "median_t1_sec", "Median T1"),
            ("min_t2_sec", "median_t2_sec", "Median T2"),
        ):
            required_value = requirement.get(required_key)
            available_value = resources.get(available_key)
            if required_value is None or available_value is None:
                continue
            satisfied = float(available_value) >= float(required_value)
            criteria.append(
                f"{label}: {float(available_value):.4g} s >= "
                f"{float(required_value):.4g} s "
                f"[{'SATISFIED' if satisfied else 'VIOLATED'}]"
            )

        required_readout = requirement.get("max_readout_error")
        available_readout = resources.get("median_readout_error")
        if required_readout is not None and available_readout is not None:
            satisfied = float(available_readout) <= float(required_readout)
            criteria.append(
                f"Median readout error: {float(available_readout):.4g} <= "
                f"{float(required_readout):.4g} "
                f"[{'SATISFIED' if satisfied else 'VIOLATED'}]"
            )

    if not criteria:
        return []
    return ["  Resource criteria:"] + [f"    {item}" for item in criteria]


@dataclass(frozen=True)
class FeasibilityRecommendation:
    compute: RecommendedCompute
    reason: str


@dataclass(frozen=True)
class FeasibilityReport:
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeasibilityReport":
        if int(data.get("schema_version", 0)) != 1:
            raise ValueError("unsupported feasibility report schema")
        if data.get("kind") != "feasibility-report":
            raise ValueError("payload kind must be 'feasibility-report'")
        return cls(dict(data))

    @property
    def final(self) -> bool:
        return bool(self.raw.get("final"))

    @property
    def classical(self) -> Mapping[str, Any]:
        value = self.raw.get("classical")
        return value if isinstance(value, Mapping) else {}

    @property
    def quantum(self) -> Mapping[str, Any]:
        value = self.raw.get("quantum")
        return value if isinstance(value, Mapping) else {}

    @property
    def recommendation(self) -> FeasibilityRecommendation:
        value = self.raw.get("recommendation")
        data = value if isinstance(value, Mapping) else {}
        return FeasibilityRecommendation(
            RecommendedCompute(str(data.get("compute", "undetermined"))),
            str(data.get("reason", "")),
        )

    @property
    def evidence(self) -> Mapping[str, Any]:
        value = self.raw.get("evidence")
        return value if isinstance(value, Mapping) else {}

    @property
    def circuit_compression(self) -> Mapping[str, Any] | None:
        value = self.raw.get("circuit_compression")
        return value if isinstance(value, Mapping) else None

    @property
    def execution_trace(self) -> Mapping[str, Any] | None:
        value = self.raw.get("execution_trace")
        return value if isinstance(value, Mapping) else None

    @property
    def computation_result(self) -> Mapping[str, Any] | None:
        value = self.raw.get("computation_result")
        return value if isinstance(value, Mapping) else None

    @property
    def computation_experiment_report(self) -> "ExperimentReport | None":
        """Return the selected/available computation as a typed ExperimentReport.

        Feasibility owns the decision about which computation is relevant. The SDK
        only deserializes the portable ExperimentReport included in the service
        response; it never reruns or reconstructs the feasibility decision.
        """

        result = self.computation_result
        if result is None:
            return None
        payload = result.get("experiment_report")
        if not isinstance(payload, Mapping):
            return None
        from .quantum_io_contract import ExperimentReport

        return ExperimentReport.from_dict(payload)

    @property
    def quantum_circuit(self) -> "QuantumCircuit | None":
        """Return the produced logical experiment circuit as a Qiskit circuit.

        The service transports the exact logical ``P + U`` circuit as QASM 3.
        Qiskit performs the reconstruction; the SDK does not implement its own
        circuit parser, representation, or drawer.
        """

        value = self.raw.get("quantum_circuit")
        if not isinstance(value, Mapping):
            return None
        if value.get("kind") != "quantum-circuit":
            raise ValueError("unsupported quantum circuit payload kind")
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("unsupported quantum circuit payload schema")
        if value.get("role") != "logical-experiment-circuit":
            raise ValueError("unsupported quantum circuit role")
        if value.get("format") != "qasm3":
            raise ValueError("unsupported quantum circuit format")
        qasm = value.get("qasm")
        if not isinstance(qasm, str) or not qasm.strip():
            raise ValueError("quantum circuit payload has no QASM 3 program")
        try:
            from qiskit import qasm3
        except ImportError as exc:
            raise RuntimeError(
                "Quantum circuit reconstruction requires Qiskit. "
                "Install the SDK qiskit extra."
            ) from exc
        return qasm3.loads(qasm)

    @property
    def experiment_diagram(self) -> "ExperimentDiagram | None":
        """Return the server-provided typed experiment diagram.

        The SDK deliberately does not reconstruct feasibility steps or statuses.
        Install the optional ``visualization`` extra to obtain the shared diagram
        model and renderers.
        """

        value = self.raw.get("experiment_diagram")
        if not isinstance(value, Mapping):
            return None
        try:
            from q_alchemy.visualization import ExperimentDiagram
        except ImportError as exc:
            raise RuntimeError(
                "Experiment diagram deserialization requires q-alchemy-visualization. "
                "Install the SDK visualization extra and configure the Q-Alchemy "
                "package index."
            ) from exc
        return ExperimentDiagram.from_dict(value)

    @property
    def classical_status(self) -> FeasibilityStatus:
        return FeasibilityStatus(str(self.classical.get("status", "unknown")))

    @property
    def quantum_status(self) -> FeasibilityStatus:
        return FeasibilityStatus(str(self.quantum.get("status", "unknown")))

    @property
    def next_evidence(self) -> tuple[Mapping[str, Any], ...]:
        value = self.raw.get("next_evidence") or ()
        return tuple(item for item in value if isinstance(item, Mapping))

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(str(item) for item in (self.raw.get("warnings") or ()))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "FeasibilityReport":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("feasibility report JSON must contain an object")
        return cls.from_dict(data)

    def format_summary(self) -> str:
        """Format the report using q-alchemy-feasibility's canonical summary.

        Keep this implementation synchronized with
        ``q_alchemy.feasibility.models.FeasibilityReport.format_summary``. The
        SDK intentionally formats locally instead of depending on the heavier
        hosted feasibility runtime.
        """

        classical = self.classical
        quantum = self.quantum
        evidence = self.evidence
        lines = ["COMPUTE FEASIBILITY", "-------------------"]

        compression = self.circuit_compression
        if compression is not None:
            lines.extend(["", "CIRCUIT COMPRESSION"])
            lines.append(f"  Attempted: {bool(compression.get('attempted', False))}")
            lines.append(f"  Applied: {bool(compression.get('applied', False))}")
            changed = compression.get("changed")
            lines.append(
                "  Changed: "
                + (str(bool(changed)) if changed is not None else "not available")
            )
            lines.append(
                f"  Circuit used: {compression.get('circuit_used') or 'not available'}"
            )
            lines.append(f"  Result: {compression.get('reason') or 'not available'}")
            lines.append(
                f"  Equivalence: {compression.get('equivalence') or 'not available'}"
            )
            input_metrics = compression.get("input_metrics")
            compressed_metrics = compression.get("compressed_metrics")
            if isinstance(input_metrics, Mapping):
                after = (
                    compressed_metrics
                    if isinstance(compressed_metrics, Mapping)
                    else input_metrics
                )
                lines.append(
                    "  1Q operations: "
                    f"{input_metrics.get('one_qubit_operations')} -> "
                    f"{after.get('one_qubit_operations')}"
                )
                lines.append(
                    "  2Q operations: "
                    f"{input_metrics.get('two_qubit_operations')} -> "
                    f"{after.get('two_qubit_operations')}"
                )
                lines.append(
                    f"  Depth: {input_metrics.get('depth')} -> {after.get('depth')}"
                )
            else:
                lines.extend(
                    [
                        "  1Q operations: not available",
                        "  2Q operations: not available",
                        "  Depth: not available",
                    ]
                )
            lines.append(
                f"  Accepted regions: {int(compression.get('accepted_regions', 0))}"
            )

        lines.extend(
            [
                "",
                "CLASSICAL",
                "  Status: " + _format_feasibility_status(classical),
            ]
        )
        classical_reason = _classical_infeasibility_reason(classical)
        if classical_reason is not None:
            lines.append(f"  Reason: {classical_reason}")
        classical_quality = str(classical.get("quality_status") or "not-assessed")
        results = _sequence_of_mappings(_mapping(classical.get("criteria")).get("results"))
        reference_only = bool(results) and all(
            _mapping(result.get("evidence")).get("source") == "quantum-io:exact-reference"
            for result in results
        )
        qualifier = " (relative to the simulated circuit reference)" if reference_only else ""
        lines.append(f"  Quality: {classical_quality}{qualifier}")
        quality_reason = (
            _quality_failure_reason(
                classical.get("criteria"), source="Available classical evidence"
            )
            if classical_quality == "violated"
            else None
        )
        if quality_reason is not None:
            lines.append(f"  Quality reason: {quality_reason}")
        lines.append(
            f"  Method: {classical.get('selected_method') or 'not selected'}"
        )
        preparation_warning = _preparation_claim_warning(evidence)
        if preparation_warning is not None:
            lines.append(f"  WARNING: {preparation_warning}")
        lines.extend(_format_classical_resource_criteria_lines(classical, evidence))
        lines.extend(_format_criteria_lines(
            classical.get("criteria"), qualify_reference=not reference_only,
        ))

        lines.extend(
            [
                "",
                "QUANTUM",
                "  Status: " + _format_feasibility_status(quantum),
                f"  Quality: {quantum.get('quality_status') or 'not-assessed'}",
            ]
        )
        quantum_reason = _quantum_infeasibility_reason(quantum)
        if quantum_reason is not None:
            lines.append(f"  Reason: {quantum_reason}")
        quality_reason = _quantum_quality_reason(quantum)
        if quality_reason is not None:
            lines.append(f"  Quality reason: {quality_reason}")
        lines.extend(
            [
                f"  Backend: {quantum.get('backend') or 'not selected'}",
                "  Assessment basis: "
                f"{quantum.get('assessment_basis') or 'not available'}",
            ]
        )

        noisy_completion = _latest_evidence(
            evidence,
            _QPU_NOISY_SIMULATION_COMPLETED,
            scopes=("quantum-model",),
        )
        if noisy_completion is not None and noisy_completion.get("value") is True:
            lines.append("  Noisy simulation performed: True")
        lines.append(
            f"  QPU execution performed: {bool(quantum.get('execution_performed', False))}"
        )

        available_resources = quantum.get("available_resources")
        if isinstance(available_resources, Mapping):
            if available_resources.get("median_two_qubit_error") is not None:
                lines.append(
                    "  Median 2Q error: "
                    f"{float(available_resources['median_two_qubit_error']):.4g}"
                )
            if available_resources.get("median_t2_sec") is not None:
                lines.append(
                    f"  Median T2: {float(available_resources['median_t2_sec']):.4g} s"
                )

        lines.extend(_format_quantum_resource_criteria_lines(quantum))
        quantum_basis = str(quantum.get("assessment_basis") or "")
        if quantum_basis.startswith("measured-qpu"):
            lines.extend(
                _format_criteria_lines(
                    quantum.get("criteria"),
                    heading="Quality criteria (measured QPU)",
                    include_all_missing=(
                        quantum.get("quality_status") == "inconclusive"
                        and bool(quantum.get("execution_performed", False))
                    ),
                )
            )
        elif quantum_basis.startswith("calibrated-model") or quantum_basis == "simulated-quantum":
            lines.extend(
                _format_criteria_lines(
                    quantum.get("model_criteria"),
                    heading=(
                        "Quality criteria (quantum simulator)"
                        if quantum_basis == "simulated-quantum"
                        else "Quality criteria (calibrated model)"
                    ),
                    include_all_missing=quantum.get("quality_status") == "inconclusive",
                )
            )

        rec = self.recommendation
        lines.extend(
            [
                "",
                "RECOMMENDATION",
                f"  {rec.compute.value}: {rec.reason}",
            ]
        )
        if self.next_evidence:
            lines.extend(["", "NEXT EVIDENCE"])
            for item in self.next_evidence:
                kind = item.get("kind", "unknown")
                reason = item.get("reason", "")
                lines.append(f"  - {kind}: {reason}")
        return "\n".join(lines)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
