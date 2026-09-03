"""Dependency-free portable contract for the hosted Q-Alchemy feasibility service."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .quantum_io_contract import ExperimentReport
    from q_alchemy.visualization import ExperimentDiagram


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
    allow_noisy_simulation: bool = True
    dense_memory_utilization_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not 0 < self.dense_memory_utilization_fraction <= 1:
            raise ValueError("dense_memory_utilization_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "classical_first": self.classical_first,
            "quantum_execution": self.quantum_execution.value,
            "allow_noisy_simulation": self.allow_noisy_simulation,
            "dense_memory_utilization_fraction": self.dense_memory_utilization_fraction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeasibilityPolicy":
        return cls(
            classical_first=bool(data.get("classical_first", True)),
            quantum_execution=QuantumExecutionPolicy(
                str(data.get("quantum_execution", QuantumExecutionPolicy.WHEN_NEEDED.value))
            ),
            allow_noisy_simulation=bool(data.get("allow_noisy_simulation", True)),
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
class EvidenceCollectionConfig:
    provider: str = "ibm"
    backend: str | None = None
    least_busy: bool = False
    shots: int = 4096
    preparation_options: Mapping[str, Any] = field(default_factory=dict)
    sparse_config: Mapping[str, Any] = field(default_factory=lambda: {"sparse_epsilon": 0.0, "final_sparse_epsilon": 0.0})
    qtucker_config: Mapping[str, Any] = field(default_factory=dict)
    resource_sweep_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    attribution_noise_scales: tuple[float, ...] = (0.75, 0.5, 0.25, 0.125)
    attribution_shot_multipliers: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    attribution_dimensions: tuple[str, ...] = ("one-qubit-gate-error", "two-qubit-gate-error", "coherence", "readout-error", "shots")

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend.strip()
        ):
            raise ValueError("backend must be a non-empty string when provided")
        if self.backend is not None and self.least_busy:
            raise ValueError("select at most one backend strategy: backend or least_busy")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        _validate_decreasing_scales(
            self.resource_sweep_scales,
            allow_one=True,
            name="resource_sweep_scales",
        )
        _validate_decreasing_scales(
            self.attribution_noise_scales,
            allow_one=False,
            name="attribution_noise_scales",
        )
        previous = 1.0
        for raw in self.attribution_shot_multipliers:
            value = float(raw)
            if value <= previous:
                raise ValueError(
                    "attribution_shot_multipliers must be strictly increasing and > 1"
                )
            previous = value
        allowed = {
            "one-qubit-gate-error",
            "two-qubit-gate-error",
            "coherence",
            "readout-error",
            "shots",
        }
        if not self.attribution_dimensions:
            raise ValueError("attribution_dimensions must not be empty")
        if len(set(self.attribution_dimensions)) != len(self.attribution_dimensions):
            raise ValueError("attribution_dimensions must be unique")
        unknown = set(self.attribution_dimensions) - allowed
        if unknown:
            raise ValueError(
                f"unsupported attribution dimensions: {sorted(unknown)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "backend": self.backend,
            "least_busy": self.least_busy,
            "shots": self.shots,
            "preparation_options": dict(self.preparation_options),
            "sparse_config": dict(self.sparse_config),
            "qtucker_config": dict(self.qtucker_config),
            "resource_sweep_scales": list(self.resource_sweep_scales),
            "attribution_noise_scales": list(self.attribution_noise_scales),
            "attribution_shot_multipliers": list(self.attribution_shot_multipliers),
            "attribution_dimensions": list(self.attribution_dimensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceCollectionConfig":
        return cls(
            provider=str(data.get("provider", "ibm")),
            backend=(str(data["backend"]) if data.get("backend") is not None else None),
            least_busy=bool(data.get("least_busy", False)),
            shots=int(data.get("shots", 4096)),
            preparation_options=dict(data.get("preparation_options", {})),
            sparse_config=dict(data.get("sparse_config", {"sparse_epsilon": 0.0, "final_sparse_epsilon": 0.0})),
            qtucker_config=dict(data.get("qtucker_config", {})),
            resource_sweep_scales=tuple(float(v) for v in data.get("resource_sweep_scales", (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125))),
            attribution_noise_scales=tuple(float(v) for v in data.get("attribution_noise_scales", (0.75, 0.5, 0.25, 0.125))),
            attribution_shot_multipliers=tuple(float(v) for v in data.get("attribution_shot_multipliers", (2.0, 4.0, 8.0, 16.0))),
            attribution_dimensions=tuple(str(v) for v in data.get("attribution_dimensions", ("one-qubit-gate-error", "two-qubit-gate-error", "coherence", "readout-error", "shots"))),
        )


def _validate_decreasing_scales(
    values: tuple[float, ...], *, allow_one: bool, name: str
) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    previous: float | None = None
    for raw in values:
        value = float(raw)
        if value <= 0 or value > 1 or (not allow_one and value == 1):
            interval = "(0, 1]" if allow_one else "(0, 1)"
            raise ValueError(f"{name} values must be in {interval}")
        if previous is not None and value >= previous:
            raise ValueError(f"{name} must be strictly decreasing")
        previous = value


@dataclass(frozen=True)
class FeasibilityRequest:
    criteria: SolutionCriteria
    policy: FeasibilityPolicy = FeasibilityPolicy()
    evidence_collection: EvidenceCollectionConfig = EvidenceCollectionConfig()
    classical_resources: ClassicalResources | None = None
    max_steps: int = 16
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported feasibility request schema_version={self.schema_version}"
            )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

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
            max_steps=int(data.get("max_steps", 16)),
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
_DENSE_STATEVECTOR_MIN_MEMORY_BYTES = (
    "classical.dense_statevector.minimum_memory_bytes"
)
_QPU_NOISY_SIMULATION_ATTEMPTED = "quantum.model.execution_attempted"


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
) -> list[str]:
    evaluation_data = _mapping(evaluation)
    results = _sequence_of_mappings(evaluation_data.get("results"))
    if not results:
        return []

    # Match q-alchemy-feasibility: suppress an all-missing table when a branch
    # was decided entirely from static/resource evidence. Once one criterion has
    # evidence, keep missing companion criteria visible.
    if not any(isinstance(result.get("evidence"), Mapping) for result in results):
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
    if str(assessment.get("status")) != "infeasible":
        return None

    basis = str(assessment.get("assessment_basis") or "")
    if basis.startswith("measured-qpu"):
        reason = _quality_failure_reason(
            assessment.get("criteria"),
            source="Measured QPU evidence",
        )
        if reason is not None:
            return reason
    if basis.startswith("calibrated-model"):
        reason = _quality_failure_reason(
            assessment.get("model_criteria"),
            source="Backend-calibrated noisy simulation",
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
        if basis.startswith("calibrated-model"):
            return _quality_failure_reason(
                assessment.get("model_criteria"),
                source="Backend-calibrated noisy simulation",
            )

    if quality_status != "inconclusive":
        return None

    evaluation = (
        assessment.get("model_criteria")
        if basis.startswith("calibrated-model")
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
    if all(record.get("generated_at") is not None for record in records):
        return max(
            enumerate(records),
            key=lambda pair: (str(pair[1].get("generated_at")), pair[0]),
        )[1]
    return records[-1]


def _format_classical_resource_criteria_lines(
    assessment: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    criteria: list[str] = []
    available_resources = _mapping(assessment.get("available_resources"))
    available = available_resources.get("available_memory_bytes")
    requirement_value = assessment.get("required_resources")
    requirement = _mapping(requirement_value) if requirement_value is not None else None

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

    dense = _latest_evidence(
        evidence,
        _DENSE_STATEVECTOR_MIN_MEMORY_BYTES,
        scopes=("classical",),
    )
    dense_value = _numeric_value(dense.get("value")) if dense is not None else None
    if dense_value is not None:
        dense_bytes = int(dense_value)
        lower_bound = " (lower bound)" if bool(dense.get("lower_bound", False)) else ""
        if available is not None and dense_bytes > int(available):
            criteria.append(
                "Dense statevector memory: "
                f"{_format_bytes(dense_bytes)} minimum{lower_bound} > "
                f"{_format_bytes(available)} available [VIOLATED]"
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

    if not criteria:
        return []
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

        pairs = (
            ("max_one_qubit_error", "median_one_qubit_error", "Median 1Q error"),
            ("max_two_qubit_error", "median_two_qubit_error", "Median 2Q error"),
            ("max_readout_error", "median_readout_error", "Median readout error"),
        )
        for required_key, available_key, label in pairs:
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
    def quantum_circuit_payload(self) -> Mapping[str, Any] | None:
        """Portable logical experiment circuit returned by feasibility core."""

        value = self.raw.get("quantum_circuit")
        return value if isinstance(value, Mapping) else None

    @property
    def quantum_circuit(self) -> Any | None:
        """Return the produced logical experiment circuit as a Qiskit circuit.

        The service transports the exact logical ``P + U`` circuit as QASM 3.
        Qiskit performs the reconstruction; the SDK does not implement its own
        circuit parser or representation.
        """

        payload = self.quantum_circuit_payload
        if payload is None:
            return None
        if payload.get("kind") != "quantum-circuit":
            raise ValueError("unsupported quantum circuit payload kind")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported quantum circuit payload schema")
        if payload.get("role") != "logical-experiment-circuit":
            raise ValueError("unsupported quantum circuit role")
        if payload.get("format") != "qasm3":
            raise ValueError("unsupported quantum circuit format")
        qasm = payload.get("qasm")
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

    def draw_quantum_circuit(self, *, output: str = "text", **kwargs: Any) -> Any:
        """Draw the produced logical experiment circuit with Qiskit."""

        circuit = self.quantum_circuit
        if circuit is None:
            raise ValueError("feasibility report does not contain a quantum_circuit")
        return circuit.draw(output=output, **kwargs)

    @property
    def experiment_diagram_payload(self) -> Mapping[str, Any] | None:
        """Renderer-neutral diagram payload returned by feasibility core."""

        value = self.raw.get("experiment_diagram")
        return value if isinstance(value, Mapping) else None

    @property
    def experiment_diagram(self) -> "ExperimentDiagram | None":
        """Deserialize the server-provided diagram with q-alchemy-visualization.

        The SDK deliberately does not reconstruct feasibility steps or statuses.
        Install the optional ``visualization`` extra to obtain the shared diagram
        model and renderers.
        """

        payload = self.experiment_diagram_payload
        if payload is None:
            return None
        try:
            from q_alchemy.visualization import ExperimentDiagram
        except ImportError as exc:
            raise RuntimeError(
                "Experiment diagram deserialization requires q-alchemy-visualization. "
                "Install the SDK visualization extra and configure the Q-Alchemy "
                "package index."
            ) from exc
        return ExperimentDiagram.from_dict(payload)

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

    def draw(self, *, output: str = "text", show_title: bool = False) -> Any:
        """Render the experiment diagram returned by the feasibility service."""

        diagram = self.experiment_diagram
        if diagram is None:
            raise ValueError("feasibility report does not contain an experiment_diagram")
        return diagram.draw(output=output, show_title=show_title)

    def format_summary(self) -> str:
        """Format the report using q-alchemy-feasibility's canonical summary."""

        classical = self.classical
        quantum = self.quantum
        evidence = self.evidence
        lines = ["COMPUTE FEASIBILITY", "-------------------"]

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
        lines.append(
            f"  Method: {classical.get('selected_method') or 'not selected'}"
        )
        lines.extend(_format_classical_resource_criteria_lines(classical, evidence))
        lines.extend(_format_criteria_lines(classical.get("criteria")))

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

        noisy_attempt = _latest_evidence(
            evidence,
            _QPU_NOISY_SIMULATION_ATTEMPTED,
            scopes=("quantum-model",),
        )
        if noisy_attempt is not None and noisy_attempt.get("value") is True:
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

        resource_gap = quantum.get("resource_gap")
        if isinstance(resource_gap, Mapping):
            factor = resource_gap.get(
                "required_balanced_quantum_noise_improvement_factor"
            )
            if factor is not None:
                prefix = (
                    "> "
                    if bool(
                        resource_gap.get(
                            "balanced_quantum_noise_improvement_is_lower_bound",
                            False,
                        )
                    )
                    else ""
                )
                lines.append(
                    "  Required balanced quantum-noise improvement: "
                    f"{prefix}{float(factor):.3g}x"
                )

        attribution = quantum.get("resource_attribution")
        if isinstance(attribution, Mapping):
            remedies = attribution.get("single_resource_remedies")
            if isinstance(remedies, (list, tuple)) and remedies:
                lines.append(
                    "  Single-resource remedies: "
                    + ", ".join(str(item) for item in remedies)
                )
            if attribution.get("dominant_dimension") is not None:
                lines.append(
                    "  Dominant demonstrated limitation: "
                    f"{attribution['dominant_dimension']}"
                )

        lines.extend(_format_quantum_resource_criteria_lines(quantum))
        quantum_basis = str(quantum.get("assessment_basis") or "")
        if quantum_basis.startswith("measured-qpu"):
            lines.extend(
                _format_criteria_lines(
                    quantum.get("criteria"),
                    heading="Quality criteria (measured QPU)",
                )
            )
        elif quantum_basis.startswith("calibrated-model"):
            lines.extend(
                _format_criteria_lines(
                    quantum.get("model_criteria"),
                    heading="Quality criteria (calibrated model)",
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
