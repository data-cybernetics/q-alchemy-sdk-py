"""Portable exact full-circuit compression settings and diagnostics.

Compression preserves the implemented P+U circuit from canonical |0...0>.
It does not certify preparation against its requested target or predict noisy
execution quality. Applied means the compressor output was selected, including
unchanged outputs. A disabled stage is represented by no compression summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


def _validated_options(options: Mapping[str, Any]) -> None:
    try:
        json.dumps(options, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("compression options must be finite JSON values") from exc


@dataclass(frozen=True)
class CircuitCompressionConfig:
    """Exact compression settings for the complete logical experiment circuit.

    Compression is deliberately applied to the full measurement-free ``P + U``
    computation, not independently to state preparation or evolution.  The full
    experiment starts from canonical ``|0...0>``, so Quantum I/O always uses the
    compressor's exact ``reachable_subspace`` semantics.

    ``options`` contains ordinary :class:`CircuitCompressorOptions` keyword
    overrides except ``equivalence`` and ``collect_report``, which Quantum I/O
    owns in order to preserve the execution contract and report diagnostics.
    """

    enabled: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("circuit_compression.enabled must be a bool")
        if not isinstance(self.options, Mapping):
            raise ValueError("circuit_compression.options must be a mapping")
        normalized = dict(self.options)
        _validated_options(normalized)
        if "equivalence" in normalized:
            if normalized["equivalence"] != "reachable_subspace":
                raise ValueError(
                    "Quantum I/O full-circuit compression requires "
                    "equivalence='reachable_subspace'"
                )
            normalized.pop("equivalence")
        if "collect_report" in normalized:
            if normalized["collect_report"] is not True:
                raise ValueError(
                    "Quantum I/O owns collect_report and requires it to be true"
                )
            normalized.pop("collect_report")
        object.__setattr__(self, "options", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionConfig":
        if not isinstance(data, Mapping):
            raise ValueError("circuit_compression must be a mapping")
        return cls(
            enabled=data.get("enabled", False),
            options=data.get("options", {}),
        )


@dataclass(frozen=True)
class CircuitCompressionMetrics:
    """Portable structural circuit-cost metrics from one compression stage."""

    operations: int
    one_qubit_operations: int
    two_qubit_operations: int
    cx: int
    depth: int
    two_qubit_depth: int
    counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionMetrics":
        return cls(**{key: data[key] for key in (
            "operations", "one_qubit_operations", "two_qubit_operations",
            "cx", "depth", "two_qubit_depth",
        )}, counts=dict(data.get("counts", {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": self.operations,
            "one_qubit_operations": self.one_qubit_operations,
            "two_qubit_operations": self.two_qubit_operations,
            "cx": self.cx,
            "depth": self.depth,
            "two_qubit_depth": self.two_qubit_depth,
            "counts": {str(k): int(v) for k, v in self.counts.items()},
        }


@dataclass(frozen=True)
class CircuitCompressionSummary:
    """Portable record of optional full-circuit compression for one assessment."""

    attempted: bool
    applied: bool
    changed: bool | None = None
    exact: bool | None = None
    equivalence: str | None = None
    input_semantics: str | None = None
    input_metrics: CircuitCompressionMetrics | None = None
    compressed_metrics: CircuitCompressionMetrics | None = None
    accepted_regions: int = 0
    reason: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionSummary":
        values = {key: data[key] for key in (
            "attempted", "applied", "changed", "exact", "equivalence",
            "input_semantics", "accepted_regions", "reason", "options",
        ) if key in data}
        for key in ("input_metrics", "compressed_metrics"):
            values[key] = (CircuitCompressionMetrics.from_dict(data[key])
                           if data.get(key) is not None else None)
        return cls(**values)

    def format_summary(self) -> str:
        """The same compression presentation used by Feasibility."""
        lines = ["CIRCUIT COMPRESSION", f"  Attempted: {self.attempted}",
                 f"  Applied: {self.applied}",
                 f"  Changed: {self.changed if self.changed is not None else 'not available'}",
                 f"  Circuit used: {self.circuit_used}",
                 f"  Result: {self.reason or 'not available'}",
                 f"  Equivalence: {self.equivalence or 'not available'}"]
        if self.input_metrics is not None:
            before = self.input_metrics
            after = self.compressed_metrics or before
            lines.extend([f"  1Q operations: {before.one_qubit_operations} -> {after.one_qubit_operations}",
                          f"  2Q operations: {before.two_qubit_operations} -> {after.two_qubit_operations}",
                          f"  Depth: {before.depth} -> {after.depth}"])
        else:
            lines.extend(["  1Q operations: not available", "  2Q operations: not available", "  Depth: not available"])
        lines.append(f"  Accepted regions: {self.accepted_regions}")
        return "\n".join(lines)

    @property
    def circuit_used(self) -> str:
        if not self.attempted:
            return "not-executed"
        return "compressor-output" if self.applied else "original"

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "applied": self.applied,
            "changed": self.changed,
            "circuit_used": self.circuit_used,
            "exact": self.exact,
            "equivalence": self.equivalence,
            "input_semantics": self.input_semantics,
            "input_metrics": (
                self.input_metrics.to_dict() if self.input_metrics is not None else None
            ),
            "compressed_metrics": (
                self.compressed_metrics.to_dict()
                if self.compressed_metrics is not None
                else None
            ),
            "accepted_regions": self.accepted_regions,
            "reason": self.reason,
            "options": dict(self.options),
        }


