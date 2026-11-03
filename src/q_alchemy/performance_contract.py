"""Portable observed wall-time diagnostics (schema 1).

Offsets share a trace origin; spans are inclusive and may overlap. Repeated
span IDs in embedded reports describe the same work. Estimated device times
and vendor-reported durations are deliberately outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class PerformanceSpan:
    span_id: str
    parent_span_id: str | None
    operation: str
    start_offset_seconds: float
    elapsed_seconds: float
    status: str = "ok"
    evidence_request_id: str | None = None
    evidence_request_kind: str | None = None
    result_ids: tuple[str, ...] = ()
    cache_hit: bool | None = None
    origin_trace_id: str | None = None
    origin_span_id: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        for name in ("start_offset_seconds", "elapsed_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite non-negative seconds")
        if not self.span_id or not self.operation:
            raise ValueError("span_id and operation must be non-empty")
        if self.status not in ("ok", "error", "running"):
            raise ValueError("unsupported performance span status")
        object.__setattr__(self, "result_ids", tuple(self.result_ids))

    def to_dict(self) -> dict[str, Any]:
        data = {name: getattr(self, name) for name in (
            "span_id", "parent_span_id", "operation", "start_offset_seconds",
            "elapsed_seconds", "status")}
        for name in ("evidence_request_id", "evidence_request_kind", "cache_hit",
                     "origin_trace_id", "origin_span_id", "error_type"):
            if getattr(self, name) is not None:
                data[name] = getattr(self, name)
        if self.result_ids:
            data["result_ids"] = list(self.result_ids)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerformanceSpan:
        return cls(**{name: (tuple(data[name]) if name == "result_ids" else data[name])
                      for name in cls.__dataclass_fields__ if name in data})


@dataclass(frozen=True)
class PerformanceDiagnostics:
    trace_id: str
    root_span_id: str | None
    spans: tuple[PerformanceSpan, ...] = ()
    complete: bool = True
    dropped_spans: int = 0
    clock: str = "perf_counter"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported performance schema_version")
        object.__setattr__(self, "spans", tuple(self.spans))
        if len({span.span_id for span in self.spans}) != len(self.spans):
            raise ValueError("performance span IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "trace_id": self.trace_id,
                "root_span_id": self.root_span_id, "clock": self.clock,
                "complete": self.complete, "dropped_spans": self.dropped_spans,
                "spans": [span.to_dict() for span in self.spans]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerformanceDiagnostics:
        return cls(trace_id=data["trace_id"], root_span_id=data.get("root_span_id"),
                   spans=tuple(PerformanceSpan.from_dict(item) for item in data.get("spans", ())),
                   complete=data.get("complete", True), dropped_spans=data.get("dropped_spans", 0),
                   clock=data.get("clock", "perf_counter"), schema_version=data.get("schema_version", 1))
