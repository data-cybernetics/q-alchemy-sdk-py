"""Portable contracts for the hosted circuit compressor (wire schema 1).

The SDK transports core options without importing the compression engine or
duplicating its resource policy. An empty options object uses the server defaults,
including canonical-zero input semantics. Use equivalence='operator' for a
subroutine whose input state is arbitrary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .quantum_io_contract import Circuit, _json_value


def _require_envelope(data: Mapping[str, Any], kind: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{kind} must be a JSON object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"{kind}.schema_version must be the integer 1")
    if data.get("kind") != kind:
        raise ValueError(f"expected kind={kind!r}")


def _circuit_envelope(circuit: Any) -> dict[str, Any]:
    """Accept the existing SDK Circuit, a service envelope, or a native circuit."""
    if isinstance(circuit, Mapping):
        _require_envelope(circuit, "quantum-circuit")
        if circuit.get("format") != "qasm3":
            raise ValueError("circuit.format must be 'qasm3'")
        qasm = circuit.get("qasm")
        if not isinstance(qasm, str) or not qasm.strip():
            raise ValueError("circuit.qasm must be a non-empty string")
        return _json_value(circuit, path="circuit")
    if not isinstance(circuit, Circuit):
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise TypeError("circuit must be an SDK Circuit or a quantum-circuit envelope; native circuits require the qiskit extra") from None
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be an SDK Circuit, a quantum-circuit envelope, or a Qiskit QuantumCircuit")
        circuit = Circuit.from_qiskit(circuit)
    return {"schema_version": 1, "kind": "quantum-circuit", "format": "qasm3", "qasm": circuit.payload}


@dataclass(frozen=True)
class CircuitCompressionRequest:
    """Core option overrides; omitted options retain server-side defaults.

    Values must be portable JSON. Option names, ranges and resource policies are
    validated by the deployed compressor. No SDK-side simulation is performed.
    """

    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")
        object.__setattr__(self, "options", _json_value(self.options, path="options"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "kind": "circuit-compression-request",
                "options": _json_value(self.options, path="options")}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionRequest":
        _require_envelope(data, "circuit-compression-request")
        return cls(options=data.get("options", {}))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "CircuitCompressionRequest":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class CircuitCompressionReport:
    """Server-reported exactness, metrics and a portable compressed circuit.

    ``exact`` describes the declared ``equivalence`` and ``input_semantics``;
    reachable-subspace exactness does not certify arbitrary input states.
    Region operation indices refer to the server's normalized baseline circuit.
    Diagnostics in ``report`` are present only when requested with collect_report.
    Unknown fields are preserved for forward compatibility.
    """

    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_envelope(self.raw, "circuit-compression-report")
        _circuit_envelope(self.raw.get("circuit"))
        for name in ("changed", "exact"):
            if type(self.raw.get(name)) is not bool:
                raise ValueError(f"compression report {name} must be a bool")
        for name in ("equivalence", "input_semantics"):
            if not isinstance(self.raw.get(name), str) or not self.raw[name]:
                raise ValueError(f"compression report {name} must be a non-empty string")
        for name in ("metrics", "options"):
            if not isinstance(self.raw.get(name), Mapping):
                raise ValueError(f"compression report {name} must be an object")
        for stage in ("input", "baseline", "compressed"):
            if not isinstance(self.raw["metrics"].get(stage), Mapping):
                raise ValueError(f"compression metrics must include {stage}")
        regions = self.raw.get("regions")
        if not isinstance(regions, (list, tuple)) or any(not isinstance(region, Mapping) for region in regions):
            raise ValueError("compression report regions must be an array of objects")
        if self.raw.get("report") is not None and not isinstance(self.raw["report"], Mapping):
            raise ValueError("compression diagnostics must be an object or null")
        object.__setattr__(self, "raw", _json_value(self.raw, path="compression report"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CircuitCompressionReport":
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self.raw, path="compression report")

    @classmethod
    def from_json(cls, payload: str) -> "CircuitCompressionReport":
        return cls.from_dict(json.loads(payload))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @property
    def circuit(self) -> Circuit:
        return Circuit.qasm3(self.raw["circuit"]["qasm"])

    def to_qiskit(self) -> Any:
        """Reconstruct the output circuit; requires the SDK's qiskit extra."""
        return self.circuit.to_qiskit()

    @property
    def changed(self) -> bool:
        return self.raw["changed"]

    @property
    def exact(self) -> bool:
        return self.raw["exact"]

    @property
    def equivalence(self) -> str:
        return self.raw["equivalence"]

    @property
    def input_semantics(self) -> str:
        return self.raw["input_semantics"]

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.raw["metrics"]

    @property
    def options(self) -> Mapping[str, Any]:
        return self.raw["options"]

    @property
    def regions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["regions"])

    @property
    def report(self) -> Mapping[str, Any] | None:
        return self.raw.get("report")

    def format_summary(self) -> str:
        lines = ["CIRCUIT COMPRESSION", f"Changed: {self.changed}", f"Exact: {self.exact}",
                 f"Equivalence: {self.equivalence}", f"Input semantics: {self.input_semantics}",
                 "Metrics: input -> baseline -> compressed"]
        for label, key in (("1Q operations", "one_qubit_operations"),
                           ("2Q operations", "two_qubit_operations"), ("CX", "cx"),
                           ("Operations", "operations"), ("Depth", "depth"),
                           ("2Q depth", "two_qubit_depth")):
            values = [str(self.metrics[stage].get(key, "unknown")) for stage in ("input", "baseline", "compressed")]
            lines.append(f"{label}: " + " -> ".join(values))
        lines.append(f"Accepted regions: {len(self.regions)}")
        return "\n".join(lines)
