"""Opt-in SDK -> deployed PineXQ -> circuit-compressor integration tests.

These tests create real jobs and use no mocked transport or local ProCon. Set
Q_ALCHEMY_RUN_LIVE_CIRCUIT_COMPRESSION=1 and Q_ALCHEMY_API_KEY (or PINEXQ_API_KEY).
Successful jobs and SDK-created WorkData are removed; failed jobs are retained.
Q_ALCHEMY_CIRCUIT_COMPRESSION_STEP_VERSION optionally pins the deployed step.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from q_alchemy import (
    CircuitCompressionParams,
    CircuitCompressionRequest,
    CircuitCompressionService,
)


_LIVE_ENABLED = os.getenv("Q_ALCHEMY_RUN_LIVE_CIRCUIT_COMPRESSION", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_API_KEY_PRESENT = bool(os.getenv("Q_ALCHEMY_API_KEY") or os.getenv("PINEXQ_API_KEY"))

pytestmark = pytest.mark.skipif(
    not (_LIVE_ENABLED and _API_KEY_PRESENT),
    reason=(
        "live circuit compression requires Q_ALCHEMY_API_KEY/PINEXQ_API_KEY "
        "and Q_ALCHEMY_RUN_LIVE_CIRCUIT_COMPRESSION=1"
    ),
)


def _service() -> CircuitCompressionService:
    return CircuitCompressionService(CircuitCompressionParams(
        step_version=os.getenv("Q_ALCHEMY_CIRCUIT_COMPRESSION_STEP_VERSION") or None,
        job_completion_timeout_sec=300,
        job_tags=["sdk-live-circuit-compression"],
        remove_data=True,
    ))


def _run(circuit, request):
    with _service() as service:
        job = service.compress(circuit, request)
        report = job.result()
        # The second call uses the cached report and retries unfinished cleanup.
        assert job.result() is report
        assert job.removed, "PineXQ did not remove this test's job and SDK-owned WorkData"
        with pytest.raises(RuntimeError, match="removed"):
            _ = job.raw_job
    return report


def test_live_zero_input_compression_reduces_two_qubit_cost():
    """Reduce actual CX cost below both the input and the O0 baseline.

    This two-qubit pure-state preparation uses two CX gates; the compressor can
    prepare the same state with one. The gates are not adjacent cancelling CXs,
    and O0 keeps the baseline at two CXs, so baseline transpilation alone cannot
    satisfy the reduction assertion. Equivalence is omitted to test the service's
    canonical-zero default.
    """
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.37, 1)
    circuit.cx(0, 1)
    circuit.rz(0.19, 0)
    report = _run(circuit, CircuitCompressionRequest(options={
        "optimization_level": 0,
        "basis_gates": ["u", "cx"],
        "collect_report": True,
        # Allow cold worker startup on shared CI hosts. These are core options,
        # separate from the SDK's 300-second wait for the complete remote job.
        "synthesis_timeout_seconds": 15.0,
        "synthesis_budget_seconds": 45.0,
    }))
    compressed = report.to_qiskit()
    assert report.exact and report.changed
    assert report.equivalence == "reachable_subspace"
    assert report.input_semantics == "canonical_zero_state"
    assert report.options["optimization_level"] == 0
    assert report.regions
    assert report.report is not None
    assert report.report["accepted_regions"] > 0

    input_cx = circuit.count_ops().get("cx", 0)
    output_cx = compressed.count_ops().get("cx", 0)
    output_2q = sum(len(instruction.qubits) == 2 for instruction in compressed.data)
    assert report.metrics["input"]["cx"] == input_cx == 2
    assert report.metrics["baseline"]["cx"] == 2
    assert report.metrics["compressed"]["cx"] == output_cx < input_cx
    assert report.metrics["compressed"]["two_qubit_operations"] == output_2q
    assert output_2q < report.metrics["baseline"]["two_qubit_operations"]
    assert output_2q < report.metrics["input"]["two_qubit_operations"]
    np.testing.assert_allclose(
        Statevector(compressed).data, Statevector(circuit).data, atol=1e-8, rtol=0,
    )
    print(report.format_summary())


def test_live_operator_compression_preserves_phase_and_controlled_use():
    """Verify all inputs and global phase after the real service round trip."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    circuit = qiskit.QuantumCircuit(1, global_phase=0.37)
    circuit.x(0)
    circuit.z(0)
    circuit.x(0)
    circuit.z(0)
    report = _run(circuit, CircuitCompressionRequest(options={
        "equivalence": "operator", "collect_report": True,
    }))
    compressed = report.to_qiskit()
    assert report.exact
    assert report.equivalence == "operator"
    assert report.input_semantics == "all_inputs"
    assert report.options["equivalence"] == "operator"
    assert report.report is not None
    np.testing.assert_allclose(
        Operator(compressed).data, Operator(circuit).data, atol=1e-8, rtol=0,
    )
    # Operator.equiv would hide a lost global phase. Controlling the circuit
    # turns such a loss into an observable relative phase on the control qubit.
    np.testing.assert_allclose(
        Operator(compressed.to_gate().control()).data,
        Operator(circuit.to_gate().control()).data, atol=1e-8, rtol=0,
    )
    print(report.format_summary())
