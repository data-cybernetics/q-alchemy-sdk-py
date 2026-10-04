"""Public SDK contract coverage; no private engine imports or network requests."""
import json
from pathlib import Path
import numpy as np
import pytest
from q_alchemy import (
    ExecutionPlan, ExperimentReport, MeasurementPlan, PauliObservable,
    QuantumExperiment, Runtime, SparseStateEstimate, State,
)
from tests.test_quantum_io import _report_payload


def test_compact_reconstruction_options_roundtrip():
    experiment = QuantumExperiment(
        State.sparse(num_qubits=72, indices=[(1 << 70) + 3], amplitudes=[1]),
        measurement_plan=MeasurementPlan.qtucker_reconstruction(block_size=4, rank=1),
    )
    plan = ExecutionPlan(
        preparation_method="iterative_tucker",
        preparation_options={"factors_size": 0, "max_fidelity_loss": 1e-3},
        acquisition=Runtime.qalchemy_sparse(),
        estimator=Runtime.qtucker(block_size=4, rank=1),
        estimation_output={"support": "indices", "indices": [(1 << 70) + 3], "target_fidelity": True},
    )
    assert QuantumExperiment.from_json(experiment.to_json()) == experiment
    assert ExecutionPlan.from_json(plan.to_json()) == plan
    assert plan.to_dict()["estimation_output"]["indices"] == [str((1 << 70) + 3)]
    assert experiment.to_dict()["measurement_plan"]["reconstruction"]["validation"] is False


def test_training_only_is_supported_and_generator_cannot_replace_explicit_data():
    observable = PauliObservable.pauli("Z", "Z")
    assert MeasurementPlan(training=(observable,)).validation == ()
    with pytest.raises(ValueError):
        MeasurementPlan(training=(observable,), reconstruction={"rank": 1})
    with pytest.raises(ValueError):
        MeasurementPlan(validation=(observable,))


def test_typed_sparse_report_preserves_probability_and_large_indices(tmp_path):
    sparse = SparseStateEstimate(72, ((1 << 70) + 3,), (.5j,), {
        "support_probability": .8, "returned_probability": .25, "renormalized": False,
    })
    payload = _report_payload()
    payload["report"]["estimate"] = {
        "estimator": "qtucker-state-estimation:fit_with_restarts",
        "statevector_materialized": False, "metadata": {},
        "sparse_state": sparse.to_dict(), "target_fidelity": .9, "target_infidelity": .1,
    }
    payload["report"]["reference_to_estimated_surrogate_fidelity"] = .95
    payload["report"]["reference_to_estimated_surrogate_infidelity"] = .05
    report = ExperimentReport.from_dict(payload)
    assert report.estimate.sparse_state == sparse
    assert report.reference_to_estimated_surrogate_fidelity == .95
    assert report.estimate.target_fidelity == .9
    assert ExperimentReport.from_json(report.to_json()).estimate == report.estimate
    assert "Target-to-estimated fidelity" in report.format_summary()
    sparse.save_npz(tmp_path / "estimate.npz")
    with np.load(tmp_path / "estimate.npz", allow_pickle=False) as saved:
        assert int(saved["indices"][0]) == (1 << 70) + 3
        assert json.loads(str(saved["metadata_json"]))["renormalized"] is False
        assert np.linalg.norm(saved["amplitudes"])**2 == .25


@pytest.mark.parametrize("index", [1.5, True, "-1", "1.5"])
def test_sparse_output_rejects_invalid_wire_indices(index):
    with pytest.raises(ValueError):
        SparseStateEstimate.from_dict({"num_qubits": 2, "indices": [index], "amplitudes": [[1, 0]]})


def test_defaults_keep_existing_wire_payloads():
    plan = ExecutionPlan().to_dict()
    assert "preparation_method" not in plan
    assert "estimation_output" not in plan
    assert "reconstruction" not in MeasurementPlan().to_dict()


def test_core_generated_report_roundtrips_and_summary_matches():
    fixture = json.loads((Path(__file__).parent / "data" /
                          "quantum_io_compact_reconstruction.json").read_text())
    report = ExperimentReport.from_dict(fixture["report"])
    assert report.to_dict() == fixture["report"]
    assert report.format_summary() == fixture["summary"]
