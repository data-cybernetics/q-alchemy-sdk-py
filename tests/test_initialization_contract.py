from __future__ import annotations

from dataclasses import fields
import json

import pytest

from q_alchemy.initialize import (
    InitializationMethods,
    OptParams,
    create_processing_input,
)


def test_initialization_methods_match_hosted_procon() -> None:
    assert {method.value for method in InitializationMethods} == {
        "auto",
        "hierarchical_tucker",
        "iterative_tucker",
    }


def test_removed_synthesis_scheme_options_are_not_part_of_opt_params() -> None:
    names = {item.name for item in fields(OptParams)}
    assert "isometry_scheme" not in names
    assert "unitary_scheme" not in names

    with pytest.raises(TypeError):
        OptParams(isometry_scheme="ccd")
    with pytest.raises(TypeError):
        OptParams(unitary_scheme="qsd")


def test_auto_keeps_basis_gates_as_first_class_procon_parameter() -> None:
    options = OptParams(
        basis_gates=["rz", "sx", "x", "ecr"],
        initialization_method=InitializationMethods.AUTO,
        extra_kwargs={"cost_function": "two_qubit_then_depth"},
    )

    processing_name, parameters = create_processing_input(options, "encoded-state")

    assert processing_name == "build_initialization_circuit_inline"
    assert parameters["basis_gates"] == ["rz", "sx", "x", "ecr"]
    assert parameters["options"]["method"] == InitializationMethods.AUTO
    assert json.loads(parameters["options"]["opt_params"]) == {
        "cost_function": "two_qubit_then_depth"
    }
