"""Protect separately resolvable modern and PennyLane bridge environments."""
from pathlib import Path
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version


def manifest():
    return tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())


def test_bridge_is_explicit_and_exclusive_with_modern_examples():
    data = manifest()
    extras = data["project"]["optional-dependencies"]
    bridge = {Requirement(text).name for text in extras["pennylane"]}
    modern = {Requirement(text).name for text in extras["examples"]}
    dev = {
        Requirement(text).name
        for text in data["dependency-groups"]["dev"]
        if isinstance(text, str)
    }
    assert "pennylane-qiskit" in bridge
    assert "pennylane-qiskit" not in modern | dev
    assert "pennylane" in modern
    assert any(
        {item["extra"] for item in group} == {"pennylane", "examples"}
        for group in data["tool"]["uv"]["conflicts"]
    )


def test_numpy_branches_preserve_python_311_and_update_modern_python():
    dependencies = [Requirement(text) for text in manifest()["project"]["dependencies"]]
    for python, version in [("3.11", "2.4.6"), ("3.14", "2.5.5")]:
        environment = default_environment()
        environment.update(python_version=python, python_full_version=python + ".0")
        selected = [
            req
            for req in dependencies
            if req.name == "numpy"
            and (req.marker is None or req.marker.evaluate(environment))
        ]
        assert len(selected) == 1
        assert Version(version) in selected[0].specifier


def test_qiskit_extras_allow_the_internal_patched_wheel():
    for group in manifest()["project"]["optional-dependencies"].values():
        for text in group:
            req = Requirement(text)
            if req.name == "qiskit":
                assert Version("2.6.1+qalchemy.3") in req.specifier
