"""Protect compatibility between the SDK's example and integration dependencies."""
from pathlib import Path
import runpy
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version


def manifest():
    return tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())


def test_examples_can_use_bridge_compatible_aer_and_runtime():
    data = manifest()
    extras = data["project"]["optional-dependencies"]
    bridge = {Requirement(text).name for text in extras["pennylane"]}
    examples = {req.name: req for req in map(Requirement, extras["examples"])}
    dev = {
        Requirement(text).name
        for text in data["dependency-groups"]["dev"]
        if isinstance(text, str)
    }
    assert "pennylane-qiskit" in bridge
    assert "pennylane-qiskit" not in examples.keys() | dev
    assert "pennylane" in examples
    assert Version("0.17.2") in examples["qiskit-aer"].specifier
    assert Version("0.45.0") in examples["qiskit-ibm-runtime"].specifier
    assert not any(
        {"pennylane", "examples"} <= {item.get("extra") for item in group}
        for group in data["tool"]["uv"].get("conflicts", [])
    )


def test_backend_example_prints_setup_when_the_bridge_is_missing(monkeypatch, capsys):
    example = Path(__file__).parents[1] / "examples" / "qiskit_backend.py"
    demo = runpy.run_path(str(example))["pennylane_demo"]
    monkeypatch.setitem(demo.__globals__, "find_spec", lambda name: None)

    demo()

    output = capsys.readouterr().out
    assert "--extra examples --extra qiskit --extra pennylane" in output
    assert "python examples/qiskit_backend.py" in output


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
