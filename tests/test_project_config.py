from __future__ import annotations

from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_qiskit_is_not_redirected_to_a_private_index() -> None:
    metadata = _metadata()
    sources = metadata.get("tool", {}).get("uv", {}).get("sources", {})

    # With no source override, uv resolves qiskit from the default public PyPI
    # index. The SDK must not depend on the private Q-Alchemy Qiskit build.
    assert "qiskit" not in sources

    assert any(
        dependency.startswith("qiskit[") or dependency.startswith("qiskit>")
        for dependency in metadata["project"]["optional-dependencies"]["qiskit"]
    )

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    qiskit = next(package for package in lock["package"] if package["name"] == "qiskit")
    assert qiskit["source"] == {"registry": "https://pypi.org/simple"}


def test_pinexq_dependency_requires_supported_client_versions() -> None:
    dependency = next(
        Requirement(entry) for entry in _metadata()["project"]["dependencies"]
        if Requirement(entry).name == "pinexq-client"
    )
    assert "2.1.0" in dependency.specifier
    assert "2.0.0" not in dependency.specifier
    assert "3.0.0" not in dependency.specifier

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    client = next(package for package in lock["package"] if package["name"] == "pinexq-client")
    assert client["version"] in dependency.specifier
    assert client["source"] == {"registry": "https://pypi.org/simple"}
