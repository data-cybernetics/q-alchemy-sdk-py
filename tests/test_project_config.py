from __future__ import annotations

from pathlib import Path
import tomllib


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
