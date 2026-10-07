from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_q_alchemy_discovers_sibling_distribution_in_source_tree(tmp_path: Path) -> None:
    """The SDK package must not hide subpackages from other Q-Alchemy distributions."""

    sibling = tmp_path / "q_alchemy" / "_namespace_probe"
    sibling.mkdir(parents=True)
    (sibling / "__init__.py").write_text("VALUE = 73\n", encoding="utf-8")

    project_src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import q_alchemy._namespace_probe as probe\n"
        "assert probe.VALUE == 73\n"
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    paths = [str(project_src), str(tmp_path)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    subprocess.run([sys.executable, "-c", code], env=env, check=True)
