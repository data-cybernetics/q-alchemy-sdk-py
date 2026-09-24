"""Refresh SDK fixtures in an environment containing the Feasibility core.

Run this script with the core's Python, not the SDK's environment: both expose
q_alchemy.feasibility under different APIs. No executions or cloud jobs occur.
The inputs are synthetic evidence; expected reports and text come from the core.
Use --check to detect drift without rewriting the fixture.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from q_alchemy.feasibility import (
    ClassicalResources, EvidenceBundle, FeasibilityAnalyzer, FeasibilityPolicy,
    QuantumResources, SolutionCriteria,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).parent / "data" / "feasibility_summary.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    fixture["core_version"] = version("q-alchemy-feasibility")
    for case in fixture["cases"]:
        inputs = case["inputs"]
        result = FeasibilityAnalyzer().analyze(
            criteria=SolutionCriteria.from_dict(inputs["criteria"]),
            evidence=EvidenceBundle.from_dict(inputs["evidence"]),
            classical_resources=ClassicalResources.from_dict(inputs["classical_resources"]),
            quantum_resources=(
                QuantumResources.from_dict(inputs["quantum_resources"])
                if inputs.get("quantum_resources") else None
            ),
            policy=FeasibilityPolicy.from_dict(inputs.get("policy", {})),
        )
        case["report"] = result.to_dict()
        case["summary"] = result.format_summary()
    text = json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    if args.check:
        if text != path.read_text(encoding="utf-8"):
            raise SystemExit("Feasibility fixtures differ; review the core contract before refreshing.")
    else:
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
