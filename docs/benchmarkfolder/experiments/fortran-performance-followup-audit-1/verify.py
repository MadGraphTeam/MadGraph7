#!/usr/bin/env python3
"""Verify this read-only audit; never import numerical code or start workers."""
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    index = json.loads((HERE / "index.json").read_text())
    for relative, expected in index["files"].items():
        path = HERE / relative
        assert path.is_file() and not path.is_symlink(), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert digest(path) == expected["sha256"], relative
    audit = json.loads((HERE / "audit.json").read_text())
    assert audit["native_execution"] is False
    assert audit["implementation"] is False
    assert audit["performance_acceptance"] is False
    for relative, expected in audit["inputs"].items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert digest(path) == expected["sha256"], relative
    profile = json.loads((ROOT / audit["profile_summary"]).read_text())
    assert profile["performance_eligible"] is False
    assert profile["all_profile_counts_expected"] is True
    for number, expected in audit["historical_profile_stage_medians_us"].items():
        actual = profile["stages"]["1:0"][number]
        assert actual["name"] == expected["name"]
        assert actual["raw"]["median"] == expected["raw_median_us"]
    counts = [sum((3+i)*(2+i)*(1+i)//6 for i in range(r+1)) for r in range(7)]
    assert counts == audit["rank_0_to_6_coefficient_counts"]
    print(json.dumps({"passed": True, "bound_inputs": len(audit["inputs"]),
                      "recommendations": len(audit["recommendations"]),
                      "native_execution": False, "measured_benefit": False}, sort_keys=True))


if __name__ == "__main__":
    main()
