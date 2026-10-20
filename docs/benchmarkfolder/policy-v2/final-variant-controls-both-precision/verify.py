#!/usr/bin/env python3
"""Read-only verification of both attempts and every retained raw stream."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_workers as workers


def read(path):
    return json.loads(path.read_text())


def main():
    counts = {}
    for directory in (HERE.parent / "final-variant-controls-fixed", HERE):
        seal = read(directory / "SHA256SUMS.json")
        assert seal["count"] == len(seal["files"])
        for name, digest in seal["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
        streams = 0
        for path in sorted((directory / "data").glob("candidate-*-0.json")):
            record = read(path)
            raw = directory / "raw" / record["case"] / str(record["backend"])
            parsed = workers.parse_outputs(raw, record["backend"], 10, record["loops"])
            assert all(record[key] == value for key, value in parsed.items()), path
            streams += 1
        counts[directory.name] = {"files": seal["count"], "raw_streams": streams}
    assert counts[HERE.name]["raw_streams"] == 12
    assert counts["final-variant-controls-fixed"]["raw_streams"] == 8
    failed = read(HERE.parent / "final-variant-controls-fixed/data/agreement-control-rebinding.json")
    assert not failed["passed"] and len(failed["failures"]) == 18
    comparisons = sorted((HERE / "data").glob("agreement-*.json"))
    assert len(comparisons) == 8 and all(read(path)["passed"] for path in comparisons)
    retained = read(HERE / "data/candidate-control-rebinding-8-0.json")
    fresh = read(HERE / "data/candidate-control-rebinding-fresh-8-0.json")
    assert retained["rows"] == fresh["rows"]
    for backend, status in ((1, 329), (6, 328), (8, 322)):
        record = read(HERE / "data" / f"candidate-control-near-singular-{backend}-0.json")
        assert record["status_counts"] == record["timed_status_counts"] == {str(status): 10}
    print(json.dumps({"passed": True, "inventories": counts,
                      "preserved_failed_coefficients": len(failed["failures"]),
                      "successful_comparisons": len(comparisons),
                      "physics_evaluations_started": False}, indent=2))


if __name__ == "__main__":
    main()
