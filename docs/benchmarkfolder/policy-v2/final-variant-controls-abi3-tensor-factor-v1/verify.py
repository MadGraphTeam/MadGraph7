#!/usr/bin/env python3
"""Recompute retained variant results without building or evaluating amplitudes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def read(path):
    return json.loads(path.read_text())


def checks():
    prepared = policy.verified(read(HERE / "prepared.json"),
                              "final-variant-controls-both-precision-preparation-v1")
    policy.check_file(prepared["generation"])
    policy.check_file(prepared["control_driver"])
    for record in prepared["sources"].values():
        policy.check_file(record)
    specification = importlib.util.spec_from_file_location(
        "retained_variant_controls", HERE / "controls_driver.py")
    controls = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(controls)
    config = read(HERE / "control-variants.json")
    records = {}
    for path in sorted((HERE / "data").glob("candidate-*-0.json")):
        record = read(path)
        raw = HERE / "raw" / record["case"] / str(record["backend"])
        parsed = workers.parse_outputs(raw, record["backend"], 10, record["loops"])
        assert all(record[key] == value for key, value in parsed.items()), path
        assert record["executable_sha256"] == prepared["variants"][record["case"]]["executable_sha256"]
        records[record["case"], record["backend"]] = record
    assert len(records) == 12
    generated = {}
    original_dump, original_report = bench.dump, bench.REPORT

    def capture_dump(path, value):
        assert path.parent == HERE / "data" and path.name not in generated
        generated[path.name] = value
        assert value == read(path), path

    bench.dump, bench.REPORT = capture_dump, HERE
    try:
        for name in ("control-rotation", "control-rebinding", "control-near-singular"):
            selected = {f"candidate-{backend}": record
                        for (case, backend), record in records.items() if case == name}
            assert controls.compare_records(name, selected, config,
                                            require_rescue=name == "control-near-singular")
        retained = records["control-rebinding", 8]
        fresh = records["control-rebinding-fresh", 8]
        assert retained["rows"] == fresh["rows"]
        assert controls.compare_records("rebinding-context-reset",
                                        {"retained-8": retained, "fresh-8": fresh}, config)
        rotation = read(HERE / "rotation-reference.json")
        policy.check_file(rotation["binding"])
        for backend in (1, 6, 7, 8):
            reference = rotation["references"][str(backend)]
            original = read(policy.check_file(reference))
            assert controls.compare_records(f"rotation-invariance-{backend}",
                {"original": original, "rotated": records["control-rotation", backend]}, config)
    finally:
        bench.dump, bench.REPORT = original_dump, original_report
    assert len(generated) == 8
    for backend, code in ((1, 329), (6, 328), (8, 322)):
        record = records["control-near-singular", backend]
        assert record["status_counts"] == record["timed_status_counts"] == {str(code): 10}
    return {"passed": True, "raw_streams": len(records), "comparisons": len(generated),
            "coefficient_comparisons": sum(item["coefficient_comparisons"] for item in generated.values()),
            "maximum_tolerance_fraction": max(item["maximum_tolerance_fraction"] for item in generated.values()),
            "fresh_context_rows_exact": True, "physics_evaluations_started": False,
            "acceptance_evidence": False}


def main():
    seal = read(HERE / "SHA256SUMS.json")
    assert seal["count"] == len(seal["files"])
    for name, digest in seal["files"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    result = checks()
    assert result == read(HERE / "validation.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
