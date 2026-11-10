"""Replay the closed two-case data using its retained Python protocols only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

CAMPAIGN = Path("docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1")
INDEX = CAMPAIGN / "n1-qualification-1/closure-index.json"
INDEX_SHA = "3888b1ca29d3e368c748c141462ea4286a8c88fd14ecc00a80928c24e8555dfa"
CASES = ("qcd_n1_uux_z", "ew_n1_udx_wp")
LANES = (("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root = Path(root).resolve()
    blob = (root / INDEX).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == INDEX_SHA, "Closed index changed")
    index = json.loads(blob)
    require(index["case_ids"] == list(CASES) and index["file_count"] == 218,
            "Unexpected closed numerical scope")
    require(index["performance_acceptance_eligible"] is False and
            index["full_matrix_complete"] is False, "Unexpected acceptance claim")
    for name, record in index["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Unsafe closed path")
        path = root / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                "Closed path escapes the supplied root")
        data = path.read_bytes()
        require(len(data) == record["bytes"] and
                hashlib.sha256(data).hexdigest() == record["sha256"],
                "Closed payload changed: " + name)

    def read(path):
        require(str(path) in index["files"], "Read outside the closed inventory")
        return json.loads((root / path).read_bytes())

    def load(name, path):
        require(str(path) in index["files"], "Protocol outside the closed inventory")
        module = types.ModuleType(name)
        module.__file__ = str(root / path)
        sys.modules[name] = module
        exec(compile((root / path).read_bytes(), module.__file__, "exec"), module.__dict__)
        return module

    names = ("tirust_benchmark_worker", "closed_n1_benchmark", "closed_n1_warmup")
    saved = {name: sys.modules.get(name) for name in names}
    original_path = sys.path[:]
    results = []
    comparisons = 0
    try:
        for case in CASES:
            directory = CAMPAIGN / "experiments/numerical" / ("full100-warmup-" + case + "-1")
            load(names[0], directory / "tirust_benchmark_worker.py")
            bench = load(names[1], directory / "tirust_benchmark.py")
            warmup = load(names[2], directory / "tirust_benchmark_warmup.py")
            aggregate = read(directory / "results.json")["results"]
            require([(v["source"], v["backend"]) for v in aggregate] == list(LANES),
                    "Closed lane order changed")
            reference = None
            for (source, backend), item in zip(LANES, aggregate):
                lane = directory / (source + "-" + str(backend))
                ready = read(lane / "raw/ready.json")
                observation = warmup.parse_warmup(root / lane / "raw", [backend], 100, ready)
                require(item == read(lane / "result.json") and item["status"] == "complete"
                        and item["observation"] == observation
                        and read(lane / "raw/observation.json") == observation,
                        "Raw observation differs from the retained result")
                if source == "baseline":
                    reference = observation
                comparison = bench.compare_rows(reference["rows"], observation["rows"], 3e-6)
                require(comparison == item["comparison"] and comparison["passed"]
                        and comparison["ordinary_target_passed"], "Numerical comparison failed")
                require(len({row["point"] for row in observation["rows"]}) == 100 and
                        observation["backend_precision_counts"] == {str(backend) + ":DP": 100},
                        "Saved-point or accepted-precision scope changed")
                comparisons += len(comparison["residuals"])
                results.append({"case": case, "source": source, "backend": backend,
                                "status_counts": observation["status_counts"]})
    finally:
        sys.path[:] = original_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return {"index_sha256": INDEX_SHA, "payloads_verified": len(index["files"]),
            "lanes_replayed": len(results), "coefficient_comparisons": comparisons,
            "accepted_point_calls": 1200, "all_selected_backend_dp": True,
            "regression_and_target_passed": True, "native_executions": 0,
            "full_matrix_complete": False, "performance_acceptance": False,
            "cases": list(CASES)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[5])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))
