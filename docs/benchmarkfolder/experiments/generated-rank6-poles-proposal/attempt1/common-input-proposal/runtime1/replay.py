#!/usr/bin/env python3
"""Verify this package and reparse frozen text; no numerical library is loaded."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = "docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def local(name):
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and ".." not in parts.parts, "Unsafe package path")
    result = HERE.joinpath(*parts.parts)
    require(all(not p.is_symlink() for p in [result, *result.parents] if p != HERE.parent),
            "Symlink in package path")
    require(result.resolve().is_relative_to(HERE), "Package path escaped root")
    return result


def replay():
    index = json.loads((HERE / "index.json").read_text())
    for name, item in index["files"].items():
        data = local(name).read_bytes()
        require(len(data) == item["bytes"] and sha(data) == item["sha256"], name)
        if name.endswith(".gz"):
            data = gzip.decompress(data)
        data.decode("utf8")
        require(b"\0" not in data, "Non-text payload: " + name)
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    require(actual == set(index["files"]) | {"index.json"}, "Unindexed package files")
    copies = json.loads(local("copies.json").read_text())
    raw = {}
    for name, item in copies.items():
        data = local(item["path"]).read_bytes()
        require(sha(data) == item["sha256"] and len(data) == item["bytes"], name)
        require(item["compression"] in ("none", "gzip"), "Unknown compression")
        if item["compression"] == "gzip":
            data = gzip.decompress(data)
        require(sha(data) == item["raw_sha256"] and len(data) == item["raw_bytes"], name)
        data.decode("utf8")
        require(b"\0" not in data, "Non-text runtime payload")
        raw[name] = data
    frozen = json.loads(local("frozen-sources.json").read_text())
    for name, item in frozen.items():
        data = local(item["path"]).read_bytes()
        require(sha(data) == item["sha256"] and len(data) == item["bytes"], name)
    source_index = json.loads(local("frozen/repo/" + SOURCE + "/source-index.json").read_text())
    for name, digest in source_index["files"].items():
        require(frozen[SOURCE + "/" + name]["sha256"] == digest, name)
    for name, digest in source_index["dependencies"].items():
        require(frozen[name]["sha256"] == digest, name)

    execution = json.loads(raw["execution.json"])
    require(execution["complete"] is True and execution["exit_code"] == 0,
            "Diagnostic did not complete")
    require(execution["progress"]["process_reaped"] is True, "Worker not reaped")
    require(execution["loader"]["exit_code"] == 0 and
            execution["loader"]["progress"]["process_reaped"] is True, "Loader failure")
    require(execution["progress"]["process_wall_seconds"] <= execution["worker_seconds"] == 900,
            "Worker bound")
    require(execution["loader"]["progress"]["process_wall_seconds"] <= 30, "Loader bound")
    require(execution["address_space_bytes"] == 4294967296 and execution["exclusive_cpu"] is False,
            "Resource metadata differs")
    require(execution["mapping"]["expected_runtime_sha256"] == execution["runtime_sha256"] ==
            execution["mapping"]["mapped_files_sha256_after_exit"], "Runtime identity differs")
    require(sha(raw["maps.txt"]) == execution["mapping"]["maps_sha256"], "Maps digest")
    require(raw["maps.txt"].decode() == execution["mapping"]["maps_text"], "Maps text")
    require(execution["source_index_sha256"] == frozen[SOURCE + "/source-index.json"]["sha256"],
            "Executed source identity differs")
    require(execution["build_sha256"] == frozen[SOURCE + "/build1/build.json"]["sha256"],
            "Executed build identity differs")
    for name, digest in {**execution["inputs"], **execution["outputs"]}.items():
        require(sha(raw[name]) == digest, "Execution payload differs: " + name)
    require(sha(raw["results.json"]) == execution["results_sha256"], "Results digest")
    with tempfile.TemporaryDirectory(prefix="tirust-common-input-text-") as scratch:
        directory = Path(scratch)
        for name in ("records.txt", "observer.txt"):
            (directory / name).write_bytes(raw[name])
        analyzer = local("frozen/repo/" + SOURCE + "/analyze.py")
        result = runpy.run_path(str(analyzer))["analyze"](directory)
    require(result == json.loads(raw["results.json"]), "Frozen analyzer replay differs")
    summary = json.loads(local("summary.json").read_text())
    require(result["cases"] == summary["tensor_rows"] == 30 and result["effective_inputs_exact"],
            "Common-input coverage differs")
    require(result["status"] == summary["status"] == "open_acceptance_items", "Qualification changed")
    require(dict(Counter(x["kind"] for x in result["open_items"])) == summary["open_item_counts"],
            "Open-item summary differs")
    require([x for x in result["checks"] if x["status"] == "mismatched"] ==
            summary["finite_comparison_mismatches"], "Mismatch summary differs")
    require(summary["physical_captures"] == 0 and summary["packaging_new_native_execution"] is False,
            "Unsupported physical qualification")
    return {"passed": True, "package_files": len(index["files"]), "runtime_files": len(copies),
            "tensor_rows": result["cases"], "result_status": result["status"],
            "effective_inputs_exact": result["effective_inputs_exact"],
            "open_items": len(result["open_items"]), "native_execution": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True))
