#!/usr/bin/env python3
"""Verify and replay archived CMS text with frozen analyzers; no native calls."""
from collections import Counter
from decimal import localcontext
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

HERE = Path(__file__).resolve().parent
ATTEMPT = Path("docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1")
CMS = ATTEMPT / "cms-width1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_text(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def safe_path(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe path: " + name)
    return path


def replay():
    index = json_text((HERE / "index.json").read_text())
    require(index["acceptance"] is False and index["native_execution"] is False,
            "Package scope changed")
    listed = set(index["files"])
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()}
    require(actual == listed | {"index.json"}, "Package file coverage differs")
    require(not any(p.is_symlink() for p in HERE.rglob("*")), "Package contains a symlink")
    for name, item in index["files"].items():
        data = (HERE / safe_path(name)).read_bytes()
        require(len(data) == item["bytes"] and sha(data) == item["sha256"], name)
    copies = json_text((HERE / "copies.json").read_text())
    raw = {}
    for name, item in copies.items():
        require(name == item["path"] and name in listed, "Unindexed copy")
        data = (HERE / safe_path(name)).read_bytes()
        require(len(data) == item["bytes"] and sha(data) == item["sha256"], name)
        require(item["compression"] in ("none", "gzip"), "Unknown compression")
        if item["compression"] == "gzip":
            data = gzip.decompress(data)
        require(len(data) == item["raw_bytes"] and sha(data) == item["raw_sha256"], name)
        data.decode("utf8")
        require(b"\0" not in data and not data.startswith(b"\x7fELF"), "Non-text copy")
        raw[name] = data

    execution = json_text(raw["data/run1/execution.json"])
    require(execution["complete"] is True and execution["exit_code"] == 0,
            "Recorded protocol did not complete")
    require(execution["loader"]["exit_code"] == 0, "Recorded loader check failed")
    for item, limit in ((execution, 900), (execution["loader"], 30)):
        require(item["progress"]["process_reaped"] is True, "Recorded process not reaped")
        require(item["progress"]["process_wall_seconds"] <= limit, "Recorded deadline exceeded")
    require(execution["cpu"] == 12 and execution["exclusive_cpu"] is False,
            "Recorded CPU policy differs")
    require(execution["address_space_bytes"] == 4294967296, "Recorded memory limit differs")
    mapping = execution["mapping"]
    require(mapping["cpu_allowed_list"] == "12", "Recorded CPU mapping differs")
    require(mapping["expected_runtime_sha256"] == mapping["mapped_files_sha256_after_exit"]
            == execution["runtime_sha256"], "Recorded library hashes differ")
    require(sha(raw["data/run1/maps.txt.gz"]) == mapping["maps_sha256"], "Maps hash differs")
    require(raw["data/run1/maps.txt.gz"].decode("utf8") == mapping["maps_text"], "Maps text differs")
    for group in ("inputs", "outputs"):
        for name, digest in execution[group].items():
            key = "data/run1/" + name + ("" if name.endswith(".json") else ".gz")
            require(sha(raw[key]) == digest, "Recorded " + group + " hash differs: " + name)
    require(sha(raw["data/run1/results.json"]) == execution["results_sha256"], "Result hash differs")
    prefix = "frozen/repo/" + CMS.as_posix() + "/"
    require(sha(raw[prefix + "source-index.json"]) == execution["source_index_sha256"],
            "Source index differs")
    require(sha(raw[prefix + "build1/build.json"]) == execution["build_sha256"],
            "Build record differs")
    source_index = json_text(raw[prefix + "source-index.json"])
    for name, digest in source_index["files"].items():
        require(sha(raw[prefix + name]) == digest, "Frozen source differs: " + name)
    for name, digest in source_index["dependencies"].items():
        require(sha(raw["frozen/repo/" + name]) == digest, "Frozen dependency differs: " + name)

    # Materialize only validated UTF-8 text. The original analyzer's relative
    # paths resolve inside this temporary tree; no ignored checkout is needed.
    with tempfile.TemporaryDirectory(prefix="tirust-cms-text-replay-") as temporary:
        root = Path(temporary)
        for name, data in raw.items():
            if name.startswith("frozen/repo/"):
                target = root / safe_path(name.removeprefix("frozen/repo/"))
            elif name.startswith("data/run1/"):
                suffix = name.removeprefix("data/run1/")
                if copies[name]["compression"] == "gzip":
                    suffix = suffix.removesuffix(".gz")
                target = root / ".tirust-build/generated-rank6-poles-1/cms-width1/run1" / safe_path(suffix)
            elif name.startswith("data/controls/"):
                suffix = name.removeprefix("data/controls/").removesuffix(".gz")
                target = root / ".tirust-build/generated-rank6-poles-1" / safe_path(suffix)
            else:
                raise ValueError("Unexpected copy group: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        analyzer = runpy.run_path(str(root / CMS / "analyze.py"))["analyze"]
        result = analyzer()
        require(result == json_text(raw["data/run1/results.json"]), "Frozen analyzer result differs")
        base = runpy.run_path(str(root / ATTEMPT / "analyze.py"))
        schedule = json_text((root / CMS / "schedule.json").read_text())
        work = root / ".tirust-build/generated-rank6-poles-1/cms-width1/run1"
        with localcontext() as context:
            context.prec = 100
            captures, cases = base["parse_records"](work / "records.txt", "ew", schedule)
            observer = base["parse_observer"](work / "observer.txt", [r["action"] for r in captures + cases])
        fixture = json_text((root / CMS / "fixture.json").read_text())
        expected = fixture["mass_squared_promoted_binary128"]
        require(len(cases) == 15 and len(captures) == 2, "New row coverage differs")
        require([r["action"] for r in cases] == list(range(31, 46)), "New actions differ")
        require(all(r["fixture"] == 105 and r["entry"]["m"][2] == expected for r in cases),
                "Positive-width fixture differs")
        require(base["binary128"](expected[2:]) < 0, "Width is not positive")
        require(all(not r["stable"] and observer[r["action"]]["status"] == 1 for r in cases[:6]),
                "DP refusal cohort differs")
        require(all(r["stable"] and observer[r["action"]]["status"] == 0 for r in cases[6:12]),
                "QP success cohort differs")
        require(all(r["stable"] and observer[r["action"]]["present"] == 0 for r in cases[12:]),
                "COLLIER cohort differs")
        require(all(r["state"][0] & 13 or r["state"][2] & 13 for r in cases),
                "Retained IEEE flags differ")

    summary = json_text((HERE / "summary.json").read_text())
    require(summary["combined_status"] == result["status"] == "open_acceptance_items",
            "Open acceptance status changed")
    require(summary["acceptance"] is False and summary["physical_amplitude_qualification"] is False
            and summary["performance_qualification"] is False, "Qualification scope changed")
    open_counts = dict(Counter(item["kind"] for item in result["open_items"]))
    check_counts = dict(Counter(item["status"] for item in result["checks"]))
    require(summary["combined_open_items"] == open_counts, "Open-item counts differ")
    require(summary["combined_check_statuses"] == check_counts, "Comparison counts differ")
    require(summary["new_rows"] == len(cases) and summary["fresh_captures"] == len(captures)
            and summary["cms_stable_rows"] == sum(row["stable"] for row in cases)
            and summary["cms_refused_rows"] == sum(not row["stable"] for row in cases),
            "New-row summary differs")
    require(summary["cms_tirust_dp_refusals"] == 6 and summary["cms_tirust_qp_successes"] == 6
            and summary["cms_collier_stable_rows"] == 3 and summary["cms_ieee_flag_rows"] == 15,
            "New-cohort summary differs")
    return {"passed": True, "scope": "offline UTF-8 hash verification and frozen analyzer replay only",
            "files": len(listed), "copies": len(copies), "cms_rows": len(cases),
            "cms_stable_rows": sum(row["stable"] for row in cases),
            "combined_open_items": len(result["open_items"]), "result_status": result["status"],
            "acceptance": False, "native_execution": False}


if __name__ == "__main__":
    print(json.dumps(replay(), sort_keys=True))
