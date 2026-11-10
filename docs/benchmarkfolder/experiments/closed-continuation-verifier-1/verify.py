#!/usr/bin/env python3
"""Replay an explicitly pinned CLOSED checkpoint using only retained files."""
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def contained(root, name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name
            and ".." not in path.parts, "Noncanonical relative path")
    target = (root / name).resolve()
    require(target.is_relative_to(root), "Path escapes supplied repository")
    return target


def extract(text, names, namespace):
    """Execute only selected pure definitions from a hash-pinned source."""
    selected = []
    for node in ast.parse(text).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name in names:
            selected.append(node)
    require(len(selected) == len(names), "Missing/duplicated parser definition")
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 "<pinned pure definitions>", "exec"), namespace)


def verify(repo_root, checkpoint_path, checkpoint_sha256):
    """Integrity failures raise ValueError; numerical misses remain outcomes."""
    root = Path(repo_root).resolve()
    bindings = json.loads((HERE / "bindings.json").read_text())
    require(isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
            and all(c in "0123456789abcdef" for c in checkpoint_sha256),
            "Expected an explicit lowercase SHA-256 from the caller's admission list")
    checkpoint_bytes = contained(root, checkpoint_path).read_bytes()
    require(sha(checkpoint_bytes) == checkpoint_sha256, "Checkpoint hash mismatch")
    checkpoint = json.loads(checkpoint_bytes)
    require(checkpoint["schema"] == "madloop-tirust-one-case-closure-v1",
            "Unexpected checkpoint schema")
    require(checkpoint["campaign"] == bindings["campaign"], "Wrong campaign")
    require(checkpoint["performance_acceptance_eligible"] is False,
            "Unexpected performance scope")
    files = checkpoint["files"]
    blobs = {}
    for name, record in files.items():
        data = contained(root, name).read_bytes()
        require(len(data) == record["bytes"] and sha(data) == record["sha256"],
                "Payload hash/size mismatch: " + name)
        blobs[name] = data
    require(len(files) == checkpoint["file_count"]
            and sum(map(len, blobs.values())) == checkpoint["total_bytes"],
            "Checkpoint count/size mismatch")

    def data(name):
        require(name in blobs, "Read outside checkpoint allowlist: " + name)
        return blobs[name]

    def load(name):
        return json.loads(data(name))

    def digest(name):
        return sha(data(name))

    class RetainedPath:
        """The copied parser sees only immutable, already verified raw bytes."""
        def __init__(self, name):
            self.name = name.name if isinstance(name, RetainedPath) else str(name)

        def __truediv__(self, name):
            return RetainedPath(self.name + "/" + name)

        def is_file(self):
            return self.name in blobs

        def open(self, encoding="utf-8", newline=None):
            require(encoding == "utf-8" and newline == ""
                    and self.name.endswith(("/raw/warmup.dat", "/raw/dimensions.dat")),
                    "Unexpected pure parser file access")
            return io.StringIO(data(self.name).decode(encoding), newline=newline)

    campaign = bindings["campaign_path"]
    case = checkpoint["case"]["id"]
    require(checkpoint_path == campaign + "/remaining-qualification-1/" + case
            + "/checkpoint.json", "Case/path mismatch")
    prefix = campaign + "/experiments/numerical/full100-warmup-" + case + "-1"
    closure = campaign + "/remaining-qualification-1/" + case
    specification = load(prefix + "/specification.json")
    require(specification["case"] == checkpoint["case"], "Case identity mismatch")
    require(specification["measurement_phase"] == "steady-warmup"
            and specification["performance_eligible"] is False,
            "Unexpected measurement scope")
    require(checkpoint["manifest_sha256"] == bindings["manifest_sha256"]
            == digest(prefix + "/campaign-manifest.json"), "Manifest mismatch")
    require(checkpoint["source_index_sha256"] == bindings["source_index_sha256"]
            == digest(campaign + "/remaining-qualification-1/source-index.json"),
            "Controller source index mismatch")
    require(specification["configuration_sha256"] == digest(prefix + "/cases.json")
            == bindings["configuration_sha256"], "Configuration mismatch")
    require(specification["protocols"] == bindings["protocols"], "Parser binding mismatch")
    for name, expected in bindings["protocols"].items():
        require(digest(prefix + "/" + name) == expected, "Protocol hash mismatch")

    worker = {}
    extract(data(prefix + "/tirust_benchmark_worker.py").decode(),
            {"BACKEND_CODES", "MeasurementError", "validate_status"}, worker)
    parser = {"math": math, "Path": RetainedPath,
              "worker": SimpleNamespace(**worker),
              "MeasurementError": worker["MeasurementError"]}
    extract(data(prefix + "/tirust_benchmark_warmup.py").decode(),
            {"_call_groups", "parse_warmup"}, parser)
    comparator = {}
    extract(data(prefix + "/tirust_benchmark.py").decode(), {"compare_rows"}, comparator)
    reference, lanes, settings, runtimes = None, [], [], {}
    norm = lambda value: {key: item for key, item in value.items() if key != "ldd"}
    for source, backend in [("baseline", 1), *[("candidate", b) for b in (1, 6, 7, 8, 9)]]:
        lane = prefix + f"/{source}-{backend}"
        for name in ("warmup.dat", "dimensions.dat", "ready.json"):
            data(lane + "/raw/" + name)
        observation = parser["parse_warmup"](
            lane + "/raw", backend, 100, load(lane + "/raw/ready.json"))
        result = load(lane + "/result.json")
        require(observation == load(lane + "/raw/observation.json")
                == result["observation"], "Retained parser output mismatch")
        require(result["status"] == "complete" and result["performance_eligible"] is False,
                "Incomplete/unexpected lane")
        if reference is None:
            reference = observation["rows"]
        comparison = comparator["compare_rows"](reference, observation["rows"], 3e-6)
        require(comparison == result["comparison"], "Retained comparison mismatch")
        manual = []
        for left, right in zip(reference, observation["rows"]):
            scale = max(abs(left["coefficients"][1]), abs(right["coefficients"][1]), 1e-99)
            for column, (a, b) in enumerate(zip(left["coefficients"], right["coefficients"])):
                magnitude = max(abs(a), abs(b))
                symmetric = abs(a - b) / abs(a + b) if a + b else 0. if not magnitude else math.inf
                pole = magnitude / scale if column >= 2 else math.inf
                manual.append((column, symmetric, pole))
        regression = all(s < 3e-6 or p < 3e-6 for _, s, p in manual)
        target = all(s < 1e-8 or p < 1e-8 for _, s, p in manual)
        require((regression, target) == (comparison["passed"], comparison["ordinary_target_passed"]),
                "Independent arithmetic/comparator disagreement")
        before, after = load(lane + f"/{source}-inputs.json"), load(lane + "/inputs-after.json")
        compilation = load(campaign + f"/inputs/{source}/{case}/compilation.json")
        generation = load(campaign + f"/inputs/{source}/{case}/generation.json")
        require(before["compilation"] == compilation and before["generation"] == generation,
                "Build/generation record mismatch")
        require(norm(before["runtime"]) == norm(after["runtime"]), "Runtime file identity changed")
        require(compilation["link_inputs_before"] == compilation["link_inputs_after"]
                == {key: before["runtime"][key] for key in ("archives", "metadata")},
                "Static dependency identity mismatch")
        require(before["numerical_cards"] == after["numerical_cards"]
                and len(set(before["numerical_cards"].values())) == 1,
                "Numerical card identity changed")
        require(before["points_sha256"] == after["points_sha256"]
                == digest(campaign + f"/points/{case}.dat")
                and before["points_metadata_sha256"] == digest(campaign + f"/points/{case}.json")
                and int(data(campaign + f"/points/{case}.dat").splitlines()[0]) == 100,
                "Saved point identity/count mismatch")
        require(before["param_card_sha256"] == digest(campaign + f"/inputs/{source}/{case}/param_card.dat")
                and compilation["executable_sha256"] == after["executable_sha256"],
                "Parameter card/executable identity changed")
        require(before["campaign_binding"] == specification["campaign_binding"], "Lane campaign mismatch")
        current_settings = dict(before["settings"])
        require(current_settings.pop("mlreductionlib") == str(backend), "Unexpected selected backend")
        settings.append(current_settings)
        runtimes[source] = norm(before["runtime"])
        target_miss_rows = [row for (_, symmetric, pole), row in zip(manual, comparison["residuals"])
                            if not (symmetric < 1e-8 or pole < 1e-8)]
        significant = [s for _, s, p in manual if p >= 1e-8]
        maximum_significant = max(significant, default=0.)
        lanes.append({"source": source, "backend": backend, "points": 100,
                      "rows": len(observation["rows"]), "scalar_comparisons": len(manual),
                      "regression_passed": regression, "ordinary_target_passed": target,
                      "status_counts": observation["status_counts"],
                      "accepted_backend_precision_counts": observation["backend_precision_counts"],
                      "maximum_symmetric_finite_error": max(s for c, s, _ in manual if c == 1),
                      "maximum_symmetric_non_negligible_residual": maximum_significant
                      if math.isfinite(maximum_significant) else None,
                      "ordinary_target_miss_count": len(target_miss_rows),
                      "target_miss_points": sorted({row["point"] for row in target_miss_rows}),
                      "target_miss_columns": {str(column): sum(row["column"] == column
                                                               for row in target_miss_rows)
                                              for column in range(4)},
                      "target_miss_residuals": target_miss_rows})
    require(all(item == settings[0] for item in settings), "Inconsistent numerical settings")
    status = load(closure + "/status.json")
    cleanup = []
    for stage in status["stages"]:
        require(stage["returncode"] == 0 and stage["status"] == "passed"
                and not stage["cleanup"]["survivors"], "Stage not terminal/clean")
        require(digest(closure + "/" + stage["id"] + "/console.log") == stage["console_sha256"],
                "Stage console mismatch")
        processes = stage["cleanup"]["observed_owned_processes"]
        keys = {(p["pid"], p["start_time_ticks"], p["session"]) for p in processes}
        pids = {p["pid"] for p in processes}
        require(len(keys) == len(processes) and all(
            p["pid"] > 0 and p["start_time_ticks"] > 0 and p["session"] in pids
            and (p["parent_pid"] in pids or p["pid"] == stage["child_pid"])
            for p in processes), "Invalid retained process ownership")
        cleanup.append({"stage": stage["id"], "observed_processes": len(processes), "survivors": []})
    groups = []
    for token in ("configuration-", "/cards-", "/verified-builds.json"):
        records = [load(name) for name in files if name.startswith(closure + "/") and token in name]
        require(records and all(item == records[0] for item in records), "Administrative inputs changed")
        groups.append(records)
    for source in ("baseline", "candidate"):
        require(groups[2][0][source + "/" + case]["runtime"] == runtimes[source],
                "Verified-build/runtime record mismatch")
    regression = all(lane["regression_passed"] for lane in lanes)
    target = all(lane["ordinary_target_passed"] for lane in lanes)
    return {"verified": True, "index_sha256": checkpoint_sha256,
            "payloads_verified": len(files), "payload_bytes": sum(map(len, blobs.values())),
            "cases": [case], "lanes_replayed": len(lanes), "lanes": lanes,
            "regression_passed": regression, "ordinary_target_passed": target,
            "regression_and_target_passed": regression and target,
            "native_executions": 0, "performance_acceptance": False,
            "fpe_flags_observed": False, "internal_retries_observed": False,
            "runtime_evidence": "retained ldd and resolved file hashes; no worker /proc maps",
            "external_objects_rehashed": False, "cleanup": cleanup}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root")
    parser.add_argument("checkpoint_path")
    parser.add_argument("checkpoint_sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, args.checkpoint_path, args.checkpoint_sha256),
                     indent=2, sort_keys=True))
