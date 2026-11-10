#!/usr/bin/env python3
"""Append-once closed provisional evidence, with portable Python-only replay.

Import has no IO side effects. Seal requires explicit terminal authorization;
replay never consults the original workspace, native libraries or executables.
"""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SCHEMA = "madloop-tirust-closed-provisional-report-v1"
RUNTIME_SCHEMA = "madloop-tirust-provisional-full100-v1"
CAMPAIGN = "split-dp-terminal-1"
MANIFEST_SHA = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
CONTROLLER_SHA = "7cccff659bdb180c070a62f9637fc4865892eaa739f681927c538bfae58e41c3"
UNAVAILABLE = {"unavailable_deadline", "unavailable_or_numerical_failure", "numerical_failure",
               "not_run_budget", "not_completed_budget"}
ROOT_FILES = {"specification.json", "campaign-manifest.json", "cases.json", "preparation.json",
              "prerequisites.json", "run-start.json", "progress.json", "summary.json",
              "administrative-closure.json", "controller-failure.json"}
PROTOCOLS = {"tirust_benchmark_provisional.py", "tirust_benchmark.py", "tirust_benchmark_campaign.py",
             "tirust_benchmark_chain_probe.py", "tirust_benchmark_compiler.py", "tirust_benchmark_worker.py",
             "tirust_benchmark_worker.f", "tirust_benchmark_warmup.py", "process_cleanup.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def parsed(raw):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw.decode("utf-8"), parse_constant=reject)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid retained path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe retained path")
    return name


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify_native_runtime(runtime, native):
    """Join recorded TIRust artifacts by exact installation-relative path.

    Other loader/library paths remain independently recorded runtime entries.
    Cargo/header pins belong to the frozen manifest, not observed runtime.
    """
    prefix = PurePosixPath(native["prefix"])
    require(prefix.is_absolute() and ".." not in prefix.parts, "Invalid recorded native prefix")
    required = {"libraries": ("lib/libmadloop_tirust.so",),
                "archives": ("lib/libtirust.a",),
                "metadata": ("lib/tirust-config.json", "lib/tirust-link-before.txt", "lib/tirust-native-libs.txt")}
    for kind, names in required.items():
        for name, checksum in runtime[kind].items():
            file = PurePosixPath(name)
            if file.is_relative_to(prefix):
                relative = file.relative_to(prefix).as_posix()
                if relative in native["files"]:
                    require(checksum == native["files"][relative], "Recorded TIRust artifact hash differs: " + relative)
        for name in names:
            require(name in native["files"] and runtime[kind].get(str(prefix / name)) == native["files"][name],
                    "Required recorded TIRust artifact differs or is absent: " + name)


def snapshot(directory):
    """Capture only this explicit report tree; no work/build/library traversal."""
    result = {}
    for file in sorted(directory.rglob("*")):
        mode = file.lstat().st_mode
        require(not stat.S_ISLNK(mode), "Symlink in retained source")
        if stat.S_ISDIR(mode):
            continue
        require(stat.S_ISREG(mode), "Nonregular retained source")
        name = safe(file.relative_to(directory).as_posix())
        pieces = PurePosixPath(name).parts
        allowed = (name in ROOT_FILES
            or (len(pieces) == 2 and pieces[0] == "protocol-sources" and pieces[1] in PROTOCOLS)
            or (len(pieces) == 2 and pieces[0] == "points" and file.suffix in (".json", ".dat"))
            or (len(pieces) >= 3 and pieces[0] == "comparisons" and file.suffix in (".json", ".dat", ".txt", ".log", ".tmp")))
        require(allowed, "File outside explicit retention allowlist: " + name)
        raw = file.read_bytes()
        require(not raw.startswith(b"\x7fELF") and b"\x00" not in raw, "Native/binary file outside retention scope")
        raw.decode("utf-8", errors="strict")
        result[name] = raw
    return result


def _baseline_metadata(retained):
    preparation = parsed(retained["preparation.json"])
    config = parsed(retained["cases.json"])
    require(len(config["cases"]) == 11 and len({case["id"] for case in config["cases"]}) == 11,
            "Exactly eleven baseline generation metadata inputs required")
    extra, allowlist = {}, {}
    for case in config["cases"]:
        prepared = preparation["clones"][case["id"]]["baseline"]
        file = Path(prepared["private_output"]) / "generation.json"
        require(file.is_absolute() and file.resolve() == file.absolute() and file.is_file(),
                "Baseline generation metadata is missing or symlinked")
        raw = file.read_bytes()
        require(sha(raw) == prepared["generation_sha256"], "Frozen baseline generation metadata differs")
        parsed(raw)
        name = "prepared-generation/baseline/" + safe(case["id"]) + ".json"
        extra[name] = raw
        allowlist[str(file)] = {"sha256": sha(raw), "bytes": len(raw), "retained_as": name}
    require(len(allowlist) == 11, "Baseline generation metadata paths must be distinct")
    return extra, allowlist


def _load(package_root=None, expected_index_sha256=None):
    root = Path(package_root or HERE).resolve()
    raw_index = (root / "index.json").read_bytes()
    if expected_index_sha256 is not None:
        require(sha(raw_index) == expected_index_sha256, "Selected index SHA-256 differs")
    index = parsed(raw_index)
    require(index.get("schema") == SCHEMA and index.get("closed") is True
            and index.get("performance_acceptance_eligible") is False, "Package is not closed descriptive evidence")
    actual = {}
    for file in root.rglob("*"):
        require(not file.is_symlink(), "Symlink in sealed package")
        if file.is_file() and file != root / "index.json":
            raw = file.read_bytes()
            actual[safe(file.relative_to(root).as_posix())] = {"sha256": sha(raw), "bytes": len(raw)}
    require(actual == index["files"], "Sealed outer inventory differs")
    payload_name = safe(index["payload"])
    require(payload_name in actual and payload_name == "payload-" + actual[payload_name]["sha256"] + ".json.gz",
            "Payload is not content addressed")
    payload = parsed(gzip.decompress((root / payload_name).read_bytes()))
    require(payload.get("schema") == 1 and set(payload) == {"schema", "content"}, "Unexpected payload schema")
    content = payload["content"]
    for checksum, value in content.items():
        require(isinstance(value, str) and sha(value.encode("utf-8")) == checksum, "Payload content SHA-256 mismatch")
    retained = {}
    for name, identity in index["retained_files"].items():
        safe(name)
        raw = content[identity["sha256"]].encode("utf-8")
        require(len(raw) == identity["bytes"] and not raw.startswith(b"\x7fELF") and b"\x00" not in raw,
                "Retained payload bytes differ")
        retained[name] = raw
    require(set(content) == {item["sha256"] for item in index["retained_files"].values()}, "Unreferenced payload content")
    return index, retained


def read_retained_json(name, package_root=None, expected_index_sha256=None):
    _, retained = _load(package_root, expected_index_sha256)
    require(safe(name) in retained, "JSON dependency absent from retained allowlist")
    return parsed(retained[name])


@contextmanager
def _portable(retained):
    """Fresh temporary repository layout; restore all imported module state."""
    previous_path, previous_bytecode = list(sys.path), sys.dont_write_bytecode
    previous_modules = {name: value for name, value in sys.modules.items() if name.startswith("tirust_benchmark")}
    for name in previous_modules:
        del sys.modules[name]
    try:
        sys.dont_write_bytecode = True
        with tempfile.TemporaryDirectory(prefix="closed-provisional-replay-") as temporary:
            root = Path(temporary)
            report = root / "docs/benchmarkfolder/campaigns" / CAMPAIGN / "retained"
            tools = root / "tools"
            tools.mkdir()
            for name, raw in retained.items():
                target = report / safe(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                if name.startswith("protocol-sources/"):
                    (tools / PurePosixPath(name).name).write_bytes(raw)
            sys.path.insert(0, str(tools))
            module = importlib.import_module("tirust_benchmark_provisional")
            yield module, report
    finally:
        for name in list(sys.modules):
            if name.startswith("tirust_benchmark"):
                del sys.modules[name]
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


def _validate(index, retained):
    def read(name):
        return parsed(retained[safe(name)])
    paths = index["paths"]
    spec, closure, manifest = (read(paths[key]) for key in ("specification", "closure", "manifest"))
    binding = {"id": CAMPAIGN, "manifest_sha256": sha(retained[paths["manifest"]])}
    require(index["campaign_binding"] == spec.get("campaign_binding") == binding, "Campaign manifest binding differs")
    require(spec.get("schema") == RUNTIME_SCHEMA and spec.get("performance_acceptance_eligible") is False
            and spec.get("pairs") == 3 and spec.get("points") == 100 and spec.get("point_subset") is None
            and spec.get("cpu_reservation") is False and spec.get("total_cap_seconds") == 1800
            and spec.get("target_batch_seconds") == 1 and spec.get("cleanup_reserve_seconds") == 10,
            "Provisional numerical/timing scope changed")
    require(closure.get("schema") == RUNTIME_SCHEMA and closure.get("status") in ("closed", "closed_budget")
            and closure.get("performance_acceptance_eligible") is False
            and closure.get("precise_acceptance") == "deferred_to_local"
            and closure.get("budget_seconds") == 1800 and positive(closure.get("elapsed_seconds")),
            "Controller lacks terminal closure")
    require(set(spec["protocols"]) == PROTOCOLS, "Incomplete frozen protocol closure")
    require(sha(retained["protocol-sources/tirust_benchmark_provisional.py"]) == CONTROLLER_SHA,
            "Reviewed provisional controller changed")
    for name, checksum in spec["protocols"].items():
        require(sha(retained["protocol-sources/" + safe(name)]) == checksum, "Frozen protocol hash differs")
    for name, checksum in spec["frozen_files"].items():
        require(sha(retained[safe(name)]) == checksum, "Frozen preparation/input differs")
    config, preparation, prerequisites = (read(name) for name in ("cases.json", "preparation.json", "prerequisites.json"))
    require(prerequisites.get("ordinary_numerical_qualification_passed") is True
            and prerequisites["campaign"] == CAMPAIGN and prerequisites["manifest_sha256"] == binding["manifest_sha256"],
            "Numerical prerequisite binding differs")
    require(preparation.get("native_evaluation") is False and preparation.get("compiled_or_relinked") is False,
            "Preparation scope differs")
    schedule, rows = spec["schedule"], closure["rows"]
    require(len(schedule) == 66 and len({row["id"] for row in schedule}) == 66
            and len(rows) == 66 and [row["comparison"] for row in rows] == schedule, "Exact 66-row schedule closure required")
    require(set(index["comparisons"]) == {row["id"] for row in schedule}, "Comparison index coverage differs")
    require(all(name.split("/")[1] in index["comparisons"] for name in retained if name.startswith("comparisons/")),
            "Unscheduled comparison evidence")
    with _portable(retained) as (module, report):
        chains = {}
        for case in config["cases"]:
            name = "prepared-generation/baseline/" + case["id"] + ".json"
            generation = read(name)
            require(sha(retained[name]) == preparation["clones"][case["id"]]["baseline"]["generation_sha256"]
                    and generation["source"] == "baseline" and generation["case"] == case,
                    "Frozen baseline default-chain provenance differs")
            chains[case["id"]] = generation["default_chain"]
        require(schedule == module.schedule(config, chains), "Preregistered schedule ordering differs")
        completed = {}
        for row in rows:
            comparison, status = row["comparison"], row["status"]
            require(status in UNAVAILABLE | {"complete"} and row.get("performance_acceptance_eligible") is False,
                    "Unrecognized closure row")
            key = comparison["id"]
            item = index["comparisons"][key]
            completion = item.get("completion")
            if completion is not None:
                require(read(completion) == row, "Closure/completion disagreement")
            else:
                require(status in {"not_run_budget", "not_completed_budget"}, "Attempted row lacks exact completion")
            if "elapsed_seconds_including_cleanup" in row:
                require(positive(row["elapsed_seconds_including_cleanup"]), "Invalid completion elapsed time")
            if status != "complete":
                continue
            require(completion is not None and row.get("returncode") == 0, "Completed worker was not successful")
            require(positive(row.get("elapsed_seconds_including_cleanup"))
                    and row["elapsed_seconds_including_cleanup"] <= comparison["cap_seconds"],
                    "Completed comparison exceeded its all-in cap")
            require(item.get("result") == "comparisons/" + key + "/result.json", "Completed result path differs")
            result = read(item["result"])
            require(result.get("status") == "complete" and result.get("numerical_pass") is True
                    and result.get("comparison") == comparison, "Completed result scope differs")
            run = report / "comparisons" / key
            require(module.verify_comparison(run) == result, "Exact raw replay differs")
            host = []
            phases = [(source, phase) for source in ("baseline", "candidate") for phase in ("calibrate-0", "calibrate-1")]
            phases += [(source, "repetition-" + str(number)) for number in range(3)
                       for source in (("baseline", "candidate") if number % 2 == 0 else ("candidate", "baseline"))]
            for source, phase in phases:
                snapshot_name = "comparisons/" + key + "/host-snapshots/" + source + "-" + phase + ".json"
                value = read(snapshot_name)
                require(value.get("source") == source and value.get("phase") == phase, "Host snapshot order differs")
                host.append(value)
            require(result.get("host_load") == host, "Result host snapshots differ")
            for source in ("baseline", "candidate"):
                prefix = "comparisons/" + key + "/" + source
                require(item["ready"][source] == prefix + "-ready.json", "READY index differs")
                ready = read(item["ready"][source])
                cold = result["initialization_and_warmup_seconds"][source]
                require(positive(cold) and cold == ready["cold_seconds"], "Initialization/READY cold_seconds differs")
                require(all(value["cold_seconds"] == cold for value in result["observations"][source])
                        and all(result["calibration"][source][phase]["cold_seconds"] == cold for phase in ("initial", "sustained")),
                        "Timed/calibration cold_seconds differs from READY")
                inputs = read(prefix + "-inputs.json")
                prepared = preparation["clones"][comparison["case"]][source]
                runtime = prerequisites["runtime"][source + "/" + comparison["case"]]
                require(all(inputs["runtime"][kind] == prepared["runtime"][kind] == runtime[kind]
                            for kind in ("libraries", "archives", "metadata")), "Numerical prerequisite/runtime binding differs")
                require(inputs["compilation"].get("campaign_binding") == inputs["generation"].get("campaign_binding") == binding
                        and inputs["compilation"].get("numerical_build_from_sources") is True
                        and inputs["compilation"]["sources"] == prepared["generated_sources"]
                        and inputs["compilation"]["executable_sha256"] == prepared["executable_sha256"],
                        "Prepared numerical build binding differs")
                identity = inputs["generation"]["source_identity"]
                if source == "candidate":
                    require(identity["files_sha256"] == manifest["candidate"]["exporter_identity"]["files_sha256"],
                            "Candidate exporter source differs from manifest")
                    verify_native_runtime(inputs["runtime"], manifest["candidate"]["native"])
                else:
                    require(identity["revision"] == manifest["baseline"]["revision"], "Baseline source differs from manifest")
            completed[key] = result
    return completed


def replay_completed(package_root=None, expected_index_sha256=None):
    index, retained = _load(package_root, expected_index_sha256)
    return _validate(index, retained)


def seal(source, confirmation, destination=None, *, expected_manifest_sha256=MANIFEST_SHA):
    """Call only after root authorization. expected_manifest_sha256 is a test seam."""
    source = Path(source).absolute()
    confirmation = parsed(Path(confirmation).read_bytes())
    require(confirmation.get("terminal_session_id") == 82678 and confirmation.get("terminal") is True
            and confirmation.get("seal_authorized") is True and confirmation.get("source_directory") == str(source)
            and confirmation.get("baseline_generation_metadata_authorized") is True,
            "Explicit root terminal/seal authorization is required before reading the source")
    destination = Path(destination or HERE / "sealed-1")
    require(not destination.exists(), "Append-once destination already exists; no overwrite or automatic retry")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        return _seal_authorized(source, confirmation, destination, expected_manifest_sha256)
    except BaseException as error:
        (destination / "seal-failure.json").write_bytes(encoded({"status": "failed", "error": repr(error),
            "terminal_confirmation": confirmation, "performance_acceptance_eligible": False}))
        raise


def _seal_authorized(source, confirmation, destination, expected_manifest_sha256):
    require(not source.is_symlink(), "Source directory is a symlink")
    source_snapshot = snapshot(source)
    retained = dict(source_snapshot)
    closure = "administrative-closure.json" if "administrative-closure.json" in retained else "summary.json"
    require(sha(retained[closure]) == confirmation["closure_sha256"], "Authorized closure hash differs")
    require(sha(retained["campaign-manifest.json"]) == expected_manifest_sha256, "Unapproved source campaign manifest")
    metadata, external_inputs = _baseline_metadata(retained)
    retained.update(metadata)
    rows = parsed(retained[closure])["rows"]
    comparisons = {}
    for row in rows:
        key = row["comparison"]["id"]
        prefix = "comparisons/" + safe(key) + "/"
        name = prefix + "completion.json"
        matching = False
        if name in retained:
            try:
                matching = parsed(retained[name]) == row
            except (ValueError, UnicodeError):
                pass
        comparisons[key] = {"completion": name if matching else None,
                            "completion_retained": name if name in retained else None,
                            "result": prefix + "result.json" if prefix + "result.json" in retained else None,
                            "ready": {source: prefix + source + "-ready.json" for source in ("baseline", "candidate")
                                      if prefix + source + "-ready.json" in retained}}
    content = {sha(raw): raw.decode("utf-8") for raw in retained.values()}
    compressed = gzip.compress(encoded({"schema": 1, "content": content}), compresslevel=9, mtime=0)
    payload_name = "payload-" + sha(compressed) + ".json.gz"
    index = {"schema": SCHEMA, "closed": True, "performance_acceptance_eligible": False,
             "campaign_binding": parsed(retained["specification.json"])["campaign_binding"],
             "paths": {"specification": "specification.json", "closure": closure, "manifest": "campaign-manifest.json"},
             "payload": payload_name, "retained_files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in retained.items()},
             "comparisons": comparisons, "replay": {"path": "retain.py", "function": "replay_completed", "json_accessor": "read_retained_json"},
             "source_directory": str(source), "terminal_confirmation": confirmation,
             "external_metadata_inputs": external_inputs,
             "retained_raw_bytes": sum(map(len, retained.values())), "unique_raw_bytes": sum(len(s.encode("utf-8")) for s in content.values())}
    # Validate before publication; unavailable/partial bytes remain in payload.
    _validate(index, retained)
    require(snapshot(source) == source_snapshot and _baseline_metadata(source_snapshot) == (metadata, external_inputs),
            "Closed source or authorized metadata changed while sealing")
    (destination / payload_name).write_bytes(compressed)
    shutil.copyfile(HERE / "retain.py", destination / "retain.py")
    (destination / "terminal-confirmation.json").write_bytes(encoded(confirmation))
    index["files"] = {file.name: {"sha256": sha(file.read_bytes()), "bytes": file.stat().st_size}
                      for file in destination.iterdir()}
    (destination / "index.json").write_bytes(encoded(index))
    replay_completed(destination)
    return {"index": str(destination / "index.json"), "sha256": sha((destination / "index.json").read_bytes()),
            "schedule_rows": len(rows), "performance_acceptance_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sealing = sub.add_parser("seal")
    sealing.add_argument("--source", required=True)
    sealing.add_argument("--terminal-confirmation", required=True)
    sub.add_parser("replay")
    args = parser.parse_args()
    result = seal(args.source, args.terminal_confirmation) if args.command == "seal" else replay_completed()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
