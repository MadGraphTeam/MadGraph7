#!/usr/bin/env python3
"""Publish a combined checkpoint only after the fresh lane closes cleanly."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def module(name, path, expected):
    if digest(path) != expected:
        raise ValueError("Sealing dependency changed")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    source, replay = read(HERE / "source-index.json"), read(HERE / "replay-index.json")
    runner = module("closed_ew_continuation", HERE / "run.py", source["files"]["run.py"]["sha256"])
    runner.verify_seal()
    runner.checked_files(replay["files"], HERE)
    verifier = module("closed_ew_replay", HERE / "verify.py", replay["files"]["verify.py"]["sha256"])
    destination = HERE / "checkpoint.json"
    if destination.exists():
        raise ValueError("A combined checkpoint already exists; no replacement")
    state = read(HERE / "status.json")
    stage = HERE / ("numerical-" + runner.CASE)
    result = read(stage / "numerical-result.json")
    if (state["status"] != "passed" or result["status"] != "passed" or len(state["stages"]) != 1
            or state["stages"][0]["status"] != "passed" or state["stages"][0]["returncode"] != 0
            or state["stages"][0]["cleanup"]["survivors"]):
        raise ValueError("Only a complete clean continuation can close the combined case")
    if len(result["lanes"]) != 6:
        raise ValueError("Expected all six selected complete lanes")
    # Reuse the original bounded, pure publication guard. This executes no
    # compiler, loader, model setup, worker or amplitude routine.
    preparation = runner.adapter.load_support()[1]
    preparation.STAGE_SECONDS, preparation.CLEANUP_RESERVE = 120., 10.
    with preparation.StageDeadline():
        inputs = read(HERE / "inputs-index.json")
        files = dict(inputs["files"])
        for name, expected in source["dependencies"].items():
            if record(ROOT / name) != expected:
                raise ValueError("Pinned runtime-adapter dependency changed")
            files[name] = expected
        for path in sorted(HERE.rglob("*")):
            if path.is_file():
                if path.is_symlink() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    raise ValueError("Unexpected file in continuation closure")
                if path.name in ("checkpoint.json", "replay-result.json"):
                    continue
                files[str(path.relative_to(ROOT))] = record(path)
        relative = str(HERE.relative_to(ROOT))
        continuation = {"input_index_path": relative + "/inputs-index.json",
            "input_index_sha256": digest(HERE / "inputs-index.json"),
            "status_path": relative + "/status.json", "stage_directory": str(stage.relative_to(ROOT)),
            "configuration_before_path": str((stage / "configuration-before.json").relative_to(ROOT)),
            "configuration_after_path": str((stage / "configuration-after.json").relative_to(ROOT))}
        checkpoint = {"schema": "madloop-tirust-combined-one-case-closure-v1",
            "campaign": runner.adapter.CAMPAIGN,
            "case": next(c for c in read(runner.REPORT / "cases.json")["cases"] if c["id"] == runner.CASE),
            "cases": [verifier.case_summary(result["lanes"])],
            "manifest_sha256": runner.adapter.MANIFEST_SHA,
            "source_index_sha256": digest(HERE / "source-index.json"),
            "replay_index_sha256": digest(HERE / "replay-index.json"),
            "files": files, "file_count": len(files), "total_bytes": sum(v["bytes"] for v in files.values()),
            "lanes": inputs["selected_original_lanes"] + [{"source": "candidate", "backend": 9,
                "directory": relative + "/lane/candidate-9", "origin": "fresh-continuation"}],
            "interruption": inputs["interruption"], "continuation": continuation,
            "performance_acceptance_eligible": False, "native_calls_after_case": 0}
        data = (json.dumps(checkpoint, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        checksum = hashlib.sha256(data).hexdigest()
        checkpoint_path = str(destination.relative_to(ROOT))
        # The verifier sees only the sealed files. Validate identical bytes at
        # its canonical path in that temporary root before publishing anything.
        with tempfile.TemporaryDirectory(prefix="ew-closure-proof-", dir=ROOT / ".benchmark-work") as temporary:
            copy = Path(temporary)
            for name, expected in files.items():
                source_path = ROOT / name
                if record(source_path) != expected:
                    raise ValueError("Evidence changed during sealing")
                target = copy / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
            target = copy / checkpoint_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            verified = verifier.verify(copy, checkpoint_path, checksum)
        if not verified["verified"] or not verified["regression_passed"] or verified["lanes_replayed"] != 6:
            raise ValueError("Combined replay did not qualify all six lanes")
        for name, expected in files.items():
            if record(ROOT / name) != expected:
                raise ValueError("Evidence changed before publication")
        with destination.open("xb") as stream:
            stream.write(data)
        with (HERE / "replay-result.json").open("x") as stream:
            json.dump(verified, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"checkpoint": checkpoint_path, "sha256": checksum,
            "payload_files": len(files), "payload_bytes": checkpoint["total_bytes"],
            "regression_passed": verified["regression_passed"],
            "ordinary_target_passed": verified["ordinary_target_passed"], "native_executions": 0}, indent=2))


if __name__ == "__main__":
    main()
