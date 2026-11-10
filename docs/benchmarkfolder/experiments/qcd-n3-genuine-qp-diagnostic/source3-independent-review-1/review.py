"""Independent sealed-source3 read-only review; no inspected binary execution."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[4]
SOURCE = BASE / "source-proposal3"
WORK = ROOT / ".tirust-build/qcd-n3-genuine-qp-diagnostic"
SEAL = "fb8a6be7e9e5f24a6f5adb03dcc357563fc1db5fc6434e61f6a1eb03c81e5a45"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_index(directory, filename, expected):
    index = directory / filename
    if sha(index) != expected:
        raise ValueError("Index seal mismatch: " + str(index))
    payload = json.loads(index.read_text())
    files = payload["files"]
    for name, record in files.items():
        path = directory / name
        if path.is_symlink() or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Indexed payload mismatch: " + str(path))
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(files) | {filename}:
        raise ValueError("Indexed payload membership mismatch: " + str(directory))
    return payload


def main():
    if (HERE / "review.json").exists():
        raise FileExistsError(HERE / "review.json")
    index = checked_index(SOURCE, "source-index.json", SEAL)
    binding = json.loads((SOURCE / "bindings.json").read_text())
    old = BASE / "source-proposal2"
    old_index = checked_index(old, "source-index.json", "44039ba25e451910d1ab7c264ac1f397437ce41f965d8e381b539dd9baf44046")
    build_static = BASE / "build2-static"
    checked_index(build_static, "evidence-index.json", "446ba17811c642753901b867fa86efd962b82511594cfd92adb5cb8b394f618f")
    failed = BASE / "failed-run2-retention-1"
    failed_index = checked_index(failed, "evidence-index.json", "7ee94d74363f35905828cab7372c634b2e658c2d831a522377be026061e76cb0")
    old_binding = json.loads((old / "bindings.json").read_text())
    if binding["runtime"] != old_binding["runtime"] or len(binding["runtime"]) != 15:
        raise ValueError("Original fifteen-library runtime changed")
    if any(binding["dependencies"].get(str(old / name)) != item["sha256"]
           for name, item in old_index["files"].items()):
        raise ValueError("Source2 payload not covered by source3 dependency pins")
    references = {}
    for name, item in failed_index["files"].items():
        if name.startswith("actual-run2/"):
            path = WORK / "run2" / Path(name).name
            if sha(path) != item["sha256"]:
                raise ValueError("Actual run2 changed against closed capture")
            references[str(path)] = item["sha256"]
    build_ref = json.loads((build_static / "evidence-index.json").read_text())["files"]
    for name in ["build.json", "plan.json"] + [f"command-{n}.{ext}" for n in range(1, 5) for ext in ("json", "log")]:
        path = WORK / "build2" / name
        if sha(path) != build_ref[name]["sha256"]:
            raise ValueError("Actual build2 metadata changed against closed capture")
        references[str(path)] = sha(path)
    map_bytes = gzip.decompress((build_static / "link.map.gz").read_bytes())
    if (WORK / "build2/link.map").read_bytes() != map_bytes:
        raise ValueError("Actual build2 map changed against compressed closed capture")
    references[str(WORK / "build2/link.map")] = hashlib.sha256(map_bytes).hexdigest()
    proof = json.loads((WORK / "build2/build.json").read_text())
    references.update(proof["objects"])
    references.update(proof["original_objects"])
    references[proof["executable"]] = proof["executable_sha256"]
    for path, wanted in references.items():
        if sha(path) != wanted:
            raise ValueError("Referenced build2/original object bytes changed")
    all_pins = {str(SOURCE / name): item["sha256"] for name, item in index["files"].items()}
    all_pins[str(SOURCE / "source-index.json")] = SEAL
    all_pins.update(binding["dependencies"])
    all_pins.update(references)
    for path, wanted in all_pins.items():
        if sha(path) != wanted:
            raise ValueError("Pre-review pin mismatch: " + path)
    forbidden_write_roots = (SOURCE, old, WORK / "build2", WORK / "run2", WORK / "run3")
    guard = {"forbidden_process_attempts": [], "protected_input_write_attempts": []}

    def audit(event, arguments):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system", "ctypes.dlopen"):
            guard["forbidden_process_attempts"].append(event)
            raise AssertionError("No compiler, loader, native executable or dynamic loading in independent review")
        path = None
        if event == "open":
            item, mode, flags = arguments
            if (isinstance(mode, str) and any(x in mode for x in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                path = item
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"):
            path = arguments[0]
        elif event in ("os.rename", "os.link", "os.symlink"):
            path = arguments[1]
        if isinstance(path, (str, bytes, os.PathLike)):
            absolute = Path(os.fsdecode(path)).absolute()
            if any(absolute == root or root in absolute.parents for root in forbidden_write_roots):
                guard["protected_input_write_attempts"].append({"event": event, "path": str(absolute)})
                raise AssertionError("Actual source2/source3/build2/run2/run3 writes forbidden")

    sys.addaudithook(audit)
    sys.path.insert(0, str(SOURCE))
    import run as controller
    log = io.StringIO()
    started = time.monotonic()
    with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No real subprocess")), mock.patch.object(os, "execve", side_effect=AssertionError("No real execve")):
        checked_binding = controller.source_check()
        _, inherited, executable = controller.inherited_build(checked_binding)
        suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern="test_*.py")
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    seconds = time.monotonic() - started
    if result.testsRun != 58 or not result.wasSuccessful():
        raise AssertionError(log.getvalue())
    after = {path: sha(path) for path in all_pins}
    if after != all_pins or guard["forbidden_process_attempts"] or guard["protected_input_write_attempts"]:
        raise ValueError("Independent review guard/pin failure")
    checked_index(SOURCE, "source-index.json", SEAL)
    checked_index(old, "source-index.json", "44039ba25e451910d1ab7c264ac1f397437ce41f965d8e381b539dd9baf44046")
    outcome = {
        "schema": "qcd-n3-source3-independent-review-1", "passed": True,
        "source3_index_sha256": SEAL, "source3_payload_count": len(index["files"]),
        "source3_payload_bytes": sum(item["bytes"] for item in index["files"].values()),
        "source3_dependency_count": len(binding["dependencies"]),
        "source2_index_sha256": sha(old / "source-index.json"),
        "source2_payloads_covered_by_dependency_pins": len(old_index["files"]),
        "closed_build2_run2_reference_hashes": references,
        "all_checked_pin_count": len(all_pins), "all_checked_pins_before_and_after": all_pins,
        "original_runtime_count": len(binding["runtime"]), "original_runtime_unchanged": True,
        "build2_executable_sha256": inherited["executable_sha256"],
        "test_count": result.testsRun, "test_failures": len(result.failures), "test_errors": len(result.errors),
        "static_source_and_inherited_build_checks_passed": True, "seconds": seconds,
        "guards": guard, "real_native_or_compiler_or_loader_invocations": 0,
        "actual_diagnostic_shim_created_by_review": False,
        "review_conclusion": "No blocking defect found. Original fifteen providers remain exact; fresh source3 is reuse-only, traces/alias guards precede native entry and the unchanged worker map checks remain observational.",
        "limitations": ["This review did not invoke run.py run, a loader or a numerical worker.",
                        "The worker mapping check remains an observational sample, not a before-main barrier or continuous monitoring.",
                        "Alias verification receipts are not guaranteed after a failed worker; failures remain rejected.",
                        "No numerical, performance or QP qualification follows from static review or pure fixture tests."],
    }
    (HERE / "test.log").write_text(log.getvalue())
    (HERE / "review.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "index.json"}
    packet = {"schema": "qcd-n3-source3-independent-review-index-1", "scope": "Sealed source membership, immutable source2/build2/run2 refs and 58 pure tests; no native/compiler/loader calls or actual diagnostic shim.", "files": files}
    (HERE / "index.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "index_sha256": sha(HERE / "index.json"), "payload_count": len(files),
                      "source3_index_sha256": SEAL, "test_count": result.testsRun,
                      "all_checked_pin_count": len(all_pins), "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
