#!/usr/bin/env python3
"""Verify the two retained, unlaunched CPU2 preparation namespaces."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LABELS = ("historical-c91-qcd-n2-chain-1", "historical-c91-ew-n2-chain-1")
INDEX = ROOT / "unlaunched-n2-cpu2-index.json"
README = ROOT / "README-unlaunched-n2-cpu2.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    files, attempts = {}, []
    for label in LABELS:
        directory = ROOT / label
        paths = sorted(path for path in directory.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("Unexpected symlink in retained preparation")
        specification = json.loads((directory / "specification.json").read_text())
        scheduling = json.loads((directory / "scheduling.json").read_text())
        preparation = json.loads((directory / "preparation.json").read_text())
        expected = {"specification.json", "preparation.json", "scheduling.json"}
        expected.update("inputs/" + name for name in specification["inputs"])
        expected.update("protocol-sources/" + name for name in specification["protocols"])
        if {path.relative_to(directory).as_posix() for path in paths} != expected:
            raise ValueError("Unexpected or missing preparation files; execution evidence cannot be silently ignored")
        if (len(paths) != 11 or specification["label"] != label
                or specification["machine"]["selected_cpu"] != 2
                or scheduling["status"] != "prepared-unlaunched"
                or scheduling["performance_acceptance_eligible"] is not False
                or specification["performance_acceptance_eligible"] is not False):
            raise ValueError("Unlaunched preparation specification differs")
        for group, subdirectory in (("inputs", "inputs"), ("protocols", "protocol-sources")):
            for name, checksum in specification[group].items():
                if digest(directory / subdirectory / name) != checksum:
                    raise ValueError(f"Retained {group} binding differs: {label}/{name}")
        for source in ("baseline", "candidate"):
            if preparation["clones"][source]["compiled_or_relinked"] is not False:
                raise ValueError("Preparation unexpectedly records compilation/relink")
        for path in paths:
            files[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        attempts.append({"label": label, "case": specification["case"]["id"],
                         "status": "prepared-unlaunched", "selected_cpu": 2,
                         "created_utc": specification["created_utc"],
                         "scheduling_decision_utc": scheduling["utc"],
                         "reason": scheduling["reason"], "successor_label": scheduling["successor_label"],
                         "native_measurements": 0, "retained_files": len(paths),
                         "performance_acceptance_eligible": False,
                         "physics_result": "not evaluated in this namespace"})
    return {"schema": "madloop-tirust-unlaunched-preparation-audit-v1",
            "scope": "Retained text inventory and internal bindings only; historical cloned binaries are not revalidated",
            "status": "passed", "attempts": attempts, "retained_files": len(files), "files": files,
            "audit_program_sha256": digest(Path(__file__).resolve()), "readme_sha256": digest(README)}


if __name__ == "__main__":
    result = collect()
    if sys.argv[1:] == ["--create"]:
        with INDEX.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
    elif sys.argv[1:]:
        raise SystemExit("Use no arguments to verify; --create publishes a new index exclusively")
    elif result != json.loads(INDEX.read_text()):
        raise SystemExit("Retained preparation audit differs from its frozen index")
    print(json.dumps({"status": "passed", "retained_files": result["retained_files"],
                      "index_sha256": digest(INDEX)}, sort_keys=True))
