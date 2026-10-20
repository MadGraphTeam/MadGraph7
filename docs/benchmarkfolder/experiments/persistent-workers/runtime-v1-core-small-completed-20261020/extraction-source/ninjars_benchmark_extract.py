#!/usr/bin/env python3
"""Retain every completed case from a failed persistent-worker run.

This derives a complete *case subset*, never changes the failed source run, and
does not promote measurements. Numerical protocol sources remain frozen.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import ninjars_benchmark_workers as workers

ROOT = workers.ROOT
SCHEMA = "completed-case-extraction-v1"
TARGETS = [("baseline", b) for b in (1, 6, 7)] + [
    ("candidate", b) for b in (1, 6, 7, 8)]


def inside_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("Extraction evidence must remain inside the repository")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def inventory(directory):
    """Hash regular files; refuse links and special files in retained evidence."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Symlink in extraction evidence: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"Nonregular extraction evidence: {path}")
        result[str(path.relative_to(directory))] = workers.digest(path)
    return result


def completed_prefix(status):
    requested, completed = status.get("requested_cases"), status.get("completed_cases")
    if status.get("status") != "failed":
        raise RuntimeError("Extraction requires a stopped failed source run")
    if (not isinstance(requested, list) or not isinstance(completed, list)
            or not completed or len(completed) >= len(requested)
            or any(not isinstance(case, str) or
                   re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case) is None
                   for case in requested)
            or len(set(requested)) != len(requested)
            or completed != requested[:len(completed)]):
        raise RuntimeError("Completed cases must be a nonempty strict prefix without duplicates")
    return completed


def rejection_evidence(source, completed):
    rejected = sorted((source / "rejected").rglob("*.json"))
    if not rejected:
        raise RuntimeError("Failed source run has no retained rejection evidence")
    diagnostics = {}
    for path in rejected:
        failure = read_json(path)
        if (failure.get("case") in completed or
                any(path.name.startswith(case + "-") for case in completed)):
            raise RuntimeError("A completed case has rejection or cleanup-failure evidence")
        log = failure.get("diagnostic_log")
        if log:
            log = inside_root(log)
            if not log.is_file() or log.is_symlink():
                raise RuntimeError("Missing or linked failure diagnostic log")
            diagnostics[str(log.relative_to(ROOT))] = workers.digest(log)
    return diagnostics


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify_extraction(destination, *, verify_source=True):
    """Recheck the derivation, unchanged source bytes and the frozen validator."""
    destination = inside_root(destination)
    manifest = read_json(destination / "derivation.json")
    payload = {key: value for key, value in manifest.items() if key != "derivation_sha256"}
    if (manifest.get("schema") != SCHEMA or
            workers.bench.content_digest(payload) != manifest.get("derivation_sha256")):
        raise RuntimeError("Invalid extraction derivation digest")
    source = inside_root(ROOT / manifest["source_run"])
    original = read_json(destination / "parent-run/run-status.json")
    completed = completed_prefix(original)
    if (completed != manifest["selected_cases"] or
            original != manifest["original_status"]):
        raise RuntimeError("Extraction omits or changes completed source cases")
    actual = inventory(destination)
    actual.pop("derivation.json")
    # The frozen promoter adds this record only after independent validation.
    actual.pop("promotion.json", None)
    if actual != manifest["files_sha256"]:
        raise RuntimeError("Extracted evidence differs from its derivation manifest")
    if inventory(destination / "parent-run") != manifest["source_files_sha256"]:
        raise RuntimeError("Retained failed source snapshot changed")
    if verify_source and inventory(source) != manifest["source_files_sha256"]:
        raise RuntimeError("Original failed source run changed after extraction")
    rejection_evidence(destination / "parent-run", completed)
    status = read_json(destination / "run-status.json")
    expected = {
        "status": "complete", "acceptance_eligible": False,
        "scope": "completed-case-subset", "completed_cases": completed,
        "requested_cases": completed, "original_status": "failed",
        "original_requested_cases": original["requested_cases"],
        "source_run": manifest["source_run"],
        "source_run_status_sha256": manifest["source_files_sha256"]["run-status.json"],
        "derivation_manifest": "derivation.json",
    }
    if status != expected:
        raise RuntimeError("Derived status misrepresents the failed source run")
    for copied, original_name in manifest["unchanged_copies"].items():
        if workers.digest(destination / copied) != manifest["source_files_sha256"][original_name]:
            raise RuntimeError("Derived protocol, measurement or build differs from source")
    for source_name, record in manifest["diagnostic_logs"].items():
        if workers.digest(destination / record["snapshot"]) != record["sha256"]:
            raise RuntimeError("Failure diagnostic snapshot changed")
        if verify_source and workers.digest(ROOT / source_name) != record["sha256"]:
            raise RuntimeError("Original failure diagnostic changed")
    _, records = workers.checked_series(destination, minimum=7, require_core=True)
    if {key[0] for key in records} != set(completed):
        raise RuntimeError("Validated series differs from the completed case subset")
    return {
        "destination": str(destination.relative_to(ROOT)),
        "selected_cases": completed,
        "measurements": len(records),
        "derivation_sha256": manifest["derivation_sha256"],
        "original_status": "failed",
        "published": False,
    }


def extract_completed(source, destination):
    source, destination = inside_root(source), inside_root(destination)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("Extraction destination already exists")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise RuntimeError("Source and destination cannot contain one another")
    original = read_json(source / "run-status.json")
    completed = completed_prefix(original)
    diagnostics = rejection_evidence(source, completed)
    source_files = inventory(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".pending-",
                                     dir=destination.parent))
    try:
        # Preserve the entire failed run, including incomplete later-case files.
        shutil.copytree(source, temporary / "parent-run")
        unchanged = {}

        def retain(name):
            copy_file(temporary / "parent-run" / name, temporary / name)
            unchanged[name] = name

        retain("protocol.json")
        for name in source_files:
            if name.startswith("sources/") or name.startswith("inputs/"):
                retain(name)
            elif any(name.startswith(f"builds/{case}-{src}-{backend}/")
                     for case in completed for src, backend in TARGETS):
                retain(name)
            elif any(name.startswith(f"data/{case}-{src}-{backend}-")
                     for case in completed for src, backend in TARGETS):
                retain(name)
        diagnostic_records = {}
        for name, checksum in diagnostics.items():
            snapshot = "parent-diagnostics/" + checksum + ".log"
            copy_file(ROOT / name, temporary / snapshot)
            diagnostic_records[name] = {"snapshot": snapshot, "sha256": checksum}
        copy_file(Path(__file__), temporary / "extraction-source" / Path(__file__).name)
        source_name = str(source.relative_to(ROOT))
        workers.bench.dump(temporary / "run-status.json", {
            "status": "complete", "acceptance_eligible": False,
            "scope": "completed-case-subset", "completed_cases": completed,
            "requested_cases": completed, "original_status": "failed",
            "original_requested_cases": original["requested_cases"],
            "source_run": source_name,
            "source_run_status_sha256": source_files["run-status.json"],
            "derivation_manifest": "derivation.json",
        })
        (temporary / "README.md").write_text(
            "# Completed-case subset of a failed benchmark run\n\n"
            f"Source: `{source_name}`. Its status remains **failed**.\n\n"
            "This directory contains every case that completed before the failure, "
            "in the original order: " + ", ".join(f"`{case}`" for case in completed) + ". "
            "No completed case, backend or repetition was selected out.\n\n"
            "`parent-run/` retains the entire original failed run byte-for-byte, "
            "including its rejected measurements, incomplete later-case data, "
            "logs and status. `derivation.json` binds the copied evidence and "
            "the source snapshot. The complete status here applies only to the "
            "explicit completed-case subset. The remaining requested cases and "
            "the original failure remain outstanding.\n\n"
            "The unchanged persistent-worker validator requires all seven targets "
            "and at least seven contiguous repetitions for every extracted case. "
            "Extraction does not promote data or imply numerical/performance passage.\n")
        manifest = {
            "schema": SCHEMA,
            "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_run": source_name,
            "original_status": original,
            "selected_cases": completed,
            "selection_policy": "All completed cases, exact original prefix; no user selection",
            "source_files_sha256": source_files,
            "unchanged_copies": unchanged,
            "diagnostic_logs": diagnostic_records,
            "files_sha256": inventory(temporary),
        }
        manifest["derivation_sha256"] = workers.bench.content_digest(manifest)
        workers.bench.dump(temporary / "derivation.json", manifest)
        verify_extraction(temporary)
        # Recheck source after copying and validating; a live/modified run cannot
        # be mistaken for the frozen failed evidence observed at entry.
        if inventory(source) != source_files:
            raise RuntimeError("Original failed source run changed during extraction")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("Extraction destination appeared during validation")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_extraction(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "extract":
        if args.source is None:
            parser.error("extract requires --source")
        result = extract_completed(args.source, args.destination)
    else:
        if args.source is not None:
            parser.error("verify does not accept --source")
        result = verify_extraction(args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
