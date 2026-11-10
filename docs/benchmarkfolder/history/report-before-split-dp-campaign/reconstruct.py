#!/usr/bin/env python3
"""Verify the original Git closure and optionally reconstruct an isolated report.

Run from the MadGraph repository root. This script never changes the canonical
report, its evidence or a previous reconstruction, and launches no physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COMMIT = "bd8eb72a179a2a6a4385af07d13507f94c52df95"
PREFIX = "docs/benchmarkfolder/"
LEDGER_SHA256 = "111f170db4a7c1359ac7760ea8d34405b682f4798206b3d0a2854d3d99c8010a"
PDF_SHA256 = "f5cc86a70738dfc719b34a1e7686b57934fe1b421c7774275f2498ee0044d2df"
COMPACT_FILES = {
    "benchmark.pdf", "report-build.json", "report-context.json", "benchmark.typ",
    "style.typ", "poles-timing.typ", "build_report.py", "intermediate_report.py",
    "collect_intermediate.py", "intermediate.typ", "intermediate-status.json",
    "qualification.json", "summary.json", "cases.json", "implementation-start.json",
    "style-provenance.json", "qualification/tirust-install/index.json",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    path = Path(name)
    if (not isinstance(name, str) or not name or path.is_absolute()
            or ".." in path.parts or "\\" in name or "\n" in name):
        raise ValueError("Unsafe retained path: " + str(name))
    return path


def compact_path(name):
    path = HERE
    for part in safe_relative(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Compact archive may not contain symlinks")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        help="New ignored directory under this repository; omission verifies only")
    args = parser.parse_args()
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    if Path.cwd().resolve() != root:
        parser.error("Run from the MadGraph repository root")
    manifest = json.loads(compact_path("archive-manifest.json").read_text())
    if (manifest.get("schema") != "madloop-tirust-report-archive-v1"
            or set(manifest.get("original_files", {})) != COMPACT_FILES
            or manifest["report_commit"] != COMMIT or manifest["git_report_prefix"] != PREFIX
            or manifest["original_ledger_sha256"] != LEDGER_SHA256
            or manifest["original_pdf_sha256"] != PDF_SHA256):
        raise ValueError("Archive identity differs from the original report")
    ledger_raw = compact_path("report-build.json").read_bytes()
    if digest(ledger_raw) != LEDGER_SHA256:
        raise ValueError("Original report ledger changed")
    ledger = json.loads(ledger_raw)
    if ledger["pdf_sha256"] != PDF_SHA256:
        raise ValueError("Original PDF binding changed")
    expected = dict(ledger["inputs"])
    for name, identity in manifest["original_files"].items():
        path = safe_relative(name)
        if identity["git_path"] != PREFIX + path.as_posix():
            raise ValueError("Archived file has a different Git path")
        raw = compact_path(name).read_bytes()
        if digest(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
            raise ValueError("Compact archive changed: " + name)
        if name in expected and expected[name] != {k: identity[k] for k in ("sha256", "bytes")}:
            raise ValueError("Compact archive conflicts with original ledger")
        expected[name] = {k: identity[k] for k in ("sha256", "bytes")}
    if (len(ledger["inputs"]) != manifest["required_input_files"]
            or sum(i["bytes"] for i in ledger["inputs"].values()) != manifest["required_input_bytes"]):
        raise ValueError("Input closure count/size changed")
    destination = None
    if args.destination is not None:
        candidate = args.destination if args.destination.is_absolute() else root / args.destination
        destination = candidate.resolve()
        if (destination == root or not destination.is_relative_to(root)
                or ".git" in destination.relative_to(root).parts or candidate.exists()
                or candidate.is_symlink()):
            raise ValueError("Destination must be a new directory inside this repository")
        if subprocess.run(["git", "check-ignore", "-q", "--", str(destination)]).returncode != 0:
            raise ValueError("Reconstruction destination must be ignored")
    # Read every required Git blob before publishing any reconstruction. Retain
    # bytes in memory (~288 MB) so a later Git/content failure leaves no partial
    # directory and no live files can change between verification and copying.
    payloads = {}
    process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    try:
        for name, identity in sorted(expected.items()):
            safe_relative(name)
            process.stdin.write((COMMIT + ":" + PREFIX + name + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            if len(header) != 3 or header[1] != "blob":
                raise ValueError("Original Git blob missing: " + name)
            raw = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b"\n":
                raise ValueError("Malformed Git batch output")
            if len(raw) != identity["bytes"] or digest(raw) != identity["sha256"]:
                raise ValueError("Git bytes differ from original report ledger: " + name)
            if destination is not None:
                payloads[name] = raw
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise ValueError("Git batch verification failed")
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        for name, raw in payloads.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
    print(json.dumps({"status": "passed", "report_commit": COMMIT,
        "required_input_files_verified": len(ledger["inputs"]),
        "required_input_bytes_verified": manifest["required_input_bytes"],
        "total_git_files_verified": len(expected), "compact_archive_verified": True,
        "original_pdf_sha256": PDF_SHA256,
        "reconstruction": str(destination) if destination is not None else None,
        "pdf_recompiled": False, "numerical_execution": False}, indent=2))


if __name__ == "__main__":
    main()
