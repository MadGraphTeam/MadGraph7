#!/usr/bin/env python3
"""Verify packaged qualification evidence, without building or running physics code.

Only the documented inventory schemas below are interpreted. Historical native
binary identities and original build paths are provenance, not copied artifacts.
--check-live-inputs additionally checks recorded repository/fixture source hashes;
that optional check can fail legitimately after later source development.
--tracked additionally requires every inventory and indexed artifact to exist in
Git's index with the same bytes, catching ignored logs and unstaged updates.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

INVENTORIES = {
    "comparison-backends/evidence-index.json": "files",
    "poles-only/index.json": "artifacts",
    "tirust-install/index.json": "artifacts",
    "tirust-bridge/evidence-index.json": "files",
    "madloop-adapter/debug/manifest.json": "portable_evidence_sha256",
    "madloop-adapter/release/manifest.json": "portable_evidence_sha256",
}
GROUPS = {name.split("/", 1)[0] for name in INVENTORIES}
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
BINARY_MAGIC = (b"\x7fELF", b"MZ", b"!<arch>\n", b"\0asm", b"\xfe\xed\xfa\xce",
                b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe")
NOTICE_ARCHIVE = "tirust-install/tirust-dependency-notices.tar.gz"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_prefix(data: bytes) -> bool:
    return b"\0" in data or any(data.startswith(magic) for magic in BINARY_MAGIC)


def verify(qualification: Path, repository: Path, live: bool = False, tracked: bool = False) -> dict:
    errors: list[str] = []
    counts: dict[str, int] = {}
    snapshots: dict[str, dict] = {}
    checked_files: set[Path] = set()
    index_files: set[Path] = set()

    def load(name: str) -> dict | None:
        path = qualification / name
        try:
            data = json.loads(path.read_text())
            index_files.add(path.resolve())
            if not isinstance(data, dict):
                raise ValueError("index must contain a JSON object")
            return data
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: cannot read index ({exc})")
            return None

    def check_file(label: str, path: Path, expected: str, size: int | None = None):
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            errors.append(f"{label}: malformed SHA256")
            return
        try:
            if not path.is_file():
                raise ValueError("missing regular file")
            before = path.stat()
            actual = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                errors.append(f"{label}: file changed while checking; retry after writer finishes")
            if actual != expected.lower():
                errors.append(f"{label}: SHA256 mismatch")
            if size is not None and (type(size) is not int or size != after.st_size):
                errors.append(f"{label}: byte-count mismatch")
            checked_files.add(path.resolve())
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: {exc}")

    def portable_path(base: Path, name: str) -> Path:
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("artifact path is not a contained relative path")
        result = base / name
        if not result.resolve().is_relative_to(base.resolve()):
            raise ValueError("artifact symlink escapes its evidence directory")
        return result

    for name, field in INVENTORIES.items():
        data = load(name)
        if data is None:
            continue
        snapshots[name] = data
        entries = data.get(field)
        if not isinstance(entries, dict) or not entries:
            errors.append(f"{name}: missing nonempty {field} inventory")
            continue
        counts[name] = len(entries)
        for relative, record in entries.items():
            label = f"{name}: {relative}"
            try:
                path = portable_path((qualification / name).parent, relative)
            except (TypeError, ValueError, OSError) as exc:
                errors.append(f"{label}: {exc}")
                continue
            if isinstance(record, str):
                expected, size = record, None
            elif isinstance(record, dict):
                expected, size = record.get("sha256"), record.get("bytes", record.get("size"))
            else:
                errors.append(f"{label}: unsupported inventory record")
                continue
            check_file(label, path, expected, size)

    # The comparison package also records README, notices and its index itself.
    sums = qualification / "comparison-backends/SHA256SUMS"
    try:
        lines = sums.read_text().splitlines()
        index_files.add(sums.resolve())
        counts["comparison-backends/SHA256SUMS"] = len(lines)
        for number, line in enumerate(lines, 1):
            try:
                expected, name = line.split(None, 1)
                path = portable_path(sums.parent, name.removeprefix("*"))
                check_file(f"{sums.name}:{number}", path, expected)
            except (ValueError, OSError) as exc:
                errors.append(f"{sums.name}:{number}: {exc}")
    except OSError as exc:
        errors.append(f"comparison-backends/SHA256SUMS: {exc}")

    # Compiled executables and libraries must stay outside the evidence package.
    # One intentional license/source-notice archive is inspected without extraction.
    scanned = 0
    for group in sorted(GROUPS):
        for path in sorted((qualification / group).rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                errors.append(f"{path.relative_to(qualification)}: generated Python binary in evidence")
                continue
            name = path.relative_to(qualification).as_posix()
            if not path.resolve().is_relative_to(qualification.resolve()):
                errors.append(f"{name}: evidence symlink escapes qualification directory")
                continue
            scanned += 1
            try:
                with path.open("rb") as stream:
                    prefix = stream.read(4096)
                if name == NOTICE_ARCHIVE:
                    with tarfile.open(path, "r:gz") as archive:
                        for member in archive:
                            entry = PurePosixPath(member.name)
                            if entry.is_absolute() or ".." in entry.parts or not (member.isfile() or member.isdir()):
                                errors.append(f"{name}: unsafe or nonregular archive member {member.name}")
                                continue
                            if member.isfile():
                                with archive.extractfile(member) as stream:
                                    if binary_prefix(stream.read(4096)):
                                        errors.append(f"{name}: binary content in notice member {member.name}")
                elif binary_prefix(prefix) or prefix.startswith((b"\x1f\x8b", b"PK\x03\x04")):
                    errors.append(f"{name}: unexpected binary/archive content")
            except (OSError, tarfile.TarError) as exc:
                errors.append(f"{name}: cannot inspect content ({exc})")

    live_count = 0
    superseded: set[str] = set()
    if live:
        entries: list[tuple[str, str]] = []
        for name, field in (("comparison-backends/evidence-index.json", "repository_inputs"),
                            ("tirust-install/index.json", "source_sha256")):
            snapshot = snapshots.get(name, {})
            sources = dict(snapshot.get(field, {}))
            # The install record preserves original build identities and names
            # separately qualified later source changes. Check the recorded
            # amendment itself before using its current-source hashes.
            if name == "tirust-install/index.json":
                for amendment in snapshot.get("post_install_changes", []):
                    try:
                        record_path = portable_path((qualification / name).parent, amendment["record"])
                        recorded = json.loads(record_path.read_text())
                        changed = amendment["source_sha256"]
                        if changed != recorded.get("source_sha256"):
                            raise ValueError("source hashes disagree with the named amendment record")
                        superseded.update(key for key in changed if key in sources)
                        sources.update(changed)
                    except (OSError, KeyError, TypeError, ValueError) as exc:
                        errors.append(f"{name}: invalid source amendment ({exc})")
            entries.extend(sources.items())
        bridge = load("tirust-bridge/qualification.json")
        if bridge:
            entries.extend((record["path"], record["sha256"])
                           for record in bridge.get("source_state", {}).get("bridge_artifacts", []))
            entries.extend((record["source"], record["sha256"])
                           for record in bridge.get("fixtures", []))
        for name, expected in entries:
            path = (repository / name).resolve()
            if not (path.is_relative_to(repository) or path.is_relative_to(repository.parent / "references")):
                errors.append(f"live input {name}: outside repository/reference roots")
                continue
            check_file(f"live input {name}", path, expected)
            live_count += 1

    tracked_count = 0
    if tracked:
        # Verify staged bytes as well as index membership: a tracked path with
        # newer unstaged content would not reproduce the audited checkout.
        required = sorted(index_files | {path for path in checked_files if path.is_relative_to(qualification)})
        try:
            names = [path.relative_to(repository).as_posix() for path in required]
            listed = subprocess.run(["git", "-C", str(repository), "ls-files", "--stage", "-z", "--", *names],
                                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            staged = {}
            for entry in listed.split(b"\0"):
                if not entry:
                    continue
                header, name = entry.split(b"\t", 1)
                mode, oid, stage = header.decode().split()
                if stage == "0":
                    staged[name.decode()] = (mode, oid)
            oids = sorted({staged[name][1] for name in names if name in staged})
            contents = {}
            if oids:
                result = subprocess.run(["git", "-C", str(repository), "cat-file", "--batch"],
                                        input=("\n".join(oids) + "\n").encode(), check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stream = io.BytesIO(result.stdout)
                for oid in oids:
                    actual_oid, kind, length = stream.readline().decode().split()
                    if actual_oid != oid or kind != "blob":
                        raise ValueError("Git index entry does not identify a blob")
                    value = stream.read(int(length))
                    if stream.read(1) != b"\n" or len(value) != int(length):
                        raise ValueError("incomplete Git blob response")
                    contents[oid] = hashlib.sha256(value).hexdigest()
            for path, name in zip(required, names):
                if name not in staged:
                    errors.append(f"{name}: indexed evidence is not tracked/staged in Git")
                    continue
                mode, oid = staged[name]
                if mode not in ("100644", "100755"):
                    errors.append(f"{name}: indexed evidence is not a regular Git file")
                elif contents[oid] != digest(path):
                    errors.append(f"{name}: staged content differs from audited working-tree bytes")
                tracked_count += 1
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"Git index verification failed ({exc})")

    return {"status": "passed" if not errors else "failed", "inventory_entries": counts,
            "unique_hashed_files": len(checked_files), "content_scanned_files": scanned,
            "live_source_inputs_checked": live_count,
            "superseded_capture_sources": sorted(superseded),
            "git_index_verification": "enabled" if tracked else "not requested; filesystem evidence only",
            "git_index_files_checked": tracked_count, "errors": errors,
            "scope": "Recorded file integrity, contained paths and binary-content inspection; not numerical or performance validation. Historical native binaries are not rerun or required."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, default=Path(__file__).resolve().parent / "qualification")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check-live-inputs", action="store_true")
    parser.add_argument("--tracked", action="store_true", help="also require matching staged Git blobs for inventories and indexed evidence")
    args = parser.parse_args()
    report = verify(args.qualification.resolve(), args.repository.resolve(), args.check_live_inputs, args.tracked)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
