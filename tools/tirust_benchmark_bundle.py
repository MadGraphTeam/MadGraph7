#!/usr/bin/env python3
"""Build, restore and verify the TIRust benchmark bundle.

The benchmark workspace ``docs/benchmarkfolder`` (campaigns, physical cards,
frozen points, report sources and evidence) is not part of the MadGraph source
tree. It is distributed as a release asset (see ``docs/tirust-benchmarks.md``)
and restored into a checkout, where it is ignored by git, before a
requalification, the local timing or a report build.

  build    --revision REV --output DIR [--include DIR] [--name NAME]
           Archive docs/benchmarkfolder at REV byte for byte, overlay the files
           of --include (which must not replace archived files), write
           RELEASE-PROVENANCE.json and SHA256SUMS, and a deterministic .tar.xz.
  restore  [--archive PATH] [--sha256 HEX] [--force]
           Check the archive against the SHA256 pinned in
           tirust_benchmark_bundle.json, then extract it into the checkout.
           Existing identical files are kept; differing files are refused
           unless --force. Downloads the pinned release when --archive is
           omitted.
  verify   Check docs/benchmarkfolder against its SHA256SUMS.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "docs/benchmarkfolder"
SUMS = "SHA256SUMS"
PROVENANCE = "RELEASE-PROVENANCE.json"

# The published release: repository, tag, archive name and SHA256.
PIN = Path(__file__).with_suffix(".json")


def pinned_release() -> dict:
    pin = json.loads(PIN.read_text())
    pin["url"] = (f"https://github.com/{pin['repository']}/releases/download/"
                  f"{pin['tag']}/{pin['archive']}")
    return pin


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True)
    return result.stdout if binary else result.stdout.decode().strip()


def member_path(name: str) -> PurePosixPath:
    """Return a safe relative path inside the workspace, or raise."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    if path.parts[:2] != tuple(WORKSPACE.split("/")) or len(path.parts) < 3:
        raise ValueError(f"archive member outside {WORKSPACE}: {name!r}")
    return path


def archived_files(root: Path, revision: str) -> dict[str, tuple[bytes, int]]:
    """Read docs/benchmarkfolder at revision as {path: (bytes, mode)}."""
    stream = git(root, "archive", "--format=tar", revision, WORKSPACE, binary=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(stream), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir() or member.name == "pax_global_header":
                continue
            if not member.isfile():
                raise ValueError(f"{member.name}: only regular files are bundled")
            member_path(member.name)
            files[member.name] = (archive.extractfile(member).read(),
                                  0o755 if member.mode & 0o111 else 0o644)
    if not files:
        raise ValueError(f"{WORKSPACE} is empty at {revision}")
    return files


def build(root: Path, revision: str, output: Path, include: Path | None = None,
          name: str | None = None) -> tuple[Path, str]:
    commit = git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    mtime = int(git(root, "show", "-s", "--format=%ct", commit))
    files = archived_files(root, commit)
    added = {}
    if include is not None:
        for path in sorted(p for p in include.rglob("*") if p.is_file()):
            target = f"{WORKSPACE}/{path.relative_to(include).as_posix()}"
            member_path(target)
            if target in files:
                raise ValueError(f"{target}: --include must not replace archived files")
            data = path.read_bytes()
            files[target] = (data, 0o755 if os.access(path, os.X_OK) else 0o644)
            added[target] = sha256(data)
    for reserved in (PROVENANCE, SUMS):
        if f"{WORKSPACE}/{reserved}" in files:
            raise ValueError(f"{WORKSPACE}/{reserved} is written by the builder")
    name = name or f"madloop-tirust-benchmark-{commit[:10]}.tar.xz"
    provenance = {
        "bundle": name,
        "source_revision": commit,
        "source_tree": WORKSPACE,
        "source_tree_object": git(root, "rev-parse", f"{commit}:{WORKSPACE}"),
        "archived_files": len(files) - len(added),
        "added_files": added,
        "builder_sha256": file_sha256(Path(__file__)),
        "note": "Archived files are byte-identical to the source revision.",
    }
    files[f"{WORKSPACE}/{PROVENANCE}"] = (
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(), 0o644)
    prefix = len(WORKSPACE) + 1
    sums = "".join(f"{sha256(data)}  {path[prefix:]}\n"
                   for path, (data, _) in sorted(files.items()))
    files[f"{WORKSPACE}/{SUMS}"] = (sums.encode(), 0o644)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / name
    with tarfile.open(archive_path, mode="w:xz", preset=9,
                      format=tarfile.PAX_FORMAT) as archive:
        for path, (data, mode) in sorted(files.items()):
            info = tarfile.TarInfo(path)
            info.size, info.mode, info.mtime = len(data), mode, mtime
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return archive_path, file_sha256(archive_path)


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tirust-benchmark-bundle"})
    with urllib.request.urlopen(request) as response, open(target.with_suffix(".part"), "wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
    target.with_suffix(".part").replace(target)
    return target


def restore(root: Path, archive_path: Path, expected: str, force: bool = False) -> dict:
    actual = file_sha256(archive_path)
    if actual != expected:
        raise ValueError(f"{archive_path}: SHA256 {actual}, expected {expected}")
    plan, conflicts = [], []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"{member.name}: only regular files are restored")
            relative = member_path(member.name)
            data = archive.extractfile(member).read()
            target = root.joinpath(*relative.parts)
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"{target}: not a regular file")
            if target.exists():
                if target.read_bytes() == data:
                    plan.append((target, None, member.mode))
                    continue
                conflicts.append(str(relative))
            plan.append((target, data, member.mode))
    if conflicts and not force:
        shown = ", ".join(conflicts[:5]) + (" ..." if len(conflicts) > 5 else "")
        raise ValueError(f"{len(conflicts)} existing files differ ({shown}); "
                         "use --force to replace them")
    counts = {"written": 0, "unchanged": 0, "replaced": len(conflicts)}
    for target, data, mode in plan:
        if data is None:
            counts["unchanged"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(data)
        os.chmod(handle.name, 0o755 if mode & 0o111 else 0o644)
        os.replace(handle.name, target)
        counts["written"] += 1
    counts["written"] -= counts["replaced"]
    verify(root)
    return counts


def verify(root: Path) -> int:
    base = root / WORKSPACE
    lines = (base / SUMS).read_text().splitlines()
    bad = []
    for line in lines:
        digest, name = line.split("  ", 1)
        path = base / name
        if not path.is_file() or file_sha256(path) != digest:
            bad.append(name)
    if bad:
        raise ValueError(f"{len(bad)} files missing or changed in {WORKSPACE} "
                         f"(first: {bad[0]})")
    return len(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    built = commands.add_parser("build")
    built.add_argument("--revision", required=True)
    built.add_argument("--output", type=Path, required=True)
    built.add_argument("--include", type=Path)
    built.add_argument("--name")
    restored = commands.add_parser("restore")
    restored.add_argument("--archive", type=Path)
    restored.add_argument("--sha256", help="expected SHA256 (default: the pinned release)")
    restored.add_argument("--force", action="store_true")
    commands.add_parser("verify")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            path, digest = build(ROOT, args.revision, args.output, args.include, args.name)
            print(f"{digest}  {path}")
        elif args.command == "restore":
            pin = pinned_release()
            archive = args.archive or download(
                pin["url"], ROOT / ".tirust-build/benchmark-bundle" / pin["archive"])
            counts = restore(ROOT, archive, args.sha256 or pin["sha256"], args.force)
            print(f"{WORKSPACE}: {counts['written']} written, {counts['replaced']} "
                  f"replaced, {counts['unchanged']} unchanged; checksums verified")
        else:
            print(f"{WORKSPACE}: {verify(ROOT)} files match {SUMS}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
