#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Check or refresh licenses/rust/ against the crates linked into the native libraries.

Linked = the normal (non-build, non-dev, non-proc-macro) dependency closure of
the wheel's extension (rusticol-python, features extension-module,numpy) and of
the C SDK archive (rusticol-capi), taken as the UNION over every release Rust
target (``_artifacts.RELEASE_TARGETS``): a crate linked on one platform only
(for example mach2 on macOS) still ships its licence texts in every wheel.

For each linked crates.io crate, licenses/rust/<crate>-<version>/ holds the
files of the crate root whose name starts with LICENSE, LICENCE, COPYING,
UNLICENSE, NOTICE or COPYRIGHT (case-insensitive, nested files included, files
of nested crates excluded), plus the manifest's license-file. A crate
published without such files keeps a hand-written LICENSE-NOTE directory.
A linked git crate is covered by its repository's notice file under licenses/
(REPOSITORY_NOTICES; a git dependency from another repository fails the
check until its notice is added); its own licence files not reproduced
verbatim in that notice also go to licenses/rust/<crate>-<version>/. Path
crates are pyAmpliCol's own (0BSD).

Usage (Cargo sources must be available; --offline is passed to Cargo):
  python -I tools/release/rust_licence_inventory.py            # check, exit 1 on drift
  python -I tools/release/rust_licence_inventory.py --apply    # copy the wanted files
  python -I tools/release/rust_licence_inventory.py --json out.json
The tool never deletes: stale directories are printed for a literal-path git rm.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _artifacts import RELEASE_TARGETS  # noqa: E402

LINKED_PACKAGES = (
    ("rusticol-python", "extension-module,numpy"),
    ("rusticol-capi", ""),
)
# Linked git repositories -> the notice file under licenses/ covering their crates.
REPOSITORY_NOTICES = {
    "ValentinHirschi/symbolica": "Symbolica.txt",
    "alphal00p/gammaloop": "GammaLoop-crates.txt",
    "alphal00p/ninjars": "NinjaRS.txt",
    "alphal00p/oneloopmaster": "oneloopmaster.txt",
    "huashengshao/TIRust": "TIRust.txt",
}
_REPOSITORY = re.compile(r"github\.com/([^/]+/[^/?#]+?)(?:\.git)?[?#)]")
_LICENCE_NAME = re.compile(
    r"^(licen[cs]e|copying|unlicense|notice|copyright)([-_.][a-z0-9.+-]*)?$", re.I
)
_CODE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".h", ".js", ".json", ".lock"}
    | {".py", ".rs", ".sh", ".toml", ".yaml", ".yml"}
)


def _cargo(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["cargo", *arguments, "--offline", "--locked"], cwd=root, text=True
    )


def linked_crates(root: Path) -> list[dict[str, object]]:
    """Linked crates over every release target, with the targets linking each."""

    metadata = json.loads(
        _cargo(root, "metadata", "--format-version", "1", "--all-features")
    )
    packages = {(p["name"], p["version"]): p for p in metadata["packages"]}
    proc_macros = {
        key
        for key, package in packages.items()
        if any("proc-macro" in target["kind"] for target in package["targets"])
    }
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for rust_target in sorted(set(RELEASE_TARGETS.values())):
        for package, features in LINKED_PACKAGES:
            command = ["tree", "-p", package, "--target", rust_target]
            command += ["-e", "normal,no-proc-macro", "--prefix", "none"]
            command += ["--format", "{p}|{l}"]
            if features:
                command += ["--features", features]
            for line in _cargo(root, *command).splitlines():
                spec, _, declared = line.replace(" (*)", "").strip().partition("|")
                if not spec:
                    continue
                name, version, *source = spec.split(" ")
                key = (name, version.removeprefix("v"))
                if key in proc_macros:
                    continue
                package_metadata = packages.get(key, {})
                row = rows.setdefault(
                    key,
                    {
                        "name": key[0],
                        "version": key[1],
                        "source": " ".join(source),
                        "license": declared or package_metadata.get("license"),
                        "license_file": package_metadata.get("license_file"),
                        "manifest": package_metadata.get("manifest_path"),
                        "targets": [],
                    },
                )
                if rust_target not in row["targets"]:
                    row["targets"].append(rust_target)
    return [rows[key] for key in sorted(rows)]


def licence_files(row: dict[str, object]) -> list[tuple[Path, Path]]:
    crate_root = Path(str(row["manifest"])).parent
    nested = {
        manifest.parent
        for manifest in crate_root.rglob("Cargo.toml")
        if manifest.parent != crate_root
    }
    files = sorted(
        path
        for path in crate_root.rglob("*")
        if path.is_file()
        and _LICENCE_NAME.match(path.name)
        and path.suffix.lower() not in _CODE_SUFFIXES
        and "target" not in path.relative_to(crate_root).parts
        and not nested.intersection(path.parents)
    )
    if row.get("license_file"):
        extra = crate_root / str(row["license_file"])
        if extra.is_file() and extra not in files:
            files.append(extra)
    return [(path, path.relative_to(crate_root)) for path in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", type=Path, help="write the linked-crate inventory")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    destination = root / "licenses" / "rust"

    inventory = linked_crates(root)
    if arguments.json:
        arguments.json.write_text(json.dumps(inventory, indent=1) + "\n")
    registry = [row for row in inventory if not row["source"]]
    git = [row for row in inventory if str(row["source"]).startswith("(http")]
    present = {path.name for path in destination.iterdir() if path.is_dir()}
    wanted: dict[str, list[tuple[Path, Path]]] = {}
    without_files: list[str] = []
    uncovered: list[str] = []
    for row in registry + git:
        key = f"{row['name']}-{row['version']}"
        files = licence_files(row)
        if row in git:
            match = _REPOSITORY.search(str(row["source"]))
            notice = REPOSITORY_NOTICES.get(match.group(1) if match else "")
            if notice is None or not (root / "licenses" / notice).is_file():
                uncovered.append(f"{key} {row['source']}")
                continue
            text = (root / "licenses" / notice).read_bytes()
            files = [
                (path, rel) for path, rel in files if path.read_bytes() not in text
            ]
            if not files:
                if key in present:
                    without_files.append(key)
                continue
        if files:
            wanted[key] = files
        else:
            without_files.append(key)
    kept = sorted(key for key in without_files if key in present)
    unrecorded = sorted(key for key in without_files if key not in present)
    stale = sorted(present - wanted.keys() - set(kept))
    added = sorted(wanted.keys() - present)
    changed = sorted(
        key
        for key in wanted.keys() & present
        if any(
            not (destination / key / relative).is_file()
            or (destination / key / relative).read_bytes() != source.read_bytes()
            for source, relative in wanted[key]
        )
    )
    print(
        f"linked crates: {len(inventory)} over {len(set(RELEASE_TARGETS.values()))} "
        f"release targets; crates.io: {len(registry)}; git: {len(git)}; "
        f"licenses/rust directories wanted: {len(wanted)}"
    )
    platform_only = sorted(
        f"{row['name']}-{row['version']} ({', '.join(row['targets'])})"
        for row in registry
        if len(row["targets"]) < len(set(RELEASE_TARGETS.values()))
    )
    print("platform-specific:", ", ".join(platform_only) or "none")
    print("kept LICENSE-NOTE directories:", ", ".join(kept) or "none")
    print("missing:", ", ".join(added) or "none")
    print("changed:", ", ".join(changed) or "none")
    print("stale:", " ".join(stale) or "none")
    print("without licence files and no directory:", ", ".join(unrecorded) or "none")
    print("git crates without a repository notice:", ", ".join(uncovered) or "none")
    if arguments.apply:
        for key in sorted(added + changed):
            for source, relative in wanted[key]:
                target = destination / key / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return 1 if stale or unrecorded or uncovered else 0
    return 1 if added or changed or stale or unrecorded or uncovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
