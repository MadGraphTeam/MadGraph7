#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Keep the pyAmpliCol versions in step with the MadGraph7 VERSION file.

The PyPI version of pyamplicol is ``<MG version>.<N>``: ``N = 1`` is the release
published with that MadGraph version, ``N >= 2`` are standalone pyamplicol
releases on top of it. Cargo only accepts semver, so the Rust crates carry the
plain MadGraph version and the build backend hands the Python version to Rust.

    sync_version.py            keep N (or reset it to 1 after a MadGraph bump)
    sync_version.py --bump     N -> N + 1, for a standalone pyamplicol release
    sync_version.py --version 1.0.0.1
    sync_version.py --check [--version V]   only report a mismatch (exit 1)
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
MG_VERSION_FILE = ROOT.parent / "VERSION"
WORKSPACE_CRATES = ("pyamplicol", "rusticol-capi", "rusticol-core", "rusticol-python")


def madgraph_version() -> str:
    text = MG_VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*(\S+)", text)
    if match is None:
        raise RuntimeError(f"no version in {MG_VERSION_FILE}")
    return match.group(1)


def package_version() -> str:
    """The pyamplicol (PyPI) version currently written in pyproject.toml."""

    with (ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"].get("version", ""))


def release_number(version: str, mg_version: str) -> int | None:
    """N for ``version == f"{mg_version}.{N}"`` with N >= 1, else None."""

    prefix = f"{mg_version}."
    if not version.startswith(prefix):
        return None
    suffix = version[len(prefix) :]
    if not suffix.isdigit() or suffix.startswith("0"):
        return None
    return int(suffix)


def default_package_version(mg_version: str) -> str:
    """Keep the current release number N, or start at 1 for a new MG version."""

    number = release_number(package_version(), mg_version)
    return f"{mg_version}.{number or 1}"


def _section_value(text: str, section: str, version: str) -> str:
    # Replace `version = "..."` in the given TOML section.
    pattern = re.compile(
        rf'(?ms)(^\[{re.escape(section)}\]\n(?:(?!^\[).)*?^version = )"[^"]*"'
    )
    return pattern.sub(rf'\g<1>"{version}"', text, count=1)


def _cargo_lock(text: str, version: str) -> str:
    for crate in WORKSPACE_CRATES:
        text = re.sub(
            rf'(?m)(^name = "{crate}"\nversion = )"[^"]*"',
            rf'\g<1>"{version}"',
            text,
        )
    return text


def _versions_py(text: str, version: str) -> str:
    return re.sub(
        r'(def package_version\(default: str = )"[^"]*"',
        rf'\g<1>"{version}"',
        text,
    )


# Files carrying the Cargo (MadGraph) version and the Python (PyPI) version.
CARGO_EDITS = {
    Path("Cargo.toml"): lambda text, v: _section_value(text, "workspace.package", v),
    Path("Cargo.lock"): _cargo_lock,
}
PYTHON_EDITS = {
    Path("pyproject.toml"): lambda text, v: _section_value(text, "project", v),
    Path("dependencies/release-lock.toml"): lambda text, v: _section_value(
        text, "project", v
    ),
    Path("src/pyamplicol/_internal/versions.py"): _versions_py,
}


# The shipped self-test artifacts record the package version that made them.
SELFTEST_ARTIFACTS = (
    Path("src/pyamplicol/assets/selftest/portable-64le/artifact"),
    Path("src/pyamplicol/assets/selftest_virtual/ew/artifact"),
    Path("src/pyamplicol/assets/selftest_virtual/li/artifact"),
)


def sync(mg_version: str, version: str, *, check: bool) -> list[Path]:
    """Return the files whose version differs, updating them unless checking."""

    from prepare_selftest_fixture import stamp_package_version

    stale: list[Path] = []
    for edits, value in ((CARGO_EDITS, mg_version), (PYTHON_EDITS, version)):
        for relative, edit in edits.items():
            path = ROOT / relative
            text = path.read_text(encoding="utf-8")
            updated = edit(text, value)
            if updated != text:
                stale.append(relative)
                if not check:
                    path.write_text(updated, encoding="utf-8")
    for relative in SELFTEST_ARTIFACTS:
        if stamp_package_version(ROOT / relative, version, check=check):
            stale.append(relative)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--version", help="pyamplicol version, <MG version>.<N>")
    group.add_argument("--bump", action="store_true", help="increase N by one")
    args = parser.parse_args(argv)
    if args.check and args.bump:
        parser.error("--bump cannot be combined with --check")

    mg_version = madgraph_version()
    if args.version:
        version = args.version
    elif args.check:
        version = package_version()
    elif args.bump:
        number = release_number(package_version(), mg_version)
        version = f"{mg_version}.{(number or 0) + 1}"
    else:
        version = default_package_version(mg_version)
    if release_number(version, mg_version) is None:
        print(
            f"pyamplicol version {version!r} is not <MadGraph version>.<N> "
            f"with MadGraph version {mg_version} and N >= 1",
            file=sys.stderr,
        )
        return 1

    stale = sync(mg_version, version, check=args.check)
    if args.check and stale:
        names = ", ".join(path.as_posix() for path in stale)
        print(
            f"pyamplicol version is not {version} (Cargo {mg_version}) in: {names}",
            file=sys.stderr,
        )
        return 1
    for path in stale:
        print(f"set version in {path.as_posix()}")
    if not args.check:
        print(f"pyamplicol {version} (Cargo crates {mg_version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
