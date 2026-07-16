#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Stage a freshly built wheel's native resources for source-tree tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
RELEASE_TOOLS = ROOT / "tools" / "release"
sys.path.insert(0, str(RELEASE_TOOLS))

from _artifacts import audit_wheel  # noqa: E402
from _common import (  # noqa: E402
    ReleaseError,
    build_mode,
    clean_environment,
    exactly_one,
    external_temporary_directory,
    is_relative_to,
    run,
)

SOURCE_PACKAGE = ROOT / "src" / "pyamplicol"
SOURCE_BUILD_INFO = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
_SDK_PREFIXES = (
    "pyamplicol/_sdk/fortran/",
    "pyamplicol/_sdk/include/",
    "pyamplicol/_sdk/lib/",
)
_SDK_FILES = {
    "pyamplicol/_sdk/link.json",
    "pyamplicol/_sdk/metadata.json",
}


def _wheel_identity(
    members: Mapping[str, bytes],
) -> tuple[str, str]:
    metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise ReleaseError("source runtime wheel must contain one METADATA file")
    metadata = BytesParser(policy=policy.default).parsebytes(members[metadata_names[0]])
    name = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", ""))
    if name != "pyamplicol" or not version:
        raise ReleaseError("source runtime wheel has an invalid distribution identity")
    return name, version


def _safe_destination(source_package: Path, member: str) -> Path:
    relative = PurePosixPath(member).relative_to("pyamplicol")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ReleaseError(f"unsafe source runtime wheel member: {member}")
    destination = source_package.joinpath(*relative.parts).resolve()
    if not is_relative_to(destination, source_package.resolve()):
        raise ReleaseError(f"source runtime wheel member escapes package: {member}")
    return destination


def _write_atomic(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.chmod(mode)
    os.replace(temporary, path)


def stage_runtime(
    wheel: Path,
    *,
    source_package: Path = SOURCE_PACKAGE,
    source_build_info: Path = SOURCE_BUILD_INFO,
    mode: str,
    audit: bool = True,
) -> dict[str, object]:
    wheel = wheel.resolve(strict=True)
    if audit:
        audit_wheel(wheel, mode=mode)
    with zipfile.ZipFile(wheel) as archive:
        member_modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist()
            if not info.is_dir()
        }
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }
    _, version = _wheel_identity(members)

    extension_names = [
        name
        for name in members
        if name.startswith("pyamplicol/_rusticol.")
        and name.rsplit(".", 1)[-1] in {"dylib", "pyd", "so"}
    ]
    if len(extension_names) != 1:
        raise ReleaseError(
            "source runtime wheel must contain one Rusticol extension, found "
            f"{len(extension_names)}"
        )
    sdk_metadata_name = "pyamplicol/_sdk/metadata.json"
    try:
        sdk_metadata = json.loads(members[sdk_metadata_name])
        target = str(sdk_metadata["target"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReleaseError("source runtime wheel has invalid SDK metadata") from error
    if not target or "/" in target or "\\" in target:
        raise ReleaseError(f"source runtime wheel has an unsafe target: {target!r}")

    selftest_prefix = f"pyamplicol/assets/selftest/{target}/"
    selected = {
        name
        for name in members
        if name in extension_names
        or name in _SDK_FILES
        or name.startswith(_SDK_PREFIXES)
        or name.startswith(selftest_prefix)
    }
    if not any(name.startswith(selftest_prefix) for name in selected):
        raise ReleaseError(
            f"source runtime wheel has no self-test fixture for {target}"
        )
    for member in sorted(selected):
        _write_atomic(
            _safe_destination(source_package, member),
            members[member],
            mode=member_modes[member] or 0o644,
        )

    build_info_name = "pyamplicol/_build_info.json"
    if build_info_name in members:
        build_info = members[build_info_name]
    else:
        build_info = (
            json.dumps(
                {
                    "schema_version": 1,
                    "publishable": mode == "release",
                    "version": version,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    _write_atomic(source_build_info, build_info)
    return {
        "mode": mode,
        "target": target,
        "version": version,
        "wheel": wheel.name,
        "staged_file_count": len(selected) + 1,
    }


def build_and_stage(*, python: Path, mode: str) -> dict[str, object]:
    with external_temporary_directory("pyamplicol-source-runtime-") as temporary:
        run(
            [python, "-m", "build", "--wheel", "--outdir", temporary],
            cwd=ROOT,
            env=clean_environment(mode=mode),
        )
        wheel = exactly_one(
            list(temporary.glob("pyamplicol-*.whl")),
            "fresh source-test wheel",
        )
        return stage_runtime(wheel, mode=mode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    report = build_and_stage(python=args.python, mode=mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
