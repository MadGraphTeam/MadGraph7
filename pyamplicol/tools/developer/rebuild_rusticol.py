#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Rebuild and restage only Rusticol for a candidate source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = ROOT / "src" / "pyamplicol"
SOURCE_RUNTIME_INFO = ROOT / ".artifacts" / "source-runtime" / "_build_info.json"
RUSTICOL_MANIFEST = Path("rust/crates/rusticol-python/Cargo.toml")
NATIVE_SUFFIXES = (".dylib", ".pyd", ".so")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build_backend"))
import _pyamplicol_build as build_backend  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "bin" / "python",
    )
    parser.add_argument(
        "--maturin",
        type=Path,
        default=ROOT / ".venv" / "bin" / "maturin",
    )
    parser.add_argument(
        "--cargo-test",
        action="append",
        default=[],
        metavar="FILTER",
        help="run a rusticol-core test filter in the contributor overlay",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="run requested cargo tests without rebuilding the extension",
    )
    return parser


def _read_candidate_info() -> dict[str, object]:
    try:
        payload = json.loads(SOURCE_RUNTIME_INFO.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            "candidate source-runtime metadata is unavailable; "
            "run `just dev-install` once"
        ) from error
    if not isinstance(payload, dict) or payload.get("publishable") is not False:
        raise RuntimeError("Rusticol-only rebuilds require a staged candidate runtime")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("candidate source-runtime metadata has no package version")
    return payload


def _native_extension() -> Path:
    extensions = sorted(
        path
        for path in SOURCE_PACKAGE.glob("_rusticol.*")
        if path.is_file() and path.name.endswith(NATIVE_SUFFIXES)
    )
    if len(extensions) != 1:
        raise RuntimeError(
            f"expected one staged Rusticol extension, found {len(extensions)}"
        )
    return extensions[0]


def _wheel_extension(wheel: Path) -> tuple[str, bytes, int]:
    with zipfile.ZipFile(wheel) as archive:
        candidates = [
            info
            for info in archive.infolist()
            if info.filename.startswith("pyamplicol/_rusticol.")
            and info.filename.endswith(NATIVE_SUFFIXES)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one Rusticol extension in {wheel}, found {len(candidates)}"
            )
        info = candidates[0]
        return Path(info.filename).name, archive.read(info), info.external_attr >> 16


def _install_extension(wheel: Path) -> Path:
    name, payload, mode = _wheel_extension(wheel)
    destination = SOURCE_PACKAGE / name
    with tempfile.NamedTemporaryFile(
        dir=SOURCE_PACKAGE,
        prefix=f".{name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.chmod(mode or 0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    for stale in SOURCE_PACKAGE.glob("_rusticol.*"):
        if stale != destination and stale.name.endswith(NATIVE_SUFFIXES):
            stale.unlink()
    return destination


def _stage_untracked_source(overlay: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not build_backend._is_allowlisted(relative)
            or build_backend._is_excluded(relative)
        ):
            continue
        source = ROOT / relative
        mode = source.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"untracked build input is not a regular file: {source}")
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)


def _publish_info(
    payload: dict[str, object],
    *,
    extension: Path,
    native_digest: str,
) -> None:
    payload["native_build_inputs_sha256"] = native_digest
    payload["source_runtime"] = {
        "extension_name": extension.name,
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "native_build_inputs_sha256": native_digest,
    }
    SOURCE_RUNTIME_INFO.parent.mkdir(parents=True, exist_ok=True)
    temporary = SOURCE_RUNTIME_INFO.with_name(
        f".{SOURCE_RUNTIME_INFO.name}.rusticol-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, SOURCE_RUNTIME_INFO)


def main() -> int:
    args = _parser().parse_args()
    _read_candidate_info()
    with build_backend._overlay("candidate") as (overlay, _):
        _stage_untracked_source(overlay)
        target = ROOT / ".artifacts" / "rusticol-rebuild" / "target"
        payload = json.loads(
            (overlay / "src" / "pyamplicol" / "_build_info.json").read_text(
                encoding="utf-8"
            )
        )
        native_digest = str(payload["native_build_inputs_sha256"])
        environment = build_backend._clean_environment(
            {
                "CARGO_TARGET_DIR": str(target),
                "PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256": native_digest,
                "PYAMPLICOL_PACKAGE_VERSION": str(payload["version"]),
            }
        )
        for test_filter in args.cargo_test:
            subprocess.run(
                [
                    "cargo",
                    "test",
                    "--release",
                    "--locked",
                    "--offline",
                    "--manifest-path",
                    str(overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"),
                    test_filter,
                ],
                cwd=overlay,
                env=environment,
                check=True,
            )
        if args.test_only:
            if not args.cargo_test:
                raise RuntimeError("--test-only requires at least one --cargo-test")
            return 0
        with tempfile.TemporaryDirectory(prefix="pyamplicol-rusticol-wheel-") as raw:
            wheel_directory = Path(raw)
            subprocess.run(
                [
                    str(args.maturin),
                    "build",
                    "--release",
                    "--locked",
                    "--offline",
                    "--manifest-path",
                    str(overlay / RUSTICOL_MANIFEST),
                    "--features",
                    "extension-module,numpy",
                    "--interpreter",
                    str(args.python),
                    "--out",
                    str(wheel_directory),
                ],
                cwd=overlay,
                env=environment,
                check=True,
            )
            wheels = sorted(wheel_directory.glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"expected one Rusticol wheel, found {len(wheels)}")
            extension = _install_extension(wheels[0])
            subprocess.run(
                [
                    str(args.python),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    str(wheels[0]),
                ],
                cwd=ROOT,
                check=True,
            )
    _publish_info(payload, extension=extension, native_digest=native_digest)
    subprocess.run(
        [
            str(args.python),
            "-c",
            (
                "from pyamplicol import _rusticol; "
                "from pyamplicol._internal.versions import verify_native_module; "
                "verify_native_module(_rusticol); "
                "print(_rusticol.package_version(), "
                "_rusticol.native_build_inputs_sha256())"
            ),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
