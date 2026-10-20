#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build an optimized loop native runtime and stage it into a source checkout.

``build`` compiles ``rusticol-python`` (the ``pyamplicol._rusticol`` extension)
and ``rusticol-loop-sdk`` (rlib/static/shared SDK libraries) with Cargo's
``release`` profile and copies them into a new, never-overwritten directory
with ``build-info.json`` (native build-input digest, source revision, effective
profile). Cargo's ``CARGO_PROFILE_RELEASE_*`` variables select the codegen; the
default here is 16 codegen units without cross-crate LTO, which keeps rebuilds
practical for long developer runs. Set them explicitly (for example
``CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_LTO=thin``) to
reproduce the shipped wheel codegen. The profile does not enter the native
build-input digest, so optimized and dev builds of one source are
interchangeable at staging.

Cargo always builds into a target directory private to this profile: a shared
``CARGO_TARGET_DIR`` is used concurrently by dev-profile builds of other
checkouts, which block on its lock and overwrite ``release/`` artifacts with
other codegen between our build and copy. ``--target-dir`` selects the
directory; by default it is ``cargo-loop-native-<profile>`` next to a set
``CARGO_TARGET_DIR``, or Cargo's checkout-local ``target/`` otherwise.
``build-info.json`` also records ``RUSTFLAGS``/``CARGO_ENCODED_RUSTFLAGS`` and
whether the source tree had uncommitted changes (provenance only).

``stage`` publishes such a directory into a checkout's ``src/pyamplicol``
through the source-runtime contract of ``prepare_source_runtime.py``. It
refuses a build whose digest differs from the checkout's native inputs and
verifies the staged module in a fresh interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_source_runtime as runtime  # noqa: E402

_CODEGEN_DEFAULTS = {
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
    "CARGO_PROFILE_RELEASE_LTO": "false",
}
# Cargo artifact file name -> name in the output directory.
_OUTPUTS = {
    "lib_rusticol.so": "_rusticol.so",
    "librusticol_loop_sdk.rlib": "librusticol_loop_sdk.rlib",
    "librusticol_loop_sdk.a": "librusticol_loop_sdk.a",
    "librusticol_loop_sdk.so": "librusticol_loop_sdk.so",
}
_VERIFY = """
import sys
sys.path.insert(0, sys.argv[1])
from pyamplicol import _rusticol
from pyamplicol._internal.versions import verify_native_module
verify_native_module(_rusticol)
assert _rusticol._loop.NativeVirtualRuntime
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_label(env: dict[str, str]) -> str:
    units = env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"]
    return f"release-cu{units}-lto-{env['CARGO_PROFILE_RELEASE_LTO']}"


def _target_directory(root: Path, env: dict[str, str], explicit: Path | None) -> Path:
    shared = (
        Path(env["CARGO_TARGET_DIR"]).resolve() if env.get("CARGO_TARGET_DIR") else None
    )
    if explicit is not None:
        target = explicit.resolve()
    elif shared is not None:
        target = shared.parent / f"cargo-loop-native-{_profile_label(env)}"
    else:
        target = (root / "target").resolve()
    if target == shared:
        raise SystemExit(
            f"error: {target} is the shared CARGO_TARGET_DIR; "
            "use a private --target-dir"
        )
    return target


def build(
    root: Path, output: Path, target_dir: Path | None = None
) -> dict[str, object]:
    output = output.resolve()
    if output.exists():
        raise SystemExit(f"error: build output already exists: {output}")
    digest = runtime._native_build_inputs_digest(root)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
        ).strip()
    )
    env = dict(os.environ, PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256=digest)
    for key, value in _CODEGEN_DEFAULTS.items():
        env.setdefault(key, value)
    target = _target_directory(root, env, target_dir)
    env.pop("CARGO_TARGET_DIR", None)
    command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--target-dir",
        str(target),
        "-p",
        "rusticol-python",
        "-p",
        "rusticol-loop-sdk",
        "--features",
        "rusticol-python/extension-module,rusticol-python/numpy",
        "--message-format=json-render-diagnostics",
    ]
    process = subprocess.Popen(
        command, cwd=root, env=env, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout is not None
    artifacts: dict[str, Path] = {}
    for line in process.stdout:
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact":
            for filename in map(Path, message["filenames"]):
                if filename.name in _OUTPUTS:
                    artifacts[_OUTPUTS[filename.name]] = filename
    if process.wait() != 0:
        raise SystemExit(f"error: cargo build failed ({process.returncode})")
    missing = sorted(set(_OUTPUTS.values()) - set(artifacts))
    if missing:
        raise SystemExit(f"error: missing cargo outputs: {missing}")
    if runtime._native_build_inputs_digest(root) != digest:
        raise SystemExit("error: native build inputs changed during the build")

    with (root / "Cargo.toml").open("rb") as stream:
        version = tomllib.load(stream)["workspace"]["package"]["version"]
    output.mkdir(parents=True)
    for name, source in artifacts.items():
        (output / name).write_bytes(source.read_bytes())
    info = {
        "native_build_inputs_sha256": digest,
        "source_revision": revision,
        "source_dirty": dirty,
        "source_checkout": str(root),
        "cargo_target_dir": str(target),
        "rustflags": {
            key: env[key]
            for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")
            if key in env
        },
        "version": version.replace("-dev.", ".dev"),
        "profile": {"name": "release"}
        | {key: env[key] for key in sorted(env) if key.startswith("CARGO_PROFILE_")},
        "rustc": subprocess.check_output(["rustc", "-V"], text=True).strip(),
        "files": {name: _sha256(output / name) for name in sorted(artifacts)},
    }
    (output / "build-info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )
    for path in output.iterdir():
        path.chmod(0o555 if path.suffix == ".so" else 0o444)
    return info


def stage(build_directory: Path, root: Path) -> dict[str, object]:
    info = json.loads((build_directory / "build-info.json").read_text())
    digest = info["native_build_inputs_sha256"]
    if runtime._native_build_inputs_digest(root) != digest:
        raise SystemExit(
            f"error: {build_directory} was built from other native inputs than {root}"
        )
    data = (build_directory / "_rusticol.so").read_bytes()
    extension_sha256 = hashlib.sha256(data).hexdigest()
    if extension_sha256 != info["files"]["_rusticol.so"]:
        raise SystemExit("error: extension does not match its build-info.json")
    package = root / "src" / "pyamplicol"
    build_info = root / ".artifacts" / "source-runtime" / "_build_info.json"
    with runtime._publication_lock(build_info.parent):
        marker = build_info.parent / ".staging"
        runtime._write_atomic(marker, b"incomplete\n", mode=0o600)
        for old in runtime._native_extension_inventory(package):
            if old.name != "_rusticol.so":
                old.unlink()
        runtime._write_atomic(package / "_rusticol.so", data, mode=0o755)
        payload = {
            "schema_version": 1,
            "publishable": False,
            "version": info["version"],
            "source_revision": info["source_revision"],
            "source_checkout": str(root),
            "native_build_inputs_sha256": digest,
            "native_build": {
                "directory": str(build_directory),
                "profile": info["profile"],
            },
            "source_runtime": {
                "mode": "candidate",
                "extension_name": "_rusticol.so",
                "extension_sha256": extension_sha256,
                "native_build_inputs_sha256": digest,
            },
        }
        runtime._write_atomic(
            build_info,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        marker.unlink()
    subprocess.run(
        [sys.executable, "-c", _VERIFY, str(root / "src")], cwd=root, check=True
    )
    return {"staged": str(package / "_rusticol.so"), "native_digest": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="build into a new directory")
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument(
        "--target-dir",
        type=Path,
        help="private Cargo target directory (default: cargo-loop-native-<profile> "
        "next to a set CARGO_TARGET_DIR, else the checkout's target/)",
    )
    stage_parser = commands.add_parser("stage", help="stage a build into a checkout")
    stage_parser.add_argument("build_directory", type=Path)
    for sub in (build_parser, stage_parser):
        sub.add_argument("--source", type=Path, default=ROOT, help="source checkout")
    args = parser.parse_args(argv)
    root = args.source.resolve()
    if args.command == "build":
        report = build(root, args.output, args.target_dir)
    else:
        report = stage(args.build_directory.resolve(), root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
