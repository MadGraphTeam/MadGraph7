# SPDX-License-Identifier: 0BSD
"""Prepare the portable 64-bit installed-package physics self-test fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "selftest"
PORTABLE_TEMPLATE = "portable-64le"
COMPATIBLE_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
)


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _artifact_id(manifest: Mapping[str, object]) -> str:
    content = dict(manifest)
    content.pop("artifact_id", None)
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_version() -> str:
    with (ROOT / "Cargo.toml").open("rb") as stream:
        cargo = tomllib.load(stream)
    try:
        version = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Cargo.toml has no workspace package version") from exc
    if not isinstance(version, str) or not version:
        raise RuntimeError("Cargo workspace package version is invalid")
    return version.replace("-dev.", ".dev")


def _walk_evaluator_manifests(value: object) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("kind") == "symjit-application-evaluator":
            found.append(value)
        for child in value.values():
            found.extend(_walk_evaluator_manifests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_evaluator_manifests(child))
    return found


def _strip_symbolica_fallbacks(artifact: Path, manifest: dict[str, Any]) -> None:
    removed: set[str] = set()
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    execution_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "evaluator-manifest"
    ]
    for payload in execution_payloads:
        relative = payload.get("path")
        if not isinstance(relative, str) or not relative.endswith("/execution.json"):
            continue
        path = artifact / relative
        execution = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise RuntimeError(f"invalid evaluator manifest: {relative}")
        for evaluator in _walk_evaluator_manifests(execution):
            state = evaluator.get("evaluator_state_path")
            if isinstance(state, str):
                removed.add((Path(relative).parent / state).as_posix())
            evaluator["evaluator_state_path"] = None
            evaluator["evaluator_state_runtime_capability"] = None
        path.write_bytes(_canonical_json(execution))
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
    if not removed:
        raise RuntimeError("source artifact has no Symbolica fallback evaluator states")
    for relative in sorted(removed):
        path = artifact / relative
        if not path.is_file():
            raise RuntimeError(f"missing Symbolica fallback state: {relative}")
        path.unlink()
    manifest["payloads"] = [
        payload
        for payload in payloads
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("path"), str)
            and payload["path"] in removed
        )
    ]


def _sanitize_configuration_paths(artifact: Path, manifest: dict[str, Any]) -> None:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    roles = {"configuration-requested", "configuration-effective"}
    found: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("role") not in roles:
            continue
        role = str(payload["role"])
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError(f"self-test {role} payload has no path")
        path = artifact / relative
        text = path.read_text(encoding="utf-8")
        updated = _sanitize_generation_output(text, role)
        path.write_text(updated, encoding="utf-8")
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
        found.add(role)
    if found != roles:
        raise RuntimeError(
            "self-test artifact lacks requested/effective configurations"
        )


def _sanitize_generation_output(text: str, role: str) -> str:
    pattern = r'(?m)^output\s*=\s*"[^"]*"$'
    matches = tuple(re.finditer(pattern, text))
    if len(matches) > 1:
        raise RuntimeError(f"self-test {role} has multiple generation outputs")
    if not matches:
        return text
    return re.sub(pattern, 'output = "."', text, count=1)


def _retarget_portable_manifest(
    manifest: dict[str, Any],
    *,
    source_target: str,
) -> None:
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test artifact producer metadata is invalid")
    target = producer.get("target")
    if not isinstance(target, dict) or target.get("triple") != source_target:
        raise RuntimeError("self-test artifact target differs from the active runtime")
    target["triple"] = PORTABLE_TEMPLATE
    target["cpu_features"] = []

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("self-test artifact payload inventory is invalid")
    evaluator_targets = 0
    for payload in payloads:
        if not isinstance(payload, dict):
            raise RuntimeError("self-test artifact payload entry is invalid")
        payload_target = payload.get("target")
        if payload_target is None:
            continue
        if (
            not isinstance(payload_target, dict)
            or payload_target.get("triple") != source_target
        ):
            raise RuntimeError("self-test evaluator target metadata is invalid")
        payload_target["triple"] = PORTABLE_TEMPLATE
        payload_target["cpu_features"] = []
        evaluator_targets += 1
    if evaluator_targets == 0:
        raise RuntimeError("self-test artifact has no target-tagged evaluator state")


def _complex_pairs(values: Sequence[object]) -> list[list[float]]:
    return [
        [float(complex(value).real), float(complex(value).imag)] for value in values
    ]


def _complex_sequences_close(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right, strict=True):
        actual = complex(left_value)
        expected = complex(right_value)
        if not (
            math.isclose(actual.real, expected.real, rel_tol=1.0e-12, abs_tol=1.0e-14)
            and math.isclose(
                actual.imag, expected.imag, rel_tol=1.0e-12, abs_tol=1.0e-14
            )
        ):
            return False
    return True


def prepare(source: Path, destination: Path | None = None) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir() or not (source / "artifact.json").is_file():
        raise RuntimeError("self-test source must be a schema-v3 artifact directory")

    from pyamplicol import Runtime

    native = importlib.import_module("pyamplicol._rusticol")
    target = str(native.target_info().triple)
    if target not in COMPATIBLE_TARGETS:
        raise RuntimeError(f"unsupported portable self-test target: {target}")
    runtime = Runtime.load(source, mute_warnings=True)
    momentum_loader = getattr(runtime._backend, "validation_momenta", None)
    if not callable(momentum_loader):
        raise RuntimeError("artifact runtime cannot expose its validation point")
    momenta = momentum_loader()
    if momenta is None:
        raise RuntimeError("artifact has no deterministic validation point")
    total = runtime.evaluate(momenta)
    resolved = runtime.evaluate_resolved(momenta)
    if not _complex_sequences_close(resolved.total(), total):
        raise RuntimeError("resolved self-test values do not reproduce evaluate()")

    output = (destination or (FIXTURE_ROOT / PORTABLE_TEMPLATE)).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"self-test destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.staging")
    if temporary.exists():
        raise RuntimeError(f"stale self-test staging directory exists: {temporary}")
    try:
        artifact = temporary / "artifact"
        shutil.copytree(source, artifact, symlinks=False)
        manifest_path = artifact / "artifact.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
            raise RuntimeError("self-test source is not a schema-v3 artifact")
        producer = manifest.get("producer")
        runtime_metadata = manifest.get("runtime")
        if not isinstance(producer, dict) or not isinstance(runtime_metadata, dict):
            raise RuntimeError(
                "self-test artifact producer/runtime metadata is invalid"
            )
        if producer.get("target", {}).get("triple") != target:
            raise RuntimeError(
                "self-test artifact target differs from the active runtime"
            )
        _strip_symbolica_fallbacks(artifact, manifest)
        _sanitize_configuration_paths(artifact, manifest)
        _retarget_portable_manifest(manifest, source_target=target)
        version = _workspace_version()
        producer["version"] = version
        runtime_metadata["engine_version"] = version
        manifest["artifact_id"] = _artifact_id(manifest)
        manifest_path.write_bytes(_canonical_json(manifest))
        expected = {
            "schema_version": 1,
            "artifact_path": "artifact",
            "target": PORTABLE_TEMPLATE,
            "compatible_targets": list(COMPATIBLE_TARGETS),
            "source_generation_target": target,
            "serialization": {
                "kind": "symjit-application-mir-v3",
                "endianness": "little",
                "word_size_bits": 64,
                "load_behavior": "recompile-mir-for-loading-host",
            },
            "process_id": runtime.physics.process_id,
            "process": runtime.physics.process,
            "momenta": momenta,
            "resolved_shape": list(resolved.shape),
            "helicity_ids": list(resolved.helicity_ids),
            "color_ids": list(resolved.color_ids),
            "total": _complex_pairs(total),
        }
        (temporary / "expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    output = prepare(args.source, args.destination)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
