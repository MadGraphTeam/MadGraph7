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
PORTABLE_OPTIMIZATION_LEVEL = 1
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


def _compiled_model_payload(
    manifest: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "compiled-model"
    ]
    if len(matches) != 1:
        raise RuntimeError("self-test artifact must contain one compiled model")
    payload = matches[0]
    relative = payload.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("self-test compiled-model payload has no path")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("self-test compiled-model payload path is unsafe")
    return payload, path


def _validate_source_compiled_model(
    source: Path,
    manifest: Mapping[str, object],
) -> None:
    """Require the source artifact to match the active model compiler exactly."""

    from pyamplicol.models.loading import load_compiled_model

    _payload, relative = _compiled_model_payload(manifest)
    load_compiled_model(source / relative)


def _normalize_compiled_model_version(
    artifact: Path,
    manifest: Mapping[str, object],
    *,
    version: str,
) -> None:
    """Retarget the embedded compiler fingerprint to the release package version."""

    payload, relative = _compiled_model_payload(manifest)
    path = artifact / relative
    try:
        compiled_model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test compiled model is invalid") from error
    if not isinstance(compiled_model, dict):
        raise RuntimeError("self-test compiled model must be an object")
    producer = compiled_model.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test compiled-model producer is invalid")
    producer["pyamplicol"] = version
    path.write_bytes(_canonical_json(compiled_model))
    payload["sha256"] = _sha256(path)
    payload["size_bytes"] = path.stat().st_size


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


def _validate_portable_evaluator_configuration(
    artifact: Path,
    manifest: Mapping[str, object],
) -> int:
    """Require architecture-neutral MIR before retargeting an evaluator state."""

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    evaluator_count = 0
    for payload in payloads:
        if (
            not isinstance(payload, dict)
            or payload.get("role") != "evaluator-manifest"
            or not str(payload.get("path", "")).endswith("/execution.json")
        ):
            continue
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("self-test evaluator manifest has no path")
        execution = json.loads((artifact / relative).read_text(encoding="utf-8"))
        for evaluator in _walk_evaluator_manifests(execution):
            evaluator_count += 1
            if (
                evaluator.get("compiler_type") != "native"
                or evaluator.get("translation_mode") != "indirect"
            ):
                raise RuntimeError(
                    "portable self-test requires native indirect SymJIT evaluators"
                )
            if evaluator.get("optimization_level") != PORTABLE_OPTIMIZATION_LEVEL:
                raise RuntimeError(
                    "portable self-test source must use SymJIT optimization level "
                    f"{PORTABLE_OPTIMIZATION_LEVEL}; O2/O3 MIR may contain "
                    "source-architecture register allocation"
                )
    if evaluator_count == 0:
        raise RuntimeError("self-test artifact has no SymJIT evaluator manifests")
    return PORTABLE_OPTIMIZATION_LEVEL


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
    removed_loose: set[str] = set()
    for relative in sorted(removed):
        path = artifact / relative
        if path.is_file():
            path.unlink()
            removed_loose.add(relative)
    removed_packed = removed - removed_loose
    if removed_packed:
        _strip_packed_symbolica_fallbacks(artifact, manifest, removed_packed)
    manifest["payloads"] = [
        payload
        for payload in payloads
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("path"), str)
            and payload["path"] in removed_loose
        )
    ]


def _strip_packed_symbolica_fallbacks(
    artifact: Path,
    manifest: dict[str, Any],
    removed: set[str],
) -> None:
    """Remove exact fallback states from a packed evaluator container."""

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )

    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict):
        raise RuntimeError("artifact extensions are invalid")
    extension = extensions.get("evaluator_payload_container")
    if not isinstance(extension, dict):
        missing = ", ".join(sorted(removed))
        raise RuntimeError(f"missing Symbolica fallback state: {missing}")
    relative = extension.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("evaluator payload container has no path")
    container_path = artifact / relative

    with PacbinReader.open(container_path, verify_payloads=True) as reader:
        by_path = {member.logical_path: member for member in reader.members}
        missing = sorted(removed - by_path.keys())
        if missing:
            raise RuntimeError(
                "missing Symbolica fallback state: " + ", ".join(missing)
            )
        invalid = sorted(
            path
            for path in removed
            if by_path[path].kind is not PacbinMemberKind.SYMBOLICA_EXACT_STATE
        )
        if invalid:
            raise RuntimeError(
                "Symbolica fallback state has an invalid packed kind: "
                + ", ".join(invalid)
            )
        retained = tuple(
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                reader.open_member_stream(member.logical_path),
            )
            for member in reader.members
            if member.logical_path not in removed
        )
        if not retained:
            raise RuntimeError("portable self-test evaluator container is empty")
        index = write_pacbin_atomic(container_path, retained)

    extension["member_count"] = len(index.members)
    extension["unpacked_size_bytes"] = sum(
        member.length for member in index.members
    )
    extension["index_sha256"] = index.index_sha256

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError("artifact must declare one evaluator payload container")
    payload = matches[0]
    payload["sha256"] = _sha256(container_path)
    payload["size_bytes"] = container_path.stat().st_size


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

    try:
        source_manifest = json.loads(
            (source / "artifact.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test source manifest is invalid") from error
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema_version") != 3
    ):
        raise RuntimeError("self-test source is not a schema-v3 artifact")
    _validate_source_compiled_model(source, source_manifest)

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
        portable_optimization_level = _validate_portable_evaluator_configuration(
            artifact,
            manifest,
        )
        _strip_symbolica_fallbacks(artifact, manifest)
        _sanitize_configuration_paths(artifact, manifest)
        _retarget_portable_manifest(manifest, source_target=target)
        version = _workspace_version()
        _normalize_compiled_model_version(
            artifact,
            manifest,
            version=version,
        )
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
                "source_optimization_level": portable_optimization_level,
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
