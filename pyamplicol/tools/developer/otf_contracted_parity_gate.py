#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run one bounded OTF-contracted numerical parity cell.

The caller is responsible for placing each invocation under the repository
memory watchdog.  A recurrence/NLC/p16-or-p200 invocation creates a local
authority; an OTF/NLC/p16 invocation authenticates that result, while OTF/full/p16
authenticates either a retained MadGraph wave record or the frozen acceptance
fixture.  Every invocation generates exactly one process, evaluates one point
once, and refuses to replace evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

KIND = "pyamplicol-otf-contracted-parity-cell-v1"
SEED = 101
TOLERANCE = Decimal("1e-10")


class GateError(RuntimeError):
    """The bounded parity cell was malformed or failed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--prepared-model", required=True, type=Path)
    parser.add_argument("--process", required=True)
    parser.add_argument("--process-name", required=True)
    parser.add_argument("--n-final", required=True, type=int)
    parser.add_argument("--family-id", type=int)
    parser.add_argument("--point", type=Path)
    parser.add_argument("--accuracy", required=True, choices=("nlc", "full"))
    parser.add_argument(
        "--mode", required=True, choices=("recurrence", "on-the-fly")
    )
    parser.add_argument("--precision", required=True, type=int, choices=(16, 200))
    parser.add_argument(
        "--query-construction-cores",
        type=int,
        choices=(1, 4),
        default=1,
    )
    authority = parser.add_mutually_exclusive_group()
    authority.add_argument("--recurrence-authority", type=Path)
    authority.add_argument("--madgraph-authority", type=Path)
    authority.add_argument("--acceptance-fixture", type=Path)
    parser.add_argument("--fixture-case-id")
    return parser


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="ascii"),
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GateError(f"non-finite JSON token {value!r} in {path}")
            ),
        )
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read JSON document {path}: {error}") from error


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError(f"{where} must be a JSON object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise GateError(f"{where} must be a JSON array")
    return value


def _decimal(value: Any, where: str) -> Decimal:
    if isinstance(value, bool):
        raise GateError(f"{where} must be a finite number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise GateError(f"{where} must be a finite number") from error
    if not result.is_finite():
        raise GateError(f"{where} must be finite")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lower_hex(value: Any, length: int, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GateError(f"{where} must be {length} lowercase hexadecimal digits")
    return value


def _candidate_identity(package: Any, distribution_version: str) -> dict[str, Any]:
    """Authenticate a contributor build without requiring a clean revision."""

    package_root = Path(package.__file__).resolve().parent
    installed_info_path = package_root / "_build_info.json"
    installed = _object(_read_json(installed_info_path), "installed build-info")
    version = installed.get("version")
    if not isinstance(version, str) or not version or version != distribution_version:
        raise GateError("installed build-info version differs from the distribution")
    native_digest = _lower_hex(
        installed.get("native_build_inputs_sha256"),
        64,
        "installed native build-input digest",
    )
    source_revision = installed.get("source_revision")
    if source_revision is not None:
        source_revision = _lower_hex(source_revision, 40, "installed source revision")
    source_checkout = installed.get("source_checkout")
    if not isinstance(source_checkout, str) or not source_checkout:
        raise GateError("installed build-info has no source checkout")
    source_root = Path(source_checkout).expanduser().resolve(strict=True)
    if source_root != Path(__file__).resolve().parents[2]:
        raise GateError("installed candidate belongs to a different source checkout")
    source_info_path = (
        source_root / ".artifacts" / "source-runtime" / "_build_info.json"
    )
    source = _object(_read_json(source_info_path), "source-runtime build-info")
    if (
        source.get("version") != version
        or source.get("source_revision") != source_revision
        or source.get("native_build_inputs_sha256") != native_digest
    ):
        raise GateError("installed and source-runtime build-info identities differ")
    runtime = _object(source.get("source_runtime"), "source-runtime contract")
    extension_sha256 = _lower_hex(
        runtime.get("extension_sha256"), 64, "source-runtime extension digest"
    )
    if (
        runtime.get("mode") != "candidate"
        or runtime.get("native_build_inputs_sha256") != native_digest
    ):
        raise GateError("source-runtime native identity differs from build-info")

    from pyamplicol import _rusticol as native

    extension_path = Path(native.__file__).resolve(strict=True)
    if runtime.get("extension_name") != extension_path.name:
        raise GateError(
            "source-runtime extension name differs from the installed module"
        )
    if _sha256(extension_path) != extension_sha256:
        raise GateError("installed native extension digest differs from build-info")
    native_version_operation = getattr(native, "package_version", None)
    native_digest_operation = getattr(native, "native_build_inputs_sha256", None)
    if not callable(native_version_operation) or not callable(native_digest_operation):
        raise GateError("native extension has no version/build-input identity")
    native_version = native_version_operation()
    if (
        not isinstance(native_version, str)
        or native_version.replace("-dev.", ".dev") != version
    ):
        raise GateError("native extension version differs from build-info")
    if native_digest_operation() != native_digest:
        raise GateError("native extension build-input digest differs from build-info")

    completed = subprocess.run(
        ("git", "-C", os.fspath(source_root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise GateError("cannot record current Git HEAD provenance")
    git_head = _lower_hex(completed.stdout.strip(), 40, "current Git HEAD")
    return {
        "kind": "installed-contributor-native-candidate-v1",
        "version": version,
        "native_build_inputs_sha256": native_digest,
        "native_extension_sha256": extension_sha256,
        "source_revision": source_revision,
        "current_git_head_provenance_only": git_head,
        "installed_build_info": {
            "path": os.fspath(installed_info_path),
            "sha256": _sha256(installed_info_path),
        },
        "source_runtime_build_info": {
            "path": os.fspath(source_info_path),
            "sha256": _sha256(source_info_path),
        },
        "native_extension": {
            "path": os.fspath(extension_path),
            "sha256": extension_sha256,
        },
    }


def _point_rows(value: Any, where: str) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for index, raw_row in enumerate(_sequence(value, where)):
        row = _sequence(raw_row, f"{where}[{index}]")
        if len(row) != 4:
            raise GateError(f"{where}[{index}] must contain four components")
        try:
            values = tuple(float(component) for component in row)
        except (TypeError, ValueError, OverflowError) as error:
            raise GateError(f"{where}[{index}] contains a non-number") from error
        if any(not math.isfinite(component) for component in values):
            raise GateError(f"{where}[{index}] contains a non-finite number")
        rows.append(values)
    if not rows:
        raise GateError(f"{where} must not be empty")
    return tuple(rows)


def _point_digest(point: tuple[tuple[float, ...], ...]) -> str:
    canonical = json.dumps(
        [[repr(component) for component in row] for row in point],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _same_point(
    left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]
) -> bool:
    return left == right


def _fixture_case(
    path: Path, case_id: str, process: str
) -> tuple[tuple[tuple[float, ...], ...], Decimal, dict[str, Any]]:
    fixture = _object(_read_json(path), "acceptance fixture")
    if (
        fixture.get("kind") != "pyamplicol-ufo-sm-numerical-acceptance"
        or fixture.get("schema_version") != Decimal(1)
    ):
        raise GateError("acceptance fixture identity is unsupported")
    comparison = _object(fixture.get("comparison"), "fixture comparison")
    if (
        comparison.get("relative_tolerance") != "0.0000000001"
        or comparison.get("absolute_tolerance") is not None
        or comparison.get("full_candidate_precision_digits") != Decimal(200)
    ):
        raise GateError("acceptance fixture does not carry the strict p200 contract")
    cases = _sequence(
        fixture.get("extra_full_colour_cases"), "fixture extra_full_colour_cases"
    )
    matches = [
        _object(raw, "fixture extra case")
        for raw in cases
        if isinstance(raw, Mapping) and raw.get("id") == case_id
    ]
    if len(matches) != 1 or matches[0].get("process") != process:
        raise GateError(f"fixture case {case_id!r} does not match {process!r}")
    case = matches[0]
    expected = _decimal(
        _object(case.get("expected"), "fixture expected").get("full"),
        "fixture full target",
    )
    point = _point_rows(case.get("momenta"), "fixture momenta")
    return point, expected, {
        "kind": "frozen-madgraph-acceptance-fixture",
        "path": os.fspath(path),
        "sha256": _sha256(path),
        "case_id": case_id,
    }


def _point_from_path(path: Path) -> tuple[tuple[float, ...], ...]:
    return _point_rows(_read_json(path), f"point {path}")


def _madgraph_target(
    path: Path,
    *,
    process: str,
    n_final: int,
    point: tuple[tuple[float, ...], ...],
) -> tuple[Decimal, dict[str, Any]]:
    authority = _object(_read_json(path), "MadGraph authority")
    if (
        authority.get("process") != process
        or authority.get("n_final") != Decimal(n_final)
        or authority.get("seed") != Decimal(SEED)
    ):
        raise GateError("retained MadGraph authority identity differs from this cell")
    declared_point = authority.get("point_path")
    if not isinstance(declared_point, str):
        raise GateError("retained MadGraph authority has no point_path")
    authority_point_path = Path(declared_point).expanduser().resolve(strict=True)
    authority_point = _point_from_path(authority_point_path)
    if not _same_point(point, authority_point):
        raise GateError("worker point differs from the retained MadGraph point")
    expected = _decimal(authority.get("madgraph_value"), "MadGraph value")
    measurement_path = path.parent / "madgraph-measurement.json"
    measurement = _object(_read_json(measurement_path), "MadGraph measurement")
    validation = _object(measurement.get("validation"), "MadGraph validation")
    if (
        measurement.get("status") != "ok"
        or validation.get("status") != "ok"
        or validation.get("method")
        != "independent-madgraph-tree-level-oracle"
        or _decimal(measurement.get("matrix_element"), "MadGraph measurement value")
        != expected
    ):
        raise GateError("retained MadGraph measurement is not a successful authority")
    return expected, {
        "kind": "retained-authenticated-madgraph-wave",
        "path": os.fspath(path),
        "sha256": _sha256(path),
        "measurement_path": os.fspath(measurement_path),
        "measurement_sha256": _sha256(measurement_path),
        "point_path": os.fspath(authority_point_path),
        "point_sha256": _sha256(authority_point_path),
    }


def _recurrence_target(
    path: Path,
    *,
    process: str,
    point_digest: str,
    producer_version: str,
) -> tuple[Decimal, dict[str, Any]]:
    record = _object(_read_json(path), "recurrence authority")
    lane = _object(record.get("lane"), "recurrence authority lane")
    actual = _object(record.get("actual"), "recurrence authority actual")
    producer = _object(record.get("producer"), "recurrence authority producer")
    authority_precision = lane.get("precision")
    if (
        record.get("kind") != KIND
        or record.get("status") != "anchor"
        or record.get("process") != process
        or record.get("point_digest") != point_digest
        or lane.get("accuracy") != "nlc"
        or lane.get("mode") != "recurrence"
        or authority_precision not in {Decimal(16), Decimal(200)}
        or producer.get("version") != producer_version
    ):
        raise GateError("recurrence authority does not match this OTF/NLC cell")
    imaginary = _decimal(actual.get("imaginary"), "recurrence imaginary part")
    if imaginary != 0:
        raise GateError("recurrence authority is not exactly real")
    return _decimal(actual.get("real"), "recurrence real part"), {
        "kind": f"same-candidate-recurrence-p{authority_precision}",
        "precision": int(authority_precision),
        "path": os.fspath(path),
        "sha256": _sha256(path),
    }


def _actual(value: Any) -> tuple[Decimal, Decimal, str]:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise GateError("matrix element is non-finite")
        return value, Decimal(0), str(value)
    try:
        number = complex(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GateError("matrix element is not numeric") from error
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise GateError("matrix element is non-finite")
    return Decimal(repr(number.real)), Decimal(repr(number.imag)), repr(value)


def _comparison(real: Decimal, imaginary: Decimal, expected: Decimal) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 512
        magnitude = (real * real + imaginary * imaginary).sqrt()
        difference = ((real - expected) ** 2 + imaginary**2).sqrt()
        scale = max(magnitude, abs(expected))
        bound = TOLERANCE * scale
        passed = difference == 0 if scale == 0 else difference <= bound
        residual = Decimal(0) if scale == 0 else difference / scale
    return {
        "expected": str(expected),
        "absolute_difference": str(difference),
        "scale": str(scale),
        "conditioned_residual": str(residual),
        "relative_tolerance": str(TOLERANCE),
        "bound": str(bound),
        "passed": passed,
    }


def _run_config(accuracy: str, mode: str, cores: int) -> Any:
    from pyamplicol.config import (
        Action,
        ColorAccuracy,
        ColorConfig,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        JITConfig,
        LCFlowLayout,
        RelationDiscoveryMode,
        RunConfig,
    )

    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy=ColorAccuracy(accuracy),
            lc_flow_layout=LCFlowLayout.TOPOLOGY_REPLAY,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode(mode),
            optimization=EvaluatorOptimizationConfig(cores=cores),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to replace evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, ensure_ascii=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _validate_arguments(args: argparse.Namespace) -> None:
    expected_lane = {
        ("nlc", "recurrence", 16): "anchor",
        ("nlc", "recurrence", 200): "anchor",
        ("nlc", "on-the-fly", 16): "recurrence",
        ("full", "on-the-fly", 16): "full",
    }.get((args.accuracy, args.mode, args.precision))
    if expected_lane is None:
        raise GateError(
            "unsupported gate lane; only NLC recurrence-p16/p200 and OTF-p16"
        )
    has_recurrence = args.recurrence_authority is not None
    has_full = (
        args.madgraph_authority is not None
        or args.acceptance_fixture is not None
    )
    if expected_lane == "anchor" and (has_recurrence or has_full):
        raise GateError("recurrence-p200 anchor must not declare an authority")
    if expected_lane == "recurrence" and not has_recurrence:
        raise GateError("OTF/NLC requires --recurrence-authority")
    if expected_lane == "full" and not has_full:
        raise GateError("OTF/full requires a MadGraph authority")
    if (args.acceptance_fixture is None) != (args.fixture_case_id is None):
        raise GateError("--acceptance-fixture and --fixture-case-id are paired")
    if args.acceptance_fixture is None and args.point is None:
        raise GateError("non-fixture cells require --point")
    if args.n_final < 1 or args.n_final > 6:
        raise GateError("n_final must be in 1..6")
    if args.mode == "recurrence" and args.query_construction_cores != 1:
        raise GateError("recurrence authorities use exactly one core")


def _authenticate_artifact(
    artifact: Path,
    *,
    process_name: str,
    mode: str,
    query_construction_cores: int,
    candidate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate cheap persisted boundaries before any runtime evaluation."""

    manifest_path = artifact / "artifact.json"
    manifest = _object(_read_json(manifest_path), "artifact manifest")
    manifest_producer = _object(manifest.get("producer"), "artifact producer")
    artifact_producer_version = manifest_producer.get("version")
    artifact_source_revision = manifest_producer.get("git_revision")
    artifact_native_digest = manifest_producer.get(
        "native_build_inputs_sha256"
    )
    if (
        manifest_producer.get("distribution") != "pyamplicol"
        or artifact_producer_version != candidate_identity["version"]
        or artifact_source_revision != candidate_identity["source_revision"]
        or artifact_native_digest
        != candidate_identity["native_build_inputs_sha256"]
    ):
        raise GateError("artifact producer differs from installed candidate")
    effective_path = artifact / "config" / "effective.toml"
    try:
        effective = tomllib.loads(effective_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise GateError(f"cannot read effective configuration: {error}") from error
    effective_cores = (
        effective.get("evaluator", {}).get("optimization", {}).get("cores")
    )
    if effective_cores != query_construction_cores:
        raise GateError(
            "effective configuration does not authenticate the requested core budget"
        )
    query_construction_threads = None
    if mode == "on-the-fly":
        execution_path = artifact / "processes" / process_name / "execution.json"
        execution = _object(_read_json(execution_path), "OTF execution manifest")
        runtime_options = _object(
            execution.get("runtime_options"), "OTF runtime options"
        )
        query_construction_threads = runtime_options.get(
            "query_construction_threads"
        )
        if query_construction_threads != Decimal(query_construction_cores):
            raise GateError(
                "OTF execution manifest does not authenticate the requested "
                "query-construction thread budget"
            )
    return {
        "producer": {
            "version": artifact_producer_version,
            "source_revision_provenance": artifact_source_revision,
            "native_build_inputs_sha256": artifact_native_digest,
            "artifact_id": manifest.get("artifact_id"),
            "artifact_manifest_sha256": _sha256(manifest_path),
        },
        "effective_config": {
            "path": os.fspath(effective_path),
            "sha256": _sha256(effective_path),
            "evaluator_optimization_cores": effective_cores,
        },
        "runtime_options": {
            "query_construction_threads": (
                None
                if query_construction_threads is None
                else int(query_construction_threads)
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_arguments(args)
    output = args.output.expanduser().resolve(strict=False)
    preflight_path = output.with_name(f"{output.stem}.preflight.json")
    artifact = args.artifact.expanduser().resolve(strict=False)
    prepared = args.prepared_model.expanduser().resolve(strict=True)
    if not prepared.is_file():
        raise GateError(f"prepared model is not a regular file: {prepared}")
    if output.exists() or output.is_symlink():
        raise GateError(f"result already exists: {output}")
    if preflight_path.exists() or preflight_path.is_symlink():
        raise GateError(f"preflight evidence already exists: {preflight_path}")
    if artifact.exists() or artifact.is_symlink():
        raise GateError(f"artifact already exists: {artifact}")

    fixture_target: Decimal | None = None
    fixture_lineage: dict[str, Any] | None = None
    if args.acceptance_fixture is not None:
        fixture_path = args.acceptance_fixture.expanduser().resolve(strict=True)
        point, fixture_target, fixture_lineage = _fixture_case(
            fixture_path, args.fixture_case_id, args.process
        )
    else:
        point_path = args.point.expanduser().resolve(strict=True)
        point = _point_from_path(point_path)
    digest = _point_digest(point)

    import pyamplicol
    from pyamplicol import Generator, ModelSource, ProcessSet, Runtime

    producer_version = importlib.metadata.version("pyamplicol")
    candidate_identity = _candidate_identity(pyamplicol, producer_version)
    expected: Decimal | None = None
    authority: dict[str, Any] | None = None
    if args.recurrence_authority is not None:
        authority_path = args.recurrence_authority.expanduser().resolve(strict=True)
        expected, authority = _recurrence_target(
            authority_path,
            process=args.process,
            point_digest=digest,
            producer_version=producer_version,
        )
    elif args.madgraph_authority is not None:
        authority_path = args.madgraph_authority.expanduser().resolve(strict=True)
        expected, authority = _madgraph_target(
            authority_path,
            process=args.process,
            n_final=args.n_final,
            point=point,
        )
    elif fixture_target is not None:
        expected, authority = fixture_target, fixture_lineage

    model_started = time.perf_counter()
    model = ModelSource.from_path(prepared).compile()
    model_seconds = time.perf_counter() - model_started
    artifact.parent.mkdir(parents=True, exist_ok=True)
    generation_started = time.perf_counter()
    Generator(
        _run_config(
            args.accuracy,
            args.mode,
            args.query_construction_cores,
        )
    ).generate(
        ProcessSet.from_expressions(
            (args.process,),
            names=(args.process_name,),
        ),
        artifact,
        model=model,
    )
    generation_seconds = time.perf_counter() - generation_started
    preflight_common = {
        "kind": "pyamplicol-otf-contracted-parity-preflight-v1",
        "schema_version": 1,
        "artifact": os.fspath(artifact),
        "candidate_identity": candidate_identity,
        "process": args.process,
        "point_digest": digest,
        "authority": authority,
        "requested_lane": {
            "accuracy": args.accuracy,
            "mode": args.mode,
            "precision": args.precision,
            "workers": 1,
            "cores": args.query_construction_cores,
        },
        "generation_seconds": generation_seconds,
        "evaluation_state": "not-started",
    }
    try:
        artifact_identity = _authenticate_artifact(
            artifact,
            process_name=args.process_name,
            mode=args.mode,
            query_construction_cores=args.query_construction_cores,
            candidate_identity=candidate_identity,
        )
    except Exception as error:
        _atomic_json(
            preflight_path,
            {
                **preflight_common,
                "status": "failed",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        raise
    _atomic_json(
        preflight_path,
        {
            **preflight_common,
            "status": "passed",
            "artifact_identity": artifact_identity,
        },
    )
    producer = artifact_identity["producer"]
    effective_config = artifact_identity["effective_config"]
    query_construction_threads = artifact_identity["runtime_options"][
        "query_construction_threads"
    ]

    load_started = time.perf_counter()
    runtime = Runtime.load(artifact, process=args.process_name)
    load_seconds = time.perf_counter() - load_started
    try:
        inspection = _object(runtime.inspect(), "compact runtime inspection")
        runtime_metadata = _object(
            inspection.get("runtime_metadata"),
            "compact runtime metadata",
        )
        observed_mode = runtime_metadata.get("execution_mode")
        observed_accuracy = runtime_metadata.get("color_accuracy")
        if (
            runtime.execution_mode != args.mode
            or observed_mode != args.mode
            or observed_accuracy != args.accuracy
        ):
            raise GateError(
                "runtime lane differs from requested "
                f"{args.mode}/{args.accuracy}: "
                f"{runtime.execution_mode}/{observed_mode}/{observed_accuracy}"
            )
        evaluation_started = time.perf_counter()
        values = runtime.evaluate((point,), precision=args.precision)
        evaluation_seconds = time.perf_counter() - evaluation_started
    finally:
        runtime.clear()
    if len(values) != 1:
        raise GateError(f"single-point evaluation returned {len(values)} values")
    real, imaginary, exact = _actual(values[0])
    if real < 0:
        raise GateError("matrix element is negative")

    comparison = None if expected is None else _comparison(real, imaginary, expected)
    status = (
        "anchor"
        if comparison is None
        else ("passed" if comparison["passed"] is True else "mismatch")
    )
    record: dict[str, Any] = {
        "kind": KIND,
        "schema_version": 1,
        "status": status,
        "seed": SEED,
        "process": args.process,
        "process_name": args.process_name,
        "n_final": args.n_final,
        "family_id": args.family_id,
        "point_digest": digest,
        "point_count": 1,
        "selector_policy": "none-contracted-singleton",
        "warmup_count": 0,
        "repetition_count": 0,
        "lane": {
            "accuracy": args.accuracy,
            "mode": args.mode,
            "precision": args.precision,
            "workers": 1,
            "cores": args.query_construction_cores,
            "query_construction_threads": (
                None
                if query_construction_threads is None
                else int(query_construction_threads)
            ),
            "relation_discovery": "off",
        },
        "actual": {
            "real": str(real),
            "imaginary": str(imaginary),
            "exact_runtime_string": exact,
        },
        "authority": authority,
        "comparison": comparison,
        "timings_seconds": {
            "prepared_model_load": model_seconds,
            "generation": generation_seconds,
            "runtime_load": load_seconds,
            "single_evaluation": evaluation_seconds,
        },
        "prepared_model": {
            "path": os.fspath(prepared),
            "sha256": _sha256(prepared),
        },
        "artifact": os.fspath(artifact),
        "preflight": {
            "path": os.fspath(preflight_path),
            "sha256": _sha256(preflight_path),
        },
        "effective_config": effective_config,
        "candidate_identity": candidate_identity,
        "producer": producer,
        "python_package": os.fspath(Path(pyamplicol.__file__).resolve()),
    }
    _atomic_json(output, record)
    if comparison is not None and comparison["passed"] is not True:
        raise GateError(
            f"strict parity failed for {args.process}/{args.accuracy}: "
            f"residual={comparison['conditioned_residual']}"
        )
    print(json.dumps(record, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
