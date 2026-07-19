#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Produce and consume an architecture-scoped eager JIT model bundle.

Heavy invocations of this script must be wrapped by::

    python tools/ci/memory_watchdog.py --limit-gib 30 -- \
      python tools/ci/eager_portability.py ...

The producer writes one built-in-SM JIT O3 bundle and a numerical transfer
fixture.  Consumers use that exact archive; they never prepare a model pack.
SymJIT application storage v3 may cross operating systems within one CPU
architecture class, but x86-64 and AArch64 packs are deliberately distinct.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

if __package__:
    from .eager_portability_contract import (
        FORBIDDEN_SUFFIXES as _FORBIDDEN_SUFFIXES,
    )
    from .eager_portability_contract import (
        PortabilityError,
        RuntimeContracts,
        architecture_class,
        audit_architecture_jit_bundle,
    )
    from .eager_portability_contract import (
        archive_manifest as _archive_manifest,
    )
    from .eager_portability_contract import (
        array_value as _array,
    )
    from .eager_portability_contract import (
        canonical_member_path as _canonical_member_path,
    )
    from .eager_portability_contract import (
        integer_value as _integer,
    )
    from .eager_portability_contract import (
        native_payload_kind as _native_payload_kind,
    )
    from .eager_portability_contract import (
        object_value as _object,
    )
    from .eager_portability_contract import (
        sha256_file as _sha256_file,
    )
    from .eager_portability_contract import (
        string_value as _string,
    )
else:
    from eager_portability_contract import (
        FORBIDDEN_SUFFIXES as _FORBIDDEN_SUFFIXES,
    )
    from eager_portability_contract import (
        PortabilityError,
        RuntimeContracts,
        architecture_class,
        audit_architecture_jit_bundle,
    )
    from eager_portability_contract import (
        archive_manifest as _archive_manifest,
    )
    from eager_portability_contract import (
        array_value as _array,
    )
    from eager_portability_contract import (
        canonical_member_path as _canonical_member_path,
    )
    from eager_portability_contract import (
        integer_value as _integer,
    )
    from eager_portability_contract import (
        native_payload_kind as _native_payload_kind,
    )
    from eager_portability_contract import (
        object_value as _object,
    )
    from eager_portability_contract import (
        sha256_file as _sha256_file,
    )
    from eager_portability_contract import (
        string_value as _string,
    )

TRANSFER_KIND = "pyamplicol-eager-jit-portability-transfer"
TRANSFER_SCHEMA_VERSION = 2
CONSUMER_REPORT_KIND = "pyamplicol-eager-jit-portability-consumer-report"
CONSUMER_REPORT_SCHEMA_VERSION = 2

DEFAULT_PROCESS = "d d~ > z"
DEFAULT_PROCESS_ID = "d_dbar_to_z"
DEFAULT_RTOL = 1.0e-12
DEFAULT_ATOL = 1.0e-15
DEFAULT_BUNDLE_NAME = "builtin-sm-jit-o3.pyamplicol-model"
DEFAULT_FIXTURE_NAME = "transfer.json"

_ROOT = Path(__file__).resolve().parents[2]
_COMPILER_COMMANDS = (
    "ar",
    "as",
    "c++",
    "cc",
    "clang",
    "clang++",
    "g++",
    "gcc",
    "ld",
    "nasm",
)


def _runtime_contracts() -> RuntimeContracts:
    # Keep imports lazy so clean-checkout release tests can exercise the archive
    # auditor without installing pyAmpliCol or its native dependencies.
    from pyamplicol._internal.versions import (
        COMPILED_MODEL_SCHEMA_VERSION,
        EAGER_DAG_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_SERIALIZATION_ABI,
        SYMJIT_APPLICATION_ABI,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        package_version,
    )
    from pyamplicol.models.prepared import (
        EAGER_KERNEL_ABI,
        PREPARED_MODEL_BUNDLE_KIND,
        PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
    )

    return RuntimeContracts(
        bundle_kind=PREPARED_MODEL_BUNDLE_KIND,
        bundle_schema_version=PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
        eager_kernel_abi=EAGER_KERNEL_ABI,
        compiled_model_schema_version=COMPILED_MODEL_SCHEMA_VERSION,
        symbolica_serialization_abi=SYMBOLICA_SERIALIZATION_ABI,
        symjit_application_abi=SYMJIT_APPLICATION_ABI,
        symjit_runtime_capability=SYMJIT_F64_RUNTIME_CAPABILITY,
        eager_runtime_capability=EAGER_DAG_F64_RUNTIME_CAPABILITY,
        package_version=package_version(),
    )


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return environment


def _python_executable(path: Path) -> Path:
    """Return an absolute launcher path without dereferencing a venv symlink."""

    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    if not absolute.is_file():
        raise PortabilityError(f"Python executable does not exist: {absolute}")
    if not os.access(absolute, os.X_OK):
        raise PortabilityError(f"Python executable is not executable: {absolute}")
    return absolute


def _host_identity(
    role: str,
    *,
    expected_system: str | None,
    expected_machine: str | None,
) -> tuple[str, str, str]:
    actual_system = platform.system()
    actual_machine = platform.machine()
    actual_architecture = architecture_class(actual_machine)
    if expected_system is not None and actual_system != expected_system:
        raise PortabilityError(
            f"{role} system is {actual_system!r}, expected {expected_system!r}"
        )
    if expected_machine is not None:
        expected_architecture = architecture_class(expected_machine)
        if actual_architecture != expected_architecture:
            raise PortabilityError(
                f"{role} architecture class is {actual_architecture!r}, "
                f"expected {expected_architecture!r}"
            )
    return actual_system, actual_machine, actual_architecture


def _run(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> None:
    print(f"+ {shlex.join(command)}", file=sys.stderr, flush=True)
    try:
        subprocess.run(
            list(command),
            cwd=_ROOT,
            env=dict(environment) if environment is not None else None,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PortabilityError(f"command failed: {shlex.join(command)}") from error


def _model_compile_command(python: Path, bundle: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "pyamplicol",
        "model",
        "compile",
        "built-in-sm",
        str(bundle),
        "--backend",
        "jit",
        "--jit-optimization-level",
        "3",
        "--cores",
        "1",
        "--no-symbolica-suggestion",
        "--progress",
        "off",
        "--format",
        "json",
    ]


def _generation_command(
    python: Path,
    *,
    model: str | Path,
    output: Path,
    execution_mode: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "pyamplicol",
        "generate",
        DEFAULT_PROCESS,
        str(output),
        "--model",
        str(model),
        "--execution-mode",
        execution_mode,
        "--backend",
        "jit",
        "--jit-optimization-level",
        "3",
        "--color-accuracy",
        "lc",
        "--mode",
        "replace",
        "--workers",
        "1",
        "--validation",
        "--validation-samples",
        "1",
        "--no-post-build-validation",
        "--no-emit-api-bundle",
        "--no-symbolica-suggestion",
        "--progress",
        "off",
        "--format",
        "json",
    ]


def _validation_momenta(artifact: Path) -> list[list[float]]:
    path = artifact / "processes" / DEFAULT_PROCESS_ID / "validation-momenta.json"
    try:
        payload = _object(json.loads(path.read_text(encoding="utf-8")), "momenta")
        points = _array(payload.get("points"), "momenta.points")
        first = _array(points[0], "momenta.points[0]")
        momenta = []
        for index, raw_particle in enumerate(first):
            particle = _object(raw_particle, f"momenta.points[0][{index}]")
            components = _array(
                particle.get("momentum"),
                f"momenta.points[0][{index}].momentum",
            )
            if len(components) != 4:
                raise PortabilityError("validation momentum must have four components")
            momenta.append([float(value) for value in components])
    except (OSError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, PortabilityError):
            raise
        raise PortabilityError("could not read generated validation momenta") from error
    return momenta


def _evaluate_artifact(artifact: Path, momenta: Sequence[Sequence[float]]) -> complex:
    from pyamplicol import Runtime

    runtime = Runtime.load(artifact, process=DEFAULT_PROCESS_ID)
    values = runtime.evaluate((momenta,), precision=16)
    if len(values) != 1:
        raise PortabilityError("runtime returned an unexpected number of values")
    return complex(values[0])


def _close(actual: complex, expected: complex, *, rtol: float, atol: float) -> bool:
    return abs(actual - expected) <= atol + rtol * abs(expected)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PortabilityError("could not identify the producer Git commit") from error
    return completed.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def produce_transfer(
    output_directory: Path,
    *,
    python: Path,
    expected_system: str | None = None,
    expected_machine: str | None = None,
) -> dict[str, object]:
    actual_system, actual_machine, actual_architecture = _host_identity(
        "producer",
        expected_system=expected_system,
        expected_machine=expected_machine,
    )

    output = output_directory.expanduser().resolve(strict=False)
    if output.exists():
        if not output.is_dir():
            raise PortabilityError(f"producer output path is not a directory: {output}")
        if any(output.iterdir()):
            raise PortabilityError(f"producer output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / DEFAULT_BUNDLE_NAME
    contracts = _runtime_contracts()
    environment = _command_environment()

    _run(_model_compile_command(python, bundle), environment=environment)
    audit = audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_architecture_class=actual_architecture,
    )
    if audit["producer_version"] != contracts.package_version:
        raise PortabilityError(
            "prepared bundle producer version differs from the installed package"
        )

    with tempfile.TemporaryDirectory(
        prefix="pyamplicol-eager-portability-producer-"
    ) as raw:
        temporary = Path(raw)
        eager_artifact = temporary / "eager"
        compiled_artifact = temporary / "compiled"
        _run(
            _generation_command(
                python,
                model=bundle,
                output=eager_artifact,
                execution_mode="eager",
            ),
            environment=environment,
        )
        _run(
            _generation_command(
                python,
                model="built-in-sm",
                output=compiled_artifact,
                execution_mode="compiled",
            ),
            environment=environment,
        )
        momenta = _validation_momenta(eager_artifact)
        eager_value = _evaluate_artifact(eager_artifact, momenta)
        expected_value = _evaluate_artifact(compiled_artifact, momenta)
        if not _close(
            eager_value,
            expected_value,
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        ):
            raise PortabilityError(
                "producer eager value does not match the compiled reference"
            )

    fixture: dict[str, object] = {
        "bundle": {
            "filename": bundle.name,
            **audit,
        },
        "expected": {
            "atol": DEFAULT_ATOL,
            "imaginary": expected_value.imag,
            "real": expected_value.real,
            "rtol": DEFAULT_RTOL,
            "source": "compiled-jit-o3-f64",
        },
        "kind": TRANSFER_KIND,
        "process": {
            "expression": DEFAULT_PROCESS,
            "id": DEFAULT_PROCESS_ID,
            "momenta": momenta,
        },
        "producer": {
            "architecture_class": actual_architecture,
            "git_commit": _git_commit(),
            "machine": actual_machine,
            "package_version": contracts.package_version,
            "python": platform.python_version(),
            "system": actual_system,
        },
        "schema_version": TRANSFER_SCHEMA_VERSION,
    }
    fixture_path = output / DEFAULT_FIXTURE_NAME
    _write_json(fixture_path, fixture)
    return {
        "bundle": str(bundle),
        "bundle_sha256": audit["bundle_sha256"],
        "expected": {"real": expected_value.real, "imaginary": expected_value.imag},
        "fixture": str(fixture_path),
        "kernel_count": audit["kernel_count"],
        "architecture_class": actual_architecture,
    }


@contextlib.contextmanager
def compiler_guard() -> Iterator[tuple[dict[str, str], Path]]:
    """Deny external compiler/linker execution during consumer generation."""

    with tempfile.TemporaryDirectory(prefix="pyamplicol-compiler-guard-") as raw:
        root = Path(raw)
        marker = root / "invocations.log"
        script = root / "deny-tool"
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$0 $*\" >> {shlex.quote(str(marker))}\n"
            "exit 97\n",
            encoding="ascii",
        )
        script.chmod(0o755)
        for name in _COMPILER_COMMANDS:
            shim = root / name
            try:
                shim.symlink_to(script.name)
            except OSError:
                shim.write_bytes(script.read_bytes())
                shim.chmod(0o755)

        environment = _command_environment()
        environment["PATH"] = os.pathsep.join((str(root), environment.get("PATH", "")))
        for variable, command in (
            ("AR", "ar"),
            ("AS", "as"),
            ("CC", "cc"),
            ("CXX", "c++"),
            ("LD", "ld"),
        ):
            environment[variable] = str(root / command)
        yield environment, marker


@contextlib.contextmanager
def _temporary_environment(environment: Mapping[str, str]) -> Iterator[None]:
    previous = os.environ.copy()
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _fixture(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortabilityError(f"could not read transfer fixture: {error}") from error
    fixture = _object(payload, "transfer fixture")
    if fixture.get("kind") != TRANSFER_KIND:
        raise PortabilityError("invalid eager portability transfer kind")
    if fixture.get("schema_version") != TRANSFER_SCHEMA_VERSION:
        raise PortabilityError("unsupported eager portability transfer schema")
    return fixture


def _manifest_payloads(
    artifact: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    try:
        manifest = _object(
            json.loads((artifact / "artifact.json").read_text(encoding="utf-8")),
            "artifact manifest",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortabilityError("generated artifact manifest is invalid") from error
    payloads: dict[str, dict[str, object]] = {}
    for index, raw_payload in enumerate(
        _array(manifest.get("payloads"), "artifact.payloads")
    ):
        record = _object(raw_payload, f"artifact.payloads[{index}]")
        path = _canonical_member_path(record.get("path"), "artifact payload path")
        source = artifact / PurePosixPath(path)
        if not source.is_file() or source.is_symlink():
            raise PortabilityError(f"generated artifact payload {path!r} is missing")
        expected_size = _integer(record.get("size_bytes"), f"payload {path!r} size")
        expected_sha = _string(record.get("sha256"), f"payload {path!r} sha256")
        if (
            source.stat().st_size != expected_size
            or _sha256_file(source) != expected_sha
        ):
            raise PortabilityError(f"generated artifact payload {path!r} is modified")
        payloads[path] = record
    return manifest, payloads


def verify_consumer_artifact(
    artifact: Path,
    bundle: Path,
    *,
    contracts: RuntimeContracts,
    bundle_audit: Mapping[str, object],
) -> dict[str, object]:
    """Prove that eager generation copied, rather than rebuilt, JIT kernels."""

    manifest, artifact_payloads = _manifest_payloads(artifact)
    if (
        manifest.get("kind") != "pyamplicol-process"
        or manifest.get("schema_version") != 3
    ):
        raise PortabilityError("consumer did not generate a schema-v3 process artifact")
    producer = _object(manifest.get("producer"), "artifact.producer")
    if producer.get("version") != contracts.package_version:
        raise PortabilityError("consumer artifact package version mismatch")
    processes = _array(manifest.get("processes"), "artifact.processes")
    if len(processes) != 1:
        raise PortabilityError("consumer artifact must contain one concrete process")
    process = _object(processes[0], "artifact.processes[0]")
    if (
        process.get("id") != DEFAULT_PROCESS_ID
        or process.get("expression") != DEFAULT_PROCESS
    ):
        raise PortabilityError("consumer artifact process identity mismatch")
    if contracts.eager_runtime_capability not in _array(
        process.get("required_runtime_capabilities"),
        "process.required_runtime_capabilities",
    ):
        raise PortabilityError("consumer artifact does not require the eager runtime")

    execution_path = artifact / "processes" / DEFAULT_PROCESS_ID / "execution.json"
    execution = _object(
        json.loads(execution_path.read_text(encoding="utf-8")),
        "eager execution manifest",
    )
    if execution.get("kind") != "pyamplicol-runtime-eager-execution":
        raise PortabilityError("consumer generated the compiled execution lane")

    extensions = _object(manifest.get("extensions"), "artifact.extensions")
    identity = _object(
        extensions.get("eager_prepared_pack"),
        "artifact.extensions.eager_prepared_pack",
    )
    if (
        identity.get("backend") != "jit"
        or identity.get("eager_kernel_abi") != contracts.eager_kernel_abi
    ):
        raise PortabilityError("consumer artifact prepared-pack identity mismatch")
    if identity.get("kernel_count") != bundle_audit.get("kernel_count"):
        raise PortabilityError("consumer artifact records an unexpected source pack")

    bundle_manifest, bundle_payloads = _archive_manifest(bundle)
    bundle_pack = _object(bundle_manifest.get("kernel_pack"), "bundle kernel pack")
    bundle_members = {
        _canonical_member_path(
            _object(record, "bundle member").get("path"),
            "bundle member path",
        ): _object(record, "bundle member")
        for record in _array(bundle_manifest.get("members"), "bundle members")
    }

    eager_pack_path = artifact / "model" / "eager-kernel-pack.json"
    filtered_pack = _object(
        json.loads(eager_pack_path.read_text(encoding="utf-8")),
        "filtered eager kernel pack",
    )
    if filtered_pack.get("backend") != "jit" or filtered_pack.get(
        "target"
    ) != bundle_pack.get("target"):
        raise PortabilityError("consumer changed the architecture-scoped JIT target")
    filtered_kernels = _array(filtered_pack.get("kernels"), "filtered pack kernels")
    if not filtered_kernels:
        raise PortabilityError("consumer artifact references no prepared kernels")

    copied_paths: set[str] = set()
    for index, raw_kernel in enumerate(filtered_kernels):
        kernel = _object(raw_kernel, f"filtered kernel {index}")
        f64 = _object(
            kernel.get("f64_evaluator_manifest"), f"filtered kernel {index} f64"
        )
        for field in ("application_path", "evaluator_state_path"):
            source_path = _canonical_member_path(
                f64.get(field), f"filtered kernel {index} {field}"
            )
            destination_path = f"model/eager-kernels/{source_path}"
            destination = artifact / PurePosixPath(destination_path)
            source_record = bundle_members.get(source_path)
            destination_record = artifact_payloads.get(destination_path)
            if source_record is None or destination_record is None:
                raise PortabilityError(
                    f"consumer omitted transferred kernel payload {source_path!r}"
                )
            source_sha = _string(source_record.get("sha256"), "source payload SHA")
            if destination_record.get("sha256") != source_sha:
                raise PortabilityError(
                    f"consumer rebuilt or changed kernel payload {source_path!r}"
                )
            if destination.read_bytes() != bundle_payloads[source_path]:
                raise PortabilityError(
                    f"consumer kernel payload {source_path!r} is not byte-identical"
                )
            copied_paths.add(source_path)

    for path, record in artifact_payloads.items():
        suffix = PurePosixPath(path).suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES:
            raise PortabilityError(
                f"consumer artifact contains native/source payload {path!r}"
            )
        if path.startswith("model/eager-kernels/"):
            payload = (artifact / PurePosixPath(path)).read_bytes()
            native_kind = _native_payload_kind(payload)
            if native_kind is not None:
                raise PortabilityError(
                    f"consumer kernel payload {path!r} contains {native_kind}"
                )
            if record.get("executable") is not False:
                raise PortabilityError(
                    f"consumer kernel payload {path!r} is executable"
                )

    return {
        "artifact_id": manifest.get("artifact_id"),
        "copied_kernel_payload_count": len(copied_paths),
        "filtered_kernel_count": len(filtered_kernels),
        "process_id": DEFAULT_PROCESS_ID,
        "runtime_kind": execution.get("kind"),
    }


def consume_transfer(
    transfer_directory: Path,
    *,
    python: Path,
    report_path: Path,
    expected_system: str | None,
    expected_machine: str | None,
) -> dict[str, object]:
    actual_system, actual_machine, actual_architecture = _host_identity(
        "consumer",
        expected_system=expected_system,
        expected_machine=expected_machine,
    )
    transfer = transfer_directory.expanduser().resolve(strict=True)
    fixture = _fixture(transfer / DEFAULT_FIXTURE_NAME)
    bundle_record = _object(fixture.get("bundle"), "transfer.bundle")
    filename = _canonical_member_path(bundle_record.get("filename"), "bundle filename")
    if len(PurePosixPath(filename).parts) != 1:
        raise PortabilityError(
            "transferred bundle filename must not contain directories"
        )
    bundle = transfer / filename
    contracts = _runtime_contracts()
    audit = audit_architecture_jit_bundle(
        bundle,
        contracts=contracts,
        expected_sha256=_string(bundle_record.get("bundle_sha256"), "bundle sha256"),
        expected_architecture_class=actual_architecture,
    )
    if bundle_record.get("architecture_class") != audit["architecture_class"]:
        raise PortabilityError(
            "transfer fixture architecture differs from the prepared bundle"
        )
    if audit["producer_version"] != contracts.package_version:
        raise PortabilityError(
            "transferred bundle producer version differs from the consumer package"
        )
    if fixture.get("producer") is None:
        raise PortabilityError("transfer fixture omits producer provenance")
    producer = _object(fixture.get("producer"), "transfer.producer")
    if producer.get("architecture_class") != audit["architecture_class"]:
        raise PortabilityError(
            "producer architecture differs from the prepared bundle target"
        )
    if producer.get("git_commit") != _git_commit():
        raise PortabilityError("producer and consumer source commits differ")

    process = _object(fixture.get("process"), "transfer.process")
    if (
        process.get("expression") != DEFAULT_PROCESS
        or process.get("id") != DEFAULT_PROCESS_ID
    ):
        raise PortabilityError(
            "transfer process fixture differs from the harness contract"
        )
    momenta = _array(process.get("momenta"), "transfer.process.momenta")
    expected_record = _object(fixture.get("expected"), "transfer.expected")
    expected = complex(
        float(expected_record.get("real")),
        float(expected_record.get("imaginary")),
    )
    rtol = float(expected_record.get("rtol"))
    atol = float(expected_record.get("atol"))
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise PortabilityError("transfer tolerances must be finite and non-negative")

    bundle_sha_before = _sha256_file(bundle)
    with tempfile.TemporaryDirectory(
        prefix="pyamplicol-eager-portability-consumer-"
    ) as raw:
        artifact = Path(raw) / "eager"
        with compiler_guard() as (environment, marker):
            command = _generation_command(
                python,
                model=bundle,
                output=artifact,
                execution_mode="eager",
            )
            _run(command, environment=environment)
            if marker.exists() and marker.stat().st_size:
                invocations = marker.read_text(encoding="utf-8", errors="replace")
                raise PortabilityError(
                    "consumer eager generation invoked an external compiler/linker:\n"
                    + invocations
                )
            artifact_summary = verify_consumer_artifact(
                artifact,
                bundle,
                contracts=contracts,
                bundle_audit=audit,
            )
            with _temporary_environment(environment):
                actual = _evaluate_artifact(artifact, momenta)
            if marker.exists() and marker.stat().st_size:
                invocations = marker.read_text(encoding="utf-8", errors="replace")
                raise PortabilityError(
                    "consumer eager load/evaluation invoked an external "
                    "compiler/linker:\n" + invocations
                )

    if _sha256_file(bundle) != bundle_sha_before:
        raise PortabilityError("consumer modified the transferred prepared bundle")
    if not _close(actual, expected, rtol=rtol, atol=atol):
        raise PortabilityError(
            "consumer f64 result differs from the producer compiled reference: "
            f"actual={actual!r}, expected={expected!r}, "
            f"difference={abs(actual - expected):.17g}"
        )

    report: dict[str, object] = {
        "artifact": artifact_summary,
        "bundle": audit,
        "consumer": {
            "architecture_class": actual_architecture,
            "git_commit": _git_commit(),
            "machine": actual_machine,
            "package_version": contracts.package_version,
            "python": platform.python_version(),
            "system": actual_system,
        },
        "kind": CONSUMER_REPORT_KIND,
        "numerical_check": {
            "absolute_difference": abs(actual - expected),
            "actual": {"imaginary": actual.imag, "real": actual.real},
            "atol": atol,
            "expected": {"imaginary": expected.imag, "real": expected.real},
            "rtol": rtol,
        },
        "producer": producer,
        "schema_version": CONSUMER_REPORT_SCHEMA_VERSION,
    }
    _write_json(report_path.expanduser().resolve(strict=False), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-test an architecture-scoped built-in-SM JIT O3 model bundle."
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Audit one prepared JIT bundle without generating a process.",
    )
    audit.add_argument("bundle", type=Path)
    audit.add_argument("--expected-sha256")
    audit.add_argument(
        "--expected-machine",
        help="Machine name whose storage-v3 architecture class must match the bundle.",
    )

    produce = subparsers.add_parser(
        "produce",
        help="Create the one transferable bundle and compiled-reference fixture.",
    )
    produce.add_argument("output_directory", type=Path)
    produce.add_argument("--python", type=Path, default=Path(sys.executable))
    produce.add_argument("--expected-system")
    produce.add_argument("--expected-machine")

    consume = subparsers.add_parser(
        "consume",
        help="Consume a transferred bundle without rebuilding its kernels.",
    )
    consume.add_argument("transfer_directory", type=Path)
    consume.add_argument("--python", type=Path, default=Path(sys.executable))
    consume.add_argument("--report", type=Path, required=True)
    consume.add_argument("--expected-system")
    consume.add_argument("--expected-machine")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action == "audit":
            expected_machine = arguments.expected_machine or platform.machine()
            result = audit_architecture_jit_bundle(
                arguments.bundle,
                contracts=_runtime_contracts(),
                expected_sha256=arguments.expected_sha256,
                expected_architecture_class=architecture_class(expected_machine),
            )
        elif arguments.action == "produce":
            result = produce_transfer(
                arguments.output_directory,
                python=_python_executable(arguments.python),
                expected_system=arguments.expected_system,
                expected_machine=arguments.expected_machine,
            )
        else:
            result = consume_transfer(
                arguments.transfer_directory,
                python=_python_executable(arguments.python),
                report_path=arguments.report,
                expected_system=arguments.expected_system,
                expected_machine=arguments.expected_machine,
            )
    except (PortabilityError, OSError, ValueError) as error:
        print(f"eager-portability: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
