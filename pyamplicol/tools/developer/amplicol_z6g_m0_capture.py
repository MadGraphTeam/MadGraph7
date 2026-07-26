#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Capture strict original-AmpliCol evidence for the qq_Z6g M0 gate.

The public ``capture`` command validates the four pyAmpliCol inputs from an
M0 request template, builds the pinned original-AmpliCol probes, and records
seven paired selected/all-flow subprocess rounds.  The private ``_sample``
command is the content-addressed executable retained in every timing record;
it wraps ``amplicol_library_benchmark`` or ``amplicol_color_probe`` and emits
one strict ``amplicol-m0-probe-result`` JSON object on stdout.

This program intentionally has no synthetic or diagnostic-success mode.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import legacy_amplicol  # noqa: E402

PROCESS_EXPRESSION = "u u~ > Z g g g g g g"
NORMALIZED_PROCESS = "u u~ > z g g g g g g"
AMPLICOL_EXTERNAL_LEG_PERMUTATION = (0, 1, 3, 4, 5, 6, 7, 8, 2)
SELECTED_ROLE = "selected-flow-helicity-sum"
UNION_ROLE = "all-flow-single-helicity"
ROLES = (SELECTED_ROLE, UNION_ROLE)
SELECTED_WORKLOAD = "single-runtime-selected-flow/helicity-sum"
UNION_WORKLOAD = "all-flows/runtime-selected-single-helicity"
MIN_SAMPLES = 7
RTOL = 1.0e-12
ATOL = 1.0e-15
DEFAULT_TARGET_SECONDS = 5.0
DEFAULT_WARMUP_POINTS = 100
DEFAULT_MINIMUM_POINTS = 100
DEFAULT_MAXIMUM_POINTS = 100_000

_M0_REQUEST_KEYS = {
    "kind",
    "schema_version",
    "captures",
    "amplicol_evidence",
    "expected",
}
_PROBE_RESULT_KEYS = {
    "kind",
    "schema_version",
    "role",
    "sample_index",
    "evaluated_point_count",
    "elapsed_seconds",
    "seconds_per_point",
    "selected_totals",
    "resolved_sums",
}


class CaptureError(RuntimeError):
    """The authoritative original-AmpliCol capture could not be completed."""


@dataclass(frozen=True, slots=True)
class CaptureContract:
    """Cross-validated physical and provenance contract from four M0 captures."""

    expected: dict[str, Any]
    input_files: tuple[dict[str, Any], ...]
    host: dict[str, Any]
    fixture: dict[str, Any]
    color_axis: dict[str, Any]
    helicity_axis: dict[str, Any]
    color_flow_request: str
    helicity_request: str
    selected_values: tuple[complex, ...]
    union_values: tuple[complex, ...]


@dataclass(frozen=True, slots=True)
class ProbeContext:
    """Immutable original-AmpliCol row and snapshotted execution inputs."""

    runtime: Path
    worker_executable: Path
    process_file: Path
    selected_binary: Path
    union_binary: Path
    linked_files: tuple[Path, ...]
    group: int
    integral: int
    source_pdgs: tuple[int, ...]
    generated_pdgs: tuple[int, ...]
    generated_color_order: tuple[int, ...]
    permutation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Pinned original-AmpliCol source/compiler identity."""

    source: dict[str, Any]
    source_files: tuple[dict[str, Any], ...]


def _die(message: str) -> NoReturn:
    raise CaptureError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        _die(f"evidence path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return _file_ref(path)


def _content_addressed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "content_sha256" in result:
        _die("content-addressing input already contains content_sha256")
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _die(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: _die(
                f"{label} contains non-finite JSON number {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        _die(f"{label} must be a JSON object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _strict_json(path.read_bytes(), label)
    except OSError as error:
        raise CaptureError(f"cannot read {label} at {path}: {error}") from error


def _load_m0() -> ModuleType:
    path = SCRIPT.with_name("eager_compiled_arena_m0.py")
    spec = importlib.util.spec_from_file_location("_amplicol_capture_m0", path)
    if spec is None or spec.loader is None:
        _die(f"cannot load M0 schema validator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise CaptureError(f"cannot load M0 schema validator: {error}") from error
    return module


def _same_complex_values(
    left: Sequence[complex],
    right: Sequence[complex],
) -> bool:
    return len(left) == len(right) and all(
        abs(lhs - rhs) <= ATOL + RTOL * abs(rhs)
        for lhs, rhs in zip(left, right, strict=True)
    )


def _contract_from_request(path: Path) -> CaptureContract:
    """Revalidate all four captures and extract their common M0 contract."""

    m0 = _load_m0()
    payload = _load_json(path, "M0 request template")
    if set(payload) != _M0_REQUEST_KEYS:
        _die("M0 request template has unknown or missing root keys")
    if (
        payload.get("kind") != m0.REQUEST_KIND
        or payload.get("schema_version") != m0.REQUEST_SCHEMA
    ):
        _die("M0 request template has the wrong kind or schema")
    amplicol = payload.get("amplicol_evidence")
    if not isinstance(amplicol, Mapping) or set(amplicol) != set(ROLES):
        _die("request template must reserve exactly both AmpliCol evidence roles")
    expected = m0._validate_expected(payload.get("expected"))
    if expected["amplicol_source_revision"] != legacy_amplicol.expected_revision():
        _die(
            "request AmpliCol revision differs from the contributor-lock pin: "
            f"{expected['amplicol_source_revision']} != "
            f"{legacy_amplicol.expected_revision()}"
        )
    if expected["external_leg_permutation"] != list(AMPLICOL_EXTERNAL_LEG_PERMUTATION):
        _die(
            "request AmpliCol source-to-generated permutation must be "
            f"{list(AMPLICOL_EXTERNAL_LEG_PERMUTATION)}"
        )

    captures_raw = payload.get("captures")
    if not isinstance(captures_raw, Mapping) or set(captures_raw) != set(m0.MODELS):
        _die("request template must reference exactly both M0 model families")
    benchmark = m0._load_benchmark_module()
    captures: dict[tuple[str, str], Any] = {}
    input_files: list[dict[str, Any]] = [_file_ref(path)]
    for model in m0.MODELS:
        layouts = captures_raw[model]
        if not isinstance(layouts, Mapping) or set(layouts) != set(m0.LAYOUTS):
            _die(f"request captures for {model} must contain both LC layouts")
        for layout in m0.LAYOUTS:
            ref = m0._file_ref(
                layouts[layout],
                base=path.resolve().parent,
                label=f"request.captures.{model}.{layout}",
            )
            loaded = m0._load_json_ref(ref, f"{model}/{layout}")
            input_files.append(
                {
                    "path": str(ref.path),
                    "size_bytes": ref.size_bytes,
                    "sha256": ref.sha256,
                }
            )
            captures[(model, layout)] = m0._validate_capture(
                loaded,
                model=model,
                layout=layout,
                expected=expected,
                benchmark=benchmark,
            )

    first = captures[("built-in-sm", "topology-replay")]
    for key, current in captures.items():
        if (
            current.source_identity != first.source_identity
            or current.runtime_identity != first.runtime_identity
            or current.host != first.host
            or current.fixture != first.fixture
            or current.normalization_sha256 != first.normalization_sha256
            or current.color_axis != first.color_axis
            or current.helicity_axis != first.helicity_axis
        ):
            _die(f"capture {key} differs from the common M0 physical contract")
    for layout in m0.LAYOUTS:
        builtin = captures[("built-in-sm", layout)]
        ufo = captures[("ufo-sm", layout)]
        if not _same_complex_values(
            builtin.validation_values,
            ufo.validation_values,
        ):
            _die(f"built-in and UFO validation values differ for {layout}")

    topology = captures[("built-in-sm", "topology-replay")]
    union = captures[("built-in-sm", "all-flow-union")]
    topology_payload = topology.loaded.payload
    fixture_payload = topology_payload["profiles"]["compiled"]["validation"]["fixture"]
    raw_file = fixture_payload["file"]
    raw_path_value = raw_file.get("resolved_path", raw_file.get("path"))
    if not isinstance(raw_path_value, str):
        _die("topology capture has no absolute raw validation-momenta path")
    fixture_path = Path(raw_path_value).resolve(strict=True)
    fixture_ref = _file_ref(fixture_path)
    if (
        fixture_ref["sha256"] != first.fixture["file_sha256"]
        or fixture_payload.get("point_count") != first.fixture["point_count"]
        or fixture_payload.get("points_sha256") != first.fixture["points_sha256"]
    ):
        _die("topology raw fixture identity differs from its capture")

    topology_selector = topology.selector
    union_selector = union.selector
    if topology_selector.get("color_flow_request") != union_selector.get(
        "color_flow_request"
    ) or topology_selector.get("helicity_request") != union_selector.get(
        "helicity_request"
    ):
        _die("the two LC layouts used different runtime selector requests")
    return CaptureContract(
        expected=dict(expected),
        input_files=tuple(input_files),
        host=dict(first.host),
        fixture={
            "path": fixture_path,
            "file": fixture_ref,
            "point_count": first.fixture["point_count"],
            "points_sha256": first.fixture["points_sha256"],
        },
        color_axis={
            "count": first.color_axis["count"],
            "ordered_ids_sha256": first.color_axis["ordered_ids_sha256"],
        },
        helicity_axis={
            "count": first.helicity_axis["count"],
            "ordered_ids_sha256": first.helicity_axis["ordered_ids_sha256"],
        },
        color_flow_request=str(topology_selector["color_flow_request"]),
        helicity_request=str(topology_selector["helicity_request"]),
        selected_values=tuple(topology.validation_values),
        union_values=tuple(union.validation_values),
    )


def _host_identity() -> dict[str, Any]:
    """Use the same host identity algorithm as the schema-6 harness."""

    cpu_model = platform.processor().strip()
    if not cpu_model and platform.system() == "Darwin":
        completed = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            cpu_model = completed.stdout.strip()
    if not cpu_model and platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.casefold().startswith(("model name", "hardware")):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    uname = platform.uname()
    return {
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_model": cpu_model or None,
        "logical_cpu_count": os.cpu_count(),
    }


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    label: str,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [os.fspath(item) for item in command]
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        env=None if environment is None else {**os.environ, **environment},
        capture_output=True,
        check=False,
        text=True,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"$ {' '.join(rendered)}\n")
            stream.write(completed.stdout)
            stream.write(completed.stderr)
            if not completed.stdout.endswith("\n"):
                stream.write("\n")
    if completed.returncode != 0:
        _die(
            f"{label} exited with {completed.returncode}: {' '.join(rendered)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


@contextlib.contextmanager
def _staged_process_file(repository: Path, process_file: Path) -> Iterator[str]:
    target = repository / "processes.txt"
    backup = repository / f".processes.txt.m0-backup-{os.getpid()}"
    existed = target.is_file()
    if backup.exists():
        _die(f"stale process-file backup exists: {backup}")
    if existed:
        shutil.copy2(target, backup)
    shutil.copy2(process_file, target)
    try:
        yield target.name
    finally:
        if existed:
            shutil.move(backup, target)
        else:
            target.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def _fixture_points(
    path: Path,
) -> tuple[tuple[int, ...], tuple[tuple[tuple[float, ...], ...], ...]]:
    payload = _load_json(path, "validation-momenta fixture")
    if (
        payload.get("kind") != "pyamplicol-rusticol-validation-momenta"
        or payload.get("schema_version") != 1
        or "points" not in payload
    ):
        _die("validation-momenta fixture has the wrong schema")
    raw_points = payload["points"]
    if not isinstance(raw_points, list) or not raw_points:
        _die("validation-momenta fixture has no points")
    pdgs: tuple[int, ...] | None = None
    points: list[tuple[tuple[float, ...], ...]] = []
    for point_index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, list) or len(raw_point) != 9:
            _die(f"fixture point {point_index} does not have nine external legs")
        point_pdgs: list[int] = []
        momenta: list[tuple[float, ...]] = []
        for particle in raw_point:
            if not isinstance(particle, Mapping):
                _die(f"fixture point {point_index} contains a non-object particle")
            momentum = particle.get("momentum")
            particle_pdg = particle.get("pdg")
            if (
                isinstance(particle_pdg, bool)
                or not isinstance(particle_pdg, int)
                or not isinstance(momentum, list)
                or len(momentum) != 4
            ):
                _die(f"fixture point {point_index} contains an invalid particle")
            converted = tuple(float(component) for component in momentum)
            if not all(math.isfinite(component) for component in converted):
                _die(f"fixture point {point_index} contains non-finite momentum")
            point_pdgs.append(particle_pdg)
            momenta.append(converted)
        current_pdgs = tuple(point_pdgs)
        if pdgs is None:
            pdgs = current_pdgs
        elif current_pdgs != pdgs:
            _die("fixture external-particle order changes between points")
        points.append(tuple(momenta))
    assert pdgs is not None
    normalized_process = " ".join(str(payload.get("process", "")).split()).casefold()
    if normalized_process != NORMALIZED_PROCESS:
        _die("validation-momenta fixture is not the qq_Z6g process")
    if legacy_amplicol.process_pdgs(NORMALIZED_PROCESS) != pdgs:
        _die("fixture external PDGs differ from the original-AmpliCol process parser")
    return pdgs, tuple(points)


def _write_ordered_momenta(
    path: Path,
    *,
    source_pdgs: Sequence[int],
    target_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
) -> None:
    ordered = legacy_amplicol._ordered_binary64_momenta(
        source_pdgs,
        target_pdgs,
        momenta,
    )
    path.write_text(
        "\n".join(
            " ".join(format(float(component), ".17g") for component in vector)
            for vector in ordered
        )
        + "\n",
        encoding="utf-8",
    )


def _generate_process_file(
    *,
    repository: Path,
    setup: Path,
    log_path: Path,
) -> Path:
    """Run the case-sensitive legacy generator with its lowercase spelling."""

    _run(
        (
            sys.executable,
            repository / "process_list.py",
            "--serial",
            NORMALIZED_PROCESS,
        ),
        cwd=setup,
        label="original-AmpliCol process generation",
        log_path=log_path,
    )
    process_file = setup / "processes.txt"
    if not process_file.is_file():
        _die("original-AmpliCol process_list.py did not write processes.txt")
    return process_file


def _copy_runtime(repository: Path, output: Path, process_file: Path) -> ProbeContext:
    runtime = output / "runtime"
    runtime.mkdir(parents=True)
    interpreter = Path(sys.executable).resolve(strict=True)
    worker_executable = runtime / "amplicol_z6g_m0_sample"
    worker_executable.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(str(interpreter))} "
        f'{shlex.quote(str(SCRIPT))} "$@"\n',
        encoding="utf-8",
    )
    worker_executable.chmod(0o755)
    selected_binary = runtime / "amplicol_library_benchmark"
    union_binary = runtime / "amplicol_color_probe"
    shutil.copy2(repository / selected_binary.name, selected_binary)
    shutil.copy2(repository / union_binary.name, union_binary)
    runtime_process = runtime / "processes.txt"
    shutil.copy2(process_file, runtime_process)
    libraries: list[Path] = []
    for pattern in ("libamp*.so", "libamp*.dylib", "libamp*.a"):
        for source in sorted(repository.glob(pattern)):
            target = runtime / source.name
            if not target.exists():
                shutil.copy2(source, target)
                libraries.append(target)
    if not libraries:
        _die("generated selected-flow library produced no libamp library")
    return ProbeContext(
        runtime=runtime.resolve(),
        worker_executable=worker_executable.resolve(),
        process_file=runtime_process.resolve(),
        selected_binary=selected_binary.resolve(),
        union_binary=union_binary.resolve(),
        linked_files=(
            interpreter,
            SCRIPT,
            selected_binary.resolve(),
            union_binary.resolve(),
            runtime_process.resolve(),
            *(path.resolve() for path in libraries),
        ),
        group=0,
        integral=0,
        source_pdgs=(),
        generated_pdgs=(),
        generated_color_order=(),
        permutation=(),
    )


def _prepare_probes(
    *,
    repository: Path,
    output: Path,
    fixture_path: Path,
    expected_flow_word: Sequence[int],
    expected_permutation: Sequence[int],
    jobs: int,
) -> ProbeContext:
    """Generate one LC row, build both probes, and snapshot execution inputs."""

    legacy_amplicol.prepare_checkout(repository)
    legacy_amplicol.validate_checkout(repository)
    source_pdgs, points = _fixture_points(fixture_path)
    setup = output / "setup"
    setup.mkdir(parents=True)
    log = setup / "build.log"
    process_file = _generate_process_file(
        repository=repository,
        setup=setup,
        log_path=log,
    )
    entries = legacy_amplicol.parse_process_file(process_file)
    entry, _matches = legacy_amplicol.select_generated_process_entry(
        entries,
        generated_process=NORMALIZED_PROCESS,
        wanted_pdgs=source_pdgs,
    )
    mapped = legacy_amplicol.source_mapped_color_order(
        entry,
        source_pdgs=source_pdgs,
    )
    colored = {
        index
        for index, pdg in enumerate(source_pdgs, start=1)
        if abs(int(pdg)) == 21 or 1 <= abs(int(pdg)) <= 6
    }
    color_word = tuple(label for label in mapped if label in colored)
    if list(color_word) != list(expected_flow_word):
        _die(
            "selected original-AmpliCol generated row differs from the stable "
            f"M0 flow word: {color_word} != {tuple(expected_flow_word)}"
        )
    permutation = tuple(legacy_amplicol._permutation(source_pdgs, entry.process_pdgs))
    if list(permutation) != list(expected_permutation):
        _die("source-to-generated external-leg permutation differs from M0 request")

    momenta_directory = repository / "Utilities" / "ME_checks"
    momenta_directory.mkdir(parents=True, exist_ok=True)
    for candidate in entries:
        target = momenta_directory / (
            f"momenta_{candidate.group}_{candidate.integral}.txt"
        )
        _write_ordered_momenta(
            target,
            source_pdgs=source_pdgs,
            target_pdgs=candidate.process_pdgs,
            momenta=points[0],
        )
    with _staged_process_file(repository, process_file) as process_argument:
        commands: tuple[tuple[str, ...], ...] = (
            ("make", "cleanlib"),
            ("make", f"-j{jobs}", "amplicol_generate"),
            (
                "./amplicol_generate",
                "--library=create",
                f"--process={process_argument}",
                "--amplicol_momenta_probe=10",
                "--amplicol_probe_quiet",
                "--timing=none",
            ),
            ("make", f"-j{jobs}", "amplicol_generate_library"),
            ("make", f"-j{jobs}", "amplicol_library_benchmark"),
            ("make", f"-j{jobs}", "amplicol_color_probe"),
        )
        for command in commands:
            _run(
                command,
                cwd=repository,
                label=f"original-AmpliCol build step {command[0]}",
                log_path=log,
            )
    legacy_amplicol.validate_checkout(repository)
    snapshot = _copy_runtime(repository, output, process_file)
    return ProbeContext(
        runtime=snapshot.runtime,
        worker_executable=snapshot.worker_executable,
        process_file=snapshot.process_file,
        selected_binary=snapshot.selected_binary,
        union_binary=snapshot.union_binary,
        linked_files=snapshot.linked_files,
        group=int(entry.group),
        integral=int(entry.integral),
        source_pdgs=tuple(source_pdgs),
        generated_pdgs=tuple(entry.process_pdgs),
        generated_color_order=tuple(entry.color_order),
        permutation=permutation,
    )


def _source_tree_sha256(identities: Sequence[Mapping[str, Any]]) -> str:
    members = sorted(
        (
            {
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
            for identity in identities
        ),
        key=lambda identity: (identity["sha256"], identity["size_bytes"]),
    )
    return _canonical_sha256(
        {
            "kind": "amplicol-source-content-set",
            "schema_version": 1,
            "members": members,
        }
    )


def _source_evidence(repository: Path) -> SourceEvidence:
    legacy_amplicol.validate_checkout(repository)
    completed = _run(
        ("git", "ls-files", "-z"),
        cwd=repository,
        label="original-AmpliCol tracked-source inventory",
    )
    source_paths: list[Path] = []
    seen: set[Path] = set()
    for item in completed.stdout.split("\0"):
        if not item:
            continue
        path = (repository / item).resolve(strict=True)
        if not path.is_file():
            _die(f"tracked original-AmpliCol source is not a file: {path}")
        if path in seen:
            _die(f"tracked original-AmpliCol sources resolve to duplicate path: {path}")
        seen.add(path)
        source_paths.append(path)
    if not source_paths:
        _die("original-AmpliCol tracked-source inventory is empty")
    source_files = tuple(_file_ref(path) for path in sorted(source_paths))
    compiler = legacy_amplicol._compiler_provenance(repository)
    source = {
        "revision": legacy_amplicol.expected_revision(),
        "dirty": False,
        "compiler": {
            "id": compiler.identity,
            "version": compiler.version,
            "target": compiler.target,
            "flags_sha256": _canonical_sha256(list(compiler.flags)),
        },
        "source_tree_sha256": _source_tree_sha256(source_files),
    }
    return SourceEvidence(source=source, source_files=source_files)


def _csv_ints(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _sample_command(
    *,
    role: str,
    round_index: int,
    points: int,
    contract: CaptureContract,
    context: ProbeContext,
) -> list[str]:
    expected = contract.expected
    selector_argument = (
        f"--color-flow-id={expected['color_flow']['id']}"
        if role == SELECTED_ROLE
        else f"--helicity-id={expected['helicity']['id']}"
    )
    return [
        str(context.worker_executable),
        "_sample",
        f"--workload={role}",
        f"--round={round_index}",
        f"--momenta={contract.fixture['path']}",
        f"--momenta-sha256={contract.fixture['file']['sha256']}",
        f"--momenta-points-sha256={contract.fixture['points_sha256']}",
        f"--source-revision={expected['amplicol_source_revision']}",
        selector_argument,
        f"--color-flow-word={_csv_ints(expected['color_flow']['word'])}",
        f"--helicity-values={_csv_ints(expected['helicity']['values'])}",
        f"--evaluated-points={points}",
        f"--runtime={context.runtime}",
        f"--selected-binary={context.selected_binary}",
        f"--union-binary={context.union_binary}",
        f"--process-file={context.process_file}",
        f"--process-file-sha256={_file_sha256(context.process_file)}",
        f"--group={context.group}",
        f"--integral={context.integral}",
        f"--source-pdgs={_csv_ints(context.source_pdgs)}",
        f"--generated-pdgs={_csv_ints(context.generated_pdgs)}",
        f"--generated-color-order={_csv_ints(context.generated_color_order)}",
        f"--source-to-generated-permutation={_csv_ints(context.permutation)}",
    ]


def _parse_probe_stdout(
    stdout: str,
    *,
    role: str,
    sample_index: int,
    evaluated_points: int,
) -> dict[str, Any]:
    payload = _strict_json(stdout.encode("utf-8"), "probe stdout")
    if set(payload) != _PROBE_RESULT_KEYS:
        _die("probe stdout has unknown or missing keys")
    if (
        payload.get("kind") != "amplicol-m0-probe-result"
        or payload.get("schema_version") != 1
        or payload.get("role") != role
        or payload.get("sample_index") != sample_index
        or payload.get("evaluated_point_count") != evaluated_points
    ):
        _die("probe stdout identity differs from its command")
    elapsed = payload.get("elapsed_seconds")
    seconds_per_point = payload.get("seconds_per_point")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
        or isinstance(seconds_per_point, bool)
        or not isinstance(seconds_per_point, int | float)
        or not math.isfinite(float(seconds_per_point))
        or float(seconds_per_point) <= 0.0
        or not math.isclose(
            float(seconds_per_point),
            float(elapsed) / evaluated_points,
            rel_tol=0.0,
            abs_tol=max(1.0e-18, abs(float(seconds_per_point)) * 1.0e-15),
        )
    ):
        _die("probe stdout timing scalars are invalid or stale")
    for field in ("selected_totals", "resolved_sums"):
        rows = payload.get(field)
        if not isinstance(rows, list) or not rows:
            _die(f"probe stdout {field} is empty")
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    for value in row
                )
            ):
                _die(f"probe stdout {field} has an invalid complex pair")
    return payload


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _complex_pairs(values: Sequence[complex]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def _pairs_as_complex(values: object, label: str) -> tuple[complex, ...]:
    if not isinstance(values, list):
        _die(f"{label} must be a list")
    result: list[complex] = []
    for row in values:
        if not isinstance(row, list) or len(row) != 2:
            _die(f"{label} has an invalid complex pair")
        result.append(complex(float(row[0]), float(row[1])))
    return tuple(result)


def _timing_rows(output: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    in_summary = False
    for line in output.splitlines():
        if "Timing summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        tokens = line.strip().split()
        for index in range(1, len(tokens)):
            try:
                value = float(tokens[index])
            except ValueError:
                continue
            label = " ".join(tokens[:index]).casefold()
            if label in rows:
                _die(f"probe timing summary repeats row {label!r}")
            if not math.isfinite(value) or value <= 0.0:
                _die(f"probe timing row {label!r} is not finite and positive")
            rows[label] = value
            break
    return rows


def _timing_value(output: str, label: str) -> float:
    rows = _timing_rows(output)
    wanted = label.casefold()
    exact = rows.get(wanted)
    if exact is not None:
        return exact
    matches = [value for row, value in rows.items() if wanted in row]
    if len(matches) != 1:
        _die(f"probe output has no unique positive timing row for {label!r}")
    return matches[0]


def _library_environment(runtime: Path) -> dict[str, str]:
    root = str(runtime.resolve())
    result: dict[str, str] = {}
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        existing = os.environ.get(name)
        result[name] = root if not existing else f"{root}{os.pathsep}{existing}"
    return result


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    prior = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _entry(arguments: argparse.Namespace) -> Any:
    return legacy_amplicol.ProcessEntry(
        group=arguments.group,
        integral=arguments.integral,
        process_pdgs=tuple(arguments.generated_pdgs),
        color_order=tuple(arguments.generated_color_order),
    )


def _timed_selected(
    arguments: argparse.Namespace,
    *,
    source_pdgs: Sequence[int],
    first_point: Sequence[Sequence[float]],
) -> tuple[float, Any]:
    with tempfile.TemporaryDirectory(prefix="pac-m0-selected-", dir="/tmp") as raw:
        momentum_path = Path(raw) / "momenta.dat"
        _write_ordered_momenta(
            momentum_path,
            source_pdgs=source_pdgs,
            target_pdgs=arguments.generated_pdgs,
            momenta=first_point,
        )
        completed = _run(
            (
                arguments.selected_binary,
                str(arguments.evaluated_points),
                str(arguments.group),
                str(arguments.integral),
                momentum_path,
            ),
            cwd=arguments.runtime,
            environment=_library_environment(arguments.runtime),
            label="selected-flow original-AmpliCol probe",
        )
    output = completed.stdout + "\n" + completed.stderr
    return (
        _timing_value(output, "amplitude evaluation"),
        legacy_amplicol._parse_selected_flow_probe_output(output),
    )


def _timed_union(
    arguments: argparse.Namespace,
    *,
    source_pdgs: Sequence[int],
    first_point: Sequence[Sequence[float]],
) -> tuple[float, Any]:
    permutation = tuple(arguments.source_to_generated_permutation)
    ordered_helicities = [arguments.helicity_values[index] for index in permutation]
    with tempfile.TemporaryDirectory(prefix="pac-m0-union-", dir="/tmp") as raw:
        work = Path(raw)
        process_copy = work / "processes.txt"
        momentum_path = work / "momenta.dat"
        shutil.copy2(arguments.process_file, process_copy)
        _write_ordered_momenta(
            momentum_path,
            source_pdgs=source_pdgs,
            target_pdgs=arguments.generated_pdgs,
            momenta=first_point,
        )
        completed = _run(
            (
                arguments.union_binary,
                str(arguments.evaluated_points),
                str(arguments.group),
                str(arguments.integral),
                "lc",
                process_copy,
                momentum_path,
                *(str(value) for value in ordered_helicities),
            ),
            cwd=work,
            label="all-flow original-AmpliCol probe",
        )
    output = completed.stdout + "\n" + completed.stderr
    return (
        _timing_value(output, "total"),
        legacy_amplicol._parse_probe_output(output),
    )


def _sample_values(
    arguments: argparse.Namespace,
    *,
    source_pdgs: Sequence[int],
    points: Sequence[Sequence[Sequence[float]]],
    timed_result: Any,
) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    entry = _entry(arguments)
    if arguments.workload == SELECTED_ROLE and (
        timed_result.group != arguments.group
        or timed_result.integral != arguments.integral
        or tuple(timed_result.process_pdgs) != tuple(arguments.generated_pdgs)
        or tuple(timed_result.color_order) != tuple(arguments.generated_color_order)
    ):
        _die("selected-flow timing probe returned a different generated row")
    selected: list[complex] = []
    resolved: list[complex] = []
    for index, momenta in enumerate(points):
        if index == 0:
            probe = timed_result
        elif arguments.workload == SELECTED_ROLE:
            with _temporary_environment(_library_environment(arguments.runtime)):
                probe = legacy_amplicol.run_selected_flow_library_probe(
                    arguments.runtime,
                    entry=entry,
                    source_pdgs=source_pdgs,
                    momenta=momenta,
                    points=1,
                )
        else:
            probe = legacy_amplicol.run_color_probe(
                arguments.runtime,
                process_file=arguments.process_file,
                entry=entry,
                source_pdgs=source_pdgs,
                momenta=momenta,
                color_accuracy="lc",
                helicities=arguments.helicity_values,
            )
        value = complex(float(probe.value), 0.0)
        selected.append(value)
        if arguments.workload == SELECTED_ROLE:
            # The generated LC library's scalar is its internal resolved
            # helicity sum after serialized hel_fac/normalization application.
            resolved.append(value)
        else:
            partition_sum = probe.lc_partition_sum
            if partition_sum is None or not probe.lc_row_partitions:
                _die("all-flow color probe emitted no resolved LC-row partition")
            resolved.append(complex(float(partition_sum), 0.0))
    if not _same_complex_values(selected, resolved):
        _die("original-AmpliCol selected total and resolved sum do not close")
    return tuple(selected), tuple(resolved)


def _sample_payload(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.source_revision != legacy_amplicol.expected_revision():
        _die("sample source revision differs from the contributor-lock pin")
    expected_flow_id = "flow:" + _csv_ints(arguments.color_flow_word)
    expected_helicity_id = "h:" + ",".join(
        f"{value:+d}" for value in arguments.helicity_values
    )
    if (
        len(arguments.color_flow_word) != 8
        or len(arguments.helicity_values) != 9
        or any(value not in (-1, 1) for value in arguments.helicity_values)
        or len(arguments.source_pdgs) != 9
        or len(arguments.generated_pdgs) != 9
        or len(arguments.source_to_generated_permutation) != 9
    ):
        _die("sample physical selector or external-leg dimensions are invalid")
    if (
        arguments.workload == SELECTED_ROLE
        and arguments.color_flow_id != expected_flow_id
    ) or (
        arguments.workload == UNION_ROLE
        and arguments.helicity_id != expected_helicity_id
    ):
        _die("sample stable selector ID differs from its explicit physical values")
    for path, digest, label in (
        (arguments.momenta, arguments.momenta_sha256, "momenta"),
        (
            arguments.process_file,
            arguments.process_file_sha256,
            "process file",
        ),
    ):
        if _file_sha256(path.resolve(strict=True)) != digest:
            _die(f"sample {label} content differs from its command digest")
    if (
        not arguments.selected_binary.is_file()
        or not os.access(arguments.selected_binary, os.X_OK)
        or not arguments.union_binary.is_file()
        or not os.access(arguments.union_binary, os.X_OK)
    ):
        _die("sample probe binary is missing or non-executable")
    source_pdgs, points = _fixture_points(arguments.momenta)
    if tuple(arguments.source_pdgs) != source_pdgs:
        _die("sample source PDGs differ from the momenta fixture")
    if tuple(arguments.source_to_generated_permutation) != tuple(
        legacy_amplicol._permutation(
            arguments.source_pdgs,
            arguments.generated_pdgs,
        )
    ):
        _die("sample source-to-generated permutation is stale")
    if _canonical_sha256(points) != arguments.momenta_points_sha256:
        _die("sample momenta point digest differs from its command")
    if arguments.workload == SELECTED_ROLE:
        elapsed, timed_result = _timed_selected(
            arguments,
            source_pdgs=source_pdgs,
            first_point=points[0],
        )
    else:
        elapsed, timed_result = _timed_union(
            arguments,
            source_pdgs=source_pdgs,
            first_point=points[0],
        )
    selected, resolved = _sample_values(
        arguments,
        source_pdgs=source_pdgs,
        points=points,
        timed_result=timed_result,
    )
    seconds_per_point = elapsed / arguments.evaluated_points
    return {
        "kind": "amplicol-m0-probe-result",
        "schema_version": 1,
        "role": arguments.workload,
        "sample_index": arguments.round,
        "evaluated_point_count": arguments.evaluated_points,
        "elapsed_seconds": elapsed,
        "seconds_per_point": seconds_per_point,
        "selected_totals": _complex_pairs(selected),
        "resolved_sums": _complex_pairs(resolved),
    }


def _interleave_group_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        {
            "kind": "pyamplicol-m0-paired-interleave",
            "schema_version": 1,
            "records": [dict(record) for record in records],
        }
    )


def _comparison_payload(
    selected: Sequence[complex],
    resolved: Sequence[complex],
) -> dict[str, Any]:
    if not _same_complex_values(selected, resolved):
        _die("cannot publish a non-closing validation comparison")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(selected, resolved, strict=True)):
        difference = abs(left - right)
        relative = difference / max(abs(right), ATOL)
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
        rows.append({"point_index": index, "passes": True})
    return {
        "selected_totals": _complex_pairs(selected),
        "resolved_sums": _complex_pairs(resolved),
        "point_comparisons": rows,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passes": True,
    }


def _manifest(
    *,
    role: str,
    contract: CaptureContract,
    context: ProbeContext,
    source: SourceEvidence,
    binary_evidence: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    interleave_group_sha256: str,
    selected: Sequence[complex],
    resolved: Sequence[complex],
) -> dict[str, Any]:
    expected = contract.expected
    timing_values = [float(sample["seconds_per_point"]) for sample in samples]
    median = statistics.median(timing_values)
    mad = statistics.median(abs(value - median) for value in timing_values)
    union = role == UNION_ROLE
    payload = {
        "kind": "pyamplicol-amplicol-m0-raw-evidence",
        "schema_version": 1,
        "complete": True,
        "evidence_scope": "authoritative-host-capture-v1",
        "workload": UNION_WORKLOAD if union else SELECTED_WORKLOAD,
        "source": source.source,
        "host": contract.host,
        "process": {
            "expression": PROCESS_EXPRESSION,
            "normalized_expression": NORMALIZED_PROCESS,
        },
        "physical_axes": {
            "color_flow": contract.color_axis,
            "helicity": contract.helicity_axis,
        },
        "selector": {
            "color_flow_request": contract.color_flow_request,
            "resolved_color_flow_id": (None if union else expected["color_flow"]["id"]),
            "color_flow_word": expected["color_flow"]["word"],
            "helicity_request": contract.helicity_request,
            "resolved_helicity_id": (expected["helicity"]["id"] if union else None),
            "helicity_values": expected["helicity"]["values"],
            "sum_axis": "color_flow" if union else "helicity",
            "source_to_generated_permutation": list(context.permutation),
            "complete_physical_axes": True,
            "generation_specialized_axes": [],
        },
        "momenta": {
            "point_count": contract.fixture["point_count"],
            "points_sha256": contract.fixture["points_sha256"],
            "raw_file": contract.fixture["file"],
        },
        "normalization_sha256": expected["normalization_sha256"],
        "timing": {
            "boundary": "direct-library-total" if union else "amplitude-evaluation",
            "batch_semantics": "scalar-normalized-per-point",
            "statistics_contract": "subprocess-median-and-raw-mad-v1",
            "interleave_group_sha256": interleave_group_sha256,
            "sample_count": len(samples),
            "median_seconds_per_point": median,
            "mad_seconds_per_point": mad,
            "samples": [dict(sample) for sample in samples],
            "samples_sha256": _canonical_sha256(samples),
        },
        "validation": _comparison_payload(selected, resolved),
        "binary_evidence": dict(binary_evidence),
    }
    return _content_addressed(payload)


def _raw_sample(
    *,
    role: str,
    sample_index: int,
    command_sha256: str,
    stdout: str,
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    return _content_addressed(
        {
            "kind": "pyamplicol-amplicol-m0-raw-sample",
            "schema_version": 1,
            "role": role,
            "sample_index": sample_index,
            "command_sha256": command_sha256,
            "evaluated_point_count": parsed["evaluated_point_count"],
            "elapsed_seconds": parsed["elapsed_seconds"],
            "seconds_per_point": parsed["seconds_per_point"],
            "stdout": stdout,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        }
    )


def _adaptive_points(
    elapsed: float,
    *,
    warmup_points: int,
    target_seconds: float,
    minimum_points: int,
    maximum_points: int,
) -> int:
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        _die("probe calibration did not report a positive finite time")
    estimate = math.ceil(target_seconds * warmup_points / elapsed)
    return max(minimum_points, min(maximum_points, estimate))


def _run_sample_command(
    command: Sequence[str],
    *,
    role: str,
    round_index: int,
    points: int,
) -> tuple[str, dict[str, Any], str, str]:
    started = _utc_now()
    completed = _run(
        command,
        cwd=ROOT,
        label=f"{role} round {round_index}",
    )
    finished = _utc_now()
    parsed = _parse_probe_stdout(
        completed.stdout,
        role=role,
        sample_index=round_index,
        evaluated_points=points,
    )
    if completed.stderr:
        _die(f"{role} round {round_index} wrote unexpected stderr")
    return completed.stdout, parsed, started, finished


def _verify_immutable_files(identities: Sequence[Mapping[str, Any]]) -> None:
    for identity in identities:
        path = Path(str(identity["path"]))
        if _file_ref(path) != dict(identity):
            _die(f"content-addressed capture input drifted: {path}")


def _capture(arguments: argparse.Namespace) -> dict[str, Any]:
    request = arguments.request_template.resolve(strict=True)
    repository = arguments.repository.resolve(strict=False)
    output = arguments.output_directory.resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        _die(f"output directory must not contain prior state: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = _contract_from_request(request)
    host = _host_identity()
    if host != contract.host:
        _die("current host identity differs from the four pyAmpliCol captures")
    context = _prepare_probes(
        repository=repository,
        output=output,
        fixture_path=contract.fixture["path"],
        expected_flow_word=contract.expected["color_flow"]["word"],
        expected_permutation=contract.expected["external_leg_permutation"],
        jobs=arguments.jobs,
    )
    source = _source_evidence(repository)
    if source.source["revision"] != contract.expected["amplicol_source_revision"]:
        _die("prepared original-AmpliCol source differs from the request pin")

    executable_ref = _file_ref(context.worker_executable)
    linked_refs = tuple(_file_ref(path) for path in context.linked_files)
    binary_evidence = {
        "executable": executable_ref,
        "linked_libraries": list(linked_refs),
        "source_files": list(source.source_files),
    }

    points_by_role: dict[str, int] = {}
    explicit = {
        SELECTED_ROLE: arguments.selected_points,
        UNION_ROLE: arguments.union_points,
    }
    for role in ROLES:
        if explicit[role] is not None:
            points_by_role[role] = explicit[role]
            continue
        warmup_command = _sample_command(
            role=role,
            round_index=-1,
            points=arguments.warmup_points,
            contract=contract,
            context=context,
        )
        _stdout, parsed, _started, _finished = _run_sample_command(
            warmup_command,
            role=role,
            round_index=-1,
            points=arguments.warmup_points,
        )
        points_by_role[role] = _adaptive_points(
            float(parsed["elapsed_seconds"]),
            warmup_points=arguments.warmup_points,
            target_seconds=arguments.target_seconds,
            minimum_points=arguments.minimum_points,
            maximum_points=arguments.maximum_points,
        )

    samples: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    values: dict[str, tuple[tuple[complex, ...], tuple[complex, ...]]] = {}
    interleave_records: list[dict[str, Any]] = []
    raw_directory = output / "raw-samples"
    for round_index in range(MIN_SAMPLES):
        for offset, role in enumerate(ROLES):
            position = 2 * round_index + offset
            points = points_by_role[role]
            command = _sample_command(
                role=role,
                round_index=round_index,
                points=points,
                contract=contract,
                context=context,
            )
            command_sha256 = _canonical_sha256(command)
            stdout, parsed, started, finished = _run_sample_command(
                command,
                role=role,
                round_index=round_index,
                points=points,
            )
            selected = _pairs_as_complex(
                parsed["selected_totals"],
                f"{role} selected totals",
            )
            resolved = _pairs_as_complex(
                parsed["resolved_sums"],
                f"{role} resolved sums",
            )
            expected_values = (
                contract.selected_values
                if role == SELECTED_ROLE
                else contract.union_values
            )
            if not _same_complex_values(selected, expected_values):
                _die(f"{role} original-AmpliCol values differ from pyAmpliCol")
            prior = values.setdefault(role, (selected, resolved))
            if not _same_complex_values(prior[0], selected) or not _same_complex_values(
                prior[1],
                resolved,
            ):
                _die(f"{role} validation values changed between subprocess rounds")
            raw_payload = _raw_sample(
                role=role,
                sample_index=round_index,
                command_sha256=command_sha256,
                stdout=stdout,
                parsed=parsed,
            )
            raw_ref = _write_json(
                raw_directory / f"{position:02d}-{role}.json",
                raw_payload,
            )
            sample = {
                "sample_index": round_index,
                "interleave_round": round_index,
                "interleave_position": position,
                "started_at_utc": started,
                "finished_at_utc": finished,
                "subprocess": True,
                "command": command,
                "command_sha256": command_sha256,
                "evaluated_point_count": parsed["evaluated_point_count"],
                "elapsed_seconds": parsed["elapsed_seconds"],
                "seconds_per_point": parsed["seconds_per_point"],
                "interrupted": False,
                "raw_output_file": raw_ref,
            }
            samples[role].append(sample)
            interleave_records.append(
                {
                    "role": role,
                    "round": round_index,
                    "position": position,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "command_sha256": command_sha256,
                }
            )

    interleave_sha256 = _interleave_group_sha256(interleave_records)
    evidence_refs: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        selected, resolved = values[role]
        manifest = _manifest(
            role=role,
            contract=contract,
            context=context,
            source=source,
            binary_evidence=binary_evidence,
            samples=samples[role],
            interleave_group_sha256=interleave_sha256,
            selected=selected,
            resolved=resolved,
        )
        evidence_refs[role] = _write_json(
            output / f"{role}.json",
            manifest,
        )

    legacy_amplicol.validate_checkout(repository)
    _verify_immutable_files(
        (
            *contract.input_files,
            executable_ref,
            contract.fixture["file"],
            *linked_refs,
            *source.source_files,
            *(
                sample["raw_output_file"]
                for role_samples in samples.values()
                for sample in role_samples
            ),
        )
    )
    index = _content_addressed(
        {
            "kind": "pyamplicol-amplicol-m0-capture-index",
            "schema_version": 1,
            "complete": True,
            "request_template": contract.input_files[0],
            "source": source.source,
            "host": contract.host,
            "process": NORMALIZED_PROCESS,
            "paired_round_count": MIN_SAMPLES,
            "evaluated_points": points_by_role,
            "interleave_group_sha256": interleave_sha256,
            "amplicol_evidence": evidence_refs,
        }
    )
    _write_json(output / "capture-index.json", index)
    return index


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("must not be empty")
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _sha256(value: str) -> str:
    invalid = any(character not in "0123456789abcdef" for character in value)
    if len(value) != 64 or invalid:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def _revision(value: str) -> str:
    invalid = any(character not in "0123456789abcdef" for character in value)
    if len(value) != 40 or invalid:
        raise argparse.ArgumentTypeError("must be a full lowercase Git SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture",
        help="build the pinned probes and capture two authoritative M0 manifests",
    )
    capture.add_argument("--request-template", type=Path, required=True)
    capture.add_argument(
        "--repository",
        type=Path,
        default=legacy_amplicol.DEFAULT_REPOSITORY,
        help="clean original-AmpliCol checkout at the contributor-lock revision",
    )
    capture.add_argument("--output-directory", type=Path, required=True)
    capture.add_argument("--jobs", type=_positive_int, default=1)
    capture.add_argument(
        "--target-seconds",
        type=_positive_float,
        default=DEFAULT_TARGET_SECONDS,
    )
    capture.add_argument(
        "--warmup-points",
        type=_positive_int,
        default=DEFAULT_WARMUP_POINTS,
    )
    capture.add_argument(
        "--minimum-points",
        type=_positive_int,
        default=DEFAULT_MINIMUM_POINTS,
    )
    capture.add_argument(
        "--maximum-points",
        type=_positive_int,
        default=DEFAULT_MAXIMUM_POINTS,
    )
    capture.add_argument("--selected-points", type=_positive_int)
    capture.add_argument("--union-points", type=_positive_int)

    sample = subparsers.add_parser("_sample", help=argparse.SUPPRESS)
    sample.add_argument("--workload", choices=ROLES, required=True)
    sample.add_argument("--round", type=int, required=True)
    sample.add_argument("--momenta", type=Path, required=True)
    sample.add_argument("--momenta-sha256", type=_sha256, required=True)
    sample.add_argument("--momenta-points-sha256", type=_sha256, required=True)
    sample.add_argument("--source-revision", type=_revision, required=True)
    selectors = sample.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--color-flow-id")
    selectors.add_argument("--helicity-id")
    sample.add_argument(
        "--color-flow-word",
        type=_comma_separated_ints,
        required=True,
    )
    sample.add_argument(
        "--helicity-values",
        type=_comma_separated_ints,
        required=True,
    )
    sample.add_argument("--evaluated-points", type=_positive_int, required=True)
    sample.add_argument("--runtime", type=Path, required=True)
    sample.add_argument("--selected-binary", type=Path, required=True)
    sample.add_argument("--union-binary", type=Path, required=True)
    sample.add_argument("--process-file", type=Path, required=True)
    sample.add_argument("--process-file-sha256", type=_sha256, required=True)
    sample.add_argument("--group", type=_positive_int, required=True)
    sample.add_argument("--integral", type=_positive_int, required=True)
    sample.add_argument(
        "--source-pdgs",
        type=_comma_separated_ints,
        required=True,
    )
    sample.add_argument(
        "--generated-pdgs",
        type=_comma_separated_ints,
        required=True,
    )
    sample.add_argument(
        "--generated-color-order",
        type=_comma_separated_ints,
        required=True,
    )
    sample.add_argument(
        "--source-to-generated-permutation",
        type=_comma_separated_ints,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "_sample":
            if (
                arguments.workload == SELECTED_ROLE and arguments.color_flow_id is None
            ) or (arguments.workload == UNION_ROLE and arguments.helicity_id is None):
                _die("sample selector kind does not match its workload")
            payload = _sample_payload(arguments)
            sys.stdout.write(_canonical_bytes(payload).decode("utf-8") + "\n")
            return 0
        if arguments.maximum_points < arguments.minimum_points:
            _die("maximum-points must not be below minimum-points")
        index = _capture(arguments)
        sys.stdout.write(_canonical_bytes(index).decode("utf-8") + "\n")
        return 0
    except CaptureError as error:
        print(f"amplicol Z+6g M0 capture failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
