# SPDX-License-Identifier: 0BSD
"""MadGraph tree-level authority adapter for full-colour report cells.

The authority is MadGraph's installation-owned, default-restricted UFO ``sm``
model.  The adapter authenticates that model's external parameter identities
and values against the campaign UFO-SM inputs before evaluating either side.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.resources
import json
import math
import os
import shutil
import statistics
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from tools.developer import madgraph_correctness as _correctness
from tools.developer.madgraph_correctness import (
    MADGRAPH_DRIVER_SOURCE_SHA256,
    CommandExecutor,
    CommandResult,
    DriverResult,
    MadGraphAdapterError,
    StandaloneMadGraphRunner,
    SubprocessExecutor,
)
from tools.developer.madgraph_correctness import (
    discover_subprocess as _discover_subprocess,
)
from tools.developer.madgraph_correctness import (
    madgraph_command_card as _correctness_command_card,
)
from tools.developer.madgraph_correctness import (
    momenta_rows as _momenta_rows,
)
from tools.developer.madgraph_correctness import (
    reject_failed_generation as _reject_generic_generation,
)
from tools.developer.madgraph_correctness import (
    validate_installation as _validate_base_installation,
)

from .agreements import DIRECT_AGREEMENT_FIELD
from .cache import empty_measurement
from .measurement import shared_validation_points
from .models import Accuracy, CellSpec, ExecutionMode, ModelKey, ResultStatus, Workload
from .phase_state import WorkerPhaseReporter
from .runner import (
    DEFAULT_TARGET_RUNTIME_SECONDS,
    MADGRAPH_RELATIVE_TOLERANCE,
    point_digest,
)

_DRIVER_SOURCE = _correctness._DRIVER_SOURCE
_parse_driver_output = _correctness.parse_driver_output

DEFAULT_WARMUP_CALLS = 20
DEFAULT_MINIMUM_CALLS = 10
DEFAULT_MAXIMUM_CALLS = 10_000_000
DEFAULT_MINIMUM_PROFILE_CHUNKS = 5


@dataclass(frozen=True, slots=True)
class MadGraphSettings:
    installation: Path
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS
    warmup_calls: int = DEFAULT_WARMUP_CALLS
    minimum_calls: int = DEFAULT_MINIMUM_CALLS
    maximum_calls: int = DEFAULT_MAXIMUM_CALLS
    minimum_profile_chunks: int = DEFAULT_MINIMUM_PROFILE_CHUNKS
    json_model_path: Path | None = None
    restriction_json_path: Path | None = None
    worker_deadline_monotonic: float | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.target_runtime_seconds)
            or self.target_runtime_seconds <= 0.0
        ):
            raise ValueError("target_runtime_seconds must be finite and positive")
        if self.warmup_calls < DEFAULT_WARMUP_CALLS:
            raise ValueError(f"warmup_calls must be at least {DEFAULT_WARMUP_CALLS}")
        if self.minimum_calls < 1:
            raise ValueError("minimum_calls must be positive")
        if self.maximum_calls < self.minimum_calls:
            raise ValueError("maximum_calls must not be below minimum_calls")
        if self.minimum_profile_chunks < DEFAULT_MINIMUM_PROFILE_CHUNKS:
            raise ValueError(
                "minimum_profile_chunks must be at least "
                f"{DEFAULT_MINIMUM_PROFILE_CHUNKS}"
            )
        if self.worker_deadline_monotonic is not None and (
            not math.isfinite(self.worker_deadline_monotonic)
            or self.worker_deadline_monotonic <= 0.0
        ):
            raise ValueError("worker_deadline_monotonic must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = tuple(
        sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and not candidate.is_symlink()
            and "__pycache__" not in candidate.parts
            and candidate.suffix != ".pyc"
        )
    )
    if not files:
        raise MadGraphAdapterError(f"UFO model directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        payload = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _default_json_model_path() -> Path:
    resource = importlib.resources.files("pyamplicol").joinpath(
        "assets", "models", "json", "sm", "sm.json"
    )
    if not isinstance(resource, os.PathLike):
        raise MadGraphAdapterError("installed UFO-SM JSON is not filesystem-backed")
    try:
        result = Path(os.fspath(resource)).resolve(strict=True)
    except OSError as error:
        raise MadGraphAdapterError("packaged UFO-SM JSON is unavailable") from error
    if not result.is_file():
        raise MadGraphAdapterError("packaged UFO-SM JSON is not a regular file")
    return result


def _default_restriction_json_path() -> Path:
    resource = importlib.resources.files("pyamplicol").joinpath(
        "assets", "models", "json", "sm", "restrict_default.json"
    )
    if not isinstance(resource, os.PathLike):
        raise MadGraphAdapterError(
            "installed UFO-SM default restriction is not filesystem-backed"
        )
    try:
        result = Path(os.fspath(resource)).resolve(strict=True)
    except OSError as error:
        raise MadGraphAdapterError(
            "packaged UFO-SM default restriction is unavailable"
        ) from error
    if not result.is_file():
        raise MadGraphAdapterError(
            "packaged UFO-SM default restriction is not a regular file"
        )
    return result


def _expected_external_parameters(
    model_path: Path,
    restriction_path: Path,
) -> dict[tuple[str, tuple[int, ...]], tuple[str, float]]:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MadGraphAdapterError("cannot read packaged UFO-SM JSON") from error
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, list):
        raise MadGraphAdapterError("packaged UFO-SM JSON has no parameter list")
    try:
        restriction = json.loads(restriction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MadGraphAdapterError(
            "cannot read packaged UFO-SM default restriction"
        ) from error
    if not isinstance(restriction, Mapping):
        raise MadGraphAdapterError("packaged UFO-SM restriction is not an object")
    result: dict[tuple[str, tuple[int, ...]], tuple[str, float]] = {}
    observed_names: set[str] = set()
    for raw in raw_parameters:
        if not isinstance(raw, Mapping) or raw.get("nature") != "external":
            continue
        block = raw.get("lhablock")
        codes = raw.get("lhacode")
        name = raw.get("name")
        value = raw.get("value")
        if (
            not isinstance(block, str)
            or not isinstance(codes, list)
            or not codes
            or any(
                isinstance(code, bool) or not isinstance(code, int) for code in codes
            )
            or not isinstance(name, str)
            or not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or not isinstance(value[0], (int, float))
            or float(value[1]) != 0.0
        ):
            raise MadGraphAdapterError(
                "packaged UFO-SM external parameter is malformed"
            )
        key = (block.upper(), tuple(codes))
        if key in result:
            raise MadGraphAdapterError("packaged UFO-SM repeats an external LHA key")
        restricted = restriction.get(name)
        if (
            not isinstance(restricted, list)
            or len(restricted) != 2
            or isinstance(restricted[0], bool)
            or not isinstance(restricted[0], (int, float))
            or float(restricted[1]) != 0.0
        ):
            raise MadGraphAdapterError(
                f"default restriction has no real value for external {name}"
            )
        observed_names.add(name)
        result[key] = (name, float(restricted[0]))
    if not result:
        raise MadGraphAdapterError("packaged UFO-SM has no external parameters")
    extra = set(restriction).difference(observed_names)
    if extra:
        raise MadGraphAdapterError(
            f"default restriction contains unknown externals: {sorted(extra)!r}"
        )
    return result


def _ufo_external_parameters(
    parameters_path: Path,
) -> dict[tuple[str, tuple[int, ...]], tuple[str, float]]:
    """Read literal UFO external declarations without importing model code."""

    try:
        tree = ast.parse(parameters_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        raise MadGraphAdapterError("cannot parse MadGraph UFO parameters.py") from error
    result: dict[tuple[str, tuple[int, ...]], tuple[str, float]] = {}
    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(node.value, ast.Call)
            or not isinstance(node.value.func, ast.Name)
            or node.value.func.id != "Parameter"
        ):
            continue
        keywords = {
            keyword.arg: keyword.value
            for keyword in node.value.keywords
            if keyword.arg is not None
        }
        try:
            nature = ast.literal_eval(keywords["nature"])
        except (KeyError, ValueError, TypeError):
            continue
        if nature != "external":
            continue
        try:
            name = ast.literal_eval(keywords["name"])
            value = ast.literal_eval(keywords["value"])
            block = ast.literal_eval(keywords["lhablock"])
            codes = ast.literal_eval(keywords["lhacode"])
        except (KeyError, ValueError, TypeError) as error:
            raise MadGraphAdapterError(
                "MadGraph UFO external parameter is not literal"
            ) from error
        if (
            not isinstance(name, str)
            or not isinstance(block, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isinstance(codes, list)
            or not codes
            or any(
                isinstance(code, bool) or not isinstance(code, int) for code in codes
            )
        ):
            raise MadGraphAdapterError("MadGraph UFO external parameter is malformed")
        key = (block.upper(), tuple(codes))
        if key in result:
            raise MadGraphAdapterError("MadGraph UFO repeats an external LHA key")
        result[key] = (name, float(value))
    if not result:
        raise MadGraphAdapterError("MadGraph UFO has no literal external parameters")
    return result


def _write_exact_param_card(
    card_path: Path,
    *,
    parameters_path: Path,
    model_path: Path,
    restriction_path: Path,
) -> dict[str, object]:
    """Write the exact external LHA inputs within MG's 20-character limit."""

    expected = _expected_external_parameters(model_path, restriction_path)
    madgraph_parameters = _ufo_external_parameters(parameters_path)
    expected_identity = {key: value[0] for key, value in expected.items()}
    madgraph_identity = {key: value[0] for key, value in madgraph_parameters.items()}
    if madgraph_identity != expected_identity:
        missing = sorted(set(expected).difference(madgraph_parameters))
        extra = sorted(set(madgraph_parameters).difference(expected))
        changed = tuple(
            (key, expected[key][0], madgraph_parameters[key][0])
            for key in sorted(set(expected).intersection(madgraph_parameters))
            if expected[key][0] != madgraph_parameters[key][0]
        )
        raise MadGraphAdapterError(
            "MadGraph UFO external identities differ from the packaged JSON model: "
            f"missing={missing!r}, extra={extra!r}, changed={changed!r}"
        )
    blocks = sorted(
        {key[0] for key in expected if key[0] != "DECAY"},
        key=lambda name: (
            {"SMINPUTS": 0, "MASS": 1}.get(name, 2),
            name,
        ),
    )
    lines = [
        "######################################################################",
        "## Exact packaged UFO-SM external parameters for MadGraph validation",
        "######################################################################",
        "",
    ]
    for block in blocks:
        lines.append(f"Block {block}")
        for key in sorted(candidate for candidate in expected if candidate[0] == block):
            name, value = expected[key]
            rendered = f"{value:.14e}"
            if len(rendered) > 20:
                raise MadGraphAdapterError(
                    f"external parameter {name} exceeds MadGraph's LHA value field"
                )
            codes = " ".join(str(code) for code in key[1])
            lines.append(f"  {codes} {rendered} # {name}")
        lines.append("")
    for key in sorted(candidate for candidate in expected if candidate[0] == "DECAY"):
        name, value = expected[key]
        rendered = f"{value:.14e}"
        if len(rendered) > 20:
            raise MadGraphAdapterError(
                f"external parameter {name} exceeds MadGraph's LHA value field"
            )
        codes = " ".join(str(code) for code in key[1])
        lines.append(f"DECAY {codes} {rendered} # {name}")
    card_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return _validate_exact_param_card(
        card_path,
        model_path,
        restriction_path,
    )


def _validate_exact_param_card(
    card_path: Path,
    model_path: Path,
    restriction_path: Path,
) -> dict[str, object]:
    expected = _expected_external_parameters(model_path, restriction_path)
    observed: dict[tuple[str, tuple[int, ...]], float] = {}
    block: str | None = None
    for raw_line in card_path.read_text(encoding="utf-8", errors="strict").splitlines():
        content = raw_line.partition("#")[0].strip()
        if not content:
            continue
        tokens = content.split()
        keyword = tokens[0].upper()
        if keyword == "BLOCK" and len(tokens) >= 2:
            block = tokens[1].upper()
            continue
        if keyword == "DECAY" and len(tokens) >= 3:
            key = ("DECAY", (int(tokens[1]),))
            value_token = tokens[2]
        elif block is not None and len(tokens) >= 2:
            try:
                codes = tuple(int(token) for token in tokens[:-1])
            except ValueError:
                continue
            key = (block, codes)
            value_token = tokens[-1]
        else:
            continue
        if key not in expected:
            continue
        if key in observed:
            raise MadGraphAdapterError(
                "exact parameter card repeats an external LHA key"
            )
        try:
            observed[key] = float(value_token.replace("D", "E").replace("d", "e"))
        except ValueError as error:
            raise MadGraphAdapterError(
                "exact parameter card contains a malformed external value"
            ) from error
    if set(observed) != set(expected):
        missing = sorted(set(expected).difference(observed))
        raise MadGraphAdapterError(
            "exact parameter card does not cover packaged UFO-SM externals: "
            f"missing={missing!r}"
        )
    mismatches = tuple(
        (key, expected[key][1], observed[key])
        for key in sorted(expected)
        if expected[key][1] != observed[key]
    )
    if mismatches:
        raise MadGraphAdapterError(
            "exact parameter card differs from packaged UFO-SM defaults: "
            f"{mismatches!r}"
        )
    canonical = json.dumps(
        {expected[key][0]: [observed[key], 0.0] for key in sorted(expected)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "external_parameter_count": len(expected),
        "external_parameters_sha256": hashlib.sha256(canonical).hexdigest(),
        "binary64_exact_match": True,
        "format": "%.14e",
    }


def _validate_installation(path: Path) -> tuple[Path, Path, Path]:
    installation, launcher = _validate_base_installation(path)
    model = installation / "models" / "sm"
    required_model_files = (model / "parameters.py", model / "restrict_default.dat")
    if (
        model.is_symlink()
        or not model.is_dir()
        or any(path.is_symlink() or not path.is_file() for path in required_model_files)
    ):
        raise MadGraphAdapterError(
            "MadGraph installation must contain its regular UFO models/sm model"
        )
    return installation, launcher, model


def madgraph_command_card(process: str) -> str:
    return _correctness_command_card(process)


def _reject_failed_generation(
    result: CommandResult,
    standalone: Path,
) -> None:
    try:
        _reject_generic_generation(
            result,
            standalone,
            expected_model_import="sm",
        )
    except MadGraphAdapterError as error:
        if "bound exclusively" not in str(error):
            raise
        raise MadGraphAdapterError(
            "standalone process card is not bound exclusively to MadGraph UFO sm"
        ) from error


class MadGraphMeasurementAdapter(StandaloneMadGraphRunner):
    """Generate, drive, and profile one standalone MadGraph matrix element."""

    @staticmethod
    def _assert_deadline(settings: MadGraphSettings) -> None:
        deadline = settings.worker_deadline_monotonic
        if deadline is not None and time.monotonic() >= deadline:
            raise MadGraphAdapterError("MadGraph worker deadline was exhausted")

    def measure(
        self,
        cell: CellSpec,
        *,
        artifact_path: Path,
        settings: MadGraphSettings,
        phase_reporter: WorkerPhaseReporter | None = None,
    ) -> dict[str, object]:
        if (
            cell.measurement.execution_mode is not ExecutionMode.MADGRAPH
            or cell.measurement.model is not ModelKey.UFO_SM
            or cell.measurement.accuracy is not Accuracy.FULL
            or cell.workload is not Workload.CONTRACTED
        ):
            raise MadGraphAdapterError(
                "MadGraph adapter requires a contracted full-colour UFO-SM "
                "reference cell"
            )
        installation, launcher, madgraph_model = _validate_installation(
            settings.installation
        )
        json_model_path = settings.json_model_path or _default_json_model_path()
        restriction_json_path = (
            settings.restriction_json_path or _default_restriction_json_path()
        )
        artifact_path.mkdir(parents=True, exist_ok=True)
        log_path = artifact_path / "madgraph.log"
        standalone = artifact_path / "standalone"
        model_record = {
            "name": "sm",
            "source_directory": os.fspath(madgraph_model),
            "source_sha256": _directory_digest(madgraph_model),
        }
        command_card = madgraph_command_card(cell.process)
        command_card_path = artifact_path / "madgraph_command_card.dat"
        command_card_path.write_text(command_card, encoding="utf-8")
        commands: list[dict[str, object]] = []

        generation_context = (
            nullcontext() if phase_reporter is None else phase_reporter.generation()
        )
        with generation_context:
            self._assert_deadline(settings)
            generation = self._run_logged(
                self.executor,
                (launcher, command_card_path),
                cwd=artifact_path,
                log_path=log_path,
            )
            commands.append(generation.record())
            _reject_failed_generation(generation, standalone)
            restriction_card_record = _validate_exact_param_card(
                madgraph_model / "restrict_default.dat",
                json_model_path,
                restriction_json_path,
            )
            param_card = standalone / "Cards" / "param_card.dat"
            rounded_param_card = (
                standalone / "Cards" / "param_card.madgraph-rounded.dat"
            )
            shutil.copy2(param_card, rounded_param_card)
            exact_param_card_record = _write_exact_param_card(
                param_card,
                parameters_path=madgraph_model / "parameters.py",
                model_path=json_model_path,
                restriction_path=restriction_json_path,
            )
            subprocess_dir = _discover_subprocess(standalone)
            executable, compilation, driver_digest = self._compile_driver(
                standalone,
                subprocess_dir,
                log_path=log_path,
            )
            commands.append(compilation.record())

        validation_points = shared_validation_points(cell.process)
        momenta = _momenta_rows(validation_points)
        momenta_path = subprocess_dir / "pyamplicol_momenta.dat"
        momenta_path.write_text(
            "".join(
                " ".join(format(component, ".17e") for component in row) + "\n"
                for row in momenta
            ),
            encoding="ascii",
        )
        if phase_reporter is not None:
            phase_reporter.profiling_started()

        self._assert_deadline(settings)
        calibration = self._run_driver(
            executable,
            subprocess_dir=subprocess_dir,
            momenta_path=momenta_path,
            points=settings.minimum_calls,
            warmup_calls=settings.warmup_calls,
            log_path=log_path,
        )
        commands.append(calibration.command.record())
        calibration_seconds = calibration.seconds
        calibration_calls = calibration.points
        target_chunk_seconds = (
            settings.target_runtime_seconds / settings.minimum_profile_chunks
        )
        calibration_target_seconds = min(0.1, 0.25 * target_chunk_seconds)
        while (
            calibration_seconds < calibration_target_seconds
            and calibration_calls < settings.maximum_calls
        ):
            if calibration_seconds <= 0.0:
                next_calls = calibration_calls * 10
            else:
                next_calls = math.ceil(
                    1.25
                    * calibration_calls
                    * calibration_target_seconds
                    / calibration_seconds
                )
            calibration_calls = min(
                settings.maximum_calls,
                max(calibration_calls + 1, next_calls),
            )
            calibration = self._run_driver(
                executable,
                subprocess_dir=subprocess_dir,
                momenta_path=momenta_path,
                points=calibration_calls,
                warmup_calls=settings.warmup_calls,
                log_path=log_path,
            )
            commands.append(calibration.command.record())
            calibration_seconds = calibration.seconds
        if calibration_seconds <= 0.0:
            raise MadGraphAdapterError("MadGraph driver timer did not advance")
        seconds_per_call = calibration_seconds / calibration.points
        profile_calls = max(
            settings.minimum_calls,
            min(
                settings.maximum_calls,
                math.ceil(target_chunk_seconds / seconds_per_call),
            ),
        )
        chunks: list[DriverResult] = []
        attempted_profile_chunks = 0
        zero_duration_profile_chunks = 0
        while not chunks:
            attempt: list[DriverResult] = []
            for _ in range(settings.minimum_profile_chunks):
                self._assert_deadline(settings)
                chunk = self._run_driver(
                    executable,
                    subprocess_dir=subprocess_dir,
                    momenta_path=momenta_path,
                    points=profile_calls,
                    warmup_calls=settings.warmup_calls,
                    log_path=log_path,
                )
                attempted_profile_chunks += 1
                commands.append(chunk.command.record())
                if not math.isclose(
                    chunk.value,
                    calibration.value,
                    rel_tol=5.0e-13,
                    abs_tol=1.0e-300,
                ):
                    raise MadGraphAdapterError(
                        "MadGraph driver value changed between independent "
                        "profile chunks"
                    )
                if chunk.seconds <= 0.0:
                    zero_duration_profile_chunks += 1
                    break
                attempt.append(chunk)
            if len(attempt) == settings.minimum_profile_chunks:
                chunks = attempt
                break
            if profile_calls >= settings.maximum_calls:
                raise MadGraphAdapterError(
                    "MadGraph profile timer did not advance at maximum_calls"
                )
            profile_calls = min(
                settings.maximum_calls,
                max(profile_calls + 1, profile_calls * 10),
            )
        if phase_reporter is not None:
            phase_reporter.validation_started()

        total_points = sum(chunk.points for chunk in chunks)
        total_seconds = math.fsum(chunk.seconds for chunk in chunks)
        if total_seconds <= 0.0:
            raise MadGraphAdapterError("MadGraph profile accumulated no positive time")
        rates = tuple(chunk.seconds / chunk.points for chunk in chunks)
        mean_rate = total_seconds / total_points
        standard_error = statistics.stdev(rates) / math.sqrt(len(rates))
        relative_standard_error = standard_error / mean_rate
        version_path = installation / "VERSION"

        measurement = empty_measurement()
        measurement.update(
            {
                "status": ResultStatus.OK.value,
                "generation_seconds": generation.elapsed_seconds,
                "wall_seconds_per_point": mean_rate,
                "execution_seconds_per_point": mean_rate,
                "matrix_element": calibration.value,
                "sample_count": total_points,
                "standard_error_seconds_per_point": standard_error,
                "relative_standard_error": relative_standard_error,
                "artifact": {
                    "path": os.fspath(artifact_path),
                    "standalone": os.fspath(standalone),
                    "subprocess": os.fspath(subprocess_dir.relative_to(artifact_path)),
                    "log_path": os.fspath(log_path),
                },
                "selector_contract": None,
                "validation": {
                    "status": ResultStatus.OK.value,
                    "method": "independent-madgraph-tree-level-oracle",
                    "point_digest": point_digest(validation_points),
                    DIRECT_AGREEMENT_FIELD: [],
                },
                "resources": {
                    "monitor": "external-cell-supervisor",
                    "peak_rss_gib": None,
                },
                "provenance": {
                    "method": "madgraph-standalone-custom-fortran-driver",
                    "installation": os.fspath(installation),
                    "version": (
                        None
                        if not version_path.is_file()
                        else version_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).strip()
                    ),
                    "version_sha256": (
                        None if not version_path.is_file() else _sha256(version_path)
                    ),
                    "model": model_record,
                    "default_restriction": restriction_card_record,
                    "command_card": command_card,
                    "command_card_sha256": _sha256(command_card_path),
                    "process_card_sha256": _sha256(
                        standalone / "Cards" / "proc_card_mg5.dat"
                    ),
                    "param_card_sha256": _sha256(param_card),
                    "madgraph_rounded_param_card_sha256": _sha256(rounded_param_card),
                    "exact_param_card": exact_param_card_record,
                    "driver_sha256": driver_digest,
                    "driver_compile_seconds": compilation.elapsed_seconds,
                    "driver_timer": "fortran-system-clock-int64",
                    "driver_warmup_calls": settings.warmup_calls,
                    "report_momenta": validation_points,
                    "profile_chunk_count": len(chunks),
                    "profile_calls_per_chunk": profile_calls,
                    "profile_attempted_chunk_count": attempted_profile_chunks,
                    "profile_discarded_chunk_count": (
                        attempted_profile_chunks - len(chunks)
                    ),
                    "profile_zero_duration_chunk_count": (zero_duration_profile_chunks),
                    "commands": commands,
                    "generation_timing_scope": (
                        "generate-output-standalone-launch-force"
                    ),
                    "generation_includes_madgraph_compilation": True,
                    "driver_compilation_in_generation_seconds": False,
                },
                "failure": None,
            }
        )
        return measurement


__all__ = [
    "DEFAULT_MAXIMUM_CALLS",
    "DEFAULT_MINIMUM_CALLS",
    "DEFAULT_MINIMUM_PROFILE_CHUNKS",
    "DEFAULT_WARMUP_CALLS",
    "MADGRAPH_DRIVER_SOURCE_SHA256",
    "MADGRAPH_RELATIVE_TOLERANCE",
    "CommandExecutor",
    "CommandResult",
    "MadGraphAdapterError",
    "MadGraphMeasurementAdapter",
    "MadGraphSettings",
    "SubprocessExecutor",
    "madgraph_command_card",
]
