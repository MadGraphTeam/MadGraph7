# SPDX-License-Identifier: 0BSD
"""Export resolved LC recurrences through the native runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pyamplicol.api.errors import ArtifactError, ConfigurationError

if TYPE_CHECKING:
    from pyamplicol.api.protocols import Momenta


def provider_process_ids(artifact: Path) -> tuple[str, ...]:
    """Enumerate generated processes, aliases and routed provider members once."""

    from pyamplicol.artifacts import load_manifest
    from pyamplicol.artifacts.process_groups import read_process_groups

    manifest = load_manifest(artifact)
    identifiers: dict[str, None] = {}
    for process in manifest.processes:
        identifiers[str(process["id"])] = None
        for alias in cast(Sequence[Mapping[str, object]], process["aliases"]):
            identifiers[str(alias["id"])] = None
    groups = read_process_groups(artifact)
    if groups is not None:
        for member in groups.members():
            identifiers[member.process_id] = None
    return tuple(identifiers)


def _check_destination(output: Path) -> None:
    """Refuse an unwritable destination before any runtime is loaded."""

    if output.is_dir():
        raise ConfigurationError(f"--output {str(output)!r} is a directory")
    parent = output.parent
    if not parent.is_dir():
        raise ConfigurationError(
            f"--output directory {str(parent)!r} does not exist or is not a directory"
        )
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ConfigurationError(f"--output directory {str(parent)!r} is not writable")


def _replace_output(output: Path, content: str) -> None:
    """Write beside the destination, then rename: readers never see a partial file."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _nonnegative_indent(value: str) -> int:
    try:
        indent = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "indent must be a nonnegative integer"
        ) from exc
    if indent < 0:
        raise argparse.ArgumentTypeError("indent must be a nonnegative integer")
    return indent


def _program_key(program: Mapping[str, object]) -> tuple[str, str]:
    return str(program["process_key"]), str(program["colour_id"])


def _program_reference(program: Mapping[str, object]) -> tuple[str, str] | None:
    representative = str(program["representative_process_key"])
    if str(program["process_key"]) == representative:
        return None
    return representative, str(program["representative_colour_id"])


def _member_reference(program: Mapping[str, object]) -> dict[str, object]:
    target = _program_reference(program)
    if target is None:
        return dict(program)
    metadata = (
        "process_key",
        "process",
        "colour_id",
        "representative_process_key",
        "representative_colour_id",
        "caller_legs",
        "crossing",
        "group_factor",
        "helicity_outputs",
        "reference_values",
    )
    return {
        **{key: program[key] for key in metadata if key in program},
        "program_reference": {"process_key": target[0], "colour_id": target[1]},
    }


def add_export_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("artifact", type=Path, metavar="LC_DIR")
    parser.add_argument("-o", "--output", type=Path, required=True, metavar="FILE")
    parser.add_argument(
        "--process",
        action="append",
        default=None,
        metavar="ID",
        help="Export one process; repeat to select several (default: all members).",
    )
    parser.add_argument(
        "--color-flow",
        action="append",
        default=None,
        metavar="ID",
        help=(
            "Export a stable colour-flow ID; repeat (default: all provider flows). "
            "Flow IDs are per process and apply to every selected process, so "
            "combine this with --process on a multi-process artifact."
        ),
    )
    parser.add_argument(
        "--model-parameters",
        type=Path,
        metavar="FILE",
        help="JSON parameter mapping to apply before resolving the export.",
    )
    parser.add_argument(
        "--reference-points",
        type=Path,
        metavar="FILE",
        help="JSON momentum batch or process-key to batch map for f64 references.",
    )
    parser.add_argument(
        "--indent",
        type=_nonnegative_indent,
        metavar="N",
        help="Indent capture JSON by N spaces (default: compact JSON).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the completion summary as JSON (the export is always JSON).",
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")


@dataclass(frozen=True, slots=True)
class RecurrenceExportInvocation:
    artifact: Path
    output: Path
    processes: tuple[str, ...] | None = None
    color_flows: tuple[str, ...] | None = None
    model_parameters: Path | None = None
    reference_points: Path | None = None
    indent: int | None = None
    output_format: Literal["human", "json"] = "human"
    output_color: Literal["auto", "always", "never"] = "auto"

    def run(self) -> object:
        from pyamplicol.api import Runtime

        from .utilities import UtilityResult

        _check_destination(self.output)
        parameters = None
        if self.model_parameters is not None:
            parameters = json.loads(self.model_parameters.read_text(encoding="utf-8"))
            if not isinstance(parameters, Mapping):
                raise ConfigurationError(
                    "--model-parameters must contain a JSON object"
                )
        references = None
        if self.reference_points is not None:
            references = json.loads(self.reference_points.read_text(encoding="utf-8"))
            if not isinstance(references, (list, Mapping)):
                raise ConfigurationError(
                    "--reference-points must contain a JSON momentum batch "
                    "or a process-key to batch mapping"
                )
        processes = (
            provider_process_ids(self.artifact)
            if self.processes is None
            else self.processes
        )
        if len(set(processes)) != len(processes):
            raise ConfigurationError("--process IDs must be unique")
        combined: dict[str, object] | None = None
        programs: dict[tuple[str, str], Mapping[str, object]] = {}
        owners: dict[tuple[str, str], str] = {}
        kernels: dict[str, object] = {}
        resolved_parameters: dict[str, Mapping[str, object]] = {}

        requests = {process_id: self.color_flows for process_id in processes}
        # Export requested processes first, then any representatives omitted by
        # a member-only selection. Only requested processes receive samples.
        for requested in (True, False):
            for process_id, colours in requests.items():
                runtime = Runtime.load(
                    self.artifact,
                    process=process_id,
                    model_parameters=parameters,
                )
                export = runtime.export_recurrence(color_flows=colours)
                if requested and references is not None:
                    from pyamplicol.api.recurrence_export import attach_reference_points

                    points = references
                    if isinstance(references, Mapping):
                        selected_programs = cast(
                            Sequence[Mapping[str, object]], export["programs"]
                        )
                        process_key = str(selected_programs[0]["process_key"])
                        try:
                            points = references[process_key]
                        except KeyError:
                            raise ConfigurationError(
                                "--reference-points has no batch for process "
                                f"{process_key!r}"
                            ) from None
                        if not isinstance(points, list):
                            raise ConfigurationError(
                                "--reference-points batch for process "
                                f"{process_key!r} must be a JSON momentum batch"
                            )
                    export = attach_reference_points(
                        export, runtime, cast("Momenta", points)
                    )
                identity = {
                    key: value
                    for key, value in export.items()
                    if key not in {"programs", "kernels", "parameters"}
                }
                if combined is None:
                    combined = identity
                elif identity != combined:
                    raise ArtifactError(
                        f"recurrence export identity differs for {process_id!r}"
                    )
                for parameter in cast(
                    Sequence[Mapping[str, object]], export["parameters"]
                ):
                    name = str(parameter["name"])
                    previous = resolved_parameters.get(name)
                    if previous is not None and previous != parameter:
                        raise ArtifactError(
                            f"recurrence export parameter {name!r} differs for "
                            f"{process_id!r}"
                        )
                    resolved_parameters[name] = parameter
                for program in cast(Sequence[Mapping[str, object]], export["programs"]):
                    key = _program_key(program)
                    previous = programs.get(key)
                    if previous is None:
                        programs[key] = program
                        owners[key] = process_id
                    elif previous != program:
                        # Two selections of one stored process in different
                        # leg orders share a key; keeping one would drop the
                        # other's crossing silently.
                        raise ConfigurationError(
                            f"--process {owners[key]!r} and {process_id!r} both "
                            f"select process {key[0]!r} in different leg orders; "
                            "export them to separate files"
                        )
                for kernel in cast(Sequence[Mapping[str, object]], export["kernels"]):
                    kernels[str(kernel["sha256"])] = kernel
            if requested:
                missing: dict[str, dict[str, None]] = {}
                for program in programs.values():
                    target = _program_reference(program)
                    if target is not None and target not in programs:
                        missing.setdefault(target[0], {})[target[1]] = None
                requests = {
                    process: tuple(colours) for process, colours in missing.items()
                }
        if combined is None:
            raise ArtifactError("artifact has no processes to export")
        combined["parameters"] = [
            resolved_parameters[name] for name in sorted(resolved_parameters)
        ]
        combined["programs"] = [
            _member_reference(program) for program in programs.values()
        ]
        combined["kernels"] = list(kernels.values())
        content = (
            json.dumps(
                combined,
                indent=self.indent,
                separators=(",", ":") if self.indent is None else None,
                allow_nan=False,
            )
            + "\n"
        )
        _replace_output(self.output, content)
        return UtilityResult("recurrence export", "complete", self.output)


def parse_export_recurrence(arguments: Sequence[str]) -> RecurrenceExportInvocation:
    parser = argparse.ArgumentParser(prog="pyamplicol export-recurrence")
    add_export_options(parser)
    args = parser.parse_args(arguments)
    return RecurrenceExportInvocation(
        artifact=args.artifact,
        output=args.output,
        processes=None if args.process is None else tuple(args.process),
        color_flows=None if args.color_flow is None else tuple(args.color_flow),
        model_parameters=args.model_parameters,
        reference_points=args.reference_points,
        indent=args.indent,
        output_format="json" if args.json else "human",
        output_color=args.color,
    )
