# SPDX-License-Identifier: 0BSD
"""Export resolved LC recurrences through the native runtime."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pyamplicol.api.errors import ArtifactError, ConfigurationError


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
        help="Export a stable colour-flow ID; repeat (default: all provider flows).",
    )
    parser.add_argument(
        "--model-parameters",
        type=Path,
        metavar="FILE",
        help="JSON parameter mapping to apply before resolving the export.",
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
    output_format: Literal["human", "json"] = "human"
    output_color: Literal["auto", "always", "never"] = "auto"

    def run(self) -> object:
        from pyamplicol.api import Runtime
        from pyamplicol.artifacts import load_manifest
        from pyamplicol.artifacts.process_groups import read_process_groups

        from .utilities import UtilityResult

        parameters = None
        if self.model_parameters is not None:
            parameters = json.loads(self.model_parameters.read_text(encoding="utf-8"))
            if not isinstance(parameters, Mapping):
                raise ConfigurationError(
                    "--model-parameters must contain a JSON object"
                )
        processes = self.processes
        if processes is None:
            manifest = load_manifest(self.artifact)
            identifiers: dict[str, None] = {}
            for process in manifest.processes:
                identifiers[str(process["id"])] = None
                for alias in cast(Sequence[Mapping[str, object]], process["aliases"]):
                    identifiers[str(alias["id"])] = None
            groups = read_process_groups(self.artifact)
            if groups is not None:
                for member in groups.members():
                    identifiers[member.process_id] = None
            processes = tuple(identifiers)
        if len(set(processes)) != len(processes):
            raise ConfigurationError("--process IDs must be unique")
        combined: dict[str, object] | None = None
        programs: list[object] = []
        kernels: dict[str, object] = {}
        resolved_parameters: dict[str, Mapping[str, object]] = {}
        for process_id in processes:
            runtime = Runtime.load(
                self.artifact,
                process=process_id,
                model_parameters=parameters,
            )
            export = runtime.export_recurrence(color_flows=self.color_flows)
            identity = {
                key: value
                for key, value in export.items()
                if key not in {"programs", "kernels", "parameters"}
            }
            if combined is None:
                combined = identity
            elif identity != combined:
                raise ArtifactError(
                    "recurrence export identity differs for "
                    f"{process_id!r}"
                )
            for parameter in cast(Sequence[Mapping[str, object]], export["parameters"]):
                name = str(parameter["name"])
                previous = resolved_parameters.get(name)
                if previous is not None and previous != parameter:
                    raise ArtifactError(
                        f"recurrence export parameter {name!r} differs for "
                        f"{process_id!r}"
                    )
                resolved_parameters[name] = parameter
            programs.extend(cast(Sequence[object], export["programs"]))
            for kernel in cast(Sequence[Mapping[str, object]], export["kernels"]):
                kernels[str(kernel["sha256"])] = kernel
        if combined is None:
            raise ArtifactError("artifact has no processes to export")
        combined["parameters"] = [
            resolved_parameters[name] for name in sorted(resolved_parameters)
        ]
        combined["programs"] = programs
        combined["kernels"] = list(kernels.values())
        content = json.dumps(combined, indent=2, allow_nan=False) + "\n"
        self.output.write_text(content, encoding="utf-8")
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
        output_format="json" if args.json else "human",
        output_color=args.color,
    )
