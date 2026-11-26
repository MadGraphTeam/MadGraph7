# SPDX-License-Identifier: 0BSD
"""Small C data emitter shared by generated UMAMI providers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _string(value: object) -> str:
    return "NULL" if value is None else json.dumps(str(value), ensure_ascii=True)


def _number(value: object) -> str:
    return format(float(value), ".17g")


def _boolean(value: object) -> str:
    return "true" if value else "false"


def _power(process: Mapping[str, Any]) -> str:
    power = process.get("alpha_s_power")
    return "-1" if power is None else str(int(power))


def helicity_export(
    data: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Sequence[int]:
    """The exported helicity of each helicity of a runtime record: from schema 6
    its ``helicity_export`` is an index into the document's ``helicity_exports``."""
    export = runtime["helicity_export"]
    if data.get("schema_version", 0) < 6:
        return export
    if isinstance(export, bool) or not isinstance(export, int):
        raise ValueError(f"expected a helicity export index, got {export!r}")
    return data["helicity_exports"][export]


def provider_header(data: Mapping[str, Any]) -> str:
    """Render the private ABI defined by ``umami_provider.h``."""
    provider = data["provider"]
    lines = [
        "/* SPDX-License-Identifier: 0BSD */",
        "/* Generated from the same records as metadata.json. */",
        '#include "umami_provider.h"',
        f"#define UMAMI_HAS_ALPHA_S {int(provider['supports_alpha_s'])}",
        f"#define UMAMI_HAS_COLOR_FLOW {int(provider['color_accuracy'] == 'lc')}",
        f"#define UMAMI_ARTIFACT_ID {_string(provider.get('artifact_id'))}",
        "static const double umami_masses[] = {"
        + ", ".join(_number(v) for v in provider["masses"])
        + "};",
    ]
    process_rows = []
    for index, process in enumerate(data["runtime_processes"]):
        prefix = f"umami_p{index}"
        lines.append(
            f"static const int {prefix}_pdgs[] = {{"
            + ", ".join(str(v) for v in process["pdgs"])
            + "};"
        )
        helicities = process["helicity_ids"]
        parameters = process["parameters"]
        if helicities:
            lines.append(
                f"static const char *const {prefix}_helicities[] = {{"
                + ", ".join(_string(v) for v in helicities)
                + "};"
            )
        if parameters:
            lines.append(f"static const UmamiParameter {prefix}_parameters[] = {{")
            for p in parameters:
                lines.append(
                    "    {"
                    + ", ".join(
                        (
                            _string(p["name"]),
                            _number(p["default_real"]),
                            _number(p.get("default_imaginary", 0.0)),
                            _boolean(p["mutable"]),
                            _boolean(
                                p.get(
                                    "is_complex", p.get("default_imaginary", 0.0) != 0.0
                                )
                            ),
                        )
                    )
                    + "},"
                )
            lines.append("};")
        process_rows.append(
            "    {"
            + ", ".join(
                (
                    _string(process["id"]),
                    f"{prefix}_pdgs",
                    _string(process["alpha_s_parameter"]),
                    _power(process),
                    _string(process["color_accuracy"]),
                    str(len(helicities)),
                    f"{prefix}_helicities" if helicities else "NULL",
                    str(len(parameters)),
                    f"{prefix}_parameters" if parameters else "NULL",
                )
            )
            + "},"
        )
    lines.extend(
        ["static const UmamiProcess umami_processes[] = {", *process_rows, "};"]
    )
    # per-entry arrays interned by content: providers repeat few distinct ones
    interned: dict[tuple[str, tuple], str] = {}
    counts: dict[str, int] = {}

    def array(kind: str, ctype: str, values: list[str]) -> str:
        key = (kind, tuple(values))
        name = interned.get(key)
        if name is None:
            counts[kind] = counts.get(kind, 0) + 1
            name = interned[key] = f"umami_{kind}{counts[kind] - 1}"
            lines.append(f"static const {ctype} {name}[] = {{" + ", ".join(values) + "};")
        return name

    contributions: list[str] = []
    channel_rows = []
    for channel_index, channel in enumerate(data["channels"]):
        first = len(contributions)
        for flavour in channel["processes"]:
            runtime = flavour["runtime"]
            permutation = array(
                "perm", "size_t", [str(v) for v in runtime["permutation"]]
            )
            exported = helicity_export(data, runtime)
            to_runtime = [-1] * len(data["helicities"])
            for runtime_index, export_index in enumerate(exported):
                to_runtime[export_index] = runtime_index
            helicity_runtime = array(
                "hrt", "int", [str(v) for v in to_runtime or [-1]]
            )
            export_array = array(
                "hex", "size_t", [str(v) for v in exported or [0]]
            )
            signs = runtime.get("momentum_signs")
            momentum_signs = None
            if signs is not None:
                momentum_signs = array(
                    "sgn", "signed char", [str(int(v)) for v in signs]
                )
            fields = [
                str(runtime["process_index"]),
                _string(runtime["color_id"]),
                str(runtime["color_index"]),
                permutation,
                _number(runtime["factor"]),
                helicity_runtime,
                export_array,
            ]
            if "assignment" in runtime:
                # Flavour-union assignment; omitted fields stay zero (NULL key).
                fields += [
                    _string(runtime["assignment_key"]),
                    str(int(runtime["assignment"])),
                    str(int(runtime["color_axis"])),
                ]
                if momentum_signs is not None:
                    fields.append(momentum_signs)
            elif signs is not None:
                raise ValueError("a crossed UMAMI contribution must be an assignment")
            contributions.append("    {" + ", ".join(fields) + "},")
        channel_rows.append((first, len(contributions) - first))
    # one contribution per flavour, all in one array
    lines.extend([
        "static const UmamiContribution umami_contributions[] = {",
        *(contributions or ["    {0},"]),
        "};",
    ])
    for channel_index, (first, count) in enumerate(channel_rows):
        lines.extend(
            [
                f"static const UmamiFlavour umami_c{channel_index}_flavours[] = {{",
                *(
                    f"    {{1, &umami_contributions[{first + k}]}},"
                    for k in range(count)
                ),
                *([] if count else ["    {0},"]),
                "};",
            ]
        )
    channel_rows = [
        f"    {{{count}, umami_c{channel_index}_flavours}},"
        for channel_index, (_, count) in enumerate(channel_rows)
    ]
    lines.extend(
        ["static const UmamiChannel umami_channels[] = {", *channel_rows, "};"]
    )
    lines.extend(
        [
            "static const UmamiProvider umami_provider = {",
            "    "
            + ", ".join(
                (
                    _string(provider["id"]),
                    str(provider["particle_count"]),
                    str(provider["incoming_count"]),
                    str(provider["helicity_count"]),
                    str(provider["color_count"]),
                    str(len(data["helicities"])),
                )
            )
            + ",",
            f"    umami_masses, {len(process_rows)}, umami_processes,",
            f"    {len(channel_rows)}, umami_channels, "
            + _boolean(provider["supports_alpha_s"]),
            "};",
            "",
        ]
    )
    return "\n".join(lines)
