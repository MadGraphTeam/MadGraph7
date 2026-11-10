# SPDX-License-Identifier: 0BSD
"""UMAMI providers of colour-accuracy ratios between two artifacts.

The ratio is the target matrix element (e.g. full colour) over the leading-colour
reference summed over all colour flows, both at the helicity of the event. It
reweights events of a leading-colour UMAMI provider generated from the same
processes, with the channel, flavour and helicity indices of that provider.

With ``switch_lines``, events generated only for the flavours with fewer quark
lines can be switched to those with at least ``switch_lines`` lines: the
provider sums the leading-colour matrix elements of all flavours at the event,
with the parton densities passed in (parameter ``flavor_switch``).
"""

from __future__ import annotations

import importlib.resources
import itertools
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .umami_c import _boolean, _power, _string

_TEMPLATES = {
    "umami_reweight.c": "umami_reweight.c",
    "reweight_provider.h": "reweight_provider.h",
    "umami.h": "umami.h",
    "umami.exports": "umami.exports",
    "umami.exports.macos": "umami.exports.macos",
    "UMAMI_LICENSE": "UMAMI_LICENSE",
    "Makefile.reweight": "Makefile",
}


_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")


def _identifier(value: object) -> str:
    # Provider ids end up in file names and in the makefile.
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid UMAMI provider id {value!r}")
    return value


def _integer(value: object) -> int:
    # Metadata values are written into generated C sources.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer in the UMAMI metadata, got {value!r}")
    return value


def load_umami_metadata(artifact: Path) -> list[dict[str, Any]]:
    """The metadata documents of all UMAMI providers of an artifact."""
    root = (Path(artifact) / "API" / "umami").resolve()
    index = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if index.get("kind") != "pyamplicol-umami-provider-index":
        return [index]
    documents = []
    for entry in index["providers"]:
        path = (root / entry["metadata"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"UMAMI metadata path {entry['metadata']!r} leaves {root}")
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    return documents


def _process_row(process: Mapping[str, Any], prefix: str, lines: list[str]) -> str:
    lines.append(
        f"static const int {prefix}_pdgs[] = {{"
        + ", ".join(str(_integer(v)) for v in process["pdgs"])
        + "};"
    )
    return (
        "    {"
        + ", ".join(
            (
                _string(process["id"]),
                f"{prefix}_pdgs",
                _string(process["alpha_s_parameter"]),
                _power(process),
                str(len(process["helicity_ids"])),
            )
        )
        + "},"
    )


def _int_array(name: str, ctype: str, values: Sequence[int]) -> str:
    return (
        f"static const {ctype} {name}[] = {{"
        + ", ".join(str(_integer(v)) for v in values or [-1])
        + "};"
    )


def quark_lines(pdgs: Sequence[int]) -> int:
    """Number of quark lines of a flavour: its quarks and antiquarks over two."""
    return sum(1 for pdg in pdgs if 0 < abs(pdg) <= 6) // 2


def _first_unused(source: Sequence[int], destination: Sequence[int]) -> list[int]:
    # destination leg k takes the first unused source leg with its pdg
    used: set[int] = set()
    result = []
    for pdg in destination:
        j = next(j for j, s in enumerate(source) if s == pdg and j not in used)
        used.add(j)
        result.append(j)
    return result


def _canonical_legs(pdgs: Sequence[int], legs: Sequence[int], incoming: int) -> tuple:
    # the matrix element is symmetric in identical outgoing legs
    legs = list(legs)
    for pdg in set(pdgs[incoming:]):
        slots = [k for k in range(incoming, len(pdgs)) if pdgs[k] == pdg]
        for k, value in zip(slots, sorted(legs[k] for k in slots)):
            legs[k] = value
    return tuple(legs)


def _is_parton(pdg: int) -> bool:
    return pdg == 21 or 0 < abs(pdg) <= 6


def _pdf_column(pdg: int) -> int:
    if pdg == 21:
        return 6
    if not 0 < abs(pdg) <= 6:
        raise ValueError(f"the flavour switch needs partons in the initial state, got {pdg}")
    return pdg + 6


def switch_catalogue(
    reference: Mapping[str, Any],
    groups: Mapping[str, Any] | None,
    switch_lines: int,
    initial_legs: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """The flavours of a leading-colour provider in event slot order, both beam
    orientations and every distinct ordering of the outgoing partons over the
    parton slots, see ``write_reweight_provider``. The other outgoing legs keep
    the slots the events have them in."""
    incoming = _integer(reference["provider"]["incoming_count"])
    if incoming != 2:
        raise ValueError("the flavour switch needs two incoming partons")
    processes = reference["runtime_processes"]
    by_id = {process["id"]: index for index, process in enumerate(processes)}
    members = []
    if groups is None:
        members = [
            (index, list(process["pdgs"]), list(range(len(process["pdgs"]))), False, 1.0)
            for index, process in enumerate(processes)
        ]
    else:
        for group in groups["groups"]:
            rep = by_id.get(group["representative"]["process_id"])
            if rep is None:
                continue
            for member in group["members"]:
                factor = member["factor"]
                members.append((
                    rep,
                    list(member["pdgs"]),
                    list(member["permutation"]),
                    bool(member["conjugated"]),
                    factor["numerator"] / factor["denominator"],
                ))
    # LHE colour tags of every flow of a runtime process, in its legs
    flows: dict[int, dict[str, int]] = {index: {} for index in by_id.values()}
    for channel in reference["channels"]:
        for entry in channel["processes"]:
            for member in entry.get("members", []):
                index = by_id.get(member["process_id"])
                if index is not None and member.get("color_id") is not None:
                    flows[index][member["color_id"]] = member["color_flows"]
    flow_ids = {index: sorted(table) for index, table in flows.items()}
    color_flows = [list(map(list, flow)) for flow in reference["color_flows"]]
    color_index = {json.dumps(flow): i for i, flow in enumerate(color_flows)}
    evaluations: dict[tuple, int] = {}
    orderings, targets, target_pdgs, seen = [], [], [], set()
    # the slots of the outgoing non-partons, the same in every row
    fixed = {
        slot: pdg
        for slot, pdg in enumerate(reference["pdg_ids"][0])
        if slot >= incoming and not _is_parton(pdg)
    }
    if any(
        {s: p for s, p in enumerate(row) if s >= incoming and not _is_parton(p)} != fixed
        for row in reference["pdg_ids"]
    ):
        raise ValueError("the flavour switch needs the non-partons in fixed slots")
    parton_slots = [
        slot for slot in range(incoming, len(reference["pdg_ids"][0])) if slot not in fixed
    ]
    n_partons = math.factorial(len(parton_slots))
    for rep, pdgs, permutation, conjugated, factor in members:
        rep_pdgs = processes[rep]["pdgs"]
        a, b = pdgs[:incoming]
        for orientation in (0, 1):
            initial = (a, b) if orientation == 0 else (b, a)
            if orientation == 1 and a == b:
                continue
            if initial_legs is not None and not (
                initial[0] in initial_legs[0] and initial[1] in initial_legs[1]
            ):
                continue
            key = (initial, tuple(sorted(pdgs[incoming:])))
            if key in seen:
                continue
            seen.add(key)
            lines = quark_lines(pdgs)
            final = pdgs[incoming:]
            partons = [p for p in final if _is_parton(p)]
            if sorted(p for p in final if not _is_parton(p)) != sorted(fixed.values()):
                raise ValueError(f"flavour {pdgs} has other non-partons than the events")
            weight = factor * math.prod(
                math.factorial(partons.count(p)) for p in set(partons)
            ) / n_partons
            for parton_order in sorted(set(itertools.permutations(partons))):
                order_slots = dict(fixed)
                order_slots.update(zip(parton_slots, parton_order))
                order = tuple(order_slots[s] for s in range(incoming, len(pdgs)))
                slots = [0, 1] if orientation == 0 else [1, 0]
                slots += [incoming + j for j in _first_unused(order, final)]
                legs = _canonical_legs(
                    rep_pdgs, [slots[r] for r in permutation], incoming
                )
                evaluation = evaluations.setdefault((rep, legs), len(evaluations))
                target = -1
                if lines >= switch_lines:
                    target = len(targets)
                    colors = []
                    for flow_id in flow_ids[rep]:
                        tags = color_flows[flows[rep][flow_id]]
                        slot_tags = [[0, 0] for _ in legs]
                        for j, slot in enumerate(legs):
                            pair = list(tags[j])
                            slot_tags[slot] = pair[::-1] if conjugated else pair
                        text = json.dumps(slot_tags)
                        if text not in color_index:
                            color_index[text] = len(color_flows)
                            color_flows.append(slot_tags)
                        colors.append(color_index[text])
                    targets.append({"evaluation": evaluation, "colors": colors})
                    target_pdgs.append([*initial, *order])
                orderings.append({
                    "evaluation": evaluation,
                    "pdf": [_pdf_column(initial[0]), _pdf_column(initial[1])],
                    "weight": weight,
                    "target": target,
                })
    return {
        "evaluations": [
            {"process": rep, "legs": list(legs)} for (rep, legs) in evaluations
        ],
        "orderings": orderings,
        "targets": targets,
        "target_pdgs": target_pdgs,
        "flow_ids": flow_ids,
        "color_flows": color_flows[len(reference["color_flows"]):],
        "switch_lines": switch_lines,
    }


def _switch_tables(
    reference: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    target_index: Any,
    lines: list[str],
) -> str:
    """C tables of the flavour switch; target_index maps a reference process
    index to its target process index."""
    for i, evaluation in enumerate(catalogue["evaluations"]):
        lines.append(_int_array(f"reweight_switch_e{i}", "size_t", evaluation["legs"]))
    lines.append("static const ReweightSwitchEvaluation reweight_switch_evaluations[] = {")
    lines.extend(
        f"    {{{_integer(e['process'])}, reweight_switch_e{i}}},"
        for i, e in enumerate(catalogue["evaluations"])
    )
    lines.append("};")
    lines.append("static const ReweightSwitchOrdering reweight_switch_orderings[] = {")
    lines.extend(
        f"    {{{o['evaluation']}, {{{o['pdf'][0]}, {o['pdf'][1]}}}, "
        f"{float(o['weight'])!r}, {o['target']}}},"
        for o in catalogue["orderings"]
    )
    lines.append("};")
    for i, target in enumerate(catalogue["targets"]):
        lines.append(_int_array(f"reweight_switch_t{i}", "int", target["colors"]))
    lines.append("static const ReweightSwitchTarget reweight_switch_targets[] = {")
    for i, target in enumerate(catalogue["targets"]):
        process = catalogue["evaluations"][target["evaluation"]]["process"]
        lines.append(
            f"    {{{target['evaluation']}, {target_index(process)}, reweight_switch_t{i}}},"
        )
    lines.append("};")
    flow_rows = []
    for index in range(len(reference["runtime_processes"])):
        ids = catalogue["flow_ids"].get(index, [])
        lines.append(
            f"static const char *const reweight_switch_f{index}[] = {{"
            + ", ".join(_string(i) for i in ids or [None])
            + "};"
        )
        flow_rows.append(f"    {{{len(ids)}, reweight_switch_f{index}}},")
    lines.append("static const ReweightSwitchFlows reweight_switch_flows[] = {")
    lines.extend(flow_rows)
    lines.append("};")
    helicities = [v for vector in reference["helicities"] for v in vector]
    lines.append(_int_array("reweight_switch_helicities", "int", helicities))
    lines.extend([
        "static const ReweightSwitch reweight_switch = {",
        f"    {len(catalogue['evaluations'])}, reweight_switch_evaluations,",
        f"    {len(catalogue['orderings'])}, reweight_switch_orderings,",
        f"    {len(catalogue['targets'])}, reweight_switch_targets,",
        "    reweight_switch_flows, reweight_switch_helicities,",
        "};",
    ])
    return "&reweight_switch"


def reweight_header(
    reference: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    reference_path: str,
    target_path: str,
    catalogue: Mapping[str, Any] | None = None,
) -> str:
    """Render the tables of one reweight provider, see ``reweight_provider.h``;
    with a ``switch_catalogue``, also those of the flavour switch."""
    target_processes: list[Mapping[str, Any]] = []
    target_ids: list[str | None] = []
    target_lookup = {}
    for document in targets:
        for process in document["runtime_processes"]:
            target_lookup.setdefault(
                tuple(process["pdgs"]), (process, document["provider"]["artifact_id"])
            )
    reference_processes = reference["runtime_processes"]
    lines = [
        "/* SPDX-License-Identifier: 0BSD */",
        "/* Generated from the UMAMI metadata of the two artifacts. */",
        '#include "reweight_provider.h"',
    ]
    rows = [
        _process_row(process, f"reweight_r{i}", lines)
        for i, process in enumerate(reference_processes)
    ]
    lines.extend(["static const ReweightProcess reweight_references[] = {", *rows, "};"])

    exported_count = len(reference["helicities"])
    channel_rows = []
    for c, channel in enumerate(reference["channels"]):
        flavour_rows = []
        for f, entry in enumerate(channel["processes"]):
            runtime = entry["runtime"]
            if "helicity_export" not in runtime:
                raise ValueError("the reference UMAMI metadata has no helicity export")
            process_index = _integer(runtime["process_index"])
            process = reference_processes[process_index]
            match = target_lookup.get(tuple(process["pdgs"]))
            if match is None:
                raise ValueError(
                    f"no process of the target artifact matches {process['id']!r}"
                )
            target, target_id = match
            if target not in target_processes:
                target_processes.append(target)
                target_ids.append(target_id)
            target_index = target_processes.index(target)
            reference_helicity = [-1] * exported_count
            target_helicity = [-1] * exported_count
            for runtime_index, exported in enumerate(runtime["helicity_export"]):
                reference_helicity[exported] = runtime_index
                identifier = process["helicity_ids"][runtime_index]
                if identifier in target["helicity_ids"]:
                    target_helicity[exported] = target["helicity_ids"].index(identifier)
            prefix = f"reweight_c{c}_f{f}"
            lines.append(
                _int_array(f"{prefix}_permutation", "size_t", runtime["permutation"])
            )
            lines.append(_int_array(f"{prefix}_reference", "int", reference_helicity))
            lines.append(_int_array(f"{prefix}_target", "int", target_helicity))
            flavour_rows.append(
                f"    {{{process_index}, {target_index}, "
                f"{prefix}_permutation, {prefix}_reference, {prefix}_target}},"
            )
        lines.extend(
            [
                f"static const ReweightFlavour reweight_c{c}_flavours[] = {{",
                *flavour_rows,
                "};",
            ]
        )
        channel_rows.append(f"    {{{len(flavour_rows)}, reweight_c{c}_flavours}},")
    def target_index(process_index: int) -> int:
        process = reference_processes[process_index]
        match = target_lookup.get(tuple(process["pdgs"]))
        if match is None:
            raise ValueError(f"no process of the target artifact matches {process['id']!r}")
        if match[0] not in target_processes:
            target_processes.append(match[0])
            target_ids.append(match[1])
        return target_processes.index(match[0])

    switch = (
        "NULL"
        if catalogue is None
        else _switch_tables(reference, catalogue, target_index, lines)
    )
    if len(set(target_ids)) > 1:
        raise ValueError("the target processes come from different artifacts")
    rows = [
        _process_row(process, f"reweight_t{i}", lines)
        for i, process in enumerate(target_processes)
    ]
    lines.extend(["static const ReweightProcess reweight_targets[] = {", *rows, "};"])
    lines.extend(
        ["static const ReweightChannel reweight_channels[] = {", *channel_rows, "};"]
    )
    supports_alpha_s = all(
        p["alpha_s_parameter"] is not None
        for p in [*reference_processes, *target_processes]
    )
    lines.extend(
        [
            f"#define UMAMI_HAS_ALPHA_S {int(supports_alpha_s)}",
            "static const ReweightProvider reweight_provider = {",
            f"    {_string(reference['provider']['id'])}, "
            f"{_integer(reference['provider']['particle_count'])}, {exported_count},",
            f"    {_string(reference_path)}, {_string(target_path)},",
            f"    {_string(reference['provider'].get('artifact_id'))}, "
            f"{_string(target_ids[0] if target_ids else None)},",
            f"    {len(reference_processes)}, reweight_references,",
            f"    {len(target_processes)}, reweight_targets,",
            f"    {len(channel_rows)}, reweight_channels, {_boolean(supports_alpha_s)},",
            f"    {switch}",
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def write_reweight_provider(
    reference_artifact: str | os.PathLike[str],
    target_artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
    switch_lines: int | None = None,
    initial_legs: Sequence[Sequence[int]] | None = None,
) -> list[str]:
    """Write the sources of the reweight providers to ``output``.

    One provider is written per UMAMI provider of the leading-colour reference
    artifact; ``make -C output`` builds ``build/libreweight_<id>.so``. The
    libraries are initialized with the absolute path of ``output``, the artifacts
    are found relative to it. Returns the provider ids.

    With ``switch_lines``, the providers can also switch events to the flavours
    with at least that many quark lines (parameter ``flavor_switch``), and
    ``flavor_switch_<id>.json`` lists the switch targets (ordered pdg ids, the
    library's flavour index) and the colour flows they add after the
    metadata's. ``initial_legs`` are the pdg ids allowed for each incoming leg.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = load_umami_metadata(Path(reference_artifact))
    targets = load_umami_metadata(Path(target_artifact))
    if any(d["provider"]["color_accuracy"] != "lc" for d in references):
        raise ValueError("the reference artifact must have leading-colour accuracy")
    reference_path = os.path.relpath(Path(reference_artifact).resolve(), output.resolve())
    target_path = os.path.relpath(Path(target_artifact).resolve(), output.resolve())
    templates = importlib.resources.files("pyamplicol.assets.api_templates") / "umami"
    for source, destination in _TEMPLATES.items():
        (output / destination).write_bytes((templates / source).read_bytes())
    groups_path = Path(reference_artifact) / "process-groups.json"
    groups = (
        json.loads(groups_path.read_text(encoding="utf-8"))
        if switch_lines is not None and groups_path.is_file()
        else None
    )
    ids = []
    for reference in references:
        identifier = _identifier(reference["provider"]["id"])
        catalogue = None
        if switch_lines is not None:
            catalogue = switch_catalogue(reference, groups, switch_lines, initial_legs)
            (output / f"flavor_switch_{identifier}.json").write_text(
                json.dumps({
                    "switch_lines": switch_lines,
                    "targets": catalogue["target_pdgs"],
                    "color_flow_offset": len(reference["color_flows"]),
                    "color_flows": catalogue["color_flows"],
                }),
                encoding="utf-8",
            )
        (output / f"reweight_{identifier}.h").write_text(
            reweight_header(reference, targets, reference_path, target_path, catalogue),
            encoding="utf-8",
        )
        ids.append(identifier)
    (output / "providers.mk").write_text(
        "UMAMI_PROVIDERS := " + " ".join(ids) + "\n", encoding="utf-8"
    )
    return ids


__all__ = [
    "load_umami_metadata",
    "quark_lines",
    "reweight_header",
    "switch_catalogue",
    "write_reweight_provider",
]
