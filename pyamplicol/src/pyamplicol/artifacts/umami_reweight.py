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

A flavour-union reference evaluates its members as assignments of their
template's runtime (one native call per template and slot permutation in the
switch). Full-colour targets are the members' own processes of the target
artifact, or assignments of its templates when it has flavour unions too.
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

from ..generation.process_grouping import ProcessGroupMap
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
    values = values or [-1]
    # plain ints need no per-value check (the hot path of large providers)
    if all(type(v) is int for v in values):
        text = ", ".join(map(str, values))
    else:
        text = ", ".join(str(_integer(v)) for v in values)
    return f"static const {ctype} {name}[] = {{" + text + "};"


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


def _canonical_tags(flow: Sequence[Sequence[int]]) -> tuple:
    # colour tags renumbered by first appearance: equal for the same physical flow
    names: dict[int, int] = {}
    return tuple(
        tuple(names.setdefault(tag, len(names) + 1) if tag else 0 for tag in pair)
        for pair in flow
    )


def _is_qcd_parton(pdg: int) -> bool:
    return pdg == 21 or 0 < abs(pdg) <= 6


def _jet_partons(reference: Mapping[str, Any]) -> frozenset[int]:
    """The gluons and quarks of a provider's rows in massless slots: the partons
    the flavour switch may exchange (never a top, a b only when massless)."""
    masses = reference["provider"].get("masses")
    if masses is None:
        raise ValueError("the flavour switch needs the leg masses of the provider")
    massless, massive = set(), set()
    for row in reference["pdg_ids"]:
        for slot, pdg in enumerate(row):
            if _is_qcd_parton(pdg):
                (massless if float(masses[slot]) == 0.0 else massive).add(pdg)
    if massless & massive:
        raise ValueError(
            f"the partons {sorted(massless & massive)} have massless and massive slots"
        )
    return frozenset(massless)


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
    orientations and every distinct ordering of the outgoing massless gluons and
    quarks over their slots, see ``write_reweight_provider``. The other outgoing
    legs (massive quarks too) keep the slots the event's row has them in: one
    set of orderings per arrangement of them in the rows."""
    incoming = _integer(reference["provider"]["incoming_count"])
    if incoming != 2:
        raise ValueError("the flavour switch needs two incoming partons")
    processes = reference["runtime_processes"]
    by_id = {process["id"]: index for index, process in enumerate(processes)}
    # (runtime process, pdgs, permutation, conjugated, factor, assignment); a
    # flavour-union assignment (>= 0) includes its factor in the runtime value
    members = []
    # per (runtime, assignment): its process id and its value class (equal
    # patterns, signs and factor give equal values, so they share evaluations)
    member_ids: dict[tuple[int, int], tuple[str, tuple]] = {}
    # per template runtime: its pattern table (endpoint pairs per pattern) and lookup
    pattern_tables: dict[int, tuple[list[list[tuple[int, int]]], dict[tuple, int]]] = {}
    if groups is None:
        members = [
            (index, list(process["pdgs"]), list(range(len(process["pdgs"]))), False,
             1.0, -1)
            for index, process in enumerate(processes)
        ]
    else:
        for group in groups["groups"]:
            template = group["representative"]["process_id"]
            rep = by_id.get(template)
            if rep is None:
                continue
            union = any("flavour_union_patterns" in m for m in group["members"])
            pairs = group.get("flavour_union_pattern_pairs")
            if union and pairs:
                table = [[(int(f), int(a)) for f, a in pattern] for pattern in pairs]
                pattern_tables[rep] = (
                    table,
                    {tuple(sorted(pattern)): p for p, pattern in enumerate(table)},
                )
            assignment = 0
            for member in group["members"]:
                if "crossing_momentum_signs" in member:
                    raise ValueError(
                        f"the flavour switch cannot use the crossed member "
                        f"{member['process_id']!r} yet"
                    )
                if union and member["process_id"] != template:
                    if "flavour_union_patterns" not in member:
                        raise ValueError(
                            f"flavour-union template {template!r} has the non-union "
                            f"member {member['process_id']!r}"
                        )
                    assignment += 1
                factor = member["factor"]
                members.append((
                    rep,
                    list(member["pdgs"]),
                    list(member["permutation"]),
                    bool(member["conjugated"]),
                    factor["numerator"] / factor["denominator"],
                    assignment if union else -1,
                ))
                if union:
                    value_class = (
                        tuple(member.get("flavour_union_patterns", ("template",))),
                        tuple(member.get("flavour_union_pattern_signs", ())),
                        factor["numerator"],
                        factor["denominator"],
                    )
                    member_ids[(rep, assignment)] = (member["process_id"], value_class)
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
    color_index = {tuple(map(tuple, flow)): i for i, flow in enumerate(color_flows)}
    # the colour flows of a target per (runtime, legs, conjugated)
    target_colors: dict[tuple, list[int]] = {}
    evaluations: dict[tuple, int] = {}
    described: list[tuple] = []
    orderings, targets, target_pdgs, seen = [], [], [], set()
    # the outgoing non-partons (massive quarks too) of every row, in the same slots
    # in every row; their flavours may differ between rows (e+ ve and mu+ vm of l+ vl)
    jet = _jet_partons(reference)
    canonical: dict[tuple, dict[int, int]] = {}
    # a row's arrangement: where its non-partons sit relative to the first row with
    # them (t~ t of conjugated flavours); its events keep it in every flavour summed
    arrangements: list[tuple] = []
    rows = reference["pdg_ids"]
    row_arrangements = [0] * len(rows)
    # rows in physical order: the numbering does not depend on the table order
    for index in sorted(range(len(rows)), key=lambda i: tuple(rows[i])):
        row = rows[index]
        fixed = {s: p for s, p in enumerate(row) if s >= incoming and p not in jet}
        first = canonical.setdefault(tuple(sorted(fixed.values())), fixed)
        if list(fixed) != list(next(iter(canonical.values()))):
            raise ValueError("the flavour switch needs the non-partons in fixed slots")
        slots = list(fixed)
        moved = _first_unused(list(fixed.values()), list(first.values()))
        # sigma: slot of a non-parton in the first row -> its slot in this row
        sigma = tuple(
            (s, slots[k]) for s, k in zip(first, moved, strict=True) if s != slots[k]
        )
        if sigma not in arrangements:
            arrangements.append(sigma)
        row_arrangements[index] = arrangements.index(sigma)
    # physical order: orderings and targets do not depend on the grouping
    members.sort(key=lambda m: tuple(m[1]))
    fixed_slots = next(iter(canonical.values()))
    parton_slots = [
        slot for slot in range(incoming, len(reference["pdg_ids"][0]))
        if slot not in fixed_slots
    ]
    n_partons = math.factorial(len(parton_slots))
    for (arrangement, sigma), (
        rep, pdgs, permutation, conjugated, factor, assignment
    ) in itertools.product(enumerate(arrangements), members):
        rep_pdgs = processes[rep]["pdgs"]
        # the member process id names a flavour-union assignment (checked at load)
        assignment_key, value_class = member_ids.get((rep, assignment), (None, None))
        if assignment >= 0:
            # the member's identical legs in template slots; the runtime has its factor
            rep_pdgs = [pdgs[leg] for leg in permutation]
            factor = 1.0
        a, b = pdgs[:incoming]
        for orientation in (0, 1):
            initial = (a, b) if orientation == 0 else (b, a)
            if orientation == 1 and a == b:
                continue
            if initial_legs is not None and not (
                initial[0] in initial_legs[0] and initial[1] in initial_legs[1]
            ):
                continue
            key = (arrangement, initial, tuple(sorted(pdgs[incoming:])))
            if key in seen:
                continue
            seen.add(key)
            lines = quark_lines(pdgs)
            final = pdgs[incoming:]
            partons = [p for p in final if p in jet]
            first = canonical.get(tuple(sorted(p for p in final if p not in jet)))
            if first is None:
                raise ValueError(f"flavour {pdgs} has other non-partons than the events")
            moves = dict(sigma)
            fixed = {moves.get(slot, slot): pdg for slot, pdg in first.items()}
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
                evaluation = evaluations.setdefault(
                    (rep, value_class, legs), len(evaluations)
                )
                if evaluation == len(described):
                    described.append((rep, assignment, legs, assignment_key))
                target = -1
                if lines >= switch_lines:
                    target = len(targets)
                    colors = target_colors.get((rep, legs, conjugated))
                    if colors is None:
                        found = []
                        for slot_index, flow_id in enumerate(flow_ids[rep]):
                            tags = color_flows[flows[rep][flow_id]]
                            slot_tags = [[0, 0] for _ in legs]
                            for j, slot in enumerate(legs):
                                pair = list(tags[j])
                                slot_tags[slot] = pair[::-1] if conjugated else pair
                            found.append((_canonical_tags(slot_tags), slot_index, slot_tags))
                        # the flows in physical order, the order of the colour draw
                        colors = target_colors[rep, legs, conjugated] = ([], [])
                        for _, slot_index, slot_tags in sorted(found):
                            key = tuple(map(tuple, slot_tags))
                            if key not in color_index:
                                color_index[key] = len(color_flows)
                                color_flows.append(slot_tags)
                            colors[0].append(color_index[key])
                            colors[1].append(slot_index)
                    targets.append({
                        "evaluation": evaluation,
                        "colors": list(colors[0]),
                        "flows": list(colors[1]),
                    })
                    target_pdgs.append([*initial, *order])
                orderings.append({
                    "evaluation": evaluation,
                    "pdf": [_pdf_column(initial[0]), _pdf_column(initial[1])],
                    "weight": weight,
                    "target": target,
                    "arrangement": arrangement,
                })
    # One native call per (runtime, legs): the assignments of a template share it.
    # With a pattern table, one per (template, leg types): an evaluation at legs l
    # is its assignment relabelled by pi[r] = c^-1[l[r]] at the call's legs c.
    calls: dict[tuple, dict[str, Any]] = {}
    entry_of: dict[tuple, int] = {}
    for index, (rep, assignment, legs, _) in enumerate(described):
        table = pattern_tables.get(rep) if assignment >= 0 else None
        if assignment < 0:
            key: tuple = (rep, legs, index)
        elif table is None:
            key = (rep, legs)
        else:
            types = [0] * len(legs)
            for r, slot in enumerate(legs):
                types[slot] = processes[rep]["pdgs"][r]
            key = (rep, tuple(types))
        call = calls.setdefault(
            key,
            {
                "process": rep,
                "legs": list(legs),
                "evaluations": [],
                "assignments": [] if assignment >= 0 else None,
                "permutations": [] if table is not None else None,
                "entries": [],
            },
        )
        call["evaluations"].append(index)
        if assignment < 0:
            call["entries"].append(0)
            continue
        if table is None:
            call["entries"].append(len(call["assignments"]))
            call["assignments"].append(assignment)
            continue
        inverse = {slot: r for r, slot in enumerate(call["legs"])}
        permutation = tuple(inverse[slot] for slot in legs)
        patterns, signs, numerator, denominator = member_ids[(rep, assignment)][1]
        if patterns == ("template",):
            derived: tuple = patterns
        else:
            pairs, lookup = table
            moved = [
                tuple(sorted((permutation[f], permutation[a]) for f, a in pairs[p]))
                for p in patterns
            ]
            derived = tuple(sorted(
                (lookup[key], sign) for key, sign in zip(moved, signs, strict=True)
            ))
        entry = entry_of.setdefault(
            (key, derived, numerator, denominator), len(call["assignments"])
        )
        if entry == len(call["assignments"]):
            call["assignments"].append(assignment)
            call["permutations"].append(list(permutation))
        call["entries"].append(entry)
    for call in calls.values():
        # identity relabellings need no runtime call
        permutations = call["permutations"]
        if permutations is not None and all(p == sorted(p) for p in permutations):
            call["permutations"] = None
    return {
        "evaluations": [
            {
                "process": rep,
                "legs": list(legs),
                "assignment": assignment,
                "assignment_key": key,
            }
            for (rep, assignment, legs, key) in described
        ],
        "calls": list(calls.values()),
        "orderings": orderings,
        "targets": targets,
        "target_pdgs": target_pdgs,
        "flow_ids": flow_ids,
        "color_flows": color_flows[len(reference["color_flows"]):],
        "switch_lines": switch_lines,
        "arrangement_count": len(arrangements),
        "row_arrangements": row_arrangements,
    }


def _pad_empty(rows: Sequence[Any]) -> list[str]:
    """A C array cannot be empty: one zero entry, the stored count stays 0."""
    return [] if rows else ["    {0},"]


def _switch_tables(
    reference: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    target_route: Any,
    lines: list[str],
) -> str:
    """C tables of the flavour switch; target_route maps a target to its target
    process index, its legs in event slots (None: the evaluation's), factor,
    flavour-union assignment (-1: none) and that assignment's process id."""
    for i, evaluation in enumerate(catalogue["evaluations"]):
        lines.append(_int_array(f"reweight_switch_e{i}", "size_t", evaluation["legs"]))
    lines.append("static const ReweightSwitchEvaluation reweight_switch_evaluations[] = {")
    lines.extend(
        f"    {{{_integer(e['process'])}, reweight_switch_e{i}, "
        f"{_integer(e.get('assignment', -1))}, {_string(e.get('assignment_key'))}}},"
        for i, e in enumerate(catalogue["evaluations"])
    )
    lines.extend(_pad_empty(catalogue["evaluations"]))
    lines.append("};")
    calls = catalogue.get("calls") or [
        {"process": e["process"], "evaluations": [i], "assignments": None}
        for i, e in enumerate(catalogue["evaluations"])
    ]
    rows = []
    for i, call in enumerate(calls):
        lines.append(_int_array(f"reweight_switch_c{i}", "size_t", call["evaluations"]))
        assignments = call["assignments"]
        if assignments is None:
            rows.append(
                f"    {{{len(call['evaluations'])}, reweight_switch_c{i}, NULL, 1, "
                "NULL, NULL},"
            )
            continue
        lines.append(_int_array(f"reweight_switch_a{i}", "uint32_t", assignments))
        entries = call.get("entries") or list(range(len(call["evaluations"])))
        lines.append(_int_array(f"reweight_switch_n{i}", "size_t", entries))
        permutations = call.get("permutations")
        if permutations is not None:
            flat = [slot for permutation in permutations for slot in permutation]
            lines.append(_int_array(f"reweight_switch_p{i}", "size_t", flat))
        rows.append(
            f"    {{{len(call['evaluations'])}, reweight_switch_c{i}, "
            f"reweight_switch_a{i}, {len(assignments)}, "
            + (f"reweight_switch_p{i}" if permutations is not None else "NULL")
            + f", reweight_switch_n{i}}},"
        )
    lines.append("static const ReweightSwitchCall reweight_switch_calls[] = {")
    lines.extend(rows)
    lines.extend(_pad_empty(calls))
    lines.append("};")
    lines.append("static const ReweightSwitchOrdering reweight_switch_orderings[] = {")
    arranged = catalogue.get("arrangement_count", 1) > 1
    lines.extend(
        f"    {{{o['evaluation']}, {{{o['pdf'][0]}, {o['pdf'][1]}}}, "
        f"{float(o['weight'])!r}, {o['target']}"
        + (f", {o['arrangement']}}}," if arranged else "},")
        for o in catalogue["orderings"]
    )
    lines.extend(_pad_empty(catalogue["orderings"]))
    lines.append("};")
    rows = []
    for i, target in enumerate(catalogue["targets"]):
        lines.append(_int_array(f"reweight_switch_t{i}", "int", target["colors"]))
        order = target.get("flows", range(len(target["colors"])))
        lines.append(_int_array(f"reweight_switch_o{i}", "int", list(order)))
        index, legs, factor, assignment, key = target_route(
            target, catalogue["target_pdgs"][i]
        )
        name = f"reweight_switch_e{target['evaluation']}"
        if legs is not None:
            name = f"reweight_switch_l{i}"
            lines.append(_int_array(name, "size_t", legs))
        rows.append(
            f"    {{{target['evaluation']}, {index}, reweight_switch_t{i}, "
            f"reweight_switch_o{i}, {name}, "
            f"{float(factor)!r}, {assignment}, {_string(key)}}},"
        )
    lines.append("static const ReweightSwitchTarget reweight_switch_targets[] = {")
    lines.extend(rows)
    lines.extend(_pad_empty(catalogue["targets"]))
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
        f"    {len(calls)}, reweight_switch_calls,",
        f"    {len(catalogue['orderings'])}, reweight_switch_orderings,",
        f"    {len(catalogue['targets'])}, reweight_switch_targets,",
        "    reweight_switch_flows, reweight_switch_helicities,",
        "};",
    ])
    return "&reweight_switch"


def _target_routes(
    targets: Sequence[Mapping[str, Any]],
) -> list[
    tuple[tuple[int, ...], Mapping[str, Any], Any, list[int], float, int, str | None]
]:
    """Every physical flavour of the target artifact: (its pdgs, its runtime
    process, artifact id, runtime slot -> own leg, factor, flavour-union
    assignment or -1, its process id), from the members of its UMAMI entries."""
    routes, seen = [], set()
    for document in targets:
        processes = document["runtime_processes"]
        for channel in document["channels"]:
            for entry in channel["processes"]:
                for member in entry.get("members", ()):
                    pdgs = tuple(int(p) for p in member["pdgs"])
                    if pdgs in seen:
                        continue
                    seen.add(pdgs)
                    runtime = member["runtime"]
                    assignment = _integer(runtime.get("assignment", -1))
                    key = runtime.get("assignment_key")
                    if assignment >= 0 and not isinstance(key, str):
                        raise ValueError(
                            f"target member {member.get('process_id')!r} has a "
                            "flavour-union assignment without its process id"
                        )
                    routes.append((
                        pdgs,
                        processes[_integer(runtime["process_index"])],
                        document["provider"]["artifact_id"],
                        [int(v) for v in runtime["permutation"]],
                        float(runtime["factor"]),
                        assignment,
                        key,
                    ))
    return routes


def _has_union_targets(targets: Sequence[Mapping[str, Any]]) -> bool:
    """Whether target flavours are assignments of flavour-union templates."""
    return any(
        "assignment" in member["runtime"]
        for document in targets
        for channel in document["channels"]
        for entry in channel["processes"]
        for member in entry.get("members", ())
    )


def _route_index(routes, incoming: int) -> dict:
    """Routes keyed by (initial pdgs, sorted final pdgs); the first route of a
    key wins, as in a scan of the route list."""
    incoming = _integer(incoming)
    index: dict = {}
    for route in routes:
        pdgs = route[0]
        index.setdefault((pdgs[:incoming], tuple(sorted(pdgs[incoming:]))), route)
    return index


def _route(index, event_pdgs: Sequence[int], incoming: int):
    """The target route of a flavour given in event slots: (process, artifact
    id, runtime slot -> event slot, factor, assignment, assignment key). Identical
    outgoing particles and, for two beams, the beam order are interchangeable."""
    incoming = _integer(incoming)
    event_pdgs = [int(p) for p in event_pdgs]
    final = tuple(sorted(event_pdgs[incoming:]))
    initial = tuple(event_pdgs[:incoming])
    for swapped in (False, True) if incoming == 2 else (False,):
        route = index.get((initial[::-1] if swapped else initial, final))
        if route is None:
            continue
        pdgs, process, artifact_id, permutation, factor, assignment, key = route
        to_event = [1 - leg if swapped else leg for leg in range(incoming)]
        to_event += [
            incoming + j for j in _first_unused(event_pdgs[incoming:], pdgs[incoming:])
        ]
        legs = [to_event[leg] for leg in permutation]
        return process, artifact_id, legs, factor, assignment, key
    raise ValueError(f"no process of the target artifact has the flavour {event_pdgs}")


def _helicity_values(identifier: str) -> tuple[int, ...]:
    # helicity ids are "h:" and the signed values in runtime leg order
    return tuple(int(value) for value in identifier[2:].split(","))


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
    routes = None
    # needed by flavour-union routes only
    incoming = reference["provider"].get("incoming_count")
    # union targets: flavours without a target process of their own are assignments,
    # and a template's own flavour is its assignment 0 (the C side needs one kind)
    union_targets = _has_union_targets(targets)
    target_routes = _target_routes(targets) if union_targets else None
    templates = (
        {route[1]["id"] for route in target_routes if route[5] >= 0}
        if union_targets
        else set()
    )

    def route_index():
        nonlocal routes
        if routes is None:
            routes = _route_index(target_routes or _target_routes(targets), incoming)
        return routes
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

    known: dict[int, int] = {}

    def target_index(target, target_id) -> int:
        if id(target) in known:
            return known[id(target)]
        if target not in target_processes:
            target_processes.append(target)
            target_ids.append(target_id)
        known[id(target)] = target_processes.index(target)
        return known[id(target)]

    # the index of each helicity id of a target process (its first occurrence)
    positions: dict[int, dict[str, int]] = {}

    def helicity_positions(target) -> dict[str, int]:
        if id(target) not in positions:
            table: dict[str, int] = {}
            for index, identifier in enumerate(target["helicity_ids"]):
                table.setdefault(identifier, index)
            positions[id(target)] = table
        return positions[id(target)]

    exported_count = len(reference["helicities"])
    # several arrangements of the switch's non-partons: each entry names its own
    arranged = catalogue is not None and catalogue.get("arrangement_count", 1) > 1
    channel_rows = []
    for c, channel in enumerate(reference["channels"]):
        flavour_rows = []
        for f, entry in enumerate(channel["processes"]):
            runtime = entry["runtime"]
            if "helicity_export" not in runtime:
                raise ValueError("the reference UMAMI metadata has no helicity export")
            process_index = _integer(runtime["process_index"])
            process = reference_processes[process_index]
            prefix = f"reweight_c{c}_f{f}"
            assignment = runtime.get("assignment")
            reference_helicity = [-1] * exported_count
            target_helicity = [-1] * exported_count
            for runtime_index, exported in enumerate(runtime["helicity_export"]):
                reference_helicity[exported] = runtime_index
            lines.append(
                _int_array(f"{prefix}_permutation", "size_t", runtime["permutation"])
            )
            target_permutation, target_factor = f"{prefix}_permutation", 1.0
            target_assignment, target_key = -1, None
            match = target_lookup.get(tuple(process["pdgs"]))
            if assignment is None and (match is not None or not union_targets):
                if match is None:
                    raise ValueError(
                        f"no process of the target artifact matches {process['id']!r}"
                    )
                target, target_id = match
                if target["id"] in templates:
                    target_assignment, target_key = 0, target["id"]
                target_helicities = helicity_positions(target)
                for runtime_index, exported in enumerate(runtime["helicity_export"]):
                    identifier = process["helicity_ids"][runtime_index]
                    if identifier in target_helicities:
                        target_helicity[exported] = target_helicities[identifier]
            else:
                # A flavour-union member or target: its own route, legs and factor.
                pdgs = reference["pdg_ids"][entry["matrix_elements"][0]["pdg_ids"]]
                route = _route(route_index(), pdgs, incoming)
                target, target_id, legs, target_factor = route[:4]
                target_assignment, target_key = route[4:]
                target_permutation = f"{prefix}_target_permutation"
                lines.append(_int_array(target_permutation, "size_t", legs))
                values = {
                    _helicity_values(identifier): index
                    for index, identifier in enumerate(target["helicity_ids"])
                }
                for exported in runtime["helicity_export"]:
                    vector = reference["helicities"][exported]
                    target_helicity[exported] = values.get(
                        tuple(vector[slot] for slot in legs), -1
                    )
            index = target_index(target, target_id)
            lines.append(_int_array(f"{prefix}_reference", "int", reference_helicity))
            lines.append(_int_array(f"{prefix}_target", "int", target_helicity))
            switch_arrangement = ""
            if arranged:
                # the arrangement of the non-partons in the entry's rows
                found = {
                    catalogue["row_arrangements"][_integer(term["pdg_ids"])]
                    for term in entry["matrix_elements"]
                }
                if len(found) != 1:
                    raise ValueError(f"flavour {f} of channel {c} mixes arrangements")
                switch_arrangement = f", {found.pop()}"
            flavour_rows.append(
                f"    {{{process_index}, {index}, "
                f"{prefix}_permutation, {prefix}_reference, {prefix}_target, "
                f"{-1 if assignment is None else _integer(assignment)}, "
                f"{_string(runtime.get('assignment_key'))}, "
                f"{target_permutation}, {target_factor!r}, {target_assignment}, "
                f"{_string(target_key)}{switch_arrangement}}},"
            )
        lines.extend(
            [
                f"static const ReweightFlavour reweight_c{c}_flavours[] = {{",
                *flavour_rows,
                "};",
            ]
        )
        channel_rows.append(f"    {{{len(flavour_rows)}, reweight_c{c}_flavours}},")

    def target_route(target, pdgs):
        evaluation = catalogue["evaluations"][target["evaluation"]]
        process = reference_processes[evaluation["process"]]
        match = target_lookup.get(tuple(process["pdgs"]))
        if evaluation.get("assignment", -1) < 0 and (
            match is not None or not union_targets
        ):
            if match is None:
                raise ValueError(
                    f"no process of the target artifact matches {process['id']!r}"
                )
            if match[0]["id"] in templates:
                return target_index(*match), None, 1.0, 0, match[0]["id"]
            return target_index(*match), None, 1.0, -1, None
        process, target_id, legs, factor, assignment, key = _route(
            route_index(), pdgs, incoming
        )
        return target_index(process, target_id), legs, factor, assignment, key

    switch = (
        "NULL"
        if catalogue is None
        else _switch_tables(reference, catalogue, target_route, lines)
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
    if groups is not None:
        # kind and schema check; switch_catalogue reads the raw records
        ProcessGroupMap.from_document(groups)
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
