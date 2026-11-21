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
switch). A crossed member (process.flavor_union_crossing) reads sign-flipped
momenta with helicities flipped by rho; a call of the switch shares one crossing.
Full-colour targets are the members' own processes of the target artifact, or
assignments of its templates when it has flavour unions too (never crossed).
"""

from __future__ import annotations

import importlib.resources
import itertools
import json
import math
import os
import re
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..generation.process_grouping import ProcessGroupMap
from .umami_c import _power
from .umami_reweight_data import (
    FlavourRow,
    ProcessRow,
    ProviderScalars,
    ReweightTables,
    SwitchCall,
    SwitchEvaluation,
    SwitchOrderings,
    SwitchRows,
    SwitchTargets,
    decode_reweight_data,
    encode_reweight_data,
    write_reweight_data,
)

_TEMPLATES = {
    "umami_reweight.c": "umami_reweight.c",
    "reweight_provider.h": "reweight_provider.h",
    "reweight_data.h": "reweight_data.h",
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
    # Metadata values reach the provider data and generated C sources.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer in the UMAMI metadata, got {value!r}")
    return value


def _runtime_records(document: Mapping[str, Any]):
    for channel in document.get("channels", ()):
        for entry in channel["processes"]:
            yield entry["runtime"]
            for member in entry.get("members", ()):
                yield member["runtime"]


def _checked_metadata(document: dict[str, Any]) -> dict[str, Any]:
    # schema 5 adds momentum_signs (crossed flavour-union members) to schema 4
    version = document.get("schema_version")
    if version not in (4, 5):
        raise ValueError(f"unsupported UMAMI metadata schema version {version!r}")
    if version == 4 and any(
        "momentum_signs" in runtime for runtime in _runtime_records(document)
    ):
        raise ValueError("UMAMI metadata schema 4 cannot carry momentum signs")
    return document


def load_umami_metadata(artifact: Path) -> list[dict[str, Any]]:
    """The metadata documents of all UMAMI providers of an artifact."""
    root = (Path(artifact) / "API" / "umami").resolve()
    index = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if index.get("kind") != "pyamplicol-umami-provider-index":
        return [_checked_metadata(index)]
    documents = []
    for entry in index["providers"]:
        path = (root / entry["metadata"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"UMAMI metadata path {entry['metadata']!r} leaves {root}")
        documents.append(_checked_metadata(json.loads(path.read_text(encoding="utf-8"))))
    return documents


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
    # the matrix element is symmetric in identical outgoing legs: the slots reading
    # an outgoing event slot (template-initial ones too when crossed)
    legs = list(legs)
    for pdg in set(pdgs):
        slots = [k for k in range(len(pdgs)) if pdgs[k] == pdg and legs[k] >= incoming]
        for k, value in zip(slots, sorted(legs[k] for k in slots), strict=True):
            legs[k] = value
    return tuple(legs)


def _member_crossing(
    member: Mapping[str, Any], incoming: int
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """(sigma, rho) in template slot order of a crossed flavour-union member
    record, sigma checked against its legs; None when uncrossed."""
    signs = member.get("crossing_momentum_signs")
    if signs is None:
        return None
    identifier = member.get("process_id")
    if member["conjugated"] or "flavour_union_patterns" not in member:
        raise ValueError(
            f"crossed member {identifier!r} must be an unconjugated union member"
        )
    permutation = [_integer(v) for v in member["permutation"]]
    initial = _integer(member["initial_count"])
    expected = [-1 if (r < incoming) != (leg < initial) else 1
                for r, leg in enumerate(permutation)]
    helicity = member.get("crossing_helicity_signs")
    consistent = (
        [_integer(v) for v in signs] == expected
        and helicity is not None
        and len(helicity) == len(expected)
        and all(
            _integer(h) in (-1, 1) and (h == 1 or s == -1)
            for h, s in zip(helicity, expected, strict=True)
        )
    )
    if not consistent:
        raise ValueError(
            f"crossed member {identifier!r} has inconsistent crossing signs"
        )
    return tuple(expected), tuple(int(h) for h in helicity)


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
    # (runtime process, pdgs, permutation, conjugated, factor, assignment, rho); a
    # flavour-union assignment (>= 0) includes its factor in the runtime value, a
    # crossed one has its helicity signs rho (None: uncrossed)
    members = []
    # per (runtime, assignment): its process id and its value class (equal
    # patterns, signs and factor give equal values, so they share evaluations)
    member_ids: dict[tuple[int, int], tuple[str, tuple]] = {}
    # per template runtime: its pattern table (endpoint pairs per pattern) and lookup
    pattern_tables: dict[int, tuple[list[list[tuple[int, int]]], dict[tuple, int]]] = {}
    if groups is None:
        members = [
            (index, list(process["pdgs"]), list(range(len(process["pdgs"]))), False,
             1.0, -1, None)
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
                crossing = _member_crossing(member, incoming)
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
                    None if crossing is None else crossing[1],
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
        rep, pdgs, permutation, conjugated, factor, assignment, rho
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
                # crossed slots: template side != event side (sigma, checked above)
                if rho is None and any(
                    (r < incoming) != (slot < incoming) for r, slot in enumerate(legs)
                ):
                    raise ValueError(f"flavour {pdgs} changes sides without a crossing")
                evaluation = evaluations.setdefault(
                    (rep, value_class, legs), len(evaluations)
                )
                if evaluation == len(described):
                    described.append((rep, assignment, legs, assignment_key, rho))
                elif described[evaluation][4] != rho:
                    raise ValueError(
                        f"flavour {pdgs} has two crossings at one evaluation"
                    )
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
                                # template flows are reversed at template-initial slots
                                pair = list(tags[j])
                                crossed = (j < incoming) != (slot < incoming)
                                flip = crossed != conjugated
                                slot_tags[slot] = pair[::-1] if flip else pair
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
    for index, (rep, assignment, legs, _, _) in enumerate(described):
        table = pattern_tables.get(rep) if assignment >= 0 else None
        if assignment < 0:
            key: tuple = (rep, legs, index)
        elif table is None:
            key = (rep, legs)
        else:
            # template particle and side per event slot: relabellings keep both
            types = [0] * len(legs)
            sides = [False] * len(legs)
            for r, slot in enumerate(legs):
                types[slot] = processes[rep]["pdgs"][r]
                sides[slot] = r < incoming
            key = (rep, tuple(types), tuple(sides))
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
                **(
                    {}
                    if rho is None
                    else {
                        "signs": [
                            -1 if (r < incoming) != (slot < incoming) else 1
                            for r, slot in enumerate(legs)
                        ],
                        "helicity_signs": list(rho),
                    }
                ),
            }
            for (rep, assignment, legs, key, rho) in described
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


def _ints(values: Sequence[int]) -> tuple[int, ...]:
    # every integer array is checked: metadata values reach the provider data
    if all(type(v) is int for v in values):
        return tuple(values)
    return tuple(_integer(v) for v in values)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _process_data(process: Mapping[str, Any]) -> ProcessRow:
    return ProcessRow(
        _text(process["id"]),
        tuple(_integer(v) for v in process["pdgs"]),
        _text(process["alpha_s_parameter"]),
        int(_power(process)),
        len(process["helicity_ids"]),
    )


def _signs(values: Sequence[int] | None, message: str) -> tuple[int, ...] | None:
    if values is None:
        return None
    values = _ints(values)
    if any(v not in (-1, 1) for v in values):
        raise ValueError(message)
    return values


def _switch_rows(
    reference: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    target_route: Any,
) -> SwitchRows:
    """The rows of the flavour switch; target_route maps a target to its target
    process index, its legs in event slots (None: the evaluation's), factor,
    flavour-union assignment (-1: none) and that assignment's process id."""
    evaluations = []
    for i, evaluation in enumerate(catalogue["evaluations"]):
        legs = _ints(evaluation["legs"])
        # a crossed evaluation: momentum signs and helicity signs (rho) per runtime slot
        signs = [
            _signs(evaluation.get(field), f"switch evaluation {i} has invalid {field}")
            for field in ("signs", "helicity_signs")
        ]
        evaluations.append(SwitchEvaluation(
            _integer(evaluation["process"]), legs,
            _integer(evaluation.get("assignment", -1)),
            _text(evaluation.get("assignment_key")), *signs,
        ))
    calls = []
    for call in catalogue.get("calls") or [
        {"process": e["process"], "evaluations": [i], "assignments": None}
        for i, e in enumerate(catalogue["evaluations"])
    ]:
        members = _ints(call["evaluations"])
        if call["assignments"] is None:
            calls.append(SwitchCall(members, None, None, None))
            continue
        entries = call.get("entries") or list(range(len(members)))
        permutations = call.get("permutations")
        calls.append(SwitchCall(
            members, _ints(call["assignments"]), _ints(entries),
            None if permutations is None
            else _ints([slot for permutation in permutations for slot in permutation]),
        ))
    arranged = catalogue.get("arrangement_count", 1) > 1
    rows = catalogue["orderings"]
    orderings = SwitchOrderings(
        array("q", (o["evaluation"] for o in rows)),
        array("q", (o["pdf"][0] for o in rows)),
        array("q", (o["pdf"][1] for o in rows)),
        array("d", (float(o["weight"]) for o in rows)),
        array("q", (o["target"] for o in rows)),
        array("q", (o["arrangement"] if arranged else 0 for o in rows)),
    )
    # equal colour, flow and leg arrays share one tuple
    shared: dict[tuple, tuple] = {}

    def share(values):
        values = _ints(values)
        return shared.setdefault(values, values)

    targets = SwitchTargets(array("q"), array("q"), [], [], [], array("d"), array("q"), [])
    for i, target in enumerate(catalogue["targets"]):
        colors = share(target["colors"])
        flows = share(target.get("flows", range(len(target["colors"]))))
        index, legs, factor, assignment, key = target_route(
            target, catalogue["target_pdgs"][i]
        )
        targets.evaluation.append(target["evaluation"])
        targets.target_process.append(index)
        targets.colors.append(colors)
        targets.flows.append(flows)
        targets.legs.append(None if legs is None else share(legs))
        targets.factor.append(float(factor))
        targets.assignment.append(assignment)
        targets.assignment_key.append(_text(key))
    # only the reference processes of the targets need their flow ids
    used = {evaluations[e].process for e in set(targets.evaluation)}
    flows = [
        tuple(str(i) for i in catalogue["flow_ids"].get(index, [])) if index in used else ()
        for index in range(len(reference["runtime_processes"]))
    ]
    helicities = _ints([v for vector in reference["helicities"] for v in vector])
    return SwitchRows(
        evaluations, calls, orderings, targets, flows, helicities,
        int(catalogue.get("arrangement_count", 1)),
    )


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


def reweight_tables(
    reference: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    reference_path: str,
    target_path: str,
    catalogue: Mapping[str, Any] | None = None,
) -> ReweightTables:
    """The tables of one reweight provider, see ``reweight_provider.h``; with a
    ``switch_catalogue``, also those of the flavour switch. Written by
    ``umami_reweight_data.write_reweight_data``."""
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
    reference_rows = [_process_data(process) for process in reference_processes]

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
    channels = []
    for c, channel in enumerate(reference["channels"]):
        flavours = []
        for f, entry in enumerate(channel["processes"]):
            runtime = entry["runtime"]
            if "helicity_export" not in runtime:
                raise ValueError("the reference UMAMI metadata has no helicity export")
            process_index = _integer(runtime["process_index"])
            process = reference_processes[process_index]
            assignment = runtime.get("assignment")
            reference_helicity = [-1] * exported_count
            target_helicity = [-1] * exported_count
            for runtime_index, exported in enumerate(runtime["helicity_export"]):
                reference_helicity[exported] = runtime_index
            permutation = _ints(runtime["permutation"])
            # a crossed union member gathers sigma * input[permutation]
            reference_signs = None
            if "momentum_signs" in runtime:
                signs = [_integer(v) for v in runtime["momentum_signs"]]
                if assignment is None or any(v not in (-1, 1) for v in signs):
                    raise ValueError(
                        f"flavour {f} of channel {c} has invalid momentum signs"
                    )
                reference_signs = tuple(signs)
            target_permutation, target_factor = None, 1.0
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
                target_permutation = _ints(legs)
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
            switch_arrangement = 0
            if arranged:
                # the arrangement of the non-partons in the entry's rows
                found = {
                    catalogue["row_arrangements"][_integer(term["pdg_ids"])]
                    for term in entry["matrix_elements"]
                }
                if len(found) != 1:
                    raise ValueError(f"flavour {f} of channel {c} mixes arrangements")
                switch_arrangement = found.pop()
            flavours.append(FlavourRow(
                process_index, index, permutation, reference_signs,
                _ints(reference_helicity), _ints(target_helicity),
                -1 if assignment is None else _integer(assignment),
                _text(runtime.get("assignment_key")),
                target_permutation, None, target_factor, target_assignment,
                _text(target_key), switch_arrangement,
            ))
        channels.append(flavours)

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

    switch = None if catalogue is None else _switch_rows(reference, catalogue, target_route)
    if len(set(target_ids)) > 1:
        raise ValueError("the target processes come from different artifacts")
    target_rows = [_process_data(process) for process in target_processes]
    supports_alpha_s = all(
        p["alpha_s_parameter"] is not None
        for p in [*reference_processes, *target_processes]
    )
    provider = ProviderScalars(
        _text(reference["provider"]["id"]),
        _integer(reference["provider"]["particle_count"]),
        exported_count,
        str(reference_path),
        str(target_path),
        _text(reference["provider"].get("artifact_id")),
        _text(target_ids[0] if target_ids else None),
        supports_alpha_s,
    )
    return ReweightTables(provider, reference_rows, target_rows, channels, switch)


def write_reweight_provider(
    reference_artifact: str | os.PathLike[str],
    target_artifact: str | os.PathLike[str],
    output: str | os.PathLike[str],
    switch_lines: int | None = None,
    initial_legs: Sequence[Sequence[int]] | None = None,
) -> list[str]:
    """Write the sources of the reweight providers to ``output``.

    One provider is written per UMAMI provider of the leading-colour reference
    artifact: its tables in ``reweight_<id>.rwd`` and a header binding the
    library to them; ``make -C output`` builds ``build/libreweight_<id>.so``. The
    libraries are initialized with the absolute path of ``output``, where they
    read their data file; the artifacts are found relative to it. Returns the
    provider ids.

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
    # the target side stays unsigned until full-colour crossing
    if any(
        "momentum_signs" in runtime
        for document in targets
        for runtime in _runtime_records(document)
    ):
        raise ValueError(
            "reweight targets with crossed flavour-union members are not supported "
            "(process.flavor_union_crossing)"
        )
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
        write_reweight_data(output, reweight_tables(
            reference, targets, reference_path, target_path, catalogue
        ))
        ids.append(identifier)
    (output / "providers.mk").write_text(
        "UMAMI_PROVIDERS := " + " ".join(ids) + "\n", encoding="utf-8"
    )
    return ids


__all__ = [
    "decode_reweight_data",
    "encode_reweight_data",
    "load_umami_metadata",
    "quark_lines",
    "reweight_tables",
    "switch_catalogue",
    "write_reweight_provider",
]
