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
With rusticol.flavour-union-relabelling.v2 a call spans template sides: it is
keyed by the all-outgoing flavour per event slot (``SWITCH_CROSS_SIDE``). A
template whose calls would derive more than ``MAX_RELABELLED_ASSIGNMENTS`` ids
keeps the per-side key; past that the catalogue is refused at generation.
Pure-QCD orderings read the summed leading colour of one member of their
charge-conjugate class (``SWITCH_C_DEDUP``); the targets keep their own.
Full-colour targets are the members' own processes of the target artifact, or
assignments of its templates when it has flavour unions too (never crossed).
"""

from __future__ import annotations

import heapq
import importlib.resources
import itertools
import json
import math
import operator
import os
import re
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._internal.versions import (
    FLAVOUR_UNION_CROSS_SIDE_RELABELLING_RUNTIME_CAPABILITY,
    FLAVOUR_UNION_RELABELLING_RUNTIME_CAPABILITY,
)
from ..generation.process_grouping import ProcessGroupMap
from .umami_c import _power, helicity_export
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

# Pure-QCD switch totals share one evaluation per charge-conjugate class (False:
# one per flavour, for A/B comparisons).
SWITCH_C_DEDUP = os.environ.get("PYAMPLICOL_SWITCH_C_DEDUP", "1") != "0"
# Switch calls of a template merge across sides by all-outgoing flavour when the
# core relabels across sides (False: per template side, for A/B comparisons).
SWITCH_CROSS_SIDE = os.environ.get("PYAMPLICOL_SWITCH_CROSS_SIDE", "1") != "0"
# Relabelled assignment ids a loaded runtime may derive (rusticol-core engine.rs).
MAX_RELABELLED_ASSIGNMENTS = 4096


def runtime_capabilities() -> frozenset[str]:
    """The runtime capabilities of the native core the providers link: the
    extension's query when it has one, else those of this build's core (the
    extension and the C SDK come from one build)."""
    try:
        from .. import _rusticol
    except ImportError:
        _rusticol = None
    query = getattr(_rusticol, "supported_runtime_capabilities", None)
    if query is not None:
        return frozenset(query())
    return frozenset({
        FLAVOUR_UNION_RELABELLING_RUNTIME_CAPABILITY,
        FLAVOUR_UNION_CROSS_SIDE_RELABELLING_RUNTIME_CAPABILITY,
    })


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
    # schema 5 adds momentum_signs (crossed flavour-union members) to schema 4;
    # schema 6 interns the helicity exports and has columnar helicity mappings
    version = document.get("schema_version")
    if version not in (4, 5, 6):
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
    slots: dict[int, list[int]] = {}
    for j in range(len(source) - 1, -1, -1):
        slots.setdefault(source[j], []).append(j)
    try:
        return [slots[pdg].pop() for pdg in destination]
    except (KeyError, IndexError):
        raise StopIteration from None


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


def _width_parameters(process: Mapping[str, Any]) -> set[int]:
    """The particles with a width parameter among the runtime's parameters."""
    widths = set()
    for parameter in process.get("parameters") or ():
        code = parameter.get("lhacode") or ()
        if parameter.get("lhablock") == "DECAY" and len(code) == 1:
            widths.add(abs(int(code[0])))
        match = re.fullmatch(r"particle\.(-?[0-9]+)\.width", str(parameter["name"]))
        if match:
            widths.add(abs(int(match.group(1))))
    return widths


def _pure_qcd(process: Mapping[str, Any]) -> bool:
    """Whether a runtime's summed leading colour is C symmetric: QCD partons only,
    the proven alpha_s power of a tree without electroweak couplings, and no width
    of a parton (a fixed top width breaks it, e.g. in g g > t t~ q q~). This
    assumes vector-like QCD vertices, as in the SM; a model with other coloured
    couplings of the same alpha_s power needs PYAMPLICOL_SWITCH_C_DEDUP=0."""
    pdgs, power = process["pdgs"], process.get("alpha_s_power")
    return (
        type(power) is int
        and power == len(pdgs) - 2
        and all(_is_qcd_parton(pdg) for pdg in pdgs)
        and not {abs(pdg) for pdg in pdgs} & _width_parameters(process)
    )


def _conjugate(pdg: int | None) -> int | None:
    return pdg if pdg is None or pdg in (21, 22, 23, 25) else -pdg


def _c_representatives(
    evaluation: array,
    weight: array,
    flavours: Mapping[tuple, tuple[int, list[int]]],
    pure: Sequence[bool],
    call_of: Sequence[int],
    costs: Sequence[float],
) -> list[int] | None:
    """The evaluation each one's switch total is read from, or None when nothing is
    shared. The charge conjugate of an ordering's flavour at the same slots has the
    same pure-QCD leading colour; a class reads one member, chosen by a greedy
    cost-weighted cover of the classes by calls."""
    parent = list(range(len(pure)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = False
    for (initial, base, _), (start, partons) in flavours.items():
        key = (
            tuple(map(_conjugate, initial)),
            tuple(map(_conjugate, base)),
            tuple(sorted(map(_conjugate, partons))),
        )
        other = flavours.get(key)
        if other is None:
            continue
        rank = {
            order: k
            for k, order in enumerate(sorted(set(itertools.permutations(other[1]))))
        }
        for k, order in enumerate(sorted(set(itertools.permutations(partons)))):
            o, p = start + k, other[0] + rank[tuple(map(_conjugate, order))]
            a, b = find(evaluation[o]), find(evaluation[p])
            # the ordering factor (non-union members) must agree as well
            if a != b and pure[evaluation[o]] and pure[evaluation[p]] and (
                weight[o] == weight[p]
            ):
                parent[max(a, b)] = min(a, b)
                merged = True
    if not merged:
        return None
    classes: dict[int, list[int]] = {}
    for e in range(len(parent)):
        classes.setdefault(find(e), []).append(e)
    members = list(classes.values())
    covering: list[set[int]] = [set() for _ in costs]
    for c, group in enumerate(members):
        for e in group:
            covering[call_of[e]].add(c)
    chosen: set[int] = set()
    covered = [False] * len(members)

    def choose(g: int) -> None:
        chosen.add(g)
        for c in covering[g]:
            covered[c] = True

    # a class within one call forces it; then most classes per cost, lowest index
    for group in members:
        calls = {call_of[e] for e in group}
        if len(calls) == 1:
            choose(calls.pop())
    open_count = [sum(not covered[c] for c in classes_) for classes_ in covering]
    heap = [
        (-n / costs[g], g) for g, n in enumerate(open_count) if n and g not in chosen
    ]
    heapq.heapify(heap)
    while heap:
        _, g = heapq.heappop(heap)
        if g in chosen:
            continue
        n = sum(not covered[c] for c in covering[g])
        if n == 0:
            continue
        if heap and (-n / costs[g], g) > heap[0]:
            heapq.heappush(heap, (-n / costs[g], g))
            continue
        choose(g)
    representative = list(range(len(parent)))
    for group in members:
        rep = min(e for e in group if call_of[e] in chosen)
        for e in group:
            representative[e] = rep
    return representative


def _pdf_column(pdg: int) -> int:
    if pdg == 21:
        return 6
    if not 0 < abs(pdg) <= 6:
        raise ValueError(f"the flavour switch needs partons in the initial state, got {pdg}")
    return pdg + 6


def _flat(flow: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(itertools.chain.from_iterable(flow))


def _flow_key(values: Sequence[int]) -> bytes:
    # int16 when it fits (LHE tags are ~500), else int32: the lengths differ
    try:
        return array("h", values).tobytes()
    except OverflowError:
        return array("i", values).tobytes()


class _Rows(Sequence):
    """Read-only rows of a columnar table, built on access."""

    __slots__ = ("_count", "_row")

    def __init__(self, count: int, row: Callable[[int], Any]) -> None:
        self._count, self._row = count, row

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._row(i) for i in range(*index.indices(self._count))]
        index = operator.index(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._row(index)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, str | bytes):
            return len(self) == len(other) and all(
                a == b for a, b in zip(self, other, strict=True)
            )
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SwitchTargetTable:
    """The switch targets of a catalogue by column: evaluation, colour set
    (colours, flow positions) and pdg ids in event slots (``width`` each)."""

    evaluation: array
    color_set: array
    color_sets: list[tuple[tuple[int, ...], tuple[int, ...]]]
    pdgs: array
    width: int

    def pdgs_of(self, index: int) -> list[int]:
        return self.pdgs[index * self.width:(index + 1) * self.width].tolist()


def _sides_crossed(legs: Sequence[int], incoming: int) -> tuple[bool, ...]:
    # template slot r reads an event slot of the other side
    return tuple((r < incoming) != (slot < incoming) for r, slot in enumerate(legs))


def _switch_calls(
    described: Sequence[tuple],
    indices: Iterable[int],
    call_key: Callable[[int], tuple],
    pattern_tables: Mapping[int, tuple],
    member_ids: Mapping[tuple[int, int], tuple[str, tuple]],
    incoming: int = 2,
    conjugated: Mapping[tuple[int, int], bool] | None = None,
    derived_ids: dict[int, set[tuple]] | None = None,
) -> list[dict[str, Any]]:
    """The native calls of the switch evaluations ``indices``. One call per
    (runtime, legs): the assignments of a template share it. With a pattern
    table, one per ``call_key`` (template and leg flavours): an evaluation at legs
    l is its assignment relabelled by pi[r] = c^-1[l[r]] at the call's legs c,
    those of its first evaluation (the legs source: the C code gathers the
    momenta and crossing signs there), uncrossed when the call has one.
    ``derived_ids`` collects per runtime the keys the core interns a new id for."""
    groups: dict[tuple, list[int]] = {}
    for index in indices:
        groups.setdefault(call_key(index), []).append(index)
    calls = []
    for key, members in groups.items():
        source = next((i for i in members if described[i][4] is None), members[0])
        rep, assignment, legs, _, rho = described[source]
        table = pattern_tables.get(rep) if assignment >= 0 else None
        call: dict[str, Any] = {
            "process": rep,
            "legs": list(legs),
            "evaluations": [],
            "assignments": [] if assignment >= 0 else None,
            "permutations": [] if table is not None else None,
            "entries": [],
        }
        calls.append(call)
        # relabelled requests take the source's crossing (sigma' = sigma_c), which
        # the core refuses for a conjugated base
        crossed = rho is not None
        entry_of: dict[tuple, int] = {}
        inverse = {slot: r for r, slot in enumerate(legs)}
        sigma_c = _sides_crossed(legs, incoming)
        for index in [source, *(i for i in members if i != source)]:
            rep_i, assignment, legs_i, _, _ = described[index]
            if rep_i != rep or (assignment < 0) != (call["assignments"] is None):
                raise AssertionError(f"switch call {key!r} mixes runtimes")
            if crossed and conjugated and conjugated.get((rep, assignment), False):
                raise AssertionError(
                    f"switch call {key!r} crosses conjugated assignment {assignment}"
                )
            call["evaluations"].append(index)
            if assignment < 0:
                call["entries"].append(0)
                continue
            if table is None:
                call["entries"].append(len(call["assignments"]))
                call["assignments"].append(assignment)
                continue
            permutation = tuple(inverse[slot] for slot in legs_i)
            # the crossing of the relabelled id: sigma_e flipped where r moves
            sigma_e = _sides_crossed(legs_i, incoming)
            if any(
                (sigma_e[r] ^ ((r < incoming) != (p < incoming))) != sigma_c[p]
                for r, p in enumerate(permutation)
            ):
                raise AssertionError(f"switch call {key!r} changes a crossing")
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
                    (lookup[pattern], sign)
                    for pattern, sign in zip(moved, signs, strict=True)
                ))
            entry = entry_of.setdefault(
                (derived, numerator, denominator), len(call["assignments"])
            )
            if entry == len(call["assignments"]):
                call["assignments"].append(assignment)
                call["permutations"].append(list(permutation))
                # the core keys a derived id by (base, crossing, patterns)
                own = patterns if patterns == ("template",) else tuple(
                    sorted(zip(patterns, signs, strict=True))
                )
                if derived_ids is not None and (derived != own or sigma_c != sigma_e):
                    keys = derived_ids.setdefault(rep, set())
                    keys.add((assignment, sigma_c, derived))
            call["entries"].append(entry)
        if call["evaluations"][0] != source or call["legs"] != list(
            described[call["evaluations"][0]][2]
        ):
            raise AssertionError(f"switch call {key!r} has another legs source")
        # identity relabellings need no runtime call
        permutations = call["permutations"]
        if permutations is not None and all(p == sorted(p) for p in permutations):
            call["permutations"] = None
    return calls


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
    # per (runtime, assignment): whether the member is conjugated (never crossed)
    conjugated_of: dict[tuple[int, int], bool] = {}
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
                    conjugated_of[(rep, assignment)] = bool(member["conjugated"])
    # LHE colour tags of every flow of a runtime process, in its legs
    flows: dict[int, dict[str, int]] = {index: {} for index in by_id.values()}
    for channel in reference["channels"]:
        for entry in channel["processes"]:
            for member in entry.get("members", []):
                index = by_id.get(member["process_id"])
                if index is not None and member.get("color_id") is not None:
                    flows[index][member["color_id"]] = member["color_flows"]
    flow_ids = {index: sorted(table) for index, table in flows.items()}
    n = len(reference["pdg_ids"][0]) if reference["pdg_ids"] else 0
    base_flows = reference["color_flows"]
    # colour flows by content: the metadata's, then those the targets add (flat,
    # two tags per leg); keys are packed bytes, not tuples (millions at 7j)
    color_index = {_flow_key(_flat(flow)): i for i, flow in enumerate(base_flows)}
    new_flows = array("i")
    # per runtime: (flow position, its flat tags + a 0 for unassigned slots)
    rep_tags = {
        rep: [(k, (*_flat(base_flows[table[flow_id]]), 0))
              for k, flow_id in enumerate(flow_ids[rep])]
        for rep, table in flows.items()
    }
    # the (colours, flows) of a target per (runtime, legs, conjugated): an index
    # into color_sets, whose equal tuples are shared
    target_colors: dict[tuple, int] = {}
    color_sets: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    color_set_index: dict[tuple, int] = {}

    def color_set(rep: int, legs: tuple[int, ...], conjugated: bool) -> int:
        # flat event-slot tag k reads flat template tag source[k]
        source = [2 * n] * (2 * n)
        for j, slot in enumerate(legs):
            # template flows are reversed at template-initial slots
            flip = ((j < incoming) != (slot < incoming)) != conjugated
            source[2 * slot] = 2 * j + flip
            source[2 * slot + 1] = 2 * j + (not flip)
        gather = operator.itemgetter(*source)
        found = []
        for k, tags in rep_tags[rep]:
            slot_tags = gather(tags)
            # colour tags renumbered by first appearance (0: none)
            names = dict(zip(dict.fromkeys((0, *slot_tags)), itertools.count()))
            found.append((tuple(map(names.__getitem__, slot_tags)), k, slot_tags))
        # the flows in physical order, the order of the colour draw
        found.sort()
        colors = []
        for _, _, slot_tags in found:
            key = _flow_key(slot_tags)
            index = color_index.get(key)
            if index is None:
                index = color_index[key] = len(base_flows) + len(new_flows) // (2 * n)
                new_flows.extend(slot_tags)
            colors.append(index)
        key = (tuple(colors), tuple(k for _, k, _ in found))
        index = color_set_index.get(key)
        if index is None:
            index = color_set_index[key] = len(color_sets)
            color_sets.append(key)
        return index

    evaluations: dict[tuple, int] = {}
    described: list[tuple] = []
    seen: set[tuple] = set()
    # per (initial, outgoing non-partons by slot, sorted partons): its first
    # ordering and partons
    flavours: dict[tuple, tuple[int, list[int]]] = {}
    # the orderings and targets by column (millions of rows at 7j)
    order_columns = SwitchOrderings(
        array("i"), array("b"), array("b"), array("d"), array("i"), array("i")
    )
    target_evaluation, target_color_set = array("i"), array("i")
    target_pdgs = array("i")
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
    parton_positions = [slot - incoming for slot in parton_slots]
    append_evaluation = order_columns.evaluation.append
    append_pdf0, append_pdf1 = order_columns.pdf0.append, order_columns.pdf1.append
    append_weight = order_columns.weight.append
    append_target = order_columns.target.append
    append_arrangement = order_columns.arrangement.append
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
        # the matrix element is symmetric in identical legs reading outgoing slots
        identical = {}
        for k, pdg in enumerate(rep_pdgs):
            identical.setdefault(pdg, []).append(k)
        identical = [ks for ks in identical.values() if len(ks) > 1]
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
            final = pdgs[incoming:]
            partons = [p for p in final if p in jet]
            first = canonical.get(tuple(sorted(p for p in final if p not in jet)))
            if first is None:
                raise ValueError(f"flavour {pdgs} has other non-partons than the events")
            moves = dict(sigma)
            fixed = {moves.get(slot, slot): pdg for slot, pdg in first.items()}
            base_order = [fixed.get(s) for s in range(incoming, len(pdgs))]
            weight = factor * math.prod(
                math.factorial(partons.count(p)) for p in set(partons)
            ) / n_partons
            pdf0, pdf1 = _pdf_column(initial[0]), _pdf_column(initial[1])
            is_target = quark_lines(pdgs) >= switch_lines
            prefix = [0, 1] if orientation == 0 else [1, 0]
            # the first ordering of the flavour; its parton orders follow sorted
            flavours[initial, tuple(base_order), tuple(sorted(partons))] = (
                len(order_columns.evaluation), partons
            )
            for parton_order in sorted(set(itertools.permutations(partons))):
                order = list(base_order)
                for position, pdg in zip(parton_positions, parton_order, strict=True):
                    order[position] = pdg
                slots = prefix + [incoming + j for j in _first_unused(order, final)]
                legs = [slots[r] for r in permutation]
                for ks in identical:
                    reading = [k for k in ks if legs[k] >= incoming]
                    if len(reading) > 1:
                        values = sorted([legs[k] for k in reading])
                        for k, value in zip(reading, values, strict=True):
                            legs[k] = value
                legs = tuple(legs)
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
                if is_target:
                    target = len(target_evaluation)
                    colors = target_colors.get((rep, legs, conjugated))
                    if colors is None:
                        colors = target_colors[rep, legs, conjugated] = color_set(
                            rep, legs, conjugated
                        )
                    target_evaluation.append(evaluation)
                    target_color_set.append(colors)
                    target_pdgs.extend(initial)
                    target_pdgs.extend(order)
                append_evaluation(evaluation)
                append_pdf0(pdf0)
                append_pdf1(pdf1)
                append_weight(weight)
                append_target(target)
                append_arrangement(arrangement)
    cross_side = SWITCH_CROSS_SIDE and (
        FLAVOUR_UNION_CROSS_SIDE_RELABELLING_RUNTIME_CAPABILITY
        in runtime_capabilities()
    )
    # templates whose cross-side calls exceed MAX_RELABELLED_ASSIGNMENTS
    side_keyed: set[int] = set()

    def call_key(index: int) -> tuple:
        rep, assignment, legs, _, _ = described[index]
        if assignment < 0:
            return (rep, legs, index)
        if pattern_tables.get(rep) is None:
            return (rep, legs)
        types = [0] * len(legs)
        if cross_side and rep not in side_keyed:
            # all-outgoing template flavour per event slot, kept across sides
            for r, slot in enumerate(legs):
                pdg = processes[rep]["pdgs"][r]
                types[slot] = _conjugate(pdg) if r < incoming else pdg
            return (rep, tuple(types))
        # template particle and side per event slot: same-side relabellings keep both
        sides = [False] * len(legs)
        for r, slot in enumerate(legs):
            types[slot] = processes[rep]["pdgs"][r]
            sides[slot] = r < incoming
        return (rep, tuple(types), tuple(sides))

    evaluation_column = array("i", order_columns.evaluation)

    def build_calls(derived: dict[int, set[tuple]]) -> list[dict[str, Any]]:
        order_columns.evaluation[:] = evaluation_column
        calls = _switch_calls(
            described, range(len(described)), call_key, pattern_tables,
            member_ids, incoming, conjugated_of, derived,
        )
        pure = [_pure_qcd(processes[e[0]]) for e in described] if SWITCH_C_DEDUP else []
        if any(pure):
            call_of = [0] * len(described)
            costs = []
            for g, call in enumerate(calls):
                for e in call["evaluations"]:
                    call_of[e] = g
                # a call costs about one template recursion per helicity and flow
                rep = call["process"]
                costs.append(float(max(1, len(processes[rep].get("helicity_ids") or ()))
                                   * max(1, len(flow_ids.get(rep, ())))))
            representative = _c_representatives(
                order_columns.evaluation, order_columns.weight, flavours, pure, call_of,
                costs,
            )
            if representative is not None:
                # the orderings read the representatives; targets keep their own
                order_columns.evaluation[:] = array(
                    "i", map(representative.__getitem__, order_columns.evaluation)
                )
                derived.clear()
                calls = _switch_calls(
                    described, sorted(set(order_columns.evaluation)), call_key,
                    pattern_tables, member_ids, incoming, conjugated_of, derived,
                )
        return calls

    def over_cap(derived: dict[int, set[tuple]]) -> set[int]:
        cap = MAX_RELABELLED_ASSIGNMENTS
        return {r for r, keys in derived.items() if len(keys) > cap}

    derived: dict[int, set[tuple]] = {}
    calls = build_calls(derived)
    over = over_cap(derived)
    if over and cross_side:
        # per template side: fewer moved slots, so fewer derived ids
        side_keyed.update(over)
        derived = {}
        calls = build_calls(derived)
        over = over_cap(derived)
    if over:
        worst = max(over, key=lambda rep: len(derived[rep]))
        raise ValueError(
            f"switch calls of {processes[worst]['id']!r} need {len(derived[worst])} "
            f"relabelled assignments, more than the {MAX_RELABELLED_ASSIGNMENTS} a "
            "runtime may derive"
        )
    targets = SwitchTargetTable(
        target_evaluation, target_color_set, color_sets, target_pdgs, n
    )
    width = 2 * n
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
        "calls": calls,
        "ordering_table": order_columns,
        "target_table": targets,
        # row views of the tables (inspection and tests)
        "orderings": _Rows(len(order_columns.evaluation), lambda i: {
            "evaluation": order_columns.evaluation[i],
            "pdf": [order_columns.pdf0[i], order_columns.pdf1[i]],
            "weight": order_columns.weight[i],
            "target": order_columns.target[i],
            "arrangement": order_columns.arrangement[i],
        }),
        "targets": _Rows(len(target_evaluation), lambda i: {
            "evaluation": target_evaluation[i],
            "colors": list(color_sets[target_color_set[i]][0]),
            "flows": list(color_sets[target_color_set[i]][1]),
        }),
        "target_pdgs": _Rows(len(target_evaluation), targets.pdgs_of),
        "flow_ids": flow_ids,
        # the flows the targets add after the metadata's, flat
        "color_flow_table": new_flows,
        "color_flows": _Rows(len(new_flows) // width if width else 0, lambda i: [
            list(new_flows[k:k + 2]) for k in range(i * width, (i + 1) * width, 2)
        ]),
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
    """The rows of the flavour switch; target_route maps a target (its
    evaluation, its pdg ids in event slots) to its target process index, its
    legs in event slots (None: the evaluation's), factor, flavour-union
    assignment (-1: none) and that assignment's process id."""
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
    orderings = catalogue["ordering_table"]
    table = catalogue["target_table"]
    # equal colour, flow and leg arrays share one tuple
    shared: dict[tuple, tuple] = {}

    def share(values):
        values = tuple(values)
        known = shared.get(values)
        if known is None:
            known = shared[values] = _ints(values)
        return known

    sets = [(share(colors), share(flows)) for colors, flows in table.color_sets]
    count = len(table.evaluation)
    targets = SwitchTargets(
        array("i"), array("i"), [None] * count, [None] * count, [None] * count,
        array("d"), array("i"), [None] * count,
    )
    for i in range(count):
        evaluation = table.evaluation[i]
        colors, flows = sets[table.color_set[i]]
        index, legs, factor, assignment, key = target_route(
            evaluation, table.pdgs_of(i)
        )
        targets.evaluation.append(evaluation)
        targets.target_process.append(index)
        targets.colors[i] = colors
        targets.flows[i] = flows
        targets.legs[i] = None if legs is None else share(legs)
        targets.factor.append(float(factor))
        targets.assignment.append(assignment)
        targets.assignment_key[i] = _text(key)
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

    # per (target process, legs): the target helicity of each exported one (-1: none)
    routed: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def routed_helicities(target, legs: tuple[int, ...]) -> list[int]:
        key = (id(target), legs)
        if key not in routed:
            values = {
                _helicity_values(identifier): index
                for index, identifier in enumerate(target["helicity_ids"])
            }
            routed[key] = [
                values.get(tuple(vector[slot] for slot in legs), -1)
                for vector in reference["helicities"]
            ]
        return routed[key]

    exported_count = len(reference["helicities"])
    # helicity maps by export (and target): flavours share few distinct ones
    inverse_exports: dict[tuple, tuple[int, ...]] = {}
    target_exports: dict[tuple, tuple[int, ...]] = {}
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
            export = tuple(helicity_export(reference, runtime))
            reference_helicity = inverse_exports.get(export)
            if reference_helicity is None:
                inverse = [-1] * exported_count
                for runtime_index, exported in enumerate(export):
                    inverse[exported] = runtime_index
                reference_helicity = inverse_exports[export] = tuple(inverse)
            target_helicity = [-1] * exported_count
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
                key = (id(target), process_index, export)
                if key not in target_exports:
                    target_helicities = helicity_positions(target)
                    for runtime_index, exported in enumerate(export):
                        identifier = process["helicity_ids"][runtime_index]
                        if identifier in target_helicities:
                            target_helicity[exported] = target_helicities[identifier]
                    target_exports[key] = tuple(target_helicity)
                target_helicity = target_exports[key]
            else:
                # A flavour-union member or target: its own route, legs and factor.
                pdgs = reference["pdg_ids"][entry["matrix_elements"][0]["pdg_ids"]]
                route = _route(route_index(), pdgs, incoming)
                target, target_id, legs, target_factor = route[:4]
                target_assignment, target_key = route[4:]
                target_permutation = _ints(legs)
                key = (id(target), target_permutation, export)
                if key not in target_exports:
                    table = routed_helicities(target, target_permutation)
                    for exported in export:
                        target_helicity[exported] = table[exported]
                    target_exports[key] = tuple(target_helicity)
                target_helicity = target_exports[key]
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
                # built from integers here: no per-value check
                reference_helicity, target_helicity,
                -1 if assignment is None else _integer(assignment),
                _text(runtime.get("assignment_key")),
                target_permutation, None, target_factor, target_assignment,
                _text(target_key), switch_arrangement,
            ))
        channels.append(flavours)

    # per switch evaluation: its route when it is the target's own process (None:
    # each target is routed by its flavour)
    own_routes: dict[int, tuple | None] = {}

    def own_route(evaluation_index):
        evaluation = catalogue["evaluations"][evaluation_index]
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
        return None

    def target_route(evaluation_index, pdgs):
        if evaluation_index not in own_routes:
            own_routes[evaluation_index] = own_route(evaluation_index)
        route = own_routes[evaluation_index]
        if route is not None:
            return route
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


def write_switch_sidecar(
    path: Path, catalogue: Mapping[str, Any], color_flow_offset: int
) -> None:
    """Write ``flavor_switch_<id>.json`` in chunks: the bytes of ``json.dumps``
    of {switch_lines, targets, color_flow_offset, color_flows}."""
    targets = catalogue["target_table"]
    flows = catalogue["color_flow_table"]
    width = targets.width

    def rows(values: array, row: str, size: int):
        # ", "-joined rows of ``size`` values, ~1 MB of text at a time
        step = max(1, 65536 // max(1, size))
        count = len(values) // size if size else 0
        for start in range(0, count, step):
            stop = min(count, start + step)
            text = ", ".join(
                row % tuple(values[k * size:(k + 1) * size]) for k in range(start, stop)
            )
            yield (", " if start else "") + text

    with open(path, "w", encoding="utf-8") as stream:
        lines = json.dumps(catalogue["switch_lines"])
        stream.write(f'{{"switch_lines": {lines}, "targets": [')
        for chunk in rows(targets.pdgs, "[" + ", ".join(["%d"] * width) + "]", width):
            stream.write(chunk)
        offset = json.dumps(color_flow_offset)
        stream.write(f'], "color_flow_offset": {offset}, "color_flows": [')
        pair = "[" + ", ".join(["[%d, %d]"] * width) + "]"
        for chunk in rows(flows, pair, 2 * width):
            stream.write(chunk)
        stream.write("]}")


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
            write_switch_sidecar(
                output / f"flavor_switch_{identifier}.json",
                catalogue,
                len(reference["color_flows"]),
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
    "write_switch_sidecar",
]
