# SPDX-License-Identifier: 0BSD
"""Leading-colour helicity and colour-flow selection of v2 virtuals (MQ-SELECT).

At leading colour the virtual is diagonal in the Born colour flows (design
§7.5): sum over flows sigma and Born helicity rows h of
N_c^P conj(A0_sigma(h)) A1_sigma(h), plus the Born-anchored counterterm and
field pieces. A selection keeps some helicity rows and some flows, with the
tree engine's conventions (``generation.physics_metadata``):

* a helicity row's ID is ``h:<signed values in leg order>`` (``_helicity_id``),
  the runtime's own ``h<index>`` names it as well;
* a colour flow's ID is ``flow:<word>`` (``_color_id``) of a sector of the
  tree engine's leading-colour plan (``color.plan.build_color_plan`` at LC,
  with the request's ``reference_color_order`` and ``max_color_sectors``,
  trace reflections folded as the tree engine folds them, each folded
  sector's reflection a flow of its own); the flows an artifact computes
  (all, or those of a generation-time selection) are numbered from 1 in
  the tree runtime's order (plan sectors, a folded sector's reflection
  after it), as the tree runtime numbers its computed flows.

A flow is the Born basis element (``colour.BornBasis``) with the same outer
flow (``colour.sector_flow``); Born basis elements of no leading-colour flow
(U(1) and Fierz pieces) have no leading-colour contribution. The loop's
leading colour is the leading N_c power of the squared matrix element: a
tree-engine flow below it (u u~ > d d~ g's flows joining each quark to its
own antiquark) has a vanishing leading-colour Born and virtual: its
selection is zero at evaluation; at generation it contributes exact zeros
next to a leading flow, and a selection without one is refused
(``subleading-colour-flows``). Between two Born orders (QCD and electroweak
Born amplitudes) a flow can interfere with another at the leading power
(u u~ > d d~: the QCD flow (2,4,3,1) with the photon's (2,1,3,4) at
QCD^2 QED^2): the evaluation-time selection keeps that term (its bra is the
flow, its ket the whole Born); a generation-time selection restricts the
Born itself, so one that would drop it is refused
(``interfering-colour-flows``). The selected
output is the observable column of those rows and basis elements
(``observables.Column``): its seeds and its Born, counterterm and field bras
keep them only, so a complete set of selections sums to the inclusive value.
Next-to-leading and full colour expose no selection.

A runtime forms and compiles each distinct evaluation-time selection's
column set on first use and keeps the ``SELECTION_CACHE_DEFAULT`` most
recently used (``SelectionCache``; ``$PYAMPLICOL_LOOP_SELECTION_CACHE``
sets the count, 0 keeps none): a kept selection holds its native plan and
executors (u u~ > g g g g at leading colour: about 66 MB and 0.9 s of setup
per (helicity, flow) pair), so sampling many distinct selections stays
bounded in memory (apart from about 1.5 MB of native heap each formation
leaves, as any rebuilt native executor does; design §8.5).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyamplicol.api.errors import CompatibilityError, EvaluationError

from . import reference_executor as rx


@dataclass(frozen=True)
class Selection:
    """A generation-time leading-colour selection (``SliceRequest.selection``;
    the public route's ``[process]`` selectors): the Born helicity of some
    legs (label, helicity) and the colour flows kept (tree-engine words;
    ``None`` keeps every flow). The generation prunes the recursion to them:
    the Born closure keeps the selected flows' tensors only, so the seed
    table, the colour pruning of the skeletons, the Born and counterterm
    bras and the tree program follow (``restrict_born_terms``), and the
    blocks keep the selected helicity rows only (``restrict_rows``).

    ``color_order`` and ``max_sectors`` are the tree engine's
    ``reference_color_order`` and ``max_color_sectors``: they build the
    leading-colour plan whose order numbers the artifact's flows
    (``request_flows``); they select nothing themselves."""

    helicities: tuple[tuple[int, int], ...] = ()
    flows: tuple[tuple[int, ...], ...] | None = None
    color_order: tuple[int, ...] | None = None
    max_sectors: int | None = None

    def __post_init__(self) -> None:
        helicities = tuple(
            sorted({(int(label), int(h)) for label, h in self.helicities})
        )
        if len({label for label, _h in helicities}) != len(helicities):
            raise ValueError("a selection names one helicity per leg")
        object.__setattr__(self, "helicities", helicities)
        if self.flows is not None:
            flows = tuple(sorted({tuple(int(x) for x in w) for w in self.flows}))
            if not flows:
                raise ValueError("a colour-flow selection keeps at least one flow")
            object.__setattr__(self, "flows", flows)
        if self.color_order is not None:
            object.__setattr__(
                self, "color_order", tuple(int(x) for x in self.color_order) or None
            )
        if self.max_sectors is not None:
            object.__setattr__(self, "max_sectors", int(self.max_sectors))

    @property
    def restricts(self) -> bool:
        """Whether the selection keeps fewer helicity rows or flows than the
        process has (``color_order`` and ``max_sectors`` alone only order)."""

        return bool(self.helicities) or self.flows is not None

    def record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "helicities": [list(pair) for pair in self.helicities],
            "flows": None if self.flows is None else [list(w) for w in self.flows],
        }
        # The plan options only when set (a plain selection's record is
        # unchanged).
        if self.color_order is not None:
            record["color_order"] = list(self.color_order)
        if self.max_sectors is not None:
            record["max_sectors"] = self.max_sectors
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Selection:
        flows = record["flows"]
        order = record.get("color_order")
        return cls(
            tuple((int(a), int(b)) for a, b in record["helicities"]),
            None if flows is None else tuple(tuple(int(x) for x in w) for w in flows),
            None if order is None else tuple(int(x) for x in order),
            record.get("max_sectors"),
        )


def restrict_born_terms(
    born_terms: Mapping[Any, tuple[Any, ...]],
    legs: Sequence[Any],
    flows: Sequence[Sequence[int]],
) -> dict[Any, tuple[Any, ...]]:
    """The Born closure's terms with the colour components of the selected
    flows only (``Selection.flows``, tree words; ``colour.outer_flow`` keys):
    the Born basis (``assemble._born_tensors``) is then those flows' tensors.

    At leading colour this is the selection's bra: within one Born order
    distinct flows do not interfere at the leading power, nor does a flow
    with a Born tensor of no flow (U(1) and Fierz pieces), so the Born of the
    kept terms is the selected flows' and the seeds, counterterm and field
    bras keep them only. A selected flow without a Born tensor is refused.
    Once the slice grades its Born (``LoopSlice._refuse_inexact_flows``) a
    selection is refused whose flows are all below the process's leading
    colour where they reach it (next to a leading flow such a flow's
    contributions are exact zeros), or one of whose flows interferes at the
    leading power with an unselected flow (possible between two Born orders,
    e.g. QCD and photon Born amplitudes)."""

    from dataclasses import replace

    from .colour import outer_flow

    representations = {leg.label: leg.representation for leg in legs}
    wanted = {}
    for word in flows:
        key = outer_flow(word, representations)
        if key is None:
            raise ValueError(f"colour flow {list(word)} has no planar reading")
        wanted[key] = tuple(word)
    out: dict[Any, tuple[Any, ...]] = {}
    seen = set()
    for order, terms in born_terms.items():
        kept = []
        for term in terms:
            items = tuple(
                item for item in term.colour.terms if _item_flow(item) in wanted
            )
            seen.update(_item_flow(item) for item in items)
            if items:
                kept.append(replace(term, colour=replace(term.colour, terms=items)))
        out[order] = tuple(kept)
    missing = [list(word) for key, word in wanted.items() if key not in seen]
    if missing:
        raise ValueError(f"the selected colour flows {missing} have no Born tensor")
    return out


def restrict_rows(blocks: Any, helicities: Sequence[tuple[int, int]]) -> Any:
    """``TreeBlocks`` whose blocks keep the helicity rows of the selected leg
    helicities only (``Selection.helicities``): the Born rows, the frames,
    the dressing rows and the tree program follow (``plan_export.
    born_helicities``, ``generator_input.leg_rows``). A block left without a
    row is dropped; a selected helicity a leg does not have is refused."""

    from dataclasses import replace

    fixed = dict(helicities)
    if not fixed:
        return blocks
    labels = {leg.label for leg in blocks.process.legs}
    if not set(fixed) <= labels:
        raise ValueError(
            f"selected helicities name legs {sorted(set(fixed) - labels)} the "
            "process does not have"
        )
    available: dict[int, set[int]] = {label: set() for label in fixed}
    kept = []
    for block in blocks.blocks:
        if block.is_source:
            for row in block.rows:
                for label, h in row.helicities:
                    if label in available:
                        available[label].add(h)
        rows = tuple(
            row
            for row in block.rows
            if all(fixed.get(label, h) == h for label, h in row.helicities)
        )
        if rows:
            kept.append(replace(block, rows=rows))
    for label, h in fixed.items():
        if h not in available[label]:
            raise ValueError(
                f"selected helicity {h} is unavailable for external leg {label} "
                f"(its helicities: {sorted(available[label])})"
            )
    return replace(blocks, blocks=tuple(kept))


def require_leg_helicities(
    compiled: Any, process: str, helicities: Sequence[tuple[int, int]]
) -> None:
    """The selected leg helicities (``Selection.helicities``) name legs of
    ``process`` and helicities they have, the tree engine's source states
    (crossed for an incoming leg, as ``generation.eager_columnar`` lists
    them): checked before any generation (the dry run), as
    ``restrict_rows`` checks the generated rows."""

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    if not helicities:
        return
    ir = build_model_process_ir(process, compiled.ir, color_accuracy="lc")
    model = CompiledUFOModel(compiled)
    available: dict[int, set[int]] = {}
    for leg in ir.legs:
        if leg.outgoing_pdg is None:
            continue
        source = model._source_ir(int(leg.outgoing_pdg))
        available[int(leg.label)] = {
            int((source.crossing.apply(state) if leg.is_initial else state).helicity)
            for state in source.states
        }
    fixed = dict(helicities)
    if unknown := sorted(set(fixed) - set(available)):
        raise ValueError(
            f"selected helicities name legs {unknown} the process does not have"
        )
    for label, h in sorted(fixed.items()):
        if h not in available[label]:
            raise ValueError(
                f"selected helicity {h} is unavailable for external leg {label} "
                f"(its helicities: {sorted(available[label])})"
            )


def _item_flow(item: Any) -> Any:
    from .colour import _canonical_rotation

    colour = item.tensor
    strings = tuple(
        sorted(
            (s.fundamental_label, *s.adjoint_labels, s.antifundamental_label)
            for s in colour.open_strings
        )
    )
    return strings, tuple(_canonical_rotation(tuple(t)) for t in colour.traces)


@dataclass(frozen=True)
class Flow:
    """One leading-colour flow: its tree-engine ID and word, its Born basis
    element (``None``: no Born tensor of that flow, so it contributes
    nothing) and its plan sector id (a folded trace's reflection has its
    sector's: the loop computes both flows, the tree engine serves the
    reflection by its sector's)."""

    id: str
    word: tuple[int, ...]
    basis: int | None
    sector: int

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "word": list(self.word),
            "basis": self.basis,
            "sector": self.sector,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Flow:
        return cls(
            str(record["id"]),
            tuple(int(x) for x in record["word"]),
            None if record["basis"] is None else int(record["basis"]),
            int(record["sector"]),
        )


def helicity_id(row: rx.Row) -> str:
    """The tree engine's ID of a Born helicity row (values in leg order)."""

    from pyamplicol.generation.physics_metadata import _helicity_id

    return _helicity_id([h for _leg, h in sorted(row)])


def flow_id(word: Sequence[int]) -> str:
    from pyamplicol.generation.physics_metadata import _color_id

    return _color_id(word)


def _tensor_flow(tensor: Any) -> Any:
    """The ``colour.sector_flow`` key of a one-term literal Born tensor."""

    (term,) = tensor.terms
    return _item_flow(term)


def leading_colour_plan(
    compiled: Any,
    process: str,
    *,
    max_sectors: int | None = None,
    reference_color_order: Sequence[int] | None = None,
) -> Any:
    """The tree engine's leading-colour plan of ``process``
    (``GenerationService._plan_concrete_process``'s ``build_color_plan``
    call: the model's trace-reflection folding)."""

    from pyamplicol.color.plan import build_color_plan
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    ir = build_model_process_ir(process, compiled.ir, color_accuracy="lc")
    model = CompiledUFOModel(compiled)
    return build_color_plan(
        ir,
        color_accuracy="lc",
        max_sectors=max_sectors,
        reference_color_order=(
            None if not reference_color_order else tuple(reference_color_order)
        ),
        fold_trace_reflections=model.lc_trace_reflection_equivalence_is_proven(ir),
    )


def plan_flows(plan: Any, basis: Any) -> tuple[Flow, ...]:
    """The physical flows of a leading-colour plan in the tree runtime's
    ``ColorFlow`` order (each sector's flow, then for a folded trace its
    reflection), each with its element of ``basis``."""

    from .colour import _canonical_rotation, sector_flow

    elements: dict[Any, int] = {}
    for b, tensor in enumerate(getattr(basis, "tensors", ())):
        elements.setdefault(_tensor_flow(tensor), b)
    found: dict[str, Flow] = {}
    for sector in plan.sectors:
        word = tuple(int(x) for x in sector.color_words[0])
        identifier = flow_id(word)
        found.setdefault(
            identifier,
            Flow(identifier, word, elements.get(sector_flow(sector)), int(sector.id)),
        )
        if not (
            plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(sector.trace_labels) > 2
        ):
            continue
        reflected = (word[0], *reversed(word[1:]))
        other = flow_id(reflected)
        if other == identifier:
            continue
        key = ((), (_canonical_rotation(reflected),))
        found.setdefault(
            other,
            Flow(other, reflected, elements.get(key), int(sector.id)),
        )
    return tuple(found.values())


def slice_flows(slice_: Any) -> tuple[Flow, ...]:
    """The leading-colour flows of an LC slice (``request_flows`` on its Born
    basis)."""

    return request_flows(slice_.compiled, slice_.request, slice_.basis)


def request_flows(compiled: Any, request: Any, basis: Any) -> tuple[Flow, ...]:
    """The leading-colour flows an LC request's artifact computes, in the tree
    runtime's order, which numbers them: ``plan_flows`` of the tree engine's
    plan of the process built with the selection's ``color_order`` and
    ``max_sectors`` (``reference_color_order``, ``max_color_sectors``), the
    selected flows only under a colour-flow selection (the tree runtime of
    ``selected_color_sector_ids`` lists and numbers the flows it computes).
    NLC and full colour have none."""

    if request.accuracy != "lc":
        return ()
    selection = request.selection
    plan = leading_colour_plan(
        compiled,
        request.process,
        max_sectors=None if selection is None else selection.max_sectors,
        reference_color_order=None if selection is None else selection.color_order,
    )
    flows = plan_flows(plan, basis)
    if selection is None or selection.flows is None:
        return flows
    kept = {flow_id(word) for word in selection.flows}
    out = tuple(flow for flow in flows if flow.id in kept)
    if len(out) != len(kept):
        missing = sorted(kept - {flow.id for flow in out})
        raise ValueError(
            f"the selected colour flows {missing} are no flows of the tree "
            "engine's leading-colour plan of the request"
        )
    return out


@dataclass(frozen=True)
class Catalogue:
    """The selectors of an LC v2 virtual: its Born helicity rows and flows."""

    rows: tuple[rx.Row, ...]
    flows: tuple[Flow, ...]

    def resolve(
        self,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
    ) -> tuple[frozenset[rx.Row] | None, frozenset[int] | None]:
        """(helicity rows, Born basis elements) of a selection; ``None`` keeps
        every one of that axis. Helicities by tree ID (``h:...``) or runtime
        ID (``h<index>``); flows by ID or ordinal (``"1"`` the first)."""

        rows = None
        if helicities:
            index: dict[str, rx.Row] = {}
            for k, row in enumerate(self.rows):
                index[helicity_id(row)] = row
                index[f"h{k}"] = row
            chosen = []
            for name in helicities:
                row = index.get(str(name).strip())
                if row is None:
                    raise EvaluationError(
                        f"unknown helicity selector {name!r}; choose a helicity "
                        "ID of the process (h:<values in leg order>)"
                    )
                chosen.append(row)
            rows = frozenset(chosen)
        basis = None
        if color_flows:
            by_id = {flow.id: flow for flow in self.flows}
            chosen_basis = set()
            for name in color_flows:
                flow = by_id.get(str(name).strip())
                if flow is None:
                    flow = self._ordinal(str(name))
                if flow.basis is None:
                    raise EvaluationError(
                        f"colour flow {flow.id!r} has no Born amplitude in this process"
                    )
                chosen_basis.add(flow.basis)
            basis = frozenset(chosen_basis)
        return rows, basis

    def _ordinal(self, name: str) -> Flow:
        try:
            ordinal = int(name, 10)
        except ValueError:
            ordinal = None
        if ordinal is None or str(ordinal) != name.strip():
            raise EvaluationError(
                f"unknown colour-flow selector {name!r}; choose a colour-flow "
                "ID of the artifact's flows (flow:<word>; a generation-time "
                "colour selection computes its flows only) or an ordinal"
            )
        if ordinal < 1 or ordinal > len(self.flows):
            raise EvaluationError(
                f"color-flow ordinal {name!r} is out of range; choose "
                f"1..{len(self.flows)} or a stable color component ID"
            )
        return self.flows[ordinal - 1]


_ACCURACY_NAMES = {"nlc": "next-to-leading", "full": "full"}


def require_leading_colour(
    accuracy: str,
    helicities: Iterable[str] | None,
    color_flows: Iterable[str] | None,
    *,
    resolved: bool = False,
) -> None:
    """Helicity and colour-flow selection exists at leading colour only: NLC
    and full colour keep their API (every helicity and colour contracted;
    ``evaluate_resolved`` without selectors per helicity)."""

    if (helicities or color_flows) and accuracy != "lc":
        instead = (
            "evaluate_resolved without selectors returns every helicity row"
            if resolved
            else "use evaluate_resolved for per-helicity outputs"
        )
        raise CompatibilityError(
            "v2 virtual runtimes evaluate every helicity and colour at "
            f"{_ACCURACY_NAMES.get(accuracy, accuracy)} colour; helicity and "
            "colour-flow selection is available at leading colour "
            f"(color.accuracy='lc'); {instead}"
        )


SELECTION = "selection"

# Distinct evaluation-time selections a runtime keeps formed and compiled
# (``SelectionCache``), unless ``$PYAMPLICOL_LOOP_SELECTION_CACHE`` sets
# another count.
SELECTION_CACHE_DEFAULT = 8
SELECTION_CACHE_ENVIRONMENT = "PYAMPLICOL_LOOP_SELECTION_CACHE"


def selection_cache_size() -> int:
    """The count of ``SelectionCache`` entries:
    ``$PYAMPLICOL_LOOP_SELECTION_CACHE`` (an integer >= 0; 0 keeps none) or
    ``SELECTION_CACHE_DEFAULT``."""

    text = os.environ.get(SELECTION_CACHE_ENVIRONMENT, "").strip()
    if not text:
        return SELECTION_CACHE_DEFAULT
    try:
        size = int(text, 10)
    except ValueError:
        size = -1
    if size < 0:
        raise EvaluationError(
            f"{SELECTION_CACHE_ENVIRONMENT} must be an integer >= 0 (the count "
            f"of evaluation-time selections a runtime keeps), not {text!r}"
        )
    return size


class SelectionCache:
    """The column runners of a runtime's evaluation-time selections, formed
    on first use: the ``size`` most recently used are kept
    (``selection_cache_size``), the least recently used dropped first, so
    their native plans and executors are released (module docstring)."""

    def __init__(self, size: int | None = None) -> None:
        self.size = selection_cache_size() if size is None else int(size)
        self._entries: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any, build: Callable[[], Any]) -> Any:
        found = self._entries.get(key)
        if found is not None:
            self._entries.move_to_end(key)
            return found
        found = build()
        if self.size > 0:
            self._entries[key] = found
            while len(self._entries) > self.size:
                self._entries.popitem(last=False)
        return found

    def items(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)


def column(rows: frozenset | None, basis: frozenset | None) -> Any:
    """The observable column of a selection."""

    from .observables import Column

    return Column(
        (SELECTION, _key(rows), None if basis is None else tuple(sorted(basis))),
        None,
        rows,
        basis,
    )


def _key(rows: frozenset | None) -> Any:
    return None if rows is None else tuple(sorted(rows))
