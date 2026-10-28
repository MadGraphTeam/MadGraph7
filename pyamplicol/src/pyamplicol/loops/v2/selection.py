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
  trace reflections folded as the tree engine folds them, each folded
  sector's reflection a flow of its own), the flows sorted by ID and numbered
  from 1 in that order (the tree runtime's ordinals).

A flow is the Born basis element (``colour.BornBasis``) with the same outer
flow (``colour.sector_flow``); Born basis elements of no leading-colour flow
(U(1) and Fierz pieces) have no leading-colour contribution. The selected
output is the observable column of those rows and basis elements
(``observables.Column``): its seeds and its Born, counterterm and field bras
keep them only, so a complete set of selections sums to the inclusive value.
Next-to-leading and full colour expose no selection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    blocks keep the selected helicity rows only (``restrict_rows``)."""

    helicities: tuple[tuple[int, int], ...] = ()
    flows: tuple[tuple[int, ...], ...] | None = None

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

    def record(self) -> dict[str, Any]:
        return {
            "helicities": [list(pair) for pair in self.helicities],
            "flows": None if self.flows is None else [list(w) for w in self.flows],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Selection:
        flows = record["flows"]
        return cls(
            tuple((int(a), int(b)) for a, b in record["helicities"]),
            None if flows is None else tuple(tuple(int(x) for x in w) for w in flows),
        )


def restrict_born_terms(
    born_terms: Mapping[Any, tuple[Any, ...]],
    legs: Sequence[Any],
    flows: Sequence[Sequence[int]],
) -> dict[Any, tuple[Any, ...]]:
    """The Born closure's terms with the colour components of the selected
    flows only (``Selection.flows``, tree words; ``colour.outer_flow`` keys):
    the Born basis (``assemble._born_tensors``) is then those flows' tensors.

    At leading colour this is the selection's bra: distinct flows do not
    interfere at the leading power, nor does a flow with a Born tensor of no
    flow (U(1) and Fierz pieces), so the Born of the kept terms is the
    selected flows' and the seeds, counterterm and field bras keep them only.
    A selected flow without a Born tensor is refused, as is a selection
    that keeps none of the closure."""

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
    nothing), whether the tree engine computes it (a folded reflection is
    served by its sector's ``representative``) and its plan sector id."""

    id: str
    word: tuple[int, ...]
    basis: int | None
    computed: bool
    representative: str
    sector: int

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "word": list(self.word),
            "basis": self.basis,
            "computed": self.computed,
            "representative": self.representative,
            "sector": self.sector,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Flow:
        return cls(
            str(record["id"]),
            tuple(int(x) for x in record["word"]),
            None if record["basis"] is None else int(record["basis"]),
            bool(record["computed"]),
            str(record["representative"]),
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
    """The physical flows of a leading-colour plan, sorted by ID (the tree
    runtime's ``ColorFlow`` order), each with its element of ``basis``."""

    from .colour import _canonical_rotation, sector_flow

    elements: dict[Any, int] = {}
    for b, tensor in enumerate(basis.tensors):
        elements.setdefault(_tensor_flow(tensor), b)
    found: dict[str, Flow] = {}
    for sector in plan.sectors:
        word = tuple(int(x) for x in sector.color_words[0])
        identifier = flow_id(word)
        found.setdefault(
            identifier,
            Flow(
                identifier,
                word,
                elements.get(sector_flow(sector)),
                True,
                identifier,
                int(sector.id),
            ),
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
            Flow(
                other, reflected, elements.get(key), False, identifier, int(sector.id)
            ),
        )
    return tuple(found[key] for key in sorted(found))


def slice_flows(slice_: Any) -> tuple[Flow, ...]:
    """The complete leading-colour flows of an LC slice (``plan_flows`` of the
    tree engine's complete plan of its process on its Born basis)."""

    if slice_.request.accuracy != "lc":
        return ()
    plan = leading_colour_plan(slice_.compiled, slice_.request.process)
    return plan_flows(plan, slice_.basis)


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
                if flow.basis is not None:
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
                "ID of the process (flow:<word>) or an ordinal"
            )
        if ordinal < 1 or ordinal > len(self.flows):
            raise EvaluationError(
                f"color-flow ordinal {name!r} is out of range; choose "
                f"1..{len(self.flows)} or a stable color component ID"
            )
        return self.flows[ordinal - 1]


def require_leading_colour(
    accuracy: str, helicities: Iterable[str] | None, color_flows: Iterable[str] | None
) -> None:
    """Helicity and colour-flow selection exists at leading colour only: NLC
    and full colour keep their API (every helicity and colour contracted)."""

    if (helicities or color_flows) and accuracy != "lc":
        raise CompatibilityError(
            "v2 virtual runtimes evaluate every helicity and colour at "
            f"{accuracy} colour; helicity and colour-flow selection is "
            "available at leading colour (color.accuracy='lc'); use "
            "evaluate_resolved for per-helicity outputs"
        )


SELECTION = "selection"


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
