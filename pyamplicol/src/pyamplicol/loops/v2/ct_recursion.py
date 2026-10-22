# SPDX-License-Identifier: 0BSD
"""Counterterms inside the tree recursion: the production form of design §6.1.

The model's counterterm (CT) vertices are lowered into its tree model as
ordinary vertices (``models.loading.lower_counterterm_vertices``), so the tree
engine itself (the Python generic DAG and the Rust recurrence alike) builds
CT = 0 and CT = 1 currents in one recursion. Its roots then hold the Born and
the CT amplitudes in the Born's own colour sectors, and the tree runtime's
squared-coupling-order partials give the Born x CT interference directly; no
CT tree and no site is listed:

* **Pseudo-orders.** Every CT vertex carries ``COUNTERTERM_INSERTION_ORDER``
  once (limit 1), plus the order of its *slot* (component x epsilon power,
  ``models.loading.counterterm_slot_order``). UV_local and R2 therefore stay
  separate down to each Laurent coefficient, and a finite-only evaluation
  never needs a pole slot.
* **Couplings.** A lowered CT vertex term has one complex runtime coupling per
  slot (two real parameters). ``CountertermRecursion.parameter_values`` binds
  it, per evaluation, to the sum over the admitted rules of that term of
  ``(number of admitted alternatives) x (Laurent coefficient)``; the Laurent
  coefficients come from the source scalar program (``loops.v2.fields``), as
  for the sites.
* **Admission** (design §6.2) is a precomputed predicate, never a site list.
  It follows the witness table of ``loops.v2.ct_sites``: per rule, the
  alternatives ``admitted_alternatives`` gives on every divergent witness of
  the rule's field multiset and orders. In QCD this is field level (leg masks
  enter only through LSZ, which the arity >= 3 of lowered rules excludes);
  a rule whose admission differs between witness signatures fails closed.
* **Not lowered yet** (``CountertermRecursion.residual`` with the reason;
  these rules stay site closures, ``loops.v2.ct_sites``, restricted with
  ``closed_rules``): two-point insertions (the unary node of design §6.1),
  colours outside the tree colour flow (d-symbol and δδ products of the
  four-gluon R2: the multi-trace colour state, ``models.ct_color_flow``),
  whole-rule colour continuations and vertices with ghost legs.

The site closure stays the oracle (O6): per component, squared order and
Laurent coefficient, the recursion's partials equal the sites of the lowered
rules (``tests/unit/test_loop_v2_ct_recursion.py``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from pyamplicol.models.ct_color_flow import (
    UnloweredCountertermColour,
    tree_colour_source,
)
from pyamplicol.models.loading import (
    COUNTERTERM_INSERTION_ORDER,
    COUNTERTERM_POWERS,
    CompiledModel,
    CountertermLoweredModel,
    CountertermLoweringError,
    CountertermStructure,
    counterterm_lorentz_error,
    counterterm_slot_order,
    lower_counterterm_vertices,
)

from .ct_sites import (
    COMPONENT_BY_KIND,
    COUNTERTERM_KINDS,
    Signature,
    Species,
    admitted_alternatives,
    divergent_witnesses,
)
from .families import FamilySet, LoopOrderSelection, Orders, add_orders, model_data


class CountertermAdmissionError(NotImplementedError):
    """A rule's admission depends on more than its fields and orders."""


# --------------------------------------------------------------------- structures


def _coefficient_is_zero(entry: Any, power: int) -> bool:
    from symbolica import E

    return entry.coupling.value.coefficient(power).expand() == E("0")


def rule_terms(rule: Any) -> tuple[tuple[Any, CountertermStructure], ...]:
    """(entry, lowered vertex term) of one rule, per entry and non-zero power.

    A Laurent coefficient that vanishes identically in the model's source
    program (e.g. the poles of an R2 coupling) has no slot.
    """

    component = COMPONENT_BY_KIND[rule.kind]
    return tuple(
        (
            entry,
            CountertermStructure(
                particles=tuple(entry.particle_names),
                colour=entry.color,
                lorentz=entry.lorentz_name,
                orders=add_orders(tuple(entry.order)),
                component=component,
                power=power,
            ),
        )
        for entry in rule.entries
        for power in COUNTERTERM_POWERS
        if not _coefficient_is_zero(entry, power)
    )


class TreeVocabulary:
    """What the tree compiler of one compiled model can lower, cached."""

    def __init__(self, compiled: CompiledModel):
        self.compiled = compiled
        self.particles = {
            str(item["name"]): item for item in compiled.model["particles"]
        }
        self._lorentz: dict[str, str | None] = {}

    def residual_reason(self, rule: Any) -> str | None:
        """Why ``rule`` cannot enter the tree recursion yet, or ``None``."""

        if rule.kind not in COUNTERTERM_KINDS:
            return f"unknown counterterm kind {rule.kind!r}"
        if len(rule.particle_pdgs) < 3:
            return "two-point insertion: needs the unary node of design §6.1"
        if rule.whole_rule_continuation is not None:
            return "whole-rule colour continuation"
        for entry in rule.entries:
            records = [self.particles[name] for name in entry.particle_names]
            if any(int(record.get("ghost_number", 0)) for record in records):
                return "ghost leg"
            try:
                tree_colour_source(entry.color, [int(r["color"]) for r in records])
            except UnloweredCountertermColour as error:
                return str(error)
            name = entry.lorentz_name
            if name not in self._lorentz:
                self._lorentz[name] = counterterm_lorentz_error(self.compiled, name)
            if self._lorentz[name] is not None:
                return f"Lorentz structure {name}: {self._lorentz[name]}"
        return None


def lower_model_counterterms(
    compiled: CompiledModel,
    rules: Mapping[str, Any],
    *,
    forbidden: Iterable[int] = (),
) -> CountertermLoweredModel:
    """The tree model of ``compiled`` with every lowerable rule's CT vertices.

    ``rules`` are prepared counterterm rules (``PreparedCountertermModel.rules``)
    of any request; preparing the model without order bounds gives all of
    them, so one lowered model serves every process of the model that forbids
    the same particles (|PDG| codes ``forbidden``, removed from the tree model).
    """

    vocabulary = TreeVocabulary(compiled)
    terms = {
        name: rule_terms(rule)
        for name, rule in rules.items()
        if vocabulary.residual_reason(rule) is None
    }
    # Terms of the same slot fed by the same couplings of the same rules share
    # one lowered coupling.
    contributions: dict[CountertermStructure, set[tuple[str, str]]] = defaultdict(set)
    for name, rows in terms.items():
        for entry, structure in rows:
            contributions[structure].add((name, entry.coupling.name))
    classes = {
        structure: tuple(sorted(rows)) for structure, rows in contributions.items()
    }
    forbidden = {abs(int(pdg)) for pdg in forbidden}
    excluded = {
        str(item["name"])
        for item in compiled.model["particles"]
        if abs(int(item["pdg_code"])) in forbidden
    }
    refused: dict[CountertermStructure, str] = {}
    while True:
        # A rule is lowered whole or not at all.
        structures = {
            structure
            for rows in terms.values()
            if not any(structure in refused for _entry, structure in rows)
            for _entry, structure in rows
        }
        try:
            lowered = lower_counterterm_vertices(
                compiled,
                structures,
                coupling_classes={s: classes[s] for s in structures},
                excluded_particles=excluded,
            )
        except CountertermLoweringError as error:
            if not error.refused or set(error.refused) <= set(refused):
                raise
            refused.update(error.refused)
            continue
        return replace(lowered, refused=refused)


# ---------------------------------------------------------------------- admission


def field_level_admission(
    rules: Mapping[str, Any],
    witnesses: Mapping[Signature, frozenset[Species]],
    native_model: Any,
    selection: LoopOrderSelection,
    *,
    forbidden: Iterable[int] = (),
) -> dict[str, tuple[int, ...]]:
    """Admitted alternatives of each rule, independent of the leg masks.

    The witness signatures of a rule are those with its field multiset and
    orders; every one of them must admit the same alternatives
    (``ct_sites.admitted_alternatives``), otherwise the recursion would need a
    mask-level predicate and ``CountertermAdmissionError`` is raised. UVtree
    rules need no witness. Rules of an unselected correction admit nothing.
    """

    data = model_data(native_model)
    names = {int(entry["pdg_code"]): entry["name"] for entry in data["particles"]}
    forbidden = frozenset(abs(int(pdg)) for pdg in forbidden)
    by_projection: dict[tuple[tuple[str, ...], Orders], list[Signature]] = defaultdict(
        list
    )
    for signature in witnesses:
        by_projection[signature.fields, signature.orders].append(signature)
    out: dict[str, tuple[int, ...]] = {}
    for name, rule in sorted(rules.items()):
        if not (
            rule.perturbation_type is None
            or rule.perturbation_type in selection.corrections
        ):
            out[name] = ()
            continue
        orders = add_orders(rule.order)
        fields = tuple(sorted(names[pdg] for pdg in rule.particle_pdgs))
        if rule.kind == "UVtree":
            out[name] = admitted_alternatives(
                rule, Signature((), orders), witnesses, selection, forbidden
            )
            continue
        found = {
            admitted_alternatives(rule, signature, witnesses, selection, forbidden)
            for signature in by_projection.get((fields, orders), ())
        }
        if len(found) > 1:
            raise CountertermAdmissionError(
                f"counterterm rule {name} admits different alternatives on "
                f"witnesses with the same fields {fields}; the tree recursion "
                "needs a leg-mask admission predicate for it"
            )
        out[name] = found.pop() if found else ()
    return out


# ---------------------------------------------------------------------- recursion


Slot = tuple[str, int]  # (component, epsilon power)


@dataclass(frozen=True)
class CountertermRecursion:
    """The CT rules of one loop request, split between recursion and sites.

    ``lowered`` is the model-level lowered tree model; ``admission`` maps
    every rule in the recursion to its admitted alternatives and ``residual``
    every other rule to the reason it stays a site closure.
    """

    lowered: CountertermLoweredModel
    rules: Mapping[str, Any]
    terms: Mapping[str, tuple[tuple[Any, CountertermStructure], ...]]
    admission: Mapping[str, tuple[int, ...]]
    residual: Mapping[str, str]
    born_orders: tuple[Orders, ...]
    loop_orders: tuple[Orders, ...]

    @classmethod
    def build(
        cls,
        lowered: CountertermLoweredModel,
        rules: Mapping[str, Any],
        families: FamilySet,
        native_model: Any,
        selection: LoopOrderSelection,
        born_orders: Iterable[Orders],
        *,
        forbidden: Iterable[int] = (),
    ) -> CountertermRecursion:
        """Split the request's ``rules`` and admit the lowered ones.

        ``families`` is the request's family set, whose divergent routings are
        the witnesses (``ct_sites.divergent_witnesses``); ``born_orders`` are
        the exact Born amplitude orders.
        """

        vocabulary = TreeVocabulary(lowered.compiled)
        residual: dict[str, str] = {}
        lowered_rules: dict[str, Any] = {}
        terms: dict[str, tuple[tuple[Any, CountertermStructure], ...]] = {}
        for name, rule in rules.items():
            reason = vocabulary.residual_reason(rule)
            if reason is None:
                terms[name] = rule_terms(rule)
                missing = [s for _e, s in terms[name] if s not in lowered.parameters]
                if missing:
                    reason = lowered.refused.get(
                        missing[0], "a vertex term is not in the lowered tree model"
                    )
                    del terms[name]
            if reason is None:
                lowered_rules[name] = rule
            else:
                residual[name] = reason
        witnesses = divergent_witnesses(families, native_model)
        admission = field_level_admission(
            lowered_rules,
            witnesses,
            native_model,
            selection,
            forbidden=forbidden,
        )
        return cls(
            lowered=lowered,
            rules=lowered_rules,
            terms=terms,
            admission=admission,
            residual=residual,
            born_orders=tuple(sorted({add_orders(o) for o in born_orders})),
            loop_orders=tuple(sorted(selection.loop_orders)),
        )

    # -- program -----------------------------------------------------------------

    def coupling_order_limits(self) -> dict[str, int]:
        """Order limits of the tree program: Born and loop orders, CT <= 1.

        Every physical order of the model is bounded (by 0 when no Born or
        loop order carries it), since the tree generator leaves an unlisted
        order unbounded.
        """

        limits: dict[str, int] = {
            str(item["name"]): 0
            for item in self.lowered.compiled.model["orders"]
            if not str(item["name"]).startswith("PYAMPLICOL_CT")
        }
        for orders in (*self.born_orders, *self.loop_orders):
            for name, power in orders:
                limits[name] = max(limits.get(name, 0), int(power))
        limits[COUNTERTERM_INSERTION_ORDER] = 1
        return limits

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(
            sorted(
                {
                    (structure.component, structure.power)
                    for terms in self.terms.values()
                    for _entry, structure in terms
                }
            )
        )

    # -- couplings ---------------------------------------------------------------

    def scalar_outputs(self) -> tuple[tuple[str, int], ...]:
        """Source scalar outputs (CT coupling name, epsilon power) to bind."""

        return tuple(
            sorted(
                {
                    (entry.coupling.name, structure.power)
                    for name, terms in self.terms.items()
                    if self.admission[name]
                    for entry, structure in terms
                }
            )
        )

    def parameter_values(
        self,
        scalars: Mapping[tuple[str, int], Any],
        *,
        poles: bool = True,
        slots: Iterable[Slot] | None = None,
    ) -> dict[str, Any]:
        """Runtime values of the lowered CT parameters at one point.

        ``scalars`` maps ``scalar_outputs()`` to the bound Laurent
        coefficients (any carrier with ``+``, ``*`` by int, ``.real`` and
        ``.imag``). Without ``poles`` the pole slots are zero, and with
        ``slots`` only those (component, power) slots are bound. Every lowered
        parameter is returned; terms of rules this request does not admit
        are zero.
        """

        selected = None if slots is None else frozenset(slots)

        values: dict[CountertermStructure, Any] = {}
        for name, terms in self.terms.items():
            multiplicity = len(self.admission[name])
            for entry, structure in terms:
                if not multiplicity or (structure.power and not poles):
                    continue
                if selected is not None and (
                    (structure.component, structure.power) not in selected
                ):
                    continue
                term = multiplicity * scalars[entry.coupling.name, structure.power]
                known = values.get(structure)
                values[structure] = term if known is None else known + term
        out: dict[str, Any] = {}
        bound: dict[tuple[str, str], Any] = {}
        for structure, names in self.lowered.parameters.items():
            value = values.get(structure)
            if names in bound and bound[names] != value:
                raise ValueError(
                    f"lowered CT coupling {names} is shared by vertex terms whose "
                    "couplings differ in this request"
                )
            bound[names] = value
            real, imaginary = names
            out[real] = 0 if value is None else value.real
            out[imaginary] = 0 if value is None else value.imag
        return out

    # -- partials ----------------------------------------------------------------

    def components(self, evaluation: Any) -> dict[tuple[str, Orders, int], tuple]:
        """Born x CT interference per (component, squared order, power).

        ``evaluation`` is the tree runtime's ``TreeOrderEvaluation`` of the
        CT program; its squared orders with one insertion are 2 Re of the Born
        times the CT amplitude of one slot (the pseudo-orders are removed from
        the returned squared order). Every amplitude-order pair of such a
        squared order must pair a Born order with a loop order of the slot,
        otherwise an unrequested order would leak into the sector and the
        split fails closed.
        """

        slots = {
            counterterm_slot_order(component, power): (component, power)
            for component, power in self.slots
        }
        born = set(self.born_orders)
        loop = set(self.loop_orders)
        out: dict[tuple[str, Orders, int], tuple] = {}
        for index, sector in enumerate(evaluation.sectors):
            orders = dict(sector.order)
            if orders.get(COUNTERTERM_INSERTION_ORDER) != 1:
                continue
            (slot,) = (name for name in orders if name in slots)
            for left, right in sector.pairs:
                tree, ct = (left, right) if _insertions(right) else (right, left)
                physical = _physical(ct)
                if _insertions(tree) or add_orders(tree) not in born:
                    raise ValueError(
                        f"CT sector {sector.order} pairs an amplitude of order "
                        f"{tree} outside the Born orders"
                    )
                if physical not in loop:
                    raise ValueError(
                        f"CT sector {sector.order} pairs a CT amplitude of order "
                        f"{physical} outside the loop orders"
                    )
            component, power = slots[slot]
            out[component, _physical(sector.order), power] = tuple(
                point[index] for point in evaluation.values
            )
        return out


def _insertions(orders: Iterable[tuple[str, int]]) -> int:
    return dict(orders).get(COUNTERTERM_INSERTION_ORDER, 0)


def _physical(orders: Iterable[tuple[str, int]]) -> Orders:
    return add_orders(
        tuple(
            (name, power)
            for name, power in orders
            if not name.startswith("PYAMPLICOL_CT_")
        )
    )


__all__ = [
    "CountertermAdmissionError",
    "CountertermRecursion",
    "TreeVocabulary",
    "field_level_admission",
    "lower_model_counterterms",
    "rule_terms",
]
