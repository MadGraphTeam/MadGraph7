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
  separate down to each Laurent coefficient.
* **Couplings.** The coupling of a lowered CT vertex term is a derived model
  coupling, ``S_slot * sum_rules M_rule * c_rule``: ``c_rule`` is the exact
  Laurent coefficient of the rule's UFO coupling (the source scalar program
  of ``loops.model``, with the model's fixed parameters substituted and the
  UFO helpers ``cond``/``reglog`` resolved, ``CoefficientLowering``),
  ``M_rule`` an integer runtime parameter (the number of admitted
  alternatives of the rule in the process, 0 when not admitted) and
  ``S_slot`` an integer slot switch (0 drops the slot: finite-only mode
  zeroes the pole slots). The tree runtime evaluates the couplings from the
  model parameters in every carrier, so no per-point binding is needed and
  arbitrary precision stays exact.
* **Admission** (design §6.2) is a precomputed predicate, never a site list.
  It follows the witness table of ``loops.v2.ct_sites``: per rule, the
  alternatives ``admitted_alternatives`` gives on every divergent witness of
  the rule's field multiset and orders. In QCD this is field level (leg masks
  enter only through LSZ, which the arity >= 3 of lowered rules excludes);
  a rule whose admission differs between witness signatures fails closed.
* **Not lowered yet** (``residual`` with the reason; these rules stay site
  closures, ``loops.v2.ct_sites``, restricted with ``closed_rules``):
  two-point insertions (the unary node of design §6.1), colours outside the
  tree colour flow (d-symbol and δδ products of the four-gluon R2: the
  multi-trace colour state, ``models.ct_color_flow``), Lorentz structures
  the tree compiler does not normalise (Levi-Civita), contact vertices
  without a colour-flow decomposition, whole-rule colour continuations,
  ghost legs and coefficients with UFO helpers other than ``cond`` and
  ``reglog``.

The site closure stays the oracle (O6): per component, squared order and
Laurent coefficient, the recursion's partials equal the sites of the lowered
rules (``tests/unit/test_loop_v2_ct_recursion.py``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import reduce
from operator import add, mul
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

Slot = tuple[str, int]  # (component, epsilon power)
RuleKey = tuple[object, ...]
_PREFIX = "PYAMPLICOL_CT_"


class CountertermAdmissionError(NotImplementedError):
    """A rule's admission depends on more than its fields and orders."""


class CountertermCouplingError(NotImplementedError):
    """A CT coupling coefficient the tree model cannot express exactly."""


# --------------------------------------------------------------------- structures


def rule_key(rule: Any) -> RuleKey:
    """Identity of a rule across preparations of the same model.

    Rule names are numbered per preparation; kind, entries (UFO vertex,
    colour and Lorentz cell, coupling) and loop-species alternatives are not.
    """

    return (
        rule.kind,
        tuple(
            sorted(
                (e.vertex_name, e.color_index, e.lorentz_index, e.coupling.name)
                for e in rule.entries
            )
        ),
        tuple(tuple(alternative) for alternative in rule.alternatives),
    )


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


# ------------------------------------------------------------------- coefficients


class CoefficientLowering:
    """Laurent coefficients of CT couplings as exact tree-model expressions.

    The coefficients are those of the source scalar program
    (``KernelBasisEntry.coupling``), in the model's UFO parameter symbols, so
    they are valid coupling expressions of the compiled payload once

    * the model's fixed parameters (restriction zeros, ``fixed_parameters`` of
      the preparation) and its internal numeric constants (``CA``, ``TF``,
      ``Ncol``, ``lhv`` in loop_sm, exact from their UFO source literals) are
      substituted; the compiled payload keeps such constants only as binary64
      values, which would cap arbitrary precision at about 1e-16, and
    * the UFO helper calls are resolved: ``cond(c, a, b)`` is ``a`` for a
      vanishing number ``c`` and ``b`` otherwise, ``reglog(z)`` is 0 for a
      vanishing number and ``log(z)`` otherwise, ``complexconjugate`` is
      ``conj``. A symbolic ``c`` or ``z`` is recorded in ``assumptions``
      (expressions that must not vanish at runtime, checked by
      ``CountertermModel.check_parameters``); any other helper raises
      ``CountertermCouplingError``.
    """

    def __init__(self, compiled: CompiledModel, fixed_parameters: Mapping[str, Any]):
        from symbolica import E, S

        if compiled.raw_loop_model is None:
            raise ValueError("counterterm coefficients need the model's NLO sidecar")
        bindings = compiled.raw_loop_model.metadata.get("function_bindings", {})
        self._helpers = {int(record): str(name) for name, record in bindings.items()}
        self._fixed = [
            (E(f"UFO::{{}}::{name}"), _number(value))
            for name, value in sorted(
                {
                    **_numeric_constants(compiled.raw_loop_model),
                    **fixed_parameters,
                }.items()
            )
        ]
        self._zero = E("0")
        self._log = S("symbolica::log")
        self._conj = S("symbolica::conj")
        self.assumptions: set[str] = set()

    def __call__(self, entry: Any, power: int) -> Any:
        expression = entry.coupling.value.coefficient(power)
        for symbol, value in self._fixed:
            expression = expression.replace(symbol, value)
        return self._resolve(expression).expand()

    def _resolve(self, atom: Any) -> Any:
        from symbolica import AtomType, S

        kind = atom.get_type()
        if kind in (AtomType.Num, AtomType.Var):
            return atom
        children = [self._resolve(child) for child in atom]
        if kind == AtomType.Add:
            return reduce(add, children)
        if kind == AtomType.Mul:
            return reduce(mul, children)
        if kind == AtomType.Pow:
            base, exponent = children
            return base**exponent
        name = atom.get_name()
        head = name.rsplit("::", 1)[-1]
        if not (
            name.startswith("pyamplicol_scalar_model_") and head.startswith("value_")
        ):
            return S(name)(*children)
        helper = self._helpers.get(int(head.removeprefix("value_")))
        if helper == "cond":
            condition, if_zero, otherwise = children
            return if_zero if self._vanishes(condition) else otherwise
        if helper == "reglog":
            (argument,) = children
            if self._vanishes(argument):
                return self._zero
            return self._log(argument)
        if helper == "complexconjugate":
            (argument,) = children
            return self._conj(argument)
        raise CountertermCouplingError(
            f"counterterm coefficient uses the UFO helper {helper!r}, which the "
            "tree model has no exact form of"
        )

    def _vanishes(self, value: Any) -> bool:
        from symbolica import AtomType

        value = value.expand()
        if value.get_type() == AtomType.Num:
            return value == self._zero
        self.assumptions.add(value.to_canonical_string())
        return False


def _numeric_constants(raw: Any) -> dict[str, str]:
    """Internal parameters of ``raw`` whose exact source is a number literal."""

    out = {}
    for record in raw.collections["parameters"]:
        fields = record["fields"]
        value = fields.get("value") or {}
        if fields.get("nature") != "internal" or value.get("kind") != "expression":
            continue
        if not (fields.get("value_source") or {}).get("exact", False):
            continue
        number = _exact_literal(str(value["expression"]))
        if number is not None:
            out[str(fields["name"])] = f"{number.numerator}/{number.denominator}"
    return out


def _exact_literal(source: str) -> Fraction | None:
    """The exact rational of an arithmetic expression of number literals."""

    import ast

    def value(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(ast.get_source_segment(source, node) or repr(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            operations = {
                ast.Add: left + right,
                ast.Sub: left - right,
                ast.Mult: left * right,
            }
            if isinstance(node.op, ast.Div) and right:
                return left / right
            if type(node.op) in operations:
                return operations[type(node.op)]
        raise ValueError(source)

    try:
        return value(ast.parse(source, mode="eval"))
    except (SyntaxError, ValueError):
        return None


def _number(value: Any) -> Any:
    from symbolica import E, Expression

    real, imaginary = value if isinstance(value, tuple) else (value, "0")
    return E(str(real)) + E(str(imaginary)) * Expression.I


# ------------------------------------------------------------------------ model


@dataclass(frozen=True)
class CountertermModel:
    """The model-level CT lowering shared by every process of one model.

    ``multiplicities`` names the integer runtime parameter of every lowered
    rule (by ``rule_key``) and ``switches`` that of every slot;
    ``assumptions`` are model-parameter expressions the lowered couplings
    assume non-zero (``CoefficientLowering``); ``residual`` maps every rule
    left to site closure to the reason.
    """

    lowered: CountertermLoweredModel
    multiplicities: Mapping[RuleKey, str]
    switches: Mapping[Slot, str]
    assumptions: tuple[str, ...]
    residual: Mapping[RuleKey, str]

    def check_parameters(self, values: Mapping[str, Any]) -> None:
        """Fail if runtime model parameter ``values`` break an assumption.

        Parameters absent from ``values`` take the compiled defaults.
        """

        from symbolica import AtomType, E, Expression

        defaults = {
            name: complex(*pair)
            for name, pair in self.lowered.compiled.parameter_defaults.items()
        }
        defaults.update(values)
        for assumption in self.assumptions:
            expression = E(assumption)
            for name, value in defaults.items():
                value = complex(value)
                expression = expression.replace(
                    E(f"UFO::{{}}::{name}"),
                    E(repr(value.real)) + E(repr(value.imag)) * Expression.I,
                )
            expression = expression.expand()
            if expression.get_type() != AtomType.Num:
                raise ValueError(f"cannot decide counterterm assumption {assumption}")
            if expression == E("0"):
                raise ValueError(
                    f"model parameters make {assumption} vanish, which the lowered "
                    "counterterm couplings assume non-zero"
                )


def lower_model_counterterms(
    compiled: CompiledModel,
    rules: Mapping[str, Any],
    *,
    fixed_parameters: Mapping[str, Any] | None = None,
    forbidden: Iterable[int] = (),
) -> CountertermModel:
    """The tree model of ``compiled`` with every lowerable rule's CT vertices.

    ``rules`` are prepared counterterm rules (``PreparedCountertermModel.rules``)
    of any request and ``fixed_parameters`` the exact fixed parameter values of
    that preparation (``structure.fixed_parameters``); preparing the model
    without order bounds gives every rule, so one lowered model serves every
    process of the model that forbids the same particles (|PDG| codes
    ``forbidden``, removed from the tree model).
    """

    vocabulary = TreeVocabulary(compiled)
    coefficients = CoefficientLowering(compiled, fixed_parameters or {})
    residual: dict[RuleKey, str] = {}
    # Exact coefficient of every rule on each of its lowered vertex terms.
    values: dict[RuleKey, dict[CountertermStructure, Any]] = {}
    for rule in rules.values():
        key = rule_key(rule)
        reason = vocabulary.residual_reason(rule)
        if reason is not None:
            residual[key] = reason
            continue
        rows: dict[CountertermStructure, Any] = {}
        try:
            for entry, structure in rule_terms(rule):
                value = coefficients(entry, structure.power)
                rows[structure] = (
                    rows[structure] + value if structure in rows else value
                )
        except CountertermCouplingError as error:
            residual[key] = str(error)
            continue
        # A coefficient the fixed parameters make vanish has no vertex term.
        values[key] = {s: value for s, value in rows.items() if not _is_zero(value)}
    multiplicities = {
        key: f"{_PREFIX}M{index}" for index, key in enumerate(sorted(values, key=repr))
    }
    switches = {
        slot: f"{_PREFIX}S_{slot[0].upper()}_EPS{-slot[1]}"
        for slot in sorted(
            {(s.component, s.power) for rows in values.values() for s in rows}
        )
    }
    forbidden = {abs(int(pdg)) for pdg in forbidden}
    excluded = {
        str(item["name"])
        for item in compiled.model["particles"]
        if abs(int(item["pdg_code"])) in forbidden
    }
    parameters = dict.fromkeys((*multiplicities.values(), *switches.values()), 1.0)
    refused: dict[CountertermStructure, str] = {}
    while True:
        try:
            lowered = lower_counterterm_vertices(
                compiled,
                _coupling_expressions(values, multiplicities, switches),
                parameters=parameters,
                excluded_particles=excluded,
            )
        except CountertermLoweringError as error:
            if not error.refused or set(error.refused) <= set(refused):
                raise
            refused.update(error.refused)
            # A rule is lowered whole or not at all.
            for key in [
                k for k, rows in values.items() if rows.keys() & refused.keys()
            ]:
                structure = next(s for s in values.pop(key) if s in refused)
                residual[key] = refused[structure]
                del multiplicities[key]
            continue
        return CountertermModel(
            lowered=replace(lowered, refused=refused),
            multiplicities=multiplicities,
            switches=switches,
            assumptions=tuple(sorted(coefficients.assumptions)),
            residual=residual,
        )


def _coupling_expressions(
    values: Mapping[RuleKey, Mapping[CountertermStructure, Any]],
    multiplicities: Mapping[RuleKey, str],
    switches: Mapping[Slot, str],
) -> dict[CountertermStructure, str]:
    """``S_slot * sum_rules M_rule * c_rule`` of every lowered vertex term."""

    from symbolica import E

    sums: dict[CountertermStructure, Any] = {}
    for key in sorted(values, key=repr):
        multiplicity = E(f"UFO::{{}}::{multiplicities[key]}")
        for structure, value in values[key].items():
            term = multiplicity * value
            sums[structure] = sums[structure] + term if structure in sums else term
    return {
        structure: (
            E(f"UFO::{{}}::{switches[structure.component, structure.power]}") * total
        )
        .expand()
        .to_canonical_string()
        for structure, total in sums.items()
        if not _is_zero(total)
    }


def _is_zero(value: Any) -> bool:
    from symbolica import E

    return value.expand() == E("0")


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
# ---------------------------------------------------------------------- recursion


@dataclass(frozen=True)
class CountertermRecursion:
    """The CT rules of one loop request, split between recursion and sites.

    ``rules`` are the request's rules in the recursion with their admitted
    alternatives (``admission``) and lowered ``terms``; ``residual`` maps
    every other rule of the request to the reason it stays a site closure.
    """

    model: CountertermModel
    rules: Mapping[str, Any]
    terms: Mapping[str, tuple[tuple[Any, CountertermStructure], ...]]
    admission: Mapping[str, tuple[int, ...]]
    residual: Mapping[str, str]
    born_orders: tuple[Orders, ...]
    loop_orders: tuple[Orders, ...]

    @classmethod
    def build(
        cls,
        model: CountertermModel,
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

        residual: dict[str, str] = {}
        lowered: dict[str, Any] = {}
        terms: dict[str, tuple[tuple[Any, CountertermStructure], ...]] = {}
        for name, rule in rules.items():
            key = rule_key(rule)
            if key in model.multiplicities:
                lowered[name] = rule
                terms[name] = rule_terms(rule)
            else:
                residual[name] = model.residual.get(
                    key, "the rule is not in the lowered model"
                )
        admission = field_level_admission(
            lowered,
            divergent_witnesses(families, native_model),
            native_model,
            selection,
            forbidden=forbidden,
        )
        return cls(
            model=model,
            rules=lowered,
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
            for item in self.model.lowered.compiled.model["orders"]
            if not str(item["name"]).startswith(_PREFIX)
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

    def parameter_values(
        self,
        *,
        poles: bool = True,
        slots: Iterable[Slot] | None = None,
        sign: int = 1,
    ) -> dict[str, int]:
        """Runtime values of the CT multiplicities and slot switches.

        Each lowered rule of the model gets its number of admitted
        alternatives in this request (0 for rules of other requests). Slot
        switches are ``sign`` for the selected ``slots`` (default: all) and
        0 otherwise; without ``poles`` the pole slots are 0 (finite-only
        mode). The values are integers, exact in every carrier; they are set
        once per process and do not depend on the phase-space point.
        """

        values = dict.fromkeys(self.model.multiplicities.values(), 0)
        for name, rule in self.rules.items():
            values[self.model.multiplicities[rule_key(rule)]] = len(
                self.admission[name]
            )
        selected = None if slots is None else frozenset(slots)
        for slot, switch in self.model.switches.items():
            on = (selected is None or slot in selected) and (poles or not slot[1])
            values[switch] = sign if on else 0
        return values

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
    "CoefficientLowering",
    "CountertermAdmissionError",
    "CountertermCouplingError",
    "CountertermModel",
    "CountertermRecursion",
    "TreeVocabulary",
    "field_level_admission",
    "lower_model_counterterms",
    "rule_key",
    "rule_terms",
]
