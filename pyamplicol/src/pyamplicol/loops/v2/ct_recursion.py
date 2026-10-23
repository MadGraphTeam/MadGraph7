# SPDX-License-Identifier: 0BSD
"""Counterterms inside the tree recursion: the production form of design §6.1.

The model's counterterm (CT) vertices are lowered into its tree model as
ordinary vertices (``models.loading.lower_counterterm_vertices``), and its
two-point CTs as clone lines (``models.ct_line_insertions``: a clone of the
field with the propagator ``P Pi P`` and copies of the field's vertices), so
the tree engine itself (the Python generic DAG and the Rust recurrence alike)
builds the currents without and with one insertion in one recursion. Its
roots then hold the Born and the CT amplitudes in the Born's own colour
sectors, and the tree runtime's squared-coupling-order partials give the
Born x CT interference directly; no CT tree and no site is listed:

* **Pseudo-orders.** One insertion carries ``COUNTERTERM_INSERTION_ORDER``
  and the order of its *slot* (component x epsilon power,
  ``models.loading.counterterm_slot_order``) twice each: a CT vertex itself,
  a clone line once on each of its two vertex copies. The insertion limit
  is 2. UV_local and R2 therefore stay separate down to each Laurent
  coefficient.
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
  enter only through LSZ, which clone lines exclude structurally: a clone is
  never external); a rule whose admission differs between witness
  signatures fails closed.
* **Not lowered** (``residual`` with the reason; these rules stay site
  closures, ``loops.v2.ct_sites``, restricted with ``closed_rules``):
  Lorentz structures the tree compiler does not normalise (Levi-Civita
  cells that do not cancel), contact vertices without a colour-flow
  decomposition, the two-point terms ``models.ct_line_insertions`` refuses
  (mixing between spins), ghost legs and coefficients with UFO helpers other
  than ``cond`` and ``reglog``; no loop_sm rule is among them. With
  ``multi_trace`` (the production closure, ``ct_closure``) the d-symbol, d·d,
  d·f and δδ colours of the four-gluon and gggV R2 (``models.ct_color_flow``)
  and the whole-rule colour continuations (``whole_rule_rows``) are lowered;
  without it (the tree-route form kept as a test of the lowering) they stay
  residual, as the tree runtime has no multi-trace colour sectors.
* **Cancelling cells.** A colour or Lorentz cell of a UFO vertex the tree
  compiler cannot lower does not make its rule residual when it cancels
  exactly between the rules of that cell (the loop groups of one vertex):
  the sum over those rules of their admitted alternatives times their
  Laurent coefficients vanishes identically (``ct_sites.vanishes_identically``),
  e.g. the Levi-Civita R2 of a Z with gluons, whose axial couplings cancel
  between the up- and down-type quark loops of complete doublets. The
  lowered model leaves the cell out (``CountertermModel.unlowered`` keeps its
  coefficients; candidates are the cells that cancel with every alternative
  admitted), and each request checks the cancellation again with its own
  admission (``CountertermRecursion.cancelled``); where it fails, every rule
  of the cell is residual for that request.

The site closure stays the oracle (O6): per component, squared order and
Laurent coefficient, the recursion's partials equal the sites of the lowered
rules (``tests/unit/test_loop_v2_ct_recursion.py``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
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
    COUNTERTERM_INSERTION_WEIGHT,
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
    Cell,
    Signature,
    Species,
    admitted_alternatives,
    divergent_witnesses,
    entry_cell,
    vanishes_identically,
)
from .families import FamilySet, LoopOrderSelection, Orders, add_orders, model_data

Slot = tuple[str, int, int]  # (component, epsilon power, N_c grade)
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
    """What the tree compiler of one compiled model can lower, cached.

    ``multi_trace`` admits the colours whose currents carry closed traces
    (``models.ct_color_flow``), which only loop-request blocks carry.
    """

    def __init__(
        self,
        compiled: CompiledModel,
        *,
        multi_trace: bool = False,
        whole_rules: bool = False,
    ):
        self.compiled = compiled
        self.multi_trace = multi_trace
        self.whole_rules = whole_rules
        self.particles = {
            str(item["name"]): item for item in compiled.model["particles"]
        }
        self._lorentz: dict[str, str | None] = {}

    def residual_reason(self, rule: Any) -> str | None:
        """Why ``rule`` cannot enter the tree recursion yet, or ``None``."""

        reason = self.rule_reason(rule)
        if reason is not None or rule.whole_rule_continuation is not None:
            return reason
        for entry in rule.entries:
            reason = self.entry_reason(entry)
            if reason is not None:
                return reason
        return None

    def rule_reason(self, rule: Any) -> str | None:
        """Why no entry of ``rule`` can enter the tree recursion, or ``None``."""

        if rule.kind not in COUNTERTERM_KINDS:
            return f"unknown counterterm kind {rule.kind!r}"
        if len(rule.particle_pdgs) < 2:
            return "one-point counterterm"
        if rule.whole_rule_continuation is not None:
            if not (self.whole_rules and self.multi_trace):
                return "whole-rule colour continuation"
            return None  # the replacement terms are lowered, not the entries
        for entry in rule.entries:
            records = [self.particles[name] for name in entry.particle_names]
            if any(int(record.get("ghost_number", 0)) for record in records):
                return "ghost leg"
        return None

    def entry_reason(self, entry: Any) -> str | None:
        """Why the colour or Lorentz cell of ``entry`` cannot be lowered."""

        records = [self.particles[name] for name in entry.particle_names]
        try:
            tree_colour_source(
                entry.color,
                [int(r["color"]) for r in records],
                multi_trace=self.multi_trace,
            )
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

    def __init__(
        self,
        compiled: CompiledModel,
        fixed_parameters: Mapping[str, Any],
        identities: Mapping[str, Any] | None = None,
    ):
        from symbolica import E

        from pyamplicol._internal.physics.symbols import symbols

        if compiled.raw_loop_model is None:
            raise ValueError("counterterm coefficients need the model's NLO sidecar")
        bindings = compiled.raw_loop_model.metadata.get("function_bindings", {})
        self._helpers = {int(record): str(name) for name, record in bindings.items()}
        identities = dict(identities or {})
        constants = {
            name: value
            for name, value in _numeric_constants(compiled.raw_loop_model).items()
            if name not in identities
        }
        # A formal-N_c continuation's colour invariants (CA -> N_c, ...)
        # replace their numeric values (design §6.7).
        self._fixed = [
            (E(f"UFO::{{}}::{name}"), value)
            for name, value in sorted(identities.items())
        ] + [
            (E(f"UFO::{{}}::{name}"), _number(value))
            for name, value in sorted({**constants, **fixed_parameters}.items())
        ]
        self._zero = E("0")
        self._log, self._conj = symbols.qualified("symbolica::log", "symbolica::conj")
        self.assumptions: set[str] = set()

    def __call__(self, entry: Any, power: int) -> Any:
        expression = entry.coupling.value.coefficient(power)
        for symbol, value in self._fixed:
            expression = expression.replace(symbol, value)
        return self._resolve(expression).expand()

    def payload_expression(self, expression: Any) -> Any:
        """An expression in source (``UFO::name``) symbols, e.g. a whole-rule
        coefficient, as a coefficient in the payload's symbols."""

        expression = _payload_symbols(expression)
        for symbol, value in self._fixed:
            expression = expression.replace(symbol, value)
        return self._resolve(expression).expand()

    def _resolve(self, atom: Any) -> Any:
        from symbolica import AtomType

        from pyamplicol._internal.physics.symbols import symbols

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
            return symbols.qualified(name)(*children)
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
    left to site closure to the reason; ``structures`` are the lowered vertex
    terms of every lowered rule (graded with a continuation); ``unlowered``
    holds, per lowered rule, the cells the lowered model leaves out because
    they cancel between rules (module docstring): cell -> (why it is not
    lowered, {Laurent power: exact coefficient}).
    """

    lowered: CountertermLoweredModel
    multiplicities: Mapping[RuleKey, str]
    switches: Mapping[Slot, str]
    assumptions: tuple[str, ...]
    residual: Mapping[RuleKey, str]
    structures: Mapping[RuleKey, tuple[CountertermStructure, ...]] = field(
        default_factory=dict
    )
    unlowered: Mapping[RuleKey, Mapping[Cell, tuple[str, Mapping[int, Any]]]] = field(
        default_factory=dict
    )

    def check_parameters(self, values: Mapping[str, Any]) -> None:
        """Fail if runtime model parameter ``values`` break an assumption.

        Parameters absent from ``values`` take the compiled defaults.
        """

        from symbolica import AtomType, E, Expression, Replacement

        if not self.assumptions:
            return
        defaults = {
            name: complex(*pair)
            for name, pair in self.lowered.compiled.parameter_defaults.items()
        }
        defaults.update(values)
        replacements = [
            Replacement(
                E(f"UFO::{{}}::{name}"),
                E(repr(complex(value).real))
                + E(repr(complex(value).imag)) * Expression.I,
            )
            for name, value in defaults.items()
        ]
        for assumption in self.assumptions:
            expression = E(assumption).replace_multiple(replacements).expand()
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
    multi_trace: bool = False,
    continuation: Any = None,
    dirac_fields: Iterable[str] = (),
    order_limits: Mapping[str, int] | None = None,
) -> CountertermModel:
    """The tree model of ``compiled`` with every lowerable rule's CT vertices.

    ``rules`` are prepared counterterm rules (``PreparedCountertermModel.rules``)
    of any request and ``fixed_parameters`` the exact fixed parameter values of
    that preparation (``structure.fixed_parameters``); preparing the model
    without order bounds gives every rule, so one lowered model serves every
    process of the model that forbids the same particles (|PDG| codes
    ``forbidden``, removed from the tree model).

    ``multi_trace`` lowers the colours whose currents carry closed traces
    (the four-gluon R2; ``models.ct_color_flow``): the model then serves the
    counterterm closure's loop-request blocks (``loops.v2.ct_closure``), not
    complete tree programs. The U(N) contact split of a ``d*d`` colour adds
    ``(2/N_c) delta delta``, which is subtracted from the rule exactly
    (``d_product_u1_term``; N_c = 3, the UFO's fixed-SU(3) colours).

    ``continuation`` (a ``SourceColorContinuation``) grades the couplings by
    N_c (design §6.7): its colour invariants enter the coefficients as
    functions of a formal N_c, every vertex term is split into its N_c
    powers (``CountertermStructure.grade``, one slot each), and the U(1) term
    of a ``d*d`` split is ``2/N_c``.

    ``dirac_fields`` (UFO names) are Weyl fields the lowered model propagates
    as Dirac fields regardless of its two-point terms
    (``lower_counterterm_vertices``).

    ``order_limits`` (physical coupling order -> largest power) bound the
    tree program the model serves (``CountertermRecursion.coupling_order_limits``
    of a request): the model then keeps only the vertices and rules within
    them, and a rule above them stays residual. ``None`` lowers the whole
    model.
    """

    dirac_fields = frozenset(dirac_fields)
    vocabulary = TreeVocabulary(
        compiled, multi_trace=multi_trace, whole_rules=continuation is not None
    )
    nc = None if continuation is None else continuation.nc
    coefficients = CoefficientLowering(
        compiled,
        fixed_parameters or {},
        None if continuation is None else _payload_identities(continuation),
    )
    residual: dict[RuleKey, str] = {}
    # Exact coefficient of every rule on each of its lowered vertex terms.
    values: dict[RuleKey, dict[CountertermStructure, Any]] = {}
    # Cells the tree compiler cannot lower, per rule: (reason, coefficients).
    unlowered: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]] = {}
    weights: dict[RuleKey, int] = {}
    for rule in rules.values():
        key = rule_key(rule)
        reason = vocabulary.rule_reason(rule)
        if reason is None and order_limits is not None:
            beyond = [
                orders
                for orders in {add_orders(tuple(e.order)) for e in rule.entries}
                if any(power > order_limits.get(name, power) for name, power in orders)
            ]
            if beyond:
                reason = f"its orders {beyond[0]} exceed the program's limits"
        if reason is not None:
            residual[key] = reason
            continue
        rows: dict[CountertermStructure, Any] = {}
        cells: dict[Cell, tuple[str, dict[int, Any]]] = {}
        try:
            if rule.whole_rule_continuation is not None:
                rows = whole_rule_rows(rule, coefficients)
            else:
                for entry in rule.entries:
                    reason = vocabulary.entry_reason(entry)
                    if reason is None:
                        continue
                    powers = cells.setdefault(entry_cell(entry), (reason, {}))[1]
                    for power in COUNTERTERM_POWERS:
                        if _coefficient_is_zero(entry, power):
                            continue
                        value = coefficients(entry, power)
                        powers[power] = (
                            powers[power] + value if power in powers else value
                        )
            for entry, structure in (
                rule_terms(rule) if rule.whole_rule_continuation is None else ()
            ):
                if entry_cell(entry) in cells:
                    continue
                value = coefficients(entry, structure.power)
                rows[structure] = (
                    rows[structure] + value if structure in rows else value
                )
        except (CountertermCouplingError, NotImplementedError) as error:
            residual[key] = str(error)
            continue
        if cells:
            unlowered[key] = cells
            weights[key] = len(rule.alternatives)
        if multi_trace:
            rows = _without_u1_terms(rows, nc)
        if nc is not None:
            try:
                rows = _graded(rows, nc)
            except (ValueError, NotImplementedError) as error:
                residual[key] = f"N_c grading of the coupling failed: {error}"
                continue
        # A coefficient the fixed parameters make vanish has no vertex term.
        values[key] = {s: value for s, value in rows.items() if not _is_zero(value)}

    def vanishes(expression: Any) -> bool:
        return vanishes_identically(compiled, expression)

    # Candidate cells cancel with every alternative of their rules admitted;
    # the other rules with an unlowered cell stay residual (all of a cell's
    # rules together, so that no share of a cell is left to sites alone).
    for key in uncancelled_rules(unlowered, weights, vanishes, frozenset(residual)):
        if key in values:
            del values[key]
            residual[key] = _uncancelled_reason(unlowered[key])
    unlowered = {key: cells for key, cells in unlowered.items() if key in values}
    multiplicities = {
        key: f"{_PREFIX}M{index}" for index, key in enumerate(sorted(values, key=repr))
    }
    switches = {
        slot: _switch_name(slot)
        for slot in sorted(
            {(s.component, s.power, s.grade) for rows in values.values() for s in rows}
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
                multi_trace=multi_trace,
                dirac_fields=dirac_fields,
                order_limits=order_limits,
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
            # The other rules of a refused rule's cancelling cells follow it.
            for key in uncancelled_rules(
                unlowered, weights, vanishes, frozenset(residual)
            ):
                if key in values:
                    del values[key]
                    residual[key] = _uncancelled_reason(unlowered[key])
                    del multiplicities[key]
            unlowered = {k: cells for k, cells in unlowered.items() if k in values}
            continue
        return CountertermModel(
            lowered=replace(lowered, refused=refused),
            multiplicities=multiplicities,
            switches=switches,
            assumptions=tuple(sorted(coefficients.assumptions)),
            residual=residual,
            structures={key: tuple(sorted(rows)) for key, rows in values.items()},
            unlowered=unlowered,
        )


def uncancelled_rules(
    unlowered: Mapping[Any, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    weights: Mapping[Any, int],
    vanishes: Any,
    residual: frozenset = frozenset(),
) -> frozenset:
    """Rules (keys of ``unlowered``) whose unlowered cells do not cancel.

    A cell cancels when the sum over its rules of ``weights`` (admitted
    alternatives) times their coefficients ``vanishes`` at every Laurent
    power, and none of its rules is ``residual`` (a residual rule's share of
    the cell is closed at sites). Every rule of a cell that does not cancel is
    returned, to a fixed point.
    """

    members: dict[Cell, list[Any]] = defaultdict(list)
    for key, cells in unlowered.items():
        for cell in cells:
            members[cell].append(key)
    cancels: dict[Cell, bool] = {}

    def cancelled(cell: Cell) -> bool:
        if cell not in cancels:
            keys = members[cell]
            cancels[cell] = all(
                vanishes(
                    reduce(
                        add,
                        (
                            weights[key] * unlowered[key][cell][1][power]
                            for key in keys
                            if power in unlowered[key][cell][1]
                        ),
                    )
                )
                for power in COUNTERTERM_POWERS
                if any(power in unlowered[key][cell][1] for key in keys)
            )
        return cancels[cell]

    out = {key for key in unlowered if key in residual}
    changed = True
    while changed:
        changed = False
        for cell, keys in members.items():
            if all(key in out for key in keys):
                continue
            if any(key in out for key in keys) or not cancelled(cell):
                out.update(keys)
                changed = True
    return frozenset(out)


def _uncancelled_reason(cells: Mapping[Cell, tuple[str, Any]]) -> str:
    reason = next(iter(cells.values()))[0]
    return f"{reason} (it does not cancel between the rules of its vertex)"


def _without_u1_terms(
    rows: Mapping[CountertermStructure, Any], nc: Any = None
) -> dict[CountertermStructure, Any]:
    """Subtract the U(1) term of every ``d*d`` colour's U(N) contact split.

    ``(2/N_c) delta delta`` times the coefficient: with N_c = 3 for UFO
    colours (those of SU(3)), with the formal ``nc`` of a continuation
    (``models.ct_color_flow.d_product_u1_term``).
    """

    from pyamplicol.color.contraction_types import NC
    from pyamplicol.models.ct_color_flow import d_product_u1_term

    factor = _number((f"-2/{NC}", "0")) if nc is None else -2 / nc
    out = dict(rows)
    for structure, value in rows.items():
        colour = d_product_u1_term(structure.colour)
        if colour is None:
            continue
        term = replace(structure, colour=colour)
        correction = value * factor
        out[term] = out[term] + correction if term in out else correction
    return out


def _graded(
    rows: Mapping[CountertermStructure, Any], nc: Any
) -> dict[CountertermStructure, Any]:
    """Every vertex term split into its N_c powers (``grade``)."""

    from pyamplicol.loops.physical_color import split_color_grades

    out: dict[CountertermStructure, Any] = {}
    items = sorted(rows.items())
    for degree, values in split_color_grades((value for _s, value in items), nc):
        for (structure, _value), value in zip(items, values, strict=True):
            if _is_zero(value):
                continue
            term = replace(structure, grade=int(degree))
            out[term] = out[term] + value if term in out else value
    return out


def whole_rule_rows(rule: Any, coefficients: CoefficientLowering) -> dict:
    """Vertex terms of a whole-rule colour continuation (design §6.1, §6.7).

    The replacement of a fixed-SU(3) rule (the general-N_c four-gluon R2,
    ``loops.qcd_r2_color``) has colour tensors in adjoint traces and Lorentz
    structures that are sums of metric products, with coefficients exact in
    a formal N_c. Each term's colour is reduced to closed traces of its legs
    and written in the SU(N) vocabulary of ``models.ct_color_flow``:
    ``tr(tau^a tau^b) = delta^{ab}`` and ``tr(tau^a tau^b tau^c tau^d) =
    delta^{ab} delta^{cd} / N_c + (d + i f)^{abx} (d + i f)^{cdx} / 2``.
    Returns {structure: coefficient in the payload's symbols and N_c}.
    """

    from fractions import Fraction

    from symbolica import E, Expression
    from symbolica.community import hepkit

    from pyamplicol.color.connections import ColorLeg
    from pyamplicol.models.loading import counterterm_metric_lorentz_name

    from .assemble import _metric_pairings
    from .colour import reduce_network, relabel, spenso_colour_network

    continuation = rule.whole_rule_continuation
    nc = continuation.nc
    particles = None
    rows: dict[CountertermStructure, Any] = {}
    (entry, *_rest) = rule.entries
    particles = tuple(entry.particle_names)
    orders = add_orders(tuple(entry.order))
    for term in continuation.terms:
        kernel = term.kernel
        legs = {
            hepkit.Symbols.half_edge()(half, 1).to_canonical_string(): index + 1
            for index, half in enumerate(kernel.leg_half_edges)
        }
        if len(legs) != 4:
            raise NotImplementedError("whole-rule continuation of a non-quartic rule")

        def rename(label, legs=legs):
            return ("leg", legs[label]) if label in legs else ("dummy", label)

        network = relabel(spenso_colour_network(kernel.color.to_expression()), rename)
        colour_legs = tuple(ColorLeg(index, 8) for index in range(1, 5))
        tensor = reduce_network(
            network, colour_legs, {("leg", i): i for i in range(1, 5)}
        )
        if tensor.normalization_power % 2:
            raise NotImplementedError("whole-rule colour with an odd power of sqrt 2")
        scale = Fraction(1, 2 ** (tensor.normalization_power // 2))
        value = coefficients.payload_expression(term.coefficient)
        for lorentz_scale, pairs in _metric_pairings(
            kernel.lorentz.to_expression(), legs
        ):
            lorentz = counterterm_metric_lorentz_name((a + 1, b + 1) for a, b in pairs)
            for item in tensor.terms:
                if item.tensor.open_strings:
                    raise NotImplementedError("whole-rule colour with open strings")
                number = item.coefficient.scaled(scale * lorentz_scale)
                weight = (
                    E(str(number.real)) + E(str(number.imag)) * Expression.I
                ) * nc**item.nc_power
                for colour, factor in _adjoint_trace_colours(item.tensor.traces, nc):
                    structure = CountertermStructure(
                        particles=particles,
                        colour=colour,
                        lorentz=lorentz,
                        orders=orders,
                        component=COMPONENT_BY_KIND[rule.kind],
                        power=0,
                    )
                    contribution = value * weight * factor
                    rows[structure] = (
                        rows[structure] + contribution
                        if structure in rows
                        else contribution
                    )
    return rows


def _adjoint_trace_colours(traces, nc) -> list[tuple[str, Any]]:
    """UFO colours (SU(N) vocabulary) with factors of a product of traces."""

    from symbolica import E, Expression

    lengths = sorted(len(trace) for trace in traces)
    if lengths == [2, 2]:
        pairs = sorted(tuple(sorted(trace)) for trace in traces)
        return [("*".join(f"Identity({a},{b})" for a, b in pairs), E("1"))]
    if lengths == [4]:
        ((a, b, c, d),) = traces
        half, i = E("1/2"), Expression.I
        # Each factor with its legs ascending: f flips sign, d does not.
        ab, cd = (min(a, b), max(a, b)), (min(c, d), max(c, d))
        sab, scd = (1 if a < b else -1), (1 if c < d else -1)
        return [
            (f"Identity({ab[0]},{ab[1]})*Identity({cd[0]},{cd[1]})", 1 / nc),
            (f"d(-1,{ab[0]},{ab[1]})*d(-1,{cd[0]},{cd[1]})", half),
            (f"d(-1,{ab[0]},{ab[1]})*f(-1,{cd[0]},{cd[1]})", half * i * scd),
            (f"f(-1,{ab[0]},{ab[1]})*d(-1,{cd[0]},{cd[1]})", half * i * sab),
            (f"f(-1,{ab[0]},{ab[1]})*f(-1,{cd[0]},{cd[1]})", -half * sab * scd),
        ]
    raise NotImplementedError(f"whole-rule colour with traces of lengths {lengths}")


def _payload_identities(continuation: Any) -> dict[str, Any]:
    """The continuation's colour invariants in the payload's symbols."""

    return {
        name: _payload_symbols(expression)
        for name, expression in continuation.parameter_identities.items()
    }


def _payload_symbols(expression: Any) -> Any:
    """Source parameter symbols (``UFO::name``) as payload ones (``UFO::{}::``)."""

    from symbolica import E

    for symbol in expression.get_all_symbols(False):
        full = symbol.get_name()
        if full.startswith("UFO::") and not full.startswith("UFO::{}::"):
            expression = expression.replace(
                symbol, E(f"UFO::{{}}::{full.removeprefix('UFO::')}")
            )
    return expression


def _slot_of(structure: CountertermStructure) -> Slot:
    return (structure.component, structure.power, structure.grade)


def _switch_name(slot: Slot) -> str:
    component, power, grade = slot
    name = f"{_PREFIX}S_{component.upper()}_EPS{-power}"
    if grade:
        name += f"_NC{grade}" if grade > 0 else f"_NCM{-grade}"
    return name


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
        structure: (E(f"UFO::{{}}::{switches[_slot_of(structure)]}") * total)
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


@dataclass(frozen=True)
class CountertermRecursion:
    """The CT rules of one loop request, split between recursion and sites.

    ``rules`` are the request's rules in the recursion with their admitted
    alternatives (``admission``) and lowered ``terms``; ``residual`` maps
    every other rule of the request to the reason it stays a site closure;
    ``cancelled`` names the cells of the recursion's rules that the lowered
    model leaves out and that cancel with this request's admission (module
    docstring), which the CT-root sites leave out too.
    """

    model: CountertermModel
    rules: Mapping[str, Any]
    terms: Mapping[str, tuple[tuple[Any, CountertermStructure], ...]]
    admission: Mapping[str, tuple[int, ...]]
    residual: Mapping[str, str]
    born_orders: tuple[Orders, ...]
    loop_orders: tuple[Orders, ...]
    cancelled: Mapping[str, frozenset[Cell]] = field(default_factory=dict)

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
                cells = model.unlowered.get(key, {})
                terms[name] = tuple(
                    (None, structure) for structure in model.structures[key]
                ) or tuple(
                    (entry, structure)
                    for entry, structure in rule_terms(rule)
                    if entry_cell(entry) not in cells
                )
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
        # The cells the lowered model leaves out must cancel with this
        # request's admission; otherwise their rules are closed at sites.
        unlowered = {
            name: model.unlowered[rule_key(rule)]
            for name, rule in lowered.items()
            if rule_key(rule) in model.unlowered
        }
        compiled = model.lowered.compiled
        failed = uncancelled_rules(
            unlowered,
            {name: len(admission[name]) for name in unlowered},
            lambda expression: vanishes_identically(compiled, expression),
        )
        for name in failed:
            residual[name] = _uncancelled_reason(unlowered[name]).replace(
                "its vertex)", "its vertex in this request)"
            )
            del lowered[name], terms[name], admission[name]
        return cls(
            model=model,
            rules=lowered,
            terms=terms,
            admission=admission,
            residual=residual,
            born_orders=tuple(sorted({add_orders(o) for o in born_orders})),
            loop_orders=tuple(sorted(selection.loop_orders)),
            cancelled={
                name: frozenset(cells)
                for name, cells in unlowered.items()
                if name not in failed
            },
        )

    # -- program -----------------------------------------------------------------

    def coupling_order_limits(self) -> dict[str, int]:
        """Order limits of the tree program: Born and loop orders, one insertion.

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
        limits[COUNTERTERM_INSERTION_ORDER] = COUNTERTERM_INSERTION_WEIGHT
        return limits

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(
            sorted(
                {
                    _slot_of(structure)
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
        CT program; its squared orders with one insertion (insertion order
        ``COUNTERTERM_INSERTION_WEIGHT``) are 2 Re of the Born times the CT
        amplitude of one slot (the pseudo-orders are removed from
        the returned squared order). Every amplitude-order pair of such a
        squared order must pair a Born order with a loop order of the slot,
        otherwise an unrequested order would leak into the sector and the
        split fails closed.
        """

        slots = {
            counterterm_slot_order(*slot): slot
            for slot in self.slots
            if not slot[2]  # the tree runtime evaluates at N_c = 3
        }
        born = set(self.born_orders)
        loop = set(self.loop_orders)
        out: dict[tuple[str, Orders, int], tuple] = {}
        for index, sector in enumerate(evaluation.sectors):
            orders = dict(sector.order)
            if orders.get(COUNTERTERM_INSERTION_ORDER) != COUNTERTERM_INSERTION_WEIGHT:
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
            component, power, _grade = slots[slot]
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
    "uncancelled_rules",
]
