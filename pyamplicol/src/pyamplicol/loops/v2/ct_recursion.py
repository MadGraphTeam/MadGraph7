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
  UFO-standard helpers ``cond``/``reglog``/``complexconjugate`` resolved,
  ``CoefficientLowering``), ``M_rule`` an integer runtime parameter (the
  number of admitted alternatives of the rule in the process, 0 when not
  admitted) and ``S_slot`` an integer slot switch (0 drops the slot:
  finite-only mode zeroes the pole slots). The tree runtime evaluates the
  couplings from the model parameters in every carrier, so arbitrary
  precision stays exact. A coupling that reads a CT scalar whose body needs
  the model's scalar library (the electroweak helpers, a rebound ``reglog``;
  WP3.2) is lowered on *reference parameters*, the real and imaginary parts
  of its own Laurent coefficients, which the CT program binds at the card
  from its exact source and its CT scalars' bodies (``coefficient_plan``,
  ``CountertermRecursion.parameter_values``): no CT body and none of the
  coupling's model parameters is inlined into a vertex, so the tree
  compiler's kernel proofs see one symbol per coefficient (EWI-3R).
* **Admission** (design §6.2) is the plain tree-level rule, without any
  matching to the loop: a rule of a selected correction enters with every
  loop-content alternative (``rule.alternatives``, the UFO ``loop_particles``)
  that holds no vetoed particle (``particle_veto``, |PDG| codes), and the
  exactly-once constraint is the insertion order. ``M_rule`` is the number
  of those alternatives (``admitted_alternatives``).
* **Not lowered** (``residual`` with the reason): Lorentz structures the
  tree compiler does not normalise,
  contact vertices without a colour-flow decomposition, the two-point terms
  ``models.ct_line_insertions`` refuses (mixing between spins), couplings
  of left-out cells (or graded by a formal N_c) that are not linear in their
  referenced CT scalars (``CoefficientLowering.exact``), and coefficients that
  are not finite at the fixed parameters (an infinity or indeterminate in a
  taken branch); no loop_sm rule is among them. A request that admits such
  a rule is refused (``UnloweredCountertermError``) when the rule's vertex
  can enter a tree of it (``ct_closure.InsertionReachability``, the tree
  engine's field and coupling-order reachability); there is no site
  fallback. Rules that can be in no tree of the program (ghost legs; orders
  above its limits; a vertex no tree of the request holds,
  REQUIREMENTS_ADDENDUM J) or have no admitted loop content (vetoed, or no
  loop of a selected correction, ``admitted_alternatives``) are ``inert``
  and contribute nothing. With
  ``multi_trace`` (the production closure, ``ct_closure``) the d-symbol, d·d,
  d·f and δδ colours of the four-gluon and gggV R2 (``models.ct_color_flow``)
  and the whole-rule colour continuations (``whole_rule_rows``) are lowered;
  without it (the tree-route form kept as a test of the lowering) they stay
  residual, as the tree runtime has no multi-trace colour sectors.
* **Cancelling cells.** A colour or Lorentz cell of a UFO vertex the tree
  compiler cannot lower does not make its rule residual when it cancels
  exactly between the rules of that cell (the loop groups of one vertex):
  the sum over those rules of ``M_rule`` times their Laurent coefficients
  vanishes identically (``ct_sites.vanishes_identically``). The lowered model
  leaves the cell out (``CountertermModel.unlowered``); one that does not
  cancel makes the cell's rules residual. Levi-Civita cells
  (``TreeVocabulary.levi_civita``: the Levi-Civita R2 of a Z with gluons,
  whose axial couplings cancel between the up- and down-type quark loops of
  complete doublets) are left out where they cancel, so loop_sm's lowered
  models keep their bytes, and lowered where they do not (an incomplete
  doublet, an anomalous model): the tree compiler lowers ``Epsilon``
  (``models.tensors.LEVI_CIVITA``, WP3.2 EWI-7b; ``_CellSettlement``).
* **Anomalous vetoes** ([U2], the user's decision: refuse by default with a
  recorded override). A Levi-Civita cell that does not cancel under the
  request's ``particle_veto`` but cancels with every flavour admitted (the
  same cancellation test with the multiplicities of no veto, the vetoed-out
  rules included: a veto that breaks a doublet, e.g. ``[d, s, b]`` or
  ``[t]`` with a Z and gluons) makes its rules residual with a reason that
  names the cell and the override (``CountertermModel.anomalous_veto``);
  a request reaching one is refused with ``AnomalousVetoError``. With
  ``allow_anomalous_veto`` (``SliceRequest``, ``generation.
  allow_anomalous_veto``, ``--allow-anomalous-veto``) the cell is lowered
  and the artifact's generation record lists it. Without a veto the test is
  not run (an incomplete doublet or an anomalous model is lowered).

The site closure with MadLoop-style witness admission (``ct_sites``) stays a
test oracle (O6): per component, squared order and Laurent coefficient, the
recursion's partials equal the sites of the lowered rules
(``tests/unit/test_loop_v2_ct_recursion.py``, ``test_loop_v2_ct_closure.py``).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import reduce
from operator import add, mul
from types import MappingProxyType
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
    counterterm_lorentz_levi_civita,
    counterterm_slot_order,
    lower_counterterm_vertices,
)

from .ct_sites import (
    COMPONENT_BY_KIND,
    COUNTERTERM_KINDS,
    Cell,
    entry_cell,
    vanishes_identically,
)
from .families import LoopOrderSelection, Orders, add_orders

Slot = tuple[str, int, int]  # (component, epsilon power, N_c grade)
RuleKey = tuple[object, ...]
_PREFIX = "PYAMPLICOL_CT_"


class UnloweredCountertermError(NotImplementedError):
    """A request admits a counterterm rule its lowered model does not hold."""

    # The listed internal refusal (``refusals.LOOP_INTERNAL_REFUSALS``).
    refusal_key = "unlowered-counterterm"


class AnomalousVetoError(UnloweredCountertermError):
    """A request reaches a Levi-Civita counterterm cell that cancels with
    every flavour admitted but not under its ``particle_veto`` ([U2]: refused
    by default; ``allow_anomalous_veto`` lowers it)."""

    # The listed admission refusal (``refusals.LOOP_ADMISSION_REFUSALS``).
    refusal_key = "anomalous-veto"


# The override of an anomalous veto, as the refusals name it.
ANOMALOUS_VETO_OVERRIDE = (
    "allow_anomalous_veto (SliceRequest.allow_anomalous_veto, TOML "
    "generation.allow_anomalous_veto, CLI --allow-anomalous-veto)"
)


class CountertermCouplingError(NotImplementedError):
    """A CT coupling coefficient the tree model cannot express exactly."""

    # The listed internal refusal (``refusals.LOOP_INTERNAL_REFUSALS``).
    refusal_key = "counterterm-coupling"


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


def rule_terms(
    rule: Any, is_zero: Any = None
) -> tuple[tuple[Any, CountertermStructure], ...]:
    """(entry, lowered vertex term) of one rule, per entry and non-zero power.

    A Laurent coefficient that vanishes identically in the model's source
    program (e.g. the poles of an R2 coupling) has no slot. ``is_zero``
    (``entry, power -> bool``, default ``_coefficient_is_zero``) decides it,
    e.g. ``CoefficientLowering.is_zero``.
    """

    is_zero = _coefficient_is_zero if is_zero is None else is_zero
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
        if not is_zero(entry, power)
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
        self._levi_civita: dict[str, bool] = {}

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
        if self.ghost_leg(rule):
            return "ghost leg"
        return None

    def ghost_leg(self, rule: Any) -> bool:
        """Whether an entry of ``rule`` has a ghost leg: no tree current of
        physical legs carries a ghost, so such a rule is inert."""

        return any(
            int(self.particles[name].get("ghost_number", 0))
            for entry in rule.entries
            for name in entry.particle_names
        )

    def entry_reason(self, entry: Any) -> str | None:
        """Why the colour or Lorentz cell of ``entry`` is not lowered as it
        stands: it cannot be, or it is a Levi-Civita cell (``levi_civita``),
        left out where it cancels between the rules of its vertex and lowered
        where it does not (module docstring)."""

        return self._cell(entry)[0]

    def levi_civita(self, entry: Any) -> bool:
        """Whether the cell of ``entry`` is a Levi-Civita cell the tree
        compiler lowers: its colour in the tree vocabulary and its Lorentz
        structure contracting ``models.tensors.LEVI_CIVITA`` (UFO
        ``Epsilon``, e.g. the axial R2 of a Z with gluons)."""

        return self._cell(entry)[1]

    def _cell(self, entry: Any) -> tuple[str | None, bool]:
        records = [self.particles[name] for name in entry.particle_names]
        try:
            tree_colour_source(
                entry.color,
                [int(r["color"]) for r in records],
                multi_trace=self.multi_trace,
            )
        except UnloweredCountertermColour as error:
            return str(error), False
        name = entry.lorentz_name
        if name not in self._lorentz:
            self._lorentz[name] = counterterm_lorentz_error(self.compiled, name)
            self._levi_civita[name] = self._lorentz[
                name
            ] is None and counterterm_lorentz_levi_civita(self.compiled, name)
        if self._lorentz[name] is not None:
            return f"Lorentz structure {name}: {self._lorentz[name]}", False
        if self._levi_civita[name]:
            return (
                f"Lorentz structure {name}: Levi-Civita tensor (Epsilon), left "
                "out where it cancels",
                True,
            )
        return None, False


# ------------------------------------------------------------------- coefficients


# The UFO-standard bodies of the helpers the lowering resolves at generation
# (MadGraph's ``function_library``), with their arguments. A model's bound
# helper is resolved by name only when its body is structurally this one (the
# scalar library lowers both to the same expression); a model that rebinds the
# name (the CMS ``reglog`` of loop_qcd_qed_sm and its G_mu variant) keeps its
# own body, as a scalar-library function of its referenced CT scalars.
_STANDARD_HELPERS = {
    "cond": (
        ("condition", "ExprTrue", "ExprFalse"),
        "(ExprTrue if condition==0.0 else ExprFalse)",
    ),
    "reglog": (("z",), "(0.0 if z==0.0 else cmath.log(z))"),
    "complexconjugate": (("z",), "z.conjugate()"),
}


def standard_helpers(library: Any) -> dict[int, str]:
    """Source records of the helpers of a ``ScalarFunctionLibrary`` whose
    bound bodies (value and diagnostic) are the UFO-standard ``cond``,
    ``reglog`` or ``complexconjugate`` (``_STANDARD_HELPERS``), with the
    name: the digest rule of design §6.1 (loop_sm's three helpers keep
    today's by-name form, so its plans stay byte-identical)."""

    out: dict[int, str] = {}
    for name, (value, diagnostic, arguments) in library.helpers.items():
        standard = _STANDARD_HELPERS.get(name)
        if standard is None or len(standard[0]) != len(arguments):
            continue
        parameters, body = library._bodies[value]
        lowered = library.lower(
            standard[1],
            locals=dict(zip(standard[0], parameters, strict=True)),
            closed=True,
        )
        if (
            lowered.value == body
            and lowered.diagnostic == library._bodies[diagnostic][1]
        ):
            out[int(value.get_name().rsplit("_", 1)[1])] = name
    return out


def reference_name(scalar: str, power: int, part: str) -> str:
    """The lowered model's external parameter holding one part (``"re"`` or
    ``"im"``) of the epsilon^``power`` coefficient of a referenced coupling
    or CT scalar ``scalar`` (``CoefficientLowering``; the UFO's scalar names
    are unique across parameters, CT parameters and couplings)."""

    tag = "0" if not power else f"{'M' if power < 0 else 'P'}{abs(power)}"
    return f"{_PREFIX}P_{scalar}_E{tag}_{part.upper()}"


def _reference_defaults(names: Iterable[str]) -> dict[str, float]:
    """The declared defaults of reference parameters: binary64 numbers in
    [1, 2) with 48 bits fixed by each name, distinct (checked), so that no
    lowered coupling vanishes at the defaults by a coincidence
    (``lower_counterterm_vertices`` treats a coupling that vanishes there as
    a structural zero; the couplings' symbolic zeros are dropped before,
    ``lower_model_counterterms``). A difference of two references cannot
    vanish, and a rational relation between them only by a 48-bit
    coincidence. The defaults are never evaluated: the CT program binds
    every reference at the card (``CountertermRecursion.parameter_values``
    in Python, the parameter program's outputs natively)."""

    import hashlib

    found = {
        name: 1.0 + int(hashlib.sha256(name.encode()).hexdigest()[:12], 16) / 2.0**48
        for name in names
    }
    if len(set(found.values())) != len(found):
        raise CountertermCouplingError(
            "two counterterm reference parameters share a default"
        )
    return found


def _linear_degree(
    expression: Any, symbols_: frozenset, conj: Callable[[str], bool] | None = None
) -> float:
    """Polynomial degree of ``expression`` in the real symbols ``symbols_``;
    ``conj`` is transparent (it maps a real linear form to a real linear
    form), and so is a unary function whose name ``conj`` accepts (e.g. the
    scalar library's UFO-standard ``complexconjugate``). Infinite when a
    symbol is in another function's argument, a denominator or a
    non-integer power."""

    from symbolica import AtomType

    kind = expression.get_type()
    if kind == AtomType.Num:
        return 0
    if kind == AtomType.Var:
        return 1 if expression in symbols_ else 0
    children = list(expression)
    if kind == AtomType.Add:
        return max(_linear_degree(child, symbols_, conj) for child in children)
    if kind == AtomType.Mul:
        return sum(_linear_degree(child, symbols_, conj) for child in children)
    if kind == AtomType.Pow:
        base, exponent = children
        degree = _linear_degree(base, symbols_, conj)
        if not degree:
            return float("inf") if _linear_degree(exponent, symbols_, conj) else 0
        number = exponent.to_canonical_string()
        return degree * int(number) if number.isdigit() else float("inf")
    name = expression.get_name()
    if len(children) == 1 and (
        name == "symbolica::conj" or (conj is not None and conj(name))
    ):
        return _linear_degree(children[0], symbols_, conj)
    return (
        float("inf") if any(_linear_degree(c, symbols_, conj) for c in children) else 0
    )


@dataclass(frozen=True)
class _Referenced:
    """A coupling read in its referenced form: its source Laurent terms
    (epsilon power, value with the CT scalars as ``UFO::name`` symbols) and
    the CT scalars it reads (name, symbol)."""

    terms: tuple[tuple[int, Any], ...]
    scalars: tuple[tuple[str, Any], ...]


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
    * the UFO-standard helper calls are resolved (``standard_helpers``):
      ``cond(c, a, b)`` is ``a`` for a vanishing number ``c`` and ``b``
      otherwise, ``reglog(z)`` is 0 for a vanishing number and ``log(z)``
      otherwise, ``complexconjugate`` is ``conj``. The fixed parameters are
      substituted at the leaves while resolving and only the selected branch
      of a ``cond`` is resolved, so the branch that is not taken is never
      formed (e.g. the 1/MB^2 of ``cond(MB, 0, ...)`` at MB = 0, which
      Symbolica would make an infinity). A ``c`` or ``z`` free of model
      parameters (numbers, builtin constants such as ``𝜋``, builtin
      functions and the scalar library's predicates and conditionals of
      them) is decided here (``decide_constant``); one that depends on model
      parameters is recorded in ``assumptions`` without its constant factors
      (the parameter-dependent product that must not vanish at runtime,
      checked by ``CountertermModel.check_parameters``). An undecidable
      constant, a coefficient that is not finite and any other helper raise
      ``CountertermCouplingError``.

    **Referenced couplings** (WP3.2 EWI-3, EWI-3R). A coupling that reads a CT
    scalar (a UFO ``CTParameter``, or a parameter with epsilon) whose body
    calls a helper that is not UFO-standard (the electroweak ``re``,
    ``recms``, ``regsqrt``, ``reglogp``, ``grreglog``, ``arg``, or a rebound
    ``reglog``) is not inlined: the epsilon^p coefficient of the coupling is
    a pair of real external parameters of the lowered model (real and
    imaginary part, ``reference_name(coupling, p, part)``; ``references``),
    and the CT program binds them at the card from the coupling's exact
    source with its CT scalars' bodies, every helper a function of the
    model's scalar library (``CountertermModel.coefficient_plan``, the scalar
    program of the Born and field couplings): one helper semantics in every
    lane, each CT body evaluated once per card and never expanded into a
    vertex. The tree compiler sees one symbol per coefficient, so its kernel
    keys and equivalence proofs never expand the coupling's model parameters
    (e.g. the G_mu couplings resolved to ``Gf``, ``MW``, ``MZ`` and the
    ``conj`` of real masses), and the coupling may be any polynomial in its
    CT scalars (the coefficient is the exact Laurent coefficient of the
    product; NLOCT couplings are linear in them). Which epsilon powers can
    be non-zero is decided from the source (``is_zero``: a power the
    coupling's terms, its CT scalars' non-zero powers and the terms' degrees
    in them cannot reach, ``_reachable_powers``); a coefficient that
    vanishes at the fixed parameters is a zero of the plan
    (``coefficient_plan``) and has no vertex term. A coupling that is not a
    polynomial in its CT scalars, or that reaches a power below the CT
    slots, raises ``CountertermCouplingError`` (fail closed: its rule is
    residual, refused when a tree of the request can hold it).

    **Exact form** (``exact``). Under a formal-N_c continuation, a referenced
    coupling whose source reads one of the continuation's colour invariants
    keeps its N_c dependence explicit, and the cells the lowered model leaves
    out (``CountertermModel.unlowered``) need the CT scalars shared between
    the couplings of their rules to cancel: there the coupling is written on
    references of its CT scalars' Laurent coefficients (``P_k^(r)``, pairs as
    above, ``reference_name(scalar, r, part)``). The NLOCT couplings are
    linear in their CT scalars, so the epsilon^p coefficient of a coupling
    ``sum_q eps^q (a_q0 + sum_k a_qk P_k)`` is
    ``a_p0 + sum_q sum_k a_qk P_k^(p - q)``; a coupling that is not linear in
    its CT scalars raises ``CountertermCouplingError`` in this form.
    """

    def __init__(
        self,
        compiled: CompiledModel,
        fixed_parameters: Mapping[str, Any],
        identities: Mapping[str, Any] | None = None,
    ):
        from symbolica import E

        from pyamplicol._internal.physics.symbols import symbols
        from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary

        if compiled.raw_loop_model is None:
            raise ValueError("counterterm coefficients need the model's NLO sidecar")
        from ..model_cms import scheme_metadata

        # The complex-mass scheme's run-time widths (``_RealArguments``).
        self.widths = frozenset(
            (scheme_metadata(compiled) or {}).get("widths", {}).values()
        )
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
        self._substitution = dict(self._fixed)
        self._zero = E("0")
        self._log, self._conj = symbols.qualified("symbolica::log", "symbolica::conj")
        self._standard = standard_helpers(
            ScalarFunctionLibrary(compiled.raw_loop_model)
        )
        self.assumptions: set[str] = set()
        # Reference parameters: name -> (coupling or CT scalar, epsilon
        # power, "re" | "im"), and the scalar program they belong to.
        self.references: dict[str, tuple[str, int, str]] = {}
        self.program: Any = None
        self._identities = frozenset(identities)
        self._couplings: dict[tuple[int, int], _Referenced | None] = {}
        self._names: dict[int, dict[str, tuple[str, int]]] = {}
        self._scalars: dict[tuple[int, tuple[str, int]], bool | None] = {}
        self._powers: dict[tuple[int, str], frozenset[int] | None] = {}
        self._coupling_powers: dict[tuple[int, int], frozenset[int] | None] = {}

    def __call__(self, entry: Any, power: int) -> Any:
        referenced = self._referenced(entry)
        if referenced is not None:
            if self._graded(referenced):
                return self._referenced_coefficient(entry, power, referenced)
            return self._pair(entry.coupling.name, power)
        return self._finite(self._resolve(entry.coupling.value.coefficient(power)))

    def exact(self, entry: Any, power: int) -> Any:
        """The epsilon^``power`` coefficient of ``entry``'s coupling in its
        exact form (class docstring): a referenced coupling on references of
        its CT scalars' coefficients, any other as ``__call__`` gives it."""

        referenced = self._referenced(entry)
        if referenced is not None:
            return self._referenced_coefficient(entry, power, referenced)
        return self(entry, power)

    def payload_expression(self, expression: Any) -> Any:
        """An expression in source (``UFO::name``) symbols, e.g. a whole-rule
        coefficient, as a coefficient in the payload's symbols."""

        return self._finite(self._resolve(_payload_symbols(expression)))

    def is_zero(self, entry: Any, power: int) -> bool:
        """Whether the epsilon^``power`` coefficient of ``entry``'s coupling
        vanishes identically: for a referenced coupling when no term can
        reach the power (``_reachable_powers``; one that vanishes at the fixed
        parameters is a zero of the coefficient plan) or, graded, in its
        exact form (no CT body is expanded), otherwise in the source scalar
        program (``_coefficient_is_zero``). A referenced coupling
        ``_reachable_powers`` cannot bound raises ``CountertermCouplingError``."""

        referenced = self._referenced(entry)
        if referenced is None:
            return _coefficient_is_zero(entry, power)
        if self._graded(referenced):
            return _is_zero(self._referenced_coefficient(entry, power, referenced))
        powers = self._reachable_powers(entry, referenced)
        return powers is not None and power not in powers

    @staticmethod
    def _finite(value: Any) -> Any:
        """``value`` expanded; a value that is not finite (an infinity or an
        indeterminate formed in a taken branch) raises."""

        value = value.expand()
        if value.is_finite() is False:
            raise CountertermCouplingError(
                f"counterterm coefficient {value} is not finite at the fixed "
                "parameters (an infinity or indeterminate in a taken branch)"
            )
        return value

    # -- referenced couplings ----------------------------------------------------

    def _referenced(self, entry: Any) -> _Referenced | None:
        """The referenced form of ``entry``'s coupling when it reads a CT
        scalar whose body needs the scalar library, else ``None``."""

        from pyamplicol.loops.scalar_frontend import SourceScalarDefinition

        program = getattr(entry, "scalar_program", None)
        coupling = entry.coupling
        if (
            program is None
            or not isinstance(coupling, SourceScalarDefinition)
            or coupling.collection != "couplings"
        ):
            return None
        key = (id(program), coupling.id)
        if key in self._couplings:
            return self._couplings[key]
        library = program.library
        source = library.raw.exact_value("couplings", coupling.id)
        terms = tuple(
            (int(power), library.lower(text).value)
            for power, text in (
                source["terms"]
                if source["kind"] == "laurent"
                else ((0, source["expression"]),)
            )
        )
        names = self._scalar_names(program)
        scalars = tuple(
            sorted(
                {
                    (name, symbol)
                    for _power, value in terms
                    for symbol in value.get_all_symbols(False)
                    for name in (symbol.get_name().rsplit("::", 1)[-1],)
                    if symbol.get_name().startswith("UFO::")
                    and name in names
                    and self._library_scalar(program, names[name]) is not None
                },
                key=lambda pair: pair[0],
            )
        )
        found = None
        if any(self._library_scalar(program, names[name]) for name, _ in scalars):
            if self.program is None:
                self.program = program
            elif self.program is not program:
                raise CountertermCouplingError(
                    "referenced counterterm scalars of two scalar programs"
                )
            found = _Referenced(terms, scalars)
        self._couplings[key] = found
        return found

    def _scalar_names(self, program: Any) -> dict[str, tuple[str, int]]:
        found = self._names.get(id(program))
        if found is None:
            found = self._names[id(program)] = {
                definition.name: key for key, definition in program.definitions.items()
            }
        return found

    def _library_scalar(self, program: Any, key: tuple[str, int]) -> bool | None:
        """``None`` unless the scalar ``key`` is a CT scalar (a UFO CT
        parameter, or a parameter with epsilon); then whether its resolved
        body calls a helper that is not UFO-standard (``standard_helpers``)."""

        from pyamplicol.loops.model_expressions import epsilon_symbol

        cached = (id(program), key)
        if cached in self._scalars:
            return self._scalars[cached]
        definition = program.definitions[key]
        found: bool | None = None
        value = definition.value.expression
        if key[0] == "counterterm_parameters" or (
            key[0] == "parameters" and epsilon_symbol() in value.get_all_symbols()
        ):
            found = any(
                name.startswith("pyamplicol_scalar_model_")
                and name.rsplit("::", 1)[-1].startswith(("value_", "diagnostic_"))
                and int(name.rsplit("_", 1)[1]) not in self._standard
                for expression in (value, definition.diagnostic)
                for name in (s.get_name() for s in expression.get_all_symbols(True))
            )
        self._scalars[cached] = found
        return found

    def _scalar_powers(self, program: Any, name: str) -> frozenset[int] | None:
        """Epsilon powers of the non-zero source terms of the CT scalar
        ``name``; ``None`` (any power) when its source reads other CT scalars,
        whose powers then combine. Powers that turn out zero at the card are
        dropped with the coefficient plan (``lower_model_counterterms``)."""

        from symbolica import E

        cached = (id(program), name)
        if cached in self._powers:
            return self._powers[cached]
        library = program.library
        names = self._scalar_names(program)
        source = library.raw.exact_value(*names[name])
        found: frozenset[int] | None = frozenset((0,))
        if source["kind"] == "laurent":
            found = frozenset(
                int(power)
                for power, text in source["terms"]
                if library.lower(text).value != E("0")
            )
        if any(
            dependency in names
            and self._library_scalar(program, names[dependency]) is not None
            for symbol in library.lower_value(source).value.get_all_symbols(False)
            for dependency in (symbol.get_name().rsplit("::", 1)[-1],)
            if symbol.get_name().startswith("UFO::")
        ):
            found = None
        self._powers[cached] = found
        return found

    def _pair(self, name: str, power: int) -> Any:
        """The reference pair ``re + i im`` of the epsilon^``power``
        coefficient of the coupling or CT scalar ``name``."""

        from symbolica import E, Expression

        parts = []
        for part in ("re", "im"):
            parameter = reference_name(name, power, part)
            self.references[parameter] = (name, power, part)
            parts.append(E(f"UFO::{{}}::{parameter}"))
        return parts[0] + parts[1] * Expression.I

    def _is_conj(self, name: str) -> bool:
        """Whether the function ``name`` is the scalar library's UFO-standard
        ``complexconjugate`` (``standard_helpers``)."""

        head = name.rsplit("::", 1)[-1]
        return (
            name.startswith("pyamplicol_scalar_model_")
            and head.startswith("value_")
            and self._standard.get(int(head.removeprefix("value_")))
            == "complexconjugate"
        )

    def _graded(self, referenced: _Referenced) -> bool:
        """Whether a referenced coupling's source reads a colour invariant of
        the formal-N_c continuation (its exact form keeps the N_c powers)."""

        return bool(self._identities) and any(
            symbol.get_name().rsplit("::", 1)[-1] in self._identities
            for _power, value in referenced.terms
            for symbol in value.get_all_symbols(False)
        )

    def _reachable_powers(
        self, entry: Any, referenced: _Referenced
    ) -> frozenset[int] | None:
        """The epsilon powers at which a referenced coupling's Laurent
        coefficient can be non-zero: a term's own power when its part free of
        CT scalars does not vanish, and its power plus every sum of ``k``
        non-zero powers of the CT scalars it reads (``_scalar_powers``), for
        ``k`` from 1 to the term's polynomial degree in them (a linear term:
        each scalar's powers; a product or power of CT scalars: their sums, a
        superset whose exact zeros the coefficient plan drops); ``None`` (any
        power) when a CT scalar's powers combine with others'.

        Fail closed (``CountertermCouplingError``, the rule then residual): a
        term that is not a polynomial in its CT scalars (one in a
        denominator, under a non-integer power or in a function other than
        ``conj``), whose Laurent coefficients this bound does not cover, and
        a reachable power below the lowest CT slot (``COUNTERTERM_POWERS``),
        which no vertex term could hold."""

        from symbolica import Replacement

        program = entry.scalar_program
        key = (id(program), entry.coupling.id)
        if key in self._coupling_powers:
            return self._coupling_powers[key]
        removed = [
            Replacement(symbol, self._zero) for _name, symbol in referenced.scalars
        ]
        found: set[int] | None = set()
        for order, value in referenced.terms:
            if not _is_zero(value.replace_multiple(removed)):
                found.add(order)
            read = value.get_all_symbols(False)
            shifts: set[int] = set()
            factors = []
            for name, symbol in referenced.scalars:
                if symbol not in read:
                    continue
                powers = self._scalar_powers(program, name)
                if powers is None:
                    found = None
                    break
                shifts.update(powers)
                factors.append(symbol)
            if found is None:
                break
            if not factors:
                continue
            degree = _linear_degree(value, frozenset(factors), self._is_conj)
            if degree == float("inf"):
                raise CountertermCouplingError(
                    f"the coupling {entry.coupling.name} is not a polynomial in "
                    "its counterterm scalars "
                    + ", ".join(name for name, _symbol in referenced.scalars)
                )
            sums = {0}
            for _factor in range(int(degree)):
                sums = {total + shift for total in sums for shift in shifts}
                found.update(order + total for total in sums)
        if found is not None and min(found, default=0) < min(COUNTERTERM_POWERS):
            raise CountertermCouplingError(
                f"the coupling {entry.coupling.name} reaches epsilon^"
                f"{min(found)}, below the counterterm slots {COUNTERTERM_POWERS}"
            )
        result = None if found is None else frozenset(found)
        self._coupling_powers[key] = result
        return result

    def _referenced_coefficient(
        self, entry: Any, power: int, referenced: _Referenced
    ) -> Any:
        """The epsilon^``power`` coefficient of a referenced coupling on the
        reference parameters of its CT scalars (class docstring)."""

        from symbolica import E, Expression, Replacement

        program = entry.scalar_program
        removed = [
            Replacement(symbol, self._zero) for _name, symbol in referenced.scalars
        ]
        total = self._zero
        symbols_ = set()
        for order, value in referenced.terms:
            shift = power - order
            replacements = []
            for name, symbol in referenced.scalars:
                powers = self._scalar_powers(program, name)
                if powers is not None and shift not in powers:
                    replacements.append(Replacement(symbol, self._zero))
                    continue
                parts = []
                for part in ("re", "im"):
                    parameter = reference_name(name, shift, part)
                    self.references[parameter] = (name, shift, part)
                    parts.append(E(f"UFO::{{}}::{parameter}"))
                    symbols_.add(parts[-1])
                replacements.append(
                    Replacement(symbol, parts[0] + parts[1] * Expression.I)
                )
            term = value.replace_multiple(replacements)
            if shift:
                # The part free of CT scalars has the coupling's own power.
                term -= value.replace_multiple(removed)
            total += term
        coefficient = self._finite(self._resolve(_payload_symbols(total)))
        if _linear_degree(coefficient, frozenset(symbols_)) > 1:
            raise CountertermCouplingError(
                f"the coupling {entry.coupling.name} is not linear in its "
                "counterterm scalars "
                + ", ".join(name for name, _symbol in referenced.scalars)
            )
        return coefficient

    # -- resolution ----------------------------------------------------------------

    def _resolve(self, atom: Any) -> Any:
        from symbolica import AtomType

        from pyamplicol._internal.physics.symbols import symbols

        kind = atom.get_type()
        if kind == AtomType.Num:
            return atom
        if kind == AtomType.Var:
            return self._substitution.get(atom, atom)
        if kind in (AtomType.Add, AtomType.Mul, AtomType.Pow):
            children = [self._resolve(child) for child in atom]
            if kind == AtomType.Add:
                return reduce(add, children)
            if kind == AtomType.Mul:
                return reduce(mul, children)
            base, exponent = children
            return base**exponent
        name = atom.get_name()
        head = name.rsplit("::", 1)[-1]
        if not (
            name.startswith("pyamplicol_scalar_model_") and head.startswith("value_")
        ):
            return symbols.qualified(name)(*(self._resolve(child) for child in atom))
        record = int(head.removeprefix("value_"))
        # Only UFO-standard bodies are resolved by name (``standard_helpers``).
        helper = self._standard.get(record)
        if helper == "cond":
            # The branch that is not taken is never resolved: its conditions
            # are no assumptions, and its fixed-parameter singularities (a
            # log of MU_R^2/MB^2 at MB = 0) are never formed.
            condition, if_zero, otherwise = list(atom)
            taken = if_zero if self._vanishes(self._resolve(condition)) else otherwise
            return self._resolve(taken)
        children = [self._resolve(child) for child in atom]
        if helper == "reglog":
            (argument,) = children
            if self._vanishes(argument):
                return self._zero
            return self._log(argument)
        if helper == "complexconjugate":
            (argument,) = children
            return self._conj(argument)
        raise CountertermCouplingError(
            "counterterm coefficient uses the UFO helper "
            f"{self._helpers.get(record, helper)!r}, which the tree model has no "
            "exact form of"
        )

    def _vanishes(self, value: Any) -> bool:
        """Whether the helper argument ``value`` vanishes: decided now when
        it is free of model parameters, else assumed not (``assumptions``)."""

        from symbolica import AtomType

        value = _decided_conditionals(value).expand()
        factors = list(value) if value.get_type() == AtomType.Mul else [value]
        # An assumption states that a parameter-dependent factor is non-zero:
        # constant factors are decided here and never become assumptions.
        varying = []
        for factor in factors:
            if not factor.is_constant() and _has_parameters(factor):
                varying.append(factor)
                continue
            decision = decide_constant(factor)
            if decision == "zero":
                return True
            if decision != "non-zero":
                raise CountertermCouplingError(
                    f"cannot decide whether the counterterm coefficient argument "
                    f"{value} vanishes: its constant factor {factor} is {decision}"
                )
        if varying:
            self.assumptions.add(reduce(mul, varying).to_canonical_string())
        return False


# The scalar library's predicates (``scalar_frontend``) and its conditional.
_PREDICATE_PREFIX = "pyamplicol_scalar_v1::"
_IF = "symbolica::if"


def _exact_number(value: Any) -> Fraction | None:
    """The exact value of a real rational number atom, else ``None``."""

    from symbolica import AtomType

    if value.get_type() != AtomType.Num:
        return None
    text = value.to_canonical_string()
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


class _Every:
    """Every index (``_RealArguments.cut_parameters`` of an unknown body)."""

    def __contains__(self, item: object) -> bool:
        return True


_EVERY = _Every()


def _symbol_names(value: Any) -> frozenset[str]:
    """The canonical names of the symbols of ``value``."""

    return frozenset(symbol.to_canonical_string() for symbol in value.get_all_symbols())


def _has_parameters(value: Any) -> bool:
    """Whether ``value`` holds a symbol other than a builtin constant."""

    from symbolica import Expression

    builtins = (Expression.PI, Expression.E, Expression.I)
    return any(s not in builtins for s in value.get_all_symbols(False))


def _decided_predicate(name: str, arguments: list[Any]) -> Any:
    """A scalar-library predicate (or real/imaginary projection) of exact
    real numbers, decided: ``1`` or ``0`` (the number for a projection);
    ``None`` when an argument is not an exact real number."""

    from symbolica import E

    head = name.removeprefix(_PREDICATE_PREFIX)
    numbers = [_exact_number(argument) for argument in arguments]
    if any(number is None for number in numbers):
        return None
    if head in ("real_part", "imaginary_part") and len(numbers) == 1:
        return arguments[0] if head == "real_part" else E("0")
    if head == "truth" and len(numbers) == 1:
        return E("1" if numbers[0] != 0 else "0")
    if len(numbers) != 2:
        return None
    left, right = numbers
    decision = {
        "equal": left == right,
        "less": left < right,
        "less_equal": left <= right,
    }.get(head)
    return None if decision is None else E("1" if decision else "0")


def _decided_conditionals(expression: Any) -> Any:
    """``expression`` with every scalar-library predicate of exact real
    numbers decided and every ``symbolica::if`` whose condition is then an
    exact number reduced to its taken branch (the branch not taken is never
    formed; ``lazily_fixed`` without replacements). E.g. the CMS switch
    ``if(equal(0, 1), 1, 0)`` of the G_mu UFO's ``CMSParam`` at its
    real-mass value 0 is 0."""

    return lazily_fixed(expression, {})


# Working precisions (decimal digits) of the numerical decision of constants.
_DECISION_DIGITS = 40


def decide_constant(value: Any) -> str:
    """Whether ``value``, an expression free of model parameters, vanishes.

    ``value`` holds numbers, builtin constants (``𝜋``, Euler's number, the
    imaginary unit) and builtin functions of them
    (``Expression.is_constant``), and the scalar library's predicates and
    conditionals of such values, which are decided exactly first
    (``_decided_conditionals``; e.g. the G_mu UFO's CMS switch
    ``if(equal(0, 1), 1, 0)``). Returns

    * ``"zero"`` or ``"non-zero"``: decided. A number (rational or
      floating-point) is compared with 0 exactly. Any other expression is
      evaluated at ``_DECISION_DIGITS`` and twice as many digits, with its
      floating-point coefficients replaced by the rationals of their stored
      values (a binary64 coefficient would cap both evaluations at 16
      digits): it is non-zero when both values agree to half the lower
      precision, relative to the more precise one, and that one is non-zero.
    * ``"singular"``: an infinite or indeterminate value.
    * ``"undecided"``: neither (e.g. ``sin(𝜋)``, which Symbolica keeps; its
      values at both precisions are rounding residues that do not agree).
      Callers refuse it, never assume it non-zero.
    """

    from decimal import Decimal, localcontext

    from symbolica import AtomType, E

    from pyamplicol.models.compiler_symbolica import _exact_binary64_coefficients

    value = _decided_conditionals(value)
    if not value.is_constant():
        if _has_parameters(value):
            raise ValueError(f"{value} depends on model parameters")
        return "undecided"  # a predicate or conditional of an inexact value
    if value.is_finite() is False:
        return "singular"
    if value.get_type() == AtomType.Num:
        return "zero" if value == E("0") else "non-zero"
    exact = _exact_binary64_coefficients(value)
    values = []
    for digits in (_DECISION_DIGITS, 2 * _DECISION_DIGITS):
        try:
            number = exact.evaluate({}, decimal_digit_precision=digits)
        except ValueError:  # e.g. an infinite coefficient
            return "singular"
        if not number.is_finite():
            return "singular"
        values.append((number.real.to_decimal(), number.imag.to_decimal()))
    with localcontext() as context:
        context.prec = 4 * _DECISION_DIGITS
        (low_re, low_im), (high_re, high_im) = values
        difference = (low_re - high_re) ** 2 + (low_im - high_im) ** 2
        size = high_re**2 + high_im**2
        if size and difference <= size * Decimal(10) ** -_DECISION_DIGITS:
            return "non-zero"
    return "undecided"


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


# Decimal digits of the references' coefficients at least (those of a
# double-double input), and the cards whose values a model keeps.
_COEFFICIENT_DIGITS = 32
# Guard digits of their evaluation: the plan's Symbolica evaluator orders its
# operations by the process's symbol registrations (e.g. whether the base
# model was compiled in this process or loaded from the cache), and the
# couplings' cancellations lose up to about 3 digits (measured at the G_mu
# target: 98 of 728 references differed between the two histories by up to
# 3e-30 relative at 32 digits). With these the values a carrier rounds do
# not depend on the history.
_COEFFICIENT_GUARD_DIGITS = 16
_BOUND_KEPT = 8


@dataclass(frozen=True)
class CountertermModel:
    """The model-level CT lowering shared by every process of one model.

    ``multiplicities`` names the integer runtime parameter of every lowered
    rule (by ``rule_key``) and ``switches`` that of every slot;
    ``assumptions`` are model-parameter expressions the lowered couplings
    assume non-zero (``CoefficientLowering``); ``residual`` maps every rule
    the lowering cannot lower to the reason, and ``inert`` names those of
    them that contribute nothing (ghost legs and orders above the program's
    limits: in no tree of it; no admitted loop content); ``structures`` are
    the lowered vertex terms of every lowered rule (graded with a
    continuation); ``unlowered`` holds, per lowered rule,
    the cells the lowered model leaves out because they cancel between rules
    with the multiplicities ``weights`` (module docstring): cell -> (why it is
    not lowered, {Laurent power: exact coefficient}). ``veto`` are the
    vetoed |PDG| codes (``particle_veto``) and ``species_orders`` the
    model's perturbative species facts of the admission
    (``admitted_alternatives``), ``None`` when the lowering had none.
    ``anomalous_veto`` maps every rule with a Levi-Civita cell that cancels
    with every flavour admitted but not under ``veto`` to those cells
    (module docstring, "Anomalous vetoes"): residual unless
    ``allow_anomalous_veto``, lowered otherwise.

    ``coefficients`` maps every reference parameter of the lowered model
    (``CoefficientLowering``: a part of a referenced coupling's Laurent
    coefficient, or in its exact form of a CT scalar's) to (coupling or
    scalar name, epsilon power, ``"re"`` or ``"im"``); ``coefficient_plan``
    (a ``scalar_bindings.ScalarBindingPlan`` of those (name, power) outputs,
    every helper a scalar-library function, the generation's fixed
    parameters substituted) binds them at the card (``coefficient_values``).
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
    weights: Mapping[RuleKey, int] = field(default_factory=dict)
    inert: frozenset[RuleKey] = frozenset()
    veto: frozenset[int] = frozenset()
    species_orders: Mapping[int, frozenset[str]] | None = field(
        default=None, repr=False
    )
    anomalous_veto: Mapping[RuleKey, frozenset[Cell]] = field(default_factory=dict)
    allow_anomalous_veto: bool = False
    coefficients: Mapping[str, tuple[str, int, str]] = field(default_factory=dict)
    coefficient_plan: Any = field(default=None, compare=False, repr=False)
    # (card, digits) -> reference values (``coefficient_values``).
    _bound: dict = field(default_factory=dict, compare=False, repr=False)

    def coefficient_values(
        self, card: Mapping[str, Any], digits: int
    ) -> dict[str, str]:
        """The reference parameters (``coefficients``) at the model parameters
        ``card`` (decimal strings; absent ones take their exact source
        defaults, the fixed ones are the generation's), as decimal strings:
        the scalar library evaluates the couplings' and CT scalars' exact
        bodies at ``max(digits, 32)`` digits and ``_COEFFICIENT_GUARD_DIGITS``
        more (a carrier rounds them once, as it rounds its couplings). A
        value or active diagnostic that is not finite raises
        ``scalar_bindings.SourceBindingFailure`` (fail closed)."""

        if not self.coefficients:
            return {}
        plan = self.coefficient_plan
        names = frozenset(plan.external_names)
        inputs = tuple(
            sorted((name, str(value)) for name, value in card.items() if name in names)
        )
        digits = max(int(digits), _COEFFICIENT_DIGITS)
        found = self._bound.get((inputs, digits))
        if found is None:
            from pyamplicol.loops.scalar_bindings import NativeScalarBindings

            state = NativeScalarBindings(
                plan,
                arithmetic="arbitrary",
                precision=digits + _COEFFICIENT_GUARD_DIGITS,
                parameters=dict(inputs),
            )
            values = dict(zip(plan.outputs, state.values.to_decimal(), strict=True))
            found = {
                name: values[scalar, power][part == "im"]
                for name, (scalar, power, part) in self.coefficients.items()
            }
            while len(self._bound) >= _BOUND_KEPT:
                self._bound.pop(next(iter(self._bound)))
            self._bound[inputs, digits] = found
        return found

    def check_parameters(self, values: Mapping[str, Any]) -> None:
        """Fail if runtime model parameter ``values`` break an assumption.

        Parameters absent from ``values`` take the compiled defaults. Every
        assumption must be decided with the parameters substituted
        (``decide_constant``): one that vanishes, is singular (e.g. a mass
        in a denominator set to 0) or cannot be decided is refused. A
        derived (internal) parameter an assumption reads (a derived mass)
        takes its resolved definition at these values.
        """

        from symbolica import E, Expression, Replacement

        from pyamplicol._internal.physics.symbols import symbols

        from .ct_sites import resolved_internal_parameters

        if not self.assumptions:
            return
        compiled = self.lowered.compiled
        defaults = {
            name: complex(*pair) for name, pair in compiled.parameter_defaults.items()
        }
        defaults.update(values)
        numbers = {
            name: E(repr(complex(value).real))
            + E(repr(complex(value).imag)) * Expression.I
            for name, value in defaults.items()
        }
        replacements = [
            Replacement(E(f"UFO::{{}}::{name}"), number)
            for name, number in numbers.items()
        ]
        payload = "UFO::{}::"

        def resolved(expression: Any) -> Any:
            # The model parameters left (internal ones, e.g. a derived mass):
            # their resolved definitions at these values.
            left = {
                name[len(payload) :]
                for name in (
                    s.to_canonical_string() for s in expression.get_all_symbols(False)
                )
                if name.startswith(payload)
            }
            if not left:
                return expression
            model_symbols = symbols.model(compiled.name)

            def symbol(name: str) -> Any:
                return E(model_symbols.symbol(name).to_canonical_string())

            on_values = [
                Replacement(symbol(name), number) for name, number in numbers.items()
            ]
            return expression.replace_multiple(
                [
                    Replacement(
                        E(f"{payload}{record.name}"),
                        resolved_internal_parameters(
                            compiled, symbol(record.name)
                        ).replace_multiple(on_values),
                    )
                    for record in compiled.ir.parameters
                    if record.nature == "internal" and record.name in left
                ]
            )

        for assumption in self.assumptions:
            expression = resolved(E(assumption).replace_multiple(replacements))
            expression = _decided_conditionals(expression).expand()
            decision = (
                "undecided"  # a symbol no parameter value replaces
                if _has_parameters(expression)
                else decide_constant(expression)
            )
            if decision == "zero":
                raise ValueError(
                    f"model parameters make {assumption} vanish, which the lowered "
                    "counterterm couplings assume non-zero"
                )
            if decision == "singular":
                raise ValueError(
                    f"model parameters make {assumption} singular, which the "
                    "lowered counterterm couplings assume finite and non-zero"
                )
            if decision != "non-zero":
                raise ValueError(f"cannot decide counterterm assumption {assumption}")


def lower_model_counterterms(
    compiled: CompiledModel,
    rules: Mapping[str, Any],
    *,
    fixed_parameters: Mapping[str, Any] | None = None,
    veto: Iterable[int] = (),
    corrections: Iterable[str] | None = None,
    multi_trace: bool = False,
    continuation: Any = None,
    dirac_fields: Iterable[str] = (),
    order_limits: Mapping[str, int] | None = None,
    species_orders: Mapping[int, frozenset[str]] | None = None,
    allow_anomalous_veto: bool = False,
) -> CountertermModel:
    """The tree model of ``compiled`` with every lowerable rule's CT vertices.

    ``rules`` are prepared counterterm rules (``PreparedCountertermModel.rules``)
    of any request and ``fixed_parameters`` the exact fixed parameter values of
    that preparation (``structure.fixed_parameters``); preparing the model
    without order bounds gives every rule, so one lowered model serves every
    process of the model with the same ``veto`` (|PDG| codes: the vetoed
    particles are removed from every tree vertex, ordinary or CT, and every
    loop-content alternative holding one is dropped, ``admitted_alternatives``).
    ``corrections`` (``None``: all) select the rules whose multiplicities the
    cancellation of unlowerable cells assumes.

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
    them, and a rule above them is inert. ``None`` lowers the whole model.

    ``species_orders`` (the prepared structure's
    ``perturbative_species_orders``) admit an alternative only as a loop of a
    selected correction (``admitted_alternatives``); a rule without one is
    inert.

    ``allow_anomalous_veto`` lowers the Levi-Civita cells a ``veto`` keeps
    from cancelling that cancel without it; by default their rules are
    residual (module docstring, "Anomalous vetoes"; [U2]).
    """

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols

    dirac_fields = frozenset(dirac_fields)
    veto = frozenset(abs(int(pdg)) for pdg in veto)
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
    inert: set[RuleKey] = set()
    # Exact coefficient of every rule on each of its lowered vertex terms.
    values: dict[RuleKey, dict[CountertermStructure, Any]] = {}
    # Cells the tree compiler cannot lower, per rule: (reason, coefficients).
    unlowered: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]] = {}
    weights: dict[RuleKey, int] = {}
    # The anomalous-veto test (module docstring): the multiplicities without
    # the veto, and the unlowered cells of the rules the veto leaves without
    # loop content (from their own lowering, so that the lowered model's
    # assumptions and references stay those of the admitted rules).
    unvetoed: dict[RuleKey, int] = {}
    vetoed_cells: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]] = {}
    probe: list[CoefficientLowering] = []

    def unlowered_cells(
        rule: Any, lowering: CoefficientLowering
    ) -> dict[Cell, tuple[str, dict[int, Any]]]:
        cells: dict[Cell, tuple[str, dict[int, Any]]] = {}
        for entry in rule.entries:
            reason = vocabulary.entry_reason(entry)
            if reason is None:
                continue
            powers = cells.setdefault(entry_cell(entry), (reason, {}))[1]
            for power in COUNTERTERM_POWERS:
                if lowering.is_zero(entry, power):
                    continue
                # Left out: its cancellation needs the CT scalars shared
                # between couplings (``CoefficientLowering.exact``).
                value = lowering.exact(entry, power)
                powers[power] = powers[power] + value if power in powers else value
        return cells

    for rule in rules.values():
        key = rule_key(rule)
        reason = vocabulary.rule_reason(rule)
        if reason is not None and vocabulary.ghost_leg(rule):
            inert.add(key)
        if reason is None and order_limits is not None:
            beyond = [
                orders
                for orders in {add_orders(tuple(e.order)) for e in rule.entries}
                if any(power > order_limits.get(name, power) for name, power in orders)
            ]
            if beyond:
                reason = f"its orders {beyond[0]} exceed the program's limits"
                inert.add(key)
        if reason is None and veto:
            unvetoed[key] = len(
                admitted_alternatives(rule, (), corrections, species_orders)
            )
        if reason is None and not admitted_alternatives(
            rule, veto, corrections, species_orders
        ):
            reason = "no admitted loop content (particle_veto or correction)"
            inert.add(key)
            if unvetoed.get(key) and rule.whole_rule_continuation is None:
                if not probe:
                    probe.append(
                        CoefficientLowering(
                            compiled,
                            fixed_parameters or {},
                            None
                            if continuation is None
                            else _payload_identities(continuation),
                        )
                    )
                try:
                    found = unlowered_cells(rule, probe[0])
                except (CountertermCouplingError, NotImplementedError):
                    found = {}
                if found:
                    vetoed_cells[key] = found
        if reason is not None:
            residual[key] = reason
            continue
        rows: dict[CountertermStructure, Any] = {}
        cells: dict[Cell, tuple[str, dict[int, Any]]] = {}
        try:
            if rule.whole_rule_continuation is not None:
                rows = whole_rule_rows(rule, coefficients)
            else:
                cells = unlowered_cells(rule, coefficients)
            for entry, structure in (
                rule_terms(rule, coefficients.is_zero)
                if rule.whole_rule_continuation is None
                else ()
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
            weights[key] = len(
                admitted_alternatives(rule, veto, corrections, species_orders)
            )
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

    plan = _referenced_scalars(
        coefficients, values, unlowered, residual, fixed_parameters or {}
    )

    internal = frozenset(
        record.name for record in compiled.ir.parameters if record.nature == "internal"
    )

    model_symbols = symbols.model(compiled.name)

    def vanishes(expression: Any) -> bool:
        # ``vanishes_identically``: without an internal parameter of the
        # payload's its resolution leaves the expansion as it is, and only
        # the re-parsed expansion is expanded again (Symbolica's expand is
        # not idempotent: a second one can expand a sum the first left in a
        # factor).
        value = expression.expand()
        if value == E("0"):
            return True
        if not any(
            symbol.get_name().rsplit("::", 1)[-1] in internal
            for symbol in value.get_all_symbols(False)
        ):
            parsed = model_symbols.expression(value.to_canonical_string())
            return parsed.expand() == E("0")
        return vanishes_identically(compiled, expression)

    def cancels_unvetoed(cell: Cell) -> bool:
        """Whether ``cell`` cancels with every flavour admitted: the
        cancellation test with the multiplicities of no veto, over the
        cell's admitted and vetoed-out rules."""

        cells = {
            key: {cell: found[cell]}
            for key, found in (*unlowered.items(), *vetoed_cells.items())
            if cell in found and unvetoed.get(key)
        }
        _members, cancelled = _cell_cancellation(cells, unvetoed, vanishes)
        return bool(cells) and cancelled(cell)

    # Unlowered cells must cancel with the rules' multiplicities; otherwise
    # a Levi-Civita cell is lowered (unless the veto keeps it from
    # cancelling: refused by default, [U2]), and any other makes every rule
    # of the cell residual (refused by a request admitting it).
    by_key = {rule_key(rule): rule for rule in rules.values()}
    anomalous: dict[RuleKey, set[Cell]] = defaultdict(set)
    settle = _CellSettlement(
        weights,
        vanishes,
        by_key,
        vocabulary,
        coefficients,
        fixed_parameters or {},
        multi_trace=multi_trace,
        nc=nc,
        anomalous=anomalous,
        cancels_unvetoed=cancels_unvetoed if veto else None,
        allow_anomalous_veto=allow_anomalous_veto,
        veto=veto,
    )
    plan = settle(plan, unlowered, residual, values)
    unlowered = {
        key: cells for key, cells in unlowered.items() if key in values and cells
    }
    multiplicities = {
        key: f"{_PREFIX}M{index}" for index, key in enumerate(sorted(values, key=repr))
    }
    switches = {
        slot: _switch_name(slot)
        for slot in sorted(
            {(s.component, s.power, s.grade) for rows in values.values() for s in rows}
        )
    }
    excluded = {
        str(item["name"])
        for item in compiled.model["particles"]
        if abs(int(item["pdg_code"])) in veto
    }
    references = frozenset(coefficients.references)
    refused: dict[CountertermStructure, str] = {}
    while True:
        parameters = dict.fromkeys((*multiplicities.values(), *switches.values()), 1.0)
        read = _reads(
            (v for rows in values.values() for v in rows.values()), references
        )
        parameters.update(_reference_defaults(sorted(read)))
        # A coupling that vanishes once the model's internal parameters are
        # resolved (an off-diagonal CKM entry of a diagonal restriction) has
        # no vertex term: dropped here, so ``lower_counterterm_vertices``
        # compiles the tree model once instead of again without it.
        couplings = {
            structure: expression
            for structure, expression in _coupling_expressions(
                values, multiplicities, switches
            ).items()
            if not vanishes(E(expression))
        }
        try:
            lowered = lower_counterterm_vertices(
                compiled,
                couplings,
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
            # The other rules of a refused rule's cancelling cells follow it
            # (a Levi-Civita cell is lowered for them instead).
            plan = settle(plan, unlowered, residual, values)
            for key in [k for k in multiplicities if k not in values]:
                del multiplicities[key]
            for slot in sorted(
                {
                    (s.component, s.power, s.grade)
                    for rows in values.values()
                    for s in rows
                }
            ):
                switches.setdefault(slot, _switch_name(slot))
            unlowered = {
                k: cells for k, cells in unlowered.items() if k in values and cells
            }
            continue
        return CountertermModel(
            lowered=replace(lowered, refused=refused),
            multiplicities=multiplicities,
            switches=switches,
            assumptions=tuple(sorted(coefficients.assumptions)),
            residual=residual,
            structures={key: tuple(sorted(rows)) for key, rows in values.items()},
            unlowered=unlowered,
            weights={key: weights[key] for key in unlowered},
            inert=frozenset(inert),
            veto=veto,
            species_orders=species_orders,
            anomalous_veto={
                key: frozenset(cells) for key, cells in sorted(anomalous.items())
            },
            allow_anomalous_veto=allow_anomalous_veto,
            coefficients={name: coefficients.references[name] for name in sorted(read)},
            coefficient_plan=plan if read else None,
        )


def _referenced_scalars(
    coefficients: CoefficientLowering,
    values: dict[RuleKey, dict[CountertermStructure, Any]],
    unlowered: Mapping[RuleKey, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    residual: dict[RuleKey, str],
    fixed_parameters: Mapping[str, Any],
) -> Any:
    """The coefficient plan of the references the lowered rows ``values``
    (and their left-out cells) read: the Laurent coefficients of referenced
    couplings and, in their exact form, of CT scalars
    (``CoefficientLowering``), at the generation's fixed parameters
    (``coefficient_plan``), or ``None`` when they read none.

    ``values`` is updated in place: a reference that vanishes at the fixed
    parameters is a structural zero (rows that vanish then have no vertex
    term, as with an inlined coefficient), and a rule that reads a reference
    that is not finite there (as an expression, or in binary64 at the model's
    default card) moves to ``residual`` (M2: fail closed; a request whose
    trees can hold it is refused)."""

    from symbolica import E, Replacement

    references = coefficients.references
    if not references:
        return None
    names = frozenset(references)
    read = _reads((v for rows in values.values() for v in rows.values()), names)
    read |= _reads(
        (
            v
            for cells in unlowered.values()
            for _reason, powers in cells.values()
            for v in powers.values()
        ),
        names,
    )
    if not read:
        return None
    found = coefficient_plan(
        coefficients.program,
        {references[name][:2] for name in read},
        fixed_parameters,
        coefficients._standard,
        widths=coefficients.widths,
    )
    zeros = [
        Replacement(E(f"UFO::{{}}::{name}"), E("0"))
        for name in sorted(read)
        if references[name][:2] in found.zero
    ]
    bad = frozenset(name for name in read if references[name][:2] in found.nonfinite)
    for key in sorted(values, key=repr):
        cells = unlowered.get(key, {})
        if bad and (
            _reads(values[key].values(), bad)
            or any(_reads(powers.values(), bad) for _r, powers in cells.values())
        ):
            del values[key]
            scalars = sorted({references[name][0] for name in bad})
            residual[key] = (
                "its counterterm coefficients of "
                + ", ".join(scalars)
                + " are not finite at the fixed parameters and the model's "
                "default card"
            )
            continue
        if zeros:
            rows = {
                s: v.replace_multiple(zeros).expand() for s, v in values[key].items()
            }
            values[key] = {s: v for s, v in rows.items() if not _is_zero(v)}
            for _reason, powers in cells.values():
                for power in list(powers):
                    powers[power] = powers[power].replace_multiple(zeros).expand()
    return found.plan


def _reads(rows: Iterable[Any], names: frozenset[str]) -> frozenset[str]:
    """The parameters of ``names`` the expressions ``rows`` read."""

    if not names:
        return frozenset()
    return frozenset(
        name
        for value in rows
        for symbol in value.get_all_symbols(False)
        for name in (symbol.get_name().rsplit("::", 1)[-1],)
        if name in names
    )


def lazily_fixed(
    expression: Any,
    values: Mapping[Any, Any],
    conds: Mapping[str, str] = MappingProxyType({}),
    memo: dict[tuple[Any, bool], Any] | None = None,
) -> Any:
    """``expression`` with the symbols of ``values`` replaced, top down: a
    conditional (``symbolica::if``, or a call of a UFO-standard ``cond``
    helper of ``conds``: head name -> ``"value"`` or ``"diagnostic"``) whose
    condition the replacement makes an exact number keeps only its taken
    branch, so the branch that is not taken is never formed (e.g. the
    1/MB^2 of ``cond(MB, ...)`` at MB = 0, an infinity no evaluator accepts),
    and a decided ``cond`` diagnostic is 0. Scalar-library predicates of
    exact numbers are decided. With ``values``, subexpressions free of their
    symbols are kept as they are; without, the whole expression is walked
    (``_decided_conditionals``). ``memo`` (one per ``values`` and ``conds``)
    keeps the walk's result per subexpression, which the expressions of one
    plan share (their CT scalars' bodies)."""

    from symbolica import AtomType, E

    from pyamplicol._internal.physics.symbols import symbols

    keys = frozenset(values)

    def walk(atom: Any, everywhere: bool) -> Any:
        if memo is None:
            return walked(atom, everywhere)
        found = memo.get((atom, everywhere))
        if found is None:
            found = memo[atom, everywhere] = walked(atom, everywhere)
        return found

    def walked(atom: Any, everywhere: bool) -> Any:
        if not everywhere and not keys.intersection(atom.get_all_symbols(False)):
            return atom
        kind = atom.get_type()
        if kind == AtomType.Num:
            return atom
        if kind == AtomType.Var:
            return values.get(atom, atom)
        children = list(atom)
        if kind in (AtomType.Add, AtomType.Mul, AtomType.Pow):
            children = [walk(child, everywhere) for child in children]
            if kind == AtomType.Add:
                return reduce(add, children)
            if kind == AtomType.Mul:
                return reduce(mul, children)
            return children[0] ** children[1]
        name = atom.get_name()
        role = "if" if name == _IF else conds.get(name)
        if role is not None and len(children) == 3:
            condition = walk(children[0], everywhere)
            number = _exact_number(condition if everywhere else walk(condition, True))
            if number is not None:
                if role == "diagnostic":
                    return E("0")
                # if(c, a, b) takes a for c != 0; cond(c, a, b) for c == 0.
                taken = (number != 0) if role == "if" else (number == 0)
                return walk(children[1] if taken else children[2], everywhere)
            children = [condition, *(walk(child, everywhere) for child in children[1:])]
        else:
            children = [walk(child, everywhere) for child in children]
        if name.startswith(_PREDICATE_PREFIX):
            decided = _decided_predicate(name, children)
            if decided is not None:
                return decided
        return symbols.qualified(name)(*children)

    return walk(expression, not keys)


@dataclass(frozen=True)
class _CoefficientPlan:
    """The referenced coefficients (of couplings or CT scalars) of one
    lowering: their scalar plan with the fixed parameters substituted
    (``lazily_fixed``), and the outputs that are exactly zero or not finite
    there."""

    plan: Any
    zero: frozenset[tuple[str, int]]
    nonfinite: frozenset[tuple[str, int]]


def coefficient_plan(
    program: Any,
    outputs: Iterable[tuple[str, int]],
    fixed_parameters: Mapping[str, Any],
    standard: Mapping[int, str] | None = None,
    *,
    widths: Container[str] = (),
) -> _CoefficientPlan:
    """The scalar plan (``scalar_bindings.prepare_scalar_bindings``) of the
    Laurent coefficients ``outputs`` ((name, epsilon power); CT scalars or
    couplings) of a source scalar ``program``, every helper a scalar-library
    function,
    with the generation's ``fixed_parameters`` substituted lazily (the
    model's UFO-standard helpers ``standard``, by default
    ``standard_helpers`` of the program's library, resolved; a ``cond``
    decided lazily): the plan reads only the other external parameters.
    Outputs that vanish identically there are ``zero`` (their references
    are structural zeros); outputs that are not finite, as expressions or in
    binary64 at the model's default card, are ``nonfinite`` (M2: never
    lowered, the rules reading them are refused when a tree of the request
    can hold them). ``widths`` are the complex-mass scheme's run-time width
    names (``_RealArguments``'s zero-width projections)."""

    from dataclasses import replace as replaced
    from decimal import Decimal

    from pyamplicol.loops.scalar_bindings import prepare_scalar_bindings
    from pyamplicol.loops.scalar_frontend import ScalarExpression

    library = program.library
    if standard is None:
        standard = standard_helpers(library)
    free = prepare_scalar_bindings(library.raw, sorted(outputs), program=program)
    # (real, imaginary) decimal strings, as ``ScalarBindingPlan.fixed_parameters``.
    fixed = {
        name: tuple(str(part) for part in value)
        if isinstance(value, tuple)
        else (str(value), "0")
        for name, value in fixed_parameters.items()
        if name in free.external_names
    }
    inputs = dict(zip(free.external_names, free.input_symbols, strict=True))
    substitution = {
        inputs[name]: library.lower(f"complex({Decimal(re)},{Decimal(im)})").value
        for name, (re, im) in fixed.items()
    }
    conds = {}
    if "cond" in standard.values():
        value, diagnostic, _arguments = library.helpers["cond"]
        conds[value.get_name()] = "value"
        conds[diagnostic.get_name()] = "diagnostic"
    kept = [i for i, name in enumerate(free.external_names) if name not in fixed]
    memo: dict[tuple[Any, bool], Any] = {}
    expressions = [
        ScalarExpression(
            lazily_fixed(e.value, substitution, conds, memo),
            lazily_fixed(e.diagnostic, substitution, conds, memo),
        )
        for e in free.expressions
    ]
    zero = frozenset(
        output
        for output, e in zip(free.outputs, expressions, strict=True)
        if e.value.expand() == library.zero
    )
    nonfinite = {
        output
        for output, e in zip(free.outputs, expressions, strict=True)
        if output not in zero
        and (e.value.is_finite() is False or e.diagnostic.is_finite() is False)
    }
    selected = [
        i
        for i, output in enumerate(free.outputs)
        if output not in zero and output not in nonfinite
    ]
    if not selected:
        return _CoefficientPlan(None, zero, frozenset(nonfinite))
    reals = {free.input_symbols[i] for i in kept if free.external_types[i] == "real"}
    real = _RealArguments(
        reals,
        library,
        widths=[
            free.input_symbols[i]
            for i in kept
            if free.external_names[i] in widths and free.input_symbols[i] in reals
        ],
    )
    plan = replaced(
        free,
        external_names=tuple(free.external_names[i] for i in kept),
        external_types=tuple(free.external_types[i] for i in kept),
        input_symbols=tuple(free.input_symbols[i] for i in kept),
        defaults=tuple(free.defaults[i] for i in kept),
        outputs=tuple(free.outputs[i] for i in selected),
        expressions=tuple(
            ScalarExpression(real(e.value), real(e.diagnostic))
            for e in (expressions[i] for i in selected)
        ),
        fixed_parameters=MappingProxyType(fixed),
    )
    nonfinite |= _nonfinite_at_defaults(plan)
    if nonfinite & set(plan.outputs):
        selected = [
            i for i, output in enumerate(plan.outputs) if output not in nonfinite
        ]
        plan = (
            replaced(
                plan,
                outputs=tuple(plan.outputs[i] for i in selected),
                expressions=tuple(plan.expressions[i] for i in selected),
            )
            if selected
            else None
        )
    return _CoefficientPlan(plan, zero, frozenset(nonfinite))


# The helpers whose MadLoop implementations (MG5_aMC's Fortran model library,
# double and quadruple precision) normalise a zero imaginary part of their
# complex arguments, of either sign, to +0 before taking a branch:
# REGLOG, REGLOGP, REGLOGM, REGSQRT and GRREGLOG (``_RealArguments``).
_MADLOOP_NORMALISED_HELPERS = ("reglog", "reglogp", "reglogm", "regsqrt", "grreglog")


class _RealArguments:
    """Branch-cut arguments that are real take the +0 side, MadLoop's.

    MadLoop's helpers ``reglog``, ``reglogp``, ``reglogm``, ``regsqrt`` and
    ``grreglog`` (MG5_aMC's Fortran model functions) replace a zero
    imaginary part of their argument, of either sign, by +0 before taking a
    branch: an argument exactly on the negative real axis takes the side of
    a positive imaginary part. That side matters at real masses (e.g.
    ``regsqrt(MZ^2 (MZ^2 - 4 MW^2))`` or the logarithms of the Higgs
    two-point counterterms, whose values change by O(1) with it), and the
    sign of an exactly real intermediate's zero imaginary part depends on
    the evaluation order (SymJIT's and Symbolica's differ). Every lane takes
    MadLoop's side:

    * an argument of a helper call (the scalar library's functions), a
      logarithm or a fractional power that is a rational function with real
      coefficients of real model inputs (and of real parts) reads it through
      the scalar library's real projection (``real_part``): value unchanged,
      imaginary part +0 in every carrier and evaluator, decided once;
    * any other argument of the five helpers above is normalised at runtime
      as MadLoop does: ``if(imaginary_part(z) == 0, real_part(z), z)``.

    With ``signed`` only branch-cut arguments that are manifestly real and
    not manifestly non-negative (``is_nonnegative``) are projected (those of
    a logarithm, a square root, a fractional power, and of the parameters a
    helper's body reads through one, ``cut_parameters``), and nothing is
    normalised at runtime: the form of the native lane's counterterm and
    field couplings (``native_trees._parameter_program``), whose other
    arguments are off the negative real axis or not exactly real.

    ``widths`` are the run-time width inputs of a complex-mass scheme. A
    branch-cut argument that is not manifestly real but is real wherever
    some of its widths vanish (its rational normal form with those widths
    at zero is manifestly real, ``zero_width_cases``) reads its real
    projection at run time when they are all exactly zero:
    ``if(equal(w1, 0) equal(w2, 0) ..., real_part(z), z)``. Its imaginary
    part is then zero in exact arithmetic, but the evaluator may compute it
    through complex factors of the other widths that cancel only exactly
    (the G_mu UFO's MZ^4 - 4 MT^2 MZ^2 written through cw^2 = MW^2 / MZ^2,
    CM01 at WZ = WT = 0 with WW != 0): its rounding residue, of either sign
    and in every precision, would decide the side of the cut. The test is
    on the exact inputs, the value is unchanged but for that residue, and
    the side is MadLoop's (whose Fortran keeps the zero imaginary part
    exact and takes +0).
    """

    def __init__(
        self,
        reals: Iterable[Any],
        library: Any,
        *,
        signed: bool = False,
        widths: Iterable[Any] = (),
    ):
        from symbolica import Expression

        self.signed = signed
        self.widths = tuple(widths)
        self._zero_width: dict[Any, tuple[tuple[Any, ...], ...]] = {}
        self._nonnegative: dict[Any, bool] = {}
        # helper head name -> (parameters, body), for ``cut_parameters``.
        self._bodies = {
            head.get_name(): body
            for head, body in getattr(library, "_bodies", {}).items()
        }
        self._cuts: dict[str, Container[int]] = {}
        self.reals = frozenset(reals) | {Expression.PI, Expression.E}
        self.real_part = library.predicates["real_part"]
        self.imaginary_part = library.predicates["imaginary_part"]
        self.equal = library.predicates["equal"]
        self.choose = library.choose
        self.zero = library.zero
        self.projections = frozenset(
            f"{_PREDICATE_PREFIX}{name}" for name in ("real_part", "imaginary_part")
        )
        self.normalised = frozenset(
            head.get_name()
            for name in _MADLOOP_NORMALISED_HELPERS
            if name in library.helpers
            for head in library.helpers[name][:2]
        )
        self._real: dict[Any, bool] = {}
        self._done: dict[Any, Any] = {}

    def is_real(self, atom: Any) -> bool:
        from symbolica import AtomType

        found = self._real.get(atom)
        if found is not None:
            return found
        kind = atom.get_type()
        if kind == AtomType.Num:
            found = _exact_number(atom) is not None
        elif kind == AtomType.Var:
            found = atom in self.reals
        elif kind in (AtomType.Add, AtomType.Mul):
            found = all(self.is_real(child) for child in atom)
        elif kind == AtomType.Pow:
            base, exponent = list(atom)
            number = _exact_number(exponent)
            found = (
                number is not None and number.denominator == 1 and self.is_real(base)
            )
        else:
            found = atom.get_name() in self.projections
        self._real[atom] = found
        return found

    def is_nonnegative(self, atom: Any) -> bool:
        """Whether ``atom`` (manifestly real) is manifestly non-negative: a
        non-negative number or pi or e, an even power of a real, a rational
        power of a non-negative, and sums and products of non-negatives."""

        from symbolica import AtomType, Expression

        found = self._nonnegative.get(atom)
        if found is not None:
            return found
        kind = atom.get_type()
        if kind == AtomType.Num:
            number = _exact_number(atom)
            found = number is not None and number >= 0
        elif kind == AtomType.Var:
            found = atom in (Expression.PI, Expression.E)
        elif kind in (AtomType.Add, AtomType.Mul):
            found = all(self.is_nonnegative(child) for child in atom)
        elif kind == AtomType.Pow:
            base, exponent = list(atom)
            number = _exact_number(exponent)
            even = number is not None and number == number.numerator
            even = even and number.numerator % 2 == 0 and self.is_real(base)
            found = even or (number is not None and self.is_nonnegative(base))
        else:
            found = False
        self._nonnegative[atom] = found
        return found

    def cut_parameters(self, name: str) -> Container[int]:
        """The indices of the parameters of the helper ``name`` that its body
        reads through a branch cut: in the argument of a logarithm, a square
        root or a fractional power, or of such a parameter of a helper it
        calls; every parameter of a helper without a known body."""

        found = self._cuts.get(name)
        if found is not None:
            return found
        known = self._bodies.get(name)
        if known is None:
            found = _EVERY
        else:
            self._cuts[name] = frozenset()  # a recursive helper reads none
            parameters, body = known
            arguments = self._cut_arguments(body)
            found = frozenset(
                index
                for index, parameter in enumerate(parameters)
                if any(
                    parameter.to_canonical_string() in _symbol_names(argument)
                    for argument in arguments
                )
            )
        self._cuts[name] = found
        return found

    def _cut_arguments(self, atom: Any) -> list[Any]:
        """The arguments ``atom`` reads through a branch cut (``cut_parameters``)."""

        from symbolica import AtomType

        kind = atom.get_type()
        if kind in (AtomType.Num, AtomType.Var):
            return []
        children = list(atom)
        out = [
            argument for child in children for argument in self._cut_arguments(child)
        ]
        if kind == AtomType.Pow:
            base, exponent = children
            number = _exact_number(exponent)
            if number is None or number.denominator != 1:
                out.append(base)
        elif kind == AtomType.Fn:
            name = atom.get_name()
            if name in ("symbolica::log", "symbolica::sqrt"):
                out.extend(children)
            elif name.startswith("pyamplicol_scalar_model_"):
                cuts = self.cut_parameters(name)
                out.extend(c for i, c in enumerate(children) if i in cuts)
        return out

    def __call__(self, atom: Any) -> Any:
        from symbolica import AtomType

        from pyamplicol._internal.physics.symbols import symbols

        kind = atom.get_type()
        if kind in (AtomType.Num, AtomType.Var):
            return atom
        found = self._done.get(atom)
        if found is not None:
            return found
        children = [self(child) for child in atom]
        if kind == AtomType.Add:
            found = reduce(add, children)
        elif kind == AtomType.Mul:
            found = reduce(mul, children)
        elif kind == AtomType.Pow:
            base, exponent = children
            number = _exact_number(exponent)
            if number is None or number.denominator != 1:
                if self._wrapped(base):
                    base = self.real_part(base)
                else:
                    base = self._at_zero_widths(next(iter(atom)), base)
            found = base**exponent
        else:
            name = atom.get_name()
            if name.startswith("pyamplicol_scalar_model_") or name in (
                "symbolica::log",
                "symbolica::sqrt",
            ):
                normalised = name in self.normalised
                cuts = (
                    self.cut_parameters(name)
                    if self.signed and name.startswith("pyamplicol_scalar_model_")
                    else None
                )
                children = [
                    self._argument(child, normalised, original)
                    if cuts is None or index in cuts
                    else child
                    for index, (child, original) in enumerate(
                        zip(children, atom, strict=True)
                    )
                ]
            found = symbols.qualified(name)(*children)
        self._done[atom] = found
        return found

    def _argument(self, atom: Any, normalised: bool, original: Any = None) -> Any:
        """A branch-cut function's argument on MadLoop's side: the real
        projection of a compound manifestly real one, and of one real at
        vanishing widths where they vanish (``_at_zero_widths``; ``original``
        is the argument before its own arguments were processed); for the
        helpers MadLoop normalises (``normalised``), a runtime +0 for a
        vanishing imaginary part of any other one that is not an exact
        number."""

        from symbolica import AtomType

        if self._wrapped(atom):
            return self.real_part(atom)
        atom = self._at_zero_widths(atom if original is None else original, atom)
        if self.signed:
            return atom
        if normalised and atom.get_type() != AtomType.Num and not self.is_real(atom):
            return self.choose(
                self.equal(self.imaginary_part(atom), self.zero),
                self.real_part(atom),
                atom,
            )
        return atom

    def zero_width_cases(self, atom: Any) -> tuple[tuple[Any, ...], ...]:
        """The minimal sets of ``widths`` at whose zero ``atom`` is real: its
        rational normal form (``together``) with them at zero is manifestly
        real (``_real_at_zero``); none for a manifestly real ``atom``."""

        from symbolica import Expression, Replacement

        found = self._zero_width.get(atom)
        if found is not None:
            return found
        cases: list[tuple[Any, ...]] = []
        present = [w for w in self.widths if atom.contains(w)]
        if present and not self.is_real(atom):
            for size in range(1, len(present) + 1):
                for subset in itertools.combinations(present, size):
                    if any(set(case) <= set(subset) for case in cases):
                        continue
                    zero = atom.replace_multiple(
                        [Replacement(w, Expression.num(0)) for w in subset]
                    )
                    if self._real_at_zero(zero.together()):
                        cases.append(subset)
        found = self._zero_width[atom] = tuple(cases)
        return found

    def _real_at_zero(self, atom: Any) -> bool:
        """``is_real``, with a fractional power of a manifestly non-negative
        real (a pole mass at zero width, (M^2)^(1/2)) real too."""

        from symbolica import AtomType

        kind = atom.get_type()
        if kind == AtomType.Pow:
            base, exponent = list(atom)
            number = _exact_number(exponent)
            if number is not None and number.denominator != 1:
                return self._real_at_zero(base) and self.is_nonnegative(base)
            return number is not None and self._real_at_zero(base)
        if kind in (AtomType.Add, AtomType.Mul):
            return all(self._real_at_zero(child) for child in atom)
        return self.is_real(atom)

    def _at_zero_widths(self, original: Any, atom: Any) -> Any:
        """``atom`` (``original`` with its arguments processed), read through
        its real projection where the widths of a ``zero_width_cases`` set
        are all exactly zero."""

        cases = self.zero_width_cases(original) if self.widths else ()
        for case in reversed(cases):
            condition = reduce(mul, (self.equal(w, self.zero) for w in case))
            atom = self.choose(condition, self.real_part(atom), atom)
        return atom

    def _wrapped(self, atom: Any) -> bool:
        """Whether ``atom`` is a compound manifestly real argument (a sum,
        product or power: numbers and inputs are exact already, and a real
        part is the projection itself)."""

        from symbolica import AtomType

        compound = (AtomType.Add, AtomType.Mul, AtomType.Pow)
        return (
            atom.get_type() in compound
            and self.is_real(atom)
            and not (self.signed and self.is_nonnegative(atom))
        )


def _nonfinite_at_defaults(plan: Any) -> set[tuple[str, int]]:
    """Outputs of ``plan`` whose value or active diagnostic is not finite at
    the model's default parameters in binary64 (the plan's own saved exact
    programs, evaluated as ``scalar_bindings.NativeScalarBindings`` does,
    without failing on the first non-finite one)."""

    from decimal import Decimal

    from pyamplicol.loops.native import load_loop_native

    backend = load_loop_native()
    prepared = plan.prepared
    options = {"arithmetic": "double", "precision": None}
    empty = backend.NativeBatch.from_decimal([], 1, 0, **options)
    arguments = empty
    if prepared.defaults_state is not None:
        defaults = backend.CoefficientEvaluator(prepared.defaults_state, **options)
        arguments = defaults.evaluate_batch(empty).take_columns(
            list(range(0, 2 * len(prepared.external_names), 2))
        )
    computed = (
        backend.CoefficientEvaluator(prepared.values_state, **options)
        .evaluate_batch(arguments)
        .to_decimal()
    )
    return {
        output
        for index, output in enumerate(plan.outputs)
        if not all(
            Decimal(x).is_finite()
            for x in (*computed[2 * index], *computed[2 * index + 1])
        )
    }


def _cell_cancellation(
    unlowered: Mapping[Any, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    weights: Mapping[Any, int],
    vanishes: Any,
) -> tuple[dict[Cell, list[Any]], Any]:
    """The rules of every cell of ``unlowered`` and whether a cell's
    coefficients, weighted by the multiplicities, vanish at every power."""

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

    return members, cancelled


def uncancelled_rules(
    unlowered: Mapping[Any, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    weights: Mapping[Any, int],
    vanishes: Any,
    residual: frozenset = frozenset(),
    lowered: frozenset = frozenset(),
) -> frozenset:
    """Rules (keys of ``unlowered``) whose unlowered cells do not cancel.

    A cell cancels when the sum over its rules of ``weights`` (multiplicities)
    times their coefficients ``vanishes`` at every Laurent power, and none of
    its rules is ``residual``. Every rule of a cell that does not cancel is
    returned, to a fixed point; a cell of ``lowered`` (a Levi-Civita cell,
    ``TreeVocabulary.levi_civita``) makes no rule residual: where it does not
    cancel it is lowered (``uncancelled_cells``).
    """

    members, cancelled = _cell_cancellation(unlowered, weights, vanishes)
    out = {key for key in unlowered if key in residual}
    changed = True
    while changed:
        changed = False
        for cell, keys in members.items():
            if cell in lowered or all(key in out for key in keys):
                continue
            if any(key in out for key in keys) or not cancelled(cell):
                out.update(keys)
                changed = True
    return frozenset(out)


def uncancelled_cells(
    unlowered: Mapping[Any, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    weights: Mapping[Any, int],
    vanishes: Any,
    residual: frozenset = frozenset(),
) -> frozenset[Cell]:
    """The cells of ``unlowered`` that do not cancel: one of their rules is
    ``residual``, or the sum over their rules of ``weights`` times the
    coefficients does not vanish at some Laurent power (``uncancelled_rules``)."""

    members, cancelled = _cell_cancellation(unlowered, weights, vanishes)
    return frozenset(
        cell
        for cell, keys in members.items()
        if any(key in residual for key in keys) or not cancelled(cell)
    )


@dataclass(frozen=True)
class _CellSettlement:
    """The settlement of the cells a lowering leaves out (module docstring,
    "Cancelling cells"; ``lower_model_counterterms``).

    A cell that cancels between the rules of its vertex stays out of the
    lowered model. One that does not cancel makes every rule of it residual
    (to a fixed point, ``uncancelled_rules``), unless it is a Levi-Civita cell
    (``TreeVocabulary.levi_civita``): the tree compiler lowers ``Epsilon``
    (``models.tensors.LEVI_CIVITA``), so the cell's vertex terms join the
    rows of each of its rules that is lowered (incomplete doublets,
    doublet-breaking vetoes, anomalous models; WP3.2 EWI-7b). Cells that
    cancel stay out, so models whose cells all cancel (loop_sm with every
    flavour) lower to the same bytes.
    """

    weights: Mapping[RuleKey, int]
    vanishes: Any
    rules: Mapping[RuleKey, Any]
    vocabulary: TreeVocabulary
    coefficients: CoefficientLowering
    fixed_parameters: Mapping[str, Any]
    multi_trace: bool
    nc: Any
    # The anomalous-veto guard ([U2]): rule -> its Levi-Civita cells that
    # cancel only without the veto (filled here), the test of such a cell
    # (``None`` without a veto), the override and the vetoed |PDG| codes.
    anomalous: dict[RuleKey, set[Cell]] = field(default_factory=dict)
    cancels_unvetoed: Any = None
    allow_anomalous_veto: bool = False
    veto: frozenset[int] = frozenset()

    def __call__(
        self,
        plan: Any,
        unlowered: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]],
        residual: dict[RuleKey, str],
        values: dict[RuleKey, dict[CountertermStructure, Any]],
    ) -> Any:
        """Settle ``unlowered``, updating ``unlowered`` (a lowered cell leaves
        its rules' entries), ``residual`` and ``values`` (the lowered rows)
        in place, and return the coefficient plan (``_referenced_scalars``),
        formed again when lowered cells read references."""

        while (
            self._settle(unlowered, residual, values) and self.coefficients.references
        ):
            kept = frozenset(values)
            plan = _referenced_scalars(
                self.coefficients, values, unlowered, residual, self.fixed_parameters
            )
            if frozenset(values) == kept:
                break
        return plan

    def _settle(
        self,
        unlowered: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]],
        residual: dict[RuleKey, str],
        values: dict[RuleKey, dict[CountertermStructure, Any]],
    ) -> frozenset[RuleKey]:
        """One settlement; the rules whose rows gained a lowered cell."""

        levi_civita = frozenset(
            cell
            for key, cells in unlowered.items()
            for cell in cells
            if all(
                self.vocabulary.levi_civita(entry)
                for entry in self.rules[key].entries
                if entry_cell(entry) == cell
            )
        )
        self._refuse_anomalous(unlowered, residual, values, levi_civita)
        for key in uncancelled_rules(
            unlowered, self.weights, self.vanishes, frozenset(residual), levi_civita
        ):
            if key in values:
                del values[key]
                residual[key] = _uncancelled_reason(
                    {c: v for c, v in unlowered[key].items() if c not in levi_civita}
                )
        lowered: set[RuleKey] = set()
        uncancelled = uncancelled_cells(
            unlowered, self.weights, self.vanishes, frozenset(residual)
        )
        for cell in sorted(uncancelled & levi_civita):
            for key in sorted(
                (k for k, cells in unlowered.items() if cell in cells and k in values),
                key=repr,
            ):
                try:
                    rows = self._cell_rows(self.rules[key], cell)
                except (
                    CountertermCouplingError,
                    NotImplementedError,
                    ValueError,
                ) as error:
                    del values[key]
                    residual[key] = (
                        f"its Levi-Civita cell {cell} does not cancel and its "
                        f"coupling cannot be lowered: {error}"
                    )
                    continue
                merged = dict(values[key])
                for structure, value in rows.items():
                    merged[structure] = (
                        merged[structure] + value if structure in merged else value
                    )
                values[key] = {s: v for s, v in merged.items() if not _is_zero(v)}
                del unlowered[key][cell]
                lowered.add(key)
        return frozenset(lowered)

    def _refuse_anomalous(
        self,
        unlowered: dict[RuleKey, dict[Cell, tuple[str, dict[int, Any]]]],
        residual: dict[RuleKey, str],
        values: dict[RuleKey, dict[CountertermStructure, Any]],
        levi_civita: frozenset[Cell],
    ) -> None:
        """Record the Levi-Civita cells that do not cancel under the veto but
        cancel without it ([U2]) and, without the override, make their rules
        residual, to a fixed point (a rule made residual can keep another of
        its cells from cancelling)."""

        if self.cancels_unvetoed is None:
            return
        tested: set[Cell] = set()
        while True:
            uncancelled = uncancelled_cells(
                unlowered, self.weights, self.vanishes, frozenset(residual)
            )
            found = sorted(
                cell
                for cell in (uncancelled & levi_civita) - tested
                if self.cancels_unvetoed(cell)
            )
            tested.update(uncancelled & levi_civita)
            if not found:
                return
            for cell in found:
                for key in sorted(
                    (k for k, cells in unlowered.items() if cell in cells), key=repr
                ):
                    self.anomalous.setdefault(key, set()).add(cell)
                    if self.allow_anomalous_veto or key not in values:
                        continue
                    entry = next(
                        e for e in self.rules[key].entries if entry_cell(e) == cell
                    )
                    del values[key]
                    residual[key] = (
                        f"its Levi-Civita cell of {entry.vertex_name} (colour "
                        f"{entry.color}, Lorentz structure {entry.lorentz_name}) "
                        "cancels between the rules of "
                        "its vertex with every flavour admitted but not under "
                        f"particle_veto {sorted(self.veto)} (an anomalous veto, "
                        "e.g. one that breaks a quark doublet); refused by default, "
                        f"lowered with {ANOMALOUS_VETO_OVERRIDE}"
                    )
            if self.allow_anomalous_veto:
                return

    def _cell_rows(self, rule: Any, cell: Cell) -> dict[CountertermStructure, Any]:
        """The vertex terms of the entries of one cell of ``rule``
        (``rule_terms``), as ``lower_model_counterterms`` forms a rule's rows."""

        rows: dict[CountertermStructure, Any] = {}
        for entry, structure in rule_terms(rule, self.coefficients.is_zero):
            if entry_cell(entry) != cell:
                continue
            value = self.coefficients(entry, structure.power)
            rows[structure] = rows[structure] + value if structure in rows else value
        if self.multi_trace:
            rows = _without_u1_terms(rows, self.nc)
        if self.nc is not None:
            rows = _graded(rows, self.nc)
        return {s: v for s, v in rows.items() if not _is_zero(v)}


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


def admitted_alternatives(
    rule: Any,
    veto: Iterable[int] = (),
    corrections: Iterable[str] | None = None,
    species_orders: Mapping[int, frozenset[str]] | None = None,
) -> tuple[int, ...]:
    """Indices of the loop-content alternatives of ``rule`` that are admitted.

    The plain tree-level rule (design §6.2): every alternative (the rule's
    UFO ``loop_particles``, |PDG| codes; an empty one is a wildcard) without
    a particle of ``veto`` (|PDG| codes), for a rule of a selected correction
    (``corrections``; ``None`` selects every correction). A UVtree rule also
    needs a selected correction among its own orders. No loop routing is
    consulted: the coupling orders and the exactly-once insertion bound the
    trees the rule enters.

    With ``species_orders`` (|PDG| -> the orders the species is perturbative
    in, MadLoop's ``is_perturbating`` facts: the prepared structure's
    ``perturbative_species_orders``) an alternative is also a loop of a
    selected correction: each of its particles is perturbative in one, as
    every line of the request's loops (the families' eligible species). The
    UFO declares no correction type of a CT vertex, so this is what tells a
    QCD loop from a QED one: under QED corrections the R2 of u u~ a with a
    u-gluon loop (orders QCD^2 QED, the QCD correction of the photon vertex)
    is not admitted, while its u-photon twin of u u~ g is. A species without
    facts is no loop species.
    """

    veto = frozenset(abs(int(pdg)) for pdg in veto)
    if corrections is not None:
        corrections = frozenset(corrections)
        if rule.perturbation_type is not None and (
            rule.perturbation_type not in corrections
        ):
            return ()
        if rule.kind == "UVtree" and not corrections & {n for n, _ in rule.order}:
            return ()

    def selected(alternative: Iterable[int]) -> bool:
        if corrections is None or species_orders is None:
            return True
        return all(
            species_orders.get(abs(int(pdg)), frozenset()) & corrections
            for pdg in alternative
        )

    return tuple(
        index
        for index, alternative in enumerate(rule.alternatives)
        if not veto.intersection(alternative) and selected(alternative)
    )


# ---------------------------------------------------------------------- recursion


@dataclass(frozen=True)
class CountertermRecursion:
    """The CT rules of one loop request in the recursion.

    ``rules`` are the request's rules in the lowered model with their
    admitted alternatives (``admission``) and lowered ``terms``; ``residual``
    maps every other rule the request admits to the reason it is not lowered
    (production refuses such a request when the rule's vertex can enter a
    tree of it, ``ct_closure.CountertermProgram``; the tree-route tests close
    them at sites). ``inert`` maps the rules the request admits that
    contribute nothing and need no lowering to the reason: in no tree of the
    program (ghost legs, orders above its limits, or a vertex no tree of the
    request can hold, ``ct_closure.InsertionReachability``; REQUIREMENTS_ADDENDUM
    J), or every loop-content alternative vetoed.
    """

    model: CountertermModel
    rules: Mapping[str, Any]
    terms: Mapping[str, tuple[tuple[Any, CountertermStructure], ...]]
    admission: Mapping[str, tuple[int, ...]]
    residual: Mapping[str, str]
    born_orders: tuple[Orders, ...]
    loop_orders: tuple[Orders, ...]
    inert: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        model: CountertermModel,
        rules: Mapping[str, Any],
        selection: LoopOrderSelection,
        born_orders: Iterable[Orders],
    ) -> CountertermRecursion:
        """Admit the request's ``rules`` (``admitted_alternatives`` with the
        model's veto and the request's corrections); ``born_orders`` are the
        exact Born amplitude orders."""

        residual: dict[str, str] = {}
        inert: dict[str, str] = {}
        lowered: dict[str, Any] = {}
        terms: dict[str, tuple[tuple[Any, CountertermStructure], ...]] = {}
        admission: dict[str, tuple[int, ...]] = {}
        for name, rule in rules.items():
            key = rule_key(rule)
            admitted = admitted_alternatives(
                rule, model.veto, selection.corrections, model.species_orders
            )
            if key in model.multiplicities:
                # The left-out cells cancel with the multiplicities the
                # lowering assumed: the loop groups of one vertex share its
                # correction type, so they are all admitted or none is.
                cells = model.unlowered.get(key, {})
                lowered[name] = rule
                admission[name] = admitted
                terms[name] = tuple(
                    (None, structure) for structure in model.structures[key]
                ) or tuple(
                    (entry, structure)
                    for entry, structure in rule_terms(rule)
                    if entry_cell(entry) not in cells
                )
            elif admitted and key in model.inert:
                inert[name] = model.residual[key]
            elif admitted:
                residual[name] = model.residual.get(
                    key, "the rule is not in the lowered model"
                )
        return cls(
            model=model,
            rules=lowered,
            terms=terms,
            admission=admission,
            residual=residual,
            born_orders=tuple(sorted({add_orders(o) for o in born_orders})),
            loop_orders=tuple(sorted(selection.loop_orders)),
            inert=inert,
        )

    def with_inert(
        self, rules: Mapping[str, Any], reachable: Any
    ) -> CountertermRecursion:
        """This recursion with every ``residual`` rule (of the request's
        ``rules``) whose vertex ``reachable`` (``rule -> bool``,
        ``ct_closure.InsertionReachability.admits``) finds in no tree of the
        program moved to ``inert`` (REQUIREMENTS_ADDENDUM J): it contributes
        nothing, so it needs no lowering."""

        residual, inert = {}, dict(self.inert)
        for name, reason in self.residual.items():
            if reachable(rules[name]):
                residual[name] = reason
            else:
                inert[name] = f"its vertex enters no tree of the request ({reason})"
        return replace(self, residual=residual, inert=inert)

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
        card: Mapping[str, Any] | None = None,
        numbers: Any = None,
    ) -> dict[str, Any]:
        """Runtime values of the CT multiplicities and slot switches, and with
        a ``card`` the model's reference parameters.

        Each lowered rule of the model gets its number of admitted
        alternatives in this request (0 for rules of other requests). Slot
        switches are ``sign`` for the selected ``slots`` (default: all) and
        0 otherwise; without ``poles`` the pole slots are 0 (finite-only
        mode). These values are integers, exact in every carrier; they are
        set once per process and do not depend on the phase-space point.

        ``card`` (the point's model parameters, decimal strings) with
        ``numbers`` (the carrier, ``assemble._Numbers``) adds the referenced
        CT scalars' coefficients at that card as decimal strings
        (``CountertermModel.coefficient_values``); a model with references
        needs them in every evaluation of its CT program.
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
        if card is not None and self.model.coefficients:
            values.update(
                self.model.coefficient_values(card, numbers.precision.working_digits)
            )
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
    "ANOMALOUS_VETO_OVERRIDE",
    "AnomalousVetoError",
    "CoefficientLowering",
    "CountertermCouplingError",
    "CountertermModel",
    "CountertermRecursion",
    "TreeVocabulary",
    "UnloweredCountertermError",
    "admitted_alternatives",
    "decide_constant",
    "lower_model_counterterms",
    "rule_key",
    "rule_terms",
    "uncancelled_rules",
]
