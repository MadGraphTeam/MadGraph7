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
  of the scalars' Laurent coefficients, which the CT program binds at the
  card from their exact bodies (``coefficient_plan``,
  ``CountertermRecursion.parameter_values``): no CT body is inlined into a
  vertex.
* **Admission** (design §6.2) is the plain tree-level rule, without any
  matching to the loop: a rule of a selected correction enters with every
  loop-content alternative (``rule.alternatives``, the UFO ``loop_particles``)
  that holds no vetoed particle (``particle_veto``, |PDG| codes), and the
  exactly-once constraint is the insertion order. ``M_rule`` is the number
  of those alternatives (``admitted_alternatives``).
* **Not lowered** (``residual`` with the reason): Lorentz structures the
  tree compiler does not normalise (Levi-Civita cells that do not cancel),
  contact vertices without a colour-flow decomposition, the two-point terms
  ``models.ct_line_insertions`` refuses (mixing between spins), couplings
  that are not linear in their referenced CT scalars, and coefficients that
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
  vanishes identically (``ct_sites.vanishes_identically``), e.g. the
  Levi-Civita R2 of a Z with gluons, whose axial couplings cancel between
  the up- and down-type quark loops of complete doublets. The lowered model
  leaves the cell out (``CountertermModel.unlowered``); a veto that breaks a
  doublet makes the cell's rules residual.

The site closure with MadLoop-style witness admission (``ct_sites``) stays a
test oracle (O6): per component, squared order and Laurent coefficient, the
recursion's partials equal the sites of the lowered rules
(``tests/unit/test_loop_v2_ct_recursion.py``, ``test_loop_v2_ct_closure.py``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
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
    ``"im"``) of the epsilon^``power`` coefficient of a referenced CT scalar
    (``CoefficientLowering``)."""

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


def _linear_degree(expression: Any, symbols_: frozenset) -> float:
    """Polynomial degree of ``expression`` in the real symbols ``symbols_``;
    ``conj`` is transparent (it maps a real linear form to a real linear
    form). Infinite when a symbol is in another function's argument, a
    denominator or a non-integer power."""

    from symbolica import AtomType

    kind = expression.get_type()
    if kind == AtomType.Num:
        return 0
    if kind == AtomType.Var:
        return 1 if expression in symbols_ else 0
    children = list(expression)
    if kind == AtomType.Add:
        return max(_linear_degree(child, symbols_) for child in children)
    if kind == AtomType.Mul:
        return sum(_linear_degree(child, symbols_) for child in children)
    if kind == AtomType.Pow:
        base, exponent = children
        degree = _linear_degree(base, symbols_)
        if not degree:
            return float("inf") if _linear_degree(exponent, symbols_) else 0
        number = exponent.to_canonical_string()
        return degree * int(number) if number.isdigit() else float("inf")
    if expression.get_name() == "symbolica::conj" and len(children) == 1:
        return _linear_degree(children[0], symbols_)
    return float("inf") if any(_linear_degree(c, symbols_) for c in children) else 0


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

    **Referenced CT scalars** (WP3.2 EWI-3). A coupling that reads a CT scalar
    (a UFO ``CTParameter``, or a parameter with epsilon) whose body calls a
    helper that is not UFO-standard (the electroweak ``re``, ``recms``,
    ``regsqrt``, ``reglogp``, ``grreglog``, ``arg``, or a rebound ``reglog``)
    is not inlined. The NLOCT couplings are linear in their CT scalars, so the
    epsilon^p coefficient of a coupling ``sum_q eps^q (a_q0 + sum_k a_qk P_k)``
    is ``a_p0 + sum_q sum_k a_qk P_k^(p - q)``: the lowering writes each
    ``P_k^(r)`` as a pair of real external parameters of the lowered model
    (real and imaginary part, ``reference_name``; ``references``), and the CT
    program binds them at the card from the scalars' exact bodies, with every
    helper a function of the model's scalar library
    (``CountertermModel.coefficient_plan``, the scalar program of the Born
    and field couplings): one helper semantics in every lane, each CT body
    evaluated once per card and never expanded into a vertex. A coupling that
    is not linear in its CT scalars raises ``CountertermCouplingError``.
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
        # Referenced CT scalars: reference parameter -> (scalar, epsilon
        # power, "re" | "im"), and the scalar program they belong to.
        self.references: dict[str, tuple[str, int, str]] = {}
        self.program: Any = None
        self._couplings: dict[tuple[int, int], _Referenced | None] = {}
        self._names: dict[int, dict[str, tuple[str, int]]] = {}
        self._scalars: dict[tuple[int, tuple[str, int]], bool | None] = {}
        self._powers: dict[tuple[int, str], frozenset[int] | None] = {}

    def __call__(self, entry: Any, power: int) -> Any:
        referenced = self._referenced(entry)
        if referenced is not None:
            return self._referenced_coefficient(entry, power, referenced)
        return self._finite(self._resolve(entry.coupling.value.coefficient(power)))

    def payload_expression(self, expression: Any) -> Any:
        """An expression in source (``UFO::name``) symbols, e.g. a whole-rule
        coefficient, as a coefficient in the payload's symbols."""

        return self._finite(self._resolve(_payload_symbols(expression)))

    def is_zero(self, entry: Any, power: int) -> bool:
        """Whether the epsilon^``power`` coefficient of ``entry``'s coupling
        vanishes identically: in its referenced form when it reads a
        referenced CT scalar (no CT body is expanded), otherwise in the source
        scalar program (``_coefficient_is_zero``)."""

        referenced = self._referenced(entry)
        if referenced is None:
            return _coefficient_is_zero(entry, power)
        return _is_zero(self._referenced_coefficient(entry, power, referenced))

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

    # -- referenced CT scalars ---------------------------------------------------

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


# Decimal digits of the referenced CT scalars' coefficients at least (those of
# a double-double input), and the cards whose values a model keeps.
_COEFFICIENT_DIGITS = 32
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

    ``coefficients`` maps every reference parameter of the lowered model
    (``CoefficientLowering``: a part of a referenced CT scalar's Laurent
    coefficient) to (scalar name, epsilon power, ``"re"`` or ``"im"``);
    ``coefficient_plan`` (a ``scalar_bindings.ScalarBindingPlan`` of those
    (scalar, power) outputs, every helper a scalar-library function, the
    generation's fixed parameters substituted) binds them at the card
    (``coefficient_values``).
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
        the scalar library evaluates the CT scalars' exact bodies at
        ``max(digits, 32)`` digits (a carrier rounds them once, as it rounds
        its couplings). A value or active diagnostic that is not finite
        raises ``scalar_bindings.SourceBindingFailure`` (fail closed)."""

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
                plan, arithmetic="arbitrary", precision=digits, parameters=dict(inputs)
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
    """

    from symbolica import E

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
        if reason is None and not admitted_alternatives(
            rule, veto, corrections, species_orders
        ):
            reason = "no admitted loop content (particle_veto or correction)"
            inert.add(key)
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
                        if coefficients.is_zero(entry, power):
                            continue
                        value = coefficients(entry, power)
                        powers[power] = (
                            powers[power] + value if power in powers else value
                        )
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

    def vanishes(expression: Any) -> bool:
        return vanishes_identically(compiled, expression)

    # Unlowered cells must cancel with the rules' multiplicities; otherwise
    # every rule of the cell is residual (refused by a request admitting it).
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
            weights={key: weights[key] for key in unlowered},
            inert=frozenset(inert),
            veto=veto,
            species_orders=species_orders,
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
    """The coefficient plan of the referenced CT scalars the lowered rows
    ``values`` (and their left-out cells) read, at the generation's fixed
    parameters (``coefficient_plan``), or ``None`` when they read none.

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
                "its counterterm scalars "
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
    (``_decided_conditionals``)."""

    from symbolica import AtomType, E

    from pyamplicol._internal.physics.symbols import symbols

    keys = frozenset(values)

    def walk(atom: Any, everywhere: bool) -> Any:
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
    """The referenced CT scalars' coefficients of one lowering: their scalar
    plan with the fixed parameters substituted (``lazily_fixed``), and the
    outputs that are exactly zero or not finite there."""

    plan: Any
    zero: frozenset[tuple[str, int]]
    nonfinite: frozenset[tuple[str, int]]


def coefficient_plan(
    program: Any,
    outputs: Iterable[tuple[str, int]],
    fixed_parameters: Mapping[str, Any],
    standard: Mapping[int, str] | None = None,
) -> _CoefficientPlan:
    """The scalar plan (``scalar_bindings.prepare_scalar_bindings``) of the
    CT scalars' Laurent coefficients ``outputs`` ((name, epsilon power)) of
    a source scalar ``program``, every helper a scalar-library function,
    with the generation's ``fixed_parameters`` substituted lazily (the
    model's UFO-standard helpers ``standard``, by default
    ``standard_helpers`` of the program's library, resolved; a ``cond``
    decided lazily): the plan reads only the other external parameters.
    Outputs that vanish identically there are ``zero`` (their references
    are structural zeros); outputs that are not finite, as expressions or in
    binary64 at the model's default card, are ``nonfinite`` (M2: never
    lowered, the rules reading them are refused when a tree of the request
    can hold them)."""

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
    expressions = [
        ScalarExpression(
            lazily_fixed(e.value, substitution, conds),
            lazily_fixed(e.diagnostic, substitution, conds),
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
    real = _RealArguments(
        {free.input_symbols[i] for i in kept if free.external_types[i] == "real"},
        library,
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
    """

    def __init__(self, reals: Iterable[Any], library: Any):
        from symbolica import Expression

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
            if (number is None or number.denominator != 1) and self._wrapped(base):
                base = self.real_part(base)
            found = base**exponent
        else:
            name = atom.get_name()
            if name.startswith("pyamplicol_scalar_model_") or name in (
                "symbolica::log",
                "symbolica::sqrt",
            ):
                normalised = name in self.normalised
                children = [self._argument(child, normalised) for child in children]
            found = symbols.qualified(name)(*children)
        self._done[atom] = found
        return found

    def _argument(self, atom: Any, normalised: bool) -> Any:
        """A branch-cut function's argument on MadLoop's side: the real
        projection of a compound manifestly real one; for the helpers
        MadLoop normalises (``normalised``), a runtime +0 for a vanishing
        imaginary part of any other one that is not an exact number."""

        from symbolica import AtomType

        if self._wrapped(atom):
            return self.real_part(atom)
        if normalised and atom.get_type() != AtomType.Num and not self.is_real(atom):
            return self.choose(
                self.equal(self.imaginary_part(atom), self.zero),
                self.real_part(atom),
                atom,
            )
        return atom

    def _wrapped(self, atom: Any) -> bool:
        """Whether ``atom`` is a compound manifestly real argument (a sum,
        product or power: numbers and inputs are exact already, and a real
        part is the projection itself)."""

        from symbolica import AtomType

        compound = (AtomType.Add, AtomType.Mul, AtomType.Pow)
        return atom.get_type() in compound and self.is_real(atom)


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


def uncancelled_rules(
    unlowered: Mapping[Any, Mapping[Cell, tuple[str, Mapping[int, Any]]]],
    weights: Mapping[Any, int],
    vanishes: Any,
    residual: frozenset = frozenset(),
) -> frozenset:
    """Rules (keys of ``unlowered``) whose unlowered cells do not cancel.

    A cell cancels when the sum over its rules of ``weights`` (multiplicities)
    times their coefficients ``vanishes`` at every Laurent power, and none of
    its rules is ``residual``. Every rule of a cell that does not cancel is
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
