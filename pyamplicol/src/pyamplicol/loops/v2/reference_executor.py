# SPDX-License-Identifier: 0BSD
"""Python reference executor of the v2 loop schedules F, B and M (design §2.2, §5.1).

The executor dresses the skeletons of one generator into Born-weighted family
numerators N_{f,s}(q), polynomials in the loop momentum in the NinjaRS layout
(``poly_affine``). It runs the three schedules of design §2.2 on the same
contents, kernels and seeds, so they are mutual oracles (O4):

* **B** walks each family's suffix trie (``schedules.SuffixTrie``) in the
  trie's frame. Leaves carry the seed of the skeleton as read in that frame;
  rows are the helicities of the undressed legs, summed as blocks attach.
* **F** walks the prefix graph (``schedules.forward_graph``) from the cut with
  helicity-explicit rows of the attached legs and applies seeds at closure.
* **M** walks the merged state graph (``schedules.merged_graph``) with seeds
  injected at the cut. It needs colour-ordered contents whose seed depends on
  the cut word only (LC, or certified classes at NLC/FC:
  ``ColourSeeds.certified``) and a state graph whose
  every cut-to-closure path is one skeleton; both are checked, and
  ``ReferenceExecutor.run`` falls back to B otherwise (generic classes).

Every dressing step runs in the fused form (X(J) per block row) or the split
form (T = sum O x J per vertex group, then the vertex tensor once per group),
in exact, arbitrary-precision or binary64 arithmetic (``Carrier``). Each run
records exact operation counts (complex multiply-adds) of both forms for every
step, the per-step cheaper form, closures and frame maps; ``numeric=False``
walks the same steps with row sets only and returns the same counts without
evaluating anything (G1 counts for large processes).

Inputs are protocols, so physics and algebra oracles share the walk:

* ``Kinematics`` gives each content's segment in the frame position it is
  dressed at. ``CatalogueKinematics`` lowers contents onto the WP1.1 kernel
  catalogue: ``PositionColour`` contents (WP1.4 seeds) use their own vertex
  colour structure and UFO legs; colour-word contents (``PlanarWords``) sum
  every colour structure of their rule on canonical legs until colour-ordered
  vertex kernels exist (WP2.4).
* ``BlockRows`` gives block currents per (block, word, chirality, helicity
  row); ``SyntheticBlockRows`` draws exact random currents, valid inputs for
  every algebraic identity between schedules, frames and forms.
* ``SeedSource`` gives the seed of a skeleton as read in a frame.
  ``ColourSeeds`` contracts the exact seed table of WP1.4 with Born amplitudes;
  a reading against the loop direction has its own seed class (its vertex
  colour legs are re-assigned like its kinematic legs).

Exact arithmetic: kernels are exact polynomials in their kinematic inputs.
Their coefficients carry model factors; rational ones (parameters with
rational values, integer powers) stay exact, while pi, fractional powers and
parameter-only sums under roots (e.g. the weak mixing angle) are evaluated by
Symbolica and fixed once per model to rational stand-ins (``KernelConstants``)
shared by every kernel. Every identity between schedules, frames,
orientations and forms therefore holds exactly; values agree with the true
numbers to the stand-in precision. The binary64 and arbitrary-precision
carriers read the same polynomials with the same factors in their own
arithmetic.

Leaf weights default to the automorphism weight times the merged-flavour
multiplicity. The skeleton's fermion sign epsilon_s and the i-factors belong
to ``conventions`` (WP1.8), which passes them as ``weight``; block currents
enter as the tree engine produced them (``Block.wick_sign`` is not applied).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations, product
from math import gcd
from typing import Any, Literal, Protocol

from .catalogue import LoopKernelCatalogue, LoopSegmentBinding, LoopSegmentKey
from .colour import (
    ColourAccuracy,
    PositionColour,
    SectorPair,
    SeedColour,
    SeedCombination,
    SeedTable,
    _exact_number,
    build_seed_table,
)
from .families import (
    BlockCurrent,
    FamilyFrame,
    FamilyKey,
    Orders,
    Relabelling,
    add_orders,
    model_data,
    relabelling,
    ring_readings,
)
from .poly_affine import (
    MonomialLayout,
    leg_momentum,
    relabel,
    sequence_relabelling,
    size,
    substitute_work,
)
from .schedules import LoopSchedules, StateGraph, forward_graph, merged_graph
from .skeletons import Content, Skeleton

Row = tuple[tuple[int, int], ...]  # ((leg, helicity), ...) sorted by leg
Form = Literal["fused", "split"]
FORMS: tuple[Form, ...] = ("fused", "split")


def restrict(row: Row, mask: int) -> Row:
    return tuple(item for item in row if mask >> item[0] & 1)


def merge(left: Row, right: Row) -> Row:
    return tuple(sorted(left + right))


# --------------------------------------------------------------------- carriers


class Complex:
    """A complex number over an exact or Decimal real type.

    The exact carrier uses ``Fraction`` parts, the arbitrary-precision carrier
    ``Decimal`` parts (rounded by the active ``decimal`` context). Integers
    mix with both.
    """

    __slots__ = ("imag", "real")

    def __init__(self, real: Any = 0, imag: Any = 0):
        self.real = real
        self.imag = imag

    def __add__(self, other: Any) -> Complex:
        if isinstance(other, Complex):
            return Complex(self.real + other.real, self.imag + other.imag)
        if isinstance(other, int | Fraction | Decimal):
            return Complex(self.real + other, self.imag)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> Complex:
        return Complex(-self.real, -self.imag)

    def __sub__(self, other: Any) -> Complex:
        return self + (-other)

    def __rsub__(self, other: Any) -> Complex:
        return (-self) + other

    def __mul__(self, other: Any) -> Complex:
        if isinstance(other, Complex):
            a, b, c, d = self.real, self.imag, other.real, other.imag
            return Complex(a * c - b * d, a * d + b * c)
        if isinstance(other, int | Fraction | Decimal):
            return Complex(self.real * other, self.imag * other)
        return NotImplemented

    __rmul__ = __mul__

    def conjugate(self) -> Complex:
        return Complex(self.real, -self.imag)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Complex):
            return self.real == other.real and self.imag == other.imag
        if isinstance(other, int | Fraction | Decimal):
            return self.imag == 0 and self.real == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.real, self.imag))

    def __bool__(self) -> bool:
        return bool(self.real) or bool(self.imag)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __abs__(self) -> float:
        return abs(complex(self))

    def __repr__(self) -> str:
        return f"Complex({self.real!r}, {self.imag!r})"


class ExactComplex:
    """The exact carrier's scalar: (re + i im) / den with integer parts.

    Values are reduced only when the denominator grows large, so sums and
    products of numbers with equal or small denominators cost a few integer
    operations instead of the two gcds of every ``Fraction`` operation.
    Equal values compare equal whatever their representation.
    """

    __slots__ = ("den", "im", "re")

    _LIMIT = 2048  # bits of denominator before a reduction

    def __init__(self, re: int, im: int = 0, den: int = 1):
        self.re = re
        self.im = im
        self.den = den

    @classmethod
    def of(cls, real: Fraction, imag: Fraction) -> ExactComplex:
        a, b = real.denominator, imag.denominator
        den = a * b // gcd(a, b)
        return cls(real.numerator * (den // a), imag.numerator * (den // b), den)

    @staticmethod
    def _made(re: int, im: int, den: int) -> ExactComplex:
        if den.bit_length() > ExactComplex._LIMIT:
            common = gcd(gcd(re, im), den)
            if common > 1:
                re, im, den = re // common, im // common, den // common
        return ExactComplex(re, im, den)

    def __add__(self, other: Any) -> ExactComplex:
        if type(other) is ExactComplex:
            d, e = self.den, other.den
            if d == e:
                return ExactComplex(self.re + other.re, self.im + other.im, d)
            common = gcd(d, e)
            f, g = e // common, d // common
            return ExactComplex._made(
                self.re * f + other.re * g, self.im * f + other.im * g, d * f
            )
        if type(other) is int:
            return ExactComplex(self.re + other * self.den, self.im, self.den)
        if isinstance(other, Fraction):
            return self + ExactComplex(other.numerator, 0, other.denominator)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> ExactComplex:
        return ExactComplex(-self.re, -self.im, self.den)

    def __sub__(self, other: Any) -> ExactComplex:
        return self + (-other)

    def __rsub__(self, other: Any) -> ExactComplex:
        return (-self) + other

    def __mul__(self, other: Any) -> ExactComplex:
        if type(other) is ExactComplex:
            a, b, c, d = self.re, self.im, other.re, other.im
            return ExactComplex._made(
                a * c - b * d, a * d + b * c, self.den * other.den
            )
        if type(other) is int:
            return ExactComplex(self.re * other, self.im * other, self.den)
        if isinstance(other, Fraction):
            return self * ExactComplex(other.numerator, 0, other.denominator)
        return NotImplemented

    __rmul__ = __mul__

    @property
    def real(self) -> Fraction:
        return Fraction(self.re, self.den)

    @property
    def imag(self) -> Fraction:
        return Fraction(self.im, self.den)

    def conjugate(self) -> ExactComplex:
        return ExactComplex(self.re, -self.im, self.den)

    def __eq__(self, other: object) -> bool:
        if type(other) is ExactComplex:
            return (
                self.re * other.den == other.re * self.den
                and self.im * other.den == other.im * self.den
            )
        if isinstance(other, int | Fraction):
            return self.im == 0 and self.re == other * self.den
        if isinstance(other, Complex):
            return self.real == other.real and self.imag == other.imag
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.real, self.imag))

    def __bool__(self) -> bool:
        return bool(self.re) or bool(self.im)

    def __complex__(self) -> complex:
        return complex(self.real, self.imag)

    def __abs__(self) -> float:
        return abs(complex(self))

    def __repr__(self) -> str:
        return f"ExactComplex({self.real}, {self.imag})"


def exact(value: Any) -> Complex:
    """An exact complex number from int, Fraction, decimal string or pair."""

    if isinstance(value, Complex | ExactComplex):
        return Complex(Fraction(value.real), Fraction(value.imag))
    if isinstance(value, tuple):
        real, imag = value
        return Complex(Fraction(real), Fraction(imag))
    if (
        hasattr(value, "real")
        and hasattr(value, "imag")
        and not isinstance(value, int | Fraction | Decimal | str)
    ):
        return Complex(Fraction(value.real), Fraction(value.imag))
    return Complex(Fraction(value), Fraction(0))


@dataclass(frozen=True)
class Carrier:
    """Arithmetic of an execution: ``exact``, ``ap`` (Decimal) or ``f64``.

    ``digits`` is the working precision of ``ap`` and, for ``exact``, the
    precision of the rational stand-ins of irrational kernel constants.
    """

    kind: Literal["exact", "ap", "f64"]
    digits: int = 64

    @classmethod
    def exact(cls, digits: int = 64) -> Carrier:
        return cls("exact", digits)

    @classmethod
    def ap(cls, digits: int) -> Carrier:
        return cls("ap", digits)

    @classmethod
    def f64(cls) -> Carrier:
        return cls("f64", 17)

    def __post_init__(self) -> None:
        if self.kind not in ("exact", "ap", "f64") or self.digits < 1:
            raise ValueError(f"unknown carrier {self.kind!r}/{self.digits}")

    def scope(self):
        """The Decimal context of ``ap`` arithmetic (a no-op otherwise)."""

        return localcontext(prec=self.digits) if self.kind == "ap" else nullcontext()

    def real(self, value: Fraction | Decimal) -> Any:
        if self.kind == "exact":
            return Fraction(value)
        if self.kind == "f64":
            return float(value)
        if isinstance(value, Fraction):
            return Decimal(value.numerator) / Decimal(value.denominator)
        return +Decimal(value)

    def number(self, value: Any) -> Any:
        """Cast an exact input (int, Fraction, exact ``Complex``) to the carrier."""

        if value is None:
            return None
        if self.kind == "exact" and type(value) is ExactComplex:
            return value
        value = exact(value)
        if self.kind == "exact":
            return ExactComplex.of(value.real, value.imag)
        if self.kind == "f64":
            return complex(float(value.real), float(value.imag))
        return Complex(self.real(value.real), self.real(value.imag))


# ------------------------------------------------------------- exact kernels

# ("pi", None) | ("param", name) | ("prime", p) | ("expr", (text, symbols)):
# "expr" is a parameter-only factor with no product form (a sum, a function),
# its text and the (symbol, parameter name) pairs it contains.
Factor = tuple[str, Hashable]


def _primes(value: int) -> dict[int, int]:
    out: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            out[divisor] = out.get(divisor, 0) + 1
            value //= divisor
        divisor += 1
    if value > 1:
        out[value] = out.get(value, 0) + 1
    return out


class KernelConstants:
    """Values of the model factors in kernel coefficients, for one carrier.

    ``parameters`` maps model parameter names to exact values. A coefficient
    is a complex rational times integer powers of parameters and primes
    (exact) times irrational factors: pi^k, base^(p/q) with 0 < p/q < 1 and
    parameter-only sums or functions (``Factor``). Symbolica evaluates every
    irrational factor (principal branch) at ``carrier.digits`` plus guard
    digits; the exact carrier replaces each by one rational stand-in, its
    value rounded to ``carrier.digits``, shared by every kernel. Bases a
    fractional power was distributed over must be positive.
    """

    def __init__(self, parameters: Mapping[str, Any], carrier: Carrier):
        self.parameters = {name: exact(value) for name, value in parameters.items()}
        self.carrier = carrier
        self._factors: dict[tuple[Factor, Fraction], Any] = {}
        self._constants: dict[Hashable, Any] = {}

    def _base(self, base: Factor) -> Complex:
        kind, value = base
        if kind == "param":
            if value not in self.parameters:
                raise KeyError(f"model parameter {value!r} has no value")
            return self.parameters[value]
        if kind == "prime":
            return exact(value)
        return exact(self._irrational(base, Fraction(1)))

    def _expression(self, base: Factor) -> Any:
        from symbolica import E, Expression, Replacement

        kind, value = base
        if kind == "pi":
            return Expression.PI
        if kind == "expr":
            text, symbols = value
            return E(text).replace_multiple(
                [
                    Replacement(
                        E(symbol), _number_expression(self._base(("param", name)))
                    )
                    for symbol, name in symbols
                ]
            )
        return _number_expression(self._base(base))

    def _irrational(self, base: Factor, exponent: Fraction) -> Any:
        key = (base, exponent)
        found = self._factors.get(key)
        if found is None:
            from symbolica import E

            power = E(f"{exponent.numerator}/{exponent.denominator}")
            value = (self._expression(base) ** power).evaluate(
                {}, self.carrier.digits + 10
            )
            real, imag = value.real.to_decimal(), value.imag.to_decimal()
            if self.carrier.kind == "exact":
                with localcontext(prec=self.carrier.digits):
                    real, imag = +real, +imag
            found = self.carrier.number(Complex(Fraction(real), Fraction(imag)))
            self._factors[key] = found
        return found

    def constant(self, key: Hashable) -> Any:
        """coefficient x exact factors x irrational factors, in the carrier."""

        found = self._constants.get(key)
        if found is None:
            coefficient, exact_factors, irrational, positive = key
            for base in positive:
                value = self._base(base)
                if value.imag or value.real <= 0:
                    raise ValueError(
                        f"a fractional power was distributed over {base}, "
                        f"which is not positive: {value!r}"
                    )
            value = exact(coefficient)
            for base, power in exact_factors:
                factor = self._base(base)
                for _ in range(abs(power)):
                    value = value * factor if power > 0 else value * _inverse(factor)
            value = self.carrier.number(value)
            for base, exponent in irrational:
                whole, fraction = divmod(exponent, 1)
                if fraction:
                    value = value * self._irrational(base, fraction)
                if whole:
                    unit = self._irrational(base, Fraction(1))
                    for _ in range(abs(int(whole))):
                        value = value * (unit if whole > 0 else _inverse(unit))
            found = self._constants[key] = value
        return found


MassFlips = tuple[tuple[str, int], ...]


def mass_flips(reading: Any, content: Content) -> MassFlips:
    """The mass parameters whose sign the tree reading flips in a content's
    segments, sorted: the outgoing loop line's mass when it is a massive
    fermion (``conventions.TreeReading.mass_signs``), else none. The Python
    reference and the native plan (``plan_export``) both read it here."""

    signs = getattr(reading, "mass_signs", None)
    return tuple(sorted(signs(content.loop_out).items())) if signs else ()


def signed_constants(constants: KernelConstants, flips: MassFlips) -> KernelConstants:
    """``constants`` with the parameters of ``flips`` multiplied by their sign."""

    if not flips:
        return constants
    parameters = dict(constants.parameters)
    for name, sign in flips:
        parameters[name] = sign * parameters[name]
    return KernelConstants(parameters, constants.carrier)


def _number_expression(value: Complex) -> Any:
    from symbolica import E, Expression

    return E(str(value.real)) + E(str(value.imag)) * Expression.I


def _inverse(value: Any) -> Any:
    if isinstance(value, complex):
        return 1 / value
    if isinstance(value, ExactComplex):
        norm = value.re * value.re + value.im * value.im
        return ExactComplex(value.re * value.den, -value.im * value.den, norm)
    norm = value.real * value.real + value.imag * value.imag
    return Complex(value.real / norm, -value.imag / norm)


class ExactKernel:
    """A prepared loop kernel as exact polynomials in its kinematic inputs.

    Each output is a list of terms (constant key, ((input position, power),
    ...)). Model parameters are not inputs: they are part of the constants.

    The terms are in a canonical order (``_term_order``), which both
    executors sum in: Symbolica orders the terms of a sum by its symbols'
    ids, which depend on every symbol created earlier in the interpreter, so
    the expansion's own order (and with it the binary64 rounding of every
    kernel output and the plan's constant numbering) moved with the process
    history (WP2.5).
    """

    def __init__(self, spec: Any):
        from symbolica import E

        self.spec = spec
        self.inputs = tuple(spec.inputs)
        positions = {
            item.symbol: index
            for index, item in enumerate(self.inputs)
            if item.role != "model-parameter"
        }
        parameters = {
            item.symbol: item.model_parameter_name
            for item in self.inputs
            if item.role == "model-parameter"
        }
        self.outputs = tuple(
            tuple(
                sorted(
                    _kernel_terms(E(expression), positions, parameters),
                    key=_term_order,
                )
            )
            for expression in spec.exact_expressions
        )
        self._bound: KernelConstants | None = None
        self._resolved: tuple[tuple[tuple[Any, Any], ...], ...] = ()

    def evaluate(self, values: Sequence[Any], constants: KernelConstants) -> list[Any]:
        # The constants of one ``KernelConstants`` are looked up once: hashing
        # the exact keys dominated repeated evaluations at one point.
        if constants is not self._bound:
            self._resolved = tuple(
                tuple((constants.constant(key), monomial) for key, monomial in terms)
                for terms in self.outputs
            )
            self._bound = constants
        out = []
        for terms in self._resolved:
            total: Any = 0
            for value, monomial in terms:
                for position, power in monomial:
                    factor = values[position]
                    for _ in range(power):
                        value = value * factor
                total = total + value
            out.append(total)
        return out


def _term_order(term: tuple[Hashable, tuple[tuple[int, int], ...]]) -> tuple:
    """The canonical order of a kernel output's terms: by monomial (input
    positions and powers), then by constant key. Both are built from exact
    numbers, parameter names and canonical strings only (``_kernel_terms``),
    so the order does not depend on Symbolica's symbol ids; the kind tag
    leading every factor makes any two keys comparable."""

    key, monomial = term
    return (monomial, key)


def _kernel_terms(expression: Any, positions, parameters):
    """(constant key, monomial) per term of a kernel output, in Symbolica's
    term order (``ExactKernel`` puts them in ``_term_order``).

    A key is (complex rational, exact factors, irrational factors, positive
    bases): integer powers of parameters and primes stay exact; pi, the
    fractional parts of powers and parameter-only sums or functions (kept
    whole) are irrational. Powers of products and nested powers are
    distributed, ``(1/p)^(1/2) -> p^(-1/2)``; that needs every base they pass
    through to be positive, which ``KernelConstants`` checks against the
    values (``positive bases``).
    """

    from symbolica import AtomType

    expanded = expression.expand()
    terms = list(expanded) if expanded.get_type() == AtomType.Add else [expanded]
    for term in terms:
        reading = _TermReading(positions, parameters)
        reading.read(term, Fraction(1), distributed=False)
        exact_factors, irrational = [], []
        for base, exponent in sorted(reading.powers.items()):
            if not exponent:
                continue
            if base[0] in ("pi", "expr"):
                irrational.append((base, exponent))
                continue
            whole, fraction = divmod(exponent, 1)
            if whole:
                exact_factors.append((base, int(whole)))
            if fraction:
                irrational.append((base, fraction))
        key = (
            reading.coefficient,
            tuple(exact_factors),
            tuple(irrational),
            tuple(sorted(reading.positive)),
        )
        yield key, tuple(sorted(reading.monomial.items()))


class _TermReading:
    """One product term of a kernel output, read factor by factor."""

    def __init__(self, positions, parameters):
        self.positions = positions
        self.parameters = parameters
        self.coefficient = (Fraction(1), Fraction(0))
        self.powers: dict[Factor, Fraction] = defaultdict(Fraction)
        self.positive: set[Factor] = set()
        self.monomial: dict[int, int] = defaultdict(int)

    def read(self, atom: Any, exponent: Fraction, distributed: bool) -> None:
        """Add ``atom^exponent``; ``distributed`` marks a fractional power
        already pushed through a product or a power on the way here."""

        from symbolica import AtomType

        kind = atom.get_type()
        if kind == AtomType.Mul:
            fractional = exponent.denominator != 1
            for factor in atom:
                self.read(factor, exponent, distributed or fractional)
            return
        if kind == AtomType.Pow:
            base, power = list(atom)
            if power.get_type() != AtomType.Num:
                raise NotImplementedError(f"symbolic exponent in {atom}")
            power = _exact_number(power)
            if power.imag:
                raise NotImplementedError(f"complex exponent in {atom}")
            fractional = exponent.denominator != 1
            self.read(base, exponent * power.real, distributed or fractional)
            return
        if kind == AtomType.Num:
            self._number(_exact_number(atom), exponent, atom)
            return
        name = atom.to_canonical_string()
        if kind == AtomType.Var and name in self.positions:
            if distributed or exponent.denominator != 1 or exponent < 1:
                raise NotImplementedError(f"kernel not polynomial in {name}")
            self.monomial[self.positions[name]] += int(exponent)
        elif kind == AtomType.Var and name in self.parameters:
            base = ("param", self.parameters[name])
            self.powers[base] += exponent
            if distributed:
                self.positive.add(base)
        elif kind == AtomType.Var and atom.get_name() == "symbolica::𝜋":
            self.powers[("pi", None)] += exponent
        elif kind == AtomType.Var:
            raise NotImplementedError(f"unknown symbol {name} in a kernel")
        else:
            self._opaque(atom, exponent, distributed)

    def _opaque(self, atom: Any, exponent: Fraction, distributed: bool) -> None:
        """A parameter-only sum or function, kept whole (a ``Factor``)."""

        from symbolica import E

        symbols = sorted(
            symbol.to_canonical_string()
            for symbol in atom.get_all_symbols(False)
            if symbol.get_name() != "symbolica::𝜋"
        )
        if any(symbol in self.positions for symbol in symbols):
            raise NotImplementedError(f"kernel not polynomial in {atom}")
        unknown = [symbol for symbol in symbols if symbol not in self.parameters]
        if unknown:
            raise NotImplementedError(f"unknown symbols {unknown} in a kernel")
        if distributed:
            self.positive.add(self._factor(atom, symbols))
        if exponent.denominator != 1:
            atom = atom ** E(f"{exponent.numerator}/{exponent.denominator}")
            exponent = Fraction(1)
        self.powers[self._factor(atom, symbols)] += exponent

    def _factor(self, atom: Any, symbols: Sequence[str]) -> Factor:
        pairs = tuple(
            (symbol, self.parameters[symbol])
            for symbol in symbols
            if symbol in self.parameters
        )
        return ("expr", (atom.to_canonical_string(), pairs))

    def _number(self, number: Any, exponent: Fraction, atom: Any) -> None:
        if exponent.denominator == 1:
            factor = (number.real, number.imag)
            if exponent < 0:
                norm = number.real**2 + number.imag**2
                factor = (number.real / norm, -number.imag / norm)
            for _ in range(abs(int(exponent))):
                self.coefficient = _complex_product(self.coefficient, factor)
            return
        if number.imag or number.real <= 0:
            raise NotImplementedError(f"fractional power of {atom}")
        ratio = number.real
        for prime, count in _primes(ratio.numerator).items():
            self.powers[("prime", prime)] += exponent * count
        for prime, count in _primes(ratio.denominator).items():
            self.powers[("prime", prime)] -= exponent * count


def _complex_product(left, right) -> tuple[Fraction, Fraction]:
    a, b = left
    c, d = right
    return (a * c - b * d, a * d + b * c)


# --------------------------------------------------------------- block rows


class BlockRows(Protocol):
    """Block currents per (block current, word, chirality, helicity row)."""

    def rows(
        self, block: BlockCurrent, word: Hashable
    ) -> Mapping[int, tuple[Row, ...]]:
        """Helicity rows of a block word, per current chirality. ``word``
        is the content colour's name of the block
        (``PositionColour.block_names``): a colour-ordered word, or (word,
        N_c grade) over graded blocks."""
        ...

    def value(
        self,
        block: BlockCurrent,
        word: Hashable,
        chirality: int,
        row: Row,
        dimension: int,
    ) -> Sequence[Any]:
        """The current's components (exact) for one helicity row."""
        ...


class SyntheticBlockRows:
    """Exact random block currents with every helicity row of their legs.

    ``helicities[leg]`` lists the helicities of external leg ``leg`` and
    ``chiralities(field)`` the current chiralities blocks of ``field`` come
    in. Like a tree-engine block, each row has one chirality: the rows of a
    leg set (``row_set`` order) cycle through the chiralities, so a massless
    fermion leg's two helicities are its two chiralities and every chirality
    combination of a loop line occurs. Any current is a valid input of the
    schedule, frame and form identities.
    """

    def __init__(
        self,
        helicities: Sequence[Sequence[int]],
        chiralities: Callable[[str], Sequence[int]] = lambda field: (0,),
        seed: int = 0,
    ):
        self.helicities = tuple(tuple(h) for h in helicities)
        self.chiralities = chiralities
        self.seed = seed
        self._values: dict[Hashable, tuple[Complex, ...]] = {}

    def row_set(self, mask: int) -> tuple[Row, ...]:
        legs = [leg for leg in range(len(self.helicities)) if mask >> leg & 1]
        return tuple(
            tuple(zip(legs, choice, strict=True))
            for choice in product(*(self.helicities[leg] for leg in legs))
        )

    def rows(
        self, block: BlockCurrent, word: Hashable
    ) -> Mapping[int, tuple[Row, ...]]:
        rows = self.row_set(block.mask)
        options = tuple(self.chiralities(block.field))
        split = {c: rows[k :: len(options)] for k, c in enumerate(options)}
        return {chirality: part for chirality, part in split.items() if part}

    def value(self, block, word, chirality, row, dimension):
        key = (block, word, chirality, row, dimension)
        found = self._values.get(key)
        if found is None:
            generator = random.Random(repr((self.seed, key)))
            found = self._values[key] = tuple(
                _random_complex(generator) for _ in range(dimension)
            )
        return found


def _random_complex(generator: random.Random) -> Complex:
    def rational() -> Fraction:
        return Fraction(generator.randint(-40, 40), generator.randint(1, 13))

    return Complex(rational(), rational())


def block_chiralities(
    catalogue: LoopKernelCatalogue,
) -> Callable[[str], tuple[int, ...]]:
    """The chiralities of tree blocks per field, as the catalogue's segments
    accept them: the Weyl chiralities (+-1) of a massless fermion, else the
    full state (0)."""

    found: dict[str, set[int]] = defaultdict(set)
    for binding in catalogue.segments:
        legs = binding.key.block_chiralities
        names = [binding.particles[leg] for leg in binding.attached_legs]
        for name, chirality in zip(names, legs, strict=True):
            found[name].add(chirality)

    def chiralities(field: str) -> tuple[int, ...]:
        weyl = tuple(sorted(c for c in found[field] if c))
        return weyl or (0,)

    return chiralities


# --------------------------------------------------------------- kinematics


Exponents = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class SplitShape:
    """One vertex group of a content's split form.

    ``key`` identifies the vertex tensor (equal keys in one dressing step
    share it); ``width`` is the number of block components; ``entries`` are
    the structural (beta, nu, beta', q exponents) of the tensor.
    """

    key: Hashable
    width: int
    entries: tuple[tuple[int, int, int, Exponents], ...]
    rows: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class ContentShape:
    """Dimensions, block rows and structural kernel entries of a content."""

    loop_in: int
    loop_out: int
    degree: int
    rows: tuple[Row, ...]
    fused: tuple[tuple[int, int, Exponents], ...]
    split: tuple[SplitShape, ...]


class Kinematics(Protocol):
    def shape(self, content: Content) -> ContentShape: ...

    def fused(self, content: Content, shift: int) -> Mapping[Row, Sequence[Any]]:
        """Values aligned with ``shape.fused`` per block row, at the frame
        position after the legs ``shift`` (carrier numbers)."""
        ...

    def split(
        self, content: Content, shift: int
    ) -> Sequence[tuple[SplitShape, Sequence[Any], Mapping[Row, Sequence[Any]]]]:
        """Per vertex group: its shape, tensor values and block vectors."""
        ...

    def matrix(
        self, content: Content, row: Row, momentum: Sequence[Any]
    ) -> Mapping[tuple[int, int], Any]:
        """The segment at a numeric incoming loop-line momentum (oracle O5)."""
        ...


@dataclass(frozen=True)
class _Slot:
    block: BlockCurrent
    word: Hashable
    chirality: int
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class _Lowered:
    """One routing term of a content with fixed block chiralities."""

    key: Hashable
    fused: tuple[LoopSegmentBinding, ...]
    split: tuple[LoopSegmentBinding, ...]
    slots: tuple[_Slot, ...]  # in the attached-leg order of the bindings
    rows: tuple[tuple[Row, tuple[Row, ...]], ...]  # merged row, per-slot rows


class CatalogueKinematics:
    """Content segments from the WP1.1 kernel catalogue.

    ``vertex_terms`` are the compiled model's vertex terms and
    ``native_model`` the native model whose field and rule names the families
    use. ``momenta`` are the external momenta, all incoming (exact).
    ``reading(rule, legs)`` maps the UFO legs of a routing term (loop in, loop
    out, blocks; the colour seeds' incoming reading) to the legs its kernels
    read; physical tree rows need ``conventions.TreeReading``, whose
    ``vertex(rule, legs)`` also names the rule whose kernels are read (the
    conjugate rule of a vertex not closed under conjugation) and its colour
    indices.
    """

    def __init__(
        self,
        catalogue: LoopKernelCatalogue,
        vertex_terms: Iterable[Any],
        native_model: Any,
        blocks: BlockRows,
        momenta: Sequence[Sequence[Any]],
        constants: KernelConstants,
        reading: Callable[[str, tuple[int, ...]], tuple[int, ...]] | None = None,
    ):
        data = model_data(native_model)
        self.catalogue = catalogue
        self.blocks = blocks
        self.constants = constants
        self.reading = reading
        self.carrier = constants.carrier
        self._signed: dict[tuple, KernelConstants] = {}
        self._rules = {rule["name"]: rule for rule in data["vertex_rules"]}
        self._antiname = {p["name"]: p["antiname"] for p in data["particles"]}
        self._terms: dict[tuple[str, int], list[Any]] = defaultdict(list)
        self._indices: dict[tuple[str, Orders], set[int]] = defaultdict(set)
        for term in vertex_terms:
            orders = add_orders(tuple(term.coupling_orders))
            self._terms[(term.vertex, term.color_index)].append(term)
            self._indices[(term.vertex, orders)].add(term.color_index)
        self._segments = {binding.key: binding for binding in catalogue.segments}
        self._kernels: dict[int, ExactKernel] = {}
        self.momenta = tuple(tuple(exact(x) for x in p) for p in momenta)
        self._shapes: dict[Content, ContentShape] = {}
        self._lowered: dict[Content, tuple[_Lowered, ...]] = {}
        self._fused: dict[tuple[Content, int], dict[Row, list[Any]]] = {}
        self._split: dict[tuple[Content, int], list] = {}

    # -- lowering ------------------------------------------------------------

    def _kernel(self, binding: LoopSegmentBinding) -> ExactKernel:
        kernel = self._kernels.get(binding.kernel_id)
        if kernel is None:
            kernel = self._kernels[binding.kernel_id] = ExactKernel(
                self.catalogue.kernels[binding.kernel_id]
            )
        return kernel

    def _routing_terms(self, content: Content):
        """(rule, colour indices, UFO legs, orders, ((block, word), ...))."""

        if isinstance(content.colour, PositionColour):
            state = content.colour
            (term,) = content.terms
            yield (
                state.rule,
                (state.colour_index,),
                state.legs,
                term.vertex.orders,
                tuple(zip(term.blocks, state.block_names, strict=True)),
            )
            return
        if not isinstance(content.colour, tuple):
            raise TypeError(f"content colour {content.colour!r} has no lowering")
        for term in content.terms:
            vertex = term.vertex
            indices = tuple(
                sorted(self._indices[(vertex.rule, add_orders(vertex.orders))])
            )
            legs = canonical_vertex_legs(
                self._rules[vertex.rule]["particles"],
                vertex.loop_in,
                self._antiname[vertex.loop_out],
                tuple(block.field for block in term.blocks),
            )
            words = _split_word(content.colour, term.blocks)
            yield (
                vertex.rule,
                indices,
                legs,
                vertex.orders,
                tuple(zip(term.blocks, words, strict=True)),
            )

    def lowered(self, content: Content) -> tuple[_Lowered, ...]:
        found = self._lowered.get(content)
        if found is not None:
            return found
        out = []
        vertex_reading = getattr(self.reading, "vertex", None)
        for rule, indices, legs, orders, attachments in self._routing_terms(content):
            read = rule
            if vertex_reading is not None:
                # The tree reading may take the conjugate rule's kernels (a
                # vertex not closed under conjugation, ``TreeReading.vertex``).
                read, legs, colour_map = vertex_reading(rule, legs)
                if colour_map is not None:
                    indices = tuple(colour_map[index] for index in indices)
            elif self.reading is not None:
                legs = self.reading(rule, legs)
            vertex_terms = [
                term
                for index in indices
                for term in self._terms[(read, index)]
                if add_orders(tuple(term.coupling_orders)) == add_orders(orders)
            ]
            if not vertex_terms:
                raise ValueError(f"no vertex term of {rule} at {orders}")
            in_leg, out_leg, *block_legs = (leg - 1 for leg in legs)
            attached = sorted(block_legs)
            ordered = [attachments[block_legs.index(leg)] for leg in attached]
            options = [
                sorted(self.blocks.rows(block, word).items()) for block, word in ordered
            ]
            for choice in product(*options):
                chiralities = tuple(chirality for chirality, _rows in choice)
                bindings = {
                    form: tuple(
                        binding
                        for term in vertex_terms
                        if (
                            binding := self._segments.get(
                                LoopSegmentKey(
                                    term.id, in_leg, out_leg, chiralities, form
                                )
                            )
                        )
                        is not None
                    )
                    for form in ("fused", "split")
                }
                if not bindings["fused"]:
                    continue  # a chirality selection rule: the term vanishes
                slots = tuple(
                    _Slot(block, word, chirality, rows)
                    for (block, word), (chirality, rows) in zip(
                        ordered, choice, strict=True
                    )
                )
                rows = tuple(
                    (merge_rows(parts), parts)
                    for parts in product(*(slot.rows for slot in slots))
                )
                key = (
                    read,
                    tuple(indices),
                    legs,
                    chiralities,
                    tuple(slot.block.mask for slot in slots),
                    content.loop_in,
                    content.loop_out,
                )
                out.append(
                    _Lowered(key, bindings["fused"], bindings["split"], slots, rows)
                )
        if not out:
            raise ValueError(f"content {content.label} lowers to no kernel")
        found = self._lowered[content] = tuple(out)
        return found

    def shape(self, content: Content) -> ContentShape:
        found = self._shapes.get(content)
        if found is not None:
            return found
        lowered = self.lowered(content)
        bindings = [binding for term in lowered for binding in term.fused]
        dimensions = {(b.loop_in_dimension, b.loop_out_dimension) for b in bindings}
        if len(dimensions) != 1:
            raise ValueError(f"content {content.label} mixes loop-line dimensions")
        ((loop_in, loop_out),) = dimensions
        degree = max(binding.degree for binding in bindings)
        if degree > content.q_degree:
            raise ValueError(
                f"content {content.label} has kernel degree {degree} above its "
                f"structural q-degree {content.q_degree}"
            )
        fused = sorted(
            {(e.loop_in, e.out, e.exponents) for b in bindings for e in b.entries}
        )
        rows = sorted({row for term in lowered for row, _parts in term.rows})
        split = []
        for term in lowered:
            width = 1
            for binding in term.split[:1]:
                for dimension in binding.block_dimensions:
                    width *= dimension
            entries = sorted(
                {
                    (e.loop_in, _flat(e.blocks, b.block_dimensions), e.out, e.exponents)
                    for b in term.split
                    for e in b.entries
                }
            )
            split.append(
                SplitShape(
                    term.key, width, tuple(entries), tuple(r for r, _ in term.rows)
                )
            )
        found = self._shapes[content] = ContentShape(
            loop_in, loop_out, content.q_degree, tuple(rows), tuple(fused), tuple(split)
        )
        return found

    # -- evaluation ------------------------------------------------------------

    def _constants(self, content: Content) -> KernelConstants:
        """Kernel constants of a content's segments: the reading may flip the
        sign of the outgoing loop line's mass (``mass_flips``)."""

        flips = mass_flips(self.reading, content)
        if not flips:
            return self.constants
        found = self._signed.get(flips)
        if found is None:
            found = self._signed[flips] = signed_constants(self.constants, flips)
        return found

    def _block_vectors(self, slot: _Slot, row: Row, dimension: int) -> list[Any]:
        values = self.blocks.value(
            slot.block, slot.word, slot.chirality, row, dimension
        )
        if len(values) != dimension:
            raise ValueError(
                f"block {slot.block} has {len(values)} components, not {dimension}"
            )
        return [self.carrier.number(value) for value in values]

    def _inputs(self, binding, kernel, shift, slots, parts) -> list[Any]:
        currents: list[Any] = []
        momenta: list[Any] = []
        for slot, part, dimension in zip(
            slots, parts, binding.block_dimensions, strict=True
        ):
            if part is not None:
                currents.extend(self._block_vectors(slot, part, dimension))
            momenta.extend(self._momentum(slot.block.mask))
        values: list[Any] = []
        for item in kernel.inputs:
            if item.role == "loop-momentum":
                values.append(shift[item.component])
            elif item.role == "block-current":
                values.append(currents[item.component])
            elif item.role == "block-momentum":
                values.append(momenta[item.component])
            elif item.role == "model-parameter":
                values.append(None)  # a kernel constant
            else:
                raise ValueError(f"segment kernel input of role {item.role!r}")
        return values

    def _momentum(self, mask: int) -> list[Any]:
        return [self.carrier.number(x) for x in leg_momentum(mask, self.momenta)]

    def fused(self, content: Content, shift: int) -> Mapping[Row, Sequence[Any]]:
        found = self._fused.get((content, shift))
        if found is not None:
            return found
        shape = self.shape(content)
        index = {entry: i for i, entry in enumerate(shape.fused)}
        loop = self._momentum(shift)
        out: dict[Row, list[Any]] = {}
        for term in self.lowered(content):
            for row, parts in term.rows:
                target = out.setdefault(row, [0] * len(index))
                for binding in term.fused:
                    kernel = self._kernel(binding)
                    values = kernel.evaluate(
                        self._inputs(binding, kernel, loop, term.slots, parts),
                        self._constants(content),
                    )
                    for entry, value in zip(binding.entries, values, strict=True):
                        slot = index[(entry.loop_in, entry.out, entry.exponents)]
                        target[slot] = target[slot] + value
        self._fused[(content, shift)] = out
        return out

    def split(self, content: Content, shift: int):
        found = self._split.get((content, shift))
        if found is not None:
            return found
        shape = self.shape(content)
        loop = self._momentum(shift)
        out = []
        for term, group in zip(self.lowered(content), shape.split, strict=True):
            index = {entry: i for i, entry in enumerate(group.entries)}
            tensor = [0] * len(index)
            no_blocks = (None,) * len(term.slots)
            for binding in term.split:
                kernel = self._kernel(binding)
                values = kernel.evaluate(
                    self._inputs(binding, kernel, loop, term.slots, no_blocks),
                    self._constants(content),
                )
                for entry, value in zip(binding.entries, values, strict=True):
                    nu = _flat(entry.blocks, binding.block_dimensions)
                    slot = index[(entry.loop_in, nu, entry.out, entry.exponents)]
                    tensor[slot] = tensor[slot] + value
            dimensions = term.split[0].block_dimensions if term.split else ()
            vectors = {}
            for row, parts in term.rows:
                factors = [
                    self._block_vectors(slot, part, dimension)
                    for slot, part, dimension in zip(
                        term.slots, parts, dimensions, strict=True
                    )
                ]
                vector = []
                for components in product(*factors):
                    value: Any = 1
                    for component in components:
                        value = value * component
                    vector.append(value)
                vectors[row] = vector
            out.append((group, tensor, vectors))
        self._split[(content, shift)] = out
        return out

    def matrix(self, content: Content, row: Row, momentum: Sequence[Any]):
        out: dict[tuple[int, int], Any] = {}
        zero = (0, 0, 0, 0)
        for term in self.lowered(content):
            for merged, parts in term.rows:
                if merged != row:
                    continue
                for binding in term.fused:
                    kernel = self._kernel(binding)
                    values = kernel.evaluate(
                        self._inputs(binding, kernel, momentum, term.slots, parts),
                        self._constants(content),
                    )
                    for entry, value in zip(binding.entries, values, strict=True):
                        if entry.exponents == zero:
                            key = (entry.loop_in, entry.out)
                            out[key] = out.get(key, 0) + value
        return out


def merge_rows(parts: Iterable[Row]) -> Row:
    return tuple(sorted(item for part in parts for item in part))


def _flat(indices: Sequence[int], dimensions: Sequence[int]) -> int:
    flat = 0
    for index, dimension in zip(indices, dimensions, strict=True):
        flat = flat * dimension + index
    return flat


def canonical_vertex_legs(
    particles: Sequence[str], loop_in: str, out_particle: str, attached: Sequence[str]
) -> tuple[int, ...]:
    """First free 1-based UFO leg of each role: loop in, loop out, blocks.

    The convention of ``colour.SeedColour``; a content read against the loop
    direction re-assigns its legs the same way, so kinematics and colour of a
    reading carry the same leg relabelling.
    """

    free = list(range(1, len(particles) + 1))
    legs = []
    for name in (loop_in, out_particle, *attached):
        leg = next((leg for leg in free if particles[leg - 1] == name), None)
        if leg is None:
            raise ValueError(f"vertex {particles} has no free leg for {name}")
        free.remove(leg)
        legs.append(leg)
    if free:
        raise ValueError(f"vertex {particles} has unassigned legs {free}")
    return tuple(legs)


def _split_word(word: tuple, blocks: Sequence[BlockCurrent]) -> tuple[tuple, ...]:
    """The words of a position's blocks in a colour-ordered (letter) word."""

    for order in permutations(range(len(blocks))):
        parts, start = {}, 0
        for index in order:
            mask = blocks[index].mask
            length = 0
            while start + length < len(word) and word[start + length] & mask:
                length += 1
            part = word[start : start + length]
            covered = 0
            for letter in part:
                covered |= letter
            if covered != mask:
                break
            parts[index] = part
            start += length
        else:
            if start == len(word):
                return tuple(parts[i] for i in range(len(blocks)))
    raise ValueError(f"word {word} does not split over {blocks}")


# --------------------------------------------------------------------- seeds


class SeedSource(Protocol):
    columns: int

    def seed(
        self, skeleton: Skeleton, contents: Sequence[Content]
    ) -> tuple[Hashable, Mapping[Row, Sequence[Any] | None]]:
        """(seed key, exact seed per full helicity row and column) of a reading."""
        ...


LeafWeight = Callable[[Skeleton, Sequence[Content]], Any]


def automorphism_weight(skeleton: Skeleton, contents: Sequence[Content]) -> Fraction:
    """Automorphism weight times the merged closed-loop flavour multiplicity."""

    return skeleton.weight * skeleton.multiplicity


def word_seed_key(skeleton: Skeleton, contents: Sequence[Content]) -> Hashable:
    """The cut species and colour word read from the cut (LC seeds, §7.5)."""

    word: tuple = ()
    for content in contents:
        if not isinstance(content.colour, tuple):
            raise TypeError("a word seed needs colour-ordered contents")
        word += content.colour
    return contents[0].loop_in, word


class SyntheticSeeds:
    """Exact random seeds per seed key and helicity row (an algebra oracle).

    ``key`` maps a reading to its seed identity (default: the skeleton's cut
    word, so every reading of a skeleton shares one seed).
    """

    def __init__(
        self,
        rows: Iterable[Row],
        key: Callable[[Skeleton, Sequence[Content]], Hashable] | None = None,
        columns: int = 1,
        seed: int = 0,
        values: bool = True,
    ):
        self.rows = tuple(rows)
        self.key = key or (
            lambda skeleton, contents: word_seed_key(skeleton, skeleton.contents)
        )
        self.columns = columns
        self.seed_value = seed
        self.values = values
        self._cache: dict[Hashable, dict[Row, Any]] = {}

    def seed(self, skeleton, contents):
        key = self.key(skeleton, contents)
        found = self._cache.get(key)
        if found is None:
            if not self.values:
                found = dict.fromkeys(self.rows)
            else:
                generator = random.Random(repr((self.seed_value, key)))
                found = {
                    row: tuple(_random_complex(generator) for _ in range(self.columns))
                    for row in self.rows
                }
            self._cache[key] = found
        return key, found


class ColourSeeds:
    """Born-weighted seeds U0[c](h) = sum_b K[c, b] conj(A0_b(h)) (design §7.3).

    ``born[row]`` maps (basis element, Born order) to the exact Born amplitude
    of one helicity row. Every reading a schedule dresses needs its class:
    build with all of them (``reading_skeletons``).
    """

    columns = 1

    def __init__(
        self,
        table: SeedTable,
        class_of: Mapping[Skeleton, SeedCombination],
        born: Mapping[Row, Mapping[tuple[int, Orders], Any]],
        pairs: Iterable[SectorPair],
        accuracy: ColourAccuracy = "full",
    ):
        self.table = table
        self.class_of = class_of
        pairs = frozenset(pairs)
        self._admitted = {(c, order) for c, order, _b, _o in table.admitted(pairs)}
        self._rows: dict[Row, dict[tuple[int, Orders], Complex]] = {}
        for row, amplitudes in born.items():
            seeds = table.seeds(
                pairs,
                {key: {0: exact(value)} for key, value in amplitudes.items()},
                accuracy,
                cast=lambda cell: Complex(cell.real, cell.imag),
                conjugate=lambda value: value.conjugate(),
            )
            self._rows[row] = seeds

    @classmethod
    def build(
        cls,
        colour: SeedColour,
        readings: Iterable[Skeleton],
        born: Mapping[Row, Mapping[tuple[int, Orders], Any]],
        pairs: Iterable[SectorPair],
        accuracy: ColourAccuracy = "full",
        *,
        compress: bool = False,
    ) -> ColourSeeds:
        table, class_of = build_seed_table(
            colour, list(dict.fromkeys(readings)), compress=compress
        )
        return cls(table, class_of, born, pairs, accuracy)

    def certified(self, skeleton: Skeleton) -> bool:
        """The colour certificate predicate of ``ReferenceExecutor.run``:
        the skeleton's seed uses certified primitive classes only
        (``SeedTable.certified``)."""

        combination = self.class_of.get(skeleton)
        return combination is not None and self.table.certified(combination)

    def seed(self, skeleton, contents):
        reading = _reading(skeleton, contents)
        combination = self.class_of.get(reading)
        if combination is None:
            raise KeyError("reading without a seed class; build with every reading")
        order = skeleton.orders
        for c, _x in combination:
            if (c, order) not in self._admitted:
                raise KeyError(f"no sector pair admits class {c} at {order}")
        # An admitted key absent from a row is a zero colour weight or Born.
        weights = [(c, Complex(x.real, x.imag)) for c, x in combination]
        return (combination, order), {
            row: (sum((w * seeds.get((c, order), 0) for c, w in weights), 0),)
            for row, seeds in self._rows.items()
        }


def _reading(skeleton: Skeleton, contents: Sequence[Content]) -> Skeleton:
    contents = tuple(contents)
    if contents == skeleton.contents:
        return skeleton
    return Skeleton(skeleton.family, contents, skeleton.weight, skeleton.multiplicity)


def reading_skeletons(plan: LoopSchedules) -> list[Skeleton]:
    """Every skeleton of a plan in its F reading and as read by its B trie."""

    out = []
    for schedule in plan.schedules.values():
        for skeleton, ring in schedule.trie.leaves:
            out.append(skeleton)
            out.append(_reading(skeleton, [plan.view.content(s.label) for s in ring]))
    return out


def plan_skeletons(plan: LoopSchedules) -> list[Skeleton]:
    """The skeletons a B plan dresses (the set F and M must dress too)."""

    return [
        skeleton
        for schedule in plan.schedules.values()
        for skeleton, _ring in schedule.trie.leaves
    ]


# ------------------------------------------------------------------- results


@dataclass(frozen=True)
class FamilyNumerator:
    """N_{f,s}(q) per observable column, dense in the layout, in ``frame``.

    ``frame`` is the reading of the family key whose shifts define q; the
    coefficients feed the v2 reducer with ``frame.apply(family)``.
    """

    family: FamilyKey
    orders: Orders
    frame: FamilyFrame
    rank: int
    columns: tuple[tuple[Any, ...], ...]

    @property
    def sequence(self) -> FamilyKey:
        return self.frame.apply(self.family)


@dataclass
class Execution:
    """Family numerators and exact operation counts of one schedule run."""

    schedule: str
    form: Form
    numerators: dict[tuple[FamilyKey, Orders], FamilyNumerator]
    counts: Counter
    fallback: str | None = None


Target = tuple[FamilyFrame, Relabelling | None]


class MScheduleUnavailable(ValueError):
    """Schedule M does not apply to these skeletons (generic fallback: B)."""


# ------------------------------------------------------------------ payloads


@dataclass
class _Payload:
    """An open-loop object: rows -> flat [alpha][beta][column][monomial].

    A ``diagonal`` payload is a seed: one value per (row, column) times the
    identity in (alpha, beta), rank 0. Without numerics rows map to None.
    """

    rows: dict[Row, Any]
    d_alpha: int
    d_beta: int
    rank: int
    diagonal: bool = False


@dataclass
class _Group:
    """Edges of one dressing step that share a split vertex tensor."""

    shape: SplitShape
    d_beta: int
    targets: set = field(default_factory=set)
    tensor: Sequence[Any] | None = None
    buffers: dict = field(default_factory=dict)


@dataclass
class ReferenceExecutor:
    """Runs schedules F, B and M on one set of kinematics and seeds.

    ``momenta`` are the external momenta (all incoming, exact) that define
    frame shifts. ``numeric=False`` walks the same steps with row sets only:
    the counts are those of a numeric run and nothing is evaluated
    (``kinematics`` then only needs ``shape`` and seeds only their rows).
    """

    kinematics: Kinematics
    seeds: SeedSource
    momenta: Sequence[Sequence[Any]] = ()
    layout: MonomialLayout | None = None
    carrier: Carrier = field(default_factory=Carrier.exact)
    form: Form = "fused"
    weight: LeafWeight = automorphism_weight
    numeric: bool = True

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"unknown kernel form {self.form!r}")
        if self.numeric and self.layout is None:
            raise ValueError("numeric execution needs a monomial layout")
        self._momenta = tuple(tuple(exact(x) for x in p) for p in self.momenta)
        self._entry_cache: dict = {}

    # -- public ------------------------------------------------------------------

    def run(
        self,
        plan: LoopSchedules,
        schedule: Literal["B", "F", "M"],
        accuracy: ColourAccuracy = "full",
        certified: Callable[[Skeleton], bool] | None = None,
    ) -> Execution:
        """Run one schedule on the skeletons of ``plan`` in its own frames.

        B accumulates in each family's trie frame, F and M in the F reading
        of the family's first skeleton (``forward_targets``). To compare F
        with B family by family, run F with ``trie_targets``.

        M is admissible at leading colour and, at ``accuracy`` NLC or full,
        only where every skeleton's colour class carries a loop-level
        certificate (``certified``, WP2.4; design §2.2). It falls back to B
        otherwise and where its structural checks fail (``run_m``);
        ``Execution.fallback`` says why.
        """

        if schedule == "B":
            return self.run_b(plan)
        skeletons = plan_skeletons(plan)
        targets = forward_targets(
            skeletons, lambda name: plan.view.catalogue.species(name).mass
        )
        if schedule == "F":
            return self.run_f(skeletons, targets)
        if schedule != "M":
            raise ValueError(f"unknown schedule {schedule!r}")
        try:
            if accuracy != "lc" and not (
                certified is not None and all(map(certified, skeletons))
            ):
                raise MScheduleUnavailable(
                    f"{accuracy} colour: M needs loop-level colour certificates "
                    "of every skeleton class"
                )
            return self.run_m(skeletons, targets)
        except MScheduleUnavailable as error:
            execution = self.run_b(plan)
            execution.fallback = str(error)
            return execution

    def run_b(self, plan: LoopSchedules) -> Execution:
        counts: Counter = Counter()
        out: dict = {}
        with self.carrier.scope():
            for key, schedule in plan.schedules.items():
                self._run_trie(plan, key, schedule.trie, counts, out)
        return Execution("B", self.form, out, counts)

    def run_f(
        self, skeletons: Sequence[Skeleton], targets: Mapping[Skeleton, Target]
    ) -> Execution:
        """Schedule F; ``targets`` gives each skeleton's family frame
        (``trie_targets`` or ``forward_targets``)."""

        counts: Counter = Counter()
        graph = forward_graph(skeletons)
        contents = _edge_contents(skeletons, "F")
        children = defaultdict(list)
        for parent, child, label in graph.edges:
            children[parent].append((child, label))
        accumulators: dict = {}
        with self.carrier.scope():
            stack = [
                (child, label, None, 0)
                for root in graph.entries
                for child, label in sorted(children[root], key=repr)
            ]
            while stack:
                state, label, payload, dressed_mask = stack.pop()
                content = contents[label]
                if payload is None:
                    loop_in = self.kinematics.shape(content).loop_in
                    payload = self._diagonal({(): (1,)}, 1, loop_in)
                dressed = self._dress(
                    [(content, payload)],
                    out_mask=None,
                    shift=dressed_mask,
                    rank_in=payload.rank,
                    columns=1,
                    counts=counts,
                )
                mask = dressed_mask | content.mask
                for skeleton in graph.closures.get(state, ()):
                    self._close_forward(
                        dressed, skeleton, targets[skeleton], counts, accumulators
                    )
                for child, child_label in sorted(children[state], key=repr):
                    stack.append((child, child_label, dressed, mask))
        return Execution("F", self.form, self._finish(accumulators), counts)

    def run_m(
        self, skeletons: Sequence[Skeleton], targets: Mapping[Skeleton, Target]
    ) -> Execution:
        """Schedule M; raises ``MScheduleUnavailable`` where it does not apply.

        Checks the structure only (colour-ordered contents, one seed per
        entry state, one skeleton per path): the colour admissibility of
        ``run`` is the caller's, so algebraic oracles may run any contents.
        """

        counts: Counter = Counter()
        try:
            graph = merged_graph(skeletons)
        except TypeError as error:
            raise MScheduleUnavailable(str(error)) from error
        contents = _edge_contents(skeletons, "M")
        parents = defaultdict(set)
        children = defaultdict(set)
        for parent, child, label in graph.edges:
            parents[child].add((parent, label))
            children[parent].add(child)
        depth = _depths(graph, parents)
        _certify_paths(graph, skeletons, depth, parents)
        roots: dict[int, tuple[Hashable, Any, Skeleton]] = {}
        for skeleton in skeletons:
            key, _rows = self.seeds.seed(skeleton, skeleton.contents)
            weight = self.weight(skeleton, skeleton.contents)
            entry = roots.setdefault(graph.starts[skeleton], (key, weight, skeleton))
            if entry[:2] != (key, weight):
                raise MScheduleUnavailable(
                    "skeletons entering one M state have different seeds or weights"
                )
        for closing in graph.closures.values():
            if len({targets[skeleton] for skeleton in closing}) != 1:
                raise MScheduleUnavailable(
                    "an M closure merges skeletons read in different frames"
                )
        full = _full_mask(skeletons[0]) if skeletons else 0
        masks: dict[int, int] = dict.fromkeys(graph.entries, 0)
        payloads: dict[int, _Payload] = {}
        remaining = {state: len(children[state]) for state in graph.states.values()}
        accumulators: dict = {}
        with self.carrier.scope():
            for root, (_key, weight, skeleton) in roots.items():
                _key, rows = self.seeds.seed(skeleton, skeleton.contents)
                loop_in = self.kinematics.shape(skeleton.contents[0]).loop_in
                payloads[root] = self._diagonal(rows, weight, loop_in)
            for state in sorted(depth, key=depth.__getitem__):
                if state in graph.entries:
                    continue
                edges = []
                shifts = set()
                for parent, label in sorted(parents[state], key=repr):
                    content = contents[label]
                    shifts.add(masks[parent])
                    if masks.setdefault(state, masks[parent] | content.mask) != (
                        masks[parent] | content.mask
                    ):
                        raise AssertionError("an M state with two dressed leg sets")
                    edges.append((content, payloads[parent]))
                if len(shifts) != 1:
                    raise AssertionError("an M state with two dressing shifts")
                payloads[state] = self._dress(
                    edges,
                    out_mask=full ^ masks[state],
                    shift=shifts.pop(),
                    rank_in=max(payload.rank for _content, payload in edges),
                    columns=self.seeds.columns,
                    counts=counts,
                )
                for parent in {parent for parent, _label in parents[state]}:
                    remaining[parent] -= 1
                    if not remaining[parent]:
                        del payloads[parent]
                closing = graph.closures.get(state)
                if closing:
                    payload = payloads.pop(state)
                    polynomial = self._trace(payload, counts)
                    self._accumulate(
                        accumulators,
                        closing[0],
                        targets[closing[0]],
                        polynomial,
                        payload.rank,
                        counts,
                    )
        return Execution("M", self.form, self._finish(accumulators), counts)

    # -- B ----------------------------------------------------------------------

    def _run_trie(self, plan, key, trie, counts, out) -> None:
        before = [0]
        for mask, _mass in trie.frame.apply(key):
            before.append(before[-1] | mask)
        full = before[-1]
        leaves = []
        for skeleton, ring in trie.leaves:
            contents = [plan.view.content(segment.label) for segment in ring]
            _key, rows = self.seeds.seed(skeleton, contents)
            loop_in = self.kinematics.shape(contents[0]).loop_in
            leaves.append(
                self._diagonal(rows, self.weight(skeleton, contents), loop_in)
            )
        values: dict[int, _Payload] = {}
        for index, node in enumerate(trie.nodes):
            edges = []
            for segment, child in node.edges:
                source = leaves[child] if node.depth == 1 else values.pop(child)
                edges.append((plan.view.content(segment.label), source))
            step = {
                "shift": before[node.depth - 1],
                "rank_in": node.rank_in,
                "columns": self.seeds.columns,
                "counts": counts,
            }
            if node.depth < len(trie.family):
                values[index] = self._dress(
                    edges, out_mask=full ^ before[node.depth], **step
                )
                continue
            # Accumulator: close each cut species apart (their lines differ).
            cuts = defaultdict(list)
            for content, source in edges:
                cuts[content.loop_out].append((content, source))
            polynomial = None
            for species in sorted(cuts):
                dressed = self._dress(cuts[species], out_mask=0, **step)
                traced = self._trace(dressed, counts)
                if self.numeric:
                    polynomial = (
                        traced
                        if polynomial is None
                        else _add_columns(polynomial, traced)
                    )
            if self.numeric:
                out[(key, node.orders)] = FamilyNumerator(
                    key, node.orders, trie.frame, trie.rank, _fit(polynomial, trie.rank)
                )

    # -- steps ------------------------------------------------------------------

    def _diagonal(self, rows, weight, dimension: int) -> _Payload:
        if not self.numeric:
            return _Payload(dict.fromkeys(rows), dimension, dimension, 0, True)
        weight = self.carrier.number(weight)
        out = {
            row: [weight * self.carrier.number(value) for value in values]
            for row, values in rows.items()
        }
        return _Payload(out, dimension, dimension, 0, True)

    def _dress(self, edges, *, out_mask, shift, rank_in, columns, counts) -> _Payload:
        """Dress one position for every edge into one payload.

        ``out_mask`` holds the legs whose rows survive (the helicities of the
        position's blocks are summed); ``None`` keeps every row and extends it
        by the block rows (schedule F). Both forms are counted; ``self.form``
        is evaluated.
        """

        shapes = [self.kinematics.shape(content) for content, _payload in edges]
        d_alpha = {payload.d_alpha for _content, payload in edges}
        d_out = {shape.loop_out for shape in shapes}
        diagonal = {payload.diagonal for _content, payload in edges}
        if len(d_alpha) != 1 or len(d_out) != 1 or len(diagonal) != 1:
            raise AssertionError("edges of one dressing step disagree in shape")
        (d_alpha,), (d_out,), (diagonal,) = d_alpha, d_out, diagonal
        rank_out = rank_in + max(shape.degree for shape in shapes)
        c_in, c_out = size(rank_in), size(rank_out)
        if self.numeric:
            self.layout.check(rank_out)
        # Multiply-adds per (row, entry): a seed is diagonal and of rank 0.
        per_entry = columns if diagonal else columns * d_alpha * c_in
        work: Counter = Counter()
        out: dict[Row, Any] = {}
        groups: dict[Hashable, _Group] = {}
        for (content, payload), shape in zip(edges, shapes, strict=True):
            if shape.loop_in != payload.d_beta:
                raise AssertionError("a segment does not fit its open-loop current")
            pairs = _row_pairs(payload, content.mask, out_mask, shape.rows)
            work["fused"] += len(pairs) * len(shape.fused) * per_entry
            width_scale = (
                columns if diagonal else columns * d_alpha * payload.d_beta * c_in
            )
            for split in shape.split:
                group = groups.setdefault(split.key, _Group(split, payload.d_beta))
                rows = set(split.rows)
                for target, _source, block in pairs:
                    if block in rows:
                        work["split-accumulate"] += split.width * width_scale
                        group.targets.add(target)
            for target, _source, _block in pairs:
                out.setdefault(target, None)
            if self.numeric:
                if self.form == "fused":
                    self._fused_edge(
                        content,
                        payload,
                        pairs,
                        shift,
                        out,
                        rank_in,
                        rank_out,
                        d_out,
                        columns,
                    )
                else:
                    self._split_edge(
                        content, payload, pairs, shift, groups, rank_in, columns
                    )
        for group in groups.values():
            work["split-vertex"] += (
                len(group.shape.entries) * per_entry * len(group.targets)
            )
        if self.numeric and self.form == "split":
            self._split_vertices(
                groups, out, rank_in, rank_out, d_alpha, d_out, columns, diagonal
            )
        kind = "seed" if diagonal else "step"
        split_work = work["split-accumulate"] + work["split-vertex"]
        counts[f"fused/{kind}"] += work["fused"]
        counts[f"split-accumulate/{kind}"] += work["split-accumulate"]
        counts[f"split-vertex/{kind}"] += work["split-vertex"]
        counts[f"best/{kind}"] += min(work["fused"], split_work)
        counts["dress-steps"] += 1
        counts["edges"] += len(edges)
        if self.numeric:
            length = d_alpha * d_out * columns * c_out
            for target, values in out.items():
                if values is None:
                    out[target] = [0] * length
        return _Payload(out, d_alpha, d_out, rank_out)

    def _fused_edge(
        self, content, payload, pairs, shift, out, rank_in, rank_out, d_out, columns
    ):
        """O'[h][a][b'] += O[hB+h][a][b] X[hB][b][b'], monomials raised."""

        d_alpha, d_beta = payload.d_alpha, payload.d_beta
        c_in, c_out = size(rank_in), size(rank_out)
        shape = self.kinematics.shape(content)
        entries = self._entries(shape.fused, rank_in)
        segment = self.kinematics.fused(content, shift)
        length = d_alpha * d_out * columns * c_out
        for target, source, block in pairs:
            buffer = out.get(target)
            if buffer is None:
                buffer = out[target] = [0] * length
            source = _padded(source, payload, rank_in)
            for (b, b2, n, products), x in zip(entries, segment[block], strict=True):
                if payload.diagonal:
                    for o in range(columns):
                        slot = ((b * d_out + b2) * columns + o) * c_out + n
                        buffer[slot] = buffer[slot] + source[o] * x
                    continue
                for a in range(d_alpha):
                    for o in range(columns):
                        start = ((a * d_beta + b) * columns + o) * c_in
                        base = ((a * d_out + b2) * columns + o) * c_out
                        for m in range(c_in):
                            slot = base + products[m]
                            buffer[slot] = buffer[slot] + source[start + m] * x

    def _split_edge(self, content, payload, pairs, shift, groups, rank_in, columns):
        """T[g][h][a][b][nu] += O[hB+h][a][b] J[hB][nu] per vertex group g."""

        d_alpha, d_beta = payload.d_alpha, payload.d_beta
        c_in = size(rank_in)
        for split, tensor, vectors in self.kinematics.split(content, shift):
            group = groups[split.key]
            group.tensor = tensor
            width = split.width
            for target, source, block in pairs:
                vector = vectors.get(block)
                if vector is None:
                    continue
                source = _padded(source, payload, rank_in)
                buffer = group.buffers.get(target)
                if buffer is None:
                    length = width * columns
                    if not payload.diagonal:
                        length *= d_alpha * d_beta * c_in
                    buffer = group.buffers[target] = [0] * length
                if payload.diagonal:
                    for nu, j in enumerate(vector):
                        for o in range(columns):
                            slot = nu * columns + o
                            buffer[slot] = buffer[slot] + source[o] * j
                    continue
                for a in range(d_alpha):
                    for b in range(d_beta):
                        for nu, j in enumerate(vector):
                            for o in range(columns):
                                start = ((a * d_beta + b) * columns + o) * c_in
                                base = (
                                    ((a * d_beta + b) * width + nu) * columns + o
                                ) * c_in
                                for m in range(c_in):
                                    slot = base + m
                                    buffer[slot] = buffer[slot] + source[start + m] * j

    def _split_vertices(
        self, groups, out, rank_in, rank_out, d_alpha, d_out, columns, diagonal
    ):
        """O'[h][a][b'] += T[g][h][a][b][nu] Y[g][b][nu][b'], once per group."""

        c_in, c_out = size(rank_in), size(rank_out)
        length = d_alpha * d_out * columns * c_out
        for group in groups.values():
            if group.tensor is None:
                continue
            split, d_beta, width = group.shape, group.d_beta, group.shape.width
            entries = self._entries(split.entries, rank_in)
            nus = [entry[1] for entry in split.entries]
            for target, accumulated in group.buffers.items():
                buffer = out.get(target)
                if buffer is None:
                    buffer = out[target] = [0] * length
                for (b, b2, n, products), nu, y in zip(
                    entries, nus, group.tensor, strict=True
                ):
                    if diagonal:
                        for o in range(columns):
                            slot = ((b * d_out + b2) * columns + o) * c_out + n
                            buffer[slot] = (
                                buffer[slot] + accumulated[nu * columns + o] * y
                            )
                        continue
                    for a in range(d_alpha):
                        for o in range(columns):
                            start = (
                                ((a * d_beta + b) * width + nu) * columns + o
                            ) * c_in
                            base = ((a * d_out + b2) * columns + o) * c_out
                            for m in range(c_in):
                                slot = base + products[m]
                                buffer[slot] = buffer[slot] + accumulated[start + m] * y

    def _entries(self, entries, rank_in):
        """(beta, beta', monomial index, raise column) of structural entries."""

        key = (entries, rank_in)
        found = self._entry_cache.get(key)
        if found is None:
            degree = max(sum(entry[-1]) for entry in entries)
            table = self.layout.product_table(rank_in, degree)
            index = self.layout.index
            found = []
            for entry in entries:
                n = index[entry[-1]]
                found.append((entry[0], entry[-2], n, tuple(row[n] for row in table)))
            self._entry_cache[key] = found
        return found

    def _trace(self, payload: _Payload, counts) -> Any:
        """Close the cut: trace over alpha = beta of a row-free payload."""

        if payload.d_alpha != payload.d_beta:
            raise AssertionError("a closure joins loop lines of different dimension")
        if set(payload.rows) - {()}:
            raise AssertionError("a closure with undressed legs")
        columns, c, d = self.seeds.columns, size(payload.rank), payload.d_alpha
        counts["closure"] += d * columns * c * len(payload.rows)
        if not self.numeric:
            return None
        out = [[0] * c for _ in range(columns)]
        values = payload.rows.get(())
        for a in range(d) if values is not None else ():
            for o in range(columns):
                base = ((a * d + a) * columns + o) * c
                column = out[o]
                for m in range(c):
                    column[m] = column[m] + values[base + m]
        return out

    def _close_forward(self, payload, skeleton, target, counts, accumulators) -> None:
        """acc_f[o] += w sum_h U0[o][h] Tr O[h] (schedule F closure)."""

        _key, rows = self.seeds.seed(skeleton, skeleton.contents)
        columns, c, d = self.seeds.columns, size(payload.rank), payload.d_alpha
        counts["closure"] += len(payload.rows) * d * columns * c
        out = None
        if self.numeric:
            weight = self.carrier.number(self.weight(skeleton, skeleton.contents))
            out = [[0] * c for _ in range(columns)]
            for row, values in payload.rows.items():
                seed = rows.get(row)
                if seed is None:
                    continue
                for o in range(columns):
                    factor = weight * self.carrier.number(seed[o])
                    column = out[o]
                    for a in range(d):
                        base = (a * d + a) * c
                        for m in range(c):
                            column[m] = column[m] + factor * values[base + m]
        self._accumulate(accumulators, skeleton, target, out, payload.rank, counts)

    def _accumulate(self, accumulators, skeleton, target, polynomial, rank, counts):
        frame, relabelled = target
        if relabelled is not None:
            counts["frame-map"] += substitute_work(rank) * self.seeds.columns
            if polynomial is not None:
                momenta = [[self.carrier.number(x) for x in p] for p in self._momenta]
                polynomial = [
                    relabel(column, self.layout, relabelled, momenta)
                    for column in polynomial
                ]
        key = (skeleton.family, skeleton.orders)
        entry = accumulators.setdefault(key, [frame, None])
        if entry[0] != frame:
            raise AssertionError(f"family {skeleton.family} accumulated in two frames")
        if polynomial is not None:
            entry[1] = (
                polynomial if entry[1] is None else _add_columns(entry[1], polynomial)
            )

    def _finish(self, accumulators) -> dict:
        if not self.numeric:
            return {}
        return {
            (family, orders): FamilyNumerator(
                family, orders, frame, _rank(polynomial), tuple(map(tuple, polynomial))
            )
            for (family, orders), (frame, polynomial) in accumulators.items()
        }


# ------------------------------------------------------------------- frames


def skeleton_sequence(skeleton: Skeleton, masses: Callable[[str], Any]) -> FamilyKey:
    """The (leg mask, mass id) ring of a skeleton in its F reading."""

    return tuple(
        (content.mask, masses(content.loop_out)) for content in skeleton.contents
    )


def forward_targets(
    skeletons: Sequence[Skeleton], masses: Callable[[str], Any]
) -> dict[Skeleton, Target]:
    """Accumulate each family in the F reading of its first skeleton.

    ``masses`` gives the mass id of a loop species (the family catalogue's
    ``species(name).mass``).
    """

    out = {}
    first: dict[FamilyKey, FamilyKey] = {}
    for skeleton in skeletons:
        sequence = skeleton_sequence(skeleton, masses)
        reference = first.setdefault(skeleton.family, sequence)
        out[skeleton] = (
            _frame_of(skeleton.family, reference),
            None
            if sequence == reference
            else sequence_relabelling(sequence, reference),
        )
    return out


def trie_targets(plan: LoopSchedules) -> dict[Skeleton, Target]:
    """Map each skeleton's F reading onto its reading in its family's B trie."""

    masses = lambda name: plan.view.catalogue.species(name).mass  # noqa: E731
    out = {}
    for key, schedule in plan.schedules.items():
        trie = schedule.trie
        target = trie.frame.apply(key)
        for skeleton, ring in trie.leaves:
            readings = plan.view.view(skeleton).readings
            start, reflected = list(ring_readings(len(ring)))[readings.index(ring)]
            sequence = skeleton_sequence(skeleton, masses)
            relabelled = relabelling(sequence, FamilyFrame(start, reflected))
            if relabelled.apply(sequence) != target:
                raise AssertionError("a trie ring is not a reading of its skeleton")
            identity = start == 0 and not reflected
            out[skeleton] = (trie.frame, None if identity else relabelled)
    return out


def _frame_of(key: FamilyKey, sequence: FamilyKey) -> FamilyFrame:
    for start, reflected in ring_readings(len(key)):
        frame = FamilyFrame(start, reflected)
        if frame.apply(key) == sequence:
            return frame
    raise ValueError(f"{sequence} is not a reading of family {key}")


# ------------------------------------------------------------------- helpers


def _full_mask(skeleton: Skeleton) -> int:
    mask = 0
    for content in skeleton.contents:
        mask |= content.mask
    return mask


def _row_pairs(payload: _Payload, mask: int, out_mask: int | None, rows: Sequence[Row]):
    """(target row, source values, block row) of one edge.

    With ``out_mask`` the block legs of each source row are summed; without
    (schedule F) every block row extends every source row.
    """

    if out_mask is None:
        return [
            (merge(source_row, block), values, block)
            for source_row, values in payload.rows.items()
            for block in rows
        ]
    available = set(rows)
    out = []
    for source_row, values in payload.rows.items():
        block = restrict(source_row, mask)
        if block in available:
            out.append((restrict(source_row, out_mask), values, block))
    return out


def _padded(values, payload: _Payload, rank_in: int):
    """Payload values at the step's input rank (a structural max-plus bound)."""

    if payload.diagonal or payload.rank == rank_in:
        return values
    have, want = size(payload.rank), size(rank_in)
    out = []
    for start in range(0, len(values), have):
        chunk = values[start : start + have]
        if have > want:
            if any(chunk[want:]):
                raise AssertionError("structural rank truncates a nonzero coefficient")
            out.extend(chunk[:want])
        else:
            out.extend(chunk)
            out.extend([0] * (want - have))
    return out


def _rank(polynomial) -> int:
    rank = 0
    while size(rank) < len(polynomial[0]):
        rank += 1
    return rank


def _fit(polynomial, rank: int) -> tuple[tuple[Any, ...], ...]:
    want = size(rank)
    out = []
    for column in polynomial:
        if len(column) > want:
            if any(column[want:]):
                raise AssertionError("family rank truncates a nonzero coefficient")
            column = column[:want]
        out.append(tuple(column) + (0,) * (want - len(column)))
    return tuple(out)


def _add_columns(left, right):
    out = []
    for a, b in zip(left, right, strict=True):
        if len(a) < len(b):
            a, b = b, a
        column = list(a)
        for i, value in enumerate(b):
            column[i] = column[i] + value
        out.append(column)
    return out


def _edge_contents(skeletons: Sequence[Skeleton], name: str) -> dict[Hashable, Content]:
    """The content of every edge label; a label must name one content."""

    contents: dict[Hashable, Content] = {}
    for skeleton in skeletons:
        for content in skeleton.contents:
            if contents.setdefault(content.label, content) != content:
                raise ValueError(
                    f"schedule {name}: label {content.label} names two contents"
                )
    return contents


def _depths(graph: StateGraph, parents) -> dict[int, int]:
    depth = dict.fromkeys(graph.entries, 0)
    pending = [state for state in graph.states.values() if state not in depth]
    while pending:
        later = []
        for state in pending:
            known = {depth.get(parent) for parent, _label in parents[state]}
            if None in known:
                later.append(state)
                continue
            if len(known) != 1:
                raise AssertionError("an M state with parents at different depths")
            depth[state] = known.pop() + 1
        if len(later) == len(pending):
            raise AssertionError("the M state graph has a cycle")
        pending = later
    return depth


def _certify_paths(graph: StateGraph, skeletons, depth, parents) -> None:
    """Every cut-to-closure path of the merged graph is exactly one skeleton.

    A merged state sums chains, so a path that is no skeleton (a prefix of
    one chain continued by the remainder of another) would be added
    spuriously. Each skeleton is one path, so equal counts suffice. Paths
    are counted over the (parent, label) edges a state sums, parallel edges
    included.
    """

    paths = Counter(dict.fromkeys(graph.entries, 1))
    for state in sorted(depth, key=depth.__getitem__):
        if state not in graph.entries:
            paths[state] = sum(paths[parent] for parent, _label in parents[state])
    closing = sum(paths[state] for state in graph.closures)
    distinct = len(set(skeletons))
    if closing != distinct:
        raise MScheduleUnavailable(
            f"the merged graph has {closing} closing paths for {distinct} skeletons"
        )


# ------------------------------------------------------------------ oracle O5


def direct_family_value(
    executor: ReferenceExecutor,
    plan: LoopSchedules,
    family: FamilyKey,
    orders: Orders,
    q: Sequence[Any],
) -> list[Any]:
    """N_f(q) of a B family at a numeric loop momentum, skeleton by skeleton.

    Every skeleton of the family's trie is read as dressed by B and every
    segment is the kernel evaluated at its numeric loop-line momentum q + V;
    helicities are summed explicitly. No polynomial, raise table or layout is
    used, so this checks the dressing algebra (design O5).
    """

    trie = plan.schedules[family].trie
    carrier = executor.carrier
    kinematics = executor.kinematics
    momenta = executor._momenta
    q = [carrier.number(value) for value in q]
    columns = executor.seeds.columns
    total: list[Any] = [0] * columns
    with carrier.scope():
        for skeleton, ring in trie.leaves:
            if skeleton.orders != orders:
                continue
            contents = [plan.view.content(segment.label) for segment in ring]
            _key, rows = executor.seeds.seed(skeleton, contents)
            weight = carrier.number(executor.weight(skeleton, contents))
            lines = []
            shift = 0
            for content in contents:
                momentum = [
                    a + carrier.number(b)
                    for a, b in zip(q, leg_momentum(shift, momenta), strict=True)
                ]
                lines.append(momentum)
                shift |= content.mask
            dimension = kinematics.shape(contents[0]).loop_in
            for row, seed in rows.items():
                matrix = {(a, a): 1 for a in range(dimension)}
                for content, momentum in zip(contents, lines, strict=True):
                    segment = kinematics.matrix(
                        content, restrict(row, content.mask), momentum
                    )
                    product_: dict[tuple[int, int], Any] = {}
                    for (a, b), x in matrix.items():
                        for (b1, c), y in segment.items():
                            if b1 == b:
                                product_[(a, c)] = product_.get((a, c), 0) + x * y
                    matrix = product_
                trace = sum((v for (a, c), v in matrix.items() if a == c), 0)
                for o in range(columns):
                    total[o] = total[o] + weight * carrier.number(seed[o]) * trace
    return total


__all__ = [
    "FORMS",
    "BlockRows",
    "Carrier",
    "CatalogueKinematics",
    "ColourSeeds",
    "Complex",
    "ContentShape",
    "ExactKernel",
    "Execution",
    "FamilyNumerator",
    "KernelConstants",
    "Kinematics",
    "MScheduleUnavailable",
    "ReferenceExecutor",
    "SeedSource",
    "SplitShape",
    "SyntheticBlockRows",
    "SyntheticSeeds",
    "automorphism_weight",
    "block_chiralities",
    "canonical_vertex_legs",
    "direct_family_value",
    "exact",
    "forward_targets",
    "plan_skeletons",
    "reading_skeletons",
    "trie_targets",
    "word_seed_key",
]
