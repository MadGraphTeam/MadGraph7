# SPDX-License-Identifier: 0BSD
"""The exact helicity-zero filter of loop-induced processes (M-LIT LIT-R2;
RULED Q10 = (a), Q13; m-lit DECISIONS 2026-11-06 ~09:00 (2) and ~13:45).

A helicity row h is an *integrand zero* when every family numerator
N_{f,l,h,t}(q) of the row (every family f, loop order l and basis element t)
and its counterterm ket vanish identically. Such a row contributes exactly
nothing, so it may be dropped to save time; rows that vanish only after the
integration (J_z selection rules, Landau-Yang) stay live (PLAN §1.5). The
filter never declares a process zero (Q13): if it would drop every row, the
rows stay live and its verdict is a diagnostic.

**Binary64 pre-screen (keep only).** One binary64 evaluation with the
reduction re-test and the term scale at a generation point: a row whose
amplitude exceeds ``ESTIMATE_FACTOR`` (the x10 rule) times its own estimate
(``induced.amplitude_estimates``: the re-test and the row's term floor) at
some loop order and Laurent coefficient is live; the screen never drops a
row.

**Exact test (prime field).** The remaining rows are tested exactly: the
plan's family numerators are dressed natively over F_p, p = 2^61 - 1
(``rusticol_core::loop_v2::prime``), at ``k`` random points of the
dressing's inputs: every block-current component, the external momenta
(momentum conservation kept: the last leg balances the others) and every
model factor of the kernel constants (the constants' exact factored form,
``reference_executor.KernelConstants``: a complex rational coefficient times
powers of parameters, primes, pi and parameter functions; each base b takes
a random u_b with b^e = u_b^(e L_b), L_b the common denominator of its
exponents, so equal constants stay equal; a prime base takes its square
root modulo p where it has one). Each numerator coefficient is then a
Laurent polynomial in these values with exact rational coefficients; times
a monomial clearing every base's negative powers (nonzero, every u_b being
drawn nonzero) it is a polynomial of total degree at most ``degree_bound``
(``d``, from the plan: a family's positions times the largest kernel term
degree of its contents, every input power, every positive constant
exponent and a split term's block components counted, plus every base's
largest negative exponent), so a row that is not an integrand zero
vanishes at one random point with probability at most d / (p - 1)
(Schwartz-Zippel); ``k`` is the least count with rows x (d / (p - 1))^k at
most the stated failure probability ``p_fail`` (both recorded). A zero
relying on relations among these values that the random point does not
keep (on-shell conditions, spinor identities of the tree currents) is not
seen: the row stays live, which costs time, never correctness.

**Counterterm kets.** The engine has no exact tree program, so a row's
counterterm ket counts as zero only where the request has no counterterm
ket at all (``native_cells.counterterm_tensors``, structural); otherwise
the rows the exact test finds zero stay live (recorded).

A dropped row keeps its (row, t) columns without a row (``export_plan``):
its trace writes nothing, its columns stay zero and are never reduced.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

# The field of the exact test (``rusticol_core::loop_v2::prime::PRIME``).
PRIME = (1 << 61) - 1
# The stated failure probability of the exact test (overridable per call).
FAILURE_PROBABILITY = 1e-12


@dataclass(frozen=True)
class FilterVerdict:
    """The filter's verdict for one process; ``record`` is what an artifact
    stores as its generation filter (the timings stay out of it)."""

    rows: int
    live_by_screen: int
    tested: tuple[str, ...]
    integrand_zero: tuple[str, ...]
    dropped: tuple[str, ...]
    every_row_zero: bool
    counterterms: str
    prime: int
    points: int
    degree_bound: int
    failure_probability: float
    failure_bound: float
    seconds: Mapping[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "live_by_screen": self.live_by_screen,
            "tested": list(self.tested),
            "integrand_zero": list(self.integrand_zero),
            "dropped": list(self.dropped),
            "every_row_zero": self.every_row_zero,
            "counterterms": self.counterterms,
            "prime": self.prime,
            "points": self.points,
            "degree_bound": self.degree_bound,
            "failure_probability": self.failure_probability,
            "failure_bound": self.failure_bound,
        }


def row_label(row: Any) -> str:
    """A helicity row as text: its (leg, helicity) pairs."""

    return " ".join(f"{leg}:{helicity:+d}" for leg, helicity in row)


# ----------------------------------------------------------- degree bound


def _bases(constants: Sequence[Any]) -> dict[Any, int]:
    """Per base (and mass-flip sign) the common denominator of its
    exponents over the plan's constants."""

    lcm: dict[Any, int] = {}
    for key, flips in constants:
        _coefficient, exact_factors, irrational, _positive = key
        signs = dict(flips)
        for base, power in (*exact_factors, *irrational):
            name = _base_name(base, signs, Fraction(power))
            denominator = Fraction(power).denominator
            lcm[name] = math.lcm(lcm.get(name, 1), denominator)
    return lcm


def _base_name(base: Any, signs: Mapping[str, int], power: Fraction) -> Any:
    """The variable of a base: its own, or, for a flipped parameter under a
    fractional power, its own flipped one (an independent variable)."""

    kind, value = base
    if kind == "param" and signs.get(value, 1) < 0 and power.denominator != 1:
        return (base, -1)
    return (base, 1)


def _constant_exponents(
    key: Any, flips: Any, lcm: Mapping[Any, int]
) -> tuple[int, dict[Any, int]]:
    """A constant as a Laurent monomial in the base variables u_b
    (b^e = u_b^(e L_b)): the sum of its positive exponents, and per base the
    size of its negative exponent."""

    _coefficient, exact_factors, irrational, _positive = key
    signs = dict(flips)
    positive = Fraction(0)
    negative: dict[Any, Fraction] = {}
    for base, power in (*exact_factors, *irrational):
        power = Fraction(power)
        name = _base_name(base, signs, power)
        scaled = power * lcm[name]
        if scaled > 0:
            positive += scaled
        elif scaled < 0:
            negative[name] = negative.get(name, Fraction(0)) - scaled
    return int(positive), {name: int(size) for name, size in negative.items()}


def degree_bound(exported: Any) -> int:
    """A bound of the total degree of every numerator coefficient in the
    exact test's random values, as a polynomial once every base's negative
    powers are cleared.

    A kernel term is its constant (a Laurent monomial in the bases,
    ``_constant_exponents``) times a monomial in the inputs (block currents,
    block momenta, loop momenta: a frame shift makes each loop power a
    polynomial of that degree in the momenta); a split vertex term also
    multiplies one block-current component per slot outside its kernels. A
    family's numerator multiplies one term per position, so with ``n`` the
    largest family's positions, ``P`` the largest term degree counting
    positive powers only and ``N_b`` the largest negative power of base b,
    the numerator times prod_b u_b^(n N_b) is a polynomial of degree at most
    ``n (P + sum_b N_b)``; it vanishes exactly where the numerator does,
    every u_b being drawn nonzero."""

    program = exported.program
    constants = exported.constants
    lcm = _bases(constants)
    exponents = [_constant_exponents(key, flips, lcm) for key, flips in constants]
    kernels: list[tuple[int, dict[Any, int]]] = []
    for kernel in program["kernels"]:
        top = 0
        cleared: dict[Any, int] = {}
        for terms in kernel["outputs"]:
            for constant, monomial in terms:
                positive, negative = exponents[constant]
                top = max(top, positive + sum(power for _input, power in monomial))
                for name, size in negative.items():
                    cleared[name] = max(cleared.get(name, 0), size)
        kernels.append((top, cleared))
    top = 0
    cleared = {}
    for record in program["contents"]:
        for term in record["terms"]:
            slots = len(term["slots"])
            for extra, bindings in ((0, term["bindings"]), (slots, term["split"])):
                for binding in bindings:
                    degree, negative = kernels[binding["kernel"]]
                    top = max(top, degree + extra)
                    for name, size in negative.items():
                        cleared[name] = max(cleared.get(name, 0), size)
    positions = max(
        (len(family["sequence"]) for family in program["families"]), default=0
    )
    return max(1, positions * (top + sum(cleared.values())))


def points_needed(degree: int, rows: int, failure: float) -> int:
    """The least k with rows x (degree / (p - 1))^k <= failure."""

    ratio = degree / (PRIME - 1)
    if ratio >= 1:
        raise ValueError("the degree bound reaches the field size")
    k = 1
    while rows * ratio**k > failure:
        k += 1
    return k


# ------------------------------------------------------------ random inputs


def _sqrt_mod(value: int) -> int | None:
    """A square root of ``value`` modulo p (p = 3 mod 4), or None."""

    value %= PRIME
    root = pow(value, (PRIME + 1) // 4, PRIME)
    return root if root * root % PRIME == value else None


class _Values:
    """One random point of the exact test: a value per base variable."""

    def __init__(self, rng: random.Random, lcm: Mapping[Any, int]):
        self.rng = rng
        self.lcm = lcm
        self.values: dict[Any, int] = {}

    def draw(self) -> int:
        return self.rng.randrange(1, PRIME)

    def base(self, name: Any) -> int:
        found = self.values.get(name)
        if found is None:
            (kind, value), _sign = name
            root = None
            if kind == "prime" and self.lcm[name] == 2:
                # A prime under a square root: its own root modulo p, where
                # it has one (then b = u^2 exactly).
                root = _sqrt_mod(int(value))
            found = self.values[name] = root if root is not None else self.draw()
        return found

    def power(self, base: Any, signs: Mapping[str, int], power: Fraction) -> int:
        name = _base_name(base, signs, power)
        kind, value = base
        if kind == "prime" and power.denominator == 1 and self.lcm[name] == 1:
            # A prime under integer powers only: itself.
            u, exponent = int(value), power
        else:
            u, exponent = self.base(name), power * self.lcm[name]
        out = pow(u, int(abs(exponent)), PRIME)
        if exponent < 0:
            out = pow(out, PRIME - 2, PRIME)
        flipped = kind == "param" and signs.get(value, 1) < 0
        if flipped and power.denominator == 1 and int(power) % 2:
            out = (PRIME - out) % PRIME
        return out

    def constant(self, key: Any, flips: Any) -> tuple[int, int]:
        coefficient, exact_factors, irrational, _positive = key
        from . import reference_executor as rx

        exact = rx.exact(coefficient)
        re = _residue(Fraction(exact.real))
        im = _residue(Fraction(exact.imag))
        signs = dict(flips)
        factor = 1
        for base, power in (*exact_factors, *irrational):
            factor = factor * self.power(base, signs, Fraction(power)) % PRIME
        return re * factor % PRIME, im * factor % PRIME


def _residue(value: Fraction) -> int:
    return (
        value.numerator
        % PRIME
        * pow(value.denominator % PRIME, PRIME - 2, PRIME)
        % PRIME
    )


def random_inputs(executor: Any, rng: random.Random, lcm: Mapping[Any, int]) -> dict:
    """The batch of one random point of the exact test (``NativeExecutor
    ._inputs``' layout, values as residues)."""

    exported = executor.exported
    legs = len(executor.slice.blocks.process.legs)
    values = _Values(rng, lcm)

    def pair() -> list[str]:
        return [str(values.draw()), str(values.draw())]

    momenta = [[values.draw() for _mu in range(4)] for _leg in range(legs - 1)]
    momenta.append([(-sum(p[mu] for p in momenta)) % PRIME for mu in range(4)])
    rows = [
        pair()
        for _block, _word, _chirality, _row, dimension in exported.block_rows
        for _component in range(dimension)
    ]
    constants = []
    for key, flips in exported.constants:
        re, im = values.constant(key, flips)
        constants.append([str(re), str(im)])
    return {
        "momenta": [[str(x) for p in momenta for x in p]],
        "scales": ["1"],
        "block_rows": [rows],
        "born": [[["1", "0"]] * len(exported.rows)],
        "constants": constants,
        "masses": [str(values.draw()) for _mass in exported.masses],
        "leg_masses": [str(values.draw()) for _leg in range(legs)],
    }


# ------------------------------------------------------------------- filter


def zero_rows_exactly(
    executor: Any, rows: Sequence[int], points: int, seed: int
) -> tuple[set[int], float]:
    """The rows (indices into ``executor.exported.rows``) whose every
    numerator coefficient vanishes at ``points`` random points over F_p, and
    the seconds taken."""

    exported = executor.exported
    size = exported.amplitude
    lcm = _bases(exported.constants)
    native = executor.native("prime", None, 1)
    rng = random.Random(seed)
    zero = set(rows)
    start = time.perf_counter()
    for _point in range(points):
        if not zero:
            break
        (point,) = native.numerators(random_inputs(executor, rng, lcm), "B")
        for _family, _order, _rank, columns in point:
            for o, column in enumerate(columns):
                row = o // size
                if row in zero and any(re != "0" or im != "0" for re, im in column):
                    zero.discard(row)
    return zero, time.perf_counter() - start


def screen(
    executor: Any, point: Any, scale: Any, parameters: Mapping[str, str]
) -> set[int]:
    """The rows the binary64 pre-screen proves live: ``ESTIMATE_FACTOR`` (the
    x10 rule) times their own estimate at some loop order and Laurent
    coefficient, the estimate of the generation-time probes
    (``induced.probes``: the calibrated re-test and generic rotation of
    ``stability.RECOMMENDED_CHECKS`` and the row term floors): the rotation
    budget covers the rounding of numerators that cancel term by term (a
    Furry zero), whose term floor is rounding itself."""

    from .assemble import Precision, _fraction
    from .induced import (
        ESTIMATE_FACTOR,
        _probe_runs,
        amplitude_estimates,
        physical_floors,
        rotation_errors,
    )
    from .stability import RECOMMENDED_CHECKS

    checks = RECOMMENDED_CHECKS
    precision = Precision.double()
    carrier, (evaluation,) = _probe_runs(
        executor, [(point, None)], scale, parameters, precision, checks.frame
    )
    exact = tuple(tuple(_fraction(x) for x in p) for p in point)
    turned = []
    if checks.rotations:
        _carrier, rotated = _probe_runs(
            executor,
            [(rotation.apply(exact), None) for rotation in checks.rotations],
            scale,
            parameters,
            precision,
            None,
        )
        turned = [rotation_errors(evaluation, run) for run in rotated]
    floors = physical_floors(executor, point, scale, parameters)
    estimates = amplitude_estimates(evaluation, carrier.unit_roundoff, floors, turned)
    index = {row: r for r, row in enumerate(executor.exported.rows)}
    live = set()
    for (_order, row), entries in estimates.items():
        if any(residue > ESTIMATE_FACTOR * estimate for residue, estimate in entries):
            live.add(index[row])
    return live


def filter_rows(
    executor: Any,
    point: Any,
    scale: Any,
    parameters: Mapping[str, str] | None = None,
    *,
    failure: float = FAILURE_PROBABILITY,
    seed: int = 0,
) -> FilterVerdict:
    """The filter's verdict for the plan of a loop-induced ``executor``
    (``plan_export.NativeExecutor``) at a generation ``point``."""

    from .native_cells import counterterm_tensors

    exported = executor.exported
    if exported.amplitude is None:
        raise ValueError("the helicity-zero filter serves loop-induced plans")
    rows = list(exported.rows)
    seconds: dict[str, float] = {}
    start = time.perf_counter()
    live = screen(executor, point, scale, dict(parameters or {}))
    seconds["screen"] = time.perf_counter() - start
    tested = [r for r in range(len(rows)) if r not in live]
    start = time.perf_counter()
    degree = degree_bound(exported)
    seconds["degree"] = time.perf_counter() - start
    k = points_needed(degree, max(len(tested), 1), failure)
    zero: set[int] = set()
    if tested:
        zero, seconds["exact"] = zero_rows_exactly(executor, tested, k, seed)
    counterterms = "present" if counterterm_tensors(executor.slice) else "absent"
    certified = sorted(zero) if counterterms == "absent" else []
    every = bool(rows) and len(certified) == len(rows)
    dropped = [] if every else certified
    ratio = degree / (PRIME - 1)
    return FilterVerdict(
        rows=len(rows),
        live_by_screen=len(live),
        tested=tuple(row_label(rows[r]) for r in tested),
        integrand_zero=tuple(row_label(rows[r]) for r in sorted(zero)),
        dropped=tuple(row_label(rows[r]) for r in dropped),
        every_row_zero=every,
        counterterms=counterterms,
        prime=PRIME,
        points=k,
        degree_bound=degree,
        failure_probability=failure,
        failure_bound=max(len(tested), 1) * ratio**k,
        seconds=seconds,
    )


def generation_filter(
    slice_: Any,
    candidates: Sequence[Sequence[Sequence[float]]],
    *,
    failure: float = FAILURE_PROBABILITY,
) -> tuple[dict[str, Any], frozenset]:
    """The generation-time filter of a loop-induced slice: its verdict at the
    candidate point farthest from every tree pole (lifted to 32 digits,
    ``induced.probe_point``) on the slice's own plan, and the rows to drop
    (``LoopSlice.zero_rows``, read by ``export_plan``). An exact colour zero
    has no row to filter."""

    from .induced import _point_scale, probe_point
    from .plan_export import NativeExecutor

    if slice_.amplitude.colour_zero or not candidates:
        return {"exact_colour_zero": bool(slice_.amplitude.colour_zero)}, frozenset()
    lifted = [probe_point(slice_, momenta, 32) for momenta in candidates]
    index = max(
        range(len(lifted)),
        key=lambda i: float("inf") if lifted[i][1] is None else lifted[i][1],
    )
    point, _distance = lifted[index]
    initial = sum(1 for leg in slice_.blocks.process.legs if leg.is_initial)
    executor = NativeExecutor(slice_, forward=False, inputs="native")
    verdict = filter_rows(
        executor, point, _point_scale(point, initial), failure=failure
    )
    labels = set(verdict.dropped)
    dropped = frozenset(
        row for row in executor.exported.rows if row_label(row) in labels
    )
    return {**verdict.record(), "candidate": index}, dropped
