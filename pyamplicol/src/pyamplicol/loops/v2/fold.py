# SPDX-License-Identifier: 0BSD
"""F1 proportional-chain folding of loop-induced dressed-row plans (M-LIT
lit-fold; USER ~22:39 "Yes, and allow exact zeros"; m-lit DECISIONS
2026-11-08 22:39).

A loop-induced plan on route R2 (``plan_export``, schema 5, ``dressed``)
dresses, per family, one chain per colour-ordered skeleton read in the
family's trie frame (the leaves of its class tries; never Feynman
diagrams): chain j adds ``U_j w_j Tr_j`` into the family's (row, t)
columns, with ``U_j`` the amplitude vector of its seed key, ``w_j`` its
exact leaf weight and ``Tr_j`` its traced numerator per helicity row and
loop order (a polynomial in the loop momentum). Its *weighted numerator*
is ``N_j = w_j Tr_j``.

**The fold.** Chains of one family that close into the same loop orders and
whose weighted numerators are exactly proportional, ``N_j = r_j N_rep`` for
every helicity row, loop order and coefficient with ``r_j`` an exact
rational, are dressed once: their sum ``sum_j U_j N_j`` is
``(sum_j r_j U_j) N_rep``. Within one class trie (equal vectors) the
representative's leaf weight becomes ``w_rep sum_j r_j``; across class tries
it keeps ``w_rep`` under the seed key ``sum_j r_j x^(j)`` (the exact
combination of the chains' seed classes, key vector ``sum_j r_j U_j``). A
group whose combined vector vanishes exactly contributes nothing: its
chains are not exported, and a process left without a chain and without a
counterterm ket is an **exact zero** (the user's ruling: a zero proven by
the engine's exact algebra; never by a physics rule).

**The exact test** (the helicity-zero filter's machinery,
``helicity_filter``; no physics rule decides a fold: no vertex type,
Lorentz structure, C-parity, model or particle name). Every input of the
dressing is a free variable: block-current components, external momenta
(momentum conservation kept), masses and every base of the kernel
constants' exact factored form (parameters, primes under fractional
powers, pi, parameter functions; ``b^e = u_b^(e L_b)``, a prime under
integer powers only is itself), so an identity that holds for every value
holds for the physical ones, and a ratio that depends on an irrational or a
parameter varies between points and is no fold (a missed fold, never a
false one).

1. *Candidates and exact ratios*: one native dressing of every chain in the
   exact rational carrier (``chain-resolved probe``: each chain closes into
   its own column, ``ProbeDressing``) at a random point of small integers;
   chains whose numerator vectors are proportional there form a candidate
   group, each ratio ``r_j`` the exact quotient of the two numerators (a
   ratio with a nonzero imaginary part is no fold: the ruling asks for an
   exact rational ratio). A chain that vanishes there is a candidate zero.
2. *Certification*: each identity ``N_j - r_j N_rep = 0`` (``N_j = 0`` for a
   candidate zero) is tested in the prime field F_p, p = 2^61 - 1, at ``k``
   fresh random points (the native prime carrier, the same probe plan). A
   nonzero identity is a polynomial of total degree at most ``d`` (the
   plan's ``degree_bound``, once the bases' negative powers are cleared), so
   it vanishes at a random point with probability at most ``d / (p - 1)``
   (Schwartz-Zippel); with ``m`` identities, ``k`` is the least count with
   ``m (d / (p - 1))^k`` at most the stated failure probability (the union
   bound; ``generation.loop_induced_fold_failure_probability``, the
   filter's default 1e-12; 0 folds nothing). An identity that fails leaves
   its chain unfolded. *What the bound covers*: the probability over the
   random points for the numerators reduced modulo p. As for the filter, a
   deterministic step precedes it and is not counted: an exact coefficient
   (of a numerator, or of an identity: ``r_j`` is exact, from step 1, never
   reconstructed from a residue) whose numerator p divides reduces to 0; a
   constant whose denominator p divides has no residue, and then nothing
   folds (``Unrepresentable``).

The record (``FoldVerdict.record``) holds the prime, the seed, the points,
the degree bound, the identities certified, the failure probability and
bound, and the counts; it is the certificate of an exact zero.

Seeded plans (Born-interfered, schema 4) and route R1 (the oracle) are never
folded; a process in which nothing folds keeps its plan byte for byte.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from pyamplicol.color.connections import ExactColorCoefficient

from . import helicity_filter as hf
from . import reference_executor as rx
from .families import FamilyKey, Orders

# The artifact's generation-filter key of the fold record.
FOLD_RECORD = "loop_induced_fold"
# The stated failure probability by default: the helicity filter's
# (``helicity_filter.FAILURE_PROBABILITY``, Q10's "stated failure
# probability"; the setting ``generation.loop_induced_fold_failure_
# probability``).
FAILURE_PROBABILITY = hf.FAILURE_PROBABILITY
# The seed of the fold's random points (recorded; a regeneration draws the
# same points and writes the same plan bytes).
SEED = 0
# Bits of the exact candidate point's random integers. Any size is sound
# (the certification decides); small integers keep the exact dressing
# cheap, and 2^-16 per coordinate makes a coincidental proportionality at
# that point unlikely (it would only fail its certification).
EXACT_BITS = 16
# Complex entries of one probe call's output (``_batches``): the memory of
# a native numerator call, which no result depends on.
PROBE_ENTRIES = 1 << 23


@dataclass(frozen=True)
class Chain:
    """One chain of a dressed-row family: an oriented skeleton read in the
    family's trie frame, its seed combination (seed classes and their exact
    weights x_c), loop orders, exact leaf weight and amplitude vector."""

    skeleton: Any
    ring: tuple
    combination: tuple
    orders: Orders
    weight: Fraction
    vector: tuple


def family_chains(slice_: Any) -> dict[FamilyKey, list[Chain]]:
    """The chains of every family of a loop-induced slice's dressed-row plan,
    in the family trie's leaf order (``plan_export._Exporter.family``)."""

    from .plan_export import amplitude_vector

    view = slice_.plan.view
    cells = slice_.amplitude.cells_at_three()
    vectors: dict[Any, tuple] = {}
    out: dict[FamilyKey, list[Chain]] = {}
    for key, schedule in slice_.plan.schedules.items():
        chains = []
        for skeleton, ring in schedule.trie.leaves:
            contents = [view.content(segment.label) for segment in ring]
            combination = slice_.class_of.get(rx._reading(skeleton, contents))
            if combination is None:
                raise KeyError("reading without a seed class")
            vector = vectors.get(combination)
            if vector is None:
                vector = vectors[combination] = amplitude_vector(combination, cells)
            chains.append(
                Chain(
                    skeleton,
                    tuple(ring),
                    combination,
                    skeleton.orders,
                    Fraction(slice_.weights(skeleton, contents)),
                    vector,
                )
            )
        out[key] = chains
    return out


# ------------------------------------------------------------------ dressings


class ProbeDressing:
    """A chain-resolved test dressing (``plan_export.export_plan``'s
    ``dressing``): every chain of the probed families is its own class trie
    whose key closes into its own column, ``j`` its index in its family, so
    the plan's family numerators hold each chain's weighted numerator
    ``N_j`` per (row, j). Families outside ``families`` are not exported."""

    def __init__(
        self,
        families: Mapping[FamilyKey, Sequence[Chain]],
        zero_rows: frozenset = frozenset(),
    ):
        self.families = families
        # Helicity rows the probe does not read (``export_plan``: columns
        # without a row, pruned at load): the exact ratios' probe reads a
        # few rows only (``_cover``).
        self.zero_rows = zero_rows
        # Columns per row: the largest probed family's chain count.
        self.amplitude = max((len(c) for c in families.values()), default=0) or 1

    def tries(self, key: FamilyKey):
        chains = self.families.get(key)
        if chains is None:
            return []
        return [
            (("probe", j), [(chain.skeleton, chain.ring, chain.weight)])
            for j, chain in enumerate(chains)
        ]

    def seed_records(self, exporter: Any, size: int) -> tuple[list, list]:
        """One class record and one seed key per probe column j: a unit
        weight on the one Born component, column j only."""

        exporter.born_components.setdefault((0, (), 0), 0)
        classes, seeds = [], []
        for key in exporter.seeds:
            _probe, j = key
            columns: list[list] = [[] for _ in range(size)]
            columns[j].append([0, "1", "0"])
            classes.append({"columns": columns})
            seeds.append({"classes": [[len(classes) - 1, "1", "0"]]})
        return classes, seeds


@dataclass
class FoldedDressing:
    """The folded class tries of a slice's families (``LoopSlice.fold``,
    read by ``plan_export.export_plan`` on route R2): per folded family its
    class tries (seed key, chains with their exact leaf weights); a family
    without a fold is absent (its own class tries, byte for byte), one whose
    every group cancels has none (not exported)."""

    families: dict[FamilyKey, list[tuple[Any, list[tuple[Any, tuple, Fraction]]]]]
    verdict: FoldVerdict

    def tries(self, key: FamilyKey):
        return self.families.get(key)

    @property
    def exact_zero(self) -> bool:
        return self.verdict.exact_zero


# ------------------------------------------------------------------- verdict


@dataclass(frozen=True)
class FoldVerdict:
    """The fold's certificate and counts for one process; ``record`` is what
    the artifact stores (``generation_filters['loop_induced_fold']``)."""

    enabled: bool
    prime: int
    seed: int
    points: int
    degree_bound: int
    identities: int
    failure_probability: float
    failure_bound: float
    families: int
    chains: int
    chains_after: int
    groups: int
    chains_folded: int
    zero_groups: int
    zero_orders: tuple[Orders, ...]
    exact_zero: bool
    failed: int = 0
    reason: str | None = None
    seconds: Mapping[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enabled": self.enabled,
            "prime": self.prime,
            "seed": self.seed,
            "points": self.points,
            "degree_bound": self.degree_bound,
            "identities": self.identities,
            "failure_probability": self.failure_probability,
            "failure_bound": self.failure_bound,
            "families": self.families,
            "chains": self.chains,
            "chains_after": self.chains_after,
            "groups": self.groups,
            "chains_folded": self.chains_folded,
            "zero_groups": self.zero_groups,
            "zero_loop_orders": [
                [[name, power] for name, power in order] for order in self.zero_orders
            ],
            "exact_zero": self.exact_zero,
            "identities_failed": self.failed,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def disabled(failure: float, reason: str) -> dict[str, Any]:
    """The record of a process the fold does not run on."""

    return {"enabled": False, "failure_probability": failure, "reason": reason}


# -------------------------------------------------------------- random points


class _FreeValues(hf._Values):
    """The filter's random values with every base free: a prime under a
    fractional power takes a random value, never its own root modulo p (an
    identity of free values holds for the true ones; a ratio through an
    irrational then varies between points)."""

    def base(self, name: Any) -> int:
        found = self.values.get(name)
        if found is None:
            found = self.values[name] = self.draw()
        return found


class _ExactValues:
    """A random point of small integers for the exact rational carrier, in
    the filter's base variables (``helicity_filter._Values``, every base
    free)."""

    def __init__(self, rng: random.Random, lcm: Mapping[Any, int]):
        self.rng = rng
        self.lcm = lcm
        self.values: dict[Any, int] = {}

    def draw(self) -> int:
        return self.rng.randrange(1, 1 << EXACT_BITS)

    def base(self, name: Any) -> int:
        found = self.values.get(name)
        if found is None:
            found = self.values[name] = self.draw()
        return found

    def power(self, base: Any, signs: Mapping[str, int], power: Fraction) -> Fraction:
        name = hf._base_name(base, signs, power)
        kind, value = base
        if kind == "prime" and power.denominator == 1 and self.lcm[name] == 1:
            u, exponent = Fraction(int(value)), power
        else:
            u, exponent = Fraction(self.base(name)), power * self.lcm[name]
        out = u ** int(exponent)
        flipped = kind == "param" and signs.get(value, 1) < 0
        if flipped and power.denominator == 1 and int(power) % 2:
            out = -out
        return out

    def constant(self, key: Any, flips: Any) -> tuple[Fraction, Fraction]:
        coefficient, exact_factors, irrational, _positive = key
        exact = rx.exact(coefficient)
        signs = dict(flips)
        factor = Fraction(1)
        for base, power in (*exact_factors, *irrational):
            factor *= self.power(base, signs, Fraction(power))
        return Fraction(exact.real) * factor, Fraction(exact.imag) * factor


def exact_inputs(exported: Any, legs: int, rng: random.Random, lcm) -> dict:
    """The batch of one random exact point (``helicity_filter.random_inputs``'
    layout, values as exact texts)."""

    from .plan_export import _fraction_text

    values = _ExactValues(rng, lcm)
    momenta = [[values.draw() for _mu in range(4)] for _leg in range(legs - 1)]
    momenta.append([-sum(p[mu] for p in momenta) for mu in range(4)])
    rows = [
        [str(values.draw()), str(values.draw())]
        for _block, _word, _chirality, _row, dimension in exported.block_rows
        for _component in range(dimension)
    ]
    constants = []
    for key, flips in exported.constants:
        re, im = values.constant(key, flips)
        constants.append([_fraction_text(re), _fraction_text(im)])
    return {
        "momenta": [[str(x) for p in momenta for x in p]],
        "scales": ["1"],
        "block_rows": [rows],
        "born": [[["1", "0"]] * len(exported.rows)],
        "constants": constants,
        "masses": [str(values.draw()) for _mass in exported.masses],
        "leg_masses": [str(values.draw()) for _leg in range(legs)],
    }


# ---------------------------------------------------------------- the probe


def _residue(text: str) -> int:
    """An exact text (``a`` or ``a/b``) modulo p."""

    numerator, _slash, denominator = text.partition("/")
    value = int(numerator) % hf.PRIME
    if denominator:
        value = value * pow(int(denominator), hf.PRIME - 2, hf.PRIME) % hf.PRIME
    return value


def _mul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    p = hf.PRIME
    return (a[0] * b[0] - a[1] * b[1]) % p, (a[0] * b[1] + a[1] * b[0]) % p


def _inverse(a: tuple[int, int]) -> tuple[int, int]:
    p = hf.PRIME
    inverse = pow((a[0] * a[0] + a[1] * a[1]) % p, p - 2, p)
    return a[0] * inverse % p, -a[1] * inverse % p


@dataclass
class _Batch:
    """Families probed by one chain-resolved plan (``ProbeDressing``)."""

    families: dict[FamilyKey, list[Chain]]
    executor: Any = None
    lcm: Mapping[Any, int] = field(default_factory=dict)
    degree: int = 0
    # Per (family, j): its candidate group's representative and the ratio's
    # residue, from the discovery point.
    groups: dict[FamilyKey, list[list[tuple[int, Any]]]] = field(default_factory=dict)
    zeros: dict[FamilyKey, list[int]] = field(default_factory=dict)
    # Per (family, j): the helicity rows its numerator is nonzero on.
    rows: dict[tuple[FamilyKey, int], set[int]] = field(default_factory=dict)


def _batches(
    families: Mapping[FamilyKey, list[Chain]], rows: int, entries: int
) -> list[_Batch]:
    """Families in probe batches, fewest chains first, each batch's dense
    output per point (rows x its widest family's chains x the coefficients
    of its families) within ``entries`` complex entries (the memory of one
    native call; no result depends on it), a family alone when it exceeds
    them."""

    from .poly_affine import size

    ordered = sorted(families.items(), key=lambda item: len(item[1]))
    out: list[_Batch] = []
    current: dict[FamilyKey, list[Chain]] = {}
    coefficients = 0
    for key, chains in ordered:
        orders = len({chain.orders for chain in chains})
        own = orders * size(max(chain.skeleton.rank for chain in chains))
        if current and rows * len(chains) * (coefficients + own) > entries:
            out.append(_Batch(current))
            current, coefficients = {}, 0
        current[key] = chains
        coefficients += own
    if current:
        out.append(_Batch(current))
    return out


def _chain_vectors(
    exported: Any, point: Any, families: Mapping[FamilyKey, Sequence[Chain]]
) -> dict[tuple[FamilyKey, int], list[tuple[int, list]]]:
    """Per (family, chain) its weighted numerator: per helicity row (plan
    order) the (re, im) texts of its column (row, j) at its loop orders."""

    width = max(len(c) for c in families.values())
    rows = len(exported.rows)
    out: dict[tuple[FamilyKey, int], list[tuple[int, list]]] = {}
    for family, order, _rank, columns in point:
        key = exported.families[family]
        orders = exported.orders[order]
        for j, chain in enumerate(families[key]):
            if chain.orders != orders:
                continue
            out[(key, j)] = [(r, columns[r * width + j]) for r in range(rows)]
    return out


def _discover(batch: _Batch, point: Any, rng: random.Random) -> None:
    """Candidate groups from one random point over F_p: chains whose
    numerators are proportional there (equal ratios of two random linear
    projections; the ratio's residue from one of them, real only), and
    candidate zeros (a numerator vanishing there)."""

    p = hf.PRIME
    exported = batch.executor.exported
    vectors = _chain_vectors(exported, point, batch.families)
    weights: list[tuple[int, int]] = []
    for key, chains in batch.families.items():
        found: dict[Any, list[tuple[int, Any]]] = {}
        for j, chain in enumerate(chains):
            first = (0, 0)
            second = (0, 0)
            nonzero: set[int] = set()
            n = 0
            for r, column in vectors.get((key, j), []):
                for re, im in column:
                    if n == len(weights):
                        weights.append((rng.randrange(p), rng.randrange(p)))
                    a, b = weights[n]
                    n += 1
                    if re == "0" and im == "0":
                        continue
                    x, y = int(re), int(im)
                    nonzero.add(r)
                    first = ((first[0] + a * x) % p, (first[1] + a * y) % p)
                    second = ((second[0] + b * x) % p, (second[1] + b * y) % p)
            if not nonzero:
                batch.zeros.setdefault(key, []).append(j)
                continue
            batch.rows[(key, j)] = nonzero
            if second == (0, 0):
                # Unlikely (probability ~ 1/p): no fold for this chain.
                continue
            shape = _mul(first, _inverse(second))
            bucket = found.setdefault((chain.orders, shape), [])
            if not bucket:
                bucket.append((j, second))
                continue
            ratio = _mul(second, _inverse(bucket[0][1]))
            if ratio[1]:
                # A ratio outside F_p is complex: no fold.
                continue
            bucket.append((j, ratio[0]))
        groups = [g for g in found.values() if len(g) > 1]
        if groups:
            batch.groups[key] = [[(g[0][0], 1), *g[1:]] for g in groups]


def _cover(batch: _Batch) -> set[int]:
    """Helicity rows on which every candidate group's representative is
    nonzero somewhere (greedy), the rows of the exact ratios' probe."""

    needed = [
        batch.rows[(key, group[0][0])]
        for key, groups in batch.groups.items()
        for group in groups
    ]
    chosen: set[int] = set()
    while needed:
        counts: dict[int, int] = {}
        for rows in needed:
            for row in rows:
                counts[row] = counts.get(row, 0) + 1
        best = min(counts, key=lambda row: (-counts[row], row))
        chosen.add(best)
        needed = [rows for rows in needed if best not in rows]
    return chosen


def _exact_ratios(slice_: Any, batch: _Batch, rng: random.Random) -> None:
    """Each candidate's exact ratio: one native dressing in the exact
    rational carrier, on the rows of ``_cover`` only, at a random point of
    small integers; the quotient of the first nonzero coefficient of the
    representative, kept where it is real and agrees with its residue."""

    from .plan_export import NativeExecutor, export_plan

    rows = _cover(batch)
    all_rows = born_rows_of(slice_)
    probe = ProbeDressing(
        batch.families,
        frozenset(row for r, row in enumerate(all_rows) if r not in rows),
    )
    exported = export_plan(slice_, forward=False, dressing=probe)
    executor = NativeExecutor(slice_, forward=False, exported=exported)
    legs = len(slice_.blocks.process.legs)
    lcm = hf._bases(exported.constants)
    (point,) = executor.native("exact", None, 1).numerators(
        exact_inputs(exported, legs, rng, lcm), "B"
    )
    vectors = _chain_vectors(exported, point, batch.families)
    del point
    for key, groups in batch.groups.items():
        kept_groups = []
        for group in groups:
            rep = group[0][0]
            reference = [
                z
                for r, column in vectors.get((key, rep), [])
                if r in rows
                for z in column
            ]
            pivot = next(
                (i for i, (re, im) in enumerate(reference) if re != "0" or im != "0"),
                None,
            )
            if pivot is None:
                continue
            b_re, b_im = (Fraction(x) for x in reference[pivot])
            norm = b_re * b_re + b_im * b_im
            kept = [(rep, Fraction(1))]
            for j, residue in group[1:]:
                values = [
                    z
                    for r, column in vectors.get((key, j), [])
                    if r in rows
                    for z in column
                ]
                a_re, a_im = (Fraction(x) for x in values[pivot])
                ratio_re = (a_re * b_re + a_im * b_im) / norm
                ratio_im = (a_im * b_re - a_re * b_im) / norm
                if ratio_im or _residue(_text(ratio_re)) != residue:
                    continue
                kept.append((j, ratio_re))
            if len(kept) > 1:
                kept_groups.append(kept)
        batch.groups[key] = kept_groups


def _text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def born_rows_of(slice_: Any) -> tuple:
    from .plan_export import born_rows

    return born_rows(slice_)


def _identities(batch: _Batch) -> list[tuple[FamilyKey, int, int | None, Fraction]]:
    """The identities of a batch to certify: (family, j, rep, r_j) for
    N_j = r_j N_rep, (family, j, None, 0) for N_j = 0."""

    out = []
    for key, groups in batch.groups.items():
        for group in groups:
            rep = group[0][0]
            for j, ratio in group[1:]:
                out.append((key, j, rep, ratio))
    for key, chains in batch.zeros.items():
        for j in chains:
            out.append((key, j, None, Fraction(0)))
    return out


def _certify(
    batch: _Batch,
    identities: Sequence[tuple[FamilyKey, int, int | None, Fraction]],
    points: int,
    rng: random.Random,
) -> set[int]:
    """The indices of the identities that fail at some of ``points`` random
    points over F_p (every base free); all of them when a constant has no
    value in F_p (``helicity_filter.Unrepresentable``)."""

    executor = batch.executor
    exported = executor.exported
    native = executor.native("prime", None, 1)
    failed: set[int] = set()
    p = hf.PRIME
    for _point in range(points):
        live = [i for i in range(len(identities)) if i not in failed]
        if not live:
            break
        try:
            inputs = hf.random_inputs(executor, rng, batch.lcm, values=_FreeValues)
        except hf.Unrepresentable:
            return set(range(len(identities)))
        (point,) = native.numerators(inputs, "B")
        vectors = _chain_vectors(exported, point, batch.families)
        del point
        for i in live:
            key, j, rep, ratio = identities[i]
            left = [z for _r, column in vectors.get((key, j), []) for z in column]
            if rep is None:
                if any(re != "0" or im != "0" for re, im in left):
                    failed.add(i)
                continue
            right = [z for _r, column in vectors.get((key, rep), []) for z in column]
            r = ratio.numerator % p * pow(ratio.denominator % p, p - 2, p) % p
            for (a_re, a_im), (b_re, b_im) in zip(left, right, strict=True):
                if (int(a_re) - r * int(b_re)) % p or (int(a_im) - r * int(b_im)) % p:
                    failed.add(i)
                    break
    return failed


# ------------------------------------------------------------------- folding


def _scaled(combination: tuple, ratio: Fraction) -> dict[int, ExactColorCoefficient]:
    factor = ExactColorCoefficient(ratio, Fraction(0))
    return {c: factor * x for c, x in combination}


def _fold_family(
    chains: Sequence[Chain],
    groups: Sequence[Sequence[tuple[int, Fraction]]],
    cells: Any,
) -> tuple[list[tuple[Any, list[tuple[Any, tuple, Fraction]]]], int]:
    """The folded class tries of one family (in first-seen order, as
    ``plan_export.vector_groups``) and the count of groups whose combined
    vector vanishes exactly (their chains are dropped)."""

    from .plan_export import amplitude_vector

    member: dict[int, tuple[int, Fraction]] = {}
    for group in groups:
        rep = group[0][0]
        for j, ratio in group:
            member[j] = (rep, ratio)
    of_rep: dict[int, list[tuple[int, Fraction]]] = {}
    for j, (rep, ratio) in member.items():
        of_rep.setdefault(rep, []).append((j, ratio))
    tries: dict[tuple, list] = {}
    keys: dict[tuple, Any] = {}
    zero_groups = 0
    for j, chain in enumerate(chains):
        if j in member and member[j][0] != j:
            continue
        weight = chain.weight
        combination = chain.combination
        if j in of_rep:
            group = of_rep[j]
            if len({chains[i].vector for i, _r in group}) == 1:
                weight = chain.weight * sum((r for _i, r in group), Fraction(0))
                vector = chain.vector if weight else ()
            else:
                merged: dict[int, ExactColorCoefficient] = {}
                for i, ratio in group:
                    for c, x in _scaled(chains[i].combination, ratio).items():
                        merged[c] = merged[c] + x if c in merged else x
                combination = tuple(sorted((c, x) for c, x in merged.items() if x))
                vector = amplitude_vector(combination, cells) if combination else ()
            if not vector:
                zero_groups += 1
                continue
        else:
            vector = chain.vector
        trie = tries.setdefault(vector, [])
        keys.setdefault(vector, (combination, chain.orders))
        trie.append((chain.skeleton, chain.ring, weight))
    return [(keys[vector], trie) for vector, trie in tries.items()], zero_groups


def fold_slice(
    slice_: Any,
    *,
    failure: float = FAILURE_PROBABILITY,
    seed: int = SEED,
    entries: int = PROBE_ENTRIES,
) -> FoldedDressing:
    """The fold of a loop-induced slice's dressed-row plan (module docs):
    its folded class tries (a family without a certified fold keeps its own)
    and its verdict. ``entries`` bounds one probe call's output (memory
    only)."""

    from .plan_export import NativeExecutor, export_plan

    if failure <= 0:
        raise ValueError("a zero failure probability folds nothing: skip the fold")
    seconds = {"probe": 0.0, "candidates": 0.0, "exact": 0.0, "certify": 0.0}
    families = {k: c for k, c in family_chains(slice_).items() if c}
    rows = len(born_rows_of(slice_))
    batches = _batches(families, rows, entries)
    rng = random.Random(seed)
    for batch in batches:
        start = time.perf_counter()
        exported = export_plan(
            slice_, forward=False, dressing=ProbeDressing(batch.families)
        )
        batch.executor = NativeExecutor(slice_, forward=False, exported=exported)
        batch.lcm = hf._bases(exported.constants)
        batch.degree = hf.degree_bound(exported)
        seconds["probe"] += time.perf_counter() - start
        start = time.perf_counter()
        try:
            inputs = hf.random_inputs(
                batch.executor, rng, batch.lcm, values=_FreeValues
            )
        except hf.Unrepresentable:
            # No value in F_p: nothing of the batch folds.
            seconds["candidates"] += time.perf_counter() - start
            continue
        (point,) = batch.executor.native("prime", None, 1).numerators(inputs, "B")
        _discover(batch, point, rng)
        del point
        seconds["candidates"] += time.perf_counter() - start
        if batch.groups:
            start = time.perf_counter()
            _exact_ratios(slice_, batch, rng)
            seconds["exact"] += time.perf_counter() - start
    identities = [_identities(batch) for batch in batches]
    count = sum(len(found) for found in identities)
    degree = max((batch.degree for batch in batches), default=1)
    k = hf.points_needed(degree, max(count, 1), failure)
    failed: list[set[int]] = []
    start = time.perf_counter()
    for batch, found in zip(batches, identities, strict=True):
        failed.append(_certify(batch, found, k, rng) if found else set())
        batch.executor = None
    seconds["certify"] = time.perf_counter() - start
    return _verdict(
        slice_,
        families,
        batches,
        identities,
        failed,
        degree,
        k,
        failure,
        seed,
        seconds,
    )


def _verdict(
    slice_, families, batches, identities, failed, degree, k, failure, seed, seconds
) -> FoldedDressing:
    """The folded class tries and the verdict of the certified identities
    (a failed identity leaves its chain unfolded)."""

    groups: dict[FamilyKey, list] = {}
    certified_zero: dict[FamilyKey, set[int]] = {}
    bad: set[tuple[FamilyKey, int]] = set()
    for found, broken in zip(identities, failed, strict=True):
        for i in broken:
            bad.add((found[i][0], found[i][1]))
        for i, (key, j, rep, _ratio) in enumerate(found):
            if rep is None and i not in broken:
                certified_zero.setdefault(key, set()).add(j)
    for batch in batches:
        groups.update(batch.groups)
    cells = slice_.amplitude.cells_at_three()
    folded: dict[FamilyKey, list] = {}
    chains_after = 0
    group_count = 0
    chains_folded = 0
    zero_groups = 0
    live_orders: set[Orders] = set()
    for key, chains in families.items():
        family_groups = []
        for group in groups.get(key, []):
            kept = [(j, r) for j, r in group if (key, j) not in bad]
            if len(kept) > 1:
                family_groups.append(kept)
        zero = certified_zero.get(key, set())
        if not family_groups and not zero:
            chains_after += len(chains)
            live_orders.update(chain.orders for chain in chains)
            continue
        index = [j for j in range(len(chains)) if j not in zero]
        renumber = {j: i for i, j in enumerate(index)}
        tries, zeros_here = _fold_family(
            [chains[j] for j in index],
            [[(renumber[j], r) for j, r in g] for g in family_groups],
            cells,
        )
        zero_groups += zeros_here + len(zero)
        group_count += len(family_groups)
        chains_folded += sum(len(g) for g in family_groups)
        folded[key] = tries
        for _seed_key, trie in tries:
            chains_after += len(trie)
            live_orders.update(skeleton.orders for skeleton, _ring, _w in trie)
    kets = _ket_orders(slice_)
    requested = _requested_orders(slice_)
    zero_orders = tuple(
        sorted(requested - live_orders - kets - set(slice_.colour_zero_orders))
    )
    count = sum(len(found) for found in identities)
    broken = sum(len(found) for found in failed)
    verdict = FoldVerdict(
        enabled=True,
        prime=hf.PRIME,
        seed=seed,
        points=k,
        degree_bound=degree,
        identities=count - broken,
        failure_probability=failure,
        failure_bound=max(count, 1) * (degree / (hf.PRIME - 1)) ** k,
        families=len(families),
        chains=sum(len(c) for c in families.values()),
        chains_after=chains_after,
        groups=group_count,
        chains_folded=chains_folded,
        zero_groups=zero_groups,
        zero_orders=zero_orders,
        exact_zero=bool(families) and not live_orders and not kets,
        failed=broken,
        seconds=seconds,
    )
    return FoldedDressing(folded, verdict)


def _ket_orders(slice_: Any) -> set[Orders]:
    """The loop orders at which the request has a counterterm ket."""

    from .native_cells import _DegreeKeys

    keys = _DegreeKeys(slice_, orbits=False)
    return {order for order, found in keys.counterterms.items() if found}


def _requested_orders(slice_: Any) -> set[Orders]:
    from .families import add_orders

    return {
        add_orders(tuple(orders.items())) for orders in slice_.request.exact_loop_orders
    }
