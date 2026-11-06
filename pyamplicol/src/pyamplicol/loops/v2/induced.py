# SPDX-License-Identifier: 0BSD
"""Loop-induced |A_loop|^2 in amplitude mode, route R1 (M-LIT LIT-AMP).

A loop-induced request has no Born amplitude within its coupling-order
limits (m-lit PLAN §1.2), so there is no bra to interfere the loop with: the
one-loop amplitude is kept as a colour vector on the amplitude basis T of
the request (``colour.AmplitudeBasis``, LIT-COL) and squared with the
full-colour metric G (PLAN §2.a.2):

    a_{l,h,t} = (i / 16 pi^2) sum_f Omega_f[N_{f,l,h,t}] + A^CT_{l,h,t},
    |A|^2_s   = (1 / avg) sum_h sum_{(l,l') in s} kappa_{ll'}
                    Re sum_{t,t'} conj(a_{l,h,t}) G_{tt'} a_{l',h,t'},

with kappa = 1 for l = l' and 2 for l != l' (loop x loop sectors: the pair
(l, l') of a sector has l <= l'), avg the slice's spin/colour average and
symmetry factor (``Normalisation.of``, unchanged), G at N_c = 3 (full colour
only, RULED Q8). The amplitude numerator of family f, helicity row h and
basis element t is today's family numerator with the leaf weight
w_s c_{c(s),t} instead of w_s U0_c(h): same tries, contents and kernels.

**Route R1** (RULED Q14 = (a)): schedule B with one-hot "resolved" columns
(h0, t), whose seed is c_{c,t} on row h0 only (the native plan writes them
as ``column_rows = {h0}`` with unit Born components; ``plan_export``). This
module is the Python oracle of that route (the reference executor) and of
the amplitude assembly and the canonical squaring the native carrier runs.

**Canonical squaring order** (DESIGN §7.10): sectors in order, their pairs
(l, l') in order, then helicity rows h in plan order, then t, then t'
(``square``). Laurent coefficients convolve as [finite, 1/eps, 1/eps^2] =
[a0^dag G b0, a0^dag G b1 + a1^dag G b0, a0^dag G b2 + a1^dag G b1 +
a2^dag G b0]: the finite part of |a|^2 is exact only if the amplitude's
poles vanish (its O(eps) terms are not computed), which the finiteness
diagnostic checks (PLAN §2.a.7 "Pole shape").

**Exact zeros** (RULED Q13): the only exact zero is the colour zero: every
loop of a requested loop order has a single coloured attachment (the
family recursion drops it as ``assemble.COLOUR_ZERO_REJECTION``, Tr T^a = 0
in any model) and no counterterm ket carries a colour tensor there
(``LoopSlice.colour_zero_orders``); every other zero is evaluated
numerically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from . import reference_executor as rx
from .families import Orders

# The interference of a loop-induced result (the artifact metadata value).
LOOP_INDUCED_INTERFERENCE = "conj(loop)*loop"


def loop_sectors(
    loop_orders: Iterable[Orders],
) -> tuple[tuple[Orders, frozenset[tuple[Orders, Orders]]], ...]:
    """The loop x loop sectors of a loop-induced request (m-lit PLAN §1.3):
    per squared order l + l' its pairs (l, l') with l <= l' (sorted), in
    squared-order order. The pair type is the Born-interfered (Born, loop)
    pair's: its first slot holds the conjugated loop order."""

    from .families import add_orders

    ordered = sorted(set(loop_orders))
    sectors: dict[Orders, set[tuple[Orders, Orders]]] = {}
    for i, left in enumerate(ordered):
        for right in ordered[i:]:
            sectors.setdefault(add_orders(left, right), set()).add((left, right))
    return tuple(
        (sector, frozenset(pairs)) for sector, pairs in sorted(sectors.items())
    )


def bind(slice_: Any, *, generation: Any = None) -> None:
    """Complete a generated loop-induced slice: its amplitude basis
    (``LoopSlice.amplitude``: the seed classes' literal tensors and the
    counterterm kets', ``colour.amplitude_basis``) and its exact colour-zero
    loop orders (``LoopSlice.colour_zero_orders``).

    A requested loop order is an exact colour zero when no counterterm ket
    carries a colour tensor there and no seed class read at that order has
    a nonzero coefficient cell: either the order's classes all vanish by
    the colour algebra (c_{c,t} = 0 at N_c = 3) or no family routes it at
    all, which is accepted only when the field-level family recursion at
    that order drops a loop for a single coloured attachment (Tr T^a = 0,
    ``COLOUR_ZERO_REJECTION``, the mechanism of ``reachable_loop_orders``).
    Any other order without a loop amplitude is refused
    (``no-loop-amplitude``): never a silent zero."""

    from .colour import amplitude_basis
    from .families import LoopOrderSelection, add_orders
    from .native_cells import _DegreeKeys

    request = slice_.request
    keys = _DegreeKeys(slice_, orbits=False)
    kets = {order for order, found in keys.counterterms.items() if found}
    tensors = sorted({tensor for found in keys.counterterms.values() for tensor, _p in found})
    basis = amplitude_basis(slice_.table, tensors)
    slice_.amplitude = basis
    requested = {add_orders(tuple(orders.items())) for orders in request.exact_loop_orders}
    live = {
        order
        for c, row in enumerate(basis.cells_at_three())
        if row
        for order in slice_.table.orders[c]
    }
    families = slice_.families
    routed = {
        add_orders(*(position.orders for position in families.routing(closure)[1]))
        for group in (families.families, families.scaleless_families)
        for family in group.values()
        for closure in family.closures
    }
    zero = requested - live - kets
    unrouted = sorted(zero - routed)
    if unrouted:
        from pyamplicol.processes.model import build_model_process_ir

        from .assemble import COLOUR_ZERO_REJECTION, _field_level_families

        process = build_model_process_ir(
            request.process, slice_.compiled.ir, color_accuracy="full"
        )
        for order in unrouted:
            found = _field_level_families(
                slice_.data,
                process,
                slice_.loop_catalogue,
                LoopOrderSelection(frozenset(request.corrections), frozenset({order})),
                request,
                slice_.native if generation is not None else None,
            )
            if found.families or not any(
                entry.reason == COLOUR_ZERO_REJECTION for entry in found.rejected
            ):
                from .refusals import LoopRequestUnsupported

                raise LoopRequestUnsupported(
                    "no-loop-amplitude",
                    f"{request.process!r}: no loop amplitude at the loop order "
                    f"{dict(order)} (no family, no counterterm ket and no exact "
                    "colour zero)",
                )
    slice_.colour_zero_orders = frozenset(zero)


def kappa(pair: tuple[Orders, Orders]) -> int:
    """1 for a diagonal pair (l, l), 2 (2 Re) for l != l'."""

    return 1 if pair[0] == pair[1] else 2


def column_labels(rows: Sequence[rx.Row], size: int) -> tuple[tuple[rx.Row, int], ...]:
    """The R1 columns (h0, t), row-major: column r * T + t."""

    return tuple((row, t) for row in rows for t in range(size))


class AmplitudeSeeds:
    """R1 leaf seeds of the Python reference executor (an ``rx.SeedSource``):
    column (h0, t) of row h0 carries sum_c x_c c_{c,t} of the skeleton's
    seed combination; every other (row, column) is zero, so column (h0, t)
    collects the amplitude numerator of row h0 alone."""

    def __init__(self, slice_: Any, rows: Sequence[rx.Row], numbers: Any):
        basis = slice_.amplitude
        self.size = len(basis)
        self.rows = tuple(rows)
        self.columns = len(self.rows) * self.size
        self.class_of = slice_.class_of
        self._numbers = numbers
        self._cells = [
            [(t, numbers.colour(value, 0)) for t, value in row]
            for row in basis.cells_at_three()
        ]
        self._cache: dict[Any, dict[rx.Row, tuple[Any, ...]]] = {}

    def _vector(self, combination) -> list[Any]:
        numbers = self._numbers
        out = [numbers.zero] * self.size
        for c, x in combination:
            weight = numbers.pair(numbers.real(x.real), numbers.real(x.imag))
            for t, cell in self._cells[c]:
                out[t] = out[t] + weight * cell
        return out

    def seed(self, skeleton, contents):
        reading = rx._reading(skeleton, contents)
        combination = self.class_of.get(reading)
        if combination is None:
            raise KeyError("reading without a seed class")
        key = tuple(combination)
        found = self._cache.get(key)
        if found is None:
            vector = self._vector(combination)
            zero = self._numbers.zero
            found = {}
            for r, row in enumerate(self.rows):
                values = [zero] * self.columns
                values[r * self.size : (r + 1) * self.size] = vector
                found[row] = tuple(values)
            self._cache[key] = found
        return (key, skeleton.orders), found


def counterterm_kets(
    slice_: Any, site_vectors: Mapping[rx.Row, Mapping], numbers: Any, width: int
) -> dict[tuple[Orders, rx.Row], list[list[Any]]]:
    """The counterterm amplitude A^CT_{l,h,t} per (loop order, row): the CT
    program's per-row colour vectors (``_PointEvaluation.site_vectors``:
    (component, Laurent power, loop order) -> {(literal tensor, N_c power):
    value}, coefficients and 1/sqrt(2) powers folded) on the amplitude
    basis at N_c = 3, its components summed (the R2 and UV parts are not
    additive in a square). Per (l, h): T lists of ``width`` Laurent
    coefficients [finite, 1/eps, 1/eps^2]."""

    basis = slice_.amplitude
    index = {tensor: t for t, tensor in enumerate(basis.tensors)}
    three = numbers.real(3)
    zero = numbers.zero
    out: dict[tuple[Orders, rx.Row], list[list[Any]]] = {}
    for row, vectors in site_vectors.items():
        for (_component, power, order), vector in vectors.items():
            k = -power
            if k >= width:
                continue
            ket = out.get((order, row))
            if ket is None:
                ket = out[(order, row)] = [[zero] * width for _ in basis.tensors]
            for (tensor, nc), value in vector.items():
                t = index.get(tensor)
                if t is None:
                    raise KeyError(
                        f"a counterterm ket tensor {tensor!r} outside the amplitude "
                        "basis (colour.amplitude_basis of the slice's CT tensors)"
                    )
                ket[t][k] = ket[t][k] + value * three**nc
    return out


def bare_amplitudes(
    families: Mapping[tuple[Any, Orders, tuple[rx.Row, int]], Any],
    size: int,
    numbers: Any,
    width: int,
) -> dict[tuple[Orders, rx.Row], list[list[Any]]]:
    """a_bare_{l,h,t} = (i/16 pi^2) sum_f Omega_f per (loop order, row) from
    the reduced lanes (family, loop order, (row, t)), families summed in
    sorted lane order (the Laurent ``coefficients`` of each lane)."""

    from .conventions import LOOP_FACTOR_DENOMINATOR

    factor = numbers.pair(0, 1) * (
        1 / (LOOP_FACTOR_DENOMINATOR * numbers.pi * numbers.pi)
    )
    zero = numbers.zero
    sums: dict[tuple[Orders, rx.Row], list[list[Any]]] = {}
    for (_family, order, (row, t)), value in sorted(families.items(), key=_lane_key):
        amplitude = sums.get((order, row))
        if amplitude is None:
            amplitude = sums[(order, row)] = [[zero] * width for _ in range(size)]
        cell = amplitude[t]
        for k, coefficient in enumerate(value.coefficients[:width]):
            cell[k] = cell[k] + coefficient
    return {
        key: [[factor * z for z in cell] for cell in amplitude]
        for key, amplitude in sums.items()
    }


def _lane_key(item):
    (family, order, (row, t)), _value = item
    return (family, order, row, t)


def add_amplitudes(*parts: Mapping, numbers: Any, size: int, width: int) -> dict:
    """The sum of per-(order, row) amplitude vectors (missing entries zero)."""

    zero = numbers.zero
    out: dict = {}
    for part in parts:
        for key, vector in part.items():
            total = out.get(key)
            if total is None:
                total = out[key] = [[zero] * width for _ in range(size)]
            for t in range(size):
                for k in range(width):
                    total[t][k] = total[t][k] + vector[t][k]
    return out


def metric(slice_: Any, numbers: Any) -> list[list[Any]]:
    """G_{tt'} = <t|t'> at N_c = 3 in the carrier (real symmetric)."""

    return [
        [numbers.colour(cell, 0) for cell in row]
        for row in slice_.amplitude.metric_at_three()
    ]


def square(
    amplitudes: Mapping[tuple[Orders, rx.Row], Sequence[Sequence[Any]]],
    gram: Sequence[Sequence[Any]],
    sectors: Sequence[tuple[Orders, frozenset[tuple[Orders, Orders]]]],
    rows: Sequence[rx.Row],
    width: int,
    numbers: Any,
    average: Any,
) -> dict[Orders, tuple[Any, ...]]:
    """|A|^2 per sector and Laurent coefficient in the canonical order
    (module docstring): sectors, pairs, rows h, t, t'; per coefficient k
    sum_{i+j=k} Re(a_{l,i}^dag G a_{l',j}) times kappa, divided by
    ``average``."""

    zero = numbers.zero
    real_zero = numbers.real(0)
    size = len(gram)
    out: dict[Orders, tuple[Any, ...]] = {}
    for sector, pairs in sectors:
        totals = [real_zero] * width
        for pair in sorted(pairs):
            left, right = pair
            weight = kappa(pair)
            for row in rows:
                a = amplitudes.get((left, row))
                b = amplitudes.get((right, row))
                if a is None or b is None:
                    continue
                for k in range(width):
                    total = zero
                    for i in range(k + 1):
                        j = k - i
                        for t in range(size):
                            inner = zero
                            for u in range(size):
                                inner = inner + gram[t][u] * b[u][j]
                            total = total + a[t][i].conjugate() * inner
                    totals[k] = totals[k] + weight * total.real
        out[sector] = tuple(value / average for value in totals)
    return out


def rows_of(evaluation: Any) -> tuple[rx.Row, ...]:
    """The helicity rows of a point evaluation in plan order (the product of
    the legs' helicities, ``plan_export.born_rows``)."""

    return tuple(evaluation.site_vectors)


def finish(evaluation: Any, retest: str | None) -> Any:
    """Dress (schedule B, R1 seeds), reduce and assemble a loop-induced point
    (``_PointEvaluation.finish`` of an amplitude-mode slice)."""

    from .poles import splits

    if splits(evaluation.mode):
        raise ValueError(
            "a loop-induced result admits the pole selections 'all' and 'none' "
            "only: its UV and IR parts are not additive in |A|^2 (m-lit PLAN "
            "§2.a.7)"
        )
    slice_ = evaluation.slice
    rows = rows_of(evaluation)
    labels = column_labels(rows, len(slice_.amplitude))
    if not slice_.plan.schedules:
        # An exact colour zero at every order: no family to dress.
        evaluation.set_lanes({}, None if retest is None else {})
    else:
        seeds = AmplitudeSeeds(slice_, rows, evaluation.numbers)
        execution = evaluation.dress(seeds=seeds)
        evaluation._reduce(execution.numerators, retest, columns=labels)
    evaluation._assemble()
    return evaluation


def assemble(evaluation: Any) -> None:
    """``_PointEvaluation._assemble`` of a loop-induced point: the amplitude
    vectors a = a_bare + A^CT per (loop order, row) (kept as
    ``evaluation.amplitude_vectors``), |A|^2 per sector (``square``) as the
    "virtual" component of the ``PointResult`` (no other component: the bare,
    R2 and UV parts are not additive in a square), Born fields empty, and
    the stability inputs (``budgets``)."""

    from .assemble import Laurent, PointResult

    numbers = evaluation.numbers
    slice_ = evaluation.slice
    width = 3 if evaluation.poles else 1
    size = len(slice_.amplitude)
    rows = rows_of(evaluation)
    with numbers.scope():
        bare = bare_amplitudes(evaluation.families, size, numbers, width)
        kets = counterterm_kets(slice_, evaluation.site_vectors, numbers, width)
        total = add_amplitudes(bare, kets, numbers=numbers, size=size, width=width)
        gram = metric(slice_, numbers)
        squared = square(
            total, gram, slice_.sectors, rows, width, numbers, evaluation.average
        )
        zero = numbers.real(0)
        virtual = {sector: Laurent(*values) for sector, values in squared.items()}
        totals = [zero]
        for k in range(width):
            totals.append(sum((values[k] for values in squared.values()), zero))
    evaluation.amplitude_vectors = total
    evaluation.bare_vectors = bare
    evaluation.counterterm_vectors = kets
    evaluation.gram = gram
    evaluation.result = PointResult(
        {},
        {"virtual": virtual},
        tuple(totals),
        dict(evaluation.families),
        None,
        colour_contraction=getattr(evaluation, "colour_contraction", "direct"),
        poles=evaluation.mode,
    )
    budgets(evaluation)


def _absolute(vector, numbers):
    return [[abs(z) for z in cell] for cell in vector]


def _deviations(direct, other, size, width, numbers):
    """Per (order, row) the absolute deviations |a - a'| per t and k."""

    zero = numbers.zero
    out = {}
    for key in dict.fromkeys([*direct, *other]):
        a = direct.get(key)
        b = other.get(key)
        out[key] = [
            [
                abs((a[t][k] if a is not None else zero) - (b[t][k] if b is not None else zero))
                for k in range(width)
            ]
            for t in range(size)
        ]
    return out


def _bound_norm(errors, gram) -> float:
    """sqrt(e^T |G| e) for absolute errors e per t (an upper bound of
    ||delta a||_G for |delta a_t| <= e_t)."""

    total = 0.0
    for t, x in enumerate(errors):
        for u, y in enumerate(errors):
            total += float(x) * abs(complex(gram[t][u])) * float(y)
    return total**0.5


def quadratic_budget(evaluation: Any, errors: Mapping) -> dict:
    """The error budget of |A|^2 per sector and Laurent coefficient from
    absolute amplitude errors ``errors`` ((order, row) -> per t and k):
    sum_pairs kappa sum_h sum_{i+j=k} (||a_i||_G ||e_j|| + ||e_i|| ||a_j||_G
    + ||e_i|| ||e_j||) / avg, colour-metric norms (PLAN §2.a.8; never a raw
    component)."""

    from .stability import Budget

    slice_ = evaluation.slice
    gram = evaluation.gram
    numbers = evaluation.numbers
    width = 3 if evaluation.poles else 1
    rows = rows_of(evaluation)
    average = float(evaluation.average)
    amplitude = evaluation.amplitude_vectors
    size = len(slice_.amplitude)

    def norms(vectors, key):
        vector = vectors.get(key)
        if vector is None:
            return [0.0] * width
        return [
            float(g_norm([vector[t][k] for t in range(size)], gram, numbers))
            for k in range(width)
        ]

    def bounds(key):
        vector = errors.get(key)
        if vector is None:
            return [0.0] * width
        return [_bound_norm([vector[t][k] for t in range(size)], gram) for k in range(width)]

    out = {}
    for sector, pairs in slice_.sectors:
        totals = [0.0] * width
        for pair in sorted(pairs):
            left, right = pair
            for row in rows:
                a, b = norms(amplitude, (left, row)), norms(amplitude, (right, row))
                da, db = bounds((left, row)), bounds((right, row))
                for k in range(width):
                    for i in range(k + 1):
                        j = k - i
                        totals[k] += kappa(pair) * (
                            a[i] * db[j] + da[i] * b[j] + da[i] * db[j]
                        )
        out[sector] = Budget(tuple(value / average for value in totals))
    return out


def lane_errors(evaluation: Any, lanes: Mapping) -> dict:
    """Absolute amplitude errors (i/16 pi^2) sum_f |delta Omega_f| per
    (order, row), t and k from per-lane absolute deviations (no
    cancellation between families)."""

    from .conventions import LOOP_FACTOR_DENOMINATOR
    import math

    width = 3 if evaluation.poles else 1
    size = len(evaluation.slice.amplitude)
    factor = 1 / (LOOP_FACTOR_DENOMINATOR * math.pi**2)
    out: dict = {}
    for (_family, order, (row, t)), values in lanes.items():
        cell = out.setdefault((order, row), [[0.0] * width for _ in range(size)])[t]
        for k, value in enumerate(tuple(values)[:width]):
            cell[k] += factor * float(abs(value))
    return out


def budgets(evaluation: Any) -> None:
    """The stability inputs of a loop-induced point: ``absolute`` (the scale
    S of the estimate's floor), ``retest_budget`` and ``family_deviations``
    from the reduction re-test (quadratic in the colour-metric norm)."""

    zero_lane = None
    evaluation.family_deviations = None
    evaluation.retest_budget = None
    if evaluation.retest_families is not None:
        deviations = {}
        for lane, value in evaluation.families.items():
            other = evaluation.retest_families[lane].coefficients
            deviations[lane] = tuple(
                abs(a - b) for a, b in zip(value.coefficients, other, strict=True)
            )
        evaluation.family_deviations = deviations
        evaluation.retest_budget = quadratic_budget(
            evaluation, lane_errors(evaluation, deviations)
        )
    del zero_lane
    evaluation.absolute = scale(evaluation)


def rotation_budget(direct: Any, rotated: Any) -> dict:
    """The error budget of one rotation check of a loop-induced point.

    A rotation keeps |A|^2 but not the helicity amplitudes: the external
    wavefunctions at the rotated momenta differ from the rotated ones by a
    little-group phase per helicity row, common to every loop order, basis
    element and lane of the row. Each row's lanes and counterterm kets are
    therefore compared after removing that phase (the phase of
    sum_{l,t} conj(a'_{l,h,t}) a_{l,h,t} of the finite parts; 1 where that
    sum vanishes), lane by lane, and the absolute deviations enter the
    quadratic budget (``quadratic_budget``) as the reduction re-test's do."""

    numbers = direct.numbers
    width = 3 if direct.poles else 1
    size = len(direct.slice.amplitude)
    with numbers.scope():
        phases = {}
        for row in rows_of(direct):
            total = numbers.zero
            for (order, key_row), vector in direct.amplitude_vectors.items():
                if key_row != row:
                    continue
                other = rotated.amplitude_vectors.get((order, row))
                if other is None:
                    continue
                for t in range(size):
                    total = total + other[t][0].conjugate() * vector[t][0]
            magnitude = abs(total)
            phases[row] = (
                numbers.pair(1)
                if not magnitude
                else total * numbers.real(1 / magnitude)
                if numbers.double
                else total * (1 / numbers.real(magnitude))
            )
        deviations = {}
        zero = numbers.zero
        for lane in dict.fromkeys([*direct.families, *rotated.families]):
            _family, _order, (row, _t) = lane
            a = direct.families.get(lane)
            b = rotated.families.get(lane)
            a = (zero,) * width if a is None else a.coefficients[:width]
            b = (zero,) * width if b is None else b.coefficients[:width]
            deviations[lane] = tuple(
                abs(y * phases[row] - x) for x, y in zip(a, b, strict=True)
            )
        errors = lane_errors(direct, deviations)
        for key in dict.fromkeys(
            [*direct.counterterm_vectors, *rotated.counterterm_vectors]
        ):
            _order, row = key
            a = direct.counterterm_vectors.get(key)
            b = rotated.counterterm_vectors.get(key)
            cell = errors.setdefault(key, [[0.0] * width for _ in range(size)])
            for t in range(size):
                for k in range(width):
                    x = zero if a is None else a[t][k]
                    y = zero if b is None else b[t][k]
                    cell[t][k] += float(abs(y * phases[row] - x))
    return quadratic_budget(direct, errors)


def scale(evaluation: Any) -> dict:
    """The scale S of the estimate's floor per sector and coefficient: the
    square of the pre-cancellation amplitude sizes, sum_h (sum_f
    ||(i/16pi^2) Omega_f||_G + ||A^CT||_G)^2 / avg per Laurent order of the
    product (PLAN §2.a.8), with colour-metric bounds."""

    from .stability import Budget

    numbers = evaluation.numbers
    slice_ = evaluation.slice
    width = 3 if evaluation.poles else 1
    size = len(slice_.amplitude)
    gram = evaluation.gram
    errors = lane_errors(
        evaluation,
        {lane: value.coefficients for lane, value in evaluation.families.items()},
    )
    kets = evaluation.counterterm_vectors
    for key, vector in kets.items():
        cell = errors.setdefault(key, [[0.0] * width for _ in range(size)])
        for t in range(size):
            for k in range(width):
                cell[t][k] += abs(complex(vector[t][k]))
    rows = rows_of(evaluation)
    average = float(evaluation.average)
    out = {}
    for sector, pairs in slice_.sectors:
        totals = [0.0] * width
        for pair in sorted(pairs):
            left, right = pair
            for row in rows:
                a = errors.get((left, row))
                b = errors.get((right, row))
                if a is None or b is None:
                    continue
                na = [_bound_norm([a[t][k] for t in range(size)], gram) for k in range(width)]
                nb = [_bound_norm([b[t][k] for t in range(size)], gram) for k in range(width)]
                for k in range(width):
                    for i in range(k + 1):
                        totals[k] += kappa(pair) * na[i] * nb[k - i]
        out[sector] = Budget(tuple(value / average for value in totals))
    del numbers
    return out


def g_norm(vector: Sequence[Any], gram: Sequence[Sequence[Any]], numbers: Any) -> Any:
    """||x||_G = sqrt(x^dag G x) (G positive semidefinite)."""

    zero = numbers.zero
    total = zero
    for t, x in enumerate(vector):
        inner = zero
        for u, y in enumerate(vector):
            inner = inner + gram[t][u] * y
        total = total + x.conjugate() * inner
    value = total.real
    if value <= 0:
        return numbers.real(0)
    return value.sqrt() if hasattr(value, "sqrt") else value**0.5


__all__ = [
    "LOOP_INDUCED_INTERFERENCE",
    "AmplitudeSeeds",
    "add_amplitudes",
    "bare_amplitudes",
    "bind",
    "column_labels",
    "counterterm_kets",
    "g_norm",
    "kappa",
    "loop_sectors",
    "metric",
    "square",
]
