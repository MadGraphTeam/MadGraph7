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
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
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
    tensors = sorted(
        {tensor for found in keys.counterterms.values() for tensor, _p in found}
    )
    basis = amplitude_basis(slice_.table, tensors)
    slice_.amplitude = basis
    requested = {
        add_orders(tuple(orders.items())) for orders in request.exact_loop_orders
    }
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


@dataclass(frozen=True)
class StoredAmplitude:
    """The amplitude basis of a stored loop-induced process (its native
    record's ``amplitude``, ``amplitude_record``): what the native route
    reads of ``colour.AmplitudeBasis`` (the basis size, the full-colour
    metric at N_c = 3, the elements the counterterm kets carry), the map of
    the tree program's counterterm ket components onto the basis
    (``ket_map``), the helicity rows of the amplitude-mode plan, the exact
    colour-zero loop orders (RULED Q13's colour-zero record) and whether the
    whole request is an exact colour zero (``colour_zero``,
    ``AmplitudeBasis.colour_zero`` at generation: no nonzero coefficient
    cell and no counterterm ket; the record keeps no cells, so it stores
    the generation's answer)."""

    size: int
    metric: tuple[tuple[Any, ...], ...]
    counterterms: tuple[int, ...]
    ket_map: tuple[tuple[int, int, Orders, int], ...]
    rows: tuple[rx.Row, ...]
    colour_zero_orders: frozenset[Orders]
    colour_zero: bool

    def __len__(self) -> int:
        return self.size

    def metric_at_three(self) -> tuple[tuple[Any, ...], ...]:
        return self.metric

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> StoredAmplitude:
        from pyamplicol.color.connections import ExactColorCoefficient

        def orders(value) -> Orders:
            return tuple((str(name), int(power)) for name, power in value)

        return cls(
            int(record["size"]),
            tuple(
                tuple(
                    ExactColorCoefficient(Fraction(re), Fraction(im)) for re, im in row
                )
                for row in record["metric"]
            ),
            tuple(int(t) for t in record["counterterms"]),
            tuple(
                (int(t), int(k), orders(order), int(nc))
                for t, k, order, nc in record["ket_map"]
            ),
            tuple(tuple(enumerate(int(h) for h in row)) for row in record["rows"]),
            frozenset(orders(order) for order in record["colour_zero_orders"]),
            bool(record["colour_zero"]),
        )


def amplitude_record(slice_: Any, exported: Any) -> dict[str, Any]:
    """The amplitude record of a stored loop-induced process
    (``native_process.NativeProcess.export``; ``StoredAmplitude``): exact
    rationals as strings, loop orders as [name, power] lists; no path, user
    or host enters it."""

    basis = slice_.amplitude

    def orders(value: Orders) -> list[list[Any]]:
        return [[name, power] for name, power in value]

    mapping = (
        ket_map(slice_, exported.trees)
        if exported.trees.program.get("counterterms")
        else []
    )
    return {
        "size": len(basis),
        "metric": [
            [[str(cell.real), str(cell.imag)] for cell in row]
            for row in basis.metric_at_three()
        ],
        "counterterms": list(basis.counterterms),
        "ket_map": [[t, k, orders(order), nc] for t, k, order, nc in mapping],
        "rows": [[h for _leg, h in row] for row in exported.rows],
        "colour_zero_orders": [
            orders(order) for order in sorted(slice_.colour_zero_orders)
        ],
        # The request's exact colour zero (``AmplitudeBasis.colour_zero``):
        # one colour-zero order beside live ones does not make it.
        "colour_zero": basis.colour_zero,
    }


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
    the legs' helicities, ``plan_export.born_rows``): the native executor's
    rows, else those of the point's counterterm vectors."""

    rows = getattr(evaluation, "amplitude_rows", None)
    return tuple(evaluation.site_vectors) if rows is None else rows


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
        # The term scale's numerator coefficients (``term_values``).
        evaluation.amplitude_numerators = execution.numerators
    evaluation._assemble()
    return evaluation


def read_native(
    evaluation: Any, output: Mapping, index: int, exported: Any, carrier, numbers
) -> None:
    """Read the native amplitude sum of point ``index`` of an amplitude-mode
    evaluation (``NativeExecutor.reduce``): per (loop order, row) the bare
    amplitude vectors (``native_bare``), with a check their re-test budget
    (``native_budget``: (i/16 pi^2) sum_f |delta Omega_f| per t and k) and,
    when the executor formed it, the term scale (``native_terms``,
    ``term_values``); binary64 squares natively (``native_square``)."""

    size = exported.amplitude
    columns = len(exported.rows) * size

    def vectors(slots, read):
        out = {}
        for o, order in enumerate(exported.orders):
            for r, row in enumerate(exported.rows):
                base = o * columns + r * size
                out[(order, row)] = [read(slots[base + t]) for t in range(size)]
        return out

    # (An empty amplitude basis, an exact colour zero, has no slot.)
    amplitudes = output.get("amplitudes")
    evaluation.native_bare = vectors(
        [] if amplitudes is None else amplitudes[index], carrier
    )
    budget = output.get("amplitude_budget")
    evaluation.native_budget = (
        None
        if budget is None
        else vectors(budget[index], lambda values: [float(x) for x in values])
    )
    terms = output.get("amplitude_scale")
    evaluation.native_terms = (
        None
        if terms is None
        else vectors(terms[index], lambda values: [float(x) for x in values])
    )
    evaluation.native_square = numbers.double
    evaluation.amplitude_rows = tuple(exported.rows)
    kets = output.get("counterterm_kets")
    if kets is not None:
        evaluation.native_kets = native_kets(
            evaluation.slice,
            exported,
            carrier(kets[index]),
            numbers,
            width_of(evaluation),
        )


def width_of(evaluation: Any) -> int:
    """Laurent coefficients per value: [finite, 1/eps, 1/eps^2], or finite."""

    return 3 if evaluation.poles else 1


def ket_map(slice_: Any, trees: Any) -> list[tuple[int, int, Orders, int]]:
    """Per counterterm ket component of a native tree program (its layout
    (ket tensor, vector, N_c power), ``native_trees._counterterm_program``)
    the amplitude basis element of its tensor, the Laurent index of its
    vector's epsilon power, the vector's loop orders and its N_c power."""

    basis = {tensor: t for t, tensor in enumerate(slice_.amplitude.tensors)}
    layout = trees.program["counterterms"]["ket"]
    out = []
    for tensor, vector, nc_power in layout:
        _component, power, order = trees.ket_vectors[vector]
        literal = trees.ket_tensors[tensor]
        t = basis.get(literal)
        if t is None:
            raise KeyError(
                f"a native counterterm ket tensor {literal!r} outside the "
                "amplitude basis"
            )
        out.append((t, -power, order, nc_power))
    return out


def native_kets(
    slice_: Any, exported: Any, values: Sequence[Any], numbers: Any, width: int
) -> dict[tuple[Orders, rx.Row], list[list[Any]]]:
    """A^CT per (loop order, row) from a point's native counterterm kets
    (``V2Executor.evaluate``'s ``counterterm_kets``: rows x ket components,
    the program's layout) on the amplitude basis at N_c = 3 (``ket_map``,
    or a stored process's recorded map; the native twin of
    ``counterterm_kets``)."""

    mapping = getattr(slice_.amplitude, "ket_map", None)
    if mapping is None:
        mapping = exported.__dict__.get("_ket_map")
    if mapping is None:
        mapping = exported.__dict__["_ket_map"] = ket_map(slice_, exported.trees)
    components = len(mapping)
    size = len(slice_.amplitude)
    zero = numbers.zero
    three = numbers.real(3)
    out: dict[tuple[Orders, rx.Row], list[list[Any]]] = {}
    for r, row in enumerate(exported.rows):
        for j, (t, k, order, nc) in enumerate(mapping):
            if k >= width:
                continue
            value = values[r * components + j]
            ket = out.get((order, row))
            if ket is None:
                ket = out[(order, row)] = [[zero] * width for _ in range(size)]
            ket[t][k] = ket[t][k] + value * three**nc
    return out


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
    native = getattr(evaluation, "native_bare", None)
    with numbers.scope():
        bare = (
            bare_amplitudes(evaluation.families, size, numbers, width)
            if native is None
            else native
        )
        kets = getattr(evaluation, "native_kets", None)
        if kets is None:
            if not hasattr(evaluation, "site_vectors"):
                if slice_.amplitude.counterterms:
                    # Never a silent loss of the R2/UV part.
                    raise ValueError(
                        "a loop-induced native point needs the executor's "
                        "counterterm kets (native trees with counterterms)"
                    )
                kets = {}
            else:
                kets = counterterm_kets(slice_, evaluation.site_vectors, numbers, width)
        total = add_amplitudes(bare, kets, numbers=numbers, size=size, width=width)
        gram = metric(slice_, numbers)
        if getattr(evaluation, "native_square", False):
            squared = native_square(
                total,
                gram,
                slice_.sectors,
                rows,
                width,
                evaluation.average,
                slice_.native,
            )
        else:
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


def native_square(
    amplitudes: Mapping[tuple[Orders, rx.Row], Sequence[Sequence[Any]]],
    gram: Sequence[Sequence[Any]],
    sectors: Sequence[tuple[Orders, frozenset[tuple[Orders, Orders]]]],
    rows: Sequence[rx.Row],
    width: int,
    average: Any,
    native: Any,
) -> dict[Orders, tuple[float, ...]]:
    """``square`` of binary64 amplitude vectors by the native canonical
    square (``square_amplitudes``, the one Rust implementation of RULED Q41
    for the Python API and the SDK): dense slots (loop order, row, t) over
    the sectors' loop orders (zero where absent)."""

    orders = sorted(
        {order for _sector, pairs in sectors for pair in pairs for order in pair}
    )
    slot = {order: k for k, order in enumerate(orders)}
    size = len(gram)
    zero = [[0.0, 0.0]] * width
    values = []
    for order in orders:
        for row in rows:
            vector = amplitudes.get((order, row))
            for t in range(size):
                if vector is None:
                    values.append(zero)
                    continue
                values.append([[z.real, z.imag] for z in vector[t][:width]])
    metric_values = [[complex(x).real, complex(x).imag] for line in gram for x in line]
    pairs = [
        sorted((slot[left], slot[right]) for left, right in pairs)
        for _sector, pairs in sectors
    ]
    squared = native.square_amplitudes(
        values,
        len(orders),
        len(rows),
        size,
        width,
        metric_values,
        pairs,
        float(average),
    )
    return {
        sector: tuple(totals)
        for (sector, _pairs), totals in zip(sectors, squared, strict=True)
    }


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
        return [
            _bound_norm([vector[t][k] for t in range(size)], gram) for k in range(width)
        ]

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

    import math

    from .conventions import LOOP_FACTOR_DENOMINATOR

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
    """The re-test inputs of a loop-induced point's stability estimate:
    ``retest_budget`` and ``family_deviations`` (quadratic in the
    colour-metric norm). The scale S of the estimate's floor is formed on
    demand (``stability_scale``: it costs the term scale's reductions)."""

    evaluation.family_deviations = None
    evaluation.retest_budget = None
    native = getattr(evaluation, "native_budget", None)
    if native is not None:
        evaluation.retest_budget = quadratic_budget(evaluation, native)
    elif evaluation.retest_families:
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
    evaluation.absolute = None


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

    return quadratic_budget(
        direct, _rotation_deviations(direct, rotated, _row_phases(direct, rotated, 1))
    )


def rotation_errors(direct: Any, rotated: Any) -> dict:
    """Per (loop order, row), t and Laurent coefficient k the absolute
    deviation bound of a probe's rotation check (``probe``): the lanes and
    counterterm kets of ``rotated`` against ``direct`` after removing each
    row's little-group phase (``rotation_budget``'s comparison), here with
    the phase from every Laurent coefficient (a probe's residues may sit in
    the poles), summed over the lanes so that no two cancel. A true value
    rotates with its row's phase and deviates by rounding only; rounding
    noise deviates by its own size: the residues of a Furry zero (g g > a a
    a), whose numerator coefficients cancel between the loop's two
    orientations, so that its term floor is at rounding too, are covered by
    this budget, not by the floor. Needs the raw lanes (or compares the
    bare amplitudes, ``rotation_budget``)."""

    width = width_of(direct)
    return _rotation_deviations(direct, rotated, _row_phases(direct, rotated, width))


def _row_phases(direct: Any, rotated: Any, width: int) -> dict:
    """Per helicity row the little-group phase of ``rotated`` against
    ``direct``: the phase of sum conj(a') a over the row's loop orders, t
    and its first ``width`` Laurent coefficients (1 where it vanishes)."""

    numbers = direct.numbers
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
                    for k in range(width):
                        total = total + other[t][k].conjugate() * vector[t][k]
            magnitude = abs(total)
            phases[row] = (
                numbers.pair(1)
                if not magnitude
                else total * numbers.real(1 / magnitude)
                if numbers.double
                else total * (1 / numbers.real(magnitude))
            )
    return phases


def _rotation_deviations(direct: Any, rotated: Any, phases: Mapping) -> dict:
    """``lane_errors`` of the phase-corrected lane deviations of ``rotated``
    from ``direct``, plus the counterterm kets' (and, without raw lanes, the
    bare amplitudes') per (loop order, row, t) deviations."""

    numbers = direct.numbers
    width = 3 if direct.poles else 1
    size = len(direct.slice.amplitude)
    with numbers.scope():
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
        # Without raw lanes (the native amplitude sum) the bare amplitudes
        # themselves are compared, per (loop order, row, t).
        parts = [(direct.counterterm_vectors, rotated.counterterm_vectors)]
        if not direct.families and getattr(direct, "native_bare", None) is not None:
            parts.append((direct.bare_vectors, rotated.bare_vectors))
        for key, a, b in (
            (key, mine.get(key), theirs.get(key))
            for mine, theirs in parts
            for key in dict.fromkeys([*mine, *theirs])
        ):
            _order, row = key
            cell = errors.setdefault(key, [[0.0] * width for _ in range(size)])
            for t in range(size):
                for k in range(width):
                    x = zero if a is None else a[t][k]
                    y = zero if b is None else b[t][k]
                    cell[t][k] += float(abs(y * phases[row] - x))
    return errors


def term_values(evaluation: Any) -> dict:
    """Per (loop order, row) the term scale of the bare amplitude per basis
    element t and Laurent coefficient k, sum_f sum_alpha |N_{f,h,t,alpha}|
    |I_{f,alpha,k}| / 16pi^2 (m-lit DECISIONS 2026-11-06 ~09:00 (1)): the
    size of each amplitude component before the reduction's own
    cancellations, from the tensor moments I_{f,alpha} (the integral of
    monomial alpha of family f, one unit-numerator reduction per monomial,
    family and point; TIRust's moments). The native executor forms it on
    request (``V2Options::term_scale``, ``native_terms``); the Python
    reference reduces the unit numerators of its own dressing (the oracle).
    Cached on the evaluation."""

    found = getattr(evaluation, "term_cache", None)
    if found is not None:
        return found
    native = getattr(evaluation, "native_bare", None)
    if native is not None:
        terms = getattr(evaluation, "native_terms", None)
        if terms is None:
            raise ValueError(
                "a native loop-induced point has no term scale: the executor "
                "forms it with the stability checks (term_scale)"
            )
        evaluation.term_cache = terms
        return terms
    out: dict = {}
    numerators = getattr(evaluation, "amplitude_numerators", None) or {}
    if numerators:
        # (Formed on demand, possibly after the point's own scope.)
        with evaluation.numbers.scope():
            moments = _moments(evaluation, numerators)
            out = _term_sums(evaluation, numerators, moments)
    evaluation.term_cache = out
    return out


def _moments(evaluation: Any, numerators: Mapping) -> dict:
    """I_{f,alpha} per (family f, numerator frame) and monomial alpha, as
    Laurent coefficient lists: the unit numerators of one of the family's
    loop orders in that frame (the integrals depend on the propagators and
    the frame alone) reduced through the evaluation's own reducer
    (``_PointEvaluation._groups``)."""

    from dataclasses import replace

    from .poles import reducer_poles

    numbers = evaluation.numbers
    units = {}
    frames = {}
    for (family, order), numerator in sorted(numerators.items()):
        if (family, numerator.frame) in frames:
            continue
        frames[(family, numerator.frame)] = (family, order)
        size = len(numerator.columns[0])
        one, zero = numbers.pair(1), numbers.zero
        units[(family, order)] = replace(
            numerator,
            columns=tuple(
                tuple(one if m == alpha else zero for m in range(size))
                for alpha in range(size)
            ),
        )
    width = max(len(numerator.columns) for numerator in units.values())
    lanes, groups = evaluation._groups(units, 0, list(range(width)))
    arithmetic = "double" if numbers.double else "arbitrary"
    digits = None if numbers.double else numbers.precision.digits
    reducer = evaluation.slice.native.V2Reducer(
        arithmetic, digits, "fast" if numbers.double else "generic"
    )
    direct, _check = reducer.reduce(
        groups, poles=reducer_poles(evaluation.mode), check=False, rescale=False
    )
    frame_of = {key: frame for (_family, frame), key in frames.items()}
    out: dict = {}
    for (family, order, alpha), batch in zip(lanes, direct, strict=True):
        key = (family, frame_of[(family, order)])
        out.setdefault(key, {})[alpha] = evaluation._values(batch, alpha)
    return out


def _term_sums(evaluation: Any, numerators: Mapping, moments: Mapping) -> dict:
    """``term_values`` of the Python reference from its numerators (columns
    (row, t), ``column_labels``) and the families' moments."""

    import math

    from .conventions import LOOP_FACTOR_DENOMINATOR

    width = width_of(evaluation)
    size = len(evaluation.slice.amplitude)
    rows = rows_of(evaluation)
    factor = 1 / (LOOP_FACTOR_DENOMINATOR * math.pi**2)
    out: dict = {}
    for (family, order), numerator in sorted(numerators.items()):
        integrals = moments[(family, numerator.frame)]
        for column, coefficients in enumerate(numerator.columns):
            if not any(coefficients):
                continue
            row, t = rows[column // size], column % size
            cell = out.setdefault((order, row), [[0.0] * width for _ in range(size)])[t]
            for alpha, value in enumerate(coefficients):
                if not value:
                    continue
                magnitude = float(abs(value))
                for k in range(width):
                    cell[k] += factor * magnitude * float(abs(integrals[alpha][k]))
    return out


def row_scales(evaluation: Any) -> dict:
    """The per-row term floor of every (loop order, row): per Laurent
    coefficient k, S_k(h) = max(S_term,k(h), S_term,0(h)) with S_term,k(h) =
    sum_t T_{h,t,k} sqrt(G_tt) + ||A^CT_{h,k}||_G (``term_values`` T; m-lit
    DECISIONS 2026-11-06 ~09:00 (1)). The floor by the row's own finite
    order covers order-k numerator content that cancels in the dressing
    (gg>hh, gg>zz 1/eps^2); the term scale covers families that integrate
    to zero by themselves (Landau-Yang), where scales measured after the
    integration are themselves at rounding. Cached on the evaluation."""

    found = getattr(evaluation, "row_scale_cache", None)
    if found is not None:
        return found
    terms = term_values(evaluation)
    gram = evaluation.gram
    numbers = evaluation.numbers
    width = width_of(evaluation)
    size = len(evaluation.slice.amplitude)
    roots = [max(float(complex(gram[t][t]).real), 0.0) ** 0.5 for t in range(size)]
    kets = evaluation.counterterm_vectors
    out = {}
    with numbers.scope():
        for key in dict.fromkeys([*terms, *kets]):
            vector = terms.get(key)
            ket = kets.get(key)
            values = []
            for k in range(width):
                value = 0.0
                if vector is not None:
                    value = sum(vector[t][k] * roots[t] for t in range(size))
                if ket is not None:
                    value += float(
                        g_norm([ket[t][k] for t in range(size)], gram, numbers)
                    )
                values.append(value)
            out[key] = [max(value, values[0]) for value in values]
    evaluation.row_scale_cache = out
    return out


def scale(evaluation: Any) -> dict:
    """The scale S of the estimate's floor per sector and coefficient
    (``StabilityEstimate.from_checks``' ``scale``): the |A|^2-level product
    of the per-row term floors, sum_pairs kappa sum_h sum_{i+j=k} S_{l,h,i}
    S_{l',h,j} / avg (``row_scales``; PLAN §2.a.8 with the DECISIONS ~09:00
    (1) scale), so that a numerical zero is reported as "zero within the
    estimate" (RULED Q13: no zero is declared)."""

    from .stability import Budget

    scales = row_scales(evaluation)
    width = width_of(evaluation)
    rows = rows_of(evaluation)
    average = float(evaluation.average)
    out = {}
    for sector, pairs in evaluation.slice.sectors:
        totals = [0.0] * width
        for pair in sorted(pairs):
            left, right = pair
            for row in rows:
                a = scales.get((left, row))
                b = scales.get((right, row))
                if a is None or b is None:
                    continue
                for k in range(width):
                    for i in range(k + 1):
                        totals[k] += kappa(pair) * a[i] * b[k - i]
        out[sector] = Budget(tuple(value / average for value in totals))
    return out


def stability_scale(evaluation: Any) -> dict:
    """``scale`` of a point (formed on first use: the Python reference
    reduces its unit numerators then; a native point needs the executor's
    term scale, requested with the stability checks)."""

    if evaluation.absolute is None:
        evaluation.absolute = scale(evaluation)
    return evaluation.absolute


def finiteness(evaluation: Any) -> dict:
    """The amplitude-level finiteness ratio per (loop order, row): max over
    the pole coefficients k = 1, 2 of ||a_{-k}||_G / S_k(h), the residue of
    the amplitude's poles against the row's term floor (``row_scales``;
    PLAN §2.a.7 L3 with the DECISIONS 2026-11-06 ~09:00 (1) scale). A
    one-loop amplitude with no tree is IR finite and UV finite after
    renormalisation, so the ratio is of the order of the carrier's unit
    roundoff (0 for a row with no amplitude). Empty without poles."""

    width = width_of(evaluation)
    if width == 1:
        return {}
    scales = row_scales(evaluation)
    gram = evaluation.gram
    numbers = evaluation.numbers
    size = len(evaluation.slice.amplitude)
    out = {}
    for key, vector in evaluation.amplitude_vectors.items():
        ratio = 0.0
        for k in range(1, width):
            with numbers.scope():
                residue = float(
                    g_norm([vector[t][k] for t in range(size)], gram, numbers)
                )
            if residue:
                bound = scales.get(key, [0.0] * width)[k]
                ratio = max(ratio, residue / bound if bound else float("inf"))
        out[key] = ratio
    return out


# The user's x10 rule (m-lit Q12): a probe residue passes within ten times
# its own stability estimate.
ESTIMATE_FACTOR = 10


def amplitude_estimates(
    evaluation: Any,
    roundoff: float,
    scales: Mapping | None = None,
    rotations: Sequence[Mapping] = (),
) -> dict:
    """Per (loop order, row) and Laurent coefficient k the pair (||a_k||_G,
    E_k): the amplitude's colour-metric norm and its absolute estimate E_k =
    SAFETY x (max over rotations of ||delta a_k||_G + ||delta a_k||_G of the
    re-test) + FLOOR_UNITS x u x S_k(h) (``stability``'s combination at the
    amplitude level, PLAN §2.a.8), with the re-test's ||delta a_k||_G
    bounded by its per-component deviations (``native_budget``, or the
    Python lanes' ``family_deviations``; no two families cancel), a
    rotation's by ``rotation_errors`` (``rotations``), and S_k(h) the row's
    term floor (``row_scales``, or ``scales``: a Ward probe's are the
    physical amplitude's, ``probe``). Needs the reduction re-test and the
    term scale (``probe``)."""

    from .stability import FLOOR_UNITS, SAFETY

    errors = getattr(evaluation, "native_budget", None)
    if errors is None:
        deviations = evaluation.family_deviations
        if deviations is None:
            raise ValueError("an amplitude estimate needs the reduction re-test")
        errors = lane_errors(evaluation, deviations)
    if scales is None:
        scales = row_scales(evaluation)
    gram = evaluation.gram
    numbers = evaluation.numbers
    width = width_of(evaluation)
    size = len(evaluation.slice.amplitude)
    out = {}
    with numbers.scope():
        items = [
            (
                key,
                [
                    g_norm([vector[t][k] for t in range(size)], gram, numbers)
                    for k in range(width)
                ],
            )
            for key, vector in evaluation.amplitude_vectors.items()
        ]
    for key, norms in items:
        error = errors.get(key)
        scale_ = scales.get(key, [0.0] * width)
        entries = []
        for k in range(width):
            residue = float(norms[k])
            budget = (
                0.0
                if error is None
                else _bound_norm([error[t][k] for t in range(size)], gram)
            )
            budget += max(
                (
                    _bound_norm([turned[key][t][k] for t in range(size)], gram)
                    for turned in rotations
                    if key in turned
                ),
                default=0.0,
            )
            entries.append(
                (residue, SAFETY * budget + FLOOR_UNITS * roundoff * scale_[k])
            )
        out[key] = tuple(entries)
    return out


@dataclass(frozen=True)
class ProbeResult:
    """A loop-induced probe of one point (``probe``): per (loop order, row)
    the largest ratio |residue| / E over the probed Laurent coefficients
    (``amplitude_estimates``) and whether every ratio is within
    ``ESTIMATE_FACTOR`` (the x10 rule; pass/fail from the probe's own
    estimate, never a fixed tolerance, m-lit Q12); ``scales`` are the
    physical amplitude's row term floors the estimates stand on."""

    kind: str
    ratios: Mapping[tuple[Orders, rx.Row], float]
    scales: Mapping[tuple[Orders, rx.Row], Sequence[float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def worst(self) -> float:
        return max(self.ratios.values(), default=0.0)

    @property
    def passed(self) -> bool:
        return self.worst <= ESTIMATE_FACTOR


def _probe_runs(executor, runs, scale, parameters, precision, frame, term_scale=False):
    """One native batch of probe runs, (point, Ward value or None) each, at
    one scale, with the reduction re-test in ``frame`` (None: none), the raw
    lanes (``rotation_errors`` sums its deviations over them) and, with
    ``term_scale``, the term scale: (the carrier's precision, the
    evaluations). The points of a batch run concurrently."""

    wards = [ward for _point, ward in runs]
    executor.term_scale = term_scale
    raw, executor.raw_lanes = executor.raw_lanes, True
    try:
        precision, evaluations, failures = executor._evaluated(
            [point for point, _ward in runs],
            [scale] * len(runs),
            dict(parameters or {}),
            precision,
            "all",
            frame,
            "B",
            None if all(ward is None for ward in wards) else wards,
        )
    finally:
        executor.term_scale = False
        executor.raw_lanes = raw
    if failures:
        index = min(failures)
        raise ArithmeticError(f"the probe point failed: {failures[index]}")
    return precision, evaluations


def physical_floors(
    executor: Any,
    point: Sequence[Sequence[Any]],
    scale: Any,
    parameters: Mapping[str, str] | None = None,
) -> dict:
    """The row term floors (``row_scales``) of the physical amplitude at a
    point, from a binary64 run with the term scale: a scale is a magnitude,
    which binary64 gives far better than the x10 rule needs, while the term
    scale's unit-numerator reductions (C(R + 4, 4) per family) cost 51 s in
    double-double against 3 s in binary64 on g g > z z g. Where the
    numerators cancel term by term (a Furry zero) the floor is rounding in
    either carrier; the rotation budget carries those probes
    (``rotation_errors``)."""

    from .assemble import Precision

    _carrier, (evaluation,) = _probe_runs(
        executor, [(point, None)], scale, parameters, Precision.double(), None, True
    )
    return row_scales(evaluation)


def ward_floors(physical: Mapping, leg: int) -> dict:
    """The term floors a Ward probe of leg ``leg`` (1-based) stands on: per
    (loop order, row) the largest physical floor over the rows that differ
    from it only in that leg's helicity (the substituted leg reads one
    vector for every helicity, so its rows repeat the same amplitude)."""

    def merged(row):
        return tuple((i, 0 if i == leg - 1 else h) for i, h in row)

    out: dict = {}
    for (order, row), values in physical.items():
        key = (order, merged(row))
        found = out.get(key)
        out[key] = (
            list(values)
            if found is None
            else [max(a, b) for a, b in zip(found, values, strict=True)]
        )
    return {(order, row): out[(order, merged(row))] for order, row in physical}


def probe(
    executor: Any,
    point: Sequence[Sequence[Any]],
    scale: Any,
    *,
    parameters: Mapping[str, str] | None = None,
    precision: Any = None,
    ward: Any = None,
    checks: Any = None,
    physical: Mapping | None = None,
) -> ProbeResult:
    """The finiteness probe (``ward`` None) or the amplitude-level Ward
    probe of a loop-induced point on a native executor of its slice
    (``plan_export.NativeExecutor``; PLAN §2.a.7, items 5-6).

    * **Finiteness**: a one-loop amplitude without a tree is IR finite and,
      renormalised, UV finite, so its pole coefficients a_{-1}, a_{-2}
      vanish per (loop order, row) in the colour-metric norm; each is
      measured against its estimate on the row's term floor (DECISIONS
      2026-11-06 ~09:00 (1)): a missing family, a wrong scaleless drop or a
      wrong master shows here (it replaces Catani for loop-induced).
    * **Ward** (``ward``: a Ward value, ``ward.gauge_ward``: epsilon_j ->
      c p_j / E_j of a massless gauge boson j): the amplitude with leg j's
      polarisation replaced vanishes at every Laurent coefficient; the bare
      part alone does not, the R2 counterterms restore it, so a missing R2
      vertex shows here (invisible to the poles). The slice must be on the
      full tree kernels (``ward.ward_slice``). Widths enter the probe as
      the caller's parameters (zero widths outside the complex-mass scheme,
      m-lit Q7). The residue is measured against the PHYSICAL amplitude's
      term floor (``ward_floors``; ``physical``: the row floors of the
      physical evaluation at the same point, ``ProbeResult.scales`` of its
      finiteness probe, ``physical_floors`` when not given), as ``ward.py``'s
      criterion measures against the physical evaluation: the substituted
      amplitude's own floor can be at rounding (the numerators of g g > h
      vanish term by term), which would make its estimate meaningless.

    ``checks`` (``stability.StabilityChecks``; default the calibrated
    ``RECOMMENDED_CHECKS``: the rescaled reduction re-test and one generic
    rotation) are the estimate's checks: the re-test runs at the probe's
    precision, each rotation re-evaluates the rotated point (a Ward value
    rotates with it, ``ward.rotated_ward``) and enters by
    ``rotation_errors``, which also covers residues that are rounding of a
    cancellation inside the family numerators (a Furry zero), where the
    term floor is at rounding itself."""

    (result,) = probes(
        executor,
        point,
        scale,
        [ward],
        parameters=parameters,
        precision=precision,
        checks=checks,
        physical=physical,
    )
    return result


def probes(
    executor: Any,
    point: Sequence[Sequence[Any]],
    scale: Any,
    wards: Sequence[Any],
    *,
    parameters: Mapping[str, str] | None = None,
    precision: Any = None,
    checks: Any = None,
    physical: Mapping | None = None,
) -> list[ProbeResult]:
    """``probe`` for every Ward value of ``wards`` (None: the finiteness
    probe) at one point: the probes' runs in one native batch with the
    re-test, their rotated runs in another without it."""

    from .assemble import Precision, _fraction
    from .stability import RECOMMENDED_CHECKS
    from .ward import rotated_ward, ward_leg

    precision = precision or Precision.double_double()
    checks = RECOMMENDED_CHECKS if checks is None else checks
    frame = checks.frame
    if frame is None:
        raise ValueError("a probe needs the reduction re-test")
    exact = tuple(tuple(_fraction(x) for x in p) for p in point)
    moved = [rotation.apply(exact) for rotation in checks.rotations]
    # Two batches: the probes' runs with the re-test, their rotated runs
    # without it (a rotation compares values, ``rotation_errors``).
    carrier, evaluations = _probe_runs(
        executor, [(point, ward) for ward in wards], scale, parameters, precision, frame
    )
    rotated = []
    if moved:
        _carrier, rotated = _probe_runs(
            executor,
            [
                (turned, None if ward is None else rotated_ward(rotation, ward))
                for ward in wards
                for rotation, turned in zip(checks.rotations, moved, strict=True)
            ],
            scale,
            parameters,
            precision,
            None,
        )
    # The term floor is the physical amplitude's, a magnitude: formed once in
    # binary64 (``physical_floors``; in double-double the term scale's unit
    # reductions would dominate the probe's cost) and shared by the Ward
    # probes of the point (``physical``).
    if physical is None:
        physical = physical_floors(executor, point, scale, parameters)
    stride = len(moved)
    out = []
    for index, ward in enumerate(wards):
        evaluation = evaluations[index]
        turned = [
            rotation_errors(evaluation, run)
            for run in rotated[index * stride : (index + 1) * stride]
        ]
        floors = physical if ward is None else ward_floors(physical, ward_leg(ward))
        estimates = amplitude_estimates(
            evaluation, carrier.unit_roundoff, floors, turned
        )
        first = 0 if ward is not None else 1
        ratios = {}
        for key, entries in estimates.items():
            ratio = 0.0
            for residue, estimate in entries[first:]:
                if residue:
                    ratio = max(ratio, residue / estimate if estimate else float("inf"))
            ratios[key] = ratio
        out.append(
            ProbeResult(
                "ward" if ward is not None else "finiteness",
                ratios,
                MappingProxyType(physical),
            )
        )
    return out


# ------------------------------------------------- generation-time probes


class ProbeFailure(ArithmeticError):
    """A generation-time probe of a loop-induced request outside ten times
    its own estimate (m-lit Q12: generation fails unless overridden)."""

    refusal_key = "loop-induced-probe"


def probe_request(slice_: Any) -> tuple[Any, tuple[str, ...]]:
    """The request of the generation-time probe slice and the widths it
    zeroes (m-lit Q7): outside the complex-mass scheme every external width
    parameter of the model that is not zero is fixed to 0 (a fixed width in
    a tree propagator breaks the Ward identity of an external photon or
    gluon line it attaches to, so the probe would fail spuriously); under
    the complex-mass scheme the widths stay (the scheme is gauge invariant).
    The production artifact keeps its widths."""

    from dataclasses import replace

    from pyamplicol.loops.model_cms import convention_record

    from .ew_policy import is_zero

    compiled, request = slice_.compiled, slice_.request
    if convention_record(compiled) is not None:
        return request, ()
    parameters = {record.name: record for record in compiled.ir.parameters}
    fixed = request.fixed_parameters
    widths = sorted(
        {
            str(record.width)
            for record in compiled.ir.particles
            if str(record.width) in parameters
            and parameters[str(record.width)].nature == "external"
            and not is_zero(str(record.width), parameters, fixed)
        }
    )
    if not widths:
        return request, ()
    zeroed = {**dict(fixed), **dict.fromkeys(widths, "0")}
    return replace(request, fixed_parameters=zeroed), tuple(widths)


def probe_point(
    slice_: Any, momenta: Sequence[Sequence[float]], digits: int
) -> tuple[list[list[str]], float | None]:
    """A probe point of the slice's process lifted from a binary64 phase-space
    point (``generation.validation``: decays at rest, scatterings in the
    centre-of-mass frame with the beams along +-z), with momentum
    conservation exact and every mass shell within 10^-digits relative, so
    the probe's residues measure the amplitude, not the point.

    The final three-momenta are read exactly, the last one replaced by minus
    the sum of the others; a scattering's final energies are the on-shell
    ones and its beams carry their total along +-z; a decay's final momenta
    are scaled (Newton, at ``digits``) so that their energies add up to the
    decaying mass. Masses are the slice's (``assemble._leg_masses`` at the
    generation parameters). Returns the decimal strings and the point's
    smallest relative distance |s_I - M^2| / M^2 over the invariants of
    every leg subset I with 2 <= |I| <= n - 2 (a tree propagator's) and
    every massive particle of the model (None without such a subset)."""

    from decimal import Decimal, localcontext
    from itertools import combinations

    from .assemble import _leg_masses

    legs = sorted(slice_.blocks.process.legs, key=lambda leg: leg.label)
    count = sum(1 for leg in legs if leg.is_initial)
    squares = _leg_masses(slice_, slice_.request.fixed_parameters, digits + 10)
    with localcontext() as context:
        context.prec = digits + 10

        def exact(value: Any) -> Fraction:
            return value if isinstance(value, Fraction) else Fraction(value)

        def root(value: Fraction) -> Fraction:
            if value <= 0:
                return Fraction(0)
            return exact(
                Decimal(value.numerator).sqrt() / Decimal(value.denominator).sqrt()
            )

        final = [
            [Fraction(float(x)) for x in momentum[1:]] for momentum in momenta[count:]
        ]
        final[-1] = [-sum(p[i] for p in final[:-1]) for i in range(3)]
        spatial = [sum(x * x for x in p) for p in final]
        masses = squares[count:]

        def energies(scale: Fraction) -> list[Fraction]:
            return [
                root(scale * scale * q + m)
                for q, m in zip(spatial, masses, strict=True)
            ]

        scale = Fraction(1)
        if count == 1:
            target = root(squares[0])
            for _ in range(digits):
                values = energies(scale)
                error = sum(values) - target
                slope = sum(
                    scale * q / e for q, e in zip(spatial, values, strict=True) if e
                )
                if not slope:
                    break
                step = exact(Decimal(error.numerator) / Decimal(error.denominator))
                step = exact(
                    Decimal(step.numerator)
                    / Decimal(step.denominator)
                    / (Decimal(slope.numerator) / Decimal(slope.denominator))
                )
                scale -= step
                if abs(step) <= Fraction(1, 10 ** (digits + 5)) * abs(scale):
                    break
        values = energies(scale)
        final_momenta = [
            [e, *(scale * x for x in p)] for e, p in zip(values, final, strict=True)
        ]
        total = sum(values)
        if count == 1:
            initial = [[total, Fraction(0), Fraction(0), Fraction(0)]]
        else:
            first, second = squares[0], squares[1]
            energy = (total * total + first - second) / (2 * total)
            pz = root(energy * energy - first)
            initial = [
                [energy, Fraction(0), Fraction(0), pz],
                [total - energy, Fraction(0), Fraction(0), -pz],
            ]
        point = [*initial, *final_momenta]
        massive = sorted(
            {
                square
                for square in (
                    _particle_square(slice_, record, digits)
                    for record in slice_.compiled.ir.particles
                )
                if square
            }
        )
        outgoing = [
            [(-x if index < count else x) for x in p] for index, p in enumerate(point)
        ]
        distance = None
        for size in range(2, len(point) - 1):
            for subset in combinations(range(len(point)), size):
                p = [sum(outgoing[i][mu] for i in subset) for mu in range(4)]
                invariant = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]
                for square in massive:
                    gap = float(abs(invariant - square) / square)
                    distance = gap if distance is None else min(distance, gap)
        text = [
            [
                str(Decimal(x.numerator) / Decimal(x.denominator))
                if x.denominator != 1
                else str(x.numerator)
                for x in p
            ]
            for p in point
        ]
    return text, distance


def _particle_square(slice_: Any, record: Any, digits: int) -> Fraction | None:
    """A model particle's squared mass at the generation parameters (None if
    massless or not a real mass, e.g. a complex-mass scheme's pole mass)."""

    from .assemble import _model_masses

    mass = str(record.mass)
    try:
        value = _model_masses(slice_.compiled, slice_.request.fixed_parameters, digits)(
            mass
        )
    except (KeyError, ValueError):
        return None
    if not isinstance(value, Fraction) or not value:
        return None
    return value * value


def generation_probes(
    slice_: Any,
    compiled: Any,
    native: Any,
    candidates: Sequence[Sequence[Sequence[float]]],
    *,
    allow_failure: bool = False,
    precision: Any = None,
) -> dict[str, Any]:
    """The generation-time probes of a loop-induced slice (m-lit Q12, PLAN
    §2.a.7 items 5-6), in double-double by default: the finiteness probe and,
    on one leg of every external massless gauge-boson species (gluons and
    photons), the amplitude-level Ward probe epsilon_j -> p_j / E_j, on
    a slice of the request generated on the full tree kernels at zero widths
    outside the complex-mass scheme (``probe_request``; the production slice
    itself when it already is one). The point is the candidate binary64
    point (``candidates``, the generation's validation points) farthest from
    every tree pole, lifted by ``probe_point``. A probe passes within ten
    times its own estimate (``ESTIMATE_FACTOR``); a failure raises
    ``ProbeFailure`` unless ``allow_failure`` (recorded). An exact colour
    zero has nothing to probe. Massive external vectors take no Ward probe
    here (their Goldstone-equivalence and both-gauges checks are LIT-EW's).

    Returns the record the artifact stores (``generation_filters``)."""

    from .assemble import Precision
    from .plan_export import NativeExecutor
    from .ward import gauge_legs, gauge_ward, ward_slice

    if slice_.amplitude.colour_zero:
        return {"exact_colour_zero": True}
    precision = precision or Precision.double_double()
    request, zeroed = probe_request(slice_)
    blocks = slice_.blocks
    if zeroed or getattr(blocks, "kernel_variant", None) != "full":
        target = ward_slice(compiled, request, native)
    else:
        target = slice_
    digits = 2 * precision.working_digits
    lifted = [probe_point(target, momenta, digits) for momenta in candidates]
    index = max(
        range(len(lifted)),
        key=lambda i: float("inf") if lifted[i][1] is None else lifted[i][1],
    )
    point, distance = lifted[index]
    executor = NativeExecutor(target, forward=False, inputs="native")
    scale = _point_scale(point, sum(1 for leg in blocks.process.legs if leg.is_initial))
    # One Ward probe per external gauge-boson species (its first leg; Bose
    # symmetry relates the others) at c = 1: the amplitude is linear in the
    # substituted vector, so c = i (needed where an interference takes a
    # real part, ``ward.WARD_FACTORS``) has the same norm.
    species = {int(leg.label): int(leg.pdg) for leg in target.blocks.process.legs}
    first: dict[int, int] = {}
    for leg in gauge_legs(target):
        first.setdefault(species[leg], leg)
    legs = sorted(first.values())
    found = probes(
        executor,
        point,
        scale,
        [None, *(gauge_ward(leg, (1, 0), point) for leg in legs)],
        precision=precision,
    )
    results = [("finiteness", None, found[0])] + [
        ("ward", leg, result) for leg, result in zip(legs, found[1:], strict=True)
    ]
    finiteness = results[0][2]
    wards = [result for kind, _, result in results if kind == "ward"]
    record: dict[str, Any] = {
        "precision": precision.arithmetic,
        "widths": "complex-mass" if not zeroed and _complex_mass(slice_) else "zero",
        **({"zeroed_widths": list(zeroed)} if zeroed else {}),
        "candidate": index,
        "pole_distance": None if distance is None else _short(distance),
        "finiteness": {"worst": _short(finiteness.worst), "passed": finiteness.passed},
        "ward": {
            "legs": list(legs),
            "worst": _short(max((r.worst for r in wards), default=0.0)),
            "passed": all(r.passed for r in wards),
        },
    }
    failed = [
        f"{kind} probe"
        + ("" if where is None else f" (leg {where})")
        + f": {result.worst:.3g} times its estimate"
        for kind, where, result in results
        if not result.passed
    ]
    if failed:
        record["failed"] = failed
        if not allow_failure:
            raise ProbeFailure(
                f"{slice_.request.process!r}: the loop-induced generation-time "
                f"probes failed ({'; '.join(failed)}; pass within "
                f"{ESTIMATE_FACTOR} times the probe's own estimate, m-lit Q12): "
                "a missing family or counterterm (finiteness) or R2 vertex "
                "(Ward). generation.allow_failed_probes = true stores the "
                "artifact anyway, with the failure recorded"
            )
        record["allowed_failure"] = True
    return record


def _short(value: float) -> float | str:
    """A ratio for the artifact record: three significant digits (a JSON
    number), "inf" for an infinite one."""

    return float(f"{value:.3g}") if value != float("inf") else "inf"


def _complex_mass(slice_: Any) -> bool:
    from pyamplicol.loops.model_cms import convention_record

    return convention_record(slice_.compiled) is not None


def _point_scale(point: Sequence[Sequence[str]], initial: int) -> Fraction:
    """The probe's mu_R^2: the squared invariant mass of the initial state."""

    total = [sum(Fraction(p[mu]) for p in point[:initial]) for mu in range(4)]
    return total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2


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
    "ESTIMATE_FACTOR",
    "LOOP_INDUCED_INTERFERENCE",
    "AmplitudeSeeds",
    "ProbeFailure",
    "ProbeResult",
    "StoredAmplitude",
    "add_amplitudes",
    "amplitude_estimates",
    "amplitude_record",
    "bare_amplitudes",
    "bind",
    "column_labels",
    "counterterm_kets",
    "finiteness",
    "g_norm",
    "generation_probes",
    "kappa",
    "loop_sectors",
    "metric",
    "physical_floors",
    "probe",
    "probe_point",
    "probe_request",
    "probes",
    "rotation_errors",
    "row_scales",
    "scale",
    "square",
    "stability_scale",
    "term_values",
    "ward_floors",
]
