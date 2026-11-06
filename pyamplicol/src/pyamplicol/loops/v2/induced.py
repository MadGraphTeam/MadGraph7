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
from dataclasses import dataclass
from fractions import Fraction
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
    (``ket_map``), the helicity rows of the amplitude-mode plan and the
    exact colour-zero loop orders (RULED Q13's colour-zero record)."""

    size: int
    metric: tuple[tuple[Any, ...], ...]
    counterterms: tuple[int, ...]
    ket_map: tuple[tuple[int, int, Orders, int], ...]
    rows: tuple[rx.Row, ...]
    colour_zero_orders: frozenset[Orders]

    def __len__(self) -> int:
        return self.size

    def metric_at_three(self) -> tuple[tuple[Any, ...], ...]:
        return self.metric

    @property
    def colour_zero(self) -> bool:
        """No coefficient cell and no counterterm ket: every amplitude
        component vanishes by colour algebra (the record's
        ``colour_zero_orders`` name the orders)."""

        return self.size == 0 or (
            not self.counterterms and bool(self.colour_zero_orders)
        )

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

    evaluation.native_bare = vectors(output["amplitudes"][index], carrier)
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
        values, len(rows), size, width, metric_values, pairs, float(average)
    )
    return {
        sector: tuple(totals)
        for (sector, _pairs), totals in zip(sectors, squared, strict=True)
    }


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
                abs(
                    (a[t][k] if a is not None else zero)
                    - (b[t][k] if b is not None else zero)
                )
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
    return quadratic_budget(direct, errors)


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
    for key in dict.fromkeys([*terms, *kets]):
        vector = terms.get(key)
        ket = kets.get(key)
        values = []
        for k in range(width):
            value = 0.0
            if vector is not None:
                value = sum(vector[t][k] * roots[t] for t in range(size))
            if ket is not None:
                value += float(g_norm([ket[t][k] for t in range(size)], gram, numbers))
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
            residue = float(g_norm([vector[t][k] for t in range(size)], gram, numbers))
            if residue:
                bound = scales.get(key, [0.0] * width)[k]
                ratio = max(ratio, residue / bound if bound else float("inf"))
        out[key] = ratio
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
