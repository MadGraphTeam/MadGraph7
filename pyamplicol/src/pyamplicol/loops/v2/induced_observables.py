# SPDX-License-Identifier: 0BSD
"""Observables of a loop-induced process (m-lit LIT-OBS; PLAN §1.5, §1.7,
§2.a.2, §3.4; RULED Q8, Q9, Q13, Q37, Q39).

A loop-induced point is one amplitude vector a_{l,h,t} per loop order l,
helicity row h and amplitude-basis element t (``induced``: the native
amplitude sum, a = bare + A^CT). Every observable is a quadratic form on
that vector: nothing is dressed or reduced again, and the forms are squared
by the one canonical square (``induced.square`` / the native
``square_amplitudes`` in binary64, RULED Q41), with the slice's average
``avg`` (``Normalisation.of``) and its loop x loop sectors. Full colour only
(RULED Q8).

* **Colour-correlated** (``correlated``). A declared history X (an ordered
  pair of colour connections, ``ColorCorrelator``) has the directed matrix
  M_X[t, u] = <bra_X(t)|ket_X(u)> of the amplitude basis at N_c = 3
  (``color.correlator_matrices.literal_correlator_matrices``, computed at
  generation and stored with the process: nothing regenerates at load). Its
  output per sector is (1/avg) sum_h sum_pairs (a_l^dag M a_l' [+ a_l'^dag M
  a_l for l != l']), the history's analogue of the Born-interfered
  B^dag K_BL L + L^dag K_LB B: complex in general, real for a dipole
  T_i.T_j (Hermitian). It is squared as M = H + i K with the Hermitian
  parts H = (M + M^dag)/2 and K = (M - M^dag)/2i, each by the canonical
  square. The history ``"born"`` is the colour metric G itself: |A|^2.
  Colour conservation holds exactly on the basis: sum_{j != i} M_{T_i.T_j}
  = -C_i G (every basis element is a colour singlet).
* **Spin-correlated** (``spin``). A spin vector v of a declared leg j
  replaces the leg's wavefunction on both sides (the Born-interfered
  convention: v takes the place of the leg's polarisation vector, its
  crossing phase applied by the engine as to every source). The amplitude
  is linear in that wavefunction, so A(v) = sum_lambda c_lambda(v)
  A(lambda) with c_lambda = -v . conj(w_lambda) on the leg's own
  polarisation vectors w_lambda at the point (``polarisations``: the tree
  program's source records, HELAS conventions; w . conj(w') = -delta):
  exact for v in their span, which is v . p_j = 0 (for a massless leg
  modulo p_j, whose part vanishes by the Ward identity, the generation-time
  probe checks it: the projection Recola's loop-induced spin correlations
  make). A part of v outside that span (v . p_j != 0: a binary64-built
  k_perp's rounding, or more) has no meaning on the helicity amplitudes:
  the output is that of the transverse vector v - r the projection reads,
  and its estimate carries r (``transverse_residual``; below). No extra
  dressing or reduction: the spin-correlated output is the colour form of
  the history on the vectors A(v) (the other legs' rows summed), every spin
  source assignment of a call on the same evaluations.
* **Resolved** (``resolved``). Per helicity row h, |A(h)|^2/avg (one
  colour-contracted component); with ``AMPLITUDE_BASIS`` per basis element
  t the anchored part Re(conj(a_t) (G a)_t)/avg (the metric
  (E_t G + G E_t)/2, E_t the projection on t): a basis-dependent split
  (T may be linearly dependent at N_c = 3) whose sum over t is the row's
  value.
* **Single-helicity selection** (``selection``, RULED Q39): the sum over
  the selected rows of |A(h)|^2/avg with the same average as the total, so
  the rows of a complete selection sum to it. No parity halving and no
  Bose row mapping (RULED Q37): every row is its own amplitude.

Stability estimates (REQUIREMENTS_ADDENDUM C) per output. ``"born"``
without spin sources over every row is |A|^2 itself: it carries the
inclusive evaluation's estimate (``plan_export.point_estimate``). Every
other output: a rotation's
budget is the output's own deviation at the rotated point (every form is
rotation invariant: the rotated spin vectors and polarisation vectors move
together, and a row's little-group phase cancels in a^dag M a); the
reduction check's budget bounds the form's change from the re-test's
absolute amplitude errors e by the elementwise |M| (sum e^T|M||b| +
|a|^T|M|e + e^T|M|e); a spin vector with a non-transverse part of size
rho (Euclidean, the gauge part along p of a massless leg excluded)
determines the transverse vector it stands for only to within rho: the
budget ``stability.TRANSVERSE`` is the same bound over every transverse
vector within rho of the one read (coefficient changes |dc_h| <= rho |w_h|,
Cauchy-Schwarz), 0 for an exactly transverse v, of the output's size for a
v far from transverse (no refusal: the point's estimate says it, and a
rescue policy fails that point alone); the scale S of the floor is the same
form of the per-row term floors (``induced.term_values`` and the
counterterm kets), never the output itself (RULED Q13: a zero is "zero
within the estimate").

Binary64 outputs run the one Rust implementation the native SDK runs
(``rusticol_loop_reduction::v2_runtime::induced_outputs``, m-lit LIT-SDK,
DECISIONS 2026-11-07 21:50 R7): the carrier forms (``Forms``; an
amplitude-basis anchor's metric formed natively from G), the spin
coefficients (``PointOutputs``) and the spin-projected vectors
(``spin_amplitudes``) through the native extension (``induced_form``,
``induced_anchor_form``, ``induced_spin_coefficients``,
``induced_spin_amplitudes``), the square natively as before; a selection's
row mask runs the executor's row liveness (``NativeExecutor.runs(rows=...)``),
which the SDK's selections run too (m-lit lit-sdk-b). The ``Decimal`` code of
this module stays the oracle (``NATIVE_BINARY64 = False`` runs it in binary64
too) and the double-double / arbitrary-precision path.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from threading import RLock
from typing import Any

from pyamplicol.color.connections import ExactColorCoefficient

from . import induced
from . import reference_executor as rx
from .families import Orders

# The resolved colour axis of a loop-induced result: one colour-contracted
# component per helicity row (its ID the accuracy, "full"), or the
# amplitude-basis anchors (IDs ``ANCHOR_ID``), a basis-dependent split.
CONTRACTED = "contracted"
AMPLITUDE_BASIS = "amplitude-basis"
COLOUR_RESOLUTIONS = (CONTRACTED, AMPLITUDE_BASIS)
ANCHOR_ID = "amplitude-{}"
INCLUSIVE = "born"
# Binary64 outputs through the native implementation the SDK shares (module
# docstring); False runs this module's Decimal oracle in binary64 too.
NATIVE_BINARY64 = True
# The guard digits of a spin leg's polarisation vectors and coefficients over
# the carrier's working digits (``Precision.working_digits``, itself floored
# at the tree replay's digits): their square roots, products and sums round
# in that context, so a coefficient enters the carrier correctly rounded
# (the native one is ``induced_outputs::SPIN_GUARD_DIGITS``).
SPIN_GUARD_DIGITS = 10

Matrix = tuple[tuple[ExactColorCoefficient, ...], ...]


# ------------------------------------------------------------------ colour


def correlator_matrices(basis: Any, histories: Sequence[Any]) -> dict[str, Matrix]:
    """M_X of every declared history X on a slice's amplitude basis
    (``colour.AmplitudeBasis``: its legs and literal tensors), exact at
    N_c = 3, by history id."""

    from pyamplicol.color.correlator_matrices import literal_correlator_matrices

    histories = [history for history in histories if history.id != INCLUSIVE]
    if not histories:
        # Only |A|^2 itself (no correlator declared): no matrix, so a stored
        # amplitude basis without its legs (an artifact without correlator
        # matrices) needs none.
        return {}
    matrices = literal_correlator_matrices(basis.legs, basis.tensors, histories)
    return {
        history.id: matrix for history, matrix in zip(histories, matrices, strict=True)
    }


def declare(slice_: Any, compiled: Any, request: Any, correlators: Any) -> None:
    """Declare the observables ``correlators`` (a ``CorrelatorConfig``) of a
    generated loop-induced slice: the declaration resolved on its amplitude
    basis's colour legs (``all_color_through_order`` expanded), its spin
    classes checked to be vector legs with four-component sources
    (``observables.require_vector_spin_sources``, the tree engine's rule) and
    every history's matrix M_X (``correlator_matrices``), kept as
    ``slice_.amplitude_correlators`` for the stored amplitude record
    (``induced.amplitude_record``). A generation-time selection that keeps
    fewer helicities is refused with declared observables, as for a
    Born-interfered virtual (``selected-correlators``)."""

    from .observables import _refuse_selected_observables, require_vector_spin_sources

    _refuse_selected_observables(request, True)
    resolved = correlators._resolve_color(slice_.amplitude.legs)
    require_vector_spin_sources(compiled, request.process, resolved.spin_correlations)
    slice_.amplitude_correlators = (
        resolved,
        correlator_matrices(slice_.amplitude, resolved.color_correlations),
    )


def correlators_record(declared: tuple[Any, Mapping[str, Matrix]]) -> dict[str, Any]:
    """The stored form of a declaration (``declare``: the resolved
    ``CorrelatorConfig`` and its matrices; the amplitude record's
    ``correlators``): the declaration's JSON and each history's matrix,
    exact rationals as strings, by history id."""

    config, matrices = declared
    return {
        "declaration": config.to_json_dict(),
        "matrices": [
            {
                "id": name,
                "matrix": [
                    [[str(cell.real), str(cell.imag)] for cell in row] for row in matrix
                ],
            }
            for name, matrix in sorted(matrices.items())
        ],
    }


def correlators_from_record(record: Mapping[str, Any]) -> tuple[Any, dict[str, Matrix]]:
    """``correlators_record`` read back: (the ``CorrelatorConfig``, the
    matrices by history id)."""

    from pyamplicol.correlators import CorrelatorConfig

    matrices = {
        str(item["id"]): tuple(
            tuple(ExactColorCoefficient(Fraction(re), Fraction(im)) for re, im in row)
            for row in item["matrix"]
        )
        for item in record["matrices"]
    }
    return CorrelatorConfig.from_json_dict(record["declaration"]), matrices


def hermitian_parts(matrix: Matrix) -> tuple[Matrix, Matrix | None]:
    """(H, K) with M = H + i K, H = (M + M^dag)/2 and K = (M - M^dag)/2i,
    both Hermitian; K is None when M is Hermitian (a dipole)."""

    size = len(matrix)
    half = Fraction(1, 2)
    h = tuple(
        tuple(
            (matrix[t][u] + matrix[u][t].conjugate()).scaled(half) for u in range(size)
        )
        for t in range(size)
    )
    minus_half_i = ExactColorCoefficient(0, Fraction(-1, 2))
    k = tuple(
        tuple(
            (matrix[t][u] + matrix[u][t].conjugate().scaled(-1)) * minus_half_i
            for u in range(size)
        )
        for t in range(size)
    )
    return h, (k if any(cell for row in k for cell in row) else None)


def anchor_metric(gram: Matrix, t: int) -> Matrix:
    """(E_t G + G E_t)/2: a^dag of it a = Re(conj(a_t) (G a)_t)."""

    size = len(gram)
    half = Fraction(1, 2)
    zero = ExactColorCoefficient()
    return tuple(
        tuple(
            (
                (gram[r][u] if r == t else zero) + (gram[r][u] if u == t else zero)
            ).scaled(half)
            for u in range(size)
        )
        for r in range(size)
    )


# -------------------------------------------------------------------- spin


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value)) if not isinstance(value, Decimal) else value


def polarisations(
    sources: Sequence[Mapping[str, Any]], leg: int, momentum: Sequence[Any]
) -> dict[int, tuple[tuple[Decimal, Decimal], ...]]:
    """The polarisation vectors w_h of vector leg ``leg`` (its label) at its
    physical momentum ``momentum``, per source helicity h (the helicity of
    the leg's rows): the tree program's source records
    (``source_fill.sources``: the momentum transform of the crossing, the
    source helicity), before the crossing phase, which the engine applies
    to a spin vector alike. A massive leg's mass is its on-shell invariant
    sqrt(p^2). In the active Decimal context."""

    from pyamplicol.runtime.symbolica_exact import _massive_vector, _massless_vector

    out: dict[int, tuple[tuple[Decimal, Decimal], ...]] = {}
    for source in sources:
        if int(source["leg_label"]) != leg:
            continue
        if source.get("wavefunction_kind") != "vector" or int(source["dimension"]) != 4:
            raise ValueError(
                f"spin-correlated leg {leg} requires a full four-component vector "
                "source"
            )
        p = [_decimal(x) for x in momentum]
        if source["applied_crossing"]["momentum_transform"] == "negate-four-momentum":
            p = [-x for x in p]
        helicity = int(source["source_helicity"])
        if source["source_ir"].get("mass_parameter") is None:
            wave = _massless_vector(p, helicity)
        else:
            mass = (p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]).sqrt()
            wave = _massive_vector(p, helicity, mass)
        out[helicity] = tuple(wave)
    if not out:
        raise ValueError(f"leg {leg} has no source of the tree program")
    return out


def spin_coefficients(
    vector: Sequence[tuple[Any, Any]],
    waves: Mapping[int, Sequence[tuple[Decimal, Decimal]]],
) -> dict[int, tuple[Decimal, Decimal]]:
    """c_h = -v . conj(w_h) (Minkowski, mostly minus) per helicity h: the
    coefficients of v on the polarisation vectors (w_h . conj(w_h') =
    -delta), so that M.v = sum_h c_h M.w_h for v in their span."""

    signs = (1, -1, -1, -1)
    out = {}
    for helicity, wave in waves.items():
        re = Decimal(0)
        im = Decimal(0)
        for sign, (vr, vi), (wr, wi) in zip(signs, vector, wave, strict=True):
            vr, vi = _decimal(vr), _decimal(vi)
            # v * conj(w) = (vr wr + vi wi) + i (vi wr - vr wi)
            re += sign * (vr * wr + vi * wi)
            im += sign * (vi * wr - vr * wi)
        out[helicity] = (-re, -im)
    return out


def massless(sources: Sequence[Mapping[str, Any]], leg: int) -> bool:
    """Whether vector leg ``leg``'s source records are massless (no mass
    parameter; ``polarisations``)."""

    return any(
        int(source["leg_label"]) == leg
        and source["source_ir"].get("mass_parameter") is None
        for source in sources
    )


def transverse_residual(
    vector: Sequence[tuple[Any, Any]],
    momentum: Sequence[Any],
    waves: Mapping[int, Sequence[tuple[Decimal, Decimal]]],
    coefficients: Mapping[int, tuple[Decimal, Decimal]],
    gauge: bool,
) -> Decimal:
    """rho: the Euclidean size of the part of a spin vector v that the
    leg's polarisation vectors w_h do not carry, r = v - sum_h c_h w_h
    (``spin_coefficients``), less its component along the leg's momentum p
    when ``gauge`` (a massless leg: that part vanishes by the Ward
    identity). The w_h (and p for a massless leg) span {v : v.p = 0}, so
    rho is 0, exactly, when v.p = 0 in exact rationals, and otherwise the
    size of the non-transverse part that the projection drops
    (``PointOutputs``: the output is that of the transverse vector v - r,
    the estimate covers every transverse vector within rho of it). In the
    active Decimal context."""

    from .assemble import _fraction

    signs = (1, -1, -1, -1)
    p = [_fraction(x) for x in momentum]
    parts = [(_fraction(re), _fraction(im)) for re, im in vector]
    if not any(
        sum(sign * part[c] * q for sign, part, q in zip(signs, parts, p, strict=True))
        for c in (0, 1)
    ):
        return Decimal(0)
    residual = [[_decimal(re), _decimal(im)] for re, im in vector]
    for h, wave in waves.items():
        cr, ci = coefficients[h]
        for mu, (wr, wi) in enumerate(wave):
            residual[mu][0] -= cr * wr - ci * wi
            residual[mu][1] -= cr * wi + ci * wr
    size = sum(re * re + im * im for re, im in residual)
    if gauge:
        q = [_decimal(x) for x in momentum]
        norm = sum(x * x for x in q)
        along = [
            sum(x * part[c] for x, part in zip(q, residual, strict=True))
            for c in (0, 1)
        ]
        size -= (along[0] * along[0] + along[1] * along[1]) / norm
    return max(size, Decimal(0)).sqrt()


def _norm(wave: Sequence[tuple[Decimal, Decimal]]) -> Decimal:
    """The Euclidean norm of a complex four-vector."""

    return sum(re * re + im * im for re, im in wave).sqrt()


def spin_amplitudes(
    amplitudes: Mapping[tuple[Orders, rx.Row], Sequence[Sequence[Any]]],
    coefficients: Mapping[int, Mapping[int, Any]],
    numbers: Any,
    width: int,
    size: int,
) -> dict[tuple[Orders, rx.Row], list[list[Any]]]:
    """A(v) per (loop order, remaining row): sum over the spin legs'
    helicities of prod_j c_j(h_j) a(row); ``coefficients`` maps each spin
    leg's 0-based index to its carrier coefficients per helicity. The
    remaining row keeps the other legs' helicities (a spin leg's entry is
    dropped). Binary64 runs natively (module docstring)."""

    if numbers.double and NATIVE_BINARY64:
        return _native_spin_amplitudes(amplitudes, coefficients, width, size)
    zero = numbers.zero
    out: dict[tuple[Orders, rx.Row], list[list[Any]]] = {}
    for (order, row), vector in amplitudes.items():
        weight = None
        rest = []
        for leg, h in row:
            found = coefficients.get(leg)
            if found is None:
                rest.append((leg, h))
                continue
            c = found.get(h)
            if c is None:
                weight = None
                break
            weight = c if weight is None else weight * c
        else:
            if weight is None:
                continue
            key = (order, tuple(rest))
            total = out.get(key)
            if total is None:
                total = out[key] = [[zero] * width for _ in range(size)]
            for t in range(size):
                for k in range(width):
                    total[t][k] = total[t][k] + weight * vector[t][k]
    return out


def _native() -> Any:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


def _native_spin_amplitudes(
    amplitudes: Mapping[tuple[Orders, rx.Row], Sequence[Sequence[Any]]],
    coefficients: Mapping[int, Mapping[int, Any]],
    width: int,
    size: int,
) -> dict[tuple[Orders, rx.Row], list[list[Any]]]:
    """``spin_amplitudes`` of binary64 vectors by the native implementation
    (``induced_spin_amplitudes``), in the same insertion order."""

    found = _native().induced_spin_amplitudes(
        [
            (key, [[(z.real, z.imag) for z in cell] for cell in vector])
            for key, vector in amplitudes.items()
        ],
        {
            leg: [(h, c.real, c.imag) for h, c in values.items()]
            for leg, values in coefficients.items()
        },
        width,
        size,
    )
    return {
        (tuple(tuple(item) for item in order), tuple(tuple(item) for item in row)): [
            [complex(re, im) for re, im in cell] for cell in vector
        ]
        for (order, row), vector in found
    }


def _native_coefficients(
    sources_json: str,
    leg: int,
    momentum: Sequence[Any],
    vector: Sequence[tuple[Any, Any]],
    digits: int,
) -> dict[int, complex] | None:
    """The binary64 spin coefficients of one spin leg by the native
    implementation (``induced_spin_coefficients``), or None where an input
    is not the decimal text or ``Decimal`` the native reads exactly as
    ``_decimal`` does (a ``Fraction``, which ``_decimal`` rounds)."""

    parts = [*momentum, *(x for pair in vector for x in pair)]
    if any(not isinstance(x, (str, Decimal, int)) for x in parts):
        return None
    found = _native().induced_spin_coefficients(
        sources_json,
        leg,
        [str(x) for x in momentum],
        [(str(re), str(im)) for re, im in vector],
        digits,
    )
    return {h: complex(re, im) for h, re, im in found}


# ----------------------------------------------------------------- outputs


@dataclass(frozen=True)
class Output:
    """One loop-induced output: the metric of its quadratic form (a history
    id, ``INCLUSIVE`` for G, or ("anchor", t)), the helicity rows it keeps
    (None: all) and the index of the call's spin source assignment it reads
    (``InducedColumns.evaluate``)."""

    key: Hashable
    metric: Any = INCLUSIVE
    rows: frozenset[rx.Row] | None = None
    spin: int = 0


class Forms:
    """The carrier matrices of the outputs: G, every declared history's
    (H, K) and the anchors' metrics, formed once per carrier precision
    (``InducedObservables.forms``) and shared by every point."""

    def __init__(self, gram: Matrix, matrices: Mapping[str, Matrix], numbers: Any):
        self._gram = gram
        self._matrices = matrices
        self._numbers = numbers
        self._found: dict[Any, tuple[Any, Any, Any]] = {}

    def __call__(self, metric: Any) -> tuple[Any, Any, Any]:
        """(H in the carrier, K in the carrier or None, |M| as floats)."""

        found = self._found.get(metric)
        if found is None:
            if metric == INCLUSIVE:
                exact, k = self._gram, None
                h = exact
            elif isinstance(metric, tuple):
                exact = anchor_metric(self._gram, metric[1])
                h, k = exact, None
            else:
                exact = self._matrices[metric]
                h, k = hermitian_parts(exact)
            numbers = self._numbers
            if numbers.double and NATIVE_BINARY64 and isinstance(metric, tuple):
                # An anchor's native form: its metric formed from G by the
                # SDK's Rust code (``induced_anchor_form``, module docstring).
                native_h = _native().induced_anchor_form(
                    [
                        [(str(cell.real), str(cell.imag)) for cell in row]
                        for row in self._gram
                    ],
                    metric[1],
                )
                carrier = [[complex(re, im) for re, im in row] for row in native_h]
                imaginary = None
            elif numbers.double and NATIVE_BINARY64:
                # The native forms (module docstring): H and K of a history's
                # matrix, else the metric itself.
                native_h, native_k = _native().induced_form(
                    [
                        [(str(cell.real), str(cell.imag)) for cell in row]
                        for row in exact
                    ],
                    metric != INCLUSIVE,
                )
                carrier = [[complex(re, im) for re, im in row] for row in native_h]
                imaginary = (
                    None
                    if native_k is None
                    else [[complex(re, im) for re, im in row] for row in native_k]
                )
            else:
                with numbers.scope():
                    carrier = [[numbers.colour(cell, 0) for cell in row] for row in h]
                    imaginary = (
                        None
                        if k is None
                        else [[numbers.colour(cell, 0) for cell in row] for row in k]
                    )
            magnitude = [[abs(complex(cell)) for cell in row] for row in exact]
            found = self._found[metric] = (carrier, imaginary, magnitude)
        return found


def _square(evaluation: Any, amplitudes, metric, rows) -> dict[Orders, tuple]:
    """The canonical square of ``amplitudes`` with ``metric`` (a carrier
    matrix) over ``rows``: natively in binary64 (``induced.native_square``),
    in the carrier's Decimal otherwise (``induced.square``)."""

    slice_ = evaluation.slice
    width = induced.width_of(evaluation)
    if getattr(evaluation, "native_square", False):
        return induced.native_square(
            amplitudes,
            metric,
            slice_.sectors,
            rows,
            width,
            evaluation.average,
            slice_.native,
        )
    return induced.square(
        amplitudes,
        metric,
        slice_.sectors,
        rows,
        width,
        evaluation.numbers,
        evaluation.average,
    )


def form(evaluation: Any, amplitudes, forms: Forms, metric: Any, rows) -> dict:
    """The output of one metric on ``amplitudes`` over ``rows``, per sector
    (a tuple of Laurent coefficients): real for a Hermitian metric, else
    H-square + i K-square."""

    h, k, _magnitude = forms(metric)
    real = _square(evaluation, amplitudes, h, rows)
    if k is None:
        return real
    imaginary = _square(evaluation, amplitudes, k, rows)
    numbers = evaluation.numbers
    return {
        sector: tuple(
            numbers.pair(x, y) if numbers.double else rx.Complex(x, y)
            for x, y in zip(values, imaginary[sector], strict=True)
        )
        for sector, values in real.items()
    }


def _bound(evaluation, magnitude, left, right, rows) -> dict:
    """sum_pairs kappa sum_rows sum_{i+j=k} x_i^T |M| y_j / avg of the
    absolute vectors ``left``/``right`` ((order, row) -> [t][k] floats)
    (``induced.sector_form``)."""

    size = len(magnitude)

    def columns(vectors):
        def of(key):
            vector = vectors.get(key)
            if vector is None:
                return None
            return [[vector[t][k] for t in range(size)] for k in range(len(vector[0]))]

        return of

    def kernel(x, y):
        total = 0.0
        for t in range(size):
            if x[t]:
                total += x[t] * sum(magnitude[t][u] * y[u] for u in range(size))
        return total

    return induced.sector_form(evaluation, rows, columns(left), columns(right), kernel)


def _absolute(vectors) -> dict:
    return {
        key: [[float(abs(z)) for z in cell] for cell in vector]
        for key, vector in vectors.items()
    }


def _add_budgets(*parts: Mapping) -> dict:
    from .stability import Budget

    out = {}
    for part in parts:
        for sector, budget in part.items():
            found = out.get(sector)
            out[sector] = (
                budget
                if found is None
                else Budget(
                    tuple(
                        x + y
                        for x, y in zip(
                            found.coefficients, budget.coefficients, strict=True
                        )
                    )
                )
            )
    return out


def _spin_errors(
    errors: Mapping,
    coefficients: Mapping[int, Mapping[int, Any]],
    changes: Mapping[int, Mapping[int, float]] | None = None,
):
    """Absolute errors of A(v) per (loop order, remaining row): from the
    rows' errors e, sum prod_j |c_j| e; with ``changes`` (delta_j per spin
    leg and helicity: coefficient changes |dc_j| <= delta_j) on e = |a|,
    sum (prod_j (|c_j| + delta_j) - prod_j |c_j|) |a|, the difference
    formed term by term (D <- D (|c_j| + delta_j) + delta_j prod_{i<j} |c_i|),
    so a delta below |c|'s rounding is kept."""

    if not coefficients:
        return errors
    magnitudes = {
        leg: {h: abs(complex(c)) for h, c in values.items()}
        for leg, values in coefficients.items()
    }
    out: dict = {}
    for (order, row), cells in errors.items():
        weight = 1.0
        moved = 0.0
        rest = []
        for leg, h in row:
            found = magnitudes.get(leg)
            if found is None:
                rest.append((leg, h))
                continue
            size = found.get(h, 0.0)
            if changes is not None:
                delta = changes.get(leg, {}).get(h, 0.0)
                moved = moved * (size + delta) + weight * delta
            weight *= size
        if changes is not None:
            weight = moved
        if not weight:
            continue
        target = out.setdefault(
            (order, tuple(rest)), [[0.0] * len(cell) for cell in cells]
        )
        for t, cell in enumerate(cells):
            for k, value in enumerate(cell):
                target[t][k] += weight * value
    return out


class PointOutputs:
    """The outputs of one point evaluation (a ``_NativePoint`` of the
    amplitude-mode executor): its amplitude vectors, the spin legs'
    coefficients of one spin assignment and the carrier forms (``Forms``,
    shared by the call's points)."""

    def __init__(
        self,
        evaluation: Any,
        forms: Forms,
        sources: Sequence[Mapping[str, Any]],
        point: Sequence[Sequence[Any]],
        spin: Mapping[int, Any] | None,
    ):
        self.evaluation = evaluation
        numbers = evaluation.numbers
        self.width = induced.width_of(evaluation)
        self.size = len(evaluation.slice.amplitude)
        self.forms = forms
        self.coefficients: dict[int, dict[int, Any]] = {}
        # delta_j(h) = rho_j |w_h| per spin leg with a non-transverse part
        # (``transverse_residual``): the coefficient changes over the
        # transverse vectors within rho_j of the one the output reads.
        self.changes: dict[int, dict[int, float]] = {}
        self._transverse: dict | None = None
        if spin:
            sources_json = (
                json.dumps(list(sources))
                if numbers.double and NATIVE_BINARY64
                else None
            )
            with localcontext() as context:
                context.prec = numbers.precision.working_digits + SPIN_GUARD_DIGITS
                for leg, vector in spin.items():
                    label = int(leg)
                    momentum = point[label - 1]
                    waves = polarisations(sources, label, momentum)
                    exact = spin_coefficients(vector, waves)
                    native = (
                        None
                        if sources_json is None
                        else _native_coefficients(
                            sources_json,
                            label,
                            momentum,
                            vector,
                            numbers.precision.working_digits,
                        )
                    )
                    self.coefficients[label - 1] = (
                        native
                        if native is not None
                        else {h: numbers.pair(re, im) for h, (re, im) in exact.items()}
                    )
                    rho = transverse_residual(
                        vector, momentum, waves, exact, massless(sources, label)
                    )
                    if rho:
                        self.changes[label - 1] = {
                            h: float(rho * _norm(wave)) for h, wave in waves.items()
                        }
        with numbers.scope():
            self.amplitudes = (
                spin_amplitudes(
                    evaluation.amplitude_vectors,
                    self.coefficients,
                    numbers,
                    self.width,
                    self.size,
                )
                if self.coefficients
                else evaluation.amplitude_vectors
            )
        self.rows = tuple(dict.fromkeys(row for _order, row in self.amplitudes))
        if not self.coefficients:
            self.rows = induced.rows_of(evaluation)

    def rows_of(self, output: Output) -> tuple[rx.Row, ...]:
        if output.rows is None:
            return self.rows
        return tuple(row for row in self.rows if row in output.rows)

    def value(self, output: Output) -> dict[Orders, tuple]:
        with self.evaluation.numbers.scope():
            return form(
                self.evaluation,
                self.amplitudes,
                self.forms,
                output.metric,
                self.rows_of(output),
            )

    def budgets(
        self,
        output: Output,
        rotated: Sequence[PointOutputs],
        frame: Any,
        cross: PointOutputs | None = None,
    ):
        """The named budgets of the output's estimate (module docstring);
        ``cross``: the point's outputs on the other reduction backend (the
        cross-backend check, compared as a rotation's)."""

        from .stability import CROSS, ROTATION, TRANSVERSE, Budget

        direct = self.value(output)
        out: dict[str, dict] = {}
        others = [(ROTATION.format(k), other) for k, other in enumerate(rotated)]
        if cross is not None:
            others.append((CROSS, cross))
        for name, other in others:
            moved = other.value(output)
            out[name] = {
                sector: Budget(
                    tuple(
                        float(abs(y - x))
                        for x, y in zip(values, moved[sector], strict=True)
                    )
                )
                for sector, values in direct.items()
            }
        if frame is not None:
            errors = self._errors()
            if errors is not None:
                out[frame] = self._perturbed(output, errors)
        if self.changes:
            if self._transverse is None:
                self._transverse = _spin_errors(
                    _absolute(self.evaluation.amplitude_vectors),
                    self.coefficients,
                    self.changes,
                )
            out[TRANSVERSE] = self._perturbed(output, self._transverse)
        return out

    def _perturbed(self, output: Output, errors: Mapping) -> dict:
        """The form's change bound for absolute errors e of its vectors b:
        e^T|M||b| + |b|^T|M|e + e^T|M|e (elementwise |M|)."""

        rows = self.rows_of(output)
        _h, _k, magnitude = self.forms(output.metric)
        values = _absolute(self.amplitudes)
        return _add_budgets(
            _bound(self.evaluation, magnitude, errors, values, rows),
            _bound(self.evaluation, magnitude, values, errors, rows),
            _bound(self.evaluation, magnitude, errors, errors, rows),
        )

    def _errors(self) -> dict | None:
        evaluation = self.evaluation
        errors = getattr(evaluation, "native_budget", None)
        if errors is None:
            deviations = getattr(evaluation, "family_deviations", None)
            if deviations is None:
                return None
            errors = induced.lane_errors(evaluation, deviations)
        return _spin_errors(errors, self.coefficients)

    def scale(self, output: Output) -> dict:
        """The floor's scale S: the output's |M| form of the per-row term
        floors (term scale per t and k with the row's finite floor, plus the
        counterterm kets' magnitudes)."""

        evaluation = self.evaluation
        terms = induced.term_values(evaluation)
        kets = _absolute(evaluation.counterterm_vectors)
        floors: dict = {}
        for key in dict.fromkeys([*terms, *kets]):
            cells = []
            for t in range(self.size):
                values = [0.0] * self.width
                for source in (terms.get(key), kets.get(key)):
                    if source is not None:
                        for k in range(self.width):
                            values[k] += float(source[t][k])
                cells.append([max(value, values[0]) for value in values])
            floors[key] = cells
        floors = _spin_errors(floors, self.coefficients)
        _h, _k, magnitude = self.forms(output.metric)
        return _bound(evaluation, magnitude, floors, floors, self.rows_of(output))

    def estimate(
        self, output: Output, rotated, frame, roundoff, cross=None, not_run=None
    ) -> Any:
        """The output's estimate; ``cross`` and ``not_run`` as ``budgets``'
        and ``StabilityEstimate.from_checks``'."""

        from .assemble import Laurent
        from .stability import StabilityEstimate

        values = {
            sector: Laurent(*laurent) for sector, laurent in self.value(output).items()
        }
        return StabilityEstimate.from_checks(
            values,
            self.budgets(output, rotated, frame, cross),
            scale=self.scale(output),
            roundoff=roundoff,
            not_run=not_run,
        )


def finiteness(evaluation: Any, rows: Sequence[rx.Row] | None) -> float | None:
    """The amplitude-level finiteness ratio of an output
    (``induced.point_finiteness``) over the helicity rows it reads (None:
    every row): under a row mask the other rows are not dressed, so their
    counterterm kets alone are no amplitude."""

    if induced.point_finiteness(evaluation) is None:
        return None
    ratios = induced.finiteness(evaluation)
    kept = None if rows is None else frozenset(rows)
    return max(
        (
            ratio
            for (_order, row), ratio in ratios.items()
            if kept is None or row in kept
        ),
        default=0.0,
    )


def point_result(
    evaluation: Any,
    values: Mapping[Orders, tuple],
    estimate=None,
    rows: Sequence[rx.Row] | None = None,
):
    """A ``PointResult`` of one output: its values as the "virtual"
    component per sector (no Born, no other component: a square has no
    additive bare/R2/UV split); ``rows`` the helicity rows it reads (its
    finiteness ratio's)."""

    from .assemble import Laurent, PointResult

    numbers = evaluation.numbers
    width = induced.width_of(evaluation)
    zero = numbers.real(0)
    virtual = {sector: Laurent(*laurent) for sector, laurent in values.items()}
    totals = [zero]
    for k in range(width):
        totals.append(sum((laurent[k] for laurent in values.values()), zero))
    return PointResult(
        {},
        {"virtual": virtual},
        tuple(totals),
        {},
        None,
        stability=estimate,
        poles=evaluation.mode,
        finiteness=finiteness(evaluation, rows),
    )


# ------------------------------------------------------------- runners


class InducedColumns:
    """The outputs of one call kind on a loop-induced process
    (``V2ObservablesRuntime``'s native column runner): ``outputs`` are the
    call's ``Output`` records, evaluated on the process's own amplitude-mode
    executor (one native batch per call, plus one per rotation, whatever
    the outputs' spin source assignments)."""

    def __init__(
        self,
        observables: InducedObservables,
        outputs: Sequence[Output],
        member: Any = None,
        *,
        resolved: bool = False,
    ):
        self.observables = observables
        self.outputs = tuple(outputs)
        self.member = member
        # Resolved outputs are keyed (row, component): a member's keys carry
        # its own helicity rows (``observables._member_key``).
        self.resolved = resolved

    def _rows(self) -> frozenset[rx.Row] | None:
        """The helicity rows the call's outputs read when every output
        restricts them (a single-helicity selection): the executor then
        dresses those rows only (``NativeExecutor.runs(rows=...)``); None
        dresses every row."""

        if any(output.rows is None for output in self.outputs):
            return None
        kept = frozenset(row for output in self.outputs for row in output.rows)
        return None if kept >= frozenset(self.observables.rows) else kept

    def evaluate(
        self,
        points,
        scales,
        *,
        spin=None,
        parameters=None,
        precision=None,
        poles="all",
        stability=None,
        ward=None,
        reduction=None,
    ):
        """Per point the outputs by key (None for a failed point:
        ``FailedPoints``). ``spin`` per point is one spin source assignment
        (a mapping or None) or a tuple of them, which the outputs index
        (``Output.spin``); ``reduction`` the call's effective backend (the
        executor refuses one it does not run)."""

        from .observables import _member_key, _rotated_point, _rotated_spin
        from .plan_export import FailedPoints, point_estimate
        from .stability import StabilityChecks

        if ward is not None:
            raise NotImplementedError(
                "the asymmetric Ward mode is a validation tool of the generated "
                "slice (ward.ward_slice)"
            )
        stability = stability or StabilityChecks()
        spins = (
            list(spin)
            if spin is not None and not isinstance(spin, Mapping)
            else [spin] * len(points)
        )
        if len(spins) != len(points):
            raise ValueError("one spin source assignment per point")
        assignments = [
            value if isinstance(value, tuple) else (value,) for value in spins
        ]
        member = self.member
        if member is not None:
            points = [member.point(point) for point in points]
            assignments = [
                tuple(member.spin(value) for value in values) for values in assignments
            ]
        observables = self.observables
        with observables.lock:
            # The executor is the artifact runtime's own (shared state: its
            # row mask and term scale), run under its lock.
            precision, references, rotated, failures = observables.executor.runs(
                points,
                scales,
                parameters=parameters,
                precision=precision,
                poles=poles,
                stability=stability,
                rows=self._rows(),
                reduction=reduction,
            )
        sources = observables.sources
        forms = observables.forms(precision)
        used = sorted({output.spin for output in self.outputs})
        out: list[Any] = []
        for index, reference in enumerate(references):
            if index in failures:
                out.append(None)
                continue
            point = points[index]
            outputs = {
                g: PointOutputs(reference, forms, sources, point, assignments[index][g])
                for g in used
            }
            turned: dict[int, list[PointOutputs]] = {}

            def rotations(
                g: int, index: int = index, point: Any = point, turned: dict = turned
            ) -> list:
                """The assignment's outputs at the rotated points (formed for
                the outputs whose estimate reads them)."""

                found = turned.get(g)
                if found is None:
                    value = assignments[index][g]
                    found = turned[g] = [
                        PointOutputs(
                            run[index],
                            forms,
                            sources,
                            _rotated_point(rotation, point),
                            _rotated_spin(rotation, value),
                        )
                        for rotation, run in zip(
                            stability.rotations, rotated, strict=True
                        )
                    ]
                return found

            crossed: dict[int, PointOutputs] = {}

            def cross(
                g: int, index: int = index, point: Any = point, crossed: dict = crossed
            ) -> tuple[PointOutputs | None, dict[str, str]]:
                """The assignment's outputs on the other reduction backend
                (the cross-backend check), or None and why it did not run."""

                from .stability import CROSS

                if not stability.cross_backend:
                    return None, {}
                run, why = references[index].cross_run
                if run is None:
                    return None, {CROSS: why}
                found = crossed.get(g)
                if found is None:
                    found = crossed[g] = PointOutputs(
                        run, forms, sources, point, assignments[index][g]
                    )
                return found, {}

            result = {}
            for output in self.outputs:
                g = output.spin
                inclusive = (
                    output.metric == INCLUSIVE
                    and output.rows is None
                    and not outputs[g].coefficients
                )
                estimate = None
                if stability.enabled and inclusive:
                    # |A|^2 itself: the inclusive evaluation's estimate
                    # (``NativeExecutor.evaluate``), the same quantity.
                    estimate = point_estimate(
                        reference,
                        [run[index] for run in rotated],
                        stability,
                        precision.unit_roundoff,
                    )
                elif stability.enabled:
                    other, skipped = cross(g)
                    estimate = outputs[g].estimate(
                        output,
                        rotations(g),
                        stability.reduction,
                        precision.unit_roundoff,
                        other,
                        skipped,
                    )
                value = point_result(
                    reference, outputs[g].value(output), estimate, output.rows
                )
                key = output.key
                if member is not None and self.resolved:
                    key = _member_key(member, key)
                result[key] = value
            out.append(result)
        if failures:
            raise FailedPoints(out, failures)
        return out


class InducedObservables:
    """The observables of a loop-induced process (the subset of
    ``observables.ObservableSlice`` that ``runtime.virtual.
    V2InducedObservablesRuntime`` reads): its declared colour histories and
    spin classes, its helicity rows, the stored correlator matrices and the
    process's own amplitude-mode executor (a stored process's, or an
    in-memory slice's native-input executor). ``current`` returns the
    executor to run (the artifact runtime's, which ``clear`` replaces;
    default the given one) and ``lock`` guards its runs (the artifact
    runtime's lock, whose inclusive calls share the executor)."""

    def __init__(
        self,
        executor: Any,
        histories: Sequence[Any] = (),
        spin_classes: Sequence[Sequence[int]] = (),
        matrices: Mapping[str, Matrix] | None = None,
        *,
        current: Callable[[], Any] | None = None,
        lock: Any = None,
    ):
        self._current = current or (lambda: executor)
        self.lock = lock if lock is not None else RLock()
        self.slice = executor.slice
        amplitude = self.slice.amplitude
        if amplitude is None:
            raise ValueError("InducedObservables needs a loop-induced process")
        declared = getattr(amplitude, "correlators", None)
        if declared is None:
            declared = getattr(self.slice, "amplitude_correlators", None)
        if declared is not None and not histories and not spin_classes:
            # The generation's declaration (``declare``) and its matrices.
            config, matrices = declared
            histories = config.color_correlations
            spin_classes = config.spin_correlations
        self.histories = tuple(histories)
        self.spin_classes = tuple(tuple(int(leg) for leg in c) for c in spin_classes)
        if matrices is None:
            matrices = (
                declared[1]
                if declared is not None
                else correlator_matrices(amplitude, self.histories)
            )
        missing = [h.id for h in self.histories if h.id not in matrices]
        if missing:
            raise ValueError(f"colour histories {missing} have no stored matrix")
        self.matrices = dict(matrices)
        rows = getattr(amplitude, "rows", None)
        if rows is None:
            rows = executor.exported.rows
        self.rows = tuple(rows)
        self.components: tuple[int, ...] = ()
        self.flows: tuple[Any, ...] = ()
        self._forms: dict[Any, Forms] = {}

    @classmethod
    def of_slice(cls, slice_: Any, correlators: Any = None, **options: Any):
        """The observables of an in-memory loop-induced slice on its own
        native-input executor (``options``: the executor's)."""

        from .plan_export import NativeExecutor

        executor = NativeExecutor(slice_, forward=False, inputs="native", **options)
        histories = () if correlators is None else correlators.color_correlations
        spin = () if correlators is None else correlators.spin_correlations
        return cls(executor, histories, spin)

    @property
    def executor(self) -> Any:
        return self._current()

    @property
    def sources(self) -> Sequence[Mapping[str, Any]]:
        return self.executor.exported.trees.program["source_fill"]["sources"]

    def forms(self, precision: Any) -> Forms:
        """The carrier forms of the outputs at ``precision`` (a ``Precision``),
        built once (point independent)."""

        from .assemble import _Numbers

        with self.lock:
            found = self._forms.get(precision)
            if found is None:
                found = self._forms[precision] = Forms(
                    self.slice.amplitude.metric_at_three(),
                    self.matrices,
                    _Numbers(precision),
                )
            return found

    def history(self, identifier: str) -> Any:
        from pyamplicol.color.correlator_matrices import ColorCorrelator

        if identifier == INCLUSIVE:
            return ColorCorrelator(INCLUSIVE)
        for history in self.histories:
            if history.id == identifier:
                return history
        raise KeyError(f"undeclared colour correlation {identifier!r}")

    def native_columns(
        self, kind: str, value: Any, *, member: Any = None, **_options: Any
    ) -> InducedColumns:
        """The runner of one call kind (``V2ObservablesRuntime._native_columns``
        keys): ``correlated`` (history ids, or (spin source assignment,
        history id) pairs: ``V2InducedObservablesRuntime.
        evaluate_correlated_many``), ``resolved`` (a colour resolution,
        ``COLOUR_RESOLUTIONS``) or ``selection`` (a ``selection.column``: its
        helicity rows, a ``member``'s own)."""

        from .selection import SELECTION

        if kind == "correlated":
            outputs = []
            for item in dict.fromkeys(value):
                spin, name = item if isinstance(item, tuple) else (0, item)
                self.history(name)
                outputs.append(Output(item, name, None, spin))
        elif kind == "resolved":
            if value not in COLOUR_RESOLUTIONS:
                raise ValueError(f"unknown loop-induced resolution {value!r}")
            anchors = (
                (None,)
                if value == CONTRACTED
                else tuple(range(len(self.slice.amplitude)))
            )
            outputs = [
                Output(
                    (row, b),
                    INCLUSIVE if b is None else ("anchor", b),
                    frozenset({row}),
                )
                for row in self.rows
                for b in anchors
            ]
        elif kind == SELECTION:
            if value.basis is not None:
                raise ValueError("a loop-induced process has no colour-flow selection")
            rows = value.rows
            if member is not None and rows is not None:
                # A member's selection names its own rows.
                representative = {member.row(row): row for row in self.rows}
                rows = frozenset(representative[row] for row in rows)
            outputs = [Output(value.key, INCLUSIVE, rows)]
        else:
            raise ValueError(f"unknown column set {kind!r}")
        return InducedColumns(self, outputs, member, resolved=kind == "resolved")


__all__ = [
    "AMPLITUDE_BASIS",
    "ANCHOR_ID",
    "COLOUR_RESOLUTIONS",
    "CONTRACTED",
    "NATIVE_BINARY64",
    "InducedColumns",
    "InducedObservables",
    "Output",
    "PointOutputs",
    "correlator_matrices",
    "correlators_from_record",
    "correlators_record",
    "declare",
    "hermitian_parts",
    "massless",
    "polarisations",
    "spin_amplitudes",
    "spin_coefficients",
    "transverse_residual",
]
