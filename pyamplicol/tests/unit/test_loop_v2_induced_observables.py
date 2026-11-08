# SPDX-License-Identifier: 0BSD
"""Observables of loop-induced processes (m-lit LIT-OBS; PLAN §3.4, §1.5,
§1.7; RULED Q8, Q9, Q37, Q39; validation row LI-O1).

Every output is a quadratic form on the point's amplitude vector
(``loops.v2.induced_observables``):

* colour-correlated |A|^2 on the amplitude basis equal Recola's loop-induced
  colour correlations (``compute_colour_correlation_rcl(..., 'NLO')`` of a
  process without a Born; ``tests/fixtures/loop/required/li/correlations``,
  ``tools/loop_reference/recola_correlations.py``) on g g > h g g and
  u u~ > h g g, and the exact colour-algebra identity sum_j T_i.T_j = -C_i
  holds exactly on the basis matrices and on the evaluated values;
* spin correlations equal Recola's loop-induced ones (v transverse, with a
  gauge component), the literal-source evaluation of the same vector (the
  asymmetric Ward mode's loop-side source on a full-kernel slice: the
  amplitude with the leg's wavefunction replaced) for a gluon and a massive
  Z, and the finite differences of the resolved values (v = the leg's own
  polarisation vectors, their sums and differences);
* the helicity-resolved rows and the single-helicity selections sum to the
  total (row LI-O1), each selection equal to its resolved row; the
  amplitude-basis anchors of a row sum to it;
* binary64 within ten times its estimate of double-double: correlated
  outputs with the recommended checks, the amplitude-basis anchors with the
  both-backends check's (``tools/loop_v2_observables_backends.py``);
* full colour only (no colour-flow selection), the colour axis option is
  loop-induced only, undeclared histories and spin legs are refused.
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tests.unit.test_loop_v2_induced import _fixture, _request, _Rows

from .loop_v2_induced_observables_rows import ROWS, coloured, generate

ROOT = Path(__file__).resolve().parents[2]
CORRELATIONS = ROOT / "tests" / "fixtures" / "loop" / "required" / "li" / "correlations"
LAX = 1e-5
# Rounding of two binary64 evaluations of the same quantity (the native
# square against Python's, the literal source against the projection).
ROUNDING = 1e-11
CASIMIR = {8: Fraction(3), 3: Fraction(4, 3), -3: Fraction(4, 3)}


def _recola(name):
    return json.loads((CORRELATIONS / f"{name}-recola.json").read_text())


# A permutation alias of u u~ > h g g: u~ u > h g g (the initial legs
# exchanged).
ALIAS = ("p_swap", "p", (1, 0, 2, 3, 4))


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li"))


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: generate(
            tmp_path_factory.mktemp(name),
            name,
            expression,
            (ALIAS,) if name == "li_uux_hgg" else (),
        )
        for name, expression in ROWS.items()
    }


def _runtime(artifacts, name):
    from pyamplicol import Runtime

    return Runtime.load(artifacts[name], process="p")


def _points(name):
    fixture = _fixture(name)
    return fixture["points"], [str(Fraction(mu) ** 2) for mu in fixture["mu_r"]]


def _spin(vector):
    return tuple((Decimal(re), Decimal(im)) for re, im in vector)


def _relative(value, reference):
    return abs(complex(value) - complex(reference)) / abs(complex(reference))


def _served_by(result):
    """The backend that served each point of an "auto" result (m-lit F3:
    TIRust for loop-induced processes; a point TIRust refuses or fails is
    the per-point fallback's, NinjaRS, its lane recording the reason)."""

    out = []
    for lane in result.diagnostics:
        record = lane.reduction
        assert record.requested == "auto"
        if record.effective == "ninjars":
            assert record.reason.startswith("auto: NinjaRS, the per-point fallback: ")
        else:
            assert (record.effective, record.fallback) == ("tirust", "ninjars")
        out.append(record.effective)
    return out


def _tirust_or_fallback(explicit):
    """Per point the backend "auto" takes: TIRust where the explicit TIRust
    call (``on_failure="return"``) serves the point, else NinjaRS."""

    return ["tirust" if value is not None else "ninjars" for value in explicit.values]


# ------------------------------------------------------------------ units


@pytest.mark.loop_fast
def test_literal_matrices_keep_the_metric_and_colour_conservation():
    """The order-zero history is the colour metric; sum_{j != i} T_i.T_j =
    -C_i on literal colour-singlet tensors (exact, N_c = 3); dipoles are
    Hermitian, a non-Hermitian history splits into H + i K."""

    from pyamplicol.color.connections import ColorLeg, ColorTensor
    from pyamplicol.color.correlator_matrices import (
        ColorCorrelator,
        literal_correlator_matrices,
    )
    from pyamplicol.loops.v2.induced_observables import hermitian_parts

    legs = (ColorLeg(1, 8), ColorLeg(2, 8), ColorLeg(3, 8), ColorLeg(4, 8))
    tensors = (
        ColorTensor(traces=((1, 2, 3, 4),)),
        ColorTensor(traces=((1, 3, 2, 4),)),
        ColorTensor(traces=((1, 2), (3, 4))),
    )
    dipoles = [
        ColorCorrelator.dipole(f"{i}{j}", i, j)
        for i in range(1, 5)
        for j in range(1, 5)
        if i != j
    ]
    identity, *matrices = literal_correlator_matrices(
        legs, tensors, [ColorCorrelator("born"), *dipoles]
    )
    by_id = dict(zip((d.id for d in dipoles), matrices, strict=True))
    size = len(tensors)
    for i in range(1, 5):
        for t in range(size):
            for u in range(size):
                total = sum(
                    (by_id[f"{i}{j}"][t][u] for j in range(1, 5) if j != i),
                    start=type(identity[t][u])(),
                )
                assert total == identity[t][u].scaled(-3)
    for matrix in matrices:
        h, k = hermitian_parts(matrix)
        assert k is None and h == matrix
    # A history of two emissions on different legs per side is not
    # Hermitian in general: M = H + i K exactly.
    from pyamplicol.color.connections import EmitGluon

    history = ColorCorrelator(
        "two",
        (EmitGluon(1, -1), EmitGluon(2, -2)),
        (EmitGluon(3, -2), EmitGluon(4, -1)),
    )
    (matrix,) = literal_correlator_matrices(legs, tensors, [history])
    h, k = hermitian_parts(matrix)
    from pyamplicol.color.connections import ExactColorCoefficient

    i = ExactColorCoefficient(0, 1)
    for t in range(size):
        for u in range(size):
            kk = k[t][u] if k is not None else ExactColorCoefficient()
            assert h[t][u] + i * kk == matrix[t][u]


@pytest.mark.loop_fast
def test_the_transverse_residual_is_the_part_the_projection_drops():
    """rho (``transverse_residual``) is the Euclidean size of v - sum_h c_h
    w_h, the gauge part along p of a massless leg excluded: exactly 0 for
    v.p = 0 in rationals (a transverse vector, p itself, their sum),
    |v.p|/|p| for a massless leg's non-transverse part (also the rounding of
    a binary64-built transverse vector) and |v.p| |p| / m^2 for a massive
    leg's (its dropped direction is p)."""

    from decimal import localcontext

    from pyamplicol.loops.v2.induced_observables import (
        spin_coefficients,
        transverse_residual,
    )
    from pyamplicol.runtime.symbolica_exact import _massive_vector, _massless_vector

    zero = (Decimal(0), Decimal(0))
    with localcontext() as context:
        context.prec = 50
        p = [Decimal(5), Decimal(3), Decimal(0), Decimal(4)]
        waves = {h: _massless_vector(p, h) for h in (-1, 1)}

        def rho(vector, momentum=p, waves=waves, gauge=True):
            coefficients = spin_coefficients(vector, waves)
            return transverse_residual(vector, momentum, waves, coefficients, gauge)

        transverse = (zero, zero, (Decimal(0), Decimal(1)), zero)
        gauge = tuple((x, Decimal(0)) for x in p)
        both = tuple(
            (a[0] + 7 * b[0], a[1]) for a, b in zip(transverse, gauge, strict=True)
        )
        assert rho(transverse) == rho(gauge) == rho(both) == 0
        timelike = ((Decimal(1), Decimal(0)), zero, zero, zero)
        norm = sum(x * x for x in p).sqrt()
        assert abs(rho(timelike) - 5 / norm) < Decimal("1e-40")
        shifted = tuple(
            (a[0] + b[0], a[1]) for a, b in zip(both, timelike, strict=True)
        )
        assert abs(rho(shifted) - 5 / norm) < Decimal("1e-40")
        # (0, 4/5, 0, -3/5) rounded to binary64: v.p = -(3 x 0.8 - 4 x 0.6)
        # in the rounded values, a nonzero rounding.
        rounded = (
            zero,
            (Decimal.from_float(0.8), Decimal(0)),
            zero,
            (Decimal.from_float(-0.6), Decimal(0)),
        )
        dot = 3 * rounded[1][0] + 4 * rounded[3][0]
        assert dot != 0
        assert abs(rho(rounded) - abs(dot) / norm) < Decimal("1e-40")
        mass = Decimal(91)
        q = [Decimal(0), Decimal(12), Decimal(-30), Decimal(38)]
        q[0] = (mass**2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2).sqrt()
        massive = {h: _massive_vector(q, h, mass) for h in (-1, 0, 1)}
        found = rho(timelike, q, massive, False)
        expected = q[0] * sum(x * x for x in q).sqrt() / mass**2
        assert abs(found - expected) < Decimal("1e-30") * expected


@pytest.mark.loop_fast
def test_spin_coefficients_invert_the_polarisation_vectors():
    """c_h(w_h') = delta: the leg's polarisation vectors are orthonormal
    (w . conj(w') = -delta), massless and massive, for both energy signs;
    p_j of a massless leg has no physical component."""

    from decimal import localcontext

    from pyamplicol.loops.v2.induced_observables import spin_coefficients
    from pyamplicol.runtime.symbolica_exact import _massive_vector, _massless_vector

    with localcontext() as context:
        context.prec = 40
        for sign in (1, -1):
            p = [Decimal(sign * 50), Decimal(12), Decimal(-30), Decimal(38)]
            p[0] = sign * (p[1] ** 2 + p[2] ** 2 + p[3] ** 2).sqrt()
            waves = {h: _massless_vector(p, h) for h in (-1, 1)}
            for h, wave in waves.items():
                for other, c in spin_coefficients(wave, waves).items():
                    assert abs(c[0] - (other == h)) < Decimal("1e-35")
                    assert abs(c[1]) < Decimal("1e-35")
            gauge = spin_coefficients([(x, Decimal(0)) for x in p], waves)
            assert all(
                abs(re) + abs(im) < Decimal("1e-35") for re, im in gauge.values()
            )
            mass = Decimal(91)
            q = [Decimal(0), Decimal(12), Decimal(-30), Decimal(38)]
            q[0] = sign * (mass**2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2).sqrt()
            waves = {h: _massive_vector(q, h, mass) for h in (-1, 0, 1)}
            for h, wave in waves.items():
                for other, c in spin_coefficients(wave, waves).items():
                    assert abs(c[0] - (other == h)) < Decimal("1e-35")
                    assert abs(c[1]) < Decimal("1e-35")


# ------------------------------------------------------------ references


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", sorted(ROWS))
def test_colour_and_spin_correlations_equal_recola(artifacts, name):
    """LI-O1: Recola's loop-induced colour correlations cc(i, j) =
    <T_i.T_j>/C_i, spin correlations <A(v)|A(v)> and spin-colour
    correlations <A(v)|T_g.T_j|A(v)>/C_A equal the correlated |A|^2 (lax
    rule; the digits are binary64's), and the colour identity sum_j
    T_i.T_j = -C_i |A|^2 holds on the evaluated values."""

    from pyamplicol.correlators import CorrelatedRequest

    runtime = _runtime(artifacts, name)
    reference = _recola(name)
    points, scales = _points(name)
    assert reference["points"] == points
    pdgs = _fixture(name)["pdgs"]
    colour = {k: (8 if pdg == 21 else 3) for k, pdg in enumerate(pdgs, 1)}
    ids = [item.id for item in runtime.available_color_correlations()]
    many = runtime.evaluate_correlated_many(
        points, {i: CorrelatedRequest(i) for i in ids}, mu_r2=scales
    )
    total = [values[0].finite for values in many["born"].values]
    assert many["born"].contribution == "loop-induced"
    # "auto" at the default target: TIRust serves every fixture point (u u~
    # > h g g point 0's method-choice lanes, ~1.2e-9 of their term scale,
    # are within its calibrated acceptance).
    assert _served_by(many["born"]) == ["tirust"] * len(points)
    worst = 0.0
    for row in reference["rows"]:
        p = row["point"]
        recola = row["nlo"][0]
        if row["kind"] == "born":
            worst = max(worst, _relative(total[p], recola))
        elif row["kind"] == "cc":
            i, j = row["legs"]
            value = many[f"T{i}T{j}"].values[p][0].finite
            worst = max(worst, _relative(value, recola * float(CASIMIR[colour[i]])))
        else:
            leg = row["legs"][0]
            history = "born" if row["kind"] == "sc" else f"T{leg}T{row['legs'][1]}"
            factor = 1.0 if row["kind"] == "sc" else float(CASIMIR[8])
            (result,) = runtime.evaluate_correlated_many(
                [points[p]],
                {"x": CorrelatedRequest(history, {leg: _spin(row["vector"])})},
                mu_r2=[scales[p]],
            )["x"].values
            worst = max(worst, _relative(result[0].finite, recola * factor))
    assert worst <= LAX, worst
    for i in coloured(pdgs):
        for p, value in enumerate(total):
            summed = sum(
                complex(many[f"T{i}T{j}"].values[p][0].finite)
                for j in coloured(pdgs)
                if j != i
            )
            assert _relative(summed, -float(CASIMIR[colour[i]]) * value) <= ROUNDING


# ------------------------------------------------------------- resolved


@pytest.mark.loop_physics
def test_resolved_rows_and_single_helicity_selections_sum_to_the_total(artifacts):
    """LI-O1 / RULED Q39: every helicity row's |A(h)|^2/avg (the total's
    average), its single-helicity selection equal to it, the rows and the
    selections summing to the total; a row's amplitude-basis anchors sum to
    it; no parity or Bose row mapping (every row its own amplitude)."""

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    points, scales = points[:1], scales[:1]
    total = runtime.evaluate(points, mu_r2=scales).values[0][0].finite
    resolved = runtime.evaluate_resolved(points, mu_r2=scales)
    assert resolved.contribution == "loop-induced"
    assert resolved.virtual_color_ids == ("full",) and resolved.born_color_ids == ()
    rows = [
        resolved.values[0][0][h][0].finite for h in range(len(resolved.helicity_ids))
    ]
    assert _relative(sum(rows), total) <= ROUNDING
    anchors = runtime.evaluate_resolved(
        points, mu_r2=scales, color_resolution="amplitude-basis"
    )
    assert anchors.virtual_color_ids[0] == "amplitude-0"
    # The _with_prec form takes the colour axis too.
    with_prec = runtime.evaluate_resolved_with_prec(
        points, 16, mu_r2=scales, color_resolution="amplitude-basis"
    )
    assert with_prec.virtual_color_ids == anchors.virtual_color_ids
    assert with_prec.values == anchors.values
    selected = []
    for h, helicity in enumerate(resolved.helicity_ids):
        parts = anchors.values[0][0][h]
        assert abs(sum(part.finite for part in parts) - rows[h]) <= ROUNDING * total
        value = runtime.evaluate(points, mu_r2=scales, helicities=[helicity])
        selected.append(value.values[0][0].finite)
        assert value.values[0][0].finite == rows[h]
    assert _relative(sum(selected), total) <= ROUNDING
    # The selection's dressing is pruned to its row (route R2's per-call
    # liveness): fewer modelled MACs and reduction lanes than every row, the
    # value bitwise the resolved row's (above).
    executor = runtime._backend._observable_runtime()._observables.executor
    pruned = dict(executor.last_report["counts"])
    runtime.evaluate_correlated(points, mu_r2=scales)
    whole = executor.last_report["counts"]
    assert "live-row links" in pruned and "live-row links" not in whole
    for count in ("reduction lanes", "dressing work (modelled complex MAC per lane)"):
        assert pruned[count] < whole[count], count
    # Q37: helicity-flipped rows are not mapped onto each other.
    assert len(set(rows)) > len(rows) // 2


@pytest.mark.loop_physics
def test_spin_correlations_are_finite_differences_of_resolved_values(artifacts):
    """The spin-correlated output against the resolved values, through the
    public calls: a leg's own polarisation vector w_h as spin vector gives
    the resolved rows with that helicity, the output is a Hermitian form in
    v (parallelogram rule on w_+ +- w_-), and a real orthonormal frame
    transverse to the leg sums to the total (completeness). These hold by
    construction of the projection on the Python polarisation vectors
    (``polarisations``), which builds w_h too: they check the forms, the
    row bookkeeping and the frame's completeness, not the vectors against
    the engine's sources, which Recola's complex transverse vectors
    (``test_colour_and_spin_correlations_equal_recola``) and the
    literal-source oracle (``test_spin_projection_equals_the_literal_source``)
    do."""

    from decimal import localcontext

    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2.induced_observables import polarisations

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    point, scale = points[0], scales[0]
    leg = 4
    observables = runtime._backend._observable_runtime()._observables
    with localcontext() as context:
        context.prec = 40
        waves = polarisations(observables.sources, leg, point[leg - 1])
    resolved = runtime.evaluate_resolved([point], mu_r2=[scale])
    row_ids = resolved.helicity_ids
    rows = runtime._backend._observable_runtime()._observables.rows

    def spin(vector):
        (value,) = runtime.evaluate_correlated_many(
            [point], {"x": CorrelatedRequest("born", {leg: vector})}, mu_r2=[scale]
        )["x"].values
        return complex(value[0].finite).real

    by_helicity = {}
    for k, row in enumerate(rows):
        h = dict(row)[leg - 1]
        by_helicity[h] = by_helicity.get(h, 0.0) + resolved.values[0][0][k][0].finite
    assert len(row_ids) == len(rows)
    for h, wave in waves.items():
        assert _relative(spin(wave), by_helicity[h]) <= ROUNDING
    plus, minus = waves[1], waves[-1]

    def combine(sign):
        return tuple(
            (a[0] + sign * b[0], a[1] + sign * b[1])
            for a, b in zip(plus, minus, strict=True)
        )

    parallelogram = spin(combine(1)) + spin(combine(-1))
    assert _relative(parallelogram, 2 * (by_helicity[1] + by_helicity[-1])) <= 1e-10
    # A real orthonormal frame transverse to the leg's momentum.
    k = [Fraction(x) for x in point[leg - 1]]
    norm = sum(x * x for x in k[1:]) ** 0.5
    n = [float(x) / float(norm) for x in k[1:]]
    a = [n[1], -n[0], 0.0]
    a = [x / sum(y * y for y in a) ** 0.5 for x in a]
    b = [
        n[1] * a[2] - n[2] * a[1],
        n[2] * a[0] - n[0] * a[2],
        n[0] * a[1] - n[1] * a[0],
    ]
    frame = [tuple((Decimal(repr(x)), Decimal(0)) for x in (0.0, *e)) for e in (a, b)]
    total = sum(by_helicity.values())
    assert _relative(spin(frame[0]) + spin(frame[1]), total) <= 1e-10


@pytest.mark.loop_physics
@pytest.mark.parametrize(("name", "leg"), [("li_gg_hg", 1), ("li_gg_zz", 3)])
def test_spin_projection_equals_the_literal_source(li, name, leg):
    """The projection on the leg's polarisation vectors equals the amplitude
    with the leg's wavefunction replaced by v (the Ward mode's loop-side
    source on a full-kernel slice), for a gluon (v with a gauge component
    along p) and a massive Z (v transverse to p)."""

    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.induced_observables import (
        InducedObservables,
        Output,
        PointOutputs,
    )
    from pyamplicol.loops.v2.ward import ward_slice

    fixture = _fixture(name)
    compiled = li.model(fixture["scheme_data"].get("restriction") or "default")
    slice_ = ward_slice(compiled, _request(fixture, compiled), li.native)
    observables = InducedObservables.of_slice(
        slice_, CorrelatorConfig.from_json_dict({"spin_correlations": [[leg]]})
    )
    point, scale, parameters = li.point(name)
    k = [Fraction(x) for x in point[leg - 1]]
    space = k[1:]
    u = (Fraction(3), Fraction(-1), Fraction(2))
    a = (
        space[1] * u[2] - space[2] * u[1],
        space[2] * u[0] - space[0] * u[2],
        space[0] * u[1] - space[1] * u[0],
    )
    b = (
        space[1] * a[2] - space[2] * a[1],
        space[2] * a[0] - space[0] * a[2],
        space[0] * a[1] - space[1] * a[0],
    )
    scale_a = Fraction(1, int(float(sum(x * x for x in a)) ** 0.5) + 1)
    scale_b = Fraction(1, int(float(sum(x * x for x in b)) ** 0.5) + 1)
    gauge = Fraction(1, 1000) if name == "li_gg_hg" else Fraction(0)
    vector = [
        (
            Fraction(8, 10) * am * scale_a
            - Fraction(4, 10) * bm * scale_b
            + gauge * km,
            Fraction(3, 10) * am * scale_a + Fraction(9, 10) * bm * scale_b,
        )
        for am, bm, km in zip((0, *a), (0, *b), k, strict=True)
    ]
    spin = {
        leg: tuple(
            (
                Decimal(re.numerator) / re.denominator,
                Decimal(im.numerator) / im.denominator,
            )
            for re, im in vector
        )
    }
    executor = observables.executor
    _p, (reference,), _r, _f = executor.runs(
        [point],
        [scale],
        parameters=parameters,
        precision=Precision.double(),
        poles="none",
    )
    projected = PointOutputs(
        reference,
        observables.forms(Precision.double()),
        observables.sources,
        point,
        spin,
    ).value(Output("x"))
    _p, (literal,), _r, _f = executor.runs(
        [point],
        [scale],
        parameters=parameters,
        precision=Precision.double(),
        poles="none",
        ward=(leg, vector),
    )
    first = dict(literal.amplitude_rows[0])[leg - 1]
    rows = tuple(r for r in literal.amplitude_rows if dict(r)[leg - 1] == first)
    direct = induced.square(
        literal.amplitude_vectors,
        induced.metric(literal.slice, literal.numbers),
        literal.slice.sectors,
        rows,
        1,
        literal.numbers,
        literal.average,
    )
    for sector, values in projected.items():
        assert _relative(values[0], direct[sector][0]) <= 1e-10


@pytest.mark.loop_physics
def test_a_permutation_alias_reads_the_representative(artifacts):
    """The alias u~ u > h g g evaluates as its representative at the
    permuted momenta, bitwise: the inclusive value, a colour history read on
    its own labels (its T_1.T_4 is the representative's T_2.T_4), a spin
    vector on a gluon leg, the resolved rows (relabelled) and its
    single-helicity selections (on its own rows)."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    name = "li_uux_hgg"
    points, scales = _points(name)
    points, scales = points[:1], scales[:1]
    _alias, _target, permutation = ALIAS
    member = [[point[permutation[k]] for k in range(len(point))] for point in points]
    representative = _runtime(artifacts, name)
    alias = Runtime.load(artifacts[name], process="p_swap")
    own = representative.evaluate(points, mu_r2=scales)
    swapped = alias.evaluate(member, mu_r2=scales)
    assert swapped.values == own.values
    # Point 0 on TIRust for both (its lanes within the default acceptance).
    assert _served_by(own) == _served_by(swapped) == ["tirust"]
    row = _recola(name)["rows"][13]
    assert row["kind"] == "sc" and row["legs"] == [4] and row["point"] == 0
    vector = _spin(row["vector"])
    requests = {
        "c": CorrelatedRequest("T2T4"),
        "s": CorrelatedRequest("T2T4", {4: vector}),
    }
    mapped = {
        "c": CorrelatedRequest("T1T4"),
        "s": CorrelatedRequest("T1T4", {4: vector}),
    }
    direct = representative.evaluate_correlated_many(points, requests, mu_r2=scales)
    relabelled = alias.evaluate_correlated_many(member, mapped, mu_r2=scales)
    for key in requests:
        assert relabelled[key].values == direct[key].values, key
    rows = representative.evaluate_resolved(points, mu_r2=scales)
    moved = alias.evaluate_resolved(member, mu_r2=scales)
    assert sorted(v[0].finite for v in moved.values[0][0]) == sorted(
        v[0].finite for v in rows.values[0][0]
    )
    # Its helicity selections name its own rows (its resolved order, its
    # tree IDs in its leg order), each the representative row whose legs it
    # permutes.
    from pyamplicol.loops.v2.selection import helicity_id

    own = alias._backend._observable_runtime().selection_catalogue().rows
    for k in (0, 3, len(own) - 1):
        theirs = tuple(sorted((permutation[leg], h) for leg, h in own[k]))
        value = alias.evaluate(member, mu_r2=scales, helicities=[f"h{k}"]).values
        assert value[0][0].finite == moved.values[0][0][k][0].finite
        assert (
            alias.evaluate(
                member, mu_r2=scales, helicities=[helicity_id(own[k])]
            ).values
            == value
            == representative.evaluate(
                points, mu_r2=scales, helicities=[helicity_id(theirs)]
            ).values
        )
        kept = alias.evaluate_resolved(member, mu_r2=scales, helicities=[f"h{k}"])
        assert kept.helicity_ids == (f"h{k}",)
        assert kept.values[0][0][0] == moved.values[0][0][k]


@pytest.mark.loop_physics
def test_one_call_is_one_evaluation_whatever_the_spin_sources(artifacts, monkeypatch):
    """``evaluate_correlated_many`` runs the executor once for requests with
    different spin sources (RULED Q9: every (assignment, history) output is
    a form on the same amplitude vectors), each output equal to its own
    call bitwise; the call's effective backend reaches the executor, which
    reduces on it (TIRust on the resolved route, m-lit lit-tir; never
    NinjaRS under a TIRust label); a spin vector with no transverse part is
    served, not refused; the observables run the runtime's current executor
    under its lock, also after ``clear``."""

    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2.reduction import ReductionUnavailable

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    points, scales = points[:1], scales[:1]
    backend = runtime._backend
    observables = backend._observable_runtime()._observables
    assert observables.lock is backend._lock
    executor = backend._executor
    original = executor.runs
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("reduction"))
        return original(*args, **kwargs)

    monkeypatch.setattr(executor, "runs", counted)
    rows = [r for r in _recola(name)["rows"] if r["kind"] == "sc" and r["point"] == 0]
    (leg_a,), (leg_b,) = rows[0]["legs"], rows[-1]["legs"]
    requests = {
        "total": CorrelatedRequest("born"),
        "T1T4": CorrelatedRequest("T1T4"),
        "s1": CorrelatedRequest("born", {leg_a: _spin(rows[0]["vector"])}),
        "s2": CorrelatedRequest("T1T4", {leg_b: _spin(rows[-1]["vector"])}),
    }
    # NinjaRS named: the call semantics, whatever "auto" takes (TIRust,
    # the loop-induced default, where it serves the carrier).
    many = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, reduction="ninjars"
    )
    assert calls == ["ninjars"]
    assert {
        lane.reduction.effective
        for result in many.values()
        for lane in result.diagnostics
    } == {"ninjars"}
    for label, request in requests.items():
        alone = runtime.evaluate_correlated_many(
            points, {label: request}, mu_r2=scales, reduction="ninjars"
        )[label]
        assert alone.values == many[label].values, label
    assert len(calls) == 1 + len(requests)
    tirust = bool(backend._native.tirust_available())
    if tirust:
        # The executor reduces on the backend it is given.
        original(points, scales, reduction="tirust")
        assert executor.last_report["counts"]["contracted lanes"] > 0
    else:
        with pytest.raises(ReductionUnavailable):
            original(points, scales, reduction="tirust")
    # A vector with no transverse part is served (its value the projection's
    # exact 0; its estimate carries it): one more evaluation, no refusal.
    zero = (Decimal(0), Decimal(0))
    timelike = ((Decimal(1), Decimal(0)), zero, zero, zero)
    (value,) = runtime.evaluate_correlated_many(
        points,
        {"x": CorrelatedRequest("born", {leg_a: timelike})},
        mu_r2=scales,
        reduction="ninjars",
    )["x"].values
    assert complex(value[0].finite) == 0
    assert len(calls) == 2 + len(requests)
    # TIRust on the resolved route: the executor reduces on it and every
    # lane records it (a point it fails closed fails typed, never NinjaRS
    # under a TIRust label); a native without TIRust refuses it typed.
    if tirust:
        served = runtime.evaluate_correlated_many(
            points, requests, mu_r2=scales, reduction="tirust", on_failure="return"
        )
        assert {
            lane.reduction.effective
            for result in served.values()
            for lane in result.diagnostics
        } == {"tirust"}
        assert executor.last_report["counts"]["contracted lanes"] > 0
    else:
        with pytest.raises(ReductionUnavailable):
            runtime.evaluate_correlated_many(
                points, requests, mu_r2=scales, reduction="tirust"
            )
    assert calls[-1] == "tirust"
    backend.clear()
    assert observables.executor is backend._executor is not executor


# ------------------------------------------------------------ carriers


@pytest.mark.loop_physics
def test_correlated_outputs_in_double_double_and_with_estimates(artifacts):
    """The quadratic forms in double-double (Decimal square of the native
    amplitude vectors) agree with binary64; with the recommended stability
    checks (``stability.RECOMMENDED_CHECKS``: one generic rotation and the
    rescaled reduction re-test) every output carries its estimate, absolute
    against its own scale, and the binary64 value is within ten times it of
    the double-double one at every fixture point (without the rescaled
    re-test the masters' own rounding is missed: T1.T4 at point 1 is then 25
    times off its estimate, DESIGN §10.4)."""

    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    requests = {i: CorrelatedRequest(i) for i in ("born", "T1T4", "T2T4", "T4T5")}
    recommended = VirtualStabilityChecks(
        RECOMMENDED_CHECKS.rotations, reduction_direction=True, rescaled=True
    )
    double = runtime.evaluate_correlated_many(points, requests, mu_r2=scales)
    dd = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, arithmetic="double-double"
    )
    checked = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, stability_checks=recommended
    )
    # "born" is |A|^2 itself: the inclusive evaluation's value and estimate.
    inclusive = runtime.evaluate(points, mu_r2=scales, stability_checks=recommended)
    assert checked["born"].values == inclusive.values
    assert checked["born"].stability == inclusive.stability
    # Without checks TIRust serves every point, in binary64 and
    # double-double; with them a point whose check rows TIRust refuses is the
    # per-point fallback's (NinjaRS), on both routes alike.
    assert _served_by(double["born"]) == _served_by(dd["born"]) == ["tirust"] * 3
    explicit = runtime.evaluate(
        points,
        mu_r2=scales,
        stability_checks=recommended,
        reduction="tirust",
        on_failure="return",
    )
    assert (
        _served_by(checked["born"])
        == _served_by(inclusive)
        == _tirust_or_fallback(explicit)
    )
    for key in requests:
        assert len(checked[key].stability) == len(points)
        for p, estimate in enumerate(checked[key].stability):
            a = complex(double[key].values[p][0].finite)
            b = complex(dd[key].values[p][0].finite.real)
            assert 0 < estimate.relative < 1e-8
            # binary64 within ten times its own estimate of the DD value.
            assert abs(a - b) <= 10 * estimate.absolute[0].finite, (key, p)
            assert _relative(a, b) <= LAX


@pytest.mark.loop_physics
def test_the_both_backends_checks_cover_the_amplitude_basis_anchors(artifacts):
    """The binary64 rule of the both-backends check (L6,
    ``tools/loop_v2_observables_backends.py``) reads the estimates of its
    ``checks()`` (the rescaled re-test and three generic rotations): every
    amplitude-basis anchor of u u~ > h g g, at every fixture point, is within
    ten times its estimate of the double-double value. The numerical-zero
    anchor amplitude-0 of h11 at point 0 is 14.7 times off the estimate of
    the recommended checks (one rotation, which moves it by 3 % of its
    error)."""

    from tools.loop_v2_observables_backends import checks

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    options = {"mu_r2": scales, "color_resolution": "amplitude-basis"}
    dd = runtime.evaluate_resolved(points, arithmetic="double-double", **options)
    checked = runtime.evaluate_resolved(points, stability_checks=checks(), **options)
    explicit = runtime.evaluate_resolved(
        points,
        stability_checks=checks(),
        reduction="tirust",
        on_failure="return",
        **options,
    )
    assert _served_by(checked) == _tirust_or_fallback(explicit)
    assert _served_by(dd) == ["tirust"] * len(points)
    for p, estimates in enumerate(checked.stability):
        for h, row in enumerate(estimates):
            for c, estimate in enumerate(row):
                a = complex(checked.values[p][0][h][c].finite)
                b = complex(dd.values[p][0][h][c].finite.real)
                assert abs(a - b) <= 10 * estimate.absolute[0].finite, (p, h, c)


@pytest.mark.loop_physics
def test_a_binary64_k_perp_is_served_with_its_transverse_budget(artifacts):
    """A Sudakov k_perp = k - (k.p/q.p) q - (k.q/q.p) p built in binary64
    is transverse to its leg only to its rounding (here beyond 4u sum |v||p|,
    the former refusal's bound): it is served in binary64 and double-double,
    its value within the estimate of the exact rational k_perp's, whose
    estimate has no spin-transverse budget; the budget carries the dropped
    part. A vector far from transverse (w_+ plus the time direction) is
    served with an estimate of the value's own size."""

    import random
    from decimal import localcontext

    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest
    from pyamplicol.loops.v2.induced_observables import polarisations

    name, leg = "li_uux_hgg", 4
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    point, scale = points[0], scales[0]

    def dot(a, b):
        return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]

    p = [Fraction(x) for x in point[leg - 1]]
    q = [Fraction(x) for x in point[0]]
    assert dot(p, p) == dot(q, q) == 0
    pf, qf = [float(x) for x in p], [float(x) for x in q]
    rng = random.Random(5)
    while True:
        size = 10 ** rng.uniform(-3, 1) * pf[0]
        k = [rng.uniform(-size, size) for _ in range(4)]
        built = [
            k[m] - dot(k, pf) / dot(qf, pf) * qf[m] - dot(k, qf) / dot(qf, pf) * pf[m]
            for m in range(4)
        ]
        exact_dot = abs(dot([Fraction(x) for x in built], p))
        rounding = sum(abs(Fraction(x)) * abs(y) for x, y in zip(built, p, strict=True))
        if exact_dot > 4 * Fraction(2.0**-53) * rounding:
            break
    kk = [Fraction(x) for x in k]
    exact = [
        kk[m] - dot(kk, p) / dot(q, p) * q[m] - dot(kk, q) / dot(q, p) * p[m]
        for m in range(4)
    ]
    assert dot(exact, p) == 0
    with localcontext() as context:
        context.prec = 60
        transverse = tuple(
            (Decimal(x.numerator) / Decimal(x.denominator), Decimal(0)) for x in exact
        )
    rounded = tuple((Decimal(repr(x)), Decimal(0)) for x in built)
    checks = VirtualStabilityChecks(1, reduction_direction=True)

    served = []

    def spin(vector, arithmetic="double"):
        result = runtime.evaluate_correlated_many(
            [point],
            {"x": CorrelatedRequest("born", {leg: vector})},
            mu_r2=[scale],
            arithmetic=arithmetic,
            stability_checks=checks,
        )["x"]
        served.extend(_served_by(result))
        (value,), (estimate,) = result.values, result.stability
        return complex(value[0].finite), estimate

    for arithmetic in ("double", "double-double"):
        reference, clean = spin(transverse, arithmetic)
        value, estimate = spin(rounded, arithmetic)
        budgets = dict(estimate.checks)
        assert dict(clean.checks).get("spin-transverse", 0.0) < 1e-40
        assert 0 < budgets["spin-transverse"] < 1e-12, budgets
        assert abs(value - reference) <= estimate.absolute[0].finite
    with localcontext() as context:
        context.prec = 40
        observables = runtime._backend._observable_runtime()._observables
        plus = polarisations(observables.sources, leg, point[leg - 1])[1]
    far = ((plus[0][0] + 1, plus[0][1]), *plus[1:])
    value, estimate = spin(far)
    alone, _estimate = spin(plus)
    assert value == alone
    assert dict(estimate.checks)["spin-transverse"] > 1
    assert estimate.absolute[0].finite > abs(value)
    # Each of the six calls records the backend that served the point
    # (TIRust, or the per-point fallback's NinjaRS with its reason).
    assert len(served) == 6 and set(served) <= {"tirust", "ninjars"}, served


# ------------------------------------------------- the per-point fallback


def _ratio(reason):
    """The largest deviation / term scale of the method-choice refusals in
    a failure text (``v2_tir``'s acceptance check), or None."""

    import re

    found = [
        float(deviation) / float(scale)
        for deviation, scale in re.findall(
            r"deviation ([0-9.eE+-]+) of term scale ([0-9.eE+-]+)", reason
        )
    ]
    return max(found) if found else None


def _per_point(auto, explicit, other):
    """Each point of an ``"auto"`` result (m-lit F3 (c)): served by TIRust
    where the explicit TIRust call serves it (bitwise its value), else by
    the per-point fallback on NinjaRS (bitwise the explicit NinjaRS value,
    its lane recording NinjaRS and the fallback's reason with TIRust's
    failure). Returns the backend of every point."""

    backends = []
    for p, lane in enumerate(auto.diagnostics):
        record = lane.reduction
        assert record.requested == "auto"
        if explicit.values[p] is not None:
            assert record.effective == "tirust" and record.fallback == "ninjars", p
            assert repr(auto.values[p]) == repr(explicit.values[p]), p
        else:
            assert record.effective == "ninjars" and record.fallback is None, p
            assert record.reason.startswith("auto: NinjaRS, the per-point fallback: ")
            assert explicit.diagnostics[p].detail in record.reason or (
                "TIRust refused or failed the point" in record.reason
            )
            assert repr(auto.values[p]) == repr(other.values[p]), p
        backends.append(record.effective)
    return backends


@pytest.mark.loop_physics
def test_auto_serves_a_point_tirust_refuses_on_ninjars(artifacts, loop_native):
    """F3 (c) at a stated target: LI-O1 u u~ > h g g point 0 has
    method-choice lanes whose two frames deviate by r ~ 1.2e-9 of their
    term scale (lit-cache step 2: 4.44e-28 / 3.71e-19). At the target T
    whose calibrated acceptance (``tir_acceptance``) is r / 2, explicit
    TIRust fails the point typed with its check's reason, and "auto" serves
    it on NinjaRS (its value bitwise the explicit NinjaRS one, its lane
    recording NinjaRS and the fallback's reason) and the other points on
    TIRust, bitwise. At the default target (1e-5, acceptance 1e-5 / 580)
    the lanes are admitted: TIRust serves the point."""

    from pyamplicol.loops.v2.reduction import default_target

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    options = {"mu_r2": scales, "poles": "none", "on_failure": "return"}
    # Every method-choice lane refused (the smallest target): its deviation.
    found = runtime.evaluate(
        points[:1],
        reduction="tirust",
        reduction_target=1e-300,
        **{**options, "mu_r2": scales[:1]},
    )
    ratio = _ratio(found.diagnostics[0].detail)
    assert ratio is not None and ratio > 0
    per_target = loop_native.tir_acceptance(1.0)
    target = ratio / 2 / per_target
    assert loop_native.tir_acceptance(target) < ratio
    explicit = runtime.evaluate(
        points, reduction="tirust", reduction_target=target, **options
    )
    assert explicit.values[0] is None
    assert "TIRust method-choice lane refused by its check" in (
        explicit.diagnostics[0].detail
    )
    other = runtime.evaluate(points, reduction="ninjars", **options)
    auto = runtime.evaluate(points, reduction_target=target, **options)
    backends = _per_point(auto, explicit, other)
    assert backends[0] == "ninjars", f"target {target:e}"
    assert {lane.reduction.target for lane in auto.diagnostics} == {target}
    # The default target admits the point's lanes (ratio below its
    # acceptance): TIRust serves it.
    default = runtime.evaluate(points, **options)
    assert ratio <= loop_native.tir_acceptance(default_target())
    assert [lane.reduction.effective for lane in default.diagnostics] == [
        "tirust"
    ] * len(points)
    # The value TIRust serves at either target is the same value.
    assert repr(default.values[1:]) == repr(auto.values[1:])


@pytest.mark.loop_physics
def test_auto_serves_a_point_the_monitoring_fails_on_ninjars(artifacts):
    """F3 (c): with the recommended stability checks at the default target,
    TIRust's re-test rows refuse some fixture points of u u~ > h g g (the
    term-scale lanes of the check, lit-tir-mp's open item 2), so the
    point's monitored evaluation fails; "auto" evaluates those points again,
    with the same checks, on NinjaRS (values and estimates bitwise the
    explicit NinjaRS call's), the others on TIRust. With the cross-backend
    check a fallback point's cross-backend check is not run, with the
    fallback's reason (R6)."""

    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.loops.v2.stability import CROSS, RECOMMENDED_CHECKS

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    checks = VirtualStabilityChecks(
        RECOMMENDED_CHECKS.rotations, reduction_direction=True, rescaled=True
    )
    options = {"mu_r2": scales, "on_failure": "return", "stability_checks": checks}
    explicit = runtime.evaluate(points, reduction="tirust", **options)
    other = runtime.evaluate(points, reduction="ninjars", **options)
    auto = runtime.evaluate(points, **options)
    backends = _per_point(auto, explicit, other)
    assert "ninjars" in backends, "the default target's checks fail no point"
    for p, backend in enumerate(backends):
        source = other if backend == "ninjars" else explicit
        assert auto.stability[p] == source.stability[p], p
    crossed = runtime.evaluate(
        points,
        **{
            **options,
            "stability_checks": VirtualStabilityChecks(
                RECOMMENDED_CHECKS.rotations,
                reduction_direction=True,
                rescaled=True,
                cross_backend=True,
            ),
        },
    )
    for p, backend in enumerate(backends):
        lane = crossed.diagnostics[p]
        assert lane.reduction.effective == backend, p
        if backend == "ninjars":
            assert (CROSS, lane.reduction.reason) in crossed.stability[p].not_run
            assert repr(crossed.values[p]) == repr(other.values[p])


@pytest.mark.loop_physics
def test_a_double_double_auto_point_falls_back_at_double_double(artifacts, monkeypatch):
    """F3 (c) CARRIERS (this package's reading of DECISIONS 2026-11-08 15:19
    'at the same precision'): the per-point fallback evaluates at the call's
    own arithmetic. TIRust refuses no LI-O1 fixture point in double-double
    at the default target (its gates there are binary64's squared), so a
    TIRust failure of point 0 is forced in the executor: "auto" serves it
    on NinjaRS in double-double, equal to the explicit NinjaRS double-double
    value, the other points on TIRust."""

    from pyamplicol.loops.v2 import plan_export
    from pyamplicol.loops.v2.reduction import current

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    options = {
        "mu_r2": scales,
        "poles": "none",
        "on_failure": "return",
        "arithmetic": "double-double",
    }
    other = runtime.evaluate(points, reduction="ninjars", **options)
    original = plan_export.NativeExecutor._evaluated

    def failing(self, *args, **kwargs):
        precision, evaluations, failed = original(self, *args, **kwargs)
        if current() == "tirust" and 0 in range(len(evaluations)):
            failed = {**failed, 0: ["refused by a conditioning gate (forced)"]}
        return precision, evaluations, failed

    monkeypatch.setattr(plan_export.NativeExecutor, "_evaluated", failing)
    explicit = runtime.evaluate(points, reduction="tirust", **options)
    assert explicit.values[0] is None and explicit.values[1] is not None
    auto = runtime.evaluate(points, **options)
    assert {lane.arithmetic for lane in auto.diagnostics} == {"double-double"}
    assert _per_point(auto, explicit, other)[:2] == ["ninjars", "tirust"]
    assert "(forced)" in auto.diagnostics[0].reduction.reason


@pytest.mark.loop_physics
def test_auto_serves_every_point_when_tirust_refuses_the_plans(artifacts, monkeypatch):
    """F3 (c), capability refusals (not governed by the acceptance): TIRust's
    refusal of the call's plans (its capability query of a family, the typed
    ``ReductionUnavailable`` before any point; forced here) leaves every point
    to the per-point fallback under "auto", each recording NinjaRS and the
    refusal; explicit TIRust is refused typed."""

    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.loops.v2 import plan_export, reduction

    name = "li_gg_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    options = {"mu_r2": scales, "poles": "none", "on_failure": "return"}
    other = runtime.evaluate(points, reduction="ninjars", **options)
    original = plan_export.prepare

    def refusing(executor):
        if reduction.current() == "tirust":
            raise ReductionUnavailable(
                "reduction 'tirust' cannot serve this artifact: (forced)"
            )
        original(executor)

    monkeypatch.setattr(plan_export, "prepare", refusing)
    with pytest.raises(ReductionUnavailable, match="forced"):
        runtime.evaluate(points, reduction="tirust", **options)
    auto = runtime.evaluate(points, **options)
    assert repr(auto.values) == repr(other.values)
    for lane in auto.diagnostics:
        assert lane.reduction.effective == "ninjars"
        assert "(forced)" in lane.reduction.reason


@pytest.mark.loop_physics
def test_the_loop_induced_default_stack_records_level_and_backend(artifacts):
    """F3 (b): ``VirtualRescuePolicy()`` on a loop-induced "auto" call is the
    loop-induced default stack, TIRust binary64 -> NinjaRS binary64 ->
    double-double -> 57 digits; every point records the level and the
    backend that served it, a point TIRust fails at level 0 is recorded at a
    later level (the NinjaRS level when NinjaRS meets the target there).
    ``reduction="ninjars"`` with the same policy runs NinjaRS at every level
    (no TIRust attempt), ``"tirust"`` TIRust at every level."""

    from pyamplicol.api import VirtualRescuePolicy

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    options = {"mu_r2": scales, "on_failure": "return"}
    stacked = runtime.evaluate(points, stability=VirtualRescuePolicy(), **options)
    for record in stacked.rescue:
        first = record.attempts[0]
        assert (first.arithmetic, first.reduction.effective) == ("double", "tirust")
        assert first.reduction.requested == "tirust"
        if len(record.attempts) > 1:
            second = record.attempts[1]
            assert (second.arithmetic, second.reduction.effective) == (
                "double",
                "ninjars",
            )
    # The levels a point reached and the backend that returned it.
    assert [
        record.reduction.effective for record in stacked.rescue if record.level == 0
    ] == ["tirust" for record in stacked.rescue if record.level == 0]
    # Some fixture point fails TIRust's level (its monitored evaluation, the
    # term-scale lanes of the re-test rows) and is served at the NinjaRS
    # level: the records below are not vacuous.
    climbed = [record for record in stacked.rescue if record.level >= 1]
    assert any(record.level == 1 for record in climbed), [
        record.level for record in stacked.rescue
    ]
    assert all(
        record.reduction.effective == "ninjars"
        for record in stacked.rescue
        if record.level == 1
    )
    for record in stacked.rescue:
        if record.attempts[0].failure is not None:
            assert record.level >= 1
    for backend in ("ninjars", "tirust"):
        one = runtime.evaluate(
            points, stability=VirtualRescuePolicy(), reduction=backend, **options
        )
        for record in one.rescue:
            assert {item.reduction.effective for item in record.attempts} == {backend}
            assert len(record.attempts) == record.level + 1


# ------------------------------------------------------------- refusals


@pytest.mark.loop_physics
def test_loop_induced_observable_refusals(artifacts):
    """Full colour only: no colour-flow selection; undeclared histories and
    spin legs are refused; the colour axis option is the loop-induced
    runtime's."""

    from pyamplicol.api.errors import CompatibilityError, EvaluationError
    from pyamplicol.correlators import CorrelatedRequest

    runtime = _runtime(artifacts, "li_gg_hgg")
    points, scales = _points("li_gg_hgg")
    points, scales = points[:1], scales[:1]
    with pytest.raises(CompatibilityError, match="no colour flows"):
        runtime.evaluate(points, mu_r2=scales, color_flows=["1"])
    with pytest.raises(CompatibilityError, match="undeclared"):
        runtime.evaluate_correlated(points, color_correlation="T1T3", mu_r2=scales)
    with pytest.raises(EvaluationError, match="not declared"):
        runtime.evaluate_correlated_many(
            points,
            {"x": CorrelatedRequest("born", {3: ((Decimal(0), Decimal(0)),) * 4})},
            mu_r2=scales,
        )
    with pytest.raises(CompatibilityError, match="colour resolution"):
        runtime.evaluate_resolved(points, mu_r2=scales, color_resolution="flows")
    # A correlated call takes no selection; the message names the
    # loop-induced full-colour helicity selection.
    with pytest.raises(CompatibilityError, match="loop-induced processes select"):
        runtime.evaluate_correlated(points, mu_r2=scales, helicities=["h0"])


@pytest.mark.loop_fast
def test_born_interfered_runtimes_refuse_the_colour_axis_option():
    """``color_resolution`` is a loop-induced process's option: the
    Born-interfered observables runtime refuses it typed (DESIGN §7.14)."""

    from types import SimpleNamespace

    from pyamplicol.api.errors import CompatibilityError
    from pyamplicol.runtime.virtual import V2ObservablesRuntime

    slice_ = SimpleNamespace(
        sectors=(),
        request=SimpleNamespace(accuracy="full"),
        blocks=SimpleNamespace(process=SimpleNamespace(legs=(1, 2, 3, 4))),
    )
    runtime = V2ObservablesRuntime(
        SimpleNamespace(slice=slice_, spin_classes=(), histories=(), rows=())
    )
    with pytest.raises(CompatibilityError, match="Born-interfered virtual"):
        runtime.evaluate_resolved([], mu_r2=[], color_resolution="amplitude-basis")
