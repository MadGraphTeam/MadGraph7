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
* full colour only (no colour-flow selection), the colour axis option is
  loop-induced only, undeclared histories and spin legs are refused.
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tests.unit.test_loop_v2_induced import _card, _fixture, _request, _Rows

ROOT = Path(__file__).resolve().parents[2]
CORRELATIONS = ROOT / "tests" / "fixtures" / "loop" / "required" / "li" / "correlations"
LAX = 1e-5
# Rounding of two binary64 evaluations of the same quantity (the native
# square against Python's, the literal source against the projection).
ROUNDING = 1e-11
CASIMIR = {8: Fraction(3), 3: Fraction(4, 3), -3: Fraction(4, 3)}


def _dipoles(legs):
    return [
        {
            "id": f"T{i}T{j}",
            "bra": [{"kind": "emit-gluon", "emitter": i, "emitted": -1}],
            "ket": [{"kind": "emit-gluon", "emitter": j, "emitted": -1}],
        }
        for i in legs
        for j in legs
        if i != j
    ]


def _recola(name):
    return json.loads((CORRELATIONS / f"{name}-recola.json").read_text())


def _coloured(pdgs):
    return [k for k, pdg in enumerate(pdgs, 1) if abs(pdg) in (1, 2, 3, 4, 5, 6, 21)]


def _gluons(pdgs):
    return [k for k, pdg in enumerate(pdgs, 1) if pdg == 21]


def _generate(root: Path, name: str, expression: str, aliases=()):
    """The loop-induced artifact of fixture row ``name`` through the public
    Generator, every dipole T_i.T_j and every gluon's spin class declared
    (``aliases``: ``ProcessAlias`` arguments on process ``p``)."""

    from pyamplicol.api import Generator, ProcessAlias, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    fixture = _fixture(name)
    pdgs = fixture["pdgs"]
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(_card(root / "artifact", (("p", expression),), fixture))
    correlators = CorrelatorConfig.from_json_dict(
        {
            "color_correlations": _dipoles(_coloured(pdgs)),
            "spin_correlations": [[leg] for leg in _gluons(pdgs)],
        }
    )
    Generator(resolve_config(card)).generate(
        ProcessSet(
            (ProcessRequest.parse(expression, name="p"),),
            tuple(ProcessAlias(*alias) for alias in aliases),
        ),
        root / "artifact",
        correlators=correlators,
    )
    return root / "artifact"


# A permutation alias of g g > h g g: g g > g h g (final legs 3 and 4
# exchanged).
ALIAS = ("p_swap", "p", (0, 1, 3, 2, 4))


ROWS = {"li_gg_hgg": "g g > h g g", "li_uux_hgg": "u u~ > h g g"}


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li"))


@pytest.fixture(scope="module")
def artifacts(loop_sm_ufo, loop_native, tmp_path_factory):
    return {
        name: _generate(
            tmp_path_factory.mktemp(name),
            name,
            expression,
            (ALIAS,) if name == "li_gg_hgg" else (),
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
    for i in _coloured(pdgs):
        for p, value in enumerate(total):
            summed = sum(
                complex(many[f"T{i}T{j}"].values[p][0].finite)
                for j in _coloured(pdgs)
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
    selected = []
    for h, helicity in enumerate(resolved.helicity_ids):
        parts = anchors.values[0][0][h]
        assert abs(sum(part.finite for part in parts) - rows[h]) <= ROUNDING * total
        value = runtime.evaluate(points, mu_r2=scales, helicities=[helicity])
        selected.append(value.values[0][0].finite)
        assert value.values[0][0].finite == rows[h]
    assert _relative(sum(selected), total) <= ROUNDING
    # Q37: helicity-flipped rows are not mapped onto each other.
    assert len(set(rows)) > len(rows) // 2


@pytest.mark.loop_physics
def test_spin_correlations_are_finite_differences_of_resolved_values(artifacts):
    """A leg's own polarisation vector w_h as spin vector gives the resolved
    rows with that helicity; the spin-correlated output is a Hermitian form
    in v (parallelogram rule on w_+ +- w_-), and a real orthonormal frame
    transverse to the leg sums to the total (completeness)."""

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
        reference, observables.matrices, observables.sources, point, spin
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
    """The alias g g > g h g evaluates as its representative at the
    permuted momenta, bitwise: the inclusive value, a served colour history
    (T_1.T_5: its image is declared), a spin vector on the alias's gluon
    leg 3 (the representative's leg 4) and the resolved rows (relabelled)."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    name = "li_gg_hgg"
    points, scales = _points(name)
    points, scales = points[:1], scales[:1]
    _alias, _target, permutation = ALIAS
    member = [[point[permutation[k]] for k in range(len(point))] for point in points]
    representative = _runtime(artifacts, name)
    alias = Runtime.load(artifacts[name], process="p_swap")
    assert (
        alias.evaluate(member, mu_r2=scales).values
        == representative.evaluate(points, mu_r2=scales).values
    )
    vector = _spin(_recola(name)["rows"][21]["vector"])
    assert _recola(name)["rows"][21]["legs"] == [4]
    requests = {
        "c": CorrelatedRequest("T1T5"),
        "s": CorrelatedRequest("born", {4: vector}),
    }
    mapped = {
        "c": CorrelatedRequest("T1T5"),
        "s": CorrelatedRequest("born", {3: vector}),
    }
    direct = representative.evaluate_correlated_many(points, requests, mu_r2=scales)
    relabelled = alias.evaluate_correlated_many(member, mapped, mu_r2=scales)
    for key in requests:
        assert relabelled[key].values == direct[key].values, key
    rows = representative.evaluate_resolved(points, mu_r2=scales)
    moved = alias.evaluate_resolved(member, mu_r2=scales)
    assert sorted(v.finite for v in (x[0] for x in moved.values[0][0])) == sorted(
        v.finite for v in (x[0] for x in rows.values[0][0])
    )


# ------------------------------------------------------------ carriers


@pytest.mark.loop_physics
def test_correlated_outputs_in_double_double_and_with_estimates(artifacts):
    """The quadratic forms in double-double (Decimal square of the native
    amplitude vectors) agree with binary64; with stability checks every
    output carries its estimate, absolute against its own scale."""

    from pyamplicol.api import VirtualStabilityChecks
    from pyamplicol.correlators import CorrelatedRequest

    name = "li_uux_hgg"
    runtime = _runtime(artifacts, name)
    points, scales = _points(name)
    points, scales = points[:1], scales[:1]
    requests = {i: CorrelatedRequest(i) for i in ("born", "T1T4", "T4T5")}
    double = runtime.evaluate_correlated_many(points, requests, mu_r2=scales)
    dd = runtime.evaluate_correlated_many(
        points, requests, mu_r2=scales, arithmetic="double-double"
    )
    checked = runtime.evaluate_correlated_many(
        points,
        requests,
        mu_r2=scales,
        stability_checks=VirtualStabilityChecks(1, reduction_direction=True),
    )
    for key in requests:
        a = complex(double[key].values[0][0].finite)
        b = complex(dd[key].values[0][0].finite.real)
        (estimate,) = checked[key].stability
        assert 0 < estimate.relative < 1e-8
        # binary64 within ten times its own estimate of the DD value.
        assert abs(a - b) <= 10 * estimate.absolute[0].finite
        assert _relative(a, b) <= LAX


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
