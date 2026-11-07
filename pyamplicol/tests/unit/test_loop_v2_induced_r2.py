# SPDX-License-Identifier: 0BSD
"""Loop-induced |A_loop|^2 on route R2 (M-LIT LIT-R2): amplitude mode in
schedule B with dressed-leg rows (``plan_export._Exporter.family``,
the native ``dressed`` plans of ``rusticol_core::loop_v2::plan``).

Route R2k (m-lit DECISIONS 2026-11-06 ~09:00 (3)): per family one suffix
trie per amplitude vector of its skeletons' seeds, in the family's frame;
each leaf seeds one constant row (its weight), the steps' rows are the
helicities of the legs dressed so far, and each trace step closes its rows
with its seed key's amplitude vector into route R1's (row, t) columns.
Checked here:

* R1 = R2 on lit-amp's rows: the bare amplitude vectors per (loop order,
  row) and |A|^2 agree to rounding in binary64 and double-double (the two
  routes sum the same terms in another order);
* an R2 plan is schema 5 with route R1's columns and ``dressed``; the
  Python exporter's dressed row links (written with ``links=True``) equal
  the native loader's derivation; the binary member decodes back to the
  plan's JSON; a seeded plan's binary member holds format 3's columns only;
* route R2 runs schedule B only;
* the exact helicity-zero filter: its degree bound, the F_p test's
  verdicts, Q13, dropped rows as columns without a row, and the drop path
  through the public generation.
"""

from __future__ import annotations

import pytest

from .test_loop_v2_induced import _fixture, _Rows, li_artifact  # noqa: F401

# Rounding of the two routes' different summation orders, relative to the
# largest amplitude component of the point (binary64, double-double).
ROUNDING = {"double": 1e-12, "double-double": 1e-27}

# lit-amp's rows (``test_loop_v2_induced.ROWS``).
ROWS = [
    "li_gg_h",
    "li_gg_h_tb",
    "li_h_gg",
    "li_gg_aa_nf5",
    "li_gg_aa",
    "li_gg_hh",
    "li_gg_zz",
    "li_gg_hg",
    "li_uux_hg",
]


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li-r2"))


def _routes(li, name):
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    slice_ = li.slice(name)
    return {
        route: NativeExecutor(slice_, forward=False, route=route)
        for route in ("R1", "R2")
    }


def _evaluated(executor, point, scale, parameters, precision):
    _precision, (evaluation,), failures = executor._evaluated(
        [point], [scale], parameters, precision, "all", None
    )
    assert not failures
    return evaluation


def _largest(vectors) -> float:
    return max(
        (
            abs(complex(z))
            for vector in vectors.values()
            for laurent in vector
            for z in laurent
        ),
        default=0.0,
    )


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ROWS)
@pytest.mark.parametrize("carrier", ["double", "double-double"])
def test_route_r2_equals_route_r1(li, name, carrier):
    from pyamplicol.loops.v2.assemble import Precision

    precision = Precision.double() if carrier == "double" else Precision.double_double()
    executors = _routes(li, name)
    assert executors["R2"].exported.program.get("dressed") is True
    assert "dressed" not in executors["R1"].exported.program
    points = len(_fixture(name)["points"])
    for index in range(points if carrier == "double" else 1):
        point, scale, parameters = li.point(name, index)
        r1, r2 = (
            _evaluated(executors[route], point, scale, parameters, precision)
            for route in ("R1", "R2")
        )
        assert r1.native_bare.keys() == r2.native_bare.keys()
        scale_ = _largest(r1.native_bare)
        assert scale_ > 0
        worst = max(
            abs(complex(a) - complex(b))
            for key, vector in r1.native_bare.items()
            for laurent_a, laurent_b in zip(vector, r2.native_bare[key], strict=True)
            for a, b in zip(laurent_a, laurent_b, strict=True)
        )
        assert worst <= ROUNDING[carrier] * scale_, (name, index, worst / scale_)
    # |A|^2 through the evaluation path: the square of vectors equal to
    # rounding, relative to the square of the largest component.
    point, scale, parameters = li.point(name)
    values = {
        route: executor.evaluate(
            [point], [scale], parameters=parameters, precision=precision
        )[0]
        for route, executor in executors.items()
    }
    total = float(values["R1"].totals[1])
    assert total > 0
    assert abs(float(values["R2"].totals[1]) / total - 1) <= 1e3 * ROUNDING[carrier]


@pytest.mark.loop_physics
def test_dressed_plans_keep_route_r1_columns_and_their_links_load(li, loop_native):
    from pyamplicol.loops.v2.plan_export import (
        AMPLITUDE_PLAN_SCHEMA,
        export_plan,
    )

    slice_ = li.slice("li_gg_hg")
    r1 = export_plan(slice_, forward=False, route="R1").program
    exported = export_plan(slice_, forward=False, route="R2", links=True)
    r2 = exported.program
    assert r2["schema"] == AMPLITUDE_PLAN_SCHEMA and r2["dressed"] is True
    for key in ("rows", "columns", "column_rows", "column_weights", "born_components"):
        assert r2[key] == r1[key], key
    # Every leaf seeds one row; the written row links (the Python dressed
    # rule) are checked against the native derivation when it loads.
    families = list(r2["families"])
    assert all(family["leaves"] for family in families)
    loop_native.v2_plan_load(exported.json().encode(), digest=False)
    # The binary member decodes to the plan's JSON.
    member = exported.binary()
    model = loop_native.V2PlanModel.from_binary(member)
    assert bytes(model.json()) == exported.json().encode()


@pytest.mark.loop_physics
def test_route_r2_runs_schedule_b_only(li):
    from pyamplicol.loops.v2.plan_export import export_plan

    slice_ = li.slice("li_gg_hg")
    with pytest.raises(ValueError, match="schedule B only"):
        export_plan(slice_, forward=True, route="R2")
    with pytest.raises(ValueError, match="amplitude-mode"):
        export_plan(slice_, forward=False, route="R3")


# ------------------------------------------------- exact helicity-zero filter


@pytest.mark.loop_fast
def test_degree_bound_clears_negative_powers_and_counts_split_blocks():
    """The exact test's degree bound on a crafted plan: three positions; a
    fused kernel with terms MT/sw x and sw x (base sw with both signs: the
    Laurent numerator times sw^3 is a polynomial) and a split kernel sw on
    two block slots (two block components multiply it outside the kernel):
    d = 3 (max(1 + 1, 1 + 1, 1 + 2) + 1) = 12; counting |exponents| per term
    (9) undercounts the cleared degree."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.helicity_filter import degree_bound, points_needed

    mt, sw = ("param", "MT"), ("param", "sw")
    constants = [
        ((1, ((mt, 1), (sw, -1)), (), True), ()),
        ((1, ((sw, 1),), (), True), ()),
    ]
    program = {
        "kernels": [
            {"outputs": [[[0, [[0, 1]]], [1, [[0, 1]]]]]},
            {"outputs": [[[1, []]]]},
        ],
        "contents": [
            {
                "terms": [
                    {
                        "slots": [1, 2],
                        "bindings": [{"kernel": 0}],
                        "split": [{"kernel": 1}],
                    }
                ]
            }
        ],
        "families": [{"sequence": [[1, 0], [2, 0], [4, 0]]}],
    }
    exported = SimpleNamespace(program=program, constants=constants)
    assert degree_bound(exported) == 12
    assert points_needed(12, 4, 1e-12) == 1


@pytest.mark.loop_fast
def test_constants_without_a_residue_keep_every_row_live():
    """A constant with no value in F_p (its denominator, or a prime base
    under a negative power, a multiple of p) is refused, never read as 0: the
    exact test then finds no zero row (as the Rust carrier's non-finite
    marker keeps a row live)."""

    import random
    from fractions import Fraction
    from types import SimpleNamespace

    from pyamplicol.loops.v2 import helicity_filter as hf

    assert hf._residue(Fraction(3, 2)) * 2 % hf.PRIME == 3
    with pytest.raises(hf.Unrepresentable):
        hf._residue(Fraction(1, hf.PRIME))
    values = hf._Values(random.Random(0), {(("prime", hf.PRIME), 1): 1})
    assert values.power(("prime", hf.PRIME), {}, Fraction(1)) == 0
    with pytest.raises(hf.Unrepresentable):
        values.power(("prime", hf.PRIME), {}, Fraction(-1))
    executor = SimpleNamespace(
        exported=SimpleNamespace(
            amplitude=1,
            constants=[((Fraction(1, hf.PRIME), (), (), True), ())],
            block_rows=[],
            rows=[(), ()],
            masses=[],
        ),
        slice=SimpleNamespace(
            blocks=SimpleNamespace(process=SimpleNamespace(legs=[None, None]))
        ),
        native=lambda *args: None,
    )
    zero, _seconds = hf.zero_rows_exactly(executor, [0, 1], 1, 0)
    assert zero == set()


def _filter(li, name):
    from pyamplicol.loops.v2.helicity_filter import generation_filter

    fixture = _fixture(name)
    candidates = [[[float(x) for x in p] for p in point] for point in fixture["points"]]
    return generation_filter(li.slice(name), candidates)


@pytest.mark.loop_physics
def test_prime_field_numerators_are_exact_and_discriminate(li):
    """The exact test (F_p, p = 2^61 - 1, random block currents, momenta and
    constant factors): no row of g g > h g, g g > z z or g g > z (whose rows
    vanish only after the integration: Landau-Yang) is an integrand zero;
    every row of g g > h a and g g > a a a is (Furry: the quark loop's two
    orientations cancel inside each family), at each of several seeds."""

    from pyamplicol.loops.v2 import helicity_filter as hf
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    for name, zero in (
        ("li_gg_hg", False),
        ("li_gg_zz", False),
        ("li_gg_z", False),
        ("li_gg_ha", True),
        ("li_gg_aaa", True),
    ):
        executor = NativeExecutor(li.slice(name), forward=False)
        rows = range(len(executor.exported.rows))
        for seed in (0, 1):
            found, _seconds = hf.zero_rows_exactly(executor, rows, 1, seed)
            assert found == (set(rows) if zero else set()), (name, seed)


@pytest.mark.loop_physics
def test_filter_verdicts_keep_every_row_of_zero_processes(li):
    """Q13: the filter never empties a process. g g > h a: no row is proven
    live by the binary64 pre-screen (its amplitude is rounding within its
    estimate), every row is an integrand zero and the request has no
    counterterm ket, so every row would drop: the rows stay live and the
    verdict is a diagnostic. g g > h: the J_z-forbidden rows (+,-) and (-,+)
    are not proven live by the screen (they vanish after the integration)
    and are not integrand zeros: live. The failure bound and k are
    recorded from the plan's degree bound."""

    from pyamplicol.loops.v2.helicity_filter import PRIME

    record, dropped = _filter(li, "li_gg_ha")
    assert record["every_row_zero"] and record["counterterms"] == "absent"
    assert record["live_by_screen"] == 0 and record["dropped"] == [] and not dropped
    assert len(record["integrand_zero"]) == record["rows"] == 8
    assert record["prime"] == PRIME and record["points"] >= 1
    assert record["failure_bound"] <= record["failure_probability"]
    record, dropped = _filter(li, "li_gg_h")
    assert not record["every_row_zero"] and not dropped
    assert record["live_by_screen"] == 2 and len(record["tested"]) == 2
    assert record["integrand_zero"] == []


@pytest.mark.loop_physics
def test_dropped_rows_are_columns_without_a_row(li):
    """A dropped row (``LoopSlice.zero_rows``, set by the filter for an
    integrand zero) keeps its (row, t) columns without a row: its amplitude
    is exactly zero and its columns are not reduced; the other rows are
    unchanged bitwise (here two rows of g g > h g dropped by hand, to show the
    mechanism; the filter drops only integrand zeros)."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    name = "li_gg_hg"
    slice_ = li.slice(name)
    point, scale, parameters = li.point(name)
    whole = NativeExecutor(slice_, forward=False)
    rows = whole.exported.rows
    dropped = frozenset(rows[:2])
    slice_.zero_rows = dropped
    try:
        pruned = NativeExecutor(slice_, forward=False)
    finally:
        slice_.zero_rows = frozenset()
    size = pruned.exported.amplitude
    assert pruned.exported.program["column_rows"][: 2 * size] == [[]] * (2 * size)
    values = {}
    lanes = {}
    for label, executor in (("whole", whole), ("pruned", pruned)):
        values[label] = _evaluated(
            executor, point, scale, parameters, Precision.double()
        ).native_bare
        executor.evaluate([point], [scale], parameters=parameters)
        lanes[label] = executor.last_report["counts"]["reduction lanes"]
    assert lanes["pruned"] < lanes["whole"]
    for (order, row), vector in values["pruned"].items():
        if row in dropped:
            assert all(complex(z) == 0 for laurent in vector for z in laurent)
        else:
            assert vector == values["whole"][(order, row)]


@pytest.mark.loop_physics
def test_generation_records_the_filter_verdict(li_artifact):  # noqa: F811
    """The public route runs the filter at generation and records its
    verdict per loop-induced process (``loop_induced_helicity_filter``):
    g g > h keeps every row (the J_z-forbidden rows are tested and are not
    integrand zeros), the exact colour zero h > g a has no row to filter;
    the stored plan keeps one row per column."""

    import json

    manifest = json.loads((li_artifact / "artifact.json").read_text())
    records = {
        record["id"]: record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
    }
    verdict = records["gg_h"]["loop_induced_helicity_filter"]
    assert verdict["rows"] == 4 and verdict["dropped"] == []
    assert verdict["live_by_screen"] == 2 and len(verdict["tested"]) == 2
    assert verdict["integrand_zero"] == [] and not verdict["every_row_zero"]
    assert verdict["failure_bound"] <= verdict["failure_probability"]
    assert records["h_ga"]["loop_induced_helicity_filter"] == {
        "exact_colour_zero": True
    }


@pytest.mark.loop_physics
def test_generation_stores_a_partial_drop(li, li_artifact, tmp_path, monkeypatch):  # noqa: F811
    """The production drop path end to end: a verdict dropping a proper
    subset of rows makes generation export the process again
    (``artifact.reexport_process``) and store the dropped rows' columns
    without a row; the public runtime evaluates the stored process. Forced
    here (no tested process drops a proper subset): g g > h's rows the
    pre-screen does not prove live, the J_z-forbidden ones, which vanish only
    after the integration, so |A|^2 equals the whole artifact's to rounding.
    A zero failure probability skips the filter."""

    import dataclasses
    import json

    from pyamplicol import Runtime
    from pyamplicol.loops.v2 import helicity_filter as hf
    from pyamplicol.loops.v2.artifact import read_plan_json

    from .test_loop_v2_induced import _generate

    real = hf.filter_rows

    def forced(*args, **kwargs):
        verdict = real(*args, **kwargs)
        return dataclasses.replace(verdict, dropped=verdict.tested)

    monkeypatch.setattr(hf, "filter_rows", forced)
    output = _generate(
        tmp_path / "dropped", (("gg_h", "g g > h"),), _fixture("li_gg_h")
    )
    manifest = json.loads((output / "artifact.json").read_text())
    (record,) = manifest["extensions"]["generation"]["concrete_processes"]
    verdict = record["filters"]["loop_induced_helicity_filter"]
    assert len(verdict["dropped"]) == 2 and verdict["dropped"] == verdict["tested"]
    assert not verdict["every_row_zero"] and verdict["rows"] == 4
    plan = json.loads(
        read_plan_json(output / "processes" / "gg_h" / "loop-runtime.pacbin")
    )
    assert sorted(map(len, plan["column_rows"])) == [0, 0, 1, 1]
    point, scale, parameters = li.point("li_gg_h")
    values = {}
    for label, artifact in (("whole", li_artifact), ("dropped", output)):
        runtime = Runtime.load(artifact, process="gg_h", model_parameters=parameters)
        values[label] = runtime.evaluate([point], mu_r2=[str(scale)]).values[0][0]
    whole, dropped = values["whole"][0], values["dropped"][0]
    assert dropped != 0 and abs(dropped - whole) <= 1e-12 * abs(whole)
    record, rows = hf.generation_filter(li.slice("li_gg_h"), [[0.0]], failure=0)
    assert record == {"failure_probability": 0.0, "dropped": []} and not rows
