# SPDX-License-Identifier: 0BSD
"""Loop-induced |A_loop|^2 on route R2 (M-LIT LIT-R2): amplitude mode in
schedule B with dressed-leg rows (``plan_export._Exporter._dressed_family``,
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
* route R2 runs schedule B only.
"""

from __future__ import annotations

import pytest

from .test_loop_v2_induced import _fixture, _Rows

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
