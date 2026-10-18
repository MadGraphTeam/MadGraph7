# SPDX-License-Identifier: 0BSD
"""Exact model scalar grades for rectangular Born/counterterm contractions."""

from __future__ import annotations

import os
from collections import Counter

import pytest

from pyamplicol.loops.color_projection import prepare_color_projection
from pyamplicol.loops.physical_color import split_color_grades


@pytest.fixture(scope="module")
def symbols():
    symbolic = pytest.importorskip("symbolica")
    from symbolica import S

    return symbolic, S("grade_test::Nc", is_real=True), S("grade_test::x")


def test_scalar_grades_preserve_alias_order_cf_and_exact_cancellation(symbols):
    s, nc, x = symbols
    expressions = (x * (nc**2 - 1) / (2 * nc), s.E("0"), 3 * x, (nc**2 - 1) / (nc - 1))
    grades = split_color_grades(expressions, nc)
    assert grades == (
        (-1, (-x / 2, s.E("0"), s.E("0"), s.E("0"))),
        (0, (s.E("0"), s.E("0"), 3 * x, s.E("1"))),
        (1, (x / 2, s.E("0"), s.E("0"), s.E("1"))),
    )
    assert all(nc not in value.get_all_symbols() for _, row in grades for value in row)
    assert split_color_grades((s.E("0"), s.E("0")), nc) == ()
    assert split_color_grades((), nc) == ()


def test_conditional_grades_keep_both_runtime_branches(symbols):
    s, nc, mass = symbols
    condition = s.S("symbolica::if")
    expression = condition(mass, (nc**2 - 1) * (mass + nc) / (2 * nc), 3 / nc)
    grades = split_color_grades((expression,), nc)
    assert [degree for degree, _ in grades] == [-1, 0, 1, 2]
    reconstructed = sum((nc**degree * values[0] for degree, values in grades), s.E("0"))
    for value in (0, 2):
        assert (
            expression.replace(mass, value) - reconstructed.replace(mass, value)
        ).together().expand() == 0
    with pytest.raises(NotImplementedError, match="Nc-dependent conditions"):
        split_color_grades((condition(nc, mass, s.E("0")),), nc)
    for value in (1 / (nc + 1), nc.log()):
        with pytest.raises(ValueError, match="finite Laurent"):
            split_color_grades((value,), nc)


def test_projection_uses_product_grade_and_one_structural_degree(symbols):
    s, nc, _ = symbols
    # A gluon loop and a quark counterterm share P=5 at fixed Nf. The
    # scalar-grade-one UV row must retain its Nc^3 piece at NLC; projecting
    # the degree-four overlap alone at P=5 would incorrectly remove its LC.
    born_overlap = nc**4 - nc**2
    rows = [[nc * born_overlap], [born_overlap]]
    projection = prepare_color_projection(nc, rows)
    assert projection.leading_degree == 5
    assert projection.project("lc") == (nc**5, s.E("0"))
    assert projection.project("nlc") == (nc**5 - nc**3, nc**4)
    assert projection.project("full") == (nc**5 - nc**3, born_overlap)


def test_loop_sm_pure_gluon_uv_and_fields_have_explicit_uniform_color_grades(symbols):
    s, nc, _ = symbols
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to the loop_sm source")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.wavefunction import prepare_external_field_rules
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(path))
    particles = source.raw.collections["particles"]
    ca, tf = s.S("UFO::CA", "UFO::TF")
    counted = Counter()
    for entry in source.counterterms("UV"):
        if any(particles[i]["fields"]["pdg_code"] != 21 for i in entry.particle_ids):
            continue
        assert entry.vertex_name in {"V_UV1eps3G", "V_UV1eps4G"}
        assert entry.color in {
            "f(1,2,3)",
            "f(-1,1,2)*f(3,4,-1)",
            "f(-1,1,3)*f(2,4,-1)",
            "f(-1,1,4)*f(2,3,-1)",
        }
        species = {
            particles[i]["fields"]["pdg_code"]
            for group in entry.loop_particles
            for i in group
        }
        gauge = species == {21}
        assert gauge or species <= {1, 2, 3, 4, 5, 6}
        assert (
            ca if gauge else tf
        ) in entry.coupling.value.expression.get_all_symbols()
        expressions = tuple(
            entry.coupling.value.coefficient(power)
            .replace(ca, nc)
            .replace(tf, s.E("1/2"))
            for power in (0, -1, -2)
        )
        grades = split_color_grades(expressions, nc)
        assert tuple(degree for degree, _ in grades) == ((1,) if gauge else (0,))
        counted[len(entry.particle_ids), gauge] += 1
    assert counted == {(3, False): 4, (3, True): 1, (4, False): 12, (4, True): 3}
    fields = [rule for rule in prepare_external_field_rules(source) if rule.pdg == 21]
    assert len(fields) == 3
    for rule in fields:
        assert tf in rule.coefficient.expression.get_all_symbols()
        expressions = tuple(
            rule.coefficient.coefficient(power).replace(tf, s.E("1/2"))
            for power in (0, -1, -2)
        )
        assert tuple(degree for degree, _ in split_color_grades(expressions, nc)) == (
            0,
        )
