# SPDX-License-Identifier: 0BSD
"""Counterterm cells without a kernel that cancel between loop groups.

loop_sm's R2 of a Z with two or three gluons has a Levi-Civita Lorentz
structure (``R2_GGZ``, the axial part of ``R2_GGGZ``), which neither the tree
compiler nor the site catalogue lowers. Its couplings are proportional to the
Z's axial couplings, one rule per quark doublet member (``AxialZUp`` for the
u, c, t loop group, ``AxialZDown`` = -``AxialZUp`` for d, s, b), so with every
flavour in the loops the cell cancels exactly (``ct_recursion`` module
docstring, "Cancelling cells"); the vector (d-symbol) part of ``R2_GGGZ``
remains and enters the recursion.

Oracles: the exact cancellation of the coefficients; the slice with its
counterterms from the recursion equals the slice with every rule closed at
sites (O6), each route checking the cancellation on its own (the lowered
model with the rules' multiplicities, the site closure per block
assignment); where the cell does not cancel (a synthetic anomalous coupling,
or a ``particle_veto`` breaking a quark doublet) both routes refuse the
request. The slice requests use the synthetic model of
``test_loop_v2_ct_two_point``, whose Z couples to quarks through the vector
structure only (axial loop couplings need the electroweak sign oracles,
WP3.2); its counterterm vertices are loop_sm's.
"""

from __future__ import annotations

import pytest

from tools.loop_reference import legacy_capture

from .test_loop_v2_ct_closure import (  # noqa: F401 (fixtures)
    SLICE_CARRIERS,
    SLICE_PARAMETERS,
    SLICE_TOLERANCE,
    _certify_furry_zeros,
    _route_differences,
    closure_ct_model,
)
from .test_loop_v2_ct_two_point import two_point_model  # noqa: F401 (fixture)

GGZ_VERTICES = frozenset({"V_GGZ", "V_GGGZ"})


def _vertices(rule) -> frozenset[str]:
    return frozenset(entry.vertex_name for entry in rule.entries)


@pytest.mark.loop_physics
def test_levi_civita_cells_cancel_between_quark_doublets(
    loop_sm_compiled_model,
    closure_ct_model,  # noqa: F811
):
    """The closure's lowering of loop_sm leaves no rule to sites: the
    Levi-Civita cells of the g g Z and g g g Z R2 are left out with their
    exact coefficients, opposite between the up- and down-type loop groups,
    and the vector d-symbol cell of g g g Z is lowered. The cancellation
    needs equal numbers of admitted up- and down-type flavours, which a top
    veto breaks."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import (
        admitted_alternatives,
        rule_key,
        uncancelled_rules,
    )
    from pyamplicol.loops.v2.ct_sites import vanishes_identically
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    rules = prepare_loop_request_model(
        compiled, fixed_parameters=legacy_capture.FIXED_PARAMETERS
    ).counterterms.rules
    model = closure_ct_model()
    assert not model.residual
    names = {rule_key(rule): name for name, rule in rules.items()}
    unlowered = {names[key]: cells for key, cells in model.unlowered.items()}
    assert {
        vertex for name in unlowered for vertex in _vertices(rules[name])
    } == GGZ_VERTICES
    # Every unlowered cell is a Levi-Civita Lorentz structure of two rules.
    cells = {cell for item in unlowered.values() for cell in item}
    assert len(cells) == 2
    for cell in cells:
        members = [name for name, item in unlowered.items() if cell in item]
        assert len(members) == 2
        assert all("Epsilon" in unlowered[n][cell][0] for n in members)
        up, down = (unlowered[name][cell][1] for name in members)
        assert set(up) == set(down) == {0}
        assert vanishes_identically(compiled, up[0] + down[0])
        assert not vanishes_identically(compiled, up[0])
        weights = dict.fromkeys(unlowered, 3)
        assert {model.weights[rule_key(rules[name])] for name in members} == {3}
        assert not uncancelled_rules(unlowered, weights, _vanishes(compiled))
        # particle_veto = [t]: the u, c, t group keeps two flavours.
        top = {name: admitted_alternatives(rules[name], (6,)) for name in members}
        assert sorted(map(len, top.values())) == [2, 3]
        weights.update({name: len(top[name]) for name in members})
        assert uncancelled_rules(unlowered, weights, _vanishes(compiled)) == set(
            members
        )
    # g g g Z keeps its vector (d-symbol) cell in the lowered model.
    gggz = [name for name in unlowered if _vertices(rules[name]) == {"V_GGGZ"}]
    assert len(gggz) == 2
    for name in gggz:
        (structure,) = model.structures[rule_key(rules[name])]
        assert (structure.colour, structure.lorentz) == ("d(1,2,3)", "R2_GGVV")


def _vanishes(compiled):
    from pyamplicol.loops.v2.ct_sites import vanishes_identically

    return lambda expression: vanishes_identically(compiled, expression)


# ``name -> (process, Born QCD, QED, leg masses, vertices whose cells cancel)``.
# u u~ -> e- e+ g g also has the g g g Z R2 inside its blocks and at the root:
# its vector (d-symbol) cell in the recursion, its axial cell cancelled.
CANCELLATION_O6 = {
    "uux_emepg": ("u u~ > e- e+ g", 1, 2, (0,) * 5, {"V_GGZ"}),
    "uux_emepgg": ("u u~ > e- e+ g g", 2, 2, (0,) * 6, {"V_GGZ", "V_GGGZ"}),
}
# The Levi-Civita Lorentz structures of loop_sm's g g Z and g g g Z R2.
LEVI_CIVITA = frozenset({"R2_GGZ", "R2_GGGVa"})


@pytest.fixture(scope="module")
def cancellation_routes(two_point_model, loop_native):  # noqa: F811
    """``(name, counterterms) -> LoopSlice`` of the vector-Z synthetic model."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(two_point_model)
    built = {}

    def build(name: str, counterterms: str):
        if (name, counterterms) not in built:
            process, qcd, qed, _masses, _vertices = CANCELLATION_O6[name]
            built[name, counterterms] = LoopSlice.generate(
                compiled,
                SliceRequest(
                    process,
                    {"QCD": qcd, "QED": qed},
                    {"QCD": qcd + 2, "QED": qed},
                    fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                    counterterms=counterterms,
                ),
                loop_native,
            )
        return built[name, counterterms]

    return build


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", SLICE_CARRIERS)
@pytest.mark.parametrize("name", tuple(CANCELLATION_O6))
def test_cancelled_counterterms_from_recursion_equal_sites(
    cancellation_routes, monkeypatch, name, carrier
):
    """O6 with the Levi-Civita cells cancelled: the g g Z (and g g g Z)
    rules are admitted in the recursion with their cell left out (no rule is
    residual; the plain admission also admits the g g g Z rules where no tree
    holds them), the site closure drops the same cells on its own, and every
    component agrees."""

    from pyamplicol.loops.v2.ct_recursion import rule_key

    _certify_furry_zeros(monkeypatch)
    recursion = cancellation_routes(name, "recursion")
    program = recursion.counterterm_program.recursion
    rules = recursion.prepared.counterterms.rules
    assert not program.residual
    cancelled = {
        vertex
        for rule in program.rules
        if program.model.unlowered.get(rule_key(rules[rule]))
        and program.admission[rule]
        for vertex in _vertices(rules[rule])
    }
    assert CANCELLATION_O6[name][4] <= cancelled <= GGZ_VERTICES
    sites = cancellation_routes(name, "sites")
    closed = {
        vertex
        for site in sites.site_terms
        if site.alternatives
        for vertex in _vertices(site.rule) & GGZ_VERTICES
    }
    assert closed == CANCELLATION_O6[name][4]
    lorentz = {
        term.entry.lorentz_name
        for site, terms in sites.site_terms.items()
        if _vertices(site.rule) & GGZ_VERTICES
        for term in terms
    }
    assert not lorentz & LEVI_CIVITA
    worst = _route_differences(
        cancellation_routes,
        name,
        carrier,
        SLICE_PARAMETERS,
        masses=CANCELLATION_O6[name][3],
    )
    assert all(value <= SLICE_TOLERANCE[carrier] for value in worst.values()), worst
    if carrier == "double" and "V_GGGZ" in CANCELLATION_O6[name][4]:
        # The vector g g g Z cell counts: the sites without it differ.
        _assert_vertex_changes_r2(sites, "V_GGGZ", CANCELLATION_O6[name][3])


def _assert_vertex_changes_r2(sites, vertex: str, masses) -> None:
    from dataclasses import replace

    from pyamplicol.loops.v2.assemble import Precision

    from .test_loop_v2_ct_closure import SLICE_SCALE_SQUARED, _slice_point

    without = replace(
        sites,
        site_terms={
            site: terms
            for site, terms in sites.site_terms.items()
            if vertex not in _vertices(site.rule)
        },
        _cache={},
    )
    point = _slice_point(masses)
    full, rest = (
        route.evaluate(
            point,
            SLICE_SCALE_SQUARED,
            parameters=SLICE_PARAMETERS,
            precision=Precision.double(),
        ).components["r2"]
        for route in (sites, without)
    )
    assert any(
        abs(float(a) - float(b)) > 1e-6 * abs(float(b))
        for sector, laurent in rest.items()
        for a, b in zip(full[sector].coefficients, laurent.coefficients, strict=True)
    )


# The vector-Z model with an anomalous g g Z R2: the down-type loop group's
# coupling is twice loop_sm's, so the Levi-Civita cell no longer cancels.
ANOMALOUS_COUPLINGS = """

PYAMPLICOL_TEST_GGZDOWN = Coupling(name = 'PYAMPLICOL_TEST_GGZDOWN',
                                   value = '4.0*AxialZDown*(G**2/(12.0*cmath.pi**2))',
                                   order = {'QCD':2,'QED':1})
"""
ANOMALOUS_VERTICES = """

V_GGZ.couplings = {(0,0,0):C.R2_GGZup,(0,0,1):C.PYAMPLICOL_TEST_GGZDOWN}
"""


@pytest.fixture(scope="module")
def anomalous_model(loop_sm_ufo, tmp_path_factory):
    """``two_point_model`` with the anomalous g g Z R2, compiled once."""

    from dataclasses import replace

    from pyamplicol.loops.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )
    from pyamplicol.models.loading import compile_model_source

    from .test_loop_v2_ct_two_point import synthetic_two_point_ufo

    root = tmp_path_factory.mktemp("anomalous-model")
    source = synthetic_two_point_ufo(loop_sm_ufo, root)
    for name, text in (
        ("CT_couplings.py", ANOMALOUS_COUPLINGS),
        ("CT_vertices.py", ANOMALOUS_VERTICES),
    ):
        with (source / name).open("a", encoding="utf-8") as stream:
            stream.write(text)
    model = compile_model_source(source, cache_dir=root / "cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / "loop-sm-anomalous.json")


@pytest.mark.loop_physics
@pytest.mark.parametrize("counterterms", ("recursion", "sites"))
def test_uncancelled_levi_civita_cells_fail_closed(
    anomalous_model, loop_native, monkeypatch, counterterms
):
    """Where the g g Z cell does not cancel (an anomalous model) it has no
    kernel: the lowering cannot lower its rules, and as the vertex enters a
    tree of u u~ > e- e+ g (two gluon currents and a Z) the recursion refuses
    the request (no site fallback); the site oracle refuses it for want of a
    kernel. Neither drops the cell."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.ct_recursion import UnloweredCountertermError
    from pyamplicol.models.loading import load_compiled_model

    _certify_furry_zeros(monkeypatch)
    compiled = load_compiled_model(anomalous_model)
    error, match = (
        (
            UnloweredCountertermError,
            r"can enter a tree.*\[G G Z\].*R2_GGZ.*does not cancel",
        )
        if counterterms == "recursion"
        else (ValueError, "no ct-site kernel")
    )
    with pytest.raises(error, match=match):
        LoopSlice.generate(
            compiled,
            SliceRequest(
                "u u~ > e- e+ g",
                {"QCD": 1, "QED": 2},
                {"QCD": 3, "QED": 2},
                fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                counterterms=counterterms,
            ),
            loop_native,
        )


@pytest.mark.loop_physics
def test_uncancelled_cells_in_no_tree_are_inert(
    anomalous_model,
    two_point_model,  # noqa: F811
    loop_native,
):
    """REQUIREMENTS_ADDENDUM J: a CT vertex that cannot enter any tree of the
    request is inert. In u u~ > e- e+ (QCD corrections) the plain rule admits
    the anomalous g g Z R2 (a QCD correction, its orders within the
    program's), whose uncancelled Levi-Civita cell cannot be lowered; but no
    tree of the request holds two gluon currents, so the vertex enters none
    (``ct_closure.InsertionReachability``). Its rules are recorded inert and
    the request generates, with the components of the model whose cell
    cancels (one point, binary64). In u u~ > e- e+ g the same vertex enters
    a tree and the request is refused (the test above)."""

    from decimal import Decimal, localcontext

    from pyamplicol.loops.v2.assemble import (
        COMPONENTS,
        LoopSlice,
        Precision,
        SliceRequest,
    )
    from pyamplicol.models.loading import load_compiled_model

    from .test_loop_v2_ct_closure import SLICE_SCALE_SQUARED, _slice_point

    request = SliceRequest(
        "u u~ > e- e+",
        {"QCD": 0, "QED": 2},
        {"QCD": 2, "QED": 2},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    )
    slices = [
        LoopSlice.generate(load_compiled_model(path), request, loop_native)
        for path in (anomalous_model, two_point_model)
    ]
    recursion = slices[0].counterterm_program.recursion
    rules = slices[0].prepared.counterterms.rules
    inert = {
        name for name, reason in recursion.inert.items() if "enters no tree" in reason
    }
    assert {vertex for name in inert for vertex in _vertices(rules[name])} == {"V_GGZ"}
    assert not recursion.residual and not inert & set(recursion.rules)
    cancelling = slices[1].counterterm_program.recursion
    assert not cancelling.residual and not cancelling.inert.keys() & inert
    point = _slice_point((0,) * 4)
    mine, reference = (
        route.evaluate(
            point,
            SLICE_SCALE_SQUARED,
            parameters=SLICE_PARAMETERS,
            precision=Precision.double(),
        )
        for route in slices
    )
    with localcontext() as context:
        context.prec = 60
        for component in COMPONENTS:
            assert mine.components[component].keys() == (
                reference.components[component].keys()
            ), component
            for sector, laurent in reference.components[component].items():
                values = [Decimal(str(x)) for x in laurent.coefficients]
                ours = [
                    Decimal(str(x))
                    for x in mine.components[component][sector].coefficients
                ]
                scale = max(map(abs, values)) or 1
                assert all(
                    abs(a - b) <= SLICE_TOLERANCE["double"] * scale
                    for a, b in zip(ours, values, strict=True)
                ), (component, sector)
    assert any(
        any(laurent.coefficients) for laurent in reference.components["r2"].values()
    )
