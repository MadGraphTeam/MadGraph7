# SPDX-License-Identifier: 0BSD
"""Levi-Civita counterterm cells: left out where they cancel between loop
groups, lowered where they do not.

loop_sm's R2 of a Z with two or three gluons has a Levi-Civita Lorentz
structure (``R2_GGZ``, the axial part of ``R2_GGGZ``; UFO ``Epsilon``, the
tree compiler's ``models.tensors.LEVI_CIVITA``). Its couplings are
proportional to the Z's axial couplings, one rule per quark doublet member
(``AxialZUp`` for the u, c, t loop group, ``AxialZDown`` = -``AxialZUp`` for
d, s, b), so with every flavour in the loops the cell cancels exactly
(``ct_recursion`` module docstring, "Cancelling cells") and the lowered model
leaves it out: loop_sm's lowered models keep their bytes. The vector
(d-symbol) part of ``R2_GGGZ`` remains and enters the recursion. Where the
cell does not cancel (a synthetic anomalous coupling, or a ``particle_veto``
breaking a quark doublet under the override ``allow_anomalous_veto``) the
recursion lowers it as tree kernels (WP3.2 EWI-7b) and the site closure
closes it with the catalogue's kernels. A veto that keeps a cell from
cancelling which cancels without it is refused by default ([U2],
``ct_recursion.AnomalousVetoError``), and a cell the tree compiler cannot
lower that does not cancel refuses the request.

Oracles: the exact cancellation of the coefficients; the slice with its
counterterms from the recursion equals the slice with every rule closed at
sites (O6), each route deciding the cancellation on its own (the lowered
model with the rules' multiplicities, the site closure per block
assignment), with the cells cancelled and with them lowered. The slice
requests use the synthetic model of ``test_loop_v2_ct_two_point``, whose Z
couples to quarks through the vector structure only; its counterterm
vertices are loop_sm's. The sign of the Levi-Civita tensor is fixed by the
gluon Ward identity and MadLoop (``test_loop_v2_epsilon_sign``).
"""

from __future__ import annotations

import re

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
    veto breaks: the cell is then lowered, and makes no rule residual."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import (
        admitted_alternatives,
        rule_key,
        uncancelled_cells,
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
        assert cell in uncancelled_cells(unlowered, weights, _vanishes(compiled))
        assert not uncancelled_rules(
            unlowered, weights, _vanishes(compiled), lowered=frozenset(cells)
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
    holds them), the site closure closes the same cells with the catalogue's
    Levi-Civita kernels (WP3.1 I-AX, ``test_loop_v2_levi_civita``), whose
    up- and down-type terms cancel, and every component agrees."""

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
    assert lorentz & LEVI_CIVITA
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

    # The site terms are built on first use (``LoopSlice.site_terms``): a
    # copy with a fresh cache holding the terms without the vertex's sites.
    without = replace(
        sites,
        _cache={
            "site-terms": {
                site: terms
                for site, terms in sites.site_terms.items()
                if vertex not in _vertices(site.rule)
            }
        },
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

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

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


# ``name -> (process, Born QCD, QED, particle_veto, model)``: the g g Z cell of
# u u~ > e- e+ g does not cancel, by the anomalous coupling or by a veto of
# the top that leaves two up-type and three down-type flavours (an anomalous
# veto: the request takes the override ``allow_anomalous_veto``).
LOWERED_O6 = {
    "anomalous": ("u u~ > e- e+ g", 1, 2, (), "anomalous"),
    "top-veto": ("u u~ > e- e+ g", 1, 2, (6,), "vector-z"),
}


@pytest.fixture(scope="module")
def lowered_routes(anomalous_model, two_point_model, loop_native):  # noqa: F811
    """``(name, counterterms) -> LoopSlice`` of the ``LOWERED_O6`` requests."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    models = {"anomalous": anomalous_model, "vector-z": two_point_model}
    built = {}

    def build(name: str, counterterms: str):
        if (name, counterterms) not in built:
            process, qcd, qed, veto, model = LOWERED_O6[name]
            built[name, counterterms] = LoopSlice.generate(
                load_compiled_model(models[model]),
                SliceRequest(
                    process,
                    {"QCD": qcd, "QED": qed},
                    {"QCD": qcd + 2, "QED": qed},
                    fixed_parameters=legacy_capture.FIXED_PARAMETERS,
                    particle_veto=veto,
                    allow_anomalous_veto=bool(veto),
                    counterterms=counterterms,
                ),
                loop_native,
            )
        return built[name, counterterms]

    return build


@pytest.mark.loop_physics
@pytest.mark.parametrize("carrier", SLICE_CARRIERS)
@pytest.mark.parametrize("name", tuple(LOWERED_O6))
def test_uncancelled_levi_civita_cells_from_recursion_equal_sites(
    lowered_routes, monkeypatch, name, carrier
):
    """O6 where the g g Z cell does not cancel (WP3.2 EWI-7b): the recursion
    lowers it as tree kernels (``R2_GGZ`` in the lowered rows of both loop
    groups' rules that are admitted; no rule residual or inert for want of a
    kernel), the site closure closes it with the catalogue's Levi-Civita
    kernels, and every component agrees. The cell counts: the R2 of the
    sites without the g g Z sites differs."""

    from pyamplicol.loops.v2.ct_recursion import rule_key

    _certify_furry_zeros(monkeypatch)
    recursion = lowered_routes(name, "recursion")
    program = recursion.counterterm_program.recursion
    rules = recursion.prepared.counterterms.rules
    assert not program.residual
    model = program.model
    ggz = [
        rule_key(rules[rule])
        for rule in program.rules
        if _vertices(rules[rule]) == {"V_GGZ"} and program.admission[rule]
    ]
    assert len(ggz) == 2
    for key in ggz:
        assert {s.lorentz for s in model.structures[key]} == {"R2_GGZ"}
        assert not model.unlowered.get(key)
    # The top veto's cells cancel with every flavour admitted: lowered under
    # the override and recorded (the model's g g Z and g g g Z rules); the
    # anomalous model has no veto.
    if LOWERED_O6[name][3]:
        by_key = {rule_key(rule): rule for rule in rules.values()}
        assert set(ggz) <= set(model.anomalous_veto)
        assert all(_vertices(by_key[k]) <= GGZ_VERTICES for k in model.anomalous_veto)
    else:
        assert not model.anomalous_veto
    sites = lowered_routes(name, "sites")
    assert {
        term.entry.lorentz_name
        for site, terms in sites.site_terms.items()
        if _vertices(site.rule) & GGZ_VERTICES
        for term in terms
    } & LEVI_CIVITA
    worst = _route_differences(
        lowered_routes, name, carrier, SLICE_PARAMETERS, masses=(0,) * 5
    )
    assert all(value <= SLICE_TOLERANCE[carrier] for value in worst.values()), worst
    if carrier == "double":
        _assert_vertex_changes_r2(sites, "V_GGZ", (0,) * 5)


@pytest.mark.loop_physics
@pytest.mark.parametrize("veto", ((6,), (1, 3, 5)))
def test_anomalous_vetoes_are_refused_unless_overridden(
    two_point_model,  # noqa: F811
    loop_native,
    veto,
):
    """[U2] (the user's decision): a ``particle_veto`` that keeps the g g Z
    cell of u u~ > e- e+ g from cancelling, while it cancels with every
    flavour admitted (the top: two up-type flavours against three
    down-type; d, s, b: the down-type loop group has no loop content left
    and is inert), is refused by default with ``AnomalousVetoError``, which
    names the cell and the override. With ``allow_anomalous_veto`` the cell
    is lowered and the lowered model records the rules (the up-type group's
    under d, s, b; both under the top veto; the O6 test above checks the
    values of the latter)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.ct_recursion import AnomalousVetoError, rule_key
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(two_point_model)
    request = SliceRequest(
        "u u~ > e- e+ g",
        {"QCD": 1, "QED": 2},
        {"QCD": 3, "QED": 2},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        particle_veto=veto,
    )
    with pytest.raises(
        AnomalousVetoError,
        match=(
            r"can enter a tree.*\[G G Z\].*Levi-Civita cell .*R2_GGZ.* cancels "
            r"between the rules of its vertex with every flavour admitted but not "
            rf"under particle_veto {re.escape(str(list(veto)))} .*allow_anomalous_veto"
        ),
    ):
        LoopSlice.generate(compiled, request, loop_native)
    if veto != (1, 3, 5):
        return
    from dataclasses import replace

    allowed = LoopSlice.generate(
        compiled, replace(request, allow_anomalous_veto=True), loop_native
    )
    program = allowed.counterterm_program.recursion
    rules = allowed.prepared.counterterms.rules
    assert not program.residual
    ggz = {
        rule_key(rules[name])
        for name in program.rules
        if _vertices(rules[name]) == {"V_GGZ"} and program.admission[name]
    }
    assert len(ggz) == 1
    assert ggz <= set(program.model.anomalous_veto)
    assert program.model.allow_anomalous_veto
    for key in ggz:
        assert {s.lorentz for s in program.model.structures[key]} == {"R2_GGZ"}


@pytest.mark.loop_physics
def test_uncancelled_cells_without_tree_kernels_are_refused(
    anomalous_model, loop_native, monkeypatch
):
    """A cell the tree compiler cannot lower that does not cancel makes the
    rules of its cell residual, and a request whose trees reach one is
    refused (``UnloweredCountertermError``, no site fallback,
    REQUIREMENTS_ADDENDUM J): the anomalous model's g g Z cell, with the tree
    compiler's ``Epsilon`` withheld (as before EWI-7b), at u u~ > e- e+ g.
    The refusal is not the anomalous-veto one (no veto)."""

    from collections import OrderedDict

    from pyamplicol.loops.v2 import ct_closure, ct_recursion
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    original = ct_recursion.counterterm_lorentz_error

    def withheld(compiled, name):
        if name == "R2_GGZ":
            return "a Levi-Civita tensor (Epsilon), withheld by the test"
        return original(compiled, name)

    monkeypatch.setattr(ct_recursion, "counterterm_lorentz_error", withheld)
    monkeypatch.setattr(ct_closure, "_LOWERED", OrderedDict())
    request = SliceRequest(
        "u u~ > e- e+ g",
        {"QCD": 1, "QED": 2},
        {"QCD": 3, "QED": 2},
        fixed_parameters=legacy_capture.FIXED_PARAMETERS,
    )
    with pytest.raises(
        ct_recursion.UnloweredCountertermError,
        match=r"can enter a tree.*\[G G Z\].*R2_GGZ.*withheld.*does not cancel",
    ) as caught:
        LoopSlice.generate(load_compiled_model(anomalous_model), request, loop_native)
    assert not isinstance(caught.value, ct_recursion.AnomalousVetoError)


@pytest.mark.loop_physics
def test_lowered_cells_in_no_tree_change_nothing(
    anomalous_model,
    two_point_model,  # noqa: F811
    loop_native,
):
    """In u u~ > e- e+ (QCD corrections) the plain rule admits the anomalous
    g g Z R2 (a QCD correction, its orders within the program's), whose
    uncancelled Levi-Civita cell is lowered (no rule residual, none inert for
    want of a kernel); no tree of the request holds two gluon currents
    (``ct_closure.InsertionReachability``), so the components equal those of
    the model whose cell cancels (one point, binary64). Rules the lowering
    cannot lower and no tree reaches are inert (REQUIREMENTS_ADDENDUM J;
    ``test_loop_v2_ct_recursion``)."""

    from decimal import Decimal, localcontext

    from pyamplicol.loops.v2.assemble import (
        COMPONENTS,
        LoopSlice,
        Precision,
        SliceRequest,
    )
    from pyamplicol.loops.v2.ct_recursion import rule_key
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
    assert not recursion.residual
    assert not any("enters no tree" in reason for reason in recursion.inert.values())
    ggz = [name for name in recursion.rules if _vertices(rules[name]) == {"V_GGZ"}]
    assert len(ggz) == 2
    for name in ggz:
        structures = recursion.model.structures[rule_key(rules[name])]
        assert {s.lorentz for s in structures} == {"R2_GGZ"}
    cancelling = slices[1].counterterm_program.recursion
    assert not cancelling.residual
    assert all(
        not cancelling.model.structures.get(rule_key(rules[name])) for name in ggz
    )
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
