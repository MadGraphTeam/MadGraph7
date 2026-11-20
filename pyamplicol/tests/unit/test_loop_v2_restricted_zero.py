# SPDX-License-Identifier: 0BSD
"""Vertices a model's restriction removes are no loop vertices (fail-closed-2).

A vertex of the loop model whose couplings at an order vanish identically at
the model's restriction contributes nothing, and the restricted tree model
holds no colour or Lorentz term of it. ``ct_sites.restricted_to_zero`` decides
that from the coupling values (the source helpers inlined with their exact
bodies, the internal parameters resolved to the restriction's external
ones), never from vertex, particle or parameter names; the loop catalogue
leaves such vertices out (``ct_sites.vanishing_loop_vertices``,
``families.ModelLoopCatalogue``) and the counterterm root closure skips them.
Any other loop vertex the tree model lacks is the listed refusal
``loop-vertex-outside-tree-model``, never an untyped error.

Reproducer (slice-status DECISIONS 2026-10-28 ~03:50): g t > t u u~ with QCD
and QED corrections on loop_qcd_qed_sm_Gmu's default restriction (diagonal
CKM: V_101 = u~ b W+ carries complexconjugate(CKM13) = 0) raised
"no colour structure of V_101 at (('QED', 1),)" 29 s into generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .loop_renamed_models import write_variant

GMU = {"compatibility": "gmu_particle_anti", "scale_parameter": "MU_R"}
FIXED = {
    "MB": "0",
    "ymb": "0",
    "MTA": "0",
    "ymtau": "0",
    "WT": "0",
    "WW": "0",
    "WZ": "0",
    "WH": "0",
    "WTau": "0",
}
REPRODUCER = (
    "g t > t u u~",
    {"QCD": 1, "QED": 2},
    ({"QCD": 1, "QED": 4},),
)
QED1 = (("QED", 1),)
UP, DOWN = ("u", "c", "t"), ("d", "s", "b")
# The vertices the loop catalogue leaves out of the reproducer's loops at the
# G_mu default restriction (``LoopSlice.generate``: the native model's vertex
# rules, by particles and orders): the W vertices of the six off-diagonal CKM
# elements and the top's charged-Goldstone Yukawas of the two off-diagonal
# elements it has (complexconjugate(CKM31) ymt and complexconjugate(CKM32)
# ymt; the b is massless there). The prepared native model keeps them, as
# the conjugate is only zero with the helper inlined. The shipped model's
# exactly unitary CKM (UFOmodeldb issue-1 patch) writes CKM13 itself as
# complexconjugate(CKMs13e), so the W- vertex of CKM13 joins them.
DIAGONAL_W = {(f"{up}~", down, "W+") for up, down in zip(UP, DOWN, strict=True)}
PRODUCTION = (
    {
        ((f"{up}~", down, "W+"), QED1)
        for up in UP
        for down in DOWN
        if (f"{up}~", down, "W+") not in DIAGONAL_W
    }
    | {(("t~", "d", "G+"), QED1), (("t~", "s", "G+"), QED1)}
    | {(("b~", "u", "W-"), QED1)}
)


def _request(compiled, process, born, loops, *, corrections=("QCD", "QED")):
    from pyamplicol.loops.v2.assemble import SliceRequest

    known = set(compiled.parameter_defaults)
    fixed = {name: value for name, value in FIXED.items() if name in known}
    return SliceRequest(
        process, born, loops, corrections=corrections, fixed_parameters=fixed, **GMU
    )


@pytest.fixture(scope="module")
def gmu_models(loop_gmu_ufo, tmp_path_factory):
    """``gmu_models(variant, restriction)``: the G_mu UFO, or its renamed
    ``ckm`` variant (CKM elements, Wolfenstein inputs and every vertex
    renamed, ``loop_renamed_models``), compiled at a restriction."""

    from pyamplicol.models.loading import compile_model_source

    root = tmp_path_factory.mktemp("restricted-zero")
    built: dict = {}

    def build(variant: str | None = None, restriction: str = "default"):
        key = (variant, restriction)
        if key not in built:
            source = Path(loop_gmu_ufo)
            if variant is not None:
                write_variant(source, root / variant, variant)
                source = root / variant
            built[key] = compile_model_source(
                source,
                restriction=restriction,
                cache_dir=root / "cache",
                require_supported=False,
            )
        return built[key]

    return build


def _vanishing(compiled) -> tuple[dict, dict]:
    """(particles, orders) -> vertex name of the loop model's tree vertices
    that vanish at the restriction, for the reproducer's request scope: all
    of them, and those of the native model's vertex rules (the set the loop
    catalogue leaves out, ``LoopSlice.generate``).

    The first set is checked here against an oracle independent of the rule:
    it is exactly the loop model's tree vertices (by vertex and orders) of
    which the compiled tree model holds no term (``ir.vertex_terms``; the
    tree compiler's own zero test)."""

    from pyamplicol.loops.v2.assemble import prepare_request_model
    from pyamplicol.loops.v2.ct_sites import vanishing_loop_vertices
    from pyamplicol.loops.v2.families import add_orders

    prepared = prepare_request_model(compiled, _request(compiled, *REPRODUCER))[0]
    kernels = prepared.model.kernels
    tree = [entry for entry in kernels if entry.kind == "tree"]
    found = vanishing_loop_vertices(compiled, kernels)
    held = {
        (term.vertex, add_orders(tuple(term.coupling_orders)))
        for term in compiled.ir.vertex_terms
    }
    loop = {(entry.vertex_name, add_orders(tuple(entry.order))) for entry in tree}
    assert found == loop - held
    native = json.loads(prepared.structure.native.to_json())["vertex_rules"]
    production = vanishing_loop_vertices(
        compiled, kernels, (rule["name"] for rule in native)
    )
    particles = {entry.vertex_name: tuple(entry.particle_names) for entry in tree}

    def named(keys) -> dict:
        return {(particles[name], orders): name for name, orders in keys}

    return named(found), named(production)


def _w_vertices(found) -> set[tuple[str, str]]:
    """The (up, down) flavours of the u~-type d-type W+ vertices in ``found``."""

    return {
        (particles[0].rstrip("~"), particles[1])
        for particles, orders in found
        if len(particles) == 3
        and particles[2] == "W+"
        and particles[0].rstrip("~") in UP
        and particles[1] in DOWN
        and orders == QED1
    }


@pytest.mark.loop_physics
def test_vanishing_vertices_follow_the_coupling_values_not_the_names(gmu_models):
    """At the G_mu UFO's default restriction (Wolfenstein inputs zero: a
    diagonal CKM matrix) the vanishing vertices are exactly those the tree
    compiler drops (``_vanishing``'s oracle), among them the W vertices of
    exactly the off-diagonal CKM elements (each a complexconjugate of a zero
    CKM element, which only the inlined helper shows), not the diagonal
    ones; of the native model's vertices, exactly ``PRODUCTION`` (with the
    top's charged-Goldstone Yukawas of the same elements). The ``ckm``
    variant, whose CKM elements, Wolfenstein inputs and vertices are all
    renamed, gives the same vertices by particles and orders under other
    names; at the restriction with a non-diagonal CKM matrix
    (``restrict_ckm``) no W vertex vanishes, again exactly the dropped
    ones."""

    original, production = _vanishing(gmu_models())
    off_diagonal = {(u, d) for u in UP for d in DOWN} - set(zip(UP, DOWN, strict=True))
    assert _w_vertices(original) == off_diagonal
    assert set(production) == PRODUCTION
    assert production[("u~", "b", "W+"), QED1] == "V_101"  # the reproducer's
    renamed, renamed_production = _vanishing(gmu_models("ckm"))
    assert set(renamed) == set(original)
    assert set(renamed_production) == PRODUCTION
    assert not set(renamed.values()) & set(original.values())
    assert _w_vertices(_vanishing(gmu_models(restriction="ckm"))[0]) == set()


@pytest.mark.loop_physics
def test_the_reproducer_s_loops_leave_the_vanishing_vertices_out(
    gmu_models, loop_native, monkeypatch
):
    """``LoopSlice.generate`` applies the rule: the reproducer's loop
    catalogue receives exactly ``PRODUCTION`` as its vanishing vertices,
    holds none of them, and the loop schedules (which read every loop
    vertex's colour from the tree model) are built without a refusal; the
    generation is stopped at the counterterm program that follows them
    (about 40 s)."""

    from pyamplicol.loops.v2 import assemble, ct_closure

    catalogues = []

    class Recorded(assemble.ModelLoopCatalogue):
        def __init__(self, data, **options):
            super().__init__(data, **options)
            rules = {rule["name"]: rule for rule in data["vertex_rules"]}
            vanishing = frozenset(options.get("vanishing", ()))
            catalogues.append((self, rules, vanishing))

    class Stopped(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise Stopped

    monkeypatch.setattr(assemble, "ModelLoopCatalogue", Recorded)
    monkeypatch.setattr(ct_closure.CountertermProgram, "build", stop)
    compiled = gmu_models()
    with pytest.raises(Stopped):
        assemble.LoopSlice.generate(
            compiled, _request(compiled, *REPRODUCER), loop_native
        )
    ((catalogue, rules, vanishing),) = catalogues
    assert {
        (tuple(rules[name]["particles"]), orders) for name, orders in vanishing
    } == PRODUCTION
    used = {
        (vertex.rule, vertex.orders)
        for species in catalogue.loop_species()
        for vertex in catalogue.vertices(species)
    }
    assert used and not used & vanishing
    # The diagonal W vertices stay loop vertices.
    assert {tuple(rules[name]["particles"]) for name, _ in used} >= DIAGONAL_W


@pytest.mark.loop_physics
def test_a_loop_vertex_the_tree_model_lacks_is_a_listed_refusal(
    gmu_models, loop_native, monkeypatch
):
    """Without the restriction rule (``vanishing_loop_vertices`` finding
    nothing) the reproducer's loops reach V_101, which the tree model does
    not hold: the generation refuses with the listed, typed
    ``loop-vertex-outside-tree-model`` naming the vertex, its particles and
    orders, where it raised an untyped ValueError (about 30 s)."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    compiled = gmu_models()
    monkeypatch.setattr(assemble, "vanishing_loop_vertices", lambda *_args: frozenset())
    with pytest.raises(LoopRequestUnsupported) as refused:
        assemble.LoopSlice.generate(
            compiled, _request(compiled, *REPRODUCER), loop_native
        )
    assert refused.value.key == "loop-vertex-outside-tree-model"
    message = str(refused.value)
    # The process in the model's spelling (its gluon is G).
    assert message.startswith("loop-vertex-outside-tree-model: 'G t > t u u~': ")
    assert (
        "the loop vertex V_101 (u~ b W+) at the coupling orders {'QED': 1} has "
        "no term in the request's tree model"
    ) in message


@pytest.mark.loop_nightly
def test_the_reproducer_generates_with_catani_poles(gmu_models, loop_native):
    """The reproducer generates (about 5 minutes, 9 GB; 150 families) and its
    virtual's poles are the Catani operator's at an exact on-shell point,
    evaluated with 32 digits (measured equal to the printed digits; binary64
    misses them by 2e-11 and 1.2e-12 of the largest coefficient, its
    conditioning at this point: the finite part moves by 2e-10)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision

    compiled = gmu_models()
    slice_ = LoopSlice.generate(compiled, _request(compiled, *REPRODUCER), loop_native)
    result = slice_.evaluate(
        POINT,
        "1000000",
        parameters={"aS": "0.118", "MT": "173", "ymt": "173"},
        precision=Precision.arbitrary(32),
    )
    (sector,) = result.components["virtual"]
    virtual = [
        complex(x).real for x in result.components["virtual"][sector].coefficients
    ]
    single, double = (complex(x).real for x in result.ir_poles[sector])
    scale = max(abs(x) for x in virtual)
    assert abs(virtual[2] - double) <= CATANI_TOLERANCE * scale
    assert abs(virtual[1] - single) <= CATANI_TOLERANCE * scale


# An exact on-shell point of g t > t u u~ (massive top at MT = 173 along the
# beam; ``tools/loop_reference/exact_points.py`` construction, terminating
# decimals), from the ew-leptons probe of the reproducer.
POINT = [
    ["315.746625", "0", "0", "315.746625"],
    ["360.034625", "0", "0", "-315.746625"],
    ["251.220354248", "104.740294205952", "138.408125607936", "55.27502079456"],
    ["121.270270752", "69.955105794048", "94.519074392064", "29.64635420544"],
    ["303.290625", "-174.6954", "-232.9272", "-84.921375"],
]
# As the electroweak Catani checks (``test_loop_v2_ew_gauge.py``: 1e-12 of the
# largest coefficient).
CATANI_TOLERANCE = 1e-12
