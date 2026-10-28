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


def _vanishing(compiled) -> dict[tuple[tuple[str, ...], tuple], str]:
    """(particles, orders) -> vertex name of the loop model's vertices that
    vanish at the restriction, for the reproducer's request scope."""

    from pyamplicol.loops.v2.assemble import prepare_request_model
    from pyamplicol.loops.v2.ct_sites import vanishing_loop_vertices

    prepared = prepare_request_model(compiled, _request(compiled, *REPRODUCER))[0]
    particles = {
        entry.vertex_name: tuple(entry.particle_names)
        for entry in prepared.model.kernels
        if entry.kind == "tree"
    }
    return {
        (particles[name], orders): name
        for name, orders in vanishing_loop_vertices(compiled, prepared.model.kernels)
    }


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
    diagonal CKM matrix) exactly the W vertices of the off-diagonal CKM
    elements vanish (each a complexconjugate of a zero CKM element, which only
    the inlined helper shows), with the charged-Goldstone Yukawas of the
    same elements; the diagonal ones do not. The ``ckm`` variant, whose CKM
    elements, Wolfenstein inputs and vertices are all renamed, gives the same
    vertices by particles and orders under other names; at the restriction
    with a non-diagonal CKM matrix (``restrict_ckm``) no W vertex vanishes."""

    original = _vanishing(gmu_models())
    off_diagonal = {(u, d) for u in UP for d in DOWN} - set(zip(UP, DOWN, strict=True))
    assert _w_vertices(original) == off_diagonal
    assert (("u~", "b", "W+"), QED1) in original  # the reproducer's V_101
    goldstones = {
        particles
        for particles, orders in original
        if "G+" in particles or "G-" in particles
    }
    assert goldstones  # e.g. t~ d G+ (I113, -I213 of CKM13 and CKM31)
    renamed = _vanishing(gmu_models("ckm"))
    assert set(renamed) == set(original)
    assert not set(renamed.values()) & set(original.values())
    assert _w_vertices(_vanishing(gmu_models(restriction="ckm"))) == set()


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
    """The reproducer generates (about 5 minutes, 9 GB) and its virtual's
    poles are the Catani operator's at an exact on-shell point (binary64)."""

    from pyamplicol.loops.v2.assemble import LoopSlice

    compiled = gmu_models()
    slice_ = LoopSlice.generate(compiled, _request(compiled, *REPRODUCER), loop_native)
    result = slice_.evaluate(
        POINT, "1000000", parameters={"aS": "0.118", "MT": "173", "ymt": "173"}
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
CATANI_TOLERANCE = 1e-9
