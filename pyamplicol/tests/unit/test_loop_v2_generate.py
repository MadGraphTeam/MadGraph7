# SPDX-License-Identifier: 0BSD
"""Native v2 generator stages equal the Python generator (WP3.1 I8/I9).

The Python stages are the oracles: families (``generate_families``) and the
skeletons, view and schedule-B tries (``build_schedules``) built natively
must decode into equal objects, in the same order, on the control model and
on loop_sm, at full colour and through the LC/NLC prune pass.
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2 import generator_input
from pyamplicol.loops.v2.families import (
    LoopOrderSelection,
    ModelBlockTable,
    ModelLoopCatalogue,
    generate_families,
)
from pyamplicol.loops.v2.generator_input import NativeGeneration
from pyamplicol.loops.v2.schedules import (
    STRUCTURED,
    SkeletonView,
    build_schedules,
    cheapest,
)
from pyamplicol.loops.v2.skeletons import DressedColour, PlanarWords


def _rows(table) -> tuple[tuple[int, ...], object]:
    """The helicity rows of the control processes' legs (all massless: two
    each) and the Python oracle's frame policy on them (the native policy
    takes the counts explicitly, ``schedules(leg_rows=...)``)."""

    counts = (2,) * len(table.legs)
    return counts, cheapest(generator_input.frame_rows(counts), STRUCTURED)


def assert_same_families(native, python) -> None:
    assert native.legs == python.legs
    assert list(native.families.items()) == list(python.families.items())
    assert native.scaleless == python.scaleless
    assert native.rejected == python.rejected
    assert native.merged == python.merged
    a, b = native.prefixes, python.prefixes
    assert a.parent == b.parent
    assert a.position == b.position
    assert a.species == b.species
    assert a.depth == b.depth
    assert a._index == b._index


def assert_same_schedules(native, python) -> None:
    assert native.view._contents == python.view._contents
    assert native.view._labels == python.view._labels
    assert list(native.schedules) == list(python.schedules)
    for key, mine in native.schedules.items():
        theirs = python.schedules[key]
        assert mine.family == theirs.family
        assert mine.trie.frame == theirs.trie.frame, key
        assert mine.trie.leaves == theirs.trie.leaves, key
        assert mine.trie.nodes == theirs.trie.nodes, key
        assert mine == theirs, key


@pytest.fixture(scope="module")
def control():
    from pyamplicol.loops.control_models import qcd_qed_control_model

    model = qcd_qed_control_model()
    return model, ModelLoopCatalogue(model)


def _selection(born, corrections=("QCD",)):
    loop = {**born, "QCD": born.get("QCD", 0) + 2}
    return LoopOrderSelection(
        frozenset(corrections), frozenset({tuple(sorted(loop.items()))})
    )


CONTROL_PROCESSES = [
    (["g", "g"], ["g", "g"], {"QCD": 2}),
    (["u", "u~"], ["g", "g"], {"QCD": 2}),
    (["u", "u~"], ["d", "d~", "g"], {"QCD": 3}),
    (["g", "g"], ["g", "g", "g"], {"QCD": 3}),
    (["u", "u~"], ["e-", "e+"], {"QED": 2}),
]


@pytest.mark.loop_fast
@pytest.mark.parametrize(("incoming", "outgoing", "born"), CONTROL_PROCESSES)
@pytest.mark.parametrize("colour_model", ["dressed", "planar"])
def test_native_stages_equal_python_on_the_control_model(
    control, loop_native, incoming, outgoing, born, colour_model
):
    model, catalogue = control
    selection = _selection(born)
    table = ModelBlockTable(model, incoming, outgoing, cap=selection.cap)
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    families = generate_families(table, catalogue, selection)
    assert_same_families(generation.families(), families)
    colour = (
        DressedColour() if colour_model == "dressed" else PlanarWords(len(table.legs))
    )
    counts, rows = _rows(table)
    python = build_schedules(families, catalogue, colour, policy=rows)
    assert_same_schedules(generation.schedules(colour, leg_rows=counts), python)
    # The prune pass: a filter and each family's stored frame.
    frames = {key: s.trie.frame for key, s in python.schedules.items()}

    def keep(skeleton):
        return skeleton.contents[0].mask % 3 != 1 or len(skeleton.contents) > 3

    def same_frame(family, candidates):
        return next(t for f, t in candidates if f == frames[family.key])

    pruned = build_schedules(
        families,
        catalogue,
        colour,
        policy=same_frame,
        view=SkeletonView(catalogue, colour, keep=keep),
    )
    assert_same_schedules(
        generation.schedules(colour, frames=frames, keep=keep), pruned
    )


@pytest.mark.loop_fast
def test_native_stages_keep_mixed_order_sectors_apart(control, loop_native):
    model, catalogue = control
    sectors = frozenset({(("QCD", 4),), (("QCD", 2), ("QED", 2))})
    selection = LoopOrderSelection(frozenset({"QCD", "QED"}), sectors)
    table = ModelBlockTable(model, ["u", "u~"], ["d", "d~"], cap=selection.cap)
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    families = generate_families(table, catalogue, selection)
    assert_same_families(generation.families(), families)
    colour = PlanarWords(len(table.legs))
    counts, rows = _rows(table)
    assert_same_schedules(
        generation.schedules(colour, leg_rows=counts),
        build_schedules(families, catalogue, colour, policy=rows),
    )


@pytest.mark.loop_fast
def test_native_family_rank_check_and_errors(control, loop_native):
    """The native recursion refuses what the Python one refuses."""

    model, catalogue = control
    selection = _selection({"QCD": 2})
    table = ModelBlockTable(model, ["g", "g"], ["g", "g"], cap=selection.cap)
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    with pytest.raises(ValueError, match="no fixed frame"):
        generation.schedules(DressedColour(), frames={})


# --------------------------------------------------------------------- loop_sm


SLICE_CASES = [
    ("gg_gg", "full"),
    ("uux_ddxg", "full"),
    ("uux_uuxg", "full"),
    ("gg_gg", "lc"),
    ("gg_gg", "nlc"),
]


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    import json

    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tests.unit.test_loop_v2_slice import _request
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    built = {}

    def get(name, accuracy):
        if (name, accuracy) not in built:
            options = {}
            if accuracy != "full":
                options = {"accuracy": accuracy, "colour_declaration": declaration}
            built[name, accuracy] = LoopSlice.generate(
                compiled, _request(name, **options), loop_native
            )
        return built[name, accuracy]

    return get


@pytest.mark.loop_physics
@pytest.mark.parametrize(("name", "accuracy"), SLICE_CASES)
def test_native_stages_equal_python_in_the_slice(slices, name, accuracy):
    """LoopSlice.generate runs the native stages; the Python stages on the
    same block table and colour model give equal families and schedules,
    including the LC/NLC prune with each family's stored frame."""

    from pyamplicol.loops.v2 import assemble

    slice_ = slices(name, accuracy)
    assert slice_.generation is not None
    table = slice_.blocks.block_table(slice_.data)
    loop = assemble.add_orders(tuple(slice_.request.loop_orders.items()))
    selection = LoopOrderSelection(
        frozenset(slice_.request.corrections), frozenset({loop})
    )
    families = generate_families(table, slice_.loop_catalogue, selection)
    assert_same_families(slice_.families, families)
    counts = generator_input.leg_rows(slice_.blocks)
    python = build_schedules(
        families,
        slice_.loop_catalogue,
        slice_.colour,
        policy=assemble._frame_policy(counts),
    )
    if accuracy == "full":
        assert_same_schedules(slice_.plan, python)
        return
    native_first = slice_.generation.schedules(
        slice_.colour, model=STRUCTURED, leg_rows=counts
    )
    assert_same_schedules(native_first, python)
    frames = {key: s.trie.frame for key, s in python.schedules.items()}
    keep = slice_.plan.view.keep

    def same_frame(family, candidates):
        return next(t for f, t in candidates if f == frames[family.key])

    pruned = build_schedules(
        families,
        slice_.loop_catalogue,
        slice_.colour,
        policy=same_frame,
        view=SkeletonView(slice_.loop_catalogue, slice_.colour, keep=keep),
    )
    assert_same_schedules(slice_.plan, pruned)


def test_native_stage_switch_names_the_stages():
    assert frozenset({"families", "schedules"}) == generator_input.NATIVE_STAGES


COMPANION_CASES = [
    ("g g > g g g", 3, "full"),
    ("u u~ > g g g", 3, "full"),
    # The companion's LC prune (``_prune_colour`` with each family's frame).
    ("u u~ > g g g", 3, "lc"),
    pytest.param("g g > g g g g", 4, "full", marks=pytest.mark.loop_nightly),
]


@pytest.mark.loop_physics
@pytest.mark.parametrize(("process", "qcd", "accuracy"), COMPANION_CASES)
def test_scaleless_companion_on_native_families_equals_python(
    loop_sm_compiled_model, loop_native, monkeypatch, process, qcd, accuracy
):
    """The companion of the certified-scaleless families (poles-uvir's
    ``LoopSlice.scaleless``, run by the "uv" and "ir" pole selections) on
    the native family recursion and native schedules equals the one on
    Python's stages: its families, prefix states and schedules (S1, S4) and
    seeds (S5), and the main slice's families with their scaleless records
    (RUST-MIGRATION.md §4 item 1(b))."""

    import json

    from pyamplicol.loops.v2 import snapshot
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    options = {}
    if accuracy != "full":
        options = {
            "accuracy": accuracy,
            "colour_declaration": LoopColorDeclaration.from_dict(
                json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
            ),
        }
    request = SliceRequest(
        process, {"QCD": qcd}, {"QCD": qcd + 2}, fixed_parameters={"MB": "0"}, **options
    )
    found = {}
    for stages in (generator_input.NATIVE_STAGES, frozenset()):
        monkeypatch.setattr(generator_input, "NATIVE_STAGES", stages)
        slice_ = LoopSlice.generate(compiled, request, loop_native)
        assert slice_.families.scaleless_families
        assert slice_.scaleless is not None
        # The native side schedules (and prunes) the companion natively.
        assert (slice_.scaleless.generation is not None) is bool(stages)
        assert slice_.scaleless.colour_pruned is (accuracy != "full")
        found[bool(stages)] = (
            snapshot.digest(snapshot.families_records(slice_.families)),
            snapshot.stage_digests(slice_.scaleless, plan=False),
        )
    assert found[True] == found[False]


# ---------------------------------------------------------- reachability (I1a)


REACHABILITY_CASES = ["gg_gg", "uux_ddxg", "uux_ggg", "dy", "tt", "uux_uuxg"]


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", REACHABILITY_CASES)
def test_loop_aware_reachability_changes_no_result(
    loop_sm_compiled_model, loop_native, monkeypatch, name
):
    """Building only the block currents the field-level families consume
    leaves the families, schedules, seeds, the plan and the complete virtual
    (counterterm sites and external fields included) unchanged."""

    from pyamplicol.loops.v2 import assemble, snapshot
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model
    from tests.unit.test_loop_v2_slice import _fixture, _parameters, _request

    compiled = load_compiled_model(loop_sm_compiled_model)
    fixture = _fixture(name)
    points, scales = fixture["points"][:1], fixture["scales_squared"][:1]
    found = {}
    for reachability in (True, False):
        monkeypatch.setattr(assemble, "LOOP_BLOCK_REACHABILITY", reachability)
        slice_ = LoopSlice.generate(compiled, _request(name), loop_native)
        digests = snapshot.stage_digests(slice_)
        found[reachability] = (
            slice_.families.families,
            slice_.families.scaleless,
            {k: v for k, v in digests.items() if k != "S1"},
            snapshot.virtual_digest(slice_, points, scales, _parameters(name)),
            len(slice_.blocks.blocks),
        )
    pruned, full = found[True], found[False]
    assert pruned[:4] == full[:4]
    assert pruned[4] <= full[4]


@pytest.mark.loop_physics
def test_born_orders_the_root_closure_misses_are_refused(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """Without a QED bound u u~ > d d~ g has a QCD 1 QED 2 Born (photon, Z and
    W exchange). The root closure at leg 1 builds it, also for QCD loops of
    the QCD Born only: loop-aware reachability adds the block currents the
    Born closure attaches (``assemble._born_block_requests``, beyond the loop
    orders' cap) and the W vertex closes through its conjugate rule (WP3.2
    EWI-2). With the QCD loops of both Born orders the renormalised virtual
    is UV-finite in every sector and the QCD sector's poles are Catani's
    (the QCD x electroweak sectors are no Catani check with QCD corrections
    alone: the loop selection leaves out the photon-gluon boxes, as
    MadLoop's does). Without those currents the closure has no terms at that
    order, and generation refuses the request rather than leave the
    interference sectors zero."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model
    from tools.loop_reference import exact_points

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "u u~ > d d~ g", {"QCD": 3}, {"QCD": 5}, fixed_parameters={"MB": "0"}
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    assert (("QCD", 1), ("QED", 2)) in slice_.blocks.born_orders
    assert set(slice_.born_terms) == set(slice_.blocks.born_orders)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "u u~ > d d~ g",
            {"QCD": 3},
            ({"QCD": 5, "QED": 0}, {"QCD": 3, "QED": 2}),
            fixed_parameters={"MB": "0"},
        ),
        loop_native,
    )
    (point,) = exact_points.generate([0] * 5, count=1)
    point = [[exact_points.decimal_string(x) for x in leg] for leg in point]
    parameters = {"aS": "0.118", "MT": "173"}
    result = slice_.evaluate(point, "1000000", parameters=parameters)
    virtual = result.components["virtual"]
    assert len(virtual) == 3
    single, double = result.ir_poles[(("QCD", 8),)]
    coefficients = [float(x) for x in virtual[(("QCD", 8),)].coefficients]
    scale = max(abs(float(single)), abs(float(double)))
    assert abs(coefficients[1] - float(single)) <= 1e-12 * scale
    assert abs(coefficients[2] - float(double)) <= 1e-12 * scale
    uv = slice_.evaluate(point, "1000000", parameters=parameters, poles="uv")
    for sector, laurent in uv.pole_origins["uv"]["virtual"].items():
        bare = [float(x) for x in uv.components["bare"][sector].coefficients]
        poles = [float(x) for x in laurent.coefficients[1:]]
        assert all(abs(x) <= 1e-12 * max(map(abs, bare)) for x in poles), sector
    monkeypatch.setattr(assemble, "_born_block_requests", lambda *args: frozenset())
    with pytest.raises(NotImplementedError, match="no terms at the Born orders"):
        LoopSlice.generate(compiled, request, loop_native)


# ------------------------------------------------- providers at 5 and 6 legs


@pytest.mark.parametrize(
    ("name", "points", "inputs"),
    [
        pytest.param("uux_ddxg", [0], "python", marks=pytest.mark.loop_physics),
        pytest.param("uux_ddxg", [0], "native", marks=pytest.mark.loop_physics),
        # g g > t t~ (massive fermion loop lines, the tree reading's mass
        # flips in the native kernel constants) is the M-QCD campaign's row
        # B01 (``test_loop_v2_mqcd_default``; PLAN §3.2 MQ-VALIDATE item 3).
        pytest.param("gg_gggg", [0], "python", marks=pytest.mark.loop_nightly),
        # gamma*/Z at seven legs: chiral couplings on loop quark lines, closed
        # quark loops with the Z, Furry-certified A g g triangles and the
        # Levi-Civita R2 rules (WP3.1 I-AX).
        pytest.param("uux_emepddxg", [1], "python", marks=pytest.mark.loop_physics),
        pytest.param(
            "uux_emepggg", [0, 1, 2], "python", marks=pytest.mark.loop_nightly
        ),
        # Point 2's binary64 finite part is ill-conditioned (5.7e-9, flagged
        # by the stability estimate; DoubleFloat 1.9e-14 of MadLoop).
        pytest.param(
            "gg_ttxemepg_qcd", [0, 1], "python", marks=pytest.mark.loop_nightly
        ),
    ],
)
def test_validate_agrees_with_the_providers(
    loop_sm_compiled_model, name, points, inputs
):
    """``tools/loop_v2_generate.py validate``: generation with the native
    stages, plan export and native evaluation against every provider of a
    required reference (the fixture's metric and tolerance), all three
    Laurent coefficients, with Python or native per-point inputs (trees,
    Born, counterterm and field components)."""

    import argparse

    from tools import loop_v2_generate

    record = loop_v2_generate.validate(
        argparse.Namespace(
            name=name,
            points=points,
            precision="double",
            stability=False,
            inputs=inputs,
            model=str(loop_sm_compiled_model),
        )
    )
    assert record["passed"], record["points"]
    compared = {
        key for p in record["points"] for row in p["errors"].values() for key in row
    }
    assert compared == {"born", "finite", "single_pole", "double_pole"}
    assert all(p["max_error"] <= 1e-10 for p in record["points"]), record["points"]


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "born", "accuracy"),
    [
        ("g g > g g g", "QCD=3", "fc"),
        ("g g > g g g", "QCD=3", "lc"),
        ("g g > g g g", "QCD=3", "nlc"),
        ("u u~ > g g g", "QCD=3", "lc"),
    ],
)
def test_poles_double_pole_is_universal(
    loop_sm_compiled_model, monkeypatch, process, born, accuracy
):
    """``tools/loop_v2_generate.py poles``: native trees and Born, the bare
    double pole over the Born is -(alpha_s/2pi) sum_i C_i, at leading colour
    with the Casimirs' leading parts (C_F -> N_c/2) over the leading-colour
    Born of the native component metric (NLC: adjoint legs only)."""

    import argparse

    from tools import loop_v2_generate

    # The tool's stage timers patch the generator's modules for good.
    monkeypatch.setattr(loop_v2_generate._Timers, "install", lambda self: None)
    record = loop_v2_generate.poles(
        argparse.Namespace(
            process=process,
            born=born,
            accuracy=accuracy,
            veto="",
            points=2,
            rotation=False,
            alpha_s="0.118",
            tolerance=1e-10,
            model=str(loop_sm_compiled_model),
        )
    )
    assert record["passed"], record["points"]


@pytest.mark.loop_physics
def test_reachability_falls_back_where_field_level_families_differ(
    loop_sm_compiled_model, loop_native, monkeypatch
):
    """With a photon, the field-level block table has currents the tree
    compiler does not build, and the pruned tree table splits fewer
    closed-loop flavour classes: the generator then builds every block
    current, so the families equal those without reachability."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    class Stop(Exception):
        pass

    # The families of the last tree program the generator builds (the
    # field-level families come first), read when the colour model of the
    # schedules is built.
    built = []
    families = generator_input.NativeGeneration.families

    def recorded(self):
        found = families(self)
        built.append(found)
        return found

    def capture(*args, **kwargs):
        raise Stop(built[-1])

    monkeypatch.setattr(generator_input.NativeGeneration, "families", recorded)
    monkeypatch.setattr(assemble, "SeedColour", capture)
    compiled = load_compiled_model(loop_sm_compiled_model)
    request = SliceRequest(
        "u u~ > e- e+ d d~ g",
        {"QCD": 3, "QED": 2},
        {"QCD": 5, "QED": 2},
        fixed_parameters={"MB": "0"},
        particle_veto=(23,),
    )
    found = {}
    for reachability in (True, False):
        monkeypatch.setattr(assemble, "LOOP_BLOCK_REACHABILITY", reachability)
        with pytest.raises(Stop) as stop:
            LoopSlice.generate(compiled, request, loop_native)
        found[reachability] = stop.value.args[0]
    assert_same_families(found[True], found[False])


# ------------------------------------------------- evaluation reuse (I1c)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("process", "born", "ignored"),
    [
        ("g g > g g g", {"QCD": 3}, ()),
        ("u u~ > d d~ s s~", {"QCD": 4}, ()),
        ("u u~ > e- e+ d d~ g", {"QCD": 3, "QED": 2}, (23,)),
    ],
)
def test_reuse_fast_path_equals_the_derivation(
    loop_sm_compiled_model, monkeypatch, process, born, ignored
):
    """The evaluation-reuse pass skips its class derivation when no two
    currents share an evaluation contract; the DAGs equal the derived ones
    (the photon process merges electroweak evaluations and derives)."""

    import pyamplicol.generation.dag_compiler as dag_compiler
    from pyamplicol.generation import dag_equivalence as equivalence
    from pyamplicol.loops.v2.blocks import build_tree_blocks
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    def derived(dag, model):
        if not dag.interactions:
            return dag
        kinds = {}
        classes = equivalence._derive_current_value_equivalences(
            dag, model, equivalence_by_kind=kinds
        )
        return equivalence._rewrite_interaction_evaluation_reuse(
            dag, model, current_equivalences=classes, equivalence_by_kind=kinds
        )

    def signature(dag):
        return (
            [
                (i.evaluation_group_id, repr(i.evaluation_factor), i.result_id)
                for i in dag.interactions
            ],
            [repr(current.index) for current in dag.currents],
        )

    compiled = load_compiled_model(loop_sm_compiled_model)
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    ir = build_model_process_ir(process, compiled.ir, color_accuracy="full")
    found = []
    for reuse in (equivalence.assign_recursive_current_evaluation_reuse, derived):
        monkeypatch.setattr(
            dag_compiler, "assign_recursive_current_evaluation_reuse", reuse
        )
        blocks = build_tree_blocks(
            ir,
            CompiledUFOModel(compiled),
            born_coupling_orders=born,
            block_coupling_orders=loop,
            ignored_particle_ids=ignored,
        )
        found.append((signature(blocks.blocks_dag), signature(blocks.born_dag)))
    assert found[0] == found[1]


# ------------------------------------------------------------- seven legs


@pytest.fixture(scope="module")
def loop_sm_request_model(loop_sm_compiled_model):
    """loop_sm prepared for QCD corrections with up to five gluon couplings."""

    import json

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    prepared = prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 5, "QED": 0},
        loop_order_bounds={"QCD": 7, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    data = json.loads(prepared.structure.native.to_json())
    catalogue = ModelLoopCatalogue(
        data, species_orders=prepared.structure.perturbative_species_orders
    )
    return compiled, data, catalogue


@pytest.mark.loop_physics
def test_native_families_at_seven_legs(loop_sm_request_model, loop_native):
    """gg -> ggggg FC: the native recursion equals Python's on the
    field-level table (9,414 families, 52,682 routings, 109,638 prefix
    states: 109,179 of the admitted families, WP3.1 plan §2.5, then the
    certified-scaleless families' for the poles companion)."""

    _compiled, data, catalogue = loop_sm_request_model
    selection = _selection({"QCD": 5})
    table = ModelBlockTable(data, ["G", "G"], ["G"] * 5, cap=selection.cap)
    native = NativeGeneration(table, catalogue, selection, loop_native).families()
    assert len(native.families) == 9414
    assert sum(len(f.closures) for f in native.families.values()) == 52682
    assert len(native.prefixes) == 109638
    assert_same_families(native, generate_families(table, catalogue, selection))


@pytest.mark.loop_nightly
def test_native_schedules_at_seven_legs(loop_sm_request_model, loop_native):
    """uu~ -> ggggg FC: the native skeletons, view and tries equal Python's
    on the reachable tree blocks with the slice's colour model."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.blocks import build_tree_blocks, requested_loop_blocks
    from pyamplicol.loops.v2.colour import SeedColour
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    compiled, data, catalogue = loop_sm_request_model
    selection = _selection({"QCD": 5})
    field = ModelBlockTable(data, ["u", "u~"], ["G"] * 5, cap=selection.cap)
    reachable = NativeGeneration(field, catalogue, selection, loop_native).families()
    process = build_model_process_ir(
        "u u~ > g g g g g", compiled.ir, color_accuracy="full"
    )
    blocks = build_tree_blocks(
        process,
        CompiledUFOModel(compiled),
        born_coupling_orders={"QCD": 5, "QED": 0},
        block_coupling_orders={"QCD": 7, "QED": 0},
        reachability=requested_loop_blocks(reachable.block_requests(), data),
    )
    table = blocks.block_table(data)
    generation = NativeGeneration(table, catalogue, selection, loop_native)
    families = generate_families(table, catalogue, selection)
    assert_same_families(generation.families(), families)
    colour = SeedColour(blocks, data, compiled.ir.vertex_terms)
    counts = generator_input.leg_rows(blocks)
    assert_same_schedules(
        generation.schedules(colour, leg_rows=counts),
        build_schedules(
            families, catalogue, colour, policy=assemble._frame_policy(counts)
        ),
    )
