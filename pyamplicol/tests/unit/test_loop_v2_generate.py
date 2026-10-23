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

ROWS = cheapest(lambda mask: 2 ** mask.bit_count(), STRUCTURED)


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
    python = build_schedules(families, catalogue, colour, policy=ROWS)
    assert_same_schedules(generation.schedules(colour), python)
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
    assert_same_schedules(
        generation.schedules(colour),
        build_schedules(families, catalogue, colour, policy=ROWS),
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
    python = build_schedules(
        families, slice_.loop_catalogue, slice_.colour, policy=assemble._FRAME_POLICY
    )
    if accuracy == "full":
        assert_same_schedules(slice_.plan, python)
        return
    native_first = slice_.generation.schedules(slice_.colour, model=STRUCTURED)
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
