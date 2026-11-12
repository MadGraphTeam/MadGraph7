# SPDX-License-Identifier: 0BSD
"""C4 stored (M-FINAL MF-8G-1 item 7): a stored native process holds the IR
dipole operators of its Born basis per relabelling orbit
(``native_process.NativeProcess.dipoles``, the portable ``dipoles.FORMAT``,
member ``point-inputs/dipoles.json.zlib``), built and checked in
rusticol-core at generation, so a loaded process forms Catani's prediction
(MF-8G-3) without regeneration. Loaded and expanded, the operators equal the
slice's own (``LoopSlice.dipoles``) exactly; a process without two coloured
legs stores none.
"""

from __future__ import annotations

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("process", "born"),
    [("u u~ > d d~ g", {"QCD": 3, "QED": 0}), ("g g > g g g", {"QCD": 3, "QED": 0})],
)
def test_stored_dipoles_equal_the_slice_operators(
    loop_sm_compiled_model, loop_native, tmp_path, process, born
):
    from pyamplicol.loops.v2 import dipoles
    from pyamplicol.loops.v2.artifact import read_native_process, read_plan, write_plan
    from pyamplicol.loops.v2.native_process import NativeProcess
    from pyamplicol.models.loading import load_compiled_model

    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(process, born, loop, fixed_parameters={"MB": "0"}),
        loop_native,
    )
    process_ = NativeProcess.of(NativeExecutor(slice_, forward=False, inputs="native"))
    assert process_.dipoles is not None and process_.dipoles.startswith("{")
    companion = process_.scaleless
    path = tmp_path / "loop-runtime.pacbin"
    write_plan(
        process_.plan,
        path,
        process_,
        None if companion is None else companion.plan,
    )
    loaded = read_native_process(path, read_plan(path))
    assert loaded.dipoles == process_.dipoles
    assert dipoles.operators(loaded.dipole_orbits()) == slice_.dipoles()


@pytest.mark.loop_fast
def test_only_a_basis_with_two_coloured_legs_stores_dipoles(monkeypatch):
    """``native_process._stored_dipoles`` counts the Born basis's coloured
    legs (the basis lists the colour singlets too): a colourless Born stores
    none and builds nothing; two coloured legs among singlets build."""

    from types import SimpleNamespace

    from pyamplicol.color.connections import ColorLeg
    from pyamplicol.loops.v2 import dipoles
    from pyamplicol.loops.v2.native_process import _stored_dipoles

    built = []

    def build(slice_):
        built.append(slice_)
        return SimpleNamespace(to_json=lambda: "{}")

    monkeypatch.setattr(dipoles, "build", build)

    def slice_of(*representations):
        legs = tuple(ColorLeg(k + 1, r) for k, r in enumerate(representations))
        return SimpleNamespace(basis=SimpleNamespace(legs=legs))

    assert _stored_dipoles(slice_of(1, 1, 1, 1)) is None
    assert built == []
    coloured = slice_of(3, -3, 1, 1)
    assert _stored_dipoles(coloured) == "{}"
    assert built == [coloured]


@pytest.mark.loop_nightly
def test_a_colourless_process_stores_no_dipoles(
    loop_qcd_qed_sm_ufo, loop_native, tmp_path
):
    """e- e+ > mu- mu+ with electroweak corrections (loop_qcd_qed_sm, widths
    zero; generation about 1 min): no dipoles member, none loaded."""

    from pyamplicol.loops.v2.artifact import read_native_process, read_plan, write_plan
    from pyamplicol.loops.v2.native_process import NativeProcess
    from pyamplicol.models.loading import compile_model_source

    widths = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
    slice_ = LoopSlice.generate(
        compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False),
        SliceRequest(
            "e- e+ > m- m+",
            {"QED": 2, "QCD": 0},
            {"QED": 4, "QCD": 0},
            corrections=("QED",),
            fixed_parameters=widths,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    assert all(leg.representation == 1 for leg in slice_.basis.legs)
    process_ = NativeProcess.of(NativeExecutor(slice_, forward=False, inputs="native"))
    assert process_.dipoles is None and "dipoles.json" not in process_.members()
    companion = process_.scaleless
    path = tmp_path / "loop-runtime.pacbin"
    write_plan(
        process_.plan,
        path,
        process_,
        None if companion is None else companion.plan,
    )
    loaded = read_native_process(path, read_plan(path))
    assert loaded.dipoles is None and loaded.dipole_orbits() is None
