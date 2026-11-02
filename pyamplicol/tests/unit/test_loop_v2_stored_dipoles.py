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
