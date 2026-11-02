# SPDX-License-Identifier: 0BSD
"""The LC/NLC prune releases the complete schedule build (M-FINAL MF-8G-1).

``LoopSlice._prune_colour`` rebuilds the pruned schedules from the complete
native build (``NativeGeneration.schedules`` with a base) and then releases
that build (``NativeGeneration.release``: the native arena entry, the
generation's references and its seed stage), since a pruned slice is never
pruned again. The pruned plan's view keeps the Python liveness instead of
the native flags that held the build. The oracle is the same generation
without the release: its exported plan and tree program are byte-identical.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import generator_input
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

FIXED = {"MB": "0"}


def _request(process: str, born: dict[str, int], accuracy: str) -> SliceRequest:
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    declaration = LoopColorDeclaration.from_dict(
        json.loads(Path(legacy_capture.COLOUR_DECLARATION).read_text())
    )
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    return SliceRequest(
        process,
        born,
        loop,
        accuracy=accuracy,
        fixed_parameters=FIXED,
        colour_declaration=declaration,
    )


def _stored(slice_: LoopSlice) -> tuple[str, bytes]:
    exported = NativeExecutor(slice_, forward=False, inputs="native").exported
    return exported.json(), exported.trees.binary()


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("process", "born", "accuracy"),
    [
        ("g g > g g g", {"QCD": 3, "QED": 0}, "lc"),
        ("u u~ > g g g", {"QCD": 3, "QED": 0}, "nlc"),
    ],
)
def test_prune_releases_the_complete_build(
    loop_sm_compiled_model, loop_native, monkeypatch, process, born, accuracy
):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    request = _request(process, born, accuracy)
    released: list[tuple[weakref.ref, int, object]] = []
    release = generator_input.NativeGeneration.release

    def spy(self, build):
        released.append((weakref.ref(build), build.handle, self._native))
        release(self, build)

    monkeypatch.setattr(generator_input.NativeGeneration, "release", spy)
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    pruned = [s for s in (slice_, slice_.scaleless) if s is not None and s.colour_pruned]
    assert slice_.colour_pruned and len(released) == len(pruned)
    gc.collect()
    for ref, handle, native in released:
        # Nothing holds the decoded complete build, and its handle is gone.
        assert ref() is None
        with pytest.raises(Exception, match="released"):
            native.readings(handle)
    for s in pruned:
        assert s.generation.base is None
        build = s.generation.build_of_plan(s.plan)
        assert build is not None and not build.complete
        assert build.readings()[2]  # the pruned plan's own build stays
    stored = _stored(slice_)

    # The oracle: the same generation keeping every build.
    monkeypatch.setattr(
        generator_input.NativeGeneration, "release", lambda self, build: None
    )
    kept = LoopSlice.generate(compiled, request, loop_native)
    assert kept.generation.base is not None and kept.generation.base.complete
    assert _stored(kept) == stored
