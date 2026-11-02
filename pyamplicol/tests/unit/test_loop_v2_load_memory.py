# SPDX-License-Identifier: 0BSD
"""Load memory (M-FINAL MF-8G-2 items 2 and 4): a stored process loads
file-backed.

``artifact.load_native_process`` reads a ``loop-runtime.pacbin`` as
``read_native_process`` does, but the plan, the companion plan and the
binary tree programs stay in the container (``artifact.StoredMember``): the
native executor reads and inflates them itself (rusticol-core
``loop_artifact::read_loop_v2_stored_member``), and the Python view holds the
small members and each tree program's head only (inflated alone,
``read_loop_v2_stored_section``). The in-memory load is the oracle:

* the file-backed view holds stored members, reads the same plan text and
  tree-program head, and decodes to the same tree program;
* its executors evaluate bitwise as the in-memory ones (no poles, all poles,
  and the "uv" selection that runs the certified-scaleless companion);
* a container rewritten after the load is refused when an executor reads it.

The Rust plan's split steps holding their links once and the interned
interaction weights are covered by the executor's form oracles
(``test_loop_v2_executor.py``) and the binary/JSON tree-program equality
(``test_loop_v2_tree_binary.py``).
"""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}


def _points(source: str, count: int) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"][:count]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu[:count]]


@pytest.fixture(scope="module")
def stored(loop_sm_compiled_model, loop_native, tmp_path_factory):
    """u u~ > d d~ g at full colour (with its certified-scaleless companion)
    stored as a ``loop-runtime.pacbin``."""

    from pyamplicol.loops.v2.artifact import write_plan
    from pyamplicol.loops.v2.native_process import NativeProcess
    from pyamplicol.models.loading import load_compiled_model

    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > d d~ g",
            {"QCD": 3, "QED": 0},
            {"QCD": 5, "QED": 0},
            fixed_parameters={"MB": "0"},
        ),
        loop_native,
    )
    process = NativeProcess.of(NativeExecutor(slice_, forward=False, inputs="native"))
    assert process.trees.is_binary and process.scaleless is not None
    path = tmp_path_factory.mktemp("load-memory") / "loop-runtime.pacbin"
    write_plan(process.plan, path, process, process.scaleless.plan)
    return path


@pytest.mark.loop_fast
def test_file_backed_load_equals_the_in_memory_load(stored, loop_native):
    from pyamplicol.loops.v2.artifact import (
        SCALELESS_PLAN_MEMBER,
        StoredMember,
        load_native_process,
        read_native_process,
        read_plan,
        read_plan_json,
    )

    memory = read_native_process(stored, read_plan(stored))
    loaded = load_native_process(stored)
    # The Python view holds stored members, never the plan or the container.
    assert isinstance(loaded.plan, StoredMember)
    assert isinstance(loaded.trees.container, StoredMember)
    assert isinstance(loaded.scaleless.plan, StoredMember)
    assert isinstance(loaded.scaleless.trees.container, StoredMember)
    assert loaded.plan.bytes() == memory.plan == read_plan(stored)
    assert loaded.plan_json == read_plan_json(stored)
    assert loaded.scaleless.plan_json == read_plan_json(stored, SCALELESS_PLAN_MEMBER)
    for mine, theirs in (
        (loaded.trees, memory.trees),
        (loaded.scaleless.trees, memory.scaleless.trees),
    ):
        assert mine.program == theirs.program
        assert mine.binary() == theirs.container
        assert mine.json() == theirs.json()
        assert mine.kernels == theirs.kernels and mine.planes == theirs.planes
    assert loaded.record == memory.record
    # Bitwise equal evaluations (the "uv" selection runs the companion).
    points, scales = _points("required/uux_ddxg.json", 2)
    file_backed = loaded.executor(loop_native)
    in_memory = memory.executor(loop_native)
    for poles in ("none", "all", "uv"):
        assert repr(
            file_backed.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        ) == repr(
            in_memory.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        )


@pytest.mark.loop_fast
def test_a_container_rewritten_after_the_load_is_refused(stored, loop_native, tmp_path):
    from pyamplicol.loops.v2.artifact import load_native_process, read_plan, write_plan

    path = tmp_path / "loop-runtime.pacbin"
    shutil.copyfile(stored, path)
    loaded = load_native_process(path)
    # Another stored plan (one more byte, same member) in a valid container.
    plan = read_plan(stored)
    write_plan(plan + (" " if isinstance(plan, str) else b" "), path)
    executor = loaded.executor(loop_native)
    points, scales = _points("required/uux_ddxg.json", 1)
    with pytest.raises(ValueError, match="changed since the process was loaded"):
        executor.evaluate(points, scales, parameters=PARAMETERS, poles="none")
