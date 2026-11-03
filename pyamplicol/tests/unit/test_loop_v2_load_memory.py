# SPDX-License-Identifier: 0BSD
"""Load memory (M-FINAL MF-8G-2 items 2 and 4): a stored process reads its
plan and binary tree programs on demand, through the handle opened at load.

``artifact.load_native_process`` reads a ``loop-runtime.pacbin`` as
``read_native_process`` does (one member enumeration,
``artifact._native_members``), but the plan, the companion plan and the
binary tree programs are read on demand (``StoredMember``): through the
container's read-only handle opened at load (its digests verified through
it), each read digest-checked; the native executor inflates them itself
(rusticol-core ``loop_artifact::inflate_loop_v2_member``), and the Python view
holds the small members and each tree program's head only (inflated alone,
``inflate_loop_v2_section``). The in-memory load is the oracle:

* the on-demand view reads the same plan text and tree-program head, and
  decodes to the same tree program;
* its executors evaluate bitwise as the in-memory ones (no poles, all poles,
  and the "uv" selection that runs the certified-scaleless companion);
* a loaded process does not depend on its path: replacing, moving or
  deleting the artifact after the load changes no result (``Runtime.load``
  included; correlated and resolved calls in ``test_loop_v2_stored_process``),
  and a file rewritten in place is refused;
* an older build's JSON store (JSON plan member, JSON tree programs) loads
  and evaluates alike.

The Rust plan's split steps holding their links once and the interned
interaction weights are covered by the executor's form oracles
(``test_loop_v2_executor.py``) and the binary/JSON tree-program equality
(``test_loop_v2_tree_binary.py``).
"""

from __future__ import annotations

import dataclasses
import json
import os
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
def test_a_loaded_process_does_not_depend_on_its_path(stored, loop_native, tmp_path):
    from pyamplicol.loops.v2.artifact import (
        PLAN_BINARY_MEMBER,
        load_native_process,
        read_native_process,
        read_plan,
        write_plan,
    )

    oracle = read_native_process(stored, read_plan(stored))
    path = tmp_path / "loop-runtime.pacbin"
    shutil.copyfile(stored, path)
    loaded = load_native_process(path)
    # Replaced by another valid container (one more byte in the plan, same
    # member), published as the generator publishes one (a new file renamed
    # over the path): the loaded process reads the file it loaded.
    plan = read_plan(stored)
    other = tmp_path / "other.pacbin"
    write_plan(plan + (" " if isinstance(plan, str) else b" "), other)
    os.replace(other, path)
    points, scales = _points("required/uux_ddxg.json", 1)
    executor = loaded.executor(loop_native)
    expected = oracle.executor(loop_native)
    assert repr(
        executor.evaluate(points, scales, parameters=PARAMETERS, poles="none")
    ) == repr(expected.evaluate(points, scales, parameters=PARAMETERS, poles="none"))
    # Deleted: the companion executor ("uv") and the plan text are read
    # after the deletion.
    path.unlink()
    assert repr(
        executor.evaluate(points, scales, parameters=PARAMETERS, poles="uv")
    ) == repr(expected.evaluate(points, scales, parameters=PARAMETERS, poles="uv"))
    assert loaded.plan_json == oracle.plan_json
    # Rewritten in place (the loaded file itself changes): refused, never
    # read as another plan.
    shutil.copyfile(stored, path)
    rewritten = load_native_process(path)
    assert rewritten.plan.member == PLAN_BINARY_MEMBER
    with path.open("r+b") as stream:
        stream.seek(rewritten.plan.offset + rewritten.plan.length // 2)
        byte = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="changed since the process was loaded"):
        rewritten.executor(loop_native).evaluate(
            points, scales, parameters=PARAMETERS, poles="none"
        )


@pytest.mark.loop_fast
def test_a_json_store_loads_as_the_in_memory_one(stored, loop_native, tmp_path):
    """An older build's store (JSON plan members and JSON tree programs)
    through ``load_native_process``: the plans stay compressed, the JSON
    tree programs are inflated, and the values equal the binary store's."""

    from pyamplicol.generation.evaluator_container import PacbinReader
    from pyamplicol.loops.v2.artifact import (
        NATIVE_TREES_BINARY_MEMBER,
        NATIVE_TREES_MEMBER,
        PLAN_BINARY_MEMBER,
        PLAN_MEMBER,
        StoredMember,
        load_native_process,
        read_native_process,
        read_plan,
        write_plan,
    )

    memory = read_native_process(stored, read_plan(stored))

    def json_trees(trees):
        return dataclasses.replace(
            trees, program=trees.full_program(), sections=(), container=None
        )

    legacy = dataclasses.replace(
        memory,
        plan=memory.plan_json,
        trees=json_trees(memory.trees),
        scaleless=dataclasses.replace(
            memory.scaleless,
            plan=memory.scaleless.plan_json,
            trees=json_trees(memory.scaleless.trees),
        ),
    )
    path = tmp_path / "legacy.pacbin"
    write_plan(legacy.plan, path, legacy, legacy.scaleless.plan)
    with PacbinReader.open(path) as reader:
        names = {member.logical_path for member in reader.members}
    assert PLAN_MEMBER in names and PLAN_BINARY_MEMBER not in names
    assert any(name.endswith(NATIVE_TREES_MEMBER) for name in names)
    assert not any(name.endswith(NATIVE_TREES_BINARY_MEMBER) for name in names)
    loaded = load_native_process(path)
    assert isinstance(loaded.plan, StoredMember)
    assert isinstance(loaded.scaleless.plan, StoredMember)
    assert loaded.plan.text() == memory.plan_json
    for mine in (loaded.trees, loaded.scaleless.trees):
        assert mine.container is None and not mine.is_binary
    assert loaded.trees.program == memory.trees.full_program()
    assert loaded.record == memory.record
    points, scales = _points("required/uux_ddxg.json", 2)
    json_store = loaded.executor(loop_native)
    binary_store = memory.executor(loop_native)
    for poles in ("none", "all", "uv"):
        assert repr(
            json_store.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        ) == repr(
            binary_store.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        )


@pytest.mark.loop_fast
def test_a_loaded_runtime_survives_moving_and_replacing_its_artifact(tmp_path):
    """``Runtime.load`` (``V2VirtualRuntime``, ``load_native_process``): the
    executors are built on the first evaluation, after the artifact moved
    away and another file took its place, and evaluate as a runtime whose
    artifact stayed in place."""

    from pyamplicol import Runtime
    from pyamplicol._internal import selftest_virtual_ew as selftest

    root = selftest.fixture_root()
    expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
    artifact = tmp_path / "artifact"
    shutil.copytree(root / "artifact", artifact)
    in_place = selftest.evaluate(artifact, expected)

    runtime = Runtime.load(artifact, process=selftest.NAME, mute_warnings=True)
    shutil.move(artifact, tmp_path / "moved")
    for pacbin in (tmp_path / "moved").glob("processes/*/loop-runtime.pacbin"):
        # Another file at the loaded container's path.
        replaced = artifact / pacbin.relative_to(tmp_path / "moved")
        replaced.parent.mkdir(parents=True)
        replaced.write_bytes(b"not a container")
    points = [[list(map(float, p)) for p in point] for point in expected["points"]]
    scales = [float(x) for x in expected["mu_r2"]]
    result = runtime.evaluate(points, mu_r2=scales)
    shutil.rmtree(tmp_path / "moved")
    for poles in ("uv", "ir"):
        assert repr(runtime.evaluate(points, mu_r2=scales, poles=poles).values)
    again = runtime.evaluate(points, mu_r2=scales)
    assert repr(again.values) == repr(result.values)
    moved = [
        tuple(
            [sum((complex(v) for v in born), 0j).real]
            + [
                sum((complex(getattr(sector, name)) for sector in laurent), 0j).real
                for name in ("finite", "single_pole", "double_pole")
            ]
        )
        for born, laurent in zip(result.born_values, result.values, strict=True)
    ]
    assert moved == in_place
