# SPDX-License-Identifier: 0BSD
"""The binary plan member as the plan's stored form (M-FINAL MF-8G-1 item 5,
``native_lowering.BINARY_PLAN``).

A generated process stores its plan and companion plan as binary columnar
plan members (``ExportedPlan.stored()``; ``schedule/loop-plan-v4.bin.zlib``)
instead of the plan JSON, and every native executor loads the member
(``LoopProgramV4::load`` takes a member through ``read_binary``). The JSON
stays the oracle and the form of artifacts written before the member:

* the member decodes to the exporter's JSON byte for byte (``member_json``)
  and loads into the same program (``native_lowering.read``);
* an executor on the member and one on the JSON evaluate bitwise alike, with
  and without poles (the companion's "uv" too);
* a stored process holds the members (no plan JSON), its ``plan_sha256``
  digests the member, and the loaded process evaluates bitwise as generated;
  a process stored with JSON plans (``BINARY_PLAN`` off) still loads and
  evaluates alike.

Default tier: g g > g g at leading colour and u u~ > d d~ g at full colour.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.plan_export import NativeExecutor

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}

PROCESSES = {
    "gg_gg_lc": ("g g > g g", {"QCD": 2, "QED": 0}, "lc", "legacy/gg_gg.json"),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        "full",
        "required/uux_ddxg.json",
    ),
}


def _points(source: str, count: int) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"][:count]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu[:count]]


def _slice(name, loop_sm_compiled_model, loop_native) -> LoopSlice:
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    process, born, accuracy, _source = PROCESSES[name]
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    request = SliceRequest(
        process,
        born,
        loop,
        accuracy=accuracy,
        fixed_parameters=FIXED,
        colour_declaration=None if accuracy == "full" else declaration,
    )
    return LoopSlice.generate(compiled, request, loop_native)


def _values(results) -> list:
    # Lane labels differ by design between a slice's executor and a loaded
    # process's (family indices): the values are compared.
    return [(r.born, r.components, r.totals) for r in results]


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ["gg_gg_lc", "uux_ddxg"])
def test_binary_plan_member_is_the_stored_form(
    loop_sm_compiled_model, loop_native, monkeypatch, tmp_path, name
):
    from pyamplicol.runtime.pacbin import PacbinReader

    from pyamplicol.loops.v2 import native_lowering
    from pyamplicol.loops.v2.artifact import (
        PLAN_BINARY_MEMBER,
        PLAN_MEMBER,
        SCALELESS_PLAN_BINARY_MEMBER,
        SCALELESS_PLAN_MEMBER,
        plan_digest,
        read_native_process,
        read_plan,
        read_plan_json,
        write_plan,
    )
    from pyamplicol.loops.v2.native_process import NativeProcess

    monkeypatch.setattr(native_lowering, "BINARY_PLAN", True)
    slice_ = _slice(name, loop_sm_compiled_model, loop_native)
    binary = NativeExecutor(slice_, forward=False, inputs="native")
    exported = binary.exported
    plan = binary._plan_json
    text = exported.json()
    # The executor loads the member; it decodes to the exporter's JSON byte
    # for byte and loads into the same program.
    assert isinstance(plan, bytes) and native_lowering.is_member(plan)
    assert plan == exported.binary()
    assert native_lowering.member_json(plan) == text
    assert native_lowering.read(plan) == native_lowering.read(text.encode())
    oracle = NativeExecutor(
        slice_, forward=False, inputs="native", exported=exported, plan_json=text
    )
    points, scales = _points(PROCESSES[name][3], 2)
    selections = ["none", "all"] + (["uv"] if slice_.scaleless is not None else [])
    generated = {}
    for poles in selections:
        generated[poles] = binary.evaluate(
            points, scales, parameters=PARAMETERS, poles=poles
        )
        assert repr(generated[poles]) == repr(
            oracle.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        )
    # Stored: the members (no plan JSON), digested as stored.
    process = NativeProcess.of(binary)
    companion = process.scaleless
    assert process.plan == plan and process.plan_json == text
    if companion is not None:
        twin = binary.scaleless
        assert native_lowering.is_member(companion.plan)
        assert companion.plan == twin.exported.binary()
        assert companion.plan_json == twin.exported.json()
    path = tmp_path / "loop-runtime.pacbin"
    write_plan(
        process.plan, path, process, None if companion is None else companion.plan
    )
    with PacbinReader.open(path) as reader:
        names = {member.logical_path for member in reader.members}
    assert PLAN_BINARY_MEMBER in names and PLAN_MEMBER not in names
    assert (SCALELESS_PLAN_BINARY_MEMBER in names) == (companion is not None)
    assert SCALELESS_PLAN_MEMBER not in names
    stored = read_plan(path)
    assert stored == plan and plan_digest(stored) == hashlib.sha256(plan).hexdigest()
    assert read_plan_json(path) == text
    if companion is not None:
        assert read_plan(path, SCALELESS_PLAN_MEMBER) == companion.plan
    # The writer refuses point inputs of another plan form.
    with pytest.raises(ValueError, match="another plan"):
        write_plan(text, tmp_path / "other.pacbin", process)
    loaded = read_native_process(path, stored)
    assert loaded.plan == plan
    executor = loaded.executor(loop_native)
    for poles in selections:
        assert _values(
            executor.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        ) == _values(generated[poles])
    # The JSON form (``BINARY_PLAN`` off; artifacts before the member) is
    # stored, read and evaluated alike.
    monkeypatch.setattr(native_lowering, "BINARY_PLAN", False)
    assert exported.stored() == text
    legacy = dataclasses.replace(
        process,
        plan=text,
        scaleless=None
        if companion is None
        else dataclasses.replace(companion, plan=companion.plan_json),
    )
    old = tmp_path / "legacy.pacbin"
    write_plan(
        legacy.plan,
        old,
        legacy,
        None if legacy.scaleless is None else legacy.scaleless.plan,
    )
    with PacbinReader.open(old) as reader:
        names = {member.logical_path for member in reader.members}
    assert PLAN_MEMBER in names and PLAN_BINARY_MEMBER not in names
    assert read_plan(old) == text and plan_digest(text) == plan_digest(text.encode())
    executor = read_native_process(old, read_plan(old)).executor(loop_native)
    for poles in selections:
        assert _values(
            executor.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
        ) == _values(generated[poles])
