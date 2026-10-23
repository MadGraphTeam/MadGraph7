# SPDX-License-Identifier: 0BSD
"""The native plan model: byte-identical plan JSON and the binary plan member
(WP3.1 package D2, ``loops/v2/native_lowering.py``).

The native model is fed with an exported plan's records (sorted keys, spaced
separators: the writer must restore the exporter's order and compact form).
Per plan:

* the native JSON writer gives ``ExportedPlan.json()`` byte for byte;
* the binary member decodes to the same model (its JSON again identical, its
  re-encoding bitwise), loads to the same executor tables as the JSON (digest
  of the loaded program, derived row links included) and holds no path;
* strings follow ``json.dumps``'s ``ensure_ascii`` escapes;
* foreign, truncated or extended members and unknown JSON fields are refused.

Plans cover schedule B with derived and written row links, schedule F
(closures, relabellings), schedule M (edge weights, closures, chunks), massive
quark lines (mass flips) and helicity-resolved observable columns
(``column_rows``). Default tier: g g > t t~ and g g > g g g; wider controls are
``loop_physics``, six legs ``loop_nightly``; seven-leg evidence is the stored
S8 digest of bounded runs (RUST-MIGRATION.md rule 6).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import native_lowering
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.merged import merged_graph
from pyamplicol.loops.v2.plan_export import export_plan

PROCESSES = {
    "gg_ttx": ("g g > t t~", {"QCD": 2, "QED": 0}, ()),
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, ()),
    "uux_uuxg": ("u u~ > u u~ g", {"QCD": 3, "QED": 0}, ()),
    "gg_ttxg": ("g g > t t~ g", {"QCD": 3, "QED": 0}, ()),
    "uux_emepgg": ("u u~ > e- e+ g g", {"QCD": 2, "QED": 2}, (23,)),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, ()),
    "uux_gggg": ("u u~ > g g g g", {"QCD": 4, "QED": 0}, ()),
}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            process, born, ignored = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                fixed_parameters={"MB": "0"},
                ignored_particle_ids=ignored,
            )
            built[name] = LoopSlice.generate(compiled, request, loop_native)
        return built[name]

    return build


def _host_markers() -> list[bytes]:
    paths = {Path.cwd(), Path.home(), Path(sys.prefix), Path(tempfile.gettempdir())}
    return [str(path).encode() for path in paths]


def check(exported, native) -> bytes:
    """The native JSON and the binary member of one exported plan."""

    text = exported.json().encode()
    model = native_lowering.plan_model(exported, native)
    assert bytes(model.json()) == text
    member = bytes(model.binary())
    assert member.startswith(b"LOOPPLAN")
    again = native.V2PlanModel.from_binary(member)
    assert bytes(again.json()) == text
    assert bytes(again.binary()) == member
    json_load, binary_load = (
        native_lowering.load(data, native, digest=True) for data in (text, member)
    )
    assert (json_load.form, binary_load.form) == ("json", "binary")
    assert binary_load.digest == json_load.digest
    for marker in _host_markers():
        assert marker not in member
    return member


@pytest.mark.loop_fast
def test_massive_plan_forward_links_and_resolved_columns(slices, loop_native):
    """g g > t t~ (mass flips): schedule F; schedule B with derived and with
    written row links (the loader checks those against its derivation); the
    observables' helicity-resolved columns (``column_rows``)."""

    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices("gg_ttx")
    check(export_plan(slice_), loop_native)
    check(export_plan(slice_, forward=False, links=True), loop_native)
    observables = ObservableSlice.of(slice_)
    plan = ColumnPlan(observables, observables.resolved_columns("helicity-colour", {}))
    resolved = export_plan(slice_, forward=False, columns=plan.seed_columns)
    assert any(rows is not None for rows in resolved.program["column_rows"])
    check(resolved, loop_native)


@pytest.mark.loop_fast
def test_merged_plan(slices, loop_native):
    """g g > g g g: schedule M's section (entries, weighted edges, closures,
    chunks) beside schedule B."""

    slice_ = slices("gg_ggg")
    merged = export_plan(slice_, forward=False, merged=merged_graph(slice_))
    assert merged.program["merged"]["states"]
    check(merged, loop_native)


@pytest.mark.loop_fast
def test_strings_follow_ensure_ascii(slices, loop_native):
    """Leaf weights and kernel input kinds are strings: escapes, control
    characters, DEL, non-ASCII and astral characters as ``json.dumps``."""

    exported = export_plan(slices("gg_ttx"), forward=False)
    program = json.loads(exported.json())
    texts = ['"', "\\", "\n\r\t\b\f", "\x01\x1f\x7f", "\u00e9/~ ", "\U0001f600\u2028"]
    leaves = [leaf for family in program["families"] for leaf in family["leaves"]]
    for leaf, text in zip(leaves, texts, strict=False):
        leaf["weight"] = text
    records = json.dumps(program, sort_keys=True).encode()
    model = loop_native.V2PlanModel.from_json(records)
    expected = json.dumps(program, separators=(",", ":")).encode()
    assert bytes(model.json()) == expected
    member = bytes(model.binary())
    assert bytes(loop_native.V2PlanModel.from_binary(member).json()) == expected


@pytest.mark.loop_fast
def test_foreign_and_damaged_members_are_refused(slices, loop_native):
    exported = export_plan(slices("gg_ttx"), forward=False)
    member = native_lowering.plan_binary(exported, loop_native)
    damaged = [
        member[:-8],
        member + bytes(8),
        b"LOOPPLAX" + member[8:],
        member[:8] + (2).to_bytes(4, "little") + member[12:],
        member[:12] + (3).to_bytes(4, "little") + member[16:],
    ]
    for data in damaged:
        with pytest.raises(ValueError):
            loop_native.V2PlanModel.from_binary(data)
    program = json.loads(exported.json())
    program["families"][0]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        loop_native.V2PlanModel.from_json(json.dumps(program).encode())


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_uuxg", "gg_ttxg", "uux_emepgg"])
def test_quark_line_massive_and_electroweak_plans(slices, loop_native, name):
    slice_ = slices(name)
    check(export_plan(slice_), loop_native)
    check(export_plan(slice_, forward=False, links=True), loop_native)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg", "uux_gggg"])
def test_six_leg_plans(slices, loop_native, name):
    """The stored form at six legs (``forward=False``) and schedule M."""

    slice_ = slices(name)
    check(export_plan(slice_, forward=False), loop_native)
    check(export_plan(slice_, forward=False, merged=merged_graph(slice_)), loop_native)
