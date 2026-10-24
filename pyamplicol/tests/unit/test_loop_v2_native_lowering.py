# SPDX-License-Identifier: 0BSD
"""The native plan model: byte-identical plan JSON and the binary plan member
(WP3.1 package D2, ``loops/v2/native_lowering.py``).

The native model is fed with an exported plan's records (sorted keys, spaced
separators: the writer must restore the exporter's order and compact form).
Per plan:

* the native JSON writer gives ``ExportedPlan.json()`` byte for byte;
* the binary member decodes to the same model (its JSON again identical, its
  re-encoding bitwise), loads to the same executor tables as the JSON (digest
  of the loaded program, derived row links included) and holds no host path
  (searched in its decoded strings, where a path would show);
* strings follow ``json.dumps``'s ``ensure_ascii`` escapes;
* foreign, truncated or extended members and unknown JSON fields are refused
  (crafted column counts: the reader's Rust unit tests);
* the bindings run in a process without a Symbolica key (``loop_nightly``).

Plans cover schedule B with derived and written row links, schedule F
(closures, relabellings), schedule M (edge weights, closures, chunks), massive
quark lines (mass flips), helicity-resolved observable columns
(``column_rows``), the stored form with the tree program of native per-point
inputs (its Born kernels join the plan's), and the certified-scaleless
companion's plans that the "uv" and "ir" pole selections run (the slice's,
with and without its tree program, and the observable columns'). Default
tier: g g > t t~ and g g > g g; wider controls (the companion's tree-program
and column plans among them) are ``loop_physics``, six legs and the no-key
process ``loop_nightly``; seven-leg evidence is the stored S8 digest of
bounded runs (RUST-MIGRATION.md rule 6).
"""

from __future__ import annotations

import json
import os
import subprocess
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
    "gg_gg": ("g g > g g", {"QCD": 2, "QED": 0}, ()),
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


def host_paths(member: bytes, decoded: bytes) -> list[bytes]:
    """The host markers a member carries. Its strings are byte columns stored
    as offsets or differences, so a path never shows in the raw bytes: they
    are searched in the member's decoded JSON (``decoded``), which holds every
    string the member carries (the member is a function of that model, its
    round trip being bitwise)."""

    return [m for m in _host_markers() if m in decoded or m in member]


def check(exported, native) -> bytes:
    """The native JSON and the binary member of one exported plan."""

    text = exported.json().encode()
    model = native_lowering.plan_model(exported, native)
    assert bytes(model.json()) == text
    member = bytes(model.binary())
    assert member.startswith(b"LOOPPLAN")
    again = native.V2PlanModel.from_binary(member)
    decoded = bytes(again.json())
    assert decoded == text
    assert bytes(again.binary()) == member
    json_load, binary_load = (
        native_lowering.load(data, native, digest=True) for data in (text, member)
    )
    assert (json_load.form, binary_load.form) == ("json", "binary")
    assert binary_load.digest == json_load.digest
    assert not host_paths(member, decoded)
    return member


def check_companion_plans(slice_, native, resolved=None) -> None:
    """The plans the "uv" and "ir" pole selections run: the certified-scaleless
    companion's plan (``LoopSlice.scaleless``, stored beside the main plan by
    ``artifact.generate``) and its tree-program form (``NativeExecutor.scaleless``
    with native inputs), and, for ``resolved`` observable columns
    (``ColumnPlan``), the companion's column plan (``ColumnPlan.scaleless``,
    ``observables.NativeObservables``)."""

    companion = slice_.scaleless
    assert companion is not None
    check(export_plan(companion, forward=False), native)
    check(
        export_plan(
            companion,
            forward=False,
            trees=True,
            born_squared=False,
            counterterms=False,
        ),
        native,
    )
    if resolved is not None:
        twin = resolved.scaleless()
        assert twin is not None
        check(export_plan(twin.slice, forward=False, columns=twin.seed_columns), native)


@pytest.mark.loop_fast
def test_massive_plan_forward_links_and_resolved_columns(slices, loop_native):
    """g g > t t~ (mass flips): schedule F; schedule B with derived and with
    written row links (the loader checks those against its derivation); the
    observables' helicity-resolved columns (``column_rows``); the companion's
    plan (its tree-program and column plans: ``loop_physics``)."""

    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices("gg_ttx")
    check(export_plan(slice_), loop_native)
    check(export_plan(slice_, forward=False, links=True), loop_native)
    observables = ObservableSlice.of(slice_)
    plan = ColumnPlan(observables, observables.resolved_columns("helicity-colour", {}))
    resolved = export_plan(slice_, forward=False, columns=plan.seed_columns)
    assert any(rows is not None for rows in resolved.program["column_rows"])
    check(resolved, loop_native)
    assert slice_.scaleless is not None
    check(export_plan(slice_.scaleless, forward=False), loop_native)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("gg_gg", marks=pytest.mark.loop_fast),
        pytest.param("gg_ggg", marks=pytest.mark.loop_physics),
    ],
)
def test_merged_plan(slices, loop_native, name):
    """Pure gluons: schedule M's section (entries, weighted edges, closures,
    chunks) beside schedule B; the certified-scaleless companion's plan."""

    slice_ = slices(name)
    merged = export_plan(slice_, forward=False, merged=merged_graph(slice_))
    section = merged.program["merged"]
    assert section["entries"] and section["chunks"]
    states = section["states"]
    assert any(state["closure"] is not None for state in states)
    assert any(
        edge.get("weight") is not None for state in states for edge in state["edges"]
    )
    check(merged, loop_native)
    assert slice_.scaleless is not None
    check(export_plan(slice_.scaleless, forward=False), loop_native)


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
    decoded = bytes(loop_native.V2PlanModel.from_binary(member).json())
    assert decoded == expected
    # A host path in a string is found in the decoded strings, not in the
    # member's raw bytes.
    marker = _host_markers()[0]
    leaves[len(texts)]["weight"] = marker.decode()
    model = loop_native.V2PlanModel.from_json(json.dumps(program).encode())
    member = bytes(model.binary())
    assert marker not in member
    decoded = bytes(loop_native.V2PlanModel.from_binary(member).json())
    assert marker in host_paths(member, decoded)


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
    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices(name)
    check(export_plan(slice_), loop_native)
    check(export_plan(slice_, forward=False, links=True), loop_native)
    # The stored main plan: its tree program and Born metric (the native
    # per-point inputs, ``artifact.generate``); its Born kernels join the
    # plan's.
    check(
        export_plan(slice_, forward=False, trees=True, born_squared=True), loop_native
    )
    observables = ObservableSlice.of(slice_)
    plan = ColumnPlan(observables, observables.resolved_columns("helicity-colour", {}))
    check(export_plan(slice_, forward=False, columns=plan.seed_columns), loop_native)
    check_companion_plans(slice_, loop_native, plan)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_gggg", "uux_gggg"])
def test_six_leg_plans(slices, loop_native, name):
    """The stored form at six legs (``forward=False``), schedule M and the
    stored and companion plans."""

    slice_ = slices(name)
    check(export_plan(slice_, forward=False), loop_native)
    check(export_plan(slice_, forward=False, merged=merged_graph(slice_)), loop_native)
    check(
        export_plan(slice_, forward=False, trees=True, born_squared=True), loop_native
    )
    check_companion_plans(slice_, loop_native)


@pytest.mark.loop_nightly
def test_plan_bindings_need_no_symbolica_licence(slices, loop_native, tmp_path):
    """``V2PlanModel`` and ``v2_plan_load`` in a process without a key
    (RUST-MIGRATION.md rule 4): the JSON written byte-identical, the binary
    round trip bitwise, and equal load digests from both encodings.

    The child imports the extension, which starts a restricted Symbolica
    instance: like ``test_native_builder_needs_no_symbolica_licence`` it must
    not overlap another unlicensed Symbolica process of the user.
    """

    exported = export_plan(slices("gg_ttx"), forward=False)
    text = exported.json().encode()
    member = native_lowering.plan_binary(exported, loop_native)
    expected = native_lowering.load(text, loop_native, digest=True).digest
    records = tmp_path / "records.json"
    records.write_text(json.dumps(exported.program, sort_keys=True))
    written = tmp_path / "written.json", tmp_path / "written.bin"
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from pyamplicol.loops.native import load_loop_native\n"
        "native = load_loop_native()\n"
        "model = native.V2PlanModel.from_json(Path(sys.argv[1]).read_bytes())\n"
        "text, member = bytes(model.json()), bytes(model.binary())\n"
        "again = native.V2PlanModel.from_binary(member)\n"
        "assert bytes(again.json()) == text and bytes(again.binary()) == member\n"
        "Path(sys.argv[2]).write_bytes(text)\n"
        "Path(sys.argv[3]).write_bytes(member)\n"
        "for data in (text, member):\n"
        "    print(native.v2_plan_load(data, digest=True)[3])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(records), *map(str, written)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert written[0].read_bytes() == text
    assert written[1].read_bytes() == member
    assert result.stdout.split()[-2:] == [expected, expected]
