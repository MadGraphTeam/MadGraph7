# SPDX-License-Identifier: 0BSD
"""The native plan model: byte-identical plan JSON and the binary plan member
(WP3.1 package D2, ``loops/v2/native_lowering.py``).

The native model is fed with an exported plan's records (sorted keys, spaced
separators: the writer must restore the exporter's order and compact form).
Per plan:

* the native JSON writer gives ``ExportedPlan.json()`` byte for byte;
* the binary member decodes to the same model (its JSON again identical, its
  re-encoding bitwise), loads to the same executor tables as the JSON (digest
  of the loaded program, derived row links included), both through the timed
  load and through ``read_binary``, the member's one production reader, and
  holds no host path (searched in its decoded strings, where a path would
  show);
* strings follow ``json.dumps``'s ``ensure_ascii`` escapes;
* foreign members and unknown JSON fields are refused, and every truncation,
  every single-byte corruption and crafted decompression bombs (width-0
  columns, shared lists copied at every reference, strings, declared sizes)
  fail closed: an error or a model within the member's declared bound, never
  an abort or a memory exhaustion (in a child under an address-space limit,
  so a regression fails instead of exhausting the host; the reader's Rust
  unit tests add the padding of plans that share beyond the bound);
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

The electroweak control vetoes the top through ``SliceRequest``'s
``ignored_particle_ids``, which ct-simplify renames ``particle_veto``: the
rename lands here in phase 1, when ct-simplify reaches this branch with
G-int (it is not on the pinned generation-scale base).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from pyamplicol.loops.v2 import native_lowering
from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.merged import merged_graph
from pyamplicol.loops.v2.plan_export import PLAN_SCHEMA, export_plan

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
# A plan without the tree program holds no counterterm data, so it is the same
# under both counterterm forms (RUST-MIGRATION.md rule 6; checked for every
# export of these controls): they take the cheaper site closure. The controls
# exported with their tree program (counterterm components) keep the
# production route.
SITES = {"gg_ttx", "gg_gg", "gg_ggg"}


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, LoopSlice] = {}

    def build(name: str) -> LoopSlice:
        if name not in built:
            process, born, ignored = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            options: dict[str, object] = {}
            if ignored:  # only the electroweak control vetoes a particle
                options["ignored_particle_ids"] = ignored
            if name in SITES:
                options["counterterms"] = "sites"
            request = SliceRequest(
                process, born, loop, fixed_parameters={"MB": "0"}, **options
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
    assert native_lowering.read(member, native) == json_load.digest
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
def test_foreign_and_damaged_members_are_refused(slices, loop_native, tmp_path):
    """Foreign formats and schemas, extended members and unknown JSON fields
    are refused; truncations, single-byte corruptions and crafted
    decompression bombs fail closed (``_fail_closed``)."""

    exported = export_plan(slices("gg_ttx"), forward=False)
    member = native_lowering.plan_binary(exported, loop_native)
    damaged = [
        member + bytes(8),
        b"LOOPPLAX" + member[8:],
        member[:8] + (1).to_bytes(4, "little") + member[12:],
        member[:8] + (3).to_bytes(4, "little") + member[12:],
        member[:12] + (3).to_bytes(4, "little") + member[16:],
    ]
    for data in damaged:
        with pytest.raises(ValueError):
            loop_native.V2PlanModel.from_binary(data)
    program = json.loads(exported.json())
    program["families"][0]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        loop_native.V2PlanModel.from_json(json.dumps(program).encode())
    _fail_closed(loop_native, member, tmp_path)


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


# The member's header (``binary.rs``): magic, format, schema, column count,
# reserved, the declared decoded size and the padding length; then per column
# a 24-byte header (width, delta flag, zeros, count, base) and its payload.
HEADER = 40
DECODED_PER_BYTE = 128


def _columns(member: bytes, names: list[str]) -> dict[str, int]:
    """The header offset of every column of a member."""

    offsets, at = {}, HEADER
    for name in names:
        offsets[name] = at
        size = member[at] * struct.unpack_from("<Q", member, at + 8)[0]
        at += 24 + size + (8 - size % 8) % 8
    assert at + struct.unpack_from("<Q", member, 32)[0] == len(member)
    return offsets


def _empty_plan() -> dict[str, object]:
    counts = ("legs", "rows", "columns", "born_components", "constants", "masses")
    lists = ("helicities", "block_rows", "kernels", "contents", "factors")
    lists += ("seed_classes", "seeds", "families")
    empty: dict[str, object] = {"schema": PLAN_SCHEMA, "average": 1}
    empty |= {key: 0 for key in counts} | {key: [] for key in lists}
    return empty | {"forward": None, "merged": None}


def _wide_member(native) -> bytes:
    """A member whose seed classes and seed keys are long enough for the
    reader's parallel decode (``binary.rs``, "Parallel decoding")."""

    plan = _empty_plan()
    plan["seed_classes"] = [
        {"columns": [[[k % 3, f"{k}/7", "0"]] * (k % 4)]} for k in range(400)
    ]
    plan["seeds"] = [{"classes": [[k % 5, "1", f"-{k}"]] * (k % 3)} for k in range(500)]
    return bytes(native.V2PlanModel.from_json(json.dumps(plan).encode()).binary())


def _crafted_members(native) -> dict[str, bytes]:
    """Decompression bombs of a few kB, rewritten from the empty plan's member
    (width-0 columns hold no payload), each declaring the most decoded bytes
    its length allows: 10^6 contents of 10^4 rows shared (the evidence audit's
    2.4e11 bytes) and 10^5 x 10^5; 2^40 contents; one string of 2^40 bytes;
    2^40 strings; 2^40 seed keys (a parallel section); and a declaration
    beyond the bound."""

    empty = _empty_plan()
    member = bytes(native.V2PlanModel.from_json(json.dumps(empty).encode()).binary())
    names = native.V2PlanModel.columns()
    assert struct.unpack_from("<I", member, 16)[0] == len(names)
    offsets = _columns(member, names)

    def craft(*columns: tuple[str, int, int], declared: int | None = None) -> bytes:
        crafted = bytearray(member)
        for name, count, base in columns:
            assert crafted[offsets[name]] == 0, name  # width 0: no payload
            struct.pack_into("<Qq", crafted, offsets[name] + 8, count, base)
        most = DECODED_PER_BYTE * len(crafted)
        struct.pack_into("<Q", crafted, 24, most if declared is None else declared)
        return bytes(crafted)

    def shared_rows(contents: int, rows: int) -> bytes:
        per_content = ("ContentDIn", "ContentDOut", "ContentDegree", "ContentRows")
        per_content += ("ContentMask", "RowHelicityRef", "FusedRef", "TermLen")
        return craft(
            ("ContentLen", 1, contents),
            *((name, contents, 0) for name in per_content),
            ("RowHelicityLen", 1, rows),
            ("RowHelicityLegs", rows, 0),
            ("FusedLen", 1, 0),
        )

    return {
        "shared 1e6 x 1e4": shared_rows(10**6, 10**4),
        "shared 1e5 x 1e5": shared_rows(10**5, 10**5),
        "shared 2e3 x 30": shared_rows(2000, 30),
        "contents 2^40": craft(("ContentLen", 1, 2**40), ("Chunk", 2**40, 0)),
        "string 2^40 bytes": craft(
            ("StringLen", 1, 2**40), ("StringByte", 2**40, ord("1"))
        ),
        "strings 2^40": craft(("StringLen", 2**40, 0)),
        "seed keys 2^40": craft(
            ("SeedKeyLen", 1, 2**40), ("SeedKeyClassLen", 2**40, 0)
        ),
        "declared beyond the bound": craft(declared=DECODED_PER_BYTE * len(member) + 1),
    }


_FAIL_CLOSED = """
import hashlib, json, resource, sys, time
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
model = load_loop_native().V2PlanModel
member = Path(sys.argv[1]).read_bytes()
crafted = {p.stem: p.read_bytes() for p in Path(sys.argv[2]).iterdir()}
wide = Path(sys.argv[3]).read_bytes()
digests = [hashlib.sha256(bytes(model.from_binary(m).json())).hexdigest()
           for m in (member, wide)]
status = Path('/proc/self/status').read_text().splitlines()
size = next(int(l.split()[1]) for l in status if l.startswith('VmSize:'))
limit = size * 1024 + 2**30
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
out = {'truncated': [], 'corrupted': [0, 0], 'crafted': {}, 'digests': digests}
start = time.perf_counter()
for length in range(len(member)):
    try:
        model.from_binary(member[:length])
        out['truncated'].append(length)
    except ValueError:
        pass
for at in range(len(member)):
    damaged = bytearray(member)
    damaged[at] ^= 0xFF
    try:
        model.from_binary(bytes(damaged))
        out['corrupted'][1] += 1
    except ValueError:
        out['corrupted'][0] += 1
for name, data in crafted.items():
    try:
        model.from_binary(data)
        out['crafted'][name] = 'DECODED'
    except ValueError as error:
        out['crafted'][name] = str(error)
out['seconds'] = time.perf_counter() - start
print(json.dumps(out))
"""


def _fail_closed(native, member: bytes, tmp_path: Path) -> None:
    """Every truncation of ``member`` is refused, every single-byte corruption
    is refused or decodes (a damaged value can be another valid one; the
    container's SHA-256 detects damage), and every crafted bomb is refused at
    its declaration or at its first charge beyond it, before allocating. In a
    child under an address-space limit of 1 GiB above its size: a reader that
    decoded a bomb would fail there instead of exhausting the host. The child
    decodes with one thread (sequentially), this process with its pool (the
    wide member in parallel chunks): equal models (RUST-MIGRATION.md rule 5).
    """

    crafted = tmp_path / "crafted"
    crafted.mkdir()
    bombs = _crafted_members(native)
    for k, data in enumerate(bombs.values()):
        assert len(data) < 8192
        (crafted / f"{k}.bin").write_bytes(data)
    stored, wide = tmp_path / "member.bin", tmp_path / "wide.bin"
    stored.write_bytes(member)
    wide.write_bytes(_wide_member(native))
    result = subprocess.run(
        [sys.executable, "-c", _FAIL_CLOSED, str(stored), str(crafted), str(wide)],
        env=os.environ | {"RAYON_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    out = json.loads(result.stdout.splitlines()[-1])
    here = [
        hashlib.sha256(bytes(native.V2PlanModel.from_binary(data).json())).hexdigest()
        for data in (member, wide.read_bytes())
    ]
    assert out["digests"] == here
    assert out["truncated"] == []
    assert sum(out["corrupted"]) == len(member) and out["corrupted"][0] > 0
    for k, name in enumerate(bombs):
        error = out["crafted"][str(k)]
        expected = "bound" if name == "declared beyond the bound" else "declared size"
        assert expected in error, (name, error)
