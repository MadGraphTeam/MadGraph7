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
* foreign members and unknown JSON fields are refused, and truncations,
  single-byte corruptions and crafted decompression bombs (width-0 columns,
  shared lists copied at every reference, strings, declared sizes) fail
  closed: an error or a model within the member's declared bound, never an
  abort or a memory exhaustion (in a child under an address-space limit, so
  a regression fails instead of exhausting the host; the reader's Rust unit
  tests add every truncation, the parallel decode's corruptions and the
  padding of plans that share beyond the bound); the parallel decode equals
  the sequential one;
* crafted plans whose loader derivations would exhaust or crash the process
  (2^26 Born rows, 2^16 columns without seed classes, a 2^40-th vertex, a
  family rank of 10^6, a fused exponent of 2^20, a power of ten of 10^9)
  are refused by ``read_binary`` before deriving: unpadded, beyond the
  member's derivation bound; padded, by the fallible probe under the child's
  address-space limit; valid list-dominated members (2^17 fused entries, 64
  contents sharing 4096) load or are refused under a scan of tighter limits,
  never abort; every plan's derivation bound (``derive_bound``) is within its
  member's, and no plan written today needs padding (the bound's soundness
  against the loader's real allocations: the reader's Rust tests,
  ``tests/loop_v2_plan_derive_bound.rs``);
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

The electroweak control vetoes the top through ``SliceRequest``'s public
``particle_veto`` (REQUIREMENTS_ADDENDUM J).
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
            process, born, veto = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            options: dict[str, object] = {}
            if veto:  # only the electroweak control vetoes a particle
                options["particle_veto"] = veto
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
    assert struct.unpack_from("<Q", member, 32)[0] == 0, "no plan needs padding"
    memory, allocated, work = again.derive_bound()
    assert 0 < memory <= allocated
    assert allocated + work <= DERIVED_PER_BYTE * len(member)
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
    _fail_closed(loop_native, member, json.loads(exported.json()), tmp_path)


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
    """``V2PlanModel``, ``v2_plan_load`` and ``v2_plan_read`` in a process
    without a key (RUST-MIGRATION.md rule 4): the JSON written byte-identical,
    the binary round trip bitwise, the derivation bound (``derive_bound``),
    and equal load digests from both encodings and from ``read_binary``.

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
        "assert all(isinstance(x, int) and x > 0 for x in again.derive_bound())\n"
        "Path(sys.argv[2]).write_bytes(text)\n"
        "Path(sys.argv[3]).write_bytes(member)\n"
        "for data in (text, member):\n"
        "    print(native.v2_plan_load(data, digest=True)[3])\n"
        "print(native.v2_plan_read(member))\n"
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
    assert result.stdout.split()[-3:] == [expected] * 3


# The member's header (``binary.rs``): magic, format, schema, column count,
# reserved, the declared decoded size and the padding length; then per column
# a 24-byte header (width, delta flag, zeros, count, base) and its payload.
HEADER = 40
DECODED_PER_BYTE = 128
DERIVED_PER_BYTE = 8192


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


def _crafted_members(native) -> dict[str, tuple[bytes, str]]:
    """Decompression bombs of a few kB, rewritten from the empty plan's member
    (width-0 columns hold no payload), each declaring the most decoded bytes
    its length allows, with the charge that refuses it: n contents sharing one
    list of m empty rows, refused at the contents' own list for n = 10^6, 10^5
    (the evidence audit's 2.4e11 bytes at m = 10^4) and at the shared list's
    copies (``RowHelicityRef``) for few contents of long lists; 2^40 contents;
    one string of 2^40 bytes; 2^40 strings; 2^40 seed keys (a parallel
    section); and a declaration beyond the bound."""

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

    beyond = "decodes beyond the member's declared size"
    return {
        "shared 1e6 x 1e4": (shared_rows(10**6, 10**4), f"ContentLen {beyond}"),
        "shared 1e5 x 1e5": (shared_rows(10**5, 10**5), f"ContentLen {beyond}"),
        "shared 2e3 x 30": (shared_rows(2000, 30), f"RowHelicityRef {beyond}"),
        "shared 3e3 x 2e3": (shared_rows(3000, 2000), f"RowHelicityRef {beyond}"),
        "shared 1e2 x 2e4": (shared_rows(100, 20000), f"RowHelicityRef {beyond}"),
        "contents 2^40": (
            craft(("ContentLen", 1, 2**40), ("Chunk", 2**40, 0)),
            beyond,
        ),
        "string 2^40 bytes": (
            craft(("StringLen", 1, 2**40), ("StringByte", 2**40, ord("1"))),
            beyond,
        ),
        "strings 2^40": (craft(("StringLen", 2**40, 0)), beyond),
        "seed keys 2^40": (
            craft(("SeedKeyLen", 1, 2**40), ("SeedKeyClassLen", 2**40, 0)),
            beyond,
        ),
        "declared beyond the bound": (
            craft(declared=DECODED_PER_BYTE * len(member) + 1),
            "declared decoded size beyond the member's bound",
        ),
    }


def _unpadded(member: bytes) -> bytes:
    """The member without its padding (and the padding length 0)."""

    padding = struct.unpack_from("<Q", member, 32)[0]
    cut = bytearray(member[: len(member) - padding])
    struct.pack_into("<Q", cut, 32, 0)
    return bytes(cut)


def _derivation_bombs(native, program: dict) -> dict[str, tuple[bytes, str]]:
    """Plans whose loader derivations (``LoopProgramV4::from_file``) would
    take far beyond the child's address space, from the g g > t t~ plan
    ``program``, with the refusal ``read_binary`` gives each.

    - 2^16 observable columns without seed classes (the evidence audit's
      reproducer, at 4096 on g g > g g g): every row link expanded per column;
    - 2^26 Born rows (26 legs of two helicities) for one family's one edge;
    - a term's vertex id 2^40 (the vertex table) and a family rank of 10^6
      (monomial tables): refused as the loader's validation would, before it
      allocates;
    - a fused exponent of 2^20 (the evidence audit's 4,240-byte crash: the
      layouts' recursive monomial index overflowed the stack before the
      loader checks the content's degree): refused before deriving;
    - a leaf weight ``1e1000000000`` (a 0.4 GB power of ten, minutes to
      compute; its written member is padded by 26 MB to pay for it):
      unpadded, beyond the member's derivation bound.

    The first two as written (the writer pads them to pay for their
    derivations: refused by the fallible reservation under the child's
    limit) and unpadded (a few kB: beyond the member's derivation bound).
    """

    def member(plan: dict) -> bytes:
        text = json.dumps(plan, separators=(",", ":")).encode()
        return bytes(native.V2PlanModel.from_json(text).binary())

    columns = json.loads(json.dumps(program))
    columns["seed_classes"] = []
    for key in columns["seeds"]:
        key["classes"] = []
    columns["columns"] = 2**16
    rows = json.loads(json.dumps(program))
    legs = 26
    rows |= {"legs": legs, "rows": 2**legs, "helicities": [[-1, 1]] * legs}
    family = rows["families"][0]
    family["steps"] = family["steps"][:1]
    family["steps"][0]["edges"] = family["steps"][0]["edges"][:1]
    rows["families"] = [family]
    vertex = json.loads(json.dumps(program))
    term = next(t for c in vertex["contents"] for t in c["terms"] if t["vertex"] >= 0)
    term["vertex"] = 2**40
    rank = json.loads(json.dumps(program))
    rank["families"][0]["rank"] = 10**6
    exponent = json.loads(json.dumps(program))
    exponent["contents"][0]["fused"][0][5] = 2**20
    power = json.loads(json.dumps(program))
    power["families"][0]["leaves"][0]["weight"] = "1e1000000000"
    refused = "which the allocator refuses"
    bound = "the loader's derivations exceed the member's bound"
    bombs = {}
    for name, plan in (("columns 2^16", columns), ("rows 2^26", rows)):
        padded = member(plan)
        assert struct.unpack_from("<Q", padded, 32)[0] > 0, name
        bombs[f"{name} padded"] = (padded, refused)
        bombs[f"{name} unpadded"] = (_unpadded(padded), bound)
    bombs["vertex 2^40"] = (member(vertex), "a vertex id beyond the terms")
    bombs["family rank 1e6"] = (member(rank), "family rank exceeds propagators")
    bombs["fused exponent 2^20"] = (member(exponent), "q-degree above its content's")
    bombs["decimal power 1e9"] = (_unpadded(member(power)), bound)
    return bombs


_FAIL_CLOSED = """
import hashlib, json, resource, sys, time
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
native = load_loop_native()
model = native.V2PlanModel
member = Path(sys.argv[1]).read_bytes()
crafted = {p.stem: p.read_bytes() for p in Path(sys.argv[2]).iterdir()}
wide = Path(sys.argv[3]).read_bytes()
derived = {p.stem: p.read_bytes() for p in Path(sys.argv[4]).iterdir()}
digests = [hashlib.sha256(bytes(model.from_binary(m).json())).hexdigest()
           for m in (member, wide)]
status = Path('/proc/self/status').read_text().splitlines()
size = next(int(l.split()[1]) for l in status if l.startswith('VmSize:'))
limit = size * 1024 + 2**30
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
out = {'truncated': [], 'corrupted': [0, 0], 'crafted': {}, 'digests': digests}
start = time.perf_counter()
# Every byte of the header and of the first column headers, then every
# seventh byte (all offsets modulo eight); the reader's Rust unit tests take
# every truncation.
offsets = [*range(min(len(member), 256)), *range(256, len(member), 7)]
out['offsets'] = len(offsets)
for length in offsets:
    try:
        model.from_binary(member[:length])
        out['truncated'].append(length)
    except ValueError:
        pass
for data in (member, wide):
    for at in [*range(min(len(data), 256)), *range(256, len(data), 7)]:
        damaged = bytearray(data)
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
# The member's production reader (read_binary) under the limit: the stored
# plan loads; the derivation bombs are refused.
out['read'] = native.v2_plan_read(member)
out['derived'] = {}
for name, data in derived.items():
    try:
        native.v2_plan_load(data, digest=False)
        out['derived'][name] = 'LOADED'
    except ValueError as error:
        out['derived'][name] = str(error)
out['seconds'] = time.perf_counter() - start
print(json.dumps(out))
"""


# Address-space headrooms (MiB above a fresh child's size) the list-dominated
# members are read under, ascending until one loads (``_under_limits``); a
# decimal-power member steps by 1 MiB.
HEADROOMS = (4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192)

# The reader under each headroom without the digest (``program_digest``
# formats the program beyond the reader's bound: a diagnostic), the digest of
# the member that loads without a limit.
_UNDER_LIMITS = """
import json, resource, sys
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
native = load_loop_native()
member = Path(sys.argv[1]).read_bytes()
rows = []
for mb in json.loads(sys.argv[2]):
    status = Path('/proc/self/status').read_text().splitlines()
    size = next(int(l.split()[1]) for l in status if l.startswith('VmSize:'))
    limit = size * 1024 + mb * 2**20
    resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))
    try:
        native.v2_plan_load(member, digest=False)
        rows.append('LOADED')
    except ValueError as error:
        rows.append(str(error))
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY,) * 2)
    if 'allocator refuses' not in rows[-1]:
        break
if rows[-1] == 'LOADED':
    rows[-1] = native.v2_plan_read(member)
print(json.dumps(rows))
"""

# A thread pool that cannot start (64 threads' stacks beyond 64 MiB of
# headroom): a ValueError at every read, also once the limit is lifted (rayon
# never starts a global pool after a failed start), never rayon's panic.
_POOL_LIMIT = """
import json, resource, sys
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
native = load_loop_native()
member = Path(sys.argv[1]).read_bytes()
status = Path('/proc/self/status').read_text().splitlines()
size = next(int(l.split()[1]) for l in status if l.startswith('VmSize:'))
resource.setrlimit(resource.RLIMIT_AS, (size * 1024 + 2**26, resource.RLIM_INFINITY))
rows = []
for limited in (True, True, False):
    if not limited:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY,) * 2)
    try:
        native.v2_plan_load(member, digest=False)
        rows.append('LOADED')
    except ValueError as error:
        rows.append(str(error))
print(json.dumps(rows))
"""


def _children(script: str, *arguments: str, **env: str) -> subprocess.Popen:
    """A fresh child running ``script`` (started now, collected by
    ``_collect``: the fail-closed children run concurrently)."""

    return subprocess.Popen(
        [sys.executable, "-c", script, *arguments],
        env=os.environ | env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect(name: str, child: subprocess.Popen, timeout: int) -> object:
    """The JSON last line ``child`` prints (killed beyond ``timeout``)."""

    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise
    assert child.returncode == 0, (name, child.returncode, stderr[-2000:])
    return json.loads(stdout.splitlines()[-1])


def _under_limits(native, tmp_path: Path) -> dict[str, tuple[subprocess.Popen, bytes]]:
    """Valid members whose loader derivations follow their lists at several
    times their decoded size (the evidence audit's abort windows: a probe of
    twice the decoded size passed and the segment layouts' sets aborted; a
    decode used the address space to its edge and the thread pool's first
    use aborted): one content of 2^17 distinct fused entries, and 64 contents
    sharing one list of 4096 (stored once, copied per content); and a weight
    of a 10^5-digit mantissa over 10^(3 x 10^6) (GMP numbers, allocated by C
    ``malloc`` outside the counting allocator of the Rust bound tests: the
    power, the product, the normalising gcd and divisions; the address space
    they take against the bound is measured there,
    ``weights_fit_their_bound_in_address_space``). Each is read in a fresh
    child (one rayon thread, the pool not yet started; one malloc arena,
    since glibc's per-thread arenas reserve 64 MB of address space each,
    which a failing allocation falls back to: hidden headroom) under
    address-space limits above its size, ascending (``HEADROOMS``; the weight
    by 1 MiB): refused by a probe ("which the allocator refuses") until it
    loads, to the parent's digest, never aborting; refused at the smallest.
    Started here, collected by ``_fail_closed``."""

    def member(contents: int, entries: int, width: int) -> bytes:
        fused = [[i // width, i % width, 0, 0, 0, 0] for i in range(entries)]
        content = {"d_in": width, "d_out": width, "degree": 0, "rows": 0, "mask": 0}
        content |= {"row_helicities": [], "fused": fused, "terms": []}
        plan = _empty_plan() | {"legs": 1, "rows": 1, "columns": 1}
        plan |= {"helicities": [[1]], "contents": [content] * contents}
        text = json.dumps(plan, separators=(",", ":")).encode()
        return bytes(native.V2PlanModel.from_json(text).binary())

    weight = ("1234567890" * 10**4) + "e-3000000"
    plan = _empty_plan() | {"legs": 1, "rows": 1, "columns": 1, "helicities": [[1]]}
    plan |= {"born_components": 1, "seed_classes": [{"columns": [[[0, weight, "0"]]]}]}
    text = json.dumps(plan, separators=(",", ":")).encode()
    members = {
        "fused 2^17": (member(1, 2**17, 512), HEADROOMS),
        "shared 64 x 4096": (member(64, 4096, 64), HEADROOMS),
        "weight 1e5 digits e-3e6": (
            bytes(native.V2PlanModel.from_json(text).binary()),
            tuple(range(1, 97)),
        ),
    }
    children = {}
    for k, (name, (data, headrooms)) in enumerate(members.items()):
        path = tmp_path / f"limits-{k}.bin"
        path.write_bytes(data)
        child = _children(
            _UNDER_LIMITS,
            str(path),
            json.dumps(headrooms),
            RAYON_NUM_THREADS="1",
            MALLOC_ARENA_MAX="1",
        )
        children[name] = (child, data)
    path = tmp_path / "pool.bin"
    path.write_bytes(members["shared 64 x 4096"][0])
    children["pool"] = (_children(_POOL_LIMIT, str(path), RAYON_NUM_THREADS="64"), b"")
    return children


def _fail_closed(native, member: bytes, program: dict, tmp_path: Path) -> None:
    """Truncations of ``member`` are refused; single-byte corruptions of it
    and of the wide member are refused or decode (a damaged value can be
    another valid one; the container's SHA-256 detects damage), at every byte
    of the first 256 and every seventh after (the Rust unit tests take every
    truncation, and corruptions in parallel chunks); and every crafted bomb is
    refused at its declaration or at its first charge beyond it, before
    allocating. ``read_binary`` loads ``member`` (``program``'s) and refuses
    the derivation bombs (``_derivation_bombs``). In a child under an
    address-space limit of 1 GiB above its size: a reader that decoded or
    derived a bomb would fail there instead of exhausting the host; the
    list-dominated and multi-precision members under tighter limits and a
    thread pool that cannot start (``_under_limits``, children run
    concurrently with it). The child
    decodes with one thread (sequentially), this process with its pool (the
    wide member in parallel chunks): equal models (RUST-MIGRATION.md rule 5).
    """

    crafted = tmp_path / "crafted"
    crafted.mkdir()
    bombs = _crafted_members(native)
    for k, (data, _) in enumerate(bombs.values()):
        assert len(data) < 8192
        (crafted / f"{k}.bin").write_bytes(data)
    derived = tmp_path / "derived"
    derived.mkdir()
    derivation_bombs = _derivation_bombs(native, program)
    for k, (data, _) in enumerate(derivation_bombs.values()):
        (derived / f"{k}.bin").write_bytes(data)
    stored, wide = tmp_path / "member.bin", tmp_path / "wide.bin"
    stored.write_bytes(member)
    wide.write_bytes(_wide_member(native))
    arguments = [str(stored), str(crafted), str(wide), str(derived)]
    child = _children(_FAIL_CLOSED, *arguments, RAYON_NUM_THREADS="1")
    limits = _under_limits(native, tmp_path)
    out = _collect("fail closed", child, 300)
    assert isinstance(out, dict)
    here = [
        hashlib.sha256(bytes(native.V2PlanModel.from_binary(data).json())).hexdigest()
        for data in (member, wide.read_bytes())
    ]
    assert out["digests"] == here
    assert out["truncated"] == [] and out["offsets"] > 256
    assert out["corrupted"][0] > 0
    for k, (name, (_, expected)) in enumerate(bombs.items()):
        assert expected in out["crafted"][str(k)], (name, out["crafted"][str(k)])
    assert out["read"] == native_lowering.read(member, native)
    for k, (name, (data, expected)) in enumerate(derivation_bombs.items()):
        assert expected in out["derived"][str(k)], (name, out["derived"][str(k)])
        # Unpadded, a bomb is the size of the plan's own member (wider values).
        small = len(data) <= len(member) + 1024
        assert small or name.endswith(" padded"), (name, len(data))
    for name, (child, data) in limits.items():
        rows = _collect(name, child, 120)
        assert isinstance(rows, list)
        if name == "pool":
            assert all("thread pool cannot start" in r for r in rows), rows
            continue
        assert len(rows) > 1 and all("allocator refuses" in r for r in rows[:-1]), rows
        assert rows[-1] == native_lowering.read(data, native), (name, rows)
