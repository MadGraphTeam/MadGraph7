# SPDX-License-Identifier: 0BSD
"""The native plan model: byte-identical plan JSON and the binary plan member
(WP3.1 package D2, D2' and D1; ``loops/v2/native_lowering.py``).

The native model is fed with an exported plan's records, read from the Python
objects (``V2PlanModel.from_program``, the route of ``ExportedPlan.json()``
and ``ExportedPlan.binary()``), and from a JSON with sorted keys and spaced
separators (the writer must restore the exporter's order and compact form).
Per plan:

* ``ExportedPlan.json()`` (the native writer, ``NATIVE_JSON``) and the native
  JSON from either route give ``json.dumps`` (``ExportedPlan.python_json()``)
  byte for byte; ``ExportedPlan.binary()`` is the model's member;
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
* the Python records convert only as the writer writes them back as
  ``json.dumps`` (a bool for a number, a float, a wrong-length entry, an
  unknown key, keys out of the writer's order, a missing key the writer
  writes, an absent option written as ``null`` included, and ``None`` under a
  key the writer omits are refused with the record's path; the keys it omits
  when absent may be absent);
* a read decodes on one thread under an address-space limit (the wide member
  with the default pool loads or is refused, never aborts), and a thread pool
  that cannot start (its start shown failed) leaves every read on one thread;
  these children read ``/proc`` and rely on Linux's ``RLIMIT_AS`` and
  ``RLIMIT_NPROC``: Linux only, reported as skipped elsewhere (and the pool's
  when the limit is not enforced, as for root);
* the leaf weights (``conventions.LeafWeights``, D1) give the full rule's
  sign on every reading of the main slice and of its companion, with its
  refusals;
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

The electroweak control vetoes the Z (``|PDG|`` 23) through
``SliceRequest``'s public ``particle_veto`` (REQUIREMENTS_ADDENDUM J).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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


@pytest.fixture(scope="module")
def stored(slices):
    """The stored form of the g g > t t~ plan (``forward=False``; read only)."""

    return export_plan(slices("gg_ttx"), forward=False)


# The fail-closed children read ``/proc/self/status`` and rely on Linux's
# ``RLIMIT_AS`` and ``RLIMIT_NPROC`` semantics.
LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the fail-closed children read /proc and rely on Linux's RLIMIT_AS "
    "and RLIMIT_NPROC",
)


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

    text = exported.python_json().encode()
    assert native_lowering.NATIVE_JSON
    assert exported.json().encode() == text
    model = native_lowering.plan_model(exported, native)
    assert bytes(model.json()) == text
    records = native_lowering.plan_model_from_json(exported, native)
    assert bytes(records.json()) == text
    member = bytes(model.binary())
    assert member.startswith(b"LOOPPLAN")
    assert exported.binary() == member
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
    if "column_weights" not in exported.program:
        # Columns sharing seed-weight columns store their weights once, while
        # the loader still expands the row links over every column: such a
        # member may be padded to pay for its derivations.
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


# The keys the exporter omits when their value is absent, by record (the
# writer's ``OMITTED_WHEN_ABSENT``); every other key it always writes.
OMITTED = {("", "column_rows"), ("", "column_weights")}
OMITTED |= {(".families[].steps[]", "rows")}
OMITTED |= {(".families[].steps[].edges[]", "links")}


def check_every_key_written(exported, native) -> int:
    """Each key of the first record of every kind (by path, list indices
    dropped) removed in turn: refused naming the key, unless the writer omits
    it when absent (then accepted and written as ``json.dumps`` writes the
    records). The number of keys tried."""

    program = json.loads(exported.python_json())
    first: dict[str, tuple] = {}

    def walk(value: object, kind: str, path: tuple) -> None:
        if isinstance(value, dict):
            first.setdefault(kind, path)
            for key, item in value.items():
                walk(item, f"{kind}.{key}", (*path, key))
        elif isinstance(value, list):
            for k, item in enumerate(value):
                walk(item, f"{kind}[]", (*path, k))

    walk(program, "", ())
    tried = 0
    for kind, path in first.items():
        record = program
        for step in path:
            record = record[step]
        items = list(record.items())
        for key, _value in items:
            del record[key]
            tried += 1
            try:
                if (kind, key) in OMITTED:
                    model = native.V2PlanModel.from_program(program)
                    expected = json.dumps(program, separators=(",", ":")).encode()
                    assert bytes(model.json()) == expected, (kind, key)
                else:
                    with pytest.raises(ValueError, match=f'missing field "{key}"'):
                        native.V2PlanModel.from_program(program)
            finally:
                record.clear()
                record.update(items)  # the key back in its place
    return tried


@pytest.mark.loop_fast
def test_massive_plan_links_and_resolved_columns(slices, stored, loop_native):
    """g g > t t~ (mass flips): schedule B, stored, with derived and with
    written row links (the loader checks those against its derivation); the
    observables' helicity-resolved columns (``column_rows``). Schedule F and
    the companion's plans (with its tree-program and column plans) are
    ``loop_physics`` (``test_quark_line_massive_and_electroweak_plans``); the
    default tier checks a companion's plan on g g > g g."""

    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices("gg_ttx")
    check(stored, loop_native)
    linked = export_plan(slice_, forward=False, links=True)
    check(linked, loop_native)
    # The keys the writer omits when absent (written here) may be absent.
    check_every_key_written(linked, loop_native)
    observables = ObservableSlice.of(slice_)
    plan = ColumnPlan(observables, observables.resolved_columns("helicity-colour", {}))
    resolved = export_plan(slice_, forward=False, columns=plan.seed_columns)
    assert any(rows is not None for rows in resolved.program["column_rows"])
    # Columns that differ only by their rows share a seed-weight column.
    weights = resolved.program["column_weights"]
    assert len(set(weights)) < len(weights) == len(plan.seed_columns)
    check(resolved, loop_native)
    check_every_key_written(resolved, loop_native)
    program = json.loads(resolved.python_json())
    program["column_weights"] = None
    with pytest.raises(ValueError, match=re.escape(".column_weights: None where")):
        loop_native.V2PlanModel.from_program(program)


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
    # Every key of schedule M's records is written (a merged state's closure
    # and an edge's weight as null when absent).
    check_every_key_written(merged, loop_native)
    assert slice_.scaleless is not None
    check(export_plan(slice_.scaleless, forward=False), loop_native)


@pytest.mark.loop_fast
def test_strings_follow_ensure_ascii(stored, loop_native):
    """Leaf weights and kernel input kinds are strings: escapes, control
    characters, DEL, non-ASCII and astral characters as ``json.dumps``."""

    program = json.loads(stored.json())
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
def test_program_records_convert_as_json_writes_them(stored, loop_native):
    """``V2PlanModel.from_program`` reads only what the writer writes back as
    ``json.dumps`` writes it: tuples as lists; a bool for a number, a number
    for a flag, a float, a too long or short entry, an unknown or non-str key,
    a missing key the writer writes (a value, or an absent option it writes
    as ``null``), keys out of the writer's order (a record's, the plan's
    reversed or sorted) and ``None`` under a key the writer omits when absent
    (``column_rows``, a step's ``rows``, an edge's ``links``) are refused, the
    error naming the record. The keys the writer omits when absent may be
    absent (``test_massive_plan_links_and_resolved_columns``)."""

    program = json.loads(stored.python_json())
    expected = stored.python_json().encode()
    tupled = json.loads(stored.python_json())
    tupled["factors"] = [tuple(factor) for factor in tupled["factors"]]
    model = loop_native.V2PlanModel.from_program(tupled)
    assert bytes(model.json()) == expected

    def refused(edit, match: str) -> None:
        crafted = json.loads(json.dumps(program))
        edit(crafted)
        with pytest.raises(ValueError, match=match):
            loop_native.V2PlanModel.from_program(crafted)

    def at(path: str, message: str) -> str:
        return re.escape(f"{path}: {message}")

    refused(lambda p: p.update(legs=True), at(".legs", "expected an integer"))
    refused(lambda p: p.update(average=2.0), at(".average", "expected an integer"))
    refused(
        lambda p: p["families"][0]["steps"][0].update(seeded=1),
        at(".families[0].steps[0].seeded", "expected a bool"),
    )
    refused(
        lambda p: p["contents"][0]["fused"][0].append(0),
        at(".contents[0].fused[0]", "1 more items"),
    )
    refused(lambda p: p["factors"][0].pop(), re.escape(".factors[0]: "))
    refused(lambda p: p["families"][0].update(extra=1), "unknown field")
    refused(
        lambda p: p["families"][0].pop("rank"),
        at(".families[0]", 'missing field "rank"'),
    )
    # A key the writer always writes is refused when missing, an absent
    # option (``None``, written as ``null``) included, and so is a key that
    # only another record's writer omits when absent (a content's ``rows``).
    assert program["forward"] is None and program["merged"] is None
    assert check_every_key_written(stored, loop_native) > 50
    refused(lambda p: p.pop("forward"), re.escape('missing field "forward"'))
    refused(lambda p: p.pop("merged"), re.escape('missing field "merged"'))
    refused(
        lambda p: p["contents"][0].pop("rows"),
        at(".contents[0]", 'missing field "rows"'),
    )
    refused(lambda p: p.update({1: 2}), "expected a str key")
    refused(
        lambda p: p["families"][0]["leaves"][0].update(weight=1),
        at(".families[0].leaves[0].weight", "expected a str"),
    )
    refused(lambda p: p.update(rows=2**64), "beyond 64 bits")
    order = "out of the writer's key order"
    with pytest.raises(ValueError, match=order):
        loop_native.V2PlanModel.from_program(dict(reversed(list(program.items()))))
    with pytest.raises(ValueError, match=order):
        loop_native.V2PlanModel.from_program(
            json.loads(json.dumps(program, sort_keys=True))
        )

    def reorder(record: dict, key: str) -> None:
        record[key] = record.pop(key)  # moved to the end

    refused(
        lambda p: reorder(p["families"][0], "rank"),
        re.escape('.families[0]: key "rank" after "steps": ' + order),
    )
    omitted = "None where the writer omits the key"
    refused(lambda p: p.update(column_rows=None), at(".column_rows", omitted))
    refused(
        lambda p: p["families"][0]["steps"][0].update(rows=None),
        at(".families[0].steps[0].rows", omitted),
    )
    refused(
        lambda p: p["families"][0]["steps"][0]["edges"][0].update(links=None),
        at(".families[0].steps[0].edges[0].links", omitted),
    )
    # Written, they follow the record's other keys as the exporter appends
    # them: accepted (``check`` of the written-links and resolved plans).


def _reading_outcome(sign, skeleton) -> object:
    try:
        return sign(skeleton)
    except (ValueError, KeyError, NotImplementedError) as error:
        return (type(error).__name__, str(error))


def check_leaf_weights(slice_, weights=None) -> int:
    """The leaf weights' sign on every reading of the slice and of its
    companion equals the full rule's (``LeafWeights._fermion_sign``),
    refusals included (asked twice: a refusal is not cached); ``weights``
    replaces the slice's (fresh, so the cache is the one under test).
    Returns how many readings took the fast path."""

    from pyamplicol.loops.v2.conventions import _FERMIONIC, LeafWeights
    from pyamplicol.loops.v2.reference_executor import reading_skeletons

    if weights is None:
        own = slice_.weights
        weights = LeafWeights.__new__(LeafWeights)
        weights.__dict__.update(own.__dict__)
        weights._signs, weights._kinds = {}, {}
    plans = [slice_.plan]
    if slice_.scaleless is not None:
        plans.append(slice_.scaleless.plan)
    fast = 0
    for plan in plans:
        for skeleton in reading_skeletons(plan):
            expected = _reading_outcome(weights._fermion_sign, skeleton)
            assert _reading_outcome(weights.fermion_sign, skeleton) == expected
            assert _reading_outcome(weights.fermion_sign, skeleton) == expected
            kinds = [weights._kinds.get(c) for c in skeleton.contents]
            fast += _FERMIONIC not in kinds
    return fast


@pytest.mark.loop_fast
def test_leaf_weights_equal_the_full_rule(slices):
    """D1: one sign per distinct contents tuple, and the fast path for
    contents without a fermion line, equal the full rule on every reading
    (g g > t t~: the top line through the loop, top loops, gluon and ghost
    loops; g g > g g: quark loops beside gluon and ghost loops), the fast
    path taken; the leaf weight of an exported leaf is unchanged. With the
    coloured fermion ends hidden, the g g > t t~ readings with the top line
    through the loop are refused, by both alike, at every call."""

    from pyamplicol.loops.v2.conventions import LeafWeights
    from pyamplicol.loops.v2.reference_executor import reading_skeletons

    tops = slices("gg_ttx")
    check_leaf_weights(tops)  # every reading has the top line or a top loop
    gluons = slices("gg_gg")
    assert check_leaf_weights(gluons) > 0
    skeleton, _ring = next(iter(gluons.plan.schedules.values())).trie.leaves[0]
    sign = gluons.weights._fermion_sign(skeleton)
    weight = gluons.weights(skeleton, skeleton.contents)
    assert weight == skeleton.weight * skeleton.multiplicity * sign
    blind = LeafWeights.__new__(LeafWeights)
    blind.__dict__.update(tops.weights.__dict__)
    blind._ends, blind._end_mask, blind._signs, blind._kinds = {}, 0, {}, {}
    check_leaf_weights(tops, blind)
    refusals = sum(
        isinstance(_reading_outcome(blind.fermion_sign, s), tuple)
        for s in reading_skeletons(tops.plan)
    )
    assert refusals > 0


@pytest.mark.loop_fast
def test_foreign_and_damaged_members_are_refused(stored, loop_native):
    """Foreign formats and schemas, extended members and unknown JSON fields
    are refused."""

    member = native_lowering.plan_binary(stored, loop_native)
    (form,) = struct.unpack_from("<I", member, 8)  # the member's format
    damaged = [
        member + bytes(8),
        b"LOOPPLAX" + member[8:],
        member[:8] + (form - 1).to_bytes(4, "little") + member[12:],
        member[:8] + (form + 1).to_bytes(4, "little") + member[12:],
        member[:12] + (3).to_bytes(4, "little") + member[16:],
    ]
    for data in damaged:
        with pytest.raises(ValueError):
            loop_native.V2PlanModel.from_binary(data)
    program = json.loads(stored.json())
    program["families"][0]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        loop_native.V2PlanModel.from_json(json.dumps(program).encode())


@pytest.mark.loop_fast
@LINUX_ONLY
def test_readers_fail_closed_under_limits(stored, loop_native, tmp_path):
    """Truncations, single-byte corruptions and crafted decompression and
    derivation bombs fail closed, and valid list-dominated members load or
    are refused under tight address-space limits, never aborting
    (``_fail_closed``, in children under ``RLIMIT_AS``)."""

    member = native_lowering.plan_binary(stored, loop_native)
    _fail_closed(loop_native, member, json.loads(stored.python_json()), tmp_path)


@pytest.mark.loop_fast
@LINUX_ONLY
def test_a_pool_that_cannot_start_leaves_reads_on_one_thread(
    stored, loop_native, tmp_path
):
    """A thread pool that cannot start (``_POOL_LIMIT``: no thread may be
    spawned): its start is shown failed (``v2_plan_read_pool``), every read
    decodes on one thread to the parent's digest, also once the limit is
    lifted, never a refusal or rayon's panic. Reported as skipped where the
    start cannot be made to fail: ``RLIMIT_NPROC`` not enforced (root,
    ``CAP_SYS_RESOURCE``), or reads that are sequential without a pool (an
    address-space or data limit, strict overcommit)."""

    member = native_lowering.plan_binary(stored, loop_native)
    path = tmp_path / "pool.bin"
    path.write_bytes(member)
    child = _children(_POOL_LIMIT, str(path), RAYON_NUM_THREADS="8")
    out = _collect("pool", child, 120)
    assert isinstance(out, dict)
    if out["sequential"]:
        pytest.skip(
            "reads decode on one thread here without starting a pool (an "
            "address-space or data limit, or strict overcommit)"
        )
    if not out["enforced"]:
        pytest.skip(
            "RLIMIT_NPROC is not enforced for this user (root or "
            "CAP_SYS_RESOURCE): the pool's start cannot be made to fail"
        )
    digest = native_lowering.read(member, loop_native)
    assert out["pool_before"] is None, out
    assert out["rows"] == [digest] * 3, out
    assert out["pool"] is False and out["pool_after"] is False, out


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_uuxg", "gg_ttxg", "uux_emepgg"])
def test_quark_line_massive_and_electroweak_plans(slices, loop_native, name):
    from pyamplicol.loops.v2.observables import ColumnPlan, ObservableSlice

    slice_ = slices(name)
    check_leaf_weights(slice_)
    forward = export_plan(slice_)
    check(forward, loop_native)
    # Every key of schedule F's records is written (a closure's relabelling
    # as null when absent).
    check_every_key_written(forward, loop_native)
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
def test_plan_bindings_need_no_symbolica_licence(stored, loop_native, tmp_path):
    """``V2PlanModel``, ``v2_plan_load``, ``v2_plan_read`` and
    ``v2_plan_read_pool`` in a process without a key (RUST-MIGRATION.md rule
    4): the JSON written byte-identical from the JSON and from the Python
    records (``from_program``), the binary round trip bitwise, the derivation
    bound (``derive_bound``), equal load digests from both encodings and from
    ``read_binary``, and the reader's record of the pool.

    The child imports the extension, which starts a restricted Symbolica
    instance: like ``test_native_builder_needs_no_symbolica_licence`` it must
    not overlap another unlicensed Symbolica process of the user.
    """

    text = stored.json().encode()
    member = native_lowering.plan_binary(stored, loop_native)
    expected = native_lowering.load(text, loop_native, digest=True).digest
    records = tmp_path / "records.json"
    records.write_text(json.dumps(stored.program, sort_keys=True))
    program = tmp_path / "program.json"
    program.write_text(stored.python_json())
    written = tmp_path / "written.json", tmp_path / "written.bin"
    environment = {
        key: value for key, value in os.environ.items() if key != "SYMBOLICA_LICENSE"
    }
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from pyamplicol.loops.native import load_loop_native\n"
        "native = load_loop_native()\n"
        "model = native.V2PlanModel.from_json(Path(sys.argv[1]).read_bytes())\n"
        "text, member = bytes(model.json()), bytes(model.binary())\n"
        "program = json.loads(Path(sys.argv[4]).read_text())\n"
        "assert bytes(native.V2PlanModel.from_program(program).json()) == text\n"
        "again = native.V2PlanModel.from_binary(member)\n"
        "assert bytes(again.json()) == text and bytes(again.binary()) == member\n"
        "assert all(isinstance(x, int) and x > 0 for x in again.derive_bound())\n"
        "Path(sys.argv[2]).write_bytes(text)\n"
        "Path(sys.argv[3]).write_bytes(member)\n"
        "for data in (text, member):\n"
        "    print(native.v2_plan_load(data, digest=True)[3])\n"
        "print(native.v2_plan_read(member))\n"
        # None where reads never start a pool (an address-space limit).
        "assert native.v2_plan_read_pool() in (True, None)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(records), *map(str, written), str(program)],
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
    reader's parallel decode (``binary.rs``, "Parallel decoding"), and which
    loads (one leg, one row, one column, three Born components)."""

    plan = _empty_plan() | {"legs": 1, "rows": 1, "columns": 1, "helicities": [[1]]}
    plan["born_components"] = 3
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
    The production JSON load (``LoopProgramV4::load``, hardened in phase 1)
    refuses the plans themselves alike, as JSON (``... json``; the decimal
    power there is 2^32, which numerica would panic on).
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
        # The JSON load (``LoopProgramV4::load``, phase 1): the same probe.
        bombs[f"{name} json"] = (_text(plan), refused)
    for name, plan, expected in (
        ("vertex 2^40", vertex, "a vertex id beyond the terms"),
        ("family rank 1e6", rank, "family rank exceeds propagators"),
        ("fused exponent 2^20", exponent, "q-degree above its content's"),
    ):
        bombs[name] = (member(plan), expected)
        bombs[f"{name} json"] = (_text(plan), expected)
    bombs["decimal power 1e9"] = (_unpadded(member(power)), bound)
    # A JSON is not bounded per byte (it cannot be padded): its decimal power
    # of 10^9 is charged and granted, so the JSON case is a power numerica
    # cannot raise (it panics at 2^32), refused before deriving.
    power["families"][0]["leaves"][0]["weight"] = "1e4294967296"
    bombs["decimal power 2^32 json"] = (_text(power), "power of ten beyond 2^32")
    return bombs


def _text(plan: dict) -> bytes:
    return json.dumps(plan, separators=(",", ":")).encode()


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

# A thread pool that cannot start (no thread may be spawned: RLIMIT_NPROC's
# soft limit at one process, shown enforced by a Python thread that cannot
# start): every read decodes on one thread, also once the limit is lifted
# (rayon never starts a global pool after a failed start); never a refusal or
# rayon's panic. The reader's record of the pool (``v2_plan_read_pool``):
# none before the first read, failed after. ``sequential``: reads here never
# start a pool (``binary.rs``, ``allocations_can_fail``).
_POOL_LIMIT = """
import json, resource, sys, threading
from pathlib import Path
from pyamplicol.loops.native import load_loop_native
native = load_loop_native()
member = Path(sys.argv[1]).read_bytes()
limited = [resource.getrlimit(r)[0] != resource.RLIM_INFINITY
           for r in (resource.RLIMIT_AS, resource.RLIMIT_DATA)]
strict = Path('/proc/sys/vm/overcommit_memory').read_text().strip() == '2'
out = {'sequential': any(limited) or strict,
       'pool_before': native.v2_plan_read_pool()}
hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
try:
    threading.Thread(target=lambda: None).start()
    out['enforced'] = False
except RuntimeError:
    out['enforced'] = True
rows = []
for limited in (True, True, False):
    if not limited:
        out['pool'] = native.v2_plan_read_pool()
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    try:
        rows.append(native.v2_plan_read(member))
    except ValueError as error:
        rows.append(str(error))
out['rows'] = rows
out['pool_after'] = native.v2_plan_read_pool()
print(json.dumps(out))
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
    The wide member (parallel sections) is read likewise with the default
    pool (8 threads, default malloc arenas): under the limit the read decodes
    on one thread (``read_threads``), so it is refused with an error until it
    loads, never aborting (the evidence audit's abort of fix round 7).
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
    wide = _wide_member(native)
    path = tmp_path / "limits-wide.bin"
    path.write_bytes(wide)
    pooled = _children(
        _UNDER_LIMITS, str(path), json.dumps(HEADROOMS), RAYON_NUM_THREADS="8"
    )
    children["wide, default pool"] = (pooled, wide)
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
    list-dominated and multi-precision members under tighter limits
    (``_under_limits``, children run concurrently with it). The child
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
        assert small or name.endswith((" padded", " json")), (name, len(data))
    for name, (child, data) in limits.items():
        rows = _collect(name, child, 120)
        assert isinstance(rows, list)
        digest = native_lowering.read(data, native)
        if name == "wide, default pool":
            assert rows[-1] == digest, rows
            continue
        assert len(rows) > 1 and all("allocator refuses" in r for r in rows[:-1]), rows
        assert rows[-1] == digest, (name, rows)
