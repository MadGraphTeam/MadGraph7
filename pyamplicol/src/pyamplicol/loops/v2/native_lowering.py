# SPDX-License-Identifier: 0BSD
"""Native plan lowering (WP3.1 package D, ``RUST-MIGRATION.md`` §3.D).

The native plan model (``rusticol_core::loop_v2::plan::binary::PlanModel``)
holds a plan v4 in the native loader's own file form and writes it in two
encodings:

* the plan JSON, byte-identical to ``json.dumps(program, separators=(",",
  ":"))`` (keys in the exporter's order);
* the binary columnar member (one column per record field, at its narrowest
  width), which ``loop_v2::plan::read_binary`` reads through the same
  validation and derivations as the JSON (the schedule-B row links stay
  derived at load). It holds plan data only, no path or host information, and
  declares its decoded size, bounded by its length: the reader refuses a
  member beyond the bound before allocating (no decompression bomb). The
  loader's derivations are bounded by the member too (``derive_bound``): the
  reader refuses values the derivations would trust beyond the plan's tables
  and a member whose derivations exceed its bound per byte (the writer pads a
  plan that needs more; none written today does), and probes the decoded
  model and the derivations' peak fallibly before making them, so a member
  the allocator cannot hold is refused with an error, never an abort.

The model is fed with the records of a Python ``ExportedPlan``
(``plan_export``), read from the Python objects themselves
(``V2PlanModel.from_program``). It accepts only records the writer writes
back as ``json.dumps`` writes them: the loader's types, every record's keys
in the writer's order (the exporter's), every key the writer always writes
present (an absent option as ``None``) and no ``None`` under a key the writer
omits when absent (a step's ``rows``, an edge's ``links``, ``column_rows``,
the only keys a record may lack); anything else is an error naming the
record, so the native JSON of an accepted program is ``json.dumps``'s byte
for byte. The writer and the member name every field of the loader's file
form, so a field added there does not compile until both carry it.
``ExportedPlan.json()`` and ``ExportedPlan.binary()`` take this route (the
D2' switch inside ``export_plan``'s result, so the main plan, the
companion's and the observables' column plans all do); ``NATIVE_JSON``
switches the JSON back to ``json.dumps`` (``ExportedPlan.python_json()``, the
oracle). The native lowering itself (D3) comes later.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# ``ExportedPlan.json()`` writes the plan through the native model (byte for
# byte ``ExportedPlan.python_json()``, RUST-MIGRATION.md rule 1; equal on every
# control, tests/unit/test_loop_v2_native_lowering.py).
NATIVE_JSON = True

# M-FINAL MF-8G-1 item 5 (rule 1): ``export_plan`` streams each schedule-B
# family record and each content record into the native model as the
# exporter makes it (``families_builder``, ``contents_builder``), so the
# export never holds the Python records of every family or content; the
# plan's ``program["families"]`` and ``program["contents"]`` are then
# read-only views that decode a record from the model when asked
# (``NativeFamilies``, ``NativeContents``) and ``ExportedPlan.json()`` /
# ``binary()`` come from that model. Off (or with ``NATIVE_JSON`` off), the
# Python records stay in the program (the oracle,
# ``ExportedPlan.python_json()``).
NATIVE_FAMILIES = True

# M-FINAL MF-8G-1 item 5: the binary plan member is the plan's stored form
# (``ExportedPlan.stored()``): an artifact's plan and companion plan members
# (``artifact.write_plan``), the native process's plan and the plan every
# native executor loads (``loop_v2::plan::read_binary``, which
# ``LoopProgramV4::load`` takes for a member). ``plan_sha256`` digests the
# stored bytes. Off, the plan JSON is stored and loaded (the oracle; the
# form of artifacts written before the member, which every reader keeps
# reading).
BINARY_PLAN = True


# The staged loop core, verified once per process (each ``load_loop_native``
# re-verifies the native build, 0.2-0.4 s). Its own cache, not
# ``dag_input._native``'s: a stored process's readers (``member_json``,
# ``native_trees``) call it at load, where no generation module runs.
_LOADED: list[Any] = []


def _native(native: Any | None = None) -> Any:
    """``native``, else the staged loop core (verified once per process)."""

    if native is not None:
        return native
    if not _LOADED:
        from pyamplicol.loops.native import load_loop_native

        _LOADED.append(load_loop_native())
    return _LOADED[0]


def _program(exported: Any) -> Any:
    return getattr(exported, "program", exported)


def plan_model(exported: Any, native: Any | None = None) -> Any:
    """The native model (``V2PlanModel``) of an exported plan's records (an
    ``ExportedPlan`` or its ``program``), read from the Python objects; the
    plan's own model when its families were streamed (``NATIVE_FAMILIES``)."""

    if getattr(exported, "streamed", False):
        return exported.model
    return _native(native).V2PlanModel.from_program(_program(exported))


def families_builder(native: Any | None = None) -> Any:
    """An empty ``V2PlanFamilies``: ``push`` one family record at a time."""

    return _native(native).V2PlanFamilies()


def contents_builder(native: Any | None = None) -> Any:
    """An empty ``V2PlanContents``: ``push`` one content record at a time."""

    return _native(native).V2PlanContents()


def streamed_model(
    program: dict[str, Any],
    families: Any,
    contents: Any | None = None,
    native: Any | None = None,
) -> Any:
    """The native model of ``program`` (its ``families``, and with
    ``contents`` its ``contents``, empty lists) with the streamed
    ``families`` (``families_builder``, taken) and ``contents``
    (``contents_builder``, taken)."""

    return _native(native).V2PlanModel.from_program_families(
        program, families, contents
    )


class _NativeRecords(Sequence[dict[str, Any]]):
    """A plan list whose records were streamed into the native model: each
    record decoded from the model when asked (its plan JSON record), nothing
    kept. Not JSON-serialisable on purpose: ``json.dumps(program)``
    (``ExportedPlan.python_json()``, the oracle) needs the Python records."""

    _count = ""
    _json = ""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __len__(self) -> int:
        return int(getattr(self.model, self._count)())

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        return json.loads(bytes(getattr(self.model, self._json)(index)))


class NativeFamilies(_NativeRecords):
    """``program["families"]`` of a plan whose families were streamed."""

    _count = "family_count"
    _json = "family_json"


class NativeContents(_NativeRecords):
    """``program["contents"]`` of a plan whose contents were streamed."""

    _count = "content_count"
    _json = "content_json"


def plan_model_from_json(exported: Any, native: Any | None = None) -> Any:
    """The native model of an exported plan's records read from a JSON with
    sorted keys and Python's default separators (the JSON reader's route, a
    test of the writer, which must restore the exporter's key order and
    compact form on its own)."""

    records = json.dumps(_program(exported), sort_keys=True).encode()
    return _native(native).V2PlanModel.from_json(records)


def plan_json(exported: Any, native: Any | None = None) -> bytes:
    """The plan JSON written natively (``ExportedPlan.json()``'s bytes)."""

    return bytes(plan_model(exported, native).json())


def plan_binary(exported: Any, native: Any | None = None) -> bytes:
    """The binary columnar plan member of an exported plan."""

    return bytes(plan_model(exported, native).binary())


# The binary plan member's magic (``loop_v2::plan::binary``).
_MAGIC = b"LOOPPLAN"


def is_member(plan: str | bytes) -> bool:
    """Whether a stored plan is a binary plan member (else a plan JSON, as
    text or bytes): ``loop_v2::plan::binary::is_binary``'s magic."""

    return not isinstance(plan, str) and bytes(plan[:8]) == _MAGIC


def member_json(plan: str | bytes, native: Any | None = None) -> str:
    """The plan JSON of a stored plan (M-FINAL MF-8G-1 item 5): the text of a
    JSON, the native model's JSON of a binary member (``V2PlanModel.
    from_binary(...).json()``, byte for byte the exporter's JSON of the same
    plan). Formed on each call, never kept: the Python readers that still
    read plan records (the observable column sets) take it from here."""

    if isinstance(plan, str):
        return plan
    data = bytes(plan)
    if not is_member(data):
        return data.decode("ascii")
    return bytes(_native(native).V2PlanModel.from_binary(data).json()).decode("ascii")


def plan_encodings(exported: Any, native: Any | None = None) -> tuple[bytes, bytes]:
    """The plan JSON and the binary member of an exported plan, from one
    native model."""

    model = plan_model(exported, native)
    return bytes(model.json()), bytes(model.binary())


@dataclass(frozen=True)
class PlanLoad:
    """One native load of a plan: its encoding, the seconds to decode it into
    the model and to validate and derive the executor tables, and (when
    asked) the SHA-256 of the loaded tables."""

    form: str
    decode_s: float
    derive_s: float
    digest: str | None


def load(data: bytes, native: Any | None = None, *, digest: bool = False) -> PlanLoad:
    """Load a plan (binary member or JSON) as the native executor does, timing
    the decode and the derivations apart."""

    return PlanLoad(*_native(native).v2_plan_load(data, digest=digest))


def read(data: bytes, native: Any | None = None) -> str:
    """The digest of the program the production readers load from a plan: a
    binary member through ``read_binary`` (its one reader), a JSON through
    ``LoopProgramV4::load``.

    A diagnostic, outside the readers' fail-closed bound: the digest formats
    the loaded program (``program_digest``: a multi-precision weight as its
    decimal digits, through GMP and C ``malloc``), so under an address-space
    limit the load fits in it can abort the process. Probes of the readers'
    contract call ``load(data, digest=False)`` instead."""

    return str(_native(native).v2_plan_read(data))


__all__ = [
    "BINARY_PLAN",
    "NATIVE_FAMILIES",
    "NATIVE_JSON",
    "NativeContents",
    "NativeFamilies",
    "PlanLoad",
    "contents_builder",
    "families_builder",
    "is_member",
    "load",
    "member_json",
    "plan_binary",
    "plan_encodings",
    "plan_json",
    "plan_model",
    "plan_model_from_json",
    "read",
    "streamed_model",
]
