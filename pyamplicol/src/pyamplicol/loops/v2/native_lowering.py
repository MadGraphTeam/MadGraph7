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
from dataclasses import dataclass
from typing import Any

# ``ExportedPlan.json()`` writes the plan through the native model (byte for
# byte ``ExportedPlan.python_json()``, RUST-MIGRATION.md rule 1; equal on every
# control, tests/unit/test_loop_v2_native_lowering.py).
NATIVE_JSON = True


def _native(native: Any | None) -> Any:
    """``native``, else the staged loop core, verified once per process: the
    one cache of ``dag_input._native`` (each ``load_loop_native`` re-verifies
    the native build, 0.2-0.4 s)."""

    if native is not None:
        return native
    from .dag_input import _native as loop_core

    return loop_core()


def _program(exported: Any) -> Any:
    return getattr(exported, "program", exported)


def plan_model(exported: Any, native: Any | None = None) -> Any:
    """The native model (``V2PlanModel``) of an exported plan's records (an
    ``ExportedPlan`` or its ``program``), read from the Python objects."""

    return _native(native).V2PlanModel.from_program(_program(exported))


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
    "NATIVE_JSON",
    "PlanLoad",
    "load",
    "plan_binary",
    "plan_encodings",
    "plan_json",
    "plan_model",
    "plan_model_from_json",
    "read",
]
