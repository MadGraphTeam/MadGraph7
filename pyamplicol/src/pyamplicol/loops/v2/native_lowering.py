# SPDX-License-Identifier: 0BSD
"""Native plan lowering (WP3.1 package D, ``RUST-MIGRATION.md`` §3.D).

The native plan model (``rusticol_core::loop_v2::plan::binary::PlanModel``)
holds a plan v4 in the native loader's own file form and writes it in two
encodings:

* the plan JSON, byte-identical to ``ExportedPlan.json()``
  (``json.dumps(program, separators=(",", ":"))``, keys in the exporter's
  order);
* the binary columnar member (one column per record field, at its narrowest
  width), which ``LoopProgramV4::load_binary`` reads through the same
  validation and derivations as the JSON (the schedule-B row links stay
  derived at load). It holds plan data only, no path or host information.

Phase 0 (this module): the model is fed with the records of a Python
``ExportedPlan``. They travel as JSON with sorted keys and Python's default
separators, so the native writer reproduces the exporter's key order and
compact form on its own rather than echoing its input. The switch inside
``export_plan`` (``ExportedPlan.json()`` from the native writer) and the
native lowering itself come in phase 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _native(native: Any | None) -> Any:
    if native is not None:
        return native
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


def plan_model(exported: Any, native: Any | None = None) -> Any:
    """The native model (``V2PlanModel``) of an exported plan's records."""

    records = json.dumps(exported.program, sort_keys=True).encode()
    return _native(native).V2PlanModel.from_json(records)


def plan_json(exported: Any, native: Any | None = None) -> bytes:
    """The plan JSON written natively (``ExportedPlan.json()``'s bytes)."""

    return bytes(plan_model(exported, native).json())


def plan_binary(exported: Any, native: Any | None = None) -> bytes:
    """The binary columnar plan member of an exported plan."""

    return bytes(plan_model(exported, native).binary())


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
    """Load a plan (binary member or JSON) as the native executor does."""

    return PlanLoad(*_native(native).v2_plan_load(data, digest=digest))


__all__ = ["PlanLoad", "load", "plan_binary", "plan_json", "plan_model"]
