# SPDX-License-Identifier: 0BSD
"""Explicit, bounded source-declaration compatibility views (test support).

Moved out of ``src/pyamplicol/loops`` (genericity audit m10, 2026-10-25): no
production route imports it. The view repairs a source by structure, never by
a particle, vertex or tensor name.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pyamplicol.loops.covariant_model import _reference
from pyamplicol.loops.model_expressions import parse_exact_ufo
from pyamplicol.models.loop_import import RawLoopModel

PROFILE = "loop_sm_scalar_arity_compatibility"


@dataclass(frozen=True, slots=True)
class CompatibleLoopDeclarations:
    original: RawLoopModel
    derived: RawLoopModel
    diagnostics: tuple[Any, ...]


def _constant(record: dict) -> Any:
    """The exact structure of a slot-independent (constant) Lorentz tensor."""

    value = parse_exact_ufo(record["fields"]["structure"])
    from symbolica import AtomType

    return value if value.get_type() == AtomType.Num else None


_COUNTS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


def _count(number: int) -> str:
    return _COUNTS[number] if number < len(_COUNTS) else str(number)


def prepare_loop_sm_scalar_declarations(raw: RawLoopModel):
    """Repair scalar vertices that reference a constant tensor of too few slots.

    MadLoop's loop_sm declares its four-G0 vertex with the constant three-slot
    tensor SSS1; SSSS1 is the same exact constant with four slots. The view
    finds every such defect structurally: a vertex whose particles are all
    scalars and whose single Lorentz tensor is a constant with fewer (scalar)
    slots than particles, together with the unique tensor of the vertex's
    spin signature and the same exact constant. Only those references change;
    a vertex without such a unique replacement, or a source without the
    defect, is refused. Generic source signature validation remains strict.
    This is not a model or process completeness certificate.
    """

    if PROFILE in raw.metadata:
        raise ValueError("scalar declarations are already derived")
    tensors = raw.collections["lorentz_structures"]
    payload = raw.to_dict()
    repairs = []
    for vertex in raw.collections["vertices"]:
        fields = vertex["fields"]
        particles = tuple(
            _reference(raw, ref, "particles") for ref in fields["particles"]
        )
        used = tuple(
            _reference(raw, ref, "lorentz_structures") for ref in fields["lorentz"]
        )
        spins = tuple(int(p["fields"]["spin"]) for p in particles)
        if len(used) != 1 or len(used[0]["fields"]["spins"]) >= len(spins):
            continue
        (old,) = used
        constant = _constant(old)
        if (
            set(spins) != {1}
            or set(old["fields"]["spins"]) != {1}
            or constant is None
            or tuple(fields["color"]) != ("1",)
        ):
            raise ValueError(
                f"source vertex {vertex['name']!r} has a Lorentz-slot arity "
                "mismatch that is not a constant scalar tensor"
            )
        candidates = [
            record
            for record in tensors
            if tuple(record["fields"]["spins"]) == spins
            and _constant(record) == constant
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"source vertex {vertex['name']!r}: no unique constant tensor "
                f"with the spin slots {spins} and the structure of {old['name']!r}"
            )
        (new,) = candidates
        target = payload["collections"]["vertices"][vertex["id"]]["fields"]
        target["lorentz"] = [
            {
                "kind": "reference",
                "collection": "lorentz_structures",
                "id": new["id"],
                "name": new["name"],
            }
        ]
        repairs.append((vertex, old, new))
    if len(repairs) != 1:
        raise ValueError(
            "source differs from the scalar arity mismatch of one vertex "
            f"(found {len(repairs)})"
        )
    ((vertex, old, new),) = repairs
    # The arity in words: the four-scalar diagnostic stays byte-identical to
    # the one recorded in saved artifacts (their raw loop model).
    arity = _count(len(new["fields"]["spins"]))
    diagnostic = {
        "profile": PROFILE,
        "vertex": [vertex["id"], vertex["name"]],
        "original_lorentz": [old["id"], old["name"]],
        "derived_lorentz": [new["id"], new["name"]],
        "reason": (
            f"{arity} scalar species require the existing {arity}-slot constant tensor"
        ),
    }
    payload["metadata"][PROFILE] = diagnostic
    return CompatibleLoopDeclarations(
        raw, RawLoopModel.from_dict(payload), (MappingProxyType(diagnostic),)
    )
