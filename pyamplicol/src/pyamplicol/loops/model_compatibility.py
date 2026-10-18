# SPDX-License-Identifier: 0BSD
"""Explicit, bounded source-declaration compatibility views."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pyamplicol.models.loop_import import RawLoopModel


@dataclass(frozen=True, slots=True)
class CompatibleLoopDeclarations:
    original: RawLoopModel
    derived: RawLoopModel
    diagnostics: tuple[Any, ...]


def prepare_loop_sm_scalar_declarations(raw: RawLoopModel):
    """Select the existing four-scalar tensor for the known loop_sm G0 vertex.

    The supplied V_1001 has four G0 fields but references the constant SSS1
    tensor with three declared scalar slots. SSSS1 is the same exact constant
    with four scalar slots. Only that reference changes in this explicit view;
    generic source signature validation remains strict. This is not a model or
    process completeness certificate.
    """
    from .covariant_model import _reference
    from .model_expressions import parse_exact_ufo

    key = "loop_sm_scalar_arity_compatibility"
    if key in raw.metadata:
        raise ValueError("loop_sm scalar declarations are already derived")
    vertex = raw.record("vertices", "V_1001")
    old = raw.record("lorentz_structures", "SSS1")
    new = raw.record("lorentz_structures", "SSSS1")
    fields = vertex["fields"]
    particles = tuple(_reference(raw, ref, "particles") for ref in fields["particles"])
    tensors = tuple(
        _reference(raw, ref, "lorentz_structures") for ref in fields["lorentz"]
    )
    if (
        len(particles) != 4
        or any(
            particle["name"] != "G0"
            or tuple(
                particle["fields"].get(name) for name in ("pdg_code", "spin", "color")
            )
            != (250, 1, 1)
            for particle in particles
        )
        or tensors != (old,)
        or tuple(old["fields"]["spins"]) != (1, 1, 1)
        or tuple(new["fields"]["spins"]) != (1, 1, 1, 1)
        or parse_exact_ufo(old["fields"]["structure"]) != 1
        or parse_exact_ufo(new["fields"]["structure"]) != 1
        or tuple(fields["color"]) != ("1",)
    ):
        raise ValueError("source differs from the loop_sm four-G0 arity mismatch")
    payload = raw.to_dict()
    target = payload["collections"]["vertices"][vertex["id"]]["fields"]
    target["lorentz"] = [
        {
            "kind": "reference",
            "collection": "lorentz_structures",
            "id": new["id"],
            "name": new["name"],
        }
    ]
    diagnostic = {
        "profile": key,
        "vertex": [vertex["id"], vertex["name"]],
        "original_lorentz": [old["id"], old["name"]],
        "derived_lorentz": [new["id"], new["name"]],
        "reason": "four scalar species require the existing four-slot constant tensor",
    }
    payload["metadata"][key] = diagnostic
    return CompatibleLoopDeclarations(
        raw, RawLoopModel.from_dict(payload), (MappingProxyType(diagnostic),)
    )
