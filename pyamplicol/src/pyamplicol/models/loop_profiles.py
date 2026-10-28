# SPDX-License-Identifier: 0BSD
"""Shipped ``[loop.model]`` declarations for loop models that declare none.

A UFO loop model states its counterterms but not their scheme. MG5_aMC's
loop_sm ships no ``pyamplicol_loop`` declaration, so a request on it needed
``[loop.model]`` and ``[loop.model.scheme]`` in its run card. pyAmpliCol
ships that declaration for the exact source it was written for: a profile
applies to a raw NLO source whose declarations (every UFO record collection,
after the structural source repairs) have one of its recorded digests
(``source_declarations_digest``), never to a model by name or path. The
shipped copy of loop_sm (``assets/models/ufo/loop_sm``), the MG5_aMC 3.7.0
directory itself and every restriction of either have the same digest: a
restriction lives in the source's metadata, not in its declarations.

An explicit ``[loop.model]`` or a declaration in the model's own metadata
always wins (``loop_declaration.resolve_loop_model_declaration``). Like every
declaration, a profile is declared, not certified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .loop_declaration import LoopModelDeclaration, LoopSchemeDeclaration

if TYPE_CHECKING:
    from .loop_import import RawLoopModel


@dataclass(frozen=True, slots=True)
class ShippedLoopProfile:
    """A shipped declaration (``declaration``) of the loop sources whose
    declarations digest (``source_declarations_digest``) is one of
    ``source_digests``; ``source`` names the source it was written for."""

    source: str
    source_digests: tuple[str, ...]
    declaration: LoopModelDeclaration


# MG5_aMC 3.7.0 models/loop_sm: QCD corrections, complete UV and R2
# counterterms, MadLoop's conventions ('t Hooft-Veltman, four-dimensional
# external states and gamma5, mass-dependent decoupling of heavy flavours,
# real on-shell masses). The declaration every pyAmpliCol loop_sm test and
# reference comparison has stated in its run card.
LOOP_SM_SCHEME = LoopSchemeDeclaration(
    dimensional_regularization="HV",
    external_states="physical-four-dimensional",
    gamma5="source-UFO-four-dimensional-prescription",
    renormalization="source-UFO mass-dependent decoupling",
    mass_scheme="real-on-shell",
    finite_restoration="source-UFO-local-rules",
    model_convention="loop_sm lhv=1 supplied QCD rules",
)
SHIPPED_LOOP_PROFILES = (
    ShippedLoopProfile(
        "MG5_aMC 3.7.0 models/loop_sm",
        ("e2512823936595ee4848974146d20b1290de6ecd7cf48c5d03bb47e1466b1a77",),
        LoopModelDeclaration(LOOP_SM_SCHEME, ("QCD",), True, True),
    ),
)


def source_declarations_digest(raw: RawLoopModel) -> str:
    """The sha256 of a raw NLO source's declarations: its schema version and
    every record collection (canonical JSON), after the structural source
    repairs (``covariant_model.repair_source_declarations``, idempotent).
    Independent of the source's path, name and restriction (metadata)."""

    from pyamplicol.loops.covariant_model import repair_source_declarations

    payload = repair_source_declarations(raw).to_dict()
    text = json.dumps(
        {
            "schema_version": payload["schema_version"],
            "collections": payload["collections"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def shipped_loop_profile(raw: RawLoopModel) -> ShippedLoopProfile | None:
    """The shipped profile of ``raw`` (``source_declarations_digest``), or
    None."""

    digest = source_declarations_digest(raw)
    for profile in SHIPPED_LOOP_PROFILES:
        if digest in profile.source_digests:
            return profile
    return None


__all__ = [
    "LOOP_SM_SCHEME",
    "SHIPPED_LOOP_PROFILES",
    "ShippedLoopProfile",
    "shipped_loop_profile",
    "source_declarations_digest",
]
