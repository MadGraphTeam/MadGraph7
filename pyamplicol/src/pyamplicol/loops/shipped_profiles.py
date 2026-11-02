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
restriction lives in the source's metadata, not in its declarations. The
same holds for H.-S. Shao's QCD+QED loop models loop_qcd_qed_sm and
loop_qcd_qed_sm_Gmu (the shipped copies: UFOmodeldb's with the model patch of
UFOmodeldb issue 1 / mg5amcnlo#475; the G_mu model's Python-3 conversion
changes no declaration, its patch's charge-counterterm appends are folded into
their constructors), whose digests are those after the Goldstone-flag source
repair. Their profiles also accept UFOmodeldb's models (60ee58bb, which took
the patch's b-flavour R2 split, and 6ca4b120) and MG5_aMC's online-database
downloads: the patch (R2_G0tt_{d,s,b} split by loop
content, exactly unitary CKM, the G_mu charge counterterm's b-mass term) and
UFOmodeldb's R2_ZZHH fix (1f13c48f / 6ca4b120) change values and loop labels,
not the scheme. A modified copy (e.g. a massive-muon variant) has other
declarations and therefore no profile.

An explicit ``[loop.model]`` or a declaration in the model's own metadata
always wins (``models.loop_declaration.resolve_loop_model_declaration``). Like every
declaration, a profile is declared, not certified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyamplicol.models.loop_declaration import (
    LoopModelDeclaration,
    LoopSchemeDeclaration,
)

if TYPE_CHECKING:
    from pyamplicol.models.loop_import import RawLoopModel


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


# H.-S. Shao's NLOCT models with QCD and QED corrections (arXiv:1804.10017),
# as MadLoop uses them: complete UV and R2 counterterms of both, MadLoop's
# conventions, on-shell electroweak renormalisation in the model's input
# scheme (alpha(MZ) for loop_qcd_qed_sm, G_mu for loop_qcd_qed_sm_Gmu), real
# on-shell masses. The declaration the electroweak tests state in their cards.
def _qcd_qed_scheme(inputs: str) -> LoopSchemeDeclaration:
    return LoopSchemeDeclaration(
        dimensional_regularization="HV",
        external_states="physical-four-dimensional",
        gamma5="source-UFO-four-dimensional-prescription",
        renormalization=(
            "source-UFO on-shell electroweak and mass-dependent QCD decoupling"
        ),
        mass_scheme="real-on-shell",
        finite_restoration="source-UFO-local-rules",
        model_convention=(f"loop_qcd_qed_sm{inputs} lhv=1 supplied QCD and QED rules"),
    )


LOOP_QCD_QED_SM_SCHEME = _qcd_qed_scheme("")
LOOP_QCD_QED_SM_GMU_SCHEME = _qcd_qed_scheme("_Gmu")
SHIPPED_LOOP_PROFILES = (
    ShippedLoopProfile(
        "MG5_aMC 3.7.0 models/loop_sm",
        ("e2512823936595ee4848974146d20b1290de6ecd7cf48c5d03bb47e1466b1a77",),
        LoopModelDeclaration(LOOP_SM_SCHEME, ("QCD",), True, True),
    ),
    # The shipped copy (UFOmodeldb 6ca4b120 with the issue-1 model patch)
    # first, then UFOmodeldb 60ee58bb (the patch's b-flavour R2 split only),
    # UFOmodeldb 6ca4b120 unpatched and MG5_aMC's online-database download
    # (R2_ZZHH before UFOmodeldb's fix).
    ShippedLoopProfile(
        "UFOmodeldb loop_qcd_qed_sm (alpha(MZ) scheme)",
        (
            "0dc17624150975c9f751b744bb8cc096538489e84bbd75f5fb65fb20c36ab86b",
            "1555635eeb5b219264839f6fa1cb59e87c9c712878bb7a16587cef71da92f0b3",
            "580c029a2ae2ce1bfc58722897edfc9da0b809c5038d382404918b412d17b1e4",
            "1b9f65500b248fb8ecf4a28dae1258a1f48613403482a72a0c794d7cb3964f16",
        ),
        LoopModelDeclaration(LOOP_QCD_QED_SM_SCHEME, ("QCD", "QED"), True, True),
    ),
    ShippedLoopProfile(
        "UFOmodeldb loop_qcd_qed_sm_Gmu (G_mu scheme)",
        (
            "97f39eeb340606a6810879079826900ddbb5f99c15c142e1ecd98f27bde354b7",
            "14ddd9435ffefdb8247ef93a4400fc39a646328e0e1cc26a618a4da58c3b3667",
            "b580d14a07e550b3ab9ba0800b486eccd35117defe4c948c052c2d9c08029f48",
            "588cb07a2e1691f8ee1038613f5fe8c4b768d68ad3c1df42b2ea28c8b2daeb6f",
        ),
        LoopModelDeclaration(LOOP_QCD_QED_SM_GMU_SCHEME, ("QCD", "QED"), True, True),
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
    "LOOP_QCD_QED_SM_GMU_SCHEME",
    "LOOP_QCD_QED_SM_SCHEME",
    "LOOP_SM_SCHEME",
    "SHIPPED_LOOP_PROFILES",
    "ShippedLoopProfile",
    "shipped_loop_profile",
    "source_declarations_digest",
]
