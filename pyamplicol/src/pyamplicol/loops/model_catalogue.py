# SPDX-License-Identifier: 0BSD
"""Exact source-basis certificates reused by the generic request compiler.

These identify locally tested tensor products, not model names or processes.
Unknown bases remain in the prepared source catalogue and fail if a physical
request actually uses them. Scalar couplings, UV scheme and graph selection
remain source-owned and separately checked.
"""

from __future__ import annotations

from types import SimpleNamespace

from .model_correspondence import CountertermPrimitiveValidation
from .model_electroweak import massive_neutral_lepton_counterterm_validations
from .model_expressions import parse_exact_ufo
from .model_gluons import (
    _FOUR_R2_COLOR,
    _FOUR_R2_LORENTZ,
    _ORDINARY_COLOR,
    _ORDINARY_LORENTZ,
    _TWO_POINT,
)
from .model_neutral_qcd import neutral_qcd_counterterm_validations
from .vertex_frames import certified_fermion_slash_source, certified_ffv_source


def certified_source_counterterm_bases(source):
    """Return exact-key conformance for the existing neutral and QCD primitives.

    Source FFV tensors retain their UFO frame and use the separately certified
    typed endpoint adapter at physical construction. No bundled-model scalar or
    native vertex representative is substituted. Gluon four-point R2 evidence is
    fixed SU(3) only; it does not provide a formal-Nc continuation.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    validations = {
        item.source_key: item
        for item in (
            *massive_neutral_lepton_counterterm_validations(source),
            *neutral_qcd_counterterm_validations(source),
        )
    }
    parsed = {}

    def expression(text):
        if text not in parsed:
            parsed[text] = parse_exact_ufo(text)
        return parsed[text]

    def any_equal(value, known):
        return any(
            (expression(value) - expression(item)).expand() == 0 for item in known
        )

    for entry in source.kernels:
        if entry.kind == "tree" or entry.key in validations:
            continue
        particles = tuple(
            source.raw.collections["particles"][index]["fields"]
            for index in entry.particle_ids
        )
        pdgs = tuple(item["pdg_code"] for item in particles)
        spins = tuple(item["spin"] for item in particles)
        colors = tuple(item["color"] for item in particles)
        control = None
        if (
            spins == (2, 2, 3)
            and pdgs[0] == -pdgs[1]
            and pdgs[1] > 0
            and (
                (colors == (-3, 3, 8) and any_equal(entry.color, ("T(3,2,1)",)))
                or (colors == (-3, 3, 1) and any_equal(entry.color, ("Identity(1,2)",)))
            )
            and certified_ffv_source(
                SimpleNamespace(
                    spins=spins, structure=wrap_indices(expression(entry.lorentz))
                )
            )
        ):
            control = "source-FFV-physical-frame-and-fundamental-color-control"
        elif (
            entry.kind == "R2"
            and spins == (2, 2)
            and pdgs[0] == -pdgs[1]
            and pdgs[1] in range(1, 7)
            and colors == (-3, 3)
            and any_equal(entry.color, ("Identity(1,2)",))
            and certified_fermion_slash_source(
                SimpleNamespace(
                    spins=spins, structure=wrap_indices(expression(entry.lorentz))
                )
            )
        ):
            control = "local-fermion-two-point-source-aloha-control"
        elif (
            all(pdg == 21 for pdg in pdgs)
            and spins == (3,) * len(pdgs)
            and colors == (8,) * len(pdgs)
        ):
            if (
                len(pdgs) == 2
                and any_equal(entry.color, ("Tr(1,2)",))
                and any_equal(entry.lorentz, _TWO_POINT)
            ):
                control = "local-gluon-two-point-aloha-control"
            elif (
                len(pdgs) == 3
                and any_equal(entry.color, _ORDINARY_COLOR[:1])
                and any_equal(entry.lorentz, _ORDINARY_LORENTZ[:1])
            ):
                control = "local-gluon-ordinary-and-counterterm-aloha-control"
            elif len(pdgs) == 4:
                if any_equal(entry.color, _ORDINARY_COLOR[1:]) and any_equal(
                    entry.lorentz, _ORDINARY_LORENTZ[1:]
                ):
                    control = "local-gluon-ordinary-and-counterterm-aloha-control"
                elif (
                    entry.kind == "R2"
                    and any_equal(entry.color, _FOUR_R2_COLOR)
                    and any_equal(entry.lorentz, _FOUR_R2_LORENTZ)
                ):
                    control = "local-four-gluon-r2-aloha-and-mg-color-control"
        if control is not None:
            validations[entry.key] = CountertermPrimitiveValidation(
                entry.key, entry.color, entry.lorentz, pdgs, spins, colors, control
            )
    return tuple(validations.values())
