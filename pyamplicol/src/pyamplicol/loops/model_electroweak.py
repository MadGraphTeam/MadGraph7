# SPDX-License-Identifier: 0BSD
"""Bounded local source conformance for neutral lepton electroweak controls."""

from types import SimpleNamespace

from .model_correspondence import CountertermPrimitiveValidation
from .model_expressions import parse_exact_ufo
from .vertex_frames import certified_ffv_source


def neutral_lepton_counterterm_validations(source):
    """Bind exact source keys to tested FFV and neutral two-vector primitives.

    This does not remove any model rule. A process using another source basis
    still fails the physical assembler's complete tensor-basis guard. Scalar
    normalization, renormalization and the full-process comparison are separate.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    vector_sources = tuple(
        map(
            parse_exact_ufo,
            (
                "P(-1,1)*P(-1,1)*Metric(1,2)",
                "P(1,1)*P(2,1)",
                "Metric(1,2)",
                "-P(-1,1)*P(-1,1)*Metric(1,2)",
                "-P(1,1)*P(2,1)",
            ),
        )
    )
    result = []
    for entry in source.kernels:
        if entry.kind == "tree" or parse_exact_ufo(entry.color) != 1:
            continue
        particles = [
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        ]
        pdgs = tuple(p["pdg_code"] for p in particles)
        spins = tuple(p["spin"] for p in particles)
        colors = tuple(p["color"] for p in particles)
        if any(color != 1 for color in colors):
            continue
        expression = parse_exact_ufo(entry.lorentz)
        ffv = (
            len(pdgs) == 3
            and pdgs[:2] in ((-11, 11), (-13, 13))
            and pdgs[2] in (22, 23)
            and certified_ffv_source(
                SimpleNamespace(spins=spins, structure=wrap_indices(expression))
            )
        )
        vector = (
            spins == (3, 3)
            and all(pdg in (22, 23) for pdg in pdgs)
            and any((expression - value).expand() == 0 for value in vector_sources)
        )
        if ffv or vector:
            result.append(
                CountertermPrimitiveValidation(
                    entry.key,
                    entry.color,
                    entry.lorentz,
                    pdgs,
                    spins,
                    colors,
                    "source-neutral-lepton-FFV-physical-frame-and-ALOHA-control"
                    if ffv
                    else "source-neutral-vector-two-point-ALOHA-control",
                )
            )
    return tuple(result)
