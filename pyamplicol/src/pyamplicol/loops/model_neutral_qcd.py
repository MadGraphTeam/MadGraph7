# SPDX-License-Identifier: 0BSD
"""Locally checked four-dimensional neutral-current QCD R2 tensor products."""

from .model_correspondence import CountertermPrimitiveValidation
from .model_expressions import parse_exact_ufo


def certified_neutral_qcd_basis(entry, raw):
    """Identify exact ordered source tensors; leave their coefficients untouched.

    ALOHA checks cover off-shell complex currents, including the installed UFO
    epsilon phase. Independent MG colour contractions cover Tr, d, f and complex
    trace probes. This is fixed-SU(3), four-dimensional model-rule evidence, not
    dimensional gamma-five, anomaly cancellation or renormalization evidence.
    """
    if entry.kind != "R2":
        return False
    particles = tuple(
        raw.collections["particles"][i]["fields"] for i in entry.particle_ids
    )
    pdgs = tuple(p["pdg_code"] for p in particles)
    colors = tuple(p["color"] for p in particles)
    if tuple(p["spin"] for p in particles) != (3,) * len(particles):
        return False
    candidates = ()
    if pdgs == (21, 21, 23) and colors == (8, 8, 1):
        candidates = (
            ("Tr(1,2)", "Epsilon(3,1,2,-1)*P(-1,2)-Epsilon(3,1,2,-1)*P(-1,1)"),
        )
    elif pdgs in ((21, 21, 21, 22), (21, 21, 21, 23)) and colors == (8, 8, 8, 1):
        candidates = (
            (
                "d(1,2,3)",
                "Metric(1,2)*Metric(3,4)+Metric(1,3)*Metric(2,4)+Metric(1,4)*Metric(2,3)",
            ),
        )
        if pdgs[-1] == 23:
            candidates += (("f(1,2,3)", "Epsilon(4,1,2,3)"),)
    return any(
        (parse_exact_ufo(entry.color) - parse_exact_ufo(color)).expand() == 0
        and (parse_exact_ufo(entry.lorentz) - parse_exact_ufo(lorentz)).expand() == 0
        for color, lorentz in candidates
    )


def neutral_qcd_counterterm_validations(source):
    result = []
    for entry in source.kernels:
        if not certified_neutral_qcd_basis(entry, source.raw):
            continue
        particles = tuple(
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        )
        result.append(
            CountertermPrimitiveValidation(
                entry.key,
                entry.color,
                entry.lorentz,
                tuple(p["pdg_code"] for p in particles),
                tuple(p["spin"] for p in particles),
                tuple(p["color"] for p in particles),
                "local-neutral-gauge-source-aloha-and-mg-color-control",
            )
        )
    return tuple(result)
