# SPDX-License-Identifier: 0BSD
"""Bounded local source conformance for colour-singlet electroweak tensors.

Entries are certified by structure (genericity audit m5): spins, colour
representations, charge conjugation and the exact colour/Lorentz tensors,
never by particle identities. A colour-singlet fermion pair is a particle and
its antiparticle (the antiparticle first, as UFO orders FFV/FFS vertices); a
neutral boson is its own antiparticle.
"""

from types import SimpleNamespace

from .model_correspondence import CountertermPrimitiveValidation
from .model_expressions import parse_exact_ufo
from .vertex_frames import certified_ffs_source, certified_ffv_source


def _neutral(particle) -> bool:
    """A self-conjugate (neutral) particle."""

    return particle["name"] == particle["antiname"]


def _singlet_fermion_pair(first, second) -> bool:
    """An antifermion and its fermion, colour singlets, the antiparticle
    first (UFO's sign convention for antiparticle PDG codes)."""

    return (
        first["spin"] == second["spin"] == 2
        and first["color"] == second["color"] == 1
        and first["antiname"] == second["name"]
        and second["antiname"] == first["name"]
        and first["name"] != second["name"]
        and first["pdg_code"] == -second["pdg_code"] < 0
    )


def neutral_scalar_vector_counterterm_validations(source):
    """Bind tested G0--A/Z momentum tensors to their exact source basis keys.

    This separate local certificate does not extend process admission, select
    counterterms, evaluate their scalar couplings, or certify a UV scheme.
    """
    structures = tuple(map(parse_exact_ufo, ("P(2,1)", "P(2,2)")))
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
        expression = parse_exact_ufo(entry.lorentz)
        if (
            spins == (1, 3)
            and colors == (1, 1)
            and all(_neutral(p) for p in particles)
            and any((expression - item).expand() == 0 for item in structures)
        ):
            result.append(
                CountertermPrimitiveValidation(
                    entry.key,
                    entry.color,
                    entry.lorentz,
                    pdgs,
                    spins,
                    colors,
                    "source-neutral-scalar-vector-two-point-ALOHA-control",
                )
            )
    return tuple(result)


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
            and _singlet_fermion_pair(*particles[:2])
            and spins[2] == 3
            and _neutral(particles[2])
            and certified_ffv_source(
                SimpleNamespace(spins=spins, structure=wrap_indices(expression))
            )
        )
        vector = (
            spins == (3, 3)
            and all(_neutral(p) for p in particles)
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


def massive_neutral_lepton_counterterm_validations(source):
    """Add tested neutral FFS and scalar/vector local bases without pruning.

    This source-key catalogue is bounded to a colour-singlet fermion pair with
    a neutral scalar, and neutral scalar/vector mixing. The physical assembly
    must still retain every matched source entry, check scalar diagnostics and
    reject any unvalidated tensor.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    result = [
        *neutral_lepton_counterterm_validations(source),
        *neutral_scalar_vector_counterterm_validations(source),
    ]
    for entry in source.kernels:
        if entry.kind == "tree" or parse_exact_ufo(entry.color) != 1:
            continue
        particles = [
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        ]
        pdgs = tuple(p["pdg_code"] for p in particles)
        spins = tuple(p["spin"] for p in particles)
        colors = tuple(p["color"] for p in particles)
        if (
            len(pdgs) == 3
            and _singlet_fermion_pair(*particles[:2])
            and spins[2] == 1
            and colors == (1, 1, 1)
            and _neutral(particles[2])
            and certified_ffs_source(
                SimpleNamespace(
                    spins=spins,
                    structure=wrap_indices(parse_exact_ufo(entry.lorentz)),
                )
            )
        ):
            result.append(
                CountertermPrimitiveValidation(
                    entry.key,
                    entry.color,
                    entry.lorentz,
                    pdgs,
                    spins,
                    colors,
                    "source-dirac-scalar-ALOHA-physical-frame-control",
                )
            )
    return tuple(result)
