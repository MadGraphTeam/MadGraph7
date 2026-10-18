# SPDX-License-Identifier: 0BSD
"""Bounded source-gluon tensor controls for the initial fixed-SU(3) model path.

The exact expressions below were compared with generated ALOHA and MadGraph
colour algebra. No coupling coefficient or general-Nc continuation is inferred.
Ordinary representatives additionally require exact native basis/coupling
identity, including ordered particle ports, before they can be reused by a CT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .model_correspondence import (
    CountertermPrimitiveValidation,
    OrdinaryBasisCatalogue,
    OrdinaryBasisRepresentative,
)
from .model_expressions import parse_exact_ufo

_TWO_POINT = (
    "P(-1,1)*P(-1,1)*Metric(1,2)",
    "P(1,1)*P(2,1)",
    "Metric(1,2)",
)
_FOUR_R2_LORENTZ = (
    "Metric(1,4)*Metric(2,3)",
    "Metric(1,3)*Metric(2,4)",
    "Metric(1,2)*Metric(3,4)",
    "Metric(1,4)*Metric(2,3)+Metric(1,3)*Metric(2,4)+Metric(1,2)*Metric(3,4)",
)
_FOUR_R2_COLOR = (
    "d(-1,1,3)*d(-1,2,4)",
    "d(-1,1,3)*f(-1,2,4)",
    "d(-1,1,4)*d(-1,2,3)",
    "d(-1,1,4)*f(-1,2,3)",
    "d(-1,2,3)*f(-1,1,4)",
    "d(-1,2,4)*f(-1,1,3)",
    "f(-1,1,2)*f(-1,3,4)",
    "f(-1,1,3)*f(-1,2,4)",
    "f(-1,1,4)*f(-1,2,3)",
    "Identity(1,2)*Identity(3,4)",
    "Identity(1,3)*Identity(2,4)",
    "Identity(1,4)*Identity(2,3)",
)
_ORDINARY_COLOR = (
    "f(1,2,3)",
    "f(-1,1,2)*f(3,4,-1)",
    "f(-1,1,3)*f(2,4,-1)",
    "f(-1,1,4)*f(2,3,-1)",
)
_ORDINARY_LORENTZ = (
    "P(3,1)*Metric(1,2)-P(3,2)*Metric(1,2)-P(2,1)*Metric(1,3)"
    "+P(2,3)*Metric(1,3)+P(1,2)*Metric(2,3)-P(1,3)*Metric(2,3)",
    "Metric(1,4)*Metric(2,3)-Metric(1,3)*Metric(2,4)",
    "Metric(1,4)*Metric(2,3)-Metric(1,2)*Metric(3,4)",
    "Metric(1,3)*Metric(2,4)-Metric(1,2)*Metric(3,4)",
)


@dataclass(frozen=True, slots=True)
class GluonCountertermBasis:
    ordinary_representatives: tuple[OrdinaryBasisRepresentative, ...]
    primitive_validations: tuple[CountertermPrimitiveValidation, ...]
    entries: tuple[Any, ...] = field(repr=False)
    formal_nc_continuation_validated: bool = field(default=False, init=False)
    scalar_coefficients_validated: bool = field(default=False, init=False)


def _equal(left, right):
    return (parse_exact_ufo(left) - parse_exact_ufo(right)).expand() == 0


def prepare_gluon_counterterm_basis(source, native_model) -> GluonCountertermBasis:
    """Select exact controlled gluon tensors, failing on an unknown pure-gluon CT.

    This is an explicitly bounded QCD tensor catalogue, not a general model
    certification. All selected CTs have only gluon external ports and one QCD
    coupling order. Loop groups/alternatives remain untouched for the process
    matcher. Model scalar coefficients, UV scheme and active four-gluon R2's
    general-Nc continuation remain separately uncertified.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    def ports(entry):
        return tuple(
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        )

    def gluon(entry):
        return len(entry.particle_ids) in (2, 3, 4) and all(
            (p["pdg_code"], p["spin"], p["color"]) == (21, 3, 8) for p in ports(entry)
        )

    payload = json.loads(native_model.to_json())
    species = {particle["name"]: particle for particle in payload["particles"]}
    lorentz = {item["name"]: item for item in payload["lorentz_structures"]}
    representatives = []
    for entry in source.kernels:
        if entry.kind != "tree" or not gluon(entry):
            continue
        if not (
            any(_equal(entry.lorentz, known) for known in _ORDINARY_LORENTZ)
            and any(_equal(entry.color, known) for known in _ORDINARY_COLOR)
        ):
            raise NotImplementedError(
                "ordinary gluon source tensor lacks local conformance control"
            )
        normalized = wrap_indices(parse_exact_ufo(entry.lorentz))
        candidates = []
        for rule in payload["vertex_rules"]:
            target_ports = tuple(species[name] for name in rule["particles"])
            if len(target_ports) != len(ports(entry)) or any(
                tuple(a[key] for key in ("pdg_code", "spin", "color"))
                != tuple(b[key] for key in ("pdg_code", "spin", "color"))
                for a, b in zip(ports(entry), target_ports, strict=True)
            ):
                continue
            for ci, color in enumerate(rule["color_structures"]):
                if not _equal(entry.color, color):
                    continue
                for li, name in enumerate(rule["lorentz_structures"]):
                    coupling = rule["couplings"][ci][li]
                    if coupling is None:
                        continue
                    if (
                        normalized
                        - native_model.lorentz_structure(
                            lorentz[name]["name"]
                        ).structure
                    ).expand() != 0:
                        continue
                    if (
                        entry.coupling.value.expression
                        - native_model.coupling(coupling).expression
                    ).expand() != 0:
                        continue
                    candidates.append(
                        OrdinaryBasisRepresentative(
                            entry.key,
                            rule["name"],
                            ci,
                            li,
                            "local-gluon-ordinary-and-counterterm-aloha-control",
                        )
                    )
        if len(candidates) != 1:
            raise ValueError(
                "ordinary gluon source requires one exact native "
                "tensor/coupling representative"
            )
        representatives.extend(candidates)
    catalogue = OrdinaryBasisCatalogue(source, native_model, representatives)
    entries = tuple(
        entry
        for entry in source.kernels
        if entry.kind != "tree"
        and gluon(entry)
        and len(entry.order) == 1
        and entry.order[0][0] == "QCD"
    )
    validations = []
    for entry in entries:
        if catalogue.counterterm(entry) is not None:
            continue
        if (
            len(entry.particle_ids) == 2
            and _equal(entry.color, "Tr(1,2)")
            and any(_equal(entry.lorentz, known) for known in _TWO_POINT)
        ):
            control = "local-gluon-two-point-aloha-control"
        elif (
            entry.kind == "R2"
            and len(entry.particle_ids) == 4
            and any(_equal(entry.color, known) for known in _FOUR_R2_COLOR)
            and any(_equal(entry.lorentz, known) for known in _FOUR_R2_LORENTZ)
        ):
            control = "local-four-gluon-r2-aloha-and-mg-color-control"
        else:
            raise NotImplementedError(
                f"pure-gluon counterterm {entry.vertex_name} "
                "lacks a controlled tensor basis"
            )
        count = len(entry.particle_ids)
        validations.append(
            CountertermPrimitiveValidation(
                entry.key,
                entry.color,
                entry.lorentz,
                (21,) * count,
                (3,) * count,
                (8,) * count,
                control,
            )
        )
    return GluonCountertermBasis(tuple(representatives), tuple(validations), entries)
