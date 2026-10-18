# SPDX-License-Identifier: 0BSD
"""Bounded quark/gluon source tensor catalogue for QCD controls.

This combines the gluon catalogue with the calibrated vectorlike quark vertex
and an explicit two-point representative obtained by its momentum contraction.
Local conformance does not certify insertion phases or a whole process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .model_correspondence import (
    CountertermPrimitiveValidation,
    CountertermTensorRepresentative,
    OrdinaryBasisCatalogue,
    OrdinaryBasisRepresentative,
)
from .model_expressions import parse_exact_ufo
from .model_gluons import prepare_gluon_counterterm_basis


@dataclass(frozen=True, slots=True)
class QCDCountertermBasis:
    ordinary_representatives: tuple[OrdinaryBasisRepresentative, ...]
    primitive_validations: tuple[CountertermPrimitiveValidation, ...]
    tensor_representatives: tuple[CountertermTensorRepresentative, ...]
    entries: tuple[Any, ...] = field(repr=False)
    quark_pdgs: tuple[int, ...]
    formal_nc_continuation_validated: bool = field(default=False, init=False)
    scalar_coefficients_validated: bool = field(default=False, init=False)
    fermion_two_point_insertion_validated: bool = field(default=False, init=False)


def _equal(left, right):
    return (parse_exact_ufo(left) - parse_exact_ufo(right)).expand() == 0


def prepare_qcd_counterterm_basis(source, native_model, *, quark_pdgs=(2,)):
    """Combine controlled gauge rules and selected massless u/d quark rules.

    The source and native vector vertices have an explicitly calibrated spinor
    slot convention. The source two-point slash is locally ALOHA-tested, and
    its explicit native representative is the contraction of that calibrated
    vector current with the first leg's momentum. Its insertion phase/routing
    still requires a separate process control. Unknown structures, massive
    two-point terms and extra QCD contact terms fail rather than silently
    acquiring this evidence. Species outside ``quark_pdgs`` remain outside
    this deliberately restricted catalogue.
    """
    from ufo_model_loader.symbolica_processing import wrap_indices

    quarks = tuple(quark_pdgs)
    if (
        not quarks
        or len(set(quarks)) != len(quarks)
        or any(
            isinstance(pdg, bool) or not isinstance(pdg, int) or pdg not in (1, 2)
            for pdg in quarks
        )
    ):
        raise ValueError("controlled massless quark PDGs must be distinct 1 or 2")
    gauge = prepare_gluon_counterterm_basis(source, native_model)
    representatives = list(gauge.ordinary_representatives)
    validations = list(gauge.primitive_validations)
    tensor_representatives = []
    entries = list(gauge.entries)
    payload = json.loads(native_model.to_json())
    particles = {p["name"]: p for p in payload["particles"]}

    def ports(entry):
        return tuple(
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        )

    def pdgs(entry):
        return tuple(p["pdg_code"] for p in ports(entry))

    def selected(entry):
        ids = pdgs(entry)
        return (
            any(abs(pdg) in quarks for pdg in ids)
            and all(abs(pdg) in quarks or pdg == 21 for pdg in ids)
            and len(entry.order) == 1
            and entry.order[0][0] == "QCD"
        )

    found = set()
    for entry in source.kernels:
        if entry.kind != "tree" or not selected(entry):
            continue
        ids = pdgs(entry)
        if not (
            len(ids) == 3
            and ids[0] == -ids[1]
            and ids[1] in quarks
            and ids[2] == 21
            and tuple(p["spin"] for p in ports(entry)) == (2, 2, 3)
            and tuple(p["color"] for p in ports(entry)) == (-3, 3, 8)
            and _equal(entry.color, "T(3,2,1)")
            and _equal(entry.lorentz, "Gamma(3,2,1)")
        ):
            raise NotImplementedError(
                "quark source ordinary tensor lacks vector-current control"
            )
        candidates = []
        for rule in payload["vertex_rules"]:
            target = tuple(particles[p] for p in rule["particles"])
            if tuple(
                tuple(p[k] for k in ("pdg_code", "spin", "color")) for p in target
            ) != tuple(
                tuple(p[k] for k in ("pdg_code", "spin", "color")) for p in ports(entry)
            ):
                continue
            for ci, color in enumerate(rule["color_structures"]):
                for li, lorentz in enumerate(rule["lorentz_structures"]):
                    coupling = rule["couplings"][ci][li]
                    if coupling is None or not _equal(color, entry.color):
                        continue
                    native_lorentz = native_model.lorentz_structure(lorentz).structure
                    if (
                        native_lorentz - wrap_indices(parse_exact_ufo("Gamma(3,1,2)"))
                    ).expand() != 0:
                        continue
                    if (
                        native_model.coupling(coupling).expression
                        - entry.scalar_value()
                    ).expand() != 0:
                        continue
                    candidates.append(
                        OrdinaryBasisRepresentative(
                            entry.key,
                            rule["name"],
                            ci,
                            li,
                            "native-qcd-qed-vector-current-control",
                        )
                    )
        if len(candidates) != 1:
            raise ValueError(
                "quark ordinary basis requires one calibrated native representative"
            )
        representatives.extend(candidates)
        found.add(ids[1])
    if found != set(quarks):
        raise ValueError("selected quark ordinary basis is missing")
    catalogue = OrdinaryBasisCatalogue(source, native_model, representatives)
    for entry in source.kernels:
        if entry.kind == "tree" or not selected(entry):
            continue
        ids = pdgs(entry)
        if catalogue.counterterm(entry) is None:
            if not (
                entry.kind == "R2"
                and len(ids) == 2
                and ids[0] == -ids[1]
                and ids[1] in quarks
                and tuple(p["spin"] for p in ports(entry)) == (2, 2)
                and tuple(p["color"] for p in ports(entry)) == (-3, 3)
                and _equal(entry.color, "Identity(1,2)")
                and _equal(entry.lorentz, "P(-1,1)*Gamma(-1,2,1)")
            ):
                raise NotImplementedError(
                    "quark counterterm lacks local conformance control"
                )
            validation = CountertermPrimitiveValidation(
                entry.key,
                entry.color,
                entry.lorentz,
                ids,
                (2, 2),
                (-3, 3),
                "local-fermion-two-point-source-aloha-control",
            )
            tensor_representatives.append(
                CountertermTensorRepresentative(
                    validation,
                    "P(-1,1)*Gamma(-1,1,2)",
                    "exact-contraction-of-calibrated-native-vector-current",
                )
            )
        entries.append(entry)
    return QCDCountertermBasis(
        tuple(representatives),
        tuple(validations),
        tuple(tensor_representatives),
        tuple(entries),
        quarks,
    )
