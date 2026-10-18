# SPDX-License-Identifier: 0BSD
"""Explicit correspondence between source UFO and calibrated ordinary kernels.

The native bundled models and standard UFO sources can use different spinor
slots. This module does not infer a transpose, a gamma-five scheme, or a new
tensor convention. An explicitly named ordinary representative may be reused by
a CT only after proving exact source-basis identity at identical ordered ports.
The caller's calibration label is provenance, not a new physics certificate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .exact import require_exact_numbers
from .model import KernelBasisEntry, LoopModelIR
from .model_expressions import parse_exact_ufo
from .orders import canonical_order


@dataclass(frozen=True, slots=True)
class OrdinaryBasisRepresentative:
    source_key: tuple[Any, ...]
    native_rule: str
    color_index: int
    lorentz_index: int
    calibration: str


@dataclass(frozen=True, slots=True)
class CountertermBasisCorrespondence:
    counterterm_key: tuple[Any, ...]
    representative: OrdinaryBasisRepresentative
    color: str
    lorentz: str


@dataclass(frozen=True, slots=True)
class CountertermPrimitiveValidation:
    """Caller-supplied evidence scope for a tested source local tensor.

    This record never changes its tensor. It binds an independent conformance
    control to exact source expressions and ordered particle representations;
    its label is provenance, not an execution of that control or a scheme proof.
    Scalar coefficients and other basis entries acquire no validation from it.
    """

    source_key: tuple[Any, ...]
    color: str
    lorentz: str
    particle_pdgs: tuple[int, ...]
    particle_spins: tuple[int, ...]
    particle_colors: tuple[int, ...]
    conformance_control: str

    def __post_init__(self):
        for name in (
            "source_key",
            "particle_pdgs",
            "particle_spins",
            "particle_colors",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not isinstance(self.conformance_control, str)
            or not self.conformance_control.strip()
        ):
            raise ValueError("primitive validation requires conformance provenance")


@dataclass(frozen=True, slots=True)
class CountertermTensorRepresentative:
    """Explicit native Lorentz representative of a controlled source primitive.

    The source evidence remains unchanged. This separate record authorizes one
    exact replacement in the supplied native model's convention; it is not a
    universal spinor transpose. The caller must establish the local relation
    and separately validate its process insertion, routing and scalar phase.
    """

    source_validation: CountertermPrimitiveValidation
    native_lorentz: str
    calibration: str

    def __post_init__(self):
        if not isinstance(self.calibration, str) or not self.calibration.strip():
            raise ValueError("tensor representative requires calibration provenance")
        if not isinstance(self.native_lorentz, str) or not self.native_lorentz.strip():
            raise ValueError(
                "tensor representative requires a native Lorentz expression"
            )
        require_exact_numbers(parse_exact_ufo(self.native_lorentz))


def validate_counterterm_primitives(
    source: LoopModelIR,
    validations: Iterable[CountertermPrimitiveValidation],
) -> dict[tuple[Any, ...], CountertermPrimitiveValidation]:
    """Reject stale or overly broad local-tensor evidence before model loading."""
    entries = {entry.key: entry for entry in source.kernels if entry.kind != "tree"}
    result = {}
    for validation in validations:
        if validation.source_key in result:
            raise ValueError("duplicate source counterterm primitive validation")
        entry = entries.get(validation.source_key)
        if entry is None:
            raise ValueError(
                "primitive validation must bind a source counterterm entry"
            )
        particles = [
            source.raw.collections["particles"][i]["fields"] for i in entry.particle_ids
        ]
        for name, field_name in (
            ("particle_pdgs", "pdg_code"),
            ("particle_spins", "spin"),
            ("particle_colors", "color"),
        ):
            values = getattr(validation, name)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ) or values != tuple(p[field_name] for p in particles):
                raise ValueError("primitive validation ordered particle ports differ")
        for name in ("color", "lorentz"):
            if (
                parse_exact_ufo(getattr(validation, name))
                - parse_exact_ufo(getattr(entry, name))
            ).expand() != 0:
                raise ValueError(
                    f"primitive validation source {name} expression differs"
                )
        result[entry.key] = validation
    return result


@dataclass(frozen=True, slots=True)
class _OrdinaryBasis:
    source: KernelBasisEntry
    representative: OrdinaryBasisRepresentative
    pdgs: tuple[int, ...]
    color: str
    lorentz: str
    source_color: Any = field(repr=False)
    source_lorentz: Any = field(repr=False)


def _pdgs(source: LoopModelIR, entry: KernelBasisEntry):
    return tuple(
        source.raw.collections["particles"][index]["fields"]["pdg_code"]
        for index in entry.particle_ids
    )


class OrdinaryBasisCatalogue:
    """Validate explicit ordinary representatives, then prove CT source identity."""

    def __init__(
        self,
        source: LoopModelIR,
        native_model: Any,
        representatives: Iterable[OrdinaryBasisRepresentative],
    ):
        self.source = source
        payload = json.loads(native_model.to_json())
        rules = {rule["name"]: rule for rule in payload["vertex_rules"]}
        particles = {particle["name"]: particle for particle in payload["particles"]}
        lorentz = {item["name"]: item for item in payload["lorentz_structures"]}
        source_entries = {
            entry.key: entry for entry in source.kernels if entry.kind == "tree"
        }
        seen, bases = set(), []
        for representative in representatives:
            if (
                not isinstance(representative.calibration, str)
                or not representative.calibration.strip()
            ):
                raise ValueError(
                    "ordinary representative requires calibration provenance"
                )
            if representative.source_key in seen:
                raise ValueError("duplicate source ordinary representative")
            seen.add(representative.source_key)
            if representative.source_key not in source_entries:
                raise ValueError(
                    "representative must bind a source ordinary basis entry"
                )
            source_entry = source_entries[representative.source_key]
            rule = rules.get(representative.native_rule)
            if rule is None:
                raise ValueError("representative native ordinary rule is absent")
            ci, li = representative.color_index, representative.lorentz_index
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in (ci, li)
            ):
                raise ValueError(
                    "ordinary representative indices must be nonnegative integers"
                )
            if ci >= len(rule["color_structures"]) or li >= len(
                rule["lorentz_structures"]
            ):
                raise ValueError("ordinary representative basis slot is absent")
            coupling_name = rule["couplings"][ci][li]
            if coupling_name is None:
                raise ValueError("ordinary representative basis slot has no coupling")
            source_particles = [
                source.raw.collections["particles"][i]["fields"]
                for i in source_entry.particle_ids
            ]
            target_particles = [particles[name] for name in rule["particles"]]
            if len(source_particles) != len(target_particles) or any(
                left[key] != right[key]
                for left, right in zip(source_particles, target_particles, strict=True)
                for key in ("pdg_code", "spin", "color")
            ):
                raise ValueError(
                    "ordinary representative ordered particle ports differ"
                )
            coupling = native_model.coupling(coupling_name)
            require_exact_numbers(coupling.expression)
            if canonical_order(tuple(coupling.orders.items())) != source_entry.order:
                raise ValueError("ordinary representative coupling order differs")
            if (
                coupling.expression - source_entry.coupling.value.expression
            ).expand() != 0:
                raise ValueError(
                    "ordinary representative exact coupling expression differs"
                )
            target_color = native_model.vertex_rule(
                representative.native_rule
            ).color_structures[ci]
            source_color = parse_exact_ufo(source_entry.color)
            require_exact_numbers(target_color)
            if (target_color - source_color).expand() != 0:
                raise ValueError(
                    "ordinary representative exact color expression differs"
                )
            target_lorentz = lorentz[rule["lorentz_structures"][li]]
            require_exact_numbers(
                native_model.lorentz_structure(target_lorentz["name"]).structure
            )
            if tuple(target_lorentz["spins"]) != tuple(
                p["spin"] for p in target_particles
            ):
                raise ValueError(
                    "ordinary representative Lorentz spin signature differs"
                )
            bases.append(
                _OrdinaryBasis(
                    source_entry,
                    representative,
                    _pdgs(source, source_entry),
                    target_color.to_canonical_string(),
                    target_lorentz["structure"],
                    source_color,
                    parse_exact_ufo(source_entry.lorentz),
                )
            )
        self._bases = tuple(bases)

    def counterterm(
        self, entry: KernelBasisEntry
    ) -> CountertermBasisCorrespondence | None:
        """Return a representative only for an exactly identical source basis."""
        if entry.kind == "tree" or not any(
            entry is candidate for candidate in self.source.kernels
        ):
            raise ValueError("entry must be a counterterm of the source model")
        pdgs = _pdgs(self.source, entry)
        color, lorentz = parse_exact_ufo(entry.color), parse_exact_ufo(entry.lorentz)
        matches = [
            basis
            for basis in self._bases
            if basis.pdgs == pdgs
            and (basis.source_color - color).expand() == 0
            and (basis.source_lorentz - lorentz).expand() == 0
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous source ordinary basis correspondence")
        if not matches:
            return None
        basis = matches[0]
        return CountertermBasisCorrespondence(
            entry.key,
            basis.representative,
            basis.color,
            basis.lorentz,
        )
