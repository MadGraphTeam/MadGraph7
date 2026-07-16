# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from ..models.contracts import (
    CompiledModelIR,
    CompiledParticleRecord,
    validate_color_representation,
)
from .core import (
    ParsedProcess,
    ProcessOptions,
    canonical_process_key,
    split_process_set,
)
from .ir import (
    CanonicalProcessIR,
    LegSide,
    ParticleClass,
    ProcessLegIR,
    QuarkLineSummary,
)


@dataclass(frozen=True)
class ModelParticleCatalog:
    model_name: str
    particles: tuple[CompiledParticleRecord, ...]

    def __post_init__(self) -> None:
        names = [particle.name for particle in self.particles]
        if len(names) != len(set(names)):
            raise ValueError("compiled model contains duplicate particle names")
        for particle in self.particles:
            validate_color_representation(
                particle.color,
                context=f"particle {particle.name!r}",
            )

    @property
    def by_name(self) -> dict[str, CompiledParticleRecord]:
        return {particle.name: particle for particle in self.particles}

    @property
    def external_particles(self) -> tuple[CompiledParticleRecord, ...]:
        return tuple(
            particle
            for particle in self.particles
            if particle.spin > 0
            and particle.ghost_number == 0
            and not particle.goldstoneboson
            and particle.propagating
        )

    def resolve(self, name: str, *, external: bool = True) -> CompiledParticleRecord:
        candidates = self.external_particles if external else self.particles
        exact = next(
            (particle for particle in candidates if particle.name == name), None
        )
        if exact is not None:
            return exact
        folded = [
            particle
            for particle in candidates
            if particle.name.casefold() == name.casefold()
        ]
        if len(folded) == 1:
            return folded[0]
        if len(folded) > 1:
            names = ", ".join(sorted(particle.name for particle in folded))
            raise ValueError(f"particle name {name!r} is case-ambiguous: {names}")
        raise ValueError(
            f"particle {name!r} is not an external state in {self.model_name}"
        )

    def antiparticle(self, particle: CompiledParticleRecord) -> CompiledParticleRecord:
        try:
            return self.by_name[particle.antiname]
        except KeyError as exc:
            raise ValueError(
                f"particle {particle.name!r} refers to absent antiparticle "
                f"{particle.antiname!r}"
            ) from exc

    def particle_class(self, particle: CompiledParticleRecord) -> ParticleClass:
        representation = validate_color_representation(
            particle.color,
            context=f"particle {particle.name!r}",
        )
        if representation == 8:
            return "gluon"
        if representation == 3:
            return "quark"
        if representation == -3:
            return "antiquark"
        if particle.spin == 3:
            return "vector"
        if self.model_name in {"sm", "built-in-sm"}:
            absolute_pdg = abs(particle.pdg_code)
            if absolute_pdg in {11, 13, 15}:
                return "charged-lepton"
            if absolute_pdg in {12, 14, 16}:
                return "neutrino"
            if absolute_pdg == 25:
                return "higgs"
        return "singlet"

    def default_multiparticles(self) -> dict[str, tuple[str, ...]]:
        if self.model_name not in {"sm", "built-in-sm"}:
            return {}
        available = {particle.name for particle in self.external_particles}
        preferred = (
            "g",
            "d",
            "d~",
            "u",
            "u~",
            "s",
            "s~",
            "c",
            "c~",
            "b",
            "b~",
        )
        values = tuple(name for name in preferred if name in available)
        return {"p": values, "j": values}


@dataclass(frozen=True)
class ModelProcessSetEntry:
    key: str
    process: str
    ir: CanonicalProcessIR


@dataclass(frozen=True)
class ModelProcessSet:
    request: str
    options: ProcessOptions
    entries: tuple[ModelProcessSetEntry, ...]


def parse_multiparticle_definitions(
    definitions: Sequence[str],
    catalog: ModelParticleCatalog,
) -> dict[str, tuple[str, ...]]:
    result = dict(catalog.default_multiparticles())
    for definition in definitions:
        name, separator, raw_items = definition.partition("=")
        name = name.strip()
        if (
            not separator
            or not name
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        ):
            raise ValueError(
                f"invalid multiparticle definition {definition!r}; "
                "expected NAME=ITEM,ITEM"
            )
        items = tuple(item.strip() for item in raw_items.split(",") if item.strip())
        if not items:
            raise ValueError(f"multiparticle {name!r} has no members")
        resolved = tuple(catalog.resolve(item).name for item in items)
        result[name] = tuple(dict.fromkeys(resolved))
    return result


def expand_model_processes(
    process_set: str,
    catalog: ModelParticleCatalog,
    *,
    multiparticles: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    aliases = {
        name: tuple(values)
        for name, values in (multiparticles or catalog.default_multiparticles()).items()
    }
    expanded: dict[str, None] = {}
    for process in split_process_set(process_set):
        initial_text, separator, final_text = process.partition(">")
        if not separator or ">" in final_text:
            raise ValueError("invalid collision format; expected 'initial > final'")
        initial_slots = _expanded_slots(initial_text, aliases, catalog)
        final_slots = _expanded_slots(final_text, aliases, catalog)
        if len(initial_slots) != 2:
            raise ValueError(
                "pyAmpliCol processes require exactly two initial particles"
            )
        if not final_slots:
            raise ValueError("pyAmpliCol processes require at least one final particle")
        for initial in itertools.product(*initial_slots):
            for final in itertools.product(*final_slots):
                concrete = f"{' '.join(initial)} > {' '.join(final)}"
                expanded.setdefault(concrete, None)
    return tuple(expanded)


def build_model_process_ir(
    process: str,
    model: CompiledModelIR,
    *,
    color_accuracy: str = "lc",
) -> CanonicalProcessIR:
    catalog = ModelParticleCatalog(model.name, model.particles)
    initial_text, separator, final_text = process.partition(">")
    if not separator or ">" in final_text:
        raise ValueError("invalid collision format; expected 'initial > final'")
    initial = tuple(catalog.resolve(token) for token in initial_text.split())
    final = tuple(catalog.resolve(token) for token in final_text.split())
    if len(initial) != 2 or not final:
        raise ValueError(
            "process must have exactly two initial and at least one final particle"
        )
    outgoing_initial = tuple(catalog.antiparticle(particle) for particle in initial)
    canonical = f"{' '.join(particle.name for particle in initial)} > " + " ".join(
        particle.name for particle in final
    )
    legs = tuple(
        [
            *(
                _model_leg(
                    label=index + 1,
                    side="initial",
                    particle=particle,
                    outgoing_particle=outgoing_initial[index],
                    catalog=catalog,
                )
                for index, particle in enumerate(initial)
            ),
            *(
                _model_leg(
                    label=index + 3,
                    side="final",
                    particle=particle,
                    outgoing_particle=particle,
                    catalog=catalog,
                )
                for index, particle in enumerate(final)
            ),
        ]
    )
    counts = Counter(leg.particle_class for leg in legs)
    parsed = ParsedProcess(
        initial_state=tuple(particle.name for particle in outgoing_initial),
        jet_count=0,
        rest=tuple(particle.name for particle in final),
    )
    return CanonicalProcessIR(
        process=canonical,
        key=canonical_process_key(canonical),
        parsed=parsed,
        color_accuracy=color_accuracy,
        legs=legs,
        quark_lines=QuarkLineSummary(
            quark_count=counts["quark"],
            antiquark_count=counts["antiquark"],
            quark_pair_count=min(counts["quark"], counts["antiquark"]),
        ),
    )


def _expanded_slots(
    text: str,
    aliases: Mapping[str, Sequence[str]],
    catalog: ModelParticleCatalog,
) -> list[tuple[str, ...]]:
    slots: list[tuple[str, ...]] = []
    for token in text.split():
        count, name = _repetition(token)
        if name in aliases:
            options = tuple(catalog.resolve(item).name for item in aliases[name])
        else:
            options = (catalog.resolve(name).name,)
        slots.extend(options for _ in range(count))
    return slots


def _repetition(token: str) -> tuple[int, str]:
    match = re.fullmatch(r"(?:(\d+)\*)?(.+)", token)
    if match is None:
        raise ValueError(f"invalid process token {token!r}")
    count = int(match.group(1) or 1)
    if count < 1:
        raise ValueError(f"process repetition must be positive: {token!r}")
    return count, match.group(2)


def _model_leg(
    *,
    label: int,
    side: str,
    particle: CompiledParticleRecord,
    outgoing_particle: CompiledParticleRecord,
    catalog: ModelParticleCatalog,
) -> ProcessLegIR:
    return ProcessLegIR(
        label=label,
        side=cast(LegSide, side),
        particle=particle.name,
        outgoing_particle=outgoing_particle.name,
        pdg=particle.pdg_code,
        outgoing_pdg=outgoing_particle.pdg_code,
        particle_class=catalog.particle_class(outgoing_particle),
    )


__all__ = [
    "ModelParticleCatalog",
    "ModelProcessSet",
    "ModelProcessSetEntry",
    "build_model_process_ir",
    "expand_model_processes",
    "parse_multiparticle_definitions",
]
