# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .core import (
    ANTI_PARTICLE,
    ANTIQUARKS,
    GLUONS,
    PDGS,
    QUARKS,
    SINGLETS,
    ParsedProcess,
    ProcessEnumerator,
    ProcessOptions,
    ProcessSetEntry,
    ProcessTuple,
    canonical_process_key,
    enumerate_generic_process_set,
    enumerate_process_set,
)

LegSide = Literal["initial", "final"]
ParticleClass = Literal[
    "quark",
    "antiquark",
    "gluon",
    "vector",
    "higgs",
    "charged-lepton",
    "neutrino",
    "singlet",
    "inclusive",
]

_VECTOR_NAMES = frozenset({"a", "z", "w+", "w-"})


@dataclass(frozen=True)
class ProcessLegIR:
    """One external leg with both physical and all-outgoing conventions."""

    label: int
    side: LegSide
    particle: str
    outgoing_particle: str
    pdg: int | None
    outgoing_pdg: int | None
    particle_class: ParticleClass

    @property
    def is_initial(self) -> bool:
        return self.side == "initial"

    @property
    def is_final(self) -> bool:
        return self.side == "final"

    @property
    def is_coloured(self) -> bool:
        return self.particle_class in {"quark", "antiquark", "gluon"}

    @property
    def is_singlet(self) -> bool:
        return not self.is_coloured and self.particle_class != "inclusive"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "side": self.side,
            "particle": self.particle,
            "outgoing_particle": self.outgoing_particle,
            "pdg": self.pdg,
            "outgoing_pdg": self.outgoing_pdg,
            "particle_class": self.particle_class,
        }


@dataclass(frozen=True)
class QuarkLineSummary:
    quark_count: int
    antiquark_count: int
    quark_pair_count: int

    @property
    def balanced(self) -> bool:
        return self.quark_count == self.antiquark_count

    def to_json_dict(self) -> dict[str, object]:
        return {
            "quark_count": self.quark_count,
            "antiquark_count": self.antiquark_count,
            "quark_pair_count": self.quark_pair_count,
            "balanced": self.balanced,
        }


@dataclass(frozen=True)
class CanonicalProcessIR:
    """Canonical process data used before model-driven current generation."""

    process: str
    key: str
    parsed: ParsedProcess
    color_accuracy: str
    legs: tuple[ProcessLegIR, ...]
    quark_lines: QuarkLineSummary

    @property
    def initial_legs(self) -> tuple[ProcessLegIR, ...]:
        return tuple(leg for leg in self.legs if leg.is_initial)

    @property
    def final_legs(self) -> tuple[ProcessLegIR, ...]:
        return tuple(leg for leg in self.legs if leg.is_final)

    @property
    def outgoing_particles(self) -> ProcessTuple:
        return tuple(leg.outgoing_particle for leg in self.legs)

    @property
    def outgoing_pdgs(self) -> tuple[int, ...]:
        return tuple(
            int(leg.outgoing_pdg) for leg in self.legs if leg.outgoing_pdg is not None
        )

    @property
    def initial_pdgs(self) -> tuple[int, ...]:
        return tuple(int(leg.pdg) for leg in self.initial_legs if leg.pdg is not None)

    @property
    def final_pdgs(self) -> tuple[int, ...]:
        return tuple(int(leg.pdg) for leg in self.final_legs if leg.pdg is not None)

    @property
    def gluon_labels(self) -> tuple[int, ...]:
        return self._labels_with_class("gluon")

    @property
    def quark_labels(self) -> tuple[int, ...]:
        return self._labels_with_class("quark")

    @property
    def antiquark_labels(self) -> tuple[int, ...]:
        return self._labels_with_class("antiquark")

    @property
    def singlet_labels(self) -> tuple[int, ...]:
        return tuple(leg.label for leg in self.legs if leg.is_singlet)

    @property
    def vector_labels(self) -> tuple[int, ...]:
        return self._labels_with_class("vector")

    @property
    def lepton_labels(self) -> tuple[int, ...]:
        return tuple(
            leg.label
            for leg in self.legs
            if leg.particle_class in {"charged-lepton", "neutrino"}
        )

    @property
    def higgs_labels(self) -> tuple[int, ...]:
        return self._labels_with_class("higgs")

    @property
    def has_inclusive_initial_state(self) -> bool:
        return any(leg.particle_class == "inclusive" for leg in self.initial_legs)

    @property
    def has_multiple_nonleptonic_singlets(self) -> bool:
        return (
            sum(
                1
                for leg in self.final_legs
                if leg.is_singlet
                and leg.particle_class not in {"charged-lepton", "neutrino"}
            )
            > 1
        )

    def _labels_with_class(self, particle_class: ParticleClass) -> tuple[int, ...]:
        return tuple(
            leg.label for leg in self.legs if leg.particle_class == particle_class
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process,
            "key": self.key,
            "color_accuracy": self.color_accuracy,
            "legs": [leg.to_json_dict() for leg in self.legs],
            "outgoing_particles": list(self.outgoing_particles),
            "outgoing_pdgs": list(self.outgoing_pdgs),
            "quark_lines": self.quark_lines.to_json_dict(),
            "labels": {
                "gluons": list(self.gluon_labels),
                "quarks": list(self.quark_labels),
                "antiquarks": list(self.antiquark_labels),
                "singlets": list(self.singlet_labels),
                "vectors": list(self.vector_labels),
                "leptons": list(self.lepton_labels),
                "higgs": list(self.higgs_labels),
            },
        }


@dataclass(frozen=True)
class ProcessSetEntryIR:
    key: str
    process: str
    ir: CanonicalProcessIR
    n_groups: int
    n_records: int
    n_unique_processes: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "process": self.process,
            "n_groups": self.n_groups,
            "n_records": self.n_records,
            "n_unique_processes": self.n_unique_processes,
            "ir": self.ir.to_json_dict(),
        }


@dataclass(frozen=True)
class ProcessSetIR:
    request: str
    color_accuracy: str
    entries: tuple[ProcessSetEntryIR, ...]

    @property
    def default_key(self) -> str:
        if not self.entries:
            raise ValueError("process set is empty")
        return self.entries[0].key

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request": self.request,
            "color_accuracy": self.color_accuracy,
            "default_key": self.default_key,
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


def build_process_ir(
    process: str,
    *,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
) -> CanonicalProcessIR:
    parsed = ProcessEnumerator(options).parse(process)
    physical_initial = tuple(
        particle if particle in {"p", "j"} else ANTI_PARTICLE[particle]
        for particle in parsed.initial_state
    )
    canonical = f"{' '.join(physical_initial)} > {' '.join(parsed.rest)}"
    legs = tuple(
        [
            *(
                _leg_ir(
                    label=index + 1,
                    side="initial",
                    particle=particle,
                    outgoing_particle=parsed.initial_state[index],
                )
                for index, particle in enumerate(physical_initial)
            ),
            *(
                _leg_ir(
                    label=index + 3,
                    side="final",
                    particle=particle,
                    outgoing_particle=particle,
                )
                for index, particle in enumerate(parsed.rest)
            ),
        ]
    )
    counts = Counter(leg.outgoing_particle for leg in legs)
    quark_count = sum(counts[name] for name in QUARKS)
    antiquark_count = sum(counts[name] for name in ANTIQUARKS)
    return CanonicalProcessIR(
        process=canonical,
        key=canonical_process_key(canonical),
        parsed=parsed,
        color_accuracy=color_accuracy,
        legs=legs,
        quark_lines=QuarkLineSummary(
            quark_count=quark_count,
            antiquark_count=antiquark_count,
            quark_pair_count=min(quark_count, antiquark_count),
        ),
    )


def build_process_set_ir(
    process_string: str,
    *,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
    generic: bool = False,
    max_quark_pairs: int | None = None,
) -> ProcessSetIR:
    enumeration = (
        enumerate_generic_process_set(
            process_string,
            options=options,
            max_quark_pairs=max_quark_pairs,
        )
        if generic
        else enumerate_process_set(process_string, options=options)
    )
    entries = tuple(
        _process_set_entry_ir(entry, color_accuracy=color_accuracy)
        for entry in enumeration.entries
    )
    return ProcessSetIR(
        request=enumeration.request,
        color_accuracy=color_accuracy,
        entries=entries,
    )


def _process_set_entry_ir(
    entry: ProcessSetEntry,
    *,
    color_accuracy: str,
) -> ProcessSetEntryIR:
    ir = build_process_ir(
        entry.process,
        color_accuracy=color_accuracy,
        options=entry.enumeration.options,
    )
    return ProcessSetEntryIR(
        key=entry.key,
        process=entry.process,
        ir=ir,
        n_groups=len(entry.enumeration.groups),
        n_records=entry.enumeration.n_records,
        n_unique_processes=len(entry.enumeration.unique_processes),
    )


def _leg_ir(
    *,
    label: int,
    side: LegSide,
    particle: str,
    outgoing_particle: str,
) -> ProcessLegIR:
    return ProcessLegIR(
        label=label,
        side=side,
        particle=particle,
        outgoing_particle=outgoing_particle,
        pdg=_pdg_or_none(particle),
        outgoing_pdg=_pdg_or_none(outgoing_particle),
        particle_class=_particle_class(outgoing_particle),
    )


def _pdg_or_none(particle: str) -> int | None:
    if particle in {"p", "j"}:
        return None
    return int(PDGS[particle])


def _particle_class(outgoing_particle: str) -> ParticleClass:
    if outgoing_particle in {"p", "j"}:
        return "inclusive"
    if outgoing_particle in QUARKS:
        return "quark"
    if outgoing_particle in ANTIQUARKS:
        return "antiquark"
    if outgoing_particle in GLUONS:
        return "gluon"
    if outgoing_particle in _VECTOR_NAMES:
        return "vector"
    if outgoing_particle == "h":
        return "higgs"
    pdg = abs(int(PDGS[outgoing_particle]))
    if pdg in {11, 13, 15}:
        return "charged-lepton"
    if pdg in {12, 14, 16}:
        return "neutrino"
    if outgoing_particle in SINGLETS:
        return "singlet"
    raise ValueError(f"unknown particle class for {outgoing_particle!r}")


__all__ = [
    "CanonicalProcessIR",
    "ParticleClass",
    "ProcessLegIR",
    "ProcessSetEntryIR",
    "ProcessSetIR",
    "QuarkLineSummary",
    "build_process_ir",
    "build_process_set_ir",
]
