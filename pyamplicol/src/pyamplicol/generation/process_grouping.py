# SPDX-License-Identifier: 0BSD
"""Flavour-profile canonicalization and process-group maps.

Two processes are candidates for one matrix element when their external legs
have the same particle types. A particle type is the spin, the SU(3) dimension,
the resolved mass and width and self-conjugacy; particles and antiparticles
share a type. The leg mapping between two candidates is derived from flavour
profiles: each flavour of a process is described by how many times it appears
as a particle and as an antiparticle in the initial and in the final state.
Sorting legs by profile, and only then by PDG, gives positions that follow any
flavour relabelling, which sorting by PDG alone does not.

A group map records, for each representative, the members whose matrix
element equals ``factor`` times the representative's at permuted momenta:
``ME_member(p) = factor * ME_representative(q)`` with
``q[r] = p[permutation[r]]``, so ``permutation`` maps representative legs to
member legs. ``conjugated`` members are related by charge conjugation, so LC
colour-flow words map with reversed order.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any, Protocol

from .._internal.physics.rationals import simplest_rational_between

PROCESS_GROUPS_PATH = "process-groups.json"
PROCESS_GROUPS_ROLE = "process-groups"
PROCESS_GROUPS_KIND = "pyamplicol-process-groups"
PROCESS_GROUPS_SCHEMA_VERSION = 1

_ZERO_EXPRESSIONS = frozenset({"0", "0.0", "-0", "-0.0", "zero"})


class GroupingBasis(StrEnum):
    IDENTITY = "identity"
    EXACT_LOOP_FLAVOR = "exact-loop-flavor"
    FLAVOR_BLIND = "flavor-blind"
    NUMERICAL = "numerical"
    FLAVOUR_UNION = "flavour-union"


# The least verified link decides the basis of a composed relation.
_BASIS_STRENGTH = {
    GroupingBasis.IDENTITY: 0,
    GroupingBasis.EXACT_LOOP_FLAVOR: 1,
    GroupingBasis.NUMERICAL: 2,
    GroupingBasis.FLAVOR_BLIND: 3,
    GroupingBasis.FLAVOUR_UNION: 4,
}


class GroupingAnalysis(StrEnum):
    NONE = "none"
    NUMERICAL = "numerical"


@dataclass(frozen=True, order=True, slots=True)
class ParticleType:
    """Everything a flavour relabelling must preserve about one particle."""

    spin: int
    color: int
    mass: str
    width: str
    self_conjugate: bool


@dataclass(frozen=True, slots=True)
class GroupingParticle:
    pdg: int
    name: str
    type: ParticleType


ParticleTable = Mapping[int, GroupingParticle]


@dataclass(frozen=True, slots=True)
class GroupingProcess:
    process_id: str
    expression: str
    pdgs: tuple[int, ...]
    initial_count: int
    generated: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.initial_count < len(self.pdgs):
            raise ValueError(
                f"process {self.process_id!r} needs initial and final legs"
            )


def resolved_parameter_name(
    name: str,
    parameters: Mapping[str, object],
) -> str:
    """Return ``ZERO`` for compile-time zero parameters, otherwise ``name``."""

    if not name or name.strip().casefold() == "zero":
        return "ZERO"
    record = parameters.get(name)
    resolved = getattr(record, "resolved_expression", None)
    if isinstance(resolved, str) and resolved.strip().casefold() in _ZERO_EXPRESSIONS:
        return "ZERO"
    return name


def particle_table(
    particles: Iterable[object],
    parameters: Mapping[str, object],
) -> dict[int, GroupingParticle]:
    """Build the grouping view of compiled particle records.

    ``particles`` are ``CompiledParticleRecord``-like objects and
    ``parameters`` maps parameter names to records with a
    ``resolved_expression``, as in ``CompiledModelIR``.
    """

    table: dict[int, GroupingParticle] = {}
    for particle in particles:
        pdg = int(particle.pdg_code)  # type: ignore[attr-defined]
        name = str(particle.name)  # type: ignore[attr-defined]
        antiname = str(particle.antiname)  # type: ignore[attr-defined]
        declared = getattr(particle, "self_conjugate", None)
        self_conjugate = bool(declared) if declared is not None else name == antiname
        table[pdg] = GroupingParticle(
            pdg=pdg,
            name=name,
            type=ParticleType(
                spin=int(particle.spin),  # type: ignore[attr-defined]
                color=abs(int(particle.color)),  # type: ignore[attr-defined]
                mass=resolved_parameter_name(
                    str(particle.mass),  # type: ignore[attr-defined]
                    parameters,
                ),
                width=resolved_parameter_name(
                    str(particle.width),  # type: ignore[attr-defined]
                    parameters,
                ),
                self_conjugate=self_conjugate,
            ),
        )
    return table


@dataclass(frozen=True, slots=True)
class _Leg:
    index: int
    initial: bool
    pdg: int
    flavour: int
    orientation: int
    type: ParticleType


def _particle(table: ParticleTable, pdg: int) -> GroupingParticle:
    try:
        return table[pdg]
    except KeyError as error:
        raise ValueError(f"PDG code {pdg} is absent from the model") from error


def _legs(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
) -> tuple[_Leg, ...]:
    legs = []
    for index, pdg in enumerate(process.pdgs):
        particle = _particle(table, pdg)
        if particle.type.self_conjugate:
            flavour, orientation = abs(pdg), 0
        else:
            flavour, orientation = abs(pdg), (1 if pdg > 0 else -1)
            if conjugate:
                orientation = -orientation
        legs.append(
            _Leg(
                index=index,
                initial=index < process.initial_count,
                pdg=pdg if not conjugate or orientation == 0 else -pdg,
                flavour=flavour,
                orientation=orientation,
                type=particle.type,
            )
        )
    return tuple(legs)


Profile = tuple[int, int, int, int]


def _profiles(legs: Sequence[_Leg]) -> dict[int, Profile]:
    counts: dict[int, list[int]] = {}
    for leg in legs:
        slot = (0 if leg.initial else 2) + (1 if leg.orientation < 0 else 0)
        counts.setdefault(leg.flavour, [0, 0, 0, 0])[slot] += 1
    return {flavour: tuple(values) for flavour, values in counts.items()}  # type: ignore[misc]


def coarse_key(process: GroupingProcess, table: ParticleTable) -> tuple[Any, ...]:
    """Particle types of each side, with particles and antiparticles merged."""

    legs = _legs(process, table)
    return (
        process.initial_count,
        tuple(sorted(leg.type for leg in legs if leg.initial)),
        tuple(sorted(leg.type for leg in legs if not leg.initial)),
    )


def profile_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
) -> tuple[Any, ...]:
    """Equal for exactly the processes related by a type-preserving relabelling."""

    legs = _legs(process, table, conjugate=conjugate)
    types = {leg.flavour: leg.type for leg in legs}
    profiles = _profiles(legs)
    return (
        process.initial_count,
        len(legs),
        tuple(
            sorted((types[flavour], profile) for flavour, profile in profiles.items())
        ),
    )


def flavor_class_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> tuple[Any, ...]:
    """Class of processes assumed equal by ``--flavor-blind``."""

    key = profile_key(process, table)
    if consider_charge_conjugation:
        key = min(key, profile_key(process, table, conjugate=True))
    return key


def _ordered_legs(
    legs: Sequence[_Leg],
    ranks: Mapping[int, int],
    profiles: Mapping[int, Profile],
) -> tuple[_Leg, ...]:
    return tuple(
        sorted(
            legs,
            key=lambda leg: (
                not leg.initial,
                leg.type,
                profiles[leg.flavour],
                ranks[leg.flavour],
                -leg.orientation,
                leg.index,
            ),
        )
    )


def _flavour_groups(legs: Sequence[_Leg]) -> dict[tuple[Any, ...], list[int]]:
    """Flavours with equal (type, profile), in a fixed order."""

    types = {leg.flavour: leg.type for leg in legs}
    profiles = _profiles(legs)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for flavour in sorted(profiles):
        groups.setdefault((types[flavour], profiles[flavour]), []).append(flavour)
    return groups


def _profile_order(
    legs: Sequence[_Leg],
    assignment: Mapping[tuple[Any, ...], Sequence[int]],
) -> tuple[_Leg, ...]:
    ranks = {
        flavour: rank
        for flavours in assignment.values()
        for rank, flavour in enumerate(flavours)
    }
    return _ordered_legs(legs, ranks, _profiles(legs))


@dataclass(frozen=True, slots=True)
class LegMapping:
    """One candidate relation ``member -> representative``."""

    permutation: tuple[int, ...]
    conjugated: bool
    method: str


def _mapping(
    member: Sequence[_Leg],
    representative: Sequence[_Leg],
    *,
    conjugated: bool,
    method: str,
) -> LegMapping:
    permutation = [0] * len(representative)
    for member_leg, representative_leg in zip(member, representative, strict=True):
        if (
            member_leg.initial != representative_leg.initial
            or member_leg.type != representative_leg.type
            or member_leg.orientation != representative_leg.orientation
        ):
            raise ValueError("leg orders do not describe the same process shape")
        permutation[representative_leg.index] = member_leg.index
    return LegMapping(tuple(permutation), conjugated, method)


def profile_mappings(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    conjugated: bool = False,
    limit: int | None = None,
) -> tuple[LegMapping, ...]:
    """Leg mappings induced by flavour relabellings: all of them, or the
    first ``limit`` (``flavor_blind_mapping`` takes the first only).

    Flavours with the same type and profile can be paired in any order; each
    pairing is one mapping, the first one pairs them by increasing ``|PDG|``.
    """

    member_legs = _legs(member, table, conjugate=conjugated)
    representative_legs = _legs(representative, table)
    if profile_key(member, table, conjugate=conjugated) != profile_key(
        representative, table
    ):
        return ()
    representative_order = _profile_order(
        representative_legs, _flavour_groups(representative_legs)
    )
    member_groups = _flavour_groups(member_legs)
    keys = list(member_groups)
    choices = [itertools.permutations(member_groups[key]) for key in keys]
    mappings: dict[LegMapping, None] = {}
    for combination in itertools.product(*choices):
        member_order = _profile_order(
            member_legs, dict(zip(keys, combination, strict=True))
        )
        mapping = _mapping(
            member_order,
            representative_order,
            conjugated=conjugated,
            method="flavor-profile",
        )
        mappings.setdefault(mapping, None)
        if limit is not None and len(mappings) >= limit:
            break
    return tuple(mappings)


def signed_pdg_mapping(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
) -> LegMapping | None:
    """Pair legs by sorting each side by type and signed PDG."""

    if coarse_key(member, table) != coarse_key(representative, table):
        return None

    def order(process: GroupingProcess) -> tuple[_Leg, ...]:
        return tuple(
            sorted(
                _legs(process, table),
                key=lambda leg: (not leg.initial, leg.type, leg.pdg, leg.index),
            )
        )

    permutation = [0] * len(representative.pdgs)
    for member_leg, representative_leg in zip(
        order(member), order(representative), strict=True
    ):
        permutation[representative_leg.index] = member_leg.index
    return LegMapping(tuple(permutation), False, "signed-pdg")


def candidate_mappings(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> tuple[LegMapping, ...]:
    """All leg mappings worth testing numerically, most natural first: every
    flavour relabelling (then its charge conjugate) and the signed-PDG one."""

    mappings = list(profile_mappings(member, representative, table))
    if consider_charge_conjugation:
        mappings.extend(
            profile_mappings(member, representative, table, conjugated=True)
        )
    fallback = signed_pdg_mapping(member, representative, table)
    if fallback is not None and all(
        (fallback.permutation, False) != (mapping.permutation, mapping.conjugated)
        for mapping in mappings
    ):
        mappings.append(fallback)
    return tuple(mappings)


def flavor_blind_mapping(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> LegMapping:
    """The mapping assumed by ``--flavor-blind`` within one flavour class."""

    mappings = profile_mappings(member, representative, table, limit=1)
    if not mappings and consider_charge_conjugation:
        mappings = profile_mappings(
            member, representative, table, conjugated=True, limit=1
        )
    if not mappings:
        raise ValueError(
            f"{member.expression!r} and {representative.expression!r} are not "
            "related by a flavour relabelling"
        )
    return replace(mappings[0], method="flavor-blind")


def representative_key(
    process: GroupingProcess,
    table: ParticleTable,
) -> tuple[Any, ...]:
    """Canonical order of processes: type signature, then PDGs.

    Smaller ``|PDG|`` comes first and particles precede antiparticles, so
    ``d d~ > g g`` represents ``u u~ > g g`` and ``u g > u g`` represents
    ``u~ g > u~ g``.
    """

    legs = sorted(
        _legs(process, table),
        key=lambda leg: (
            not leg.initial,
            leg.type,
            abs(leg.pdg),
            leg.pdg < 0,
            leg.index,
        ),
    )
    return (
        coarse_key(process, table),
        tuple((leg.initial, abs(leg.pdg), leg.pdg < 0) for leg in legs),
        process.process_id,
    )


def flavor_blind_classes(
    processes: Sequence[GroupingProcess],
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> list[list[GroupingProcess]]:
    """Partition ``processes`` into flavour classes, representative first."""

    classes: dict[tuple[Any, ...], list[GroupingProcess]] = {}
    for process in processes:
        key = flavor_class_key(
            process,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
        )
        classes.setdefault(key, []).append(process)
    result = []
    for members in classes.values():
        members.sort(key=lambda process: representative_key(process, table))
        result.append(members)
    return result


# --------------------------------------------------------------------------
# Relations and group maps


def identity_permutation(count: int) -> tuple[int, ...]:
    return tuple(range(count))


def compose_permutations(
    member_to_middle: Sequence[int],
    middle_to_representative: Sequence[int],
) -> tuple[int, ...]:
    """Compose ``member -> middle`` with ``middle -> representative``."""

    return tuple(member_to_middle[index] for index in middle_to_representative)


def invert_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for representative_index, member_index in enumerate(permutation):
        inverse[member_index] = representative_index
    return tuple(inverse)


@dataclass(frozen=True, slots=True)
class GroupMember:
    process_id: str
    expression: str
    pdgs: tuple[int, ...]
    initial_count: int
    factor: Fraction = Fraction(1)
    permutation: tuple[int, ...] = ()
    conjugated: bool = False
    basis: GroupingBasis = GroupingBasis.IDENTITY
    generated: bool = True
    warning: str | None = None
    # Flavour-union member: template pairing-rule ids summed before squaring.
    patterns: tuple[int, ...] | None = None
    pattern_signs: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.patterns is None) != (self.pattern_signs is None):
            raise ValueError("patterns and pattern_signs must be given together")
        if self.patterns is not None and self.pattern_signs is not None:
            if not self.patterns or len(self.patterns) != len(self.pattern_signs):
                raise ValueError(
                    f"process-group member {self.process_id!r} needs one sign "
                    "per flavour-union pattern"
                )
            if len(set(self.patterns)) != len(self.patterns) or min(self.patterns) < 0:
                raise ValueError(
                    f"process-group member {self.process_id!r} has invalid "
                    f"flavour-union patterns {self.patterns!r}"
                )
            if any(sign not in (-1, 1) for sign in self.pattern_signs):
                raise ValueError(
                    f"process-group member {self.process_id!r} has flavour-union "
                    "pattern signs other than +1 and -1"
                )

    @classmethod
    def identity(cls, process: GroupingProcess) -> GroupMember:
        return cls(
            process_id=process.process_id,
            expression=process.expression,
            pdgs=process.pdgs,
            initial_count=process.initial_count,
            permutation=identity_permutation(len(process.pdgs)),
            generated=process.generated,
        )

    @property
    def process(self) -> GroupingProcess:
        return GroupingProcess(
            process_id=self.process_id,
            expression=self.expression,
            pdgs=self.pdgs,
            initial_count=self.initial_count,
            generated=self.generated,
        )

    def related(
        self,
        mapping: LegMapping,
        *,
        factor: Fraction,
        basis: GroupingBasis,
        warning: str | None = None,
    ) -> GroupMember:
        return replace(
            self,
            factor=Fraction(factor),
            permutation=tuple(mapping.permutation),
            conjugated=mapping.conjugated,
            basis=basis,
            warning=warning,
        )

    def through(self, middle: GroupMember) -> GroupMember:
        """Re-express ``self -> A`` through ``A -> B`` as ``self -> B``.

        ``middle`` is the relation of the old representative ``A`` to the new
        representative ``B``.
        """

        if self.patterns is not None or middle.patterns is not None:
            raise ValueError("flavour-union relations do not compose")
        return replace(
            self,
            factor=self.factor * middle.factor,
            permutation=compose_permutations(self.permutation, middle.permutation),
            conjugated=self.conjugated != middle.conjugated,
            basis=max(self.basis, middle.basis, key=_BASIS_STRENGTH.__getitem__),
            warning=self.warning or middle.warning,
        )

    def inverted_to(self, representative: GroupMember) -> GroupMember:
        """Relation of ``representative`` to ``self``, the inverse of ``self``."""

        if self.patterns is not None:
            raise ValueError("flavour-union relations are not invertible")
        return replace(
            representative,
            factor=1 / self.factor,
            permutation=invert_permutation(self.permutation),
            conjugated=self.conjugated,
            basis=self.basis,
            warning=self.warning,
        )

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "process_id": self.process_id,
            "process": self.expression,
            "pdgs": list(self.pdgs),
            "initial_count": self.initial_count,
            "factor": {
                "numerator": self.factor.numerator,
                "denominator": self.factor.denominator,
            },
            "permutation": list(self.permutation),
            "conjugated": self.conjugated,
            "basis": str(self.basis),
            "generated": self.generated,
        }
        if self.warning:
            record["warning"] = self.warning
        if self.patterns is not None and self.pattern_signs is not None:
            record["flavour_union_patterns"] = list(self.patterns)
            record["flavour_union_pattern_signs"] = list(self.pattern_signs)
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GroupMember:
        factor = value["factor"]
        if not isinstance(factor, Mapping):
            raise ValueError("process-group factor must be an object")
        return cls(
            process_id=str(value["process_id"]),
            expression=str(value["process"]),
            pdgs=tuple(int(pdg) for pdg in value["pdgs"]),  # type: ignore[attr-defined]
            initial_count=int(value["initial_count"]),  # type: ignore[call-overload]
            factor=Fraction(
                int(factor["numerator"]),
                int(factor["denominator"]),
            ),
            permutation=tuple(int(i) for i in value["permutation"]),  # type: ignore[attr-defined]
            conjugated=bool(value["conjugated"]),
            basis=GroupingBasis(str(value["basis"])),
            generated=bool(value["generated"]),
            warning=(str(value["warning"]) if value.get("warning") else None),
            patterns=_optional_int_tuple(value.get("flavour_union_patterns")),
            pattern_signs=_optional_int_tuple(value.get("flavour_union_pattern_signs")),
        )


def _optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("flavour-union patterns and signs must be integer lists")
    return tuple(value)


@dataclass(slots=True)
class ProcessGroup:
    """A representative and every member, the representative first."""

    members: list[GroupMember]

    @property
    def representative(self) -> GroupMember:
        return self.members[0]

    def as_dict(self) -> dict[str, object]:
        representative = self.representative
        return {
            "representative": {
                "process_id": representative.process_id,
                "process": representative.expression,
            },
            "members": [member.as_dict() for member in self.members],
        }


@dataclass(slots=True)
class ProcessGroupMap:
    """Process groups keyed by representative process ID."""

    groups: dict[str, ProcessGroup] = field(default_factory=dict)
    analyses: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def trivial(cls, processes: Iterable[GroupingProcess]) -> ProcessGroupMap:
        result = cls()
        for process in processes:
            result.add_group([GroupMember.identity(process)])
        return result

    def add_group(self, members: Sequence[GroupMember]) -> None:
        representative = members[0]
        if not representative.generated:
            raise ValueError(
                f"representative {representative.process_id!r} was not generated"
            )
        for member in members:
            if self.group_of(member.process_id) is not None:
                raise ValueError(
                    f"process {member.process_id!r} already belongs to a group"
                )
        self.groups[representative.process_id] = ProcessGroup(list(members))

    def group_of(self, process_id: str) -> ProcessGroup | None:
        for group in self.groups.values():
            if any(member.process_id == process_id for member in group.members):
                return group
        return None

    @property
    def representatives(self) -> tuple[GroupMember, ...]:
        return tuple(group.representative for group in self.groups.values())

    def members(self) -> Iterator[GroupMember]:
        for group in self.groups.values():
            yield from group.members

    def merge(
        self,
        member_representative: str,
        relation: GroupMember,
        table: ParticleTable,
    ) -> str:
        """Join two groups given ``relation``, the relation of one
        representative (``relation.process_id``) to another
        (``member_representative`` is the target). Returns the representative
        of the merged group, the canonically smallest generated one."""

        if relation.process_id == member_representative:
            raise ValueError("cannot merge a group with itself")
        source = self.groups.pop(relation.process_id)
        target = self.groups.pop(member_representative)
        source_key = representative_key(source.representative.process, table)
        target_key = representative_key(target.representative.process, table)
        if target_key <= source_key:
            keep, moved, link = target, source, relation
        else:
            keep, moved = source, target
            link = relation.inverted_to(target.representative)
        merged = list(keep.members)
        merged.extend(member.through(link) for member in moved.members)
        self.groups[keep.representative.process_id] = ProcessGroup(merged)
        return keep.representative.process_id

    def as_document(self) -> dict[str, object]:
        groups = sorted(
            self.groups.values(), key=lambda group: group.representative.process_id
        )
        return {
            "kind": PROCESS_GROUPS_KIND,
            "schema_version": PROCESS_GROUPS_SCHEMA_VERSION,
            "analyses": list(self.analyses),
            "groups": [group.as_dict() for group in groups],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_document(), indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> ProcessGroupMap:
        if document.get("kind") != PROCESS_GROUPS_KIND:
            raise ValueError("document is not a pyAmpliCol process-group map")
        if document.get("schema_version") != PROCESS_GROUPS_SCHEMA_VERSION:
            raise ValueError("unsupported process-group map schema version")
        result = cls(analyses=[dict(entry) for entry in document.get("analyses", [])])  # type: ignore[attr-defined]
        for group in document.get("groups", []):  # type: ignore[attr-defined]
            members = [GroupMember.from_dict(member) for member in group["members"]]
            result.add_group(members)
        return result

    @classmethod
    def from_json(cls, text: str) -> ProcessGroupMap:
        return cls.from_document(json.loads(text))


# --------------------------------------------------------------------------
# Flavour unions


class PairingRuleRow(Protocol):
    """A template fermion pairing rule (``FermionPairingRuleRowV1``)."""

    @property
    def rule_id(self) -> int: ...

    @property
    def endpoint_pairings(self) -> tuple[tuple[int, int], ...]: ...


UnionPairingRules = Callable[[GroupingProcess], Sequence[PairingRuleRow]]
"""The template's fermion pairing rules, built as its recurrence projection does."""
UnionNormalization = Callable[[Sequence[int], Sequence[int]], tuple[int, int]]
"""``(average, identical)`` factors of physical initial and final PDGs."""
UnionWarning = Callable[[GroupingProcess, frozenset[int], bool], str | None]
"""Why merging ``flavours`` in the template (and conjugating, if set) may not be
a symmetry of the allowed vertices, or ``None``."""


def is_union_quark_type(particle_type: ParticleType) -> bool:
    """Massless colour-triplet Dirac fermions: the flavours a union merges."""

    return (
        particle_type.spin == 2
        and particle_type.color == 3
        and not particle_type.self_conjugate
        and particle_type.mass == "ZERO"
    )


def _union_slot(leg: _Leg) -> tuple[Any, ...]:
    flavour = 0 if is_union_quark_type(leg.type) else leg.flavour
    return (leg.initial, leg.type, leg.orientation, flavour)


def _union_signature(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
) -> tuple[Any, ...]:
    legs = _legs(process, table, conjugate=conjugate)
    return (
        process.initial_count,
        tuple(sorted(_union_slot(leg) for leg in legs)),
    )


def flavour_union_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> tuple[Any, ...]:
    """Orientation-aware signature: per leg its side, type and particle or
    antiparticle orientation, plus the flavour of every non-union leg."""

    key = _union_signature(process, table)
    if consider_charge_conjugation:
        key = min(key, _union_signature(process, table, conjugate=True))
    return key


@dataclass(frozen=True, slots=True)
class FlavourUnion:
    """Subprocesses that differ only in the flavours of their union quarks.

    ``template`` is the canonically smallest member with every union quark
    relabelled to the smallest union flavour of its type; its legs are the
    union's slots. A member evaluates the template's pairing patterns:
    ``ME_member(p) = factor * |sum_P sign_P A_P(q)|^2`` with
    ``q[r] = p[permutation[r]]``, summed per colour flow and helicity.
    """

    template: GroupingProcess
    members: tuple[GroupMember, ...]
    pattern_count: int

    def group(self) -> list[GroupMember]:
        """The template as generated representative, then the routed members."""

        return [
            GroupMember.identity(self.template),
            *(
                member
                for member in self.members
                if member.process_id != self.template.process_id
            ),
        ]


_CONJUGATION_REFUSED = object()


def _union_permutation(
    member: Sequence[_Leg],
    template: Sequence[_Leg],
) -> tuple[int, ...]:
    # Equal slots are paired in leg order, so equal layouts map identically.
    free: dict[tuple[Any, ...], list[int]] = {}
    for leg in member:
        free.setdefault(_union_slot(leg), []).append(leg.index)
    permutation = []
    for leg in template:
        legs = free.get(_union_slot(leg))
        if not legs:
            raise ValueError("member does not have the union's slot shape")
        permutation.append(legs.pop(0))
    return tuple(permutation)


def _union_rules(
    template: GroupingProcess,
    legs: Sequence[_Leg],
    pairing_rules: UnionPairingRules,
) -> tuple[PairingRuleRow, ...]:
    rules = tuple(pairing_rules(template))
    if [int(rule.rule_id) for rule in rules] != list(range(len(rules))):
        raise ValueError(
            f"pairing rules of {template.expression!r} are not numbered 0..n-1"
        )
    # Colour-fundamental union slots: final quarks and initial antiquarks.
    fundamental = {
        leg.index
        for leg in legs
        if is_union_quark_type(leg.type) and (leg.orientation > 0) != leg.initial
    }
    antifundamental = {
        leg.index for leg in legs if is_union_quark_type(leg.type)
    } - fundamental
    paired = {
        (int(a), int(b))
        for rule in rules
        for a, b in rule.endpoint_pairings
        if a in fundamental and b in antifundamental
    }
    if paired != set(itertools.product(fundamental, antifundamental)):
        raise ValueError(
            f"pairing rules of {template.expression!r} do not pair every union "
            "quark with every union antiquark"
        )
    return rules


def _flavour_union(
    group: Sequence[GroupingProcess],
    table: ParticleTable,
    *,
    pairing_rules: UnionPairingRules,
    normalization: UnionNormalization,
    union_warning: UnionWarning | None,
    consider_charge_conjugation: bool,
) -> FlavourUnion | object | None:
    lowest: dict[ParticleType, int] = {}
    for process in group:
        for leg in _legs(process, table):
            if is_union_quark_type(leg.type):
                lowest[leg.type] = min(lowest.get(leg.type, leg.flavour), leg.flavour)
    classes = {
        flavor_class_key(
            process, table, consider_charge_conjugation=consider_charge_conjugation
        )
        for process in group
    }
    if not lowest or len(classes) < 2:
        return None
    ordered = sorted(group, key=lambda process: representative_key(process, table))
    anchor = ordered[0]
    pdgs = tuple(
        (lowest[leg.type] if leg.pdg > 0 else -lowest[leg.type])
        if is_union_quark_type(leg.type)
        else leg.pdg
        for leg in _legs(anchor, table)
    )
    names = [_particle(table, pdg).name for pdg in pdgs]
    count = anchor.initial_count
    template = GroupingProcess(
        process_id=(
            anchor.process_id
            if pdgs == anchor.pdgs
            else f"{anchor.process_id}_flavour_union"
        ),
        expression=f"{' '.join(names[:count])} > {' '.join(names[count:])}",
        pdgs=pdgs,
        initial_count=count,
    )
    signature = _union_signature(anchor, table)
    conjugated = {
        process.process_id: _union_signature(process, table) != signature
        for process in group
    }
    flavours = frozenset(
        leg.flavour
        for process in group
        for leg in _legs(process, table)
        if is_union_quark_type(leg.type)
    )
    if union_warning is not None:
        if union_warning(template, flavours, False) is not None:
            return None
        if any(conjugated.values()) and union_warning(template, flavours, True):
            return _CONJUGATION_REFUSED
    template_legs = _legs(template, table)
    rules = _union_rules(template, template_legs, pairing_rules)
    union_slots = tuple(is_union_quark_type(leg.type) for leg in template_legs)
    template_norm = math.prod(normalization(pdgs[:count], pdgs[count:]))
    members = []
    for process in ordered:
        conjugate = conjugated[process.process_id]
        permutation = _union_permutation(
            _legs(process, table, conjugate=conjugate), template_legs
        )
        flavour = tuple(abs(process.pdgs[leg]) for leg in permutation)
        patterns = tuple(
            int(rule.rule_id)
            for rule in rules
            if all(
                flavour[a] == flavour[b] or not (union_slots[a] or union_slots[b])
                for a, b in rule.endpoint_pairings
            )
        )
        if not patterns:
            raise ValueError(
                f"{process.expression!r} admits no pairing of {template.expression!r}"
            )
        initial = process.pdgs[: process.initial_count]
        final = process.pdgs[process.initial_count :]
        members.append(
            GroupMember(
                process_id=process.process_id,
                expression=process.expression,
                pdgs=process.pdgs,
                initial_count=process.initial_count,
                factor=Fraction(
                    template_norm, math.prod(normalization(initial, final))
                ),
                permutation=permutation,
                conjugated=conjugate,
                basis=GroupingBasis.FLAVOUR_UNION,
                generated=False,
                patterns=patterns,
                pattern_signs=(1,) * len(patterns),
            )
        )
    return FlavourUnion(template, tuple(members), len(rules))


def flavour_unions(
    processes: Sequence[GroupingProcess],
    table: ParticleTable,
    *,
    pairing_rules: UnionPairingRules,
    normalization: UnionNormalization,
    union_warning: UnionWarning | None = None,
    consider_charge_conjugation: bool = True,
) -> list[FlavourUnion]:
    """Join ``processes`` that differ only in union-quark flavours.

    ``processes`` are concrete subprocesses (flavour-class representatives
    and their members alike). A union forms only where it joins at least two
    flavour classes; other processes keep their flavour-class routing. With
    ``union_warning`` (strict mode) a union whose flavour merge is no symmetry
    is dropped, and one whose conjugation is none splits by orientation.
    """

    groups: dict[tuple[Any, ...], list[GroupingProcess]] = {}
    for process in processes:
        key = flavour_union_key(
            process, table, consider_charge_conjugation=consider_charge_conjugation
        )
        groups.setdefault(key, []).append(process)
    options = {
        "pairing_rules": pairing_rules,
        "normalization": normalization,
        "union_warning": union_warning,
    }
    result: list[FlavourUnion] = []
    for group in groups.values():
        union = _flavour_union(
            group,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
            **options,  # type: ignore[arg-type]
        )
        if union is _CONJUGATION_REFUSED:
            parts: dict[tuple[Any, ...], list[GroupingProcess]] = {}
            for process in group:
                parts.setdefault(_union_signature(process, table), []).append(process)
            for part in parts.values():
                union = _flavour_union(
                    part,
                    table,
                    consider_charge_conjugation=False,
                    **options,  # type: ignore[arg-type]
                )
                if isinstance(union, FlavourUnion):
                    result.append(union)
        elif isinstance(union, FlavourUnion):
            result.append(union)
    return result


def rational_factor(value: float, *, tolerance: float) -> Fraction | None:
    """``value`` as the simplest rational within its relative ``tolerance``
    (the precision to which numerical grouping compares matrix elements),
    with no bound on the numerator or the denominator."""

    if value == 0 or not math.isfinite(value):
        return None
    exact = Fraction(value)
    spread = Fraction(tolerance) * abs(exact)
    fraction = simplest_rational_between(exact - spread, exact + spread)
    return None if fraction == 0 else fraction


__all__ = [
    "PROCESS_GROUPS_KIND",
    "PROCESS_GROUPS_PATH",
    "PROCESS_GROUPS_ROLE",
    "PROCESS_GROUPS_SCHEMA_VERSION",
    "FlavourUnion",
    "GroupMember",
    "GroupingAnalysis",
    "GroupingBasis",
    "GroupingParticle",
    "GroupingProcess",
    "LegMapping",
    "ParticleTable",
    "ParticleType",
    "ProcessGroup",
    "ProcessGroupMap",
    "candidate_mappings",
    "coarse_key",
    "compose_permutations",
    "flavor_blind_classes",
    "flavor_blind_mapping",
    "flavor_class_key",
    "flavour_union_key",
    "flavour_unions",
    "identity_permutation",
    "invert_permutation",
    "is_union_quark_type",
    "particle_table",
    "profile_key",
    "profile_mappings",
    "rational_factor",
    "representative_key",
    "resolved_parameter_name",
    "signed_pdg_mapping",
]
