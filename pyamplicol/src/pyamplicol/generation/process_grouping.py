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
# Written only when a flavour-union member is crossed (carries crossing signs).
PROCESS_GROUPS_CROSSING_SCHEMA_VERSION = 2
_CROSSING_KEYS = ("crossing_momentum_signs", "crossing_helicity_signs")

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


def crossing_signs(
    permutation: Sequence[int],
    template_initial_count: int,
    member_initial_count: int,
) -> tuple[int, ...]:
    """Momentum sign per template slot: -1 where slot ``r`` and member leg
    ``permutation[r]`` lie on different sides of the ``>``."""

    return tuple(
        -1 if (slot < template_initial_count) != (leg < member_initial_count) else 1
        for slot, leg in enumerate(permutation)
    )


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
    # Crossed union member, per template slot: momentum signs (sigma, derived
    # from the sides) and helicity signs (rho), lambda_M[pi_r] = rho_r lambda_T[r].
    crossing_momentum_signs: tuple[int, ...] | None = None
    crossing_helicity_signs: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.patterns is None) != (self.pattern_signs is None):
            raise ValueError("patterns and pattern_signs must be given together")
        self._check_crossing()
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

    def _check_crossing(self) -> None:
        momentum, helicity = self.crossing_momentum_signs, self.crossing_helicity_signs
        if momentum is None and helicity is None:
            return
        name = f"process-group member {self.process_id!r}"
        if momentum is None or helicity is None:
            raise ValueError(f"{name} needs both crossing sign vectors or neither")
        if self.patterns is None:
            raise ValueError(f"{name} is crossed but not a flavour-union member")
        if len(momentum) != len(self.pdgs) or len(helicity) != len(self.pdgs):
            raise ValueError(f"{name} needs one crossing sign per leg")
        if any(sign not in (-1, 1) for sign in (*momentum, *helicity)):
            raise ValueError(f"{name} has crossing signs other than +1 and -1")
        if -1 not in momentum:
            raise ValueError(f"{name} carries crossing signs but is not crossed")
        if self.conjugated:
            raise ValueError(f"{name} cannot be both crossed and conjugated")
        if any(r != 1 for s, r in zip(momentum, helicity, strict=True) if s == 1):
            raise ValueError(f"{name} flips a helicity on an uncrossed slot")

    @property
    def crossed(self) -> bool:
        """Whether the template evaluates this member at sign-flipped momenta."""

        return self.crossing_momentum_signs is not None

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
        if self.crossing_momentum_signs is not None:
            record["crossing_momentum_signs"] = list(self.crossing_momentum_signs)
            record["crossing_helicity_signs"] = list(self.crossing_helicity_signs or ())
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
            crossing_momentum_signs=_optional_int_tuple(
                value.get("crossing_momentum_signs")
            ),
            crossing_helicity_signs=_optional_int_tuple(
                value.get("crossing_helicity_signs")
            ),
        )


def _pattern_pairs(value: object) -> PatternPairs | None:
    if value is None:
        return None
    try:
        return tuple(
            tuple((int(f), int(a)) for f, a in pattern)  # type: ignore[union-attr]
            for pattern in value  # type: ignore[union-attr]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "flavour-union pattern pairs must be [[f, a], ...] lists"
        ) from exc


def _optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError("flavour-union patterns and signs must be integer lists")
    return tuple(value)


PatternPairs = tuple[tuple[tuple[int, int], ...], ...]
"""Per pairing pattern of a union template, its (fundamental, antifundamental) slots."""


@dataclass(slots=True)
class ProcessGroup:
    """A representative and every member, the representative first.

    A flavour-union template carries its pattern table ``pattern_pairs``.
    """

    members: list[GroupMember]
    pattern_pairs: PatternPairs | None = None

    def __post_init__(self) -> None:
        representative = self.members[0]
        for member in self.members:
            if member.crossed and member.crossing_momentum_signs != crossing_signs(
                member.permutation, representative.initial_count, member.initial_count
            ):
                raise ValueError(
                    f"process-group member {member.process_id!r} has crossing "
                    "momentum signs that disagree with its permutation and sides"
                )

    @property
    def representative(self) -> GroupMember:
        return self.members[0]

    def as_dict(self) -> dict[str, object]:
        representative = self.representative
        record: dict[str, object] = {
            "representative": {
                "process_id": representative.process_id,
                "process": representative.expression,
            },
            "members": [member.as_dict() for member in self.members],
        }
        if self.pattern_pairs is not None:
            record["flavour_union_pattern_pairs"] = [
                [list(pair) for pair in pattern] for pattern in self.pattern_pairs
            ]
        return record


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

    def add_group(
        self,
        members: Sequence[GroupMember],
        pattern_pairs: PatternPairs | None = None,
    ) -> None:
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
        self.groups[representative.process_id] = ProcessGroup(
            list(members), pattern_pairs
        )

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
        crossed = any(member.crossed for member in self.members())
        return {
            "kind": PROCESS_GROUPS_KIND,
            "schema_version": PROCESS_GROUPS_CROSSING_SCHEMA_VERSION
            if crossed
            else PROCESS_GROUPS_SCHEMA_VERSION,
            "analyses": list(self.analyses),
            "groups": [group.as_dict() for group in groups],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_document(), indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> ProcessGroupMap:
        if document.get("kind") != PROCESS_GROUPS_KIND:
            raise ValueError("document is not a pyAmpliCol process-group map")
        schema = document.get("schema_version")
        if isinstance(schema, bool) or schema not in (
            PROCESS_GROUPS_SCHEMA_VERSION,
            PROCESS_GROUPS_CROSSING_SCHEMA_VERSION,
        ):
            raise ValueError("unsupported process-group map schema version")
        result = cls(analyses=[dict(entry) for entry in document.get("analyses", [])])  # type: ignore[attr-defined]
        for group in document.get("groups", []):  # type: ignore[attr-defined]
            if schema == PROCESS_GROUPS_SCHEMA_VERSION and any(
                key in member for member in group["members"] for key in _CROSSING_KEYS
            ):
                raise ValueError(
                    "process-group map schema version 1 cannot carry crossing signs"
                )
            members = [GroupMember.from_dict(member) for member in group["members"]]
            result.add_group(
                members, _pattern_pairs(group.get("flavour_union_pattern_pairs"))
            )
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
UnionMemberFilter = Callable[[GroupingProcess, GroupingProcess], bool]
"""Whether a process may join the union of a template (e.g. equal coupling orders)."""
"""Why merging ``flavours`` in the template (and conjugating, if set) may not be
a symmetry of the allowed vertices, or ``None``."""
UnionClasses = Mapping[str, tuple[str, GroupMember]]
"""Flavour classes: process ID -> (representative ID, process -> representative)."""
UnionCrossingRelation = Callable[
    [GroupingProcess, GroupingProcess, tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]] | None,
]
"""``(sigma, rho)`` per template slot of a crossed member routed by the given
permutation, or ``None`` when the member cannot be crossed (it keeps its routing)."""


def is_union_quark_type(particle_type: ParticleType) -> bool:
    """Massless colour-triplet Dirac fermions: the flavours a union merges."""

    return (
        particle_type.spin == 2
        and particle_type.color == 3
        and not particle_type.self_conjugate
        and particle_type.mass == "ZERO"
    )


def is_union_gluon_type(particle_type: ParticleType) -> bool:
    """Massless self-conjugate colour octet vectors (gluons)."""

    return (
        particle_type.spin == 3
        and particle_type.color == 8
        and particle_type.self_conjugate
        and particle_type.mass == "ZERO"
    )


def crossing_eligible(process: GroupingProcess, table: ParticleTable) -> bool:
    """Stage 1 of crossing: every leg is a union quark or a gluon."""

    types = (_particle(table, pdg).type for pdg in process.pdgs)
    return all(is_union_quark_type(t) or is_union_gluon_type(t) for t in types)


def _outgoing_orientation(leg: _Leg) -> int:
    return -leg.orientation if leg.initial else leg.orientation


def _union_slot(leg: _Leg, crossing: bool = False) -> tuple[Any, ...]:
    flavour = 0 if is_union_quark_type(leg.type) else leg.flavour
    if crossing:
        return (leg.type, _outgoing_orientation(leg), flavour)
    return (leg.initial, leg.type, leg.orientation, flavour)


def _union_signature(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
    crossing: bool = False,
) -> tuple[Any, ...]:
    legs = _legs(process, table, conjugate=conjugate)
    return (
        process.initial_count,
        tuple(sorted(_union_slot(leg, crossing) for leg in legs)),
    )


def flavour_union_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
    crossing: bool = False,
) -> tuple[Any, ...]:
    """Orientation-aware signature: per leg its side, type and particle or
    antiparticle orientation, plus the flavour of every non-union leg.

    With ``crossing``, a crossing-eligible process is keyed by its all-outgoing
    content instead (no sides), so every crossing joins one union."""

    crossed = crossing and crossing_eligible(process, table)
    key = _union_signature(process, table, crossing=crossed)
    if consider_charge_conjugation:
        key = min(
            key, _union_signature(process, table, conjugate=True, crossing=crossed)
        )
    return ("all-outgoing", key) if crossed else key


@dataclass(frozen=True, slots=True)
class FlavourUnion:
    """Subprocesses that differ only in the flavours of their union quarks.

    ``template`` is the canonically smallest member with a single flavour per
    union-quark type (a union without one is not formed); its legs are the
    union's slots. Members without a pairing pattern or rejected by the member
    filter keep their own routing. A member evaluates the template's pairing patterns:
    ``ME_member(p) = factor * |sum_P sign_P A_P(q)|^2`` with
    ``q[r] = p[permutation[r]]``, summed per colour flow and helicity.
    """

    template: GroupingProcess
    members: tuple[GroupMember, ...]
    pattern_count: int
    # Endpoint pairs of each pattern (pairing rule) in template slot order.
    pattern_pairs: PatternPairs = ()

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
_CROSSING_REFUSED = object()


def _union_permutation(
    member: Sequence[_Leg],
    template: Sequence[_Leg],
    crossing: bool = False,
) -> tuple[int, ...]:
    # Equal slots are paired in leg order, so equal layouts map identically.
    free: dict[tuple[Any, ...], list[_Leg]] = {}
    for leg in member:
        free.setdefault(_union_slot(leg, crossing), []).append(leg)
    permutation: list[int | None] = [None] * len(template)
    # Same side first (an uncrossed member keeps its side-aware route), then across.
    for same_side in (True, False) if crossing else (False,):
        for slot, leg in enumerate(template):
            if permutation[slot] is not None:
                continue
            legs = free.get(_union_slot(leg, crossing), [])
            match = next(
                (m for m in legs if not same_side or m.initial == leg.initial), None
            )
            if match is None:
                if same_side:
                    continue
                raise ValueError("member does not have the union's slot shape")
            legs.remove(match)
            permutation[slot] = match.index
    return tuple(int(leg) for leg in permutation)  # type: ignore[arg-type]


def _class_union_permutations(
    ordered: Sequence[GroupingProcess],
    template: GroupingProcess,
    template_legs: Sequence[_Leg],
    conjugated: Mapping[str, bool],
    table: ParticleTable,
    classes: UnionClasses,
    allow_conjugation: bool,
    crossing: bool = False,
) -> dict[str, tuple[tuple[int, ...], bool]]:
    """Union routes ``(permutation, conjugated)`` that follow the flavour classes.

    Each class has one anchor (the template, else its representative, else
    its first process) routed by ``_union_permutation``. Every other process
    composes its relation to the anchor with that route, so its members keep
    the class's relative permutations, conjugations, patterns and colour orders.
    """

    by_class: dict[str, list[GroupingProcess]] = {}
    for process in ordered:
        home = classes.get(process.process_id, (process.process_id, None))[0]
        by_class.setdefault(home, []).append(process)
    identity = identity_permutation(len(template_legs))
    template_initial = sum(1 for leg in template_legs if leg.initial)

    def to_representative(process: GroupingProcess) -> tuple[tuple[int, ...], bool]:
        relation = classes.get(process.process_id, (None, None))[1]
        if relation is None:
            return identity, False
        return tuple(relation.permutation), relation.conjugated

    result: dict[str, tuple[tuple[int, ...], bool]] = {}
    for home, processes in by_class.items():
        anchor = next(
            (p for p in processes if p.process_id == template.process_id),
            next((p for p in processes if p.process_id == home), processes[0]),
        )
        route = _union_permutation(
            _legs(anchor, table, conjugate=conjugated[anchor.process_id]),
            template_legs,
            crossing,
        )
        anchor_permutation, anchor_conjugated = to_representative(anchor)
        from_representative = invert_permutation(anchor_permutation)
        for process in processes:
            permutation, conjugate = to_representative(process)
            # template slot -> anchor leg -> representative leg -> process leg
            composed = tuple(permutation[from_representative[leg]] for leg in route)
            conjugate = (conjugate != anchor_conjugated) != conjugated[
                anchor.process_id
            ]
            legs = _legs(process, table, conjugate=conjugate)
            # a self-conjugate slot shape may take its class's conjugation
            if (
                (conjugate != conjugated[process.process_id] and not allow_conjugation)
                or any(
                    _union_slot(legs[leg], crossing) != _union_slot(slot, crossing)
                    for leg, slot in zip(composed, template_legs, strict=True)
                )
                # a member is never both crossed and conjugated
                or (
                    conjugate
                    and -1
                    in crossing_signs(composed, template_initial, process.initial_count)
                )
            ):
                # no consistent class route: the process is routed on its own
                conjugate = conjugated[process.process_id]
                composed = _union_permutation(
                    _legs(process, table, conjugate=conjugate), template_legs, crossing
                )
            result[process.process_id] = (composed, conjugate)
    return result


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
    member_filter: UnionMemberFilter | None = None,
    classes: UnionClasses | None = None,
    crossing: bool = False,
    crossing_relation: UnionCrossingRelation | None = None,
) -> FlavourUnion | object | None:
    def class_count(processes: Iterable[GroupingProcess]) -> int:
        return len(
            {
                flavor_class_key(
                    process,
                    table,
                    consider_charge_conjugation=consider_charge_conjugation,
                )
                for process in processes
            }
        )

    if class_count(group) < 2:
        return None
    ordered = sorted(group, key=lambda process: representative_key(process, table))
    # The template is a requested process with one flavour per union-quark
    # type (smallest first), so it never adds an unrequested subprocess.
    single = [process for process in ordered if _single_union_flavour(process, table)]
    if not single:
        return None
    anchor = (
        _choose_union_template(single, ordered, table, consider_charge_conjugation)
        if crossing
        else single[0]
    )
    pdgs = anchor.pdgs
    count = anchor.initial_count
    template = GroupingProcess(
        process_id=anchor.process_id,
        expression=anchor.expression,
        pdgs=pdgs,
        initial_count=count,
    )
    signature = _union_signature(anchor, table)
    conjugated = {
        process.process_id: _union_signature(process, table) != signature
        for process in group
    }
    if crossing:
        # conjugate only where that makes the member uncrossed; else it crosses
        conjugated = {
            process.process_id: consider_charge_conjugation
            and conjugated[process.process_id]
            and _union_signature(process, table, conjugate=True) == signature
            for process in group
        }
    flavours = frozenset(
        leg.flavour
        for process in group
        for leg in _legs(process, table)
        if is_union_quark_type(leg.type)
    )
    if union_warning is not None and union_warning(template, flavours, False):
        return None
    template_legs = _legs(template, table)
    routes = _class_union_permutations(
        ordered,
        template,
        template_legs,
        conjugated,
        table,
        classes or {},
        consider_charge_conjugation,
        crossing,
    )
    if (
        union_warning is not None
        and any(conjugate for _, conjugate in routes.values())
        and union_warning(template, flavours, True)
    ):
        return _CONJUGATION_REFUSED
    rules = _union_rules(template, template_legs, pairing_rules)
    union_slots = tuple(is_union_quark_type(leg.type) for leg in template_legs)
    template_norm = math.prod(normalization(pdgs[:count], pdgs[count:]))
    members = []
    for process in ordered:
        permutation, conjugate = routes[process.process_id]
        flavour = tuple(abs(process.pdgs[leg]) for leg in permutation)
        patterns = tuple(
            int(rule.rule_id)
            for rule in rules
            if all(
                flavour[a] == flavour[b] or not (union_slots[a] or union_slots[b])
                for a, b in rule.endpoint_pairings
            )
        )
        # A flavour flow the template cannot pair (e.g. W-only) keeps its routing.
        if not patterns or (
            member_filter is not None and not member_filter(template, process)
        ):
            continue
        sigma = crossing_signs(permutation, count, process.initial_count)
        rho: tuple[int, ...] | None = None
        if -1 in sigma:
            relation = (
                None
                if conjugate or crossing_relation is None
                else crossing_relation(template, process, permutation)
            )
            if relation is None:
                return _CROSSING_REFUSED
            if tuple(relation[0]) != sigma:
                raise ValueError(
                    f"crossing of {process.expression!r} into "
                    f"{template.expression!r} has momentum signs {relation[0]!r}, "
                    f"its permutation gives {sigma!r}"
                )
            rho = tuple(relation[1])
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
                crossing_momentum_signs=None if rho is None else sigma,
                crossing_helicity_signs=rho,
            )
        )
    if class_count(member.process for member in members) < 2:
        return None
    pattern_pairs = tuple(
        tuple((int(f), int(a)) for f, a in rule.endpoint_pairings) for rule in rules
    )
    return FlavourUnion(template, tuple(members), len(rules), pattern_pairs)


def _choose_union_template(
    single: Sequence[GroupingProcess],
    group: Sequence[GroupingProcess],
    table: ParticleTable,
    consider_charge_conjugation: bool,
) -> GroupingProcess:
    """The single-flavour process that leaves the most processes uncrossed
    (directly or conjugated); ties go to the first in ``single``."""

    plain = [_union_signature(process, table) for process in group]
    conjugate = [
        _union_signature(process, table, conjugate=True)
        if consider_charge_conjugation
        else None
        for process in group
    ]

    def uncrossed(candidate: GroupingProcess) -> int:
        key = _union_signature(candidate, table)
        return sum(1 for a, b in zip(plain, conjugate, strict=True) if key in (a, b))

    template = max(single, key=uncrossed)
    assert any(is_union_quark_type(table[pdg].type) for pdg in template.pdgs)
    return template


def _single_union_flavour(process: GroupingProcess, table: ParticleTable) -> bool:
    flavours: dict[ParticleType, set[int]] = {}
    for leg in _legs(process, table):
        if is_union_quark_type(leg.type):
            flavours.setdefault(leg.type, set()).add(leg.flavour)
    return bool(flavours) and all(len(values) == 1 for values in flavours.values())


def flavour_unions(
    processes: Sequence[GroupingProcess],
    table: ParticleTable,
    *,
    pairing_rules: UnionPairingRules,
    normalization: UnionNormalization,
    union_warning: UnionWarning | None = None,
    consider_charge_conjugation: bool = True,
    member_filter: UnionMemberFilter | None = None,
    classes: UnionClasses | None = None,
    crossing: bool = False,
    crossing_relation: UnionCrossingRelation | None = None,
) -> list[FlavourUnion]:
    """Join ``processes`` that differ only in union-quark flavours.

    ``processes`` are concrete subprocesses (flavour-class representatives
    and their members alike). A union forms only where it joins at least two
    flavour classes; other processes keep their flavour-class routing. With
    ``union_warning`` (strict mode) a union whose flavour merge is no symmetry
    is dropped, and one whose conjugation is none splits by orientation.
    With ``classes``, the members of one flavour class share its routing.

    With ``crossing``, crossing-eligible processes (union quarks and gluons
    only) are keyed by all-outgoing content: a member that matches the template
    neither directly nor conjugated is crossed, with the signs that
    ``crossing_relation`` returns. If it returns ``None`` for any member, that
    content falls back to its side-aware unions. A refused conjugation crosses
    the members instead of splitting the union.
    """

    if crossing and crossing_relation is None:
        raise ValueError("crossed flavour unions need a crossing relation")
    groups: dict[tuple[Any, ...], list[GroupingProcess]] = {}
    for process in processes:
        key = flavour_union_key(
            process,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
            crossing=crossing,
        )
        groups.setdefault(key, []).append(process)
    options = {
        "pairing_rules": pairing_rules,
        "normalization": normalization,
        "union_warning": union_warning,
        "member_filter": member_filter,
        "classes": classes,
        "crossing_relation": crossing_relation,
    }
    result: list[FlavourUnion] = []

    def side_aware(group: Sequence[GroupingProcess]) -> list[FlavourUnion]:
        # a refused crossing: this content keeps the unions it has without crossing
        return flavour_unions(
            group,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
            **{**options, "crossing_relation": None},  # type: ignore[arg-type]
        )

    for key, group in groups.items():
        crossed = crossing and key[0] == "all-outgoing"
        union = _flavour_union(
            group,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
            crossing=crossed,
            **options,  # type: ignore[arg-type]
        )
        if union is _CROSSING_REFUSED:
            result.extend(side_aware(group))
        elif union is _CONJUGATION_REFUSED:
            parts: dict[tuple[Any, ...], list[GroupingProcess]] = {}
            for process in group:
                parts.setdefault(
                    _union_signature(process, table, crossing=crossed), []
                ).append(process)
            for part in parts.values():
                union = _flavour_union(
                    part,
                    table,
                    consider_charge_conjugation=False,
                    crossing=crossed,
                    **options,  # type: ignore[arg-type]
                )
                if union is _CROSSING_REFUSED:
                    result.extend(side_aware(part))
                elif isinstance(union, FlavourUnion):
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
    "PROCESS_GROUPS_CROSSING_SCHEMA_VERSION",
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
    "PatternPairs",
    "ProcessGroup",
    "ProcessGroupMap",
    "UnionCrossingRelation",
    "candidate_mappings",
    "coarse_key",
    "compose_permutations",
    "crossing_eligible",
    "crossing_signs",
    "flavor_blind_classes",
    "flavor_blind_mapping",
    "flavor_class_key",
    "flavour_union_key",
    "flavour_unions",
    "identity_permutation",
    "invert_permutation",
    "is_union_gluon_type",
    "is_union_quark_type",
    "particle_table",
    "profile_key",
    "profile_mappings",
    "rational_factor",
    "representative_key",
    "resolved_parameter_name",
    "signed_pdg_mapping",
]
