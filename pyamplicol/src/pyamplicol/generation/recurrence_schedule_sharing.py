# SPDX-License-Identifier: 0BSD
"""Content-addressed recurrence schedules and compact process bindings."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Generic, Protocol, TypeVar

from .recurrence_columnar import (
    RecurrenceBuilderLogicalInputV1,
    RecurrenceExternalLegV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceSourceStateV1,
)

RECURRENCE_SCHEDULE_SHARING_KIND = "pyamplicol-recurrence-schedule-sharing"
RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION = 3
RECURRENCE_SCHEDULE_INDEX_PATH = "recurrence/schedule-index.json"
RECURRENCE_PROCESS_BINDING_ABI = "pyamplicol-recurrence-process-binding-v2"
RECURRENCE_PROCESS_BINDING_MAGIC = b"PACRDBN2"
_PROCESS_BINDING_VERSION = 2
_PROCESS_BINDING_FIXED_SIZE = 160
_MAX_SOURCE_BIJECTIONS = 4096

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class RecurrenceScheduleSharingError(ValueError):
    """Raised when recurrence schedules cannot be shared exactly."""


@dataclass(frozen=True, slots=True)
class RecurrenceProcessRemap:
    """Authenticated root-schedule to concrete-process binding.

    Dense source, flow, and sector maps are deliberately explicit. Model-wide
    template/executor/parameter domains use sparse changes over an identity
    bijection so a binding remains small even for a large prepared catalog.
    """

    source_slots: tuple[int, ...]
    source_momentum_signs: tuple[int, ...]
    source_helicity_signs: tuple[int, ...]
    source_state_offsets: tuple[int, ...]
    source_state_indices: tuple[int, ...]
    public_flow_ids: tuple[int, ...]
    physical_sector_ids: tuple[int, ...]
    state_template_count: int
    source_template_count: int
    direct_executor_count: int
    parameter_slot_count: int
    state_template_changes: tuple[tuple[int, int], ...] = ()
    source_template_changes: tuple[tuple[int, int], ...] = ()
    direct_executor_changes: tuple[tuple[int, int], ...] = ()
    parameter_slot_changes: tuple[tuple[int, int], ...] = ()
    bijection_digest: str = ""

    def __post_init__(self) -> None:
        _permutation(self.source_slots, "source-slot remap")
        _signs(self.source_momentum_signs, len(self.source_slots), "momentum signs")
        _signs(self.source_helicity_signs, len(self.source_slots), "helicity signs")
        _ragged_permutations(
            self.source_state_offsets,
            self.source_state_indices,
            len(self.source_slots),
            "source-state remap",
        )
        _permutation(self.public_flow_ids, "public-flow remap")
        _permutation(self.physical_sector_ids, "physical-sector remap")
        for count, label in (
            (self.state_template_count, "state-template count"),
            (self.source_template_count, "source-template count"),
            (self.direct_executor_count, "direct-executor count"),
            (self.parameter_slot_count, "parameter-slot count"),
        ):
            _nonnegative_integer(count, label)
        for changes, count, label in (
            (
                self.state_template_changes,
                self.state_template_count,
                "state-template remap",
            ),
            (
                self.source_template_changes,
                self.source_template_count,
                "source-template remap",
            ),
            (
                self.direct_executor_changes,
                self.direct_executor_count,
                "direct-executor remap",
            ),
            (
                self.parameter_slot_changes,
                self.parameter_slot_count,
                "parameter-slot remap",
            ),
        ):
            _sparse_permutation(changes, count, label)
        if self.bijection_digest:
            _require_sha256(self.bijection_digest, "process-bijection digest")

    @classmethod
    def identity(
        cls,
        logical: RecurrenceBuilderLogicalInputV1,
        *,
        direct_executor_count: int,
        parameter_slot_count: int,
    ) -> RecurrenceProcessRemap:
        state_count = _template_count(logical, "current-state")
        source_count = _template_count(logical, "source")
        source_state_offsets = [0]
        source_state_indices: list[int] = []
        for leg in logical.external_legs:
            source_state_indices.extend(range(len(leg.source_states)))
            source_state_offsets.append(len(source_state_indices))
        result = cls(
            source_slots=tuple(range(len(logical.external_legs))),
            source_momentum_signs=(1,) * len(logical.external_legs),
            source_helicity_signs=(1,) * len(logical.external_legs),
            source_state_offsets=tuple(source_state_offsets),
            source_state_indices=tuple(source_state_indices),
            public_flow_ids=tuple(range(len(logical.public_flows))),
            physical_sector_ids=tuple(range(_physical_sector_domain_count(logical))),
            state_template_count=state_count,
            source_template_count=source_count,
            direct_executor_count=_nonnegative_integer(
                direct_executor_count, "direct-executor count"
            ),
            parameter_slot_count=_nonnegative_integer(
                parameter_slot_count, "parameter-slot count"
            ),
        )
        return result.with_digest(
            _canonical_digest(
                {
                    "contract": "pyamplicol-recurrence-process-bijection-v1",
                    "process": logical.process_id,
                    "remap": result._digest_payload(),
                    "relation": "identity",
                }
            )
        )

    def with_digest(self, digest: str) -> RecurrenceProcessRemap:
        return RecurrenceProcessRemap(
            source_slots=self.source_slots,
            source_momentum_signs=self.source_momentum_signs,
            source_helicity_signs=self.source_helicity_signs,
            source_state_offsets=self.source_state_offsets,
            source_state_indices=self.source_state_indices,
            public_flow_ids=self.public_flow_ids,
            physical_sector_ids=self.physical_sector_ids,
            state_template_count=self.state_template_count,
            source_template_count=self.source_template_count,
            direct_executor_count=self.direct_executor_count,
            parameter_slot_count=self.parameter_slot_count,
            state_template_changes=self.state_template_changes,
            source_template_changes=self.source_template_changes,
            direct_executor_changes=self.direct_executor_changes,
            parameter_slot_changes=self.parameter_slot_changes,
            bijection_digest=_require_sha256(digest, "process-bijection digest"),
        )

    def remap_resolved_helicities(
        self,
        rows: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = []
        for row in rows:
            if len(row) != len(self.source_slots):
                raise RecurrenceScheduleSharingError(
                    "resolved helicity width disagrees with process binding"
                )
            target = [0] * len(row)
            for root_slot, target_slot in enumerate(self.source_slots):
                target[target_slot] = (
                    row[root_slot] * self.source_helicity_signs[root_slot]
                )
            result.append(tuple(target))
        if len(set(result)) != len(result):
            raise RecurrenceScheduleSharingError(
                "process binding collapses distinct resolved helicities"
            )
        return tuple(result)

    def to_mapping(self) -> dict[str, object]:
        return {
            "bijection_digest": self.bijection_digest,
            "source_slots": list(self.source_slots),
            "source_momentum_signs": list(self.source_momentum_signs),
            "source_helicity_signs": list(self.source_helicity_signs),
            "source_state_offsets": list(self.source_state_offsets),
            "source_state_indices": list(self.source_state_indices),
            "public_flow_ids": list(self.public_flow_ids),
            "physical_sector_ids": list(self.physical_sector_ids),
            "state_templates": _sparse_mapping(
                self.state_template_count, self.state_template_changes
            ),
            "source_templates": _sparse_mapping(
                self.source_template_count, self.source_template_changes
            ),
            "direct_executors": _sparse_mapping(
                self.direct_executor_count, self.direct_executor_changes
            ),
            "parameter_slots": _sparse_mapping(
                self.parameter_slot_count, self.parameter_slot_changes
            ),
        }

    def _digest_payload(self) -> dict[str, object]:
        payload = self.to_mapping()
        payload["bijection_digest"] = None
        return payload


@dataclass(frozen=True, slots=True)
class RecurrenceSharedLoweringResult(Generic[_T]):
    output: _T
    schedule_digest: str
    remap: RecurrenceProcessRemap


@dataclass(slots=True)
class _PendingLowering(Generic[_T]):
    logical: RecurrenceBuilderLogicalInputV1
    direct_executor_count: int
    parameter_slot_count: int
    schedule_digest: str
    future: Future[_T]


class RecurrenceScheduleLoweringCache(Generic[_T]):
    """Thread-safe single-flight cache keyed before native lowering."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: dict[str, Future[_T]] = {}
        self._processes: list[_PendingLowering[_T]] = []

    def lower_once(self, digest: str, lower: Callable[[], _T]) -> _T:
        key = _require_sha256(digest, "pre-lowering schedule digest")
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                pending = Future()
                self._pending[key] = pending
                owner = True
            else:
                owner = False
        if not owner:
            return pending.result()
        try:
            result = lower()
        except BaseException as exc:
            pending.set_exception(exc)
            try:
                pending.exception()
            finally:
                raise
        pending.set_result(result)
        return result

    def lower_process(
        self,
        logical: RecurrenceBuilderLogicalInputV1,
        *,
        schedule_digest: str,
        direct_executor_count: int,
        parameter_slot_count: int,
        lower: Callable[[], _T],
    ) -> RecurrenceSharedLoweringResult[_T]:
        """Lower once across exact process-isomorphic recurrence schedules."""

        digest = _require_sha256(schedule_digest, "pre-lowering schedule digest")
        executor_count = _nonnegative_integer(
            direct_executor_count, "direct-executor count"
        )
        parameter_count = _nonnegative_integer(
            parameter_slot_count, "parameter-slot count"
        )
        with self._lock:
            owner: _PendingLowering[_T] | None = None
            remap: RecurrenceProcessRemap | None = None
            for candidate in self._processes:
                if (
                    candidate.schedule_digest != digest
                    or candidate.direct_executor_count != executor_count
                    or candidate.parameter_slot_count != parameter_count
                ):
                    continue
                remap = exact_recurrence_process_bijection(
                    candidate.logical,
                    logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                )
                if remap is not None:
                    owner = candidate
                    break
            if owner is None:
                future: Future[_T] = Future()
                owner = _PendingLowering(
                    logical=logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                    schedule_digest=digest,
                    future=future,
                )
                self._processes.append(owner)
                remap = RecurrenceProcessRemap.identity(
                    logical,
                    direct_executor_count=executor_count,
                    parameter_slot_count=parameter_count,
                )
                owns_lowering = True
            else:
                owns_lowering = False
        assert remap is not None
        if owns_lowering:
            try:
                output = lower()
            except BaseException as exc:
                owner.future.set_exception(exc)
                try:
                    owner.future.exception()
                finally:
                    raise
            owner.future.set_result(output)
        else:
            output = owner.future.result()
        return RecurrenceSharedLoweringResult(
            output=output,
            schedule_digest=owner.schedule_digest,
            remap=remap,
        )


class RecurrenceScheduleProcess(Protocol):
    process_id: str
    recurrence_schedule_path: Path
    recurrence_schedule_digest: str
    recurrence_native_schedule_semantic_digest: str
    recurrence_schedule_size_bytes: int
    recurrence_schedule_sha256: str
    recurrence_schedule_member_count: int
    recurrence_schedule_unpacked_size_bytes: int
    recurrence_schedule_index_sha256: str
    builder_input_sha256: str
    process_support_mask: int
    recurrence_process_remap: RecurrenceProcessRemap


def _recurrence_schedule_identity_payload(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
) -> dict[str, object]:
    return {
        "contract": "pyamplicol-recurrence-prelower-schedule-identity-v1",
        "logical": _schedule_plain(logical),
        "prepared_kernel_pack_digest": _require_sha256(
            prepared_kernel_pack_digest, "prepared-kernel pack digest"
        ),
        "direct_template_catalog_digest": _require_sha256(
            direct_template_catalog_digest, "direct-template catalog digest"
        ),
        "point_tile_size": _positive_integer(point_tile_size, "point tile size"),
        "workspace_mib": _positive_integer(workspace_mib, "workspace MiB"),
    }


def recurrence_native_schedule_semantic_digest(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
) -> str:
    """Return the relation-independent semantic identity embedded by native lowering.

    Only concrete process ownership is normalized. All model, crossing,
    selector, color, proof, parameter, and runtime-option semantics remain in
    the digest. Relation-discovery policy is deliberately excluded: it governs
    the lowering request, while certified relations themselves are authenticated
    separately and a request that applies none must retain the source schedule's
    byte identity.
    """

    return _canonical_digest(
        _recurrence_schedule_identity_payload(
            logical,
            prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            direct_template_catalog_digest=direct_template_catalog_digest,
            point_tile_size=point_tile_size,
            workspace_mib=workspace_mib,
        )
    )


def recurrence_schedule_semantic_digest(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    prepared_kernel_pack_digest: str,
    direct_template_catalog_digest: str,
    point_tile_size: int,
    workspace_mib: int,
    relation_discovery: Mapping[str, object] | None = None,
) -> str:
    """Return the policy-bearing request/cache identity for schedule lowering.

    This identity isolates discovery policies in temporary paths, lowering
    cache ownership, and artifact grouping. Native serialization receives
    :func:`recurrence_native_schedule_semantic_digest` instead, so policy alone
    cannot perturb a schedule when no certified relation is applied.
    """

    payload = _recurrence_schedule_identity_payload(
        logical,
        prepared_kernel_pack_digest=prepared_kernel_pack_digest,
        direct_template_catalog_digest=direct_template_catalog_digest,
        point_tile_size=point_tile_size,
        workspace_mib=workspace_mib,
    )
    if relation_discovery is not None:
        payload["relation_discovery"] = _schedule_plain(relation_discovery)
    return _canonical_digest(payload)


def exact_recurrence_process_bijection(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
    *,
    direct_executor_count: int,
    parameter_slot_count: int,
) -> RecurrenceProcessRemap | None:
    """Return an exact root-to-target schedule binding or fail closed.

    V1 sharing is intentionally limited to complete topology-replay schedules.
    The relation is an all-outgoing external-state isomorphism, not a particle
    label erasure: every source state, color forest, replay selector, parameter
    projection, and fermion-pairing row must map bijectively.
    """

    if (
        root.layout != "topology-replay"
        or target.layout != "topology-replay"
        or len(root.external_legs) != len(target.external_legs)
        or root.coupling_limits != target.coupling_limits
        or root.parameter_projection != target.parameter_projection
        or root.semantic_template_references != target.semantic_template_references
        or root.selected_public_flow_ids is not None
        or target.selected_public_flow_ids is not None
        or root.selected_source_coverage is not None
        or target.selected_source_coverage is not None
        or not _fixed_semantic_digests_match(root, target)
    ):
        return None

    candidates = _source_slot_bijections(root.external_legs, target.external_legs)
    for source_slots in candidates:
        source_relation = _external_source_relation(
            root.external_legs,
            target.external_legs,
            source_slots,
        )
        if source_relation is None:
            continue
        (
            momentum_signs,
            helicity_signs,
            source_state_offsets,
            source_state_indices,
            source_template_changes,
        ) = source_relation
        sector_ids = _sector_bijection(
            root.physical_sectors,
            target.physical_sectors,
            source_slots,
        )
        if sector_ids is None:
            continue
        flow_ids = _flow_bijection(
            root.public_flows,
            target.public_flows,
            source_slots,
            sector_ids,
        )
        if flow_ids is None:
            continue
        # Direct-Arena proof metadata still owns physical sector IDs. Until
        # that metadata has an explicit remap ABI, share only schedules whose
        # physical/public sector numbering is already identical.
        if sector_ids != tuple(range(len(sector_ids))) or flow_ids != tuple(
            range(len(flow_ids))
        ):
            continue
        if not _replay_contracts_match(
            root,
            target,
            source_slots=source_slots,
            sector_ids=sector_ids,
        ):
            continue
        if not _fermion_pairing_contracts_match(
            root.fermion_pairing_catalog,
            target.fermion_pairing_catalog,
            source_slots,
        ):
            continue

        result = RecurrenceProcessRemap(
            source_slots=source_slots,
            source_momentum_signs=momentum_signs,
            source_helicity_signs=helicity_signs,
            source_state_offsets=source_state_offsets,
            source_state_indices=source_state_indices,
            public_flow_ids=flow_ids,
            physical_sector_ids=tuple(range(_physical_sector_domain_count(root))),
            state_template_count=_template_count(root, "current-state"),
            source_template_count=_template_count(root, "source"),
            direct_executor_count=_nonnegative_integer(
                direct_executor_count, "direct-executor count"
            ),
            parameter_slot_count=_nonnegative_integer(
                parameter_slot_count, "parameter-slot count"
            ),
            source_template_changes=source_template_changes,
        )
        relation_digest = _canonical_digest(
            {
                "contract": "pyamplicol-recurrence-process-bijection-v1",
                "root_process": root.process_id,
                "target_process": target.process_id,
                "root_variable_semantics": _variable_semantic_digests(root),
                "target_variable_semantics": _variable_semantic_digests(target),
                "root_pairing": _pairing_proof_digests(root.fermion_pairing_catalog),
                "target_pairing": _pairing_proof_digests(
                    target.fermion_pairing_catalog
                ),
                "root_sector_proofs": [
                    [sector.closure_proof_algorithm, sector.closure_proof_digest]
                    for sector in root.physical_sectors
                ],
                "target_sector_proofs": [
                    [sector.closure_proof_algorithm, sector.closure_proof_digest]
                    for sector in target.physical_sectors
                ],
                "remap": result._digest_payload(),
            }
        )
        return result.with_digest(relation_digest)
    return None


def _fixed_semantic_digests_match(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
) -> bool:
    variable = {
        "process",
        "color-plan",
        "fermion-pairing-semantic",
        "fermion-pairing-topology",
        "closure-reconstruction",
    }
    root_rows = {row.role: row.digest for row in root.semantic_digests}
    target_rows = {row.role: row.digest for row in target.semantic_digests}
    return set(root_rows) == set(target_rows) and {
        role: digest for role, digest in root_rows.items() if role not in variable
    } == {role: digest for role, digest in target_rows.items() if role not in variable}


def _variable_semantic_digests(
    logical: RecurrenceBuilderLogicalInputV1,
) -> dict[str, str]:
    fixed = {"model-catalog", "prepared-catalog"}
    return {
        row.role: row.digest
        for row in logical.semantic_digests
        if row.role not in fixed
    }


def _source_slot_bijections(
    root: tuple[RecurrenceExternalLegV1, ...],
    target: tuple[RecurrenceExternalLegV1, ...],
) -> tuple[tuple[int, ...], ...]:
    root_groups: dict[tuple[object, ...], list[int]] = {}
    target_groups: dict[tuple[object, ...], list[int]] = {}
    for leg in root:
        root_groups.setdefault(_external_leg_orbit_key(leg), []).append(leg.source_slot)
    for leg in target:
        target_groups.setdefault(_external_leg_orbit_key(leg), []).append(
            leg.source_slot
        )
    if set(root_groups) != set(target_groups):
        return ()
    groups: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    count = 1
    for key in sorted(root_groups, key=repr):
        left = tuple(root_groups[key])
        right = tuple(target_groups[key])
        if len(left) != len(right):
            return ()
        permutations = tuple(itertools.permutations(right))
        count *= len(permutations)
        if count > _MAX_SOURCE_BIJECTIONS:
            return ()
        groups.append((left, permutations))
    result: list[tuple[int, ...]] = []
    for selection in itertools.product(*(rows[1] for rows in groups)):
        mapping = [0] * len(root)
        for (root_slots, _), target_slots in zip(groups, selection, strict=True):
            for root_slot, target_slot in zip(root_slots, target_slots, strict=True):
                mapping[root_slot] = target_slot
        result.append(tuple(mapping))
    return tuple(result)


def _external_leg_orbit_key(leg: RecurrenceExternalLegV1) -> tuple[object, ...]:
    return (
        leg.outgoing_pdg,
        leg.is_fermionic,
        tuple(
            sorted(
                (
                    state.current_state_template_id,
                    state.chirality,
                    state.spin_state,
                    state.crossing_phase.canonical_key,
                )
                for state in leg.source_states
            )
        ),
    )


def _external_source_relation(
    root: tuple[RecurrenceExternalLegV1, ...],
    target: tuple[RecurrenceExternalLegV1, ...],
    source_slots: tuple[int, ...],
) -> (
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
    ]
    | None
):
    momentum_signs: list[int] = []
    helicity_signs: list[int] = []
    source_state_offsets = [0]
    source_state_indices: list[int] = []
    source_templates: dict[int, int] = {}
    reverse_source_templates: dict[int, int] = {}
    for root_slot, target_slot in enumerate(source_slots):
        left = root[root_slot]
        right = target[target_slot]
        if (
            left.outgoing_pdg != right.outgoing_pdg
            or left.is_fermionic != right.is_fermionic
            or len(left.source_states) != len(right.source_states)
        ):
            return None
        right_states: dict[tuple[object, ...], RecurrenceSourceStateV1] = {}
        for state in right.source_states:
            key = _source_state_execution_key(state)
            if key in right_states:
                return None
            right_states[key] = state
        momentum_factor: int | None = None
        helicity_factor: int | None = None
        state_indices = [0] * len(left.source_states)
        for state in left.source_states:
            other = right_states.get(_source_state_execution_key(state))
            if other is None:
                return None
            state_indices[state.state_index] = other.state_index
            candidate_momentum = other.momentum_sign * state.momentum_sign
            if momentum_factor is None:
                momentum_factor = candidate_momentum
            elif momentum_factor != candidate_momentum:
                return None
            candidate_helicity = _signed_value_relation(
                state.public_helicity,
                other.public_helicity,
            )
            if candidate_helicity is None:
                return None
            if helicity_factor is None:
                helicity_factor = candidate_helicity
            elif helicity_factor != candidate_helicity:
                return None
            previous = source_templates.setdefault(
                state.source_template_id, other.source_template_id
            )
            reverse = reverse_source_templates.setdefault(
                other.source_template_id, state.source_template_id
            )
            if (
                previous != other.source_template_id
                or reverse != state.source_template_id
            ):
                return None
        momentum_signs.append(1 if momentum_factor is None else momentum_factor)
        helicity_signs.append(1 if helicity_factor is None else helicity_factor)
        if not _is_permutation(state_indices):
            return None
        source_state_indices.extend(state_indices)
        source_state_offsets.append(len(source_state_indices))
    changes = tuple(
        (source, target_id)
        for source, target_id in sorted(source_templates.items())
        if source != target_id
    )
    return (
        tuple(momentum_signs),
        tuple(helicity_signs),
        tuple(source_state_offsets),
        tuple(source_state_indices),
        changes,
    )


def _source_state_execution_key(
    state: RecurrenceSourceStateV1,
) -> tuple[object, ...]:
    return (
        state.current_state_template_id,
        state.chirality,
        state.spin_state,
        state.crossing_phase.canonical_key,
    )


def _signed_value_relation(left: int, right: int) -> int | None:
    if left == 0 or right == 0:
        return 1 if left == right else None
    if right == left:
        return 1
    if right == -left:
        return -1
    return None


def _sector_bijection(
    root: tuple[RecurrencePhysicalLCSectorV1, ...],
    target: tuple[RecurrencePhysicalLCSectorV1, ...],
    source_slots: tuple[int, ...],
) -> tuple[int, ...] | None:
    inverse = _inverse_permutation(source_slots)
    targets: dict[tuple[object, ...], int] = {}
    for sector in target:
        key = _sector_key(sector, inverse)
        if key in targets:
            return None
        targets[key] = sector.sector_id
    result: list[int] = []
    for sector in root:
        target_id = targets.get(_sector_key(sector, tuple(range(len(source_slots)))))
        if target_id is None:
            return None
        result.append(target_id)
    return tuple(result) if _is_permutation(result) else None


def _sector_key(
    sector: RecurrencePhysicalLCSectorV1,
    slots: tuple[int, ...],
) -> tuple[object, ...]:
    def mapped(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(slots[value] for value in values)

    return (
        sector.kind,
        slots[sector.closure_source_slot],
        sector.closure_proof_algorithm,
        tuple(
            (
                slots[row.fundamental_source_slot],
                slots[row.antifundamental_source_slot],
                mapped(row.adjoint_source_slots),
                mapped(row.singlet_source_slots),
            )
            for row in sector.open_strings
        ),
        mapped(sector.trace_source_slots),
        mapped(sector.singlet_source_slots),
        mapped(sector.word_source_slots),
    )


def _flow_bijection(
    root: tuple[RecurrencePublicLCFlowV1, ...],
    target: tuple[RecurrencePublicLCFlowV1, ...],
    source_slots: tuple[int, ...],
    sector_ids: tuple[int, ...],
) -> tuple[int, ...] | None:
    inverse_slots = _inverse_permutation(source_slots)
    inverse_sectors = _inverse_permutation(sector_ids)
    targets: dict[tuple[object, ...], int] = {}
    for flow in target:
        permutation = tuple(
            inverse_slots[flow.source_slot_permutation[source_slots[root_slot]]]
            for root_slot in range(len(source_slots))
        )
        key = (
            inverse_sectors[flow.construction_sector_id],
            tuple(inverse_slots[value] for value in flow.word_source_slots),
            permutation,
            flow.reduction_weight.canonical_key,
        )
        if key in targets:
            return None
        targets[key] = flow.flow_id
    result = []
    for flow in root:
        key = (
            flow.construction_sector_id,
            flow.word_source_slots,
            flow.source_slot_permutation,
            flow.reduction_weight.canonical_key,
        )
        target_id = targets.get(key)
        if target_id is None:
            return None
        result.append(target_id)
    return tuple(result) if _is_permutation(result) else None


def _replay_contracts_match(
    root: RecurrenceBuilderLogicalInputV1,
    target: RecurrenceBuilderLogicalInputV1,
    *,
    source_slots: tuple[int, ...],
    sector_ids: tuple[int, ...],
) -> bool:
    inverse_sectors = _inverse_permutation(sector_ids)
    left = tuple(
        _replay_partition_key(
            row,
            tuple(range(len(source_slots))),
            tuple(range(len(source_slots))),
            tuple(range(len(sector_ids))),
        )
        for row in root.replay_partitions
    )
    right = tuple(
        _replay_partition_key(
            row,
            source_slots,
            _inverse_permutation(source_slots),
            inverse_sectors,
        )
        for row in target.replay_partitions
    )
    return left == right


def _replay_partition_key(
    row: object,
    root_to_concrete: tuple[int, ...],
    concrete_to_root: tuple[int, ...],
    sectors: tuple[int, ...],
) -> tuple[object, ...]:
    def conjugate(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            concrete_to_root[values[root_to_concrete[root_slot]]]
            for root_slot in range(len(root_to_concrete))
        )

    return (
        sectors[row.representative_sector_id],
        sectors[row.materialized_sector_id],
        row.proof_algorithm,
        tuple(
            (
                sectors[target.sector_id],
                conjugate(target.external_permutation),
                conjugate(target.source_slot_permutation),
                target.amplitude_phase.canonical_key,
                target.fermion_sign,
            )
            for target in row.targets
        ),
    )


def _fermion_pairing_contracts_match(
    root: object,
    target: object,
    source_slots: tuple[int, ...],
) -> bool:
    if (root is None) != (target is None):
        return False
    if root is None:
        return True
    inverse = _inverse_permutation(source_slots)
    return _pairing_key(root, tuple(range(len(source_slots)))) == _pairing_key(
        target, inverse
    )


def _pairing_key(catalog: object, slots: tuple[int, ...]) -> tuple[object, ...]:
    concrete_to_root = slots
    root_to_concrete = _inverse_permutation(concrete_to_root)
    endpoints = tuple(
        sorted(
            (
                concrete_to_root[row.source_slot],
                row.species_id,
                row.particle_orientation,
                row.color_orientation,
                row.state_template_ids,
                row.anti_state_template_ids,
                row.basis_ids,
                row.color_representations,
            )
            for row in catalog.endpoints
        )
    )
    classes = tuple(
        sorted(
            (
                row.species_id,
                tuple(
                    concrete_to_root[value] for value in row.fundamental_source_slots
                ),
                tuple(
                    concrete_to_root[value]
                    for value in row.antifundamental_source_slots
                ),
                tuple(
                    (
                        concrete_to_root[left],
                        concrete_to_root[right],
                    )
                    for left, right in row.reference_pairings
                ),
                row.pairing_count,
            )
            for row in catalog.pairing_classes
        )
    )
    rules = tuple(
        sorted(
            (
                row.class_pairing_indices,
                tuple(
                    (
                        concrete_to_root[left],
                        concrete_to_root[right],
                    )
                    for left, right in row.endpoint_pairings
                ),
                tuple(
                    concrete_to_root[
                        row.source_slot_permutation[root_to_concrete[root_slot]]
                    ]
                    for root_slot in range(len(slots))
                ),
                tuple(
                    row.lineage_by_source_slot[root_to_concrete[root_slot]]
                    for root_slot in range(len(slots))
                ),
                row.fermion_parity,
                (
                    row.exact_factor.real_numerator,
                    row.exact_factor.real_denominator,
                    row.exact_factor.imag_numerator,
                    row.exact_factor.imag_denominator,
                ),
                row.multiplicity,
                row.proof_algorithm,
            )
            for row in catalog.rules
        )
    )
    return catalog.source_count, endpoints, classes, rules


def _pairing_proof_digests(catalog: object) -> object:
    if catalog is None:
        return None
    return {
        "semantic": catalog.semantic_digest,
        "topology": catalog.topology_digest,
        "endpoints": [row.contract_digest for row in catalog.endpoints],
        "classes": [row.proof_digest for row in catalog.pairing_classes],
        "rules": [row.proof_digest for row in catalog.rules],
    }


class RecurrenceSharedSchedule:
    """One immutable root Direct-Arena schedule."""

    __slots__ = (
        "digest",
        "index_sha256",
        "member_count",
        "process_ids",
        "sha256",
        "size_bytes",
        "source_path",
        "unpacked_size_bytes",
    )

    def __init__(
        self,
        *,
        digest: str,
        source_path: Path,
        sha256: str,
        size_bytes: int,
        member_count: int,
        unpacked_size_bytes: int,
        index_sha256: str,
        process_ids: tuple[str, ...],
    ) -> None:
        self.digest = _require_sha256(digest, "schedule digest")
        self.source_path = source_path
        self.sha256 = _require_sha256(sha256, "schedule payload SHA-256")
        self.size_bytes = _positive_integer(size_bytes, "schedule payload size")
        self.member_count = _positive_integer(member_count, "schedule member count")
        self.unpacked_size_bytes = _positive_integer(
            unpacked_size_bytes, "schedule unpacked size"
        )
        self.index_sha256 = _require_sha256(
            index_sha256, "schedule PACBIN index SHA-256"
        )
        self.process_ids = process_ids

    @property
    def artifact_path(self) -> str:
        return f"recurrence/schedules/{self.digest}/recurrence-runtime.pacbin"

    def to_mapping(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "path": self.artifact_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "member_count": self.member_count,
            "unpacked_size_bytes": self.unpacked_size_bytes,
            "index_sha256": self.index_sha256,
            "process_ids": list(self.process_ids),
        }


class RecurrenceProcessBinding:
    """One independent concrete-process binding to a root schedule."""

    __slots__ = (
        "native_schedule_semantic_digest",
        "payload",
        "process_id",
        "process_semantic_digest",
        "process_support_mask",
        "process_support_words",
        "remap",
        "schedule_digest",
        "sha256",
    )

    def __init__(
        self,
        *,
        process_id: str,
        schedule_digest: str,
        native_schedule_semantic_digest: str,
        process_semantic_digest: str,
        process_support_mask: int,
        remap: RecurrenceProcessRemap,
    ) -> None:
        self.process_id = _nonempty_text(process_id, "process ID")
        self.schedule_digest = _require_sha256(schedule_digest, "schedule digest")
        self.native_schedule_semantic_digest = _require_sha256(
            native_schedule_semantic_digest,
            "native schedule semantic digest",
        )
        self.process_semantic_digest = _require_sha256(
            process_semantic_digest, "process semantic digest"
        )
        self.process_support_mask = _single_bit(
            process_support_mask, "process support mask"
        )
        self.process_support_words = tuple(
            (self.process_support_mask >> shift) & ((1 << 64) - 1)
            for shift in range(0, max(1, self.process_support_mask.bit_length()), 64)
        )
        if not isinstance(remap, RecurrenceProcessRemap):
            raise RecurrenceScheduleSharingError(
                "recurrence process binding requires an exact process remap"
            )
        if not remap.bijection_digest:
            raise RecurrenceScheduleSharingError(
                "recurrence process remap has no authenticated bijection digest"
            )
        self.remap = remap
        self.payload = encode_recurrence_process_binding(
            process_id=self.process_id,
            schedule_digest=self.schedule_digest,
            process_semantic_digest=self.process_semantic_digest,
            process_support_mask=self.process_support_mask,
            remap=self.remap,
        )
        self.sha256 = hashlib.sha256(self.payload).hexdigest()

    @property
    def artifact_path(self) -> str:
        return f"processes/{self.process_id}/recurrence-binding.bin"

    def to_mapping(self) -> dict[str, object]:
        return {
            "abi": RECURRENCE_PROCESS_BINDING_ABI,
            "process_id": self.process_id,
            "schedule_digest": self.schedule_digest,
            "native_schedule_semantic_digest": (
                self.native_schedule_semantic_digest
            ),
            "process_semantic_digest": self.process_semantic_digest,
            "process_support_words": list(self.process_support_words),
            "remap": self.remap.to_mapping(),
            "path": "recurrence-binding.bin",
            "size_bytes": len(self.payload),
            "sha256": self.sha256,
        }


class RecurrenceScheduleSharingPlan:
    __slots__ = ("bindings", "schedules")

    def __init__(
        self,
        *,
        schedules: tuple[RecurrenceSharedSchedule, ...],
        bindings: tuple[RecurrenceProcessBinding, ...],
    ) -> None:
        self.schedules = schedules
        self.bindings = bindings

    def schedule(self, digest: str) -> RecurrenceSharedSchedule:
        for schedule in self.schedules:
            if schedule.digest == digest:
                return schedule
        raise RecurrenceScheduleSharingError(f"unknown recurrence schedule {digest!r}")

    def binding(self, process_id: str) -> RecurrenceProcessBinding:
        for binding in self.bindings:
            if binding.process_id == process_id:
                return binding
        raise RecurrenceScheduleSharingError(
            f"process {process_id!r} has no recurrence binding"
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": RECURRENCE_SCHEDULE_SHARING_KIND,
            "schema_version": RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
            "schedule_count": len(self.schedules),
            "binding_count": len(self.bindings),
            "schedule_alias_count": len(self.bindings) - len(self.schedules),
            "runtime_ownership": "root-schedule-plus-process-binding",
            "interning_phase": "before-direct-lowering",
            "schedules": [item.to_mapping() for item in self.schedules],
            "bindings": [item.to_mapping() for item in self.bindings],
        }

    def extension_mapping(self, *, index_sha256: str) -> dict[str, object]:
        return {
            "kind": RECURRENCE_SCHEDULE_SHARING_KIND,
            "schema_version": RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
            "index_path": RECURRENCE_SCHEDULE_INDEX_PATH,
            "index_sha256": _require_sha256(index_sha256, "schedule index SHA-256"),
            "schedule_count": len(self.schedules),
            "binding_count": len(self.bindings),
            "schedule_alias_count": len(self.bindings) - len(self.schedules),
            "runtime_ownership": "root-schedule-plus-process-binding",
            "interning_phase": "before-direct-lowering",
        }


def intern_recurrence_schedules(
    processes: Sequence[RecurrenceScheduleProcess],
) -> RecurrenceScheduleSharingPlan:
    """Validate and intern already pre-lowered root schedules."""

    schedules: dict[str, RecurrenceSharedSchedule] = {}
    schedule_processes: dict[str, list[str]] = {}
    schedule_native_semantics: dict[str, str] = {}
    bindings: list[RecurrenceProcessBinding] = []
    process_ids: set[str] = set()
    support_masks: set[int] = set()

    for process in processes:
        process_id = _nonempty_text(process.process_id, "recurrence process ID")
        if process_id in process_ids:
            raise RecurrenceScheduleSharingError(
                f"duplicate recurrence process binding {process_id!r}"
            )
        process_ids.add(process_id)
        support_mask = _single_bit(
            process.process_support_mask, f"process {process_id!r} support mask"
        )
        if support_mask in support_masks:
            raise RecurrenceScheduleSharingError(
                f"recurrence support mask {support_mask} is reused"
            )
        support_masks.add(support_mask)

        digest = _require_sha256(
            process.recurrence_schedule_digest,
            f"process {process_id!r} schedule digest",
        )
        native_semantic_digest = _require_sha256(
            process.recurrence_native_schedule_semantic_digest,
            f"process {process_id!r} native schedule semantic digest",
        )
        candidate = RecurrenceSharedSchedule(
            digest=digest,
            source_path=process.recurrence_schedule_path,
            sha256=process.recurrence_schedule_sha256,
            size_bytes=process.recurrence_schedule_size_bytes,
            member_count=process.recurrence_schedule_member_count,
            unpacked_size_bytes=process.recurrence_schedule_unpacked_size_bytes,
            index_sha256=process.recurrence_schedule_index_sha256,
            process_ids=(),
        )
        previous = schedules.get(digest)
        if previous is not None and (
            previous.sha256 != candidate.sha256
            or previous.size_bytes != candidate.size_bytes
            or previous.member_count != candidate.member_count
            or previous.unpacked_size_bytes != candidate.unpacked_size_bytes
            or previous.index_sha256 != candidate.index_sha256
        ):
            raise RecurrenceScheduleSharingError(
                f"schedule digest {digest} maps to different payloads"
            )
        previous_native_semantic_digest = schedule_native_semantics.get(digest)
        if (
            previous_native_semantic_digest is not None
            and previous_native_semantic_digest != native_semantic_digest
        ):
            raise RecurrenceScheduleSharingError(
                f"schedule digest {digest} maps to different native semantics"
            )
        schedules.setdefault(digest, candidate)
        schedule_native_semantics.setdefault(digest, native_semantic_digest)
        schedule_processes.setdefault(digest, []).append(process_id)
        bindings.append(
            RecurrenceProcessBinding(
                process_id=process_id,
                schedule_digest=digest,
                native_schedule_semantic_digest=native_semantic_digest,
                process_semantic_digest=process.builder_input_sha256,
                process_support_mask=support_mask,
                remap=process.recurrence_process_remap,
            )
        )

    shared = tuple(
        RecurrenceSharedSchedule(
            digest=digest,
            source_path=schedule.source_path,
            sha256=schedule.sha256,
            size_bytes=schedule.size_bytes,
            member_count=schedule.member_count,
            unpacked_size_bytes=schedule.unpacked_size_bytes,
            index_sha256=schedule.index_sha256,
            process_ids=tuple(sorted(schedule_processes[digest])),
        )
        for digest, schedule in sorted(schedules.items())
    )
    return RecurrenceScheduleSharingPlan(
        schedules=shared,
        bindings=tuple(sorted(bindings, key=lambda item: item.process_id)),
    )


def encode_recurrence_process_binding(
    *,
    process_id: str,
    schedule_digest: str,
    process_semantic_digest: str,
    process_support_mask: int,
    remap: RecurrenceProcessRemap,
) -> bytes:
    process = _nonempty_text(process_id, "process ID").encode("utf-8")
    if len(process) > 4096:
        raise RecurrenceScheduleSharingError("process ID exceeds 4096 UTF-8 bytes")
    mask = _single_bit(process_support_mask, "process support mask")
    words = tuple(
        (mask >> shift) & ((1 << 64) - 1)
        for shift in range(0, max(1, mask.bit_length()), 64)
    )
    if not isinstance(remap, RecurrenceProcessRemap) or not remap.bijection_digest:
        raise RecurrenceScheduleSharingError(
            "recurrence process binding requires an authenticated process remap"
        )
    counts = (
        len(remap.source_slots),
        len(remap.public_flow_ids),
        len(remap.physical_sector_ids),
        remap.state_template_count,
        remap.source_template_count,
        remap.direct_executor_count,
        remap.parameter_slot_count,
        len(remap.state_template_changes),
        len(remap.source_template_changes),
        len(remap.direct_executor_changes),
        len(remap.parameter_slot_changes),
    )
    for value in counts:
        if value > (1 << 32) - 1:
            raise RecurrenceScheduleSharingError(
                "recurrence process remap exceeds the u32 binding ABI"
            )
    header = b"".join(
        (
            RECURRENCE_PROCESS_BINDING_MAGIC,
            struct.pack(
                "<III",
                _PROCESS_BINDING_VERSION,
                len(process),
                len(words),
            ),
            bytes.fromhex(_require_sha256(schedule_digest, "schedule digest")),
            bytes.fromhex(
                _require_sha256(process_semantic_digest, "process semantic digest")
            ),
            bytes.fromhex(
                _require_sha256(
                    remap.bijection_digest,
                    "process-bijection digest",
                )
            ),
            struct.pack("<11I", *counts),
        )
    )
    if len(header) != _PROCESS_BINDING_FIXED_SIZE:
        raise AssertionError("recurrence process-binding header size drifted")
    sparse_changes = (
        remap.state_template_changes,
        remap.source_template_changes,
        remap.direct_executor_changes,
        remap.parameter_slot_changes,
    )
    return b"".join(
        (
            header,
            process,
            struct.pack(f"<{len(words)}Q", *words),
            struct.pack(f"<{len(remap.source_slots)}I", *remap.source_slots),
            struct.pack(
                f"<{len(remap.source_momentum_signs)}i",
                *remap.source_momentum_signs,
            ),
            struct.pack(
                f"<{len(remap.source_helicity_signs)}i",
                *remap.source_helicity_signs,
            ),
            struct.pack(
                f"<{len(remap.source_state_offsets)}I",
                *remap.source_state_offsets,
            ),
            struct.pack(
                f"<{len(remap.source_state_indices)}I",
                *remap.source_state_indices,
            ),
            struct.pack(
                f"<{len(remap.public_flow_ids)}I",
                *remap.public_flow_ids,
            ),
            struct.pack(
                f"<{len(remap.physical_sector_ids)}I",
                *remap.physical_sector_ids,
            ),
            *(
                struct.pack(
                    f"<{2 * len(changes)}I",
                    *(value for pair in changes for value in pair),
                )
                for changes in sparse_changes
            ),
        )
    )


def _schedule_plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for item in fields(value):
            if item.name == "process_id":
                result[item.name] = "<shared-schedule>"
            elif item.name in {"process_support_mask", "support_mask"}:
                result[item.name] = 1
            else:
                result[item.name] = _schedule_plain(getattr(value, item.name))
        return result
    if isinstance(value, Mapping):
        return {
            str(key): _schedule_plain(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_schedule_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_schedule_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _nonempty_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrenceScheduleSharingError(f"{context} must be nonempty text")
    return value


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecurrenceScheduleSharingError(
            f"{context} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrenceScheduleSharingError(f"{context} must be a positive integer")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrenceScheduleSharingError(f"{context} must be a nonnegative integer")
    return value


def _single_bit(value: object, context: str) -> int:
    result = _positive_integer(value, context)
    if result & (result - 1):
        raise RecurrenceScheduleSharingError(f"{context} must contain one bit")
    return result


def _permutation(values: Sequence[int], context: str) -> tuple[int, ...]:
    result = tuple(_nonnegative_integer(value, f"{context} value") for value in values)
    if not _is_permutation(result):
        raise RecurrenceScheduleSharingError(
            f"{context} must be a dense zero-based permutation"
        )
    return result


def _is_permutation(values: Sequence[int]) -> bool:
    return tuple(sorted(values)) == tuple(range(len(values)))


def _inverse_permutation(values: Sequence[int]) -> tuple[int, ...]:
    permutation = _permutation(values, "permutation")
    inverse = [0] * len(permutation)
    for source, target in enumerate(permutation):
        inverse[target] = source
    return tuple(inverse)


def _signs(
    values: Sequence[int],
    expected_count: int,
    context: str,
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != expected_count or any(value not in {-1, 1} for value in result):
        raise RecurrenceScheduleSharingError(
            f"{context} must contain {expected_count} signs in {{-1, 1}}"
        )
    return result


def _ragged_permutations(
    offsets: Sequence[int],
    values: Sequence[int],
    row_count: int,
    context: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    checked_offsets = tuple(
        _nonnegative_integer(value, f"{context} offset") for value in offsets
    )
    checked_values = tuple(
        _nonnegative_integer(value, f"{context} value") for value in values
    )
    if (
        len(checked_offsets) != row_count + 1
        or not checked_offsets
        or checked_offsets[0] != 0
        or checked_offsets[-1] != len(checked_values)
        or any(left > right for left, right in itertools.pairwise(checked_offsets))
    ):
        raise RecurrenceScheduleSharingError(
            f"{context} offsets do not partition {row_count} rows"
        )
    for row, (start, stop) in enumerate(itertools.pairwise(checked_offsets)):
        if not _is_permutation(checked_values[start:stop]):
            raise RecurrenceScheduleSharingError(
                f"{context} row {row} is not a dense zero-based permutation"
            )
    return checked_offsets, checked_values


def _sparse_permutation(
    changes: Sequence[tuple[int, int]],
    count: int,
    context: str,
) -> tuple[tuple[int, int], ...]:
    size = _nonnegative_integer(count, f"{context} domain size")
    result = tuple(changes)
    if result != tuple(sorted(result)):
        raise RecurrenceScheduleSharingError(f"{context} changes must be ordered")
    sources: set[int] = set()
    mapping = list(range(size))
    for source, target in result:
        source = _nonnegative_integer(source, f"{context} source")
        target = _nonnegative_integer(target, f"{context} target")
        if source >= size or target >= size:
            raise RecurrenceScheduleSharingError(
                f"{context} change is outside its declared domain"
            )
        if source == target:
            raise RecurrenceScheduleSharingError(
                f"{context} must omit identity changes"
            )
        if source in sources:
            raise RecurrenceScheduleSharingError(f"{context} repeats source {source}")
        sources.add(source)
        mapping[source] = target
    if not _is_permutation(mapping):
        raise RecurrenceScheduleSharingError(
            f"{context} changes do not define a bijection"
        )
    return result


def _sparse_mapping(
    count: int,
    changes: Sequence[tuple[int, int]],
) -> dict[str, object]:
    return {
        "count": _nonnegative_integer(count, "sparse mapping count"),
        "changes": [list(pair) for pair in changes],
    }


def _template_count(
    logical: RecurrenceBuilderLogicalInputV1,
    kind: str,
) -> int:
    identifiers = tuple(
        row.template_id
        for row in logical.semantic_template_references
        if row.kind == kind
    )
    if not identifiers:
        raise RecurrenceScheduleSharingError(
            f"recurrence process has no {kind!r} template domain"
        )
    if identifiers != tuple(range(len(identifiers))):
        raise RecurrenceScheduleSharingError(
            f"recurrence {kind!r} template IDs are not dense and ordered"
        )
    return len(identifiers)


def _physical_sector_domain_count(
    logical: RecurrenceBuilderLogicalInputV1,
) -> int:
    if logical.layout == "topology-replay":
        return len(logical.public_flows)
    return len(logical.physical_sectors)


__all__ = [
    "RECURRENCE_PROCESS_BINDING_ABI",
    "RECURRENCE_PROCESS_BINDING_MAGIC",
    "RECURRENCE_SCHEDULE_INDEX_PATH",
    "RECURRENCE_SCHEDULE_SHARING_KIND",
    "RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION",
    "RecurrenceProcessBinding",
    "RecurrenceProcessRemap",
    "RecurrenceScheduleLoweringCache",
    "RecurrenceScheduleSharingError",
    "RecurrenceScheduleSharingPlan",
    "RecurrenceSharedLoweringResult",
    "RecurrenceSharedSchedule",
    "encode_recurrence_process_binding",
    "exact_recurrence_process_bijection",
    "intern_recurrence_schedules",
    "recurrence_native_schedule_semantic_digest",
    "recurrence_schedule_semantic_digest",
]
