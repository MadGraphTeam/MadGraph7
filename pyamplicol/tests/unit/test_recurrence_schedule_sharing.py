# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceReplayPartitionV1,
    RecurrenceReplayTargetV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    NO_FERMION_LINE,
    ExternalFermionEndpointRowV1,
    FermionPairingCatalogV1,
    FermionPairingClassRowV1,
    FermionPairingRuleRowV1,
)
from pyamplicol.generation.recurrence_schedule_sharing import (
    RECURRENCE_PROCESS_BINDING_MAGIC,
    RecurrenceProcessRemap,
    RecurrenceScheduleLoweringCache,
    RecurrenceScheduleSharingError,
    encode_recurrence_process_binding,
    exact_recurrence_process_bijection,
    intern_recurrence_schedules,
    recurrence_schedule_semantic_digest,
)
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1 as ModelExactComplexRationalV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity_remap(
    *,
    source_count: int = 2,
    flow_count: int = 1,
    sector_count: int = 1,
) -> RecurrenceProcessRemap:
    return RecurrenceProcessRemap(
        source_slots=tuple(range(source_count)),
        source_momentum_signs=(1,) * source_count,
        source_helicity_signs=(1,) * source_count,
        source_state_offsets=tuple(range(source_count + 1)),
        source_state_indices=(0,) * source_count,
        public_flow_ids=tuple(range(flow_count)),
        physical_sector_ids=tuple(range(sector_count)),
        state_template_count=3,
        source_template_count=3,
        direct_executor_count=4,
        parameter_slot_count=1,
    ).with_digest(_digest("identity remap"))


@dataclass(frozen=True)
class _Leg:
    physical_pdg: int
    support_mask: int


@dataclass(frozen=True)
class _Logical:
    process_id: str
    process_support_mask: int
    external_legs: tuple[_Leg, ...]
    proof_digest: str


def _schedule_key(logical: _Logical) -> str:
    return recurrence_schedule_semantic_digest(
        cast(RecurrenceBuilderLogicalInputV1, logical),
        prepared_kernel_pack_digest=_digest("pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
    )


def test_prelower_identity_normalizes_only_process_ownership() -> None:
    first = _Logical("subprocess-a", 1, (_Leg(1, 1), _Leg(-1, 1)), _digest("p"))
    alias = _Logical("subprocess-b", 8, (_Leg(1, 8), _Leg(-1, 8)), _digest("p"))
    different = _Logical("subprocess-c", 2, (_Leg(2, 2), _Leg(-2, 2)), _digest("p"))

    assert _schedule_key(first) == _schedule_key(alias)
    assert _schedule_key(first) != _schedule_key(different)
    assert recurrence_schedule_semantic_digest(
        cast(RecurrenceBuilderLogicalInputV1, first),
        prepared_kernel_pack_digest=_digest("different pack"),
        direct_template_catalog_digest=_digest("templates"),
        point_tile_size=1024,
        workspace_mib=256,
    ) != _schedule_key(first)


def test_identical_process_schedules_are_lowered_once() -> None:
    cache = RecurrenceScheduleLoweringCache[str]()
    calls = 0

    def lower() -> str:
        nonlocal calls
        calls += 1
        return "root schedule"

    digest = _digest("schedule")
    assert cache.lower_once(digest, lower) == "root schedule"
    assert cache.lower_once(digest, lower) == "root schedule"
    assert calls == 1


def _pairing(
    process_id: str,
    *,
    fundamental: int,
    antifundamental: int,
    source_count: int,
) -> FermionPairingCatalogV1:
    endpoints = (
        ExternalFermionEndpointRowV1(
            endpoint_id=0,
            source_slot=antifundamental,
            public_label=antifundamental + 1,
            species_class_id=0,
            species_id="model:d",
            particle_orientation="antiparticle",
            color_orientation="antifundamental",
            state_template_ids=("dbar",),
            anti_state_template_ids=("d",),
            basis_ids=("dirac",),
            color_representations=(-3,),
            contract_digest=_digest(f"{process_id}:anti"),
        ),
        ExternalFermionEndpointRowV1(
            endpoint_id=1,
            source_slot=fundamental,
            public_label=fundamental + 1,
            species_class_id=0,
            species_id="model:d",
            particle_orientation="particle",
            color_orientation="fundamental",
            state_template_ids=("d",),
            anti_state_template_ids=("dbar",),
            basis_ids=("dirac",),
            color_representations=(3,),
            contract_digest=_digest(f"{process_id}:fund"),
        ),
    )
    pairing_class = FermionPairingClassRowV1(
        class_id=0,
        species_class_id=0,
        species_id="model:d",
        fundamental_source_slots=(fundamental,),
        antifundamental_source_slots=(antifundamental,),
        reference_pairings=((fundamental, antifundamental),),
        pairing_count=1,
        proof_digest=_digest(f"{process_id}:class"),
    )
    lineage = [NO_FERMION_LINE] * source_count
    lineage[fundamental] = 0
    lineage[antifundamental] = 0
    rule = FermionPairingRuleRowV1(
        rule_id=0,
        class_pairing_indices=((0, 0),),
        endpoint_pairings=((fundamental, antifundamental),),
        source_slot_permutation=tuple(range(source_count)),
        lineage_by_source_slot=tuple(lineage),
        fermion_parity=1,
        exact_factor=ModelExactComplexRationalV1.one(),
        multiplicity=1,
        proof_algorithm="canonical-external-fermion-pairing-v1",
        proof_digest=_digest(f"{process_id}:rule"),
    )
    return FermionPairingCatalogV1(
        process_key=process_id,
        source_count=source_count,
        endpoints=endpoints,
        pairing_classes=(pairing_class,),
        rules=(rule,),
        topology_digest=_digest(f"{process_id}:pairing-topology"),
        semantic_digest=_digest(f"{process_id}:pairing-semantic"),
    )


def _source_state(
    *,
    state_template_id: int,
    public_helicity: int,
    momentum_sign: int,
) -> RecurrenceSourceStateV1:
    return RecurrenceSourceStateV1(
        state_index=0,
        public_helicity=public_helicity,
        chirality=0,
        spin_state=public_helicity,
        current_state_template_id=state_template_id,
        source_template_id=state_template_id,
        momentum_sign=momentum_sign,
        crossing_phase=ExactComplexRationalV1(1),
    )


def _crossed_logical(*, target: bool) -> RecurrenceBuilderLogicalInputV1:
    process_id = "d_g_to_d_g" if target else "d_dbar_to_g_g"
    if target:
        outgoing = (-1, 21, 1, 21)
        physical = (1, 21, 1, 21)
        initial = (True, True, False, False)
        helicities = (-1, -1, 1, 1)
        momentum_signs = (-1, -1, 1, 1)
        state_ids = (0, 2, 1, 2)
        source_map = (0, 2, 1, 3)
    else:
        outgoing = (-1, 1, 21, 21)
        physical = (1, -1, 21, 21)
        initial = (True, True, False, False)
        helicities = (-1, 1, -1, 1)
        momentum_signs = (-1, -1, 1, 1)
        state_ids = (0, 1, 2, 2)
        source_map = (0, 1, 2, 3)
    external = tuple(
        RecurrenceExternalLegV1(
            source_slot=slot,
            public_label=slot + 1,
            physical_pdg=physical[slot],
            outgoing_pdg=outgoing[slot],
            is_initial=initial[slot],
            is_fermionic=abs(outgoing[slot]) == 1,
            source_states=(
                _source_state(
                    state_template_id=state_ids[slot],
                    public_helicity=helicities[slot],
                    momentum_sign=momentum_signs[slot],
                ),
            ),
            momentum_mask=1 << slot,
            support_mask=2 if target else 1,
        )
        for slot in range(4)
    )
    root_word = (1, 2, 3, 0)
    word = tuple(source_map[slot] for slot in root_word)
    open_string = RecurrenceLCOpenStringV1(
        fundamental_source_slot=source_map[1],
        antifundamental_source_slot=source_map[0],
        adjoint_source_slots=(source_map[2], source_map[3]),
    )
    sector = RecurrencePhysicalLCSectorV1(
        sector_id=0,
        public_id=f"{process_id}:sector",
        kind="open-lines",
        closure_source_slot=source_map[0],
        closure_proof_algorithm="canonical-lc-closure-anchor-v2",
        closure_proof_digest=_digest(f"{process_id}:closure"),
        open_strings=(open_string,),
        word_source_slots=word,
        support_mask=2 if target else 1,
    )
    flow = RecurrencePublicLCFlowV1(
        flow_id=0,
        public_id="flow:" + ",".join(str(slot + 1) for slot in word),
        construction_sector_id=0,
        word_source_slots=word,
        source_slot_permutation=(0, 1, 2, 3),
    )
    replay_target = RecurrenceReplayTargetV1(
        sector_id=0,
        external_permutation=(0, 1, 2, 3),
        source_slot_permutation=(0, 1, 2, 3),
    )
    replay = RecurrenceReplayPartitionV1(
        representative_sector_id=0,
        materialized_sector_id=0,
        proof_algorithm="exact-crossing-replay-v1",
        proof_digest=_digest(f"{process_id}:replay"),
        targets=(replay_target,),
    )
    references = tuple(
        RecurrenceSemanticTemplateReferenceV1(
            kind=kind,
            template_id=index,
            semantic_digest=_digest(f"{kind}:{index}"),
        )
        for kind in ("current-state", "source")
        for index in range(3)
    )
    semantics = (
        *(
            RecurrenceSemanticDigestV1(role, _digest(f"{process_id}:{role}"))
            for role in (
                "process",
                "color-plan",
                "fermion-pairing-semantic",
                "fermion-pairing-topology",
                "closure-reconstruction",
            )
        ),
        RecurrenceSemanticDigestV1("model-catalog", _digest("model")),
        RecurrenceSemanticDigestV1("prepared-catalog", _digest("prepared")),
    )
    return RecurrenceBuilderLogicalInputV1(
        process_id=process_id,
        layout="topology-replay",
        semantic_digests=semantics,
        external_legs=external,
        physical_sectors=(sector,),
        public_flows=(flow,),
        semantic_template_references=references,
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(2 if target else 1),
            "binding-local-normalization",
            _digest(f"{process_id}:normalization"),
        ),
        fermion_pairing_catalog=_pairing(
            process_id,
            fundamental=source_map[1],
            antifundamental=source_map[0],
            source_count=4,
        ),
        replay_partitions=(replay,),
        coupling_limits=(RecurrenceCouplingLimitV1("QCD", 2, 2),),
        parameter_projection=(RecurrenceParameterProjectionV1(0, "alpha_s", 0, 0),),
        process_support_mask=2 if target else 1,
    )


def test_crossed_processes_share_only_through_an_exact_bijection() -> None:
    root = _crossed_logical(target=False)
    target = _crossed_logical(target=True)
    remap = exact_recurrence_process_bijection(
        root,
        target,
        direct_executor_count=4,
        parameter_slot_count=1,
    )
    assert remap is not None
    assert remap.source_slots == (0, 2, 1, 3)
    assert remap.source_momentum_signs == (1, -1, -1, 1)
    assert remap.source_helicity_signs == (1, 1, 1, 1)
    assert remap.source_state_offsets == (0, 1, 2, 3, 4)
    assert remap.source_state_indices == (0, 0, 0, 0)
    assert remap.public_flow_ids == (0,)
    assert remap.physical_sector_ids == (0,)

    cache = RecurrenceScheduleLoweringCache[str]()
    calls = 0

    def lower(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    first = cache.lower_process(
        root,
        schedule_digest=_digest("root schedule"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("root"),
    )
    second = cache.lower_process(
        target,
        schedule_digest=_digest("target schedule"),
        direct_executor_count=4,
        parameter_slot_count=1,
        lower=lambda: lower("target"),
    )
    assert calls == 1
    assert first.output == second.output == "root"
    assert second.schedule_digest == first.schedule_digest
    assert second.remap == remap


def test_cross_process_aliasing_fails_closed_on_contract_changes() -> None:
    root = _crossed_logical(target=False)
    target = _crossed_logical(target=True)

    changed_parameter = replace(
        target,
        parameter_projection=(
            replace(target.parameter_projection[0], prepared_parameter_id=None),
        ),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            changed_parameter,
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )

    changed_flow = replace(
        target,
        public_flows=(
            replace(
                target.public_flows[0],
                reduction_weight=ExactComplexRationalV1(2),
            ),
        ),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            changed_flow,
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )

    changed_templates = list(target.semantic_template_references)
    changed_templates[0] = replace(
        changed_templates[0],
        semantic_digest=_digest("different kernel contract"),
    )
    assert (
        exact_recurrence_process_bijection(
            root,
            replace(target, semantic_template_references=tuple(changed_templates)),
            direct_executor_count=4,
            parameter_slot_count=1,
        )
        is None
    )


def _process(
    root: Path,
    *,
    process_id: str,
    schedule_digest: str,
    payload: bytes,
    support_mask: int,
) -> SimpleNamespace:
    path = root / f"{schedule_digest}-{process_id}.pacbin"
    path.write_bytes(payload)
    return SimpleNamespace(
        process_id=process_id,
        recurrence_schedule_path=path,
        recurrence_schedule_digest=schedule_digest,
        recurrence_schedule_size_bytes=len(payload),
        recurrence_schedule_sha256=hashlib.sha256(payload).hexdigest(),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=len(payload),
        recurrence_schedule_index_sha256=_digest(f"index-{schedule_digest}"),
        builder_input_sha256=_digest(f"binding-{process_id}"),
        process_support_mask=support_mask,
        recurrence_process_remap=_identity_remap(),
    )


def test_bounded_process_set_interns_before_publication(tmp_path: Path) -> None:
    shared = _digest("shared schedule")
    distinct = _digest("distinct schedule")
    plan = intern_recurrence_schedules(
        (
            _process(
                tmp_path,
                process_id="u_ubar_to_g_g",
                schedule_digest=shared,
                payload=b"shared",
                support_mask=1,
            ),
            _process(
                tmp_path,
                process_id="d_dbar_to_g_g",
                schedule_digest=shared,
                payload=b"shared",
                support_mask=2,
            ),
            _process(
                tmp_path,
                process_id="g_g_to_g_g",
                schedule_digest=distinct,
                payload=b"distinct",
                support_mask=4,
            ),
        )
    )

    assert len(plan.bindings) == 3
    assert len(plan.schedules) == 2
    assert plan.to_mapping()["runtime_ownership"] == (
        "root-schedule-plus-process-binding"
    )
    assert plan.to_mapping()["interning_phase"] == "before-direct-lowering"
    assert plan.binding("u_ubar_to_g_g").artifact_path.endswith(
        "/recurrence-binding.bin"
    )


def test_binding_payload_is_compact_and_process_owned() -> None:
    schedule = _digest("schedule")
    semantic = _digest("process")
    payload = encode_recurrence_process_binding(
        process_id="u_ubar_to_g_g",
        schedule_digest=schedule,
        process_semantic_digest=semantic,
        process_support_mask=1 << 70,
        remap=_identity_remap(),
    )
    version, process_len, word_count = struct.unpack_from("<III", payload, 8)
    assert payload[:8] == RECURRENCE_PROCESS_BINDING_MAGIC
    assert version == 2
    assert word_count == 2
    assert payload[20:52] == bytes.fromhex(schedule)
    assert payload[52:84] == bytes.fromhex(semantic)
    assert payload[84:116] == bytes.fromhex(_digest("identity remap"))
    assert payload[160 : 160 + process_len] == b"u_ubar_to_g_g"
    assert len(payload) < 512


def test_same_schedule_digest_cannot_name_different_payloads(
    tmp_path: Path,
) -> None:
    digest = _digest("schedule")
    with pytest.raises(
        RecurrenceScheduleSharingError,
        match="different payloads",
    ):
        intern_recurrence_schedules(
            (
                _process(
                    tmp_path,
                    process_id="first",
                    schedule_digest=digest,
                    payload=b"one",
                    support_mask=1,
                ),
                _process(
                    tmp_path,
                    process_id="second",
                    schedule_digest=digest,
                    payload=b"two",
                    support_mask=2,
                ),
            )
        )


def test_process_support_bits_are_independent() -> None:
    first = SimpleNamespace(
        process_id="first",
        recurrence_schedule_path=Path(__file__),
        recurrence_schedule_digest=_digest("schedule"),
        recurrence_schedule_size_bytes=Path(__file__).stat().st_size,
        recurrence_schedule_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=Path(__file__).stat().st_size,
        recurrence_schedule_index_sha256=_digest("index"),
        builder_input_sha256=_digest("first"),
        process_support_mask=1,
        recurrence_process_remap=_identity_remap(),
    )
    second = SimpleNamespace(**{**vars(first), "process_id": "second"})
    with pytest.raises(RecurrenceScheduleSharingError, match="support mask"):
        intern_recurrence_schedules((first, second))


@pytest.mark.skip(
    reason=(
        "full p p > j j j j generation belongs to the 30 GiB guarded "
        "post-build process-set acceptance run"
    )
)
def test_full_pp_to_four_jets_has_fewer_schedules_than_subprocesses() -> None:
    """Acceptance placeholder retained until the native split build is available."""
