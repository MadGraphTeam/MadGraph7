# SPDX-License-Identifier: 0BSD
"""Contracts for the Python-to-Rust recurrence builder input."""

from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import pyamplicol.generation.recurrence_columnar as recurrence_columnar
from pyamplicol.generation.recurrence_columnar import (
    RECURRENCE_BUILDER_INPUT_ABI,
    ExactComplexRationalV1,
    RecurrenceBuilderInputV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceColumn,
    RecurrenceColumnarInputError,
    RecurrenceColumnarTable,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceReplayPartitionV1,
    RecurrenceReplayTargetV1,
    RecurrenceSelectedSourceCoverageV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
    build_recurrence_builder_input_v1,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _template(
    kind: str,
    template_id: int,
    *,
    prepared_kernel_id: int | None = None,
) -> RecurrenceSemanticTemplateReferenceV1:
    return RecurrenceSemanticTemplateReferenceV1(
        kind=kind,
        template_id=template_id,
        semantic_digest=_sha256(f"{kind}:{template_id}"),
        prepared_kernel_id=prepared_kernel_id,
    )


def _logical_input(
    *,
    reverse_unordered_records: bool = False,
) -> RecurrenceBuilderLogicalInputV1:
    legs = (
        RecurrenceExternalLegV1(
            source_slot=0,
            public_label=1,
            physical_pdg=1,
            outgoing_pdg=1,
            is_initial=True,
            is_fermionic=True,
            source_states=(
                RecurrenceSourceStateV1(0, -1, 0, -1, 10, 20),
                RecurrenceSourceStateV1(1, 1, 0, 1, 10, 21),
            ),
            momentum_mask=(1 << 130) | (1 << 65) | 1,
            support_mask=1,
        ),
        RecurrenceExternalLegV1(
            source_slot=1,
            public_label=2,
            physical_pdg=-1,
            outgoing_pdg=-1,
            is_initial=True,
            is_fermionic=True,
            source_states=(
                RecurrenceSourceStateV1(0, -1, 0, -1, 11, 22),
                RecurrenceSourceStateV1(1, 1, 0, 1, 11, 23),
            ),
            momentum_mask=1 << 1,
            support_mask=1 << 1,
        ),
        RecurrenceExternalLegV1(
            source_slot=2,
            public_label=3,
            physical_pdg=21,
            outgoing_pdg=21,
            is_initial=False,
            is_fermionic=False,
            source_states=(
                RecurrenceSourceStateV1(0, -1, 0, -1, 12, 24),
                RecurrenceSourceStateV1(1, 1, 0, 1, 12, 25),
            ),
            momentum_mask=1 << 2,
            support_mask=1 << 2,
        ),
        RecurrenceExternalLegV1(
            source_slot=3,
            public_label=4,
            physical_pdg=21,
            outgoing_pdg=21,
            is_initial=False,
            is_fermionic=False,
            source_states=(
                RecurrenceSourceStateV1(0, -1, 0, -1, 13, 26),
                RecurrenceSourceStateV1(1, 1, 0, 1, 13, 27),
            ),
            momentum_mask=1 << 3,
            support_mask=1 << 3,
        ),
    )
    sectors = (
        RecurrencePhysicalLCSectorV1(
            sector_id=0,
            public_id="flow:1,3,4,2",
            kind="open-lines",
            closure_source_slot=1,
            closure_proof_algorithm="canonical-lc-closure-anchor-v1",
            closure_proof_digest=_sha256("closure-anchor:0"),
            open_strings=(RecurrenceLCOpenStringV1(0, 1, (2, 3)),),
            word_source_slots=(0, 2, 3, 1),
            support_mask=0b1111,
        ),
        RecurrencePhysicalLCSectorV1(
            sector_id=1,
            public_id="flow:1,4,3,2",
            kind="open-lines",
            closure_source_slot=1,
            closure_proof_algorithm="canonical-lc-closure-anchor-v1",
            closure_proof_digest=_sha256("closure-anchor:1"),
            open_strings=(RecurrenceLCOpenStringV1(0, 1, (3, 2)),),
            word_source_slots=(0, 3, 2, 1),
            support_mask=0b1111,
        ),
    )
    public_flows = (
        RecurrencePublicLCFlowV1(
            0,
            "flow:1,3,4,2",
            0,
            (0, 2, 3, 1),
            (0, 1, 2, 3),
        ),
        RecurrencePublicLCFlowV1(
            1,
            "flow:1,4,3,2",
            1,
            (0, 3, 2, 1),
            (0, 1, 2, 3),
        ),
    )
    targets = (
        RecurrenceReplayTargetV1(
            sector_id=0,
            external_permutation=(0, 1, 2, 3),
            source_slot_permutation=(0, 1, 2, 3),
        ),
        RecurrenceReplayTargetV1(
            sector_id=1,
            external_permutation=(0, 1, 3, 2),
            source_slot_permutation=(0, 1, 3, 2),
            amplitude_phase=ExactComplexRationalV1(-1),
            fermion_sign=-1,
        ),
    )
    partition = RecurrenceReplayPartitionV1(
        representative_sector_id=0,
        materialized_sector_id=0,
        proof_algorithm="exact-source-permutation-v1",
        proof_digest=_sha256("replay-proof"),
        targets=targets[::-1] if reverse_unordered_records else targets,
    )
    templates = (
        *(_template("current-state", value) for value in range(10, 14)),
        *(_template("source", value) for value in range(20, 28)),
        *(
            _template("transition", 40, prepared_kernel_id=400),
            _template("propagator", 41, prepared_kernel_id=401),
            _template("closure", 42, prepared_kernel_id=402),
            _template("parameter", 50),
            _template("parameter", 51),
        ),
    )
    semantic_digests = (
        RecurrenceSemanticDigestV1("process", _sha256("process")),
        RecurrenceSemanticDigestV1("model-catalog", _sha256("model")),
        RecurrenceSemanticDigestV1("prepared-catalog", _sha256("prepared")),
        RecurrenceSemanticDigestV1("color-plan", _sha256("color")),
    )
    coupling_limits = (
        RecurrenceCouplingLimitV1("QCD", 0, 4),
        RecurrenceCouplingLimitV1("QED", 0, 2),
    )
    parameters = (
        RecurrenceParameterProjectionV1(0, "alpha_s", 50, 0),
        RecurrenceParameterProjectionV1(1, "mass_6", 51, 1),
    )
    if reverse_unordered_records:
        legs = legs[::-1]
        sectors = sectors[::-1]
        public_flows = public_flows[::-1]
        templates = templates[::-1]
        semantic_digests = semantic_digests[::-1]
        coupling_limits = coupling_limits[::-1]
        parameters = parameters[::-1]
    return RecurrenceBuilderLogicalInputV1(
        process_id="d_dbar_to_g_g",
        layout="topology-replay",
        semantic_digests=semantic_digests,
        external_legs=legs,
        physical_sectors=sectors,
        public_flows=public_flows,
        semantic_template_references=templates,
        normalization=RecurrenceNormalizationV1(
            factor=ExactComplexRationalV1(1, 4),
            convention="spin-colour-average-v1",
            semantic_digest=_sha256("normalization"),
        ),
        replay_partitions=(partition,),
        selected_public_flow_ids=(1, 0),
        selected_source_coverage=(
            RecurrenceSelectedSourceCoverageV1(1, (1,)),
            RecurrenceSelectedSourceCoverageV1(0, (0,)),
        ),
        coupling_limits=coupling_limits,
        parameter_projection=parameters,
        process_support_mask=(1 << 131) - 1,
    )


@pytest.fixture
def logical_input() -> RecurrenceBuilderLogicalInputV1:
    return _logical_input()


@pytest.fixture
def columnar_input(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> RecurrenceBuilderInputV1:
    return build_recurrence_builder_input_v1(logical_input)


def _table_bytes(value: RecurrenceBuilderInputV1) -> tuple[object, ...]:
    return tuple(
        (
            table.name,
            table.row_count,
            tuple(
                (
                    column.name,
                    column.values.dtype.str,
                    column.values.shape,
                    column.values.tobytes(order="C"),
                )
                for column in table.columns
            ),
        )
        for table in value.tables
    )


def _replace_column(
    value: RecurrenceBuilderInputV1,
    table_name: str,
    column_name: str,
    replacement: np.ndarray[object, object],
) -> tuple[RecurrenceColumnarTable, ...]:
    replacement.flags.writeable = False
    old = value.table(table_name)
    columns = tuple(
        RecurrenceColumn(column.name, replacement)
        if column.name == column_name
        else column
        for column in old.columns
    )
    changed = RecurrenceColumnarTable(old.name, old.row_count, columns)
    return tuple(
        changed if table.name == table_name else table for table in value.tables
    )


def _rebuild_bitset(value: RecurrenceBuilderInputV1, bitset_id: int) -> int:
    ranges = value.table("bitset_ranges")
    words = value.table("bitset_words").column("value")
    start = int(ranges.column("start")[bitset_id])
    count = int(ranges.column("count")[bitset_id])
    return sum(int(words[start + index]) << (64 * index) for index in range(count))


def test_contract_has_owned_read_only_little_endian_tables(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    assert columnar_input.abi == RECURRENCE_BUILDER_INPUT_ABI
    assert tuple(table.name for table in columnar_input.tables) == tuple(
        sorted(table.name for table in columnar_input.tables)
    )
    expected = {
        "header",
        "header_digests",
        "external_legs",
        "source_states",
        "physical_lc_sectors",
        "public_lc_flows",
        "lc_open_strings",
        "replay_partitions",
        "replay_targets",
        "selected_public_flow_coverage",
        "selected_source_coverage",
        "coupling_limits",
        "bitset_ranges",
        "bitset_words",
        "semantic_template_references",
        "normalization",
        "parameter_projection",
        "string_ranges",
        "string_bytes",
        "u32_sequence_ranges",
        "u32_sequence_values",
    }
    assert expected <= {table.name for table in columnar_input.tables}
    header = columnar_input.table("header")
    assert int(header.column("public_flow_count")[0]) == 2
    sectors = columnar_input.table("physical_lc_sectors")
    assert tuple(int(value) for value in sectors.column("closure_source_slot")) == (
        1,
        1,
    )
    assert sectors.column("closure_proof_digest_id").dtype.str == "<u4"
    assert tuple(
        int(value)
        for value in columnar_input.table("external_legs").column("is_fermionic")
    ) == (1, 1, 0, 0)
    for table in columnar_input.tables:
        for column in table.columns:
            assert column.values.dtype != np.dtype("O")
            assert column.values.flags.c_contiguous
            assert column.values.flags.owndata
            assert not column.values.flags.writeable
            assert column.values.dtype.str[0] in {"<", "|"}


def test_digest_and_bytes_ignore_nonsemantic_record_order() -> None:
    first = build_recurrence_builder_input_v1(
        _logical_input(reverse_unordered_records=False)
    )
    second = build_recurrence_builder_input_v1(
        _logical_input(reverse_unordered_records=True)
    )

    assert first.canonical_digest == second.canonical_digest
    assert _table_bytes(first) == _table_bytes(second)
    first.require_digest(first.canonical_digest)
    with pytest.raises(RecurrenceColumnarInputError, match="does not match"):
        first.require_digest(_sha256("stale"))

    changed_sector = replace(
        _logical_input().physical_sectors[0],
        closure_proof_digest=_sha256("different-closure-anchor-proof"),
    )
    changed = build_recurrence_builder_input_v1(
        replace(
            _logical_input(),
            physical_sectors=(changed_sector, *_logical_input().physical_sectors[1:]),
        )
    )
    assert changed.canonical_digest != first.canonical_digest


def test_multiword_masks_are_flat_little_endian_u64(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    legs = columnar_input.table("external_legs")
    mask_id = int(legs.column("momentum_mask_id")[0])
    ranges = columnar_input.table("bitset_ranges")
    start = int(ranges.column("start")[mask_id])
    count = int(ranges.column("count")[mask_id])
    words = columnar_input.table("bitset_words").column("value")

    assert count == 3
    assert words.dtype.str == "<u8"
    assert int(words[start]) == 1
    assert int(words[start + 1]) == 2
    assert int(words[start + 2]) == 4
    assert _rebuild_bitset(columnar_input, mask_id) == (1 << 130) | (1 << 65) | 1


def test_flat_catalog_ranges_cover_values_exactly(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    for range_name, value_name in (
        ("string_ranges", "string_bytes"),
        ("u32_sequence_ranges", "u32_sequence_values"),
        ("bitset_ranges", "bitset_words"),
    ):
        ranges = columnar_input.table(range_name)
        cursor = 0
        for row in range(ranges.row_count):
            assert int(ranges.column("start")[row]) == cursor
            cursor += int(ranges.column("count")[row])
        assert cursor == columnar_input.table(value_name).row_count


def test_contract_is_independent_of_generic_dag() -> None:
    source = inspect.getsource(recurrence_columnar)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(
        isinstance(node, ast.Name) and node.id == "GenericDAG"
        for node in ast.walk(tree)
    )
    assert not any("dag_types" in module for module in imported_modules)
    header = build_recurrence_builder_input_v1(_logical_input()).table("header")
    assert header.row_count == 1


def test_exact_factor_requires_canonical_reduced_fractions() -> None:
    with pytest.raises(RecurrenceColumnarInputError, match="must be reduced"):
        ExactComplexRationalV1(2, 4)
    with pytest.raises(RecurrenceColumnarInputError, match="encoded as 0/1"):
        ExactComplexRationalV1(0, 2)
    with pytest.raises(RecurrenceColumnarInputError, match="must be positive"):
        ExactComplexRationalV1(1, -2)


def test_checked_integer_bounds_reject_overflow_and_negative_masks() -> None:
    with pytest.raises(RecurrenceColumnarInputError, match="does not fit u32"):
        RecurrenceSourceStateV1(1 << 32, -1, 0, -1, 0, 0)
    with pytest.raises(RecurrenceColumnarInputError, match="must be nonnegative"):
        replace(_logical_input().external_legs[0], momentum_mask=-1)
    with pytest.raises(RecurrenceColumnarInputError, match="does not fit u64"):
        RecurrenceColumnarTable("overflow", 1 << 64, ())


def test_column_rejects_mutable_view_noncontiguous_and_big_endian_storage() -> None:
    writable = np.array([1], dtype="<u4")
    with pytest.raises(RecurrenceColumnarInputError, match="read-only"):
        RecurrenceColumn("value", writable)

    base = np.arange(4, dtype="<u4")
    view = base[::2]
    view.flags.writeable = False
    with pytest.raises(RecurrenceColumnarInputError, match="C-contiguous"):
        RecurrenceColumn("value", view)

    owning_view = base.view()
    owning_view.flags.writeable = False
    with pytest.raises(RecurrenceColumnarInputError, match="own its storage"):
        RecurrenceColumn("value", owning_view)

    big_endian = np.array([1], dtype=">u4")
    big_endian.flags.writeable = False
    with pytest.raises(RecurrenceColumnarInputError, match="non-little-endian"):
        RecurrenceColumn("value", big_endian)


def test_malformed_flat_range_is_rejected(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    counts = columnar_input.table("string_ranges").column("count").copy()
    counts[0] += 1
    tables = _replace_column(columnar_input, "string_ranges", "count", counts)

    with pytest.raises(RecurrenceColumnarInputError, match="ranges are not contiguous"):
        RecurrenceBuilderInputV1(columnar_input.abi, tables)


def test_malformed_reference_and_header_count_are_rejected(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    digest_ids = columnar_input.table("header_digests").column("digest_id").copy()
    digest_ids[0] = columnar_input.table("digest_catalog").row_count
    tables = _replace_column(
        columnar_input,
        "header_digests",
        "digest_id",
        digest_ids,
    )
    with pytest.raises(RecurrenceColumnarInputError, match="absent row"):
        RecurrenceBuilderInputV1(columnar_input.abi, tables)

    counts = columnar_input.table("header").column("external_leg_count").copy()
    counts[0] += 1
    tables = _replace_column(
        columnar_input,
        "header",
        "external_leg_count",
        counts,
    )
    with pytest.raises(RecurrenceColumnarInputError, match="does not match"):
        RecurrenceBuilderInputV1(columnar_input.abi, tables)


def test_malformed_little_endian_column_width_is_rejected(
    columnar_input: RecurrenceBuilderInputV1,
) -> None:
    wrong_width = np.array(
        columnar_input.table("header").column("schema_version"),
        dtype="<u8",
        copy=True,
    )
    tables = _replace_column(
        columnar_input,
        "header",
        "schema_version",
        wrong_width,
    )

    with pytest.raises(RecurrenceColumnarInputError, match="expected '<u4'"):
        RecurrenceBuilderInputV1(columnar_input.abi, tables)


def test_logical_cross_references_fail_closed(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    with pytest.raises(RecurrenceColumnarInputError, match="absent current-state"):
        build_recurrence_builder_input_v1(
            replace(
                logical_input,
                semantic_template_references=tuple(
                    item
                    for item in logical_input.semantic_template_references
                    if not (item.kind == "current-state" and item.template_id == 10)
                ),
            )
        )

    with pytest.raises(RecurrenceColumnarInputError, match="selected public flow"):
        build_recurrence_builder_input_v1(
            replace(logical_input, selected_public_flow_ids=(99,))
        )

    bad_anchor = replace(logical_input.physical_sectors[0], closure_source_slot=99)
    with pytest.raises(RecurrenceColumnarInputError, match="closure source slot"):
        build_recurrence_builder_input_v1(
            replace(
                logical_input,
                physical_sectors=(bad_anchor, *logical_input.physical_sectors[1:]),
            )
        )

    bad_target = replace(
        logical_input.replay_partitions[0].targets[0],
        external_permutation=(0, 1, 2, 2),
    )
    partition = replace(
        logical_input.replay_partitions[0],
        targets=(bad_target, logical_input.replay_partitions[0].targets[1]),
    )
    with pytest.raises(RecurrenceColumnarInputError, match="must be a permutation"):
        build_recurrence_builder_input_v1(
            replace(logical_input, replay_partitions=(partition,))
        )

    bad_flow = replace(
        logical_input.public_flows[0],
        construction_sector_id=99,
    )
    with pytest.raises(
        RecurrenceColumnarInputError,
        match="public flow construction sector",
    ):
        build_recurrence_builder_input_v1(
            replace(
                logical_input,
                public_flows=(bad_flow, *logical_input.public_flows[1:]),
            )
        )


def test_missing_required_header_digest_is_rejected(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    with pytest.raises(RecurrenceColumnarInputError, match="missing semantic digests"):
        build_recurrence_builder_input_v1(
            replace(
                logical_input,
                semantic_digests=tuple(
                    item
                    for item in logical_input.semantic_digests
                    if item.role != "prepared-catalog"
                ),
            )
        )


def test_replay_permutations_use_documented_gather_orientation(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    cycle = (0, 2, 3, 1)
    representative_word = logical_input.physical_sectors[0].word_source_slots
    target_word = tuple(cycle[source_slot] for source_slot in representative_word)
    target_sector = replace(
        logical_input.physical_sectors[1],
        word_source_slots=target_word,
        closure_source_slot=target_word[-1],
    )
    target_flow = replace(
        logical_input.public_flows[1],
        word_source_slots=target_word,
    )
    targets = tuple(
        replace(
            target,
            external_permutation=cycle,
            source_slot_permutation=cycle,
        )
        if target.sector_id == 1
        else target
        for target in logical_input.replay_partitions[0].targets
    )
    logical = replace(
        logical_input,
        physical_sectors=(logical_input.physical_sectors[0], target_sector),
        public_flows=(logical_input.public_flows[0], target_flow),
        replay_partitions=(
            replace(logical_input.replay_partitions[0], targets=targets),
        ),
    )

    assert len(build_recurrence_builder_input_v1(logical).canonical_digest) == 64
    inverse = (0, 3, 1, 2)
    wrong_targets = tuple(
        replace(
            target,
            external_permutation=inverse,
            source_slot_permutation=inverse,
        )
        if target.sector_id == 1
        else target
        for target in targets
    )
    with pytest.raises(RecurrenceColumnarInputError, match="gather permutation"):
        build_recurrence_builder_input_v1(
            replace(
                logical,
                replay_partitions=(
                    replace(logical.replay_partitions[0], targets=wrong_targets),
                ),
            )
        )


def test_replay_permutation_must_transport_all_singlet_closure_anchor(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    singlet_sectors = tuple(
        replace(
            sector,
            kind="singlet",
            closure_source_slot=0,
            open_strings=(),
            singlet_source_slots=(0, 1, 2, 3),
            word_source_slots=(),
        )
        for sector in logical_input.physical_sectors
    )
    singlet_flows = tuple(
        replace(flow, word_source_slots=()) for flow in logical_input.public_flows
    )
    targets = tuple(
        replace(
            target,
            external_permutation=(1, 0, 2, 3),
            source_slot_permutation=(1, 0, 2, 3),
        )
        if target.sector_id == 1
        else target
        for target in logical_input.replay_partitions[0].targets
    )

    with pytest.raises(
        RecurrenceColumnarInputError,
        match="representative closure anchor",
    ):
        build_recurrence_builder_input_v1(
            replace(
                logical_input,
                physical_sectors=singlet_sectors,
                public_flows=singlet_flows,
                replay_partitions=(
                    replace(logical_input.replay_partitions[0], targets=targets),
                ),
            )
        )


def test_union_layout_does_not_require_replay_partitions(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    result = build_recurrence_builder_input_v1(
        replace(
            logical_input,
            layout="all-flow-union",
            replay_partitions=(),
        )
    )

    assert result.table("replay_partitions").row_count == 0
    assert int(result.table("header").column("layout")[0]) == 1


def test_topology_replay_layout_allows_exact_residual_only_sectors(
    logical_input: RecurrenceBuilderLogicalInputV1,
) -> None:
    result = build_recurrence_builder_input_v1(
        replace(logical_input, replay_partitions=())
    )

    assert result.table("replay_partitions").row_count == 0
    assert result.table("replay_targets").row_count == 0
    assert int(result.table("header").column("layout")[0]) == 0
