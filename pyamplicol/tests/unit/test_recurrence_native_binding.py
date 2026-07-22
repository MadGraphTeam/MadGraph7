# SPDX-License-Identifier: 0BSD
"""Private Python/Rust recurrence builder-input boundary tests."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from pyamplicol import _rusticol
from pyamplicol.generation.recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderInputV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceExternalLegV1,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
    build_recurrence_builder_input_v1,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _input() -> RecurrenceBuilderInputV1:
    logical = RecurrenceBuilderLogicalInputV1(
        process_id="d_dbar_to_g",
        layout="all-flow-union",
        semantic_digests=tuple(
            RecurrenceSemanticDigestV1(role, _sha256(role))
            for role in (
                "process",
                "model-catalog",
                "prepared-catalog",
                "color-plan",
            )
        ),
        external_legs=(
            RecurrenceExternalLegV1(
                source_slot=0,
                public_label=1,
                physical_pdg=1,
                outgoing_pdg=1,
                is_initial=True,
                is_fermionic=True,
                source_states=(
                    RecurrenceSourceStateV1(0, -1, 0, -1, 10, 20),
                ),
                momentum_mask=1,
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
                    RecurrenceSourceStateV1(0, 1, 0, 1, 11, 21),
                ),
                momentum_mask=1 << 1,
                support_mask=1 << 1,
            ),
        ),
        physical_sectors=(
            RecurrencePhysicalLCSectorV1(
                sector_id=0,
                public_id="flow:1,2",
                kind="open-lines",
                closure_source_slot=1,
                closure_proof_algorithm="canonical-lc-closure-anchor-v1",
                closure_proof_digest=_sha256("closure-anchor:flow:1,2"),
                open_strings=(RecurrenceLCOpenStringV1(0, 1),),
                word_source_slots=(0, 1),
                support_mask=0b11,
            ),
        ),
        public_flows=(
            RecurrencePublicLCFlowV1(
                0,
                "flow:1,2",
                0,
                (0, 1),
                (0, 1),
            ),
        ),
        semantic_template_references=(
            RecurrenceSemanticTemplateReferenceV1(
                "current-state", 10, _sha256("current-state:10")
            ),
            RecurrenceSemanticTemplateReferenceV1(
                "current-state", 11, _sha256("current-state:11")
            ),
            RecurrenceSemanticTemplateReferenceV1(
                "source", 20, _sha256("source:20"), prepared_kernel_id=100
            ),
            RecurrenceSemanticTemplateReferenceV1(
                "source", 21, _sha256("source:21"), prepared_kernel_id=101
            ),
        ),
        normalization=RecurrenceNormalizationV1(
            ExactComplexRationalV1(1, 4),
            "spin-colour-average-v1",
            _sha256("normalization"),
        ),
        process_support_mask=(1 << 131) - 1,
    )
    return build_recurrence_builder_input_v1(logical)


def test_native_recurrence_identity_validation_is_deterministic() -> None:
    value = _input()

    first = _rusticol._validate_recurrence_builder_input_v1(value)
    second = _rusticol._validate_recurrence_builder_input_v1(value)

    assert first == second
    assert first["kind"] == "pyamplicol-recurrence-builder-validation-result"
    assert first["builder_input_sha256"] == value.canonical_digest
    assert first["validation_status"] == "validated-identity-only"
    assert not first["schedule_constructed"]
    summary = first["inspection_summary"]
    assert summary["process_id"] == "d_dbar_to_g"
    assert summary["lc_flow_layout"] == "all-flow-union"
    assert summary["public_flow_count"] == 1
    assert summary["maximum_mask_bit"] == 130
    assert summary["selector_mask_word_count"] >= 3


def test_native_recurrence_binding_rejects_stale_digest() -> None:
    value = _input()
    stale = SimpleNamespace(abi=value.abi, tables=value.tables, digest="0" * 64)

    with pytest.raises(ValueError, match="digest mismatch"):
        _rusticol._validate_recurrence_builder_input_v1(stale)


def test_native_recurrence_binding_rejects_wrong_primitive_dtype() -> None:
    value = _input()
    tables: list[object] = []
    for table in value.tables:
        columns: list[object] = []
        for column in table.columns:
            values = column.values
            if table.name == "header" and column.name == "schema_version":
                values = np.asarray(values, dtype="<u8").copy()
                values.flags.writeable = False
            columns.append(SimpleNamespace(name=column.name, values=values))
        tables.append(
            SimpleNamespace(
                name=table.name,
                row_count=table.row_count,
                columns=tuple(columns),
            )
        )
    malformed = SimpleNamespace(
        abi=value.abi,
        tables=tuple(tables),
        digest=value.canonical_digest,
    )

    with pytest.raises(ValueError, match=r"header\.schema_version.*expected .*<u4"):
        _rusticol._validate_recurrence_builder_input_v1(malformed)
