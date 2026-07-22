# SPDX-License-Identifier: 0BSD
"""Columnar Python-to-Rust contract for recurrence construction.

This module owns only the compact, model-generic input boundary described by
``pyamplicol-recurrence-builder-input-v1``.  Callers provide immutable logical
records produced after process expansion and LC colour planning.  The builder
normalises those records into deterministic, owned NumPy columns; it neither
imports nor constructs the process ``GenericDAG`` representation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

RECURRENCE_BUILDER_INPUT_ABI: Final = "pyamplicol-recurrence-builder-input-v1"
RECURRENCE_BUILDER_INPUT_SCHEMA_VERSION: Final = 1
MISSING_U32: Final = (1 << 32) - 1

RecurrenceLCFlowLayout = Literal["topology-replay", "all-flow-union"]
LCColorSectorKind = Literal["singlet", "open-lines", "single-trace"]

_LAYOUT_CODES: Final = {"topology-replay": 0, "all-flow-union": 1}
_SECTOR_KIND_CODES: Final = {"singlet": 0, "open-lines": 1, "single-trace": 2}
_REQUIRED_HEADER_DIGEST_ROLES: Final = frozenset(
    {"process", "model-catalog", "prepared-catalog", "color-plan"}
)

_U8 = np.dtype("u1")
_U32 = np.dtype("<u4")
_U64 = np.dtype("<u8")
_I32 = np.dtype("<i4")
_ALLOWED_DTYPES = frozenset(dtype.str for dtype in (_U8, _U32, _U64, _I32))


class RecurrenceColumnarInputError(ValueError):
    """A logical record or primitive recurrence column violates ABI v1."""


@dataclass(frozen=True, slots=True)
class ExactComplexRationalV1:
    """Canonical exact complex rational used by recurrence proofs."""

    real_numerator: int
    real_denominator: int = 1
    imag_numerator: int = 0
    imag_denominator: int = 1

    def __post_init__(self) -> None:
        _validate_fraction(
            self.real_numerator,
            self.real_denominator,
            "real recurrence factor",
        )
        _validate_fraction(
            self.imag_numerator,
            self.imag_denominator,
            "imaginary recurrence factor",
        )

    @property
    def canonical_key(self) -> tuple[int, int, int, int]:
        return (
            self.real_numerator,
            self.real_denominator,
            self.imag_numerator,
            self.imag_denominator,
        )


@dataclass(frozen=True, slots=True)
class RecurrenceSemanticDigestV1:
    """One named semantic SHA-256 bound by the builder header."""

    role: str
    digest: str

    def __post_init__(self) -> None:
        _nonempty_text(self.role, "semantic digest role")
        _sha256_bytes(self.digest, f"semantic digest {self.role!r}")


@dataclass(frozen=True, slots=True)
class RecurrenceSourceStateV1:
    """One public source state retained for an external leg."""

    state_index: int
    spin_state: int
    source_template_id: int

    def __post_init__(self) -> None:
        _checked_u32(self.state_index, "source-state index")
        _checked_i32(self.spin_state, "source spin state")
        _checked_u32(self.source_template_id, "source template ID")


@dataclass(frozen=True, slots=True)
class RecurrenceExternalLegV1:
    """External process leg and complete generated source-state coverage."""

    source_slot: int
    public_label: int
    physical_pdg: int
    outgoing_pdg: int
    particle_state_template_id: int
    source_states: tuple[RecurrenceSourceStateV1, ...]
    momentum_mask: int
    support_mask: int

    def __post_init__(self) -> None:
        _checked_u32(self.source_slot, "external source slot")
        _checked_u32(self.public_label, "external public label")
        _checked_i32(self.physical_pdg, "physical PDG")
        _checked_i32(self.outgoing_pdg, "outgoing PDG")
        _checked_u32(
            self.particle_state_template_id,
            "particle-state template ID",
        )
        _validate_nonnegative_mask(self.momentum_mask, "external momentum mask")
        _validate_nonnegative_mask(self.support_mask, "external support mask")
        if not self.source_states:
            raise RecurrenceColumnarInputError(
                "every external leg must retain at least one source state"
            )
        indices = tuple(state.state_index for state in self.source_states)
        if indices != tuple(range(len(indices))):
            raise RecurrenceColumnarInputError(
                "source-state indices must be dense and ordered from zero"
            )
        spins = tuple(state.spin_state for state in self.source_states)
        if len(set(spins)) != len(spins):
            raise RecurrenceColumnarInputError(
                "one external leg cannot expose a spin state more than once"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceLCOpenStringV1:
    """One ordered open fundamental colour string, using source slots."""

    fundamental_source_slot: int
    antifundamental_source_slot: int
    adjoint_source_slots: tuple[int, ...] = ()
    singlet_source_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _checked_u32(self.fundamental_source_slot, "fundamental source slot")
        _checked_u32(
            self.antifundamental_source_slot,
            "antifundamental source slot",
        )
        _validate_u32_sequence(self.adjoint_source_slots, "adjoint source slots")
        _validate_u32_sequence(self.singlet_source_slots, "singlet source slots")


@dataclass(frozen=True, slots=True)
class RecurrencePhysicalLCSectorV1:
    """One physical LC sector before recurrence state construction."""

    sector_id: int
    public_id: str
    kind: LCColorSectorKind
    open_strings: tuple[RecurrenceLCOpenStringV1, ...] = ()
    trace_source_slots: tuple[int, ...] = ()
    singlet_source_slots: tuple[int, ...] = ()
    word_source_slots: tuple[int, ...] = ()
    support_mask: int = 0

    def __post_init__(self) -> None:
        _checked_u32(self.sector_id, "physical LC sector ID")
        _nonempty_text(self.public_id, "physical LC sector public ID")
        if self.kind not in _SECTOR_KIND_CODES:
            raise RecurrenceColumnarInputError(
                f"unsupported physical LC sector kind {self.kind!r}"
            )
        _validate_u32_sequence(self.trace_source_slots, "trace source slots")
        _validate_u32_sequence(self.singlet_source_slots, "singlet source slots")
        _validate_u32_sequence(self.word_source_slots, "colour-word source slots")
        _validate_nonnegative_mask(self.support_mask, "LC sector support mask")
        if self.kind == "open-lines" and not self.open_strings:
            raise RecurrenceColumnarInputError(
                "an open-lines LC sector must contain at least one open string"
            )
        if self.kind != "open-lines" and self.open_strings:
            raise RecurrenceColumnarInputError(
                f"an LC sector of kind {self.kind!r} cannot contain open strings"
            )
        if self.kind == "single-trace" and not self.trace_source_slots:
            raise RecurrenceColumnarInputError(
                "a single-trace LC sector must contain a trace word"
            )
        if self.kind != "single-trace" and self.trace_source_slots:
            raise RecurrenceColumnarInputError(
                f"an LC sector of kind {self.kind!r} cannot contain a trace word"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceReplayTargetV1:
    """Exact mapping from a replay representative to one physical sector."""

    sector_id: int
    external_permutation: tuple[int, ...]
    source_state_bijection: tuple[int, ...]
    closure_mapping: tuple[int, ...]
    factor: ExactComplexRationalV1 = field(
        default_factory=lambda: ExactComplexRationalV1(1)
    )
    fermion_sign: int = 1

    def __post_init__(self) -> None:
        _checked_u32(self.sector_id, "replay target sector ID")
        _validate_u32_sequence(
            self.external_permutation,
            "replay external permutation",
        )
        _validate_u32_sequence(
            self.source_state_bijection,
            "replay source-state bijection",
        )
        _validate_u32_sequence(self.closure_mapping, "replay closure mapping")
        if self.fermion_sign not in {-1, 1}:
            raise RecurrenceColumnarInputError("replay fermion sign must be -1 or 1")


@dataclass(frozen=True, slots=True)
class RecurrenceReplayPartitionV1:
    """One exactly certified physical-flow replay partition."""

    representative_sector_id: int
    materialized_sector_id: int
    proof_algorithm: str
    proof_digest: str
    targets: tuple[RecurrenceReplayTargetV1, ...]

    def __post_init__(self) -> None:
        _checked_u32(self.representative_sector_id, "replay representative sector")
        _checked_u32(self.materialized_sector_id, "replay materialized sector")
        _nonempty_text(self.proof_algorithm, "replay proof algorithm")
        _sha256_bytes(self.proof_digest, "replay proof digest")
        if not self.targets:
            raise RecurrenceColumnarInputError(
                "a replay partition must contain at least one target"
            )
        target_ids = tuple(target.sector_id for target in self.targets)
        if len(set(target_ids)) != len(target_ids):
            raise RecurrenceColumnarInputError(
                "a replay partition contains duplicate target sectors"
            )
        if self.representative_sector_id not in target_ids:
            raise RecurrenceColumnarInputError(
                "a replay partition must contain its representative sector"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceSelectedSourceCoverageV1:
    """Generation-retained local source states for one external leg."""

    source_slot: int
    source_state_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _checked_u32(self.source_slot, "selected source slot")
        _validate_u32_sequence(
            self.source_state_indices,
            "selected source-state indices",
        )
        if not self.source_state_indices:
            raise RecurrenceColumnarInputError(
                "selected source coverage cannot be empty"
            )
        if len(set(self.source_state_indices)) != len(self.source_state_indices):
            raise RecurrenceColumnarInputError(
                "selected source coverage contains duplicate states"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceCouplingLimitV1:
    """Resolved inclusive bounds for one coupling-order axis."""

    name: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "coupling-order name")
        minimum = _checked_u32(self.minimum, f"coupling-order {self.name} minimum")
        maximum = _checked_u32(self.maximum, f"coupling-order {self.name} maximum")
        if minimum > maximum:
            raise RecurrenceColumnarInputError(
                f"coupling-order {self.name!r} minimum exceeds its maximum"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceSemanticTemplateReferenceV1:
    """Content-addressed recurrence template and optional callable binding."""

    kind: str
    template_id: int
    semantic_digest: str
    prepared_kernel_id: int | None = None

    def __post_init__(self) -> None:
        _nonempty_text(self.kind, "semantic-template kind")
        _checked_u32(self.template_id, "semantic-template ID")
        _sha256_bytes(self.semantic_digest, "semantic-template digest")
        if self.prepared_kernel_id is not None:
            if self.prepared_kernel_id == MISSING_U32:
                raise RecurrenceColumnarInputError(
                    "prepared-kernel ID reserves the u32 sentinel"
                )
            _checked_u32(self.prepared_kernel_id, "prepared-kernel ID")


@dataclass(frozen=True, slots=True)
class RecurrenceNormalizationV1:
    """Exact process normalization and its semantic convention."""

    factor: ExactComplexRationalV1
    convention: str
    semantic_digest: str

    def __post_init__(self) -> None:
        _nonempty_text(self.convention, "normalization convention")
        _sha256_bytes(self.semantic_digest, "normalization semantic digest")


@dataclass(frozen=True, slots=True)
class RecurrenceParameterProjectionV1:
    """Map one public runtime parameter slot to a prepared-model slot."""

    runtime_slot: int
    runtime_name: str
    parameter_template_id: int
    prepared_parameter_id: int
    component: int = 0

    def __post_init__(self) -> None:
        _checked_u32(self.runtime_slot, "runtime parameter slot")
        _nonempty_text(self.runtime_name, "runtime parameter name")
        _checked_u32(self.parameter_template_id, "parameter-template ID")
        _checked_u32(self.prepared_parameter_id, "prepared parameter ID")
        _checked_u32(self.component, "parameter component")


@dataclass(frozen=True, slots=True)
class RecurrenceBuilderLogicalInputV1:
    """Compact logical records accepted by the recurrence column encoder."""

    process_id: str
    layout: RecurrenceLCFlowLayout
    semantic_digests: tuple[RecurrenceSemanticDigestV1, ...]
    external_legs: tuple[RecurrenceExternalLegV1, ...]
    physical_sectors: tuple[RecurrencePhysicalLCSectorV1, ...]
    semantic_template_references: tuple[RecurrenceSemanticTemplateReferenceV1, ...]
    normalization: RecurrenceNormalizationV1
    replay_partitions: tuple[RecurrenceReplayPartitionV1, ...] = ()
    selected_sector_ids: tuple[int, ...] | None = None
    selected_source_coverage: tuple[RecurrenceSelectedSourceCoverageV1, ...] | None = (
        None
    )
    coupling_limits: tuple[RecurrenceCouplingLimitV1, ...] = ()
    parameter_projection: tuple[RecurrenceParameterProjectionV1, ...] = ()
    process_support_mask: int = 0

    def __post_init__(self) -> None:
        _nonempty_text(self.process_id, "recurrence process ID")
        if self.layout not in _LAYOUT_CODES:
            raise RecurrenceColumnarInputError(
                f"unsupported recurrence LC flow layout {self.layout!r}"
            )
        _validate_nonnegative_mask(self.process_support_mask, "process support mask")


@dataclass(frozen=True, slots=True)
class RecurrenceColumn:
    """One owned, immutable primitive recurrence column."""

    name: str
    values: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "column name")
        values = self.values
        if not isinstance(values, np.ndarray):
            raise TypeError(f"column {self.name!r} must be a NumPy array")
        if values.ndim < 1:
            raise RecurrenceColumnarInputError(
                f"column {self.name!r} must have a row dimension"
            )
        if not values.flags.c_contiguous:
            raise RecurrenceColumnarInputError(
                f"column {self.name!r} must be C-contiguous"
            )
        if not values.flags.owndata:
            raise RecurrenceColumnarInputError(
                f"column {self.name!r} must own its storage"
            )
        if values.flags.writeable:
            raise RecurrenceColumnarInputError(
                f"column {self.name!r} must be read-only"
            )
        if values.dtype.str not in _ALLOWED_DTYPES:
            raise RecurrenceColumnarInputError(
                f"column {self.name!r} has unsupported or non-little-endian "
                f"dtype {values.dtype.str!r}"
            )


@dataclass(frozen=True, slots=True)
class RecurrenceColumnarTable:
    """One deterministic structure-of-arrays recurrence table."""

    name: str
    row_count: int
    columns: tuple[RecurrenceColumn, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "table name")
        _checked_u64(self.row_count, f"{self.name}.row_count")
        names = tuple(column.name for column in self.columns)
        if len(set(names)) != len(names):
            raise RecurrenceColumnarInputError(
                f"table {self.name!r} contains duplicate column names"
            )
        for column in self.columns:
            if len(column.values) != self.row_count:
                raise RecurrenceColumnarInputError(
                    f"table {self.name!r} column {column.name!r} has "
                    f"{len(column.values)} rows, expected {self.row_count}"
                )

    def column(self, name: str) -> np.ndarray[Any, Any]:
        for column in self.columns:
            if column.name == name:
                return column.values
        raise KeyError(f"table {self.name!r} has no column {name!r}")


_ColumnSpec = tuple[str, tuple[int, ...]]

_TABLE_SCHEMAS: Final[dict[str, tuple[_ColumnSpec, ...]]] = {
    "bitset_ranges": (
        ("id", ()),
        ("start", ()),
        ("count", ()),
        ("bit_count", ()),
    ),
    "bitset_words": (("value", ()),),
    "coupling_limits": (
        ("name_string_id", ()),
        ("minimum", ()),
        ("maximum", ()),
    ),
    "digest_catalog": (("id", ()), ("value", (32,))),
    "exact_factors": (
        ("id", ()),
        ("real_numerator_string_id", ()),
        ("real_denominator_string_id", ()),
        ("imag_numerator_string_id", ()),
        ("imag_denominator_string_id", ()),
    ),
    "external_legs": (
        ("source_slot", ()),
        ("public_label", ()),
        ("physical_pdg", ()),
        ("outgoing_pdg", ()),
        ("particle_state_template_id", ()),
        ("source_state_start", ()),
        ("source_state_count", ()),
        ("momentum_mask_id", ()),
        ("support_mask_id", ()),
    ),
    "header": (
        ("schema_version", ()),
        ("abi_string_id", ()),
        ("process_id_string_id", ()),
        ("layout", ()),
        ("selected_sector_mode", ()),
        ("selected_source_mode", ()),
        ("external_leg_count", ()),
        ("physical_sector_count", ()),
        ("replay_partition_count", ()),
        ("coupling_limit_count", ()),
        ("parameter_projection_count", ()),
        ("process_support_mask_id", ()),
    ),
    "header_digests": (("role_string_id", ()), ("digest_id", ())),
    "lc_open_strings": (
        ("sector_id", ()),
        ("ordinal", ()),
        ("fundamental_source_slot", ()),
        ("antifundamental_source_slot", ()),
        ("adjoint_sequence_id", ()),
        ("singlet_sequence_id", ()),
    ),
    "normalization": (
        ("factor_id", ()),
        ("convention_string_id", ()),
        ("semantic_digest_id", ()),
    ),
    "parameter_projection": (
        ("runtime_slot", ()),
        ("runtime_name_string_id", ()),
        ("parameter_template_id", ()),
        ("prepared_parameter_id", ()),
        ("component", ()),
    ),
    "physical_lc_sectors": (
        ("sector_id", ()),
        ("public_id_string_id", ()),
        ("kind", ()),
        ("open_string_start", ()),
        ("open_string_count", ()),
        ("trace_sequence_id", ()),
        ("singlet_sequence_id", ()),
        ("word_sequence_id", ()),
        ("support_mask_id", ()),
    ),
    "replay_partitions": (
        ("partition_id", ()),
        ("representative_sector_id", ()),
        ("materialized_sector_id", ()),
        ("target_start", ()),
        ("target_count", ()),
        ("proof_algorithm_string_id", ()),
        ("proof_digest_id", ()),
    ),
    "replay_targets": (
        ("partition_id", ()),
        ("sector_id", ()),
        ("external_permutation_sequence_id", ()),
        ("source_state_bijection_sequence_id", ()),
        ("closure_mapping_sequence_id", ()),
        ("factor_id", ()),
        ("fermion_sign", ()),
    ),
    "selected_sector_coverage": (("sector_id", ()),),
    "selected_source_coverage": (
        ("source_slot", ()),
        ("source_state_index", ()),
    ),
    "semantic_template_references": (
        ("kind_string_id", ()),
        ("template_id", ()),
        ("semantic_digest_id", ()),
        ("prepared_kernel_id", ()),
    ),
    "source_states": (
        ("source_slot", ()),
        ("state_index", ()),
        ("spin_state", ()),
        ("source_template_id", ()),
    ),
    "string_bytes": (("value", ()),),
    "string_ranges": (("start", ()), ("count", ())),
    "u32_sequence_ranges": (("start", ()), ("count", ())),
    "u32_sequence_values": (("value", ()),),
}

_U8_COLUMN_KEYS: Final = frozenset(
    {
        ("digest_catalog", "value"),
        ("header", "layout"),
        ("header", "selected_sector_mode"),
        ("header", "selected_source_mode"),
        ("physical_lc_sectors", "kind"),
        ("string_bytes", "value"),
    }
)
_U64_COLUMN_KEYS: Final = frozenset(
    {
        ("bitset_ranges", "start"),
        ("bitset_ranges", "count"),
        ("bitset_ranges", "bit_count"),
        ("bitset_words", "value"),
        ("external_legs", "source_state_start"),
        ("external_legs", "source_state_count"),
        ("physical_lc_sectors", "open_string_start"),
        ("physical_lc_sectors", "open_string_count"),
        ("replay_partitions", "target_start"),
        ("replay_partitions", "target_count"),
        ("string_ranges", "start"),
        ("string_ranges", "count"),
        ("u32_sequence_ranges", "start"),
        ("u32_sequence_ranges", "count"),
    }
)
_I32_COLUMN_KEYS: Final = frozenset(
    {
        ("external_legs", "physical_pdg"),
        ("external_legs", "outgoing_pdg"),
        ("replay_targets", "fermion_sign"),
        ("source_states", "spin_state"),
    }
)


@dataclass(frozen=True, slots=True)
class RecurrenceBuilderInputV1:
    """Validated immutable payload passed to the future Rust builder."""

    abi: str
    tables: tuple[RecurrenceColumnarTable, ...]

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_BUILDER_INPUT_ABI:
            raise RecurrenceColumnarInputError(
                f"unsupported recurrence builder input ABI {self.abi!r}"
            )
        names = tuple(table.name for table in self.tables)
        expected_names = tuple(sorted(_TABLE_SCHEMAS))
        if names != expected_names:
            raise RecurrenceColumnarInputError(
                "recurrence columnar tables must match the sorted v1 inventory; "
                f"got {names!r}"
            )
        self._validate_schemas()
        self._validate_catalogs()
        self._validate_references()

    def table(self, name: str) -> RecurrenceColumnarTable:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"recurrence builder input has no table {name!r}")

    @property
    def canonical_digest(self) -> str:
        """SHA-256 over ABI, table schemas, shapes, and canonical bytes."""

        digest = hashlib.sha256()
        _hash_text(digest, self.abi)
        digest.update(len(self.tables).to_bytes(8, "little"))
        for table in self.tables:
            _hash_text(digest, table.name)
            digest.update(table.row_count.to_bytes(8, "little"))
            digest.update(len(table.columns).to_bytes(4, "little"))
            for column in table.columns:
                _hash_text(digest, column.name)
                _hash_text(digest, column.values.dtype.str)
                digest.update(len(column.values.shape).to_bytes(1, "little"))
                for dimension in column.values.shape:
                    digest.update(int(dimension).to_bytes(8, "little"))
                digest.update(column.values.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def digest(self) -> str:
        """Compatibility spelling for consumers passing the digest to Rust."""

        return self.canonical_digest

    def require_digest(self, expected: str) -> None:
        """Fail before GIL release when a supplied input digest is stale."""

        _sha256_bytes(expected, "expected recurrence input digest")
        if self.canonical_digest != expected:
            raise RecurrenceColumnarInputError(
                "recurrence builder input canonical digest does not match"
            )

    def _validate_schemas(self) -> None:
        for table in self.tables:
            expected = _TABLE_SCHEMAS[table.name]
            names = tuple(column.name for column in table.columns)
            expected_names = tuple(name for name, _shape in expected)
            if names != expected_names:
                raise RecurrenceColumnarInputError(
                    f"table {table.name!r} columns do not match ABI v1: {names!r}"
                )
            for column, (_name, tail_shape) in zip(
                table.columns,
                expected,
                strict=True,
            ):
                expected_dtype = _column_dtype(table.name, column.name)
                if column.values.dtype.str != expected_dtype.str:
                    raise RecurrenceColumnarInputError(
                        f"table {table.name!r} column {column.name!r} has dtype "
                        f"{column.values.dtype.str!r}, expected "
                        f"{expected_dtype.str!r}"
                    )
                if column.values.shape[1:] != tail_shape:
                    raise RecurrenceColumnarInputError(
                        f"table {table.name!r} column {column.name!r} has tail "
                        f"shape {column.values.shape[1:]!r}, expected {tail_shape!r}"
                    )

    def _validate_catalogs(self) -> None:
        strings = _decode_flat_strings(
            self.table("string_ranges"),
            self.table("string_bytes"),
        )
        if strings != tuple(sorted(strings, key=lambda value: value.encode("utf-8"))):
            raise RecurrenceColumnarInputError(
                "flat string catalog is not in canonical byte order"
            )
        if len(set(strings)) != len(strings):
            raise RecurrenceColumnarInputError(
                "flat string catalog contains duplicates"
            )

        _validate_flat_ranges(
            self.table("u32_sequence_ranges"),
            self.table("u32_sequence_values").row_count,
            "u32 sequence catalog",
        )
        _validate_flat_ranges(
            self.table("bitset_ranges"),
            self.table("bitset_words").row_count,
            "bitset catalog",
        )
        _validate_dense_ids(self.table("bitset_ranges"), "id")
        _validate_dense_ids(self.table("digest_catalog"), "id")
        _validate_dense_ids(self.table("exact_factors"), "id")

        words = self.table("bitset_words").column("value")
        ranges = self.table("bitset_ranges")
        for row in range(ranges.row_count):
            start = int(ranges.column("start")[row])
            count = int(ranges.column("count")[row])
            if count and int(words[start + count - 1]) == 0:
                raise RecurrenceColumnarInputError(
                    "bitset catalog contains a noncanonical leading zero word"
                )
            value = sum(
                int(words[start + offset]) << (64 * offset) for offset in range(count)
            )
            if value.bit_count() != int(ranges.column("bit_count")[row]):
                raise RecurrenceColumnarInputError(
                    "bitset catalog bit count does not match its words"
                )

    def _validate_references(self) -> None:
        string_count = self.table("string_ranges").row_count
        digest_count = self.table("digest_catalog").row_count
        sequence_count = self.table("u32_sequence_ranges").row_count
        bitset_count = self.table("bitset_ranges").row_count
        factor_count = self.table("exact_factors").row_count
        external_count = self.table("external_legs").row_count
        sector_count = self.table("physical_lc_sectors").row_count
        partition_count = self.table("replay_partitions").row_count

        _validate_dense_ids(self.table("external_legs"), "source_slot")
        _validate_dense_ids(self.table("physical_lc_sectors"), "sector_id")
        _validate_dense_ids(self.table("replay_partitions"), "partition_id")
        _validate_dense_ids(self.table("parameter_projection"), "runtime_slot")

        for table, columns in (
            (self.table("header"), ("abi_string_id", "process_id_string_id")),
            (self.table("header_digests"), ("role_string_id",)),
            (self.table("coupling_limits"), ("name_string_id",)),
            (
                self.table("exact_factors"),
                (
                    "real_numerator_string_id",
                    "real_denominator_string_id",
                    "imag_numerator_string_id",
                    "imag_denominator_string_id",
                ),
            ),
            (self.table("physical_lc_sectors"), ("public_id_string_id",)),
            (self.table("replay_partitions"), ("proof_algorithm_string_id",)),
            (self.table("semantic_template_references"), ("kind_string_id",)),
            (self.table("normalization"), ("convention_string_id",)),
            (self.table("parameter_projection"), ("runtime_name_string_id",)),
        ):
            _validate_bounded_columns(table, columns, string_count)

        for table, columns in (
            (self.table("header_digests"), ("digest_id",)),
            (self.table("replay_partitions"), ("proof_digest_id",)),
            (self.table("semantic_template_references"), ("semantic_digest_id",)),
            (self.table("normalization"), ("semantic_digest_id",)),
        ):
            _validate_bounded_columns(table, columns, digest_count)

        for table, columns in (
            (
                self.table("lc_open_strings"),
                (
                    "adjoint_sequence_id",
                    "singlet_sequence_id",
                ),
            ),
            (
                self.table("physical_lc_sectors"),
                (
                    "trace_sequence_id",
                    "singlet_sequence_id",
                    "word_sequence_id",
                ),
            ),
            (
                self.table("replay_targets"),
                (
                    "external_permutation_sequence_id",
                    "source_state_bijection_sequence_id",
                    "closure_mapping_sequence_id",
                ),
            ),
        ):
            _validate_bounded_columns(table, columns, sequence_count)

        for table, columns in (
            (self.table("header"), ("process_support_mask_id",)),
            (self.table("external_legs"), ("momentum_mask_id", "support_mask_id")),
            (self.table("physical_lc_sectors"), ("support_mask_id",)),
        ):
            _validate_bounded_columns(table, columns, bitset_count)

        _validate_bounded_columns(
            self.table("replay_targets"),
            ("factor_id",),
            factor_count,
        )
        _validate_bounded_columns(
            self.table("normalization"),
            ("factor_id",),
            factor_count,
        )

        for table, columns in (
            (
                self.table("lc_open_strings"),
                (
                    "fundamental_source_slot",
                    "antifundamental_source_slot",
                ),
            ),
            (self.table("source_states"), ("source_slot",)),
            (self.table("selected_source_coverage"), ("source_slot",)),
        ):
            _validate_bounded_columns(table, columns, external_count)

        for table, columns in (
            (self.table("lc_open_strings"), ("sector_id",)),
            (
                self.table("replay_partitions"),
                (
                    "representative_sector_id",
                    "materialized_sector_id",
                ),
            ),
            (self.table("replay_targets"), ("sector_id",)),
            (self.table("selected_sector_coverage"), ("sector_id",)),
        ):
            _validate_bounded_columns(table, columns, sector_count)
        _validate_bounded_columns(
            self.table("replay_targets"),
            ("partition_id",),
            partition_count,
        )

        _validate_parent_ranges(
            self.table("external_legs"),
            "source_state_start",
            "source_state_count",
            self.table("source_states").row_count,
            "external source-state",
        )
        _validate_parent_ranges(
            self.table("physical_lc_sectors"),
            "open_string_start",
            "open_string_count",
            self.table("lc_open_strings").row_count,
            "LC open-string",
        )
        _validate_parent_ranges(
            self.table("replay_partitions"),
            "target_start",
            "target_count",
            self.table("replay_targets").row_count,
            "replay target",
        )

        header = self.table("header")
        if header.row_count != 1:
            raise RecurrenceColumnarInputError("recurrence header must have one row")
        if (
            int(header.column("schema_version")[0])
            != RECURRENCE_BUILDER_INPUT_SCHEMA_VERSION
        ):
            raise RecurrenceColumnarInputError(
                "recurrence header has the wrong schema version"
            )
        expected_counts = {
            "external_leg_count": external_count,
            "physical_sector_count": sector_count,
            "replay_partition_count": partition_count,
            "coupling_limit_count": self.table("coupling_limits").row_count,
            "parameter_projection_count": self.table("parameter_projection").row_count,
        }
        for column, expected in expected_counts.items():
            if int(header.column(column)[0]) != expected:
                raise RecurrenceColumnarInputError(
                    f"recurrence header {column} does not match its table"
                )

        strings = _decode_flat_strings(
            self.table("string_ranges"),
            self.table("string_bytes"),
        )
        abi_id = int(header.column("abi_string_id")[0])
        if strings[abi_id] != self.abi:
            raise RecurrenceColumnarInputError(
                "recurrence header ABI string does not match the payload ABI"
            )
        roles = tuple(
            strings[int(value)]
            for value in self.table("header_digests").column("role_string_id")
        )
        if len(set(roles)) != len(roles):
            raise RecurrenceColumnarInputError(
                "recurrence header contains duplicate semantic digest roles"
            )
        if not set(roles) >= _REQUIRED_HEADER_DIGEST_ROLES:
            missing = sorted(_REQUIRED_HEADER_DIGEST_ROLES - set(roles))
            raise RecurrenceColumnarInputError(
                f"recurrence header is missing semantic digests: {missing!r}"
            )

        selected_sectors = self.table("selected_sector_coverage").row_count
        selected_sources = self.table("selected_source_coverage").row_count
        if bool(int(header.column("selected_sector_mode")[0])) != bool(
            selected_sectors
        ):
            raise RecurrenceColumnarInputError(
                "selected-sector header mode disagrees with its coverage table"
            )
        if bool(int(header.column("selected_source_mode")[0])) != bool(
            selected_sources
        ):
            raise RecurrenceColumnarInputError(
                "selected-source header mode disagrees with its coverage table"
            )


class _StringCatalog:
    def __init__(self, values: Iterable[str]) -> None:
        unique = {_nonempty_text(value, "string catalog value") for value in values}
        self.values = tuple(sorted(unique, key=lambda value: value.encode("utf-8")))
        self.ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: str) -> int:
        try:
            return self.ids[value]
        except KeyError as error:
            raise RecurrenceColumnarInputError(
                f"string {value!r} was not collected before table construction"
            ) from error

    def tables(self) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
        encoded = tuple(value.encode("utf-8") for value in self.values)
        ranges = _allocate(len(encoded), start=_U64, count=_U64)
        flat = _allocate(sum(len(value) for value in encoded), value=_U8)
        cursor = 0
        for row, value in enumerate(encoded):
            ranges["start"][row] = cursor
            ranges["count"][row] = len(value)
            if value:
                flat["value"][cursor : cursor + len(value)] = np.frombuffer(
                    value,
                    dtype=_U8,
                )
            cursor += len(value)
        return (
            _freeze_table("string_ranges", ranges),
            _freeze_table("string_bytes", flat),
        )


class _DigestCatalog:
    def __init__(self, values: Iterable[str]) -> None:
        raw = {_sha256_bytes(value, "semantic digest") for value in values}
        self.values = tuple(sorted(raw))
        self.ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: str) -> int:
        return self.ids[_sha256_bytes(value, "semantic digest")]

    def table(self) -> RecurrenceColumnarTable:
        columns = _allocate(len(self.values), id=_U32, value=(_U8, (32,)))
        for row, value in enumerate(self.values):
            columns["id"][row] = row
            columns["value"][row] = np.frombuffer(value, dtype=_U8)
        return _freeze_table("digest_catalog", columns)


class _U32SequenceCatalog:
    def __init__(self, values: Iterable[Sequence[int]]) -> None:
        canonical: set[tuple[int, ...]] = set()
        for value in values:
            item = tuple(value)
            _validate_u32_sequence(item, "u32 sequence catalog value")
            canonical.add(item)
        self.values = tuple(sorted(canonical))
        self.ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: Sequence[int]) -> int:
        item = tuple(value)
        try:
            return self.ids[item]
        except KeyError as error:
            raise RecurrenceColumnarInputError(
                f"u32 sequence {item!r} was not collected"
            ) from error

    def tables(self) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
        ranges = _allocate(len(self.values), start=_U64, count=_U64)
        flat = _allocate(sum(len(value) for value in self.values), value=_U32)
        cursor = 0
        for row, value in enumerate(self.values):
            ranges["start"][row] = cursor
            ranges["count"][row] = len(value)
            if value:
                flat["value"][cursor : cursor + len(value)] = value
            cursor += len(value)
        return (
            _freeze_table("u32_sequence_ranges", ranges),
            _freeze_table("u32_sequence_values", flat),
        )


class _BitsetCatalog:
    def __init__(self, values: Iterable[int]) -> None:
        canonical: set[int] = set()
        for value in values:
            _validate_nonnegative_mask(value, "bitset catalog value")
            canonical.add(value)
        self.values = tuple(sorted(canonical))
        self.ids = {value: index for index, value in enumerate(self.values)}

    def id(self, value: int) -> int:
        return self.ids[value]

    def tables(self) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
        ranges = _allocate(
            len(self.values),
            id=_U32,
            start=_U64,
            count=_U64,
            bit_count=_U64,
        )
        total_words = sum((value.bit_length() + 63) // 64 for value in self.values)
        words = _allocate(total_words, value=_U64)
        cursor = 0
        for row, value in enumerate(self.values):
            count = (value.bit_length() + 63) // 64
            ranges["id"][row] = row
            ranges["start"][row] = cursor
            ranges["count"][row] = count
            ranges["bit_count"][row] = value.bit_count()
            for offset in range(count):
                words["value"][cursor] = (value >> (64 * offset)) & ((1 << 64) - 1)
                cursor += 1
        return (
            _freeze_table("bitset_ranges", ranges),
            _freeze_table("bitset_words", words),
        )


class _FactorCatalog:
    def __init__(self, values: Iterable[ExactComplexRationalV1]) -> None:
        unique = {value.canonical_key: value for value in values}
        self.values = tuple(unique[key] for key in sorted(unique))
        self.ids = {
            value.canonical_key: index for index, value in enumerate(self.values)
        }

    def id(self, value: ExactComplexRationalV1) -> int:
        return self.ids[value.canonical_key]

    def table(self, strings: _StringCatalog) -> RecurrenceColumnarTable:
        columns = _allocate(
            len(self.values),
            id=_U32,
            real_numerator_string_id=_U32,
            real_denominator_string_id=_U32,
            imag_numerator_string_id=_U32,
            imag_denominator_string_id=_U32,
        )
        for row, value in enumerate(self.values):
            columns["id"][row] = row
            columns["real_numerator_string_id"][row] = strings.id(
                str(value.real_numerator)
            )
            columns["real_denominator_string_id"][row] = strings.id(
                str(value.real_denominator)
            )
            columns["imag_numerator_string_id"][row] = strings.id(
                str(value.imag_numerator)
            )
            columns["imag_denominator_string_id"][row] = strings.id(
                str(value.imag_denominator)
            )
        return _freeze_table("exact_factors", columns)


def build_recurrence_builder_input_v1(
    logical: RecurrenceBuilderLogicalInputV1,
) -> RecurrenceBuilderInputV1:
    """Validate and flatten compact logical recurrence records.

    The caller retains no mutable storage reachable from the returned payload.
    Record collections whose order is not semantic are canonicalised before
    catalog and table construction.
    """

    if not isinstance(logical, RecurrenceBuilderLogicalInputV1):
        raise TypeError("recurrence columnar extraction requires logical v1 records")

    external_legs = tuple(
        sorted(logical.external_legs, key=lambda leg: leg.source_slot)
    )
    sectors = tuple(
        sorted(logical.physical_sectors, key=lambda sector: sector.sector_id)
    )
    partitions = tuple(
        sorted(
            logical.replay_partitions,
            key=lambda partition: (
                partition.representative_sector_id,
                partition.materialized_sector_id,
            ),
        )
    )
    digests = tuple(sorted(logical.semantic_digests, key=lambda item: item.role))
    template_refs = tuple(
        sorted(
            logical.semantic_template_references,
            key=lambda item: (item.kind, item.template_id, item.semantic_digest),
        )
    )
    coupling_limits = tuple(sorted(logical.coupling_limits, key=lambda item: item.name))
    parameter_projection = tuple(
        sorted(logical.parameter_projection, key=lambda item: item.runtime_slot)
    )
    selected_sectors = (
        None
        if logical.selected_sector_ids is None
        else tuple(sorted(logical.selected_sector_ids))
    )
    selected_sources = (
        None
        if logical.selected_source_coverage is None
        else tuple(
            sorted(
                logical.selected_source_coverage,
                key=lambda item: item.source_slot,
            )
        )
    )

    _validate_logical_relations(
        logical,
        external_legs=external_legs,
        sectors=sectors,
        partitions=partitions,
        digests=digests,
        template_refs=template_refs,
        coupling_limits=coupling_limits,
        parameter_projection=parameter_projection,
        selected_sectors=selected_sectors,
        selected_sources=selected_sources,
    )

    all_factors = [logical.normalization.factor]
    all_factors.extend(
        target.factor for partition in partitions for target in partition.targets
    )
    factor_catalog = _FactorCatalog(all_factors)

    string_values: list[str] = [
        RECURRENCE_BUILDER_INPUT_ABI,
        logical.process_id,
        logical.normalization.convention,
    ]
    string_values.extend(item.role for item in digests)
    string_values.extend(item.public_id for item in sectors)
    string_values.extend(partition.proof_algorithm for partition in partitions)
    string_values.extend(limit.name for limit in coupling_limits)
    string_values.extend(item.kind for item in template_refs)
    string_values.extend(item.runtime_name for item in parameter_projection)
    for factor in factor_catalog.values:
        string_values.extend(str(value) for value in factor.canonical_key)
    strings = _StringCatalog(string_values)

    digest_values = [item.digest for item in digests]
    digest_values.extend(partition.proof_digest for partition in partitions)
    digest_values.extend(item.semantic_digest for item in template_refs)
    digest_values.append(logical.normalization.semantic_digest)
    digest_catalog = _DigestCatalog(digest_values)

    sequence_values: list[Sequence[int]] = [()]
    for sector in sectors:
        sequence_values.extend(
            (
                sector.trace_source_slots,
                sector.singlet_source_slots,
                sector.word_source_slots,
            )
        )
        for line in sector.open_strings:
            sequence_values.extend(
                (line.adjoint_source_slots, line.singlet_source_slots)
            )
    for partition in partitions:
        for target in partition.targets:
            sequence_values.extend(
                (
                    target.external_permutation,
                    target.source_state_bijection,
                    target.closure_mapping,
                )
            )
    sequences = _U32SequenceCatalog(sequence_values)

    bitsets = _BitsetCatalog(
        (
            logical.process_support_mask,
            *(leg.momentum_mask for leg in external_legs),
            *(leg.support_mask for leg in external_legs),
            *(sector.support_mask for sector in sectors),
        )
    )

    tables: list[RecurrenceColumnarTable] = []
    tables.extend(bitsets.tables())
    tables.append(_build_coupling_limits(coupling_limits, strings))
    tables.append(digest_catalog.table())
    tables.append(factor_catalog.table(strings))
    external_table, source_state_table = _build_external_tables(
        external_legs,
        bitsets,
    )
    tables.extend((external_table, source_state_table))
    tables.append(
        _build_header(
            logical,
            strings,
            bitsets,
            external_count=len(external_legs),
            sector_count=len(sectors),
            partition_count=len(partitions),
            coupling_limit_count=len(coupling_limits),
            parameter_count=len(parameter_projection),
            selected_sector_mode=selected_sectors is not None,
            selected_source_mode=selected_sources is not None,
        )
    )
    tables.append(_build_header_digests(digests, strings, digest_catalog))
    sector_table, open_string_table = _build_sector_tables(
        sectors,
        strings,
        sequences,
        bitsets,
    )
    tables.extend((sector_table, open_string_table))
    tables.append(
        _build_normalization(
            logical.normalization,
            strings,
            digest_catalog,
            factor_catalog,
        )
    )
    tables.append(_build_parameter_projection(parameter_projection, strings))
    partition_table, target_table = _build_replay_tables(
        partitions,
        strings,
        digest_catalog,
        sequences,
        factor_catalog,
    )
    tables.extend((partition_table, target_table))
    tables.append(_build_selected_sectors(selected_sectors))
    tables.append(_build_selected_sources(selected_sources))
    tables.append(_build_template_references(template_refs, strings, digest_catalog))
    string_ranges, string_bytes = strings.tables()
    tables.extend((string_ranges, string_bytes))
    sequence_ranges, sequence_values_table = sequences.tables()
    tables.extend((sequence_ranges, sequence_values_table))

    return RecurrenceBuilderInputV1(
        abi=RECURRENCE_BUILDER_INPUT_ABI,
        tables=tuple(sorted(tables, key=lambda table: table.name)),
    )


def _validate_logical_relations(
    logical: RecurrenceBuilderLogicalInputV1,
    *,
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    sectors: tuple[RecurrencePhysicalLCSectorV1, ...],
    partitions: tuple[RecurrenceReplayPartitionV1, ...],
    digests: tuple[RecurrenceSemanticDigestV1, ...],
    template_refs: tuple[RecurrenceSemanticTemplateReferenceV1, ...],
    coupling_limits: tuple[RecurrenceCouplingLimitV1, ...],
    parameter_projection: tuple[RecurrenceParameterProjectionV1, ...],
    selected_sectors: tuple[int, ...] | None,
    selected_sources: tuple[RecurrenceSelectedSourceCoverageV1, ...] | None,
) -> None:
    source_slots = tuple(leg.source_slot for leg in external_legs)
    if source_slots != tuple(range(len(source_slots))):
        raise RecurrenceColumnarInputError(
            "external source slots must be dense from zero"
        )
    public_labels = tuple(leg.public_label for leg in external_legs)
    if len(set(public_labels)) != len(public_labels):
        raise RecurrenceColumnarInputError("external public labels must be unique")
    sector_ids = tuple(sector.sector_id for sector in sectors)
    if sector_ids != tuple(range(len(sector_ids))):
        raise RecurrenceColumnarInputError(
            "physical LC sector IDs must be dense from zero"
        )
    public_sector_ids = tuple(sector.public_id for sector in sectors)
    if len(set(public_sector_ids)) != len(public_sector_ids):
        raise RecurrenceColumnarInputError(
            "physical LC sector public IDs must be unique"
        )
    if not sectors:
        raise RecurrenceColumnarInputError(
            "recurrence v1 requires at least one physical LC sector"
        )

    roles = tuple(item.role for item in digests)
    if len(set(roles)) != len(roles):
        raise RecurrenceColumnarInputError(
            "semantic header digest roles must be unique"
        )
    if not set(roles) >= _REQUIRED_HEADER_DIGEST_ROLES:
        missing = sorted(_REQUIRED_HEADER_DIGEST_ROLES - set(roles))
        raise RecurrenceColumnarInputError(
            f"recurrence logical input is missing semantic digests: {missing!r}"
        )

    names = tuple(limit.name for limit in coupling_limits)
    if len(set(names)) != len(names):
        raise RecurrenceColumnarInputError("coupling-order names must be unique")
    slots = tuple(item.runtime_slot for item in parameter_projection)
    if slots != tuple(range(len(slots))):
        raise RecurrenceColumnarInputError(
            "runtime parameter slots must be dense from zero"
        )
    runtime_names = tuple(item.runtime_name for item in parameter_projection)
    if len(set(runtime_names)) != len(runtime_names):
        raise RecurrenceColumnarInputError("runtime parameter names must be unique")
    template_keys = tuple((item.kind, item.template_id) for item in template_refs)
    if len(set(template_keys)) != len(template_keys):
        raise RecurrenceColumnarInputError(
            "semantic-template kind/ID pairs must be unique"
        )
    available_templates = set(template_keys)
    for leg in external_legs:
        if ("current-state", leg.particle_state_template_id) not in available_templates:
            raise RecurrenceColumnarInputError(
                "external leg references an absent current-state template"
            )
        for state in leg.source_states:
            if ("source", state.source_template_id) not in available_templates:
                raise RecurrenceColumnarInputError(
                    "external source coverage references an absent source template"
                )
    for projection in parameter_projection:
        if ("parameter", projection.parameter_template_id) not in available_templates:
            raise RecurrenceColumnarInputError(
                "parameter projection references an absent parameter template"
            )

    for sector in sectors:
        referenced_slots: list[int] = [
            *sector.trace_source_slots,
            *sector.singlet_source_slots,
            *sector.word_source_slots,
        ]
        for line in sector.open_strings:
            referenced_slots.extend(
                (
                    line.fundamental_source_slot,
                    line.antifundamental_source_slot,
                    *line.adjoint_source_slots,
                    *line.singlet_source_slots,
                )
            )
        for source_slot in referenced_slots:
            _require_index(source_slot, len(external_legs), "LC source slot")

    covered_targets: set[int] = set()
    for partition in partitions:
        _require_index(
            partition.representative_sector_id,
            len(sectors),
            "replay representative sector",
        )
        _require_index(
            partition.materialized_sector_id,
            len(sectors),
            "replay materialized sector",
        )
        for target in sorted(partition.targets, key=lambda item: item.sector_id):
            _require_index(target.sector_id, len(sectors), "replay target sector")
            if target.sector_id in covered_targets:
                raise RecurrenceColumnarInputError(
                    "replay partitions overlap one physical sector"
                )
            covered_targets.add(target.sector_id)
            _validate_permutation(
                target.external_permutation,
                len(external_legs),
                "replay external permutation",
            )
            if len(target.source_state_bijection) != len(external_legs):
                raise RecurrenceColumnarInputError(
                    "replay source-state bijection must have one entry per external leg"
                )
            _validate_permutation(
                target.source_state_bijection,
                len(external_legs),
                "replay source-state bijection",
            )

    if selected_sectors is not None:
        if not selected_sectors:
            raise RecurrenceColumnarInputError(
                "selected generation sector coverage cannot be empty"
            )
        if len(set(selected_sectors)) != len(selected_sectors):
            raise RecurrenceColumnarInputError(
                "selected generation sector coverage contains duplicates"
            )
        for sector_id in selected_sectors:
            _require_index(sector_id, len(sectors), "selected physical sector")

    if selected_sources is not None:
        if not selected_sources:
            raise RecurrenceColumnarInputError(
                "selected source generation coverage cannot be empty"
            )
        selected_slots = tuple(item.source_slot for item in selected_sources)
        if len(set(selected_slots)) != len(selected_slots):
            raise RecurrenceColumnarInputError(
                "selected source generation coverage repeats a source slot"
            )
        for item in selected_sources:
            source_slot = _require_index(
                item.source_slot,
                len(external_legs),
                "selected source slot",
            )
            state_count = len(external_legs[source_slot].source_states)
            for state_index in item.source_state_indices:
                _require_index(
                    state_index,
                    state_count,
                    "selected source-state index",
                )

    if logical.layout == "topology-replay" and not partitions:
        raise RecurrenceColumnarInputError(
            "topology-replay recurrence input requires replay partitions"
        )


def _build_header(
    logical: RecurrenceBuilderLogicalInputV1,
    strings: _StringCatalog,
    bitsets: _BitsetCatalog,
    *,
    external_count: int,
    sector_count: int,
    partition_count: int,
    coupling_limit_count: int,
    parameter_count: int,
    selected_sector_mode: bool,
    selected_source_mode: bool,
) -> RecurrenceColumnarTable:
    columns = _allocate(
        1,
        schema_version=_U32,
        abi_string_id=_U32,
        process_id_string_id=_U32,
        layout=_U8,
        selected_sector_mode=_U8,
        selected_source_mode=_U8,
        external_leg_count=_U32,
        physical_sector_count=_U32,
        replay_partition_count=_U32,
        coupling_limit_count=_U32,
        parameter_projection_count=_U32,
        process_support_mask_id=_U32,
    )
    columns["schema_version"][0] = RECURRENCE_BUILDER_INPUT_SCHEMA_VERSION
    columns["abi_string_id"][0] = strings.id(RECURRENCE_BUILDER_INPUT_ABI)
    columns["process_id_string_id"][0] = strings.id(logical.process_id)
    columns["layout"][0] = _LAYOUT_CODES[logical.layout]
    columns["selected_sector_mode"][0] = int(selected_sector_mode)
    columns["selected_source_mode"][0] = int(selected_source_mode)
    columns["external_leg_count"][0] = _checked_u32(external_count, "external count")
    columns["physical_sector_count"][0] = _checked_u32(sector_count, "sector count")
    columns["replay_partition_count"][0] = _checked_u32(
        partition_count,
        "replay partition count",
    )
    columns["coupling_limit_count"][0] = _checked_u32(
        coupling_limit_count,
        "coupling-limit count",
    )
    columns["parameter_projection_count"][0] = _checked_u32(
        parameter_count,
        "parameter-projection count",
    )
    columns["process_support_mask_id"][0] = bitsets.id(logical.process_support_mask)
    return _freeze_table("header", columns)


def _build_header_digests(
    values: tuple[RecurrenceSemanticDigestV1, ...],
    strings: _StringCatalog,
    digests: _DigestCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate(len(values), role_string_id=_U32, digest_id=_U32)
    for row, value in enumerate(values):
        columns["role_string_id"][row] = strings.id(value.role)
        columns["digest_id"][row] = digests.id(value.digest)
    return _freeze_table("header_digests", columns)


def _build_external_tables(
    values: tuple[RecurrenceExternalLegV1, ...],
    bitsets: _BitsetCatalog,
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    legs = _allocate(
        len(values),
        source_slot=_U32,
        public_label=_U32,
        physical_pdg=_I32,
        outgoing_pdg=_I32,
        particle_state_template_id=_U32,
        source_state_start=_U64,
        source_state_count=_U64,
        momentum_mask_id=_U32,
        support_mask_id=_U32,
    )
    state_count = sum(len(value.source_states) for value in values)
    states = _allocate(
        state_count,
        source_slot=_U32,
        state_index=_U32,
        spin_state=_I32,
        source_template_id=_U32,
    )
    cursor = 0
    for row, value in enumerate(values):
        legs["source_slot"][row] = value.source_slot
        legs["public_label"][row] = value.public_label
        legs["physical_pdg"][row] = value.physical_pdg
        legs["outgoing_pdg"][row] = value.outgoing_pdg
        legs["particle_state_template_id"][row] = value.particle_state_template_id
        legs["source_state_start"][row] = cursor
        legs["source_state_count"][row] = len(value.source_states)
        legs["momentum_mask_id"][row] = bitsets.id(value.momentum_mask)
        legs["support_mask_id"][row] = bitsets.id(value.support_mask)
        for state in value.source_states:
            states["source_slot"][cursor] = value.source_slot
            states["state_index"][cursor] = state.state_index
            states["spin_state"][cursor] = state.spin_state
            states["source_template_id"][cursor] = state.source_template_id
            cursor += 1
    return (
        _freeze_table("external_legs", legs),
        _freeze_table("source_states", states),
    )


def _build_sector_tables(
    values: tuple[RecurrencePhysicalLCSectorV1, ...],
    strings: _StringCatalog,
    sequences: _U32SequenceCatalog,
    bitsets: _BitsetCatalog,
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    sectors = _allocate(
        len(values),
        sector_id=_U32,
        public_id_string_id=_U32,
        kind=_U8,
        open_string_start=_U64,
        open_string_count=_U64,
        trace_sequence_id=_U32,
        singlet_sequence_id=_U32,
        word_sequence_id=_U32,
        support_mask_id=_U32,
    )
    line_count = sum(len(value.open_strings) for value in values)
    lines = _allocate(
        line_count,
        sector_id=_U32,
        ordinal=_U32,
        fundamental_source_slot=_U32,
        antifundamental_source_slot=_U32,
        adjoint_sequence_id=_U32,
        singlet_sequence_id=_U32,
    )
    cursor = 0
    for row, value in enumerate(values):
        sectors["sector_id"][row] = value.sector_id
        sectors["public_id_string_id"][row] = strings.id(value.public_id)
        sectors["kind"][row] = _SECTOR_KIND_CODES[value.kind]
        sectors["open_string_start"][row] = cursor
        sectors["open_string_count"][row] = len(value.open_strings)
        sectors["trace_sequence_id"][row] = sequences.id(value.trace_source_slots)
        sectors["singlet_sequence_id"][row] = sequences.id(value.singlet_source_slots)
        sectors["word_sequence_id"][row] = sequences.id(value.word_source_slots)
        sectors["support_mask_id"][row] = bitsets.id(value.support_mask)
        for ordinal, line in enumerate(value.open_strings):
            lines["sector_id"][cursor] = value.sector_id
            lines["ordinal"][cursor] = ordinal
            lines["fundamental_source_slot"][cursor] = line.fundamental_source_slot
            lines["antifundamental_source_slot"][cursor] = (
                line.antifundamental_source_slot
            )
            lines["adjoint_sequence_id"][cursor] = sequences.id(
                line.adjoint_source_slots
            )
            lines["singlet_sequence_id"][cursor] = sequences.id(
                line.singlet_source_slots
            )
            cursor += 1
    return (
        _freeze_table("physical_lc_sectors", sectors),
        _freeze_table("lc_open_strings", lines),
    )


def _build_replay_tables(
    values: tuple[RecurrenceReplayPartitionV1, ...],
    strings: _StringCatalog,
    digests: _DigestCatalog,
    sequences: _U32SequenceCatalog,
    factors: _FactorCatalog,
) -> tuple[RecurrenceColumnarTable, RecurrenceColumnarTable]:
    partitions = _allocate(
        len(values),
        partition_id=_U32,
        representative_sector_id=_U32,
        materialized_sector_id=_U32,
        target_start=_U64,
        target_count=_U64,
        proof_algorithm_string_id=_U32,
        proof_digest_id=_U32,
    )
    target_count = sum(len(value.targets) for value in values)
    targets = _allocate(
        target_count,
        partition_id=_U32,
        sector_id=_U32,
        external_permutation_sequence_id=_U32,
        source_state_bijection_sequence_id=_U32,
        closure_mapping_sequence_id=_U32,
        factor_id=_U32,
        fermion_sign=_I32,
    )
    cursor = 0
    for partition_id, value in enumerate(values):
        partitions["partition_id"][partition_id] = partition_id
        partitions["representative_sector_id"][partition_id] = (
            value.representative_sector_id
        )
        partitions["materialized_sector_id"][partition_id] = (
            value.materialized_sector_id
        )
        partitions["target_start"][partition_id] = cursor
        partitions["target_count"][partition_id] = len(value.targets)
        partitions["proof_algorithm_string_id"][partition_id] = strings.id(
            value.proof_algorithm
        )
        partitions["proof_digest_id"][partition_id] = digests.id(value.proof_digest)
        for target in sorted(value.targets, key=lambda item: item.sector_id):
            targets["partition_id"][cursor] = partition_id
            targets["sector_id"][cursor] = target.sector_id
            targets["external_permutation_sequence_id"][cursor] = sequences.id(
                target.external_permutation
            )
            targets["source_state_bijection_sequence_id"][cursor] = sequences.id(
                target.source_state_bijection
            )
            targets["closure_mapping_sequence_id"][cursor] = sequences.id(
                target.closure_mapping
            )
            targets["factor_id"][cursor] = factors.id(target.factor)
            targets["fermion_sign"][cursor] = target.fermion_sign
            cursor += 1
    return (
        _freeze_table("replay_partitions", partitions),
        _freeze_table("replay_targets", targets),
    )


def _build_selected_sectors(
    values: tuple[int, ...] | None,
) -> RecurrenceColumnarTable:
    selected = () if values is None else values
    columns = _allocate(len(selected), sector_id=_U32)
    if selected:
        columns["sector_id"][:] = selected
    return _freeze_table("selected_sector_coverage", columns)


def _build_selected_sources(
    values: tuple[RecurrenceSelectedSourceCoverageV1, ...] | None,
) -> RecurrenceColumnarTable:
    rows = (
        ()
        if values is None
        else tuple(
            (item.source_slot, state_index)
            for item in values
            for state_index in sorted(item.source_state_indices)
        )
    )
    columns = _allocate(len(rows), source_slot=_U32, source_state_index=_U32)
    for row, (source_slot, state_index) in enumerate(rows):
        columns["source_slot"][row] = source_slot
        columns["source_state_index"][row] = state_index
    return _freeze_table("selected_source_coverage", columns)


def _build_coupling_limits(
    values: tuple[RecurrenceCouplingLimitV1, ...],
    strings: _StringCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate(
        len(values),
        name_string_id=_U32,
        minimum=_U32,
        maximum=_U32,
    )
    for row, value in enumerate(values):
        columns["name_string_id"][row] = strings.id(value.name)
        columns["minimum"][row] = value.minimum
        columns["maximum"][row] = value.maximum
    return _freeze_table("coupling_limits", columns)


def _build_template_references(
    values: tuple[RecurrenceSemanticTemplateReferenceV1, ...],
    strings: _StringCatalog,
    digests: _DigestCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate(
        len(values),
        kind_string_id=_U32,
        template_id=_U32,
        semantic_digest_id=_U32,
        prepared_kernel_id=_U32,
    )
    for row, value in enumerate(values):
        columns["kind_string_id"][row] = strings.id(value.kind)
        columns["template_id"][row] = value.template_id
        columns["semantic_digest_id"][row] = digests.id(value.semantic_digest)
        columns["prepared_kernel_id"][row] = (
            MISSING_U32
            if value.prepared_kernel_id is None
            else value.prepared_kernel_id
        )
    return _freeze_table("semantic_template_references", columns)


def _build_normalization(
    value: RecurrenceNormalizationV1,
    strings: _StringCatalog,
    digests: _DigestCatalog,
    factors: _FactorCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate(
        1,
        factor_id=_U32,
        convention_string_id=_U32,
        semantic_digest_id=_U32,
    )
    columns["factor_id"][0] = factors.id(value.factor)
    columns["convention_string_id"][0] = strings.id(value.convention)
    columns["semantic_digest_id"][0] = digests.id(value.semantic_digest)
    return _freeze_table("normalization", columns)


def _build_parameter_projection(
    values: tuple[RecurrenceParameterProjectionV1, ...],
    strings: _StringCatalog,
) -> RecurrenceColumnarTable:
    columns = _allocate(
        len(values),
        runtime_slot=_U32,
        runtime_name_string_id=_U32,
        parameter_template_id=_U32,
        prepared_parameter_id=_U32,
        component=_U32,
    )
    for row, value in enumerate(values):
        columns["runtime_slot"][row] = value.runtime_slot
        columns["runtime_name_string_id"][row] = strings.id(value.runtime_name)
        columns["parameter_template_id"][row] = value.parameter_template_id
        columns["prepared_parameter_id"][row] = value.prepared_parameter_id
        columns["component"][row] = value.component
    return _freeze_table("parameter_projection", columns)


def _allocate(
    row_count: int,
    **specifications: np.dtype[Any] | tuple[np.dtype[Any], tuple[int, ...]],
) -> dict[str, np.ndarray[Any, Any]]:
    _checked_u64(row_count, "table row count")
    result: dict[str, np.ndarray[Any, Any]] = {}
    for name, specification in specifications.items():
        if isinstance(specification, tuple):
            dtype, tail_shape = specification
        else:
            dtype, tail_shape = specification, ()
        result[name] = np.empty((row_count, *tail_shape), dtype=dtype, order="C")
    return result


def _freeze_table(
    name: str,
    columns: Mapping[str, np.ndarray[Any, Any]],
) -> RecurrenceColumnarTable:
    row_counts = {len(values) for values in columns.values()}
    row_count = row_counts.pop() if row_counts else 0
    if row_counts:
        raise RecurrenceColumnarInputError(
            f"table {name!r} columns have unequal lengths"
        )
    frozen: list[RecurrenceColumn] = []
    for column_name, values in columns.items():
        owned = np.array(values, dtype=values.dtype, order="C", copy=True)
        owned.flags.writeable = False
        frozen.append(RecurrenceColumn(column_name, owned))
    return RecurrenceColumnarTable(name, row_count, tuple(frozen))


def _decode_flat_strings(
    ranges: RecurrenceColumnarTable,
    values: RecurrenceColumnarTable,
) -> tuple[str, ...]:
    _validate_flat_ranges(ranges, values.row_count, "string catalog")
    flat = values.column("value")
    result: list[str] = []
    for row in range(ranges.row_count):
        start = int(ranges.column("start")[row])
        count = int(ranges.column("count")[row])
        try:
            result.append(bytes(flat[start : start + count]).decode("utf-8"))
        except UnicodeDecodeError as error:
            raise RecurrenceColumnarInputError(
                "flat string catalog contains invalid UTF-8"
            ) from error
    return tuple(result)


def _column_dtype(table_name: str, column_name: str) -> np.dtype[Any]:
    key = (table_name, column_name)
    if key in _U8_COLUMN_KEYS:
        return _U8
    if key in _U64_COLUMN_KEYS:
        return _U64
    if key in _I32_COLUMN_KEYS:
        return _I32
    return _U32


def _validate_flat_ranges(
    ranges: RecurrenceColumnarTable,
    value_count: int,
    context: str,
) -> None:
    cursor = 0
    for row in range(ranges.row_count):
        start = int(ranges.column("start")[row])
        count = int(ranges.column("count")[row])
        if start != cursor:
            raise RecurrenceColumnarInputError(
                f"{context} ranges are not contiguous and canonical"
            )
        stop = start + count
        if stop > value_count:
            raise RecurrenceColumnarInputError(
                f"{context} range references values outside its flat catalog"
            )
        cursor = stop
    if cursor != value_count:
        raise RecurrenceColumnarInputError(
            f"{context} does not account for every flat value"
        )


def _validate_parent_ranges(
    table: RecurrenceColumnarTable,
    start_column: str,
    count_column: str,
    child_count: int,
    context: str,
) -> None:
    cursor = 0
    for row in range(table.row_count):
        start = int(table.column(start_column)[row])
        count = int(table.column(count_column)[row])
        if start != cursor or start + count > child_count:
            raise RecurrenceColumnarInputError(
                f"{context} ranges are not contiguous or are out of bounds"
            )
        cursor = start + count
    if cursor != child_count:
        raise RecurrenceColumnarInputError(
            f"{context} ranges do not cover every child row"
        )


def _validate_dense_ids(table: RecurrenceColumnarTable, column_name: str) -> None:
    expected = np.arange(table.row_count, dtype=_U32)
    if not np.array_equal(table.column(column_name), expected):
        raise RecurrenceColumnarInputError(
            f"table {table.name!r} column {column_name!r} is not dense"
        )


def _validate_bounded_columns(
    table: RecurrenceColumnarTable,
    columns: Sequence[str],
    bound: int,
) -> None:
    for column_name in columns:
        values = table.column(column_name)
        if values.size and int(values.max()) >= bound:
            raise RecurrenceColumnarInputError(
                f"{table.name}.{column_name} references an absent row"
            )


def _validate_fraction(numerator: int, denominator: int, context: str) -> None:
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError(f"{context} numerator must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        raise TypeError(f"{context} denominator must be an integer")
    if denominator <= 0:
        raise RecurrenceColumnarInputError(f"{context} denominator must be positive")
    if numerator == 0 and denominator != 1:
        raise RecurrenceColumnarInputError(f"{context} zero must be encoded as 0/1")
    if math.gcd(abs(numerator), denominator) != 1:
        raise RecurrenceColumnarInputError(f"{context} must be reduced")


def _nonempty_text(value: str, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    if not value:
        raise RecurrenceColumnarInputError(f"{context} must not be empty")
    return value


def _sha256_bytes(value: str, context: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrenceColumnarInputError(f"{context} must be a lowercase SHA-256")
    return bytes.fromhex(value)


def _validate_nonnegative_mask(value: int, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise RecurrenceColumnarInputError(f"{context} must be nonnegative")


def _validate_u32_sequence(values: Sequence[int], context: str) -> None:
    for index, value in enumerate(values):
        _checked_u32(value, f"{context}[{index}]")


def _validate_permutation(values: Sequence[int], size: int, context: str) -> None:
    if tuple(sorted(values)) != tuple(range(size)):
        raise RecurrenceColumnarInputError(
            f"{context} must be a permutation of [0, {size})"
        )


def _require_index(value: int, bound: int, context: str) -> int:
    result = _checked_u32(value, context)
    if result >= bound:
        raise RecurrenceColumnarInputError(
            f"{context} {result} is outside [0, {bound})"
        )
    return result


def _checked_u32(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MISSING_U32
    ):
        raise RecurrenceColumnarInputError(f"{context} does not fit u32: {value!r}")
    return value


def _checked_u64(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise RecurrenceColumnarInputError(f"{context} does not fit u64: {value!r}")
    return value


def _checked_i32(value: int, context: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(1 << 31) <= value < 1 << 31
    ):
        raise RecurrenceColumnarInputError(f"{context} does not fit i32: {value!r}")
    return value


def _hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


__all__ = [
    "MISSING_U32",
    "RECURRENCE_BUILDER_INPUT_ABI",
    "RECURRENCE_BUILDER_INPUT_SCHEMA_VERSION",
    "ExactComplexRationalV1",
    "LCColorSectorKind",
    "RecurrenceBuilderInputV1",
    "RecurrenceBuilderLogicalInputV1",
    "RecurrenceColumn",
    "RecurrenceColumnarInputError",
    "RecurrenceColumnarTable",
    "RecurrenceCouplingLimitV1",
    "RecurrenceExternalLegV1",
    "RecurrenceLCFlowLayout",
    "RecurrenceLCOpenStringV1",
    "RecurrenceNormalizationV1",
    "RecurrenceParameterProjectionV1",
    "RecurrencePhysicalLCSectorV1",
    "RecurrenceReplayPartitionV1",
    "RecurrenceReplayTargetV1",
    "RecurrenceSelectedSourceCoverageV1",
    "RecurrenceSemanticDigestV1",
    "RecurrenceSemanticTemplateReferenceV1",
    "RecurrenceSourceStateV1",
    "build_recurrence_builder_input_v1",
]
