// SPDX-License-Identifier: 0BSD

//! Private Python boundary for recurrence builder-input v1.
//!
//! This milestone authenticates and validates the compact columnar input.  It
//! deliberately stops before recurrence-state construction or schedule
//! lowering; those operations will consume the same owned input in a later
//! milestone.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rusticol_core::recurrence::process;
use rusticol_core::recurrence::template;
use rusticol_core::recurrence::{
    AuthenticatedRecurrenceBuilderInput, CheckedTableRange, CurrentSourceBinding,
    RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_BUILDER_RESULT_ABI, RECURRENCE_LC_COLOR_CAPABILITY,
    RECURRENCE_PLAN_ABI, RECURRENCE_RUNTIME_CAPABILITY, RECURRENCE_RUNTIME_KIND,
    RECURRENCE_RUNTIME_LAYOUT_ABI, RECURRENCE_TEMPLATE_ABI, RecurrenceStrategy, SemanticDigest,
    checked_usize,
};
use rusticol_core::recurrence::{RecurrenceExecutionPlan, RecurrenceExecutionRuntime};
use rusticol_core::{
    EagerComplex64, NativeRecurrenceKernelBackend, NativeRecurrenceKernelBackendSummary,
    RusticolError, RusticolResult,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::python_error;

const INPUT_SCHEMA_VERSION: u32 = 1;
const RESULT_KIND: &str = "pyamplicol-recurrence-builder-validation-result";
const RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimitiveKind {
    U8,
    U32,
    U64,
    I32,
}

impl PrimitiveKind {
    const fn dtype(self) -> &'static str {
        match self {
            Self::U8 => "|u1",
            Self::U32 => "<u4",
            Self::U64 => "<u8",
            Self::I32 => "<i4",
        }
    }

    fn from_dtype(value: &str) -> Option<Self> {
        match value {
            "|u1" => Some(Self::U8),
            "<u4" => Some(Self::U32),
            "<u8" => Some(Self::U64),
            "<i4" => Some(Self::I32),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct ColumnSpec {
    name: &'static str,
    kind: PrimitiveKind,
    tail_shape: &'static [usize],
}

#[derive(Clone, Copy)]
struct TableSpec {
    name: &'static str,
    columns: &'static [ColumnSpec],
}

const fn column(name: &'static str, kind: PrimitiveKind) -> ColumnSpec {
    ColumnSpec {
        name,
        kind,
        tail_shape: &[],
    }
}

const fn shaped_column(
    name: &'static str,
    kind: PrimitiveKind,
    tail_shape: &'static [usize],
) -> ColumnSpec {
    ColumnSpec {
        name,
        kind,
        tail_shape,
    }
}

const TABLE_SPECS: &[TableSpec] = &[
    TableSpec {
        name: "bitset_ranges",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
            column("bit_count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "bitset_words",
        columns: &[column("value", PrimitiveKind::U64)],
    },
    TableSpec {
        name: "coupling_limits",
        columns: &[
            column("name_string_id", PrimitiveKind::U32),
            column("minimum", PrimitiveKind::U32),
            column("maximum", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "digest_catalog",
        columns: &[
            column("id", PrimitiveKind::U32),
            shaped_column("value", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "exact_factors",
        columns: &[
            column("id", PrimitiveKind::U32),
            column("real_numerator_string_id", PrimitiveKind::U32),
            column("real_denominator_string_id", PrimitiveKind::U32),
            column("imag_numerator_string_id", PrimitiveKind::U32),
            column("imag_denominator_string_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "external_legs",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("public_label", PrimitiveKind::U32),
            column("physical_pdg", PrimitiveKind::I32),
            column("outgoing_pdg", PrimitiveKind::I32),
            column("is_initial", PrimitiveKind::U8),
            column("is_fermionic", PrimitiveKind::U8),
            column("source_state_start", PrimitiveKind::U64),
            column("source_state_count", PrimitiveKind::U64),
            column("momentum_mask_id", PrimitiveKind::U32),
            column("support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "header",
        columns: &[
            column("schema_version", PrimitiveKind::U32),
            column("abi_string_id", PrimitiveKind::U32),
            column("process_id_string_id", PrimitiveKind::U32),
            column("layout", PrimitiveKind::U8),
            column("selected_flow_mode", PrimitiveKind::U8),
            column("selected_source_mode", PrimitiveKind::U8),
            column("external_leg_count", PrimitiveKind::U32),
            column("physical_sector_count", PrimitiveKind::U32),
            column("public_flow_count", PrimitiveKind::U32),
            column("replay_partition_count", PrimitiveKind::U32),
            column("coupling_limit_count", PrimitiveKind::U32),
            column("parameter_projection_count", PrimitiveKind::U32),
            column("process_support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "header_digests",
        columns: &[
            column("role_string_id", PrimitiveKind::U32),
            column("digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "lc_open_strings",
        columns: &[
            column("sector_id", PrimitiveKind::U32),
            column("ordinal", PrimitiveKind::U32),
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
            column("adjoint_sequence_id", PrimitiveKind::U32),
            column("singlet_sequence_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "normalization",
        columns: &[
            column("factor_id", PrimitiveKind::U32),
            column("convention_string_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "parameter_projection",
        columns: &[
            column("runtime_slot", PrimitiveKind::U32),
            column("runtime_name_string_id", PrimitiveKind::U32),
            column("parameter_template_id", PrimitiveKind::U32),
            column("prepared_parameter_id", PrimitiveKind::U32),
            column("component", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "physical_lc_sectors",
        columns: &[
            column("sector_id", PrimitiveKind::U32),
            column("public_id_string_id", PrimitiveKind::U32),
            column("kind", PrimitiveKind::U8),
            column("closure_source_slot", PrimitiveKind::U32),
            column("closure_proof_algorithm_string_id", PrimitiveKind::U32),
            column("closure_proof_digest_id", PrimitiveKind::U32),
            column("open_string_start", PrimitiveKind::U64),
            column("open_string_count", PrimitiveKind::U64),
            column("trace_sequence_id", PrimitiveKind::U32),
            column("singlet_sequence_id", PrimitiveKind::U32),
            column("word_sequence_id", PrimitiveKind::U32),
            column("support_mask_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "public_lc_flows",
        columns: &[
            column("flow_id", PrimitiveKind::U32),
            column("public_id_string_id", PrimitiveKind::U32),
            column("construction_sector_id", PrimitiveKind::U32),
            column("word_sequence_id", PrimitiveKind::U32),
            column("source_slot_permutation_sequence_id", PrimitiveKind::U32),
            column("reduction_weight_factor_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "replay_partitions",
        columns: &[
            column("partition_id", PrimitiveKind::U32),
            column("representative_sector_id", PrimitiveKind::U32),
            column("materialized_sector_id", PrimitiveKind::U32),
            column("target_start", PrimitiveKind::U64),
            column("target_count", PrimitiveKind::U64),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("proof_digest_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "replay_targets",
        columns: &[
            column("partition_id", PrimitiveKind::U32),
            column("sector_id", PrimitiveKind::U32),
            column("external_permutation_sequence_id", PrimitiveKind::U32),
            column("source_slot_permutation_sequence_id", PrimitiveKind::U32),
            column("amplitude_phase_factor_id", PrimitiveKind::U32),
            column("fermion_sign", PrimitiveKind::I32),
        ],
    },
    TableSpec {
        name: "selected_public_flow_coverage",
        columns: &[column("flow_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "selected_source_coverage",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("source_state_index", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "semantic_template_references",
        columns: &[
            column("kind_string_id", PrimitiveKind::U32),
            column("template_id", PrimitiveKind::U32),
            column("semantic_digest_id", PrimitiveKind::U32),
            column("prepared_kernel_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "source_states",
        columns: &[
            column("source_slot", PrimitiveKind::U32),
            column("state_index", PrimitiveKind::U32),
            column("public_helicity", PrimitiveKind::I32),
            column("chirality", PrimitiveKind::I32),
            column("spin_state", PrimitiveKind::I32),
            column("current_state_template_id", PrimitiveKind::U32),
            column("source_template_id", PrimitiveKind::U32),
            column("momentum_sign", PrimitiveKind::I32),
            column("crossing_phase_factor_id", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "string_bytes",
        columns: &[column("value", PrimitiveKind::U8)],
    },
    TableSpec {
        name: "string_ranges",
        columns: &[
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "u32_sequence_ranges",
        columns: &[
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "u32_sequence_values",
        columns: &[column("value", PrimitiveKind::U32)],
    },
];

#[derive(Clone)]
enum OwnedValues {
    U8(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
}

impl OwnedValues {
    fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }

    fn raw_bytes(&self) -> &[u8] {
        fn bytes<T>(values: &[T]) -> &[u8] {
            // Accepted multi-byte arrays are explicitly little-endian and this
            // module rejects big-endian hosts before extraction.
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            }
        }
        match self {
            Self::U8(values) => values,
            Self::U32(values) => bytes(values),
            Self::U64(values) => bytes(values),
            Self::I32(values) => bytes(values),
        }
    }

    fn as_u8(&self, context: &str) -> RusticolResult<&[u8]> {
        match self {
            Self::U8(values) => Ok(values),
            _ => Err(wrong_type(context, "u8")),
        }
    }

    fn as_u32(&self, context: &str) -> RusticolResult<&[u32]> {
        match self {
            Self::U32(values) => Ok(values),
            _ => Err(wrong_type(context, "u32")),
        }
    }

    fn as_u64(&self, context: &str) -> RusticolResult<&[u64]> {
        match self {
            Self::U64(values) => Ok(values),
            _ => Err(wrong_type(context, "u64")),
        }
    }

    fn as_i32(&self, context: &str) -> RusticolResult<&[i32]> {
        match self {
            Self::I32(values) => Ok(values),
            _ => Err(wrong_type(context, "i32")),
        }
    }
}

#[derive(Clone)]
struct OwnedColumn {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    values: OwnedValues,
}

#[derive(Clone)]
struct OwnedTable {
    name: String,
    row_count: u64,
    columns: Vec<OwnedColumn>,
    column_by_name: BTreeMap<String, usize>,
}

impl OwnedTable {
    fn row_count(&self) -> RusticolResult<usize> {
        checked_usize(self.row_count, &format!("{} row count", self.name))
    }

    fn column(&self, name: &str) -> RusticolResult<&OwnedColumn> {
        let index = self.column_by_name.get(name).ok_or_else(|| {
            invalid(format!(
                "recurrence builder table {:?} has no column {name:?}",
                self.name
            ))
        })?;
        Ok(&self.columns[*index])
    }

    fn u8(&self, column: &str) -> RusticolResult<&[u8]> {
        self.column(column)?
            .values
            .as_u8(&format!("{}.{column}", self.name))
    }

    fn u32(&self, column: &str) -> RusticolResult<&[u32]> {
        self.column(column)?
            .values
            .as_u32(&format!("{}.{column}", self.name))
    }

    fn u64(&self, column: &str) -> RusticolResult<&[u64]> {
        self.column(column)?
            .values
            .as_u64(&format!("{}.{column}", self.name))
    }

    fn i32(&self, column: &str) -> RusticolResult<&[i32]> {
        self.column(column)?
            .values
            .as_i32(&format!("{}.{column}", self.name))
    }
}

#[derive(Clone)]
struct OwnedInput {
    abi: String,
    declared_digest: String,
    tables: Vec<OwnedTable>,
    table_by_name: BTreeMap<String, usize>,
}

impl OwnedInput {
    fn table(&self, name: &str) -> RusticolResult<&OwnedTable> {
        let index = self
            .table_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("recurrence builder input has no table {name:?}")))?;
        Ok(&self.tables[*index])
    }

    fn column(&self, table: &str, column: &str) -> RusticolResult<&OwnedColumn> {
        self.table(table)?.column(column)
    }

    fn u8(&self, table: &str, column: &str) -> RusticolResult<&[u8]> {
        self.column(table, column)?
            .values
            .as_u8(&format!("{table}.{column}"))
    }

    fn u32(&self, table: &str, column: &str) -> RusticolResult<&[u32]> {
        self.column(table, column)?
            .values
            .as_u32(&format!("{table}.{column}"))
    }

    fn u64(&self, table: &str, column: &str) -> RusticolResult<&[u64]> {
        self.column(table, column)?
            .values
            .as_u64(&format!("{table}.{column}"))
    }

    fn i32(&self, table: &str, column: &str) -> RusticolResult<&[i32]> {
        self.column(table, column)?
            .values
            .as_i32(&format!("{table}.{column}"))
    }
}

const TEMPLATE_TABLE_INVENTORY: &[(&str, usize)] = &[
    ("catalog_header", 18),
    ("closures", 20),
    ("color_contractions", 14),
    ("color_nc_terms", 3),
    ("coupling_order_ranges", 3),
    ("coupling_order_terms", 3),
    ("current_states", 17),
    ("digest_catalog", 2),
    ("evaluator_bindings", 14),
    ("exact_factors", 5),
    ("flavour_flow_ranges", 3),
    ("flavour_flow_values", 1),
    ("i32_sequence_ranges", 3),
    ("i32_sequence_values", 1),
    ("lc_color_transition_witnesses", 13),
    ("parameters", 11),
    ("propagators", 12),
    ("quantum_flows", 16),
    ("quantum_number_flow_ranges", 3),
    ("quantum_number_flow_terms", 3),
    ("runtime_helicity_contracts", 8),
    ("runtime_helicity_embeddings", 4),
    ("runtime_helicity_projections", 3),
    ("runtime_helicity_variants", 9),
    ("sources", 20),
    ("string_bytes", 1),
    ("string_ranges", 2),
    ("symmetry_proofs", 9),
    ("transitions", 18),
    ("u32_sequence_ranges", 3),
    ("u32_sequence_values", 1),
];

struct PreparedTemplateInput {
    input: OwnedInput,
    canonical_digest_property: String,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
}

impl PreparedTemplateInput {
    fn canonical_digest(&self) -> RusticolResult<String> {
        let mut digest = Sha256::new();
        hash_text(&mut digest, &self.input.abi)?;
        digest.update(self.catalog_digest.as_bytes());
        digest.update(self.compiled_model_digest.as_bytes());
        digest.update(self.prepared_kernel_pack_digest.as_bytes());
        hash_tables(&mut digest, &self.input.tables)?;
        Ok(hex_digest(digest.finalize()))
    }

    fn into_core(self) -> RusticolResult<template::OwnedRecurrenceTemplateInput> {
        let actual_digest = self.canonical_digest()?;
        if actual_digest != self.input.declared_digest
            || actual_digest != self.canonical_digest_property
        {
            return Err(invalid(format!(
                "recurrence template input digest mismatch: declared {}, found {actual_digest}",
                self.input.declared_digest
            )));
        }
        decode_template_input(
            &self.input,
            self.catalog_digest,
            self.compiled_model_digest,
            self.prepared_kernel_pack_digest,
        )
    }
}

struct NativeValidationResult {
    digest: String,
    process_id: String,
    strategy: RecurrenceStrategy,
    table_count: usize,
    column_count: usize,
    total_row_count: u64,
    primitive_element_count: u64,
    primitive_byte_count: u64,
    string_count: usize,
    semantic_digest_count: usize,
    exact_factor_count: usize,
    external_leg_count: usize,
    source_state_count: usize,
    physical_sector_count: usize,
    public_flow_count: usize,
    open_string_count: usize,
    replay_partition_count: usize,
    replay_target_count: usize,
    selector_mask_count: usize,
    selector_mask_word_count: usize,
    maximum_mask_bit: Option<u64>,
    template_reference_count: usize,
    parameter_projection_count: usize,
    composite_authenticated: bool,
    template_catalog_digest: Option<SemanticDigest>,
    compiled_model_digest: Option<SemanticDigest>,
    prepared_kernel_pack_digest: Option<SemanticDigest>,
    prepared_template_count: Option<usize>,
    schedule_summary: Option<NativeScheduleSummary>,
}

#[derive(Clone, Debug)]
struct NativeScheduleSummary {
    dynamic_color_state_count: usize,
    current_count: usize,
    source_current_count: usize,
    current_count_by_support_size: Vec<usize>,
    contribution_count: usize,
    contribution_count_by_transition_template_id: Vec<(u32, usize)>,
    contribution_count_by_quantum_flow_template_id: Vec<(u32, usize)>,
    referenced_quantum_flow_template_ids: Vec<u32>,
    finalization_count: usize,
    identity_finalization_count_by_support_size: Vec<usize>,
    propagated_finalization_count_by_support_size: Vec<usize>,
    retained_helicity_count: u64,
    resolved_helicity_count: usize,
    structural_zero_helicity_count: u64,
    amplitude_destination_count: usize,
    target_sector_count: usize,
    closure_term_count: usize,
    source_layout: Vec<NativeSourceLayout>,
}

#[derive(Clone, Copy, Debug)]
struct NativeSourceLayout {
    current_id: u32,
    source_slot: u32,
    source_template_id: u32,
    component_start: usize,
    component_count: usize,
}

#[pyfunction(signature = (builder_input, prepared_template_input=None, *, construct_schedule=false))]
pub(crate) fn _validate_recurrence_builder_input_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: Option<&Bound<'_, PyAny>>,
    construct_schedule: bool,
) -> PyResult<Py<PyAny>> {
    let owned = parse_input(builder_input)?;
    let prepared_template = prepared_template_input
        .map(parse_prepared_template_input)
        .transpose()?;
    let native = py
        .detach(move || validate_input(owned, prepared_template, construct_schedule))
        .map_err(python_error)?;
    result_mapping(py, native)
}

#[pyfunction(
    signature = (
        builder_input,
        prepared_template_input,
        prepared_kernel_manifest,
        prepared_kernel_pack_digest,
        payload_root,
        source_values,
        external_momenta,
        model_parameters,
        *,
        point_count,
        point_tile_size=1024
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _evaluate_recurrence_one_helicity_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    prepared_kernel_manifest: Vec<u8>,
    prepared_kernel_pack_digest: String,
    payload_root: PathBuf,
    source_values: Vec<(f64, f64)>,
    external_momenta: Vec<f64>,
    model_parameters: Vec<(f64, f64)>,
    point_count: usize,
    point_tile_size: usize,
) -> PyResult<Py<PyAny>> {
    let owned = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let native = py
        .detach(move || {
            evaluate_one_helicity(
                owned,
                prepared_template,
                &prepared_kernel_manifest,
                &prepared_kernel_pack_digest,
                &payload_root,
                &source_values,
                &external_momenta,
                &model_parameters,
                point_count,
                point_tile_size,
                false,
            )
        })
        .map_err(python_error)?;
    recurrence_evaluation_mapping(py, native)
}

#[pyfunction(
    signature = (
        builder_input,
        prepared_template_input,
        prepared_kernel_manifest,
        prepared_kernel_pack_digest,
        payload_root,
        source_values,
        external_momenta,
        model_parameters,
        *,
        point_count,
        point_tile_size=1024
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _evaluate_recurrence_all_helicities_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    prepared_kernel_manifest: Vec<u8>,
    prepared_kernel_pack_digest: String,
    payload_root: PathBuf,
    source_values: Vec<(f64, f64)>,
    external_momenta: Vec<f64>,
    model_parameters: Vec<(f64, f64)>,
    point_count: usize,
    point_tile_size: usize,
) -> PyResult<Py<PyAny>> {
    let owned = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let native = py
        .detach(move || {
            evaluate_one_helicity(
                owned,
                prepared_template,
                &prepared_kernel_manifest,
                &prepared_kernel_pack_digest,
                &payload_root,
                &source_values,
                &external_momenta,
                &model_parameters,
                point_count,
                point_tile_size,
                true,
            )
        })
        .map_err(python_error)?;
    recurrence_evaluation_mapping(py, native)
}

#[pyfunction(
    signature = (
        builder_input,
        prepared_template_input,
        prepared_kernel_manifest,
        prepared_kernel_pack_digest,
        payload_root,
        source_values,
        external_momenta,
        model_parameters,
        *,
        point_count,
        point_tile_size=1024
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _evaluate_recurrence_helicity_sum_norm_sqr_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    prepared_kernel_manifest: Vec<u8>,
    prepared_kernel_pack_digest: String,
    payload_root: PathBuf,
    source_values: Vec<(f64, f64)>,
    external_momenta: Vec<f64>,
    model_parameters: Vec<(f64, f64)>,
    point_count: usize,
    point_tile_size: usize,
) -> PyResult<Py<PyAny>> {
    let owned = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let native = py
        .detach(move || {
            evaluate_helicity_sum_norm_sqr(
                owned,
                prepared_template,
                &prepared_kernel_manifest,
                &prepared_kernel_pack_digest,
                &payload_root,
                &source_values,
                &external_momenta,
                &model_parameters,
                point_count,
                point_tile_size,
            )
        })
        .map_err(python_error)?;
    recurrence_norm_sqr_mapping(py, native)
}

#[derive(Clone, Debug)]
struct NativeRecurrenceEvaluation {
    amplitudes: Vec<(f64, f64)>,
    source_layout: Vec<rusticol_core::recurrence::RecurrenceSourceLayout>,
    sector_count: usize,
    resolved_helicities: Vec<Vec<i32>>,
    amplitude_destinations: Vec<NativeAmplitudeDestination>,
    resolved: bool,
    backend: NativeRecurrenceKernelBackendSummary,
}

#[derive(Clone, Debug)]
struct NativeRecurrenceNormSqrEvaluation {
    norm_sqr: Vec<f64>,
    source_layout: Vec<rusticol_core::recurrence::RecurrenceSourceLayout>,
    sector_count: usize,
    backend: NativeRecurrenceKernelBackendSummary,
}

#[derive(Clone, Copy, Debug)]
struct NativeAmplitudeDestination {
    id: u32,
    target_sector_id: u32,
    target_helicity_id: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_one_helicity(
    input: OwnedInput,
    prepared_template: PreparedTemplateInput,
    prepared_kernel_manifest: &[u8],
    prepared_kernel_pack_digest: &str,
    payload_root: &std::path::Path,
    source_values: &[(f64, f64)],
    external_momenta: &[f64],
    model_parameters: &[(f64, f64)],
    point_count: usize,
    point_tile_size: usize,
    resolved: bool,
) -> RusticolResult<NativeRecurrenceEvaluation> {
    validate_inventory(&input)?;
    let actual_digest = canonical_digest(&input)?;
    if actual_digest != input.declared_digest {
        return Err(invalid("recurrence builder input digest mismatch"));
    }
    let process = decode_process_input(&input)?.validate()?;
    let template = prepared_template.into_core()?.validate()?;
    let authenticated = AuthenticatedRecurrenceBuilderInput::new(process, template)?;
    if authenticated
        .template()
        .summary()
        .prepared_kernel_pack_digest
        .to_string()
        != prepared_kernel_pack_digest
    {
        return Err(RusticolError::integrity(
            "recurrence prepared-kernel pack digest does not match the authenticated template",
        ));
    }
    let program = authenticated.build()?;
    let mut backend = NativeRecurrenceKernelBackend::load(prepared_kernel_manifest, payload_root)?;
    let plan =
        RecurrenceExecutionPlan::new(program, authenticated.template(), backend.kernel_specs())?;
    let source_layout = plan.source_layout();
    let sector_count = plan.sector_count();
    let resolved_helicities = plan
        .program()
        .resolved_helicities()
        .iter()
        .map(|row| row.public_helicities().to_vec())
        .collect::<Vec<_>>();
    let amplitude_destinations = plan
        .program()
        .amplitude_destinations()
        .iter()
        .copied()
        .map(|destination| NativeAmplitudeDestination {
            id: destination.id(),
            target_sector_id: destination.target_sector_id(),
            target_helicity_id: destination.target_helicity_id(),
        })
        .collect::<Vec<_>>();
    let output_component_count = if resolved {
        plan.resolved_component_count()?
    } else {
        sector_count
    };
    let mut runtime = RecurrenceExecutionRuntime::new(plan, point_tile_size)?;
    let source_values = source_values
        .iter()
        .map(|(real, imag)| EagerComplex64::new(*real, *imag))
        .collect::<Vec<_>>();
    let model_parameters = model_parameters
        .iter()
        .map(|(real, imag)| EagerComplex64::new(*real, *imag))
        .collect::<Vec<_>>();
    let output_len = output_component_count
        .checked_mul(point_count)
        .ok_or_else(|| invalid("recurrence output length overflows"))?;
    let mut amplitudes = vec![EagerComplex64::new(0.0, 0.0); output_len];
    if resolved {
        runtime.evaluate_resolved_amplitudes_into(
            &mut backend,
            point_count,
            &source_values,
            external_momenta,
            &model_parameters,
            &mut amplitudes,
        )?;
    } else {
        runtime.evaluate_one_helicity_amplitudes_into(
            &mut backend,
            point_count,
            &source_values,
            external_momenta,
            &model_parameters,
            &mut amplitudes,
        )?;
    }
    Ok(NativeRecurrenceEvaluation {
        amplitudes: amplitudes
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
        source_layout,
        sector_count,
        resolved_helicities,
        amplitude_destinations,
        resolved,
        backend: backend.summary().clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_helicity_sum_norm_sqr(
    input: OwnedInput,
    prepared_template: PreparedTemplateInput,
    prepared_kernel_manifest: &[u8],
    prepared_kernel_pack_digest: &str,
    payload_root: &std::path::Path,
    source_values: &[(f64, f64)],
    external_momenta: &[f64],
    model_parameters: &[(f64, f64)],
    point_count: usize,
    point_tile_size: usize,
) -> RusticolResult<NativeRecurrenceNormSqrEvaluation> {
    validate_inventory(&input)?;
    let actual_digest = canonical_digest(&input)?;
    if actual_digest != input.declared_digest {
        return Err(invalid("recurrence builder input digest mismatch"));
    }
    let process = decode_process_input(&input)?.validate()?;
    let template = prepared_template.into_core()?.validate()?;
    let authenticated = AuthenticatedRecurrenceBuilderInput::new(process, template)?;
    if authenticated
        .template()
        .summary()
        .prepared_kernel_pack_digest
        .to_string()
        != prepared_kernel_pack_digest
    {
        return Err(RusticolError::integrity(
            "recurrence prepared-kernel pack digest does not match the authenticated template",
        ));
    }
    let program = authenticated.build()?;
    let mut backend = NativeRecurrenceKernelBackend::load(prepared_kernel_manifest, payload_root)?;
    let plan =
        RecurrenceExecutionPlan::new(program, authenticated.template(), backend.kernel_specs())?;
    let source_layout = plan.source_layout();
    let sector_count = plan.sector_count();
    let mut runtime = RecurrenceExecutionRuntime::new(plan, point_tile_size)?;
    let source_values = source_values
        .iter()
        .map(|(real, imag)| EagerComplex64::new(*real, *imag))
        .collect::<Vec<_>>();
    let model_parameters = model_parameters
        .iter()
        .map(|(real, imag)| EagerComplex64::new(*real, *imag))
        .collect::<Vec<_>>();
    let output_len = sector_count
        .checked_mul(point_count)
        .ok_or_else(|| invalid("recurrence norm-squared output length overflows"))?;
    let mut norm_sqr = vec![0.0; output_len];
    runtime.evaluate_helicity_sum_norm_sqr_into(
        &mut backend,
        point_count,
        &source_values,
        external_momenta,
        &model_parameters,
        &mut norm_sqr,
    )?;
    Ok(NativeRecurrenceNormSqrEvaluation {
        norm_sqr,
        source_layout,
        sector_count,
        backend: backend.summary().clone(),
    })
}

fn recurrence_evaluation_mapping(
    py: Python<'_>,
    native: NativeRecurrenceEvaluation,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", "pyamplicol-recurrence-private-evaluation-v1")?;
    result.set_item("sector_count", native.sector_count)?;
    result.set_item("resolved", native.resolved)?;
    result.set_item("resolved_helicities", native.resolved_helicities)?;
    let destinations = PyList::empty(py);
    for destination in native.amplitude_destinations {
        let row = PyDict::new(py);
        row.set_item("id", destination.id)?;
        row.set_item("target_sector_id", destination.target_sector_id)?;
        row.set_item("target_helicity_id", destination.target_helicity_id)?;
        destinations.append(row)?;
    }
    result.set_item("amplitude_destinations", destinations)?;
    result.set_item("amplitudes", native.amplitudes)?;
    let source_layout = PyList::empty(py);
    for source in native.source_layout {
        let row = PyDict::new(py);
        row.set_item("current_id", source.current_id)?;
        row.set_item("source_slot", source.source_slot)?;
        row.set_item("source_template_id", source.source_template_id)?;
        row.set_item("component_start", source.component_start)?;
        row.set_item("component_count", source.component_count)?;
        source_layout.append(row)?;
    }
    result.set_item("source_layout", source_layout)?;
    let backend = PyDict::new(py);
    backend.set_item("manifest_sha256", native.backend.manifest_sha256)?;
    backend.set_item("backend", native.backend.backend)?;
    backend.set_item("target_triple", native.backend.target_triple)?;
    backend.set_item("target_cpu_features", native.backend.target_cpu_features)?;
    backend.set_item("target_portable", native.backend.target_portable)?;
    backend.set_item("kernel_count", native.backend.kernel_count)?;
    result.set_item("prepared_backend", backend)?;
    Ok(result.into_any().unbind())
}

fn recurrence_norm_sqr_mapping(
    py: Python<'_>,
    native: NativeRecurrenceNormSqrEvaluation,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item(
        "kind",
        "pyamplicol-recurrence-private-helicity-sum-norm-sqr-v1",
    )?;
    result.set_item("sector_count", native.sector_count)?;
    result.set_item("norm_sqr", native.norm_sqr)?;
    let source_layout = PyList::empty(py);
    for source in native.source_layout {
        let row = PyDict::new(py);
        row.set_item("current_id", source.current_id)?;
        row.set_item("source_slot", source.source_slot)?;
        row.set_item("source_template_id", source.source_template_id)?;
        row.set_item("component_start", source.component_start)?;
        row.set_item("component_count", source.component_count)?;
        source_layout.append(row)?;
    }
    result.set_item("source_layout", source_layout)?;
    let backend = PyDict::new(py);
    backend.set_item("manifest_sha256", native.backend.manifest_sha256)?;
    backend.set_item("backend", native.backend.backend)?;
    backend.set_item("target_triple", native.backend.target_triple)?;
    backend.set_item("target_cpu_features", native.backend.target_cpu_features)?;
    backend.set_item("target_portable", native.backend.target_portable)?;
    backend.set_item("kernel_count", native.backend.kernel_count)?;
    result.set_item("prepared_backend", backend)?;
    Ok(result.into_any().unbind())
}

fn parse_input(input: &Bound<'_, PyAny>) -> PyResult<OwnedInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence builder input v1 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != RECURRENCE_BUILDER_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported recurrence builder input ABI {abi:?}"
        )));
    }
    let declared_digest = required_string(input, "digest")?;
    validate_sha256_text(&declared_digest, "recurrence builder input digest")?;

    let table_objects = iterable_attribute(input, "tables", "recurrence builder tables")?;
    if table_objects.len() != TABLE_SPECS.len() {
        return Err(PyValueError::new_err(format!(
            "recurrence builder table inventory has {} tables, expected {}",
            table_objects.len(),
            TABLE_SPECS.len()
        )));
    }

    let mut tables = Vec::with_capacity(TABLE_SPECS.len());
    let mut table_by_name = BTreeMap::new();
    for (table_object, spec) in table_objects.into_iter().zip(TABLE_SPECS) {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != spec.name {
            return Err(PyValueError::new_err(format!(
                "recurrence builder table inventory mismatch: found {table_name:?}, expected {:?}",
                spec.name
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence builder table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence builder table {table_name:?} columns"),
        )?;
        if column_objects.len() != spec.columns.len() {
            return Err(PyValueError::new_err(format!(
                "recurrence builder table {table_name:?} has {} columns, expected {}",
                column_objects.len(),
                spec.columns.len()
            )));
        }

        let mut columns = Vec::with_capacity(spec.columns.len());
        let mut column_by_name = BTreeMap::new();
        for (column_object, column_spec) in column_objects.into_iter().zip(spec.columns) {
            let column_name = required_nonempty_string(&column_object, "name", "column name")?;
            if column_name != column_spec.name {
                return Err(PyValueError::new_err(format!(
                    "recurrence builder table {table_name:?} column mismatch: found {column_name:?}, expected {:?}",
                    column_spec.name
                )));
            }
            let context = format!("{table_name}.{column_name}");
            let values_object = column_object.getattr("values")?;
            let dtype = values_object
                .getattr("dtype")?
                .getattr("str")?
                .extract::<String>()?;
            if dtype != column_spec.kind.dtype() {
                return Err(PyValueError::new_err(format!(
                    "{context} has dtype {dtype:?}, expected {:?}",
                    column_spec.kind.dtype()
                )));
            }
            let flags = values_object.getattr("flags")?;
            if flags.getattr("writeable")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must be read-only"
                )));
            }
            if !flags.getattr("owndata")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must own its storage"
                )));
            }
            let (values, shape) = extract_owned_values(&values_object, column_spec.kind, &context)?;
            if shape.first().copied() != Some(row_count) {
                return Err(PyValueError::new_err(format!(
                    "{context} first dimension does not match row_count {row_count}"
                )));
            }
            let actual_tail = shape
                .iter()
                .skip(1)
                .map(|value| usize::try_from(*value))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| PyValueError::new_err(format!("{context} shape exceeds usize")))?;
            if actual_tail != column_spec.tail_shape {
                return Err(PyValueError::new_err(format!(
                    "{context} has tail shape {actual_tail:?}, expected {:?}",
                    column_spec.tail_shape
                )));
            }
            column_by_name.insert(column_name.clone(), columns.len());
            columns.push(OwnedColumn {
                name: column_name,
                dtype: column_spec.kind.dtype(),
                shape,
                values,
            });
        }
        table_by_name.insert(table_name.clone(), tables.len());
        tables.push(OwnedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }

    Ok(OwnedInput {
        abi,
        declared_digest,
        tables,
        table_by_name,
    })
}

fn parse_prepared_template_input(input: &Bound<'_, PyAny>) -> PyResult<PreparedTemplateInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence template input v1 requires a little-endian target",
        ));
    }
    let abi = required_string(input, "abi")?;
    if abi != template::RECURRENCE_TEMPLATE_INPUT_ABI {
        return Err(PyValueError::new_err(format!(
            "unsupported recurrence template input ABI {abi:?}"
        )));
    }
    let declared_digest = required_string(input, "digest")?;
    validate_sha256_text(&declared_digest, "recurrence template input digest")?;
    let canonical_digest_property = required_string(input, "canonical_digest")?;
    validate_sha256_text(
        &canonical_digest_property,
        "recurrence template canonical digest",
    )?;
    let catalog_digest = semantic_digest_python_attribute(input, "catalog_digest")?;
    let compiled_model_digest = semantic_digest_python_attribute(input, "compiled_model_digest")?;
    let prepared_kernel_pack_digest =
        semantic_digest_python_attribute(input, "prepared_kernel_pack_digest")?;

    let table_objects = iterable_attribute(input, "tables", "recurrence template tables")?;
    if table_objects.len() != TEMPLATE_TABLE_INVENTORY.len() {
        return Err(PyValueError::new_err(format!(
            "recurrence template table inventory has {} tables, expected {}",
            table_objects.len(),
            TEMPLATE_TABLE_INVENTORY.len()
        )));
    }

    let mut tables = Vec::with_capacity(TEMPLATE_TABLE_INVENTORY.len());
    let mut table_by_name = BTreeMap::new();
    for (table_object, (expected_name, expected_column_count)) in
        table_objects.into_iter().zip(TEMPLATE_TABLE_INVENTORY)
    {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != *expected_name {
            return Err(PyValueError::new_err(format!(
                "recurrence template table inventory mismatch: found {table_name:?}, expected {expected_name:?}"
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence template table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence template table {table_name:?} columns"),
        )?;
        if column_objects.len() != *expected_column_count {
            return Err(PyValueError::new_err(format!(
                "recurrence template table {table_name:?} has {} columns, expected {expected_column_count}",
                column_objects.len()
            )));
        }

        let mut columns = Vec::with_capacity(*expected_column_count);
        let mut column_by_name = BTreeMap::new();
        for column_object in column_objects {
            let column_name = required_nonempty_string(&column_object, "name", "column name")?;
            if column_by_name.contains_key(&column_name) {
                return Err(PyValueError::new_err(format!(
                    "recurrence template table {table_name:?} repeats column {column_name:?}"
                )));
            }
            let context = format!("{table_name}.{column_name}");
            let values_object = column_object.getattr("values")?;
            let dtype = values_object
                .getattr("dtype")?
                .getattr("str")?
                .extract::<String>()?;
            let kind = PrimitiveKind::from_dtype(&dtype).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{context} has unsupported recurrence-template dtype {dtype:?}"
                ))
            })?;
            let flags = values_object.getattr("flags")?;
            if flags.getattr("writeable")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must be read-only"
                )));
            }
            if !flags.getattr("owndata")?.extract::<bool>()? {
                return Err(PyValueError::new_err(format!(
                    "{context} must own its storage"
                )));
            }
            let (values, shape) = extract_owned_values(&values_object, kind, &context)?;
            if shape.first().copied() != Some(row_count) {
                return Err(PyValueError::new_err(format!(
                    "{context} first dimension does not match row_count {row_count}"
                )));
            }
            column_by_name.insert(column_name.clone(), columns.len());
            columns.push(OwnedColumn {
                name: column_name,
                dtype: kind.dtype(),
                shape,
                values,
            });
        }
        table_by_name.insert(table_name.clone(), tables.len());
        tables.push(OwnedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }

    Ok(PreparedTemplateInput {
        input: OwnedInput {
            abi,
            declared_digest,
            tables,
            table_by_name,
        },
        canonical_digest_property,
        catalog_digest,
        compiled_model_digest,
        prepared_kernel_pack_digest,
    })
}

fn extract_owned_values(
    value: &Bound<'_, PyAny>,
    kind: PrimitiveKind,
    context: &str,
) -> PyResult<(OwnedValues, Vec<u64>)> {
    macro_rules! extract {
        ($ty:ty, $variant:ident) => {{
            let array = value
                .extract::<PyReadonlyArrayDyn<'_, $ty>>()
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "{context} is not a NumPy array with dtype {:?}",
                        kind.dtype()
                    ))
                })?;
            if !array.is_c_contiguous() {
                return Err(PyValueError::new_err(format!(
                    "{context} must be C-contiguous"
                )));
            }
            let shape = array
                .shape()
                .iter()
                .map(|value| {
                    u64::try_from(*value)
                        .map_err(|_| PyValueError::new_err(format!("{context} shape exceeds u64")))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let values = array
                .as_slice()
                .map_err(|error| {
                    PyValueError::new_err(format!("{context} must be contiguous: {error}"))
                })?
                .to_vec();
            (OwnedValues::$variant(values), shape)
        }};
    }
    Ok(match kind {
        PrimitiveKind::U8 => extract!(u8, U8),
        PrimitiveKind::U32 => extract!(u32, U32),
        PrimitiveKind::U64 => extract!(u64, U64),
        PrimitiveKind::I32 => extract!(i32, I32),
    })
}

fn validate_input(
    input: OwnedInput,
    prepared_template: Option<PreparedTemplateInput>,
    construct_schedule: bool,
) -> RusticolResult<NativeValidationResult> {
    validate_inventory(&input)?;
    let actual_digest = canonical_digest(&input)?;
    if actual_digest != input.declared_digest {
        return Err(invalid(format!(
            "recurrence builder input digest mismatch: declared {}, found {actual_digest}",
            input.declared_digest
        )));
    }

    let mut total_row_count = 0_u64;
    let mut primitive_element_count = 0_u64;
    let mut primitive_byte_count = 0_u64;
    let mut column_count = 0_usize;
    for table in &input.tables {
        total_row_count = total_row_count
            .checked_add(table.row_count)
            .ok_or_else(|| invalid("recurrence total row count exceeds u64"))?;
        column_count = column_count
            .checked_add(table.columns.len())
            .ok_or_else(|| invalid("recurrence column count exceeds usize"))?;
        for column in &table.columns {
            primitive_element_count = primitive_element_count
                .checked_add(u64::try_from(column.values.len()).map_err(|_| {
                    invalid(format!(
                        "{}.{} element count exceeds u64",
                        table.name, column.name
                    ))
                })?)
                .ok_or_else(|| invalid("recurrence primitive element count exceeds u64"))?;
            primitive_byte_count = primitive_byte_count
                .checked_add(u64::try_from(column.values.raw_bytes().len()).map_err(|_| {
                    invalid(format!(
                        "{}.{} byte count exceeds u64",
                        table.name, column.name
                    ))
                })?)
                .ok_or_else(|| invalid("recurrence primitive byte count exceeds u64"))?;
        }
    }

    let validated_process = decode_process_input(&input)?.validate()?;
    let process_summary = validated_process.summary().clone();
    let process_input = validated_process.input();
    let bitset_ranges = process_input
        .bitset_ranges
        .iter()
        .map(|row| row.range)
        .collect::<Vec<_>>();
    let maximum_mask_bit = maximum_mask_bit(&bitset_ranges, &process_input.bitset_words)?;
    let string_count = process_input.string_ranges.len();
    let semantic_digest_count = process_input.digest_catalog.len();
    let exact_factor_count = process_input.exact_factors.len();
    let open_string_count = process_input.lc_open_strings.len();
    let selector_mask_count = process_input.bitset_ranges.len();
    let selector_mask_word_count = process_input.bitset_words.len();
    let parameter_projection_count = process_input.parameter_projection.len();

    let (
        composite_authenticated,
        template_catalog_digest,
        compiled_model_digest,
        prepared_kernel_pack_digest,
        prepared_template_count,
        schedule_summary,
    ) = if let Some(prepared_template) = prepared_template {
        let validated_template = prepared_template.into_core()?.validate()?;
        let authenticated =
            AuthenticatedRecurrenceBuilderInput::new(validated_process, validated_template)?;
        let template_summary = authenticated.template().summary();
        let template_catalog_digest = template_summary.catalog_digest;
        let compiled_model_digest = template_summary.compiled_model_digest;
        let prepared_kernel_pack_digest = template_summary.prepared_kernel_pack_digest;
        let prepared_template_count = template_summary.parameter_count as usize
            + template_summary.current_state_count as usize
            + template_summary.source_count as usize
            + template_summary.quantum_flow_count as usize
            + template_summary.transition_count as usize
            + template_summary.propagator_count as usize
            + template_summary.closure_count as usize
            + template_summary.color_contraction_count as usize
            + template_summary.symmetry_proof_count as usize;
        let schedule_summary = if construct_schedule {
            let program = authenticated.build()?;
            let mut source_component_start = 0usize;
            let mut source_layout = Vec::new();
            for current in program.currents().iter().filter(|row| row.is_source()) {
                let CurrentSourceBinding::FixedTemplate(source_template_id) =
                    current.key().source_binding()
                else {
                    return Err(invalid(
                        "topology-replay source lacks a fixed source template",
                    ));
                };
                let component_count = authenticated
                    .template()
                    .input()
                    .current_states
                    .get(current.key().current_state_template_id() as usize)
                    .ok_or_else(|| invalid("source current state is absent"))?
                    .dimension as usize;
                source_layout.push(NativeSourceLayout {
                    current_id: current.id(),
                    source_slot: current.key().support_source_slots()[0],
                    source_template_id,
                    component_start: source_component_start,
                    component_count,
                });
                source_component_start = source_component_start
                    .checked_add(component_count)
                    .ok_or_else(|| invalid("source component layout overflows"))?;
            }
            let mut current_count_by_support_size = Vec::<usize>::new();
            for current in program.currents() {
                let support_size = current.key().support_source_slots().len();
                if current_count_by_support_size.len() <= support_size {
                    current_count_by_support_size.resize(support_size + 1, 0);
                }
                current_count_by_support_size[support_size] += 1;
            }
            let mut referenced_quantum_flow_template_ids = program
                .contributions()
                .iter()
                .map(|row| row.key().quantum_flow_witness_id())
                .collect::<BTreeSet<_>>();
            let mut contribution_count_by_transition_template_id = BTreeMap::<u32, usize>::new();
            let mut contribution_count_by_quantum_flow_template_id = BTreeMap::<u32, usize>::new();
            for contribution in program.contributions() {
                *contribution_count_by_transition_template_id
                    .entry(contribution.key().transition_template_id())
                    .or_default() += 1;
                *contribution_count_by_quantum_flow_template_id
                    .entry(contribution.key().quantum_flow_witness_id())
                    .or_default() += 1;
            }
            referenced_quantum_flow_template_ids.extend(
                program
                    .closure_terms()
                    .iter()
                    .filter_map(|row| row.quantum_flow_template_id()),
            );
            let mut identity_finalization_count_by_support_size =
                vec![0usize; current_count_by_support_size.len()];
            let mut propagated_finalization_count_by_support_size =
                vec![0usize; current_count_by_support_size.len()];
            for finalization in program.finalizations() {
                let support_size = program.currents()[finalization.current_id() as usize]
                    .key()
                    .support_source_slots()
                    .len();
                if finalization.propagator_template_id().is_some() {
                    propagated_finalization_count_by_support_size[support_size] += 1;
                } else {
                    identity_finalization_count_by_support_size[support_size] += 1;
                }
            }
            Some(NativeScheduleSummary {
                dynamic_color_state_count: program.dynamic_color_states().len(),
                current_count: program.currents().len(),
                source_current_count: program
                    .currents()
                    .iter()
                    .filter(|current| current.is_source())
                    .count(),
                current_count_by_support_size,
                contribution_count: program.contributions().len(),
                contribution_count_by_transition_template_id:
                    contribution_count_by_transition_template_id
                        .into_iter()
                        .collect(),
                contribution_count_by_quantum_flow_template_id:
                    contribution_count_by_quantum_flow_template_id
                        .into_iter()
                        .collect(),
                referenced_quantum_flow_template_ids: referenced_quantum_flow_template_ids
                    .into_iter()
                    .collect(),
                finalization_count: program.finalizations().len(),
                identity_finalization_count_by_support_size,
                propagated_finalization_count_by_support_size,
                retained_helicity_count: program.retained_helicity_count(),
                resolved_helicity_count: program.resolved_helicities().len(),
                structural_zero_helicity_count: program.retained_helicity_count()
                    - program.resolved_helicities().len() as u64,
                amplitude_destination_count: program.amplitude_destinations().len(),
                target_sector_count: program.physical_sector_count() as usize,
                closure_term_count: program.closure_terms().len(),
                source_layout,
            })
        } else {
            None
        };
        (
            true,
            Some(template_catalog_digest),
            Some(compiled_model_digest),
            Some(prepared_kernel_pack_digest),
            Some(prepared_template_count),
            schedule_summary,
        )
    } else {
        (false, None, None, None, None, None)
    };

    Ok(NativeValidationResult {
        digest: actual_digest,
        process_id: process_summary.process_id().to_owned(),
        strategy: process_summary.strategy(),
        table_count: input.tables.len(),
        column_count,
        total_row_count,
        primitive_element_count,
        primitive_byte_count,
        string_count,
        semantic_digest_count,
        exact_factor_count,
        external_leg_count: process_summary.external_leg_count() as usize,
        source_state_count: process_summary.source_state_count() as usize,
        physical_sector_count: process_summary.physical_sector_count() as usize,
        public_flow_count: process_summary.public_flow_count() as usize,
        open_string_count,
        replay_partition_count: process_summary.replay_partition_count() as usize,
        replay_target_count: process_summary.replay_target_count() as usize,
        selector_mask_count,
        selector_mask_word_count,
        maximum_mask_bit,
        template_reference_count: process_summary.template_reference_count() as usize,
        parameter_projection_count,
        composite_authenticated,
        template_catalog_digest,
        compiled_model_digest,
        prepared_kernel_pack_digest,
        prepared_template_count,
        schedule_summary,
    })
}

fn decode_process_input(
    input: &OwnedInput,
) -> RusticolResult<process::OwnedRecurrenceProcessInput> {
    let bitset_ranges = decode_process_rows(input, "bitset_ranges", |row| {
        Ok(process::ProcessBitsetRangeRow {
            id: input.u32("bitset_ranges", "id")?[row],
            range: CheckedTableRange {
                start: input.u64("bitset_ranges", "start")?[row],
                count: input.u64("bitset_ranges", "count")?[row],
            },
            bit_count: input.u64("bitset_ranges", "bit_count")?[row],
        })
    })?;
    let coupling_limits = decode_process_rows(input, "coupling_limits", |row| {
        Ok(process::ProcessCouplingLimitRow {
            name_string_id: input.u32("coupling_limits", "name_string_id")?[row],
            minimum: input.u32("coupling_limits", "minimum")?[row],
            maximum: input.u32("coupling_limits", "maximum")?[row],
        })
    })?;
    let digest_values = input.u8("digest_catalog", "value")?;
    let digest_catalog = decode_process_rows(input, "digest_catalog", |row| {
        let start = row
            .checked_mul(32)
            .ok_or_else(|| invalid("process digest catalog offset exceeds usize"))?;
        let value = digest_values
            .get(start..start + 32)
            .ok_or_else(|| invalid("process digest catalog row is truncated"))?
            .try_into()
            .map_err(|_| invalid("process digest catalog row must contain 32 bytes"))?;
        Ok(process::ProcessDigestCatalogRow {
            id: input.u32("digest_catalog", "id")?[row],
            value,
        })
    })?;
    let exact_factors = decode_process_rows(input, "exact_factors", |row| {
        Ok(process::ProcessExactFactorRow {
            id: input.u32("exact_factors", "id")?[row],
            real_numerator_string_id: input.u32("exact_factors", "real_numerator_string_id")?[row],
            real_denominator_string_id: input.u32("exact_factors", "real_denominator_string_id")?
                [row],
            imag_numerator_string_id: input.u32("exact_factors", "imag_numerator_string_id")?[row],
            imag_denominator_string_id: input.u32("exact_factors", "imag_denominator_string_id")?
                [row],
        })
    })?;
    let external_legs = decode_process_rows(input, "external_legs", |row| {
        Ok(process::ProcessExternalLegRow {
            source_slot: input.u32("external_legs", "source_slot")?[row],
            public_label: input.u32("external_legs", "public_label")?[row],
            physical_pdg: input.i32("external_legs", "physical_pdg")?[row],
            outgoing_pdg: input.i32("external_legs", "outgoing_pdg")?[row],
            is_initial: input.u8("external_legs", "is_initial")?[row],
            is_fermionic: input.u8("external_legs", "is_fermionic")?[row],
            source_state_range: CheckedTableRange {
                start: input.u64("external_legs", "source_state_start")?[row],
                count: input.u64("external_legs", "source_state_count")?[row],
            },
            momentum_mask_id: input.u32("external_legs", "momentum_mask_id")?[row],
            support_mask_id: input.u32("external_legs", "support_mask_id")?[row],
        })
    })?;
    let header = decode_process_rows(input, "header", |row| {
        Ok(process::ProcessHeaderRow {
            schema_version: input.u32("header", "schema_version")?[row],
            abi_string_id: input.u32("header", "abi_string_id")?[row],
            process_id_string_id: input.u32("header", "process_id_string_id")?[row],
            layout: input.u8("header", "layout")?[row],
            selected_flow_mode: input.u8("header", "selected_flow_mode")?[row],
            selected_source_mode: input.u8("header", "selected_source_mode")?[row],
            external_leg_count: input.u32("header", "external_leg_count")?[row],
            physical_sector_count: input.u32("header", "physical_sector_count")?[row],
            public_flow_count: input.u32("header", "public_flow_count")?[row],
            replay_partition_count: input.u32("header", "replay_partition_count")?[row],
            coupling_limit_count: input.u32("header", "coupling_limit_count")?[row],
            parameter_projection_count: input.u32("header", "parameter_projection_count")?[row],
            process_support_mask_id: input.u32("header", "process_support_mask_id")?[row],
        })
    })?;
    let header_digests = decode_process_rows(input, "header_digests", |row| {
        Ok(process::ProcessHeaderDigestRow {
            role_string_id: input.u32("header_digests", "role_string_id")?[row],
            digest_id: input.u32("header_digests", "digest_id")?[row],
        })
    })?;
    let lc_open_strings = decode_process_rows(input, "lc_open_strings", |row| {
        Ok(process::ProcessLCOpenStringRow {
            sector_id: input.u32("lc_open_strings", "sector_id")?[row],
            ordinal: input.u32("lc_open_strings", "ordinal")?[row],
            fundamental_source_slot: input.u32("lc_open_strings", "fundamental_source_slot")?[row],
            antifundamental_source_slot: input
                .u32("lc_open_strings", "antifundamental_source_slot")?[row],
            adjoint_sequence_id: input.u32("lc_open_strings", "adjoint_sequence_id")?[row],
            singlet_sequence_id: input.u32("lc_open_strings", "singlet_sequence_id")?[row],
        })
    })?;
    let normalization = decode_process_rows(input, "normalization", |row| {
        Ok(process::ProcessNormalizationRow {
            factor_id: input.u32("normalization", "factor_id")?[row],
            convention_string_id: input.u32("normalization", "convention_string_id")?[row],
            semantic_digest_id: input.u32("normalization", "semantic_digest_id")?[row],
        })
    })?;
    let parameter_projection = decode_process_rows(input, "parameter_projection", |row| {
        Ok(process::ProcessParameterProjectionRow {
            runtime_slot: input.u32("parameter_projection", "runtime_slot")?[row],
            runtime_name_string_id: input.u32("parameter_projection", "runtime_name_string_id")?
                [row],
            parameter_template_id: input.u32("parameter_projection", "parameter_template_id")?[row],
            prepared_parameter_id: input.u32("parameter_projection", "prepared_parameter_id")?[row],
            component: input.u32("parameter_projection", "component")?[row],
        })
    })?;
    let physical_lc_sectors = decode_process_rows(input, "physical_lc_sectors", |row| {
        Ok(process::ProcessPhysicalLCSectorRow {
            sector_id: input.u32("physical_lc_sectors", "sector_id")?[row],
            public_id_string_id: input.u32("physical_lc_sectors", "public_id_string_id")?[row],
            kind: input.u8("physical_lc_sectors", "kind")?[row],
            closure_source_slot: input.u32("physical_lc_sectors", "closure_source_slot")?[row],
            closure_proof_algorithm_string_id: input
                .u32("physical_lc_sectors", "closure_proof_algorithm_string_id")?[row],
            closure_proof_digest_id: input.u32("physical_lc_sectors", "closure_proof_digest_id")?
                [row],
            open_string_range: CheckedTableRange {
                start: input.u64("physical_lc_sectors", "open_string_start")?[row],
                count: input.u64("physical_lc_sectors", "open_string_count")?[row],
            },
            trace_sequence_id: input.u32("physical_lc_sectors", "trace_sequence_id")?[row],
            singlet_sequence_id: input.u32("physical_lc_sectors", "singlet_sequence_id")?[row],
            word_sequence_id: input.u32("physical_lc_sectors", "word_sequence_id")?[row],
            support_mask_id: input.u32("physical_lc_sectors", "support_mask_id")?[row],
        })
    })?;
    let public_lc_flows = decode_process_rows(input, "public_lc_flows", |row| {
        Ok(process::ProcessPublicLCFlowRow {
            flow_id: input.u32("public_lc_flows", "flow_id")?[row],
            public_id_string_id: input.u32("public_lc_flows", "public_id_string_id")?[row],
            construction_sector_id: input.u32("public_lc_flows", "construction_sector_id")?[row],
            word_sequence_id: input.u32("public_lc_flows", "word_sequence_id")?[row],
            source_slot_permutation_sequence_id: input
                .u32("public_lc_flows", "source_slot_permutation_sequence_id")?[row],
            reduction_weight_factor_id: input
                .u32("public_lc_flows", "reduction_weight_factor_id")?[row],
        })
    })?;
    let replay_partitions = decode_process_rows(input, "replay_partitions", |row| {
        Ok(process::ProcessReplayPartitionRow {
            partition_id: input.u32("replay_partitions", "partition_id")?[row],
            representative_sector_id: input.u32("replay_partitions", "representative_sector_id")?
                [row],
            materialized_sector_id: input.u32("replay_partitions", "materialized_sector_id")?[row],
            target_range: CheckedTableRange {
                start: input.u64("replay_partitions", "target_start")?[row],
                count: input.u64("replay_partitions", "target_count")?[row],
            },
            proof_algorithm_string_id: input
                .u32("replay_partitions", "proof_algorithm_string_id")?[row],
            proof_digest_id: input.u32("replay_partitions", "proof_digest_id")?[row],
        })
    })?;
    let replay_targets = decode_process_rows(input, "replay_targets", |row| {
        Ok(process::ProcessReplayTargetRow {
            partition_id: input.u32("replay_targets", "partition_id")?[row],
            sector_id: input.u32("replay_targets", "sector_id")?[row],
            external_permutation_sequence_id: input
                .u32("replay_targets", "external_permutation_sequence_id")?[row],
            source_slot_permutation_sequence_id: input
                .u32("replay_targets", "source_slot_permutation_sequence_id")?[row],
            amplitude_phase_factor_id: input.u32("replay_targets", "amplitude_phase_factor_id")?
                [row],
            fermion_sign: input.i32("replay_targets", "fermion_sign")?[row],
        })
    })?;
    let selected_public_flow_coverage =
        decode_process_rows(input, "selected_public_flow_coverage", |row| {
            Ok(process::ProcessSelectedPublicFlowRow {
                flow_id: input.u32("selected_public_flow_coverage", "flow_id")?[row],
            })
        })?;
    let selected_source_coverage = decode_process_rows(input, "selected_source_coverage", |row| {
        Ok(process::ProcessSelectedSourceStateRow {
            source_slot: input.u32("selected_source_coverage", "source_slot")?[row],
            source_state_index: input.u32("selected_source_coverage", "source_state_index")?[row],
        })
    })?;
    let semantic_template_references =
        decode_process_rows(input, "semantic_template_references", |row| {
            Ok(process::ProcessSemanticTemplateReferenceRow {
                kind_string_id: input.u32("semantic_template_references", "kind_string_id")?[row],
                template_id: input.u32("semantic_template_references", "template_id")?[row],
                semantic_digest_id: input
                    .u32("semantic_template_references", "semantic_digest_id")?[row],
                prepared_kernel_id: input
                    .u32("semantic_template_references", "prepared_kernel_id")?[row],
            })
        })?;
    let source_states = decode_process_rows(input, "source_states", |row| {
        Ok(process::ProcessSourceStateRow {
            source_slot: input.u32("source_states", "source_slot")?[row],
            state_index: input.u32("source_states", "state_index")?[row],
            public_helicity: input.i32("source_states", "public_helicity")?[row],
            chirality: input.i32("source_states", "chirality")?[row],
            spin_state: input.i32("source_states", "spin_state")?[row],
            current_state_template_id: input.u32("source_states", "current_state_template_id")?
                [row],
            source_template_id: input.u32("source_states", "source_template_id")?[row],
            momentum_sign: input.i32("source_states", "momentum_sign")?[row],
            crossing_phase_factor_id: input.u32("source_states", "crossing_phase_factor_id")?[row],
        })
    })?;

    Ok(process::OwnedRecurrenceProcessInput {
        input_abi: input.abi.clone(),
        declared_input_digest: semantic_digest_from_hex(
            &input.declared_digest,
            "recurrence process input digest",
        )?,
        bitset_ranges,
        bitset_words: input.u64("bitset_words", "value")?.to_vec(),
        coupling_limits,
        digest_catalog,
        exact_factors,
        external_legs,
        header,
        header_digests,
        lc_open_strings,
        normalization,
        parameter_projection,
        physical_lc_sectors,
        public_lc_flows,
        replay_partitions,
        replay_targets,
        selected_public_flow_coverage,
        selected_source_coverage,
        semantic_template_references,
        source_states,
        string_ranges: plain_ranges(input, "string_ranges")?,
        string_bytes: input.u8("string_bytes", "value")?.to_vec(),
        u32_sequence_ranges: plain_ranges(input, "u32_sequence_ranges")?,
        u32_sequence_values: input.u32("u32_sequence_values", "value")?.to_vec(),
    })
}

fn decode_process_rows<T>(
    input: &OwnedInput,
    table_name: &str,
    mut decode: impl FnMut(usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = checked_usize(
        input.table(table_name)?.row_count,
        &format!("{table_name} row count"),
    )?;
    (0..row_count).map(&mut decode).collect()
}

fn plain_ranges(input: &OwnedInput, table_name: &str) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_process_rows(input, table_name, |row| {
        Ok(CheckedTableRange {
            start: input.u64(table_name, "start")?[row],
            count: input.u64(table_name, "count")?[row],
        })
    })
}

fn decode_template_input(
    input: &OwnedInput,
    catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
) -> RusticolResult<template::OwnedRecurrenceTemplateInput> {
    use template::{
        CatalogHeaderRow, ClosureRow, ColorContractionRow, ColorNcTermRow, CouplingOrderTermRow,
        CurrentStateRow, EvaluatorBindingRow, ExactFactorRow, LCColorTransitionWitnessRow,
        OwnedRecurrenceTemplateInput, ParameterRow, PropagatorRow, QuantumFlowRow,
        QuantumNumberFlowTermRow, RuntimeHelicityContractRow, RuntimeHelicityEmbeddingRow,
        RuntimeHelicityProjectionRow, RuntimeHelicityVariantRow, SourceRow, SymmetryProofRow,
        TransitionRow,
    };
    let catalog_header = decode_template_rows(input.table("catalog_header")?, |table, row| {
        Ok(CatalogHeaderRow {
            schema_version: table.u32("schema_version")?[row],
            abi_string_id: table.u32("abi_string_id")?[row],
            canonicalization_abi_string_id: table.u32("canonicalization_abi_string_id")?[row],
            exact_scalar_abi_string_id: table.u32("exact_scalar_abi_string_id")?[row],
            compiled_model_digest_id: table.u32("compiled_model_digest_id")?[row],
            prepared_kernel_pack_digest_id: table.u32("prepared_kernel_pack_digest_id")?[row],
            catalog_digest_id: table.u32("catalog_digest_id")?[row],
            parameter_count: table.u32("parameter_count")?[row],
            current_state_count: table.u32("current_state_count")?[row],
            source_count: table.u32("source_count")?[row],
            quantum_flow_count: table.u32("quantum_flow_count")?[row],
            transition_count: table.u32("transition_count")?[row],
            propagator_count: table.u32("propagator_count")?[row],
            closure_count: table.u32("closure_count")?[row],
            color_contraction_count: table.u32("color_contraction_count")?[row],
            symmetry_proof_count: table.u32("symmetry_proof_count")?[row],
            runtime_helicity_contract_count: table.u32("runtime_helicity_contract_count")?[row],
            evaluator_binding_count: table.u32("evaluator_binding_count")?[row],
        })
    })?;
    let coupling_order_ranges = template_indexed_ranges(input.table("coupling_order_ranges")?)?;
    let coupling_order_terms =
        decode_template_rows(input.table("coupling_order_terms")?, |table, row| {
            Ok(CouplingOrderTermRow {
                set_id: table.u32("set_id")?[row],
                name_string_id: table.u32("name_string_id")?[row],
                power: table.u32("power")?[row],
            })
        })?;
    let current_states = decode_template_rows(input.table("current_states")?, |table, row| {
        Ok(CurrentStateRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            particle_id: table.i32("particle_id")?[row],
            anti_particle_id: table.i32("anti_particle_id")?[row],
            species_string_id: table.u32("species_string_id")?[row],
            orientation: table.u8("orientation")?[row],
            statistics: table.u8("statistics")?[row],
            color_representation: table.i32("color_representation")?[row],
            basis_string_id: table.u32("basis_string_id")?[row],
            tensor_ordering_sequence_id: table.u32("tensor_ordering_sequence_id")?[row],
            dimension: table.u32("dimension")?[row],
            chirality: table.i32("chirality")?[row],
            lc_color_shape_string_id: table.u32("lc_color_shape_string_id")?[row],
            auxiliary_kind_string_id: table.u32("auxiliary_kind_string_id")?[row],
            mass_parameter_id: table.u32("mass_parameter_id")?[row],
            width_parameter_id: table.u32("width_parameter_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let digest_catalog = decode_template_digest_catalog(input.table("digest_catalog")?)?;
    let evaluator_bindings =
        decode_template_rows(input.table("evaluator_bindings")?, |table, row| {
            Ok(EvaluatorBindingRow {
                id: table.u32("id")?[row],
                resolver_key_string_id: table.u32("resolver_key_string_id")?[row],
                prepared_kernel_id: table.u32("prepared_kernel_id")?[row],
                contract_kind: table.u8("contract_kind")?[row],
                callable_signature_digest_id: table.u32("callable_signature_digest_id")?[row],
                input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
                output_state_template_id: table.u32("output_state_template_id")?[row],
                input_layout_sequence_id: table.u32("input_layout_sequence_id")?[row],
                output_layout_sequence_id: table.u32("output_layout_sequence_id")?[row],
                exact_expression_digest_sequence_id: table
                    .u32("exact_expression_digest_sequence_id")?[row],
                semantic_template_sequence_id: table.u32("semantic_template_sequence_id")?[row],
                callable_kind: table.u8("callable_kind")?[row],
                runtime_template_string_id: table.u32("runtime_template_string_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
    let exact_factors = decode_template_rows(input.table("exact_factors")?, |table, row| {
        Ok(ExactFactorRow {
            id: table.u32("id")?[row],
            real_numerator_string_id: table.u32("real_numerator_string_id")?[row],
            real_denominator_string_id: table.u32("real_denominator_string_id")?[row],
            imag_numerator_string_id: table.u32("imag_numerator_string_id")?[row],
            imag_denominator_string_id: table.u32("imag_denominator_string_id")?[row],
        })
    })?;
    let flavour_flow_ranges = template_indexed_ranges(input.table("flavour_flow_ranges")?)?;
    let flavour_flow_values = input.table("flavour_flow_values")?.i32("value")?.to_vec();
    let i32_sequence_ranges = template_indexed_ranges(input.table("i32_sequence_ranges")?)?;
    let i32_sequence_values = input.table("i32_sequence_values")?.i32("value")?.to_vec();
    let parameters = decode_template_rows(input.table("parameters")?, |table, row| {
        Ok(ParameterRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            name_string_id: table.u32("name_string_id")?[row],
            kind: table.u8("kind")?[row],
            value_type: table.u8("value_type")?[row],
            mutable: table.u8("mutable")?[row],
            default_factor_id: table.u32("default_factor_id")?[row],
            exact_expression_digest_id: table.u32("exact_expression_digest_id")?[row],
            dependency_sequence_id: table.u32("dependency_sequence_id")?[row],
            prepared_parameter_id: table.u32("prepared_parameter_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let propagators = decode_template_rows(input.table("propagators")?, |table, row| {
        Ok(PropagatorRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            state_template_id: table.u32("state_template_id")?[row],
            applies_propagator: table.u8("applies_propagator")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            numerator_expression_digest_id: table.u32("numerator_expression_digest_id")?[row],
            denominator_expression_digest_id: table.u32("denominator_expression_digest_id")?[row],
            mass_parameter_id: table.u32("mass_parameter_id")?[row],
            width_parameter_id: table.u32("width_parameter_id")?[row],
            gauge_string_id: table.u32("gauge_string_id")?[row],
            linearity_proof_template_id: table.u32("linearity_proof_template_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let quantum_flows = decode_template_rows(input.table("quantum_flows")?, |table, row| {
        Ok(QuantumFlowRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            input_spin_sequence_id: table.u32("input_spin_sequence_id")?[row],
            input_flavour_sequence_id: table.u32("input_flavour_sequence_id")?[row],
            input_quantum_sequence_id: table.u32("input_quantum_sequence_id")?[row],
            flavour_flow_operation_string_id: table.u32("flavour_flow_operation_string_id")?[row],
            quantum_number_flow_operation_string_id: table
                .u32("quantum_number_flow_operation_string_id")?[row],
            coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
            result_state_template_id: table.u32("result_state_template_id")?[row],
            result_spin_state: table.i32("result_spin_state")?[row],
            result_flavour_flow_id: table.u32("result_flavour_flow_id")?[row],
            result_quantum_number_flow_id: table.u32("result_quantum_number_flow_id")?[row],
            exact_coupling_factor_id: table.u32("exact_coupling_factor_id")?[row],
            predicate_digest_id: table.u32("predicate_digest_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let quantum_number_flow_ranges =
        template_indexed_ranges(input.table("quantum_number_flow_ranges")?)?;
    let quantum_number_flow_terms =
        decode_template_rows(input.table("quantum_number_flow_terms")?, |table, row| {
            Ok(QuantumNumberFlowTermRow {
                flow_id: table.u32("flow_id")?[row],
                name_string_id: table.u32("name_string_id")?[row],
                expression_string_id: table.u32("expression_string_id")?[row],
            })
        })?;
    let runtime_helicity_contracts =
        decode_template_rows(input.table("runtime_helicity_contracts")?, |table, row| {
            Ok(RuntimeHelicityContractRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                full_state_template_id: table.u32("full_state_template_id")?[row],
                variant_range: CheckedTableRange::new(
                    table.u32("variant_offset")?[row].into(),
                    table.u32("variant_count")?[row].into(),
                ),
                proof_algorithm_string_id: table.u32("proof_algorithm_string_id")?[row],
                proof_digest_id: table.u32("proof_digest_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
    let runtime_helicity_embeddings =
        decode_template_rows(input.table("runtime_helicity_embeddings")?, |table, row| {
            Ok(RuntimeHelicityEmbeddingRow {
                variant_id: table.u32("variant_id")?[row],
                full_component: table.u32("full_component")?[row],
                source_component: table.u32("source_component")?[row],
                factor_id: table.u32("factor_id")?[row],
            })
        })?;
    let runtime_helicity_projections = decode_template_rows(
        input.table("runtime_helicity_projections")?,
        |table, row| {
            Ok(RuntimeHelicityProjectionRow {
                variant_id: table.u32("variant_id")?[row],
                source_component: table.u32("source_component")?[row],
                full_component: table.u32("full_component")?[row],
            })
        },
    )?;
    let runtime_helicity_variants =
        decode_template_rows(input.table("runtime_helicity_variants")?, |table, row| {
            Ok(RuntimeHelicityVariantRow {
                id: table.u32("id")?[row],
                contract_id: table.u32("contract_id")?[row],
                source_template_id: table.u32("source_template_id")?[row],
                source_state_template_id: table.u32("source_state_template_id")?[row],
                embedding_range: CheckedTableRange::new(
                    table.u32("embedding_offset")?[row].into(),
                    table.u32("embedding_count")?[row].into(),
                ),
                projection_range: CheckedTableRange::new(
                    table.u32("projection_offset")?[row].into(),
                    table.u32("projection_count")?[row].into(),
                ),
                proof_digest_id: table.u32("proof_digest_id")?[row],
            })
        })?;
    let sources = decode_template_rows(input.table("sources")?, |table, row| {
        Ok(SourceRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            state_template_id: table.u32("state_template_id")?[row],
            crossing_string_id: table.u32("crossing_string_id")?[row],
            wavefunction_family_string_id: table.u32("wavefunction_family_string_id")?[row],
            helicity: table.i32("helicity")?[row],
            spin_state: table.i32("spin_state")?[row],
            flavour_flow_id: table.u32("flavour_flow_id")?[row],
            quantum_number_flow_id: table.u32("quantum_number_flow_id")?[row],
            lc_color_seed_operation: table.u8("lc_color_seed_operation")?[row],
            lc_color_seed_shape_string_id: table.u32("lc_color_seed_shape_string_id")?[row],
            lc_color_seed_component_kind: table.u8("lc_color_seed_component_kind")?[row],
            lc_color_seed_component_role: table.u8("lc_color_seed_component_role")?[row],
            lc_color_seed_proof_digest_id: table.u32("lc_color_seed_proof_digest_id")?[row],
            lc_color_seed_provenance_sequence_id: table
                .u32("lc_color_seed_provenance_sequence_id")?[row],
            wavefunction_expression_digest_id: table.u32("wavefunction_expression_digest_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            mass_parameter_id: table.u32("mass_parameter_id")?[row],
            width_parameter_id: table.u32("width_parameter_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let string_ranges = template_plain_ranges(input.table("string_ranges")?)?;
    let string_bytes = input.table("string_bytes")?.u8("value")?.to_vec();
    let symmetry_proofs = decode_template_rows(input.table("symmetry_proofs")?, |table, row| {
        Ok(SymmetryProofRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            proof_algorithm_string_id: table.u32("proof_algorithm_string_id")?[row],
            subject_template_sequence_id: table.u32("subject_template_sequence_id")?[row],
            input_permutation_sequence_id: table.u32("input_permutation_sequence_id")?[row],
            exact_phase_factor_id: table.u32("exact_phase_factor_id")?[row],
            expression_digest_sequence_id: table.u32("expression_digest_sequence_id")?[row],
            witness_digest_id: table.u32("witness_digest_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let transitions = decode_template_rows(input.table("transitions")?, |table, row| {
        Ok(TransitionRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            result_state_template_id: table.u32("result_state_template_id")?[row],
            quantum_flow_template_id: table.u32("quantum_flow_template_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            canonical_input_order_sequence_id: table.u32("canonical_input_order_sequence_id")?[row],
            momentum_convention_sequence_id: table.u32("momentum_convention_sequence_id")?[row],
            coupling_parameter_sequence_id: table.u32("coupling_parameter_sequence_id")?[row],
            coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
            color_contraction_template_id: table.u32("color_contraction_template_id")?[row],
            binding_coupling_factor_id: table.u32("binding_coupling_factor_id")?[row],
            exact_factor_id: table.u32("exact_factor_id")?[row],
            output_factor_source: table.u8("output_factor_source")?[row],
            equivalence_class_string_id: table.u32("equivalence_class_string_id")?[row],
            input_exchange_factor_id: table.u32("input_exchange_factor_id")?[row],
            output_projection_string_id: table.u32("output_projection_string_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let closures = decode_template_rows(input.table("closures")?, |table, row| {
        Ok(ClosureRow {
            id: table.u32("id")?[row],
            template_string_id: table.u32("template_string_id")?[row],
            input_state_sequence_id: table.u32("input_state_sequence_id")?[row],
            result_state_template_id: table.u32("result_state_template_id")?[row],
            evaluator_binding_id: table.u32("evaluator_binding_id")?[row],
            canonical_input_order_sequence_id: table.u32("canonical_input_order_sequence_id")?[row],
            coupling_parameter_sequence_id: table.u32("coupling_parameter_sequence_id")?[row],
            coupling_order_set_id: table.u32("coupling_order_set_id")?[row],
            eligible_quantum_flow_sequence_id: table.u32("eligible_quantum_flow_sequence_id")?[row],
            color_contraction_template_id: table.u32("color_contraction_template_id")?[row],
            binding_coupling_factor_id: table.u32("binding_coupling_factor_id")?[row],
            exact_factor_id: table.u32("exact_factor_id")?[row],
            output_factor_source: table.u8("output_factor_source")?[row],
            equivalence_class_string_id: table.u32("equivalence_class_string_id")?[row],
            input_exchange_factor_id: table.u32("input_exchange_factor_id")?[row],
            projection_string_id: table.u32("projection_string_id")?[row],
            component_coefficient_sequence_id: table.u32("component_coefficient_sequence_id")?[row],
            chirality_relation_string_id: table.u32("chirality_relation_string_id")?[row],
            metric_signature_string_id: table.u32("metric_signature_string_id")?[row],
            semantic_digest_id: table.u32("semantic_digest_id")?[row],
        })
    })?;
    let color_contractions =
        decode_template_rows(input.table("color_contractions")?, |table, row| {
            Ok(ColorContractionRow {
                id: table.u32("id")?[row],
                template_string_id: table.u32("template_string_id")?[row],
                rule_kind_string_id: table.u32("rule_kind_string_id")?[row],
                input_representation_sequence_id: table.u32("input_representation_sequence_id")?
                    [row],
                has_output_representation: table.u8("has_output_representation")?[row],
                output_representation: table.i32("output_representation")?[row],
                ordered_open_string_arity: table.u32("ordered_open_string_arity")?[row],
                exact_coefficient_factor_id: table.u32("exact_coefficient_factor_id")?[row],
                witness_start: table.u64("witness_start")?[row],
                witness_count: table.u64("witness_count")?[row],
                nc_term_start: table.u64("nc_term_start")?[row],
                nc_term_count: table.u64("nc_term_count")?[row],
                expression_digest_id: table.u32("expression_digest_id")?[row],
                semantic_digest_id: table.u32("semantic_digest_id")?[row],
            })
        })?;
    let lc_color_transition_witnesses = decode_template_rows(
        input.table("lc_color_transition_witnesses")?,
        |table, row| {
            Ok(LCColorTransitionWitnessRow {
                color_contraction_id: table.u32("color_contraction_id")?[row],
                ordinal: table.u32("ordinal")?[row],
                left_shape_string_id: table.u32("left_shape_string_id")?[row],
                right_shape_string_id: table.u32("right_shape_string_id")?[row],
                input_permutation: table.u8("input_permutation")?[row],
                reverse_parent_mask: table.u8("reverse_parent_mask")?[row],
                component_operation: table.u8("component_operation")?[row],
                result_component_kind: table.u8("result_component_kind")?[row],
                result_component_role: table.u8("result_component_role")?[row],
                result_shape_string_id: table.u32("result_shape_string_id")?[row],
                exact_factor_id: table.u32("exact_factor_id")?[row],
                proof_digest_id: table.u32("proof_digest_id")?[row],
                provenance_sequence_id: table.u32("provenance_sequence_id")?[row],
            })
        },
    )?;
    let color_nc_terms = decode_template_rows(input.table("color_nc_terms")?, |table, row| {
        Ok(ColorNcTermRow {
            color_contraction_id: table.u32("color_contraction_id")?[row],
            exponent: table.i32("exponent")?[row],
            factor_id: table.u32("factor_id")?[row],
        })
    })?;
    let u32_sequence_ranges = template_indexed_ranges(input.table("u32_sequence_ranges")?)?;
    let u32_sequence_values = input.table("u32_sequence_values")?.u32("value")?.to_vec();

    Ok(OwnedRecurrenceTemplateInput {
        input_abi: input.abi.clone(),
        catalog_digest: catalog_digest,
        compiled_model_digest: compiled_model_digest,
        prepared_kernel_pack_digest: prepared_kernel_pack_digest,
        catalog_header,
        coupling_order_ranges,
        coupling_order_terms,
        current_states,
        digest_catalog,
        evaluator_bindings,
        exact_factors,
        flavour_flow_ranges,
        flavour_flow_values,
        i32_sequence_ranges,
        i32_sequence_values,
        parameters,
        propagators,
        quantum_flows,
        quantum_number_flow_ranges,
        quantum_number_flow_terms,
        runtime_helicity_contracts,
        runtime_helicity_variants,
        runtime_helicity_embeddings,
        runtime_helicity_projections,
        sources,
        string_ranges,
        string_bytes,
        symmetry_proofs,
        transitions,
        closures,
        color_contractions,
        lc_color_transition_witnesses,
        color_nc_terms,
        u32_sequence_ranges,
        u32_sequence_values,
    })
}

fn decode_template_rows<T>(
    table: &OwnedTable,
    mut decode: impl FnMut(&OwnedTable, usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = table.row_count()?;
    (0..row_count).map(|row| decode(table, row)).collect()
}

fn template_indexed_ranges(table: &OwnedTable) -> RusticolResult<Vec<template::IndexedRangeRow>> {
    decode_template_rows(table, |table, row| {
        Ok(template::IndexedRangeRow {
            id: table.u32("id")?[row],
            range: CheckedTableRange::new(table.u64("start")?[row], table.u64("count")?[row]),
        })
    })
}

fn template_plain_ranges(table: &OwnedTable) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_template_rows(table, |table, row| {
        Ok(CheckedTableRange::new(
            table.u64("start")?[row],
            table.u64("count")?[row],
        ))
    })
}

fn decode_template_digest_catalog(
    table: &OwnedTable,
) -> RusticolResult<Vec<template::DigestCatalogRow>> {
    let ids = table.u32("id")?;
    let values = table.u8("value")?;
    let row_count = table.row_count()?;
    let expected = row_count
        .checked_mul(32)
        .ok_or_else(|| invalid("digest catalog byte count exceeds usize"))?;
    if values.len() != expected {
        return Err(invalid(format!(
            "digest_catalog.value has {} bytes, expected {expected}",
            values.len()
        )));
    }
    (0..row_count)
        .map(|row| {
            let start = row * 32;
            let mut value = [0_u8; 32];
            value.copy_from_slice(&values[start..start + 32]);
            Ok(template::DigestCatalogRow {
                id: ids[row],
                value,
            })
        })
        .collect()
}

fn validate_inventory(input: &OwnedInput) -> RusticolResult<()> {
    if input.abi != RECURRENCE_BUILDER_INPUT_ABI {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence builder input ABI {:?}; expected {:?}",
            input.abi, RECURRENCE_BUILDER_INPUT_ABI
        )));
    }
    if input.tables.len() != TABLE_SPECS.len() {
        return Err(invalid(
            "recurrence builder table inventory changed after extraction",
        ));
    }
    for (table, spec) in input.tables.iter().zip(TABLE_SPECS) {
        if table.name != spec.name || table.columns.len() != spec.columns.len() {
            return Err(invalid(format!(
                "recurrence builder table {:?} schema changed after extraction",
                table.name
            )));
        }
        for (column, column_spec) in table.columns.iter().zip(spec.columns) {
            if column.name != column_spec.name || column.dtype != column_spec.kind.dtype() {
                return Err(invalid(format!(
                    "recurrence builder column {}.{} schema changed after extraction",
                    table.name, column.name
                )));
            }
            if column.shape.first().copied() != Some(table.row_count) {
                return Err(invalid(format!(
                    "recurrence builder column {}.{} row count changed after extraction",
                    table.name, column.name
                )));
            }
        }
    }
    Ok(())
}

fn canonical_digest(input: &OwnedInput) -> RusticolResult<String> {
    let mut digest = Sha256::new();
    hash_text(&mut digest, &input.abi)?;
    hash_tables(&mut digest, &input.tables)?;
    Ok(hex_digest(digest.finalize()))
}

fn hash_tables(digest: &mut Sha256, tables: &[OwnedTable]) -> RusticolResult<()> {
    digest.update(
        u64::try_from(tables.len())
            .map_err(|_| invalid("recurrence table count exceeds u64"))?
            .to_le_bytes(),
    );
    for table in tables {
        hash_text(digest, &table.name)?;
        digest.update(table.row_count.to_le_bytes());
        digest.update(
            u32::try_from(table.columns.len())
                .map_err(|_| invalid("recurrence table column count exceeds u32"))?
                .to_le_bytes(),
        );
        for column in &table.columns {
            hash_text(digest, &column.name)?;
            hash_text(digest, column.dtype)?;
            digest.update(
                u8::try_from(column.shape.len())
                    .map_err(|_| invalid("recurrence column rank exceeds u8"))?
                    .to_le_bytes(),
            );
            for dimension in &column.shape {
                digest.update(dimension.to_le_bytes());
            }
            digest.update(column.values.raw_bytes());
        }
    }
    Ok(())
}

fn maximum_mask_bit(ranges: &[CheckedTableRange], words: &[u64]) -> RusticolResult<Option<u64>> {
    let mut maximum = None;
    for (mask_id, range) in ranges.iter().copied().enumerate() {
        let mask_words =
            &words[range.as_usize_range(words.len(), &format!("multiword mask {mask_id}"))?];
        if let Some(last) = mask_words.last().copied() {
            let word_index = u64::try_from(mask_words.len() - 1)
                .map_err(|_| invalid("multiword mask length exceeds u64"))?;
            let bit = word_index
                .checked_mul(64)
                .and_then(|value| value.checked_add(u64::from(63 - last.leading_zeros())))
                .ok_or_else(|| invalid("multiword mask maximum bit exceeds u64"))?;
            maximum = Some(maximum.map_or(bit, |value: u64| value.max(bit)));
        }
    }
    Ok(maximum)
}

fn result_mapping(py: Python<'_>, native: NativeValidationResult) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", RESULT_KIND)?;
    result.set_item("schema_version", RESULT_SCHEMA_VERSION)?;
    result.set_item("execution_mode", "recurrence")?;
    result.set_item(
        "validation_status",
        if native.composite_authenticated {
            "validated-composite-input"
        } else {
            "validated-identity-only"
        },
    )?;
    result.set_item("schedule_constructed", native.schedule_summary.is_some())?;
    result.set_item("composite_authenticated", native.composite_authenticated)?;
    result.set_item("builder_input_abi", RECURRENCE_BUILDER_INPUT_ABI)?;
    result.set_item("builder_input_schema_version", INPUT_SCHEMA_VERSION)?;
    result.set_item("builder_input_sha256", native.digest)?;
    result.set_item("builder_result_abi", RECURRENCE_BUILDER_RESULT_ABI)?;
    result.set_item("recurrence_template_abi", RECURRENCE_TEMPLATE_ABI)?;
    result.set_item(
        "template_catalog_digest",
        native
            .template_catalog_digest
            .map(|value| value.to_string()),
    )?;
    result.set_item(
        "compiled_model_digest",
        native.compiled_model_digest.map(|value| value.to_string()),
    )?;
    result.set_item(
        "prepared_kernel_pack_digest",
        native
            .prepared_kernel_pack_digest
            .map(|value| value.to_string()),
    )?;
    result.set_item("recurrence_plan_abi", RECURRENCE_PLAN_ABI)?;
    result.set_item("runtime_kind", RECURRENCE_RUNTIME_KIND)?;
    result.set_item("runtime_layout_abi", RECURRENCE_RUNTIME_LAYOUT_ABI)?;
    result.set_item(
        "required_runtime_capabilities",
        PyList::new(
            py,
            [
                RECURRENCE_RUNTIME_CAPABILITY,
                RECURRENCE_LC_COLOR_CAPABILITY,
            ],
        )?,
    )?;

    let summary = PyDict::new(py);
    summary.set_item("process_id", native.process_id)?;
    summary.set_item("lc_flow_layout", native.strategy.as_str())?;
    summary.set_item("table_count", native.table_count)?;
    summary.set_item("column_count", native.column_count)?;
    summary.set_item("total_row_count", native.total_row_count)?;
    summary.set_item("primitive_element_count", native.primitive_element_count)?;
    summary.set_item("primitive_byte_count", native.primitive_byte_count)?;
    summary.set_item("string_count", native.string_count)?;
    summary.set_item("semantic_digest_count", native.semantic_digest_count)?;
    summary.set_item("exact_factor_count", native.exact_factor_count)?;
    summary.set_item("external_leg_count", native.external_leg_count)?;
    summary.set_item("source_state_count", native.source_state_count)?;
    summary.set_item("physical_sector_count", native.physical_sector_count)?;
    summary.set_item("public_flow_count", native.public_flow_count)?;
    summary.set_item("open_string_count", native.open_string_count)?;
    summary.set_item("replay_partition_count", native.replay_partition_count)?;
    summary.set_item("replay_target_count", native.replay_target_count)?;
    summary.set_item("selector_mask_count", native.selector_mask_count)?;
    summary.set_item("selector_mask_word_count", native.selector_mask_word_count)?;
    summary.set_item("maximum_mask_bit", native.maximum_mask_bit)?;
    summary.set_item("template_reference_count", native.template_reference_count)?;
    summary.set_item(
        "parameter_projection_count",
        native.parameter_projection_count,
    )?;
    summary.set_item("prepared_template_count", native.prepared_template_count)?;
    if let Some(schedule) = native.schedule_summary {
        let schedule_summary = PyDict::new(py);
        schedule_summary.set_item(
            "dynamic_color_state_count",
            schedule.dynamic_color_state_count,
        )?;
        schedule_summary.set_item("current_count", schedule.current_count)?;
        schedule_summary.set_item("source_current_count", schedule.source_current_count)?;
        schedule_summary.set_item(
            "current_count_by_support_size",
            schedule.current_count_by_support_size,
        )?;
        schedule_summary.set_item("contribution_count", schedule.contribution_count)?;
        schedule_summary.set_item(
            "contribution_count_by_transition_template_id",
            schedule.contribution_count_by_transition_template_id,
        )?;
        schedule_summary.set_item(
            "contribution_count_by_quantum_flow_template_id",
            schedule.contribution_count_by_quantum_flow_template_id,
        )?;
        schedule_summary.set_item(
            "referenced_quantum_flow_template_ids",
            schedule.referenced_quantum_flow_template_ids,
        )?;
        schedule_summary.set_item("finalization_count", schedule.finalization_count)?;
        schedule_summary.set_item(
            "identity_finalization_count_by_support_size",
            schedule.identity_finalization_count_by_support_size,
        )?;
        schedule_summary.set_item(
            "propagated_finalization_count_by_support_size",
            schedule.propagated_finalization_count_by_support_size,
        )?;
        schedule_summary.set_item("retained_helicity_count", schedule.retained_helicity_count)?;
        schedule_summary.set_item("resolved_helicity_count", schedule.resolved_helicity_count)?;
        schedule_summary.set_item(
            "structural_zero_helicity_count",
            schedule.structural_zero_helicity_count,
        )?;
        schedule_summary.set_item(
            "amplitude_destination_count",
            schedule.amplitude_destination_count,
        )?;
        schedule_summary.set_item("target_sector_count", schedule.target_sector_count)?;
        schedule_summary.set_item("closure_term_count", schedule.closure_term_count)?;
        let source_layout = PyList::empty(py);
        for source in schedule.source_layout {
            let row = PyDict::new(py);
            row.set_item("current_id", source.current_id)?;
            row.set_item("source_slot", source.source_slot)?;
            row.set_item("source_template_id", source.source_template_id)?;
            row.set_item("component_start", source.component_start)?;
            row.set_item("component_count", source.component_count)?;
            source_layout.append(row)?;
        }
        schedule_summary.set_item("source_layout", source_layout)?;
        summary.set_item("schedule", schedule_summary)?;
    }
    result.set_item("inspection_summary", summary)?;
    Ok(result.into_any().unbind())
}

fn iterable_attribute<'py>(
    value: &Bound<'py, PyAny>,
    attribute: &str,
    context: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    value
        .getattr(attribute)?
        .try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{context} must be iterable")))?
        .collect()
}

fn required_string(value: &Bound<'_, PyAny>, attribute: &str) -> PyResult<String> {
    value.getattr(attribute)?.extract::<String>().map_err(|_| {
        PyTypeError::new_err(format!(
            "recurrence builder input {attribute} must be a string"
        ))
    })
}

fn required_nonempty_string(
    value: &Bound<'_, PyAny>,
    attribute: &str,
    context: &str,
) -> PyResult<String> {
    let result = required_string(value, attribute)?;
    if result.is_empty() {
        return Err(PyValueError::new_err(format!(
            "recurrence builder {context} must not be empty"
        )));
    }
    Ok(result)
}

fn validate_sha256_text(value: &str, context: &str) -> PyResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PyValueError::new_err(format!(
            "{context} must be a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn semantic_digest_python_attribute(
    value: &Bound<'_, PyAny>,
    attribute: &str,
) -> PyResult<SemanticDigest> {
    let digest = required_string(value, attribute)?;
    semantic_digest_from_hex(&digest, attribute).map_err(python_error)
}

fn semantic_digest_from_hex(value: &str, context: &str) -> RusticolResult<SemanticDigest> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(invalid(format!(
            "{context} must be a lowercase SHA-256 digest"
        )));
    }
    let mut result = [0_u8; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| invalid(format!("{context} is not hexadecimal")))?;
    }
    SemanticDigest::new(result)
}

fn hash_text(digest: &mut Sha256, value: &str) -> RusticolResult<()> {
    digest.update(
        u64::try_from(value.len())
            .map_err(|_| invalid("recurrence digest text length exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(value.as_bytes());
    Ok(())
}

fn hex_digest(value: impl AsRef<[u8]>) -> String {
    let mut result = String::with_capacity(value.as_ref().len() * 2);
    for byte in value.as_ref() {
        use std::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn wrong_type(context: &str, expected: &str) -> RusticolError {
    invalid(format!("{context} is not a {expected} recurrence column"))
}
