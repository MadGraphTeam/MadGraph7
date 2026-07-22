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
use rusticol_core::recurrence::{
    CheckedTableRange, ExactComplexRational, MultiwordMaskCatalogView,
    RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_BUILDER_RESULT_ABI, RECURRENCE_LC_COLOR_CAPABILITY,
    RECURRENCE_PLAN_ABI, RECURRENCE_RUNTIME_CAPABILITY, RECURRENCE_RUNTIME_KIND,
    RECURRENCE_RUNTIME_LAYOUT_ABI, RECURRENCE_TEMPLATE_ABI, RecurrenceStrategy, checked_u32_len,
    checked_usize, validate_packed_ranges, validate_u32_references,
};
use rusticol_core::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

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
    fn column(&self, name: &str) -> RusticolResult<&OwnedColumn> {
        let index = self.column_by_name.get(name).ok_or_else(|| {
            invalid(format!(
                "recurrence builder table {:?} has no column {name:?}",
                self.name
            ))
        })?;
        Ok(&self.columns[*index])
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
}

#[pyfunction]
pub(crate) fn _validate_recurrence_builder_input_v1(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let owned = parse_input(builder_input)?;
    let native = py
        .detach(move || validate_input(owned))
        .map_err(python_error)?;
    result_mapping(py, native)
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

fn validate_input(input: OwnedInput) -> RusticolResult<NativeValidationResult> {
    validate_inventory(&input)?;
    let actual_digest = canonical_digest(&input)?;
    if actual_digest != input.declared_digest {
        return Err(invalid(format!(
            "recurrence builder input digest mismatch: declared {}, found {actual_digest}",
            input.declared_digest
        )));
    }

    let strings = validate_flat_catalogs(&input)?;
    let strategy = validate_header(&input, &strings)?;
    validate_dense_ids_and_references(&input)?;
    validate_exact_factors(&input, &strings)?;
    validate_parent_ranges(&input)?;
    validate_source_and_template_contracts(&input, &strings)?;
    validate_replay_signs(&input)?;

    let bitset_ranges = checked_ranges(&input, "bitset_ranges", "start", "count")?;
    let bitset_populations = input.u64("bitset_ranges", "bit_count")?;
    let bitset_words = input.u64("bitset_words", "value")?;
    let masks = MultiwordMaskCatalogView {
        ranges: &bitset_ranges,
        populations: bitset_populations,
        words: bitset_words,
    };
    masks.validate(true)?;
    let maximum_mask_bit = maximum_mask_bit(&bitset_ranges, bitset_words)?;

    let process_id = string_at(
        &strings,
        input.u32("header", "process_id_string_id")?[0],
        "header.process_id_string_id",
    )?
    .to_owned();
    if process_id.is_empty() {
        return Err(invalid("recurrence process id must not be empty"));
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

    Ok(NativeValidationResult {
        digest: actual_digest,
        process_id,
        strategy,
        table_count: input.tables.len(),
        column_count,
        total_row_count,
        primitive_element_count,
        primitive_byte_count,
        string_count: strings.len(),
        semantic_digest_count: row_count(&input, "digest_catalog")?,
        exact_factor_count: row_count(&input, "exact_factors")?,
        external_leg_count: row_count(&input, "external_legs")?,
        source_state_count: row_count(&input, "source_states")?,
        physical_sector_count: row_count(&input, "physical_lc_sectors")?,
        public_flow_count: row_count(&input, "public_lc_flows")?,
        open_string_count: row_count(&input, "lc_open_strings")?,
        replay_partition_count: row_count(&input, "replay_partitions")?,
        replay_target_count: row_count(&input, "replay_targets")?,
        selector_mask_count: bitset_ranges.len(),
        selector_mask_word_count: bitset_words.len(),
        maximum_mask_bit,
        template_reference_count: row_count(&input, "semantic_template_references")?,
        parameter_projection_count: row_count(&input, "parameter_projection")?,
    })
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
    digest.update(
        u64::try_from(input.tables.len())
            .map_err(|_| invalid("recurrence table count exceeds u64"))?
            .to_le_bytes(),
    );
    for table in &input.tables {
        hash_text(&mut digest, &table.name)?;
        digest.update(table.row_count.to_le_bytes());
        digest.update(
            u32::try_from(table.columns.len())
                .map_err(|_| invalid("recurrence table column count exceeds u32"))?
                .to_le_bytes(),
        );
        for column in &table.columns {
            hash_text(&mut digest, &column.name)?;
            hash_text(&mut digest, column.dtype)?;
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
    Ok(hex_digest(digest.finalize()))
}

fn validate_flat_catalogs(input: &OwnedInput) -> RusticolResult<Vec<String>> {
    let string_ranges = checked_ranges(input, "string_ranges", "start", "count")?;
    let string_bytes = input.u8("string_bytes", "value")?;
    validate_packed_ranges("recurrence string", &string_ranges, string_bytes.len())?;
    let mut strings = Vec::with_capacity(string_ranges.len());
    for (index, range) in string_ranges.iter().copied().enumerate() {
        let bytes = &string_bytes
            [range.as_usize_range(string_bytes.len(), &format!("recurrence string {index}"))?];
        let value = std::str::from_utf8(bytes)
            .map_err(|error| invalid(format!("recurrence string {index} is not UTF-8: {error}")))?;
        strings.push(value.to_owned());
    }
    if !strings.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(invalid(
            "recurrence string catalog must be unique and in canonical byte order",
        ));
    }

    let sequence_ranges = checked_ranges(input, "u32_sequence_ranges", "start", "count")?;
    validate_packed_ranges(
        "recurrence u32 sequence",
        &sequence_ranges,
        input.u32("u32_sequence_values", "value")?.len(),
    )?;

    let bitset_ranges = checked_ranges(input, "bitset_ranges", "start", "count")?;
    validate_packed_ranges(
        "recurrence multiword mask",
        &bitset_ranges,
        input.u64("bitset_words", "value")?.len(),
    )?;
    Ok(strings)
}

fn validate_header(input: &OwnedInput, strings: &[String]) -> RusticolResult<RecurrenceStrategy> {
    if row_count(input, "header")? != 1 {
        return Err(invalid(
            "recurrence builder header must contain exactly one row",
        ));
    }
    if input.u32("header", "schema_version")?[0] != INPUT_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence builder input schema version {}; expected {INPUT_SCHEMA_VERSION}",
            input.u32("header", "schema_version")?[0]
        )));
    }
    let abi = string_at(
        strings,
        input.u32("header", "abi_string_id")?[0],
        "header.abi_string_id",
    )?;
    if abi != input.abi {
        return Err(invalid(
            "recurrence builder header ABI string does not match payload ABI",
        ));
    }
    for column in ["selected_flow_mode", "selected_source_mode"] {
        let value = input.u8("header", column)?[0];
        if value > 1 {
            return Err(invalid(format!("header.{column} must be zero or one")));
        }
    }
    let strategy = RecurrenceStrategy::try_from(u32::from(input.u8("header", "layout")?[0]))?;
    let expected_counts = [
        ("external_leg_count", "external_legs"),
        ("physical_sector_count", "physical_lc_sectors"),
        ("public_flow_count", "public_lc_flows"),
        ("replay_partition_count", "replay_partitions"),
        ("coupling_limit_count", "coupling_limits"),
        ("parameter_projection_count", "parameter_projection"),
    ];
    for (column, table) in expected_counts {
        let expected = checked_u32_len(row_count(input, table)?, table)?;
        let found = input.u32("header", column)?[0];
        if found != expected {
            return Err(invalid(format!(
                "header.{column} is {found}, but table {table:?} contains {expected} rows"
            )));
        }
    }

    for column in ["is_initial"] {
        for (row, value) in input
            .u8("external_legs", column)?
            .iter()
            .copied()
            .enumerate()
        {
            if value > 1 {
                return Err(invalid(format!(
                    "external_legs.{column} row {row} must be zero or one"
                )));
            }
        }
    }

    let role_ids = input.u32("header_digests", "role_string_id")?;
    let mut roles = BTreeSet::new();
    for (row, role_id) in role_ids.iter().copied().enumerate() {
        let role = string_at(strings, role_id, &format!("header_digests row {row} role"))?;
        if !roles.insert(role) {
            return Err(invalid(format!(
                "recurrence header contains duplicate semantic digest role {role:?}"
            )));
        }
    }
    for required in ["process", "model-catalog", "prepared-catalog", "color-plan"] {
        if !roles.contains(required) {
            return Err(invalid(format!(
                "recurrence header is missing semantic digest role {required:?}"
            )));
        }
    }

    let selected_flow_rows = row_count(input, "selected_public_flow_coverage")?;
    let selected_source_rows = row_count(input, "selected_source_coverage")?;
    if (input.u8("header", "selected_flow_mode")?[0] != 0) != (selected_flow_rows != 0) {
        return Err(invalid(
            "selected-flow header mode disagrees with its coverage table",
        ));
    }
    if (input.u8("header", "selected_source_mode")?[0] != 0) != (selected_source_rows != 0) {
        return Err(invalid(
            "selected-source header mode disagrees with its coverage table",
        ));
    }
    Ok(strategy)
}

fn validate_dense_ids_and_references(input: &OwnedInput) -> RusticolResult<()> {
    for (table, column) in [
        ("bitset_ranges", "id"),
        ("digest_catalog", "id"),
        ("exact_factors", "id"),
        ("external_legs", "source_slot"),
        ("physical_lc_sectors", "sector_id"),
        ("public_lc_flows", "flow_id"),
        ("replay_partitions", "partition_id"),
        ("parameter_projection", "runtime_slot"),
    ] {
        validate_dense_ids(input.u32(table, column)?, &format!("{table}.{column}"))?;
    }

    let string_count = row_count(input, "string_ranges")?;
    validate_reference_columns(
        input,
        &[
            ("header", "abi_string_id"),
            ("header", "process_id_string_id"),
            ("header_digests", "role_string_id"),
            ("coupling_limits", "name_string_id"),
            ("exact_factors", "real_numerator_string_id"),
            ("exact_factors", "real_denominator_string_id"),
            ("exact_factors", "imag_numerator_string_id"),
            ("exact_factors", "imag_denominator_string_id"),
            ("physical_lc_sectors", "public_id_string_id"),
            ("public_lc_flows", "public_id_string_id"),
            ("replay_partitions", "proof_algorithm_string_id"),
            ("semantic_template_references", "kind_string_id"),
            ("normalization", "convention_string_id"),
            ("parameter_projection", "runtime_name_string_id"),
        ],
        string_count,
    )?;

    let digest_count = row_count(input, "digest_catalog")?;
    validate_reference_columns(
        input,
        &[
            ("header_digests", "digest_id"),
            ("replay_partitions", "proof_digest_id"),
            ("semantic_template_references", "semantic_digest_id"),
            ("normalization", "semantic_digest_id"),
        ],
        digest_count,
    )?;
    let digest_bytes = input.u8("digest_catalog", "value")?;
    for (row, digest) in digest_bytes.chunks_exact(32).enumerate() {
        if digest.iter().all(|byte| *byte == 0) {
            return Err(invalid(format!(
                "digest_catalog row {row} contains an all-zero semantic digest"
            )));
        }
    }

    let sequence_count = row_count(input, "u32_sequence_ranges")?;
    validate_reference_columns(
        input,
        &[
            ("lc_open_strings", "adjoint_sequence_id"),
            ("lc_open_strings", "singlet_sequence_id"),
            ("physical_lc_sectors", "trace_sequence_id"),
            ("physical_lc_sectors", "singlet_sequence_id"),
            ("physical_lc_sectors", "word_sequence_id"),
            ("replay_targets", "external_permutation_sequence_id"),
            ("replay_targets", "source_slot_permutation_sequence_id"),
            ("public_lc_flows", "word_sequence_id"),
            ("public_lc_flows", "source_slot_permutation_sequence_id"),
        ],
        sequence_count,
    )?;

    let bitset_count = row_count(input, "bitset_ranges")?;
    validate_reference_columns(
        input,
        &[
            ("header", "process_support_mask_id"),
            ("external_legs", "momentum_mask_id"),
            ("external_legs", "support_mask_id"),
            ("physical_lc_sectors", "support_mask_id"),
        ],
        bitset_count,
    )?;

    let factor_count = row_count(input, "exact_factors")?;
    validate_reference_columns(
        input,
        &[
            ("replay_targets", "amplitude_phase_factor_id"),
            ("public_lc_flows", "reduction_weight_factor_id"),
            ("source_states", "crossing_phase_factor_id"),
            ("normalization", "factor_id"),
        ],
        factor_count,
    )?;

    let external_count = row_count(input, "external_legs")?;
    validate_reference_columns(
        input,
        &[
            ("lc_open_strings", "fundamental_source_slot"),
            ("lc_open_strings", "antifundamental_source_slot"),
            ("source_states", "source_slot"),
            ("selected_source_coverage", "source_slot"),
        ],
        external_count,
    )?;

    let sector_count = row_count(input, "physical_lc_sectors")?;
    validate_reference_columns(
        input,
        &[
            ("lc_open_strings", "sector_id"),
            ("replay_partitions", "representative_sector_id"),
            ("replay_partitions", "materialized_sector_id"),
            ("replay_targets", "sector_id"),
            ("public_lc_flows", "construction_sector_id"),
        ],
        sector_count,
    )?;

    let public_flow_count = row_count(input, "public_lc_flows")?;
    validate_reference_columns(
        input,
        &[("selected_public_flow_coverage", "flow_id")],
        public_flow_count,
    )?;

    let partition_count = row_count(input, "replay_partitions")?;
    validate_reference_columns(
        input,
        &[("replay_targets", "partition_id")],
        partition_count,
    )?;
    Ok(())
}

fn validate_source_and_template_contracts(
    input: &OwnedInput,
    strings: &[String],
) -> RusticolResult<()> {
    let kind_ids = input.u32("semantic_template_references", "kind_string_id")?;
    let template_ids = input.u32("semantic_template_references", "template_id")?;
    let mut templates: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
    for (row, (kind_id, template_id)) in kind_ids
        .iter()
        .copied()
        .zip(template_ids.iter().copied())
        .enumerate()
    {
        let kind = string_at(
            strings,
            kind_id,
            &format!("semantic_template_references row {row} kind"),
        )?;
        if kind.is_empty() {
            return Err(invalid(format!(
                "semantic_template_references row {row} has an empty kind"
            )));
        }
        if !templates.entry(kind).or_default().insert(template_id) {
            return Err(invalid(format!(
                "semantic_template_references contains duplicate typed ID ({kind:?}, {template_id})"
            )));
        }
    }

    let require_template = |kind: &str, template_id: u32, context: &str| -> RusticolResult<()> {
        if !templates
            .get(kind)
            .is_some_and(|values| values.contains(&template_id))
        {
            return Err(invalid(format!(
                "{context} references absent semantic template ({kind:?}, {template_id})"
            )));
        }
        Ok(())
    };

    let source_slots = input.u32("source_states", "source_slot")?;
    let state_indices = input.u32("source_states", "state_index")?;
    let public_helicities = input.i32("source_states", "public_helicity")?;
    let momentum_signs = input.i32("source_states", "momentum_sign")?;
    let current_state_ids = input.u32("source_states", "current_state_template_id")?;
    let source_template_ids = input.u32("source_states", "source_template_id")?;
    let source_ranges = checked_ranges(
        input,
        "external_legs",
        "source_state_start",
        "source_state_count",
    )?;
    for (source_slot, range) in source_ranges.iter().copied().enumerate() {
        let rows = range.as_usize_range(
            source_slots.len(),
            &format!("external_legs source-state range {source_slot}"),
        )?;
        if rows.is_empty() {
            return Err(invalid(format!(
                "external_legs row {source_slot} has no source states"
            )));
        }
        let mut seen_helicities = BTreeSet::new();
        for (local_index, row) in rows.enumerate() {
            let expected_slot = u32::try_from(source_slot)
                .map_err(|_| invalid("external source slot exceeds u32"))?;
            if source_slots[row] != expected_slot {
                return Err(invalid(format!(
                    "source_states row {row} belongs to source slot {}, expected {expected_slot}",
                    source_slots[row]
                )));
            }
            let expected_index = u32::try_from(local_index)
                .map_err(|_| invalid("source-state index exceeds u32"))?;
            if state_indices[row] != expected_index {
                return Err(invalid(format!(
                    "source_states.state_index row {row} contains {}, expected {expected_index}",
                    state_indices[row]
                )));
            }
            if !seen_helicities.insert(public_helicities[row]) {
                return Err(invalid(format!(
                    "external source slot {source_slot} repeats public helicity {}",
                    public_helicities[row]
                )));
            }
            if !matches!(momentum_signs[row], -1 | 1) {
                return Err(invalid(format!(
                    "source_states.momentum_sign row {row} must be -1 or 1"
                )));
            }
            require_template(
                "current-state",
                current_state_ids[row],
                &format!("source_states.current_state_template_id row {row}"),
            )?;
            require_template(
                "source",
                source_template_ids[row],
                &format!("source_states.source_template_id row {row}"),
            )?;
        }
    }

    for (row, template_id) in input
        .u32("parameter_projection", "parameter_template_id")?
        .iter()
        .copied()
        .enumerate()
    {
        require_template(
            "parameter",
            template_id,
            &format!("parameter_projection.parameter_template_id row {row}"),
        )?;
    }

    let external_count = row_count(input, "external_legs")?;
    let sequence_ranges = checked_ranges(input, "u32_sequence_ranges", "start", "count")?;
    let sequence_values = input.u32("u32_sequence_values", "value")?;
    validate_permutation_references(
        input,
        &sequence_ranges,
        sequence_values,
        "replay_targets",
        "external_permutation_sequence_id",
        external_count,
    )?;
    validate_permutation_references(
        input,
        &sequence_ranges,
        sequence_values,
        "replay_targets",
        "source_slot_permutation_sequence_id",
        external_count,
    )?;
    validate_permutation_references(
        input,
        &sequence_ranges,
        sequence_values,
        "public_lc_flows",
        "source_slot_permutation_sequence_id",
        external_count,
    )?;
    validate_public_flow_words(input, &sequence_ranges, sequence_values)?;
    validate_replay_words(input, &sequence_ranges, sequence_values)?;

    let public_flow_ids = input.u32("public_lc_flows", "public_id_string_id")?;
    if public_flow_ids.is_empty() {
        return Err(invalid(
            "recurrence builder input requires at least one public LC flow",
        ));
    }
    let mut seen_public_flows = BTreeSet::new();
    for (row, string_id) in public_flow_ids.iter().copied().enumerate() {
        let public_id = string_at(
            strings,
            string_id,
            &format!("public_lc_flows.public_id_string_id row {row}"),
        )?;
        if public_id.is_empty() || !seen_public_flows.insert(public_id) {
            return Err(invalid(format!(
                "public_lc_flows row {row} has an empty or duplicate public identifier {public_id:?}"
            )));
        }
    }

    let mut selected_flows = BTreeSet::new();
    for (row, flow_id) in input
        .u32("selected_public_flow_coverage", "flow_id")?
        .iter()
        .copied()
        .enumerate()
    {
        if !selected_flows.insert(flow_id) {
            return Err(invalid(format!(
                "selected_public_flow_coverage row {row} repeats public flow {flow_id}"
            )));
        }
    }

    let selected_slots = input.u32("selected_source_coverage", "source_slot")?;
    let selected_states = input.u32("selected_source_coverage", "source_state_index")?;
    for (row, (source_slot, state_index)) in selected_slots
        .iter()
        .copied()
        .zip(selected_states.iter().copied())
        .enumerate()
    {
        let source_range = source_ranges.get(source_slot as usize).ok_or_else(|| {
            invalid(format!(
                "selected_source_coverage row {row} references absent source slot {source_slot}"
            ))
        })?;
        if u64::from(state_index) >= source_range.count {
            return Err(invalid(format!(
                "selected_source_coverage row {row} references absent state {state_index} for source slot {source_slot}"
            )));
        }
    }
    Ok(())
}

fn validate_public_flow_words(
    input: &OwnedInput,
    sequence_ranges: &[CheckedTableRange],
    sequence_values: &[u32],
) -> RusticolResult<()> {
    let sector_word_ids = input.u32("physical_lc_sectors", "word_sequence_id")?;
    let construction_sector_ids = input.u32("public_lc_flows", "construction_sector_id")?;
    let public_word_ids = input.u32("public_lc_flows", "word_sequence_id")?;
    let permutation_ids = input.u32("public_lc_flows", "source_slot_permutation_sequence_id")?;
    for row in 0..construction_sector_ids.len() {
        let sector_id = construction_sector_ids[row];
        let sector_word_id = *sector_word_ids.get(sector_id as usize).ok_or_else(|| {
            invalid(format!(
                "public_lc_flows row {row} references absent construction sector {sector_id}"
            ))
        })?;
        let construction_word = sequence_at(
            sequence_ranges,
            sequence_values,
            sector_word_id,
            &format!("public_lc_flows row {row} construction-sector word"),
        )?;
        let public_word = sequence_at(
            sequence_ranges,
            sequence_values,
            public_word_ids[row],
            &format!("public_lc_flows row {row} public word"),
        )?;
        let permutation = sequence_at(
            sequence_ranges,
            sequence_values,
            permutation_ids[row],
            &format!("public_lc_flows row {row} gather permutation"),
        )?;
        validate_mapped_word(
            construction_word,
            permutation,
            public_word,
            &format!("public_lc_flows row {row}"),
        )?;
    }
    Ok(())
}

fn validate_replay_words(
    input: &OwnedInput,
    sequence_ranges: &[CheckedTableRange],
    sequence_values: &[u32],
) -> RusticolResult<()> {
    let sector_word_ids = input.u32("physical_lc_sectors", "word_sequence_id")?;
    let representative_sector_ids = input.u32("replay_partitions", "representative_sector_id")?;
    let partition_ids = input.u32("replay_targets", "partition_id")?;
    let target_sector_ids = input.u32("replay_targets", "sector_id")?;
    let external_permutation_ids =
        input.u32("replay_targets", "external_permutation_sequence_id")?;
    let source_permutation_ids =
        input.u32("replay_targets", "source_slot_permutation_sequence_id")?;

    for row in 0..partition_ids.len() {
        let partition_id = partition_ids[row];
        let representative_sector_id = *representative_sector_ids
            .get(partition_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "replay_targets row {row} references absent partition {partition_id}"
                ))
            })?;
        let target_sector_id = target_sector_ids[row];
        let representative_word_id = *sector_word_ids
            .get(representative_sector_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "replay partition {partition_id} references absent representative sector {representative_sector_id}"
                ))
            })?;
        let target_word_id = *sector_word_ids
            .get(target_sector_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "replay_targets row {row} references absent target sector {target_sector_id}"
                ))
            })?;
        let external_permutation = sequence_at(
            sequence_ranges,
            sequence_values,
            external_permutation_ids[row],
            &format!("replay_targets row {row} external permutation"),
        )?;
        let source_permutation = sequence_at(
            sequence_ranges,
            sequence_values,
            source_permutation_ids[row],
            &format!("replay_targets row {row} source-slot permutation"),
        )?;
        if external_permutation != source_permutation {
            return Err(invalid(format!(
                "replay_targets row {row} has different external and source-slot permutations"
            )));
        }
        let representative_word = sequence_at(
            sequence_ranges,
            sequence_values,
            representative_word_id,
            &format!("replay_targets row {row} representative-sector word"),
        )?;
        let target_word = sequence_at(
            sequence_ranges,
            sequence_values,
            target_word_id,
            &format!("replay_targets row {row} target-sector word"),
        )?;
        validate_mapped_word(
            representative_word,
            source_permutation,
            target_word,
            &format!("replay_targets row {row}"),
        )?;
    }
    Ok(())
}

fn sequence_at<'a>(
    ranges: &[CheckedTableRange],
    values: &'a [u32],
    sequence_id: u32,
    context: &str,
) -> RusticolResult<&'a [u32]> {
    let range = ranges.get(sequence_id as usize).ok_or_else(|| {
        invalid(format!(
            "{context} references absent sequence id {sequence_id}"
        ))
    })?;
    Ok(&values[range.as_usize_range(values.len(), context)?])
}

fn validate_mapped_word(
    source_word: &[u32],
    permutation: &[u32],
    target_word: &[u32],
    context: &str,
) -> RusticolResult<()> {
    if source_word.len() != target_word.len() {
        return Err(invalid(format!(
            "{context} maps a word of length {} onto a word of length {}",
            source_word.len(),
            target_word.len()
        )));
    }
    for (position, (source_slot, expected_target)) in source_word
        .iter()
        .copied()
        .zip(target_word.iter().copied())
        .enumerate()
    {
        let actual_target = permutation.get(source_slot as usize).ok_or_else(|| {
            invalid(format!(
                "{context} word position {position} references out-of-range source slot {source_slot}"
            ))
        })?;
        if *actual_target != expected_target {
            return Err(invalid(format!(
                "{context} gather permutation maps word position {position} to {actual_target}, expected {expected_target}"
            )));
        }
    }
    Ok(())
}

fn validate_permutation_references(
    input: &OwnedInput,
    sequence_ranges: &[CheckedTableRange],
    sequence_values: &[u32],
    table: &str,
    column: &str,
    expected_len: usize,
) -> RusticolResult<()> {
    for (row, sequence_id) in input.u32(table, column)?.iter().copied().enumerate() {
        let range = sequence_ranges.get(sequence_id as usize).ok_or_else(|| {
            invalid(format!(
                "{table}.{column} row {row} references absent sequence id {sequence_id}"
            ))
        })?;
        let values = &sequence_values[range.as_usize_range(
            sequence_values.len(),
            &format!("{table}.{column} row {row}"),
        )?];
        if values.len() != expected_len {
            return Err(invalid(format!(
                "{table}.{column} row {row} has {} entries, expected {expected_len}",
                values.len()
            )));
        }
        let mut seen = vec![false; expected_len];
        for value in values.iter().copied() {
            let index = usize::try_from(value)
                .map_err(|_| invalid(format!("{table}.{column} value exceeds usize")))?;
            let slot = seen.get_mut(index).ok_or_else(|| {
                invalid(format!(
                    "{table}.{column} row {row} contains out-of-range source slot {value}"
                ))
            })?;
            if *slot {
                return Err(invalid(format!(
                    "{table}.{column} row {row} repeats source slot {value}"
                )));
            }
            *slot = true;
        }
    }
    Ok(())
}

fn validate_exact_factors(input: &OwnedInput, strings: &[String]) -> RusticolResult<()> {
    let numerator_real = input.u32("exact_factors", "real_numerator_string_id")?;
    let denominator_real = input.u32("exact_factors", "real_denominator_string_id")?;
    let numerator_imaginary = input.u32("exact_factors", "imag_numerator_string_id")?;
    let denominator_imaginary = input.u32("exact_factors", "imag_denominator_string_id")?;
    for row in 0..numerator_real.len() {
        let real_numerator = string_at(strings, numerator_real[row], "exact real numerator")?;
        let real_denominator = string_at(strings, denominator_real[row], "exact real denominator")?;
        let imaginary_numerator = string_at(
            strings,
            numerator_imaginary[row],
            "exact imaginary numerator",
        )?;
        let imaginary_denominator = string_at(
            strings,
            denominator_imaginary[row],
            "exact imaginary denominator",
        )?;
        let factor = ExactComplexRational::parse_parts(
            real_numerator,
            real_denominator,
            imaginary_numerator,
            imaginary_denominator,
        )
        .map_err(|error| {
            invalid(format!(
                "exact_factors row {row} is not a canonical exact complex rational: {error}"
            ))
        })?;
        if factor.real().numerator().to_string() != real_numerator
            || factor.real().denominator().to_string() != real_denominator
            || factor.imag().numerator().to_string() != imaginary_numerator
            || factor.imag().denominator().to_string() != imaginary_denominator
        {
            return Err(invalid(format!(
                "exact_factors row {row} is not in canonical reduced decimal form"
            )));
        }
    }
    Ok(())
}

fn validate_parent_ranges(input: &OwnedInput) -> RusticolResult<()> {
    validate_parent_range(
        input,
        "external_legs",
        "source_state_start",
        "source_state_count",
        "source_states",
        "source_slot",
    )?;
    validate_parent_range(
        input,
        "physical_lc_sectors",
        "open_string_start",
        "open_string_count",
        "lc_open_strings",
        "sector_id",
    )?;
    validate_parent_range(
        input,
        "replay_partitions",
        "target_start",
        "target_count",
        "replay_targets",
        "partition_id",
    )?;
    Ok(())
}

fn validate_parent_range(
    input: &OwnedInput,
    parent_table: &str,
    start_column: &str,
    count_column: &str,
    child_table: &str,
    child_parent_column: &str,
) -> RusticolResult<()> {
    let ranges = checked_ranges(input, parent_table, start_column, count_column)?;
    let child_count = row_count(input, child_table)?;
    validate_packed_ranges(
        &format!("{parent_table} to {child_table}"),
        &ranges,
        child_count,
    )?;
    let child_parent_ids = input.u32(child_table, child_parent_column)?;
    for (parent_id, range) in ranges.iter().copied().enumerate() {
        let expected = u32::try_from(parent_id)
            .map_err(|_| invalid(format!("{parent_table} row index exceeds u32")))?;
        for child_id in &child_parent_ids[range.as_usize_range(
            child_parent_ids.len(),
            &format!("{parent_table} row {parent_id}"),
        )?] {
            if *child_id != expected {
                return Err(invalid(format!(
                    "{child_table}.{child_parent_column} contains {child_id} inside parent row {parent_id}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_replay_signs(input: &OwnedInput) -> RusticolResult<()> {
    for (row, sign) in input
        .i32("replay_targets", "fermion_sign")?
        .iter()
        .copied()
        .enumerate()
    {
        if !matches!(sign, -1 | 1) {
            return Err(invalid(format!(
                "replay_targets.fermion_sign row {row} must be -1 or 1"
            )));
        }
    }
    Ok(())
}

fn checked_ranges(
    input: &OwnedInput,
    table: &str,
    start_column: &str,
    count_column: &str,
) -> RusticolResult<Vec<CheckedTableRange>> {
    let starts = input.u64(table, start_column)?;
    let counts = input.u64(table, count_column)?;
    if starts.len() != counts.len() {
        return Err(invalid(format!(
            "{table} range columns have different lengths"
        )));
    }
    Ok(starts
        .iter()
        .copied()
        .zip(counts.iter().copied())
        .map(|(start, count)| CheckedTableRange::new(start, count))
        .collect())
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

fn validate_dense_ids(values: &[u32], label: &str) -> RusticolResult<()> {
    checked_u32_len(values.len(), label)?;
    for (row, value) in values.iter().copied().enumerate() {
        let expected =
            u32::try_from(row).map_err(|_| invalid(format!("{label} row index exceeds u32")))?;
        if value != expected {
            return Err(invalid(format!(
                "{label} row {row} contains id {value}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_reference_columns(
    input: &OwnedInput,
    columns: &[(&str, &str)],
    target_count: usize,
) -> RusticolResult<()> {
    for (table, column) in columns {
        validate_u32_references(
            input.u32(table, column)?,
            target_count,
            &format!("{table}.{column}"),
        )?;
    }
    Ok(())
}

fn row_count(input: &OwnedInput, table: &str) -> RusticolResult<usize> {
    checked_usize(input.table(table)?.row_count, &format!("{table} row count"))
}

fn string_at<'a>(strings: &'a [String], id: u32, context: &str) -> RusticolResult<&'a str> {
    strings
        .get(id as usize)
        .map(String::as_str)
        .ok_or_else(|| invalid(format!("{context} references absent string id {id}")))
}

fn result_mapping(py: Python<'_>, native: NativeValidationResult) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", RESULT_KIND)?;
    result.set_item("schema_version", RESULT_SCHEMA_VERSION)?;
    result.set_item("execution_mode", "recurrence")?;
    result.set_item("validation_status", "validated-identity-only")?;
    result.set_item("schedule_constructed", false)?;
    result.set_item("builder_input_abi", RECURRENCE_BUILDER_INPUT_ABI)?;
    result.set_item("builder_input_schema_version", INPUT_SCHEMA_VERSION)?;
    result.set_item("builder_input_sha256", native.digest)?;
    result.set_item("builder_result_abi", RECURRENCE_BUILDER_RESULT_ABI)?;
    result.set_item("recurrence_template_abi", RECURRENCE_TEMPLATE_ABI)?;
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
