// SPDX-License-Identifier: 0BSD

//! Private Python boundary for Direct-Arena recurrence lowering.
//!
//! Python supplies authenticated fixed-width builder and prepared-template
//! columns. Rust constructs the compact recurrence, lowers its direct arena
//! plan, and publishes the plan-v2 PACBIN without exposing an intermediate
//! packet schedule.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use rusticol_core::recurrence::process;
use rusticol_core::recurrence::template;
use rusticol_core::recurrence::{
    AuthenticatedRecurrenceBuilderInput, CheckedTableRange, DirectExecutorRole,
    DirectRecurrencePlan, DirectRecurrenceRuntimeOptions, PreparedDirectExecutorBinding,
    PreparedDirectExecutorCatalog, RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_DIRECT_PLAN_ABI,
    RECURRENCE_DIRECT_PLAN_MEMBER, RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI, RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_LC_COLOR_CAPABILITY, RecurrenceBuildProgress, RecurrenceStrategy, SemanticDigest,
    checked_usize, lower_recurrence_direct_plan_v2, write_recurrence_direct_plan_pacbin,
};
use rusticol_core::{RusticolError, RusticolResult};
use serde_json::{Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use crate::python_error;

const RUNTIME_CONTAINER_KIND: &str = "pyamplicol-recurrence-runtime-container";
const RUNTIME_CONTAINER_SCHEMA_VERSION: u32 = 1;
const STORAGE_ABI: &str = "pacbin-v1";
const DIRECT_BUILDER_INPUT_ABI: &str = "pyamplicol-recurrence-builder-input-v2";
const DIRECT_LOWERING_RESULT_KIND: &str = "pyamplicol-recurrence-direct-lowering-result";
const DIRECT_LOWERING_RESULT_SCHEMA_VERSION: u32 = 2;
const DIRECT_CANONICALIZATION_ABI: &str = "pyamplicol-canonical-json-v1";
const DIRECT_BACKEND_ABI: &str = "rusticol.recurrence-direct-backend.v1";
const DIRECT_PAYLOAD_BINDING_ABI: &str = "pyamplicol-recurrence-direct-payload-binding-v1";
const DIRECT_IDENTITY_FINALIZER: &str = "rusticol.identity-finalize-in-place.v1";

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

const FERMION_PAIRING_TABLE_SPECS: &[TableSpec] = &[
    TableSpec {
        name: "header",
        columns: &[
            column("schema_version", PrimitiveKind::U32),
            column("abi_string_id", PrimitiveKind::U32),
            column("process_key_string_id", PrimitiveKind::U32),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            column("source_count", PrimitiveKind::U32),
            column("endpoint_count", PrimitiveKind::U32),
            column("pairing_class_count", PrimitiveKind::U32),
            column("rule_count", PrimitiveKind::U32),
            column("endpoint_state_template_count", PrimitiveKind::U64),
            column("endpoint_anti_state_template_count", PrimitiveKind::U64),
            column("endpoint_basis_count", PrimitiveKind::U64),
            column("endpoint_color_representation_count", PrimitiveKind::U64),
            column("class_fundamental_slot_count", PrimitiveKind::U64),
            column("class_antifundamental_slot_count", PrimitiveKind::U64),
            column("class_reference_pairing_count", PrimitiveKind::U64),
            column("rule_class_pairing_index_count", PrimitiveKind::U64),
            column("rule_endpoint_pairing_count", PrimitiveKind::U64),
            column("rule_source_permutation_count", PrimitiveKind::U64),
            column("rule_lineage_count", PrimitiveKind::U64),
            column("exact_integer_count", PrimitiveKind::U32),
            column("exact_integer_limb_count", PrimitiveKind::U64),
            column("string_count", PrimitiveKind::U32),
            column("string_byte_count", PrimitiveKind::U64),
            column("no_fermion_line", PrimitiveKind::U32),
            shaped_column("topology_digest", PrimitiveKind::U8, &[32]),
            shaped_column("semantic_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "endpoints",
        columns: &[
            column("endpoint_id", PrimitiveKind::U32),
            column("source_slot", PrimitiveKind::U32),
            column("public_label", PrimitiveKind::U32),
            column("species_class_id", PrimitiveKind::U32),
            column("species_string_id", PrimitiveKind::U32),
            column("particle_orientation", PrimitiveKind::U8),
            column("color_orientation", PrimitiveKind::U8),
            column("state_template_start", PrimitiveKind::U64),
            column("state_template_count", PrimitiveKind::U64),
            column("anti_state_template_start", PrimitiveKind::U64),
            column("anti_state_template_count", PrimitiveKind::U64),
            column("basis_start", PrimitiveKind::U64),
            column("basis_count", PrimitiveKind::U64),
            column("color_representation_start", PrimitiveKind::U64),
            column("color_representation_count", PrimitiveKind::U64),
            shaped_column("contract_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "endpoint_state_template_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_anti_state_template_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_basis_ids",
        columns: &[column("string_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "endpoint_color_representations",
        columns: &[column("value", PrimitiveKind::I32)],
    },
    TableSpec {
        name: "pairing_classes",
        columns: &[
            column("class_id", PrimitiveKind::U32),
            column("species_class_id", PrimitiveKind::U32),
            column("species_string_id", PrimitiveKind::U32),
            column("fundamental_slot_start", PrimitiveKind::U64),
            column("fundamental_slot_count", PrimitiveKind::U64),
            column("antifundamental_slot_start", PrimitiveKind::U64),
            column("antifundamental_slot_count", PrimitiveKind::U64),
            column("reference_pairing_start", PrimitiveKind::U64),
            column("reference_pairing_count", PrimitiveKind::U64),
            column("pairing_count", PrimitiveKind::U64),
            shaped_column("proof_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "class_fundamental_slots",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "class_antifundamental_slots",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "class_reference_pairings",
        columns: &[
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "rules",
        columns: &[
            column("rule_id", PrimitiveKind::U32),
            column("class_pairing_index_start", PrimitiveKind::U64),
            column("class_pairing_index_count", PrimitiveKind::U64),
            column("endpoint_pairing_start", PrimitiveKind::U64),
            column("endpoint_pairing_count", PrimitiveKind::U64),
            column("source_permutation_start", PrimitiveKind::U64),
            column("source_permutation_count", PrimitiveKind::U64),
            column("lineage_start", PrimitiveKind::U64),
            column("lineage_count", PrimitiveKind::U64),
            column("fermion_parity", PrimitiveKind::I32),
            column("real_numerator_integer_id", PrimitiveKind::U32),
            column("real_denominator_integer_id", PrimitiveKind::U32),
            column("imag_numerator_integer_id", PrimitiveKind::U32),
            column("imag_denominator_integer_id", PrimitiveKind::U32),
            column("multiplicity", PrimitiveKind::U64),
            column("proof_algorithm_string_id", PrimitiveKind::U32),
            shaped_column("proof_digest", PrimitiveKind::U8, &[32]),
        ],
    },
    TableSpec {
        name: "rule_class_pairing_indices",
        columns: &[
            column("class_id", PrimitiveKind::U32),
            column("pairing_index", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "rule_endpoint_pairings",
        columns: &[
            column("fundamental_source_slot", PrimitiveKind::U32),
            column("antifundamental_source_slot", PrimitiveKind::U32),
        ],
    },
    TableSpec {
        name: "rule_source_slot_permutations",
        columns: &[column("source_slot", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "rule_lineages",
        columns: &[column("line_id", PrimitiveKind::U32)],
    },
    TableSpec {
        name: "exact_integers",
        columns: &[
            column("integer_id", PrimitiveKind::U32),
            column("sign", PrimitiveKind::I32),
            column("limb_start", PrimitiveKind::U64),
            column("limb_count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "exact_integer_limbs",
        columns: &[column("value", PrimitiveKind::U64)],
    },
    TableSpec {
        name: "string_ranges",
        columns: &[
            column("string_id", PrimitiveKind::U32),
            column("start", PrimitiveKind::U64),
            column("count", PrimitiveKind::U64),
        ],
    },
    TableSpec {
        name: "string_bytes",
        columns: &[column("value", PrimitiveKind::U8)],
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
    declared_fermion_pairing_digest: Option<String>,
    fermion_pairing_tables: Vec<OwnedTable>,
    fermion_pairing_table_by_name: BTreeMap<String, usize>,
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

    fn fermion_pairing_table(&self, name: &str) -> RusticolResult<&OwnedTable> {
        let index = self
            .fermion_pairing_table_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("recurrence fermion pairing has no table {name:?}")))?;
        Ok(&self.fermion_pairing_tables[*index])
    }

    fn pairing_u8(&self, table: &str, column: &str) -> RusticolResult<&[u8]> {
        self.fermion_pairing_table(table)?.u8(column)
    }

    fn pairing_u32(&self, table: &str, column: &str) -> RusticolResult<&[u32]> {
        self.fermion_pairing_table(table)?.u32(column)
    }

    fn pairing_u64(&self, table: &str, column: &str) -> RusticolResult<&[u64]> {
        self.fermion_pairing_table(table)?.u64(column)
    }

    fn pairing_i32(&self, table: &str, column: &str) -> RusticolResult<&[i32]> {
        self.fermion_pairing_table(table)?.i32(column)
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
    ("lc_color_transition_witnesses", 15),
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

#[derive(Clone, Copy, Debug, Default)]
struct DirectLoweringTimings {
    python_extraction_seconds: f64,
    catalog_authentication_seconds: f64,
    semantic_construction_seconds: f64,
    direct_lowering_seconds: f64,
    serialization_seconds: f64,
    native_total_seconds: f64,
}

#[derive(Clone, Debug)]
struct NativeDirectLoweringResult {
    builder_input_digest: String,
    template_input_digest: String,
    prepared_kernel_pack_digest: String,
    direct_template_catalog_digest: String,
    process_id: String,
    strategy: RecurrenceStrategy,
    semantic_digest: String,
    runtime_layout_digest: String,
    member_count: u64,
    unpacked_size_bytes: u64,
    index_sha256: String,
    container_size: u64,
    plan_payload_size: u64,
    plan_sha256: String,
    current_count: usize,
    source_row_count: usize,
    contribution_count: usize,
    finalization_count: usize,
    closure_count: usize,
    row_group_count: usize,
    momentum_form_count: usize,
    selector_domain_count: usize,
    replay_target_count: usize,
    amplitude_destination_count: usize,
    resolved_helicity_count: usize,
    retained_helicity_count: u64,
    exact_factor_count: usize,
    semantic_component_count: u64,
    current_arena_components: u32,
    parameter_value_count: u32,
    physical_sector_count: u32,
    direct_executor_count: u32,
    prepared_kernel_count: usize,
    resolved_helicities: Vec<Vec<i32>>,
    timings: DirectLoweringTimings,
}

struct AuthenticatedDirectTemplateCatalog {
    catalog: PreparedDirectExecutorCatalog,
    catalog_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
    prepared_kernel_count: usize,
}

#[pyfunction(signature = (
    builder_input,
    prepared_template_input,
    direct_template_catalog_json,
    prepared_kernel_pack_digest,
    destination,
    *,
    point_tile_size,
    workspace_mib,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _lower_recurrence_direct_v2(
    py: Python<'_>,
    builder_input: &Bound<'_, PyAny>,
    prepared_template_input: &Bound<'_, PyAny>,
    direct_template_catalog_json: &Bound<'_, PyBytes>,
    prepared_kernel_pack_digest: String,
    destination: PathBuf,
    point_tile_size: u32,
    workspace_mib: u32,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let extraction_started = Instant::now();
    let owned = parse_input(builder_input)?;
    let prepared_template = parse_prepared_template_input(prepared_template_input)?;
    let direct_template_catalog_json = direct_template_catalog_json.as_bytes().to_vec();
    validate_sha256_text(&prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    let python_extraction_seconds = extraction_started.elapsed().as_secs_f64();

    let native = py
        .detach(move || {
            let mut report =
                |progress| report_recurrence_build_progress(progress_callback.as_ref(), progress);
            lower_recurrence_direct(
                owned,
                prepared_template,
                &direct_template_catalog_json,
                &prepared_kernel_pack_digest,
                &destination,
                point_tile_size,
                workspace_mib,
                python_extraction_seconds,
                &mut report,
            )
        })
        .map_err(python_error)?;
    direct_lowering_mapping(py, native)
}

#[allow(clippy::too_many_arguments)]
fn lower_recurrence_direct(
    input: OwnedInput,
    prepared_template: PreparedTemplateInput,
    direct_template_catalog_json: &[u8],
    prepared_kernel_pack_digest: &str,
    destination: &std::path::Path,
    point_tile_size: u32,
    workspace_mib: u32,
    python_extraction_seconds: f64,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<NativeDirectLoweringResult> {
    let native_started = Instant::now();
    if std::fs::symlink_metadata(destination).is_ok() {
        return Err(invalid(format!(
            "recurrence direct-plan destination already exists: {}",
            destination.display()
        )));
    }

    validate_inventory(&input)?;
    let builder_input_digest = canonical_digest(&input)?;
    if builder_input_digest != input.declared_digest {
        return Err(invalid(format!(
            "recurrence builder input digest mismatch: declared {}, found {builder_input_digest}",
            input.declared_digest
        )));
    }
    let template_input_digest = prepared_template.canonical_digest()?;
    let expected_pack_digest =
        semantic_digest_from_hex(prepared_kernel_pack_digest, "prepared kernel pack digest")?;
    if prepared_template.prepared_kernel_pack_digest != expected_pack_digest {
        return Err(RusticolError::integrity(
            "recurrence prepared-kernel pack digest does not match the authenticated template",
        ));
    }

    let catalog_started = Instant::now();
    let direct_catalog = parse_direct_template_catalog(
        direct_template_catalog_json,
        expected_pack_digest,
        prepared_template.catalog_digest,
        prepared_template.compiled_model_digest,
    )?;
    let catalog_authentication_seconds = catalog_started.elapsed().as_secs_f64();

    let process = decode_process_input(&input)?.validate()?;
    let process_id = process.summary().process_id().to_owned();
    let strategy = process.summary().strategy();
    let semantic_digest = process.semantic_identity().process_digest();
    let template = prepared_template.into_core()?.validate()?;
    let authenticated = AuthenticatedRecurrenceBuilderInput::new(process, template)?;

    let semantic_started = Instant::now();
    let program = authenticated.build_with_progress(progress)?;
    let semantic_construction_seconds = semantic_started.elapsed().as_secs_f64();

    let direct_lowering_started = Instant::now();
    let runtime_options = DirectRecurrenceRuntimeOptions::new(point_tile_size, workspace_mib)?;
    let plan = lower_recurrence_direct_plan_v2(
        &program,
        authenticated.template(),
        &direct_catalog.catalog,
        semantic_digest,
        expected_pack_digest,
        direct_catalog.catalog_digest,
        runtime_options,
    )?;
    let direct_lowering_seconds = direct_lowering_started.elapsed().as_secs_f64();

    let resolved_helicities = resolved_helicities_from_direct_plan(&plan)?;
    let semantic_component_count = plan.currents().iter().try_fold(0_u64, |total, current| {
        total
            .checked_add(u64::from(current.component_count))
            .ok_or_else(|| invalid("recurrence semantic component count exceeds u64"))
    })?;

    let serialization_started = Instant::now();
    let metadata = write_recurrence_direct_plan_pacbin(destination, &plan)?;
    let serialization_seconds = serialization_started.elapsed().as_secs_f64();
    let native_total_seconds = native_started.elapsed().as_secs_f64();

    Ok(NativeDirectLoweringResult {
        builder_input_digest,
        template_input_digest,
        prepared_kernel_pack_digest: direct_catalog.prepared_kernel_pack_digest.to_string(),
        direct_template_catalog_digest: direct_catalog.catalog_digest.to_string(),
        process_id,
        strategy,
        semantic_digest: plan.semantic_digest().to_string(),
        runtime_layout_digest: plan.runtime_layout_digest().to_string(),
        member_count: metadata.member_count,
        unpacked_size_bytes: metadata.unpacked_size_bytes,
        index_sha256: hex_digest(metadata.index_sha256),
        container_size: metadata.container_size,
        plan_payload_size: metadata.plan_payload_size,
        plan_sha256: hex_digest(metadata.plan_sha256),
        current_count: plan.currents().len(),
        source_row_count: plan.sources().len(),
        contribution_count: plan.contributions().len(),
        finalization_count: plan.finalizations().len(),
        closure_count: plan.closures().len(),
        row_group_count: plan.row_groups().len(),
        momentum_form_count: plan.momentum_forms().len(),
        selector_domain_count: plan.selector_domains().len(),
        replay_target_count: plan.replay_targets().len(),
        amplitude_destination_count: plan.amplitude_destinations().len(),
        resolved_helicity_count: plan.resolved_helicities().len(),
        retained_helicity_count: plan.retained_helicity_count(),
        exact_factor_count: plan.exact_factors().len(),
        semantic_component_count,
        current_arena_components: plan.current_arena_components(),
        parameter_value_count: plan.parameter_value_count(),
        physical_sector_count: plan.physical_sector_count(),
        direct_executor_count: plan.direct_executor_count(),
        prepared_kernel_count: direct_catalog.prepared_kernel_count,
        resolved_helicities,
        timings: DirectLoweringTimings {
            python_extraction_seconds,
            catalog_authentication_seconds,
            semantic_construction_seconds,
            direct_lowering_seconds,
            serialization_seconds,
            native_total_seconds,
        },
    })
}

fn resolved_helicities_from_direct_plan(
    plan: &DirectRecurrencePlan,
) -> RusticolResult<Vec<Vec<i32>>> {
    let public = plan.public_helicities();
    plan.resolved_helicities()
        .iter()
        .enumerate()
        .map(|(index, row)| {
            if row.id != index as u32 {
                return Err(invalid(format!(
                    "direct resolved helicity {index} has non-canonical ID {}",
                    row.id
                )));
            }
            let start = usize::try_from(row.public_helicity_start)
                .map_err(|_| invalid("direct resolved-helicity offset exceeds usize"))?;
            let count = usize::try_from(row.public_helicity_count)
                .map_err(|_| invalid("direct resolved-helicity count exceeds usize"))?;
            let end = start
                .checked_add(count)
                .ok_or_else(|| invalid("direct resolved-helicity range exceeds usize"))?;
            public
                .get(start..end)
                .map(<[i32]>::to_vec)
                .ok_or_else(|| invalid("direct resolved-helicity range is out of bounds"))
        })
        .collect()
}

fn parse_direct_template_catalog(
    bytes: &[u8],
    expected_pack_digest: SemanticDigest,
    expected_template_catalog_digest: SemanticDigest,
    expected_compiled_model_digest: SemanticDigest,
) -> RusticolResult<AuthenticatedDirectTemplateCatalog> {
    if bytes.is_empty() {
        return Err(invalid("direct-template catalog JSON must not be empty"));
    }
    let value: JsonValue = serde_json::from_slice(bytes).map_err(|error| {
        invalid(format!(
            "direct-template catalog is not valid JSON: {error}"
        ))
    })?;
    let canonical = canonical_json_bytes(&value, "direct-template catalog")?;
    if canonical != bytes {
        return Err(invalid(
            "direct-template catalog JSON is not canonical ASCII JSON",
        ));
    }
    let object = json_object(&value, "direct-template catalog")?;
    require_json_fields(
        object,
        &[
            "abi",
            "backend",
            "backend_abi",
            "canonicalization_abi",
            "catalog_digest",
            "compiled_model_digest",
            "optimization_level",
            "optimization_settings_digest",
            "portable",
            "prepared_kernel_contract_digest",
            "prepared_kernel_pack_digest",
            "prepared_kernel_payload_digest",
            "recurrence_template_catalog_digest",
            "target_triple",
            "templates",
        ],
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "abi",
        RECURRENCE_DIRECT_TEMPLATE_ABI,
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "backend_abi",
        DIRECT_BACKEND_ABI,
        "direct-template catalog",
    )?;
    require_json_string_value(
        object,
        "canonicalization_abi",
        DIRECT_CANONICALIZATION_ABI,
        "direct-template catalog",
    )?;

    let catalog_digest = json_sha256(object, "catalog_digest", "direct-template catalog digest")?;
    let actual_catalog_digest = digest_json_without_field(
        &value,
        "catalog_digest",
        "direct-template catalog semantic payload",
    )?;
    if catalog_digest != actual_catalog_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog digest does not match its canonical payload",
        ));
    }
    let prepared_kernel_pack_digest = json_sha256(
        object,
        "prepared_kernel_pack_digest",
        "direct-template prepared-kernel pack digest",
    )?;
    if prepared_kernel_pack_digest != expected_pack_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog prepared-kernel pack digest does not match the requested pack",
        ));
    }
    let recurrence_template_catalog_digest = json_sha256(
        object,
        "recurrence_template_catalog_digest",
        "direct-template semantic catalog digest",
    )?;
    if recurrence_template_catalog_digest != expected_template_catalog_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog does not match the authenticated recurrence template",
        ));
    }
    let compiled_model_digest = json_sha256(
        object,
        "compiled_model_digest",
        "direct-template compiled-model digest",
    )?;
    if compiled_model_digest != expected_compiled_model_digest {
        return Err(RusticolError::integrity(
            "direct-template catalog does not match the authenticated compiled model",
        ));
    }
    for (field, context) in [
        (
            "prepared_kernel_contract_digest",
            "direct-template prepared-kernel contract digest",
        ),
        (
            "prepared_kernel_payload_digest",
            "direct-template prepared-kernel payload digest",
        ),
        (
            "optimization_settings_digest",
            "direct-template optimization-settings digest",
        ),
    ] {
        json_sha256(object, field, context)?;
    }

    let backend = json_string(object, "backend", "direct-template backend")?;
    if !matches!(backend, "jit" | "cpp" | "asm") {
        return Err(invalid(format!(
            "direct-template catalog has unsupported backend {backend:?}"
        )));
    }
    let target_triple =
        json_nonempty_string(object, "target_triple", "direct-template target triple")?;
    let portable = json_bool(object, "portable", "direct-template portable flag")?;
    let optimization_level = json_u32(
        object,
        "optimization_level",
        "direct-template optimization level",
    )?;
    match backend {
        "jit" if !portable || optimization_level != 2 => {
            return Err(invalid(
                "prepared direct JIT catalogs must use portable SymJIT O2",
            ));
        }
        "cpp" | "asm" if portable => {
            return Err(invalid(
                "prepared direct C++/ASM catalogs must be target-native",
            ));
        }
        _ => {}
    }

    let templates = json_array(object, "templates", "direct-template templates")?;
    if templates.is_empty() {
        return Err(invalid(
            "direct-template executor catalog must not be empty",
        ));
    }
    let mut bindings = Vec::with_capacity(templates.len());
    let mut prepared_kernel_ids = BTreeSet::new();
    let mut identity_finalizer_seen = false;
    for (expected_executor_id, template_value) in templates.iter().enumerate() {
        let context = format!("direct template {expected_executor_id}");
        let template = json_object(template_value, &context)?;
        require_json_fields(
            template,
            &[
                "abi",
                "alignment_bytes",
                "backend",
                "coupling_slot_count",
                "destination_aliasing",
                "destination_component_count",
                "destination_operation",
                "direct_executor_id",
                "evaluator_binding_id",
                "evaluator_resolver_key",
                "exact_expression_digest",
                "momentum_operand_count",
                "optimization_level",
                "parameter_slot_count",
                "parent_arity",
                "parent_component_counts",
                "payload_binding",
                "portable",
                "role",
                "semantic_digest",
                "semantic_template_ids",
                "simd_axis",
                "target_triple",
                "template_id",
            ],
            &context,
        )?;
        require_json_string_value(template, "abi", RECURRENCE_DIRECT_TEMPLATE_ABI, &context)?;
        let actual_template_digest =
            digest_json_without_field(template_value, "semantic_digest", &context)?;
        let template_digest = json_sha256(
            template,
            "semantic_digest",
            &format!("{context} semantic digest"),
        )?;
        if template_digest != actual_template_digest {
            return Err(RusticolError::integrity(format!(
                "{context} semantic digest does not match its canonical payload"
            )));
        }

        let direct_executor_id = json_u32(
            template,
            "direct_executor_id",
            &format!("{context} executor ID"),
        )?;
        let expected_executor_id = u32::try_from(expected_executor_id)
            .map_err(|_| invalid("direct-template executor count exceeds u32"))?;
        if direct_executor_id != expected_executor_id {
            return Err(invalid(format!(
                "{context} has executor ID {direct_executor_id}, expected dense ID {expected_executor_id}"
            )));
        }
        let evaluator_binding_id = json_u32(
            template,
            "evaluator_binding_id",
            &format!("{context} evaluator-binding ID"),
        )?;
        let role_text = json_string(template, "role", &format!("{context} role"))?;
        let role = direct_executor_role(role_text, &context)?;
        let expected_operation = match role {
            DirectExecutorRole::Source => "initialize",
            DirectExecutorRole::Contribution => "add",
            DirectExecutorRole::Finalization => "finalize-in-place",
            DirectExecutorRole::Closure => "closure-add",
        };
        require_json_string_value(
            template,
            "destination_operation",
            expected_operation,
            &context,
        )?;
        require_json_string_value(template, "backend", backend, &context)?;
        require_json_string_value(template, "target_triple", target_triple, &context)?;
        if json_bool(template, "portable", &format!("{context} portable flag"))? != portable
            || json_u32(
                template,
                "optimization_level",
                &format!("{context} optimization level"),
            )? != optimization_level
        {
            return Err(invalid(format!(
                "{context} backend policy does not match its catalog"
            )));
        }
        require_json_string_value(template, "simd_axis", "points-contiguous", &context)?;
        let destination_aliasing = json_bool(
            template,
            "destination_aliasing",
            &format!("{context} destination aliasing"),
        )?;
        if destination_aliasing != (role == DirectExecutorRole::Finalization) {
            return Err(invalid(format!(
                "{context} has an invalid destination-aliasing contract"
            )));
        }
        json_nonempty_string(template, "template_id", &format!("{context} template ID"))?;
        json_nonempty_string(
            template,
            "evaluator_resolver_key",
            &format!("{context} evaluator resolver key"),
        )?;
        json_sha256(
            template,
            "exact_expression_digest",
            &format!("{context} exact-expression digest"),
        )?;
        validate_direct_template_shapes(template, &context)?;

        let payload = json_object(
            json_field(template, "payload_binding", &context)?,
            &format!("{context} payload binding"),
        )?;
        require_json_fields(
            payload,
            &[
                "abi",
                "destination_operation",
                "direct_application_abi",
                "exact_factor_scalar_slots",
                "input_plane_count",
                "input_plane_projections",
                "kind",
                "output_alias_inputs",
                "parameter_bindings",
                "payload_digest",
                "payload_paths",
                "prepared_kernel_id",
                "prepared_template_semantic_digest",
                "role",
                "runtime_template",
                "scalar_input_count",
                "scalar_projections",
                "source_application_abi",
                "source_application_path",
                "source_application_sha256",
                "state_plane_indices",
            ],
            &format!("{context} payload binding"),
        )?;
        require_json_string_value(
            payload,
            "abi",
            DIRECT_PAYLOAD_BINDING_ABI,
            &format!("{context} payload binding"),
        )?;
        json_sha256(
            payload,
            "payload_digest",
            &format!("{context} payload digest"),
        )?;
        let payload_kind = json_string(payload, "kind", &format!("{context} payload kind"))?;
        if !matches!(
            payload_kind,
            "rusticol-intrinsic" | "prepared-direct-call" | "pending-direct-call-abi"
        ) {
            return Err(invalid(format!(
                "{context} has unsupported payload kind {payload_kind:?}"
            )));
        }
        if payload_kind == "pending-direct-call-abi" {
            return Err(RusticolError::compatibility(format!(
                "{context} has no executable Direct-Arena payload; rebuild the prepared model"
            )));
        }
        let runtime_template = json_optional_string(
            payload,
            "runtime_template",
            &format!("{context} runtime template"),
        )?;
        let prepared_kernel_id = json_optional_u32(
            payload,
            "prepared_kernel_id",
            &format!("{context} prepared-kernel ID"),
        )?;
        validate_string_array(
            json_array(
                payload,
                "payload_paths",
                &format!("{context} payload paths"),
            )?,
            &format!("{context} payload paths"),
        )?;

        let identity_finalizer =
            runtime_template.is_some_and(|name| name == DIRECT_IDENTITY_FINALIZER);
        if identity_finalizer {
            if identity_finalizer_seen {
                return Err(invalid(
                    "direct-template catalog contains more than one generic identity finalizer",
                ));
            }
            if role != DirectExecutorRole::Finalization
                || payload_kind != "rusticol-intrinsic"
                || prepared_kernel_id.is_some()
            {
                return Err(invalid(format!(
                    "{context} has an invalid identity-finalizer contract"
                )));
            }
            identity_finalizer_seen = true;
            bindings.push(PreparedDirectExecutorBinding::identity_finalizer(
                direct_executor_id,
            ));
        } else {
            if let Some(kernel_id) = prepared_kernel_id {
                prepared_kernel_ids.insert(kernel_id);
            }
            bindings.push(PreparedDirectExecutorBinding::evaluator(
                role,
                evaluator_binding_id,
                direct_executor_id,
            ));
        }
    }

    let catalog = PreparedDirectExecutorCatalog::new(catalog_digest, bindings)?;
    Ok(AuthenticatedDirectTemplateCatalog {
        catalog,
        catalog_digest,
        prepared_kernel_pack_digest,
        prepared_kernel_count: prepared_kernel_ids.len(),
    })
}

fn validate_direct_template_shapes(
    template: &JsonMap<String, JsonValue>,
    context: &str,
) -> RusticolResult<()> {
    let parent_arity = json_u32(template, "parent_arity", &format!("{context} parent arity"))?;
    let parent_counts = json_array(
        template,
        "parent_component_counts",
        &format!("{context} parent component counts"),
    )?;
    if parent_counts.len() != parent_arity as usize {
        return Err(invalid(format!(
            "{context} parent component counts do not match parent arity"
        )));
    }
    for (index, value) in parent_counts.iter().enumerate() {
        let count = json_value_u32(value, &format!("{context} parent component {index}"))?;
        if count == 0 {
            return Err(invalid(format!(
                "{context} parent component {index} is empty"
            )));
        }
    }
    for field in [
        "coupling_slot_count",
        "momentum_operand_count",
        "parameter_slot_count",
    ] {
        json_u32(template, field, &format!("{context} {field}"))?;
    }
    if json_u32(
        template,
        "destination_component_count",
        &format!("{context} destination component count"),
    )? == 0
    {
        return Err(invalid(format!(
            "{context} destination component count must be positive"
        )));
    }
    let alignment = json_u32(template, "alignment_bytes", &format!("{context} alignment"))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(format!(
            "{context} alignment must be a positive power of two"
        )));
    }
    validate_string_array(
        json_array(
            template,
            "semantic_template_ids",
            &format!("{context} semantic template IDs"),
        )?,
        &format!("{context} semantic template IDs"),
    )
}

fn direct_executor_role(value: &str, context: &str) -> RusticolResult<DirectExecutorRole> {
    match value {
        "source" => Ok(DirectExecutorRole::Source),
        "contribution" => Ok(DirectExecutorRole::Contribution),
        "finalization" => Ok(DirectExecutorRole::Finalization),
        "closure" => Ok(DirectExecutorRole::Closure),
        _ => Err(invalid(format!(
            "{context} has unsupported direct role {value:?}"
        ))),
    }
}

fn canonical_json_bytes(value: &JsonValue, context: &str) -> RusticolResult<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|error| invalid(format!("could not canonicalize {context}: {error}")))
}

fn digest_json_without_field(
    value: &JsonValue,
    field: &str,
    context: &str,
) -> RusticolResult<SemanticDigest> {
    let mut semantic = value.clone();
    let object = semantic
        .as_object_mut()
        .ok_or_else(|| invalid(format!("{context} must be a JSON object")))?;
    if object.remove(field).is_none() {
        return Err(invalid(format!("{context} has no {field:?} field")));
    }
    let digest: [u8; 32] = Sha256::digest(canonical_json_bytes(&semantic, context)?).into();
    SemanticDigest::new(digest)
}

fn json_object<'a>(
    value: &'a JsonValue,
    context: &str,
) -> RusticolResult<&'a JsonMap<String, JsonValue>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{context} must be a JSON object")))
}

fn require_json_fields(
    object: &JsonMap<String, JsonValue>,
    expected: &[&str],
    context: &str,
) -> RusticolResult<()> {
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual == expected {
        return Ok(());
    }
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    Err(invalid(format!(
        "{context} fields do not match direct-template-v1; missing={missing:?}, unexpected={unexpected:?}"
    )))
}

fn json_field<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a JsonValue> {
    object
        .get(field)
        .ok_or_else(|| invalid(format!("{context} has no {field:?} field")))
}

fn json_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a str> {
    json_field(object, field, context)?
        .as_str()
        .ok_or_else(|| invalid(format!("{context} must be a string")))
}

fn json_nonempty_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a str> {
    let value = json_string(object, field, context)?;
    if value.is_empty() {
        return Err(invalid(format!("{context} must not be empty")));
    }
    Ok(value)
}

fn require_json_string_value(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    expected: &str,
    context: &str,
) -> RusticolResult<()> {
    let actual = json_string(object, field, context)?;
    if actual != expected {
        return Err(invalid(format!(
            "{context} {field:?} is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn json_bool(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<bool> {
    json_field(object, field, context)?
        .as_bool()
        .ok_or_else(|| invalid(format!("{context} must be a boolean")))
}

fn json_u32(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<u32> {
    json_value_u32(json_field(object, field, context)?, context)
}

fn json_value_u32(value: &JsonValue, context: &str) -> RusticolResult<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative u32")))
}

fn json_optional_u32(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Option<u32>> {
    let value = json_field(object, field, context)?;
    if value.is_null() {
        return Ok(None);
    }
    json_value_u32(value, context).map(Some)
}

fn json_optional_string<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Option<&'a str>> {
    let value = json_field(object, field, context)?;
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(Some)
        .ok_or_else(|| invalid(format!("{context} must be null or a nonempty string")))
}

fn json_array<'a>(
    object: &'a JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<&'a [JsonValue]> {
    json_field(object, field, context)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("{context} must be a JSON array")))
}

fn validate_string_array(values: &[JsonValue], context: &str) -> RusticolResult<()> {
    if values
        .iter()
        .all(|value| value.as_str().is_some_and(|value| !value.is_empty()))
    {
        return Ok(());
    }
    Err(invalid(format!(
        "{context} must contain only nonempty strings"
    )))
}

fn json_sha256(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<SemanticDigest> {
    semantic_digest_from_hex(json_string(object, field, context)?, context)
}

fn report_recurrence_build_progress(
    callback: Option<&Py<PyAny>>,
    progress: RecurrenceBuildProgress,
) -> RusticolResult<()> {
    let Some(callback) = callback else {
        return Ok(());
    };
    Python::attach(|py| -> PyResult<()> {
        let payload = PyDict::new(py);
        payload.set_item("step", progress.phase)?;
        payload.set_item("phase_index", progress.phase_index)?;
        payload.set_item("phase_total", progress.phase_total)?;
        payload.set_item("stage_total", progress.stage_total)?;
        payload.set_item(
            "candidate_parent_pair_count",
            progress.candidate_parent_pair_count,
        )?;
        payload.set_item("current_count", progress.current_count)?;
        payload.set_item("contribution_count", progress.contribution_count)?;
        payload.set_item(
            "dynamic_color_state_count",
            progress.dynamic_color_state_count,
        )?;
        payload.set_item(
            "color_target_prune_count",
            progress.color_target_prune_count,
        )?;
        if let Some(stage_index) = progress.stage_index {
            payload.set_item("stage_index", stage_index)?;
        }
        if let Some(subset_size) = progress.subset_size {
            payload.set_item("subset_size", subset_size)?;
        }
        if let Some(total) = progress.candidate_parent_pair_total {
            payload.set_item("candidate_parent_pair_total", total)?;
        }
        callback.bind(py).call1((payload,))?;
        Ok(())
    })
    .map_err(|error| invalid(format!("recurrence progress callback failed: {error}")))
}

fn direct_lowering_mapping(
    py: Python<'_>,
    native: NativeDirectLoweringResult,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("kind", DIRECT_LOWERING_RESULT_KIND)?;
    result.set_item("schema_version", DIRECT_LOWERING_RESULT_SCHEMA_VERSION)?;
    result.set_item("builder_input_abi", DIRECT_BUILDER_INPUT_ABI)?;
    result.set_item("builder_input_sha256", native.builder_input_digest)?;
    result.set_item(
        "template_input_abi",
        template::RECURRENCE_TEMPLATE_INPUT_ABI,
    )?;
    result.set_item("template_input_sha256", native.template_input_digest)?;
    result.set_item(
        "prepared_kernel_pack_digest",
        native.prepared_kernel_pack_digest,
    )?;
    result.set_item("direct_template_abi", RECURRENCE_DIRECT_TEMPLATE_ABI)?;
    result.set_item(
        "direct_template_catalog_digest",
        native.direct_template_catalog_digest,
    )?;
    result.set_item("recurrence_plan_abi", RECURRENCE_DIRECT_PLAN_ABI)?;
    result.set_item("runtime_layout_abi", RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI)?;
    result.set_item(
        "required_runtime_capabilities",
        PyList::new(
            py,
            [
                RECURRENCE_LC_COLOR_CAPABILITY,
                RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
            ],
        )?,
    )?;

    let container = PyDict::new(py);
    container.set_item("kind", RUNTIME_CONTAINER_KIND)?;
    container.set_item("schema_version", RUNTIME_CONTAINER_SCHEMA_VERSION)?;
    container.set_item("storage_abi", STORAGE_ABI)?;
    container.set_item("member_count", native.member_count)?;
    container.set_item("unpacked_size_bytes", native.unpacked_size_bytes)?;
    container.set_item("index_sha256", native.index_sha256)?;
    result.set_item("runtime_container", container)?;

    let inspection = PyDict::new(py);
    inspection.set_item("execution_mode", "recurrence")?;
    inspection.set_item("recurrence_plan_abi", RECURRENCE_DIRECT_PLAN_ABI)?;
    inspection.set_item("runtime_layout_abi", RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI)?;
    inspection.set_item("direct_template_abi", RECURRENCE_DIRECT_TEMPLATE_ABI)?;
    inspection.set_item("process_id", native.process_id)?;
    inspection.set_item("lc_flow_layout", native.strategy.as_str())?;
    inspection.set_item("prepared_kernel_count", native.prepared_kernel_count)?;
    inspection.set_item("parameter_count", native.parameter_value_count)?;
    inspection.set_item("sector_count", native.physical_sector_count)?;
    inspection.set_item("direct_executor_count", native.direct_executor_count)?;
    inspection.set_item("semantic_digest", native.semantic_digest)?;
    inspection.set_item("runtime_layout_digest", native.runtime_layout_digest)?;

    let schedule = PyDict::new(py);
    schedule.set_item("current_count", native.current_count)?;
    schedule.set_item("source_row_count", native.source_row_count)?;
    schedule.set_item("contribution_count", native.contribution_count)?;
    schedule.set_item("finalization_count", native.finalization_count)?;
    schedule.set_item("closure_term_count", native.closure_count)?;
    schedule.set_item(
        "amplitude_destination_count",
        native.amplitude_destination_count,
    )?;
    schedule.set_item("replay_target_count", native.replay_target_count)?;
    schedule.set_item("resolved_helicity_count", native.resolved_helicity_count)?;
    schedule.set_item("retained_helicity_count", native.retained_helicity_count)?;
    schedule.set_item("exact_factor_count", native.exact_factor_count)?;
    inspection.set_item("schedule", schedule)?;

    let direct_arena = PyDict::new(py);
    direct_arena.set_item("semantic_component_count", native.semantic_component_count)?;
    direct_arena.set_item("current_arena_components", native.current_arena_components)?;
    direct_arena.set_item(
        "arena_component_reuse_count",
        native
            .semantic_component_count
            .saturating_sub(u64::from(native.current_arena_components)),
    )?;
    direct_arena.set_item("momentum_form_count", native.momentum_form_count)?;
    direct_arena.set_item("selector_domain_count", native.selector_domain_count)?;
    direct_arena.set_item("row_group_count", native.row_group_count)?;
    direct_arena.set_item("packed_input_bytes", 0)?;
    direct_arena.set_item("packed_output_bytes", 0)?;
    direct_arena.set_item("scatter_bytes", 0)?;
    inspection.set_item("direct_arena", direct_arena)?;

    let member = PyDict::new(py);
    member.set_item("path", RECURRENCE_DIRECT_PLAN_MEMBER)?;
    member.set_item("size_bytes", native.plan_payload_size)?;
    member.set_item("sha256", native.plan_sha256)?;
    member.set_item("container_size_bytes", native.container_size)?;
    inspection.set_item("runtime_container_member", member)?;

    let timings = PyDict::new(py);
    timings.set_item(
        "python_extraction",
        native.timings.python_extraction_seconds,
    )?;
    timings.set_item(
        "catalog_authentication",
        native.timings.catalog_authentication_seconds,
    )?;
    timings.set_item(
        "semantic_construction",
        native.timings.semantic_construction_seconds,
    )?;
    timings.set_item("direct_lowering", native.timings.direct_lowering_seconds)?;
    timings.set_item("serialization", native.timings.serialization_seconds)?;
    timings.set_item("native_total", native.timings.native_total_seconds)?;
    inspection.set_item("generation_timings_seconds", timings)?;

    result.set_item("inspection_summary", inspection)?;
    result.set_item("resolved_helicities", native.resolved_helicities)?;
    Ok(result.into_any().unbind())
}

fn parse_input(input: &Bound<'_, PyAny>) -> PyResult<OwnedInput> {
    if cfg!(not(target_endian = "little")) {
        return Err(PyValueError::new_err(
            "recurrence builder input v2 requires a little-endian target",
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

    let pairing_table_objects = if input.hasattr("fermion_pairing_tables")? {
        iterable_attribute(
            input,
            "fermion_pairing_tables",
            "recurrence fermion-pairing tables",
        )?
    } else {
        Vec::new()
    };
    if !pairing_table_objects.is_empty()
        && pairing_table_objects.len() != FERMION_PAIRING_TABLE_SPECS.len()
    {
        return Err(PyValueError::new_err(format!(
            "recurrence fermion-pairing table inventory has {} tables, expected {}",
            pairing_table_objects.len(),
            FERMION_PAIRING_TABLE_SPECS.len()
        )));
    }
    let mut fermion_pairing_tables = Vec::with_capacity(pairing_table_objects.len());
    let mut fermion_pairing_table_by_name = BTreeMap::new();
    for (table_object, spec) in pairing_table_objects
        .into_iter()
        .zip(FERMION_PAIRING_TABLE_SPECS)
    {
        let table_name = required_nonempty_string(&table_object, "name", "table name")?;
        if table_name != spec.name {
            return Err(PyValueError::new_err(format!(
                "recurrence fermion-pairing table inventory mismatch: found {table_name:?}, expected {:?}",
                spec.name
            )));
        }
        let row_count = table_object
            .getattr("row_count")?
            .extract::<u64>()
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "recurrence fermion-pairing table {table_name:?} row_count must be u64"
                ))
            })?;
        let column_objects = iterable_attribute(
            &table_object,
            "columns",
            &format!("recurrence fermion-pairing table {table_name:?} columns"),
        )?;
        if column_objects.len() != spec.columns.len() {
            return Err(PyValueError::new_err(format!(
                "recurrence fermion-pairing table {table_name:?} has {} columns, expected {}",
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
                    "recurrence fermion-pairing table {table_name:?} column mismatch: found {column_name:?}, expected {:?}",
                    column_spec.name
                )));
            }
            let context = format!("fermion_pairing.{table_name}.{column_name}");
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
        fermion_pairing_table_by_name.insert(table_name.clone(), fermion_pairing_tables.len());
        fermion_pairing_tables.push(OwnedTable {
            name: table_name,
            row_count,
            columns,
            column_by_name,
        });
    }
    let declared_fermion_pairing_digest = if fermion_pairing_tables.is_empty() {
        None
    } else {
        if !input.hasattr("fermion_pairing_digest")? {
            return Err(PyValueError::new_err(
                "recurrence fermion-pairing tables require a canonical digest",
            ));
        }
        let value = input.getattr("fermion_pairing_digest")?;
        if value.is_none() {
            return Err(PyValueError::new_err(
                "recurrence fermion-pairing tables require a canonical digest",
            ));
        }
        let digest = value.extract::<String>().map_err(|_| {
            PyTypeError::new_err("recurrence fermion-pairing digest must be a string")
        })?;
        validate_sha256_text(&digest, "recurrence fermion-pairing digest")?;
        Some(digest)
    };

    Ok(OwnedInput {
        abi,
        declared_digest,
        tables,
        table_by_name,
        declared_fermion_pairing_digest,
        fermion_pairing_tables,
        fermion_pairing_table_by_name,
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
            declared_fermion_pairing_digest: None,
            fermion_pairing_tables: Vec::new(),
            fermion_pairing_table_by_name: BTreeMap::new(),
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

fn decode_process_input(
    input: &OwnedInput,
) -> RusticolResult<process::OwnedRecurrenceProcessInput> {
    let fermion_pairing = decode_fermion_pairing_input(input)?;
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
        fermion_pairing,
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

fn decode_fermion_pairing_input(
    input: &OwnedInput,
) -> RusticolResult<Option<process::OwnedFermionPairingInput>> {
    let Some(declared_digest) = input.declared_fermion_pairing_digest.as_deref() else {
        if !input.fermion_pairing_tables.is_empty() {
            return Err(invalid(
                "recurrence fermion-pairing tables have no declared digest",
            ));
        }
        return Ok(None);
    };
    if input.fermion_pairing_tables.is_empty() {
        return Err(invalid(
            "recurrence fermion-pairing digest has no fixed-width tables",
        ));
    }

    let header = decode_pairing_rows(input, "header", |row| {
        Ok(process::FermionPairingHeaderRow {
            schema_version: input.pairing_u32("header", "schema_version")?[row],
            abi_string_id: input.pairing_u32("header", "abi_string_id")?[row],
            process_key_string_id: input.pairing_u32("header", "process_key_string_id")?[row],
            proof_algorithm_string_id: input.pairing_u32("header", "proof_algorithm_string_id")?
                [row],
            source_count: input.pairing_u32("header", "source_count")?[row],
            endpoint_count: input.pairing_u32("header", "endpoint_count")?[row],
            pairing_class_count: input.pairing_u32("header", "pairing_class_count")?[row],
            rule_count: input.pairing_u32("header", "rule_count")?[row],
            endpoint_state_template_count: input
                .pairing_u64("header", "endpoint_state_template_count")?[row],
            endpoint_anti_state_template_count: input
                .pairing_u64("header", "endpoint_anti_state_template_count")?[row],
            endpoint_basis_count: input.pairing_u64("header", "endpoint_basis_count")?[row],
            endpoint_color_representation_count: input
                .pairing_u64("header", "endpoint_color_representation_count")?[row],
            class_fundamental_slot_count: input
                .pairing_u64("header", "class_fundamental_slot_count")?[row],
            class_antifundamental_slot_count: input
                .pairing_u64("header", "class_antifundamental_slot_count")?[row],
            class_reference_pairing_count: input
                .pairing_u64("header", "class_reference_pairing_count")?[row],
            rule_class_pairing_index_count: input
                .pairing_u64("header", "rule_class_pairing_index_count")?[row],
            rule_endpoint_pairing_count: input
                .pairing_u64("header", "rule_endpoint_pairing_count")?[row],
            rule_source_permutation_count: input
                .pairing_u64("header", "rule_source_permutation_count")?[row],
            rule_lineage_count: input.pairing_u64("header", "rule_lineage_count")?[row],
            exact_integer_count: input.pairing_u32("header", "exact_integer_count")?[row],
            exact_integer_limb_count: input.pairing_u64("header", "exact_integer_limb_count")?[row],
            string_count: input.pairing_u32("header", "string_count")?[row],
            string_byte_count: input.pairing_u64("header", "string_byte_count")?[row],
            no_fermion_line: input.pairing_u32("header", "no_fermion_line")?[row],
            topology_digest: pairing_digest_row(input, "header", "topology_digest", row)?,
            semantic_digest: pairing_digest_row(input, "header", "semantic_digest", row)?,
        })
    })?;
    let endpoints = decode_pairing_rows(input, "endpoints", |row| {
        Ok(process::FermionPairingEndpointRow {
            endpoint_id: input.pairing_u32("endpoints", "endpoint_id")?[row],
            source_slot: input.pairing_u32("endpoints", "source_slot")?[row],
            public_label: input.pairing_u32("endpoints", "public_label")?[row],
            species_class_id: input.pairing_u32("endpoints", "species_class_id")?[row],
            species_string_id: input.pairing_u32("endpoints", "species_string_id")?[row],
            particle_orientation: input.pairing_u8("endpoints", "particle_orientation")?[row],
            color_orientation: input.pairing_u8("endpoints", "color_orientation")?[row],
            state_template_range: pairing_range(input, "endpoints", "state_template", row)?,
            anti_state_template_range: pairing_range(
                input,
                "endpoints",
                "anti_state_template",
                row,
            )?,
            basis_range: pairing_range(input, "endpoints", "basis", row)?,
            color_representation_range: pairing_range(
                input,
                "endpoints",
                "color_representation",
                row,
            )?,
            contract_digest: pairing_digest_row(input, "endpoints", "contract_digest", row)?,
        })
    })?;
    let pairing_classes = decode_pairing_rows(input, "pairing_classes", |row| {
        Ok(process::FermionPairingClassRow {
            class_id: input.pairing_u32("pairing_classes", "class_id")?[row],
            species_class_id: input.pairing_u32("pairing_classes", "species_class_id")?[row],
            species_string_id: input.pairing_u32("pairing_classes", "species_string_id")?[row],
            fundamental_slot_range: pairing_range(
                input,
                "pairing_classes",
                "fundamental_slot",
                row,
            )?,
            antifundamental_slot_range: pairing_range(
                input,
                "pairing_classes",
                "antifundamental_slot",
                row,
            )?,
            reference_pairing_range: pairing_range(
                input,
                "pairing_classes",
                "reference_pairing",
                row,
            )?,
            pairing_count: input.pairing_u64("pairing_classes", "pairing_count")?[row],
            proof_digest: pairing_digest_row(input, "pairing_classes", "proof_digest", row)?,
        })
    })?;
    let rules = decode_pairing_rows(input, "rules", |row| {
        Ok(process::FermionPairingRuleRow {
            rule_id: input.pairing_u32("rules", "rule_id")?[row],
            class_pairing_index_range: pairing_range(input, "rules", "class_pairing_index", row)?,
            endpoint_pairing_range: pairing_range(input, "rules", "endpoint_pairing", row)?,
            source_permutation_range: pairing_range(input, "rules", "source_permutation", row)?,
            lineage_range: pairing_range(input, "rules", "lineage", row)?,
            fermion_parity: input.pairing_i32("rules", "fermion_parity")?[row],
            real_numerator_integer_id: input.pairing_u32("rules", "real_numerator_integer_id")?
                [row],
            real_denominator_integer_id: input
                .pairing_u32("rules", "real_denominator_integer_id")?[row],
            imag_numerator_integer_id: input.pairing_u32("rules", "imag_numerator_integer_id")?
                [row],
            imag_denominator_integer_id: input
                .pairing_u32("rules", "imag_denominator_integer_id")?[row],
            multiplicity: input.pairing_u64("rules", "multiplicity")?[row],
            proof_algorithm_string_id: input.pairing_u32("rules", "proof_algorithm_string_id")?
                [row],
            proof_digest: pairing_digest_row(input, "rules", "proof_digest", row)?,
        })
    })?;
    let rule_class_pairing_indices =
        decode_pairing_rows(input, "rule_class_pairing_indices", |row| {
            Ok(process::FermionPairingClassPairingIndexRow {
                class_id: input.pairing_u32("rule_class_pairing_indices", "class_id")?[row],
                pairing_index: input.pairing_u64("rule_class_pairing_indices", "pairing_index")?
                    [row],
            })
        })?;
    let decode_pairs = |table_name: &str| {
        decode_pairing_rows(input, table_name, |row| {
            Ok(process::FermionPairingEndpointPairRow {
                fundamental_source_slot: input
                    .pairing_u32(table_name, "fundamental_source_slot")?[row],
                antifundamental_source_slot: input
                    .pairing_u32(table_name, "antifundamental_source_slot")?[row],
            })
        })
    };
    let exact_integers = decode_pairing_rows(input, "exact_integers", |row| {
        Ok(process::FermionPairingExactIntegerRow {
            integer_id: input.pairing_u32("exact_integers", "integer_id")?[row],
            sign: input.pairing_i32("exact_integers", "sign")?[row],
            limb_range: pairing_range(input, "exact_integers", "limb", row)?,
        })
    })?;

    Ok(Some(process::OwnedFermionPairingInput {
        input_abi: process::RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI.to_owned(),
        declared_columnar_digest: semantic_digest_from_hex(
            declared_digest,
            "recurrence fermion-pairing digest",
        )?,
        header,
        endpoints,
        endpoint_state_template_ids: input
            .pairing_u32("endpoint_state_template_ids", "string_id")?
            .to_vec(),
        endpoint_anti_state_template_ids: input
            .pairing_u32("endpoint_anti_state_template_ids", "string_id")?
            .to_vec(),
        endpoint_basis_ids: input
            .pairing_u32("endpoint_basis_ids", "string_id")?
            .to_vec(),
        endpoint_color_representations: input
            .pairing_i32("endpoint_color_representations", "value")?
            .to_vec(),
        pairing_classes,
        class_fundamental_slots: input
            .pairing_u32("class_fundamental_slots", "source_slot")?
            .to_vec(),
        class_antifundamental_slots: input
            .pairing_u32("class_antifundamental_slots", "source_slot")?
            .to_vec(),
        class_reference_pairings: decode_pairs("class_reference_pairings")?,
        rules,
        rule_class_pairing_indices,
        rule_endpoint_pairings: decode_pairs("rule_endpoint_pairings")?,
        rule_source_slot_permutations: input
            .pairing_u32("rule_source_slot_permutations", "source_slot")?
            .to_vec(),
        rule_lineages: input.pairing_u32("rule_lineages", "line_id")?.to_vec(),
        exact_integers,
        exact_integer_limbs: input.pairing_u64("exact_integer_limbs", "value")?.to_vec(),
        string_ranges: pairing_plain_ranges(input, "string_ranges")?,
        string_bytes: input.pairing_u8("string_bytes", "value")?.to_vec(),
    }))
}

fn decode_pairing_rows<T>(
    input: &OwnedInput,
    table_name: &str,
    mut decode: impl FnMut(usize) -> RusticolResult<T>,
) -> RusticolResult<Vec<T>> {
    let row_count = checked_usize(
        input.fermion_pairing_table(table_name)?.row_count,
        &format!("fermion-pairing {table_name} row count"),
    )?;
    (0..row_count).map(&mut decode).collect()
}

fn pairing_range(
    input: &OwnedInput,
    table_name: &str,
    prefix: &str,
    row: usize,
) -> RusticolResult<CheckedTableRange> {
    Ok(CheckedTableRange {
        start: input.pairing_u64(table_name, &format!("{prefix}_start"))?[row],
        count: input.pairing_u64(table_name, &format!("{prefix}_count"))?[row],
    })
}

fn pairing_plain_ranges(
    input: &OwnedInput,
    table_name: &str,
) -> RusticolResult<Vec<CheckedTableRange>> {
    decode_pairing_rows(input, table_name, |row| {
        Ok(CheckedTableRange {
            start: input.pairing_u64(table_name, "start")?[row],
            count: input.pairing_u64(table_name, "count")?[row],
        })
    })
}

fn pairing_digest_row(
    input: &OwnedInput,
    table_name: &str,
    column_name: &str,
    row: usize,
) -> RusticolResult<[u8; 32]> {
    let values = input.pairing_u8(table_name, column_name)?;
    let start = row
        .checked_mul(32)
        .ok_or_else(|| invalid("fermion-pairing digest offset exceeds usize"))?;
    values
        .get(start..start + 32)
        .ok_or_else(|| invalid("fermion-pairing digest row is truncated"))?
        .try_into()
        .map_err(|_| invalid("fermion-pairing digest row must contain 32 bytes"))
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
                input_port_pairing_sequence_id: table.u32("input_port_pairing_sequence_id")?[row],
                result_port_binding_sequence_id: table.u32("result_port_binding_sequence_id")?[row],
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

#[cfg(test)]
mod direct_binding_tests {
    use super::*;
    use serde_json::json;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn refresh_digest(value: &mut JsonValue, field: &str, context: &str) {
        value.as_object_mut().unwrap().remove(field);
        let digest: [u8; 32] = Sha256::digest(canonical_json_bytes(value, context).unwrap()).into();
        value
            .as_object_mut()
            .unwrap()
            .insert(field.to_owned(), json!(hex_digest(digest)));
    }

    fn canonical_direct_catalog() -> (Vec<u8>, SemanticDigest, SemanticDigest, SemanticDigest) {
        let prepared_pack = semantic_digest_from_hex(&digest(1), "test pack").unwrap();
        let semantic_catalog =
            semantic_digest_from_hex(&digest(2), "test semantic catalog").unwrap();
        let compiled_model = semantic_digest_from_hex(&digest(3), "test compiled model").unwrap();
        let mut template = json!({
            "abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "alignment_bytes": 64,
            "backend": "jit",
            "coupling_slot_count": 0,
            "destination_aliasing": false,
            "destination_component_count": 4,
            "destination_operation": "initialize",
            "direct_executor_id": 0,
            "evaluator_binding_id": 7,
            "evaluator_resolver_key": "source:7",
            "exact_expression_digest": digest(4),
            "momentum_operand_count": 1,
            "optimization_level": 2,
            "parameter_slot_count": 0,
            "parent_arity": 0,
            "parent_component_counts": [],
            "payload_binding": {
                "abi": DIRECT_PAYLOAD_BINDING_ABI,
                "kind": "rusticol-intrinsic",
                "payload_digest": digest(5),
                "payload_paths": [],
                "prepared_kernel_id": null,
                "runtime_template": "rusticol.source-fill.v1"
            },
            "portable": true,
            "role": "source",
            "semantic_template_ids": ["source:test"],
            "simd_axis": "points-contiguous",
            "target_triple": "symjit-storage-v3-portable",
            "template_id": "direct:source:test"
        });
        let template_digest: [u8; 32] =
            Sha256::digest(canonical_json_bytes(&template, "test template").unwrap()).into();
        template.as_object_mut().unwrap().insert(
            "semantic_digest".to_owned(),
            json!(hex_digest(template_digest)),
        );

        let mut catalog = json!({
            "abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "backend": "jit",
            "backend_abi": DIRECT_BACKEND_ABI,
            "canonicalization_abi": DIRECT_CANONICALIZATION_ABI,
            "compiled_model_digest": compiled_model.to_string(),
            "optimization_level": 2,
            "optimization_settings_digest": digest(6),
            "portable": true,
            "prepared_kernel_contract_digest": digest(7),
            "prepared_kernel_pack_digest": prepared_pack.to_string(),
            "prepared_kernel_payload_digest": digest(8),
            "recurrence_template_catalog_digest": semantic_catalog.to_string(),
            "target_triple": "symjit-storage-v3-portable",
            "templates": [template]
        });
        let catalog_digest: [u8; 32] =
            Sha256::digest(canonical_json_bytes(&catalog, "test catalog").unwrap()).into();
        catalog.as_object_mut().unwrap().insert(
            "catalog_digest".to_owned(),
            json!(hex_digest(catalog_digest)),
        );
        (
            canonical_json_bytes(&catalog, "test catalog").unwrap(),
            prepared_pack,
            semantic_catalog,
            compiled_model,
        )
    }

    #[test]
    fn canonical_catalog_authenticates_dense_role_binding() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let parsed = parse_direct_template_catalog(&bytes, pack, semantic, model).unwrap();

        assert_eq!(parsed.catalog.direct_executor_count(), 1);
        assert_eq!(
            parsed
                .catalog
                .resolve_evaluator(DirectExecutorRole::Source, 7)
                .unwrap(),
            0
        );
        assert_eq!(parsed.prepared_kernel_count, 0);
    }

    #[test]
    fn noncanonical_catalog_bytes_are_rejected() {
        let (mut bytes, pack, semantic, model) = canonical_direct_catalog();
        bytes.push(b'\n');

        let error = match parse_direct_template_catalog(&bytes, pack, semantic, model) {
            Ok(_) => panic!("noncanonical direct catalog was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("not canonical"));
    }

    #[test]
    fn catalog_digest_authenticates_executor_mapping() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let mut value: JsonValue = serde_json::from_slice(&bytes).unwrap();
        value["templates"][0]["evaluator_binding_id"] = json!(8);
        let tampered = canonical_json_bytes(&value, "tampered catalog").unwrap();

        let error = match parse_direct_template_catalog(&tampered, pack, semantic, model) {
            Ok(_) => panic!("tampered direct catalog was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("catalog digest"));
    }

    #[test]
    fn pending_direct_payload_is_rejected_before_lowering() {
        let (bytes, pack, semantic, model) = canonical_direct_catalog();
        let mut value: JsonValue = serde_json::from_slice(&bytes).unwrap();
        let template = &mut value["templates"][0];
        template["payload_binding"]["kind"] = json!("pending-direct-call-abi");
        template["payload_binding"]["runtime_template"] = JsonValue::Null;
        template["payload_binding"]["prepared_kernel_id"] = json!(7);
        refresh_digest(template, "semantic_digest", "pending direct template");
        refresh_digest(&mut value, "catalog_digest", "pending direct catalog");
        let pending = canonical_json_bytes(&value, "pending direct catalog").unwrap();

        let error = match parse_direct_template_catalog(&pending, pack, semantic, model) {
            Ok(_) => panic!("pending direct payload was accepted"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("no executable Direct-Arena payload")
        );
    }
}
