// SPDX-License-Identifier: 0BSD

//! Model-generic compact recurrence ABI foundations.
//!
//! This module deliberately contains no runtime or artifact-writing code.  It
//! freezes the checked value and input contracts shared by the future Python
//! column encoder and Rust recurrence builder.

mod builder;
mod color;
mod exact;
mod input;
mod layout;
pub mod process;
pub mod template;

pub use builder::AuthenticatedRecurrenceBuilderInput;
pub use color::{
    DynamicLCColorState, DynamicLCColorStateInterner, LCColorComponent, LCColorComponentKind,
    LCColorComponentOperation, LCColorComponentRole, LCColorSourceSeed, LCColorSourceSeedOperation,
    LCColorTransitionWitness,
};
pub use exact::{ExactComplexRational, ExactRational};
pub use input::{
    CanonicalInputSection, CheckedTableRange, MultiwordMaskCatalogView,
    RecurrenceBuilderInputHeader, canonical_input_digest, checked_u32_len, checked_u64_len,
    checked_usize, validate_equal_column_lengths, validate_header_and_sections,
    validate_packed_ranges, validate_ranges_within, validate_u32_references,
};
pub use layout::{
    CanonicalMomentumLinearForm, ContributionKey, CurrentCoreKey, CurrentHelicityIdentity,
    CurrentSourceBinding, DynamicLCColorStateId, LCColorWitnessTermId, MomentumTerm,
    RecurrenceNodeKind, RecurrenceStrategy, SemanticDigest, SourceStateAssignment,
};

/// Semantic prepared-model companion ABI.
pub const RECURRENCE_TEMPLATE_ABI: &str = "pyamplicol-recurrence-template-v1";
/// Python-to-Rust recurrence builder input ABI.
pub const RECURRENCE_BUILDER_INPUT_ABI: &str = "pyamplicol-recurrence-builder-input-v1";
/// Bounded Rust-to-Python recurrence builder result ABI.
pub const RECURRENCE_BUILDER_RESULT_ABI: &str = "pyamplicol-recurrence-builder-result-v1";
/// Semantic recurrence plan ABI.
pub const RECURRENCE_PLAN_ABI: &str = "pyamplicol-recurrence-plan-v1";
/// Compact native recurrence layout ABI.
pub const RECURRENCE_RUNTIME_LAYOUT_ABI: &str = "pyamplicol-recurrence-runtime-layout-v1";
/// Process runtime kind stored in execution metadata.
pub const RECURRENCE_RUNTIME_KIND: &str = "pyamplicol-runtime-recurrence-execution";
/// Native complex-f64 recurrence capability.
pub const RECURRENCE_RUNTIME_CAPABILITY: &str = "rusticol.recurrence-runtime.complex-f64.v1";
/// LC color capability required by recurrence v1.
pub const RECURRENCE_LC_COLOR_CAPABILITY: &str = "rusticol.recurrence-color.lc.v1";
/// The builder input always consists of explicitly little-endian columns.
pub const RECURRENCE_INPUT_ENDIANNESS: &str = "little";

#[cfg(test)]
mod tests;
