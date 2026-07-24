// SPDX-License-Identifier: 0BSD

//! Model-generic compact recurrence construction and execution.

mod arena;
mod builder;
mod color;
mod construct;
pub mod direct_backend;
mod direct_codec;
mod direct_lowering;
mod direct_pacbin;
mod direct_plan;
pub mod direct_runtime;
mod exact;
mod input;
mod layout;
pub mod process;
mod program;
pub mod template;

pub use arena::{
    DirectArenaAssignment, DirectArenaInterval, DirectArenaLayout, assign_direct_arena,
    recurrence_direct_arena_layout,
};
pub use builder::AuthenticatedRecurrenceBuilderInput;
pub use color::{
    DynamicLCColorState, DynamicLCColorStateInterner, LCColorComponent, LCColorComponentKind,
    LCColorComponentOperation, LCColorComponentRole, LCColorEndpoint, LCColorParentPort,
    LCColorPortBinding, LCColorPortWiring, LCColorSourceSeed, LCColorSourceSeedOperation,
    LCColorTransitionWitness,
};
pub use construct::RecurrenceBuildProgress;
pub use direct_codec::{decode_recurrence_direct_plan_v2, encode_recurrence_direct_plan_v2};
pub use direct_lowering::{
    DirectRecurrenceRuntimeOptions, PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog,
    PreparedDirectExecutorKey, lower_recurrence_direct_plan_v2, lower_recurrence_direct_v2,
};
pub use direct_pacbin::{
    RECURRENCE_DIRECT_PLAN_MEMBER, RecurrenceDirectPacbinMetadata,
    load_recurrence_direct_plan_pacbin, write_recurrence_direct_plan_pacbin,
};
pub use direct_plan::{
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectResolvedSourceSelection, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
    DirectSourceRow, DirectSourceStateAssignment, RECURRENCE_DIRECT_PLAN_ABI,
    RECURRENCE_DIRECT_RUNTIME_CAPABILITY, RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
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
pub use program::{
    RecurrenceAmplitudeDestination, RecurrenceClosureTerm, RecurrenceContribution,
    RecurrenceCurrent, RecurrenceFinalization, RecurrenceProgram, RecurrenceReplayTarget,
    RecurrenceResolvedHelicity,
};
/// Semantic prepared-model companion ABI.
pub const RECURRENCE_TEMPLATE_ABI: &str = "pyamplicol-recurrence-template-v1";
/// Python-to-Rust recurrence builder input ABI.
pub const RECURRENCE_BUILDER_INPUT_ABI: &str = "pyamplicol-recurrence-builder-input-v2";
/// Bounded Rust-to-Python recurrence builder result ABI.
pub const RECURRENCE_BUILDER_RESULT_ABI: &str = "pyamplicol-recurrence-builder-result-v2";
/// Direct-arena recurrence plan ABI.
pub const RECURRENCE_PLAN_ABI: &str = RECURRENCE_DIRECT_PLAN_ABI;
/// Direct-arena native recurrence layout ABI.
pub const RECURRENCE_RUNTIME_LAYOUT_ABI: &str = RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI;
/// Process runtime kind stored in execution metadata.
pub const RECURRENCE_RUNTIME_KIND: &str = "pyamplicol-runtime-recurrence-execution";
/// Native complex-f64 recurrence capability.
pub const RECURRENCE_RUNTIME_CAPABILITY: &str = RECURRENCE_DIRECT_RUNTIME_CAPABILITY;
/// LC color capability required by recurrence.
pub const RECURRENCE_LC_COLOR_CAPABILITY: &str = "rusticol.recurrence-color.lc.v1";
/// The builder input always consists of explicitly little-endian columns.
pub const RECURRENCE_INPUT_ENDIANNESS: &str = "little";

#[cfg(test)]
mod direct_backend_tests;
#[cfg(test)]
mod direct_codec_tests;
#[cfg(test)]
mod direct_lowering_tests;
#[cfg(test)]
mod direct_pacbin_tests;
#[cfg(test)]
mod direct_plan_tests;
#[cfg(test)]
mod direct_runtime_tests;
#[cfg(test)]
mod tests;
