// SPDX-License-Identifier: 0BSD

//! Python-independent Rusticol runtime core.

#[cfg(not(any(
    feature = "f64-compiled",
    feature = "f64-symjit",
    feature = "symbolica-runtime"
)))]
compile_error!("rusticol-core requires at least one evaluator runtime feature");

mod artifact;
pub mod eager_layout;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_runtime;
mod eager_tables;
mod engine;
mod error;
mod metadata;
pub mod pacbin;

pub use artifact::{
    ArtifactKind, ArtifactManifest, ArtifactProcess, ArtifactSelection, Payload, PayloadRole,
    ProcessAlias, Target, VerifiedArtifact, runtime_target_info,
};
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub use eager_runtime::{
    DEFAULT_EAGER_POINT_TILE_SIZE, DEFAULT_EAGER_WORKSPACE_MIB,
    EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF, EAGER_INDEPENDENT_BLOCK_SIZE, EagerComplex64,
    EagerDirectClosureSpec, EagerExecutionPlan, EagerExecutionRuntime, EagerKernelBackend,
    EagerKernelCall, EagerKernelInput, EagerKernelRole, EagerKernelSpec, EagerPlanDefinition,
    EagerPlanDimensions, EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup,
    EagerRuntimeOptions, EagerSelectorPayloads, EagerSelectorStagePayload, EagerStagePayload,
};
pub use eager_tables::{
    EAGER_KERNEL_ABI, EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG, EAGER_OUTPUT_FACTOR_COUPLING_REAL, EAGER_OUTPUT_FACTOR_NONE,
    EAGER_PLAN_ABI, EAGER_RUNTIME_CAPABILITY, EAGER_SELECTOR_DOMAINS_ABI, EagerAttachmentRow,
    EagerClosureRow, EagerCouplingRow, EagerFinalizationRow, EagerInvocationRow,
    EagerSelectorDomainIdRow, EagerSelectorDomainRow, EagerSelectorGroupRow, MISSING_U32,
};
pub use engine::{
    NativeColorComponent, NativeDecimalEvaluation, NativeDecimalResolvedEvaluation,
    NativeExternalParticle, NativeHelicityConfiguration, NativeModelParameter,
    NativeProfiledEvaluation, NativeResolvedEvaluation, NativeRuntime, NativeRuntimeMetadata,
    NativeRuntimeProfile, RuntimeCapability, preflight_prepared_kernel_pack,
    supported_runtime_capabilities,
};
pub use error::{RusticolError, RusticolErrorKind, RusticolResult};
pub use metadata::{
    ColorAccuracy, ColorComponent, ContractedColor, Coverage, ExternalParticle, Helicity,
    LcColorFlow, ModelParameter, ParameterKind, ParticleRole, ProcessPhysics, Reduction,
    ReductionGroup, ReductionKind, SelectorCapabilities,
};

pub const PYTHON_API_VERSION: u32 = 1;
pub const TOML_SCHEMA_VERSION: u32 = 1;
pub const COMPILED_MODEL_SCHEMA_VERSION: u32 = 9;
pub const PROCESS_ARTIFACT_SCHEMA_VERSION: u32 = 3;
pub const RUNTIME_PHYSICS_SCHEMA_VERSION: u32 = 1;
pub const C_ABI_VERSION: u32 = 1;
pub const SYMBOLICA_SERIALIZATION_ABI: &str = "symbolica-bincode2-v1";
pub const ARTIFACT_MANIFEST_FILE: &str = "artifact.json";
