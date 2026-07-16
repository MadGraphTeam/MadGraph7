// SPDX-License-Identifier: 0BSD

//! Python-independent Rusticol runtime core.

#[cfg(not(any(feature = "f64-symjit", feature = "symbolica-runtime")))]
compile_error!("rusticol-core requires at least one evaluator runtime feature");

mod artifact;
mod engine;
mod error;
mod metadata;

pub use artifact::{
    ArtifactKind, ArtifactManifest, ArtifactProcess, ArtifactSelection, Payload, PayloadRole,
    ProcessAlias, Target, VerifiedArtifact, runtime_target_info,
};
pub use engine::{
    NativeColorComponent, NativeDecimalEvaluation, NativeDecimalResolvedEvaluation,
    NativeExternalParticle, NativeHelicityConfiguration, NativeModelParameter,
    NativeResolvedEvaluation, NativeRuntime, NativeRuntimeMetadata, RuntimeCapability,
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
pub const COMPILED_MODEL_SCHEMA_VERSION: u32 = 6;
pub const PROCESS_ARTIFACT_SCHEMA_VERSION: u32 = 3;
pub const RUNTIME_PHYSICS_SCHEMA_VERSION: u32 = 1;
pub const C_ABI_VERSION: u32 = 1;
pub const SYMBOLICA_SERIALIZATION_ABI: &str = "candidate-e4167e7-bincode2";
pub const ARTIFACT_MANIFEST_FILE: &str = "artifact.json";
