//! Batched one-loop tensor integral reduction.
//!
//! Graph import and numerator preparation belong to `iregi-prepare`.
#![forbid(unsafe_code)]

mod affine_descent_guard;
mod amplitude;
mod affine_power;
pub mod basis;
pub mod certified_planar;
pub mod certified_scalar;
pub mod certified_tensor;
pub mod contraction;
mod convention;
pub mod direct;
pub mod dispatch;
mod exact_zero_certificate;
pub mod execution;
mod mass_taylor;
pub mod seven_point;
pub mod six_point;
mod thin_n6;
mod thin_n7;
pub use affine_power::{
    AffinePowerBudget, AffinePowerChildMethod, AffinePowerExecutionStats, AffinePowerReductionPlan,
    AffinePowerResult, AffinePowerWorkspace,
};
pub use mass_taylor::{
    MassTaylorBudget, MassTaylorExecutionStats, MassTaylorReductionPlan, MassTaylorWorkspace,
};
mod certified_affine;
pub mod ibp;
mod invariant_certificate;
pub mod plan;
pub mod pv;
pub mod regulators;
pub mod rescue;
pub mod scalar;
pub mod snapping;
mod stability;
pub mod store;
pub mod tensor_rescue;
pub mod types;
pub use certified_planar::{CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace};
pub use certified_scalar::{
    CertificateExecutionStats, CertifiedScalarReductionPlan, CertifiedScalarWorkspace,
};
pub use certified_tensor::{
    CertifiedTensorReductionPlan, CertifiedTensorWorkspace, TensorCertificateExecutionStats,
};
pub use amplitude::{AmplitudeOptions, AmplitudeWorkspace, SubsetSharingStats};
pub use contraction::{ContractionPlan, ContractionResult, ContractionWorkspace, NumeratorBatch};
pub use direct::{DirectCapabilities, DirectMasters, DirectProfile, DirectReductionPlan, DirectWorkspace, DirectPoleWorkspace, PoleBatchResult, PoleTensorStore};
pub use dispatch::{
    DirectExecutionStats, DirectRescueTier, EmbedderOptions, LaneRoute, PrecisionRescue, RotationTest,
    TensorEngineExecution, TensorEngineKind, TensorReductionPlan, TensorWorkspace,
    TensorWorkspaceOptions,
};
pub use execution::{ExecutionProfile, ExecutionStats, Workspace, reduce_batch};
pub use invariant_certificate::ExactInvariantScalar;
pub use plan::ReductionPlan;
pub use regulators::{RegulatorNormalization, Regulators};
pub use rescue::{
    PrecisionRescueWorkspace, RescueScalar, RescueSettings, RescueStats, RescueStatus,
    RescueTrigger,
};
pub use scalar::{ScalarKinematicsBatch, ScalarReductionPlan, ScalarWorkspace};
pub use store::{MasterFamily, MasterProvider, TensorStore, TensorStoreOptions, TensorStoreStatistics};
pub use seven_point::{SevenPointExecutionStats, SevenPointReductionPlan, SevenPointWorkspace};
pub use six_point::{SixPointExecutionStats, SixPointReductionPlan, SixPointWorkspace};
pub use tensor_rescue::TensorRescueWorkspace;
pub use types::*;

/// Scalar-master selection for explicitly configured workspaces.
pub use iregi_masters::MasterExecutionPolicy;
