// SPDX-License-Identifier: 0BSD

#[cfg(not(feature = "symbolica-runtime"))]
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(feature = "symbolica-runtime")]
use symbolica::evaluate::JITCompiledEvaluator;
#[cfg(feature = "symbolica-runtime")]
use symbolica::prelude::{
    BatchEvaluator, Complex, DoubleFloat, EvaluationDomain, ExpressionEvaluator, Float,
    JITCompilationSettings, Rational, Real, RealLike,
};

#[cfg(feature = "symbolica-runtime")]
use crate::artifact::EvaluatorPayloadSource;
use crate::artifact::EvaluatorPayloadStore;
use crate::{
    ColorComponent as PhysicsColorComponentV1, PROCESS_ARTIFACT_SCHEMA_VERSION, PayloadRole,
    ProcessPhysics as ProcessPhysicsV1, RusticolError, RusticolResult, VerifiedArtifact,
};

// Keep replay state within the useful cache working set. Larger batches make
// stage-local gather/scatter substantially slower for high-flow LC workloads.
const MAX_LC_TOPOLOGY_REPLAY_EXPANDED_POINTS: usize = 2048;
const LC_SECTOR_SELECTOR_PARAMETER: &str = "runtime.lc_sector_id";
const HELICITY_RECURRENCE_CONTRACT_VERSION: u32 = 1;
const HELICITY_RECURRENCE_KIND: &str = "pyamplicol-helicity-recurrence";
const HELICITY_RECURRENCE_PROOF_ALGORITHM: &str = "canonical-source-transition-dependency-shape-v1";
const HELICITY_MATERIALIZATION_CONTRACT_VERSION: u32 = 1;
const HELICITY_MATERIALIZATION_KIND: &str = "pyamplicol-helicity-recurrence-materialization";

type LcTopologyReplayMappings = Vec<Vec<(usize, usize)>>;

#[derive(Clone, Debug, Default)]
struct LcTopologyReplayData {
    mappings: LcTopologyReplayMappings,
    routes: Vec<Vec<LcTopologyReplaySectorRoute>>,
    materialized_sector_ids: BTreeSet<i64>,
}

#[derive(Clone, Debug)]
struct LcTopologyReplaySectorRoute {
    physical_sector_id: i64,
    materialized_sector_id: i64,
    weight: f64,
    sign: i8,
    amplitude_factor: [f64; 2],
    residual: bool,
}

impl LcTopologyReplaySectorRoute {
    fn squared_reduction_weight(&self) -> f64 {
        // The replay sign is an amplitude-level relation.  LC resolved output
        // is diagonal in the physical flow, so sign^2 = 1.  Multiplying the
        // signed factor by the sign retains that convention explicitly.
        self.amplitude_factor[0] * f64::from(self.sign)
    }
}

#[derive(Clone, Copy, Debug)]
struct LcMaterializedSector {
    color_index: usize,
    reduction_weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayRoute {
    source_index: usize,
    target_index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayEntry {
    routes: Vec<LcResolvedReplayRoute>,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayPlan {
    #[cfg(test)]
    entries: Vec<LcResolvedReplayEntry>,
    routes_by_target: Vec<Vec<LcResolvedReplayTargetRoute>>,
    color_count: usize,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayTargetRoute {
    mapping_index: usize,
    source_index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplaySelection {
    #[cfg(test)]
    mapping_indices: Vec<usize>,
    #[cfg(test)]
    entries: Vec<LcResolvedReplayEntry>,
    #[cfg(test)]
    source_helicity_indices: Vec<Vec<usize>>,
    #[cfg(test)]
    source_color_indices: Vec<Vec<usize>>,
    source_groups: Vec<LcResolvedReplaySourceGroup>,
    helicity_indices: Vec<usize>,
    color_indices: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LcResolvedReplaySelectionKey {
    helicity_indices: Option<Vec<usize>>,
    color_indices: Option<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct LcResolvedReplaySourceGroup {
    mapping_indices: Vec<usize>,
    entries: Vec<LcResolvedReplayEntry>,
    helicity_ids: BTreeSet<String>,
    color_ids: BTreeSet<String>,
    source_component_count: usize,
}

pub const SYMJIT_APPLICATION_RUNTIME_CAPABILITY: &str = "symjit.application.complex-f64.v1";
pub const SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY: &str =
    "symbolica.legacy-jit-container.complex-f64.v1";
pub const SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY: &str = "symbolica.compiled-cpp.complex-f64.v1";
pub const SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY: &str = "symbolica.compiled-asm.complex-f64.v1";
pub const EAGER_DAG_RUNTIME_CAPABILITY: &str = crate::EAGER_RUNTIME_CAPABILITY;
pub const EAGER_RUNTIME_LAYOUT_CAPABILITY: &str = crate::eager_layout::EAGER_RUNTIME_CAPABILITY;
pub const EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY: &str =
    crate::EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY;
pub const COMPILED_RUNTIME_SELECTORS_CAPABILITY: &str = "rusticol.compiled.runtime-selectors.v1";
pub const COMPILED_HELICITY_DUAL_LANE_CAPABILITY: &str = "rusticol.compiled.helicity-dual-lane.v1";
pub const COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY: &str =
    "rusticol.compiled.helicity-selector-union.v1";
pub const COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY: &str =
    "rusticol.compiled.helicity-primary-recurrence.v1";
pub const COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY: &str =
    "rusticol.compiled.color-topology-lanes.v1";
#[cfg(feature = "f64-symjit")]
pub const SYMJIT_APPLICATION_STORAGE_ABI: &str = "symjit-application-storage-v3";

#[doc(hidden)]
pub fn preflight_prepared_kernel_pack(
    manifest_path: &Path,
    payload_root: &Path,
) -> RusticolResult<usize> {
    let bytes = fs::read(manifest_path).map_err(|error| {
        RusticolError::artifact(format!(
            "could not read prepared kernel pack {}: {error}",
            manifest_path.display()
        ))
    })?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse prepared kernel pack {}: {error}",
            manifest_path.display()
        ))
    })?;
    pack.validate()?;
    PreparedEvaluatorBackend::preflight_all(&pack, payload_root)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeCapability {
    CompiledColorTopologyLanesV1,
    CompiledHelicityDualLaneV1,
    CompiledHelicityPrimaryRecurrenceV1,
    CompiledHelicitySelectorUnionV1,
    CompiledRuntimeSelectorsV1,
    EagerDagComplexF64V1,
    EagerRuntimeLayoutComplexF64V1,
    EagerLcTopologyReplayComplexF64V1,
    SymjitApplicationComplexF64V1,
    SymbolicaLegacyJitContainerComplexF64V1,
    SymbolicaCompiledCppComplexF64V1,
    SymbolicaCompiledAsmComplexF64V1,
}

impl RuntimeCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompiledColorTopologyLanesV1 => COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
            Self::CompiledHelicityDualLaneV1 => COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
            Self::CompiledHelicityPrimaryRecurrenceV1 => {
                COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
            }
            Self::CompiledHelicitySelectorUnionV1 => COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
            Self::CompiledRuntimeSelectorsV1 => COMPILED_RUNTIME_SELECTORS_CAPABILITY,
            Self::EagerDagComplexF64V1 => EAGER_DAG_RUNTIME_CAPABILITY,
            Self::EagerRuntimeLayoutComplexF64V1 => EAGER_RUNTIME_LAYOUT_CAPABILITY,
            Self::EagerLcTopologyReplayComplexF64V1 => EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
            Self::SymjitApplicationComplexF64V1 => SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            Self::SymbolicaLegacyJitContainerComplexF64V1 => {
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
            }
            Self::SymbolicaCompiledCppComplexF64V1 => SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            Self::SymbolicaCompiledAsmComplexF64V1 => SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
        }
    }
}

pub fn supported_runtime_capabilities() -> Vec<&'static str> {
    let mut capabilities = vec![
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        EAGER_DAG_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        EAGER_RUNTIME_LAYOUT_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-symjit")]
        SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        #[cfg(feature = "symbolica-runtime")]
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-compiled")]
        SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-compiled")]
        SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
    ];
    capabilities.sort_unstable();
    capabilities
}

pub(crate) fn ensure_runtime_capabilities_supported<'a>(
    capabilities: impl IntoIterator<Item = &'a str>,
) -> RusticolResult<()> {
    let supported = supported_runtime_capabilities()
        .into_iter()
        .collect::<BTreeSet<_>>();
    for capability in capabilities {
        if !supported.contains(capability) {
            return Err(RusticolError::unsupported_runtime_capability(
                capability,
                format!("this Rusticol build supports {supported:?}"),
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize)]
struct ExecutionManifestHeader {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    kind: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSetManifest {
    schema_version: u32,
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    processes: Vec<ExecutionSetEntry>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSetEntry {
    process_id: String,
    manifest_path: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputCrossingMapEntry {
    target_index: usize,
    source_index: usize,
    sign: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionManifest {
    schema_version: u32,
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    compiled: EvaluatorSetManifest,
    dag_summary: ExecutionSummary,
    runtime_schema: ExecutionPlan,
    #[serde(default)]
    physics_reduction: Option<crate::Reduction>,
    #[serde(default)]
    helicity_sum_execution: Option<Box<ExecutionManifest>>,
    #[serde(default)]
    helicity_selector_executions: Vec<HelicitySelectorExecutionManifest>,
    #[serde(default)]
    color_selector_executions: Vec<ColorSelectorExecutionManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorExecutionManifest {
    selector_domain_ids: Vec<usize>,
    #[serde(default)]
    schedule_mode: HelicitySelectorScheduleMode,
    execution: Box<ExecutionManifest>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum HelicitySelectorScheduleMode {
    #[default]
    ParentClosure,
    NestedRuntime,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorSelectorExecutionManifest {
    materialized_sector_id: i64,
    execution: Box<ExecutionManifest>,
}

#[derive(Clone, Debug)]
struct EvaluatorSetManifest {
    kind: String,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    model_parameter_evaluator: Option<GenericModelParameterEvaluatorManifest>,
    stage_evaluators: Option<GenericStageEvaluatorArtifactsManifest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorSetManifestWire {
    kind: String,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    #[serde(default)]
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    // Current Python artifacts mirror this additive contract under `compiled`.
    // Runtime loading uses the authoritative runtime_schema copy below.
    #[serde(default)]
    helicity_recurrence: Option<HelicityRecurrenceManifest>,
    #[serde(default)]
    model_parameter_evaluator: Option<GenericModelParameterEvaluatorManifest>,
    stage_evaluators: Option<GenericStageEvaluatorArtifactsManifest>,
}

impl<'de> Deserialize<'de> for EvaluatorSetManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = EvaluatorSetManifestWire::deserialize(deserializer)?;
        let _validated_additive_mirror = wire.helicity_recurrence;
        Ok(Self {
            kind: wire.kind,
            runtime_available: wire.runtime_available,
            runtime_unavailable_message: wire.runtime_unavailable_message,
            lc_topology_replay: wire.lc_topology_replay,
            model_parameter_evaluator: wire.model_parameter_evaluator,
            stage_evaluators: wire.stage_evaluators,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericModelParameterEvaluatorManifest {
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    input_parameter_indices: Vec<usize>,
    outputs: Vec<GenericDerivedParameterOutputManifest>,
    evaluator: EvaluatorManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericDerivedParameterOutputManifest {
    runtime_name: String,
    output_index: usize,
    real_parameter_index: usize,
    imag_parameter_index: usize,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayManifest {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    contract_version: Option<u32>,
    #[serde(default)]
    physical_sector_count: Option<usize>,
    #[serde(default)]
    replayed_sector_count: usize,
    #[serde(default)]
    materialized_sector_ids: Vec<i64>,
    #[serde(default)]
    residual_sector_ids: Vec<i64>,
    #[serde(default)]
    groups: Vec<LcTopologyReplayGroupManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayGroupManifest {
    representative_sector_id: i64,
    materialized_sector_id: i64,
    #[serde(default)]
    active_sector_ids: Vec<i64>,
    #[serde(default)]
    proof: Option<LcTopologyReplayProofManifest>,
    #[serde(default)]
    sector_permutations: Vec<LcTopologyReplaySectorPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayProofManifest {
    #[serde(default)]
    status: String,
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    digest: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplaySectorPermutationManifest {
    sector_id: i64,
    #[serde(default = "default_lc_topology_replay_weight")]
    weight: f64,
    #[serde(default = "default_lc_topology_replay_sign")]
    sign: i8,
    #[serde(default)]
    factor: Option<Vec<f64>>,
    #[serde(default)]
    label_permutation: Vec<LcTopologyReplayLabelPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayLabelPermutationManifest {
    representative_label: usize,
    sector_label: usize,
}

fn default_lc_topology_replay_weight() -> f64 {
    1.0
}

fn default_lc_topology_replay_sign() -> i8 {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceManifest {
    kind: String,
    contract_version: u32,
    proof_algorithm: String,
    current_count: usize,
    amplitude_root_count: usize,
    proof_counts: HelicityRecurrenceProofCountsManifest,
    selector_domains: Vec<HelicitySelectorDomainManifest>,
    source_state_mappings: Vec<HelicitySourceStateMappingManifest>,
    recurrence_classes: Vec<HelicityRecurrenceClassManifest>,
    amplitude_classes: Vec<HelicityAmplitudeReplayClassManifest>,
    residual_current_ids: Vec<usize>,
    residual_root_ids: Vec<usize>,
    structural_zero_selector_domain_ids: Vec<usize>,
    diagnostics: Vec<String>,
    #[serde(default)]
    materialization: Option<HelicityRecurrenceMaterializationManifest>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceProofCountsManifest {
    recurrence_class_count: usize,
    optimized_recurrence_class_count: usize,
    optimized_current_count: usize,
    residual_current_count: usize,
    amplitude_class_count: usize,
    optimized_amplitude_class_count: usize,
    residual_amplitude_count: usize,
    source_state_mapping_count: usize,
    physical_helicity_count: usize,
    structural_zero_helicity_count: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorDomainManifest {
    id: usize,
    complete: bool,
    source_states: Vec<HelicitySelectorSourceStateManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorSourceStateManifest {
    external_label: usize,
    helicity: i32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityCurrentReplayMemberManifest {
    current_id: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceClassManifest {
    class_id: String,
    representative_current_id: usize,
    external_labels: Vec<usize>,
    source_class: bool,
    members: Vec<HelicityCurrentReplayMemberManifest>,
    proof: HelicityRecurrenceProofManifest,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceProofManifest {
    status: String,
    algorithm: String,
    digest: String,
    transition_contract_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicitySourceStateMappingManifest {
    current_id: usize,
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    recurrence_class_id: String,
    representative_current_id: usize,
    source_contract_digest: String,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityAmplitudeReplayMemberManifest {
    root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityAmplitudeReplayClassManifest {
    class_id: String,
    representative_root_id: usize,
    members: Vec<HelicityAmplitudeReplayMemberManifest>,
    proof: HelicityRecurrenceProofManifest,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceMaterializationManifest {
    kind: String,
    contract_version: u32,
    #[serde(default)]
    strategy: HelicityMaterializationStrategy,
    proof_current_count: usize,
    proof_root_count: usize,
    materialized_current_count: usize,
    materialized_root_count: usize,
    proof_to_materialized_current: Vec<usize>,
    source_routes: Vec<HelicityMaterializedSourceRouteManifest>,
    amplitude_routes: Vec<HelicityMaterializedAmplitudeRouteManifest>,
    selector_schedules: Vec<HelicityMaterializedSelectorScheduleManifest>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum HelicityMaterializationStrategy {
    #[default]
    Quotient,
    RetainedProofGraph,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedSourceRouteManifest {
    materialized_current_id: usize,
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedAmplitudeRouteManifest {
    materialized_root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
    residual: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedSelectorScheduleManifest {
    selector_domain_id: usize,
    active_current_ids: Vec<usize>,
    active_root_ids: Vec<usize>,
    structural_zero: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceRuntime {
    selector_domains: Vec<HelicitySelectorDomainRuntime>,
    source_state_mappings: Vec<HelicitySourceStateMappingRuntime>,
    recurrence_classes: Vec<HelicityRecurrenceClassRuntime>,
    amplitude_classes: Vec<HelicityAmplitudeReplayClassRuntime>,
    residual_current_ids: Vec<usize>,
    residual_root_ids: Vec<usize>,
    structural_zero_selector_domain_ids: Vec<usize>,
    materialization: Option<HelicityRecurrenceMaterializationRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicitySelectorDomainRuntime {
    complete: bool,
    source_states: Vec<(usize, i32)>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityCurrentReplayMemberRuntime {
    current_id: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceClassRuntime {
    representative_current_id: usize,
    external_labels: Vec<usize>,
    source_class: bool,
    members: Vec<HelicityCurrentReplayMemberRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicitySourceStateMappingRuntime {
    current_id: usize,
    external_index: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    recurrence_class_index: usize,
    representative_current_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityAmplitudeReplayMemberRuntime {
    root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityAmplitudeReplayClassRuntime {
    representative_root_id: usize,
    members: Vec<HelicityAmplitudeReplayMemberRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceMaterializationRuntime {
    strategy: HelicityMaterializationStrategy,
    proof_to_materialized_current: Vec<usize>,
    source_routes: Vec<HelicityMaterializedSourceRouteRuntime>,
    amplitude_routes: Vec<HelicityMaterializedAmplitudeRouteRuntime>,
    selector_schedules: Vec<HelicityMaterializedSelectorScheduleRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedSourceRouteRuntime {
    materialized_current_id: usize,
    external_index: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedAmplitudeRouteRuntime {
    materialized_root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
    residual: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedSelectorScheduleRuntime {
    selector_domain_id: usize,
    active_current_ids: Vec<usize>,
    active_root_ids: Vec<usize>,
    active_stage_chunk_indices: Vec<Vec<usize>>,
    active_amplitude_chunk_indices: Vec<usize>,
    structural_zero: bool,
}

#[derive(Clone, Debug)]
struct CompiledColorSelectorSchedule {
    active_stage_chunk_indices: Vec<Vec<usize>>,
    active_amplitude_chunk_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct CompiledColorExecutionPlan {
    schedules_by_materialized_sector: BTreeMap<i64, Arc<CompiledColorSelectorSchedule>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageEvaluatorArtifactsManifest {
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    parameter_count: usize,
    value_parameter_count: usize,
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    real_valued_inputs: Vec<usize>,
    parameter_layout: String,
    stage_count: usize,
    stages: Vec<GenericSerializedStageEvaluatorManifest>,
    amplitude_stage: GenericSerializedStageEvaluatorManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSerializedStageEvaluatorManifest {
    stage_index: usize,
    stage_kind: String,
    subset_size: Option<usize>,
    evaluator_label: String,
    parameter_layout: String,
    output_length: usize,
    output_slots: Vec<GenericStageOutputSlotManifest>,
    input_value_slot_ids: Vec<usize>,
    output_value_slot_ids: Vec<usize>,
    interaction_ids: Vec<usize>,
    #[serde(default)]
    input_components: Vec<GenericStageInputComponentManifest>,
    #[serde(default)]
    parameter_count: usize,
    #[serde(default)]
    value_parameter_count: usize,
    #[serde(default)]
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    #[serde(default)]
    real_valued_inputs: Vec<usize>,
    expression_ready: bool,
    blockers: Vec<String>,
    evaluator: EvaluatorManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageInputComponentManifest {
    kind: String,
    source_id: usize,
    component: usize,
    global_component: usize,
    parameter_index: usize,
    #[serde(default)]
    real_valued: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageOutputSlotManifest {
    value_slot_id: isize,
    current_id: isize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    output_start: usize,
    output_stop: usize,
    #[serde(default)]
    color_selector_domain_ids: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSummary {
    current_count: usize,
    source_count: usize,
    interaction_count: usize,
    amplitude_root_count: usize,
    truncated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionPlan {
    schema_version: u32,
    kind: String,
    process_key: String,
    process: String,
    external_particles: Vec<GenericExternalParticleManifest>,
    #[serde(default)]
    model: Option<GenericRuntimeModelManifest>,
    #[serde(default)]
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    #[serde(default)]
    normalization: Option<GenericRuntimeNormalizationManifest>,
    parameter_layout: GenericParameterLayoutManifest,
    current_storage: GenericCurrentStorageManifest,
    value_storage: GenericValueStorageManifest,
    source_fill: GenericSourceFillManifest,
    momentum_slots: Vec<GenericMomentumSlotManifest>,
    stages: Vec<GenericStageManifest>,
    amplitude_stage: GenericAmplitudeStageManifest,
    #[serde(default)]
    helicity_recurrence: Option<HelicityRecurrenceManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericExternalParticleManifest {
    label: usize,
    index: usize,
    pdg: i32,
    outgoing_pdg: i32,
    role: String,
    momentum_slot: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeModelManifest {
    #[serde(default)]
    particles: Vec<GenericRuntimeParticleManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeModelParameterManifest {
    name: String,
    kind: String,
    parameter_index: usize,
    #[serde(default)]
    default: f64,
    #[serde(default)]
    pdg: Option<i32>,
    #[serde(default)]
    runtime_name: Option<String>,
    #[serde(default)]
    complex_component: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeParticleManifest {
    pdg: i32,
    #[serde(default)]
    mass: f64,
    #[serde(default)]
    mass_parameter: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeNormalizationManifest {
    #[serde(default = "default_one_f64")]
    color_factor: f64,
    #[serde(default = "default_one_f64")]
    global_coupling_factor: f64,
    #[serde(default = "default_one_f64")]
    average_factor: f64,
    #[serde(default = "default_one_f64")]
    identical_factor: f64,
    #[serde(default)]
    qcd_coupling_power: usize,
    #[serde(default)]
    electroweak_coupling_power: usize,
}

fn default_one_f64() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericParameterLayoutManifest {
    source_component_parameter_count: usize,
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    parameter_count_if_flattened: usize,
    value_component_count: usize,
    source_components_complex: bool,
    momentum_components_real: bool,
    real_valued_inputs: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericCurrentStorageManifest {
    component_count: usize,
    number_type: String,
    #[serde(default)]
    metadata_compacted: bool,
    current_slots: Vec<GenericCurrentSlotManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericCurrentSlotManifest {
    current_id: usize,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
    is_source: bool,
    particle_id: i32,
    external_mask: u64,
    #[serde(default)]
    external_labels: Vec<usize>,
    #[serde(default)]
    helicity_ancestry: Value,
    chirality: i32,
    #[serde(default)]
    spin_state: Value,
    #[serde(default)]
    flavour_flow: Vec<i32>,
    #[serde(default)]
    color_state: Value,
    momentum_mask: u64,
    auxiliary_kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericValueStorageManifest {
    component_count: usize,
    number_type: String,
    #[serde(default)]
    metadata_compacted: bool,
    value_slots: Vec<GenericValueSlotManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericValueSlotManifest {
    value_slot_id: usize,
    current_id: usize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
    current_component_start: usize,
    current_component_stop: usize,
    is_source: bool,
    applies_propagator: bool,
    particle_id: i32,
    external_mask: u64,
    #[serde(default)]
    external_labels: Vec<usize>,
    momentum_mask: u64,
    chirality: i32,
    propagator: GenericPropagatorIrManifest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorKindManifest {
    Identity,
    Scalar,
    WeylFermion,
    DiracFermion,
    Vector,
    Spin2,
    Custom,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorMassClassManifest {
    Massless,
    Massive,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorGaugeManifest {
    Feynman,
    Unitary,
    DeDonder,
    FierzPauli,
    ModelSupplied,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericGoldstonePolicyManifest {
    NotApplicable,
    Absorbed,
    Explicit,
    ModelSupplied,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericPropagatorIrManifest {
    identity: GenericParticleIdentityIrManifest,
    particle_id: i32,
    chirality: i32,
    kind: GenericPropagatorKindManifest,
    backend: String,
    basis: String,
    applies_propagator: bool,
    kernel: String,
    full_tensor_network_ready: bool,
    mass_class: GenericPropagatorMassClassManifest,
    gauge: Option<GenericPropagatorGaugeManifest>,
    numerator: Option<String>,
    denominator: Option<String>,
    mass_parameter: Option<String>,
    width_parameter: Option<String>,
    custom_source: Option<String>,
    auxiliary_policy: Option<String>,
    goldstone_policy: GenericGoldstonePolicyManifest,
    description: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceFillManifest {
    source_count: usize,
    sources: Vec<GenericSourceRecordManifest>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum GenericSourceOrientationManifest {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GenericParticleStatisticsManifest {
    Boson,
    Fermion,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GenericWavefunctionFamilyManifest {
    Scalar,
    Fermion,
    Vector,
    Spin2,
    Ghost,
    Auxiliary,
}

impl GenericWavefunctionFamilyManifest {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Fermion => "fermion",
            Self::Vector => "vector",
            Self::Spin2 => "spin2",
            Self::Ghost => "ghost",
            Self::Auxiliary => "auxiliary",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum GenericMomentumTransformManifest {
    Identity,
    NegateFourMomentum,
}

impl GenericMomentumTransformManifest {
    const fn legacy_projection(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::NegateFourMomentum => "negate-incoming-momentum",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
enum GenericSourceSpinStateManifest {
    Scalar(i32),
    Components(Vec<i32>),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceStateIrManifest {
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
}

impl GenericSourceStateIrManifest {
    fn transformed(&self, crossing: &GenericCrossingIrManifest) -> Result<Self, &'static str> {
        let helicity = self
            .helicity
            .checked_mul(crossing.helicity_factor)
            .ok_or("source crossing overflows the helicity state")?;
        let chirality = self
            .chirality
            .checked_mul(crossing.chirality_factor)
            .ok_or("source crossing overflows the chirality state")?;
        let spin_state = match (&self.spin_state, crossing.spin_state_factor) {
            (state, 1) => state.clone(),
            (GenericSourceSpinStateManifest::Scalar(state), factor) => {
                GenericSourceSpinStateManifest::Scalar(
                    state
                        .checked_mul(factor)
                        .ok_or("source crossing overflows the spin state")?,
                )
            }
            (GenericSourceSpinStateManifest::Components(_), _) => {
                return Err("crossing cannot multiply a structured source spin state");
            }
        };
        Ok(Self {
            helicity,
            chirality,
            spin_state,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericCrossingIrManifest {
    momentum_transform: GenericMomentumTransformManifest,
    helicity_factor: i32,
    chirality_factor: i32,
    spin_state_factor: i32,
    phase: [f64; 2],
}

impl GenericCrossingIrManifest {
    fn is_identity(&self) -> bool {
        self.momentum_transform == GenericMomentumTransformManifest::Identity
            && self.helicity_factor == 1
            && self.chirality_factor == 1
            && self.spin_state_factor == 1
            && self.phase == [1.0, 0.0]
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericParticleIdentityIrManifest {
    canonical_id: String,
    species_id: String,
    anti_canonical_id: String,
    display_name: String,
    anti_display_name: String,
    pdg_label: i32,
    anti_pdg_label: i32,
    orientation: GenericSourceOrientationManifest,
    self_conjugate: bool,
}

fn deserialize_required_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => Err(<D::Error as serde::de::Error>::custom(
            "expected a string or null",
        )),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceIrManifest {
    identity: GenericParticleIdentityIrManifest,
    statistics: GenericParticleStatisticsManifest,
    wavefunction_family: GenericWavefunctionFamilyManifest,
    component_dimension: usize,
    states: Vec<GenericSourceStateIrManifest>,
    crossing: GenericCrossingIrManifest,
    basis: String,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    mass_parameter: Option<String>,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    width_parameter: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceRecordManifest {
    source_id: usize,
    current_id: usize,
    current_component_start: usize,
    current_component_stop: usize,
    value_slot: GenericValueSlotRefManifest,
    source_parameter_start: usize,
    source_parameter_stop: usize,
    leg_label: usize,
    input_momentum_slot: usize,
    side: String,
    crossing: String,
    physical_pdg: i32,
    outgoing_pdg: i32,
    particle_id: i32,
    anti_particle_id: i32,
    source_kind: String,
    wavefunction_kind: String,
    source_orientation: GenericSourceOrientationManifest,
    source_basis: String,
    source_ir: GenericSourceIrManifest,
    applied_crossing: GenericCrossingIrManifest,
    source_helicity: i32,
    chirality: i32,
    spin_state: Value,
    dimension: usize,
    helicity_ancestry: Value,
    color_state: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericMomentumSlotManifest {
    momentum_slot_id: usize,
    momentum_mask: u64,
    external_labels: Vec<usize>,
    component_start: usize,
    component_stop: usize,
    real_valued: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageManifest {
    stage_index: usize,
    stage_kind: String,
    subset_size: usize,
    input_current_ids: Vec<usize>,
    output_current_ids: Vec<usize>,
    input_value_slot_ids: Vec<usize>,
    output_value_slot_ids: Vec<usize>,
    interaction_count: usize,
    #[serde(default)]
    interactions_compacted: bool,
    #[serde(default)]
    interaction_ids: Vec<usize>,
    #[serde(default)]
    interactions: Vec<GenericInteractionManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericInteractionManifest {
    interaction_id: usize,
    vertex_kind: i32,
    vertex_particles: Vec<i32>,
    left_current_id: usize,
    right_current_id: usize,
    result_current_id: usize,
    left_slot: GenericSlotRefManifest,
    right_slot: GenericSlotRefManifest,
    result_slot: GenericSlotRefManifest,
    left_value_slot: GenericValueSlotRefManifest,
    right_value_slot: GenericValueSlotRefManifest,
    result_value_slots: Vec<GenericValueSlotRefManifest>,
    result_requires_propagated_value: bool,
    result_requires_unpropagated_value: bool,
    momentum_slots: GenericInteractionMomentumSlotsManifest,
    coupling: Vec<f64>,
    color_weight: Vec<f64>,
    accumulation: String,
    lowering: GenericLoweringManifest,
    full_tensor_network_ready: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericLoweringManifest {
    kind: i32,
    backend: String,
    tensor_names: Vec<String>,
    expression_head: String,
    full_tensor_network_ready: bool,
    description: String,
    kernel: String,
    input_roles: Vec<String>,
    output_role: String,
    coupling_mode: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericInteractionMomentumSlotsManifest {
    left: usize,
    right: usize,
    result: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSlotRefManifest {
    current_id: usize,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericAmplitudeStageManifest {
    stage_kind: String,
    output_count: usize,
    #[serde(default)]
    color_contraction: Option<GenericColorContractionManifest>,
    roots: Vec<GenericAmplitudeRootManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorContractionManifest {
    supported: bool,
    #[serde(default)]
    reason: Option<String>,
    group_count: usize,
    #[serde(default)]
    includes_color_factor: bool,
    entries: Vec<GenericColorContractionEntryManifest>,
    #[serde(default)]
    repeated_block: Option<GenericRepeatedColorContractionBlockManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorContractionEntryManifest {
    left_group_id: i64,
    right_group_id: i64,
    weight: Vec<f64>,
    #[serde(default = "default_symmetry_factor")]
    symmetry_factor: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRepeatedColorContractionBlockManifest {
    component_count: usize,
    component_group_ids: Vec<i64>,
    entries: Vec<GenericRepeatedColorContractionEntryManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRepeatedColorContractionEntryManifest {
    left_group_index: usize,
    right_group_index: usize,
    weight: Vec<f64>,
    #[serde(default = "default_symmetry_factor")]
    symmetry_factor: f64,
}

fn default_symmetry_factor() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericAmplitudeRootManifest {
    output_index: usize,
    root_id: usize,
    kind: String,
    left_current_id: usize,
    right_current_id: usize,
    left_slot: GenericSlotRefManifest,
    right_slot: GenericSlotRefManifest,
    left_value_slot: GenericValueSlotRefManifest,
    right_value_slot: GenericValueSlotRefManifest,
    vertex_kind: Option<i32>,
    vertex_particles: Option<Vec<i32>>,
    coupling: Vec<f64>,
    color_weight: Vec<f64>,
    #[serde(default)]
    color_sector_id: Option<i64>,
    contraction: String,
    contraction_ir: GenericContractionIrManifest,
    coherent_group_id: Option<Value>,
    helicity_weight: f64,
    #[serde(default)]
    all_sector_weight: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum GenericContractionChiralityRelationManifest {
    Any,
    Equal,
    Opposite,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericContractionIrManifest {
    name: String,
    left_basis: String,
    right_basis: String,
    coefficients: Vec<[f64; 2]>,
    chirality_relation: GenericContractionChiralityRelationManifest,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    metric_signature: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericValueSlotRefManifest {
    value_slot_id: usize,
    current_id: usize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind")]
enum EvaluatorManifest {
    #[serde(rename = "symjit-application-evaluator")]
    SymjitApplication {
        runtime_capability: String,
        application_path: String,
        application_abi: String,
        input_len: usize,
        output_len: usize,
        element_layout: String,
        batch_layout: String,
        compiler_type: String,
        translation_mode: String,
        optimization_level: u8,
        word_bits: u8,
        endianness: String,
        required_defuns: Vec<String>,
        evaluator_state_path: Option<String>,
        evaluator_state_runtime_capability: Option<String>,
    },
    #[serde(rename = "jit-symbolica-evaluator")]
    Jit {
        runtime_capability: String,
        input_len: usize,
        output_len: usize,
        evaluator_state_path: String,
    },
    #[serde(rename = "compiled-complex-evaluator")]
    CompiledComplex {
        runtime_capability: String,
        function_name: String,
        input_len: usize,
        output_len: usize,
        library_path: String,
        evaluator_state_path: Option<String>,
        number_type: String,
    },
    #[serde(rename = "chunked-symbolica-evaluator")]
    Chunked {
        required_runtime_capabilities: Vec<String>,
        #[serde(default)]
        input_len: Option<usize>,
        #[serde(default)]
        chunk_input_indices: Option<Vec<Vec<usize>>>,
        chunks: Vec<EvaluatorManifest>,
    },
}

impl EvaluatorManifest {
    fn io_len(&self) -> RusticolResult<(usize, usize)> {
        match self {
            Self::SymjitApplication {
                input_len,
                output_len,
                ..
            }
            | Self::Jit {
                input_len,
                output_len,
                ..
            }
            | Self::CompiledComplex {
                input_len,
                output_len,
                ..
            } => Ok((*input_len, *output_len)),
            Self::Chunked {
                input_len,
                chunk_input_indices,
                chunks,
                ..
            } => {
                if chunks.is_empty() {
                    return Err(RusticolError::artifact(
                        "generic serialized evaluator chunk list is empty",
                    ));
                }
                let mut child_layouts = Vec::with_capacity(chunks.len());
                let mut output_len = 0usize;
                for chunk in chunks {
                    let layout = chunk.io_len()?;
                    output_len = output_len.checked_add(layout.1).ok_or_else(|| {
                        RusticolError::artifact(
                            "generic serialized evaluator output length overflows usize",
                        )
                    })?;
                    child_layouts.push(layout);
                }
                match (input_len, chunk_input_indices) {
                    (None, None) => {
                        let parent_input_len = child_layouts[0].0;
                        if child_layouts
                            .iter()
                            .any(|(child_input_len, _)| *child_input_len != parent_input_len)
                        {
                            return Err(RusticolError::artifact(
                                "legacy chunked evaluator children have inconsistent input lengths",
                            ));
                        }
                        Ok((parent_input_len, output_len))
                    }
                    (Some(parent_input_len), Some(input_indices)) => {
                        if input_indices.len() != chunks.len() {
                            return Err(RusticolError::artifact(
                                "chunked evaluator input maps do not match evaluator chunks",
                            ));
                        }
                        for (indices, (child_input_len, _)) in
                            input_indices.iter().zip(&child_layouts)
                        {
                            if indices.len() != *child_input_len
                                || indices.iter().any(|index| *index >= *parent_input_len)
                                || indices.windows(2).any(|pair| pair[0] >= pair[1])
                            {
                                return Err(RusticolError::artifact(
                                    "chunked evaluator input map is inconsistent with child inputs",
                                ));
                            }
                        }
                        Ok((*parent_input_len, output_len))
                    }
                    _ => Err(RusticolError::artifact(
                        "chunked evaluator input metadata is incomplete",
                    )),
                }
            }
        }
    }

    fn leaf_input_indices(&self) -> RusticolResult<Vec<Vec<usize>>> {
        fn append_leaf_inputs(
            evaluator: &EvaluatorManifest,
            parent_inputs: &[usize],
            leaf_inputs: &mut Vec<Vec<usize>>,
        ) -> RusticolResult<()> {
            match evaluator {
                EvaluatorManifest::Chunked {
                    input_len,
                    chunk_input_indices,
                    chunks,
                    ..
                } => {
                    evaluator.io_len()?;
                    if let Some(input_len) = input_len
                        && *input_len != parent_inputs.len()
                    {
                        return Err(RusticolError::artifact(
                            "chunked evaluator parent input mapping has an inconsistent length",
                        ));
                    }
                    match chunk_input_indices {
                        Some(chunk_inputs) => {
                            for (chunk, indices) in chunks.iter().zip(chunk_inputs) {
                                let mapped = indices
                                    .iter()
                                    .map(|index| {
                                        parent_inputs.get(*index).copied().ok_or_else(|| {
                                            RusticolError::artifact(
                                                "chunked evaluator input map references an absent parent input",
                                            )
                                        })
                                    })
                                    .collect::<RusticolResult<Vec<_>>>()?;
                                append_leaf_inputs(chunk, &mapped, leaf_inputs)?;
                            }
                        }
                        None => {
                            for chunk in chunks {
                                append_leaf_inputs(chunk, parent_inputs, leaf_inputs)?;
                            }
                        }
                    }
                }
                _ => {
                    let input_len = evaluator.io_len()?.0;
                    if input_len != parent_inputs.len() {
                        return Err(RusticolError::artifact(
                            "evaluator leaf input mapping has an inconsistent length",
                        ));
                    }
                    leaf_inputs.push(parent_inputs.to_vec());
                }
            }
            Ok(())
        }

        let root_input_len = self.io_len()?.0;
        let root_inputs = (0..root_input_len).collect::<Vec<_>>();
        let mut leaf_inputs = Vec::new();
        append_leaf_inputs(self, &root_inputs, &mut leaf_inputs)?;
        Ok(leaf_inputs)
    }
}

struct EvaluatorGroup {
    evaluators: Vec<LoadedEvaluator>,
    input_len: usize,
    input_mappings: Vec<Option<Vec<usize>>>,
    input_mapping_spans: Vec<Vec<(usize, usize, usize)>>,
    output_len: usize,
    chunk_parameter_scratch_f64: Vec<Complex<f64>>,
    chunk_scratch_f64: Vec<Complex<f64>>,
    chunk_parameter_scratch_aosoa_f64: Vec<f64>,
    chunk_scratch_aosoa_f64: Vec<f64>,
    chunk_input_mapping_scratch: Vec<usize>,
}

// Evaluator groups already own these values behind a Vec allocation. Boxing
// only the SymJIT variant would add an indirection to every hot kernel call.
#[allow(clippy::large_enum_variant)]
enum F64Evaluator {
    #[cfg(feature = "f64-symjit")]
    SymjitApplication(SymjitApplicationEvaluator),
    #[cfg(feature = "f64-compiled")]
    Compiled(CompiledComplexF64Evaluator),
    #[cfg(feature = "symbolica-runtime")]
    Jit(JITCompiledEvaluator<Complex<f64>>),
}

struct LoadedEvaluator {
    eval: F64Evaluator,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval: Option<ExpressionEvaluator<Complex<Rational>>>,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval_source: Option<EvaluatorPayloadSource>,
    #[cfg(feature = "symbolica-runtime")]
    double_eval: Option<ExpressionEvaluator<Complex<DoubleFloat>>>,
    #[cfg(feature = "symbolica-runtime")]
    arb_eval: Option<(u32, ExpressionEvaluator<Complex<Float>>)>,
    input_len: usize,
    output_len: usize,
}

#[cfg(feature = "symbolica-runtime")]
trait RusticolHighPrecisionNumber:
    Real + RealLike + From<f64> + PartialOrd + Clone + EvaluationDomain
where
    Complex<Self>: Real + EvaluationDomain,
{
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<()>;
}

#[cfg(feature = "symbolica-runtime")]
impl RusticolHighPrecisionNumber for DoubleFloat {
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        _binary_precision: Option<u32>,
    ) -> RusticolResult<()> {
        if evaluator.double_eval.is_none() {
            let exact = evaluator.exact_evaluator()?.clone();
            evaluator.double_eval =
                Some(exact.map_coeff(&|c| {
                    Complex::new(DoubleFloat::from(&c.re), DoubleFloat::from(&c.im))
                }));
        }
        evaluator
            .double_eval
            .as_mut()
            .expect("double evaluator initialized")
            .evaluate(params, out);
        Ok(())
    }
}

#[cfg(feature = "symbolica-runtime")]
impl RusticolHighPrecisionNumber for Float {
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<()> {
        let binary_precision = binary_precision.ok_or_else(|| {
            RusticolError::invalid_argument(
                "arbitrary-precision evaluation needs a binary precision",
            )
        })?;
        let rebuild = evaluator
            .arb_eval
            .as_ref()
            .map(|(precision, _)| *precision != binary_precision)
            .unwrap_or(true);
        if rebuild {
            let exact = evaluator.exact_evaluator()?.clone();
            evaluator.arb_eval = Some((
                binary_precision,
                exact.map_coeff_with_prec(
                    &|c| {
                        Complex::new(
                            c.re.to_multi_prec_float(binary_precision),
                            c.im.to_multi_prec_float(binary_precision),
                        )
                    },
                    binary_precision,
                ),
            ));
        }
        evaluator
            .arb_eval
            .as_mut()
            .expect("arbitrary-precision evaluator initialized")
            .1
            .evaluate(params, out);
        Ok(())
    }
}

struct RawSumGroup {
    id: i64,
    indices: Vec<usize>,
    weight: f64,
    all_sector_weight: f64,
    sector_ids: Vec<i64>,
}

struct ColorContractionRuntime {
    group_count: usize,
    entries: Vec<ColorContractionEntry>,
    repeated_block: Option<RepeatedColorContractionBlock>,
    group_scratch_f64: Vec<Complex<f64>>,
}

#[derive(Clone, Copy)]
struct ColorContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re: f64,
    weight_im: f64,
    symmetry_factor: f64,
}

/// One color matrix shared by several disconnected contraction components.
///
/// Helicity-summed NLC/full-color plans contain one isomorphic color block per
/// physical helicity. Keeping only one canonical block avoids streaming the
/// same sparse matrix metadata once per helicity and lays out group values so
/// that the repeated component dimension is contiguous.
struct RepeatedColorContractionBlock {
    component_count: usize,
    component_group_indices: Vec<usize>,
    singleton_output_indices: Option<Vec<usize>>,
    entries: Vec<ColorContractionEntry>,
    all_weights_real: bool,
}

impl ColorContractionRuntime {
    fn new(groups: &[RawSumGroup], entries: Vec<ColorContractionEntry>) -> Self {
        let repeated_block = repeated_color_contraction_block(groups, &entries);
        Self {
            group_count: groups.len(),
            entries,
            repeated_block,
            group_scratch_f64: Vec::new(),
        }
    }

    fn from_repeated_block(
        groups: &[RawSumGroup],
        component_count: usize,
        component_group_indices: Vec<usize>,
        entries: Vec<ColorContractionEntry>,
    ) -> Self {
        let singleton_output_indices = component_group_indices
            .iter()
            .map(
                |group_index| match groups[*group_index].indices.as_slice() {
                    [output_index] => Some(*output_index),
                    _ => None,
                },
            )
            .collect::<Option<Vec<_>>>();
        let repeated_block = RepeatedColorContractionBlock {
            component_count,
            component_group_indices,
            singleton_output_indices,
            all_weights_real: entries.iter().all(|entry| entry.weight_im == 0.0),
            entries,
        };
        Self {
            group_count: groups.len(),
            entries: Vec::new(),
            repeated_block: Some(repeated_block),
            group_scratch_f64: Vec::new(),
        }
    }

    fn logical_entry_count(&self) -> RusticolResult<usize> {
        if self.entries.is_empty() {
            let Some(block) = self.repeated_block.as_ref() else {
                return Ok(0);
            };
            block
                .component_count
                .checked_mul(block.entries.len())
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "repeated colour contraction logical entry count overflows",
                    )
                })
        } else {
            Ok(self.entries.len())
        }
    }

    fn logical_entries(&self) -> ColorContractionEntries<'_> {
        if self.entries.is_empty() {
            if let Some(block) = self.repeated_block.as_ref() {
                return ColorContractionEntries::Repeated {
                    block,
                    component_index: 0,
                    entry_index: 0,
                };
            }
        }
        ColorContractionEntries::Expanded(self.entries.iter().copied())
    }
}

enum ColorContractionEntries<'a> {
    Expanded(std::iter::Copied<std::slice::Iter<'a, ColorContractionEntry>>),
    Repeated {
        block: &'a RepeatedColorContractionBlock,
        component_index: usize,
        entry_index: usize,
    },
}

impl Iterator for ColorContractionEntries<'_> {
    type Item = ColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Expanded(entries) => entries.next(),
            Self::Repeated {
                block,
                component_index,
                entry_index,
            } => {
                if *component_index >= block.component_count || block.entries.is_empty() {
                    return None;
                }
                let entry = block.entries[*entry_index];
                let component = *component_index;
                let left_group_index = block.component_group_indices
                    [entry.left_group_index * block.component_count + component];
                let right_group_index = block.component_group_indices
                    [entry.right_group_index * block.component_count + component];
                *entry_index += 1;
                if *entry_index == block.entries.len() {
                    *entry_index = 0;
                    *component_index += 1;
                }
                Some(ColorContractionEntry {
                    left_group_index,
                    right_group_index,
                    ..entry
                })
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalColorContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re_bits: u64,
    weight_im_bits: u64,
    symmetry_factor_bits: u64,
}

fn color_component_root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn repeated_color_contraction_block(
    groups: &[RawSumGroup],
    entries: &[ColorContractionEntry],
) -> Option<RepeatedColorContractionBlock> {
    if groups.len() < 2 || entries.is_empty() {
        return None;
    }

    let mut parent = (0..groups.len()).collect::<Vec<_>>();
    let mut component_size = vec![1usize; groups.len()];
    for entry in entries {
        let mut left = color_component_root(&mut parent, entry.left_group_index);
        let mut right = color_component_root(&mut parent, entry.right_group_index);
        if left == right {
            continue;
        }
        if component_size[left] < component_size[right] {
            std::mem::swap(&mut left, &mut right);
        }
        parent[right] = left;
        component_size[left] += component_size[right];
    }

    let mut components_by_root = BTreeMap::<usize, Vec<usize>>::new();
    for group_index in 0..groups.len() {
        let root = color_component_root(&mut parent, group_index);
        components_by_root
            .entry(root)
            .or_default()
            .push(group_index);
    }
    let mut components = components_by_root.into_values().collect::<Vec<_>>();
    components.sort_by_key(|component| component.iter().copied().min().unwrap_or(usize::MAX));
    if components.len() < 2 {
        return None;
    }
    let groups_per_component = components[0].len();
    if groups_per_component == 0
        || components
            .iter()
            .any(|component| component.len() != groups_per_component)
    {
        return None;
    }

    let component_maps = components
        .iter()
        .map(|component| {
            let mut by_sector_signature = BTreeMap::<Vec<i64>, usize>::new();
            for group_index in component {
                let mut signature = groups[*group_index].sector_ids.clone();
                signature.sort_unstable();
                if by_sector_signature
                    .insert(signature, *group_index)
                    .is_some()
                {
                    return None;
                }
            }
            Some(by_sector_signature)
        })
        .collect::<Option<Vec<_>>>()?;
    let canonical_signatures = component_maps[0].keys().cloned().collect::<Vec<_>>();
    if component_maps
        .iter()
        .skip(1)
        .any(|component| component.keys().ne(canonical_signatures.iter()))
    {
        return None;
    }

    let mut component_index_by_group = vec![usize::MAX; groups.len()];
    let mut local_index_by_group = vec![usize::MAX; groups.len()];
    let mut component_group_indices = Vec::with_capacity(groups.len());
    for (local_index, signature) in canonical_signatures.iter().enumerate() {
        for (component_index, component) in component_maps.iter().enumerate() {
            let group_index = component[signature];
            component_index_by_group[group_index] = component_index;
            local_index_by_group[group_index] = local_index;
            component_group_indices.push(group_index);
        }
    }

    let mut entries_by_component =
        vec![Vec::<CanonicalColorContractionEntry>::new(); components.len()];
    for entry in entries {
        let component_index = component_index_by_group[entry.left_group_index];
        if component_index == usize::MAX
            || component_index != component_index_by_group[entry.right_group_index]
        {
            return None;
        }
        entries_by_component[component_index].push(CanonicalColorContractionEntry {
            left_group_index: local_index_by_group[entry.left_group_index],
            right_group_index: local_index_by_group[entry.right_group_index],
            weight_re_bits: entry.weight_re.to_bits(),
            weight_im_bits: entry.weight_im.to_bits(),
            symmetry_factor_bits: entry.symmetry_factor.to_bits(),
        });
    }
    for component_entries in &mut entries_by_component {
        component_entries.sort_unstable();
    }
    if entries_by_component
        .iter()
        .skip(1)
        .any(|component_entries| component_entries != &entries_by_component[0])
    {
        return None;
    }

    let entries = entries_by_component[0]
        .iter()
        .map(|entry| ColorContractionEntry {
            left_group_index: entry.left_group_index,
            right_group_index: entry.right_group_index,
            weight_re: f64::from_bits(entry.weight_re_bits),
            weight_im: f64::from_bits(entry.weight_im_bits),
            symmetry_factor: f64::from_bits(entry.symmetry_factor_bits),
        })
        .collect::<Vec<_>>();
    let singleton_output_indices = component_group_indices
        .iter()
        .map(
            |group_index| match groups[*group_index].indices.as_slice() {
                [output_index] => Some(*output_index),
                _ => None,
            },
        )
        .collect::<Option<Vec<_>>>();
    Some(RepeatedColorContractionBlock {
        component_count: components.len(),
        component_group_indices,
        singleton_output_indices,
        all_weights_real: entries.iter().all(|entry| entry.weight_im == 0.0),
        entries,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct EvaluatorBatchProfile {
    leaf_input_pack_s: f64,
    legacy_evaluator_call_s: f64,
    evaluator_call_s: f64,
    output_gather_s: f64,
    leaf_input_copy_component_count: u64,
    output_gather_component_count: u64,
    backend_call_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct StageEvaluationProfile {
    input_pack_s: f64,
    evaluator: EvaluatorBatchProfile,
    output_assign_s: f64,
    input_copy_component_count: u64,
    output_assign_component_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AmplitudeEvaluationProfile {
    input_pack_s: f64,
    evaluator: EvaluatorBatchProfile,
    output_remap_s: f64,
    input_copy_component_count: u64,
    output_remap_component_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Debug, Default)]
struct RuntimeProfile {
    orchestration_s: f64,
    state_prepare_s: f64,
    state_clear_s: f64,
    source_fill_s: f64,
    momentum_input_setup_s: f64,
    momentum_setup_s: f64,
    model_parameter_setup_s: f64,
    stage_input_pack_s: f64,
    stage_leaf_input_pack_s: f64,
    stage_evaluator_call_s: f64,
    stage_backend_call_s: f64,
    stage_evaluator_output_gather_s: f64,
    stage_evaluator_s: f64,
    output_assign_s: f64,
    amplitude_input_pack_s: f64,
    amplitude_leaf_input_pack_s: f64,
    amplitude_evaluator_call_s: f64,
    amplitude_backend_call_s: f64,
    amplitude_evaluator_output_gather_s: f64,
    amplitude_output_remap_s: f64,
    amplitude_evaluator_s: f64,
    reduction_s: f64,
    resolved_reduction_materialization_s: f64,
    total_materialization_s: f64,
    final_output_copy_s: f64,
    total_s: f64,
    stage_input_pack_by_stage_s: Vec<f64>,
    stage_leaf_input_pack_by_stage_s: Vec<f64>,
    stage_evaluator_call_by_stage_s: Vec<f64>,
    stage_backend_call_by_stage_s: Vec<f64>,
    stage_evaluator_output_gather_by_stage_s: Vec<f64>,
    stage_output_assign_by_stage_s: Vec<f64>,
    state_component_count: u64,
    state_clear_component_count: u64,
    source_component_count: u64,
    momentum_component_count: u64,
    model_parameter_component_count: u64,
    stage_input_copy_component_count: u64,
    stage_leaf_input_copy_component_count: u64,
    stage_evaluator_output_gather_component_count: u64,
    stage_output_assign_component_count: u64,
    amplitude_input_copy_component_count: u64,
    amplitude_leaf_input_copy_component_count: u64,
    amplitude_evaluator_output_gather_component_count: u64,
    amplitude_output_remap_component_count: u64,
    evaluator_backend_call_count: u64,
    reduction_input_component_count: u64,
    resolved_materialized_component_count: u64,
    total_materialized_value_count: u64,
    final_output_copy_value_count: u64,
    scratch_reallocation_count: u64,
    eager_initialize_s: f64,
    eager_gather_s: f64,
    eager_kernel_call_s: f64,
    eager_invocation_scatter_s: f64,
    eager_finalization_s: f64,
    eager_scatter_finalization_s: f64,
    eager_closure_s: f64,
    eager_reduction_s: f64,
    eager_copy_out_s: f64,
}

// Saved SymJIT applications are native payloads and do not currently guarantee
// Rust's AArch64 callee-saved FP registers. Keep elapsed values integer-backed
// until the payload has returned, then use an opaque conversion boundary so
// LLVM cannot retain an f64 across that call.
#[inline(never)]
fn profile_duration_seconds(duration: Duration) -> f64 {
    std::hint::black_box(duration).as_secs_f64()
}

impl RuntimeProfile {
    fn add_sector(&mut self, sector: &RuntimeProfile) {
        self.orchestration_s += sector.orchestration_s;
        self.state_prepare_s += sector.state_prepare_s;
        self.state_clear_s += sector.state_clear_s;
        self.source_fill_s += sector.source_fill_s;
        self.momentum_input_setup_s += sector.momentum_input_setup_s;
        self.momentum_setup_s += sector.momentum_setup_s;
        self.model_parameter_setup_s += sector.model_parameter_setup_s;
        self.stage_input_pack_s += sector.stage_input_pack_s;
        self.stage_leaf_input_pack_s += sector.stage_leaf_input_pack_s;
        self.stage_evaluator_call_s += sector.stage_evaluator_call_s;
        self.stage_backend_call_s += sector.stage_backend_call_s;
        self.stage_evaluator_output_gather_s += sector.stage_evaluator_output_gather_s;
        self.stage_evaluator_s += sector.stage_evaluator_s;
        self.output_assign_s += sector.output_assign_s;
        self.amplitude_input_pack_s += sector.amplitude_input_pack_s;
        self.amplitude_leaf_input_pack_s += sector.amplitude_leaf_input_pack_s;
        self.amplitude_evaluator_call_s += sector.amplitude_evaluator_call_s;
        self.amplitude_backend_call_s += sector.amplitude_backend_call_s;
        self.amplitude_evaluator_output_gather_s += sector.amplitude_evaluator_output_gather_s;
        self.amplitude_output_remap_s += sector.amplitude_output_remap_s;
        self.amplitude_evaluator_s += sector.amplitude_evaluator_s;
        self.reduction_s += sector.reduction_s;
        self.resolved_reduction_materialization_s += sector.resolved_reduction_materialization_s;
        self.total_materialization_s += sector.total_materialization_s;
        self.final_output_copy_s += sector.final_output_copy_s;
        self.state_component_count += sector.state_component_count;
        self.state_clear_component_count += sector.state_clear_component_count;
        self.source_component_count += sector.source_component_count;
        self.momentum_component_count += sector.momentum_component_count;
        self.model_parameter_component_count += sector.model_parameter_component_count;
        self.stage_input_copy_component_count += sector.stage_input_copy_component_count;
        self.stage_leaf_input_copy_component_count += sector.stage_leaf_input_copy_component_count;
        self.stage_evaluator_output_gather_component_count +=
            sector.stage_evaluator_output_gather_component_count;
        self.stage_output_assign_component_count += sector.stage_output_assign_component_count;
        self.amplitude_input_copy_component_count += sector.amplitude_input_copy_component_count;
        self.amplitude_leaf_input_copy_component_count +=
            sector.amplitude_leaf_input_copy_component_count;
        self.amplitude_evaluator_output_gather_component_count +=
            sector.amplitude_evaluator_output_gather_component_count;
        self.amplitude_output_remap_component_count +=
            sector.amplitude_output_remap_component_count;
        self.evaluator_backend_call_count += sector.evaluator_backend_call_count;
        self.reduction_input_component_count += sector.reduction_input_component_count;
        self.resolved_materialized_component_count += sector.resolved_materialized_component_count;
        self.total_materialized_value_count += sector.total_materialized_value_count;
        self.final_output_copy_value_count += sector.final_output_copy_value_count;
        self.scratch_reallocation_count += sector.scratch_reallocation_count;
        self.eager_initialize_s += sector.eager_initialize_s;
        self.eager_gather_s += sector.eager_gather_s;
        self.eager_kernel_call_s += sector.eager_kernel_call_s;
        self.eager_invocation_scatter_s += sector.eager_invocation_scatter_s;
        self.eager_finalization_s += sector.eager_finalization_s;
        self.eager_scatter_finalization_s += sector.eager_scatter_finalization_s;
        self.eager_closure_s += sector.eager_closure_s;
        self.eager_reduction_s += sector.eager_reduction_s;
        self.eager_copy_out_s += sector.eager_copy_out_s;
        add_profile_vector(
            &mut self.stage_input_pack_by_stage_s,
            &sector.stage_input_pack_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_leaf_input_pack_by_stage_s,
            &sector.stage_leaf_input_pack_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_evaluator_call_by_stage_s,
            &sector.stage_evaluator_call_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_backend_call_by_stage_s,
            &sector.stage_backend_call_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_evaluator_output_gather_by_stage_s,
            &sector.stage_evaluator_output_gather_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_output_assign_by_stage_s,
            &sector.stage_output_assign_by_stage_s,
        );
    }
}

fn add_profile_vector(target: &mut Vec<f64>, source: &[f64]) {
    if target.len() < source.len() {
        target.resize(source.len(), 0.0);
    }
    for (index, value) in source.iter().enumerate() {
        target[index] += value;
    }
}

struct ExecutionRuntime {
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    external_count: usize,
    parameter_count: usize,
    value_parameter_count: usize,
    momentum_parameter_count: usize,
    current_count: usize,
    source_count: usize,
    interaction_count: usize,
    stage_count: usize,
    amplitude_output_count: usize,
    lc_topology_replay_enabled: bool,
    lc_topology_replay_mappings: Arc<LcTopologyReplayMappings>,
    lc_topology_replay_public_mappings: LcTopologyReplayMappings,
    lc_topology_replay_routes: Vec<Vec<LcTopologyReplaySectorRoute>>,
    lc_topology_replay_materialized_sector_ids: BTreeSet<i64>,
    lc_resolved_replay_plan: Option<Arc<LcResolvedReplayPlan>>,
    lc_resolved_replay_selection_cache:
        Option<(LcResolvedReplaySelectionKey, Arc<LcResolvedReplaySelection>)>,
    #[allow(dead_code)] // Loaded now and consumed by the subsequent selector-execution milestone.
    helicity_recurrence: Option<HelicityRecurrenceRuntime>,
    compiled_helicity_execution_plan: Option<CompiledHelicityExecutionPlan>,
    compiled_color_execution_plan: Option<CompiledColorExecutionPlan>,
    helicity_sum_runtime: Option<Box<ExecutionRuntime>>,
    // Lane runtimes are large recursive owners; boxing keeps their addresses stable and avoids
    // moving them when this selector index grows.
    #[allow(clippy::vec_box)]
    helicity_selector_runtimes: Vec<Box<ExecutionRuntime>>,
    helicity_selector_runtime_schedule_modes: Vec<HelicitySelectorScheduleMode>,
    helicity_selector_lane_by_domain: BTreeMap<usize, usize>,
    color_selector_runtimes: BTreeMap<i64, Box<ExecutionRuntime>>,
    runtime_unavailable_message: Option<String>,
    sources: Vec<GenericSourceRecordManifest>,
    momentum_slots: Vec<GenericMomentumSlotManifest>,
    external_is_initial: Vec<bool>,
    particle_masses: BTreeMap<i32, f64>,
    particle_mass_parameter_names: BTreeMap<i32, String>,
    normalization_factor: f64,
    normalization_color_factor: f64,
    normalization_average_factor: f64,
    normalization_identical_factor: f64,
    normalization_qcd_coupling_power: usize,
    normalization_electroweak_coupling_power: usize,
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    model_parameter_name_to_index: BTreeMap<String, usize>,
    model_parameter_runtime_slots: BTreeMap<String, RuntimeParameterSlots>,
    model_parameter_values_f64: Vec<f64>,
    model_parameter_evaluator: Option<ModelParameterEvaluatorRuntime>,
    physics_reduction_override: Option<crate::Reduction>,
    physics: Option<Arc<PhysicsRuntime>>,
    stages: Option<Vec<StageRuntime>>,
    amplitude_stage: Option<AmplitudeRuntime>,
    state_scratch_f64: Vec<Complex<f64>>,
    state_scratch_f64_requires_clear: bool,
    values_scratch_f64: Vec<f64>,
}

#[derive(Clone)]
struct PhysicsRuntime {
    manifest: ProcessPhysicsV1,
    helicity_index_by_id: BTreeMap<String, usize>,
    color_index_by_id: BTreeMap<String, usize>,
    reduction_by_group_id: BTreeMap<i64, crate::ReductionGroup>,
}

#[derive(Clone, Debug)]
struct ResolvedValues<T> {
    values: Vec<T>,
    point_count: usize,
    helicity_indices: Vec<usize>,
    color_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeRuntimeMetadata {
    pub abi_version: u32,
    pub schema_version: u32,
    pub execution_mode: String,
    pub prepared_backend: Option<String>,
    pub eager_effective_point_tile_size: Option<usize>,
    pub eager_workspace_bytes: Option<usize>,
    pub process: String,
    pub process_key: String,
    pub representative_process: String,
    pub representative_process_key: String,
    pub final_state_permutation_alias_of: Option<String>,
    pub color_accuracy: String,
    pub external_pdg_order: Vec<i32>,
    pub external_count: usize,
    pub current_count: usize,
    pub source_count: usize,
    pub interaction_count: usize,
    pub stage_count: usize,
    pub amplitude_output_count: usize,
}

/// Exact-required sections decoded lazily from one authenticated eager plan-v3
/// artifact. This bridge is intentionally private to pyAmpliCol's Python exact
/// executor; f64 execution never constructs it.
#[derive(Debug)]
pub struct NativeEagerExactSections {
    pub process_id: String,
    pub exact_schema: Value,
    pub reduction_groups: Value,
    pub selector_group_ids: Vec<u32>,
    pub selector_domains: Vec<Vec<u32>>,
    pub couplings: Vec<NativeEagerExactCoupling>,
    pub stages: Vec<NativeEagerExactStage>,
    pub invocations: Vec<NativeEagerExactInvocation>,
    pub attachments: Vec<NativeEagerExactAttachment>,
    pub finalizations: Vec<NativeEagerExactFinalization>,
    pub closures: Vec<NativeEagerExactClosure>,
}

#[derive(Debug)]
pub struct NativeEagerExactCoupling {
    pub real_parameter_id: u32,
    pub imaginary_parameter_id: u32,
    pub constant_real: String,
    pub constant_imaginary: String,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactStage {
    pub stage_index: u32,
    pub invocation_start: u64,
    pub invocation_count: u64,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub finalization_start: u64,
    pub finalization_count: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactInvocation {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub left_momentum_slot_id: u32,
    pub right_momentum_slot_id: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u8,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub selector_domain_id: u32,
}

#[derive(Debug)]
pub struct NativeEagerExactAttachment {
    pub result_current_id: u32,
    pub factor_numerators: Vec<[String; 2]>,
    pub factor_denominator: Option<[String; 2]>,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactFinalization {
    pub kernel_id: u32,
    pub current_id: u32,
    pub unpropagated_value_slot_id: u32,
    pub propagated_value_slot_id: u32,
    pub momentum_slot_id: u32,
    pub unpropagated_selector_domain_id: u32,
    pub propagated_selector_domain_id: u32,
}

#[derive(Debug)]
pub struct NativeEagerExactClosure {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub amplitude_index: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u8,
    pub factor_numerators: Vec<[String; 2]>,
    pub factor_denominator: Option<[String; 2]>,
    pub direct_coefficients: Option<Vec<[String; 2]>>,
    pub coherent_group_id: u32,
    pub selector_domain_id: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeRuntimeProfile {
    pub native_input_pack_s: f64,
    pub native_input_crossing_s: f64,
    pub orchestration_s: f64,
    pub state_prepare_s: f64,
    pub state_clear_s: f64,
    pub source_fill_s: f64,
    /// Exclusive momentum-input setup. Unlike `momentum_setup_s`, this does
    /// not include model-parameter setup.
    pub momentum_input_setup_s: f64,
    /// Backward-compatible aggregate of momentum-input and model-parameter
    /// setup.
    pub momentum_setup_s: f64,
    pub model_parameter_setup_s: f64,
    /// Top-level stage input envelope. In composed selected-chunk paths this
    /// owns the leaf gather; in full-stage paths it owns only the parent
    /// stage-input gather.
    pub stage_input_pack_s: f64,
    /// Internal attribution. This is owned by `stage_input_pack_s` for
    /// composed selected-chunk paths and by `stage_evaluator_call_s` for
    /// full-stage paths, so it must not be added to the top-level sum.
    pub stage_leaf_input_pack_s: f64,
    /// Top-level evaluator envelope. This includes leaf gathering for
    /// full-stage paths and excludes it for composed selected-chunk paths.
    pub stage_evaluator_call_s: f64,
    pub stage_backend_call_s: f64,
    pub stage_evaluator_output_gather_s: f64,
    pub stage_evaluator_s: f64,
    pub output_assign_s: f64,
    /// Top-level amplitude input envelope. In composed selected-chunk paths
    /// this owns the amplitude leaf gather.
    pub amplitude_input_pack_s: f64,
    /// Internal attribution owned either by `amplitude_input_pack_s` or
    /// `amplitude_evaluator_call_s`; never an additional top-level phase.
    pub amplitude_leaf_input_pack_s: f64,
    /// Top-level amplitude evaluator envelope. Full-stage paths include leaf
    /// gathering; composed selected-chunk paths exclude it.
    pub amplitude_evaluator_call_s: f64,
    pub amplitude_backend_call_s: f64,
    pub amplitude_evaluator_output_gather_s: f64,
    pub amplitude_output_remap_s: f64,
    pub amplitude_evaluator_s: f64,
    pub reduction_s: f64,
    /// Inclusive attribution: the resolved-result construction occurs inside
    /// `reduction_s` and must not be added to exclusive top-level phases.
    pub resolved_reduction_materialization_s: f64,
    pub total_materialization_s: f64,
    pub final_output_copy_s: f64,
    pub total_s: f64,
    pub stage_input_pack_by_stage_s: Vec<f64>,
    pub stage_leaf_input_pack_by_stage_s: Vec<f64>,
    pub stage_evaluator_call_by_stage_s: Vec<f64>,
    pub stage_backend_call_by_stage_s: Vec<f64>,
    pub stage_evaluator_output_gather_by_stage_s: Vec<f64>,
    pub stage_output_assign_by_stage_s: Vec<f64>,
    pub eager_initialize_s: f64,
    pub eager_gather_s: f64,
    pub eager_kernel_call_s: f64,
    pub eager_invocation_scatter_s: f64,
    pub eager_finalization_s: f64,
    pub eager_scatter_finalization_s: f64,
    pub eager_closure_s: f64,
    pub eager_reduction_s: f64,
    pub eager_copy_out_s: f64,
    pub selector_planner_s: f64,
    pub selector_gather_s: f64,
    pub selector_scatter_s: f64,
    pub selector_plan_kind: String,
    pub selector_group_sizes: Vec<usize>,
    pub selector_reordered_point_count: usize,
    pub selector_simd_lane_width: usize,
    pub selector_simd_occupancy: f64,
    pub native_input_component_count: u64,
    pub native_input_pack_bytes: u64,
    pub native_input_crossing_bytes: u64,
    /// Explicit nested native-input containers allocated for this call.
    pub native_input_container_allocation_count: u64,
    pub state_component_count: u64,
    pub state_clear_component_count: u64,
    pub source_component_count: u64,
    pub momentum_component_count: u64,
    pub model_parameter_component_count: u64,
    pub stage_input_copy_component_count: u64,
    pub stage_leaf_input_copy_component_count: u64,
    pub stage_evaluator_output_gather_component_count: u64,
    pub stage_output_assign_component_count: u64,
    pub amplitude_input_copy_component_count: u64,
    pub amplitude_leaf_input_copy_component_count: u64,
    pub amplitude_evaluator_output_gather_component_count: u64,
    pub amplitude_output_remap_component_count: u64,
    pub evaluator_backend_call_count: u64,
    pub reduction_input_component_count: u64,
    pub selector_gather_point_count: u64,
    pub selector_gather_bytes: u64,
    pub selector_scatter_value_count: u64,
    pub resolved_materialized_component_count: u64,
    pub total_materialized_value_count: u64,
    pub final_output_copy_value_count: u64,
    /// Capacity-changing reallocations observed in instrumented reusable hot
    /// buffers. This is intentionally not a process-wide allocation count.
    pub observed_scratch_reallocation_count: u64,
    /// Explicit final native output vector allocated for this call.
    pub native_output_allocation_count: u64,
}

impl From<RuntimeProfile> for NativeRuntimeProfile {
    fn from(profile: RuntimeProfile) -> Self {
        Self {
            native_input_pack_s: 0.0,
            native_input_crossing_s: 0.0,
            orchestration_s: profile.orchestration_s,
            state_prepare_s: profile.state_prepare_s,
            state_clear_s: profile.state_clear_s,
            source_fill_s: profile.source_fill_s,
            momentum_input_setup_s: profile.momentum_input_setup_s,
            momentum_setup_s: profile.momentum_setup_s,
            model_parameter_setup_s: profile.model_parameter_setup_s,
            stage_input_pack_s: profile.stage_input_pack_s,
            stage_leaf_input_pack_s: profile.stage_leaf_input_pack_s,
            stage_evaluator_call_s: profile.stage_evaluator_call_s,
            stage_backend_call_s: profile.stage_backend_call_s,
            stage_evaluator_output_gather_s: profile.stage_evaluator_output_gather_s,
            stage_evaluator_s: profile.stage_evaluator_s,
            output_assign_s: profile.output_assign_s,
            amplitude_input_pack_s: profile.amplitude_input_pack_s,
            amplitude_leaf_input_pack_s: profile.amplitude_leaf_input_pack_s,
            amplitude_evaluator_call_s: profile.amplitude_evaluator_call_s,
            amplitude_backend_call_s: profile.amplitude_backend_call_s,
            amplitude_evaluator_output_gather_s: profile.amplitude_evaluator_output_gather_s,
            amplitude_output_remap_s: profile.amplitude_output_remap_s,
            amplitude_evaluator_s: profile.amplitude_evaluator_s,
            reduction_s: profile.reduction_s,
            resolved_reduction_materialization_s: profile.resolved_reduction_materialization_s,
            total_materialization_s: profile.total_materialization_s,
            final_output_copy_s: profile.final_output_copy_s,
            total_s: profile.total_s,
            stage_input_pack_by_stage_s: profile.stage_input_pack_by_stage_s,
            stage_leaf_input_pack_by_stage_s: profile.stage_leaf_input_pack_by_stage_s,
            stage_evaluator_call_by_stage_s: profile.stage_evaluator_call_by_stage_s,
            stage_backend_call_by_stage_s: profile.stage_backend_call_by_stage_s,
            stage_evaluator_output_gather_by_stage_s: profile
                .stage_evaluator_output_gather_by_stage_s,
            stage_output_assign_by_stage_s: profile.stage_output_assign_by_stage_s,
            eager_initialize_s: profile.eager_initialize_s,
            eager_gather_s: profile.eager_gather_s,
            eager_kernel_call_s: profile.eager_kernel_call_s,
            eager_invocation_scatter_s: profile.eager_invocation_scatter_s,
            eager_finalization_s: profile.eager_finalization_s,
            eager_scatter_finalization_s: profile.eager_scatter_finalization_s,
            eager_closure_s: profile.eager_closure_s,
            eager_reduction_s: profile.eager_reduction_s,
            eager_copy_out_s: profile.eager_copy_out_s,
            selector_planner_s: 0.0,
            selector_gather_s: 0.0,
            selector_scatter_s: 0.0,
            selector_plan_kind: "none".to_string(),
            selector_group_sizes: Vec::new(),
            selector_reordered_point_count: 0,
            selector_simd_lane_width: 1,
            selector_simd_occupancy: 1.0,
            native_input_component_count: 0,
            native_input_pack_bytes: 0,
            native_input_crossing_bytes: 0,
            native_input_container_allocation_count: 0,
            state_component_count: profile.state_component_count,
            state_clear_component_count: profile.state_clear_component_count,
            source_component_count: profile.source_component_count,
            momentum_component_count: profile.momentum_component_count,
            model_parameter_component_count: profile.model_parameter_component_count,
            stage_input_copy_component_count: profile.stage_input_copy_component_count,
            stage_leaf_input_copy_component_count: profile.stage_leaf_input_copy_component_count,
            stage_evaluator_output_gather_component_count: profile
                .stage_evaluator_output_gather_component_count,
            stage_output_assign_component_count: profile.stage_output_assign_component_count,
            amplitude_input_copy_component_count: profile.amplitude_input_copy_component_count,
            amplitude_leaf_input_copy_component_count: profile
                .amplitude_leaf_input_copy_component_count,
            amplitude_evaluator_output_gather_component_count: profile
                .amplitude_evaluator_output_gather_component_count,
            amplitude_output_remap_component_count: profile.amplitude_output_remap_component_count,
            evaluator_backend_call_count: profile.evaluator_backend_call_count,
            reduction_input_component_count: profile.reduction_input_component_count,
            selector_gather_point_count: 0,
            selector_gather_bytes: 0,
            selector_scatter_value_count: 0,
            resolved_materialized_component_count: profile.resolved_materialized_component_count,
            total_materialized_value_count: profile.total_materialized_value_count,
            final_output_copy_value_count: profile.final_output_copy_value_count,
            observed_scratch_reallocation_count: profile.scratch_reallocation_count,
            native_output_allocation_count: 0,
        }
    }
}

impl NativeRuntimeProfile {
    fn validate_eager_top_level_accounting(&self) -> RusticolResult<()> {
        self.validate_top_level_accounting(true)
    }

    fn validate_compiled_top_level_accounting(&self) -> RusticolResult<()> {
        self.validate_top_level_accounting(false)
    }

    fn validate_top_level_accounting(&self, eager: bool) -> RusticolResult<()> {
        let mut phases = vec![
            ("native input pack", self.native_input_pack_s),
            ("native input crossing", self.native_input_crossing_s),
            ("runtime orchestration", self.orchestration_s),
            ("state preparation", self.state_prepare_s),
            ("state clearing", self.state_clear_s),
            ("source fill", self.source_fill_s),
            ("momentum input setup", self.momentum_input_setup_s),
            ("model parameter setup", self.model_parameter_setup_s),
        ];
        if eager {
            phases.push(("inclusive eager execution", self.stage_evaluator_call_s));
        } else {
            phases.extend([
                ("stage input pack", self.stage_input_pack_s),
                ("stage evaluator calls", self.stage_evaluator_call_s),
                ("stage output assignment", self.output_assign_s),
                ("amplitude input pack", self.amplitude_input_pack_s),
                ("amplitude evaluator calls", self.amplitude_evaluator_call_s),
                ("reduction", self.reduction_s),
            ]);
        }
        phases.extend([
            ("total materialization", self.total_materialization_s),
            ("final output copy", self.final_output_copy_s),
            ("selector planning", self.selector_planner_s),
            ("selector gather", self.selector_gather_s),
            ("selector scatter", self.selector_scatter_s),
        ]);
        if !self.total_s.is_finite() || self.total_s < 0.0 {
            return Err(RusticolError::internal(format!(
                "native profile has invalid wall time {:.9e}s",
                self.total_s,
            )));
        }
        for (label, value) in &phases {
            if !value.is_finite() || *value < 0.0 {
                return Err(RusticolError::internal(format!(
                    "native profile has invalid {label} time {value:.9e}s"
                )));
            }
        }
        let accounted = phases.iter().map(|(_, value)| value).sum::<f64>();
        let tolerance = 1.0e-9_f64.max(self.total_s * 1.0e-12);
        if accounted > self.total_s + tolerance {
            return Err(RusticolError::internal(format!(
                "native profile exclusive top-level phases account for {accounted:.9e}s, exceeding wall time {wall:.9e}s",
                wall = self.total_s,
            )));
        }
        Ok(())
    }

    #[inline(never)]
    fn accumulate(&mut self, other: &Self) {
        self.native_input_pack_s += other.native_input_pack_s;
        self.native_input_crossing_s += other.native_input_crossing_s;
        self.orchestration_s += other.orchestration_s;
        self.state_prepare_s += other.state_prepare_s;
        self.state_clear_s += other.state_clear_s;
        self.source_fill_s += other.source_fill_s;
        self.momentum_input_setup_s += other.momentum_input_setup_s;
        self.momentum_setup_s += other.momentum_setup_s;
        self.model_parameter_setup_s += other.model_parameter_setup_s;
        self.stage_input_pack_s += other.stage_input_pack_s;
        self.stage_leaf_input_pack_s += other.stage_leaf_input_pack_s;
        self.stage_evaluator_call_s += other.stage_evaluator_call_s;
        self.stage_backend_call_s += other.stage_backend_call_s;
        self.stage_evaluator_output_gather_s += other.stage_evaluator_output_gather_s;
        self.stage_evaluator_s += other.stage_evaluator_s;
        self.output_assign_s += other.output_assign_s;
        self.amplitude_input_pack_s += other.amplitude_input_pack_s;
        self.amplitude_leaf_input_pack_s += other.amplitude_leaf_input_pack_s;
        self.amplitude_evaluator_call_s += other.amplitude_evaluator_call_s;
        self.amplitude_backend_call_s += other.amplitude_backend_call_s;
        self.amplitude_evaluator_output_gather_s += other.amplitude_evaluator_output_gather_s;
        self.amplitude_output_remap_s += other.amplitude_output_remap_s;
        self.amplitude_evaluator_s += other.amplitude_evaluator_s;
        self.reduction_s += other.reduction_s;
        self.resolved_reduction_materialization_s += other.resolved_reduction_materialization_s;
        self.total_materialization_s += other.total_materialization_s;
        self.final_output_copy_s += other.final_output_copy_s;
        self.total_s += other.total_s;
        accumulate_profile_stages(
            &mut self.stage_input_pack_by_stage_s,
            &other.stage_input_pack_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_leaf_input_pack_by_stage_s,
            &other.stage_leaf_input_pack_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_evaluator_call_by_stage_s,
            &other.stage_evaluator_call_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_backend_call_by_stage_s,
            &other.stage_backend_call_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_evaluator_output_gather_by_stage_s,
            &other.stage_evaluator_output_gather_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_output_assign_by_stage_s,
            &other.stage_output_assign_by_stage_s,
        );
        self.eager_initialize_s += other.eager_initialize_s;
        self.eager_gather_s += other.eager_gather_s;
        self.eager_kernel_call_s += other.eager_kernel_call_s;
        self.eager_invocation_scatter_s += other.eager_invocation_scatter_s;
        self.eager_finalization_s += other.eager_finalization_s;
        self.eager_scatter_finalization_s += other.eager_scatter_finalization_s;
        self.eager_closure_s += other.eager_closure_s;
        self.eager_reduction_s += other.eager_reduction_s;
        self.eager_copy_out_s += other.eager_copy_out_s;
        self.selector_planner_s += other.selector_planner_s;
        self.selector_gather_s += other.selector_gather_s;
        self.selector_scatter_s += other.selector_scatter_s;
        self.native_input_component_count += other.native_input_component_count;
        self.native_input_pack_bytes += other.native_input_pack_bytes;
        self.native_input_crossing_bytes += other.native_input_crossing_bytes;
        self.native_input_container_allocation_count +=
            other.native_input_container_allocation_count;
        self.state_component_count += other.state_component_count;
        self.state_clear_component_count += other.state_clear_component_count;
        self.source_component_count += other.source_component_count;
        self.momentum_component_count += other.momentum_component_count;
        self.model_parameter_component_count += other.model_parameter_component_count;
        self.stage_input_copy_component_count += other.stage_input_copy_component_count;
        self.stage_leaf_input_copy_component_count += other.stage_leaf_input_copy_component_count;
        self.stage_evaluator_output_gather_component_count +=
            other.stage_evaluator_output_gather_component_count;
        self.stage_output_assign_component_count += other.stage_output_assign_component_count;
        self.amplitude_input_copy_component_count += other.amplitude_input_copy_component_count;
        self.amplitude_leaf_input_copy_component_count +=
            other.amplitude_leaf_input_copy_component_count;
        self.amplitude_evaluator_output_gather_component_count +=
            other.amplitude_evaluator_output_gather_component_count;
        self.amplitude_output_remap_component_count += other.amplitude_output_remap_component_count;
        self.evaluator_backend_call_count += other.evaluator_backend_call_count;
        self.reduction_input_component_count += other.reduction_input_component_count;
        self.selector_gather_point_count += other.selector_gather_point_count;
        self.selector_gather_bytes += other.selector_gather_bytes;
        self.selector_scatter_value_count += other.selector_scatter_value_count;
        self.resolved_materialized_component_count += other.resolved_materialized_component_count;
        self.total_materialized_value_count += other.total_materialized_value_count;
        self.final_output_copy_value_count += other.final_output_copy_value_count;
        self.observed_scratch_reallocation_count += other.observed_scratch_reallocation_count;
        self.native_output_allocation_count += other.native_output_allocation_count;
        if self.selector_plan_kind == "none" && other.selector_plan_kind != "none" {
            self.selector_plan_kind
                .clone_from(&other.selector_plan_kind);
            self.selector_group_sizes
                .clone_from(&other.selector_group_sizes);
            self.selector_reordered_point_count = other.selector_reordered_point_count;
            self.selector_simd_lane_width = other.selector_simd_lane_width;
            self.selector_simd_occupancy = other.selector_simd_occupancy;
        }
    }
}

fn accumulate_profile_stages(target: &mut Vec<f64>, source: &[f64]) {
    if target.len() < source.len() {
        target.resize(source.len(), 0.0);
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeProfiledEvaluation {
    pub values: Vec<f64>,
    pub profile: NativeRuntimeProfile,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeResolvedEvaluation {
    /// Row-major storage with layout `[point][helicity][color]`.
    pub values: Vec<f64>,
    pub point_count: usize,
    pub helicity_ids: Vec<String>,
    pub color_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeDecimalEvaluation {
    pub values: Vec<String>,
    pub decimal_digits: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeDecimalResolvedEvaluation {
    /// Row-major storage with layout `[point][helicity][color]`.
    pub values: Vec<String>,
    pub totals: Vec<String>,
    pub point_count: usize,
    pub helicity_ids: Vec<String>,
    pub color_ids: Vec<String>,
    pub decimal_digits: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeExternalParticle {
    pub label: usize,
    pub index: usize,
    pub side: String,
    pub role: String,
    pub particle: String,
    pub outgoing_particle: String,
    pub pdg: i32,
    pub outgoing_pdg: i32,
    pub particle_class: String,
    pub momentum_slot: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeHelicityConfiguration {
    pub id: String,
    pub index: usize,
    pub helicities: Vec<i32>,
    pub representative_id: String,
    pub computed: bool,
    pub structural_zero: bool,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeColorComponent {
    pub id: String,
    pub index: usize,
    pub kind: String,
    pub word: Vec<usize>,
    pub representative_id: String,
    pub computed: bool,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeModelParameter {
    pub name: String,
    pub kind: String,
    pub parameter_index: usize,
    pub default: f64,
    pub default_imaginary: f64,
    pub mutable: bool,
}

impl NativeResolvedEvaluation {
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.point_count,
            self.helicity_ids.len(),
            self.color_ids.len(),
        )
    }

    pub fn totals(&self) -> Vec<f64> {
        let component_count = self.helicity_ids.len() * self.color_ids.len();
        self.values
            .chunks(component_count)
            .map(|point| point.iter().sum())
            .collect()
    }
}

impl NativeDecimalResolvedEvaluation {
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.point_count,
            self.helicity_ids.len(),
            self.color_ids.len(),
        )
    }
}

/// Python-independent schema-v3 process runtime.
///
/// The input momentum layout is `[point][external particle][E, px, py, pz]`.
/// Instances are mutable and must not be called concurrently; independent
/// instances can be used from separate threads.
pub struct NativeRuntime {
    root: PathBuf,
    runtime: ExecutionRuntime,
    execution_lane: NativeExecutionLane,
    process: String,
    process_key: String,
    input_crossing_map: Option<Vec<InputCrossingMapEntry>>,
    final_state_permutation_alias_of: Option<String>,
    physics_v1: ProcessPhysicsV1,
    warnings_muted: bool,
    warned_kinds: BTreeSet<String>,
    pending_warnings: Vec<String>,
    point_selector_scratch: PointSelectorExecutionScratch,
    selector_simd_lane_width: usize,
}

enum NativeExecutionLane {
    Compiled,
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    Eager(Box<EagerNativeRuntime>),
}

impl NativeExecutionLane {
    fn is_eager(&self) -> bool {
        match self {
            Self::Compiled => false,
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            Self::Eager(_) => true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RuntimeParameterSlots {
    real: usize,
    imaginary: Option<usize>,
}

struct ModelParameterEvaluatorRuntime {
    input_parameter_indices: Vec<usize>,
    outputs: Vec<GenericDerivedParameterOutputManifest>,
    evaluator: EvaluatorGroup,
}

struct StageRuntime {
    outputs: Vec<(usize, usize)>,
    output_spans: Vec<(usize, usize, usize)>,
    chunk_outputs: Vec<Vec<(usize, usize)>>,
    chunk_output_spans: Vec<Vec<(usize, usize, usize)>>,
    input_components: Option<Vec<usize>>,
    input_spans: Vec<(usize, usize, usize)>,
    parameter_scratch_f64: Vec<Complex<f64>>,
    output_scratch_f64: Vec<Complex<f64>>,
    evaluator: EvaluatorGroup,
}

struct AmplitudeRuntime {
    output_length: usize,
    raw_sum_weights: Vec<f64>,
    raw_sum_all_sector_weights: Vec<f64>,
    raw_sum_color_sector_ids: Vec<Option<i64>>,
    raw_sum_groups: Vec<RawSumGroup>,
    has_coherent_groups: bool,
    color_contraction: Option<ColorContractionRuntime>,
    input_components: Option<Vec<usize>>,
    input_spans: Vec<(usize, usize, usize)>,
    parameter_scratch_f64: Vec<Complex<f64>>,
    evaluator_output_scratch_f64: Vec<Complex<f64>>,
    output_scratch_f64: Vec<Complex<f64>>,
    resolved_source_row_scratch_f64: Vec<f64>,
    resolved_target_row_scratch_f64: Vec<f64>,
    evaluator_output_order: Option<Vec<usize>>,
    evaluator: EvaluatorGroup,
}

mod runtime_load;
use runtime_load::*;

mod model_parameters;
use model_parameters::*;

mod evaluation;
use evaluation::write_resolved_f64_totals;
mod helicity_lane;
use helicity_lane::*;
mod momentum;
use momentum::*;
mod sources;

mod validation;
use validation::*;

mod native_runtime;

mod artifact_load;
use artifact_load::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_backend;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_backend::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[allow(dead_code)]
mod eager_manifest;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_manifest::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_common;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_decode;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_manifest;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[allow(dead_code)]
mod eager_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_load::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_lane;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_lane::*;

mod physics;

mod point_selectors;
use point_selectors::*;

#[path = "evaluator.rs"]
mod evaluator;
use evaluator::*;

#[path = "wavefunctions.rs"]
mod wavefunctions;
use wavefunctions::*;

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "engine/source_metadata_tests.rs"]
mod source_metadata_tests;

#[cfg(test)]
#[path = "engine/quantum_number_flow_tests.rs"]
mod quantum_number_flow_tests;

#[cfg(test)]
#[path = "engine/contraction_metadata_tests.rs"]
mod contraction_metadata_tests;

#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
#[path = "engine/eager_integration_tests.rs"]
mod eager_integration_tests;

#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
#[path = "engine/eager_v3_manifest_tests.rs"]
mod eager_v3_manifest_tests;
