// SPDX-License-Identifier: 0BSD

#[cfg(not(feature = "symbolica-runtime"))]
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
#[cfg(feature = "symbolica-runtime")]
use symbolica::evaluate::JITCompiledEvaluator;
#[cfg(feature = "symbolica-runtime")]
use symbolica::prelude::{
    BatchEvaluator, CompiledComplexEvaluator, Complex, DoubleFloat, EvaluationDomain,
    ExpressionEvaluator, Float, JITCompilationSettings, Rational, Real, RealLike,
};

use crate::{
    ColorComponent as PhysicsColorComponentV1, PROCESS_ARTIFACT_SCHEMA_VERSION, PayloadRole,
    ProcessPhysics as ProcessPhysicsV1, RusticolError, RusticolResult, VerifiedArtifact,
};

const MAX_LC_TOPOLOGY_REPLAY_EXPANDED_POINTS: usize = 8192;
const LC_SECTOR_SELECTOR_PARAMETER: &str = "runtime.lc_sector_id";

type LcTopologyReplayMappings = Vec<Vec<(usize, usize)>>;
type LcTopologyReplayData = (LcTopologyReplayMappings, Vec<f64>);

pub const SYMJIT_APPLICATION_RUNTIME_CAPABILITY: &str = "symjit.application.complex-f64.v1";
pub const SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY: &str =
    "symbolica.legacy-jit-container.complex-f64.v1";
pub const SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY: &str = "symbolica.compiled-cpp.complex-f64.v1";
pub const SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY: &str = "symbolica.compiled-asm.complex-f64.v1";
pub const SYMJIT_APPLICATION_STORAGE_ABI: &str = "symjit-application-storage-v3";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeCapability {
    SymjitApplicationComplexF64V1,
    SymbolicaLegacyJitContainerComplexF64V1,
    SymbolicaCompiledCppComplexF64V1,
    SymbolicaCompiledAsmComplexF64V1,
}

impl RuntimeCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
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
        #[cfg(feature = "f64-symjit")]
        SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        #[cfg(feature = "symbolica-runtime")]
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        #[cfg(feature = "symbolica-runtime")]
        SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
        #[cfg(feature = "symbolica-runtime")]
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
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorSetManifest {
    kind: String,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    #[serde(default)]
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    #[serde(default)]
    model_parameter_evaluator: Option<GenericModelParameterEvaluatorManifest>,
    stage_evaluators: Option<GenericStageEvaluatorArtifactsManifest>,
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

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayManifest {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    replayed_sector_count: usize,
    #[serde(default)]
    groups: Vec<LcTopologyReplayGroupManifest>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayGroupManifest {
    representative_sector_id: i64,
    materialized_sector_id: i64,
    #[serde(default)]
    active_sector_ids: Vec<i64>,
    #[serde(default)]
    sector_permutations: Vec<LcTopologyReplaySectorPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplaySectorPermutationManifest {
    sector_id: i64,
    #[serde(default = "default_lc_topology_replay_weight")]
    weight: f64,
    #[serde(default)]
    label_permutation: Vec<LcTopologyReplayLabelPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayLabelPermutationManifest {
    representative_label: usize,
    sector_label: usize,
}

fn default_lc_topology_replay_weight() -> f64 {
    1.0
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

#[derive(Clone, Debug, Deserialize)]
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
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceFillManifest {
    source_count: usize,
    sources: Vec<GenericSourceRecordManifest>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericSourceOrientationManifest {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum GenericParticleStatisticsManifest {
    Boson,
    Fermion,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
enum GenericSourceSpinStateManifest {
    Scalar(i32),
    Components(Vec<i32>),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
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

#[derive(Clone, Debug, Deserialize, PartialEq)]
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

#[derive(Clone, Debug, Deserialize)]
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

#[derive(Clone, Debug, Deserialize)]
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
    coherent_group_id: Option<Value>,
    helicity_weight: f64,
    #[serde(default)]
    all_sector_weight: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
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
        chunks: Vec<EvaluatorManifest>,
    },
}

struct EvaluatorGroup {
    evaluators: Vec<LoadedEvaluator>,
    output_len: usize,
    chunk_scratch_f64: Vec<Complex<f64>>,
}

enum F64Evaluator {
    #[cfg(feature = "f64-symjit")]
    SymjitApplication(SymjitApplicationEvaluator),
    #[cfg(feature = "symbolica-runtime")]
    Compiled(CompiledComplexEvaluator),
    #[cfg(feature = "symbolica-runtime")]
    Jit(JITCompiledEvaluator<Complex<f64>>),
}

struct LoadedEvaluator {
    eval: F64Evaluator,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval: Option<ExpressionEvaluator<Complex<Rational>>>,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval_path: Option<PathBuf>,
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
    group_scratch_f64: Vec<Complex<f64>>,
}

struct ColorContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re: f64,
    weight_im: f64,
    symmetry_factor: f64,
}

#[derive(Clone, Debug, Default)]
struct RuntimeProfile {
    source_fill_s: f64,
    momentum_setup_s: f64,
    stage_input_pack_s: f64,
    stage_evaluator_call_s: f64,
    stage_evaluator_s: f64,
    output_assign_s: f64,
    amplitude_input_pack_s: f64,
    amplitude_evaluator_call_s: f64,
    amplitude_evaluator_s: f64,
    reduction_s: f64,
    total_s: f64,
    stage_input_pack_by_stage_s: Vec<f64>,
    stage_evaluator_call_by_stage_s: Vec<f64>,
    stage_output_assign_by_stage_s: Vec<f64>,
}

impl RuntimeProfile {
    fn add_sector(&mut self, sector: &RuntimeProfile) {
        self.source_fill_s += sector.source_fill_s;
        self.momentum_setup_s += sector.momentum_setup_s;
        self.stage_input_pack_s += sector.stage_input_pack_s;
        self.stage_evaluator_call_s += sector.stage_evaluator_call_s;
        self.stage_evaluator_s += sector.stage_evaluator_s;
        self.output_assign_s += sector.output_assign_s;
        self.amplitude_input_pack_s += sector.amplitude_input_pack_s;
        self.amplitude_evaluator_call_s += sector.amplitude_evaluator_call_s;
        self.amplitude_evaluator_s += sector.amplitude_evaluator_s;
        self.reduction_s += sector.reduction_s;
        add_profile_vector(
            &mut self.stage_input_pack_by_stage_s,
            &sector.stage_input_pack_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_evaluator_call_by_stage_s,
            &sector.stage_evaluator_call_by_stage_s,
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
    lc_topology_replay_mappings: LcTopologyReplayMappings,
    lc_topology_replay_weights: Vec<f64>,
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
    physics: Option<PhysicsRuntime>,
    stages: Option<Vec<StageRuntime>>,
    amplitude_stage: Option<AmplitudeRuntime>,
    state_scratch_f64: Vec<Complex<f64>>,
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
    process: String,
    process_key: String,
    input_crossing_map: Option<Vec<InputCrossingMapEntry>>,
    final_state_permutation_alias_of: Option<String>,
    physics_v1: ProcessPhysicsV1,
    warnings_muted: bool,
    warned_kinds: BTreeSet<String>,
    pending_warnings: Vec<String>,
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
    output_scratch_f64: Vec<Complex<f64>>,
    evaluator: EvaluatorGroup,
}

mod runtime_load;
use runtime_load::*;

mod model_parameters;
use model_parameters::*;

mod evaluation;
mod sources;

mod validation;
use validation::*;

mod native_runtime;

mod artifact_load;
use artifact_load::*;

mod physics;

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
