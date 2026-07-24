// SPDX-License-Identifier: 0BSD

//! Bounded parsing and semantic validation for recurrence execution manifests.

use crate::recurrence::direct_backend::RECURRENCE_DIRECT_BACKEND_ABI;
use crate::recurrence::template::RECURRENCE_TEMPLATE_INPUT_ABI;
use crate::recurrence::{
    RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_DIRECT_PLAN_MEMBER, RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_LC_COLOR_CAPABILITY, RECURRENCE_PLAN_ABI, RECURRENCE_RUNTIME_CAPABILITY,
    RECURRENCE_RUNTIME_KIND, RECURRENCE_RUNTIME_LAYOUT_ABI,
};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolError, RusticolResult};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;

pub(super) const RECURRENCE_RUNTIME_STORAGE_ABI: &str = "pacbin-v1";
pub(super) const RECURRENCE_RUNTIME_CONTAINER_KIND: &str =
    "pyamplicol-recurrence-runtime-container";
pub(super) const RECURRENCE_RUNTIME_CONTAINER_SCHEMA: u16 = 1;
pub(super) const RECURRENCE_RUNTIME_CONTAINER_PATH: &str = "recurrence-runtime.pacbin";
pub(super) const RECURRENCE_KERNEL_PACK_MANIFEST_PATH: &str = "model/eager-kernel-pack.json";
pub(super) const RECURRENCE_KERNEL_PAYLOAD_ROOT: &str = "model/eager-kernels";

const MAX_EXECUTION_MANIFEST_BYTES: usize = 1 << 20;
const MAX_RUNTIME_CONTAINER_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const MAX_POINT_TILE_SIZE: u64 = 1_048_576;
const MAX_WORKSPACE_MIB: u64 = 4096;
const MAX_SUMMARY_COUNT: u64 = 1 << 48;
const MAX_METADATA_ROWS: usize = 1 << 20;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_SOURCE_COMPONENTS: u64 = 1 << 20;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) builder_input_abi: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) direct_backend_abi: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) kernel_pack: RecurrenceKernelPackReference,
    pub(super) runtime_options: RecurrenceRuntimeOptions,
    pub(super) runtime_metadata: RecurrenceRuntimeMetadata,
    pub(super) plan: RecurrencePlanSummary,
    pub(super) recurrence_summary: RecurrenceSummary,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceKernelPackReference {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeOptions {
    pub(super) point_tile_size: u64,
    pub(super) workspace_mib: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrencePlanSummary {
    pub(super) kind: String,
    pub(super) builder_input_abi: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) direct_backend_abi: String,
    pub(super) builder_input_sha256: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) runtime_container: RecurrenceRuntimeContainer,
    pub(super) inspection_summary: RecurrenceInspectionSummary,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeContainer {
    pub(super) kind: String,
    pub(super) schema_version: u16,
    pub(super) storage_abi: String,
    pub(super) path: String,
    pub(super) plan_member_path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) member_count: u64,
    pub(super) unpacked_size_bytes: u64,
    pub(super) index_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceLcFlowLayout {
    TopologyReplay,
    AllFlowUnion,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceInspectionSummary {
    pub(super) execution_mode: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) process_id: String,
    pub(super) lc_flow_layout: RecurrenceLcFlowLayout,
    pub(super) parameter_count: u64,
    pub(super) sector_count: u64,
    pub(super) prepared_kernel_count: u64,
    pub(super) direct_executor_count: u64,
    pub(super) semantic_digest: String,
    pub(super) runtime_layout_digest: String,
    pub(super) schedule: RecurrenceScheduleSummary,
    pub(super) direct_arena: RecurrenceDirectArenaSummary,
    pub(super) runtime_container_member: RecurrenceRuntimeContainerMember,
    pub(super) generation_timings_seconds: RecurrenceGenerationTimings,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceScheduleSummary {
    pub(super) current_count: u64,
    pub(super) source_row_count: u64,
    pub(super) contribution_count: u64,
    pub(super) finalization_count: u64,
    pub(super) amplitude_destination_count: u64,
    pub(super) closure_term_count: u64,
    pub(super) replay_target_count: u64,
    pub(super) retained_helicity_count: u64,
    pub(super) resolved_helicity_count: u64,
    pub(super) exact_factor_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceDirectArenaSummary {
    pub(super) semantic_component_count: u64,
    pub(super) current_arena_components: u64,
    pub(super) arena_component_reuse_count: u64,
    pub(super) momentum_form_count: u64,
    pub(super) selector_domain_count: u64,
    pub(super) row_group_count: u64,
    pub(super) packed_input_bytes: u64,
    pub(super) packed_output_bytes: u64,
    pub(super) scatter_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeContainerMember {
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) container_size_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenerationTimings {
    pub(super) python_extraction: f64,
    pub(super) catalog_authentication: f64,
    pub(super) semantic_construction: f64,
    pub(super) direct_lowering: f64,
    pub(super) serialization: f64,
    pub(super) native_total: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSummary {
    pub(super) lc_flow_layout: RecurrenceLcFlowLayout,
    pub(super) builder_input_abi: String,
    pub(super) template_input_abi: String,
    pub(super) current_count: u64,
    pub(super) contribution_count: u64,
    pub(super) closure_term_count: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeMetadata {
    pub(super) public_color_flows: Vec<RecurrencePublicColorFlow>,
    pub(super) runtime_parameters: Vec<RecurrenceRuntimeParameter>,
    pub(super) prepared_parameter_defaults: Vec<[f64; 2]>,
    pub(super) parameter_projection: Vec<RecurrenceParameterProjection>,
    pub(super) source_templates: Vec<RecurrenceSourceTemplate>,
    pub(super) external_legs: Vec<RecurrenceExternalLeg>,
    pub(super) particle_masses: Vec<RecurrenceParticleMass>,
    pub(super) normalization: RecurrenceNormalization,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrencePublicColorFlow {
    pub(super) public_id: String,
    pub(super) construction_sector_id: u32,
    pub(super) target_sector_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeParameter {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) parameter_index: u32,
    pub(super) default: f64,
    #[serde(default)]
    pub(super) runtime_name: Option<String>,
    #[serde(default)]
    pub(super) complex_component: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceParameterProjection {
    pub(super) runtime_slot: u32,
    pub(super) runtime_name: String,
    pub(super) parameter_template_id: u32,
    #[serde(deserialize_with = "deserialize_required_nullable_u32")]
    pub(super) prepared_parameter_id: Option<u32>,
    pub(super) component: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSourceTemplate {
    pub(super) source_template_id: u32,
    pub(super) current_state_template_id: u32,
    pub(super) dimension: u64,
    pub(super) helicity: i32,
    pub(super) chirality: i32,
    pub(super) spin_state: i32,
    pub(super) source_ir: RecurrenceGenericSourceIr,
    pub(super) crossing: RecurrenceGenericCrossingIr,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceExternalLeg {
    pub(super) source_slot: u32,
    pub(super) public_label: u32,
    pub(super) physical_pdg: i32,
    pub(super) outgoing_pdg: i32,
    pub(super) is_initial: bool,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceParticleMass {
    pub(super) outgoing_pdg: i32,
    pub(super) mass: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceNormalization {
    pub(super) color_accuracy: String,
    pub(super) color_factor: f64,
    pub(super) average_factor: f64,
    pub(super) identical_factor: f64,
    pub(super) global_coupling_factor: f64,
    #[serde(default, deserialize_with = "deserialize_optional_u64_if_present")]
    pub(super) qcd_coupling_power: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_u64_if_present")]
    pub(super) electroweak_coupling_power: Option<u64>,
    pub(super) couplings_in_stage_evaluators: bool,
    pub(super) coupling_policy: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceSourceOrientation {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum RecurrenceParticleStatistics {
    Boson,
    Fermion,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum RecurrenceWavefunctionFamily {
    Scalar,
    Fermion,
    Vector,
    Spin2,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceMomentumTransform {
    Identity,
    NegateFourMomentum,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(untagged)]
pub(super) enum RecurrenceSourceSpinState {
    Scalar(i32),
    Components(Vec<i32>),
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericSourceStateIr {
    pub(super) helicity: i32,
    pub(super) chirality: i32,
    pub(super) spin_state: RecurrenceSourceSpinState,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericCrossingIr {
    pub(super) momentum_transform: RecurrenceMomentumTransform,
    pub(super) helicity_factor: i32,
    pub(super) chirality_factor: i32,
    pub(super) spin_state_factor: i32,
    #[serde(deserialize_with = "deserialize_recurrence_phase")]
    pub(super) phase: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericParticleIdentityIr {
    pub(super) canonical_id: String,
    pub(super) species_id: String,
    pub(super) anti_canonical_id: String,
    pub(super) display_name: String,
    pub(super) anti_display_name: String,
    pub(super) pdg_label: i32,
    pub(super) anti_pdg_label: i32,
    pub(super) orientation: RecurrenceSourceOrientation,
    pub(super) self_conjugate: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericSourceIr {
    pub(super) identity: RecurrenceGenericParticleIdentityIr,
    pub(super) statistics: RecurrenceParticleStatistics,
    pub(super) wavefunction_family: RecurrenceWavefunctionFamily,
    pub(super) component_dimension: u64,
    pub(super) states: Vec<RecurrenceGenericSourceStateIr>,
    pub(super) crossing: RecurrenceGenericCrossingIr,
    pub(super) basis: String,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub(super) mass_parameter: Option<String>,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub(super) width_parameter: Option<String>,
}

impl RecurrenceExecutionManifest {
    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != RECURRENCE_RUNTIME_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "recurrence execution manifest does not match outer process {:?}",
                outer.id
            )));
        }
        validate_capabilities(
            &self.required_runtime_capabilities,
            "recurrence execution manifest",
        )?;
        validate_capabilities(
            &outer.required_runtime_capabilities,
            "outer recurrence process",
        )?;
        validate_direct_contract(
            &self.builder_input_abi,
            &self.recurrence_plan_abi,
            &self.runtime_layout_abi,
            &self.direct_template_abi,
            &self.direct_backend_abi,
            "recurrence execution manifest",
        )?;
        parse_sha256(
            &self.prepared_kernel_pack_digest,
            "recurrence prepared-kernel pack",
        )?;
        parse_sha256(
            &self.direct_template_catalog_digest,
            "recurrence direct-template catalog",
        )?;
        if self.kernel_pack.manifest_path != RECURRENCE_KERNEL_PACK_MANIFEST_PATH
            || self.kernel_pack.payload_root != RECURRENCE_KERNEL_PAYLOAD_ROOT
        {
            return Err(RusticolError::security(
                "recurrence kernel-pack paths are not canonical",
            ));
        }
        bounded_len("external PDG order", self.external_pdg_order.len())?;
        if self.external_pdg_order.is_empty() {
            return Err(RusticolError::artifact(
                "recurrence execution manifest has no external particles",
            ));
        }
        self.runtime_options.validate()?;
        self.plan.validate(
            &self.key,
            &self.prepared_kernel_pack_digest,
            &self.direct_template_catalog_digest,
        )?;
        self.recurrence_summary.validate()?;
        self.runtime_metadata.validate(
            &self.external_pdg_order,
            &self.color_accuracy,
            &self.plan.inspection_summary,
        )?;

        let inspection = &self.plan.inspection_summary;
        let schedule = inspection.schedule;
        if inspection.lc_flow_layout != self.recurrence_summary.lc_flow_layout
            || schedule.current_count != self.recurrence_summary.current_count
            || schedule.contribution_count != self.recurrence_summary.contribution_count
            || schedule.closure_term_count != self.recurrence_summary.closure_term_count
        {
            return Err(RusticolError::integrity(
                "recurrence inspection counts disagree with the recurrence summary",
            ));
        }
        Ok(())
    }
}

impl RecurrenceRuntimeOptions {
    fn validate(self) -> RusticolResult<()> {
        if !(1..=MAX_POINT_TILE_SIZE).contains(&self.point_tile_size) {
            return Err(RusticolError::artifact(format!(
                "recurrence point_tile_size must be in 1..={MAX_POINT_TILE_SIZE}"
            )));
        }
        if !(1..=MAX_WORKSPACE_MIB).contains(&self.workspace_mib) {
            return Err(RusticolError::artifact(format!(
                "recurrence workspace_mib must be in 1..={MAX_WORKSPACE_MIB}"
            )));
        }
        Ok(())
    }
}

impl RecurrencePlanSummary {
    fn validate(
        &self,
        process_key: &str,
        prepared_kernel_pack_digest: &str,
        direct_template_catalog_digest: &str,
    ) -> RusticolResult<()> {
        if self.kind != RECURRENCE_RUNTIME_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence plan kind {:?}",
                self.kind
            )));
        }
        validate_direct_contract(
            &self.builder_input_abi,
            &self.recurrence_plan_abi,
            &self.runtime_layout_abi,
            &self.direct_template_abi,
            &self.direct_backend_abi,
            "nested recurrence plan",
        )?;
        parse_sha256(&self.builder_input_sha256, "recurrence builder input")?;
        parse_sha256(
            &self.prepared_kernel_pack_digest,
            "nested recurrence prepared-kernel pack",
        )?;
        parse_sha256(
            &self.direct_template_catalog_digest,
            "nested recurrence direct-template catalog",
        )?;
        if self.prepared_kernel_pack_digest != prepared_kernel_pack_digest {
            return Err(RusticolError::integrity(
                "recurrence prepared-kernel pack digest differs between execution and plan metadata",
            ));
        }
        if self.direct_template_catalog_digest != direct_template_catalog_digest {
            return Err(RusticolError::integrity(
                "recurrence direct-template catalog digest differs between execution and plan metadata",
            ));
        }
        validate_capabilities(
            &self.required_runtime_capabilities,
            "nested recurrence plan",
        )?;
        self.runtime_container.validate()?;
        self.inspection_summary.validate(process_key)?;
        let member = &self.inspection_summary.runtime_container_member;
        if member.path != self.runtime_container.plan_member_path
            || member.container_size_bytes != self.runtime_container.size_bytes
            || member.size_bytes != self.runtime_container.unpacked_size_bytes
        {
            return Err(RusticolError::integrity(
                "Direct-Arena v2 plan member metadata disagrees with its recurrence runtime container",
            ));
        }
        Ok(())
    }
}

impl RecurrenceRuntimeContainer {
    fn validate(&self) -> RusticolResult<()> {
        if self.kind != RECURRENCE_RUNTIME_CONTAINER_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence runtime container kind {:?}",
                self.kind
            )));
        }
        if self.schema_version != RECURRENCE_RUNTIME_CONTAINER_SCHEMA {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence runtime container schema {}",
                self.schema_version
            )));
        }
        if self.storage_abi != RECURRENCE_RUNTIME_STORAGE_ABI {
            return Err(unsupported_abi(
                "recurrence runtime storage",
                &self.storage_abi,
            ));
        }
        if self.path != RECURRENCE_RUNTIME_CONTAINER_PATH {
            return Err(RusticolError::security(
                "recurrence runtime container path must be exactly recurrence-runtime.pacbin",
            ));
        }
        if self.plan_member_path != RECURRENCE_DIRECT_PLAN_MEMBER {
            return Err(RusticolError::security(format!(
                "Direct-Arena v2 plan member path must be exactly {RECURRENCE_DIRECT_PLAN_MEMBER:?}"
            )));
        }
        if self.member_count != 1 {
            return Err(RusticolError::integrity(
                "Direct-Arena v2 recurrence runtime container must declare exactly one member",
            ));
        }
        if self.size_bytes == 0 || self.size_bytes > MAX_RUNTIME_CONTAINER_BYTES {
            return Err(RusticolError::artifact(format!(
                "recurrence runtime container size must be in 1..={MAX_RUNTIME_CONTAINER_BYTES} bytes"
            )));
        }
        if self.unpacked_size_bytes == 0
            || self.unpacked_size_bytes > self.size_bytes
            || self.unpacked_size_bytes > MAX_RUNTIME_CONTAINER_BYTES
        {
            return Err(RusticolError::artifact(
                "recurrence runtime unpacked size is outside canonical bounds",
            ));
        }
        parse_sha256(&self.sha256, "recurrence runtime container")?;
        parse_sha256(&self.index_sha256, "recurrence runtime index")?;
        Ok(())
    }
}

impl RecurrenceInspectionSummary {
    fn validate(&self, process_key: &str) -> RusticolResult<()> {
        if self.execution_mode != "recurrence"
            || self.recurrence_plan_abi != RECURRENCE_PLAN_ABI
            || self.runtime_layout_abi != RECURRENCE_RUNTIME_LAYOUT_ABI
            || self.direct_template_abi != RECURRENCE_DIRECT_TEMPLATE_ABI
        {
            return Err(RusticolError::compatibility(
                "recurrence inspection summary does not match Direct-Arena v2",
            ));
        }
        if self.process_id != process_key {
            return Err(RusticolError::integrity(
                "recurrence inspection process id does not match its execution manifest",
            ));
        }
        validate_counts(
            "recurrence inspection summary",
            &[
                self.parameter_count,
                self.sector_count,
                self.prepared_kernel_count,
                self.direct_executor_count,
            ],
        )?;
        parse_sha256(&self.semantic_digest, "recurrence semantic plan")?;
        parse_sha256(
            &self.runtime_layout_digest,
            "recurrence Direct-Arena runtime layout",
        )?;
        self.schedule.validate()?;
        self.direct_arena.validate()?;
        self.runtime_container_member.validate()?;
        self.generation_timings_seconds.validate()?;
        if self.schedule.source_row_count > self.schedule.current_count
            || self.schedule.finalization_count > self.schedule.current_count
        {
            return Err(RusticolError::integrity(
                "recurrence inspection schedule disagrees with its top-level counts",
            ));
        }
        Ok(())
    }
}

impl RecurrenceScheduleSummary {
    fn validate(self) -> RusticolResult<()> {
        validate_counts(
            "recurrence schedule summary",
            &[
                self.current_count,
                self.source_row_count,
                self.contribution_count,
                self.finalization_count,
                self.amplitude_destination_count,
                self.closure_term_count,
                self.replay_target_count,
                self.retained_helicity_count,
                self.resolved_helicity_count,
                self.exact_factor_count,
            ],
        )?;
        if self.resolved_helicity_count > self.retained_helicity_count {
            return Err(RusticolError::integrity(
                "recurrence resolved helicities exceed retained helicities",
            ));
        }
        Ok(())
    }
}

impl RecurrenceDirectArenaSummary {
    fn validate(self) -> RusticolResult<()> {
        validate_counts(
            "recurrence Direct-Arena summary",
            &[
                self.semantic_component_count,
                self.current_arena_components,
                self.arena_component_reuse_count,
                self.momentum_form_count,
                self.selector_domain_count,
                self.row_group_count,
                self.packed_input_bytes,
                self.packed_output_bytes,
                self.scatter_bytes,
            ],
        )?;
        let expected_semantic = self
            .current_arena_components
            .checked_add(self.arena_component_reuse_count)
            .ok_or_else(|| {
                RusticolError::artifact("recurrence Direct-Arena component count overflows u64")
            })?;
        if self.semantic_component_count != expected_semantic {
            return Err(RusticolError::integrity(
                "recurrence Direct-Arena component reuse count is inconsistent",
            ));
        }
        if self.packed_input_bytes != 0 || self.packed_output_bytes != 0 || self.scatter_bytes != 0
        {
            return Err(RusticolError::compatibility(
                "Direct-Arena v2 must not declare packed input, packed output, or scatter bytes",
            ));
        }
        Ok(())
    }
}

impl RecurrenceRuntimeContainerMember {
    fn validate(&self) -> RusticolResult<()> {
        if self.path != RECURRENCE_DIRECT_PLAN_MEMBER {
            return Err(RusticolError::security(format!(
                "Direct-Arena v2 inspection member path must be exactly {RECURRENCE_DIRECT_PLAN_MEMBER:?}"
            )));
        }
        if self.size_bytes == 0 || self.size_bytes > MAX_RUNTIME_CONTAINER_BYTES {
            return Err(RusticolError::artifact(
                "Direct-Arena v2 plan member size is outside canonical bounds",
            ));
        }
        if self.container_size_bytes == 0
            || self.container_size_bytes > MAX_RUNTIME_CONTAINER_BYTES
            || self.size_bytes > self.container_size_bytes
        {
            return Err(RusticolError::artifact(
                "Direct-Arena v2 plan member/container sizes are inconsistent",
            ));
        }
        parse_sha256(&self.sha256, "recurrence Direct-Arena plan member")?;
        Ok(())
    }
}

impl RecurrenceGenerationTimings {
    fn validate(self) -> RusticolResult<()> {
        let values = [
            self.python_extraction,
            self.catalog_authentication,
            self.semantic_construction,
            self.direct_lowering,
            self.serialization,
            self.native_total,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(RusticolError::artifact(
                "recurrence Direct-Arena generation timings must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

impl RecurrenceSummary {
    fn validate(&self) -> RusticolResult<()> {
        if self.builder_input_abi != RECURRENCE_BUILDER_INPUT_ABI {
            return Err(unsupported_abi(
                "recurrence builder input",
                &self.builder_input_abi,
            ));
        }
        if self.template_input_abi != RECURRENCE_TEMPLATE_INPUT_ABI {
            return Err(unsupported_abi(
                "recurrence template input",
                &self.template_input_abi,
            ));
        }
        validate_counts(
            "recurrence summary",
            &[
                self.current_count,
                self.contribution_count,
                self.closure_term_count,
            ],
        )
    }
}

impl RecurrenceRuntimeMetadata {
    fn validate(
        &self,
        external_pdgs: &[i32],
        color_accuracy: &str,
        inspection: &RecurrenceInspectionSummary,
    ) -> RusticolResult<()> {
        for (context, len) in [
            ("public color flows", self.public_color_flows.len()),
            ("runtime parameters", self.runtime_parameters.len()),
            (
                "prepared parameter defaults",
                self.prepared_parameter_defaults.len(),
            ),
            ("parameter projection", self.parameter_projection.len()),
            ("source templates", self.source_templates.len()),
            ("external legs", self.external_legs.len()),
            ("particle masses", self.particle_masses.len()),
        ] {
            bounded_len(context, len)?;
        }
        if self.public_color_flows.is_empty() {
            return Err(RusticolError::artifact(
                "recurrence runtime metadata has no public color flows",
            ));
        }
        if self.public_color_flows.len() as u64 != inspection.sector_count {
            return Err(RusticolError::integrity(
                "recurrence public color-flow count does not match the Direct-Arena sector count",
            ));
        }
        let mut public_ids = BTreeSet::new();
        for (target_sector_id, flow) in self.public_color_flows.iter().enumerate() {
            validate_text(&flow.public_id, "recurrence public color-flow ID")?;
            if !public_ids.insert(flow.public_id.as_str()) {
                return Err(RusticolError::artifact(
                    "recurrence public color-flow IDs are not unique",
                ));
            }
            if u64::from(flow.construction_sector_id) >= inspection.sector_count {
                return Err(RusticolError::integrity(
                    "recurrence public color flow references an absent construction sector",
                ));
            }
            if flow.target_sector_id as usize != target_sector_id
                || u64::from(flow.target_sector_id) >= inspection.sector_count
            {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow target sectors are not dense or in range",
                ));
            }
        }
        if self.prepared_parameter_defaults.len() as u64 != inspection.parameter_count {
            return Err(RusticolError::integrity(
                "recurrence prepared-parameter defaults do not match the plan parameter count",
            ));
        }
        if self
            .prepared_parameter_defaults
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            return Err(RusticolError::artifact(
                "recurrence prepared-parameter defaults must be finite complex-f64 values",
            ));
        }
        self.validate_parameter_projection()?;
        self.validate_runtime_parameters(&self.runtime_parameters)?;

        if self.source_templates.is_empty() {
            return Err(RusticolError::artifact(
                "recurrence runtime metadata has no source templates",
            ));
        }
        let mut previous_source_id = None;
        let mut source_pdgs = BTreeSet::new();
        for (index, source) in self.source_templates.iter().enumerate() {
            if previous_source_id.is_some_and(|previous| previous >= source.source_template_id) {
                return Err(RusticolError::artifact(
                    "recurrence source templates are not in strict source-template ID order",
                ));
            }
            previous_source_id = Some(source.source_template_id);
            source.validate(index)?;
            source_pdgs.insert(source.source_ir.identity.pdg_label);
        }

        if self.external_legs.len() != external_pdgs.len() {
            return Err(RusticolError::integrity(
                "recurrence runtime external legs do not match the execution PDG order",
            ));
        }
        let mut public_labels = BTreeSet::new();
        let mut saw_final = false;
        for (source_slot, (leg, expected_pdg)) in
            self.external_legs.iter().zip(external_pdgs).enumerate()
        {
            if leg.source_slot as usize != source_slot
                || leg.public_label == 0
                || !public_labels.insert(leg.public_label)
            {
                return Err(RusticolError::artifact(
                    "recurrence external source slots or public labels are not canonical",
                ));
            }
            if leg.physical_pdg != *expected_pdg {
                return Err(RusticolError::integrity(
                    "recurrence external physical PDGs disagree with the execution manifest",
                ));
            }
            if saw_final && leg.is_initial {
                return Err(RusticolError::artifact(
                    "recurrence initial-state legs must precede final-state legs",
                ));
            }
            saw_final |= !leg.is_initial;
            if !source_pdgs.contains(&leg.outgoing_pdg) {
                return Err(RusticolError::integrity(format!(
                    "recurrence external outgoing PDG {} has no source template",
                    leg.outgoing_pdg
                )));
            }
        }

        let mut mass_pdgs = BTreeSet::new();
        let mut previous_mass_pdg = None;
        for row in &self.particle_masses {
            if previous_mass_pdg.is_some_and(|previous| previous >= row.outgoing_pdg)
                || !mass_pdgs.insert(row.outgoing_pdg)
            {
                return Err(RusticolError::artifact(
                    "recurrence particle masses are not in strict outgoing-PDG order",
                ));
            }
            previous_mass_pdg = Some(row.outgoing_pdg);
            if !row.mass.is_finite() || row.mass < 0.0 {
                return Err(RusticolError::artifact(
                    "recurrence particle masses must be finite and nonnegative",
                ));
            }
        }
        if mass_pdgs != source_pdgs {
            return Err(RusticolError::integrity(
                "recurrence particle masses do not cover exactly the source-template particles",
            ));
        }
        self.normalization.validate(color_accuracy)
    }

    fn validate_runtime_parameters(
        &self,
        parameters: &[RecurrenceRuntimeParameter],
    ) -> RusticolResult<()> {
        if parameters.len() != self.parameter_projection.len() {
            return Err(RusticolError::integrity(
                "recurrence runtime parameters do not cover the parameter projection",
            ));
        }
        let mut previous_runtime_name: Option<&str> = None;
        let mut previous_kind: Option<&str> = None;
        for (parameter_index, (parameter, projection)) in parameters
            .iter()
            .zip(&self.parameter_projection)
            .enumerate()
        {
            if parameter.parameter_index as usize != parameter_index
                || parameter.parameter_index != projection.runtime_slot
            {
                return Err(RusticolError::artifact(
                    "recurrence runtime parameter indices must be dense and match projection slots",
                ));
            }
            validate_text(&parameter.name, "recurrence runtime parameter name")?;
            validate_text(&parameter.kind, "recurrence runtime parameter kind")?;
            if !parameter.default.is_finite() {
                return Err(RusticolError::artifact(
                    "recurrence runtime parameter defaults must be finite",
                ));
            }

            match (&parameter.runtime_name, &parameter.complex_component) {
                (Some(runtime_name), Some(component)) => {
                    validate_text(runtime_name, "recurrence runtime parameter public name")?;
                    let expected_component = match projection.component {
                        0 => "real",
                        1 => "imag",
                        _ => {
                            return Err(RusticolError::integrity(
                                "recurrence runtime parameter has an invalid projected component",
                            ));
                        }
                    };
                    if component != expected_component
                        || projection.runtime_name != *runtime_name
                        || parameter.name != format!("{runtime_name}.{component}")
                    {
                        return Err(RusticolError::integrity(
                            "recurrence complex runtime parameter disagrees with its projection",
                        ));
                    }
                    if previous_runtime_name == Some(runtime_name.as_str())
                        && previous_kind != Some(parameter.kind.as_str())
                    {
                        return Err(RusticolError::integrity(
                            "recurrence complex runtime parameter components have different kinds",
                        ));
                    }
                    previous_runtime_name = Some(runtime_name);
                    previous_kind = Some(&parameter.kind);
                }
                (None, None) => {
                    if projection.component != 0 || parameter.name != projection.runtime_name {
                        return Err(RusticolError::integrity(
                            "recurrence real runtime parameter disagrees with its projection",
                        ));
                    }
                    previous_runtime_name = None;
                    previous_kind = None;
                }
                _ => {
                    return Err(RusticolError::artifact(
                        "recurrence runtime_name and complex_component must be both present or both absent",
                    ));
                }
            }

            if let Some(prepared_id) = projection.prepared_parameter_id {
                let expected_default = self.prepared_parameter_defaults[prepared_id as usize]
                    [projection.component as usize];
                if parameter.default != expected_default {
                    return Err(RusticolError::integrity(
                        "recurrence runtime parameter default disagrees with its prepared default",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_parameter_projection(&self) -> RusticolResult<()> {
        let parameter_count = self.prepared_parameter_defaults.len();
        let mut previous_key: Option<(&str, u32)> = None;
        let mut current_name: Option<&str> = None;
        let mut current_template = None;
        let mut current_prepared = None;
        for (runtime_slot, row) in self.parameter_projection.iter().enumerate() {
            if row.runtime_slot as usize != runtime_slot {
                return Err(RusticolError::artifact(
                    "recurrence runtime parameter slots must be dense and ordered from zero",
                ));
            }
            validate_text(&row.runtime_name, "recurrence runtime parameter name")?;
            if row.component > 1 {
                return Err(RusticolError::artifact(
                    "recurrence runtime parameter component must be zero or one",
                ));
            }
            if row.parameter_template_id as usize >= parameter_count
                || row
                    .prepared_parameter_id
                    .is_some_and(|id| id as usize >= parameter_count)
            {
                return Err(RusticolError::integrity(
                    "recurrence parameter projection references an absent parameter",
                ));
            }
            let key = (row.runtime_name.as_str(), row.component);
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(RusticolError::artifact(
                    "recurrence parameter projection is not in strict name/component order",
                ));
            }
            previous_key = Some(key);
            if current_name == Some(row.runtime_name.as_str()) {
                if row.component != 1
                    || current_template != Some(row.parameter_template_id)
                    || current_prepared != Some(row.prepared_parameter_id)
                {
                    return Err(RusticolError::integrity(
                        "recurrence complex parameter projection rows are inconsistent",
                    ));
                }
            } else {
                if row.component != 0 {
                    return Err(RusticolError::artifact(
                        "recurrence parameter projection must begin each parameter at component zero",
                    ));
                }
                current_name = Some(row.runtime_name.as_str());
                current_template = Some(row.parameter_template_id);
                current_prepared = Some(row.prepared_parameter_id);
            }
        }
        Ok(())
    }
}

impl RecurrenceSourceTemplate {
    fn validate(&self, index: usize) -> RusticolResult<()> {
        if self.dimension == 0 || self.dimension > MAX_SOURCE_COMPONENTS {
            return Err(RusticolError::artifact(format!(
                "recurrence source template {index} has an unsupported dimension"
            )));
        }
        self.source_ir.validate(index)?;
        self.crossing.validate(index, "source-template")?;
        if self.dimension != self.source_ir.component_dimension
            || self.crossing != self.source_ir.crossing
        {
            return Err(RusticolError::integrity(format!(
                "recurrence source template {index} disagrees with its typed SourceIR"
            )));
        }
        let declared = self.source_ir.states.iter().any(|state| {
            state.helicity == self.helicity
                && state.chirality == self.chirality
                && state.spin_state == RecurrenceSourceSpinState::Scalar(self.spin_state)
        });
        if !declared {
            return Err(RusticolError::artifact(format!(
                "recurrence source template {index} state is not declared by its typed SourceIR"
            )));
        }
        Ok(())
    }
}

impl RecurrenceGenericSourceIr {
    fn validate(&self, index: usize) -> RusticolResult<()> {
        let identity = &self.identity;
        for (label, value) in [
            ("canonical id", identity.canonical_id.as_str()),
            ("species id", identity.species_id.as_str()),
            ("anti-canonical id", identity.anti_canonical_id.as_str()),
            ("display name", identity.display_name.as_str()),
            ("anti-display name", identity.anti_display_name.as_str()),
        ] {
            validate_text(value, &format!("recurrence source {index} {label}"))?;
        }
        if identity.anti_pdg_label == 0 {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has an invalid antiparticle relation"
            )));
        }
        let canonical_self_conjugate = identity.canonical_id == identity.anti_canonical_id;
        let pdg_self_conjugate = identity.pdg_label == identity.anti_pdg_label;
        if identity.self_conjugate != canonical_self_conjugate
            || identity.self_conjugate != pdg_self_conjugate
            || (identity.orientation == RecurrenceSourceOrientation::SelfConjugate)
                != identity.self_conjugate
        {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} orientation is inconsistent with its antiparticle relation"
            )));
        }
        validate_text(&self.basis, &format!("recurrence source {index} basis"))?;
        for (label, value) in [
            ("mass parameter", self.mass_parameter.as_deref()),
            ("width parameter", self.width_parameter.as_deref()),
        ] {
            if let Some(value) = value {
                validate_text(value, &format!("recurrence source {index} {label}"))?;
            }
        }
        if self.component_dimension == 0 || self.component_dimension > MAX_SOURCE_COMPONENTS {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has an unsupported component dimension"
            )));
        }
        bounded_len("recurrence SourceIR states", self.states.len())?;
        if self.states.is_empty() {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has no declared SourceIR states"
            )));
        }
        self.crossing.validate(index, "declared")?;

        let expected_statistics = match self.wavefunction_family {
            RecurrenceWavefunctionFamily::Fermion => RecurrenceParticleStatistics::Fermion,
            RecurrenceWavefunctionFamily::Ghost => RecurrenceParticleStatistics::Ghost,
            RecurrenceWavefunctionFamily::Auxiliary => RecurrenceParticleStatistics::Auxiliary,
            RecurrenceWavefunctionFamily::Scalar
            | RecurrenceWavefunctionFamily::Vector
            | RecurrenceWavefunctionFamily::Spin2 => RecurrenceParticleStatistics::Boson,
        };
        if self.statistics != expected_statistics {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} statistics disagree with its wavefunction family"
            )));
        }
        match (self.wavefunction_family, self.component_dimension) {
            (RecurrenceWavefunctionFamily::Fermion, 2 | 4) => {
                if identity.self_conjugate {
                    return Err(RusticolError::artifact(format!(
                        "recurrence source {index} is an unsupported self-conjugate fermion source"
                    )));
                }
            }
            (RecurrenceWavefunctionFamily::Scalar, 1)
            | (RecurrenceWavefunctionFamily::Vector, 4)
            | (RecurrenceWavefunctionFamily::Spin2, 16) => {}
            _ => {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} has an unsupported wavefunction family/dimension pair"
                )));
            }
        }

        let max_helicity = if self.wavefunction_family == RecurrenceWavefunctionFamily::Spin2 {
            2
        } else {
            1
        };
        let mut states = BTreeSet::new();
        for (state_index, state) in self.states.iter().enumerate() {
            if state.chirality.unsigned_abs() > 1 || state.helicity.unsigned_abs() > max_helicity {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} SourceIR state {state_index} is outside the supported helicity/chirality range"
                )));
            }
            if let RecurrenceSourceSpinState::Components(components) = &state.spin_state {
                bounded_len("structured source spin state", components.len())?;
                if components.is_empty() || components.len() as u64 > self.component_dimension {
                    return Err(RusticolError::artifact(format!(
                        "recurrence source {index} SourceIR state {state_index} has invalid structured spin metadata"
                    )));
                }
            }
            if !states.insert(state) {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} contains duplicate SourceIR states"
                )));
            }
        }
        Ok(())
    }
}

impl RecurrenceGenericCrossingIr {
    fn validate(&self, index: usize, label: &str) -> RusticolResult<()> {
        if ![-1, 1].contains(&self.helicity_factor)
            || ![-1, 1].contains(&self.chirality_factor)
            || ![-1, 1].contains(&self.spin_state_factor)
            || !self.phase.iter().all(|component| component.is_finite())
            || self.phase == [0.0, 0.0]
        {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has invalid {label} CrossingIR metadata"
            )));
        }
        Ok(())
    }
}

impl RecurrenceNormalization {
    fn validate(&self, color_accuracy: &str) -> RusticolResult<()> {
        if self.color_accuracy != color_accuracy {
            return Err(RusticolError::integrity(
                "recurrence normalization color accuracy disagrees with the process",
            ));
        }
        if [
            self.color_factor,
            self.average_factor,
            self.identical_factor,
            self.global_coupling_factor,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(RusticolError::artifact(
                "recurrence normalization factors must be finite",
            ));
        }
        if self.average_factor <= 0.0 || self.identical_factor <= 0.0 {
            return Err(RusticolError::artifact(
                "recurrence averaging and identical-particle factors must be positive",
            ));
        }
        if self.qcd_coupling_power.is_some() != self.electroweak_coupling_power.is_some() {
            return Err(RusticolError::artifact(
                "recurrence normalization coupling powers must be both present or both absent",
            ));
        }
        validate_counts(
            "recurrence normalization",
            &[
                self.qcd_coupling_power.unwrap_or(0),
                self.electroweak_coupling_power.unwrap_or(0),
            ],
        )?;
        validate_text(
            &self.coupling_policy,
            "recurrence normalization coupling policy",
        )?;
        Ok(())
    }
}

/// Parse and validate one bounded recurrence `execution.json`.
pub(super) fn parse_recurrence_execution_manifest(
    bytes: &[u8],
    path: &str,
    outer: &ArtifactProcess,
) -> RusticolResult<RecurrenceExecutionManifest> {
    if bytes.len() >= MAX_EXECUTION_MANIFEST_BYTES {
        return Err(RusticolError::artifact(format!(
            "recurrence execution manifest {path:?} must be smaller than {MAX_EXECUTION_MANIFEST_BYTES} bytes"
        )));
    }
    let manifest: RecurrenceExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse bounded recurrence execution manifest {path:?}: {error}"
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

fn validate_capabilities(capabilities: &[String], context: &str) -> RusticolResult<()> {
    if capabilities.len() != 2
        || capabilities[0] != RECURRENCE_LC_COLOR_CAPABILITY
        || capabilities[1] != RECURRENCE_RUNTIME_CAPABILITY
    {
        return Err(RusticolError::compatibility(format!(
            "{context} must require exactly [{RECURRENCE_LC_COLOR_CAPABILITY:?}, {RECURRENCE_RUNTIME_CAPABILITY:?}]"
        )));
    }
    Ok(())
}

fn validate_direct_contract(
    builder_input_abi: &str,
    recurrence_plan_abi: &str,
    runtime_layout_abi: &str,
    direct_template_abi: &str,
    direct_backend_abi: &str,
    context: &str,
) -> RusticolResult<()> {
    for (label, actual, expected) in [
        (
            "builder input",
            builder_input_abi,
            RECURRENCE_BUILDER_INPUT_ABI,
        ),
        ("plan", recurrence_plan_abi, RECURRENCE_PLAN_ABI),
        (
            "runtime layout",
            runtime_layout_abi,
            RECURRENCE_RUNTIME_LAYOUT_ABI,
        ),
        (
            "direct template",
            direct_template_abi,
            RECURRENCE_DIRECT_TEMPLATE_ABI,
        ),
        (
            "direct backend",
            direct_backend_abi,
            RECURRENCE_DIRECT_BACKEND_ABI,
        ),
    ] {
        if actual != expected {
            return Err(RusticolError::compatibility(format!(
                "{context} has unsupported Direct-Arena v2 {label} ABI {actual:?}; regenerate the recurrence artifact"
            )));
        }
    }
    Ok(())
}

fn bounded_len(context: &str, len: usize) -> RusticolResult<()> {
    if len > MAX_METADATA_ROWS {
        return Err(RusticolError::artifact(format!(
            "{context} count exceeds the supported bound {MAX_METADATA_ROWS}"
        )));
    }
    Ok(())
}

fn validate_counts(context: &str, counts: &[u64]) -> RusticolResult<()> {
    if counts.iter().any(|count| *count > MAX_SUMMARY_COUNT) {
        return Err(RusticolError::artifact(format!(
            "{context} count exceeds the supported bound {MAX_SUMMARY_COUNT}"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, context: &str) -> RusticolResult<()> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES {
        return Err(RusticolError::artifact(format!(
            "{context} is empty or exceeds {MAX_TEXT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn unsupported_abi(context: &str, value: &str) -> RusticolError {
    RusticolError::compatibility(format!("unsupported {context} ABI {value:?}"))
}

fn parse_sha256(value: &str, context: &str) -> RusticolResult<[u8; 32]> {
    if value.len() != 64
        || value
            .as_bytes()
            .iter()
            .any(|byte| !matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(RusticolError::artifact(format!(
            "{context} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(digest)
}

fn hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        _ => unreachable!("digest validation precedes hexadecimal decoding"),
    }
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

fn deserialize_required_nullable_u32<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<u32>::deserialize(deserializer)
}

fn deserialize_recurrence_phase<'de, D>(deserializer: D) -> Result<[f64; 2], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if let Ok(pair) = serde_json::from_value::<[f64; 2]>(value.clone()) {
        return Ok(pair);
    }
    let object = value.as_object().ok_or_else(|| {
        serde::de::Error::custom("recurrence crossing phase must be a complex-f64 pair")
    })?;
    let component = |prefix: &str| -> Result<f64, D::Error> {
        let numerator = object
            .get(&format!("{prefix}_numerator"))
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("invalid exact recurrence phase"))?
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom("invalid exact recurrence phase numerator"))?;
        let denominator = object
            .get(&format!("{prefix}_denominator"))
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("invalid exact recurrence phase"))?
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom("invalid exact recurrence phase denominator"))?;
        let value = numerator / denominator;
        if denominator == 0.0 || !value.is_finite() {
            return Err(serde::de::Error::custom(
                "exact recurrence phase is not finite",
            ));
        }
        Ok(value)
    };
    Ok([component("real")?, component("imag")?])
}

fn deserialize_optional_u64_if_present<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    u64::deserialize(deserializer).map(Some)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn capabilities() -> Value {
        json!([
            RECURRENCE_LC_COLOR_CAPABILITY,
            RECURRENCE_RUNTIME_CAPABILITY
        ])
    }

    fn crossing() -> Value {
        json!({
            "momentum_transform": "identity",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0]
        })
    }

    fn source_ir() -> Value {
        json!({
            "identity": {
                "canonical_id": "model:test:state:x",
                "species_id": "model:test:species:x",
                "anti_canonical_id": "model:test:state:x",
                "display_name": "x",
                "anti_display_name": "x",
                "pdg_label": 25,
                "anti_pdg_label": 25,
                "orientation": "self-conjugate",
                "self_conjugate": true
            },
            "statistics": "boson",
            "wavefunction_family": "scalar",
            "component_dimension": 1,
            "states": [{"helicity": 0, "chirality": 0, "spin_state": 0}],
            "crossing": crossing(),
            "basis": "scalar",
            "mass_parameter": null,
            "width_parameter": null
        })
    }

    fn runtime_metadata() -> Value {
        json!({
            "public_color_flows": [{
                "public_id": "flow:1",
                "construction_sector_id": 0,
                "target_sector_id": 0
            }],
            "runtime_parameters": [{
                "name": "mass.x",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 1.0,
                "runtime_name": null,
                "complex_component": null
            }],
            "prepared_parameter_defaults": [[1.0, 0.0]],
            "parameter_projection": [{
                "runtime_slot": 0,
                "runtime_name": "mass.x",
                "parameter_template_id": 0,
                "prepared_parameter_id": 0,
                "component": 0
            }],
            "source_templates": [{
                "source_template_id": 0,
                "current_state_template_id": 0,
                "dimension": 1,
                "helicity": 0,
                "chirality": 0,
                "spin_state": 0,
                "source_ir": source_ir(),
                "crossing": crossing()
            }],
            "external_legs": [{
                "source_slot": 0,
                "public_label": 1,
                "physical_pdg": 25,
                "outgoing_pdg": 25,
                "is_initial": false
            }],
            "particle_masses": [{"outgoing_pdg": 25, "mass": 0.0}],
            "normalization": {
                "color_accuracy": "lc",
                "color_factor": 1.0,
                "average_factor": 1.0,
                "identical_factor": 1.0,
                "global_coupling_factor": 1.0,
                "qcd_coupling_power": 0,
                "electroweak_coupling_power": 0,
                "couplings_in_stage_evaluators": true,
                "coupling_policy": "test"
            }
        })
    }

    fn inspection_summary() -> Value {
        json!({
            "execution_mode": "recurrence",
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "process_id": "x_to_x",
            "lc_flow_layout": "topology-replay",
            "parameter_count": 1,
            "sector_count": 1,
            "prepared_kernel_count": 1,
            "direct_executor_count": 4,
            "semantic_digest": "66".repeat(32),
            "runtime_layout_digest": "77".repeat(32),
            "schedule": {
                "current_count": 1,
                "source_row_count": 1,
                "contribution_count": 0,
                "finalization_count": 1,
                "amplitude_destination_count": 1,
                "closure_term_count": 0,
                "replay_target_count": 1,
                "retained_helicity_count": 1,
                "resolved_helicity_count": 1,
                "exact_factor_count": 1
            },
            "direct_arena": {
                "semantic_component_count": 1,
                "current_arena_components": 1,
                "arena_component_reuse_count": 0,
                "momentum_form_count": 1,
                "selector_domain_count": 1,
                "row_group_count": 4,
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0
            },
            "runtime_container_member": {
                "path": RECURRENCE_DIRECT_PLAN_MEMBER,
                "size_bytes": 512,
                "sha256": "88".repeat(32),
                "container_size_bytes": 4096
            },
            "generation_timings_seconds": {
                "python_extraction": 0.1,
                "catalog_authentication": 0.1,
                "semantic_construction": 0.1,
                "direct_lowering": 0.1,
                "serialization": 0.1,
                "native_total": 0.5
            }
        })
    }

    pub(crate) fn manifest() -> Value {
        let plan = json!({
            "kind": RECURRENCE_RUNTIME_KIND,
            "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
            "builder_input_sha256": "11".repeat(32),
            "prepared_kernel_pack_digest": "44".repeat(32),
            "direct_template_catalog_digest": "55".repeat(32),
            "required_runtime_capabilities": capabilities(),
            "runtime_container": {
                "kind": RECURRENCE_RUNTIME_CONTAINER_KIND,
                "schema_version": RECURRENCE_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": RECURRENCE_RUNTIME_STORAGE_ABI,
                "path": RECURRENCE_RUNTIME_CONTAINER_PATH,
                "plan_member_path": RECURRENCE_DIRECT_PLAN_MEMBER,
                "size_bytes": 4096,
                "sha256": "22".repeat(32),
                "member_count": 1,
                "unpacked_size_bytes": 512,
                "index_sha256": "33".repeat(32)
            },
            "inspection_summary": inspection_summary()
        });
        json!({
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": RECURRENCE_RUNTIME_KIND,
            "required_runtime_capabilities": capabilities(),
            "process": "x > x",
            "key": "x_to_x",
            "color_accuracy": "lc",
            "external_pdg_order": [25],
            "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
            "prepared_kernel_pack_digest": "44".repeat(32),
            "direct_template_catalog_digest": "55".repeat(32),
            "kernel_pack": {
                "manifest_path": RECURRENCE_KERNEL_PACK_MANIFEST_PATH,
                "payload_root": RECURRENCE_KERNEL_PAYLOAD_ROOT
            },
            "runtime_options": {"point_tile_size": 8, "workspace_mib": 64},
            "runtime_metadata": runtime_metadata(),
            "plan": plan,
            "recurrence_summary": {
                "lc_flow_layout": "topology-replay",
                "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
                "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
                "current_count": 1,
                "contribution_count": 0,
                "closure_term_count": 0
            }
        })
    }

    pub(crate) fn outer() -> ArtifactProcess {
        ArtifactProcess {
            id: "x_to_x".to_owned(),
            expression: "x > x".to_owned(),
            color_accuracy: "lc".to_owned(),
            external_pdgs: vec![25],
            physics_path: "processes/x_to_x/physics.json".to_owned(),
            required_runtime_capabilities: vec![
                RECURRENCE_LC_COLOR_CAPABILITY.to_owned(),
                RECURRENCE_RUNTIME_CAPABILITY.to_owned(),
            ],
            aliases: Vec::new(),
        }
    }

    fn parse(value: &Value) -> RusticolResult<RecurrenceExecutionManifest> {
        parse_recurrence_execution_manifest(
            &serde_json::to_vec(value).unwrap(),
            "processes/x_to_x/execution.json",
            &outer(),
        )
    }

    #[test]
    fn accepts_strict_typed_recurrence_manifest() {
        let parsed = parse(&manifest()).unwrap();
        assert_eq!(parsed.runtime_metadata.public_color_flows.len(), 1);
        assert_eq!(parsed.runtime_metadata.source_templates.len(), 1);
        assert_eq!(parsed.plan.runtime_container.member_count, 1);
        assert_eq!(
            parsed.plan.inspection_summary.direct_arena.row_group_count,
            4
        );
    }

    #[test]
    fn rejects_unknown_runtime_metadata_fields() {
        let mut value = manifest();
        value["runtime_metadata"]["opaque"] = json!({});
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("unknown field `opaque`"));
    }

    #[test]
    fn rejects_public_color_flow_outside_construction_sectors() {
        let mut value = manifest();
        value["runtime_metadata"]["public_color_flows"][0]["construction_sector_id"] = json!(1);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("absent construction sector"));
    }

    #[test]
    fn rejects_noncanonical_public_color_flow_target_sector() {
        let mut value = manifest();
        value["runtime_metadata"]["public_color_flows"][0]["target_sector_id"] = json!(1);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("not dense or in range"));
    }

    #[test]
    fn rejects_source_ir_and_crossing_disagreement() {
        let mut value = manifest();
        value["runtime_metadata"]["source_templates"][0]["crossing"]["phase"] = json!([0.0, 1.0]);
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("disagrees with its typed SourceIR")
        );
    }

    #[test]
    fn rejects_noncanonical_container_and_capability_metadata() {
        let mut bad_path = manifest();
        bad_path["plan"]["runtime_container"]["path"] = json!("../recurrence-runtime.pacbin");
        assert!(
            parse(&bad_path)
                .unwrap_err()
                .to_string()
                .contains("exactly")
        );

        let mut bad_capabilities = manifest();
        bad_capabilities["required_runtime_capabilities"] = json!([
            RECURRENCE_RUNTIME_CAPABILITY,
            RECURRENCE_LC_COLOR_CAPABILITY
        ]);
        assert!(
            parse(&bad_capabilities)
                .unwrap_err()
                .to_string()
                .contains("must require exactly")
        );

        let mut bad_member_path = manifest();
        bad_member_path["plan"]["runtime_container"]["plan_member_path"] =
            json!("plan/recurrence-plan-v1.bin");
        assert!(
            parse(&bad_member_path)
                .unwrap_err()
                .to_string()
                .contains("Direct-Arena v2 plan member path")
        );
    }

    #[test]
    fn rejects_pre_direct_arena_abis() {
        let mut value = manifest();
        value["recurrence_plan_abi"] = json!("pyamplicol-recurrence-plan-v1");
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("Direct-Arena v2 plan ABI"));
        assert!(error.to_string().contains("regenerate"));
    }

    #[test]
    fn rejects_cross_record_direct_digest_mismatches() {
        let mut bad_pack = manifest();
        bad_pack["plan"]["prepared_kernel_pack_digest"] = json!("99".repeat(32));
        assert!(
            parse(&bad_pack)
                .unwrap_err()
                .to_string()
                .contains("prepared-kernel pack digest differs")
        );

        let mut bad_catalog = manifest();
        bad_catalog["plan"]["direct_template_catalog_digest"] = json!("aa".repeat(32));
        assert!(
            parse(&bad_catalog)
                .unwrap_err()
                .to_string()
                .contains("direct-template catalog digest differs")
        );
    }

    #[test]
    fn rejects_malformed_direct_authentication_digests() {
        let mut value = manifest();
        value["direct_template_catalog_digest"] = json!("A5".repeat(32));
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("64 lowercase hexadecimal characters")
        );
    }

    #[test]
    fn rejects_inconsistent_direct_container_member_metadata() {
        let mut value = manifest();
        value["plan"]["inspection_summary"]["runtime_container_member"]["container_size_bytes"] =
            json!(4095);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("plan member metadata disagrees"));
    }

    #[test]
    fn rejects_runtime_metadata_cross_record_mismatches() {
        let mut bad_parameter_count = manifest();
        bad_parameter_count["plan"]["inspection_summary"]["parameter_count"] = json!(2);
        assert!(
            parse(&bad_parameter_count)
                .unwrap_err()
                .to_string()
                .contains("plan parameter count")
        );

        let mut bad_external = manifest();
        bad_external["runtime_metadata"]["external_legs"][0]["physical_pdg"] = json!(24);
        assert!(
            parse(&bad_external)
                .unwrap_err()
                .to_string()
                .contains("physical PDGs")
        );
    }
}
