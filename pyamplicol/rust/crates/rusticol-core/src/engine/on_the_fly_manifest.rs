// SPDX-License-Identifier: 0BSD

//! Compact execution-manifest validation for the private on-the-fly LC lane.
//!
//! The verified outer artifact owns file sizes and digests, while PACBIN owns
//! its member index and payload digests. This manifest therefore records only
//! the identities and canonical paths needed to select the lane.

use super::recurrence_manifest::{
    RecurrenceExternalLeg, RecurrenceNormalization, RecurrenceParameterProjection,
    RecurrenceParticleMass, RecurrenceRuntimeParameter,
};
use super::{
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY, ON_THE_FLY_RUNTIME_CAPABILITY, confined_internal_path,
};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolError, RusticolResult};
use serde::Deserialize;
use std::collections::BTreeSet;

pub(super) const ON_THE_FLY_EXECUTION_KIND: &str = "pyamplicol-runtime-on-the-fly-execution";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_KIND: &str =
    "pyamplicol-on-the-fly-runtime-container";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA: u16 = 1;
pub(super) const ON_THE_FLY_RUNTIME_STORAGE_ABI: &str = "pacbin-v1";
pub(super) const ON_THE_FLY_RUNTIME_CONTAINER_PATH: &str = "on-the-fly-runtime.pacbin";
pub(super) const ON_THE_FLY_PROCESS_SEED_MEMBER: &str = "on-the-fly/process-seed-v1.bin";
pub(super) const ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH: &str = "model/eager-kernel-pack.json";
pub(super) const ON_THE_FLY_KERNEL_PAYLOAD_ROOT: &str = "model/eager-kernels";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) kernel_pack: OnTheFlyKernelPackReference,
    pub(super) runtime_options: OnTheFlyRuntimeOptions,
    pub(super) selector_policy: OnTheFlySelectorPolicy,
    pub(super) runtime_metadata: OnTheFlyRuntimeMetadata,
    pub(super) runtime_container: OnTheFlyRuntimeContainer,
}

/// Irreducible process/runtime state shared with the established recurrence
/// lane.  Dense public color-flow catalogs, source-template replicas, and
/// contracted-color metadata deliberately do not belong to an on-the-fly
/// artifact: selectors and source dispatch are derived from the compact seed.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeMetadata {
    pub(super) runtime_parameters: Vec<RecurrenceRuntimeParameter>,
    pub(super) prepared_parameter_defaults: Vec<[f64; 2]>,
    pub(super) parameter_projection: Vec<RecurrenceParameterProjection>,
    pub(super) external_legs: Vec<RecurrenceExternalLeg>,
    pub(super) particle_masses: Vec<RecurrenceParticleMass>,
    pub(super) normalization: RecurrenceNormalization,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyKernelPackReference {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeOptions {
    pub(super) point_tile_size: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum OnTheFlyColorCoverage {
    Complete,
}

/// Compact facts that affect established public LC axis order but cannot be
/// reconstructed from external color roles alone.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlySelectorPolicy {
    pub(super) color_coverage: OnTheFlyColorCoverage,
    #[serde(default)]
    pub(super) reference_color_word: Option<Vec<u32>>,
    pub(super) trace_reflections_folded: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OnTheFlyRuntimeContainer {
    pub(super) kind: String,
    pub(super) schema_version: u16,
    pub(super) storage_abi: String,
    pub(super) path: String,
    pub(super) seed_member_path: String,
}

impl OnTheFlyExecutionManifest {
    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != ON_THE_FLY_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported on-the-fly execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != "lc"
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "on-the-fly execution manifest does not match LC outer process {:?}",
                outer.id
            )));
        }
        let actual = self
            .required_runtime_capabilities
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        ]);
        if actual.len() != self.required_runtime_capabilities.len() || actual != expected {
            return Err(RusticolError::integrity(format!(
                "on-the-fly execution must require exactly {expected:?}"
            )));
        }
        if self.kernel_pack.manifest_path != ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH
            || self.kernel_pack.payload_root != ON_THE_FLY_KERNEL_PAYLOAD_ROOT
        {
            return Err(RusticolError::security(
                "on-the-fly prepared-kernel paths are not canonical",
            ));
        }
        if self.runtime_options.point_tile_size == 0 {
            return Err(RusticolError::artifact(
                "on-the-fly point_tile_size must be positive",
            ));
        }
        self.selector_policy.validate()?;
        self.runtime_metadata
            .validate(&self.external_pdg_order, &self.color_accuracy)?;
        self.runtime_container.validate()
    }
}

impl OnTheFlySelectorPolicy {
    fn validate(&self) -> RusticolResult<()> {
        if let Some(word) = &self.reference_color_word {
            if word.is_empty()
                || word.contains(&0)
                || word.iter().copied().collect::<BTreeSet<_>>().len() != word.len()
            {
                return Err(RusticolError::artifact(
                    "on-the-fly reference color word must contain unique positive external labels",
                ));
            }
        }
        Ok(())
    }
}

impl OnTheFlyRuntimeMetadata {
    fn validate(&self, external_pdgs: &[i32], color_accuracy: &str) -> RusticolResult<()> {
        for (context, len) in [
            ("runtime parameters", self.runtime_parameters.len()),
            (
                "prepared parameter defaults",
                self.prepared_parameter_defaults.len(),
            ),
            ("parameter projection", self.parameter_projection.len()),
            ("external legs", self.external_legs.len()),
            ("particle masses", self.particle_masses.len()),
        ] {
            u32::try_from(len).map_err(|_| {
                RusticolError::artifact(format!(
                    "on-the-fly {context} count exceeds the u32 identifier domain"
                ))
            })?;
        }
        if self
            .prepared_parameter_defaults
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        {
            return Err(RusticolError::artifact(
                "on-the-fly prepared-parameter defaults must be finite complex-f64 values",
            ));
        }
        if self.runtime_parameters.len() != self.parameter_projection.len() {
            return Err(RusticolError::integrity(
                "on-the-fly runtime parameters do not cover the parameter projection",
            ));
        }
        let mut previous_projection_key: Option<(&str, u32)> = None;
        let mut current_name: Option<&str> = None;
        let mut current_template = None;
        let mut current_prepared = None;
        for (runtime_slot, row) in self.parameter_projection.iter().enumerate() {
            if row.runtime_slot as usize != runtime_slot
                || row.runtime_name.is_empty()
                || row.component > 1
                || row.parameter_template_id as usize >= self.prepared_parameter_defaults.len()
                || row
                    .prepared_parameter_id
                    .is_some_and(|id| id as usize >= self.prepared_parameter_defaults.len())
            {
                return Err(RusticolError::integrity(
                    "on-the-fly parameter projection is outside its prepared-parameter domain",
                ));
            }
            let key = (row.runtime_name.as_str(), row.component);
            if previous_projection_key.is_some_and(|previous| previous >= key) {
                return Err(RusticolError::artifact(
                    "on-the-fly parameter projection is not in strict name/component order",
                ));
            }
            previous_projection_key = Some(key);
            if current_name == Some(row.runtime_name.as_str()) {
                if row.component != 1
                    || current_template != Some(row.parameter_template_id)
                    || current_prepared != Some(row.prepared_parameter_id)
                {
                    return Err(RusticolError::integrity(
                        "on-the-fly complex parameter projection rows are inconsistent",
                    ));
                }
            } else {
                if row.component != 0 {
                    return Err(RusticolError::artifact(
                        "on-the-fly parameter projection must begin at component zero",
                    ));
                }
                current_name = Some(&row.runtime_name);
                current_template = Some(row.parameter_template_id);
                current_prepared = Some(row.prepared_parameter_id);
            }
        }
        let mut previous_runtime_name: Option<&str> = None;
        let mut previous_kind: Option<&str> = None;
        for (parameter_index, (parameter, projection)) in self
            .runtime_parameters
            .iter()
            .zip(&self.parameter_projection)
            .enumerate()
        {
            if parameter.parameter_index as usize != parameter_index
                || parameter.parameter_index != projection.runtime_slot
                || parameter.name.is_empty()
                || parameter.kind.is_empty()
                || !parameter.default.is_finite()
            {
                return Err(RusticolError::integrity(
                    "on-the-fly runtime parameter identity/default is invalid",
                ));
            }
            match (&parameter.runtime_name, &parameter.complex_component) {
                (Some(runtime_name), Some(component)) => {
                    let expected_component = match projection.component {
                        0 => "real",
                        1 => "imag",
                        _ => unreachable!("projection component was validated"),
                    };
                    if runtime_name.is_empty()
                        || component != expected_component
                        || projection.runtime_name != *runtime_name
                        || parameter.name != format!("{runtime_name}.{component}")
                        || (previous_runtime_name == Some(runtime_name)
                            && previous_kind != Some(parameter.kind.as_str()))
                    {
                        return Err(RusticolError::integrity(
                            "on-the-fly complex runtime parameter disagrees with its projection",
                        ));
                    }
                    previous_runtime_name = Some(runtime_name);
                    previous_kind = Some(&parameter.kind);
                }
                (None, None) => {
                    if projection.component != 0 || parameter.name != projection.runtime_name {
                        return Err(RusticolError::integrity(
                            "on-the-fly real runtime parameter disagrees with its projection",
                        ));
                    }
                    previous_runtime_name = None;
                    previous_kind = None;
                }
                _ => {
                    return Err(RusticolError::artifact(
                        "on-the-fly runtime_name and complex_component must appear together",
                    ));
                }
            }
            if let Some(prepared_id) = projection.prepared_parameter_id
                && parameter.default
                    != self.prepared_parameter_defaults[prepared_id as usize]
                        [projection.component as usize]
            {
                return Err(RusticolError::integrity(
                    "on-the-fly runtime parameter default disagrees with its prepared default",
                ));
            }
        }

        if self.external_legs.len() != external_pdgs.len() {
            return Err(RusticolError::integrity(
                "on-the-fly external legs do not match the execution PDG order",
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
                || leg.physical_pdg != *expected_pdg
                || (saw_final && leg.is_initial)
            {
                return Err(RusticolError::integrity(
                    "on-the-fly external-leg order or identity is not canonical",
                ));
            }
            saw_final |= !leg.is_initial;
        }
        let mut previous_mass_pdg = None;
        for row in &self.particle_masses {
            if previous_mass_pdg.is_some_and(|previous| previous >= row.outgoing_pdg)
                || !row.mass.is_finite()
                || row.mass < 0.0
            {
                return Err(RusticolError::artifact(
                    "on-the-fly particle masses are not strictly ordered finite nonnegative values",
                ));
            }
            previous_mass_pdg = Some(row.outgoing_pdg);
        }
        let normalization = &self.normalization;
        if normalization.color_accuracy != color_accuracy
            || [
                normalization.color_factor,
                normalization.average_factor,
                normalization.identical_factor,
                normalization.global_coupling_factor,
            ]
            .iter()
            .any(|value| !value.is_finite())
            || normalization.average_factor <= 0.0
            || normalization.identical_factor <= 0.0
            || normalization.qcd_coupling_power.is_some()
                != normalization.electroweak_coupling_power.is_some()
            || normalization.coupling_policy.is_empty()
        {
            return Err(RusticolError::integrity(
                "on-the-fly normalization metadata is invalid",
            ));
        }
        Ok(())
    }
}

impl OnTheFlyRuntimeContainer {
    fn validate(&self) -> RusticolResult<()> {
        if self.kind != ON_THE_FLY_RUNTIME_CONTAINER_KIND
            || self.schema_version != ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA
            || self.storage_abi != ON_THE_FLY_RUNTIME_STORAGE_ABI
        {
            return Err(RusticolError::compatibility(
                "unsupported on-the-fly runtime-container contract",
            ));
        }
        if self.path != ON_THE_FLY_RUNTIME_CONTAINER_PATH
            || self.seed_member_path != ON_THE_FLY_PROCESS_SEED_MEMBER
        {
            return Err(RusticolError::security(
                "on-the-fly runtime container or seed path is not canonical",
            ));
        }
        confined_internal_path(&self.path, "on-the-fly runtime container")?;
        confined_internal_path(&self.seed_member_path, "on-the-fly process seed")?;
        Ok(())
    }
}

pub(super) fn parse_on_the_fly_execution_manifest(
    bytes: &[u8],
    outer: &ArtifactProcess,
) -> RusticolResult<OnTheFlyExecutionManifest> {
    let manifest: OnTheFlyExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse on-the-fly execution manifest: {error}"
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn outer() -> ArtifactProcess {
        ArtifactProcess {
            id: "d_dbar_to_t_tbar_g_g".into(),
            expression: "d d~ > t t~ g g".into(),
            color_accuracy: "lc".into(),
            external_pdgs: vec![1, -1, 6, -6, 21, 21],
            physics_path: "processes/d_dbar_to_t_tbar_g_g/physics.json".into(),
            required_runtime_capabilities: vec![
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY.into(),
                ON_THE_FLY_RUNTIME_CAPABILITY.into(),
            ],
            aliases: Vec::new(),
        }
    }

    fn manifest() -> Value {
        let outer = outer();
        json!({
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": ON_THE_FLY_EXECUTION_KIND,
            "required_runtime_capabilities": [
                ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
                ON_THE_FLY_RUNTIME_CAPABILITY,
            ],
            "process": outer.expression,
            "key": outer.id,
            "color_accuracy": "lc",
            "external_pdg_order": outer.external_pdgs,
            "kernel_pack": {
                "manifest_path": ON_THE_FLY_KERNEL_PACK_MANIFEST_PATH,
                "payload_root": ON_THE_FLY_KERNEL_PAYLOAD_ROOT,
            },
            "runtime_options": {"point_tile_size": 64},
            "selector_policy": {
                "color_coverage": "complete",
                "reference_color_word": null,
                "trace_reflections_folded": false
            },
            "runtime_metadata": {
                "runtime_parameters": [],
                "prepared_parameter_defaults": [],
                "parameter_projection": [],
                "external_legs": [
                    {"source_slot": 0, "public_label": 1, "physical_pdg": 1, "outgoing_pdg": -1, "is_initial": true},
                    {"source_slot": 1, "public_label": 2, "physical_pdg": -1, "outgoing_pdg": 1, "is_initial": true},
                    {"source_slot": 2, "public_label": 3, "physical_pdg": 6, "outgoing_pdg": 6, "is_initial": false},
                    {"source_slot": 3, "public_label": 4, "physical_pdg": -6, "outgoing_pdg": -6, "is_initial": false},
                    {"source_slot": 4, "public_label": 5, "physical_pdg": 21, "outgoing_pdg": 21, "is_initial": false},
                    {"source_slot": 5, "public_label": 6, "physical_pdg": 21, "outgoing_pdg": 21, "is_initial": false}
                ],
                "particle_masses": [],
                "normalization": {
                    "color_accuracy": "lc",
                    "color_factor": 1.0,
                    "average_factor": 1.0,
                    "identical_factor": 1.0,
                    "global_coupling_factor": 1.0,
                    "qcd_coupling_power": 0,
                    "electroweak_coupling_power": 0,
                    "couplings_in_stage_evaluators": true,
                    "coupling_policy": "local"
                }
            },
            "runtime_container": {
                "kind": ON_THE_FLY_RUNTIME_CONTAINER_KIND,
                "schema_version": ON_THE_FLY_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": ON_THE_FLY_RUNTIME_STORAGE_ABI,
                "path": ON_THE_FLY_RUNTIME_CONTAINER_PATH,
                "seed_member_path": ON_THE_FLY_PROCESS_SEED_MEMBER,
            },
        })
    }

    fn parse(value: &Value) -> RusticolResult<OnTheFlyExecutionManifest> {
        parse_on_the_fly_execution_manifest(&serde_json::to_vec(value).unwrap(), &outer())
    }

    #[test]
    fn accepts_only_the_compact_lc_contract() {
        let parsed = parse(&manifest()).unwrap();
        assert_eq!(parsed.runtime_options.point_tile_size, 64);
        assert_eq!(
            parsed.runtime_container.seed_member_path,
            ON_THE_FLY_PROCESS_SEED_MEMBER
        );

        let mut nlc = manifest();
        nlc["color_accuracy"] = json!("nlc");
        assert!(parse(&nlc).is_err());

        let mut extra_capability = manifest();
        extra_capability["required_runtime_capabilities"]
            .as_array_mut()
            .unwrap()
            .push(json!("unrelated.capability.v1"));
        assert!(parse(&extra_capability).is_err());
    }

    #[test]
    fn rejects_noncanonical_runtime_and_kernel_paths() {
        let mut runtime_escape = manifest();
        runtime_escape["runtime_container"]["seed_member_path"] = json!("../seed.bin");
        assert!(parse(&runtime_escape).is_err());

        let mut kernel_alias = manifest();
        kernel_alias["kernel_pack"]["manifest_path"] = json!("model/other.json");
        assert!(parse(&kernel_alias).is_err());
    }
}
