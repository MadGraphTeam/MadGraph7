// SPDX-License-Identifier: 0BSD

//! Native loader for compact eager plan-v3 artifacts.

use super::eager_manifest::PreparedKernelPackManifest;
use super::eager_v3_decode::DecodedEagerRuntimeV3;
use super::eager_v3_manifest::EagerV3ExecutionManifest;
use super::*;
use crate::eager_runtime::EagerPlanV3Sections;
use crate::{
    EagerCouplingRow, EagerPlanCouplingRow, EagerPlanReductionEntryKind, EagerRuntimeOptions,
    MISSING_U32,
};
use serde::Deserialize;
use serde_json::json;

const NATIVE_REDUCTION_GROUPS_EXTENSION_KEY: &str = "native_reduction_groups";
const NATIVE_REDUCTION_GROUPS_KIND: &str = "pyamplicol-eager-plan-v3-reduction-groups";
const NATIVE_REDUCTION_GROUPS_SCHEMA_VERSION: u32 = 1;
const NATIVE_REDUCTION_GROUPS_STORAGE_ABI: &str = "pacbin-v1";
const NATIVE_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI: &str = "pyamplicol-eager-runtime-layout-v1";
const NATIVE_REDUCTION_GROUPS_CONTAINER_PATH: &str = "eager-runtime.pacbin";
const NATIVE_REDUCTION_GROUPS_GROUP_MEMBER: &str = "reductions/groups.bin";
const NATIVE_REDUCTION_GROUPS_ENTRY_MEMBER: &str = "reductions/entries.bin";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeReductionGroupsDescriptor {
    kind: String,
    schema_version: u32,
    storage_abi: String,
    runtime_layout_abi: String,
    container_path: String,
    group_member: String,
    entry_member: String,
    group_count: u64,
}

pub(super) struct EagerV3PreparedPack {
    pub(super) manifest: PreparedKernelPackManifest,
    pub(super) payload_root: PathBuf,
}

pub(super) struct LoadedEagerV3Runtime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: EagerNativeRuntime,
}

pub(super) fn load_eager_v3_prepared_pack(
    artifact: &VerifiedArtifact,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<EagerV3PreparedPack> {
    let manifest_path = confined_internal_path(
        &manifest.kernel_pack.manifest_path,
        "eager plan-v3 prepared kernel-pack manifest path",
    )?;
    let manifest_path = manifest_path.to_str().ok_or_else(|| {
        RusticolError::security("prepared kernel-pack manifest path is not valid UTF-8")
    })?;
    if artifact.payload(manifest_path)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(format!(
            "prepared kernel-pack path {manifest_path:?} is not an evaluator-manifest payload"
        )));
    }
    let bytes = artifact.read_payload(manifest_path)?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse prepared kernel pack {manifest_path:?}: {error}"
        ))
    })?;
    pack.validate()?;
    let payload_root = confined_internal_path(
        &manifest.kernel_pack.payload_root,
        "eager plan-v3 prepared kernel payload root",
    )?;
    let payload_root = artifact.root().join(payload_root);
    validate_prepared_kernel_references(artifact, &payload_root, &pack)?;
    Ok(EagerV3PreparedPack {
        manifest: pack,
        payload_root,
    })
}

pub(super) fn load_eager_v3_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &EagerV3ExecutionManifest,
    physics: &mut ProcessPhysicsV1,
) -> RusticolResult<LoadedEagerV3Runtime> {
    let pack = load_eager_v3_prepared_pack(artifact, manifest)?;
    let container = open_verified_eager_v3_runtime_container(artifact, evaluator_root, manifest)?;
    let decoded =
        super::eager_v3_decode::decode_eager_v3_runtime(&container, manifest, &pack.manifest)?;
    hydrate_native_reduction_groups(
        physics,
        &decoded.reduction_groups,
        &decoded.reduction_entries,
    )?;
    let mut common =
        super::eager_v3_common::build_eager_v3_common_runtime(&decoded, manifest, physics.clone())?;

    let kernel_payloads = artifact.evaluator_payload_store(&pack.payload_root)?;
    let (parameter_projection, couplings, model_parameter_evaluator) =
        prepare_plan_v3_parameter_state(
            &pack.manifest,
            &decoded,
            &common.model_parameters,
            &kernel_payloads,
        )?;
    common.model_parameter_evaluator = model_parameter_evaluator;
    common.refresh_derived_model_parameters()?;

    let prepared_parameter_count = u32::try_from(parameter_projection.parameter_count)
        .map_err(|_| RusticolError::artifact("prepared parameter count exceeds u32"))?;
    let plan = crate::EagerExecutionPlan::from_plan_v3_sections(EagerPlanV3Sections {
        kernels: &decoded.kernel_specs,
        prepared_parameter_count,
        currents: &decoded.currents,
        values: &decoded.values,
        momenta: &decoded.momenta,
        parameters: &decoded.parameters,
        stages: &decoded.stages,
        couplings: &couplings,
        invocations: &decoded.invocations,
        attachments: &decoded.attachments,
        finalizations: &decoded.finalizations,
        closures: &decoded.closures,
        direct_coefficients: &decoded.direct_coefficients,
        selector_domains: &decoded.selector_domains,
        selector_memberships: &decoded.selector_memberships,
        reduction_groups: &decoded.reduction_groups,
        reduction_entries: &decoded.reduction_entries,
        exact_factors: &decoded.exact_factors,
        color_contraction_entry_start: decoded.color_contraction_entry_start,
        color_contraction_entry_count: decoded.color_contraction_entry_count,
    })?;
    let scheduler = crate::EagerExecutionRuntime::new(
        plan,
        EagerRuntimeOptions {
            point_tile_size: decoded.runtime_options.point_tile_size,
            workspace_bytes: decoded.runtime_options.workspace_bytes,
        },
    )?;
    let backend = PreparedEvaluatorBackend::load_from_store(&pack.manifest, &kernel_payloads)?;
    let (raw_sum_groups, color_contraction) = reduction_runtime(&decoded, manifest)?;
    if let Some(selector_group_ids) = scheduler.selector_group_ids() {
        let known_group_ids = raw_sum_groups
            .iter()
            .map(|group| {
                u32::try_from(group.id).map_err(|_| {
                    RusticolError::integrity(format!(
                        "eager coherent group {} does not fit the selector-domain ABI",
                        group.id
                    ))
                })
            })
            .collect::<RusticolResult<BTreeSet<_>>>()?;
        if let Some(unknown) = selector_group_ids
            .iter()
            .find(|group_id| !known_group_ids.contains(group_id))
        {
            return Err(RusticolError::integrity(format!(
                "eager selector domains reference unknown coherent group {unknown}"
            )));
        }
    }
    Ok(LoadedEagerV3Runtime {
        common,
        lane: EagerNativeRuntime::new(
            scheduler,
            backend,
            pack.manifest.backend,
            parameter_projection,
            raw_sum_groups,
            color_contraction,
        ),
    })
}

pub(super) fn load_eager_v3_exact_sections(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &EagerV3ExecutionManifest,
    physics: &mut ProcessPhysicsV1,
) -> RusticolResult<NativeEagerExactSections> {
    let pack = load_eager_v3_prepared_pack(artifact, manifest)?;
    let container = open_verified_eager_v3_runtime_container(artifact, evaluator_root, manifest)?;
    let decoded =
        super::eager_v3_decode::decode_eager_v3_runtime(&container, manifest, &pack.manifest)?;
    hydrate_native_reduction_groups(
        physics,
        &decoded.reduction_groups,
        &decoded.reduction_entries,
    )?;

    let amplitude_stage = exact_amplitude_stage(&decoded, manifest)?;
    let exact_schema = super::eager_v3_common::build_eager_v3_exact_schema(
        &decoded,
        manifest,
        physics.clone(),
        amplitude_stage,
    )?;
    let reduction_groups = serde_json::to_value(&physics.reduction.groups).map_err(|error| {
        RusticolError::serialization(format!(
            "could not serialize compact eager exact reduction groups: {error}"
        ))
    })?;
    let selector_group_ids = decoded
        .reduction_groups
        .iter()
        .map(|group| group.coherent_group_id)
        .collect::<Vec<_>>();
    let selector_domains = decoded
        .selector_domains
        .iter()
        .map(|domain| {
            Ok(table_range(
                &decoded.selector_memberships,
                domain.member_start,
                domain.member_count,
                "selector-domain members",
            )?
            .to_vec())
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let couplings = decoded
        .couplings
        .iter()
        .map(|row| {
            let [constant_real, constant_imaginary] =
                exact_factor_pair(&decoded, row.constant_factor_id, "coupling")?;
            Ok(NativeEagerExactCoupling {
                real_parameter_id: row.real_parameter_id,
                imaginary_parameter_id: row.imaginary_parameter_id,
                constant_real,
                constant_imaginary,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let stages = decoded
        .stages
        .iter()
        .map(|row| NativeEagerExactStage {
            stage_index: row.stage_index,
            invocation_start: row.invocation_start,
            invocation_count: row.invocation_count,
            attachment_start: row.attachment_start,
            attachment_count: row.attachment_count,
            finalization_start: row.finalization_start,
            finalization_count: row.finalization_count,
        })
        .collect();
    let invocations = decoded
        .invocations
        .iter()
        .map(|row| NativeEagerExactInvocation {
            kernel_id: row.kernel_id,
            left_value_slot_id: row.left_value_slot_id,
            right_value_slot_id: row.right_value_slot_id,
            left_momentum_slot_id: row.left_momentum_slot_id,
            right_momentum_slot_id: row.right_momentum_slot_id,
            coupling_slot_id: row.coupling_slot_id,
            output_factor_source: row.output_factor_source,
            attachment_start: row.attachment_start,
            attachment_count: row.attachment_count,
            selector_domain_id: row.selector_domain_id,
        })
        .collect();
    let attachments = decoded
        .attachments
        .iter()
        .map(|row| {
            let representative = exact_factor(
                &decoded,
                row.representative_evaluation_factor_id,
                "attachment representative",
            )?;
            if representative == Complex::new(0.0, 0.0) {
                return Err(RusticolError::integrity(
                    "eager exact attachment representative factor is zero",
                ));
            }
            Ok(NativeEagerExactAttachment {
                result_current_id: row.result_current_id,
                factor_numerators: vec![
                    exact_factor_pair(&decoded, row.color_factor_id, "attachment color")?,
                    exact_factor_pair(&decoded, row.evaluation_factor_id, "attachment evaluation")?,
                    exact_factor_pair(
                        &decoded,
                        row.normalization_factor_id,
                        "attachment normalization",
                    )?,
                ],
                factor_denominator: Some(exact_factor_pair(
                    &decoded,
                    row.representative_evaluation_factor_id,
                    "attachment representative",
                )?),
                selector_domain_id: row.selector_domain_id,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let finalizations = decoded
        .finalizations
        .iter()
        .map(|row| NativeEagerExactFinalization {
            kernel_id: row.kernel_id,
            current_id: row.current_id,
            unpropagated_value_slot_id: row.unpropagated_value_slot_id,
            propagated_value_slot_id: row.propagated_value_slot_id,
            momentum_slot_id: row.momentum_slot_id,
            unpropagated_selector_domain_id: row.unpropagated_selector_domain_id,
            propagated_selector_domain_id: row.propagated_selector_domain_id,
        })
        .collect();
    let closures = decoded
        .closures
        .iter()
        .map(|row| exact_closure(&decoded, row))
        .collect::<RusticolResult<Vec<_>>>()?;

    Ok(NativeEagerExactSections {
        process_id: manifest.key.clone(),
        exact_schema,
        reduction_groups,
        selector_group_ids,
        selector_domains,
        couplings,
        stages,
        invocations,
        attachments,
        finalizations,
        closures,
    })
}

fn exact_amplitude_stage(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<Value> {
    let color_sector_by_group = coherent_group_color_sector_ids(decoded)?;
    let group_weights = decoded
        .reduction_groups
        .iter()
        .map(|group| {
            exact_real_factor(decoded, group.helicity_weight_factor_id, "helicity weight")?;
            exact_real_factor(
                decoded,
                group.all_sector_weight_factor_id,
                "all-sector weight",
            )?;
            Ok((
                group.coherent_group_id,
                (
                    group.helicity_weight_factor_id,
                    group.all_sector_weight_factor_id,
                ),
            ))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let roots = decoded
        .closures
        .iter()
        .map(|root| {
            let (helicity_weight_id, all_sector_weight_id) = group_weights
                .get(&root.coherent_group_id)
                .ok_or_else(|| {
                RusticolError::integrity("eager exact closure references an unknown coherent group")
            })?;
            Ok(json!({
                "output_index": root.amplitude_index,
                "root_id": root.root_id,
                "kind": if root.kernel_id == MISSING_U32 {
                    "direct-contraction"
                } else {
                    "kernel-closure"
                },
                "coherent_group_id": root.coherent_group_id,
                "color_sector_id": color_sector_by_group
                    .get(&root.coherent_group_id)
                    .copied()
                    .ok_or_else(|| RusticolError::integrity(
                        "eager exact closure references an unknown coherent-group color sector"
                    ))?,
                "helicity_weight": exact_real_factor_number(decoded, *helicity_weight_id, "helicity weight")?,
                "all_sector_weight": exact_real_factor_number(decoded, *all_sector_weight_id, "all-sector weight")?,
            }))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let color_contraction = if manifest.color_accuracy == "lc" {
        Value::Null
    } else {
        let entries = reduction_range(
            &decoded.reduction_entries,
            decoded.color_contraction_entry_start,
            decoded.color_contraction_entry_count,
            "exact color contraction",
        )?
        .iter()
        .map(|row| {
            if row.kind != EagerPlanReductionEntryKind::ColorContraction {
                return Err(RusticolError::integrity(
                    "eager exact color-contraction range contains another entry kind",
                ));
            }
            let weight = exact_factor_pair(decoded, row.factor_id, "color-contraction weight")?;
            let symmetry = if row.auxiliary_factor_id == MISSING_U32 {
                exact_number(1.0)
            } else {
                exact_real_factor_number(decoded, row.auxiliary_factor_id, "color symmetry factor")?
            };
            Ok(json!({
                "left_group_id": row.left_id,
                "right_group_id": row.right_id,
                "weight": weight,
                "symmetry_factor": symmetry,
            }))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
        json!({"entries": entries})
    };
    Ok(json!({
        "stage_kind": "amplitude-roots",
        "output_count": decoded.dimensions.amplitude_count,
        "roots": roots,
        "color_contraction": color_contraction,
    }))
}

fn coherent_group_color_sector_ids(
    decoded: &DecodedEagerRuntimeV3,
) -> RusticolResult<BTreeMap<u32, u32>> {
    let table = decoded
        .retained_tables
        .iter()
        .find(|table| table.name.as_ref() == "coherent_groups")
        .ok_or_else(|| RusticolError::integrity("eager coherent-group table is absent"))?;
    let column = table
        .columns
        .iter()
        .find(|column| column.name.as_ref() == "color_sector_id")
        .ok_or_else(|| {
            RusticolError::integrity("eager coherent-group color-sector column is absent")
        })?;
    if column.elements_per_row != 1 {
        return Err(RusticolError::integrity(
            "eager coherent-group color-sector column is not scalar",
        ));
    }
    let super::eager_v3_decode::DecodedEagerPrimitiveColumn::U32(color_sector_ids) = &column.values
    else {
        return Err(RusticolError::integrity(
            "eager coherent-group color-sector column has the wrong primitive type",
        ));
    };
    if color_sector_ids.len() != decoded.reduction_groups.len()
        || color_sector_ids.len()
            != usize::try_from(table.row_count).map_err(|_| {
                RusticolError::artifact("eager coherent-group row count exceeds usize")
            })?
    {
        return Err(RusticolError::integrity(
            "eager coherent-group color-sector coverage is inconsistent",
        ));
    }
    decoded
        .reduction_groups
        .iter()
        .zip(color_sector_ids)
        .map(|(group, sector_id)| Ok((group.coherent_group_id, *sector_id)))
        .collect()
}

fn exact_closure(
    decoded: &DecodedEagerRuntimeV3,
    row: &crate::EagerPlanClosureRow,
) -> RusticolResult<NativeEagerExactClosure> {
    let color = exact_factor_pair(decoded, row.color_factor_id, "closure color")?;
    let (factor_numerators, direct_coefficients) = if row.kernel_id == MISSING_U32 {
        let coefficients = table_range(
            &decoded.direct_coefficients,
            row.direct_coefficient_start,
            row.direct_coefficient_count,
            "direct closure coefficients",
        )?
        .iter()
        .enumerate()
        .map(|(component, coefficient)| {
            if coefficient.component_index as usize != component {
                return Err(RusticolError::integrity(
                    "eager exact direct coefficients are not component ordered",
                ));
            }
            exact_factor_pair(decoded, coefficient.factor_id, "direct coefficient")
        })
        .collect::<RusticolResult<Vec<_>>>()?;
        (vec![color], Some(coefficients))
    } else {
        (
            vec![
                color,
                exact_factor_pair(
                    decoded,
                    row.normalization_factor_id,
                    "closure normalization",
                )?,
            ],
            None,
        )
    };
    Ok(NativeEagerExactClosure {
        kernel_id: row.kernel_id,
        left_value_slot_id: row.left_value_slot_id,
        right_value_slot_id: row.right_value_slot_id,
        amplitude_index: row.amplitude_index,
        coupling_slot_id: row.coupling_slot_id,
        output_factor_source: row.output_factor_source,
        factor_numerators,
        factor_denominator: None,
        direct_coefficients,
        coherent_group_id: row.coherent_group_id,
        selector_domain_id: row.selector_domain_id,
    })
}

fn exact_number(value: f64) -> String {
    format!("binary64:{:016x}", value.to_bits())
}

fn exact_factor_pair(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<[String; 2]> {
    let factor = decoded
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| RusticolError::integrity(format!("eager {context} factor is absent")))?;
    let value = exact_factor(decoded, factor_id, context)?;
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(RusticolError::integrity(format!(
            "eager {context} factor is not finite"
        )));
    }
    Ok([
        format!("binary64:{:016x}", factor.real_bits),
        format!("binary64:{:016x}", factor.imaginary_bits),
    ])
}

fn exact_real_factor_number(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<String> {
    exact_real_factor(decoded, factor_id, context)?;
    Ok(exact_factor_pair(decoded, factor_id, context)?[0].clone())
}

pub(super) fn reject_native_reduction_groups_for_compiled(
    physics: &ProcessPhysicsV1,
) -> RusticolResult<()> {
    if physics
        .extensions
        .contains_key(NATIVE_REDUCTION_GROUPS_EXTENSION_KEY)
    {
        return Err(RusticolError::compatibility(
            "compact PACBIN-backed reduction groups require eager plan-v3 execution; regenerate this compiled artifact",
        ));
    }
    Ok(())
}

fn hydrate_native_reduction_groups(
    physics: &mut ProcessPhysicsV1,
    groups: &[crate::EagerPlanReductionGroupRow],
    entries: &[crate::EagerPlanReductionEntryRow],
) -> RusticolResult<()> {
    let Some(value) = physics
        .extensions
        .get(NATIVE_REDUCTION_GROUPS_EXTENSION_KEY)
    else {
        return Ok(());
    };
    if !physics.reduction.groups.is_empty() {
        return Err(RusticolError::integrity(
            "compact native reduction metadata may not duplicate expanded reduction groups",
        ));
    }
    let descriptor: NativeReductionGroupsDescriptor = serde_json::from_value(value.clone())
        .map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse compact native reduction descriptor: {error}"
            ))
        })?;
    descriptor.validate()?;
    let declared_group_count = usize::try_from(descriptor.group_count).map_err(|_| {
        RusticolError::artifact("compact native reduction group count exceeds usize")
    })?;
    if declared_group_count != groups.len() {
        return Err(RusticolError::integrity(format!(
            "compact native reduction descriptor declares {declared_group_count} groups, PACBIN contains {}",
            groups.len()
        )));
    }

    let mut hydrated = Vec::with_capacity(groups.len());
    let mut seen_group_ids = BTreeSet::new();
    for group in groups {
        if !seen_group_ids.insert(group.coherent_group_id) {
            return Err(RusticolError::integrity(format!(
                "compact native reduction metadata contains duplicate coherent group {}",
                group.coherent_group_id
            )));
        }
        let selectors = reduction_range(
            entries,
            group.selector_entry_start,
            group.selector_entry_count,
            "selector members",
        )?;
        let representative = selectors.first().ok_or_else(|| {
            RusticolError::integrity(format!(
                "compact native reduction group {} has no selector members",
                group.coherent_group_id
            ))
        })?;

        let mut physical_helicity_ids = Vec::new();
        let mut physical_color_ids = Vec::new();
        let mut seen_helicities = BTreeSet::new();
        let mut seen_colors = BTreeSet::new();
        for selector in selectors {
            if selector.kind != EagerPlanReductionEntryKind::SelectorMember
                || selector.owner_id != group.coherent_group_id
            {
                return Err(RusticolError::integrity(format!(
                    "compact native reduction group {} contains an invalid selector member",
                    group.coherent_group_id
                )));
            }
            let helicity_index = usize::try_from(selector.left_id).map_err(|_| {
                RusticolError::artifact("compact native helicity selector ID exceeds usize")
            })?;
            let color_index = usize::try_from(selector.right_id).map_err(|_| {
                RusticolError::artifact("compact native color selector ID exceeds usize")
            })?;
            let helicity = physics.helicities.get(helicity_index).ok_or_else(|| {
                RusticolError::integrity(format!(
                    "compact native reduction group {} references unknown helicity selector {}",
                    group.coherent_group_id, selector.left_id
                ))
            })?;
            let color = physics.color_components.get(color_index).ok_or_else(|| {
                RusticolError::integrity(format!(
                    "compact native reduction group {} references unknown color selector {}",
                    group.coherent_group_id, selector.right_id
                ))
            })?;
            if seen_helicities.insert(selector.left_id) {
                physical_helicity_ids.push(helicity.id.clone());
            }
            if seen_colors.insert(selector.right_id) {
                physical_color_ids.push(color.id().to_string());
            }
        }

        let representative_helicity_id = physics
            .helicities
            .get(representative.left_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity("compact native representative helicity is absent")
            })?
            .id
            .clone();
        let representative_color_id = physics
            .color_components
            .get(representative.right_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity("compact native representative color is absent")
            })?
            .id()
            .to_string();
        hydrated.push(crate::ReductionGroup {
            id: format!("reduction:{}", group.coherent_group_id),
            representative_helicity_id,
            representative_color_id,
            physical_helicity_ids,
            physical_color_ids,
        });
    }
    physics.reduction.groups = hydrated;
    physics.validate()?;
    Ok(())
}

impl NativeReductionGroupsDescriptor {
    fn validate(&self) -> RusticolResult<()> {
        let valid = self.kind == NATIVE_REDUCTION_GROUPS_KIND
            && self.schema_version == NATIVE_REDUCTION_GROUPS_SCHEMA_VERSION
            && self.storage_abi == NATIVE_REDUCTION_GROUPS_STORAGE_ABI
            && self.runtime_layout_abi == NATIVE_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI
            && self.container_path == NATIVE_REDUCTION_GROUPS_CONTAINER_PATH
            && self.group_member == NATIVE_REDUCTION_GROUPS_GROUP_MEMBER
            && self.entry_member == NATIVE_REDUCTION_GROUPS_ENTRY_MEMBER;
        if !valid {
            return Err(RusticolError::compatibility(
                "unsupported compact native reduction descriptor; regenerate the eager artifact",
            ));
        }
        Ok(())
    }
}

fn open_verified_eager_v3_runtime_container(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<crate::pacbin::PacbinReader> {
    let container = &manifest.plan.runtime_container;
    let path = evaluator_root.join(&container.path);
    let relative = path.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("eager runtime container escapes the verified artifact root")
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        let std::path::Component::Normal(part) = component else {
            return Err(RusticolError::security(
                "eager runtime container path is not canonical",
            ));
        };
        parts.push(part.to_str().ok_or_else(|| {
            RusticolError::security("eager runtime container path is not valid UTF-8")
        })?);
    }
    let logical_path = parts.join("/");
    let payload = artifact.payload(&logical_path)?;
    if payload.role != crate::PayloadRole::EvaluatorState
        || payload.media_type != "application/octet-stream"
        || payload.process_id.as_deref() != Some(manifest.key.as_str())
        || payload.executable
    {
        return Err(RusticolError::integrity(
            "eager runtime container has an invalid outer payload declaration",
        ));
    }
    if payload.size_bytes != container.size_bytes || payload.sha256 != container.sha256 {
        return Err(RusticolError::integrity(
            "eager runtime container metadata disagrees with its authenticated outer payload",
        ));
    }
    super::eager_v3_manifest::open_eager_v3_runtime_container(evaluator_root, manifest)
}

fn prepare_plan_v3_parameter_state(
    pack: &PreparedKernelPackManifest,
    decoded: &DecodedEagerRuntimeV3,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<(
    EagerParameterProjection,
    Vec<EagerPlanCouplingRow>,
    Option<ModelParameterEvaluatorRuntime>,
)> {
    let legacy_rows = decoded
        .couplings
        .iter()
        .map(|row| {
            let constant = exact_factor(decoded, row.constant_factor_id, "coupling constant")?;
            Ok(EagerCouplingRow {
                real_parameter_id: row.real_parameter_id,
                imag_parameter_id: row.imaginary_parameter_id,
                constant_real: constant.re,
                constant_imag: constant.im,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let encoded = EagerCouplingRow::encode_table(&legacy_rows)?;
    let (projection, remapped, evaluator) = super::eager_load::prepare_eager_parameter_state(
        pack,
        runtime_parameters,
        &encoded,
        payloads,
    )?;
    let remapped = EagerCouplingRow::decode_table(&remapped)?;
    if remapped.len() != decoded.couplings.len() {
        return Err(RusticolError::integrity(
            "projected eager coupling count changed",
        ));
    }
    let couplings = decoded
        .couplings
        .iter()
        .zip(remapped)
        .map(|(source, projected)| EagerPlanCouplingRow {
            coupling_id: source.coupling_id,
            real_parameter_id: projected.real_parameter_id,
            imaginary_parameter_id: projected.imag_parameter_id,
            constant_factor_id: source.constant_factor_id,
        })
        .collect();
    Ok((projection, couplings, evaluator))
}

fn reduction_runtime(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<(Vec<RawSumGroup>, Option<ColorContractionRuntime>)> {
    let color_sector_by_group = (manifest.color_accuracy == "lc")
        .then(|| coherent_group_color_sector_ids(decoded))
        .transpose()?;
    let mut groups = Vec::with_capacity(decoded.reduction_groups.len());
    for group in &decoded.reduction_groups {
        let amplitudes = reduction_range(
            &decoded.reduction_entries,
            group.amplitude_entry_start,
            group.amplitude_entry_count,
            "amplitude members",
        )?;
        let selectors = reduction_range(
            &decoded.reduction_entries,
            group.selector_entry_start,
            group.selector_entry_count,
            "selector members",
        )?;
        if amplitudes
            .iter()
            .any(|entry| entry.kind != EagerPlanReductionEntryKind::AmplitudeMember)
            || selectors
                .iter()
                .any(|entry| entry.kind != EagerPlanReductionEntryKind::SelectorMember)
        {
            return Err(RusticolError::integrity(
                "eager reduction group range has inconsistent entry kinds",
            ));
        }
        let weight =
            exact_real_factor(decoded, group.helicity_weight_factor_id, "helicity weight")?;
        let all_sector_weight = exact_real_factor(
            decoded,
            group.all_sector_weight_factor_id,
            "all-sector weight",
        )?;
        // Selector IDs index the sorted public color axis. Replay uses the
        // original color-plan sector identity retained by each coherent group.
        let sector_ids = eager_raw_sum_sector_ids(
            manifest.color_accuracy.as_str(),
            group.coherent_group_id,
            color_sector_by_group.as_ref(),
        )?;
        groups.push(RawSumGroup {
            id: i64::from(group.coherent_group_id),
            indices: amplitudes
                .iter()
                .map(|entry| entry.left_id as usize)
                .collect(),
            weight,
            all_sector_weight,
            sector_ids,
        });
    }
    if manifest.color_accuracy == "lc" {
        return Ok((groups, None));
    }
    let group_index_by_id = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.id, index))
        .collect::<BTreeMap<_, _>>();
    let contraction_rows = reduction_range(
        &decoded.reduction_entries,
        decoded.color_contraction_entry_start,
        decoded.color_contraction_entry_count,
        "color contraction",
    )?;
    let mut entries = Vec::with_capacity(contraction_rows.len());
    for row in contraction_rows {
        if row.kind != EagerPlanReductionEntryKind::ColorContraction {
            return Err(RusticolError::integrity(
                "eager color-contraction range contains another entry kind",
            ));
        }
        let left_id = i64::from(row.left_id);
        let right_id = i64::from(row.right_id);
        let coefficient = exact_factor(decoded, row.factor_id, "color-contraction weight")?;
        let symmetry_factor = if row.auxiliary_factor_id == MISSING_U32 {
            1.0
        } else {
            exact_real_factor(decoded, row.auxiliary_factor_id, "color symmetry factor")?
        };
        entries.push(ColorContractionEntry {
            left_group_index: *group_index_by_id.get(&left_id).ok_or_else(|| {
                RusticolError::integrity("color contraction references an unknown left group")
            })?,
            right_group_index: *group_index_by_id.get(&right_id).ok_or_else(|| {
                RusticolError::integrity("color contraction references an unknown right group")
            })?,
            weight_re: coefficient.re,
            weight_im: coefficient.im,
            symmetry_factor,
        });
    }
    let contraction = ColorContractionRuntime::new(&groups, entries);
    Ok((groups, Some(contraction)))
}

fn eager_raw_sum_sector_ids(
    color_accuracy: &str,
    coherent_group_id: u32,
    color_sector_by_group: Option<&BTreeMap<u32, u32>>,
) -> RusticolResult<Vec<i64>> {
    if color_accuracy != "lc" {
        return Ok(Vec::new());
    }
    let sector_id = color_sector_by_group
        .and_then(|sectors| sectors.get(&coherent_group_id))
        .copied()
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "eager LC coherent group {coherent_group_id} has no color-sector identity"
            ))
        })?;
    Ok(vec![i64::from(sector_id)])
}

fn exact_factor(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<Complex<f64>> {
    let factor = decoded
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| RusticolError::integrity(format!("eager {context} factor is absent")))?;
    let value = Complex::new(
        f64::from_bits(factor.real_bits),
        f64::from_bits(factor.imaginary_bits),
    );
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(RusticolError::integrity(format!(
            "eager {context} factor is not finite"
        )));
    }
    Ok(value)
}

fn exact_real_factor(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<f64> {
    let value = exact_factor(decoded, factor_id, context)?;
    if value.im != 0.0 {
        return Err(RusticolError::integrity(format!(
            "eager {context} factor is not real"
        )));
    }
    Ok(value.re)
}

fn reduction_range<'a>(
    entries: &'a [crate::EagerPlanReductionEntryRow],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [crate::EagerPlanReductionEntryRow]> {
    table_range(entries, start, count, context)
}

fn table_range<'a, T>(
    entries: &'a [T],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let start = usize::try_from(start)
        .map_err(|_| RusticolError::artifact(format!("eager {context} start exceeds usize")))?;
    let count = usize::try_from(count)
        .map_err(|_| RusticolError::artifact(format!("eager {context} count exceeds usize")))?;
    let stop = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("eager {context} range overflows")))?;
    entries
        .get(start..stop)
        .ok_or_else(|| RusticolError::integrity(format!("eager {context} range is out of bounds")))
}

#[cfg(test)]
mod compact_reduction_tests {
    use super::*;
    use crate::{
        ColorAccuracy, ColorComponent, ContractedColor, Coverage, ExternalParticle, Helicity,
        ParticleRole, Reduction, ReductionKind, SelectorCapabilities,
    };
    use serde_json::json;

    fn compact_physics() -> ProcessPhysicsV1 {
        ProcessPhysicsV1 {
            schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
            kind: "pyamplicol-resolved-physics".to_string(),
            process_id: "p0".to_string(),
            process: "a b > c".to_string(),
            color_accuracy: ColorAccuracy::Full,
            coverage: Coverage {
                helicities: "complete".to_string(),
                color: "contracted".to_string(),
                color_kind: "contracted-color".to_string(),
                structural_zero_helicity_count: 0,
            },
            external_particles: (0..3)
                .map(|index| ExternalParticle {
                    index,
                    label: index + 1,
                    particle: format!("particle-{index}"),
                    pdg: index as i32 + 1,
                    role: if index < 2 {
                        ParticleRole::Initial
                    } else {
                        ParticleRole::Final
                    },
                    momentum_slot: index,
                    momentum_components: [
                        "E".to_string(),
                        "px".to_string(),
                        "py".to_string(),
                        "pz".to_string(),
                    ],
                })
                .collect(),
            helicities: vec![
                Helicity {
                    id: "helicity:0".to_string(),
                    index: 0,
                    values: vec![1, -1, 1],
                    computed: true,
                    structural_zero: false,
                    representative_id: "helicity:0".to_string(),
                    coefficient: 1.0,
                },
                Helicity {
                    id: "helicity:1".to_string(),
                    index: 1,
                    values: vec![-1, 1, -1],
                    computed: true,
                    structural_zero: false,
                    representative_id: "helicity:1".to_string(),
                    coefficient: 1.0,
                },
            ],
            color_components: vec![ColorComponent::ContractedColor(ContractedColor {
                id: "contracted".to_string(),
                index: 0,
                description: "coherent contracted color".to_string(),
            })],
            reduction: Reduction {
                kind: ReductionKind::ContractedColor,
                groups: Vec::new(),
            },
            model_parameters: Vec::new(),
            selectors: SelectorCapabilities {
                helicity: true,
                color_flow: false,
                contracted_color: false,
            },
            extensions: BTreeMap::from([(
                NATIVE_REDUCTION_GROUPS_EXTENSION_KEY.to_string(),
                json!({
                    "kind": NATIVE_REDUCTION_GROUPS_KIND,
                    "schema_version": NATIVE_REDUCTION_GROUPS_SCHEMA_VERSION,
                    "storage_abi": NATIVE_REDUCTION_GROUPS_STORAGE_ABI,
                    "runtime_layout_abi": NATIVE_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI,
                    "container_path": NATIVE_REDUCTION_GROUPS_CONTAINER_PATH,
                    "group_member": NATIVE_REDUCTION_GROUPS_GROUP_MEMBER,
                    "entry_member": NATIVE_REDUCTION_GROUPS_ENTRY_MEMBER,
                    "group_count": 1,
                }),
            )]),
        }
    }

    fn group() -> crate::EagerPlanReductionGroupRow {
        crate::EagerPlanReductionGroupRow {
            coherent_group_id: 7,
            amplitude_entry_start: 0,
            amplitude_entry_count: 0,
            selector_entry_start: 0,
            selector_entry_count: 3,
            helicity_weight_factor_id: 0,
            all_sector_weight_factor_id: 0,
        }
    }

    fn selector(
        owner_id: u32,
        helicity_id: u32,
        color_id: u32,
    ) -> crate::EagerPlanReductionEntryRow {
        crate::EagerPlanReductionEntryRow {
            kind: EagerPlanReductionEntryKind::SelectorMember,
            owner_id,
            left_id: helicity_id,
            right_id: color_id,
            factor_id: MISSING_U32,
            auxiliary_factor_id: MISSING_U32,
        }
    }

    #[test]
    fn hydrates_groups_from_selector_rows_in_first_seen_order() {
        let mut physics = compact_physics();
        let entries = [selector(7, 0, 0), selector(7, 1, 0), selector(7, 0, 0)];

        hydrate_native_reduction_groups(&mut physics, &[group()], &entries).unwrap();

        let hydrated = &physics.reduction.groups[0];
        assert_eq!(hydrated.id, "reduction:7");
        assert_eq!(hydrated.representative_helicity_id, "helicity:0");
        assert_eq!(hydrated.representative_color_id, "contracted");
        assert_eq!(hydrated.physical_helicity_ids, ["helicity:0", "helicity:1"]);
        assert_eq!(hydrated.physical_color_ids, ["contracted"]);
    }

    #[test]
    fn rejects_descriptor_group_count_mismatch() {
        let mut physics = compact_physics();
        physics
            .extensions
            .get_mut(NATIVE_REDUCTION_GROUPS_EXTENSION_KEY)
            .unwrap()["group_count"] = json!(2);

        let error = hydrate_native_reduction_groups(&mut physics, &[group()], &[]).unwrap_err();

        assert!(error.to_string().contains("declares 2 groups"));
    }

    #[test]
    fn rejects_unknown_descriptor_contract() {
        let mut physics = compact_physics();
        physics
            .extensions
            .get_mut(NATIVE_REDUCTION_GROUPS_EXTENSION_KEY)
            .unwrap()["storage_abi"] = json!("pacbin-v0");

        let error = hydrate_native_reduction_groups(&mut physics, &[group()], &[]).unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("regenerate"));
    }

    #[test]
    fn rejects_selector_owner_and_axis_mismatches() {
        let mut physics = compact_physics();
        let owner_error = hydrate_native_reduction_groups(
            &mut physics,
            &[group()],
            &[selector(6, 0, 0), selector(6, 1, 0), selector(6, 0, 0)],
        )
        .unwrap_err();
        assert!(owner_error.to_string().contains("invalid selector member"));

        let mut physics = compact_physics();
        let axis_error = hydrate_native_reduction_groups(
            &mut physics,
            &[group()],
            &[selector(7, 99, 0), selector(7, 0, 0), selector(7, 0, 0)],
        )
        .unwrap_err();
        assert!(
            axis_error
                .to_string()
                .contains("unknown helicity selector 99")
        );
    }

    #[test]
    fn lc_raw_sum_uses_retained_color_plan_sector_identity() {
        let sectors = BTreeMap::from([(7, 1)]);

        assert_eq!(
            eager_raw_sum_sector_ids("lc", 7, Some(&sectors)).unwrap(),
            vec![1]
        );
        assert!(eager_raw_sum_sector_ids("lc", 8, Some(&sectors)).is_err());
        assert!(
            eager_raw_sum_sector_ids("full", 7, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn compiled_lane_rejects_native_reduction_marker() {
        let physics = compact_physics();

        let error = reject_native_reduction_groups_for_compiled(&physics).unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("require eager plan-v3"));
    }
}
