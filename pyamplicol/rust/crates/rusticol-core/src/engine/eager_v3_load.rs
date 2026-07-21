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
    physics: ProcessPhysicsV1,
) -> RusticolResult<LoadedEagerV3Runtime> {
    let pack = load_eager_v3_prepared_pack(artifact, manifest)?;
    let container = open_verified_eager_v3_runtime_container(artifact, evaluator_root, manifest)?;
    let decoded =
        super::eager_v3_decode::decode_eager_v3_runtime(&container, manifest, &pack.manifest)?;
    let mut common =
        super::eager_v3_common::build_eager_v3_common_runtime(&decoded, manifest, physics)?;

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
        groups.push(RawSumGroup {
            id: i64::from(group.coherent_group_id),
            indices: amplitudes
                .iter()
                .map(|entry| entry.left_id as usize)
                .collect(),
            weight,
            all_sector_weight,
            sector_ids: selectors
                .iter()
                .filter_map(|entry| {
                    (entry.right_id != MISSING_U32).then_some(i64::from(entry.right_id))
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
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
    Ok((
        groups,
        Some(ColorContractionRuntime {
            group_count: group_index_by_id.len(),
            entries,
            group_scratch_f64: Vec::new(),
        }),
    ))
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
