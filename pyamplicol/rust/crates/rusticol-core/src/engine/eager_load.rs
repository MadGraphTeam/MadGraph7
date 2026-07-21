// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    EagerAttachmentRow, EagerClosureRow, EagerCouplingRow, EagerExecutionPlan,
    EagerFinalizationRow, EagerInvocationRow, EagerPlanPayloads, EagerSelectorDomainIdRow,
    EagerSelectorDomainRow, EagerSelectorGroupRow, EagerSelectorPayloads,
    EagerSelectorStagePayload, EagerStagePayload,
};

pub(super) fn validate_eager_payload_references(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &EagerExecutionManifest,
) -> RusticolResult<()> {
    manifest.validate_header()?;
    validate_capability_list_match(
        &artifact.manifest().runtime.required_runtime_capabilities,
        &manifest.required_runtime_capabilities,
        "outer runtime and eager execution manifest",
    )?;
    validate_eager_stage_contract(manifest)?;
    for (table, expected) in eager_table_contracts(manifest) {
        validate_eager_table_reference(artifact, evaluator_root, table, expected)?;
    }
    let (pack, payload_root) = load_prepared_kernel_pack(artifact, manifest)?;
    validate_prepared_kernel_references(artifact, &payload_root, &pack)?;
    Ok(())
}

pub(super) fn load_eager_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    _evaluator_payloads: &EvaluatorPayloadStore,
    manifest: &EagerExecutionManifest,
    common: &mut ExecutionRuntime,
) -> RusticolResult<EagerNativeRuntime> {
    validate_eager_payload_references(artifact, evaluator_root, manifest)?;
    let (pack, payload_root) = load_prepared_kernel_pack(artifact, manifest)?;
    let kernel_payloads = artifact.evaluator_payload_store(&payload_root)?;
    let coupling_bytes = read_eager_table(
        artifact,
        evaluator_root,
        &manifest.plan.couplings,
        EagerCouplingRow::ENCODED_LEN,
    )?;
    let (parameter_projection, couplings, model_parameter_evaluator) =
        prepare_eager_parameter_state(
            &pack,
            &manifest.runtime_schema.model_parameters,
            &coupling_bytes,
            &kernel_payloads,
        )?;
    common.model_parameter_evaluator = model_parameter_evaluator;
    common.refresh_derived_model_parameters()?;
    let prepared_parameter_count = u32::try_from(parameter_projection.parameter_count)
        .map_err(|_| RusticolError::artifact("prepared parameter count exceeds u32"))?;
    let definition = manifest.plan_definition(&pack, prepared_parameter_count)?;
    let closures = read_eager_table(
        artifact,
        evaluator_root,
        &manifest.plan.closures,
        EagerClosureRow::ENCODED_LEN,
    )?;
    let mut stage_bytes = Vec::with_capacity(manifest.plan.stages.len());
    for stage in &manifest.plan.stages {
        stage_bytes.push((
            read_eager_table(
                artifact,
                evaluator_root,
                &stage.invocations,
                EagerInvocationRow::ENCODED_LEN,
            )?,
            read_eager_table(
                artifact,
                evaluator_root,
                &stage.attachments,
                EagerAttachmentRow::ENCODED_LEN,
            )?,
            read_eager_table(
                artifact,
                evaluator_root,
                &stage.finalizations,
                EagerFinalizationRow::ENCODED_LEN,
            )?,
        ));
    }
    let stages = manifest
        .plan
        .stages
        .iter()
        .zip(&stage_bytes)
        .map(|(stage, bytes)| EagerStagePayload {
            stage_index: stage.stage_index,
            invocations: &bytes.0,
            attachments: &bytes.1,
            finalizations: &bytes.2,
        })
        .collect::<Vec<_>>();
    let selector_bytes = manifest
        .plan
        .selector_closures
        .as_ref()
        .map(|selector| read_eager_selector_tables(artifact, evaluator_root, selector))
        .transpose()?;
    let selector_stage_payloads = selector_bytes
        .as_ref()
        .map(|bytes| {
            manifest
                .plan
                .selector_closures
                .as_ref()
                .expect("selector manifest accompanies selector bytes")
                .stages
                .iter()
                .zip(&bytes.stages)
                .map(|(stage, payloads)| EagerSelectorStagePayload {
                    stage_index: stage.stage_index,
                    invocation_domains: &payloads.invocation_domains,
                    attachment_domains: &payloads.attachment_domains,
                    unpropagated_finalization_domains: &payloads.unpropagated_finalization_domains,
                    propagated_finalization_domains: &payloads.propagated_finalization_domains,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let selector_payloads = selector_bytes.as_ref().map(|bytes| EagerSelectorPayloads {
        domains: &bytes.domains,
        domain_group_ids: &bytes.domain_group_ids,
        stages: &selector_stage_payloads,
        closure_domains: &bytes.closure_domains,
    });
    let plan = EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &couplings,
            stages: &stages,
            closures: &closures,
            selector_domains: selector_payloads,
        },
    )?;
    let options = manifest.runtime_options.validate()?;
    let scheduler = crate::EagerExecutionRuntime::new(plan, options)?;
    let backend = PreparedEvaluatorBackend::load_from_store(&pack, &kernel_payloads)?;
    let (raw_sum_groups, color_contraction) = manifest.raw_reduction_runtime()?;
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
    Ok(EagerNativeRuntime::new(
        scheduler,
        backend,
        pack.backend.clone(),
        parameter_projection,
        raw_sum_groups,
        color_contraction,
    ))
}

#[derive(Clone, Copy, Debug)]
struct RuntimeLogicalParameterSlots {
    real: usize,
    imaginary: Option<usize>,
}

pub(super) fn prepare_eager_parameter_state(
    pack: &PreparedKernelPackManifest,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    coupling_bytes: &[u8],
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<(
    EagerParameterProjection,
    Vec<u8>,
    Option<ModelParameterEvaluatorRuntime>,
)> {
    let logical_slots = eager_runtime_logical_parameter_slots(runtime_parameters)?;
    let canonical_indices = prepared_parameter_indices(pack)?;
    let mut entries = Vec::with_capacity(canonical_indices.len());
    let mut base_parameter_count = 0usize;
    for (name, prepared_index) in &canonical_indices {
        let slots = logical_slots.get(name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared parameter {name:?} is absent from the process runtime schema"
            ))
        })?;
        entries.push(EagerParameterProjectionEntry {
            prepared_index: *prepared_index,
            runtime_real_index: slots.real,
            runtime_imaginary_index: slots.imaginary,
        });
        base_parameter_count = base_parameter_count.max(prepared_index + 1);
    }

    let mut couplings = EagerCouplingRow::decode_table(coupling_bytes)?;
    let mut synthetic_by_runtime_index = BTreeMap::new();
    for row in &mut couplings {
        for parameter_id in [&mut row.real_parameter_id, &mut row.imag_parameter_id] {
            if *parameter_id == crate::MISSING_U32 {
                continue;
            }
            let runtime_index = usize::try_from(*parameter_id).map_err(|_| {
                RusticolError::artifact("eager coupling parameter index does not fit usize")
            })?;
            if runtime_index >= runtime_parameters.len() {
                return Err(RusticolError::integrity(format!(
                    "eager coupling references runtime parameter {runtime_index}, but only {} exist",
                    runtime_parameters.len()
                )));
            }
            let prepared_index = if let Some(index) = synthetic_by_runtime_index.get(&runtime_index)
            {
                *index
            } else {
                let index = base_parameter_count
                    .checked_add(synthetic_by_runtime_index.len())
                    .ok_or_else(|| {
                        RusticolError::artifact("eager synthetic coupling parameters overflow")
                    })?;
                synthetic_by_runtime_index.insert(runtime_index, index);
                entries.push(EagerParameterProjectionEntry {
                    prepared_index: index,
                    runtime_real_index: runtime_index,
                    runtime_imaginary_index: None,
                });
                index
            };
            *parameter_id = u32::try_from(prepared_index).map_err(|_| {
                RusticolError::artifact("eager synthetic coupling parameter exceeds u32")
            })?;
        }
    }
    let parameter_count = base_parameter_count
        .checked_add(synthetic_by_runtime_index.len())
        .ok_or_else(|| RusticolError::artifact("eager prepared parameter count overflows"))?;
    entries.sort_unstable_by_key(|entry| entry.prepared_index);
    let model_parameter_evaluator = load_prepared_model_parameter_evaluator(
        pack,
        runtime_parameters,
        &logical_slots,
        payloads,
    )?;
    Ok((
        EagerParameterProjection {
            parameter_count,
            entries,
        },
        EagerCouplingRow::encode_table(&couplings)?,
        model_parameter_evaluator,
    ))
}

fn eager_runtime_logical_parameter_slots(
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeLogicalParameterSlots>> {
    let mut direct = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in runtime_parameters {
        if let Some(runtime_name) = &parameter.runtime_name {
            let slots = complex.entry(runtime_name.clone()).or_default();
            let target = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                other => {
                    return Err(RusticolError::integrity(format!(
                        "eager runtime parameter {runtime_name:?} has invalid component {other:?}"
                    )));
                }
            };
            if target.replace(parameter.parameter_index).is_some() {
                return Err(RusticolError::integrity(format!(
                    "eager runtime parameter {runtime_name:?} repeats a complex component"
                )));
            }
        } else if direct
            .insert(
                parameter.name.clone(),
                RuntimeLogicalParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "eager runtime parameter {:?} is duplicated",
                parameter.name
            )));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| {
            RusticolError::integrity(format!(
                "eager complex runtime parameter {name:?} lacks a real component"
            ))
        })?;
        if direct
            .insert(
                name.clone(),
                RuntimeLogicalParameterSlots { real, imaginary },
            )
            .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "eager runtime parameter {name:?} has conflicting scalar and complex records"
            )));
        }
    }
    Ok(direct)
}

fn prepared_parameter_indices(
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<BTreeMap<String, usize>> {
    let mut by_name = BTreeMap::new();
    let mut by_index = BTreeMap::new();
    for kernel in &pack.kernels {
        for input in &kernel.input_contracts {
            if input.role != "model-parameter" {
                continue;
            }
            let name = input.model_parameter_name.as_ref().ok_or_else(|| {
                RusticolError::integrity("prepared model-parameter input lacks its name")
            })?;
            let index = usize::try_from(input.model_parameter_index.ok_or_else(|| {
                RusticolError::integrity("prepared model-parameter input lacks its stable index")
            })?)
            .map_err(|_| RusticolError::artifact("prepared parameter index does not fit usize"))?;
            if by_name
                .insert(name.clone(), index)
                .is_some_and(|old| old != index)
            {
                return Err(RusticolError::integrity(format!(
                    "prepared parameter {name:?} has conflicting stable indices"
                )));
            }
            if by_index
                .insert(index, name.clone())
                .is_some_and(|old| old != *name)
            {
                return Err(RusticolError::integrity(format!(
                    "prepared parameter index {index} names multiple parameters"
                )));
            }
        }
    }
    Ok(by_name)
}

fn load_prepared_model_parameter_evaluator(
    pack: &PreparedKernelPackManifest,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    logical_slots: &BTreeMap<String, RuntimeLogicalParameterSlots>,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<Option<ModelParameterEvaluatorRuntime>> {
    let mut kernels = pack
        .kernels
        .iter()
        .filter(|kernel| kernel.contract_kind == "model-parameter");
    let kernel = kernels.next();
    if kernels.next().is_some() {
        return Err(RusticolError::integrity(
            "prepared kernel pack contains multiple model-parameter kernels",
        ));
    }
    let derived_names = runtime_parameters
        .iter()
        .filter(|parameter| parameter.kind == "derived_parameter_component")
        .filter_map(|parameter| parameter.runtime_name.clone())
        .collect::<BTreeSet<_>>();
    let Some(kernel) = kernel else {
        if derived_names.is_empty() {
            return Ok(None);
        }
        return Err(RusticolError::integrity(
            "eager runtime schema contains derived parameters but its prepared pack has no derivation kernel",
        ));
    };

    let mut input_parameter_indices = Vec::with_capacity(kernel.input_contracts.len());
    for input in &kernel.input_contracts {
        let name = input.model_parameter_name.as_ref().ok_or_else(|| {
            RusticolError::integrity("prepared derivation input lacks its parameter name")
        })?;
        let slots = logical_slots.get(name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared derivation input {name:?} is absent from the runtime schema"
            ))
        })?;
        if slots.imaginary.is_some() {
            return Err(RusticolError::compatibility(format!(
                "prepared derivation input {name:?} is complex, but runtime schema v3 derivation inputs are scalar"
            )));
        }
        input_parameter_indices.push(slots.real);
    }

    let output_index_by_name = kernel
        .output_layout
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value.strip_prefix("model-parameter:").ok_or_else(|| {
                RusticolError::integrity(format!(
                    "prepared derivation output layout {value:?} is invalid"
                ))
            })?;
            Ok((name.to_string(), index))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut outputs = Vec::with_capacity(derived_names.len());
    for name in derived_names {
        let slots = logical_slots.get(&name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "derived runtime parameter {name:?} has no logical slots"
            ))
        })?;
        let imaginary = slots.imaginary.ok_or_else(|| {
            RusticolError::integrity(format!(
                "derived runtime parameter {name:?} lacks an imaginary component"
            ))
        })?;
        let output_index = *output_index_by_name.get(&name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared derivation kernel does not output runtime parameter {name:?}"
            ))
        })?;
        outputs.push(GenericDerivedParameterOutputManifest {
            runtime_name: name,
            output_index,
            real_parameter_index: slots.real,
            imag_parameter_index: imaginary,
        });
    }
    let evaluator_manifest = kernel.runtime_evaluator_manifest()?;
    let evaluator = EvaluatorGroup::load_from_store(&evaluator_manifest, payloads)?;
    Ok(Some(ModelParameterEvaluatorRuntime {
        input_parameter_indices,
        outputs,
        evaluator,
    }))
}

fn validate_eager_stage_contract(manifest: &EagerExecutionManifest) -> RusticolResult<()> {
    if manifest.plan.stages.len() != manifest.runtime_schema.stages.len() {
        return Err(RusticolError::integrity(
            "eager binary stage count does not match runtime_schema",
        ));
    }
    for (table, schema) in manifest
        .plan
        .stages
        .iter()
        .zip(&manifest.runtime_schema.stages)
    {
        if usize::try_from(table.stage_index).ok() != Some(schema.stage_index)
            || table.subset_size != schema.subset_size
        {
            return Err(RusticolError::integrity(format!(
                "eager stage {} does not match runtime_schema stage {}",
                table.stage_index, schema.stage_index
            )));
        }
    }
    Ok(())
}

struct EagerSelectorTableBytes {
    domains: Vec<u8>,
    domain_group_ids: Vec<u8>,
    stages: Vec<EagerSelectorStageTableBytes>,
    closure_domains: Vec<u8>,
}

struct EagerSelectorStageTableBytes {
    invocation_domains: Vec<u8>,
    attachment_domains: Vec<u8>,
    unpropagated_finalization_domains: Vec<u8>,
    propagated_finalization_domains: Vec<u8>,
}

fn read_eager_selector_tables(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    selector: &EagerSelectorDomainsManifest,
) -> RusticolResult<EagerSelectorTableBytes> {
    let mut stages = Vec::with_capacity(selector.stages.len());
    for stage in &selector.stages {
        stages.push(EagerSelectorStageTableBytes {
            invocation_domains: read_eager_table(
                artifact,
                evaluator_root,
                &stage.invocation_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            )?,
            attachment_domains: read_eager_table(
                artifact,
                evaluator_root,
                &stage.attachment_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            )?,
            unpropagated_finalization_domains: read_eager_table(
                artifact,
                evaluator_root,
                &stage.unpropagated_finalization_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            )?,
            propagated_finalization_domains: read_eager_table(
                artifact,
                evaluator_root,
                &stage.propagated_finalization_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            )?,
        });
    }
    Ok(EagerSelectorTableBytes {
        domains: read_eager_table(
            artifact,
            evaluator_root,
            &selector.domains,
            EagerSelectorDomainRow::ENCODED_LEN,
        )?,
        domain_group_ids: read_eager_table(
            artifact,
            evaluator_root,
            &selector.domain_group_ids,
            EagerSelectorGroupRow::ENCODED_LEN,
        )?,
        stages,
        closure_domains: read_eager_table(
            artifact,
            evaluator_root,
            &selector.closure_domains,
            EagerSelectorDomainIdRow::ENCODED_LEN,
        )?,
    })
}

fn eager_table_contracts(manifest: &EagerExecutionManifest) -> Vec<(&EagerTableManifest, usize)> {
    let selector_table_count = manifest
        .plan
        .selector_closures
        .as_ref()
        .map_or(0, |selector| 3 + selector.stages.len() * 4);
    let mut result = Vec::with_capacity(2 + manifest.plan.stages.len() * 3 + selector_table_count);
    result.push((&manifest.plan.couplings, EagerCouplingRow::ENCODED_LEN));
    for stage in &manifest.plan.stages {
        result.push((&stage.invocations, EagerInvocationRow::ENCODED_LEN));
        result.push((&stage.attachments, EagerAttachmentRow::ENCODED_LEN));
        result.push((&stage.finalizations, EagerFinalizationRow::ENCODED_LEN));
    }
    result.push((&manifest.plan.closures, EagerClosureRow::ENCODED_LEN));
    if let Some(selector) = &manifest.plan.selector_closures {
        result.push((&selector.domains, EagerSelectorDomainRow::ENCODED_LEN));
        result.push((
            &selector.domain_group_ids,
            EagerSelectorGroupRow::ENCODED_LEN,
        ));
        for stage in &selector.stages {
            result.push((
                &stage.invocation_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            ));
            result.push((
                &stage.attachment_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            ));
            result.push((
                &stage.unpropagated_finalization_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            ));
            result.push((
                &stage.propagated_finalization_domains,
                EagerSelectorDomainIdRow::ENCODED_LEN,
            ));
        }
        result.push((
            &selector.closure_domains,
            EagerSelectorDomainIdRow::ENCODED_LEN,
        ));
    }
    result
}

fn validate_eager_table_reference(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    table: &EagerTableManifest,
    expected_row_size: usize,
) -> RusticolResult<()> {
    if table.row_size != expected_row_size {
        return Err(RusticolError::compatibility(format!(
            "eager table {:?} row size {} does not match runtime size {expected_row_size}",
            table.path, table.row_size
        )));
    }
    let path = evaluator_relative_payload_path(artifact, evaluator_root, &table.path)?;
    let payload = artifact.payload(&path)?;
    if payload.role != PayloadRole::EvaluatorState {
        return Err(RusticolError::security(format!(
            "eager table {path:?} has payload role {:?}, expected evaluator-state",
            payload.role
        )));
    }
    let expected_size = table
        .count
        .checked_mul(table.row_size)
        .ok_or_else(|| RusticolError::artifact("eager table byte length overflows usize"))?;
    if usize::try_from(payload.size_bytes).ok() != Some(expected_size) {
        return Err(RusticolError::integrity(format!(
            "eager table {path:?} declares {} rows but has {} bytes",
            table.count, payload.size_bytes
        )));
    }
    artifact.payload_path(&path)?;
    Ok(())
}

fn read_eager_table(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    table: &EagerTableManifest,
    expected_row_size: usize,
) -> RusticolResult<Vec<u8>> {
    validate_eager_table_reference(artifact, evaluator_root, table, expected_row_size)?;
    let path = evaluator_relative_payload_path(artifact, evaluator_root, &table.path)?;
    artifact.read_payload(&path)
}

fn evaluator_relative_payload_path(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    value: &str,
) -> RusticolResult<String> {
    let relative_root = evaluator_root.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("eager evaluator root escapes the verified artifact")
    })?;
    let confined = confined_internal_path(value, "eager table path")?;
    relative_root
        .join(confined)
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| RusticolError::security("eager table path is not valid UTF-8"))
}

fn load_prepared_kernel_pack(
    artifact: &VerifiedArtifact,
    manifest: &EagerExecutionManifest,
) -> RusticolResult<(PreparedKernelPackManifest, PathBuf)> {
    let manifest_path = confined_internal_path(
        &manifest.kernel_pack.manifest_path,
        "prepared kernel-pack manifest path",
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
        "prepared kernel payload root",
    )?;
    Ok((pack, artifact.root().join(payload_root)))
}

pub(super) fn validate_prepared_kernel_references(
    artifact: &VerifiedArtifact,
    payload_root: &Path,
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<()> {
    let relative_root = payload_root.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("prepared kernel payload root escapes the verified artifact")
    })?;
    for kernel in &pack.kernels {
        validate_evaluator_state_path(artifact, relative_root, &kernel.exact_evaluator_state_path)?;
        let evaluator = kernel.runtime_evaluator_manifest()?;
        validate_evaluator_reference(artifact, relative_root, &evaluator)?;
        for path in kernel.extra_evaluator_payload_paths()? {
            validate_evaluator_state_path(artifact, relative_root, path)?;
        }
    }
    for variant in &pack.kernel_variants {
        let evaluator = variant.runtime_evaluator_manifest()?;
        validate_evaluator_reference(artifact, relative_root, &evaluator)?;
        for path in variant.extra_evaluator_payload_paths()? {
            validate_evaluator_state_path(artifact, relative_root, path)?;
        }
    }
    Ok(())
}
