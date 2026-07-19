// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    EagerAttachmentRow, EagerClosureRow, EagerCouplingRow, EagerExecutionPlan,
    EagerFinalizationRow, EagerInvocationRow, EagerPlanPayloads, EagerStagePayload,
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
    manifest: &EagerExecutionManifest,
) -> RusticolResult<EagerNativeRuntime> {
    validate_eager_payload_references(artifact, evaluator_root, manifest)?;
    let (pack, payload_root) = load_prepared_kernel_pack(artifact, manifest)?;
    let definition = manifest.plan_definition(&pack)?;
    let couplings = read_eager_table(
        artifact,
        evaluator_root,
        &manifest.plan.couplings,
        EagerCouplingRow::ENCODED_LEN,
    )?;
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
    let plan = EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &couplings,
            stages: &stages,
            closures: &closures,
        },
    )?;
    let options = manifest.runtime_options.validate()?;
    let scheduler = crate::EagerExecutionRuntime::new(plan, options)?;
    let backend = PreparedEvaluatorBackend::load(&pack, &payload_root)?;
    let has_derived_parameter_kernel = pack
        .kernels
        .iter()
        .any(|kernel| kernel.contract_kind == "model-parameter");
    Ok(EagerNativeRuntime::new(
        scheduler,
        backend,
        pack.backend.clone(),
        has_derived_parameter_kernel,
    ))
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

fn eager_table_contracts(manifest: &EagerExecutionManifest) -> Vec<(&EagerTableManifest, usize)> {
    let mut result = Vec::with_capacity(2 + manifest.plan.stages.len() * 3);
    result.push((&manifest.plan.couplings, EagerCouplingRow::ENCODED_LEN));
    for stage in &manifest.plan.stages {
        result.push((&stage.invocations, EagerInvocationRow::ENCODED_LEN));
        result.push((&stage.attachments, EagerAttachmentRow::ENCODED_LEN));
        result.push((&stage.finalizations, EagerFinalizationRow::ENCODED_LEN));
    }
    result.push((&manifest.plan.closures, EagerClosureRow::ENCODED_LEN));
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

fn validate_prepared_kernel_references(
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
    Ok(())
}
