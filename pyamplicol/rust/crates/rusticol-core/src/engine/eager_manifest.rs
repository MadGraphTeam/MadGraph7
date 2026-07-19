// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    EAGER_KERNEL_ABI, EAGER_PLAN_ABI, EagerDirectClosureSpec, EagerKernelInput, EagerKernelRole,
    EagerKernelSpec, EagerPlanDefinition, EagerPlanDimensions, EagerReductionEntry,
    EagerReductionGroup,
};

pub(super) const EAGER_EXECUTION_KIND: &str = "pyamplicol-runtime-eager-execution";
pub(super) const MAX_EAGER_POINT_TILE_SIZE: usize = 1_048_576;
pub(super) const MAX_EAGER_WORKSPACE_MIB: usize = 4096;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    #[serde(default)]
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) eager_plan_abi: String,
    pub(super) kernel_pack: EagerKernelPackReferenceManifest,
    pub(super) runtime_options: EagerRuntimeOptionsManifest,
    pub(super) plan: EagerPlanManifest,
    pub(super) dag_summary: ExecutionSummary,
    pub(super) runtime_schema: ExecutionPlan,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerKernelPackReferenceManifest {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerRuntimeOptionsManifest {
    pub(super) point_tile_size: usize,
    pub(super) workspace_mib: usize,
}

impl EagerRuntimeOptionsManifest {
    pub(super) fn validate(self) -> RusticolResult<crate::EagerRuntimeOptions> {
        if self.point_tile_size == 0 || self.point_tile_size > MAX_EAGER_POINT_TILE_SIZE {
            return Err(RusticolError::artifact(format!(
                "eager point_tile_size must be in 1..={MAX_EAGER_POINT_TILE_SIZE}"
            )));
        }
        if self.workspace_mib == 0 || self.workspace_mib > MAX_EAGER_WORKSPACE_MIB {
            return Err(RusticolError::artifact(format!(
                "eager workspace_mib must be in 1..={MAX_EAGER_WORKSPACE_MIB}"
            )));
        }
        crate::EagerRuntimeOptions::from_mib(self.point_tile_size, self.workspace_mib)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerPlanManifest {
    pub(super) kind: String,
    pub(super) eager_plan_abi: String,
    #[serde(default)]
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process_key: String,
    pub(super) couplings: EagerTableManifest,
    pub(super) stages: Vec<EagerStageTablesManifest>,
    pub(super) closures: EagerTableManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerStageTablesManifest {
    pub(super) stage_index: u32,
    pub(super) subset_size: usize,
    pub(super) invocations: EagerTableManifest,
    pub(super) attachments: EagerTableManifest,
    pub(super) finalizations: EagerTableManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerTableManifest {
    pub(super) path: String,
    pub(super) count: usize,
    pub(super) row_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelPackManifest {
    pub(super) eager_kernel_abi: String,
    pub(super) backend: String,
    pub(super) optimization_settings: Value,
    pub(super) producer: Value,
    pub(super) dependency_abis: Value,
    pub(super) provenance: Value,
    pub(super) target: PreparedKernelTargetManifest,
    pub(super) resolver_manifest: Value,
    pub(super) kernels: Vec<PreparedKernelManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelTargetManifest {
    pub(super) portable: bool,
    pub(super) word_bits: u8,
    pub(super) endianness: String,
    pub(super) target_triple: String,
    #[serde(default)]
    pub(super) cpu_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelManifest {
    pub(super) kernel_id: u32,
    pub(super) contract_kind: String,
    pub(super) canonical_signature: String,
    pub(super) input_arity: usize,
    pub(super) output_arity: u32,
    pub(super) input_layout: Vec<String>,
    pub(super) input_contracts: Vec<PreparedKernelInputManifest>,
    pub(super) output_layout: Vec<String>,
    pub(super) exact_expressions: Vec<String>,
    pub(super) exact_evaluator_state_path: String,
    pub(super) f64_evaluator_manifest: Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparedKernelInputManifest {
    pub(super) role: String,
    pub(super) component: u32,
    pub(super) symbol: String,
    pub(super) model_parameter_name: Option<String>,
    pub(super) model_parameter_index: Option<u32>,
}

impl EagerExecutionManifest {
    pub(super) fn validate_header(&self) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != EAGER_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.eager_plan_abi != EAGER_PLAN_ABI
            || self.plan.eager_plan_abi != EAGER_PLAN_ABI
            || self.plan.kind != EAGER_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager plan ABI {:?}",
                self.eager_plan_abi
            )));
        }
        if self.plan.process_key != self.key {
            return Err(RusticolError::integrity(
                "eager plan process key does not match its execution manifest",
            ));
        }
        validate_capability_list_match(
            &self.required_runtime_capabilities,
            &self.plan.required_runtime_capabilities,
            "eager execution and plan",
        )?;
        if self.required_runtime_capabilities != [EAGER_DAG_RUNTIME_CAPABILITY.to_string()] {
            return Err(RusticolError::compatibility(format!(
                "eager execution requires exactly {EAGER_DAG_RUNTIME_CAPABILITY:?}"
            )));
        }
        self.runtime_options.validate()?;
        Ok(())
    }

    pub(super) fn compiled_metadata_manifest(&self) -> ExecutionManifest {
        ExecutionManifest {
            schema_version: self.schema_version,
            kind: "pyamplicol-runtime-execution".to_string(),
            required_runtime_capabilities: Vec::new(),
            process: self.process.clone(),
            key: self.key.clone(),
            color_accuracy: self.color_accuracy.clone(),
            external_pdg_order: self.external_pdg_order.clone(),
            compiled: EvaluatorSetManifest {
                kind: "eager-runtime-metadata".to_string(),
                runtime_available: true,
                runtime_unavailable_message: None,
                lc_topology_replay: None,
                model_parameter_evaluator: None,
                stage_evaluators: None,
            },
            dag_summary: self.dag_summary.clone(),
            runtime_schema: self.runtime_schema.clone(),
        }
    }

    pub(super) fn plan_definition(
        &self,
        pack: &PreparedKernelPackManifest,
        prepared_parameter_count: u32,
    ) -> RusticolResult<EagerPlanDefinition> {
        let dimensions = EagerPlanDimensions {
            value_slot_component_counts: contiguous_value_slot_widths(&self.runtime_schema)?,
            momentum_slot_component_counts: contiguous_momentum_slot_widths(&self.runtime_schema)?,
            current_component_counts: contiguous_current_slot_widths(&self.runtime_schema)?,
            parameter_count: prepared_parameter_count,
            amplitude_count: u32::try_from(self.runtime_schema.amplitude_stage.output_count)
                .map_err(|_| RusticolError::artifact("eager amplitude count exceeds u32"))?,
        };
        let kernels = pack.kernel_specs()?;
        let direct_closures = self.direct_closure_specs()?;
        let (reduction_groups, reduction_entries) = self.reduction_plan()?;
        Ok(EagerPlanDefinition {
            dimensions,
            kernels,
            direct_closures,
            reduction_groups,
            reduction_entries,
        })
    }

    fn direct_closure_specs(&self) -> RusticolResult<Vec<EagerDirectClosureSpec>> {
        self.runtime_schema
            .amplitude_stage
            .roots
            .iter()
            .enumerate()
            .filter(|(_, root)| root.kind == "direct-contraction")
            .map(|(index, root)| {
                let coefficients = root
                    .contraction_ir
                    .coefficients
                    .iter()
                    .map(|value| crate::EagerComplex64::new(value[0], value[1]))
                    .collect();
                Ok(EagerDirectClosureSpec {
                    closure_index: u32::try_from(index).map_err(|_| {
                        RusticolError::artifact("eager direct closure index exceeds u32")
                    })?,
                    coefficients,
                })
            })
            .collect()
    }

    fn reduction_plan(
        &self,
    ) -> RusticolResult<(Vec<EagerReductionGroup>, Vec<EagerReductionEntry>)> {
        let (groups, contraction) = self.raw_reduction_runtime()?;
        let reduction_groups = groups
            .iter()
            .map(|group| {
                Ok(EagerReductionGroup {
                    amplitude_indices: group
                        .indices
                        .iter()
                        .map(|index| {
                            u32::try_from(*index).map_err(|_| {
                                RusticolError::artifact(
                                    "eager reduction amplitude index exceeds u32",
                                )
                            })
                        })
                        .collect::<RusticolResult<Vec<_>>>()?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut entries = Vec::new();
        if let Some(contraction) = contraction {
            entries
                .try_reserve_exact(contraction.entries.len())
                .map_err(|error| {
                    RusticolError::artifact(format!(
                        "cannot reserve eager color-contraction entries: {error}"
                    ))
                })?;
            for entry in contraction.entries {
                let coefficient = crate::EagerComplex64::new(entry.weight_re, entry.weight_im)
                    * entry.symmetry_factor;
                entries.push(EagerReductionEntry {
                    left_group_index: u32::try_from(entry.left_group_index).map_err(|_| {
                        RusticolError::artifact(
                            "eager color-contraction left group index exceeds u32",
                        )
                    })?,
                    right_group_index: u32::try_from(entry.right_group_index).map_err(|_| {
                        RusticolError::artifact(
                            "eager color-contraction right group index exceeds u32",
                        )
                    })?,
                    coefficient,
                });
            }
        } else {
            entries.try_reserve_exact(groups.len()).map_err(|error| {
                RusticolError::artifact(format!(
                    "cannot reserve eager diagonal reduction entries: {error}"
                ))
            })?;
            for (group_index, group) in groups.iter().enumerate() {
                let group_index = u32::try_from(group_index)
                    .map_err(|_| RusticolError::artifact("eager reduction group exceeds u32"))?;
                entries.push(EagerReductionEntry {
                    left_group_index: group_index,
                    right_group_index: group_index,
                    coefficient: crate::EagerComplex64::new(group.all_sector_weight, 0.0),
                });
            }
        }
        Ok((reduction_groups, entries))
    }

    pub(super) fn raw_reduction_runtime(
        &self,
    ) -> RusticolResult<(Vec<RawSumGroup>, Option<ColorContractionRuntime>)> {
        let roots = &self.runtime_schema.amplitude_stage.roots;
        let output_count = self.runtime_schema.amplitude_stage.output_count;
        let weights = roots
            .iter()
            .map(|root| root.helicity_weight)
            .collect::<Vec<_>>();
        let all_sector_weights = roots
            .iter()
            .map(|root| root.all_sector_weight.unwrap_or(root.helicity_weight))
            .collect::<Vec<_>>();
        let color_sector_ids = roots
            .iter()
            .map(|root| root.color_sector_id)
            .collect::<Vec<_>>();
        let group_ids = roots
            .iter()
            .map(generic_root_group_id)
            .collect::<RusticolResult<Vec<_>>>()?;
        let groups = build_raw_sum_groups(
            output_count,
            &weights,
            &all_sector_weights,
            &group_ids,
            &color_sector_ids,
        )?;
        let contraction = build_color_contraction_runtime(
            self.runtime_schema
                .amplitude_stage
                .color_contraction
                .as_ref(),
            &groups,
        )?;
        Ok((groups, contraction))
    }
}

impl PreparedKernelPackManifest {
    pub(super) fn validate(&self) -> RusticolResult<()> {
        if self.eager_kernel_abi != EAGER_KERNEL_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager kernel ABI {:?}",
                self.eager_kernel_abi
            )));
        }
        if !matches!(self.backend.as_str(), "jit" | "asm" | "cpp") {
            return Err(RusticolError::artifact(format!(
                "unsupported prepared backend {:?}",
                self.backend
            )));
        }
        for (name, value) in [
            ("optimization_settings", &self.optimization_settings),
            ("producer", &self.producer),
            ("dependency_abis", &self.dependency_abis),
            ("provenance", &self.provenance),
            ("resolver_manifest", &self.resolver_manifest),
        ] {
            if value.as_object().is_none_or(|mapping| mapping.is_empty()) {
                return Err(RusticolError::artifact(format!(
                    "prepared kernel pack {name} must be a nonempty object"
                )));
            }
        }
        if self.resolver_manifest.get("abi").and_then(Value::as_str)
            != Some("pyamplicol-prepared-kernel-catalog-v1")
        {
            return Err(RusticolError::compatibility(
                "prepared kernel resolver manifest has an unsupported ABI",
            ));
        }
        if self.kernels.is_empty() {
            return Err(RusticolError::artifact("prepared kernel pack is empty"));
        }
        if self.target.word_bits != 64 || self.target.endianness != "little" {
            return Err(RusticolError::compatibility(
                "prepared eager kernels require a 64-bit little-endian target",
            ));
        }
        if self
            .target
            .cpu_features
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RusticolError::integrity(
                "prepared target CPU features must be sorted and unique",
            ));
        }
        if self.backend == "jit" {
            if !self.target.portable
                || self.target.target_triple != "portable-symjit-mir"
                || !self.target.cpu_features.is_empty()
            {
                return Err(RusticolError::compatibility(
                    "prepared JIT kernels must use the portable SymJIT MIR target contract",
                ));
            }
        } else {
            let host = crate::runtime_target_info();
            let host_features = host.cpu_features.into_iter().collect::<BTreeSet<_>>();
            if self.target.portable
                || self.target.target_triple != host.triple
                || self
                    .target
                    .cpu_features
                    .iter()
                    .any(|feature| !host_features.contains(feature))
            {
                return Err(RusticolError::compatibility(format!(
                    "prepared {} kernels target {:?} with features {:?}, incompatible with host {:?}",
                    self.backend, self.target.target_triple, self.target.cpu_features, host.triple,
                )));
            }
        }
        let mut ids = BTreeSet::new();
        let mut signatures = BTreeSet::new();
        for kernel in &self.kernels {
            if !ids.insert(kernel.kernel_id) {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel pack repeats kernel id {}",
                    kernel.kernel_id
                )));
            }
            if kernel.canonical_signature.is_empty()
                || !signatures.insert(kernel.canonical_signature.as_str())
            {
                return Err(RusticolError::integrity(
                    "prepared kernel signatures must be nonempty and unique",
                ));
            }
            if kernel.input_arity != kernel.input_contracts.len()
                || kernel.input_arity != kernel.input_layout.len()
                || usize::try_from(kernel.output_arity).ok() != Some(kernel.output_layout.len())
                || usize::try_from(kernel.output_arity).ok() != Some(kernel.exact_expressions.len())
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} has inconsistent input/output arities",
                    kernel.kernel_id
                )));
            }
            if kernel.exact_evaluator_state_path.is_empty() {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} lacks exact evaluator state",
                    kernel.kernel_id
                )));
            }
            kernel.validate_evaluator_metadata(self)?;
        }
        Ok(())
    }

    pub(super) fn kernel_specs(&self) -> RusticolResult<Vec<EagerKernelSpec>> {
        self.kernels
            .iter()
            .filter(|kernel| kernel.contract_kind != "model-parameter")
            .map(|kernel| {
                let role = match kernel.contract_kind.as_str() {
                    "vertex" => EagerKernelRole::Vertex,
                    "propagator" => EagerKernelRole::Finalization,
                    "closure" => EagerKernelRole::Closure,
                    other => {
                        return Err(RusticolError::artifact(format!(
                            "unsupported prepared kernel contract kind {other:?}"
                        )));
                    }
                };
                let inputs = kernel
                    .input_contracts
                    .iter()
                    .map(PreparedKernelInputManifest::to_eager_input)
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(EagerKernelSpec {
                    kernel_id: kernel.kernel_id,
                    role,
                    inputs,
                    output_component_count: kernel.output_arity,
                })
            })
            .collect()
    }
}

impl PreparedKernelManifest {
    pub(super) fn runtime_evaluator_manifest(&self) -> RusticolResult<EvaluatorManifest> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator manifest must be an object",
                self.kernel_id
            ))
        })?;
        let kind = object.get("kind").and_then(Value::as_str).ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator kind must be a nonempty string",
                self.kernel_id
            ))
        })?;
        let metadata_fields: &[&str] = match kind {
            "symjit-application-evaluator" => &["backend", "label", "settings", "build_timing"],
            "compiled-complex-evaluator" => &["backend", "settings", "source_path", "build_timing"],
            other => {
                return Err(RusticolError::compatibility(format!(
                    "prepared kernel {} has unsupported f64 evaluator kind {other:?}",
                    self.kernel_id
                )));
            }
        };
        validate_prepared_evaluator_keys(self.kernel_id, kind, object)?;
        let mut runtime = object.clone();
        for field in metadata_fields {
            runtime.remove(*field);
        }
        serde_json::from_value(Value::Object(runtime)).map_err(|error| {
            RusticolError::serialization(format!(
                "prepared kernel {} has invalid runtime evaluator metadata: {error}",
                self.kernel_id
            ))
        })
    }

    pub(super) fn extra_evaluator_payload_paths(&self) -> RusticolResult<Vec<&str>> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact("prepared f64 evaluator manifest must be an object")
        })?;
        match object.get("kind").and_then(Value::as_str) {
            Some("symjit-application-evaluator") => Ok(Vec::new()),
            Some("compiled-complex-evaluator") => Ok(vec![required_nonempty_string(
                object,
                "source_path",
                self.kernel_id,
            )?]),
            _ => Err(RusticolError::compatibility(format!(
                "prepared kernel {} has an unsupported f64 evaluator kind",
                self.kernel_id
            ))),
        }
    }

    fn validate_evaluator_metadata(&self, pack: &PreparedKernelPackManifest) -> RusticolResult<()> {
        let object = self.f64_evaluator_manifest.as_object().ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} f64 evaluator manifest must be an object",
                self.kernel_id
            ))
        })?;
        let kind = required_nonempty_string(object, "kind", self.kernel_id)?;
        let backend = required_nonempty_string(object, "backend", self.kernel_id)?;
        let (expected_kind, expected_backend, expected_capability) = match pack.backend.as_str() {
            "jit" => (
                "symjit-application-evaluator",
                "jit",
                SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            ),
            "asm" => (
                "compiled-complex-evaluator",
                "compiled-complex",
                SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            ),
            "cpp" => (
                "compiled-complex-evaluator",
                "compiled-complex",
                SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            ),
            _ => unreachable!("prepared pack backend validated before its kernels"),
        };
        if kind != expected_kind || backend != expected_backend {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator ({kind:?}, {backend:?}) does not match pack backend {:?}, expected ({expected_kind:?}, {expected_backend:?})",
                self.kernel_id, pack.backend,
            )));
        }
        let settings = object.get("settings").ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {} evaluator lacks settings metadata",
                self.kernel_id
            ))
        })?;
        if settings != &pack.optimization_settings {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator settings do not match its pack",
                self.kernel_id
            )));
        }
        let build_timing = object
            .get("build_timing")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "prepared kernel {} evaluator build_timing must be an object",
                    self.kernel_id
                ))
            })?;
        if build_timing.values().any(|value| {
            value
                .as_f64()
                .is_none_or(|seconds| !seconds.is_finite() || seconds < 0.0)
        }) {
            return Err(RusticolError::artifact(format!(
                "prepared kernel {} evaluator build timings must be finite nonnegative numbers",
                self.kernel_id
            )));
        }
        let runtime = self.runtime_evaluator_manifest()?;
        let capabilities = evaluator_runtime_capabilities(&runtime)?;
        if capabilities != BTreeSet::from([expected_capability.to_string()]) {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} pack backend {:?} declares evaluator capabilities {capabilities:?}, expected {expected_capability:?}",
                self.kernel_id, pack.backend,
            )));
        }
        let (input_len, output_len) = runtime.io_len()?;
        if input_len != self.input_arity
            || output_len != usize::try_from(self.output_arity).unwrap_or(usize::MAX)
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} evaluator I/O ({input_len}, {output_len}) does not match ({}, {})",
                self.kernel_id, self.input_arity, self.output_arity
            )));
        }
        match kind {
            "symjit-application-evaluator" => {
                if pack.backend != "jit" {
                    return Err(RusticolError::integrity(format!(
                        "prepared kernel {} uses SymJIT under backend {:?}",
                        self.kernel_id, pack.backend
                    )));
                }
                if required_nonempty_string(object, "label", self.kernel_id)?.is_empty() {
                    return Err(RusticolError::artifact("prepared evaluator label is empty"));
                }
                let application_abi =
                    required_nonempty_string(object, "application_abi", self.kernel_id)?;
                if pack
                    .dependency_abis
                    .get("symjit_application")
                    .and_then(Value::as_str)
                    != Some(application_abi)
                {
                    return Err(RusticolError::compatibility(format!(
                        "prepared kernel {} SymJIT application ABI does not match its pack",
                        self.kernel_id
                    )));
                }
            }
            "compiled-complex-evaluator" => {
                if !matches!(pack.backend.as_str(), "asm" | "cpp") {
                    return Err(RusticolError::integrity(format!(
                        "prepared kernel {} uses a compiled evaluator under backend {:?}",
                        self.kernel_id, pack.backend
                    )));
                }
                required_nonempty_string(object, "source_path", self.kernel_id)?;
            }
            _ => unreachable!("runtime evaluator projection validated the kind"),
        }
        if object.get("evaluator_state_path").and_then(Value::as_str)
            != Some(self.exact_evaluator_state_path.as_str())
        {
            return Err(RusticolError::integrity(format!(
                "prepared kernel {} exact evaluator state does not match its f64 evaluator fallback",
                self.kernel_id
            )));
        }
        Ok(())
    }
}

fn validate_prepared_evaluator_keys(
    kernel_id: u32,
    kind: &str,
    object: &serde_json::Map<String, Value>,
) -> RusticolResult<()> {
    let expected = match kind {
        "symjit-application-evaluator" => [
            "application_abi",
            "application_path",
            "backend",
            "batch_layout",
            "build_timing",
            "compiler_type",
            "element_layout",
            "endianness",
            "evaluator_state_path",
            "evaluator_state_runtime_capability",
            "input_len",
            "kind",
            "label",
            "optimization_level",
            "output_len",
            "required_defuns",
            "runtime_capability",
            "settings",
            "translation_mode",
            "word_bits",
        ]
        .as_slice(),
        "compiled-complex-evaluator" => [
            "backend",
            "build_timing",
            "evaluator_state_path",
            "function_name",
            "input_len",
            "kind",
            "library_path",
            "number_type",
            "output_len",
            "runtime_capability",
            "settings",
            "source_path",
        ]
        .as_slice(),
        _ => {
            return Err(RusticolError::compatibility(format!(
                "prepared kernel {kernel_id} has unsupported evaluator kind {kind:?}"
            )));
        }
    };
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(RusticolError::artifact(format!(
            "prepared kernel {kernel_id} evaluator fields {actual:?} do not match {expected:?}"
        )));
    }
    Ok(())
}

fn required_nonempty_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    kernel_id: u32,
) -> RusticolResult<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "prepared kernel {kernel_id} evaluator field {field:?} must be a nonempty string"
            ))
        })
}

impl PreparedKernelInputManifest {
    fn to_eager_input(&self) -> RusticolResult<EagerKernelInput> {
        if self.symbol.is_empty() {
            return Err(RusticolError::integrity(
                "prepared kernel input symbol must be nonempty",
            ));
        }
        let input = match self.role.as_str() {
            "left-current" => EagerKernelInput::FirstCurrentComponent(self.component),
            "right-current" => EagerKernelInput::SecondCurrentComponent(self.component),
            "left-momentum" => EagerKernelInput::FirstMomentumComponent(self.component),
            "right-momentum" => EagerKernelInput::SecondMomentumComponent(self.component),
            "current" => EagerKernelInput::FirstCurrentComponent(self.component),
            "momentum" => EagerKernelInput::FirstMomentumComponent(self.component),
            "coupling-real" => EagerKernelInput::CouplingReal,
            "coupling-imag" => EagerKernelInput::CouplingImag,
            "model-parameter" => {
                EagerKernelInput::ModelParameter(self.model_parameter_index.ok_or_else(|| {
                    RusticolError::integrity(
                        "prepared model-parameter input lacks its stable index",
                    )
                })?)
            }
            other => {
                return Err(RusticolError::artifact(format!(
                    "unsupported prepared kernel input role {other:?}"
                )));
            }
        };
        if self.role == "model-parameter" {
            if self
                .model_parameter_name
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                return Err(RusticolError::integrity(
                    "prepared model-parameter input lacks its name",
                ));
            }
        } else if self.model_parameter_name.is_some() || self.model_parameter_index.is_some() {
            return Err(RusticolError::integrity(
                "only prepared model-parameter inputs may carry parameter metadata",
            ));
        }
        Ok(input)
    }
}

fn contiguous_value_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .value_storage
        .value_slots
        .iter()
        .map(|slot| (slot.value_slot_id, slot.dimension))
        .collect::<Vec<_>>();
    contiguous_widths("value", pairs)
}

fn contiguous_momentum_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .momentum_slots
        .iter()
        .map(|slot| {
            let width = slot
                .component_stop
                .checked_sub(slot.component_start)
                .ok_or_else(|| {
                    RusticolError::artifact("eager momentum slot has an inverted component range")
                })?;
            Ok((slot.momentum_slot_id, width))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    contiguous_widths("momentum", pairs)
}

fn contiguous_current_slot_widths(plan: &ExecutionPlan) -> RusticolResult<Vec<u32>> {
    let pairs = plan
        .current_storage
        .current_slots
        .iter()
        .map(|slot| (slot.current_id, slot.dimension))
        .collect::<Vec<_>>();
    contiguous_widths("current", pairs)
}

fn contiguous_widths(name: &str, mut pairs: Vec<(usize, usize)>) -> RusticolResult<Vec<u32>> {
    pairs.sort_unstable_by_key(|(id, _)| *id);
    if pairs
        .iter()
        .enumerate()
        .any(|(expected, (id, _))| expected != *id)
    {
        return Err(RusticolError::artifact(format!(
            "eager {name} slot ids must be contiguous from zero"
        )));
    }
    pairs
        .into_iter()
        .map(|(id, width)| {
            if width == 0 {
                return Err(RusticolError::artifact(format!(
                    "eager {name} slot {id} has zero width"
                )));
            }
            u32::try_from(width)
                .map_err(|_| RusticolError::artifact(format!("eager {name} width exceeds u32")))
        })
        .collect()
}
