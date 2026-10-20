// SPDX-License-Identifier: 0BSD

//! Additive native virtual handle over the verified process-artifact boundary.
//! The shared core owns complete execution; this consumer supplies its existing
//! NinjaRS backend. No Python interpreter or generation code is involved.

use crate::runtime::{Batch, physical::PhysicalBackend};
use crate::verification::{
    ArithmeticSpec, SourceBindingValidation, VerificationPolicy, VerificationWorkspace,
    VerifiedLoopResult, evaluate_observable_verified,
};
use rusticol_core::loop_artifact::{
    LoadedLoopArtifact, LoopExecutionMetadata, ORDINARY_INTERFERENCE, open_loop_artifact,
};
use rusticol_core::loop_execution::{CompleteLoopResult, LoopExecutor, LoopInputs, LoopObservable};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner};
use rusticol_core::{RusticolError, RusticolResult};
use std::{path::Path, sync::Arc};

pub struct VirtualRuntime {
    artifact: Arc<LoadedLoopArtifact>,
    selected_metadata: LoopExecutionMetadata,
    backend: PhysicalBackend,
    owner: Arc<SourceOwner<PhysicalBackend>>,
    executor: Option<LoopExecutor<PhysicalBackend>>,
    arithmetic: String,
    precision: Option<u32>,
    requested_strategy: String,
    verification: VerificationWorkspace,
    policy: Option<VerificationPolicy>,
    binding_validation: SourceBindingValidation,
}

fn strategy(requested: &str) -> RusticolResult<()> {
    match requested {
        "auto" | "model" => Ok(()),
        "direct" => Err(RusticolError::unsupported_runtime_capability(
            "direct R2",
            "this artifact contains only the complete model-rule R2 branch",
        )),
        _ => Err(RusticolError::invalid_argument(
            "r2_strategy must be auto, model, or direct",
        )),
    }
}

/// Reorder original exact strings once, before any numerical ingress or rescue.
/// The immutable selected group relation has already validated this permutation.
fn bind_public_momenta(
    mut inputs: LoopInputs,
    permutation: &[usize],
) -> RusticolResult<LoopInputs> {
    if permutation.iter().copied().eq(0..permutation.len()) {
        return Ok(inputs);
    }
    if !inputs.spin_sources.is_empty() {
        return Err(RusticolError::unsupported_runtime_capability(
            "Bose permutation vector sources",
            "this member has only ordinary inclusive transport",
        ));
    }
    let width = permutation
        .len()
        .checked_mul(4)
        .ok_or_else(|| RusticolError::invalid_argument("external momentum width overflow"))?;
    let count =
        inputs.point_ids.len().checked_mul(width).ok_or_else(|| {
            RusticolError::invalid_argument("external momentum batch size overflow")
        })?;
    if inputs.momenta.len() != count || permutation.iter().any(|&index| index >= permutation.len())
    {
        return Err(RusticolError::invalid_argument(
            "original external momentum layout differs",
        ));
    }
    let mut values = Vec::with_capacity(count);
    for point in inputs.momenta.chunks_exact(width) {
        for &leg in permutation {
            values.extend_from_slice(&point[4 * leg..4 * (leg + 1)]);
        }
    }
    inputs.momenta = values;
    Ok(inputs)
}

impl VirtualRuntime {
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        Self::open_with_policy(
            path,
            process,
            arithmetic,
            precision,
            parameters,
            r2_strategy,
            None,
        )
    }

    /// Configure source admission and complete-result verification together.
    /// Without a policy, model realization retains eager primary validation.
    pub fn open_with_policy(
        path: impl AsRef<Path>,
        process: Option<&str>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Self> {
        crate::runtime::register_scalar_primitives();
        Self::bind(
            Arc::new(open_loop_artifact(path, process)?),
            arithmetic,
            precision,
            parameters,
            r2_strategy,
            policy,
        )
    }

    fn bind(
        artifact: Arc<LoadedLoopArtifact>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Self> {
        strategy(r2_strategy)?;
        let engine = if arithmetic == "double" {
            "fast"
        } else {
            "generic"
        };
        let backend = PhysicalBackend::new(arithmetic, precision, "interpreted", engine)
            .map_err(|e| RusticolError::invalid_argument(e.to_string()))?;
        let binding_validation = SourceBindingValidation::new(
            &ArithmeticSpec {
                arithmetic: arithmetic.into(),
                precision,
            },
            policy,
        )?;
        let owner = binding_validation.owner(artifact.runtime.clone(), &backend, parameters)?;
        let executor = LoopExecutor::with_source_owner(owner.clone(), backend.clone())?;
        let mut selected_metadata = artifact.metadata.clone();
        if !artifact
            .selection
            .external_permutation
            .iter()
            .copied()
            .eq(0..artifact.runtime.plan().state_legs.len())
        {
            selected_metadata.resolved = false;
            selected_metadata.correlations = false;
            selected_metadata.interference = ORDINARY_INTERFERENCE.into();
        }
        Ok(Self {
            artifact,
            selected_metadata,
            backend,
            owner,
            executor: Some(executor),
            arithmetic: arithmetic.into(),
            precision,
            requested_strategy: r2_strategy.into(),
            verification: VerificationWorkspace::default(),
            policy: policy.cloned(),
            binding_validation,
        })
    }

    /// Capabilities of the selected public process. The saved representative
    /// metadata is unchanged and remains separately exposed in metadata_json.
    pub fn metadata(&self) -> &LoopExecutionMetadata {
        &self.selected_metadata
    }
    pub fn metadata_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&serde_json::json!({
            "artifact_id": self.artifact.artifact_id,
            "process_id": self.process_id(),
            "representative_process_id": self.artifact.selection.process.id,
            "expression": self.artifact.selection.public_expression,
            "external_pdgs": self.artifact.selection.external_pdgs,
            "external_permutation": self.artifact.selection.external_permutation,
            "initial_average": self.artifact.runtime.plan().initial_average,
            "physics": self.metadata(),
            "physics_scope": "selected-process",
            "representative_physics": self.artifact.metadata,
            "plan": self.artifact.runtime.plan(),
            "color_contraction": self.color_contraction_metadata(),
        }))
        .map_err(|e| RusticolError::serialization(e.to_string()))
    }
    pub fn plan_metadata_json(&self) -> RusticolResult<String> {
        self.artifact.runtime.metadata_json()
    }
    pub fn color_contraction_metadata(&self) -> serde_json::Value {
        let plan = self.artifact.runtime.plan();
        serde_json::json!({
            "scope": "bare-loop dual Born weights; Born-square and UV/R2/field contractions remain direct",
            "arithmetic": self.arithmetic,
            "ordinary": plan.color_weight_execution(&self.arithmetic, false, false, None),
            "resolved": self.metadata().resolved.then(|| plan.helicity_projection.as_ref().map(|_| plan.color_weight_execution(&self.arithmetic, true, false, None))).flatten(),
            "directed": self.metadata().correlations.then(|| plan.directed_projection.as_ref().map(|_| plan.color_weight_execution(&self.arithmetic, false, true, None))).flatten(),
            "spin_sources": plan.spin_projections.iter().flatten().filter(|_| self.metadata().correlations).enumerate().map(|(i,p)| serde_json::json!({
                "legs": p.legs,
                "operation": plan.color_weight_execution(&self.arithmetic, false, true, Some(i)),
            })).collect::<Vec<_>>(),
        })
    }
    pub fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    pub fn precision(&self) -> Option<u32> {
        self.precision
    }
    pub fn external_count(&self) -> usize {
        self.artifact.runtime.plan().state_legs.len()
    }
    pub fn process_id(&self) -> &str {
        &self.artifact.selection.requested_id
    }
    pub fn precision_bits(&self) -> u32 {
        self.backend.precision_bits()
    }
    pub fn requested_r2_strategy(&self) -> &str {
        &self.requested_strategy
    }
    pub fn source_revision(&self) -> RusticolResult<u64> {
        Ok(self.owner.snapshot()?.revision())
    }
    pub fn parameter_overrides(&self) -> RusticolResult<SourceOverrides> {
        Ok(self.owner.snapshot()?.overrides().clone())
    }
    pub fn source_owner(&self) -> &Arc<SourceOwner<PhysicalBackend>> {
        &self.owner
    }
    pub fn backend(&self) -> &PhysicalBackend {
        &self.backend
    }
    pub fn verification_policy(&self) -> Option<&VerificationPolicy> {
        self.policy.as_ref()
    }
    pub fn source_binding_validation(&self) -> &SourceBindingValidation {
        &self.binding_validation
    }
    pub fn update_parameters(&self, changes: &SourceOverrides) -> RusticolResult<u64> {
        Ok(self
            .binding_validation
            .update(&self.owner, &self.backend, changes)?
            .revision())
    }

    /// Clear process numerical workspaces; exact parameter state and existing
    /// result buffers remain valid. The next call binds fresh native workspaces.
    pub fn clear(&mut self) {
        self.executor = None;
        self.verification.clear();
    }

    pub fn rebind(
        &self,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: Option<&str>,
    ) -> RusticolResult<Self> {
        self.rebind_with_policy(arithmetic, precision, parameters, r2_strategy, None)
    }

    /// An omitted replacement inherits the handle policy, checked against the
    /// new primary arithmetic before any source state is published.
    pub fn rebind_with_policy(
        &self,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Self> {
        let mut exact = self.parameter_overrides()?;
        if let Some(changes) = parameters {
            exact.extend(changes.clone());
        }
        Self::bind(
            self.artifact.clone(),
            arithmetic,
            precision,
            Some(&exact),
            r2_strategy.unwrap_or(&self.requested_strategy),
            policy.or(self.policy.as_ref()),
        )
    }

    pub fn evaluate(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        self.evaluate_mode(inputs, r2_strategy, LoopObservable::Inclusive, false)
    }
    pub fn evaluate_resolved(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        self.evaluate_mode(inputs, r2_strategy, LoopObservable::Resolved, false)
    }
    /// Successful complete points plus typed reduction-channel failures.
    pub fn evaluate_partial(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        resolved: bool,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        self.evaluate_mode(
            inputs,
            r2_strategy,
            LoopObservable::from_resolved(resolved),
            true,
        )
    }
    /// All saved ordered colour histories, preserving both complex cross terms.
    pub fn evaluate_directed(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        partial: bool,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        self.evaluate_mode(inputs, r2_strategy, LoopObservable::Directed, partial)
    }
    pub fn evaluate_verified(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        resolved: bool,
        policy: &VerificationPolicy,
    ) -> RusticolResult<VerifiedLoopResult> {
        self.evaluate_verified_mode(
            inputs,
            r2_strategy,
            LoopObservable::from_resolved(resolved),
            policy,
        )
    }
    pub fn evaluate_directed_verified(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        policy: &VerificationPolicy,
    ) -> RusticolResult<VerifiedLoopResult> {
        self.evaluate_verified_mode(inputs, r2_strategy, LoopObservable::Directed, policy)
    }

    fn require_observable(&self, observable: LoopObservable) -> RusticolResult<()> {
        if observable == LoopObservable::Resolved && !self.metadata().resolved {
            return Err(RusticolError::unsupported_runtime_capability(
                "resolved virtual evaluation",
                "this selected process has no admitted resolved transport",
            ));
        }
        if observable == LoopObservable::Directed && !self.metadata().correlations {
            return Err(RusticolError::unsupported_runtime_capability(
                "directed colour correlations",
                "this selected process has no admitted directed transport",
            ));
        }
        Ok(())
    }

    fn evaluate_verified_mode(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        observable: LoopObservable,
        policy: &VerificationPolicy,
    ) -> RusticolResult<VerifiedLoopResult> {
        self.require_observable(observable)?;
        strategy(r2_strategy.unwrap_or(&self.requested_strategy))?;
        let inputs = bind_public_momenta(inputs, &self.artifact.selection.external_permutation)?;
        if self.executor.is_none() {
            self.executor = Some(LoopExecutor::with_source_owner(
                self.owner.clone(),
                self.backend.clone(),
            )?);
        }
        evaluate_observable_verified(
            self.executor.as_mut().expect("workspace bound"),
            &mut self.verification,
            inputs,
            &ArithmeticSpec {
                arithmetic: self.arithmetic.clone(),
                precision: self.precision,
            },
            policy,
            observable,
        )
    }

    fn evaluate_mode(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        observable: LoopObservable,
        partial: bool,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        self.require_observable(observable)?;
        strategy(r2_strategy.unwrap_or(&self.requested_strategy))?;
        let inputs = bind_public_momenta(inputs, &self.artifact.selection.external_permutation)?;
        if self.policy.is_some() {
            return Err(RusticolError::invalid_argument(
                "configured stability requires verified evaluation to preserve rescue diagnostics",
            ));
        }
        if !inputs.model_values.is_empty() {
            return Err(RusticolError::invalid_argument(
                "source model parameters require named atomic updates",
            ));
        }
        if self.executor.is_none() {
            self.executor = Some(LoopExecutor::with_source_owner(
                self.owner.clone(),
                self.backend.clone(),
            )?);
        }
        let executor = self.executor.as_mut().expect("workspaces bound");
        if partial {
            executor.evaluate_observable_partial_with_snapshot(inputs, None, observable)
        } else {
            executor.evaluate_observable_with_snapshot(inputs, None, observable)
        }
    }
}

#[cfg(test)]
mod bose_input_tests {
    use super::bind_public_momenta;
    use rusticol_core::loop_execution::LoopInputs;
    use std::collections::BTreeMap;

    fn inputs() -> LoopInputs {
        LoopInputs {
            momenta: (0..48)
                .map(|i| format!("{i}.000000000000000000000000000000000000001"))
                .collect(),
            scales_squared: vec!["900000".into(), "1000000".into(), "1100000".into()],
            model_values: Vec::new(),
            point_ids: vec![u64::MAX, 7, 91],
            spin_sources: BTreeMap::new(),
        }
    }

    #[test]
    fn exact_bose_binding_commutes_with_original_id_subset_replay() {
        let original = inputs();
        let permutation = [0, 1, 3, 2];
        let rebound = bind_public_momenta(original.clone(), &permutation).unwrap();
        assert_eq!(rebound.point_ids, original.point_ids);
        assert_eq!(rebound.scales_squared, original.scales_squared);
        assert_eq!(rebound.model_values, original.model_values);
        assert_eq!(&rebound.momenta[8..12], &original.momenta[12..16]);
        assert_eq!(&rebound.momenta[12..16], &original.momenta[8..12]);
        let after = rebound.select_points(&[91, u64::MAX], 16).unwrap();
        let before = bind_public_momenta(
            original.select_points(&[91, u64::MAX], 16).unwrap(),
            &permutation,
        )
        .unwrap();
        assert_eq!(after.momenta, before.momenta);
        assert_eq!(after.point_ids, vec![91, u64::MAX]);
        assert_eq!(after.scales_squared, before.scales_squared);
        let empty =
            bind_public_momenta(original.select_points(&[], 16).unwrap(), &permutation).unwrap();
        assert!(empty.momenta.is_empty());
        assert!(empty.point_ids.is_empty());
    }

    #[test]
    fn exact_binding_uses_representative_to_member_direction_for_a_three_cycle() {
        let original = LoopInputs {
            momenta: (0..40)
                .map(|i| format!("{i}.000000000000000000000000000000000000001"))
                .collect(),
            scales_squared: vec!["900000".into(), "1100000".into()],
            model_values: Vec::new(),
            point_ids: vec![91, 7],
            spin_sources: BTreeMap::new(),
        };
        let permutation = [0, 1, 3, 4, 2];
        let rebound = bind_public_momenta(original.clone(), &permutation).unwrap();
        for row in 0..2 {
            let start = row * 20;
            assert_eq!(
                &rebound.momenta[start + 8..start + 12],
                &original.momenta[start + 12..start + 16]
            );
            assert_eq!(
                &rebound.momenta[start + 12..start + 16],
                &original.momenta[start + 16..start + 20]
            );
            assert_eq!(
                &rebound.momenta[start + 16..start + 20],
                &original.momenta[start + 8..start + 12]
            );
        }
        let after = rebound.select_points(&[7], 20).unwrap();
        let before =
            bind_public_momenta(original.select_points(&[7], 20).unwrap(), &permutation).unwrap();
        assert_eq!(after.momenta, before.momenta);
        assert_eq!(after.point_ids, before.point_ids);
    }

    #[test]
    fn bose_binding_rejects_unproved_sources_and_malformed_momentum_layout() {
        let mut malformed = inputs();
        malformed.momenta.pop();
        assert!(bind_public_momenta(malformed, &[0, 1, 3, 2]).is_err());
        let mut sources = inputs();
        sources
            .spin_sources
            .insert(3, vec![["1".into(), "0".into()]; 12]);
        assert!(bind_public_momenta(sources.clone(), &[0, 1, 3, 2]).is_err());
        // An identity member keeps the existing source path untouched.
        assert_eq!(
            bind_public_momenta(sources.clone(), &[0, 1, 2, 3])
                .unwrap()
                .spin_sources,
            sources.spin_sources
        );
    }
}
