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
use rusticol_core::loop_execution::{
    CompleteLoopResult, LoopExecutor, LoopInputs, LoopLaneFailure, LoopObservable,
};
use rusticol_core::loop_runtime::DirectedHistory;
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner};
use rusticol_core::{RusticolError, RusticolResult};
use std::{path::Path, sync::Arc};

pub struct VirtualRuntime {
    artifact: Arc<LoadedLoopArtifact>,
    selected_metadata: LoopExecutionMetadata,
    selected_histories: Vec<DirectedHistory>,
    selected_spin_classes: Vec<Vec<usize>>,
    history_routes: Option<Vec<usize>>,
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
    source_classes: &[Vec<usize>],
) -> RusticolResult<LoopInputs> {
    let key = inputs.spin_sources.keys().copied().collect::<Vec<_>>();
    if !key.is_empty() && !source_classes.contains(&key) {
        return Err(RusticolError::unsupported_runtime_capability(
            "selected vector sources",
            "this member has no admitted transport for the requested joint class",
        ));
    }
    if permutation.iter().copied().eq(0..permutation.len()) {
        return Ok(inputs);
    }
    let mut inverse = vec![0; permutation.len()];
    for (index, &target) in permutation.iter().enumerate() {
        if target >= permutation.len() {
            return Err(RusticolError::invalid_argument(
                "invalid external permutation",
            ));
        }
        inverse[target] = index;
    }
    inputs.spin_sources = inputs
        .spin_sources
        .into_iter()
        .map(|(leg, values)| {
            leg.checked_sub(1)
                .and_then(|index| inverse.get(index))
                .map(|&index| (index + 1, values))
                .ok_or_else(|| RusticolError::invalid_argument("foreign member source leg"))
        })
        .collect::<RusticolResult<_>>()?;
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

/// Relabel only typed fields. Every whole-bank failure remains visible, even
/// when its original history was not selected; aliases retain separate IDs.
fn bind_member_failures(
    failures: Vec<LoopLaneFailure>,
    permutation: &[usize],
    routes: Option<&[usize]>,
    histories: Option<&[DirectedHistory]>,
) -> RusticolResult<Vec<LoopLaneFailure>> {
    let mut output = Vec::new();
    for mut failure in failures {
        if let Some(leg) = failure.external_leg {
            failure.external_leg =
                Some(*permutation.get(leg).ok_or_else(|| {
                    RusticolError::invalid_argument("foreign failure external leg")
                })?);
        }
        let Some((routes, index)) = routes.zip(failure.history_index) else {
            output.push(failure);
            continue;
        };
        let history = histories
            .and_then(|h| h.get(index))
            .ok_or_else(|| RusticolError::invalid_argument("foreign failure history index"))?;
        let selected = routes
            .iter()
            .enumerate()
            .filter_map(|(row, &route)| (route == index).then_some(row))
            .collect::<Vec<_>>();
        if selected.is_empty() {
            failure.history_index = None;
            failure.detail = format!(
                "shared representative history {:?} (index {index}): {}",
                history.id, failure.detail
            );
            output.push(failure);
        } else {
            output.extend(selected.into_iter().map(|row| {
                let mut value = failure.clone();
                value.history_index = Some(row);
                value
            }));
        }
    }
    Ok(output)
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
        // Saved exact evaluators are Symbolica state (pyAmpliCol's library key).
        let _unlock = rusticol_core::licence::unlock();
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
        let plan = artifact.runtime.plan();
        let mut selected_histories = plan
            .directed_projection
            .as_ref()
            .map_or_else(Vec::new, |p| p.histories.clone());
        let mut selected_spin_classes = plan
            .spin_projections
            .iter()
            .flatten()
            .map(|p| p.legs.clone())
            .collect::<Vec<_>>();
        let mut history_routes = None;
        if !artifact
            .selection
            .external_permutation
            .iter()
            .copied()
            .eq(0..plan.state_legs.len())
        {
            selected_metadata.resolved = false;
            let member = plan
                .admitted_flavor_members
                .iter()
                .flatten()
                .find(|member| member.process_id == artifact.selection.requested_id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("selected Bose admission is absent")
                })?;
            if let Some(scope) = &member.bose_correlations {
                history_routes = Some(plan.bose_history_routes(member)?);
                selected_histories = scope
                    .history_ids
                    .iter()
                    .map(|id| {
                        selected_histories
                            .iter()
                            .find(|history| &history.id == id)
                            .cloned()
                            .ok_or_else(|| {
                                RusticolError::invalid_argument("selected history absent")
                            })
                    })
                    .collect::<RusticolResult<_>>()?;
                selected_spin_classes = scope.spin_classes.clone();
            } else {
                selected_metadata.correlations = false;
                selected_metadata.interference = ORDINARY_INTERFERENCE.into();
                selected_histories.clear();
                selected_spin_classes.clear();
            }
        }
        Ok(Self {
            artifact,
            selected_metadata,
            selected_histories,
            selected_spin_classes,
            history_routes,
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
            "selected_correlations": {
                "histories": self.selected_histories,
                "spin_classes": self.selected_spin_classes,
            },
            "plan": self.artifact.runtime.plan(),
            "color_contraction": self.color_contraction_metadata(),
        }))
        .map_err(|e| RusticolError::serialization(e.to_string()))
    }
    pub fn selected_histories(&self) -> &[DirectedHistory] {
        &self.selected_histories
    }
    pub fn selected_spin_classes(&self) -> &[Vec<usize>] {
        &self.selected_spin_classes
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
            "spin_sources": self.selected_spin_classes.iter().map(|key| {
                let mut mapped = self.artifact.selection.external_permutation.iter().enumerate()
                    .filter_map(|(index, member)| key.contains(&(member + 1)).then_some(index + 1))
                    .collect::<Vec<_>>();
                mapped.sort_unstable();
                let index = plan.spin_projections.iter().flatten().position(|p| p.legs == mapped);
                serde_json::json!({
                    "legs": key,
                    "operation": plan.color_weight_execution(&self.arithmetic, false, true, index),
                })
            }).collect::<Vec<_>>(),
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
        let _unlock = rusticol_core::licence::unlock();
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
        // Saved exact evaluators are Symbolica state (pyAmpliCol's library key).
        let _unlock = rusticol_core::licence::unlock();
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
        // Saved exact evaluators are Symbolica state (pyAmpliCol's library key).
        let _unlock = rusticol_core::licence::unlock();
        self.require_observable(observable)?;
        strategy(r2_strategy.unwrap_or(&self.requested_strategy))?;
        let inputs = bind_public_momenta(
            inputs,
            &self.artifact.selection.external_permutation,
            &self.selected_spin_classes,
        )?;
        if self.executor.is_none() {
            self.executor = Some(LoopExecutor::with_source_owner(
                self.owner.clone(),
                self.backend.clone(),
            )?);
        }
        let mut result = evaluate_observable_verified(
            self.executor.as_mut().expect("workspace bound"),
            &mut self.verification,
            inputs,
            &ArithmeticSpec {
                arithmetic: self.arithmetic.clone(),
                precision: self.precision,
            },
            policy,
            observable,
        )?;
        result.result = self.bind_result(result.result)?;
        for diagnostic in &mut result.diagnostics {
            for attempt in &mut diagnostic.attempts {
                attempt.failures = self.bind_failures(std::mem::take(&mut attempt.failures))?;
            }
        }
        Ok(result)
    }

    fn evaluate_mode(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
        observable: LoopObservable,
        partial: bool,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        // Saved exact evaluators are Symbolica state (pyAmpliCol's library key).
        let _unlock = rusticol_core::licence::unlock();
        self.require_observable(observable)?;
        strategy(r2_strategy.unwrap_or(&self.requested_strategy))?;
        let inputs = bind_public_momenta(
            inputs,
            &self.artifact.selection.external_permutation,
            &self.selected_spin_classes,
        )?;
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
        let result = if partial {
            executor.evaluate_observable_partial_with_snapshot(inputs, None, observable)
        } else {
            executor.evaluate_observable_with_snapshot(inputs, None, observable)
        }?;
        self.bind_result(result)
    }

    fn bind_failures(
        &self,
        failures: Vec<LoopLaneFailure>,
    ) -> RusticolResult<Vec<LoopLaneFailure>> {
        bind_member_failures(
            failures,
            &self.artifact.selection.external_permutation,
            self.history_routes.as_deref(),
            self.artifact
                .runtime
                .plan()
                .directed_projection
                .as_ref()
                .map(|p| p.histories.as_slice()),
        )
    }

    fn bind_result(
        &self,
        mut result: CompleteLoopResult<Arc<Batch>>,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        result.failures = self.bind_failures(result.failures)?;
        if let Some(cross) = &mut result.directed {
            cross.spin_legs = cross
                .spin_legs
                .iter()
                .map(|&leg| self.artifact.selection.external_permutation[leg - 1] + 1)
                .collect();
            cross.spin_legs.sort_unstable();
        }
        let Some(routes) = self
            .history_routes
            .as_ref()
            .filter(|_| result.directed.is_some())
        else {
            return Ok(result);
        };
        let plan = self.artifact.runtime.plan();
        let gather = |values: &Arc<Batch>, width: usize| -> RusticolResult<Arc<Batch>> {
            let columns = routes
                .iter()
                .flat_map(|&history| (0..width).map(move |column| history * width + column))
                .collect();
            values
                .take_columns(columns)
                .map(Arc::new)
                .map_err(|e| RusticolError::invalid_argument(e.to_string()))
        };
        result.born = gather(&result.born, plan.order_layout.born.len())?;
        for values in &mut result.components {
            *values = gather(values, plan.order_layout.r#virtual.len() * 3)?;
        }
        let count = plan
            .directed_projection
            .as_ref()
            .expect("validated scope")
            .histories
            .len();
        if count == 0 || result.totals.columns() % count != 0 {
            return Err(RusticolError::invalid_argument(
                "directed total columns differ",
            ));
        }
        result.totals = gather(&result.totals, result.totals.columns() / count)?;
        let cross = result.directed.as_mut().expect("directed result checked");
        cross.first = gather(&cross.first, plan.order_layout.r#virtual.len() * 3)?;
        cross.second = gather(&cross.second, plan.order_layout.r#virtual.len() * 3)?;
        cross.history_ids = self
            .selected_histories
            .iter()
            .map(|h| h.id.clone())
            .collect();
        let mut offset = 0;
        let mut columns = Vec::new();
        for family in &plan.families {
            let width = 2 * family.sector_indices.len();
            columns.extend(routes.iter().flat_map(|&history| {
                (0..width).map(move |column| offset + history * width + column)
            }));
            offset += count * width;
        }
        for row in &mut result.statuses {
            if row.len() != offset {
                return Err(RusticolError::invalid_argument(
                    "directed status channels differ from saved plan",
                ));
            }
            *row = columns.iter().map(|&index| row[index].clone()).collect();
        }
        Ok(result)
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
        let rebound = bind_public_momenta(original.clone(), &permutation, &[]).unwrap();
        assert_eq!(rebound.point_ids, original.point_ids);
        assert_eq!(rebound.scales_squared, original.scales_squared);
        assert_eq!(rebound.model_values, original.model_values);
        assert_eq!(&rebound.momenta[8..12], &original.momenta[12..16]);
        assert_eq!(&rebound.momenta[12..16], &original.momenta[8..12]);
        let after = rebound.select_points(&[91, u64::MAX], 16).unwrap();
        let before = bind_public_momenta(
            original.select_points(&[91, u64::MAX], 16).unwrap(),
            &permutation,
            &[],
        )
        .unwrap();
        assert_eq!(after.momenta, before.momenta);
        assert_eq!(after.point_ids, vec![91, u64::MAX]);
        assert_eq!(after.scales_squared, before.scales_squared);
        let empty =
            bind_public_momenta(original.select_points(&[], 16).unwrap(), &permutation, &[])
                .unwrap();
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
        let rebound = bind_public_momenta(original.clone(), &permutation, &[]).unwrap();
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
            bind_public_momenta(original.select_points(&[7], 20).unwrap(), &permutation, &[])
                .unwrap();
        assert_eq!(after.momenta, before.momenta);
        assert_eq!(after.point_ids, before.point_ids);
    }

    #[test]
    fn bose_binding_rejects_unproved_sources_and_malformed_momentum_layout() {
        let mut malformed = inputs();
        malformed.momenta.pop();
        assert!(bind_public_momenta(malformed, &[0, 1, 3, 2], &[]).is_err());
        let mut sources = inputs();
        sources
            .spin_sources
            .insert(3, vec![["1".into(), "0".into()]; 12]);
        assert!(bind_public_momenta(sources.clone(), &[0, 1, 3, 2], &[]).is_err());
        // An identity member keeps the existing source path untouched.
        assert_eq!(
            bind_public_momenta(sources.clone(), &[0, 1, 2, 3], &[vec![3]])
                .unwrap()
                .spin_sources,
            sources.spin_sources
        );
    }
    #[test]
    fn exact_sources_follow_inverse_cycle_once_and_survive_subset_replay() {
        let mut original = LoopInputs {
            momenta: (0..40)
                .map(|i| format!("{i}.000000000000000000000000000000000000001"))
                .collect(),
            scales_squared: vec!["900000".into(), "1100000".into()],
            model_values: Vec::new(),
            point_ids: vec![91, u64::MAX],
            spin_sources: BTreeMap::new(),
        };
        for leg in [3, 4, 5] {
            original.spin_sources.insert(
                leg,
                (0..8)
                    .map(|component| {
                        [
                            format!("{leg}{component}.000000000000000000000000000000000000003"),
                            format!("-{component}.7"),
                        ]
                    })
                    .collect(),
            );
        }
        let permutation = [0, 1, 3, 4, 2];
        let classes = [vec![3, 4, 5]];
        let actual = bind_public_momenta(original.clone(), &permutation, &classes).unwrap();
        assert_eq!(actual.spin_sources[&3], original.spin_sources[&4]);
        assert_eq!(actual.spin_sources[&4], original.spin_sources[&5]);
        assert_eq!(actual.spin_sources[&5], original.spin_sources[&3]);
        let after = actual.select_points(&[u64::MAX], 20).unwrap();
        let before = bind_public_momenta(
            original.select_points(&[u64::MAX], 20).unwrap(),
            &permutation,
            &classes,
        )
        .unwrap();
        assert_eq!(after.momenta, before.momenta);
        assert_eq!(after.spin_sources, before.spin_sources);
        assert_eq!(after.point_ids, before.point_ids);
        assert_eq!(after.scales_squared, before.scales_squared);
        assert!(bind_public_momenta(original, &permutation, &[]).is_err());
    }

    #[test]
    fn aliases_and_unselected_failures_keep_whole_bank_context() {
        use super::bind_member_failures;
        use rusticol_core::loop_execution::LoopLaneFailure;
        use rusticol_core::loop_runtime::DirectedHistory;
        let histories = ["born", "unselected", "shared"].map(|id| DirectedHistory {
            id: id.into(),
            bra: vec![],
            ket: vec![],
        });
        let failure = |history| LoopLaneFailure {
            point_id: u64::MAX,
            helicity: None,
            color_component_index: None,
            history_index: Some(history),
            direction: Some(1),
            order_index: Some(2),
            family_index: Some(7),
            external_leg: Some(2),
            stage: "numerator coefficients".into(),
            detail: "nonfinite".into(),
        };
        let output = bind_member_failures(
            vec![failure(2), failure(1)],
            &[0, 1, 3, 4, 2],
            Some(&[2, 0, 2]),
            Some(&histories),
        )
        .unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(
            output.iter().map(|f| f.history_index).collect::<Vec<_>>(),
            vec![Some(0), Some(2), None]
        );
        assert!(output.iter().all(|f| f.external_leg == Some(3)
            && f.point_id == u64::MAX
            && f.direction == Some(1)
            && f.family_index == Some(7)
            && f.order_index == Some(2)));
        assert!(output[2].detail.contains("unselected"));
        assert!(output[2].detail.contains("index 1"));
        assert!(output[2].detail.contains("nonfinite"));
        assert!(
            bind_member_failures(
                vec![failure(3)],
                &[0, 1, 3, 4, 2],
                Some(&[2]),
                Some(&histories)
            )
            .is_err()
        );
    }
}
