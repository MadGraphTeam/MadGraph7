// SPDX-License-Identifier: 0BSD
//! Explicit cross-precision verification and exact-input subset rescue.
//! Comparisons/conversions use the shared dependency-backed Batch primitives.
use crate::runtime::{Batch, ExecutionError, arithmetic_mode, physical::PhysicalBackend};
use rusticol_core::loop_execution::{
    CompleteLoopResult, LoopExecutor, LoopInputs, LoopLaneFailure,
};
use rusticol_core::{RusticolError, RusticolResult};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

type Complete = CompleteLoopResult<Arc<Batch>>;
fn invalid(e: impl ToString) -> RusticolError {
    RusticolError::invalid_argument(e.to_string())
}
fn execution(e: ExecutionError) -> RusticolError {
    RusticolError::evaluation(e.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ArithmeticSpec {
    pub arithmetic: String,
    pub precision: Option<u32>,
}
impl ArithmeticSpec {
    pub fn bits(&self) -> RusticolResult<u32> {
        arithmetic_mode(&self.arithmetic, self.precision)
            .map(|(_, bits)| bits)
            .map_err(invalid)
    }
}
#[derive(Clone, Debug)]
pub struct VerificationPolicy {
    pub higher: Vec<ArithmeticSpec>,
    pub relative_tolerance: String,
    pub absolute_tolerance: String,
    pub verify_successful: bool,
}
impl VerificationPolicy {
    fn validate(&self, primary: &ArithmeticSpec) -> RusticolResult<()> {
        if self.higher.is_empty() {
            return Err(invalid("verification requires higher arithmetic levels"));
        }
        let mut bits = primary.bits()?;
        for spec in &self.higher {
            let next = spec.bits()?;
            if next <= bits {
                return Err(invalid(
                    "verification levels must strictly increase actual precision bits",
                ));
            }
            bits = next;
        }
        let empty = Batch::typed::<f64>(vec![], vec![], 0, 53).map_err(execution)?;
        empty
            .agreement_rows(&empty, &self.relative_tolerance, &self.absolute_tolerance)
            .map_err(invalid)?;
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct VerificationAttempt {
    pub arithmetic: ArithmeticSpec,
    pub precision_bits: u32,
    pub numerical_success: bool,
    pub compared_precision_bits: Option<u32>,
    pub agreement: Option<bool>,
    pub conversion_agreement: Option<bool>,
    pub failures: Vec<LoopLaneFailure>,
}
#[derive(Clone, Debug, Serialize)]
pub struct VerificationDiagnostic {
    pub point_id: u64,
    pub status: String,
    pub accuracy_established: bool,
    pub agreeing_bits: Option<[u32; 2]>,
    pub attempts: Vec<VerificationAttempt>,
    pub detail: String,
}
pub struct VerifiedLoopResult {
    pub result: Complete,
    pub diagnostics: Vec<VerificationDiagnostic>,
}
impl VerifiedLoopResult {
    pub fn diagnostics_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.diagnostics)
            .map_err(|e| RusticolError::serialization(e.to_string()))
    }
}
#[derive(Default)]
pub struct VerificationWorkspace {
    higher: BTreeMap<ArithmeticSpec, LoopExecutor<PhysicalBackend>>,
}
impl VerificationWorkspace {
    pub fn clear(&mut self) {
        self.higher.clear();
    }
}

fn select(batch: &Batch, ids: &[u64]) -> RusticolResult<Batch> {
    let locations: HashMap<_, _> = batch
        .ids()
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let indices = ids
        .iter()
        .map(|id| {
            locations
                .get(id)
                .copied()
                .ok_or_else(|| invalid("verification row missing"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    batch.take_rows(indices, None).map_err(execution)
}
fn packet(result: &Complete, rows_per_point: usize) -> RusticolResult<Batch> {
    let mut batches = vec![result.born.as_ref()];
    batches.extend(result.components.iter().map(AsRef::as_ref));
    batches.push(result.totals.as_ref());
    let joined = Batch::concat_columns(&batches).map_err(execution)?;
    if rows_per_point == 0
        || result.point_ids.len().checked_mul(rows_per_point) != Some(joined.rows())
        || result.statuses.len() != joined.rows()
        || joined
            .ids()
            .iter()
            .enumerate()
            .any(|(row, &id)| id != row as u64)
    {
        return Err(invalid("incomplete resolved point"));
    }
    let columns = joined
        .columns()
        .checked_mul(rows_per_point)
        .ok_or_else(|| invalid("verification dimensions overflow"))?;
    joined
        .reshape_rows(result.point_ids.len(), columns, result.point_ids.clone())
        .map_err(execution)
}
fn record(
    diagnostics: &mut HashMap<u64, VerificationDiagnostic>,
    result: &Complete,
    spec: &ArithmeticSpec,
) -> RusticolResult<()> {
    let successful: HashSet<_> = result.point_ids.iter().copied().collect();
    for id in &result.requested_point_ids {
        diagnostics
            .get_mut(id)
            .ok_or_else(|| invalid("foreign verification point"))?
            .attempts
            .push(VerificationAttempt {
                arithmetic: spec.clone(),
                precision_bits: spec.bits()?,
                numerical_success: successful.contains(id),
                compared_precision_bits: None,
                agreement: None,
                conversion_agreement: None,
                failures: result
                    .failures
                    .iter()
                    .filter(|f| f.point_id == *id)
                    .cloned()
                    .collect(),
            });
    }
    Ok(())
}

/// All retry inputs come from `inputs`, never from previously rounded buffers.
/// One captured source revision is re-evaluated independently in each carrier.
pub fn evaluate_verified(
    executor: &mut LoopExecutor<PhysicalBackend>,
    workspace: &mut VerificationWorkspace,
    inputs: LoopInputs,
    primary_spec: &ArithmeticSpec,
    policy: &VerificationPolicy,
    resolved: bool,
) -> RusticolResult<VerifiedLoopResult> {
    policy.validate(primary_spec)?;
    if !inputs.model_values.is_empty() {
        return Err(invalid("verification requires named source parameters"));
    }
    let loaded = executor.loaded().clone();
    let owner = executor
        .source_owner()
        .cloned()
        .ok_or_else(|| invalid("verification requires source-owned exact parameters"))?;
    let snapshot = owner.snapshot()?;
    let rows_per_point = if resolved {
        loaded.plan().resolved_rows_per_point()?
    } else {
        1
    };
    let width = loaded
        .plan()
        .state_legs
        .len()
        .checked_mul(4)
        .ok_or_else(|| invalid("input width overflows"))?;
    let primary = executor.evaluate_partial_with_snapshot(
        inputs.clone(),
        Some(snapshot.clone()),
        resolved,
    )?;
    let initial = packet(&primary, rows_per_point)?;
    let mut diagnostics: HashMap<_, _> = inputs
        .point_ids
        .iter()
        .map(|&id| {
            (
                id,
                VerificationDiagnostic {
                    point_id: id,
                    status: "failed".into(),
                    accuracy_established: false,
                    agreeing_bits: None,
                    attempts: vec![],
                    detail: "no successful agreeing precision pair".into(),
                },
            )
        })
        .collect();
    record(&mut diagnostics, &primary, primary_spec)?;
    let primary_ids: HashSet<_> = initial.ids().iter().copied().collect();
    let mut accepted = Vec::<Batch>::new();
    let mut status_rows = HashMap::new();
    let mut pending: Vec<_> = inputs
        .point_ids
        .iter()
        .copied()
        .filter(|id| policy.verify_successful || !primary_ids.contains(id))
        .collect();
    for (&id, statuses) in primary
        .point_ids
        .iter()
        .zip(primary.statuses.chunks(rows_per_point))
    {
        status_rows.insert(id, statuses.to_vec());
    }
    if !policy.verify_successful {
        accepted.push(initial.clone());
        for id in initial.ids() {
            let d = diagnostics.get_mut(id).expect("original point");
            d.status = "success".into();
            d.detail = "successful initial evaluation; accuracy not established".into();
        }
    }
    // A group contains the latest successful evaluation of each still-pending
    // point. Groups can have different carriers, but their IDs remain disjoint.
    let mut previous = if policy.verify_successful {
        vec![(0_usize, initial.clone())]
    } else {
        vec![]
    };
    for (level, spec) in policy.higher.iter().enumerate() {
        if pending.is_empty() {
            break;
        }
        if !workspace.higher.contains_key(spec) {
            let backend =
                PhysicalBackend::new(&spec.arithmetic, spec.precision, "interpreted", "generic")
                    .map_err(invalid)?;
            let target =
                LoopExecutor::with_parameters(loaded.clone(), backend, Some(snapshot.overrides()))?;
            workspace.higher.insert(spec.clone(), target);
        }
        let target = workspace
            .higher
            .get_mut(spec)
            .expect("higher workspace bound");
        if !Arc::ptr_eq(target.loaded(), &loaded) {
            return Err(invalid(
                "verification workspace belongs to a different program",
            ));
        }
        let target_owner = target.source_owner().expect("source owner");
        let reconstructed =
            target_owner.reconstruct_snapshot(target.backend(), &owner, &snapshot)?;
        let attempt = target.evaluate_partial_with_snapshot(
            inputs.select_points(&pending, width)?,
            Some(reconstructed),
            resolved,
        )?;
        record(&mut diagnostics, &attempt, spec)?;
        let current = packet(&attempt, rows_per_point)?;
        let current_ids: HashSet<_> = current.ids().iter().copied().collect();
        let mut now_accepted = HashSet::new();
        for (prior_level, prior) in &previous {
            let common: Vec<_> = prior
                .ids()
                .iter()
                .copied()
                .filter(|id| current_ids.contains(id))
                .collect();
            if common.is_empty() {
                continue;
            }
            let left = select(prior, &common)?;
            let right = select(&current, &common)?;
            let agreements = left
                .agreement_rows(
                    &right,
                    &policy.relative_tolerance,
                    &policy.absolute_tolerance,
                )
                .map_err(execution)?;
            for (id, ok) in &agreements {
                let a = diagnostics
                    .get_mut(id)
                    .expect("original point")
                    .attempts
                    .last_mut()
                    .expect("current attempt");
                a.compared_precision_bits = Some(prior.precision_bits());
                a.agreement = Some(*ok);
            }
            let agreed: Vec<_> = agreements
                .into_iter()
                .filter_map(|(id, ok)| ok.then_some(id))
                .collect();
            if agreed.is_empty() {
                continue;
            }
            // Preserve a verified primary value; rescue accepts the newer higher
            // evaluation only after checking its requested-carrier conversion.
            let chosen = if *prior_level == 0 {
                select(&left, &agreed)?
            } else {
                select(&right, &agreed)?
            };
            let converted = chosen
                .convert_arithmetic(&primary_spec.arithmetic, primary_spec.precision)
                .map_err(execution)?;
            let conversion = converted
                .agreement_rows(
                    &chosen,
                    &policy.relative_tolerance,
                    &policy.absolute_tolerance,
                )
                .map_err(execution)?;
            for (id, ok) in &conversion {
                diagnostics
                    .get_mut(id)
                    .expect("original point")
                    .attempts
                    .last_mut()
                    .expect("current attempt")
                    .conversion_agreement = Some(*ok);
            }
            let approved: Vec<_> = conversion
                .into_iter()
                .filter_map(|(id, ok)| ok.then_some(id))
                .collect();
            if approved.is_empty() {
                continue;
            }
            accepted.push(select(&converted, &approved)?);
            for id in approved {
                now_accepted.insert(id);
                let d = diagnostics.get_mut(&id).expect("original point");
                d.status = if *prior_level == 0 {
                    "success"
                } else {
                    "rescued"
                }
                .into();
                d.accuracy_established = true;
                d.agreeing_bits = Some([prior.precision_bits(), current.precision_bits()]);
                d.detail = "empirical cross-precision agreement, including requested-carrier conversion; not an independent physics proof".into();
                if *prior_level != 0 {
                    let point = attempt
                        .point_ids
                        .iter()
                        .position(|&x| x == id)
                        .expect("successful point");
                    let start = point * rows_per_point;
                    status_rows
                        .insert(id, attempt.statuses[start..start + rows_per_point].to_vec());
                }
            }
        }
        pending.retain(|id| !now_accepted.contains(id));
        let pending_set: HashSet<_> = pending.iter().copied().collect();
        let mut next = Vec::new();
        for (prior_level, prior) in previous {
            let keep: Vec<_> = prior
                .ids()
                .iter()
                .copied()
                .filter(|id| pending_set.contains(id) && !current_ids.contains(id))
                .collect();
            if !keep.is_empty() {
                next.push((prior_level, select(&prior, &keep)?));
            }
        }
        let keep: Vec<_> = current
            .ids()
            .iter()
            .copied()
            .filter(|id| pending_set.contains(id))
            .collect();
        if !keep.is_empty() {
            next.push((level + 1, select(&current, &keep)?));
        }
        previous = next;
    }
    // Include a typed empty group even if every point failed.
    accepted.push(initial.take_rows(vec![], None).map_err(execution)?);
    let merged = Batch::merge_rows(
        inputs.point_ids.clone(),
        &accepted.iter().collect::<Vec<_>>(),
    )
    .map_err(execution)?;
    let rows = merged
        .rows()
        .checked_mul(rows_per_point)
        .ok_or_else(|| invalid("verification row dimensions overflow"))?;
    let ids: Vec<_> = (0..rows).map(|row| row as u64).collect();
    let flat = merged
        .reshape_rows(ids.len(), merged.columns() / rows_per_point, ids.clone())
        .map_err(execution)?;
    let widths: Vec<_> = std::iter::once(primary.born.columns())
        .chain(primary.components.iter().map(|b| b.columns()))
        .chain(std::iter::once(primary.totals.columns()))
        .collect();
    let mut start = 0;
    let mut blocks = Vec::new();
    for columns in widths {
        blocks.push(Arc::new(
            flat.take_columns((start..start + columns).collect())
                .map_err(execution)?,
        ));
        start += columns;
    }
    let born = blocks.remove(0);
    let totals = blocks.pop().expect("totals");
    let components = blocks
        .try_into()
        .map_err(|_| invalid("component count differs"))?;
    let mut failures = Vec::new();
    for id in &pending {
        let d = diagnostics.get_mut(id).expect("original point");
        if d.attempts.iter().any(|a| a.numerical_success) {
            d.status = "accuracy-not-established".into();
            d.detail = "numerical values were obtained, but no precision pair and requested-carrier conversion met the agreement bound".into();
        } else {
            d.detail = "all configured arithmetic levels failed numerically".into();
        }
        failures.extend(d.attempts.iter().flat_map(|a| a.failures.iter().cloned()));
        failures.push(LoopLaneFailure {
            point_id: *id,
            helicity: None,
            order_index: None,
            family_index: None,
            external_leg: None,
            color_component_index: None,
            stage: "precision verification".into(),
            detail: d.detail.clone(),
        });
    }
    let mut statuses: Vec<Vec<(u64, String)>> = merged
        .ids()
        .iter()
        .map(|id| {
            status_rows
                .remove(id)
                .ok_or_else(|| invalid("accepted point has no reduction status"))
        })
        .collect::<RusticolResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    for (&id, row) in ids.iter().zip(&mut statuses) {
        for (status_id, _) in row {
            *status_id = id;
        }
    }
    Ok(VerifiedLoopResult {
        result: CompleteLoopResult {
            requested_point_ids: inputs.point_ids.clone(),
            point_ids: merged.ids().to_vec(),
            born,
            components,
            totals,
            statuses,
            r2_strategy: primary.r2_strategy,
            source_revision: Some(snapshot.revision()),
            failures,
        },
        diagnostics: inputs
            .point_ids
            .iter()
            .map(|id| diagnostics.remove(id).expect("original point"))
            .collect(),
    })
}
