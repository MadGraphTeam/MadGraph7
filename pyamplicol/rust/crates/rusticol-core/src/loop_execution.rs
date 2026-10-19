// SPDX-License-Identifier: 0BSD

//! One graph-free complete loop execution schedule, independent of Python and
//! of the numerical reduction implementation. Only validated runtime plans enter
//! this boundary. Native success is not an accuracy certificate.

use crate::loop_runtime::{LoadedLoopRuntime, ModelBinding, ScalarBinding, ScheduleGroup};
use crate::source_bindings::{SourceOverrides, SourceOwner, SourceScale, SourceSnapshot};
use crate::{RusticolError, RusticolResult};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Existing typed numerical operations supplied by the loop consumer. The core
/// owns their order and dataflow; the backend owns arithmetic and workspaces.
pub trait LoopBackend {
    type Batch: Clone;
    type Evaluator;
    type ExactEvaluator;
    type Schedule;
    type Reducer;

    fn evaluator(&self, state: &[u8]) -> RusticolResult<Self::Evaluator>;
    fn exact_evaluator(&self, state: &[u8]) -> RusticolResult<Self::ExactEvaluator>;
    fn schedule(
        &self,
        states: Vec<Vec<u8>>,
        inputs: Vec<Vec<usize>>,
        input_count: usize,
        outputs: Vec<usize>,
    ) -> RusticolResult<Self::Schedule>;
    fn reducer(
        &self,
        propagators: usize,
        numerators: [String; 3],
        coefficients: usize,
    ) -> RusticolResult<Self::Reducer>;
    fn ingress(
        &self,
        values: &[String],
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Self::Batch>;
    /// Exact complex source parameters use the same dependency-owned ingress.
    fn ingress_complex(
        &self,
        values: &[[String; 2]],
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Self::Batch>;
    fn exact(
        &self,
        evaluator: &mut Self::ExactEvaluator,
        values: &[String],
        rows: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Self::Batch>;
    fn evaluate(
        evaluator: &mut Self::Evaluator,
        input: &Self::Batch,
    ) -> RusticolResult<Self::Batch>;
    fn execute(schedule: &mut Self::Schedule, input: &Self::Batch) -> RusticolResult<Self::Batch>;
    fn states(
        momenta: &Self::Batch,
        helicities: Vec<i32>,
        kinds: Vec<String>,
        components: usize,
    ) -> RusticolResult<Self::Batch>;
    fn reduce(
        reducer: &mut Self::Reducer,
        shifts: &Self::Batch,
        masses: &Self::Batch,
        scales: &Self::Batch,
        coefficients: &Self::Batch,
        matrix: &Self::Batch,
    ) -> RusticolResult<(Self::Batch, Vec<(u64, String)>)>;
    fn rows(batch: &Self::Batch) -> usize;
    fn row_ids(batch: &Self::Batch) -> &[u64];
    fn columns(batch: &Self::Batch) -> usize;
    fn select(batch: &Self::Batch, columns: Vec<usize>) -> RusticolResult<Self::Batch>;
    fn gather(batch: &Self::Batch, rows: Vec<usize>, ids: Vec<u64>) -> RusticolResult<Self::Batch>;
    fn reshape(
        batch: &Self::Batch,
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Self::Batch>;
    fn concat(batches: &[&Self::Batch]) -> RusticolResult<Self::Batch>;
    fn require_finite(
        batch: &Self::Batch,
        stage: &str,
        rows_per_point: usize,
    ) -> RusticolResult<()>;
}

/// Original decimal inputs remain available for exact family invariants. The
/// legacy model has one global real input row; source-owned states are separate.
pub struct LoopInputs {
    pub momenta: Vec<String>,
    pub scales_squared: Vec<String>,
    pub model_values: Vec<String>,
    pub point_ids: Vec<u64>,
}

/// Columns retain the validated order layout. Each virtual sector is ordered
/// finite, single pole, double pole. Components are full, bare, UV, R2, local UV,
/// and external fields. Values remain in their requested native arithmetic.
pub struct CompleteLoopResult<T> {
    pub point_ids: Vec<u64>,
    pub born: T,
    pub components: [T; 6],
    pub totals: T,
    pub statuses: Vec<Vec<(u64, String)>>,
    pub r2_strategy: String,
    pub source_revision: Option<u64>,
}

struct FamilyWorkspace<B: LoopBackend> {
    exact: B::ExactEvaluator,
    masses: B::Evaluator,
    projector: B::Evaluator,
    helicity_projector: Option<B::Evaluator>,
    reducer: B::Reducer,
    rational: Option<B::Reducer>,
}

struct HelicityWorkspace<B: LoopBackend> {
    born: B::Evaluator,
    counterterms: B::Evaluator,
}

struct SourceWorkspace<B: LoopBackend> {
    owner: Arc<SourceOwner<B>>,
    ordinary: Option<SourceScale<B>>,
    counterterms: SourceScale<B>,
}

/// Fresh mutable arithmetic workspaces over one immutable, validated descriptor.
/// No graphs, symbolic optimization, or model reimport occur during binding.
pub struct LoopExecutor<B: LoopBackend> {
    loaded: Arc<LoadedLoopRuntime>,
    backend: B,
    schedules: Vec<B::Schedule>,
    families: Vec<FamilyWorkspace<B>>,
    bare: BTreeMap<String, B::Evaluator>,
    model: BTreeMap<String, B::Evaluator>,
    counterterms: Vec<Option<B::Evaluator>>,
    fields: Vec<Option<B::Evaluator>>,
    source: Option<SourceWorkspace<B>>,
    helicity: Option<HelicityWorkspace<B>>,
}

fn invalid(message: &str) -> RusticolError {
    RusticolError::invalid_argument(message)
}
fn count(a: usize, b: usize) -> RusticolResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("loop batch dimensions overflow"))
}
fn lane_id(point: u64, width: usize, offset: usize) -> RusticolResult<u64> {
    point
        .checked_mul(width as u64)
        .and_then(|x| x.checked_add(offset as u64))
        .ok_or_else(|| invalid("loop batch row identity overflow"))
}

impl<B: LoopBackend> LoopExecutor<B> {
    pub fn new(loaded: Arc<LoadedLoopRuntime>, backend: B) -> RusticolResult<Self> {
        Self::with_parameters(loaded, backend, None)
    }
    pub fn with_parameters(
        loaded: Arc<LoadedLoopRuntime>,
        backend: B,
        parameters: Option<&SourceOverrides>,
    ) -> RusticolResult<Self> {
        let owner = match &loaded.plan().model_binding {
            ModelBinding::Legacy => {
                if parameters.is_some() {
                    return Err(invalid("named parameters require a source model owner"));
                }
                None
            }
            ModelBinding::Source { .. } => {
                if loaded.plan().r2_strategy != "model" {
                    return Err(RusticolError::unsupported_runtime_capability(
                        "native source direct R2",
                        "source coordination currently requires model R2",
                    ));
                }
                Some(Arc::new(SourceOwner::new(loaded.clone(), &backend, parameters)?))
            }
        };
        Self::bind(loaded, backend, owner)
    }

    /// Recreate process workspaces while retaining the same global parameter
    /// owner and revision. Used after clear; arithmetic changes require a new
    /// owner constructed from the retained exact overrides instead.
    pub fn with_source_owner(owner: Arc<SourceOwner<B>>, backend: B) -> RusticolResult<Self> {
        let snapshot = owner.snapshot()?;
        let empty = backend.ingress(&[], 1, 0, vec![0])?;
        // The backend's ordinary concatenation checks the numerical carrier and
        // precision, so an owner cannot silently cross arithmetic modes.
        B::concat(&[snapshot.arguments(), &empty])?;
        Self::bind(owner.loaded().clone(), backend, Some(owner))
    }

    fn bind(
        loaded: Arc<LoadedLoopRuntime>,
        backend: B,
        owner: Option<Arc<SourceOwner<B>>>,
    ) -> RusticolResult<Self> {
        let plan = loaded.plan();
        if owner.is_some() && plan.r2_strategy != "model" {
            return Err(RusticolError::unsupported_runtime_capability(
                "native source direct R2",
                "source coordination currently requires model R2",
            ));
        }
        let source = owner.map(|owner| {
            let ordinary = SourceScale::ordinary(&owner, &backend)?;
            let counterterms = SourceScale::counterterms(&owner, &backend)?;
            Ok::<_, RusticolError>(SourceWorkspace { owner, ordinary, counterterms })
        }).transpose()?;
        let eval = |name: &str| backend.evaluator(loaded.evaluator_state(name)?);
        let schedules = plan
            .schedules
            .iter()
            .map(|item| {
                let states = item
                    .states
                    .iter()
                    .map(|name| loaded.evaluator_state(name).map(Vec::from))
                    .collect::<RusticolResult<Vec<_>>>()?;
                backend.schedule(
                    states,
                    item.input_slots.clone(),
                    item.parameter_count,
                    item.output_slots.clone(),
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let families = plan
            .families
            .iter()
            .enumerate()
            .map(|(index, item)| {
                Ok(FamilyWorkspace {
                    exact: backend.exact_evaluator(loaded.evaluator_state(&item.exact_state)?)?,
                    masses: eval(&item.mass_state)?,
                    projector: eval(&item.projector_state)?,
                    helicity_projector: plan.helicity_projection.as_ref()
                        .map(|projection| eval(&projection.families[index]))
                        .transpose()?,
                    reducer: backend.reducer(
                        item.propagators,
                        item.numerators.clone(),
                        item.powers.len(),
                    )?,
                    rational: if item.rational_numerators.iter().all(|v| v == "0") {
                        None
                    } else {
                        Some(backend.reducer(
                            item.propagators,
                            item.rational_numerators.clone(),
                            item.powers.len(),
                        )?)
                    },
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let bare = plan
            .bare_evaluators
            .iter()
            .map(|(key, name)| Ok((key.clone(), eval(name)?)))
            .collect::<RusticolResult<_>>()?;
        let model = plan
            .model_evaluators
            .iter()
            .map(|(key, name)| Ok((key.clone(), eval(name)?)))
            .collect::<RusticolResult<_>>()?;
        let counterterms = plan
            .currents
            .iter()
            .map(|item| item.scalar.legacy_state().map(&eval).transpose())
            .collect::<RusticolResult<_>>()?;
        let fields = plan
            .fields
            .iter()
            .map(|item| item.legacy_state().map(&eval).transpose())
            .collect::<RusticolResult<_>>()?;
        let helicity = plan.helicity_projection.as_ref().map(|projection| {
            Ok::<_, RusticolError>(HelicityWorkspace {
                born: eval(&projection.born)?,
                counterterms: eval(&projection.counterterms)?,
            })
        }).transpose()?;
        Ok(Self {
            loaded,
            backend,
            schedules,
            families,
            bare,
            model,
            counterterms,
            fields,
            source,
            helicity,
        })
    }
    pub fn loaded(&self) -> &Arc<LoadedLoopRuntime> {
        &self.loaded
    }

    pub fn source_owner(&self) -> Option<&Arc<SourceOwner<B>>> {
        self.source.as_ref().map(|source| &source.owner)
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn evaluate(&mut self, input: LoopInputs) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_with_snapshot(input, None)
    }
    /// Explicit snapshots are an internal retained-execution facility. Ordinary
    /// calls capture one current revision before any state or current evaluation.
    pub fn evaluate_with_snapshot(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, snapshot, false)
    }
    /// Retain each helicity as a native projection/reduction row. Currents and
    /// source scalars still execute through the same shared batched schedule.
    pub fn evaluate_resolved(
        &mut self,
        input: LoopInputs,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_resolved_with_snapshot(input, None)
    }
    pub fn evaluate_resolved_with_snapshot(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, snapshot, true)
    }
    fn evaluate_mode(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        resolved: bool,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        if resolved && self.helicity.is_none() {
            return Err(RusticolError::unsupported_runtime_capability(
                "helicity-resolved virtual evaluation",
                "this saved artifact has no helicity projection kernels",
            ));
        }
        let snapshot = match (&self.source, snapshot) {
            (Some(source), Some(snapshot)) => {
                source.owner.require_snapshot(&snapshot)?;
                Some(snapshot)
            }
            (Some(source), None) => Some(source.owner.snapshot()?),
            (None, Some(_)) => return Err(invalid("a source snapshot requires its source owner")),
            (None, None) => None,
        };
        if snapshot.is_some() && !input.model_values.is_empty() {
            return Err(invalid(
                "source model parameters require named owner updates, not positional model_values",
            ));
        }
        let plan = self.loaded.plan();
        let points = input.point_ids.len();
        let legs = plan.state_legs.len();
        let helicities = plan.helicities.len();
        let rows = count(points, helicities)?;
        let momentum_width = count(legs, 4)?;
        if input.momenta.len() != count(points, momentum_width)?
            || input.scales_squared.len() != points
            || (snapshot.is_none() && input.model_values.len() != plan.model_inputs.len())
            || input.point_ids.iter().collect::<BTreeSet<_>>().len() != points
        {
            return Err(invalid(
                "loop original input shapes or unique point identities differ",
            ));
        }
        let mut ids = Vec::with_capacity(rows);
        let mut point_rows = Vec::with_capacity(rows);
        for (row, &id) in input.point_ids.iter().enumerate() {
            for h in 0..helicities {
                ids.push(lane_id(id, helicities, h)?);
                point_rows.push(row);
            }
        }
        let projection_points = if resolved { rows } else { points };
        let projection_helicities = if resolved { 1 } else { helicities };
        let projection_rows_per_point = if resolved { helicities } else { 1 };
        let projection_ids = if resolved {
            ids.clone()
        } else {
            input.point_ids.clone()
        };
        let physical = self.backend.ingress(
            &input.momenta,
            points,
            momentum_width,
            input.point_ids.clone(),
        )?;
        let lanes = B::gather(&physical, point_rows.clone(), ids.clone())?;
        let scales =
            self.backend
                .ingress(&input.scales_squared, points, 1, input.point_ids.clone())?;
        let states = prepare_states::<B>(&lanes, &ids, plan)?;
        let (model_arguments, model, pointwise_model) = if let Some(snapshot) = &snapshot {
            let source = self.source.as_mut().expect("captured source owner");
            let ModelBinding::Source {
                model_output_indices,
                ..
            } = &plan.model_binding
            else {
                unreachable!("validated source binding")
            };
            let selected = if let Some(view) = &mut source.ordinary {
                view.evaluate(snapshot, &scales)?
            } else {
                snapshot.values().clone()
            };
            (
                snapshot.arguments().clone(),
                B::select(&selected, model_output_indices.clone())?,
                source.ordinary.is_some(),
            )
        } else {
            let arguments =
                self.backend
                    .ingress(&input.model_values, 1, input.model_values.len(), vec![0])?;
            let model = eval::<B>(&mut self.bare, "model_evaluator", &arguments)?;
            B::require_finite(&model, "global model formulas", 1)?;
            (arguments, model, false)
        };
        let model_rows = if pointwise_model {
            point_rows.clone()
        } else {
            vec![0; rows]
        };
        let bound = B::concat(&[
            &lanes,
            &states,
            &B::gather(&model, model_rows, ids.clone())?,
        ])?;
        let born_outputs = evaluate_group::<B>(&mut self.schedules, &plan.born, &bound)?;
        let born_amplitudes = B::concat(&born_outputs.iter().collect::<Vec<_>>())?;
        let loop_outputs = evaluate_group::<B>(&mut self.schedules, &plan.loops, &bound)?;
        let born_input = B::reshape(
            &born_amplitudes,
            projection_points,
            count(projection_helicities, B::columns(&born_amplitudes))?,
            projection_ids.clone(),
        )?;
        let born = if resolved {
            B::evaluate(
                &mut self.helicity.as_mut().expect("checked helicity kernels").born,
                &born_input,
            )?
        } else {
            eval::<B>(&mut self.bare, "born_projector", &born_input)?
        };
        B::require_finite(&born, "Born projection", projection_rows_per_point)?;
        let sector_count = plan.order_layout.r#virtual.len();
        let mut integrated = Vec::new();
        let mut rational_integrated = Vec::new();
        let mut statuses = vec![Vec::new(); projection_points];
        for (family, workspace) in plan.families.iter().zip(&mut self.families) {
            let mut parts = vec![&born_amplitudes];
            parts.extend(family.graph_indices.iter().map(|&i| &loop_outputs[i]));
            let projection = B::concat(&parts)?;
            let projection = B::reshape(
                &projection,
                projection_points,
                count(projection_helicities, B::columns(&projection))?,
                projection_ids.clone(),
            )?;
            let projector = if resolved {
                workspace.helicity_projector.as_mut()
                    .expect("checked helicity kernel")
            } else {
                &mut workspace.projector
            };
            let coefficients = B::evaluate(projector, &projection)?;
            let exact = self.backend.exact(
                &mut workspace.exact,
                &input.momenta,
                points,
                input.point_ids.clone(),
            )?;
            let n = family.propagators;
            let shifts = B::select(&exact, (0..4 * n).collect())?;
            let matrix = B::select(&exact, (4 * n..4 * n + n * n).collect())?;
            let masses = B::evaluate(
                &mut workspace.masses,
                &B::select(&model, family.mass_columns.clone())?,
            )?;
            let channels = family.sector_indices.len();
            let channel_rows = count(projection_points, channels)?;
            let mut reduction_ids = Vec::with_capacity(channel_rows);
            let mut gather = Vec::with_capacity(channel_rows);
            for (row, &id) in projection_ids.iter().enumerate() {
                for &sector in &family.sector_indices {
                    reduction_ids.push(lane_id(id, sector_count, sector)?);
                    gather.push(if resolved { row / helicities } else { row });
                }
            }
            let coefficients = B::reshape(
                &coefficients,
                channel_rows,
                family.powers.len(),
                reduction_ids.clone(),
            )?;
            let shifts = B::gather(&shifts, gather.clone(), reduction_ids.clone())?;
            let matrix = B::gather(&matrix, gather.clone(), reduction_ids.clone())?;
            let mass_rows = if pointwise_model {
                gather.clone()
            } else {
                vec![0; channel_rows]
            };
            let masses = B::gather(&masses, mass_rows, reduction_ids.clone())?;
            let channel_scales = B::gather(&scales, gather, reduction_ids)?;
            let (values, status) = B::reduce(
                &mut workspace.reducer,
                &shifts,
                &masses,
                &channel_scales,
                &coefficients,
                &matrix,
            )
            .map_err(|e| RusticolError::evaluation(format!("family {} N4: {e}", family.index)))?;
            // The ordinary API raises for any failed channel. Successful statuses
            // retain original point IDs and family/sector ordering.
            for (row, (_, state)) in status.into_iter().enumerate() {
                statuses[row / channels].push((projection_ids[row / channels], state));
            }
            integrated.push(B::reshape(
                &values,
                projection_points,
                channels * 3,
                projection_ids.clone(),
            )?);
            if plan.dimensional_numerators {
                let values = if let Some(reducer) = &mut workspace.rational {
                    B::reduce(
                        reducer,
                        &shifts,
                        &masses,
                        &channel_scales,
                        &coefficients,
                        &matrix,
                    )
                    .map_err(|e| {
                        RusticolError::evaluation(format!(
                            "family {} dimensional remainder: {e}",
                            family.index
                        ))
                    })?
                    .0
                } else {
                    eval::<B>(
                        &mut self.bare,
                        "zero_poles",
                        &B::select(&channel_scales, vec![])?,
                    )?
                };
                rational_integrated.push(B::reshape(
                    &values,
                    projection_points,
                    channels * 3,
                    projection_ids.clone(),
                )?);
            }
        }
        let virtual_value = eval::<B>(
            &mut self.bare,
            "postprocess",
            &B::concat(&integrated.iter().collect::<Vec<_>>())?,
        )?;
        B::require_finite(&virtual_value, "bare virtual assembly", projection_rows_per_point)?;
        let rational = if plan.dimensional_numerators {
            let value = eval::<B>(
                &mut self.bare,
                "postprocess",
                &B::concat(&rational_integrated.iter().collect::<Vec<_>>())?,
            )?;
            B::require_finite(&value, "direct rational assembly", projection_rows_per_point)?;
            Some(value)
        } else {
            None
        };
        let source_values = if let Some(snapshot) = &snapshot {
            Some(
                self.source
                    .as_mut()
                    .expect("captured source owner")
                    .counterterms
                    .evaluate(snapshot, &scales)?,
            )
        } else {
            None
        };
        let coefficient_inputs = if source_values.is_none() {
            Some(B::concat(&[
                &B::gather(&model_arguments, vec![0; points], input.point_ids.clone())?,
                &scales,
            ])?)
        } else {
            None
        };
        let mut amplitudes = vec![born_amplitudes];
        for (current, evaluator) in plan.currents.iter().zip(&mut self.counterterms) {
            let coefficients = scalar_values::<B>(
                &current.scalar,
                evaluator.as_mut(),
                coefficient_inputs.as_ref(),
                source_values.as_ref(),
                "counterterm scalar coefficients and active diagnostics",
            )?;
            let coefficients = B::gather(&coefficients, point_rows.clone(), ids.clone())?;
            let combined = B::concat(&[&bound, &coefficients])?;
            let amplitude = B::execute(
                &mut self.schedules[current.schedule],
                &B::select(&combined, current.input_columns.clone())?,
            )?;
            B::require_finite(&amplitude, "counterterm current", helicities)?;
            amplitudes.push(amplitude);
        }
        for (field, evaluator) in plan.fields.iter().zip(&mut self.fields) {
            let value = scalar_values::<B>(
                field,
                evaluator.as_mut(),
                coefficient_inputs.as_ref(),
                source_values.as_ref(),
                "external-field scalar coefficients and active diagnostics",
            )?;
            amplitudes.push(B::gather(&value, point_rows.clone(), ids.clone())?);
        }
        let projected = B::concat(&amplitudes.iter().collect::<Vec<_>>())?;
        let projected = B::reshape(
            &projected,
            projection_points,
            count(projection_helicities, B::columns(&projected))?,
            projection_ids.clone(),
        )?;
        let mut counterterms = if resolved {
            B::evaluate(
                &mut self.helicity.as_mut().expect("checked helicity kernels").counterterms,
                &projected,
            )?
        } else {
            eval::<B>(&mut self.model, "projector", &projected)?
        };
        B::require_finite(&counterterms, "counterterm projection", projection_rows_per_point)?;
        let width = 3 * sector_count;
        if let Some(rational) = rational {
            counterterms = B::concat(&[
                &B::select(&counterterms, (0..width).collect())?,
                &rational,
                &B::select(&counterterms, (2 * width..3 * width).collect())?,
            ])?;
        }
        let assembled = eval::<B>(
            &mut self.model,
            "assemble",
            &B::concat(&[&virtual_value, &counterterms])?,
        )?;
        B::require_finite(&assembled, "complete virtual assembly", projection_rows_per_point)?;
        let components = [
            B::select(&assembled, (0..width).collect())?,
            virtual_value,
            B::select(&assembled, (width..2 * width).collect())?,
            B::select(&counterterms, (width..2 * width).collect())?,
            B::select(&counterterms, (0..width).collect())?,
            B::select(&counterterms, (2 * width..3 * width).collect())?,
        ];
        let mut values = vec![&born];
        values.extend(components.iter());
        let totals = eval::<B>(&mut self.model, "sum_order_values", &B::concat(&values)?)?;
        B::require_finite(&totals, "sum over completed perturbative orders", projection_rows_per_point)?;
        Ok(CompleteLoopResult {
            point_ids: projection_ids,
            born,
            components,
            totals,
            statuses,
            r2_strategy: plan.r2_strategy.clone(),
            source_revision: snapshot.as_ref().map(|value| value.revision()),
        })
    }
}

fn scalar_values<B: LoopBackend>(
    binding: &ScalarBinding,
    evaluator: Option<&mut B::Evaluator>,
    arguments: Option<&B::Batch>,
    source: Option<&B::Batch>,
    stage: &str,
) -> RusticolResult<B::Batch> {
    match binding {
        ScalarBinding::Evaluator { .. } => {
            let values = B::evaluate(
                evaluator.expect("validated scalar evaluator"),
                arguments.expect("legacy scalar inputs"),
            )?;
            B::require_finite(&values, stage, 1)?;
            B::select(&values, (0..B::columns(&values)).step_by(2).collect())
        }
        ScalarBinding::SourceColumns { columns } => B::select(
            source.expect("checked source scale outputs"),
            columns.clone(),
        ),
    }
}

fn eval<B: LoopBackend>(
    evaluators: &mut BTreeMap<String, B::Evaluator>,
    name: &str,
    input: &B::Batch,
) -> RusticolResult<B::Batch> {
    B::evaluate(
        evaluators
            .get_mut(name)
            .expect("validated evaluator reference"),
        input,
    )
}
fn evaluate_group<B: LoopBackend>(
    schedules: &mut [B::Schedule],
    group: &ScheduleGroup,
    input: &B::Batch,
) -> RusticolResult<Vec<B::Batch>> {
    let mut evaluated = BTreeMap::new();
    for &index in &group.representatives {
        evaluated.insert(
            index,
            B::execute(
                &mut schedules[group.schedules[index]],
                &B::select(input, group.input_columns[index].clone())?,
            )?,
        );
    }
    Ok(group
        .representative_for
        .iter()
        .map(|index| evaluated[index].clone())
        .collect())
}
fn prepare_states<B: LoopBackend>(
    lanes: &B::Batch,
    ids: &[u64],
    plan: &crate::loop_runtime::LoopRuntimePlan,
) -> RusticolResult<B::Batch> {
    let mut blocks = Vec::new();
    let mut original_columns = Vec::new();
    let leg_count = plan.state_legs.len();
    let mut offset = 0;
    for components in [1usize, 4] {
        let legs = plan
            .state_legs
            .iter()
            .filter(|leg| (if leg.kind == "scalar" { 1 } else { 4 }) == components)
            .collect::<Vec<_>>();
        if legs.is_empty() {
            continue;
        }
        let momenta = B::select(
            lanes,
            legs.iter()
                .flat_map(|leg| 4 * leg.index..4 * leg.index + 4)
                .collect(),
        )?;
        let mut state_ids = Vec::new();
        let mut kinds = Vec::new();
        let mut helicities = Vec::new();
        for (row, &id) in ids.iter().enumerate() {
            for leg in &legs {
                state_ids.push(lane_id(id, leg_count, leg.index)?);
                kinds.push(leg.kind.clone());
                helicities.push(plan.helicities[row % plan.helicities.len()][leg.index]);
            }
        }
        let momenta = B::reshape(&momenta, count(ids.len(), legs.len())?, 4, state_ids)?;
        let states = B::states(&momenta, helicities, kinds, components)?;
        blocks.push(B::reshape(
            &states,
            ids.len(),
            components * legs.len(),
            ids.to_vec(),
        )?);
        for (index, leg) in legs.iter().enumerate() {
            for component in 0..components {
                original_columns.push((
                    leg.offset + component,
                    offset + index * components + component,
                ));
            }
        }
        offset += components * legs.len();
    }
    original_columns.sort_unstable();
    B::select(
        &B::concat(&blocks.iter().collect::<Vec<_>>())?,
        original_columns.into_iter().map(|(_, i)| i).collect(),
    )
}
