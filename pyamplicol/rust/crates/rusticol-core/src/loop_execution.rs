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
    fn states_partial(
        momenta: &Self::Batch,
        helicities: Vec<i32>,
        kinds: Vec<String>,
        components: usize,
    ) -> RusticolResult<(Self::Batch, Vec<(u64, String)>)>;
    fn reduce(
        reducer: &mut Self::Reducer,
        shifts: &Self::Batch,
        masses: &Self::Batch,
        scales: &Self::Batch,
        coefficients: &Self::Batch,
        matrix: &Self::Batch,
    ) -> RusticolResult<(Self::Batch, Vec<(u64, String)>)>;
    fn rows(batch: &Self::Batch) -> usize;
    fn nonfinite_rows(batch: &Self::Batch) -> Vec<u64>;
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
#[derive(Clone)]
pub struct LoopInputs {
    pub momenta: Vec<String>,
    pub scales_squared: Vec<String>,
    pub model_values: Vec<String>,
    pub point_ids: Vec<u64>,
}

impl LoopInputs {
    pub fn select_points(&self, ids: &[u64], momentum_width: usize) -> RusticolResult<Self> {
        let positions: BTreeMap<_, _> = self
            .point_ids
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row))
            .collect();
        if positions.len() != self.point_ids.len()
            || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
            || self.momenta.len() != count(self.point_ids.len(), momentum_width)?
            || self.scales_squared.len() != self.point_ids.len()
        {
            return Err(invalid("original exact input layout differs"));
        }
        let rows = ids
            .iter()
            .map(|id| {
                positions
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("selected original point is absent"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            momenta: rows
                .iter()
                .flat_map(|&row| {
                    self.momenta[row * momentum_width..(row + 1) * momentum_width]
                        .iter()
                        .cloned()
                })
                .collect(),
            scales_squared: rows
                .iter()
                .map(|&row| self.scales_squared[row].clone())
                .collect(),
            model_values: self.model_values.clone(),
            point_ids: ids.to_vec(),
        })
    }
}

/// Columns retain the validated order layout. Each virtual sector is ordered
/// finite, single pole, double pole. Components are full, bare, UV, R2, local UV,
/// and external fields. Values remain in their requested native arithmetic.
pub struct CompleteLoopResult<T> {
    pub requested_point_ids: Vec<u64>,
    pub point_ids: Vec<u64>,
    pub born: T,
    pub components: [T; 6],
    pub totals: T,
    pub statuses: Vec<Vec<(u64, String)>>,
    pub r2_strategy: String,
    pub source_revision: Option<u64>,
    pub failures: Vec<LoopLaneFailure>,
}

/// A scheduled numerical channel failure; source identities never come from
/// parsing a reducer exception. Failed numerical values are not exposed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopLaneFailure {
    pub point_id: u64,
    pub helicity: Option<usize>,
    pub order_index: Option<usize>,
    pub family_index: Option<usize>,
    pub external_leg: Option<usize>,
    pub stage: String,
    pub detail: String,
}

impl<T> CompleteLoopResult<T> {
    fn require_complete(self) -> RusticolResult<Self> {
        if self.failures.is_empty() {
            Ok(self)
        } else {
            Err(RusticolError::evaluation(format!(
                "unrecovered loop numerical channels: {:?}",
                self.failures
            )))
        }
    }
}

enum LoopStepError {
    Invalid(RusticolError),
    Numerical(Vec<LoopLaneFailure>),
}
impl From<RusticolError> for LoopStepError {
    fn from(error: RusticolError) -> Self {
        Self::Invalid(error)
    }
}
type StepResult<T> = Result<T, LoopStepError>;

/// Collect every bad row in a numerical stage before dropping whole points.
fn check_finite<B: LoopBackend>(
    values: &B::Batch,
    stage: &str,
    helicities: Option<usize>,
    family_index: Option<usize>,
    failures: &mut Vec<LoopLaneFailure>,
) -> StepResult<()> {
    let bad = B::nonfinite_rows(values);
    if bad.is_empty() {
        return Ok(());
    }
    for id in bad {
        failures.push(LoopLaneFailure {
            point_id: helicities.map_or(id, |h| id / h as u64),
            helicity: helicities.map(|h| (id % h as u64) as usize),
            external_leg: None,
            order_index: None,
            family_index,
            stage: stage.into(),
            detail: "nonfinite numerical value".into(),
        });
    }
    Err(LoopStepError::Numerical(std::mem::take(failures)))
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
                Some(Arc::new(SourceOwner::new(
                    loaded.clone(),
                    &backend,
                    parameters,
                )?))
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
        let source = owner
            .map(|owner| {
                let ordinary = SourceScale::ordinary(&owner, &backend)?;
                let counterterms = SourceScale::counterterms(&owner, &backend)?;
                Ok::<_, RusticolError>(SourceWorkspace {
                    owner,
                    ordinary,
                    counterterms,
                })
            })
            .transpose()?;
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
                    helicity_projector: plan
                        .helicity_projection
                        .as_ref()
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
        let helicity = plan
            .helicity_projection
            .as_ref()
            .map(|projection| {
                Ok::<_, RusticolError>(HelicityWorkspace {
                    born: eval(&projection.born)?,
                    counterterms: eval(&projection.counterterms)?,
                })
            })
            .transpose()?;
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
        self.evaluate_mode(input, snapshot, false)?
            .require_complete()
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
        self.evaluate_mode(input, snapshot, true)?
            .require_complete()
    }
    /// Compact complete physical points and retain typed state, scale and
    /// reduction failures. Invalid original inputs and global formulas still fail.
    pub fn evaluate_partial_with_snapshot(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        resolved: bool,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, snapshot, resolved)
    }
    fn evaluate_mode(
        &mut self,
        mut input: LoopInputs,
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
        let requested = input.point_ids.clone();
        let momentum_width = count(self.loaded.plan().state_legs.len(), 4)?;
        let mut failures = Vec::new();
        loop {
            match self.evaluate_inner(input.clone(), snapshot.clone(), resolved) {
                Ok(mut result) => {
                    failures.append(&mut result.failures);
                    result.requested_point_ids = requested;
                    result.failures = failures;
                    return Ok(result);
                }
                Err(LoopStepError::Invalid(error)) => return Err(error),
                Err(LoopStepError::Numerical(mut bad)) => {
                    let failed: BTreeSet<_> = bad.iter().map(|failure| failure.point_id).collect();
                    let retained: Vec<_> = input
                        .point_ids
                        .iter()
                        .copied()
                        .filter(|id| !failed.contains(id))
                        .collect();
                    if retained.len() == input.point_ids.len() {
                        return Err(invalid(
                            "numerical stage failure made no physical-point progress",
                        ));
                    }
                    failures.append(&mut bad);
                    input = input.select_points(&retained, momentum_width)?;
                }
            }
        }
    }

    fn evaluate_inner(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        resolved: bool,
    ) -> StepResult<CompleteLoopResult<B::Batch>> {
        if snapshot.is_some() && !input.model_values.is_empty() {
            return Err(invalid(
                "source model parameters require named owner updates, not positional model_values",
            )
            .into());
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
            return Err(
                invalid("loop original input shapes or unique point identities differ").into(),
            );
        }
        // Exact ingress below validates every original momentum literal before
        // any row is removed. Scales must also be positive before rounding.
        for value in &input.scales_squared {
            if crate::source_scalar::decimal_rational(value).map_err(|error| invalid(&error))?
                <= symbolica::domains::rational::Rational::from(0)
            {
                return Err(
                    invalid("renormalization scales must be positive exact real values").into(),
                );
            }
        }
        let requested_point_ids = input.point_ids.clone();
        let mut ids = Vec::with_capacity(rows);
        let mut point_rows = Vec::with_capacity(rows);
        for (row, &id) in input.point_ids.iter().enumerate() {
            for h in 0..helicities {
                ids.push(lane_id(id, helicities, h)?);
                point_rows.push(row);
            }
        }
        let physical = self.backend.ingress(
            &input.momenta,
            points,
            momentum_width,
            input.point_ids.clone(),
        )?;
        let lanes = B::gather(&physical, point_rows, ids.clone())?;
        let scales =
            self.backend
                .ingress(&input.scales_squared, points, 1, input.point_ids.clone())?;
        let mut failures = Vec::new();
        let states = prepare_states::<B>(&lanes, &ids, plan, &mut failures)?;
        let scales = finite_rows::<B>(&scales, "renormalization scales", &mut failures)?;
        let (model_arguments, model, pointwise_model, source_values) = if let Some(snapshot) =
            &snapshot
        {
            let source = self.source.as_mut().expect("captured source owner");
            let ModelBinding::Source {
                model_output_indices,
                ..
            } = &plan.model_binding
            else {
                unreachable!("validated source binding");
            };
            let selected = if let Some(view) = &mut source.ordinary {
                let (values, mut bad) = view.evaluate_partial(snapshot, &scales)?;
                for failure in &mut bad {
                    failure.stage = format!("ordinary {}", failure.stage);
                }
                failures.extend(bad);
                values
            } else {
                snapshot.values().clone()
            };
            let (counterterms, mut bad) =
                source.counterterms.evaluate_partial(snapshot, &scales)?;
            for failure in &mut bad {
                failure.stage = format!("counterterm {}", failure.stage);
            }
            failures.extend(bad);
            (
                snapshot.arguments().clone(),
                B::select(&selected, model_output_indices.clone())?,
                source.ordinary.is_some(),
                Some(counterterms),
            )
        } else {
            let arguments =
                self.backend
                    .ingress(&input.model_values, 1, input.model_values.len(), vec![0])?;
            let model = eval::<B>(&mut self.bare, "model_evaluator", &arguments)?;
            B::require_finite(&model, "global model formulas", 1)?;
            (arguments, model, false, None)
        };
        let failed: BTreeSet<_> = failures.iter().map(|f| f.point_id).collect();
        let accepted: Vec<_> = input
            .point_ids
            .iter()
            .copied()
            .filter(|id| !failed.contains(id))
            .collect();
        let input = input.select_points(&accepted, momentum_width)?;
        let points = input.point_ids.len();
        let rows = count(points, helicities)?;
        let mut ids = Vec::with_capacity(rows);
        let mut point_rows = Vec::with_capacity(rows);
        for (row, &id) in input.point_ids.iter().enumerate() {
            for h in 0..helicities {
                ids.push(lane_id(id, helicities, h)?);
                point_rows.push(row);
            }
        }
        let lanes = select_ids::<B>(&lanes, &ids)?;
        let states = select_ids::<B>(&states, &ids)?;
        let scales = select_ids::<B>(&scales, &accepted)?;
        let model = if pointwise_model {
            select_ids::<B>(&model, &accepted)?
        } else {
            model
        };
        let source_values = source_values
            .as_ref()
            .map(|values| select_ids::<B>(values, &accepted))
            .transpose()?;
        let projection_points = if resolved { rows } else { points };
        let projection_helicities = if resolved { 1 } else { helicities };
        let projection_ids = if resolved {
            ids.clone()
        } else {
            input.point_ids.clone()
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
        let born_outputs = evaluate_group::<B>(
            &mut self.schedules,
            &plan.born,
            &bound,
            helicities,
            &mut failures,
        )?;
        let born_amplitudes = B::concat(&born_outputs.iter().collect::<Vec<_>>())?;
        let loop_outputs = evaluate_group::<B>(
            &mut self.schedules,
            &plan.loops,
            &bound,
            helicities,
            &mut failures,
        )?;
        let born_input = B::reshape(
            &born_amplitudes,
            projection_points,
            count(projection_helicities, B::columns(&born_amplitudes))?,
            projection_ids.clone(),
        )?;
        let born = if resolved {
            B::evaluate(
                &mut self
                    .helicity
                    .as_mut()
                    .expect("checked helicity kernels")
                    .born,
                &born_input,
            )?
        } else {
            eval::<B>(&mut self.bare, "born_projector", &born_input)?
        };
        check_finite::<B>(
            &born,
            "Born projection",
            resolved.then_some(helicities),
            None,
            &mut failures,
        )?;
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
                workspace
                    .helicity_projector
                    .as_mut()
                    .expect("checked helicity kernel")
            } else {
                &mut workspace.projector
            };
            let coefficients = B::evaluate(projector, &projection)?;
            check_finite::<B>(
                &coefficients,
                "family numerator projection",
                resolved.then_some(helicities),
                Some(family.index),
                &mut failures,
            )?;
            let exact = self.backend.exact(
                &mut workspace.exact,
                &input.momenta,
                points,
                input.point_ids.clone(),
            )?;
            let n = family.propagators;
            let shifts = B::select(&exact, (0..4 * n).collect())?;
            let mut matrix = B::select(&exact, (4 * n..4 * n + n * n).collect())?;
            let masses = B::evaluate(
                &mut workspace.masses,
                &B::select(&model, family.mass_columns.clone())?,
            )?;
            let mass_check = if pointwise_model {
                masses.clone()
            } else {
                B::gather(&masses, vec![0; points], input.point_ids.clone())?
            };
            check_finite::<B>(
                &mass_check,
                "propagator masses",
                None,
                Some(family.index),
                &mut failures,
            )?;
            if let Some(kinematics) = &family.kinematics {
                if !kinematics.on_shell.is_empty() {
                    let mass_rows = if pointwise_model {
                        (0..points).collect()
                    } else {
                        vec![0; points]
                    };
                    let point_masses = B::gather(&masses, mass_rows, input.point_ids.clone())?;
                    let mut columns: Vec<_> = (0..n * n).collect();
                    for &[entry, mass] in &kinematics.on_shell {
                        columns[entry] = n * n + mass;
                    }
                    matrix = B::select(&B::concat(&[&matrix, &point_masses])?, columns)?;
                }
            }
            // Test only the physical matrix after structural on-shell bindings;
            // discarded raw entries must not cause a false numerical failure.
            check_finite::<B>(
                &shifts,
                "family momentum shifts",
                None,
                Some(family.index),
                &mut failures,
            )?;
            check_finite::<B>(
                &matrix,
                "physical family invariant matrix",
                None,
                Some(family.index),
                &mut failures,
            )?;
            let masses = B::select(&masses, (0..n).collect())?;
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
            let channel_scales = B::gather(&scales, gather, reduction_ids.clone())?;
            let (values, status) = B::reduce(
                &mut workspace.reducer,
                &shifts,
                &masses,
                &channel_scales,
                &coefficients,
                &matrix,
            )
            .map_err(|e| RusticolError::evaluation(format!("family {} N4: {e}", family.index)))?;
            integrated.push(complete_family_rows::<B>(
                values,
                status,
                &projection_ids,
                &reduction_ids,
                &family.sector_indices,
                helicities,
                resolved,
                family.index,
                "ordinary numerator reduction",
                &mut failures,
                &mut statuses,
            )?);
            if plan.dimensional_numerators {
                let (values, status) = if let Some(reducer) = &mut workspace.rational {
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
                } else {
                    (
                        eval::<B>(
                            &mut self.bare,
                            "zero_poles",
                            &B::select(&channel_scales, vec![])?,
                        )?,
                        reduction_ids
                            .iter()
                            .map(|&id| (id, "success".into()))
                            .collect(),
                    )
                };
                rational_integrated.push(complete_family_rows::<B>(
                    values,
                    status,
                    &projection_ids,
                    &reduction_ids,
                    &family.sector_indices,
                    helicities,
                    resolved,
                    family.index,
                    "dimensional remainder reduction",
                    &mut failures,
                    &mut statuses,
                )?);
            }
        }
        let failed_points: BTreeSet<_> = failures.iter().map(|f| f.point_id).collect();
        let accepted: Vec<_> = projection_ids
            .iter()
            .copied()
            .filter(|&id| {
                !failed_points.contains(&(if resolved { id / helicities as u64 } else { id }))
            })
            .collect();
        let born = select_ids::<B>(&born, &accepted)?;
        for values in integrated.iter_mut().chain(&mut rational_integrated) {
            *values = select_ids::<B>(values, &accepted)?;
        }
        let virtual_value = eval::<B>(
            &mut self.bare,
            "postprocess",
            &B::concat(&integrated.iter().collect::<Vec<_>>())?,
        )?;
        check_finite::<B>(
            &virtual_value,
            "bare virtual assembly",
            resolved.then_some(helicities),
            None,
            &mut failures,
        )?;
        let rational = if plan.dimensional_numerators {
            let value = eval::<B>(
                &mut self.bare,
                "postprocess",
                &B::concat(&rational_integrated.iter().collect::<Vec<_>>())?,
            )?;
            check_finite::<B>(
                &value,
                "direct rational assembly",
                resolved.then_some(helicities),
                None,
                &mut failures,
            )?;
            Some(value)
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
                &mut failures,
            )?;
            let coefficients = B::gather(&coefficients, point_rows.clone(), ids.clone())?;
            let combined = B::concat(&[&bound, &coefficients])?;
            let amplitude = B::execute(
                &mut self.schedules[current.schedule],
                &B::select(&combined, current.input_columns.clone())?,
            )?;
            check_finite::<B>(
                &amplitude,
                "counterterm current",
                Some(helicities),
                None,
                &mut failures,
            )?;
            amplitudes.push(amplitude);
        }
        for (field, evaluator) in plan.fields.iter().zip(&mut self.fields) {
            let value = scalar_values::<B>(
                field,
                evaluator.as_mut(),
                coefficient_inputs.as_ref(),
                source_values.as_ref(),
                "external-field scalar coefficients and active diagnostics",
                &mut failures,
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
        let counterterms = if resolved {
            B::evaluate(
                &mut self
                    .helicity
                    .as_mut()
                    .expect("checked helicity kernels")
                    .counterterms,
                &projected,
            )?
        } else {
            eval::<B>(&mut self.model, "projector", &projected)?
        };
        let mut counterterms = select_ids::<B>(&counterterms, &accepted)?;
        check_finite::<B>(
            &counterterms,
            "counterterm projection",
            resolved.then_some(helicities),
            None,
            &mut failures,
        )?;
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
        check_finite::<B>(
            &assembled,
            "complete virtual assembly",
            resolved.then_some(helicities),
            None,
            &mut failures,
        )?;
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
        check_finite::<B>(
            &totals,
            "sum over completed perturbative orders",
            resolved.then_some(helicities),
            None,
            &mut failures,
        )?;
        Ok(CompleteLoopResult {
            requested_point_ids,
            point_ids: accepted,
            born,
            components,
            totals,
            statuses: statuses
                .into_iter()
                .zip(&projection_ids)
                .filter_map(|(status, &id)| {
                    (!failed_points.contains(&(if resolved { id / helicities as u64 } else { id })))
                        .then_some(status)
                })
                .collect(),
            r2_strategy: plan.r2_strategy.clone(),
            source_revision: snapshot.as_ref().map(|value| value.revision()),
            failures,
        })
    }
}

/// Keep successful rows without numeric placeholders. Callers aggregate failures
/// at the physical-point boundary before combining independently compacted data.
pub(crate) fn finite_rows<B: LoopBackend>(
    values: &B::Batch,
    stage: &str,
    failures: &mut Vec<LoopLaneFailure>,
) -> RusticolResult<B::Batch> {
    let failed: BTreeSet<_> = B::nonfinite_rows(values).into_iter().collect();
    for &point_id in &failed {
        failures.push(LoopLaneFailure {
            point_id,
            helicity: None,
            order_index: None,
            family_index: None,
            external_leg: None,
            stage: stage.into(),
            detail: "nonfinite numerical value or active scalar diagnostic".into(),
        });
    }
    let ids: Vec<_> = B::row_ids(values)
        .iter()
        .copied()
        .filter(|id| !failed.contains(id))
        .collect();
    select_ids::<B>(values, &ids)
}

fn select_ids<B: LoopBackend>(values: &B::Batch, ids: &[u64]) -> RusticolResult<B::Batch> {
    let positions: BTreeMap<_, _> = B::row_ids(values)
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row))
        .collect();
    if positions.len() != B::rows(values) {
        return Err(invalid("native result row identities are duplicated"));
    }
    let rows = ids
        .iter()
        .map(|id| {
            positions
                .get(id)
                .copied()
                .ok_or_else(|| invalid("required successful native row is absent"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    B::gather(values, rows, ids.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn complete_family_rows<B: LoopBackend>(
    values: B::Batch,
    mut channel_statuses: Vec<(u64, String)>,
    projection_ids: &[u64],
    channel_ids: &[u64],
    sectors: &[usize],
    helicities: usize,
    resolved: bool,
    family: usize,
    stage: &str,
    failures: &mut Vec<LoopLaneFailure>,
    statuses: &mut [Vec<(u64, String)>],
) -> RusticolResult<B::Batch> {
    if channel_statuses.len() != channel_ids.len()
        || channel_statuses
            .iter()
            .map(|(id, _)| id)
            .ne(channel_ids.iter())
        || B::columns(&values) != 3
    {
        return Err(invalid("reducer status identities or value columns differ"));
    }
    let successful: Vec<_> = channel_statuses
        .iter()
        .filter(|(_, state)| state == "success")
        .map(|(id, _)| *id)
        .collect();
    if B::row_ids(&values) != successful {
        return Err(invalid(
            "compact reducer values differ from successful statuses",
        ));
    }
    let nonfinite: BTreeSet<_> = B::nonfinite_rows(&values).into_iter().collect();
    for (id, state) in &mut channel_statuses {
        if state == "success" && nonfinite.contains(id) {
            *state = "nonfinite reduced Laurent coefficients".into();
        }
    }
    let mut failed = BTreeSet::new();
    for (channel, (_, detail)) in channel_statuses.into_iter().enumerate() {
        let row = channel / sectors.len();
        let projection_id = projection_ids[row];
        statuses[row].push((projection_id, detail.clone()));
        if detail != "success" {
            let point_id = if resolved {
                projection_id / helicities as u64
            } else {
                projection_id
            };
            failed.insert(point_id);
            failures.push(LoopLaneFailure {
                point_id,
                helicity: resolved.then_some((projection_id % helicities as u64) as usize),
                order_index: Some(sectors[channel % sectors.len()]),
                family_index: Some(family),
                external_leg: None,
                stage: stage.into(),
                detail,
            });
        }
    }
    let rows: Vec<_> = projection_ids
        .iter()
        .enumerate()
        .filter_map(|(row, &id)| {
            let point = if resolved { id / helicities as u64 } else { id };
            (!failed.contains(&point)).then_some(row)
        })
        .collect();
    let ids: Vec<_> = rows.iter().map(|&row| projection_ids[row]).collect();
    let channels: Vec<_> = rows
        .into_iter()
        .flat_map(|row| {
            channel_ids[row * sectors.len()..(row + 1) * sectors.len()]
                .iter()
                .copied()
        })
        .collect();
    B::reshape(
        &select_ids::<B>(&values, &channels)?,
        ids.len(),
        count(3, sectors.len())?,
        ids,
    )
}

fn scalar_values<B: LoopBackend>(
    binding: &ScalarBinding,
    evaluator: Option<&mut B::Evaluator>,
    arguments: Option<&B::Batch>,
    source: Option<&B::Batch>,
    stage: &str,
    failures: &mut Vec<LoopLaneFailure>,
) -> StepResult<B::Batch> {
    match binding {
        ScalarBinding::Evaluator { .. } => {
            let values = B::evaluate(
                evaluator.expect("validated scalar evaluator"),
                arguments.expect("legacy scalar inputs"),
            )?;
            check_finite::<B>(&values, stage, None, None, failures)?;
            Ok(B::select(
                &values,
                (0..B::columns(&values)).step_by(2).collect(),
            )?)
        }
        ScalarBinding::SourceColumns { columns } => Ok(B::select(
            source.expect("checked source scale outputs"),
            columns.clone(),
        )?),
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
    helicities: usize,
    failures: &mut Vec<LoopLaneFailure>,
) -> StepResult<Vec<B::Batch>> {
    let mut evaluated = BTreeMap::new();
    for &index in &group.representatives {
        let values = B::execute(
            &mut schedules[group.schedules[index]],
            &B::select(input, group.input_columns[index].clone())?,
        )?;
        check_finite::<B>(
            &values,
            "coefficient current",
            Some(helicities),
            None,
            failures,
        )?;
        evaluated.insert(index, values);
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
    failures: &mut Vec<LoopLaneFailure>,
) -> RusticolResult<B::Batch> {
    let mut blocks = Vec::new();
    let mut original_columns = Vec::new();
    let leg_count = plan.state_legs.len();
    let helicity_count = plan.helicities.len();
    let mut offset = 0;
    let mut failed_points = BTreeSet::new();
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
        for &id in ids {
            for leg in &legs {
                state_ids.push(lane_id(id, leg_count, leg.index)?);
                kinds.push(leg.kind.clone());
                helicities.push(plan.helicities[(id % helicity_count as u64) as usize][leg.index]);
            }
        }
        let momenta = B::reshape(
            &momenta,
            count(ids.len(), legs.len())?,
            4,
            state_ids.clone(),
        )?;
        let (states, statuses) = B::states_partial(&momenta, helicities, kinds, components)?;
        if statuses.iter().map(|(id, _)| id).ne(state_ids.iter()) {
            return Err(invalid(
                "state status identities differ from scheduled rows",
            ));
        }
        let successful: Vec<_> = statuses
            .iter()
            .filter_map(|(id, state)| (state == "success").then_some(*id))
            .collect();
        if B::row_ids(&states) != successful {
            return Err(invalid(
                "compact state values differ from successful statuses",
            ));
        }
        let nonfinite: BTreeSet<_> = B::nonfinite_rows(&states).into_iter().collect();
        for (state_id, detail) in statuses {
            if detail == "success" && !nonfinite.contains(&state_id) {
                continue;
            }
            let id = state_id / leg_count as u64;
            let point_id = id / helicity_count as u64;
            failed_points.insert(point_id);
            failures.push(LoopLaneFailure {
                point_id,
                helicity: Some((id % helicity_count as u64) as usize),
                external_leg: Some((state_id % leg_count as u64) as usize),
                order_index: None,
                family_index: None,
                stage: "external state preparation".into(),
                detail: if detail == "success" {
                    "nonfinite external state".into()
                } else {
                    detail
                },
            });
        }
        blocks.push((
            states,
            legs.iter().map(|leg| leg.index).collect::<Vec<_>>(),
            components,
        ));
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
    let accepted: Vec<_> = ids
        .iter()
        .copied()
        .filter(|id| !failed_points.contains(&(id / helicity_count as u64)))
        .collect();
    let blocks = blocks
        .into_iter()
        .map(|(states, legs, components)| {
            let state_ids = accepted
                .iter()
                .flat_map(|&id| legs.iter().map(move |&leg| lane_id(id, leg_count, leg)))
                .collect::<RusticolResult<Vec<_>>>()?;
            B::reshape(
                &select_ids::<B>(&states, &state_ids)?,
                accepted.len(),
                components * legs.len(),
                accepted.clone(),
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    original_columns.sort_unstable();
    B::select(
        &B::concat(&blocks.iter().collect::<Vec<_>>())?,
        original_columns.into_iter().map(|(_, i)| i).collect(),
    )
}
