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
    type DualBornWeights;
    type ExactEvaluator;
    type Schedule;
    type Reducer;
    fn arithmetic(&self) -> &str;

    fn evaluator(&self, state: &[u8]) -> RusticolResult<Self::Evaluator>;
    fn dual_born_weights(
        &self,
        state: &[u8],
        rectangle: Option<&crate::loop_runtime::RectangularDualWeights>,
        constants: Option<&[u8]>,
        columns: &[[usize; 3]],
    ) -> RusticolResult<Self::DualBornWeights>;
    fn evaluate_dual_born_weights(
        weights: &mut Self::DualBornWeights,
        input: &Self::Batch,
    ) -> RusticolResult<Self::Batch>;
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
    /// Literal complex vector sources, keyed by one-based physical leg. Each
    /// value contains four components per physical point; no rounding precedes
    /// numerical ingress and subset replay keeps these original strings.
    pub spin_sources: BTreeMap<usize, Vec<[String; 2]>>,
}

impl LoopInputs {
    pub fn select_points(&self, ids: &[u64], momentum_width: usize) -> RusticolResult<Self> {
        let positions: BTreeMap<_, _> = self
            .point_ids
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row))
            .collect();
        let source_width = count(self.point_ids.len(), 4)?;
        if positions.len() != self.point_ids.len()
            || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
            || self.momenta.len() != count(self.point_ids.len(), momentum_width)?
            || self.scales_squared.len() != self.point_ids.len()
            || self
                .spin_sources
                .values()
                .any(|values| values.len() != source_width)
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
            spin_sources: self
                .spin_sources
                .iter()
                .map(|(&leg, values)| {
                    (
                        leg,
                        rows.iter()
                            .flat_map(|&row| values[4 * row..4 * (row + 1)].iter().cloned())
                            .collect(),
                    )
                })
                .collect(),
        })
    }
}

/// Columns retain the validated order layout. Each virtual sector is ordered
/// finite, single pole, double pole. Components are full, bare, UV, R2, local UV,
/// and external fields. Values remain in their requested native arithmetic.
pub struct CompleteLoopResult<T> {
    pub primary_color_contraction: crate::loop_runtime::ColorWeightExecution,
    pub requested_point_ids: Vec<u64>,
    /// Successful original physical IDs, one per point in both inclusive and
    /// resolved output. Numerical batches use dense rows in point/helicity/colour
    /// order; opaque caller IDs never encode those additional axes.
    pub point_ids: Vec<u64>,
    pub born: T,
    pub components: [T; 6],
    pub totals: T,
    /// Per-numerical-row reducer statuses, labelled by the final dense row ID.
    pub statuses: Vec<Vec<(u64, String)>>,
    pub r2_strategy: String,
    pub source_revision: Option<u64>,
    pub directed: Option<DirectedCrossTerms<T>>,
    pub failures: Vec<LoopLaneFailure>,
}

/// Both integrated cross terms, retaining history/order/pole columns. Histories
/// are never summed and the saved labels identify their exact ordered operators.
pub struct DirectedCrossTerms<T> {
    pub history_ids: Vec<String>,
    pub spin_legs: Vec<usize>,
    pub first: T,
    pub second: T,
}

/// Presence of directed kernels never changes ordinary evaluation semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopObservable {
    Inclusive,
    Resolved,
    Directed,
}
impl LoopObservable {
    pub fn from_resolved(resolved: bool) -> Self {
        if resolved {
            Self::Resolved
        } else {
            Self::Inclusive
        }
    }
}

/// A scheduled numerical channel failure; source identities never come from
/// parsing a reducer exception. Failed numerical values are not exposed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct LoopLaneFailure {
    pub point_id: u64,
    pub helicity: Option<usize>,
    pub color_component_index: Option<usize>,
    pub history_index: Option<usize>,
    /// Zero denotes the first integrated cross term, one the second.
    pub direction: Option<usize>,
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
    check_component_finite::<B>(values, stage, helicities, None, family_index, failures)
}

fn check_component_finite<B: LoopBackend>(
    values: &B::Batch,
    stage: &str,
    helicities: Option<usize>,
    components: Option<usize>,
    family_index: Option<usize>,
    failures: &mut Vec<LoopLaneFailure>,
) -> StepResult<()> {
    let bad = B::nonfinite_rows(values);
    if bad.is_empty() {
        return Ok(());
    }
    for id in bad {
        failures.push(LoopLaneFailure {
            point_id: helicities.map_or(id, |h| id / components.unwrap_or(1) as u64 / h as u64),
            helicity: helicities.map(|h| (id / components.unwrap_or(1) as u64 % h as u64) as usize),
            color_component_index: components.map(|c| (id % c as u64) as usize),
            history_index: None,
            direction: None,
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
    directed_projector: Option<B::Evaluator>,
    spin_projectors: Vec<B::Evaluator>,
    reducer: B::Reducer,
    rational: Option<B::Reducer>,
}

struct HelicityWorkspace<B: LoopBackend> {
    born: B::Evaluator,
    counterterms: B::Evaluator,
    dual_born_weights: Option<B::DualBornWeights>,
}

struct SourceWorkspace<B: LoopBackend> {
    owner: Arc<SourceOwner<B>>,
    ordinary: Option<SourceScale<B>>,
    counterterms: SourceScale<B>,
}

struct DirectedWorkspace<B: LoopBackend> {
    born: B::Evaluator,
    dual_born_weights: Option<B::DualBornWeights>,
    postprocess: B::Evaluator,
    counterterms: B::Evaluator,
    assemble: B::Evaluator,
    sum_order_values: B::Evaluator,
}

struct SpinWorkspace<B: LoopBackend> {
    directed: DirectedWorkspace<B>,
    phases: Vec<B::Evaluator>,
}

/// Fresh mutable arithmetic workspaces over one immutable, validated descriptor.
/// No graphs, symbolic optimization, or model reimport occur during binding.
pub struct LoopExecutor<B: LoopBackend> {
    loaded: Arc<LoadedLoopRuntime>,
    backend: B,
    schedules: Vec<B::Schedule>,
    families: Vec<FamilyWorkspace<B>>,
    bare: BTreeMap<String, B::Evaluator>,
    dual_born_weights: Option<B::DualBornWeights>,
    model: BTreeMap<String, B::Evaluator>,
    counterterms: Vec<Option<B::Evaluator>>,
    fields: Vec<Option<B::Evaluator>>,
    source: Option<SourceWorkspace<B>>,
    helicity: Option<HelicityWorkspace<B>>,
    directed: Option<DirectedWorkspace<B>>,
    spin: Vec<SpinWorkspace<B>>,
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
        owner.require_backend(&backend)?;
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
        let bind_weights = |w: &crate::loop_runtime::DualBornWeights| {
            backend.dual_born_weights(
                loaded.evaluator_state(&w.state)?,
                w.rectangular.as_ref(),
                w.constants_state
                    .as_ref()
                    .map(|name| loaded.evaluator_state(name))
                    .transpose()?,
                &w.columns,
            )
        };
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
                    directed_projector: plan
                        .directed_projection
                        .as_ref()
                        .map(|projection| eval(&projection.families[index]))
                        .transpose()?,
                    spin_projectors: plan
                        .spin_projections
                        .iter()
                        .flatten()
                        .map(|projection| eval(&projection.directed.families[index]))
                        .collect::<RusticolResult<_>>()?,
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
                    dual_born_weights: projection
                        .dual_born_weights
                        .as_ref()
                        .map(&bind_weights)
                        .transpose()?,
                })
            })
            .transpose()?;
        let bind_directed = |projection: &crate::loop_runtime::DirectedProjection| {
            Ok::<_, RusticolError>(DirectedWorkspace {
                born: eval(&projection.born)?,
                dual_born_weights: projection
                    .dual_born_weights
                    .as_ref()
                    .map(&bind_weights)
                    .transpose()?,
                postprocess: eval(&projection.postprocess)?,
                counterterms: eval(&projection.counterterms)?,
                assemble: eval(&projection.assemble)?,
                sum_order_values: eval(&projection.sum_order_values)?,
            })
        };
        let directed = plan
            .directed_projection
            .as_ref()
            .map(&bind_directed)
            .transpose()?;
        let spin = plan
            .spin_projections
            .iter()
            .flatten()
            .map(|class| {
                Ok(SpinWorkspace {
                    directed: bind_directed(&class.directed)?,
                    phases: class
                        .phases
                        .iter()
                        .map(|phase| eval(phase))
                        .collect::<RusticolResult<_>>()?,
                })
            })
            .collect::<RusticolResult<_>>()?;
        let dual_born_weights = plan
            .dual_born_weights
            .as_ref()
            .map(&bind_weights)
            .transpose()?;
        Ok(Self {
            loaded,
            backend,
            schedules,
            families,
            bare,
            dual_born_weights,
            model,
            counterterms,
            fields,
            source,
            helicity,
            directed,
            spin,
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
        self.evaluate_mode(input, snapshot, LoopObservable::Inclusive)?
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
        self.evaluate_mode(input, snapshot, LoopObservable::Resolved)?
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
        self.evaluate_mode(input, snapshot, LoopObservable::from_resolved(resolved))
    }
    pub fn evaluate_directed(
        &mut self,
        input: LoopInputs,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, None, LoopObservable::Directed)?
            .require_complete()
    }
    pub fn evaluate_observable_partial_with_snapshot(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        observable: LoopObservable,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, snapshot, observable)
    }
    pub fn evaluate_observable_with_snapshot(
        &mut self,
        input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        observable: LoopObservable,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        self.evaluate_mode(input, snapshot, observable)?
            .require_complete()
    }
    fn evaluate_mode(
        &mut self,
        mut input: LoopInputs,
        snapshot: Option<Arc<SourceSnapshot<B::Batch>>>,
        observable: LoopObservable,
    ) -> RusticolResult<CompleteLoopResult<B::Batch>> {
        let resolved = observable == LoopObservable::Resolved;
        if !input.spin_sources.is_empty() {
            if observable != LoopObservable::Directed {
                return Err(invalid("spin sources require explicit directed evaluation"));
            }
            let legs = input.spin_sources.keys().copied().collect::<Vec<_>>();
            if !self
                .loaded
                .plan()
                .spin_projections
                .iter()
                .flatten()
                .any(|p| p.legs == legs)
            {
                return Err(RusticolError::unsupported_runtime_capability(
                    "joint loop spin source class",
                    "this exact joint source class has no saved projection kernels",
                ));
            }
            let width = count(input.point_ids.len(), 4)?;
            if input
                .spin_sources
                .values()
                .any(|values| values.len() != width)
            {
                return Err(invalid("spin source rows differ from physical point count"));
            }
        }
        if observable == LoopObservable::Directed && self.directed.is_none() {
            return Err(RusticolError::unsupported_runtime_capability(
                "directed-complex loop evaluation",
                "this artifact has no directed projection kernels",
            ));
        }
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
        if requested.iter().collect::<BTreeSet<_>>().len() != requested.len() {
            return Err(invalid("loop original point identities are not unique"));
        }
        // Use buffer ordinals for every scheduled PH/PHC/order channel. Original
        // u64 identities are opaque and may use their complete representable
        // range, independently of the number of generated channels.
        input.point_ids = dense_ids(requested.len())?;
        let rows_per_point = if resolved {
            self.loaded.plan().resolved_rows_per_point()?
        } else {
            1
        };
        let momentum_width = count(self.loaded.plan().state_legs.len(), 4)?;
        let mut failures = Vec::new();
        loop {
            match self.evaluate_inner(input.clone(), snapshot.clone(), observable) {
                Ok(mut result) => {
                    failures.append(&mut result.failures);
                    let original_id = |ordinal: u64| {
                        usize::try_from(ordinal)
                            .ok()
                            .and_then(|index| requested.get(index))
                            .copied()
                            .ok_or_else(|| invalid("foreign internal loop point identity"))
                    };
                    for failure in &mut failures {
                        failure.point_id = original_id(failure.point_id)?;
                    }
                    let rows = result.point_ids.len();
                    if rows % rows_per_point != 0 || result.statuses.len() != rows {
                        return Err(invalid("completed loop point/row layout differs"));
                    }
                    let physical_ids = result
                        .point_ids
                        .chunks(rows_per_point)
                        .map(|group| {
                            let ordinal = group[0] / rows_per_point as u64;
                            for (row, &id) in group.iter().enumerate() {
                                if id != lane_id(ordinal, rows_per_point, row)? {
                                    return Err(invalid("completed loop channel order differs"));
                                }
                            }
                            original_id(ordinal)
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let row_ids = dense_ids(rows)?;
                    let dense = |batch: &B::Batch| {
                        if B::rows(batch) != rows
                            || B::row_ids(batch) != result.point_ids.as_slice()
                        {
                            return Err(invalid("completed loop numerical rows differ"));
                        }
                        B::reshape(batch, rows, B::columns(batch), row_ids.clone())
                    };
                    result.born = dense(&result.born)?;
                    for component in &mut result.components {
                        *component = dense(component)?;
                    }
                    result.totals = dense(&result.totals)?;
                    if let Some(directed) = &mut result.directed {
                        directed.first = dense(&directed.first)?;
                        directed.second = dense(&directed.second)?;
                    }
                    for (&id, statuses) in row_ids.iter().zip(&mut result.statuses) {
                        for (status_id, _) in statuses {
                            *status_id = id;
                        }
                    }
                    result.point_ids = physical_ids;
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
        observable: LoopObservable,
    ) -> StepResult<CompleteLoopResult<B::Batch>> {
        let resolved = observable == LoopObservable::Resolved;
        let directed = observable == LoopObservable::Directed;
        if snapshot.is_some() && !input.model_values.is_empty() {
            return Err(invalid(
                "source model parameters require named owner updates, not positional model_values",
            )
            .into());
        }
        let plan = self.loaded.plan();
        let source_legs = input.spin_sources.keys().copied().collect::<Vec<_>>();
        let spin_index = if source_legs.is_empty() {
            None
        } else {
            plan.spin_projections
                .as_ref()
                .and_then(|classes| classes.iter().position(|p| p.legs == source_legs))
        };
        let helicity_layout = spin_index.map_or(plan.helicities.as_slice(), |index| {
            plan.spin_projections.as_ref().expect("checked spin class")[index]
                .helicities
                .as_slice()
        });
        let histories = if directed {
            plan.directed_projection
                .as_ref()
                .expect("checked directed kernels")
                .histories
                .len()
        } else {
            1
        };
        let directions = if directed { count(2, histories)? } else { 1 };
        let points = input.point_ids.len();
        let legs = plan.state_legs.len();
        let helicities = helicity_layout.len();
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
        if let Some(failure) = snapshot.as_ref().and_then(|s| s.binding_failure()) {
            // A global model failure must not conceal invalid original input
            // syntax. Validate exact literals before reporting numerical lanes.
            for value in input
                .momenta
                .iter()
                .chain(input.spin_sources.values().flatten().flatten())
            {
                crate::source_scalar::decimal_rational(value).map_err(|error| invalid(&error))?;
            }
            if points != 0 {
                return Err(LoopStepError::Numerical(
                    input
                        .point_ids
                        .iter()
                        .map(|&point_id| LoopLaneFailure {
                            point_id,
                            helicity: None,
                            color_component_index: None,
                            history_index: None,
                            direction: None,
                            order_index: None,
                            family_index: None,
                            external_leg: None,
                            stage: failure.stage.clone(),
                            detail: "global source realization is nonfinite in this arithmetic"
                                .into(),
                        })
                        .collect(),
                ));
            }
            // No fake successful model values are needed to return correctly
            // shaped empty output after all original points were rejected.
            let empty = |columns| self.backend.ingress(&[], 0, columns, vec![]);
            let width = count(count(3, plan.order_layout.r#virtual.len())?, histories)?;
            let component = empty(width)?;
            return Ok(CompleteLoopResult {
                primary_color_contraction: plan.color_weight_execution(
                    self.backend.arithmetic(),
                    resolved,
                    directed,
                    spin_index,
                ),
                requested_point_ids: vec![],
                point_ids: vec![],
                born: empty(count(plan.order_layout.born.len(), histories)?)?,
                components: std::array::from_fn(|_| component.clone()),
                totals: empty(count(
                    1 + 3 * (6 + if directed { 2 } else { 0 }),
                    histories,
                )?)?,
                statuses: vec![],
                r2_strategy: plan.r2_strategy.clone(),
                source_revision: snapshot.as_ref().map(|s| s.revision()),
                directed: directed.then(|| DirectedCrossTerms {
                    history_ids: plan
                        .directed_projection
                        .as_ref()
                        .expect("checked directed kernels")
                        .histories
                        .iter()
                        .map(|h| h.id.clone())
                        .collect(),
                    spin_legs: source_legs,
                    first: component.clone(),
                    second: component,
                }),
                failures: vec![],
            });
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
        let lanes = B::gather(&physical, point_rows.clone(), ids.clone())?;
        let scales =
            self.backend
                .ingress(&input.scales_squared, points, 1, input.point_ids.clone())?;
        let mut failures = Vec::new();
        let mut sources = BTreeMap::new();
        if let Some(index) = spin_index {
            for ((&leg, values), phase) in
                input.spin_sources.iter().zip(&mut self.spin[index].phases)
            {
                let values =
                    self.backend
                        .ingress_complex(values, points, 4, input.point_ids.clone())?;
                check_finite::<B>(&values, "spin source ingress", None, None, &mut failures)?;
                let values = B::evaluate(phase, &values)?;
                check_finite::<B>(&values, "spin source phase", None, None, &mut failures)?;
                sources.insert(leg, B::gather(&values, point_rows.clone(), ids.clone())?);
            }
        }
        let states =
            prepare_states::<B>(&lanes, &ids, plan, helicity_layout, &sources, &mut failures)?;
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
                snapshot.values()?.clone()
            };
            let (counterterms, mut bad) =
                source.counterterms.evaluate_partial(snapshot, &scales)?;
            for failure in &mut bad {
                failure.stage = format!("counterterm {}", failure.stage);
            }
            failures.extend(bad);
            (
                snapshot.arguments()?.clone(),
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
        let projection_input_points = if resolved { rows } else { points };
        let projection_helicities = if resolved { 1 } else { helicities };
        let projection_input_ids = if resolved {
            ids.clone()
        } else {
            input.point_ids.clone()
        };
        let color_components = if resolved {
            plan.resolved_color_component_count()
        } else {
            1
        };
        let projection_points = count(projection_input_points, color_components)?;
        let resolved_rows_per_point = count(helicities, color_components)?;
        let mut projection_ids = Vec::with_capacity(projection_points);
        for &id in &projection_input_ids {
            for component in 0..color_components {
                projection_ids.push(lane_id(id, color_components, component)?);
            }
        }
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
            projection_input_points,
            count(projection_helicities, B::columns(&born_amplitudes))?,
            projection_input_ids.clone(),
        )?;
        let mut directed_workspace = if directed {
            Some(if let Some(index) = spin_index {
                &mut self.spin[index].directed
            } else {
                self.directed.as_mut().expect("checked directed kernels")
            })
        } else {
            None
        };
        let born = if directed {
            B::evaluate(
                &mut directed_workspace
                    .as_mut()
                    .expect("checked directed kernels")
                    .born,
                &born_input,
            )?
        } else if resolved {
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
        let born = B::reshape(
            &born,
            projection_points,
            count(histories, plan.order_layout.born.len())?,
            projection_ids.clone(),
        )?;
        check_component_finite::<B>(
            &born,
            "Born projection",
            resolved.then_some(helicities),
            resolved.then_some(color_components),
            None,
            &mut failures,
        )?;
        let sector_count = plan.order_layout.r#virtual.len();
        let weight_plan = if directed {
            if let Some(index) = spin_index {
                plan.spin_projections.as_ref().expect("checked spin class")[index]
                    .directed
                    .dual_born_weights
                    .as_ref()
            } else {
                plan.directed_projection
                    .as_ref()
                    .expect("checked directed kernels")
                    .dual_born_weights
                    .as_ref()
            }
        } else if resolved {
            plan.helicity_projection
                .as_ref()
                .expect("checked helicity kernels")
                .dual_born_weights
                .as_ref()
        } else {
            plan.dual_born_weights.as_ref()
        };
        let weight_evaluator = if directed {
            directed_workspace
                .as_mut()
                .expect("checked directed kernels")
                .dual_born_weights
                .as_mut()
        } else if resolved {
            self.helicity
                .as_mut()
                .expect("checked helicity kernels")
                .dual_born_weights
                .as_mut()
        } else {
            self.dual_born_weights.as_mut()
        };
        let dual_weights = weight_evaluator
            .map(|evaluator| B::evaluate_dual_born_weights(evaluator, &born_amplitudes))
            .transpose()?;
        if let Some(weights) = &dual_weights {
            check_finite::<B>(
                weights,
                "dual Born weights",
                Some(helicities),
                None,
                &mut failures,
            )?;
        }
        let mut integrated = Vec::new();
        let mut rational_integrated = Vec::new();
        let mut statuses = vec![Vec::new(); projection_points];
        for (family_index, (family, workspace)) in
            plan.families.iter().zip(&mut self.families).enumerate()
        {
            let selected = match (&dual_weights, weight_plan) {
                (Some(weights), Some(plan)) => Some(B::select(
                    weights,
                    plan.family_columns[family_index].clone(),
                )?),
                (None, None) => None,
                _ => return Err(invalid("dual Born weight workspace differs from its plan").into()),
            };
            let mut parts = vec![selected.as_ref().unwrap_or(&born_amplitudes)];
            parts.extend(family.graph_indices.iter().map(|&i| &loop_outputs[i]));
            let projection = B::concat(&parts)?;
            let projection = B::reshape(
                &projection,
                projection_input_points,
                count(projection_helicities, B::columns(&projection))?,
                projection_input_ids.clone(),
            )?;
            let projector = if directed {
                if let Some(index) = spin_index {
                    &mut workspace.spin_projectors[index]
                } else {
                    workspace
                        .directed_projector
                        .as_mut()
                        .expect("checked directed kernel")
                }
            } else if resolved {
                workspace
                    .helicity_projector
                    .as_mut()
                    .expect("checked helicity kernel")
            } else {
                &mut workspace.projector
            };
            let coefficients = B::evaluate(projector, &projection)?;
            let coefficients = B::reshape(
                &coefficients,
                projection_points,
                count(
                    count(directions, family.sector_indices.len())?,
                    family.powers.len(),
                )?,
                projection_ids.clone(),
            )?;
            check_component_finite::<B>(
                &coefficients,
                "family numerator projection",
                resolved.then_some(helicities),
                resolved.then_some(color_components),
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
            let channels = count(directions, family.sector_indices.len())?;
            let channel_rows = count(projection_points, channels)?;
            let mut reduction_ids = Vec::with_capacity(channel_rows);
            let mut gather = Vec::with_capacity(channel_rows);
            for (row, &id) in projection_ids.iter().enumerate() {
                for direction in 0..directions {
                    for &sector in &family.sector_indices {
                        reduction_ids.push(lane_id(
                            id,
                            count(directions, sector_count)?,
                            direction * sector_count + sector,
                        )?);
                        gather.push(if resolved {
                            row / resolved_rows_per_point
                        } else {
                            row
                        });
                    }
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
                directions,
                directed,
                helicities,
                color_components,
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
                    directions,
                    directed,
                    helicities,
                    color_components,
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
                !failed_points.contains(
                    &(if resolved {
                        id / resolved_rows_per_point as u64
                    } else {
                        id
                    }),
                )
            })
            .collect();
        let born = select_ids::<B>(&born, &accepted)?;
        for values in integrated.iter_mut().chain(&mut rational_integrated) {
            *values = select_ids::<B>(values, &accepted)?;
        }
        let integrated_input = B::concat(&integrated.iter().collect::<Vec<_>>())?;
        let virtual_value = if directed {
            B::evaluate(
                &mut directed_workspace
                    .as_mut()
                    .expect("checked directed kernels")
                    .postprocess,
                &integrated_input,
            )?
        } else {
            eval::<B>(&mut self.bare, "postprocess", &integrated_input)?
        };
        check_component_finite::<B>(
            &virtual_value,
            "bare virtual assembly",
            resolved.then_some(helicities),
            resolved.then_some(color_components),
            None,
            &mut failures,
        )?;
        let rational = if plan.dimensional_numerators {
            let value = eval::<B>(
                &mut self.bare,
                "postprocess",
                &B::concat(&rational_integrated.iter().collect::<Vec<_>>())?,
            )?;
            check_component_finite::<B>(
                &value,
                "direct rational assembly",
                resolved.then_some(helicities),
                resolved.then_some(color_components),
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
            projection_input_points,
            count(projection_helicities, B::columns(&projected))?,
            projection_input_ids.clone(),
        )?;
        let counterterms = if directed {
            B::evaluate(
                &mut directed_workspace
                    .as_mut()
                    .expect("checked directed kernels")
                    .counterterms,
                &projected,
            )?
        } else if resolved {
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
        let counterterms = B::reshape(
            &counterterms,
            projection_points,
            count(
                if directed { count(18, histories)? } else { 9 },
                sector_count,
            )?,
            projection_ids.clone(),
        )?;
        let mut counterterms = select_ids::<B>(&counterterms, &accepted)?;
        check_component_finite::<B>(
            &counterterms,
            "counterterm projection",
            resolved.then_some(helicities),
            resolved.then_some(color_components),
            None,
            &mut failures,
        )?;
        let width = count(count(3, sector_count)?, histories)?;
        if let Some(rational) = rational {
            counterterms = B::concat(&[
                &B::select(&counterterms, (0..width).collect())?,
                &rational,
                &B::select(&counterterms, (2 * width..3 * width).collect())?,
            ])?;
        }
        let assembled = if directed {
            let cross_terms = B::select(&virtual_value, (width..3 * width).collect())?;
            B::evaluate(
                &mut directed_workspace
                    .as_mut()
                    .expect("checked directed kernels")
                    .assemble,
                &B::concat(&[&cross_terms, &counterterms])?,
            )?
        } else {
            eval::<B>(
                &mut self.model,
                "assemble",
                &B::concat(&[&virtual_value, &counterterms])?,
            )?
        };
        check_component_finite::<B>(
            &assembled,
            "complete virtual assembly",
            resolved.then_some(helicities),
            resolved.then_some(color_components),
            None,
            &mut failures,
        )?;
        let (components, directed_values) = if directed {
            (
                [
                    B::select(&assembled, (0..width).collect())?,
                    B::select(&virtual_value, (0..width).collect())?,
                    B::select(&assembled, (width..2 * width).collect())?,
                    B::select(&assembled, (2 * width..3 * width).collect())?,
                    B::select(&assembled, (3 * width..4 * width).collect())?,
                    B::select(&assembled, (4 * width..5 * width).collect())?,
                ],
                Some(DirectedCrossTerms {
                    spin_legs: source_legs,
                    history_ids: plan
                        .directed_projection
                        .as_ref()
                        .expect("checked directed kernels")
                        .histories
                        .iter()
                        .map(|history| history.id.clone())
                        .collect(),
                    first: B::select(&assembled, (5 * width..6 * width).collect())?,
                    second: B::select(&assembled, (6 * width..7 * width).collect())?,
                }),
            )
        } else {
            (
                [
                    B::select(&assembled, (0..width).collect())?,
                    virtual_value,
                    B::select(&assembled, (width..2 * width).collect())?,
                    B::select(&counterterms, (width..2 * width).collect())?,
                    B::select(&counterterms, (0..width).collect())?,
                    B::select(&counterterms, (2 * width..3 * width).collect())?,
                ],
                None,
            )
        };
        let mut values = vec![&born];
        values.extend(components.iter());
        let totals = if let Some(cross) = &directed_values {
            values.extend([&cross.first, &cross.second]);
            B::evaluate(
                &mut directed_workspace
                    .as_mut()
                    .expect("checked directed kernels")
                    .sum_order_values,
                &B::concat(&values)?,
            )?
        } else {
            eval::<B>(&mut self.model, "sum_order_values", &B::concat(&values)?)?
        };
        check_component_finite::<B>(
            &totals,
            "sum over completed perturbative orders",
            resolved.then_some(helicities),
            resolved.then_some(color_components),
            None,
            &mut failures,
        )?;
        Ok(CompleteLoopResult {
            primary_color_contraction: plan.color_weight_execution(
                self.backend.arithmetic(),
                resolved,
                directed,
                spin_index,
            ),
            requested_point_ids,
            point_ids: accepted,
            born,
            components,
            totals,
            statuses: statuses
                .into_iter()
                .zip(&projection_ids)
                .filter_map(|(status, &id)| {
                    (!failed_points.contains(
                        &(if resolved {
                            id / resolved_rows_per_point as u64
                        } else {
                            id
                        }),
                    ))
                    .then_some(status)
                })
                .collect(),
            r2_strategy: plan.r2_strategy.clone(),
            source_revision: snapshot.as_ref().map(|value| value.revision()),
            directed: directed_values,
            failures,
        })
    }
}

fn dense_ids(rows: usize) -> RusticolResult<Vec<u64>> {
    (0..rows)
        .map(|row| u64::try_from(row).map_err(|_| invalid("loop row dimension exceeds u64")))
        .collect()
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
            color_component_index: None,
            order_index: None,
            family_index: None,
            history_index: None,
            direction: None,
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
    directions: usize,
    directed: bool,
    helicities: usize,
    color_components: usize,
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
        let row = channel / count(directions, sectors.len())?;
        let projection_id = projection_ids[row];
        statuses[row].push((projection_id, detail.clone()));
        if detail != "success" {
            let point_id = if resolved {
                projection_id / color_components as u64 / helicities as u64
            } else {
                projection_id
            };
            failed.insert(point_id);
            failures.push(LoopLaneFailure {
                point_id,
                helicity: resolved.then_some(
                    (projection_id / color_components as u64 % helicities as u64) as usize,
                ),
                color_component_index: resolved
                    .then_some((projection_id % color_components as u64) as usize),
                order_index: Some(sectors[channel % sectors.len()]),
                family_index: Some(family),
                history_index: directed.then_some(channel / sectors.len() % directions / 2),
                direction: directed.then_some(channel / sectors.len() % 2),
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
            let point = if resolved {
                id / color_components as u64 / helicities as u64
            } else {
                id
            };
            (!failed.contains(&point)).then_some(row)
        })
        .collect();
    let ids: Vec<_> = rows.iter().map(|&row| projection_ids[row]).collect();
    let channels: Vec<_> = rows
        .into_iter()
        .flat_map(|row| {
            channel_ids[row * sectors.len() * directions..(row + 1) * sectors.len() * directions]
                .iter()
                .copied()
        })
        .collect();
    B::reshape(
        &select_ids::<B>(&values, &channels)?,
        ids.len(),
        count(count(3, sectors.len())?, directions)?,
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
    helicity_layout: &[Vec<i32>],
    sources: &BTreeMap<usize, B::Batch>,
    failures: &mut Vec<LoopLaneFailure>,
) -> RusticolResult<B::Batch> {
    let mut blocks = Vec::new();
    let mut original_columns = Vec::new();
    let leg_count = plan.state_legs.len();
    let helicity_count = helicity_layout.len();
    let mut offset = 0;
    let mut failed_points = BTreeSet::new();
    for components in [1usize, 4] {
        let legs = plan
            .state_legs
            .iter()
            .filter(|leg| {
                !sources.contains_key(&(leg.index + 1))
                    && (if leg.kind == "scalar" { 1 } else { 4 }) == components
            })
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
                helicities.push(helicity_layout[(id % helicity_count as u64) as usize][leg.index]);
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
                color_component_index: None,
                history_index: None,
                direction: None,
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
    let mut blocks = blocks
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
    for (&label, source) in sources {
        let leg = &plan.state_legs[label - 1];
        blocks.push(select_ids::<B>(source, &accepted)?);
        for component in 0..4 {
            original_columns.push((leg.offset + component, offset + component));
        }
        offset += 4;
    }
    original_columns.sort_unstable();
    B::select(
        &B::concat(&blocks.iter().collect::<Vec<_>>())?,
        original_columns.into_iter().map(|(_, i)| i).collect(),
    )
}

/// The native evaluation path of v2 plans (DESIGN.md §5.1, WP2.1).
///
/// v2 plans replace the per-graph schedule of [`LoopExecutor::evaluate_inner`]
/// by the shared recursion: per chunk of SIMD lanes the seeds are formed, then
/// family buckets `(propagators, rank)` are streamed in chunks of families —
/// fused factors of the chunk, families dressed on parallel threads
/// (schedule B, or F as the oracle), closures scattered into per-point
/// lanes with exact invariants — into the reduction backend, which owns the
/// NinjaRS engines and master caches. Points whose numerators or reductions
/// fail are compacted out of the later buckets and reported, never branched
/// on inside a kernel. Counterterm, R2 and field components are local
/// insertions assembled by the caller.
#[cfg(feature = "loop-v2")]
pub mod v2 {
    use crate::loop_v2::closure::FamilyNumerator;
    use crate::loop_v2::dress::{Carrier, ChunkInputs, LaneData, Reads, V2Batch};
    use crate::loop_v2::forward::{forward_factors, run_forward};
    pub use crate::loop_v2::fourier::SeedFourier;
    use crate::loop_v2::fourier::seeds_fourier;
    use crate::loop_v2::merged::{DEFAULT_PART_CHUNKS, MergedStats, parts, run_merged};
    use crate::loop_v2::plan::{LoopProgramV4, Merged};
    use crate::loop_v2::retest::{ReductionCheck, add_to_budget, deviation};
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::run_family;
    use crate::{RusticolError, RusticolResult};
    use rayon::prelude::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Instant;
    use symbolica::domains::float::{Complex, Real};

    /// Default SIMD lanes per chunk (gate G0, DESIGN.md §5.4).
    pub const DEFAULT_LANES: usize = 16;
    /// Default families per streamed bucket chunk (bounds the live fused
    /// factors and family numerators).
    pub const DEFAULT_CHUNK_FAMILIES: usize = 512;

    /// The dressing schedule.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Schedule {
        /// Per-family suffix tries (production).
        B,
        /// The forward prefix tree (oracle).
        F,
        /// The merged planar state graph for the plan's certified families,
        /// schedule B for the others (all of them when the plan's graph is
        /// empty: no certified family). Plans exported without a graph
        /// refuse it (WP2.8).
        M,
    }

    /// The Laurent columns of every reduction lane (DESIGN.md §12.4).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Poles {
        /// The finite part only (finite-only mode).
        None,
        /// `[finite, 1/ε, 1/ε²]` with `ε_UV = ε_IR`.
        #[default]
        All,
        /// `[finite, 1/ε, 1/ε², 1/ε_UV]`: the combined Laurent integrals and
        /// the ultraviolet part of the single pole (the double pole is
        /// infrared), for the `uv` and `ir` pole selections.
        Split,
    }

    impl Poles {
        /// Values per lane.
        pub fn width(self) -> usize {
            match self {
                Self::None => 1,
                Self::All => 3,
                Self::Split => 4,
            }
        }
    }

    /// Options of one evaluation.
    #[derive(Clone, Copy, Debug)]
    pub struct V2Options {
        /// The Laurent columns of every lane (finite-only, combined, split).
        pub poles: Poles,
        /// Reduce every lane a second time in this frame (stability estimate).
        pub check: Option<ReductionCheck>,
        pub chunk_families: usize,
        /// Schedule B, or M where the plan has it (F is an oracle of
        /// numerators only).
        pub schedule: Schedule,
        /// Schedule-M chunks per streamed part.
        pub part_chunks: usize,
        /// The seed contraction (DESIGN.md §7.3).
        pub contraction: Contraction,
    }

    impl Default for V2Options {
        fn default() -> Self {
            Self {
                poles: Poles::All,
                check: None,
                chunk_families: DEFAULT_CHUNK_FAMILIES,
                schedule: Schedule::B,
                part_chunks: DEFAULT_PART_CHUNKS,
                contraction: Contraction::Auto,
            }
        }
    }

    /// The seed contraction of an evaluation (DESIGN.md §7.3): `Auto`
    /// contracts the certified orbit pairs of the executor's seed Fourier
    /// record ([`V2Executor::with_fourier`]) with the symmetric-group FFT
    /// and everything else directly (all of it without a record); `Direct`
    /// contracts every weight directly; `Fft` requires a record.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Contraction {
        #[default]
        Auto,
        Direct,
        Fft,
    }

    /// Report count of the seed-FFT jobs an evaluation ran (one per job and
    /// lane chunk; absent when every seed was contracted directly).
    pub const FFT_SEED_JOBS: &str = "symmetric-group-fft seed jobs";
    /// Report stage of the seed contraction alone (direct or FFT), within
    /// the "seeds" stage, which also gathers each chunk's lane data.
    pub const SEED_CONTRACTION: &str = "seed contraction";

    /// A reduction lane of one point: (family, loop orders, observable column).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct LaneKey {
        pub family: usize,
        pub order: usize,
        pub column: usize,
    }

    /// The reduction lanes of one family, lane-major: `shifts` has `4N`
    /// values per lane, `masses` `N`, `scales` one, `s_matrix` `N²` and
    /// `coefficients` `C(R+4, 4)` at the family rank (the NinjaRS adapter's
    /// `FamilyGroup`).
    pub struct FamilyLanes<T> {
        pub family: usize,
        pub propagators: usize,
        pub rank: usize,
        pub points: Vec<usize>,
        pub keys: Vec<LaneKey>,
        pub shifts: Vec<Complex<T>>,
        pub masses: Vec<Complex<T>>,
        pub scales: Vec<Complex<T>>,
        pub s_matrix: Vec<Complex<T>>,
        pub coefficients: Vec<Complex<T>>,
    }

    /// One reduction lane: its raw Laurent integrals in the columns of
    /// [`Poles`], or the reducer status of a failed lane.
    pub type LaneOutcome<T> = Result<Vec<Complex<T>>, String>;

    /// Direct outcomes, and the re-test outcomes when a check was asked; one
    /// vector per group in its lane order.
    pub type Reduced<T> = (Vec<Vec<LaneOutcome<T>>>, Option<Vec<Vec<LaneOutcome<T>>>>);

    /// Development counters a [`V2Backend`] collects inside the executor's
    /// `reduction` stage: seconds of its sub-stages (e.g. the master
    /// integrals) and counts. They are added to the run's [`Report`].
    #[derive(Clone, Debug, Default)]
    pub struct BackendStatistics {
        pub seconds: Vec<(&'static str, f64)>,
        pub counts: Vec<(&'static str, u64)>,
    }

    /// The reduction side of the v2 path: NinjaRS engines and caches.
    pub trait V2Backend<T> {
        /// Start a chunk of points (master caches hold one chunk's masters).
        fn begin(&mut self);
        fn reduce(
            &mut self,
            groups: &[FamilyLanes<T>],
            poles: Poles,
            check: Option<ReductionCheck>,
        ) -> RusticolResult<Reduced<T>>;
        /// The counters collected since the last call (none by default).
        fn take_statistics(&mut self) -> BackendStatistics {
            BackendStatistics::default()
        }
    }

    /// Modelled kernel work of the chosen forms of `steps` (complex
    /// multiply-adds per lane, as the `loop_v2_schedules` bench counts it).
    fn modelled_work<'a>(steps: impl Iterator<Item = &'a crate::loop_v2::plan::Step>) -> u64 {
        steps
            .map(|step| match (step.split.is_some(), step.costs) {
                (true, (_, Some(split))) => split.work,
                (_, (fused, _)) => fused.work,
            })
            .sum()
    }

    /// Report count of the modelled dressing work (complex MAC per lane).
    const DRESSING_WORK: &str = "dressing work (modelled complex MAC per lane)";

    /// One point's family numerators: (family, loop orders, rank, per
    /// column the coefficients in the family's trie frame).
    pub type PointNumerators<T> = Vec<(usize, usize, usize, Vec<Vec<Complex<T>>>)>;

    /// One point of an evaluation.
    #[derive(Clone)]
    pub struct PointResult<T> {
        /// Raw Laurent integrals of every reduced lane.
        pub families: Vec<(LaneKey, Vec<Complex<T>>)>,
        /// The re-test integrals, when a check was asked.
        pub check: Option<Vec<(LaneKey, Vec<Complex<T>>)>>,
        /// Per observable column, the bare virtual `2 Re(i/(16π² avg) ΣΩ)`
        /// per Laurent coefficient.
        pub bare: Vec<Vec<T>>,
        /// Per observable column, the re-test error budget.
        pub budget: Option<Vec<Vec<T>>>,
        /// Numerical failures; a failed point has no values.
        pub failures: Vec<String>,
    }

    /// Development measurements of one run: seconds per stage, work counts
    /// and the peak reals held per lane (chunk inputs: fused factors, split
    /// vertex tensors and currents; live open-loop objects of one family
    /// walk; numerators of one bucket chunk). Bytes per lane are the reals
    /// times `real_bytes` (heap limbs of arbitrary-precision reals excluded).
    #[derive(Clone, Debug, Default)]
    pub struct Report {
        pub seconds: BTreeMap<&'static str, f64>,
        pub counts: BTreeMap<&'static str, u64>,
        pub reals_per_lane: BTreeMap<&'static str, u64>,
        pub real_bytes: usize,
        pub lanes: usize,
        pub threads: usize,
    }

    impl Report {
        fn time(&mut self, stage: &'static str, start: Instant) {
            *self.seconds.entry(stage).or_default() += start.elapsed().as_secs_f64();
        }
        fn count(&mut self, name: &'static str, value: u64) {
            *self.counts.entry(name).or_default() += value;
        }
        fn peak(&mut self, name: &'static str, reals: usize, lanes: usize) {
            let value = (reals / lanes) as u64;
            let entry = self.reals_per_lane.entry(name).or_default();
            *entry = (*entry).max(value);
        }
    }

    /// A loaded plan with its lane count, thread pool and, optionally, the
    /// certified seed FFT of its seeds ([`SeedFourier`]).
    pub struct V2Executor {
        plan: Arc<LoopProgramV4>,
        lanes: usize,
        pool: rayon::ThreadPool,
        fourier: Option<Arc<SeedFourier>>,
    }

    macro_rules! with_lanes {
        ($lanes:expr, $f:ident :: <$t:ty>($($arg:expr),*)) => {
            match $lanes {
                1 => $f::<$t, 1>($($arg),*),
                2 => $f::<$t, 2>($($arg),*),
                4 => $f::<$t, 4>($($arg),*),
                8 => $f::<$t, 8>($($arg),*),
                16 => $f::<$t, 16>($($arg),*),
                32 => $f::<$t, 32>($($arg),*),
                other => Err(RusticolError::invalid_argument(format!(
                    "unsupported SIMD lane count {other} (1, 2, 4, 8, 16 or 32)"
                ))),
            }
        };
    }

    impl V2Executor {
        /// `threads = None` uses the machine's parallelism.
        pub fn new(
            plan: Arc<LoopProgramV4>,
            lanes: usize,
            threads: Option<usize>,
        ) -> RusticolResult<Self> {
            if ![1, 2, 4, 8, 16, 32].contains(&lanes) {
                return Err(RusticolError::invalid_argument(format!(
                    "unsupported SIMD lane count {lanes} (1, 2, 4, 8, 16 or 32)"
                )));
            }
            // The DoubleFloat and arbitrary-precision reduction builds
            // Symbolica expressions on these workers: pyAmpliCol's library
            // key unlocks them for their lifetime (`crate::licence`).
            let mut builder =
                rayon::ThreadPoolBuilder::new().start_handler(|_| crate::licence::unlock_thread());
            if let Some(threads) = threads {
                builder = builder.num_threads(threads.max(1));
            }
            let pool = builder
                .build()
                .map_err(|e| RusticolError::internal(e.to_string()))?;
            Ok(Self {
                plan,
                lanes,
                pool,
                fourier: None,
            })
        }

        /// Attach the plan's seed Fourier record (`plan_export.fourier_record`),
        /// certified on the plan's weights at load ([`SeedFourier::load`]).
        pub fn with_fourier(mut self, record: &[u8]) -> RusticolResult<Self> {
            self.fourier = Some(Arc::new(SeedFourier::load(record, &self.plan)?));
            Ok(self)
        }

        /// The certified seed FFT, if the executor has one.
        pub fn fourier(&self) -> Option<&SeedFourier> {
            self.fourier.as_deref()
        }

        pub fn plan(&self) -> &LoopProgramV4 {
            &self.plan
        }

        pub fn lanes(&self) -> usize {
            self.lanes
        }

        /// Run `op` on this executor's thread pool (its `threads` bound):
        /// parallel work inside it (the tree program's stages, its
        /// interferences) stays on these workers instead of the global pool.
        pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
            self.pool.install(op)
        }

        /// Every family numerator of every point (oracles O2, O4 and the
        /// comparison with the Python reference executor).
        pub fn numerators<T: Carrier>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            schedule: Schedule,
        ) -> RusticolResult<Vec<PointNumerators<T>>> {
            batch.validate(&self.plan)?;
            self.pool.install(|| {
                with_lanes!(
                    self.lanes,
                    numerators_lanes::<T>(&self.plan, batch, zero, schedule)
                )
            })
        }

        /// The certified seed FFT an evaluation with `contraction` runs.
        fn contraction(&self, contraction: Contraction) -> RusticolResult<Option<&SeedFourier>> {
            Ok(match contraction {
                Contraction::Direct => None,
                Contraction::Auto => self.fourier.as_deref(),
                Contraction::Fft => Some(self.fourier.as_deref().ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "the seed contraction 'fft' needs the plan's certified seed \
                         Fourier record (the plan has no certified orbit pair)",
                    )
                })?),
            })
        }

        /// The seeds `U0` of every point (the oracle of the seed
        /// contraction, DESIGN.md §7.3): per point, per seed key, its
        /// `rows × columns` values in object order, contracted as an
        /// evaluation with `contraction` contracts them.
        pub fn seed_values<T: Carrier + Real>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            contraction: Contraction,
        ) -> RusticolResult<Vec<Vec<Vec<Complex<T>>>>> {
            batch.validate(&self.plan)?;
            let fourier = self.contraction(contraction)?;
            self.pool.install(|| {
                with_lanes!(
                    self.lanes,
                    seed_values_lanes::<T>(&self.plan, fourier, batch, zero)
                )
            })
        }

        /// Dress, reduce and assemble the bare virtual of every point.
        pub fn evaluate<T: Carrier + Real, B: V2Backend<T> + Send>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            backend: &mut B,
            options: V2Options,
        ) -> RusticolResult<(Vec<PointResult<T>>, Report)> {
            batch.validate(&self.plan)?;
            let threads = self.pool.current_num_threads();
            let fourier = self.contraction(options.contraction)?;
            self.pool.install(|| {
                let (results, mut report) = with_lanes!(
                    self.lanes,
                    evaluate_lanes::<T>(&self.plan, fourier, batch, zero, backend, options)
                )?;
                report.threads = threads;
                Ok((results, report))
            })
        }
    }

    fn seed_values_lanes<T: Carrier + Real, const L: usize>(
        plan: &LoopProgramV4,
        fourier: Option<&SeedFourier>,
        batch: &V2Batch<T>,
        zero: &T,
    ) -> RusticolResult<Vec<Vec<Vec<Complex<T>>>>> {
        let v = crate::loop_v2::kernels::lane_vector_len(L);
        let mut out = vec![Vec::new(); batch.points()];
        for chunk in chunks(batch.points(), L) {
            let data = LaneData::<T, L>::new(plan, batch, &chunk, zero)?;
            let seeds = match fourier {
                Some(fourier) => seeds_fourier(plan, fourier, &data),
                None => seeds(plan, &data),
            };
            for (lane, &point) in chunk.iter().enumerate() {
                out[point] = seeds
                    .values
                    .iter()
                    .map(|key| {
                        key.chunks(v)
                            .map(|vector| {
                                Complex::new(vector[lane].clone(), vector[L + lane].clone())
                            })
                            .collect()
                    })
                    .collect();
            }
        }
        Ok(out)
    }

    fn chunks(points: usize, lanes: usize) -> Vec<Vec<usize>> {
        (0..points)
            .collect::<Vec<_>>()
            .chunks(lanes)
            .map(<[usize]>::to_vec)
            .collect()
    }

    /// A bucket chunk: its families, the inputs they read, and the inputs
    /// read by it or a later chunk (an input lives from its first to its
    /// last chunk).
    struct BucketChunk {
        families: Vec<usize>,
        reads: Reads,
        live: Reads,
    }

    /// The schedule-M graph an evaluation runs: the plan's under schedule M
    /// (possibly empty: no certified family), none otherwise.
    fn merged_of(plan: &LoopProgramV4, schedule: Schedule) -> RusticolResult<Option<&Merged>> {
        match schedule {
            Schedule::M => plan.merged.as_ref().map(Some).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "the plan has no schedule M (it was exported without a merged graph)",
                )
            }),
            _ => Ok(None),
        }
    }

    /// The families schedule B dresses: all, or those schedule M leaves.
    fn b_families(plan: &LoopProgramV4, merged: Option<&Merged>) -> Vec<usize> {
        (0..plan.families.len())
            .filter(|f| merged.is_none_or(|m| m.families.binary_search(f).is_err()))
            .collect()
    }

    /// Bucket chunks of at most `size` of `families` (ascending) in bucket
    /// order.
    fn bucket_chunks(plan: &LoopProgramV4, size: usize, families: &[usize]) -> Vec<BucketChunk> {
        let parts: Vec<Vec<usize>> = plan
            .buckets()
            .values()
            .map(|members| {
                members
                    .iter()
                    .copied()
                    .filter(|f| families.binary_search(f).is_ok())
                    .collect::<Vec<_>>()
            })
            .filter(|members| !members.is_empty())
            .flat_map(|members| {
                members
                    .chunks(size.max(1))
                    .map(<[usize]>::to_vec)
                    .collect::<Vec<_>>()
            })
            .collect();
        let reads: Vec<Reads> = parts.iter().map(|part| Reads::of(plan, part)).collect();
        let last = |count: usize, ids: &dyn Fn(&Reads) -> &[usize]| {
            let mut last = vec![0; count];
            for (k, chunk) in reads.iter().enumerate() {
                for &id in ids(chunk) {
                    last[id] = k;
                }
            }
            last
        };
        let factors = last(plan.factors.len(), &|r| &r.factors);
        let vertices = last(plan.vertex_instances.len(), &|r| &r.vertices);
        let currents = last(plan.currents.len(), &|r| &r.currents);
        let alive = |last: &[usize], k: usize| -> Vec<usize> {
            (0..last.len()).filter(|&id| last[id] >= k).collect()
        };
        parts
            .into_iter()
            .zip(reads)
            .enumerate()
            .map(|(k, (families, reads))| BucketChunk {
                families,
                reads,
                live: Reads {
                    factors: alive(&factors, k),
                    vertices: alive(&vertices, k),
                    currents: alive(&currents, k),
                },
            })
            .collect()
    }

    /// Dress every family of `chunk` on one chunk of lanes (schedule B).
    fn dress_families<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        chunk: &BucketChunk,
        inputs: &mut ChunkInputs<T>,
        seeds: &crate::loop_v2::seeds::Seeds<T>,
        data: &LaneData<T, L>,
        report: &mut Report,
    ) -> RusticolResult<Vec<FamilyNumerator<T>>> {
        let families = &chunk.families;
        let start = Instant::now();
        inputs.prepare(plan, data, &chunk.reads, &chunk.live);
        report.time("factors", start);
        report.peak(
            "chunk inputs (factors, vertex tensors, currents)",
            inputs.reals(),
            L,
        );
        report.count("factors", chunk.reads.factors.len() as u64);
        report.count("vertex tensors", chunk.reads.vertices.len() as u64);
        let start = Instant::now();
        let inputs = &*inputs;
        let runs = families
            .par_iter()
            .map(|&f| run_family::<T, L>(plan, f, inputs, seeds, data))
            .collect::<RusticolResult<Vec<_>>>()?;
        report.time("dressing", start);
        let peak = runs.iter().map(|run| run.peak_reals).max().unwrap_or(0);
        report.peak("open objects (one family walk)", peak, L);
        report.count("families", families.len() as u64);
        let steps = || families.iter().flat_map(|&f| &plan.families[f].steps);
        report.count("steps", steps().count() as u64);
        report.count(
            "split steps",
            steps().filter(|s| s.split.is_some()).count() as u64,
        );
        report.count(DRESSING_WORK, modelled_work(steps()));
        let numerators: Vec<_> = runs.into_iter().map(|run| run.numerator).collect();
        report.peak(
            "family numerators (one bucket chunk)",
            numerators.iter().map(FamilyNumerator::reals).sum(),
            L,
        );
        Ok(numerators)
    }

    fn numerators_lanes<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        batch: &V2Batch<T>,
        zero: &T,
        schedule: Schedule,
    ) -> RusticolResult<Vec<PointNumerators<T>>> {
        let mut out: Vec<PointNumerators<T>> = vec![Vec::new(); batch.points()];
        let mut report = Report::default();
        let merged = merged_of(plan, schedule)?;
        for points in chunks(batch.points(), L) {
            let data = LaneData::<T, L>::new(plan, batch, &points, zero)?;
            let seeds = seeds(plan, &data);
            let mut inputs = ChunkInputs::new(plan);
            let numerators = match schedule {
                Schedule::B | Schedule::M => {
                    let families = b_families(plan, merged);
                    let reads = Reads::of(plan, &families);
                    let chunk = BucketChunk {
                        families,
                        live: reads.clone(),
                        reads,
                    };
                    let mut numerators =
                        dress_families(plan, &chunk, &mut inputs, &seeds, &data, &mut report)?;
                    if let Some(merged) = merged {
                        let parts = parts(merged, DEFAULT_PART_CHUNKS, plan);
                        run_merged(plan, &parts, &mut inputs, &seeds, &data, |done| {
                            numerators.extend(done);
                            Ok(())
                        })?;
                    }
                    numerators
                }
                Schedule::F => {
                    let needed = Reads {
                        factors: forward_factors(plan),
                        ..Reads::default()
                    };
                    inputs.prepare(plan, &data, &needed, &needed);
                    run_forward(plan, &inputs.factors, &seeds, &data)?.0
                }
            };
            for (lane, &point) in points.iter().enumerate() {
                for numerator in &numerators {
                    let family = &plan.families[numerator.family];
                    for (k, &order) in family.orders.iter().enumerate() {
                        let columns = (0..plan.columns)
                            .map(|o| numerator.lane::<L>(k, o, lane))
                            .collect();
                        out[point].push((numerator.family, order, family.rank, columns));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The reduction lanes of one family numerator on a chunk of lanes,
    /// skipping points that failed already (a non-finite numerator fails its
    /// point here). Shifts `p(mask)` and invariants `s[mask]` (`p(mask)²`, or
    /// the squared mass of the leg whose mass shell fixes it) are the chunk's
    /// exact invariants cast once per mask ([`LaneData`]); an on-shell
    /// invariant thus equals the cast squared mass of its propagators bit for
    /// bit, as the masters' exact on-shell tests require.
    fn lanes_of<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        masses: &[Complex<T>],
        scales: &[Complex<T>],
        numerator: &FamilyNumerator<T>,
        data: &LaneData<T, L>,
        failures: &mut [Vec<String>],
    ) -> FamilyLanes<T> {
        let family = &plan.families[numerator.family];
        let n = family.propagators();
        let masks = family.shift_masks();
        let v = 2 * L;
        let real = |x: &T| Complex::new(x.clone(), data.zero.clone());
        let shifts: Vec<_> = masks.iter().map(|&mask| data.momentum(mask)).collect();
        let squares: Vec<_> = masks
            .iter()
            .flat_map(|&a| masks.iter().map(move |&b| a ^ b))
            .map(|mask| data.square(mask))
            .collect();
        let mut lanes = FamilyLanes {
            family: numerator.family,
            propagators: n,
            rank: family.rank,
            points: Vec::new(),
            keys: Vec::new(),
            shifts: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            s_matrix: Vec::new(),
            coefficients: Vec::new(),
        };
        for lane in 0..data.active {
            let point = data.points[lane];
            if !failures[point].is_empty() {
                continue;
            }
            if !numerator.finite::<L>(lane) {
                failures[point].push(format!("family {}: non-finite numerator", numerator.family));
                continue;
            }
            for (k, &order) in family.orders.iter().enumerate() {
                for column in 0..plan.columns {
                    let coefficients = numerator.lane::<L>(k, column, lane);
                    if coefficients
                        .iter()
                        .all(|z| z.re.is_fully_zero() && z.im.is_fully_zero())
                    {
                        continue;
                    }
                    lanes.points.push(point);
                    lanes.keys.push(LaneKey {
                        family: numerator.family,
                        order,
                        column,
                    });
                    for shift in &shifts {
                        lanes
                            .shifts
                            .extend((0..4).map(|mu| real(&shift[mu * v + lane])));
                    }
                    lanes.masses.extend(
                        family
                            .sequence
                            .iter()
                            .map(|&(_, mass)| masses[mass].clone()),
                    );
                    lanes.scales.push(scales[lane].clone());
                    lanes
                        .s_matrix
                        .extend(squares.iter().map(|square| real(&square[lane])));
                    lanes.coefficients.extend(coefficients);
                }
            }
        }
        lanes
    }

    /// What the reductions of one evaluation collect per point.
    struct Lanes<T> {
        failures: Vec<Vec<String>>,
        families: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
        checks: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
    }

    /// The reduction side of one chunk of lanes: scatters family numerators
    /// into NinjaRS lanes (skipping failed points), reduces them and records
    /// each lane's outcome.
    struct Reduction<'a, T, B, const L: usize> {
        plan: &'a LoopProgramV4,
        masses: &'a [Complex<T>],
        /// `mu_R²` per lane.
        scales: &'a [Complex<T>],
        data: &'a LaneData<T, L>,
        backend: &'a mut B,
        options: V2Options,
        sink: &'a mut Lanes<T>,
    }

    impl<T: Carrier, B: V2Backend<T>, const L: usize> Reduction<'_, T, B, L> {
        fn reduce(
            &mut self,
            numerators: &[FamilyNumerator<T>],
            report: &mut Report,
        ) -> RusticolResult<()> {
            let start = Instant::now();
            let groups: Vec<FamilyLanes<T>> = numerators
                .iter()
                .map(|numerator| {
                    lanes_of(
                        self.plan,
                        self.masses,
                        self.scales,
                        numerator,
                        self.data,
                        &mut self.sink.failures,
                    )
                })
                .filter(|group| !group.keys.is_empty())
                .collect();
            report.time("scatter", start);
            report.count(
                "reduction lanes",
                groups.iter().map(|g| g.keys.len() as u64).sum(),
            );
            if groups.is_empty() {
                return Ok(());
            }
            let start = Instant::now();
            let (direct, check) =
                self.backend
                    .reduce(&groups, self.options.poles, self.options.check)?;
            report.time("reduction", start);
            let statistics = self.backend.take_statistics();
            for (stage, seconds) in statistics.seconds {
                *report.seconds.entry(stage).or_default() += seconds;
            }
            for (name, value) in statistics.counts {
                report.count(name, value);
            }
            let sink = &mut *self.sink;
            for (index, group) in groups.iter().enumerate() {
                for (row, (&point, &key)) in group.points.iter().zip(&group.keys).enumerate() {
                    match &direct[index][row] {
                        Ok(value) => sink.families[point].push((key, value.clone())),
                        Err(status) => sink.failures[point]
                            .push(format!("family {} lane {key:?}: {status}", group.family)),
                    }
                    if let Some(check) = &check {
                        match &check[index][row] {
                            Ok(value) => sink.checks[point].push((key, value.clone())),
                            Err(status) => sink.failures[point].push(format!(
                                "re-test of family {} lane {key:?}: {status}",
                                group.family
                            )),
                        }
                    }
                }
            }
            Ok(())
        }
    }

    /// Record a schedule-M run in the report (reals per lane).
    fn merged_report(report: &mut Report, stats: &MergedStats, lanes: usize) {
        *report.seconds.entry("factors").or_default() += stats.seconds_inputs;
        *report.seconds.entry("dressing").or_default() += stats.seconds_dressing;
        report.count("merged states", stats.states);
        report.count("merged split states", stats.split_states);
        report.count("merged chunks", stats.chunks);
        report.peak(
            "chunk inputs (factors, vertex tensors, currents)",
            stats.peak_inputs,
            lanes,
        );
        report.peak("open objects (one merged chunk)", stats.peak_objects, lanes);
        report.peak("merged family accumulators", stats.peak_accumulators, lanes);
    }

    fn evaluate_lanes<T: Carrier + Real, const L: usize>(
        plan: &LoopProgramV4,
        fourier: Option<&SeedFourier>,
        batch: &V2Batch<T>,
        zero: &T,
        backend: &mut (impl V2Backend<T> + Send),
        options: V2Options,
    ) -> RusticolResult<(Vec<PointResult<T>>, Report)> {
        let points = batch.points();
        let width = options.poles.width();
        let mut report = Report {
            lanes: L,
            real_bytes: std::mem::size_of::<T>(),
            ..Report::default()
        };
        if options.schedule == Schedule::F {
            return Err(RusticolError::invalid_argument(
                "schedule F is an oracle of family numerators only",
            ));
        }
        let merged = merged_of(plan, options.schedule)?;
        let b_families = b_families(plan, merged);
        let chunks_of_buckets = bucket_chunks(plan, options.chunk_families, &b_families);
        let merged_parts = merged
            .map(|merged| parts(merged, options.part_chunks, plan))
            .unwrap_or_default();
        let mut sink = Lanes {
            failures: vec![Vec::new(); points],
            families: vec![Vec::new(); points],
            checks: vec![Vec::new(); points],
        };
        for chunk in chunks(points, L) {
            backend.begin();
            let start = Instant::now();
            let data = LaneData::<T, L>::new(plan, batch, &chunk, zero)?;
            let contraction = Instant::now();
            let seeds = match fourier {
                Some(fourier) => {
                    report.count(FFT_SEED_JOBS, fourier.jobs() as u64);
                    seeds_fourier(plan, fourier, &data)
                }
                None => seeds(plan, &data),
            };
            // The contraction alone ("seeds" also gathers the lane data).
            report.time(SEED_CONTRACTION, contraction);
            report.time("seeds", start);
            let mut inputs = ChunkInputs::new(plan);
            let masses: Vec<Complex<T>> = batch
                .masses
                .iter()
                .map(|m| Complex::new(data.cast(m), data.zero.clone()))
                .collect();
            let scales: Vec<Complex<T>> = data
                .scales
                .iter()
                .map(|scale| Complex::new(data.cast(scale), data.zero.clone()))
                .collect();
            let mut reduction = Reduction {
                plan,
                masses: &masses,
                scales: &scales,
                data: &data,
                backend: &mut *backend,
                options,
                sink: &mut sink,
            };
            for part in &chunks_of_buckets {
                let numerators =
                    dress_families(plan, part, &mut inputs, &seeds, &data, &mut report)?;
                reduction.reduce(&numerators, &mut report)?;
            }
            if !merged_parts.is_empty() {
                let stats = run_merged(plan, &merged_parts, &mut inputs, &seeds, &data, |done| {
                    reduction.reduce(&done, &mut report)
                })?;
                merged_report(&mut report, &stats, L);
                if let Some(merged) = merged {
                    report.count(
                        DRESSING_WORK,
                        modelled_work(merged.states.iter().map(|state| &state.step)),
                    );
                }
            }
        }
        let Lanes {
            mut failures,
            mut families,
            mut checks,
        } = sink;
        let zero = zero.zero();
        let pi = zero.pi();
        let factor = Complex::new(
            zero.clone(),
            zero.one() / (zero.from_usize(16) * &pi * &pi * zero.from_usize(plan.average as usize)),
        );
        let mut results = Vec::with_capacity(points);
        for point in 0..points {
            if !failures[point].is_empty() {
                results.push(PointResult {
                    families: Vec::new(),
                    check: None,
                    bare: Vec::new(),
                    budget: None,
                    failures: std::mem::take(&mut failures[point]),
                });
                continue;
            }
            let mut sums =
                vec![vec![Complex::new(zero.clone(), zero.clone()); width]; plan.columns];
            let mut lanes = std::mem::take(&mut families[point]);
            lanes.sort_by_key(|(key, _)| *key);
            for (key, value) in &lanes {
                for (s, z) in sums[key.column].iter_mut().zip(value) {
                    *s += z;
                }
            }
            let two = zero.from_usize(2);
            let bare = sums
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .map(|z| two.clone() * (factor.clone() * z).re)
                        .collect()
                })
                .collect();
            let (check, budget) = if options.check.is_some() {
                let mut tests = std::mem::take(&mut checks[point]);
                tests.sort_by_key(|(key, _)| *key);
                let mut budget = vec![vec![zero.clone(); width]; plan.columns];
                for ((key, direct), (other, test)) in lanes.iter().zip(&tests) {
                    debug_assert_eq!(key, other);
                    add_to_budget(
                        &mut budget[key.column],
                        &deviation(direct, test),
                        plan.average,
                    );
                }
                (Some(tests), Some(budget))
            } else {
                (None, None)
            };
            results.push(PointResult {
                families: lanes,
                check,
                bare,
                budget,
                failures: Vec::new(),
            });
        }
        report.count("points", points as u64);
        Ok((results, report))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::loop_v2::dress::V2Batch;
        use symbolica::domains::rational::Rational;

        /// Two scalar bubble families of a two-leg "process": one seeded
        /// trace step each, the factor a constant, so each family numerator
        /// is `2 conj(A0)` at rank 0.
        const PLAN: &[u8] =
            br#"{"schema":4,"legs":2,"rows":1,"helicities":[[0],[0]],"columns":1,"average":1,
            "born_components":1,"constants":1,"masses":1,"block_rows":[],
            "kernels":[{"inputs":[],"outputs":[[[0,[]]]]}],
            "contents":[{"d_in":1,"d_out":1,"degree":0,"rows":1,"mask":0,"row_helicities":[[]],
                "fused":[[0,0,0,0,0,0]],
                "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
                    "bindings":[{"kernel":0,"inputs":[],"entries":[0]}]}]}],
            "factors":[[0,0]],
            "seed_classes":[{"columns":[[[0,"1","0"]]]}],"seeds":[{"classes":[[0,"1","0"]]}],
            "families":[FAMILY,FAMILY],"forward":null}"#;
        const FAMILY: &str = r#"{"sequence":[[1,0],[2,0]],"rank":0,"orders":[0],
            "leaves":[{"seed":0,"weight":"1"}],
            "steps":[{"seeded":true,"trace":0,"out":0,"rows":1,"d_alpha":1,"d_out":1,"degree":0,
                "rank_in":0,"edges":[{"leaf":0,"child":-1,"factor":0,"links":[0,0,0]}]}]}"#;

        /// Fails every lane of point 0 in its first reduction and records the
        /// points each call receives.
        #[derive(Default)]
        struct Failing {
            calls: Vec<Vec<usize>>,
        }

        impl V2Backend<f64> for Failing {
            fn begin(&mut self) {}

            fn reduce(
                &mut self,
                groups: &[FamilyLanes<f64>],
                poles: Poles,
                check: Option<ReductionCheck>,
            ) -> RusticolResult<Reduced<f64>> {
                assert!(poles == Poles::All && check.is_none());
                let first = self.calls.is_empty();
                self.calls
                    .push(groups.iter().flat_map(|g| g.points.clone()).collect());
                let direct = groups
                    .iter()
                    .map(|group| {
                        group
                            .points
                            .iter()
                            .zip(group.coefficients.iter())
                            .map(|(&point, c)| -> LaneOutcome<f64> {
                                if first && point == 0 {
                                    Err("ImpreciseResult".into())
                                } else {
                                    let zero = Complex::new(0.0, 0.0);
                                    Ok(vec![*c, zero, zero])
                                }
                            })
                            .collect()
                    })
                    .collect();
                Ok((direct, None))
            }
        }

        #[test]
        fn failed_points_are_compacted_out_of_later_buckets() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let plan = LoopProgramV4::load(text.as_bytes()).unwrap();
            let executor = V2Executor::new(Arc::new(plan), DEFAULT_LANES, Some(1)).unwrap();
            let momentum = |e: i64| -> Vec<Rational> {
                [e, 0, 0, e, e, 0, 0, -e].map(Rational::from).to_vec()
            };
            let batch = V2Batch {
                momenta: vec![momentum(1), momentum(2)],
                scales: vec![Rational::from(1); 2],
                masses: vec![Rational::from(0)],
                leg_masses: Vec::new(),
                constants: vec![Complex::new(2.0, 0.0)],
                block_rows: vec![Vec::new(); 2],
                born: vec![vec![Complex::new(1.0, 1.0)], vec![Complex::new(3.0, 0.0)]],
            };
            let options = V2Options {
                chunk_families: 1,
                ..V2Options::default()
            };
            let mut backend = Failing::default();
            let (results, report) = executor
                .evaluate(&batch, &0.0, &mut backend, options)
                .unwrap();
            // One bucket chunk per family; point 0 fails in the first and is
            // not reduced again.
            assert_eq!(backend.calls, [vec![0, 1], vec![1]]);
            assert_eq!(results[0].failures.len(), 1);
            assert!(results[0].failures[0].contains("ImpreciseResult"));
            assert!(results[0].bare.is_empty());
            let lanes = &results[1].families;
            assert!(results[1].failures.is_empty() && lanes.len() == 2);
            for (_key, value) in lanes {
                assert_eq!(value[0], Complex::new(6.0, 0.0));
            }
            assert_eq!(report.counts["reduction lanes"], 3);
        }

        /// Schedule M needs a plan exported with a merged graph; an empty
        /// graph (no certified family) runs every family on schedule B.
        #[test]
        fn schedule_m_falls_back_to_b_without_certified_families() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let batch = V2Batch {
                momenta: vec![[1, 0, 0, 1, 1, 0, 0, -1].map(Rational::from).to_vec()],
                scales: vec![Rational::from(1)],
                masses: vec![Rational::from(0)],
                leg_masses: Vec::new(),
                constants: vec![Complex::new(2.0, 0.0)],
                block_rows: vec![Vec::new()],
                born: vec![vec![Complex::new(1.0, 1.0)]],
            };
            let executor = |text: &str| {
                let plan = LoopProgramV4::load(text.as_bytes()).unwrap();
                V2Executor::new(Arc::new(plan), 2, Some(1)).unwrap()
            };
            let without = executor(&text);
            let error = without
                .numerators(&batch, &0.0, Schedule::M)
                .expect_err("schedule M without a merged graph");
            assert!(error.message().contains("no schedule M"), "{error}");
            let options = V2Options {
                schedule: Schedule::M,
                ..V2Options::default()
            };
            assert!(
                without
                    .evaluate(&batch, &0.0, &mut Failing::default(), options)
                    .is_err()
            );
            let empty = text.replace(
                r#""forward":null}"#,
                r#""forward":null,"merged":{"families":[],"entries":[],"states":[],"chunks":[]}}"#,
            );
            let with = executor(&empty);
            assert_eq!(with.plan().merged.as_ref().unwrap().families.len(), 0);
            let b = with.numerators(&batch, &0.0, Schedule::B).unwrap();
            let m = with.numerators(&batch, &0.0, Schedule::M).unwrap();
            assert_eq!(m.len(), 1);
            assert_eq!(m[0].len(), 2);
            for (mine, theirs) in m[0].iter().zip(&b[0]) {
                assert_eq!((mine.0, mine.1, mine.2), (theirs.0, theirs.1, theirs.2));
                assert_eq!(mine.3, theirs.3);
            }
        }
    }
}
