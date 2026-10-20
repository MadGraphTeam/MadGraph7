//! Integrated amplitudes for any number of propagators with lane-local
//! accumulation. Five-cuts are the highest cuts: for n > 5 every pentagon,
//! box, triangle, bubble and tadpole simply has more uncut denominators.

use crate::{
    BatchShape, BoxCoefficientMode, BoxMuMode, BoxReducer, BoxReductionKernels,
    BoxReductionOptions, BoxResidue, BubbleReducer, BubbleReductionKernels, BubbleResidue,
    CutBasis, CutFailure, CutOutcome, CutProvider, Error, FloatLike, FourMomentum, HighCutProvider,
    IntegratedContribution, IntegratedTadpoleResidue, IntegrationKernels, IntegrationStatus,
    MasterProvider, Numerator, PentagonReducer, PentagonReductionKernels, PentagonResidue,
    ResidueIntegrationRequest, ResidueIntegrator, Result, SampleId, TadpoleReducer,
    TadpoleReductionKernels, Topology, TriangleReducer, TriangleReductionKernels, TriangleResidue,
};

/// Factory and generated arithmetic used by the integrated coordinator.
pub trait AmplitudeKernels<T> {
    type Boxes: BoxReductionKernels<T>;
    type Pentagons: PentagonReductionKernels<T>;
    type Triangles: TriangleReductionKernels<T>;
    type Bubbles: BubbleReductionKernels<T>;
    type Tadpoles: TadpoleReductionKernels<T>;
    type Integration: IntegrationKernels<T>;
    fn propagators(&self) -> usize;
    fn rank(&self) -> usize;
    fn box_options(&self) -> BoxReductionOptions {
        BoxReductionOptions {
            coefficients: BoxCoefficientMode::Integrated,
            mu_extraction: BoxMuMode::Expansion,
        }
    }
    fn box_kernels(&self) -> Result<Self::Boxes>;
    fn pentagon_kernels(&self) -> Result<Self::Pentagons>;
    fn triangle_kernels(&self) -> Result<Self::Triangles>;
    fn bubble_kernels(&self) -> Result<Self::Bubbles>;
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles>;
    fn integration_kernels(&self, cut_size: usize) -> Result<Self::Integration>;
    /// Input rows: previous five components, then five components per topology
    /// cut in cached order. Output rows have the five accumulated components.
    fn accumulate_batch(&mut self, input: &[T], output: &mut [T], rows: usize) -> Result<()>;
    fn values_are_finite(&self, values: &[T]) -> bool;
    /// Compare complete carriers, including every derivative component.
    fn values_equal(&self, left: &T, right: &T) -> bool;
    /// `|Re v| + |Im v|` of the primal value of `value`, as a carrier without
    /// imaginary part or derivative components. The amplitude sums these over
    /// cut contributions to measure their cancellation
    /// ([`Amplitude::contribution_moduli`]).
    fn modulus(&self, value: &T) -> T;
    /// Whether one lane's cut contributions can be summed without losing the
    /// result to cancellation (see [`EvaluationStatus::Unstable`]). The default
    /// accepts every lane.
    fn cancellation_stable<'a>(
        &self,
        contributions: impl Iterator<Item = &'a IntegratedContribution<T>>,
    ) -> bool
    where
        T: 'a,
    {
        let _ = contributions;
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationStatus {
    NotEvaluated,
    Success,
    CutFailed {
        cut_index: usize,
        failure: CutFailure,
    },
    IntegrationFailed {
        cut_index: usize,
        status: IntegrationStatus,
    },
    NonFiniteResult,
    /// Two propagators of this lane coincide numerically (equal shifts and
    /// masses), e.g. at a vanishing external momentum. Raised propagators are
    /// unsupported; other lanes of the batch are unaffected. The lane's rows
    /// of the cut outcomes are not meaningful.
    RepeatedPropagator {
        first: usize,
        second: usize,
    },
    /// Every cut succeeded, but the cut contributions cancel so strongly that
    /// rounding leaves fewer than half of the carrier's significant bits: the
    /// summed magnitudes of the contributions exceed `2^(p/2)` times the largest
    /// Laurent coefficient of their sum (`p` mantissa bits). Ninja's
    /// floating-point checks only detect exactly singular cuts (the stage
    /// kernels reject those as cut failures); this lane needs more precision.
    /// Reported only when the kernels opt in ([`AmplitudeKernels::cancellation_stable`]).
    Unstable,
}

/// Borrowed lane-major numerical inputs, including an optional exact S-matrix.
pub struct AmplitudeInput<'a, T> {
    pub shifts: &'a [FourMomentum<T>],
    pub masses_squared: &'a [T],
    pub scales_squared: &'a [T],
    pub sampling_scales: &'a [T],
    /// Lane-major n×n matrices of (V_i−V_j)². Symmetry/zero diagonals
    /// include every Dual component. Rounded momenta need not reproduce them.
    pub s_matrix: Option<&'a [T]>,
}

/// Ninja-style accumulation over a fixed phase-space batch.
///
/// This implementation accepts any number of propagators n >= 1 and ranks up
/// to n+2 (the kernels decide the supported classes; see `RankClass`).
/// Calls add successful lanes to existing results. Every field of a failed lane
/// remains unchanged; its last-attempt status is recorded separately. Structural
/// errors leave results and last-attempt statuses unchanged; a lane with
/// numerically coincident propagators fails alone. Shifts and masses
/// are lane-major; one positive real squared scale is supplied per lane.
/// Tadpole diagnostics contain only the coefficients needed for integration.
pub struct Amplitude<T, K: AmplitudeKernels<T>> {
    topology: Topology,
    lanes: usize,
    minimum_cut: usize,
    kernels: K,
    zero: T,
    box_stage: Option<BoxReducer<T, K::Boxes>>,
    pentagon: Option<PentagonReducer<T, K::Pentagons>>,
    triangle: Option<TriangleReducer<T, K::Triangles>>,
    bubble: Option<BubbleReducer<T, K::Bubbles>>,
    tadpole: Option<TadpoleReducer<T, K::Tadpoles>>,
    integration: [Option<ResidueIntegrator<T, K::Integration>>; 4],
    boxes: Vec<CutOutcome<BoxResidue<T>>>,
    pentagons: Vec<CutOutcome<PentagonResidue<T>>>,
    triangles: Vec<CutOutcome<TriangleResidue<T>>>,
    bubbles: Vec<CutOutcome<BubbleResidue<T>>>,
    tadpoles: Vec<CutOutcome<IntegratedTadpoleResidue<T>>>,
    results: Vec<IntegratedContribution<T>>,
    moduli: Vec<crate::Laurent<T>>,
    statuses: Vec<EvaluationStatus>,
    contributions: Vec<IntegratedContribution<T>>,
    input: Vec<T>,
    output: Vec<T>,
    // Private staging remains distinct from published statuses and results.
    next_statuses: Vec<EvaluationStatus>,
    active: Vec<usize>,
    integration_cuts: [Vec<usize>; 4],
    integration_values: Vec<IntegratedContribution<T>>,
    integration_statuses: Vec<IntegrationStatus>,
}

impl<T: FloatLike, K: AmplitudeKernels<T>> Amplitude<T, K> {
    pub fn new(kernels: K, prototype: T, lanes: usize) -> Result<Self> {
        let topology = Topology::new(kernels.propagators(), kernels.rank())?;
        let active = |size| {
            size <= topology.propagators()
                && (size == 4 || topology.rank().saturating_add(size) >= topology.propagators())
        };
        let box_stage = if active(4) {
            Some(BoxReducer::new(kernels.box_kernels()?, prototype.clone())?)
        } else {
            None
        };
        let pentagon = if kernels
            .box_options()
            .needs_pentagons(topology.propagators(), topology.rank())
        {
            Some(PentagonReducer::new(
                kernels.pentagon_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let triangle = if active(3) {
            Some(TriangleReducer::new(
                kernels.triangle_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let bubble = if active(2) {
            Some(BubbleReducer::new(
                kernels.bubble_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let tadpole = if active(1) {
            Some(TadpoleReducer::new(
                kernels.tadpole_kernels()?,
                prototype.clone(),
            )?)
        } else {
            None
        };
        let mut integration = [None, None, None, None];
        for size in 1..=4 {
            if active(size) {
                integration[size - 1] = Some(ResidueIntegrator::new(
                    kernels.integration_kernels(size)?,
                    prototype.clone(),
                ));
            }
        }
        let integration_cuts = std::array::from_fn(|size| {
            topology
                .cuts()
                .iter()
                .enumerate()
                .filter_map(|(index, cut)| (cut.partition.len() == size + 1).then_some(index))
                .collect()
        });
        let zero = prototype.zero();
        Ok(Self {
            topology,
            lanes,
            minimum_cut: 1,
            kernels,
            box_stage,
            pentagon,
            triangle,
            bubble,
            tadpole,
            integration,
            boxes: Vec::new(),
            pentagons: Vec::new(),
            triangles: Vec::new(),
            bubbles: Vec::new(),
            tadpoles: Vec::new(),
            results: vec![std::array::from_fn(|_| zero.clone()); lanes],
            moduli: vec![std::array::from_fn(|_| zero.clone()); lanes],
            statuses: vec![EvaluationStatus::NotEvaluated; lanes],
            contributions: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            next_statuses: Vec::new(),
            active: Vec::new(),
            integration_cuts,
            integration_values: Vec::new(),
            integration_statuses: Vec::new(),
            zero,
        })
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    /// Stop after this cut size, assuming all lower cuts contribute zero.
    /// Values 0/1 retain the full reduction. Existing accumulated values remain.
    pub fn set_cut_stop(&mut self, minimum_cut: usize) {
        self.minimum_cut = minimum_cut.max(1);
    }
    pub fn cut_stop(&self) -> usize {
        self.minimum_cut
    }
    /// `[finite, single pole, double pole, cut-constructible finite, rational]`.
    pub fn results(&self) -> &[IntegratedContribution<T>] {
        &self.results
    }
    /// Laurent coefficients in original Ninja order: finite, single, double pole.
    pub fn laurent(&self, lane: usize) -> Option<&crate::Laurent<T>> {
        self.results.get(lane)?.first_chunk::<3>()
    }
    /// For each Laurent coefficient, the sum of the moduli
    /// ([`AmplitudeKernels::modulus`]) of the cut contributions accumulated
    /// into [`Self::results`], updated and reset with them, whether or not
    /// the kernels test cancellation ([`EvaluationStatus::Unstable`]). Their
    /// ratio to [`Self::result_moduli`] measures how strongly the cuts
    /// cancel: summing them alone leaves a rounding error of order `2^-p`
    /// times these sums (`p` mantissa bits).
    pub fn contribution_moduli(&self) -> &[crate::Laurent<T>] {
        &self.moduli
    }
    /// The moduli ([`AmplitudeKernels::modulus`]) of one lane's Laurent
    /// coefficients.
    pub fn result_moduli(&self, lane: usize) -> Option<crate::Laurent<T>> {
        let laurent = self.laurent(lane)?;
        Some(std::array::from_fn(|k| self.kernels.modulus(&laurent[k])))
    }
    pub fn rational_part(&self, lane: usize) -> Option<&T> {
        self.results.get(lane).map(|value| &value[4])
    }
    pub fn cut_constructible_part(&self, lane: usize) -> Option<&T> {
        self.results.get(lane).map(|value| &value[3])
    }
    pub fn statuses(&self) -> &[EvaluationStatus] {
        &self.statuses
    }
    #[doc(hidden)]
    pub fn debug_contributions(&self) -> &[IntegratedContribution<T>] {
        &self.contributions
    }
    pub fn triangles(&self) -> &[CutOutcome<TriangleResidue<T>>] {
        &self.triangles
    }
    pub fn boxes(&self) -> &[CutOutcome<BoxResidue<T>>] {
        &self.boxes
    }
    pub fn pentagons(&self) -> &[CutOutcome<PentagonResidue<T>>] {
        &self.pentagons
    }
    pub fn bubbles(&self) -> &[CutOutcome<BubbleResidue<T>>] {
        &self.bubbles
    }
    pub fn tadpoles(&self) -> &[CutOutcome<IntegratedTadpoleResidue<T>>] {
        &self.tadpoles
    }

    pub fn reset(&mut self) {
        self.reset_batch(self.lanes);
    }
    /// Explicitly discard accumulated values when changing the lane count.
    pub fn reset_batch(&mut self, lanes: usize) {
        self.lanes = lanes;
        self.results
            .resize_with(lanes, || std::array::from_fn(|_| self.zero.clone()));
        self.results
            .fill(std::array::from_fn(|_| self.zero.clone()));
        self.moduli
            .resize_with(lanes, || std::array::from_fn(|_| self.zero.clone()));
        self.moduli.fill(std::array::from_fn(|_| self.zero.clone()));
        self.statuses.resize(lanes, EvaluationStatus::NotEvaluated);
        self.statuses.fill(EvaluationStatus::NotEvaluated);
        if let Some(stage) = &mut self.triangle {
            stage.discard_outcomes(&mut self.triangles, 0);
        } else {
            self.triangles.clear();
        }
        if let Some(stage) = &mut self.box_stage {
            stage.discard_outcomes(&mut self.boxes, 0);
        } else {
            self.boxes.clear();
        }
        self.pentagons.clear();
        if let Some(stage) = &mut self.bubble {
            stage.discard_outcomes(&mut self.bubbles, 0);
        } else {
            self.bubbles.clear();
        }
        if let Some(stage) = &mut self.tadpole {
            stage.discard_outcomes(&mut self.tadpoles, 0);
        } else {
            self.tadpoles.clear();
        }
    }

    pub fn evaluate_batch<N: Numerator<T>, C: CutProvider<T>, M: MasterProvider<T>>(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        scales_squared: &[T],
        numerator: &mut N,
        geometry: &mut C,
        masters: &mut M,
    ) -> Result<()> {
        self.evaluate_batch_with_low_cuts(
            AmplitudeInput {
                shifts,
                masses_squared,
                scales_squared,
                sampling_scales: &[],
                s_matrix: None,
            },
            numerator,
            geometry,
            masters,
        )
    }

    /// Borrowed-input variant for evaluations requiring no four/five-cut stages.
    /// This also accepts a supplied S-matrix without constructing a high-cut provider.
    pub fn evaluate_batch_with_low_cuts<
        N: Numerator<T>,
        C: CutProvider<T>,
        M: MasterProvider<T>,
    >(
        &mut self,
        input: AmplitudeInput<'_, T>,
        numerator: &mut N,
        geometry: &mut C,
        masters: &mut M,
    ) -> Result<()> {
        if (self.minimum_cut <= 4 && self.box_stage.is_some())
            || (self.minimum_cut <= 5 && self.pentagon.is_some())
        {
            return Err(Error::Unsupported(
                "four or more propagators require evaluate_batch_with_high_cuts".into(),
            ));
        }
        self.evaluate_batch_with_inputs(
            input,
            numerator,
            geometry,
            &mut UnavailableHighCuts,
            masters,
        )
    }

    /// Evaluate with four/five-cut geometry. One sampling scale per lane is
    /// required when boxes are present; it is independent of the master scale.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_with_high_cuts<
        N: Numerator<T>,
        C: CutProvider<T>,
        H: HighCutProvider<T>,
        M: MasterProvider<T>,
    >(
        &mut self,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
        scales_squared: &[T],
        sampling_scales: &[T],
        numerator: &mut N,
        geometry: &mut C,
        high_geometry: &mut H,
        masters: &mut M,
    ) -> Result<()> {
        self.evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts,
                masses_squared,
                scales_squared,
                sampling_scales,
                s_matrix: None,
            },
            numerator,
            geometry,
            high_geometry,
            masters,
        )
    }

    /// Evaluate with borrowed invariant overrides. The matrix feeds both master
    /// arguments and rational-weight kernels; geometry still uses the momenta.
    pub fn evaluate_batch_with_inputs<
        N: Numerator<T>,
        C: CutProvider<T>,
        H: HighCutProvider<T>,
        M: MasterProvider<T>,
    >(
        &mut self,
        input: AmplitudeInput<'_, T>,
        numerator: &mut N,
        geometry: &mut C,
        high_geometry: &mut H,
        masters: &mut M,
    ) -> Result<()> {
        let AmplitudeInput {
            shifts,
            masses_squared,
            scales_squared,
            sampling_scales,
            s_matrix,
        } = input;
        let n = self.topology.propagators();
        let lanes = self.lanes;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("amplitude shifts", shifts.len())?;
        BatchShape {
            rows: lanes,
            columns: n,
        }
        .validate("amplitude masses", masses_squared.len())?;
        BatchShape {
            rows: lanes,
            columns: 1,
        }
        .validate("amplitude scales", scales_squared.len())?;
        if let Some(matrix) = s_matrix {
            BatchShape {
                rows: lanes,
                columns: n * n,
            }
            .validate("amplitude S-matrix", matrix.len())?;
        }
        if self.minimum_cut <= 4 && self.box_stage.is_some() {
            BatchShape {
                rows: lanes,
                columns: 1,
            }
            .validate("amplitude box sampling scales", sampling_scales.len())?;
        }
        if numerator.rank() > self.topology.rank() {
            return Err(Error::Unsupported(format!(
                "numerator rank {} exceeds configured rank {}",
                numerator.rank(),
                self.topology.rank()
            )));
        }
        let repeated = (0..lanes)
            .filter_map(|lane| {
                self.repeated_propagators(lane, shifts, masses_squared)
                    .map(|pair| (lane, pair))
            })
            .collect::<Vec<_>>();
        if !repeated.is_empty() {
            return self.evaluate_without_repeated(
                &repeated,
                AmplitudeInput {
                    shifts,
                    masses_squared,
                    scales_squared,
                    sampling_scales,
                    s_matrix,
                },
                numerator,
                geometry,
                high_geometry,
                masters,
            );
        }
        if self.minimum_cut > 5 {
            self.pentagons.clear();
        }
        if self.minimum_cut > 4 {
            if let Some(stage) = &mut self.box_stage {
                stage.discard_outcomes(&mut self.boxes, 0);
            } else {
                self.boxes.clear();
            }
        }
        if self.minimum_cut > 3 {
            if let Some(stage) = &mut self.triangle {
                stage.discard_outcomes(&mut self.triangles, 0);
            } else {
                self.triangles.clear();
            }
        }
        if self.minimum_cut > 2 {
            if let Some(stage) = &mut self.bubble {
                stage.discard_outcomes(&mut self.bubbles, 0);
            } else {
                self.bubbles.clear();
            }
        }
        if self.minimum_cut > 1 {
            if let Some(stage) = &mut self.tadpole {
                stage.discard_outcomes(&mut self.tadpoles, 0);
            } else {
                self.tadpoles.clear();
            }
        }
        if self.minimum_cut <= 5
            && let Some(stage) = &mut self.pentagon
        {
            self.pentagons.resize(
                stage.cut_count() * lanes,
                CutOutcome::Unstable(CutFailure::NonFiniteResult),
            );
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                numerator,
                high_geometry,
                &mut self.pentagons,
            )?;
        }
        if self.minimum_cut <= 4
            && let Some(stage) = &mut self.box_stage
        {
            let rows = stage.cut_count() * lanes;
            if self.boxes.len() > rows {
                stage.discard_outcomes(&mut self.boxes, rows);
            }
            self.boxes
                .resize(rows, CutOutcome::Unstable(CutFailure::NonFiniteResult));
            stage.evaluate_batch(
                shifts,
                masses_squared,
                sampling_scales,
                lanes,
                &self.topology,
                &self.pentagons,
                numerator,
                high_geometry,
                &mut self.boxes,
            )?;
        }
        if self.minimum_cut <= 3
            && let Some(stage) = &mut self.triangle
        {
            let rows = stage.cut_count() * lanes;
            if self.triangles.len() > rows {
                stage.discard_outcomes(&mut self.triangles, rows);
            }
            self.triangles
                .resize(rows, CutOutcome::Unstable(CutFailure::NonFiniteResult));
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                numerator,
                geometry,
                &mut self.triangles,
            )?;
        }
        if self.minimum_cut <= 2
            && let Some(stage) = &mut self.bubble
        {
            let rows = stage.cut_count() * lanes;
            if self.bubbles.len() > rows {
                stage.discard_outcomes(&mut self.bubbles, rows);
            }
            self.bubbles
                .resize(rows, CutOutcome::Unstable(CutFailure::NonFiniteResult));
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                &self.topology,
                &self.triangles,
                numerator,
                geometry,
                &mut self.bubbles,
            )?;
        }
        if self.minimum_cut <= 1
            && let Some(stage) = &mut self.tadpole
        {
            let rows = stage.cut_count() * lanes;
            if self.tadpoles.len() > rows {
                stage.discard_outcomes(&mut self.tadpoles, rows);
            }
            self.tadpoles
                .resize(rows, CutOutcome::Unstable(CutFailure::NonFiniteResult));
            stage.evaluate_batch(
                shifts,
                masses_squared,
                lanes,
                &self.topology,
                &self.triangles,
                &self.topology,
                &self.bubbles,
                numerator,
                geometry,
                &mut self.tadpoles,
            )?;
        }
        let statuses = &mut self.next_statuses;
        statuses.resize(lanes, EvaluationStatus::Success);
        statuses.fill(EvaluationStatus::Success);
        // Five-cuts have no direct master contribution, but an executed failed
        // five-cut must still invalidate its lane, including when stopping at 5.
        if lanes != 0 {
            for ((cut_index, _), outcomes) in self
                .topology
                .cuts()
                .iter()
                .enumerate()
                .filter(|(_, cut)| cut.partition.len() == 5)
                .zip(self.pentagons.chunks_exact(lanes))
            {
                for (lane, outcome) in outcomes.iter().enumerate() {
                    if let CutOutcome::Unstable(failure) = outcome
                        && statuses[lane] == EvaluationStatus::Success
                    {
                        statuses[lane] = EvaluationStatus::CutFailed {
                            cut_index,
                            failure: *failure,
                        };
                    }
                }
            }
        }
        let cuts = self.topology.cuts().len();
        self.contributions
            .resize_with(cuts * lanes, || std::array::from_fn(|_| self.zero.clone()));
        self.contributions
            .fill(std::array::from_fn(|_| self.zero.clone()));
        for size in (self.minimum_cut..=4).rev() {
            let Some(integration) = &mut self.integration[size - 1] else {
                continue;
            };
            match size {
                4 => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    s_matrix,
                    &self.boxes,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    statuses,
                    &self.integration_cuts[size - 1],
                    &mut self.integration_values,
                    &mut self.integration_statuses,
                )?,
                3 => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    s_matrix,
                    &self.triangles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    statuses,
                    &self.integration_cuts[size - 1],
                    &mut self.integration_values,
                    &mut self.integration_statuses,
                )?,
                2 => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    s_matrix,
                    &self.bubbles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    statuses,
                    &self.integration_cuts[size - 1],
                    &mut self.integration_values,
                    &mut self.integration_statuses,
                )?,
                _ => integrate_stage(
                    &self.topology,
                    lanes,
                    shifts,
                    masses_squared,
                    scales_squared,
                    s_matrix,
                    &self.tadpoles,
                    integration,
                    masters,
                    &self.zero,
                    &mut self.contributions,
                    statuses,
                    &self.integration_cuts[size - 1],
                    &mut self.integration_values,
                    &mut self.integration_statuses,
                )?,
            }
        }
        for (lane, status) in statuses.iter_mut().enumerate() {
            if *status == EvaluationStatus::Success
                && !self.kernels.cancellation_stable(
                    (0..cuts).map(|cut| &self.contributions[cut * lanes + lane]),
                )
            {
                *status = EvaluationStatus::Unstable;
            }
        }
        let active = &mut self.active;
        active.clear();
        active.extend(
            statuses
                .iter()
                .enumerate()
                .filter_map(|(i, s)| (*s == EvaluationStatus::Success).then_some(i)),
        );
        self.input.clear();
        for &lane in active.iter() {
            self.input.extend_from_slice(&self.results[lane]);
            for cut in 0..cuts {
                self.input
                    .extend_from_slice(&self.contributions[cut * lanes + lane]);
            }
        }
        self.output.resize(active.len() * 5, self.zero.clone());
        self.kernels
            .accumulate_batch(&self.input, &mut self.output, active.len())?;
        for (index, &lane) in active.iter().enumerate() {
            let row = &self.output[index * 5..(index + 1) * 5];
            if self.kernels.values_are_finite(row) {
                self.results[lane].clone_from_slice(row);
                for cut in 0..cuts {
                    let contribution = &self.contributions[cut * lanes + lane];
                    for (modulus, value) in self.moduli[lane].iter_mut().zip(contribution) {
                        *modulus += &self.kernels.modulus(value);
                    }
                }
            } else {
                statuses[lane] = EvaluationStatus::NonFiniteResult;
            }
        }
        std::mem::swap(&mut self.statuses, statuses);
        Ok(())
    }

    /// The first pair of numerically coincident propagators of one lane.
    fn repeated_propagators(
        &self,
        lane: usize,
        shifts: &[FourMomentum<T>],
        masses_squared: &[T],
    ) -> Option<(usize, usize)> {
        let n = self.topology.propagators();
        (0..n)
            .flat_map(|first| (first + 1..n).map(move |second| (first, second)))
            .find(|&(first, second)| {
                let (a, b) = (lane * n + first, lane * n + second);
                self.kernels
                    .values_equal(&masses_squared[a], &masses_squared[b])
                    && shifts[a]
                        .components()
                        .into_iter()
                        .zip(shifts[b].components())
                        .all(|(x, y)| self.kernels.values_equal(x, y))
            })
    }

    /// Raised or repeated propagators are unsupported, but they fail only their
    /// own lanes. Those lanes are evaluated with a valid donor lane's kinematic
    /// inputs, so no stage sees the degenerate configuration; the numerator
    /// still receives the failed lane's own parameters. Their results stay
    /// unchanged, while their rows of the cut outcomes hold that evaluation.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_without_repeated<
        N: Numerator<T>,
        C: CutProvider<T>,
        H: HighCutProvider<T>,
        M: MasterProvider<T>,
    >(
        &mut self,
        repeated: &[(usize, (usize, usize))],
        input: AmplitudeInput<'_, T>,
        numerator: &mut N,
        geometry: &mut C,
        high_geometry: &mut H,
        masters: &mut M,
    ) -> Result<()> {
        let lanes = self.lanes;
        let status = |&(first, second): &(usize, usize)| EvaluationStatus::RepeatedPropagator {
            first,
            second,
        };
        let Some(donor) = (0..lanes).find(|lane| repeated.iter().all(|(failed, _)| failed != lane))
        else {
            for (lane, pair) in repeated {
                self.statuses[*lane] = status(pair);
            }
            return Ok(());
        };
        let n = self.topology.propagators();
        fn substitute<V: Clone>(
            values: &[V],
            width: usize,
            donor: usize,
            repeated: &[(usize, (usize, usize))],
        ) -> Vec<V> {
            let mut values = values.to_vec();
            for (lane, _) in repeated {
                for column in 0..width {
                    values[lane * width + column] = values[donor * width + column].clone();
                }
            }
            values
        }
        let shifts = substitute(input.shifts, n, donor, repeated);
        let masses_squared = substitute(input.masses_squared, n, donor, repeated);
        let scales_squared = substitute(input.scales_squared, 1, donor, repeated);
        let sampling_scales = substitute(
            input.sampling_scales,
            usize::from(input.sampling_scales.len() == lanes),
            donor,
            repeated,
        );
        let s_matrix = input
            .s_matrix
            .map(|matrix| substitute(matrix, n * n, donor, repeated));
        let saved = repeated
            .iter()
            .map(|(lane, _)| (self.results[*lane].clone(), self.moduli[*lane].clone()))
            .collect::<Vec<_>>();
        self.evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses_squared,
                scales_squared: &scales_squared,
                sampling_scales: &sampling_scales,
                s_matrix: s_matrix.as_deref(),
            },
            numerator,
            geometry,
            high_geometry,
            masters,
        )?;
        for ((lane, pair), (results, moduli)) in repeated.iter().zip(saved) {
            self.results[*lane] = results;
            self.moduli[*lane] = moduli;
            self.statuses[*lane] = status(pair);
        }
        Ok(())
    }
}

trait ResidueView<T> {
    fn id(&self) -> SampleId;
    fn index(&self) -> usize;
    fn coefficients(&self) -> &[T];
    fn basis(&self) -> Option<&CutBasis<T>>;
}
macro_rules! residue_view {
    ($name:ident) => {
        impl<T> ResidueView<T> for $name<T> {
            fn id(&self) -> SampleId {
                self.id
            }
            fn index(&self) -> usize {
                self.cut_index
            }
            fn coefficients(&self) -> &[T] {
                &self.coefficients
            }
            fn basis(&self) -> Option<&CutBasis<T>> {
                Some(&self.geometry.basis)
            }
        }
    };
}
residue_view!(TriangleResidue);
residue_view!(BubbleResidue);
residue_view!(IntegratedTadpoleResidue);

impl<T> ResidueView<T> for BoxResidue<T> {
    fn id(&self) -> SampleId {
        self.id
    }
    fn index(&self) -> usize {
        self.cut_index
    }
    fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
    fn basis(&self) -> Option<&CutBasis<T>> {
        None
    }
}

/// Allows existing low-cut clients to keep their provider implementation.
struct UnavailableHighCuts;
impl<T> HighCutProvider<T> for UnavailableHighCuts {
    fn box_batch(
        &mut self,
        _: &[crate::BoxCutRequest<'_, T>],
        _: &mut [CutOutcome<crate::BoxCut<T>>],
    ) -> Result<()> {
        Err(Error::Unsupported("box geometry was not provided".into()))
    }
    fn pentagon_batch(
        &mut self,
        _: &[crate::PentagonCutRequest<'_, T>],
        _: &mut [CutOutcome<crate::PentagonCut<T>>],
    ) -> Result<()> {
        Err(Error::Unsupported(
            "pentagon geometry was not provided".into(),
        ))
    }
    fn sample_box_batch(
        &mut self,
        _: &[crate::BoxSampleRequest<'_, T>],
        _: &mut [CutOutcome<[FourMomentum<T>; 2]>],
    ) -> Result<()> {
        Err(Error::Unsupported("box geometry was not provided".into()))
    }
}

#[allow(clippy::too_many_arguments)]
fn integrate_stage<
    T: FloatLike,
    K: IntegrationKernels<T>,
    M: MasterProvider<T>,
    R: ResidueView<T>,
>(
    topology: &Topology,
    lanes: usize,
    shifts: &[FourMomentum<T>],
    masses: &[T],
    scales: &[T],
    s_matrix: Option<&[T]>,
    residues: &[CutOutcome<R>],
    integration: &mut ResidueIntegrator<T, K>,
    provider: &mut M,
    zero: &T,
    contributions: &mut [IntegratedContribution<T>],
    statuses: &mut [EvaluationStatus],
    cuts: &[usize],
    values: &mut Vec<IntegratedContribution<T>>,
    integration_status: &mut Vec<IntegrationStatus>,
) -> Result<()> {
    let n = topology.propagators();

    // Classify every cut first so a later failed cut cannot leave earlier
    // requests for the same lane queued for expensive master evaluation.
    for (row, outcome) in residues.iter().enumerate() {
        let lane = row % lanes;
        if let CutOutcome::Unstable(failure) = outcome
            && statuses[lane] == EvaluationStatus::Success
        {
            statuses[lane] = EvaluationStatus::CutFailed {
                cut_index: cuts[row / lanes],
                failure: *failure,
            };
        }
    }
    let mut requests = Vec::with_capacity(residues.len());
    for (row, outcome) in residues.iter().enumerate() {
        let lane = row % lanes;
        match outcome {
            CutOutcome::Stable(residue) if statuses[lane] == EvaluationStatus::Success => requests
                .push(ResidueIntegrationRequest {
                    id: residue.id(),
                    partition: &topology.cuts()[residue.index()].partition,
                    shifts: &shifts[lane * n..(lane + 1) * n],
                    masses_squared: &masses[lane * n..(lane + 1) * n],
                    coefficients: residue.coefficients(),
                    basis: residue.basis(),
                    scale_squared: &scales[lane],
                    s_matrix: s_matrix.map(|matrix| &matrix[lane * n * n..(lane + 1) * n * n]),
                }),
            _ => (),
        }
    }
    values.resize_with(requests.len(), || std::array::from_fn(|_| zero.clone()));
    integration_status.resize(requests.len(), IntegrationStatus::InvalidInput);
    integration_status.fill(IntegrationStatus::InvalidInput);
    integration.evaluate_batch(&requests, provider, values, integration_status)?;
    for ((request, value), &status) in requests
        .iter()
        .zip(values.iter_mut())
        .zip(integration_status.iter())
    {
        if status == IntegrationStatus::Success {
            std::mem::swap(
                &mut contributions[request.id.sample * lanes + request.id.lane],
                value,
            );
        } else if statuses[request.id.lane] == EvaluationStatus::Success {
            statuses[request.id.lane] = EvaluationStatus::IntegrationFailed {
                cut_index: request.id.sample,
                status,
            };
        }
    }
    Ok(())
}
