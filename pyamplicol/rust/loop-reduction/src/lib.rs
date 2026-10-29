// SPDX-License-Identifier: 0BSD

//! Batched consumers of NinjaRS, without reduction or numerator algebra of our own.
//!
//! These are **unconverted** Ninja/OneLOop integrals, not physical amplitudes:
//! μ_R^(2ε)/(i π^(2−ε) rΓ) times the dimensionally regulated integral.
//! For example, A0(m²) = [m²(1−log(m²/μ_R²)), m², 0].
//! Model factors, interference, and the public amplitude normalization belong
//! outside this boundary. Every successful evaluation replaces previous values.

#![forbid(unsafe_code)]

pub mod runtime;
pub mod v2_runtime;
pub mod verification;


use std::ops::Add;
use std::time::{Duration, Instant};

use ninjars::fast::GenericNumerator;
use ninjars::fast::MuTensorNumerator;
pub use ninjars::fast::{C64, FastInput, FastStatus};
use ninjars::fast::{
    CMom, FastAmplitude, FastMaster, FastMasters, FastNumerator, FastOptions, FastResult,
};
use ninjars::{Amplitude, Numerator, Real, Result};
pub use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, FourMomentum, SamplingScaleProvider,
    SamplingScaleSource, SamplingScaleStatus,
};
use ninjars_graph::{CoefficientEvaluator, PolynomialCoefficientPlan, TensorCoefficientPlan};
pub use ninjars_symbolica::PoleSelection;
pub use ninjars_symbolica::SymbolicaSamplingScales;
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
pub use oneloop::NativeFloat;
use symbolica::{atom::Atom, evaluate::EvaluationDomain};

/// Named coefficients avoid confusing Ninja's finite-first convention with
/// pole-first public APIs. The coefficient of ε^k is not multiplied by ε^k.
#[derive(Clone, Debug, PartialEq)]
pub struct Laurent<T> {
    pub finite: T,
    pub single_pole: T,
    pub double_pole: T,
}

impl<T> From<[T; 3]> for Laurent<T> {
    fn from([finite, single_pole, double_pole]: [T; 3]) -> Self {
        Self {
            finite,
            single_pole,
            double_pole,
        }
    }
}

impl<T: Add<Output = T>> Laurent<T> {
    /// Combine reductions of N0 + ε N1 + ε² N2 through finite order.
    /// All sectors must use the same integral normalization and scheme.
    pub fn epsilon_sectors(zero: Self, one: Self, two: Self) -> Self {
        Self {
            finite: zero.finite + one.single_pole + two.double_pole,
            single_pole: zero.single_pole + one.double_pole,
            double_pole: zero.double_pole,
        }
    }
}

/// `id` is the caller's original lane identity, not a compact reduction index.
/// Failed lanes never expose a stale value from a preceding evaluation.
#[derive(Clone, Debug)]
pub struct Lane<T, S> {
    pub id: u64,
    pub status: S,
    pub value: Option<Laurent<T>>,
}

/// Optional wall-clock measurements at the existing fast-engine/provider seam.
/// Engine time includes numerator callbacks; masters time includes only the
/// delegated native provider. Counts include failed calls and actual batch rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastProfile {
    pub engine_time: Duration,
    pub masters_time: Duration,
    pub numerator_callback_time: Duration,
    pub numerator_preparation_time: Duration,
    pub engine_calls: u64,
    pub master_calls: u64,
    pub master_rows: u64,
    pub numerator_calls: u64,
    pub numerator_rows: u64,
}

impl FastProfile {
    pub fn merge(&mut self, other: Self) {
        self.engine_time += other.engine_time;
        self.masters_time += other.masters_time;
        self.numerator_callback_time += other.numerator_callback_time;
        self.numerator_preparation_time += other.numerator_preparation_time;
        self.engine_calls += other.engine_calls;
        self.master_calls += other.master_calls;
        self.master_rows += other.master_rows;
        self.numerator_calls += other.numerator_calls;
        self.numerator_rows += other.numerator_rows;
    }
}

struct ProfiledNumerator<'a, N> {
    inner: &'a mut N,
    profile: &'a mut FastProfile,
}

impl<N: FastNumerator> ProfiledNumerator<'_, N> {
    fn record(&mut self, started: Instant, rows: usize) {
        self.profile.numerator_callback_time += started.elapsed();
        self.profile.numerator_calls += 1;
        self.profile.numerator_rows += rows as u64;
    }
}

impl<N: FastNumerator> FastNumerator for ProfiledNumerator<'_, N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn evaluate(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        q: &[CMom],
        mu_squared: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self.inner.evaluate(cut, lanes, q, mu_squared, out);
        self.record(started, lanes.len());
        result
    }
    fn mu_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: Option<&[CMom]>,
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self.inner.mu_expansion(cut, lanes, v0, v1, out);
        self.record(started, lanes.len());
        result
    }
    fn mu_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        v0: &[CMom],
        v1: &[CMom],
        beta: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self.inner.mu_expansion_x2(cut, lanes, v0, v1, beta, out);
        self.record(started, lanes.len());
        result
    }
    fn t3_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self
            .inner
            .t3_expansion(cut, lanes, a, e3, e4, param, mindeg, out);
        self.record(started, lanes.len());
        result
    }
    fn t2_expansion(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        mindeg: usize,
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self
            .inner
            .t2_expansion(cut, lanes, a0, a1, e3, e4, param, mindeg, out);
        self.record(started, lanes.len());
        result
    }
    fn t3_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self
            .inner
            .t3_expansion_x2(cut, lanes, a, e3, e4, param, out);
        self.record(started, lanes.len());
        result
    }
    fn t2_expansion_x2(
        &mut self,
        cut: &[usize],
        lanes: &[usize],
        a0: &[CMom],
        a1: &[CMom],
        e3: &[CMom],
        e4: &[CMom],
        param: &[[C64; 3]],
        out: &mut [C64],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self
            .inner
            .t2_expansion_x2(cut, lanes, a0, a1, e3, e4, param, out);
        self.record(started, lanes.len());
        result
    }
    // Per-lane rank: the engine lowers lanes declared at n+2 only through
    // these; the trait defaults would silently keep the rank n+2 engine.
    fn lane_rank(&mut self, lane: usize) -> Option<usize> {
        self.inner.lane_rank(lane)
    }
    fn lower_rank(&mut self, lanes: &[usize], rank: usize) -> Result<()> {
        self.inner.lower_rank(lanes, rank)
    }
    fn restore_rank(&mut self) {
        self.inner.restore_rank()
    }
}

struct ProfiledMasters<'a> {
    inner: &'a mut OneLoopNativeMasters,
    profile: &'a mut FastProfile,
}

impl FastMasters for ProfiledMasters<'_> {
    fn requires_scaleless_bubbles(&self) -> bool {
        self.inner.requires_scaleless_bubbles()
    }

    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        output: &mut [[C64; 3]],
        statuses: &mut [bool],
    ) -> Result<()> {
        let started = Instant::now();
        let result = self.inner.evaluate(family, args, output, statuses);
        self.profile.masters_time += started.elapsed();
        self.profile.master_calls += 1;
        self.profile.master_rows += statuses.len() as u64;
        result
    }
}

fn check_lanes(ids: &[u64], lanes: usize) -> Result<()> {
    if ids.len() != lanes {
        return Err(ninjars::Error::Shape {
            name: "original lane IDs",
            expected: lanes,
            actual: ids.len(),
        });
    }
    Ok(())
}

/// A reusable polynomial reduction plan for real `f64`, `DoubleFloat`, or
/// arbitrary precision `Float` coefficient carriers. Numerical batches stay in
/// NinjaRS/Symbolica from parameter binding through scalar masters.
///
/// The selected upstream generic coordinator supports any n ≥ 1 and rank ≤ n+2.
/// μ² has rank weight two. Invalid or repeated topologies are rejected upstream.
pub struct PolynomialReducer<T: NativeFloat>
where
    Complex<T>: Real + EvaluationDomain,
{
    numerator: PolynomialNumerator<T>,
    amplitude: Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>,
    cuts: SymbolicaCuts<T>,
    high_cuts: Option<SymbolicaHighCuts<T>>,
    masters: SymbolicaMasters<T>,
}

impl<T: NativeFloat> PolynomialReducer<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// `variables` are (q0,q1,q2,q3,μ²). `onshell_threshold=None` disables
    /// approximate snapping of small master arguments; a chosen tolerance must
    /// be supplied explicitly in the working precision.
    pub fn new(
        propagators: usize,
        expression: Atom,
        variables: [Atom; 5],
        parameters: Vec<Atom>,
        prototype: T,
        onshell_threshold: Option<T>,
    ) -> Result<Self> {
        let numerator =
            PolynomialNumerator::new(expression, variables, parameters, prototype.clone())?;
        let amplitude =
            SymbolicaAmplitudeKernels::new(propagators, numerator.rank(), prototype.clone())?
                .into_amplitude(0)?;
        Ok(Self {
            numerator,
            amplitude,
            cuts: SymbolicaCuts::new(prototype.clone())?,
            high_cuts: if propagators >= 4 {
                Some(SymbolicaHighCuts::new(prototype.clone())?)
            } else {
                None
            },
            masters: SymbolicaMasters::new(prototype).with_onshell_threshold(onshell_threshold),
        })
    }

    /// Select the masters' UV or IR pole residues (oneloopmaster's
    /// `PoleSelection`, `BOTH` by default). This generic engine retains the
    /// scaleless bubbles that carry separated residues; finite parts are
    /// unchanged.
    pub fn set_pole_selection(&mut self, selection: PoleSelection) {
        self.masters.set_pole_selection(selection);
    }

    pub fn pole_selection(&self) -> PoleSelection {
        self.masters.pole_selection()
    }

    /// Inputs and parameter rows are lane-major in the order of `ids`.
    /// Parameters are bound on every call, including empty and compact batches.
    pub fn evaluate(
        &mut self,
        ids: &[u64],
        input: AmplitudeInput<'_, Complex<T>>,
        parameters: &[Complex<T>],
    ) -> Result<Vec<Lane<Complex<T>, EvaluationStatus>>> {
        let lanes = input.scales_squared.len();
        check_lanes(ids, lanes)?;
        self.numerator.set_parameter_batch(parameters, lanes)?;
        // Ninja's generic API accumulates, while its fast API overwrites. This
        // boundary deliberately gives both paths identical replacement semantics.
        self.amplitude.reset_batch(lanes);
        if let Some(high) = &mut self.high_cuts {
            self.amplitude.evaluate_batch_with_inputs(
                input,
                &mut self.numerator,
                &mut self.cuts,
                high,
                &mut self.masters,
            )?;
        } else {
            self.amplitude.evaluate_batch_with_low_cuts(
                input,
                &mut self.numerator,
                &mut self.cuts,
                &mut self.masters,
            )?;
        }
        Ok(ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let status = self.amplitude.statuses()[row];
                Lane {
                    id,
                    status,
                    value: (status == EvaluationStatus::Success)
                        .then(|| self.amplitude.laurent(row).unwrap().clone().into()),
                }
            })
            .collect())
    }
}

/// Binary64 specialization using NinjaRS's own tensor or generic numerator.
/// Real shifts and complex masses follow `FastInput`'s documented contract.
/// The native master provider applies its fixed 1e-9 on-shell threshold and
/// the engine its `infrared_eps`, absolute constants in the model's units:
/// `runtime::PolynomialPlan`'s fast plans route every lane that reaches them
/// to the generic binary64 reducer (`runtime::reaches_fast_threshold`), and
/// without the invariant matrix refuse a lane with a light squared mass
/// (`runtime::light_mass_needs_invariants`); a direct caller does the same
/// when it needs exact-zero classification.
pub struct FastReducer<N: FastNumerator> {
    numerator: N,
    amplitude: FastAmplitude,
    masters: OneLoopNativeMasters,
    values: Vec<FastResult>,
    statuses: Vec<FastStatus>,
    profiling: bool,
    profile: FastProfile,
}

impl<N: FastNumerator> FastReducer<N> {
    pub fn new(propagators: usize, numerator: N) -> Result<Self> {
        let amplitude = FastAmplitude::new(propagators, numerator.rank(), FastOptions::default())?;
        Self::with_amplitude(numerator, amplitude)
    }

    fn with_amplitude(numerator: N, amplitude: FastAmplitude) -> Result<Self> {
        Ok(Self {
            amplitude,
            numerator,
            masters: OneLoopNativeMasters::new()?,
            values: Vec::new(),
            statuses: Vec::new(),
            profiling: false,
            profile: FastProfile::default(),
        })
    }

    /// Bind coefficients on an upstream `TensorNumerator`, or access
    /// `GenericNumerator::inner_mut()` to bind explicit-μ² polynomial parameters.
    pub fn numerator_mut(&mut self) -> &mut N {
        &mut self.numerator
    }

    /// Profiling is explicit and resets earlier measurements on every toggle.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        self.profile = FastProfile::default();
    }

    pub fn take_profile(&mut self) -> FastProfile {
        std::mem::take(&mut self.profile)
    }

    pub fn evaluate(
        &mut self,
        ids: &[u64],
        input: FastInput<'_>,
    ) -> Result<Vec<Lane<C64, FastStatus>>> {
        check_lanes(ids, input.lanes)?;
        self.values.resize(input.lanes, FastResult::default());
        self.statuses.resize(input.lanes, FastStatus::NotEvaluated);
        if self.profiling {
            let mut numerator_profile = FastProfile::default();
            let started = Instant::now();
            let result = self.amplitude.evaluate(
                &input,
                &mut ProfiledNumerator {
                    inner: &mut self.numerator,
                    profile: &mut numerator_profile,
                },
                &mut ProfiledMasters {
                    inner: &mut self.masters,
                    profile: &mut self.profile,
                },
                &mut self.values,
                &mut self.statuses,
            );
            self.profile.engine_time += started.elapsed();
            self.profile.engine_calls += 1;
            self.profile.merge(numerator_profile);
            result?;
        } else {
            self.amplitude.evaluate(
                &input,
                &mut self.numerator,
                &mut self.masters,
                &mut self.values,
                &mut self.statuses,
            )?;
        }
        Ok(ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let status = self.statuses[row];
                Lane {
                    id,
                    status,
                    value: (status == FastStatus::Success).then(|| self.values[row].laurent.into()),
                }
            })
            .collect())
    }
}

/// Binary64 fast reduction using NinjaRS's generic-polynomial callback adapter.
/// Explicit mu2 is retained; no coefficient is converted through textual form.
pub struct FastPolynomialReducer {
    reducer: FastReducer<GenericNumerator<PolynomialNumerator<f64>>>,
}
impl FastPolynomialReducer {
    pub fn set_profiling(&mut self, enabled: bool) {
        self.reducer.set_profiling(enabled);
    }
    pub fn take_profile(&mut self) -> FastProfile {
        self.reducer.take_profile()
    }
    pub fn new(
        propagators: usize,
        expression: Atom,
        variables: [Atom; 5],
        parameters: Vec<Atom>,
    ) -> Result<Self> {
        let numerator = PolynomialNumerator::new(expression, variables, parameters, 0_f64)?;
        Ok(Self {
            reducer: FastReducer::new(propagators, GenericNumerator::new(numerator))?,
        })
    }
    pub fn evaluate(
        &mut self,
        ids: &[u64],
        input: FastInput<'_>,
        parameters: &[Complex<f64>],
    ) -> Result<Vec<Lane<C64, FastStatus>>> {
        let started = self.reducer.profiling.then(Instant::now);
        let binding = self
            .reducer
            .numerator_mut()
            .inner_mut()
            .set_parameter_batch(parameters, input.lanes);
        if let Some(started) = started {
            self.reducer.profile.numerator_preparation_time += started.elapsed();
        }
        binding?;
        self.reducer.evaluate(ids, input)
    }
}

/// Binary64 dense μ² numerator using only upstream coefficient preparation,
/// tensor ordering/scatter and truncated-jet expansion operations. This is a
/// polynomial representation; it does not select a dimensional scheme or add R2.
pub struct FastTensorPolynomialReducer {
    reducer: FastReducer<MuTensorNumerator>,
    evaluator: CoefficientEvaluator<f64>,
    layout: TensorCoefficientPlan,
    heads: Vec<Complex<f64>>,
    native_heads: Vec<C64>,
}

impl FastTensorPolynomialReducer {
    pub fn set_profiling(&mut self, enabled: bool) {
        self.reducer.set_profiling(enabled);
    }
    pub fn take_profile(&mut self) -> FastProfile {
        self.reducer.take_profile()
    }
    pub fn new(
        propagators: usize,
        expression: Atom,
        variables: [Atom; 5],
        parameters: Vec<Atom>,
    ) -> Result<Self> {
        let error = |e: ninjars_graph::CoefficientError| ninjars::Error::Symbolic(e.to_string());
        let plan =
            PolynomialCoefficientPlan::new(expression, variables, parameters).map_err(error)?;
        // Reject unsupported rank/topology through the owner before allocating
        // a dense tensor layout, whose size grows quickly with rank.
        let amplitude = FastAmplitude::new(propagators, plan.rank(), FastOptions::default())?;
        let layout = TensorCoefficientPlan::with_mu_squared(&plan, None).map_err(error)?;
        let numerator = MuTensorNumerator::new(layout.rank(), layout.mu_powers(), 0)?;
        Ok(Self {
            reducer: FastReducer::with_amplitude(numerator, amplitude)?,
            evaluator: plan.compile(0_f64).map_err(error)?,
            layout,
            heads: Vec::new(),
            native_heads: Vec::new(),
        })
    }

    pub fn evaluate(
        &mut self,
        ids: &[u64],
        input: FastInput<'_>,
        parameters: &[Complex<f64>],
    ) -> Result<Vec<Lane<C64, FastStatus>>> {
        check_lanes(ids, input.lanes)?;
        let started = self.reducer.profiling.then(Instant::now);
        let prepared = self.prepare_coefficients(input.lanes, parameters);
        if let Some(started) = started {
            self.reducer.profile.numerator_preparation_time += started.elapsed();
        }
        prepared?;
        self.reducer.evaluate(ids, input)
    }

    fn prepare_coefficients(&mut self, lanes: usize, parameters: &[Complex<f64>]) -> Result<()> {
        let error = |e: ninjars_graph::CoefficientError| ninjars::Error::Symbolic(e.to_string());
        self.heads
            .resize(lanes * self.layout.head_count(), Complex::new(0., 0.));
        self.evaluator
            .evaluate_batch(parameters, &mut self.heads, lanes)
            .map_err(error)?;
        self.native_heads.clear();
        self.native_heads
            .extend(self.heads.iter().map(|z| C64::new(z.re, z.im)));
        let numerator = self.reducer.numerator_mut();
        numerator.set_lanes(lanes);
        // scatter writes every occupied and zero slot on every call. Resizing
        // upstream storage alone would retain coefficients from an older batch.
        self.layout
            .scatter(
                &self.native_heads,
                numerator.coefficients_mut(),
                lanes,
                &C64::ZERO,
            )
            .map_err(error)
    }
}
