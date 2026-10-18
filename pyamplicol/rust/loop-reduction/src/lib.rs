// SPDX-License-Identifier: 0BSD

//! Batched consumers of NinjaRS, without reduction or numerator algebra of our own.
//!
//! These are **unconverted** Ninja/OneLOop integrals, not physical amplitudes:
//! μ_R^(2ε)/(i π^(2−ε) rΓ) times the dimensionally regulated integral.
//! For example, A0(m²) = [m²(1−log(m²/μ_R²)), m², 0].
//! Model factors, interference, and the public amplitude normalization belong
//! outside this boundary. Every successful evaluation replaces previous values.

#![forbid(unsafe_code)]

use std::ops::Add;

use ninjars::fast::GenericNumerator;
use ninjars::fast::MuTensorNumerator;
pub use ninjars::fast::{C64, FastInput, FastStatus};
use ninjars::fast::{FastAmplitude, FastNumerator, FastOptions, FastResult};
use ninjars::{Amplitude, Numerator, Real, Result};
pub use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, FourMomentum, SamplingScaleProvider,
    SamplingScaleSource, SamplingScaleStatus,
};
use ninjars_graph::{CoefficientEvaluator, PolynomialCoefficientPlan, TensorCoefficientPlan};
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
/// The upstream generic coordinator currently supports n ≤ 6 and rank ≤ n+1.
/// μ² has rank weight two. Unsupported topologies are rejected upstream.
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
/// The native master provider applies its fixed 1e-9 on-shell threshold. Use
/// the generic binary64 reducer when the caller needs exact-zero classification.
pub struct FastReducer<N: FastNumerator> {
    numerator: N,
    amplitude: FastAmplitude,
    masters: OneLoopNativeMasters,
    values: Vec<FastResult>,
    statuses: Vec<FastStatus>,
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
        })
    }

    /// Bind coefficients on an upstream `TensorNumerator`, or access
    /// `GenericNumerator::inner_mut()` to bind explicit-μ² polynomial parameters.
    pub fn numerator_mut(&mut self) -> &mut N {
        &mut self.numerator
    }

    pub fn evaluate(
        &mut self,
        ids: &[u64],
        input: FastInput<'_>,
    ) -> Result<Vec<Lane<C64, FastStatus>>> {
        check_lanes(ids, input.lanes)?;
        self.values.resize(input.lanes, FastResult::default());
        self.statuses.resize(input.lanes, FastStatus::NotEvaluated);
        self.amplitude.evaluate(
            &input,
            &mut self.numerator,
            &mut self.masters,
            &mut self.values,
            &mut self.statuses,
        )?;
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
        self.reducer
            .numerator_mut()
            .inner_mut()
            .set_parameter_batch(parameters, input.lanes)?;
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
        let error = |e: ninjars_graph::CoefficientError| ninjars::Error::Symbolic(e.to_string());
        self.heads
            .resize(input.lanes * self.layout.head_count(), Complex::new(0., 0.));
        self.evaluator
            .evaluate_batch(parameters, &mut self.heads, input.lanes)
            .map_err(error)?;
        self.native_heads.clear();
        self.native_heads
            .extend(self.heads.iter().map(|z| C64::new(z.re, z.im)));
        let numerator = self.reducer.numerator_mut();
        numerator.set_lanes(input.lanes);
        // scatter writes every occupied and zero slot on every call. Resizing
        // upstream storage alone would retain coefficients from an older batch.
        self.layout
            .scatter(
                &self.native_heads,
                numerator.coefficients_mut(),
                input.lanes,
                &C64::ZERO,
            )
            .map_err(error)?;
        self.reducer.evaluate(ids, input)
    }
}
