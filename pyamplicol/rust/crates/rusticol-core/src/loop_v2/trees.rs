// SPDX-License-Identifier: 0BSD

//! Native per-point inputs of the v2 executor (WP3.1 I3/I4,
//! `workspace/reports/engine-v2/wp31/PLAN.md`): the block rows of the
//! loop-request tree program and the Born amplitude components of its root
//! closure, without Python per point.
//!
//! **Trees.** The slice's tree program (`loops/v2/assemble.TreeRows`: the
//! blocks DAG and its runtime schema) runs on one complex state per point:
//! the external wavefunctions (the tree runtime's own source records and
//! formulas, `engine::SchemaSources`), the all-outgoing momenta of every
//! momentum slot and the model parameters fill the state. Then, per runtime
//! stage of the interaction program (`native_trees._interaction_program`),
//! every interaction adds its exact colour weight times its vertex kernel on
//! its two input rows to its result current, and each output slot copies the
//! sum or applies the current's propagator kernel; kernels are a few saved
//! exact Symbolica evaluators, one per vertex and propagator shape, run
//! batched over the interactions sharing them in the carrier. A block row is
//! a signed state slice (its Wick sign, `loops/v2/blocks.Block.wick_sign`).
//!
//! **Born.** The Born is the recursion's last step: every root-closure term
//! (`assemble._born_terms`) closes an ordinary vertex holding leg 1 on the
//! block rows of its attached blocks with a plan kernel, times its sign, its
//! coupling per N_c grade and the exact colour weights of its literal
//! tensors, accumulated per Born helicity row into the components
//! `A0[b, Born order, grade]`. The plan's Born components (the seeds' inputs)
//! are a subset. With a colour metric (exact weights per pair of Born
//! components), `|A0|²` per squared Born order is formed as well.
//!
//! **Counterterms and fields.** The counterterm closure's terms run like the
//! Born's into ket vectors, interfered with the Born per sector
//! (`super::interference`).
//!
//! **Loop-side sources** (the asymmetric Ward mode, `loops/v2/ward.py`): per
//! point a literal complex four-vector may replace a vector leg's
//! wavefunction on the loop side only (`TreeProgram::inputs_loop_side`). The
//! trees then run twice: the physical run gives the Born bra (the plan's Born
//! components, the seeds' input, and `|A0|²`), the loop-side run the block
//! rows, the counterterm kets (with the counterterm program's own state
//! overridden) and the Born ket of the external-field insertions; the Born
//! interference of the physical bra with the loop-side ket is formed per
//! squared Born order (`TreeInputs::loop_born`). With `v = c p_j` on a gluon
//! leg every output vanishes by gauge invariance.
//!
//! **Parameters.** Every parameter-dependent value derives from the external
//! model parameters through one saved exact evaluator of the program (its
//! last saved evaluator): the tree program's derived model parameters, the
//! Born couplings per N_c grade and the plan's kernel constants, in the
//! carrier; the masses (external spinors, loop propagators) are the exact
//! mass parameters, and the renormalisation scale's parameter is sqrt(mu_R²)
//! of each point. A batch therefore hands over only momenta, scales and the
//! external parameter values (`TreeProgram::bind`).
//!
//! Every value derives from the stored program and the external parameters:
//! no Python per batch or point.
//!
//! **Carriers.** Each kernel and the parameter program come as a saved exact
//! evaluator and, from the generation (`native_trees.py`), as a SymJIT
//! storage-v4 plane application (IR, recompiled for the loading CPU and
//! sealed, `evaluator::symjit_plane`; DESIGN §3.7): binary64 runs the plane
//! applications and touches no Symbolica state, so it needs no licence;
//! DoubleFloat and arbitrary precision decode the exact states in the
//! carrier, under pyAmpliCol's Symbolica library key (`crate::licence`), as
//! binary64 does on an x86-64 CPU without AVX and FMA3 (where SymJIT cannot
//! compile).

use super::dress::{Carrier, cast, parse_rational};
use super::interference::{Entry, Interference, InterferenceRuntime, Ket, MetricCells};
use super::plan::LoopProgramV4;
use crate::engine::{PlaneDescriptor, SchemaSources, SymjitPlaneKernel, SymjitPlaneLayout};
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use symbolica::domains::float::{Complex, DoubleFloat, Float, Real};
use symbolica::domains::rational::Rational;
use symbolica::evaluate::{BatchExpressionEvaluator, EvaluationDomain};

/// The tree-program schema this executor runs (4: the counterterm program's
/// slot table and per-slot reachability masks, WP3.1 I5).
pub const TREE_SCHEMA: u32 = 4;

/// A saved exact evaluator with its coefficients cast to binary64, as
/// [`TreeProgram::runtime`] casts them (the evaluator binary64 would run
/// without a plane application). Call under the library key.
#[cfg(feature = "python-generation-bridge")]
fn binary64_evaluator(
    saved: &[u8],
) -> RusticolResult<symbolica::evaluate::ExpressionEvaluator<Complex<f64>>> {
    let zero = 0.0_f64;
    let complex = |re: &Rational, im: &Rational| Complex::new(cast(&zero, re), cast(&zero, im));
    Ok(
        crate::engine::decode_exact_evaluator_state(saved)?.map_coeff_with_prec(
            &|r| complex(&r.re, &r.im),
            symbolica::domains::float::FloatLike::get_precision(&zero),
        ),
    )
}

/// Generation: the binary64 SymJIT plane application of a saved exact
/// evaluator of a tree program (a kernel or the parameter program), with its
/// input and output counts. The evaluator is the one binary64 would run
/// without it (its coefficients, registered constants included, cast as
/// [`TreeProgram::runtime`] casts them), compiled at `opt_level` into
/// storage-v4 IR (`evaluator::symjit_plane`). `real_points` are rows of real
/// input values at which the program runs (the parameter program: the
/// request's parameter values), the plane check's real-axis rows
/// ([`verify_plane`]). An application that disagrees with its evaluator at a
/// probe row is compiled again at optimisation level 0 and kept if that one
/// passes the check: SymJIT's optimised AVX/AVX-512 code (2.27.3 through
/// v228 `c0ccfa4`) gave NaN where its scalar code and level 0 are right
/// (CM05's parameter program, M-SM MS-MASTERS; upstream report
/// `upstream/symjit-simd-complex-times-alias`). The plane
/// check's refusal is a value: `Ok(Err(mismatch))` when the plane
/// application disagrees with its exact evaluator at the check's probes at
/// both levels (the first level's mismatch), or when an output's real-axis
/// value cannot be certified without a real point (the generation refuses the
/// program, naming the output: `native_trees.NATIVE_EXPORT_REFUSALS` key
/// `plane-mismatch`); any other failure is an error.
#[cfg(feature = "python-generation-bridge")]
pub fn compile_plane_application_checked(
    saved: &[u8],
    opt_level: u8,
    compress: bool,
    real_points: &[Vec<f64>],
) -> RusticolResult<Result<(Vec<u8>, usize, usize), PlaneMismatch>> {
    let _unlock = crate::licence::unlock();
    let evaluator = binary64_evaluator(saved)?;
    let compile = |level: u8| {
        crate::engine::compile_symbolica_evaluator_to_plane_application_bytes(
            &evaluator, level, compress,
        )
    };
    let (plane, inputs, outputs) = compile(opt_level)?;
    if let Some(point) = real_points.iter().find(|point| point.len() != inputs) {
        return Err(invalid(format!(
            "plane check: a real point of {} values for a program of {inputs} inputs",
            point.len()
        )));
    }
    let sensitive = if real_points.is_empty() {
        crate::engine::real_axis_sensitive_outputs(&evaluator)?
    } else {
        Vec::new()
    };
    if let Some(&output) = sensitive.first() {
        return Ok(Err(PlaneMismatch {
            output,
            probe: PlaneProbe::NoRealPoint,
        }));
    }
    let mut batch = evaluator.clone().into_batch_evaluator();
    let shape = (inputs, outputs);
    Ok(level_zero_fallback(
        plane,
        opt_level,
        || Ok(compile(0)?.0),
        |plane| verify_plane(plane, &mut batch, shape, real_points),
    )?
    .map(|plane| (plane, inputs, outputs)))
}

/// [`compile_plane_application_checked`]'s choice: `plane` (compiled at
/// `opt_level`) when `verify` passes it; after a probe-row disagreement at
/// `opt_level > 0`, the level-0 compilation (`level0`) when `verify` passes
/// that one; otherwise the first disagreement. A program without a real
/// point to certify a branch side (`PlaneProbe::NoRealPoint`) is not
/// recompiled: no level certifies it.
#[cfg(any(test, feature = "python-generation-bridge"))]
fn level_zero_fallback<P>(
    plane: P,
    opt_level: u8,
    level0: impl FnOnce() -> RusticolResult<P>,
    mut verify: impl FnMut(&P) -> RusticolResult<Option<PlaneMismatch>>,
) -> RusticolResult<Result<P, PlaneMismatch>> {
    let Some(mismatch) = verify(&plane)? else {
        return Ok(Ok(plane));
    };
    if opt_level > 0 && matches!(mismatch.probe, PlaneProbe::Row { .. }) {
        let plane = level0()?;
        if verify(&plane)?.is_none() {
            return Ok(Ok(plane));
        }
    }
    Ok(Err(mismatch))
}

/// What the plane check refuses ([`compile_plane_application_checked`]): the
/// output and how.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneMismatch {
    pub output: usize,
    pub probe: PlaneProbe,
}

/// How a plane check refuses an output ([`PlaneMismatch`]).
#[derive(Clone, Debug, PartialEq)]
pub enum PlaneProbe {
    /// The plane application and its exact evaluator disagree at a probe row
    /// (off the real axis, or one of the caller's real points), in the real or
    /// imaginary part, with the two binary64 values.
    Row {
        row: usize,
        real_axis: bool,
        imaginary: bool,
        plane: f64,
        exact: f64,
    },
    /// The output's value at real inputs depends on a branch side or a
    /// discontinuity (`engine::real_axis_sensitive_outputs`) and the caller
    /// gave no real point to certify it at.
    NoRealPoint,
}

impl std::fmt::Display for PlaneMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.probe {
            PlaneProbe::Row {
                row,
                real_axis,
                imaginary,
                plane,
                exact,
            } => write!(
                f,
                "the SymJIT plane application disagrees with its exact evaluator at \
                 output {} ({} part, {} probe row {row}: {plane:e} against {exact:e})",
                self.output,
                if *imaginary { "imaginary" } else { "real" },
                if *real_axis { "real-axis" } else { "complex" },
            ),
            PlaneProbe::NoRealPoint => write!(
                f,
                "output {} reads a branch-cut or discontinuous function of the \
                 inputs, whose side at real inputs only a real point of the \
                 program can certify, and none was given",
                self.output
            ),
        }
    }
}

/// The plane check: the first disagreement of a plane application with its
/// evaluator, or `None`. A translation error (a constant or a function read
/// wrongly) changes an output at O(1), while the two evaluations' rounding
/// differs by orders of magnitude less (the criterion of the tree engine's
/// exporter, `symbolica_helpers._verify_symjit_plane_constants`).
///
/// Its rows: two complex rows off the real axis (real parts in (0.5, 1.5),
/// imaginary parts of either sign in 0.25..0.75 in magnitude, a deterministic
/// low-discrepancy sequence), where no input-dependent argument sits on a
/// branch cut, so the check certifies the translation whatever the inputs'
/// order, names and values; then `real_points`, the real inputs at which the
/// program runs, which certify the real-axis evaluation there (a signed
/// zero's side of a cut, a predicate's branch). The former real probes,
/// assigned by input index, put arguments on cuts at arbitrary points and
/// made the verdict depend on the parameters' declaration order; at real
/// inputs the plane takes exact arithmetic's side of a cut by construction
/// (`symjit_plane::canonical_cut_row`, M-SM MS-MASTERS: before it, SymJIT's
/// negated real had a -0 imaginary part where Symbolica's evaluator has +0,
/// `sqrt(-x)` at x > 0: -i sqrt(x) against +i sqrt(x)).
/// The rows run together, then each in a call of its own, as a batch of
/// equal rows runs them: a SIMD block of equal lanes (the padding copies
/// the last row) runs SymJIT's SIMD code, which a block whose lanes take
/// different branches replays in scalar code. CM05's parameter program gave
/// NaN in SIMD only (M-SM MS-MASTERS, DESIGN risk 13).
#[cfg(feature = "python-generation-bridge")]
fn verify_plane(
    plane: &[u8],
    evaluator: &mut BatchExpressionEvaluator<Complex<f64>>,
    shape: (usize, usize),
    real_points: &[Vec<f64>],
) -> RusticolResult<Option<PlaneMismatch>> {
    const OFF_AXIS: usize = 2;
    let (inputs, outputs) = shape;
    let rows = OFF_AXIS + real_points.len();
    let mut values = off_axis_probes(inputs, OFF_AXIS);
    for point in real_points {
        values.extend(point.iter().map(|&x| Complex::new(x, 0.0)));
    }
    let mut expected = vec![Complex::new(0.0, 0.0); rows * outputs];
    evaluator
        .evaluate_batch(&values, &mut expected, rows)
        .map_err(|e| invalid(format!("plane check: {e}")))?;
    // One load (one compilation) for every call, as the runtime's kernel.
    let mut kernel = BinaryPlane::load(plane, shape)?;
    let mut got = vec![Complex::new(0.0, 0.0); rows * outputs];
    kernel.evaluate(&values, &mut got, rows)?;
    let mixed = first_mismatch(&expected, &got, outputs).map(|mut mismatch| {
        if let PlaneProbe::Row { row, real_axis, .. } = &mut mismatch.probe {
            if *row >= OFF_AXIS {
                *row -= OFF_AXIS;
                *real_axis = true;
            }
        }
        mismatch
    });
    if mixed.is_some() {
        return Ok(mixed);
    }
    let scales = output_scales(&expected, outputs);
    let mut got = vec![Complex::new(0.0, 0.0); outputs];
    for row in 0..rows {
        kernel.evaluate(&values[row * inputs..][..inputs], &mut got, 1)?;
        if let Some(mut mismatch) = first_mismatch_at(
            &expected[row * outputs..][..outputs],
            &got,
            outputs,
            &scales,
        ) {
            if let PlaneProbe::Row {
                row: probe,
                real_axis,
                ..
            } = &mut mismatch.probe
            {
                *real_axis = row >= OFF_AXIS;
                *probe = if row >= OFF_AXIS { row - OFF_AXIS } else { row };
            }
            return Ok(Some(mismatch));
        }
    }
    Ok(None)
}

/// `rows` rows of `inputs` complex probe values off the real axis: real parts
/// 0.5 + frac(k / phi) and imaginary parts 0.25 + 0.5 frac(k / rho) (phi the
/// golden ratio, rho the plastic number: a two-dimensional low-discrepancy
/// sequence, every value distinct), positive in even rows and negative in
/// odd rows (both half-planes).
#[cfg(any(test, feature = "python-generation-bridge"))]
fn off_axis_probes(inputs: usize, rows: usize) -> Vec<Complex<f64>> {
    (0..rows * inputs)
        .map(|k| {
            let n = (k + 1) as f64;
            let sign = if (k / inputs.max(1)) % 2 == 0 { 1.0 } else { -1.0 };
            Complex::new(
                0.5 + (0.618_033_988_749_894_8 * n).fract(),
                sign * (0.25 + 0.5 * (0.754_877_666_246_692_7 * n).fract()),
            )
        })
        .collect()
}

/// The first value where `got` (the plane application's rows) and `expected`
/// (the evaluator's) disagree beyond rounding, row-major over `outputs`
/// complex outputs per row: a part agrees bitwise, both NaN, within 1e-6 of
/// itself, or within 1e-12 of its output's scale (the largest finite part of
/// that output over the rows: a part that cancels to rounding, e.g. the zero
/// imaginary part of a real coupling, against its output's size, and a small
/// output not against a large one's).
#[cfg(any(test, feature = "python-generation-bridge"))]
fn first_mismatch(
    expected: &[Complex<f64>],
    got: &[Complex<f64>],
    outputs: usize,
) -> Option<PlaneMismatch> {
    first_mismatch_at(expected, got, outputs, &output_scales(expected, outputs))
}

/// Per output the largest finite part of `expected` (row-major over
/// `outputs` complex outputs per row): [`first_mismatch`]'s scales.
#[cfg(any(test, feature = "python-generation-bridge"))]
fn output_scales(expected: &[Complex<f64>], outputs: usize) -> Vec<f64> {
    let width = outputs.max(1);
    let mut scales = vec![0.0_f64; width];
    for (index, z) in expected.iter().enumerate() {
        for x in [z.re, z.im] {
            if x.is_finite() {
                scales[index % width] = scales[index % width].max(x.abs());
            }
        }
    }
    scales
}

/// [`first_mismatch`] with given per-output `scales` (the plane check's
/// single-row SIMD calls take those of all its rows, so a part that cancels
/// to rounding in one row is measured against its output's size).
#[cfg(any(test, feature = "python-generation-bridge"))]
fn first_mismatch_at(
    expected: &[Complex<f64>],
    got: &[Complex<f64>],
    outputs: usize,
    scales: &[f64],
) -> Option<PlaneMismatch> {
    let parts = |values: &[Complex<f64>]| -> Vec<f64> {
        values.iter().flat_map(|z| [z.re, z.im]).collect()
    };
    let width = outputs.max(1);
    let (expected, got) = (parts(expected), parts(got));
    for (index, (&e, &g)) in expected.iter().zip(&got).enumerate() {
        let scale = scales[(index / 2) % width];
        let agree = e.to_bits() == g.to_bits()
            || (e.is_nan() && g.is_nan())
            || (e - g).abs() <= 1e-6 * e.abs()
            || (e - g).abs() <= 1e-12 * scale;
        if !agree {
            let value = index / 2;
            return Some(PlaneMismatch {
                output: value % outputs.max(1),
                probe: PlaneProbe::Row {
                    row: value / outputs.max(1),
                    real_axis: false,
                    imaginary: index % 2 == 1,
                    plane: g,
                    exact: e,
                },
            });
        }
    }
    None
}

/// The binary64 rows of a plane application (`compile_plane_application_checked`)
/// through the production loader, row-major: a cold-path check of an export
/// against its exact evaluator.
#[cfg(feature = "python-generation-bridge")]
pub fn evaluate_plane_application(
    plane: &[u8],
    shape: (usize, usize),
    inputs: &[Complex<f64>],
    rows: usize,
) -> RusticolResult<Vec<Complex<f64>>> {
    let mut kernel = BinaryPlane::load(plane, shape)?;
    let mut outputs = vec![Complex::new(0.0, 0.0); rows * shape.1];
    kernel.evaluate(inputs, &mut outputs, rows)?;
    Ok(outputs)
}

/// The binary64 rows of a saved exact evaluator (Symbolica's evaluation of
/// the state), row-major: the reference of `evaluate_plane_application`.
#[cfg(feature = "python-generation-bridge")]
pub fn evaluate_exact_state(
    saved: &[u8],
    inputs: &[Complex<f64>],
    rows: usize,
) -> RusticolResult<Vec<Complex<f64>>> {
    let mut kernel = exact_kernel(saved, &0.0_f64)?;
    let mut outputs = vec![Complex::new(0.0, 0.0); rows * kernel.output_len()];
    let _unlock = crate::licence::unlock();
    kernel
        .evaluate_batch(inputs, &mut outputs, rows)
        .map_err(invalid)?;
    Ok(outputs)
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(format!("loop v2 trees: {}", message.into()))
}

fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn exact(value: &str) -> RusticolResult<Rational> {
    parse_rational(value).map_err(|error| invalid(error.message()))
}

/// The key of the complex-mass master refusal (M-SM MS-MASTERS).
pub const COMPLEX_MASS_MASTER: &str = "complex-mass-master";
/// The smallest |Im m²| / Re m² a complex pole mass may take at a batch's
/// parameters in a carrier that guards it
/// ([`TreeCarrier::COMPLEX_MASS_MASTER_GUARD`]).
pub const COMPLEX_MASS_MASTER_RATIO: f64 = 1e-150;

/// The complex-mass master guard (M-SM MS-MASTERS) on a pole mass's square
/// `re + i im` (plan mass `index`) at a batch's parameters: refused, as a
/// model-parameter error naming `complex-mass-master`, when
/// `0 < |im| < 1e-150 re`. Only double-double calls it
/// ([`TreeCarrier::COMPLEX_MASS_MASTER_GUARD`]): its C0 turned non-finite
/// there (CM01 at 1e-155, 1e-160 and 1e-300 of the masses, intermittently),
/// while binary64 and arbitrary precision equal the zero-width value. A
/// deliberate refusal of unphysical widths, with no upstream track (no CM
/// row reaches it). An exactly real pole mass (`im = 0`, a zero width) is
/// the real-mass limit and passes.
fn require_complex_mass_master(index: usize, re: &Rational, im: &Rational) -> RusticolResult<()> {
    if im.is_zero() || re.is_zero() {
        return Ok(());
    }
    let ratio = (im.abs() / re.abs()).to_f64();
    if ratio >= COMPLEX_MASS_MASTER_RATIO {
        return Ok(());
    }
    let (re64, im64) = (re.to_f64(), im.to_f64());
    Err(RusticolError::model_parameter(format!(
        "{COMPLEX_MASS_MASTER}: the complex pole mass m = {:.6} (m^2 = {re64:e} \
         {im64:+e} i; loop mass {index}) has |Im m^2| / Re m^2 = {ratio:e} at \
         these model parameters, below 1e-150: its width is below 1e-150 of \
         the mass, where the double-double complex-mass master integrals are \
         unreliable. Instead: evaluate in binary64 or arbitrary precision, give \
         that width at least 1e-150 of its mass, or zero",
        re64.abs().sqrt()
    )))
}

// ------------------------------------------------------------------ file form

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeFile {
    schema: u32,
    legs: usize,
    initial: Vec<bool>,
    state_size: usize,
    model_offset: usize,
    model_count: usize,
    /// (absolute component start, leg mask; bit i = label i + 1).
    momentum_slots: Vec<(usize, u64)>,
    source_fill: serde_json::Value,
    /// The interaction program: its kernels' saved evaluators follow in
    /// order, vertex kernels then propagator kernels.
    interactions: InteractionFile,
    /// (state component start, dimension, Wick sign) per plan block row.
    block_rows: Vec<(usize, usize, i32)>,
    /// Further (plan block row, state component start, Wick sign) terms of
    /// rows whose block current the tree engine splits into several trees
    /// (by flavour flow): each adds its current to the row, in order.
    #[serde(default)]
    block_row_terms: Vec<(usize, usize, i32)>,
    born: BornFile,
    parameters: ParameterFile,
    /// The counterterm and external-field components (none: Born and bare
    /// virtual only).
    counterterms: Option<CountertermFile>,
    /// The counterterm program of the tree recursion (WP2.2), a second
    /// interaction program on a state of its own; its kernels' saved
    /// evaluators follow the trees'. Closure blocks at or past `offset` (the
    /// trees' state size) read its state.
    #[serde(default)]
    counterterm_trees: Option<CountertermTreesFile>,
    /// Kernel input layouts of the closure terms: ("current", slot,
    /// component) or ("momentum", slot, mu).
    layouts: Vec<Vec<(String, usize, usize)>>,
    /// Colour weights of the closure terms: (re, im, sqrt(2) power p), the
    /// weight times 2^(-p/2).
    weights: Vec<(String, String, i32)>,
}

/// The parameter program: one saved exact evaluator (the program's last
/// saved evaluator) from the external parameters (real, in `names` order)
/// to complex outputs.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterFile {
    names: Vec<String>,
    /// Exact default value per external parameter (decimal or fraction).
    defaults: Vec<String>,
    outputs: usize,
    /// Per tree-program model parameter: ("input", external index, false)
    /// or ("output", output index, imaginary part).
    model: Vec<(String, usize, bool)>,
    /// The same per model parameter of the counterterm program.
    #[serde(default)]
    counterterm_model: Vec<(String, usize, bool)>,
    /// Output index per Born coupling slot.
    couplings: Vec<usize>,
    /// Output index per counterterm or field coupling slot (may depend on
    /// the renormalisation scale).
    counterterm_couplings: Vec<usize>,
    /// Outputs that must be finite (active diagnostics of the scalar
    /// library's helper functions).
    finite: Vec<usize>,
    /// Output index per plan kernel constant.
    constants: Vec<usize>,
    /// Outputs that must be real and positive (bases a fractional power
    /// was distributed over).
    positive: Vec<usize>,
    /// (PDG code, external index) of every massive external particle.
    masses: Vec<(i32, usize)>,
    /// (PDG code, output index) of every massive external particle whose
    /// mass is a derived model parameter (e.g. loop_sm's MW): a carrier
    /// value of the program, real and positive (also in `positive`). Its
    /// legs are bound like an exact mass's, to the square of the carrier
    /// value's exact rational (the value its sources take; DESIGN §11.1).
    #[serde(default)]
    derived_masses: Vec<(i32, usize)>,
    /// Per plan mass (squared): the external index of its mass parameter,
    /// `["output", o]` for a derived mass parameter (output `o`, real and
    /// positive, also in `positive`; bound as `derived_masses` are),
    /// `["complex", o]` for a complex one (a complex-mass scheme's pole mass:
    /// output `o`, bound to the square of its carrier value's exact complex
    /// rational, `BoundParameters::masses_imaginary`), or none (massless).
    plan_masses: Vec<Option<MassFile>>,
    /// The renormalisation scale's parameter, if the program reads it:
    /// sqrt(mu_R^2) of each point.
    scale: Option<usize>,
    /// Every output that depends on the scale's parameter, with its name
    /// (generation's structural analysis, `native_trees._scale_outputs`;
    /// empty without a scale input). Absent in an artifact written before
    /// the analysis: its batches are then checked per point.
    #[serde(default)]
    scale_outputs: Option<Vec<(usize, String)>>,
    /// The on-shell external photons of an electroweak request (WP3.2
    /// EWI-6b, the alpha(0) convention): the output index of
    /// alpha(0) / alpha (real and positive, also in `positive`) and the
    /// photons' 1-based leg labels. Every source of those legs, spin and
    /// loop-side sources included, takes sqrt(alpha(0) / alpha) in the
    /// trees and the counterterm program, so the Born, the loops, the
    /// counterterm kets and the field insertions carry (alpha(0) / alpha)^n
    /// and the average stays the slice's. Written only when present: an
    /// older native refuses the record (unknown field).
    #[serde(default)]
    photons: Option<(usize, Vec<usize>)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CountertermTreesFile {
    /// The trees' state size: closure block starts at or past it are in
    /// this program's state.
    offset: usize,
    state_size: usize,
    model_offset: usize,
    model_count: usize,
    momentum_slots: Vec<(usize, u64)>,
    source_fill: serde_json::Value,
    /// The slot table: (pseudo-order, component, epsilon power, N_c grade)
    /// per counterterm slot (`native_trees.counterterm_slots`).
    slots: Vec<(String, String, i32, i32)>,
    interactions: InteractionFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractionFile {
    /// (inputs, outputs, model parameters) per vertex kernel: inputs
    /// `[left, right, p_left, p_right, model]`, `model` the listed model
    /// parameters (indices into the state's model block, increasing).
    kernels: Vec<(usize, usize, Vec<usize>)>,
    /// (inputs, outputs, model parameters) per propagator kernel: inputs
    /// `[value, p, model]`.
    propagators: Vec<(usize, usize, Vec<usize>)>,
    steps: Vec<StepFile>,
    /// A counterterm program's slot sets (indices into its slot table,
    /// `CountertermTreesFile::slots`); each step's `accumulator_masks` index
    /// them. Absent for the trees.
    #[serde(default)]
    masks: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepFile {
    /// (kernel, left start, left dimension, right start, right dimension,
    /// left momentum start, right momentum start, accumulator, weight re,
    /// weight im).
    #[allow(clippy::type_complexity)]
    interactions: Vec<(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        String,
        String,
    )>,
    /// Dimension of each accumulator (one per output current).
    accumulators: Vec<usize>,
    /// (accumulator, propagator or -1 to copy, momentum start, state start,
    /// dimension).
    outputs: Vec<(usize, i64, usize, usize, usize)>,
    /// Per accumulator the slot set (`InteractionFile::masks`) its result
    /// current reaches (a counterterm program; empty for the trees): an
    /// evaluation skips the interactions and outputs of the currents that
    /// reach no slot it computes (finite-only: no slot of epsilon power 0).
    #[serde(default)]
    accumulator_masks: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BornFile {
    rows: usize,
    components: usize,
    /// Per plan Born component its index in `components` (none: zero).
    plan_components: Vec<Option<usize>>,
    /// Per closure block, per Born row its state component start.
    blocks: Vec<Vec<Option<usize>>>,
    couplings: usize,
    terms: Vec<TermFile>,
    /// Per amplitude (basis element, Born order): (component, exact factor).
    amplitudes: Vec<Vec<(usize, String)>>,
    /// Born orders per amplitude row: amplitude = basis * orders + order.
    orders: usize,
    /// Per squared Born order its (left, right) Born-order pairs.
    squared: Vec<Vec<(usize, usize)>>,
    /// Per squared Born order the metric over Born components, (squared
    /// order, component, component, re, im): the exact Gram cell times N_c
    /// to the components' powers at N_c = 3, the powers a leading or
    /// next-to-leading colour projection keeps (sparse; none: no |A0|²).
    metric: Option<Vec<MetricFileCell>>,
    average: String,
}

/// A Born metric cell as stored: (squared order, component, component, re,
/// im), the parts exact decimal strings.
type MetricFileCell = (usize, usize, usize, String, String);

/// A Born metric cell: (squared order, component, component, [re, im]).
type MetricCell = (usize, usize, usize, [Rational; 2]);

/// The counterterm closure and the interferences of the counterterm and
/// external-field components (`native_trees._counterterm_program`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CountertermFile {
    couplings: usize,
    /// Epsilon power of each coupling slot.
    coupling_powers: Vec<i32>,
    /// (ket tensor, vector, N_c power) per ket component.
    ket: Vec<(usize, usize, i32)>,
    vectors: usize,
    /// Ket tensors (the Born basis first).
    tensors: usize,
    /// (basis element, ket tensor, N_c power, re, im): exact overlaps.
    cells: Vec<(usize, usize, i32, String, String)>,
    /// (basis element, Born order, N_c power) per Born component.
    born: Vec<(usize, usize, i32)>,
    /// Counterterm closure terms: contributions (coupling slot, ket
    /// component, weight re, weight im, sqrt(2) power).
    terms: Vec<TermFile>,
    /// (kind "vector" or "born", bra Born order, vector or ket Born order,
    /// coupling slot, N_c grade, output, sector degree).
    #[allow(clippy::type_complexity)]
    entries: Vec<(String, usize, usize, Option<usize>, i32, usize, Option<i32>)>,
    /// Epsilon power per output.
    outputs: Vec<i32>,
    /// Kept N_c powers below the degree: none (full colour), 0 or 2.
    depth: Option<i32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TermFile {
    kernel: usize,
    /// The term's kernel input layout (`TreeFile::layouts`).
    layout: usize,
    /// Closure block of each slot.
    blocks: Vec<usize>,
    /// Leg mask of each slot.
    masks: Vec<u64>,
    sign: i32,
    /// (coupling, component, colour weight): the term adds coupling *
    /// weight * value to the component (`TreeFile::weights`).
    contributions: Vec<(usize, usize, usize)>,
}

// ------------------------------------------------------------ program form

#[derive(Clone, Copy, Debug)]
enum TermInput {
    Current { slot: usize, component: usize },
    Momentum { slot: usize, mu: usize },
}

/// One interaction of the interaction program.
struct Edge {
    left: usize,
    left_dimension: usize,
    right: usize,
    right_dimension: usize,
    left_momentum: usize,
    right_momentum: usize,
    /// Offset of the result's accumulator in the step's accumulator row.
    accumulator: usize,
    weight: [Rational; 2],
    /// Whether only epsilon-pole terms read its result.
    pole: bool,
}

/// One runtime stage of the interaction program.
struct Step {
    /// Per vertex kernel the interactions it evaluates.
    groups: Vec<(usize, Vec<Edge>)>,
    /// Accumulator components per point.
    width: usize,
    /// (accumulator offset, propagator, momentum start, state start,
    /// dimension, only epsilon-pole terms read it).
    outputs: Vec<(usize, Option<usize>, usize, usize, usize, bool)>,
}

struct Interactions {
    /// Saved exact evaluators: vertex kernels, then propagator kernels.
    saved: Vec<Vec<u8>>,
    /// Per kernel (vertex kernels, then propagator kernels) the model
    /// parameters it reads: indices into the state's model block, so
    /// state components `model_offset + index` (tree schema 3). No kernel
    /// reads the whole block.
    models: Vec<Vec<usize>>,
    /// Their binary64 plane applications, in the same order (none: binary64
    /// runs the exact states).
    planes: Option<Vec<Vec<u8>>>,
    /// (inputs, outputs) of each kernel, in the same order.
    shapes: Vec<(usize, usize)>,
    steps: Vec<Step>,
}

/// The state of an interaction program: its size and model parameters.
#[derive(Clone, Copy)]
struct StateLayout {
    size: usize,
    /// The model block: state components `model_offset ..
    /// model_offset + model_block`, filled per point. Kernels read the
    /// subsets `Interactions::models` of it (schema 3; the field was
    /// `model_count` while every kernel read the whole block).
    model_offset: usize,
    model_block: usize,
}

/// The counterterm program (`CountertermTreesFile`).
struct CountertermTrees {
    layout: StateLayout,
    momentum_slots: Vec<(usize, u64)>,
    sources: SchemaSources,
    interactions: Interactions,
}

struct Term {
    kernel: usize,
    layout: usize,
    blocks: Vec<usize>,
    /// Per slot its leg mask's index in the program's `masks`.
    masks: Vec<usize>,
    sign: i32,
    /// (coupling, component, colour weight).
    contributions: Vec<(usize, usize, usize)>,
}

/// The shared tables of the closure terms.
struct Closures<'a> {
    /// Per layout its inputs and the number of slots it reads.
    layouts: &'a [(Vec<TermInput>, usize)],
    weights: usize,
    masks: &'a mut BTreeMap<u64, usize>,
}

/// Where a tree-program model parameter comes from.
#[derive(Clone, Copy, Debug)]
enum ModelSource {
    Input(usize),
    Output { index: usize, imaginary: bool },
}

/// A plan mass's source in the record (`ParameterFile::plan_masses`):
/// `["output", o]` or `["complex", o]` (the kind is checked on loading).
#[derive(Deserialize)]
#[serde(untagged)]
enum MassFile {
    Input(usize),
    Output(String, usize),
}

/// A plan mass's source: an external parameter (exact) or a parameter
/// program output (a derived mass, the exact value of its carrier: real and
/// positive, or complex).
#[derive(Clone, Copy)]
enum MassSource {
    Input(usize),
    Output(usize),
    Complex(usize),
}

struct Parameters {
    names: Vec<String>,
    defaults: Vec<Rational>,
    saved: Vec<u8>,
    /// The binary64 plane application of the program (none: binary64 runs
    /// the exact state).
    plane: Option<Vec<u8>>,
    outputs: usize,
    model: Vec<ModelSource>,
    counterterm_model: Vec<ModelSource>,
    couplings: Vec<usize>,
    counterterm_couplings: Vec<usize>,
    finite: Vec<usize>,
    constants: Vec<usize>,
    positive: Vec<usize>,
    masses: Vec<(i32, usize)>,
    derived_masses: Vec<(i32, usize)>,
    plan_masses: Vec<Option<MassSource>>,
    scale: Option<usize>,
    scale_outputs: Option<Vec<(usize, String)>>,
    /// `ParameterFile::photons`.
    photons: Option<(usize, Vec<usize>)>,
}

impl Parameters {
    /// `TreeProgram::scale_dependent_shared`.
    fn scale_dependent_shared(&self) -> Option<Vec<&str>> {
        let program = self;
        let dependent = program.scale_outputs.as_ref()?;
        let shared: BTreeSet<usize> = program
            .couplings
            .iter()
            .chain(&program.constants)
            .chain(program.derived_masses.iter().map(|(_, o)| o))
            .chain(program.plan_masses.iter().filter_map(|mass| match mass {
                Some(MassSource::Output(o) | MassSource::Complex(o)) => Some(o),
                _ => None,
            }))
            .copied()
            .collect();
        Some(
            dependent
                .iter()
                .filter(|(o, _)| shared.contains(o))
                .map(|(_, name)| name.as_str())
                .collect(),
        )
    }
}

/// The counterterm closure of a program and its interferences.
struct Counterterms {
    couplings: usize,
    /// Per coupling slot: whether it is the epsilon^0 coefficient.
    finite_slot: Vec<bool>,
    components: usize,
    terms: Vec<Term>,
    interference: Interference,
    /// The interference layout observable columns share
    /// (`TreeProgram::set_columns`): Born components, ket components, ket
    /// vectors and tensors, Born orders, the projection depth.
    layout: InterferenceLayout,
}

struct InterferenceLayout {
    born: Vec<(usize, usize, i32)>,
    ket: Vec<(usize, usize, i32)>,
    vectors: usize,
    tensors: usize,
    basis: usize,
    orders: usize,
    depth: Option<i32>,
}

/// One observable column of a process (`loops/v2/native_observables.py`):
/// its Born and counterterm/field interferences over the Born basis and the
/// ket tensors with its own metric (a colour history's operator) and
/// degrees, the swapped metric of a directed history, and the Born rows and
/// bra basis elements it keeps (helicity-resolved columns). Columns with the
/// same entries and metric share one interference (`Columns::programs`).
struct Column {
    direct: usize,
    swapped: Option<usize>,
    /// Outputs `0 .. born_outputs` are Born interferences (per squared Born
    /// order, the direct metric only); the others counterterm and field
    /// outputs (direct plus the conjugate swapped value, or twice the real
    /// part of the direct one without a swapped metric).
    born_outputs: usize,
    rows: Option<Vec<bool>>,
    basis: Option<Vec<bool>>,
}

/// The spin-source slots of one interaction program's state: per
/// (1-based) leg label its vector source slots with the source's crossing
/// phase (`correlated_exact._fill_correlated_sources`).
type SourceSlots = BTreeMap<usize, Vec<(usize, [Rational; 2])>>;

/// The observable columns of a process (`TreeProgram::set_columns`).
pub struct Columns {
    /// The distinct interferences (entries and metric) of the columns.
    programs: Vec<Interference>,
    columns: Vec<Column>,
    /// Spin-source slots of the trees' state and the counterterm program's.
    trees: SourceSlots,
    counterterms: SourceSlots,
}

/// The overlap cells of one colour metric: its distinct cells (per N_c
/// power, re, im) and, per ket tensor and Born basis element, the index of
/// its cell (`native_observables._Metrics`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CellsFile {
    values: Vec<Vec<(i32, String, String)>>,
    ids: Vec<Vec<u32>>,
}

/// One interference program of the columns: its entries (kind, bra Born
/// order, vector or Born order, coupling slot, N_c grade, output, degree),
/// the epsilon power of each output and the number of Born outputs.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnProgramFile {
    #[allow(clippy::type_complexity)]
    entries: Vec<(String, usize, usize, Option<usize>, i32, usize, Option<i32>)>,
    outputs: Vec<i32>,
    born_outputs: usize,
}

/// One column: its program, its direct and (directed histories) swapped
/// metric (indices into `ColumnsFile::cells`, whose None entry is the Born
/// metric of the counterterm interference itself), the Born rows and bra
/// basis elements it keeps (None: all).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnFile {
    program: usize,
    cells: usize,
    swapped: Option<usize>,
    rows: Option<Vec<usize>>,
    basis: Option<Vec<usize>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnsFile {
    cells: Vec<Option<CellsFile>>,
    programs: Vec<ColumnProgramFile>,
    columns: Vec<ColumnFile>,
    /// (leg label, state component start, phase re, phase im) per vector
    /// source slot of the trees' state and of the counterterm program's.
    trees: Vec<(usize, usize, String, String)>,
    counterterms: Vec<(usize, usize, String, String)>,
}

/// The spin sources of one point: per (1-based) leg label a literal complex
/// four-vector replacing the leg's wavefunction in the trees and the
/// counterterm program, and the Born rows the point keeps (a source leg's
/// fixed helicity row; `None`: every row).
pub struct PointSources<T> {
    pub vectors: Vec<(usize, [Complex<T>; 4])>,
    pub rows: Option<Vec<bool>>,
}

/// The loop-side sources of one point (`TreeProgram::inputs_loop_side`):
/// per (1-based) vector leg label a literal complex four-vector replacing the
/// leg's wavefunction in the loop side's trees and counterterm program only
/// (empty: the point's loop side is its physical one).
pub type LoopSources<T> = Vec<(usize, [Complex<T>; 4])>;

/// A loaded tree program (one process, any carrier).
pub struct TreeProgram {
    legs: usize,
    initial: Vec<bool>,
    state_size: usize,
    model_offset: usize,
    model_count: usize,
    momentum_slots: Vec<(usize, u64)>,
    sources: SchemaSources,
    interactions: Interactions,
    block_rows: Vec<(usize, usize, i32)>,
    /// `TreeFile::block_row_terms` and each block row's offset in a point's
    /// rows.
    block_row_terms: Vec<(usize, usize, i32)>,
    block_row_offsets: Vec<usize>,
    rows: usize,
    components: usize,
    plan_components: Vec<Option<usize>>,
    blocks: Vec<Vec<Option<usize>>>,
    couplings: usize,
    terms: Vec<Term>,
    squared: Vec<Vec<(usize, usize)>>,
    metric: Option<Vec<MetricCell>>,
    average: Rational,
    parameters: Parameters,
    counterterms: Option<Counterterms>,
    counterterm_trees: Option<CountertermTrees>,
    /// Observable columns (`set_columns`), formed per point when set.
    columns: Option<Columns>,
    /// The leg masks of the closure terms' slots.
    masks: Vec<u64>,
    /// The closure terms' kernel input layouts.
    layouts: Vec<Vec<TermInput>>,
    /// The closure terms' colour weights and their sqrt(2) powers.
    weights: Vec<([Rational; 2], i32)>,
    /// The rows the plan's columns keep and the block rows its kept links
    /// read, when they restrict them (`rows::Restriction`): applied by
    /// `set_columns` (observable columns).
    restriction: Option<rows::Restriction>,
    /// The Born rows the root closures evaluate (none: every row; set by
    /// the row restriction, `rows`).
    closure_rows: Option<Vec<bool>>,
}

/// The parameter-only inputs of one batch.
pub struct TreeBatch<T> {
    /// Per point the model parameter values in the schema's order (real
    /// carrier values; the renormalisation scale may change per point).
    pub model: Vec<Vec<T>>,
    /// Exact masses of massive external particles by PDG code.
    pub masses: BTreeMap<i32, Rational>,
    /// Carrier masses of massive external particles whose mass is a derived
    /// model parameter, by PDG code (the parameter program's outputs).
    pub derived_masses: BTreeMap<i32, T>,
    /// Couplings per (name, N_c grade) slot.
    pub couplings: Vec<Complex<T>>,
    /// Per point the counterterm and field couplings per (name, epsilon
    /// power, N_c grade) slot.
    pub counterterm_couplings: Vec<Vec<Complex<T>>>,
    /// Per point the counterterm program's model parameter values.
    pub counterterm_model: Vec<Vec<T>>,
    /// Per point the source scale sqrt(alpha(0) / alpha) of the on-shell
    /// external photons (`ParameterFile::photons`; empty without them).
    pub photon_scales: Vec<T>,
}

/// The values a batch's external parameters bind (`TreeProgram::bind`).
pub struct BoundParameters<T> {
    /// The tree program's per-batch inputs.
    pub trees: TreeBatch<T>,
    /// The plan's kernel constants.
    pub constants: Vec<Complex<T>>,
    /// The plan's squared masses per mass id (their real parts).
    pub masses: Vec<Rational>,
    /// The imaginary parts of the plan's squared masses (a complex-mass
    /// scheme's pole masses, `["complex", o]` plan masses;
    /// `V2Batch::masses_imaginary`), or empty when the program has no
    /// complex plan mass.
    pub masses_imaginary: Vec<Rational>,
    /// The squared mass of each leg's external state (`V2Batch::leg_masses`),
    /// from the same exact mass parameters as the plan's and the sources'.
    pub leg_masses: Vec<Option<Rational>>,
}

/// Native inputs of a batch, per point.
pub struct TreeInputs<T> {
    /// The plan's block rows, flattened in plan order.
    pub block_rows: Vec<Vec<Complex<T>>>,
    /// The plan's Born components per Born row (`rows × plan components`).
    pub born: Vec<Vec<Complex<T>>>,
    /// Every Born component per Born row (`rows × components`).
    pub amplitudes: Vec<Vec<Complex<T>>>,
    /// `|A0|²` per squared Born order (with a colour metric).
    pub born_squared: Option<Vec<Vec<T>>>,
    /// With loop-side sources and a colour metric: `Re <A0|M|A0'>` per
    /// squared Born order, the physical Born bra `A0` against the loop-side
    /// Born ket `A0'` (`TreeProgram::inputs_loop_side`).
    pub loop_born: Option<Vec<Vec<T>>>,
    /// The counterterm and external-field outputs (component, sector,
    /// epsilon power), divided by the average (with counterterms).
    pub counterterms: Option<Vec<Vec<T>>>,
    /// Per point and observable column its outputs (`set_columns`: the Born
    /// interferences, then the counterterm and field ones, these divided by
    /// the average); none without columns.
    pub columns: Option<Vec<Vec<Vec<Complex<T>>>>>,
    /// Wall seconds per stage: the interaction programs (`"trees:
    /// programs"`, with the states), the Born and counterterm root closures
    /// (`"trees: closures"`) and the rest (`"trees: interferences"`: block
    /// rows, interferences, Born components and |A0|²).
    pub seconds: Vec<(&'static str, f64)>,
}

/// The carrier form of a program: its kernel evaluators and exact weights.
pub struct TreeRuntime<T> {
    /// The parameter program.
    parameters: Kernel<T>,
    /// The interaction program's kernels and, per step and kernel group,
    /// its interactions' weights.
    kernels: Vec<Kernel<T>>,
    edge_weights: Vec<Vec<Vec<Complex<T>>>>,
    /// The counterterm program's kernels and interaction weights.
    counterterm_kernels: Vec<Kernel<T>>,
    counterterm_edge_weights: Vec<Vec<Vec<Complex<T>>>>,
    /// The closure terms' colour weights.
    weights: Vec<Complex<T>>,
    metric: Option<Vec<Complex<T>>>,
    average: T,
    interference: Option<InterferenceRuntime<T>>,
    /// The metric runtime of each distinct column interference
    /// (`Columns::programs`).
    columns: Vec<InterferenceRuntime<T>>,
}

/// One kernel (or the parameter program) in carrier `T`: its exact
/// evaluator in the carrier, or in binary64 its SymJIT plane application.
pub enum Kernel<T> {
    Exact(BatchExpressionEvaluator<Complex<T>>),
    Plane(Box<dyn PlaneRows<T> + Send>),
}

impl<T> TreeRuntime<T> {
    /// Whether any program runs an exact Symbolica evaluator (DoubleFloat,
    /// arbitrary precision, or binary64 without SymJIT on this CPU): its
    /// evaluations hold pyAmpliCol's library key; plane applications need
    /// none.
    fn runs_exact_states(&self) -> bool {
        std::iter::once(&self.parameters)
            .chain(&self.kernels)
            .chain(&self.counterterm_kernels)
            .any(|kernel| matches!(kernel, Kernel::Exact(_)))
    }
}

/// A plane application (binary64 only): row-major batch evaluation, and its
/// split-complex planes for callers that fill them directly (the
/// interaction program, `run_interactions`).
pub trait PlaneRows<T> {
    /// `rows` evaluations: inputs and outputs row-major.
    fn evaluate(
        &mut self,
        inputs: &[Complex<T>],
        outputs: &mut [Complex<T>],
        rows: usize,
    ) -> RusticolResult<()>;
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    /// The planes of a call of `rows` rows, `stride` values apart: input
    /// `k`'s real parts start at `2k * stride`, its imaginary parts at
    /// `(2k + 1) * stride`, the outputs' planes follow the inputs' in the
    /// same form. The caller fills the first `rows` values of every input
    /// plane, then calls [`Self::run`]; other contents are unspecified.
    fn planes(&mut self, rows: usize) -> (&mut [T], usize);
    /// Run the kernel on the `rows` rows filled after [`Self::planes`]:
    /// the planes, the outputs' first `rows` values written.
    fn run(&mut self, rows: usize) -> RusticolResult<&[T]>;
}

impl<T> Kernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// `rows` evaluations: inputs and outputs row-major.
    fn evaluate_batch(
        &mut self,
        inputs: &[Complex<T>],
        outputs: &mut [Complex<T>],
        rows: usize,
    ) -> Result<(), String> {
        match self {
            Kernel::Exact(evaluator) => evaluator
                .evaluate_batch(inputs, outputs, rows)
                .map_err(|e| e.to_string()),
            Kernel::Plane(plane) => plane
                .evaluate(inputs, outputs, rows)
                .map_err(|e| e.message().to_owned()),
        }
    }

    fn input_len(&self) -> usize {
        match self {
            Kernel::Exact(evaluator) => evaluator.get_input_len(),
            Kernel::Plane(plane) => plane.input_len(),
        }
    }

    fn output_len(&self) -> usize {
        match self {
            Kernel::Exact(evaluator) => evaluator.get_output_len(),
            Kernel::Plane(plane) => plane.output_len(),
        }
    }
}

/// A saved exact evaluator in carrier `T`, under pyAmpliCol's library key
/// (decoding and mapping create Symbolica state).
fn exact_kernel<T: Carrier + Real>(saved: &[u8], zero: &T) -> RusticolResult<Kernel<T>>
where
    Complex<T>: Real + EvaluationDomain,
{
    let _unlock = crate::licence::unlock();
    let complex = |re: &Rational, im: &Rational| Complex::new(cast(zero, re), cast(zero, im));
    Ok(Kernel::Exact(
        crate::engine::decode_exact_evaluator_state(saved)?
            .map_coeff_with_prec(&|r| complex(&r.re, &r.im), zero.get_precision())
            .into_batch_evaluator(),
    ))
}

/// A binary64 plane application on split-complex planes (inputs re, im,
/// then outputs), filled by the caller ([`PlaneRows::planes`]) or
/// transposed from row-major rows ([`PlaneRows::evaluate`]).
///
/// SymJIT's SIMD and scalar kernels of one application round differently in
/// the last bits, and a call runs its aligned blocks of `simd_lanes` rows in
/// SIMD and the rest in scalar code. Rows are therefore padded to whole SIMD
/// blocks (copies of the last row, discarded), so every row runs in SIMD and
/// a point's value does not depend on the batch it is evaluated in. (A block
/// whose lanes take different branches of a branching program is replayed
/// in scalar code by SymJIT; the programs branch on parameters only, which
/// the points of a batch share.)
///
/// The planes lie `stride` values apart, one cache line more than the padded
/// rows: with a power-of-two plane length (4096 rows) every plane's row maps
/// to the same cache set, and the kernel and the fill touch all planes at
/// one row together. The buffer only grows; nothing is zeroed per call.
struct BinaryPlane {
    kernel: SymjitPlaneKernel,
    inputs: usize,
    outputs: usize,
    planes: Vec<f64>,
    /// The plane stride of the current call ([`PlaneRows::planes`]).
    stride: usize,
}

/// `f64` values per cache line.
const PLANE_LINE: usize = 8;

impl BinaryPlane {
    fn load(bytes: &[u8], (inputs, outputs): (usize, usize)) -> RusticolResult<Self> {
        let layout = SymjitPlaneLayout::complex(inputs, outputs);
        Ok(Self {
            kernel: SymjitPlaneKernel::load_bytes(bytes, "<loop v2 tree kernel>", layout)?,
            inputs,
            outputs,
            planes: Vec::new(),
            stride: 0,
        })
    }

    /// `rows` padded to whole SIMD blocks.
    fn padded(&self, rows: usize) -> usize {
        let lanes = self.kernel.simd_lanes();
        rows.div_ceil(lanes) * lanes
    }

    fn plane_count(&self) -> usize {
        2 * (self.inputs + self.outputs)
    }
}

impl PlaneRows<f64> for BinaryPlane {
    fn evaluate(
        &mut self,
        inputs: &[Complex<f64>],
        outputs: &mut [Complex<f64>],
        rows: usize,
    ) -> RusticolResult<()> {
        require(
            inputs.len() == rows * self.inputs && outputs.len() == rows * self.outputs,
            "plane kernel rows",
        )?;
        if rows == 0 {
            return Ok(());
        }
        let (width_in, width_out) = (self.inputs, self.outputs);
        let (planes, stride) = self.planes(rows);
        // One plane at a time: contiguous stores.
        for input in 0..width_in {
            let (re, im) = planes[2 * input * stride..][..2 * stride].split_at_mut(stride);
            for (row, values) in inputs.chunks_exact(width_in).enumerate() {
                re[row] = values[input].re;
                im[row] = values[input].im;
            }
        }
        let planes = self.run(rows)?;
        let base = 2 * width_in * stride;
        for output in 0..width_out {
            let re = &planes[base + 2 * output * stride..][..rows];
            let im = &planes[base + (2 * output + 1) * stride..][..rows];
            for (row, values) in outputs.chunks_exact_mut(width_out).enumerate() {
                values[output] = Complex::new(re[row], im[row]);
            }
        }
        Ok(())
    }

    fn input_len(&self) -> usize {
        self.inputs
    }

    fn output_len(&self) -> usize {
        self.outputs
    }

    fn planes(&mut self, rows: usize) -> (&mut [f64], usize) {
        let stride = self.padded(rows).div_ceil(PLANE_LINE) * PLANE_LINE + PLANE_LINE;
        let total = self.plane_count() * stride;
        if self.planes.len() < total {
            self.planes.resize(total, 0.0);
        }
        self.stride = stride;
        (&mut self.planes[..total], stride)
    }

    fn run(&mut self, rows: usize) -> RusticolResult<&[f64]> {
        let (stride, padded) = (self.stride, self.padded(rows));
        let total = self.plane_count() * stride;
        require(
            rows > 0 && padded < stride && self.planes.len() >= total,
            "plane kernel rows",
        )?;
        let planes = &mut self.planes[..total];
        // Whole SIMD blocks: padding rows repeat the last row.
        for plane in planes[..2 * self.inputs * stride].chunks_exact_mut(stride) {
            let last = plane[rows - 1];
            plane[rows..padded].fill(last);
        }
        let descriptors = planes
            .chunks_exact_mut(stride)
            .map(|plane| PlaneDescriptor::from_slice(&mut plane[..padded]))
            .collect();
        let mut table = self.kernel.build_raw_table(descriptors)?;
        self.kernel.execute_table(&mut table, 0, padded)?;
        drop(table);
        Ok(&self.planes[..total])
    }
}

/// Per slot set of a counterterm program (`InteractionFile::masks`: slot
/// indices, increasing, into the slot table whose epsilon powers are
/// `powers`) whether it holds no slot of epsilon power 0: a current reaching
/// only such a set is pole-only work, skipped by finite-only evaluations.
fn pole_slot_sets(masks: &[Vec<usize>], powers: &[i32]) -> RusticolResult<Vec<bool>> {
    require(
        !masks.is_empty()
            && masks.iter().all(|mask| {
                !mask.is_empty()
                    && mask.windows(2).all(|pair| pair[0] < pair[1])
                    && mask.iter().all(|&slot| slot < powers.len())
            }),
        "counterterm slot masks",
    )?;
    Ok(masks
        .iter()
        .map(|mask| mask.iter().all(|&slot| powers[slot] != 0))
        .collect())
}

/// Validate and group an interaction program (`InteractionFile`); `saved`
/// holds its kernels' evaluators, vertex kernels first.
/// `slots` is a counterterm program's slot table (none for the trees): the
/// epsilon power of each slot. A step's interaction or output is a pole one
/// (finite-only evaluations skip it) when its accumulator's slot set holds
/// no slot of epsilon power 0.
fn load_interactions(
    program: InteractionFile,
    saved: &[Vec<u8>],
    planes: Option<&[Vec<u8>]>,
    state_size: usize,
    model_count: usize,
    slots: Option<&[i32]>,
) -> RusticolResult<Interactions> {
    // Per slot set whether it reaches no finite slot (the finite-only skip).
    let pole_masks: Vec<bool> = match slots {
        Some(powers) => pole_slot_sets(&program.masks, powers)?,
        None => {
            require(
                program.masks.is_empty(),
                "slot masks outside a counterterm program",
            )?;
            Vec::new()
        }
    };
    let kernels = program.kernels.len();
    require(
        saved.len() == kernels + program.propagators.len()
            && planes.is_none_or(|planes| planes.len() == saved.len()),
        "interaction kernel evaluator count",
    )?;
    // Without plane applications every carrier decodes the exact states:
    // validate them now. With them, a carrier that needs the exact states
    // decodes them when its runtime is built (binary64 never does).
    if planes.is_none() {
        for bytes in saved {
            drop(crate::engine::decode_exact_evaluator_state(bytes)?);
        }
    }
    let mut models = Vec::with_capacity(saved.len());
    let mut shapes = Vec::with_capacity(saved.len());
    for (inputs, outputs, model) in program.kernels.iter().chain(&program.propagators) {
        require(
            model.windows(2).all(|pair| pair[0] < pair[1])
                && model.last().is_none_or(|&last| last < model_count),
            "kernel model parameter indices",
        )?;
        models.push(model.clone());
        shapes.push((*inputs, *outputs));
    }
    let within = |start: usize, length: usize| start + length <= state_size;
    let mut steps = Vec::with_capacity(program.steps.len());
    for step in program.steps {
        let mut offsets = Vec::with_capacity(step.accumulators.len());
        let mut width = 0;
        for &dimension in &step.accumulators {
            offsets.push(width);
            width += dimension;
        }
        // Per accumulator whether only pole slots read its current.
        let pole: Vec<bool> = if slots.is_some() {
            require(
                step.accumulator_masks.len() == step.accumulators.len()
                    && step
                        .accumulator_masks
                        .iter()
                        .all(|&mask| mask < pole_masks.len()),
                "accumulator slot masks",
            )?;
            step.accumulator_masks
                .iter()
                .map(|&mask| pole_masks[mask])
                .collect()
        } else {
            require(
                step.accumulator_masks.is_empty(),
                "slot masks outside a counterterm program",
            )?;
            vec![false; step.accumulators.len()]
        };
        let mut groups: BTreeMap<usize, Vec<Edge>> = BTreeMap::new();
        for (kernel, left, ld, right, rd, pl, pr, accumulator, re, im) in step.interactions {
            let (inputs, outputs, model) = program
                .kernels
                .get(kernel)
                .ok_or_else(|| invalid("interaction kernel index"))?;
            require(
                accumulator < step.accumulators.len()
                    && *outputs == step.accumulators[accumulator]
                    && *inputs == ld + rd + 8 + model.len()
                    && within(left, ld)
                    && within(right, rd)
                    && within(pl, 4)
                    && within(pr, 4),
                "interaction",
            )?;
            groups.entry(kernel).or_default().push(Edge {
                left,
                left_dimension: ld,
                right,
                right_dimension: rd,
                left_momentum: pl,
                right_momentum: pr,
                accumulator: offsets[accumulator],
                weight: [exact(&re)?, exact(&im)?],
                pole: pole[accumulator],
            });
        }
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for (accumulator, propagator, momentum, start, dimension) in step.outputs {
            require(
                accumulator < step.accumulators.len() && within(start, dimension),
                "interaction output",
            )?;
            let propagator = if propagator < 0 {
                require(
                    step.accumulators[accumulator] == dimension,
                    "an unpropagated output's dimension",
                )?;
                None
            } else {
                let index = usize::try_from(propagator).map_err(|_| invalid("propagator"))?;
                let (inputs, outputs, model) = program
                    .propagators
                    .get(index)
                    .ok_or_else(|| invalid("propagator index"))?;
                require(
                    *inputs == step.accumulators[accumulator] + 4 + model.len()
                        && *outputs == dimension
                        && within(momentum, 4),
                    "propagator output",
                )?;
                Some(kernels + index)
            };
            outputs.push((
                offsets[accumulator],
                propagator,
                momentum,
                start,
                dimension,
                pole[accumulator],
            ));
        }
        steps.push(Step {
            groups: groups.into_iter().collect(),
            width,
            outputs,
        });
    }
    Ok(Interactions {
        saved: saved.to_vec(),
        models,
        planes: planes.map(<[Vec<u8>]>::to_vec),
        shapes,
        steps,
    })
}

/// Validate a parameter program (`ParameterFile`) against the tree program
/// and the plan; `saved` is its exact evaluator.
#[allow(clippy::too_many_arguments)]
fn load_parameters(
    file: ParameterFile,
    saved: &[u8],
    plane: Option<&[u8]>,
    model_count: usize,
    counterterm_model_count: usize,
    couplings: usize,
    counterterm_couplings: usize,
    plan: &LoopProgramV4,
) -> RusticolResult<Parameters> {
    let inputs = file.names.len();
    require(
        file.defaults.len() == inputs,
        "a default per external parameter",
    )?;
    require(file.model.len() == model_count, "model parameter sources")?;
    require(
        file.counterterm_model.len() == counterterm_model_count,
        "counterterm program model parameter sources",
    )?;
    require(
        file.couplings.len() == couplings
            && file.counterterm_couplings.len() == counterterm_couplings,
        "parameter program couplings",
    )?;
    require(
        file.constants.len() == plan.constants,
        "parameter program kernel constants",
    )?;
    require(
        file.plan_masses.len() == plan.masses,
        "parameter program plan masses",
    )?;
    let output = |index: usize| index < file.outputs;
    let plan_masses = file
        .plan_masses
        .iter()
        .map(|mass| match mass {
            None => Ok(None),
            Some(MassFile::Input(i)) if *i < inputs && Some(*i) != file.scale => {
                Ok(Some(MassSource::Input(*i)))
            }
            Some(MassFile::Output(kind, o))
                if kind == "output" && output(*o) && file.positive.contains(o) =>
            {
                Ok(Some(MassSource::Output(*o)))
            }
            Some(MassFile::Output(kind, o)) if kind == "complex" && output(*o) => {
                Ok(Some(MassSource::Complex(*o)))
            }
            Some(_) => Err(invalid("parameter program plan mass source")),
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    require(
        file.couplings
            .iter()
            .chain(&file.counterterm_couplings)
            .chain(&file.finite)
            .chain(&file.constants)
            .chain(&file.positive)
            .chain(file.derived_masses.iter().map(|(_, o)| o))
            .all(|&o| output(o))
            && file
                .derived_masses
                .iter()
                .all(|(_, o)| file.positive.contains(o))
            && file
                .derived_masses
                .iter()
                .all(|(pdg, _)| !file.masses.iter().any(|(exact, _)| exact == pdg))
            && file.masses.iter().all(|&(_, i)| i < inputs)
            && file.scale.is_none_or(|i| i < inputs)
            && file.masses.iter().all(|&(_, i)| Some(i) != file.scale)
            && file.scale_outputs.as_ref().is_none_or(|dependent| {
                dependent.iter().all(|(o, _)| output(*o))
                    && (file.scale.is_some() || dependent.is_empty())
            })
            && file.photons.as_ref().is_none_or(|(o, legs)| {
                output(*o) && file.positive.contains(o) && !legs.is_empty()
            }),
        "parameter program index",
    )?;
    let sources = |records: &[(String, usize, bool)]| {
        records
            .iter()
            .map(|(kind, index, imaginary)| match kind.as_str() {
                "input" if *index < inputs && !imaginary => Ok(ModelSource::Input(*index)),
                "output" if output(*index) => Ok(ModelSource::Output {
                    index: *index,
                    imaginary: *imaginary,
                }),
                _ => Err(invalid(format!("model parameter source {kind:?} {index}"))),
            })
            .collect::<RusticolResult<Vec<_>>>()
    };
    let model = sources(&file.model)?;
    let counterterm_model = sources(&file.counterterm_model)?;
    if plane.is_none() {
        // Validated when its runtime is built otherwise (`load_interactions`).
        drop(crate::engine::decode_exact_evaluator_state(saved)?);
    }
    Ok(Parameters {
        names: file.names,
        defaults: file
            .defaults
            .iter()
            .map(|value| exact(value))
            .collect::<RusticolResult<_>>()?,
        saved: saved.to_vec(),
        plane: plane.map(<[u8]>::to_vec),
        outputs: file.outputs,
        model,
        counterterm_model,
        couplings: file.couplings,
        counterterm_couplings: file.counterterm_couplings,
        finite: file.finite,
        constants: file.constants,
        positive: file.positive,
        masses: file.masses,
        derived_masses: file.derived_masses,
        plan_masses,
        scale: file.scale,
        scale_outputs: file.scale_outputs,
        photons: file.photons,
    })
}

/// Validate one closure term (`TermFile`) over `blocks` closure blocks.
fn load_term(
    term: TermFile,
    plan: &LoopProgramV4,
    blocks: usize,
    couplings: usize,
    components: usize,
    legs: usize,
    closures: &mut Closures<'_>,
) -> RusticolResult<Term> {
    let kernel = plan
        .kernels
        .get(term.kernel)
        .ok_or_else(|| invalid("closure term kernel index"))?;
    let (inputs, slots) = closures
        .layouts
        .get(term.layout)
        .ok_or_else(|| invalid("closure term layout"))?;
    require(
        kernel.outputs.len() == 1 && kernel.inputs == inputs.len(),
        "a closure kernel has one output and the term's inputs",
    )?;
    require(
        term.blocks.len() == term.masks.len()
            && *slots <= term.blocks.len()
            && term.blocks.iter().all(|&block| block < blocks)
            && term.masks.iter().all(|&mask| mask >> legs == 0)
            && term.sign.abs() == 1,
        "closure term blocks",
    )?;
    require(
        term.contributions
            .iter()
            .all(|&(coupling, component, weight)| {
                coupling < couplings && component < components && weight < closures.weights
            }),
        "closure term contribution",
    )?;
    let masks = term
        .masks
        .iter()
        .map(|mask| {
            let next = closures.masks.len();
            *closures.masks.entry(*mask).or_insert(next)
        })
        .collect();
    Ok(Term {
        kernel: term.kernel,
        layout: term.layout,
        blocks: term.blocks,
        masks,
        sign: term.sign,
        contributions: term.contributions,
    })
}

/// Validate the closure terms' input layouts: per layout its inputs and
/// the number of slots it reads.
fn load_layouts(
    layouts: &[Vec<(String, usize, usize)>],
) -> RusticolResult<Vec<(Vec<TermInput>, usize)>> {
    layouts
        .iter()
        .map(|layout| {
            let inputs = layout
                .iter()
                .map(|(kind, slot, component)| match kind.as_str() {
                    "current" => Ok(TermInput::Current {
                        slot: *slot,
                        component: *component,
                    }),
                    "momentum" if *component < 4 => Ok(TermInput::Momentum {
                        slot: *slot,
                        mu: *component,
                    }),
                    _ => Err(invalid(format!("closure term input {kind:?}"))),
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let slots = layout
                .iter()
                .map(|&(_, slot, _)| slot + 1)
                .max()
                .unwrap_or(0);
            Ok((inputs, slots))
        })
        .collect()
}

/// Validate a counterterm program (`CountertermFile`).
#[allow(clippy::too_many_arguments)]
fn load_counterterms(
    file: CountertermFile,
    plan: &LoopProgramV4,
    blocks: usize,
    born_components: usize,
    basis: usize,
    orders: usize,
    legs: usize,
    closures: &mut Closures<'_>,
) -> RusticolResult<Counterterms> {
    require(
        file.coupling_powers.len() == file.couplings,
        "counterterm coupling powers",
    )?;
    require(
        file.born.len() == born_components,
        "a (basis element, Born order, N_c power) per Born component",
    )?;
    let components = file.ket.len();
    let terms = file
        .terms
        .into_iter()
        .map(|term| {
            load_term(
                term,
                plan,
                blocks,
                file.couplings,
                components,
                legs,
                closures,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let cells = file
        .cells
        .iter()
        .map(|(b, t, power, re, im)| Ok((*b, *t, *power, [exact(re)?, exact(im)?])))
        .collect::<RusticolResult<Vec<_>>>()?;
    let entries = file
        .entries
        .iter()
        .map(|(kind, bra, ket, coupling, grade, output, degree)| {
            let ket = match kind.as_str() {
                "vector" => Ket::Vector(*ket),
                "born" => Ket::Born(*ket),
                _ => return Err(invalid(format!("interference entry {kind:?}"))),
            };
            Ok(Entry {
                bra: *bra,
                ket,
                coupling: *coupling,
                grade: *grade,
                output: *output,
                degree: *degree,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    require(
        file.depth.is_none_or(|depth| depth == 0 || depth == 2),
        "colour projection depth",
    )?;
    let interference = Interference::new(
        basis,
        orders,
        file.tensors,
        MetricCells::List(cells),
        &file.born,
        &file.ket,
        file.vectors,
        entries,
        file.outputs,
        file.couplings,
        file.depth,
    )?;
    Ok(Counterterms {
        couplings: file.couplings,
        finite_slot: file
            .coupling_powers
            .iter()
            .map(|&power| power == 0)
            .collect(),
        components,
        terms,
        interference,
        layout: InterferenceLayout {
            born: file.born,
            ket: file.ket,
            vectors: file.vectors,
            tensors: file.tensors,
            basis,
            orders,
            depth: file.depth,
        },
    })
}

/// Interference entries as stored: (kind, bra Born order, vector or Born
/// order, coupling slot, N_c grade, output, degree).
#[allow(clippy::type_complexity)]
fn load_entries(
    entries: &[(String, usize, usize, Option<usize>, i32, usize, Option<i32>)],
) -> RusticolResult<Vec<Entry>> {
    entries
        .iter()
        .map(|(kind, bra, ket, coupling, grade, output, degree)| {
            let ket = match kind.as_str() {
                "vector" => Ket::Vector(*ket),
                "born" => Ket::Born(*ket),
                _ => return Err(invalid(format!("interference entry {kind:?}"))),
            };
            Ok(Entry {
                bra: *bra,
                ket,
                coupling: *coupling,
                grade: *grade,
                output: *output,
                degree: *degree,
            })
        })
        .collect()
}

fn load_cells(cells: CellsFile) -> RusticolResult<MetricCells> {
    let values = cells
        .values
        .iter()
        .map(|cell| {
            cell.iter()
                .map(|(power, re, im)| Ok((*power, [exact(re)?, exact(im)?])))
                .collect()
        })
        .collect::<RusticolResult<_>>()?;
    Ok(MetricCells::Table {
        values,
        ids: cells.ids,
    })
}

impl TreeProgram {
    /// Load a program exported by `loops/v2/native_trees.py` for `plan`.
    /// `kernels` holds the interaction program's saved exact evaluators and,
    /// last, the parameter program's; `planes`, when given, their binary64
    /// SymJIT plane applications in the same order.
    pub fn load(
        json: &[u8],
        kernels: &[Vec<u8>],
        planes: Option<&[Vec<u8>]>,
        plan: &LoopProgramV4,
    ) -> RusticolResult<Self> {
        Self::load_with_components(json, kernels, planes, plan, None)
    }

    /// `load` for another plan of the same process (an observable column
    /// set's plan, `loops/v2/native_observables`): `plan_components` (per
    /// plan Born component its tree component, or none) replaces the
    /// program's own.
    pub fn load_with_components(
        json: &[u8],
        kernels: &[Vec<u8>],
        planes: Option<&[Vec<u8>]>,
        plan: &LoopProgramV4,
        plan_components: Option<Vec<Option<usize>>>,
    ) -> RusticolResult<Self> {
        let mut file: TreeFile =
            serde_json::from_slice(json).map_err(|e| invalid(e.to_string()))?;
        if let Some(components) = plan_components {
            file.born.plan_components = components;
        }
        require(
            file.schema == TREE_SCHEMA,
            "unsupported tree-program schema",
        )?;
        require(
            file.legs == plan.legs && file.legs > 0 && file.legs < 64,
            "leg count",
        )?;
        require(file.initial.len() == file.legs, "initial flags")?;
        require(
            file.model_offset + file.model_count <= file.state_size,
            "model parameters outside the state",
        )?;
        for &(start, _mask) in &file.momentum_slots {
            require(
                start + 4 <= file.state_size,
                "momentum slot outside the state",
            )?;
        }
        let (parameter_evaluator, kernels) = kernels
            .split_last()
            .ok_or_else(|| invalid("the parameter program's evaluator is missing"))?;
        require(
            planes.is_none_or(|planes| planes.len() == kernels.len() + 1),
            "a plane application per saved evaluator",
        )?;
        let (parameter_plane, planes) = match planes.and_then(<[Vec<u8>]>::split_last) {
            Some((parameter, planes)) => (Some(parameter.as_slice()), Some(planes)),
            None => (None, None),
        };
        let own = file.interactions.kernels.len() + file.interactions.propagators.len();
        require(kernels.len() >= own, "interaction kernel evaluator count")?;
        let (kernels, counterterm_kernels) = kernels.split_at(own);
        let (planes, counterterm_planes) = match planes {
            Some(planes) => {
                let (own, counterterm) = planes.split_at(own);
                (Some(own), Some(counterterm))
            }
            None => (None, None),
        };
        let interactions = load_interactions(
            file.interactions,
            kernels,
            planes,
            file.state_size,
            file.model_count,
            None,
        )?;
        let counterterm_trees = match file.counterterm_trees {
            Some(ct) => {
                require(
                    ct.offset == file.state_size
                        && ct.model_offset + ct.model_count <= ct.state_size,
                    "counterterm program state",
                )?;
                for &(start, _mask) in &ct.momentum_slots {
                    require(
                        start + 4 <= ct.state_size,
                        "counterterm program momentum slot outside its state",
                    )?;
                }
                require(
                    !ct.slots.is_empty()
                        && ct
                            .slots
                            .iter()
                            .map(|slot| &slot.0)
                            .collect::<std::collections::BTreeSet<_>>()
                            .len()
                            == ct.slots.len(),
                    "counterterm slot table",
                )?;
                let powers: Vec<i32> = ct.slots.iter().map(|slot| slot.2).collect();
                Some(CountertermTrees {
                    layout: StateLayout {
                        size: ct.state_size,
                        model_offset: ct.model_offset,
                        model_block: ct.model_count,
                    },
                    momentum_slots: ct.momentum_slots,
                    sources: SchemaSources::from_value(&ct.source_fill)?,
                    interactions: load_interactions(
                        ct.interactions,
                        counterterm_kernels,
                        counterterm_planes,
                        ct.state_size,
                        ct.model_count,
                        Some(&powers),
                    )?,
                })
            }
            None => {
                require(
                    counterterm_kernels.is_empty(),
                    "interaction kernel evaluator count",
                )?;
                None
            }
        };
        // Closure blocks index the trees' state, then the counterterm
        // program's.
        let closure_size =
            file.state_size + counterterm_trees.as_ref().map_or(0, |ct| ct.layout.size);
        let parameters = load_parameters(
            file.parameters,
            parameter_evaluator,
            parameter_plane,
            file.model_count,
            counterterm_trees
                .as_ref()
                .map_or(0, |ct| ct.layout.model_block),
            file.born.couplings,
            file.counterterms.as_ref().map_or(0, |ct| ct.couplings),
            plan,
        )?;
        let sources = SchemaSources::from_value(&file.source_fill)?;
        if let Some((_, legs)) = &parameters.photons {
            // Each photon a leg with sources in the trees and, with one, the
            // counterterm program.
            require(
                legs.iter().all(|&leg| {
                    (1..=file.legs).contains(&leg)
                        && !sources.leg_slots(leg).is_empty()
                        && counterterm_trees
                            .as_ref()
                            .is_none_or(|ct| !ct.sources.leg_slots(leg).is_empty())
                }),
                "a photon leg without sources",
            )?;
        }
        require(
            file.block_rows.len() == plan.block_rows.len(),
            "block rows differ from the plan's",
        )?;
        for (&(start, dimension, sign), &expected) in file.block_rows.iter().zip(&plan.block_rows) {
            require(
                start + dimension <= file.state_size && dimension == expected && sign.abs() == 1,
                "block row",
            )?;
        }
        for &(row, start, sign) in &file.block_row_terms {
            require(
                file.block_rows
                    .get(row)
                    .is_some_and(|&(_, dimension, _)| start + dimension <= file.state_size)
                    && sign.abs() == 1,
                "block row term",
            )?;
        }
        let block_row_offsets = file
            .block_rows
            .iter()
            .scan(0, |offset, &(_, dimension, _)| {
                let start = *offset;
                *offset += dimension;
                Some(start)
            })
            .collect();
        let born = file.born;
        require(born.rows == plan.rows, "Born rows differ from the plan's")?;
        require(
            born.plan_components.len() == plan.born_components
                && born
                    .plan_components
                    .iter()
                    .flatten()
                    .all(|&component| component < born.components),
            "plan Born components",
        )?;
        for block in &born.blocks {
            require(block.len() == born.rows, "closure block rows")?;
            require(
                block.iter().flatten().all(|&start| start < closure_size),
                "closure block row outside the state",
            )?;
        }
        let mut masks = BTreeMap::new();
        let layouts = load_layouts(&file.layouts)?;
        let weights = file
            .weights
            .iter()
            .map(|(re, im, power)| Ok(([exact(re)?, exact(im)?], *power)))
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut closures = Closures {
            layouts: &layouts,
            weights: weights.len(),
            masks: &mut masks,
        };
        let blocks = born.blocks.len();
        let terms = born
            .terms
            .into_iter()
            .map(|term| {
                load_term(
                    term,
                    plan,
                    blocks,
                    born.couplings,
                    born.components,
                    file.legs,
                    &mut closures,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        require(born.orders > 0, "Born orders")?;
        let amplitudes = born
            .amplitudes
            .iter()
            .map(|row| {
                row.iter()
                    .map(|(component, factor)| {
                        require(*component < born.components, "amplitude component")?;
                        Ok((*component, exact(factor)?))
                    })
                    .collect::<RusticolResult<Vec<_>>>()
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        require(amplitudes.len() % born.orders == 0, "amplitude rows")?;
        for pairs in &born.squared {
            require(
                pairs
                    .iter()
                    .all(|&(left, right)| left < born.orders && right < born.orders),
                "squared Born orders",
            )?;
        }
        let basis = amplitudes.len() / born.orders;
        let metric = born
            .metric
            .map(|entries| {
                entries
                    .iter()
                    .map(|(squared, left, right, re, im)| {
                        require(
                            *squared < born.squared.len()
                                && *left < born.components
                                && *right < born.components,
                            "component colour metric entry",
                        )?;
                        Ok((*squared, *left, *right, [exact(re)?, exact(im)?]))
                    })
                    .collect::<RusticolResult<Vec<_>>>()
            })
            .transpose()?;
        let counterterms = file
            .counterterms
            .map(|ct| {
                load_counterterms(
                    ct,
                    plan,
                    blocks,
                    born.components,
                    basis,
                    born.orders,
                    file.legs,
                    &mut closures,
                )
            })
            .transpose()?;
        Ok(Self {
            legs: file.legs,
            initial: file.initial,
            state_size: file.state_size,
            model_offset: file.model_offset,
            model_count: file.model_count,
            momentum_slots: file.momentum_slots,
            sources,
            interactions,
            block_rows: file.block_rows,
            block_row_terms: file.block_row_terms,
            block_row_offsets,
            rows: born.rows,
            components: born.components,
            plan_components: born.plan_components,
            blocks: born.blocks,
            couplings: born.couplings,
            terms,
            squared: born.squared,
            metric,
            average: exact(&born.average)?,
            parameters,
            counterterms,
            counterterm_trees,
            columns: None,
            restriction: rows::Restriction::of(plan),
            closure_rows: None,
            layouts: layouts.into_iter().map(|(inputs, _)| inputs).collect(),
            weights,
            masks: {
                let mut table = vec![0; masks.len()];
                for (mask, index) in masks {
                    table[index] = mask;
                }
                table
            },
        })
    }

    /// The external parameters a batch supplies per point, in order.
    pub fn parameter_names(&self) -> &[String] {
        &self.parameters.names
    }

    /// Exact default value per external parameter.
    pub fn parameter_defaults(&self) -> &[Rational] {
        &self.parameters.defaults
    }

    /// The renormalisation scale's external parameter, if the program reads
    /// it (each point's sqrt(mu_R^2) binds it).
    pub fn scale_parameter(&self) -> Option<&str> {
        self.parameters
            .scale
            .map(|index| self.parameters.names[index].as_str())
    }

    /// The names of the batch-shared outputs (Born couplings, kernel
    /// constants, derived external and loop masses) that depend on the
    /// renormalisation scale, by generation's record (`None`: an artifact
    /// written before the record). Empty for the usual NLO models, whose
    /// scale enters the counterterm couplings only; a running-coupling or
    /// MSbar-mass model lists its Born couplings or masses here, and a batch
    /// of such a model takes one mu_R^2 (`V2Runtime` partitions by it).
    pub fn scale_dependent_shared(&self) -> Option<Vec<&str>> {
        self.parameters.scale_dependent_shared()
    }

    /// The names of the outputs `indices` in the generation record (by
    /// output index without one).
    fn output_names(&self, indices: &[usize]) -> Vec<String> {
        let named: BTreeMap<usize, &str> = self
            .parameters
            .scale_outputs
            .iter()
            .flatten()
            .map(|(o, name)| (*o, name.as_str()))
            .collect();
        indices
            .iter()
            .map(|o| {
                named
                    .get(o)
                    .map_or_else(|| format!("output {o}"), |n| (*n).to_owned())
            })
            .collect()
    }

    /// Bind a batch's external parameters (`values`, exact, in
    /// `parameter_names` order; the scale parameter's entry is ignored) at
    /// each point's `scales` (mu_R^2): the scale parameter is sqrt(mu_R^2)
    /// in the carrier. Returns the tree program's model parameters per
    /// point, and the Born couplings, kernel constants and masses, which
    /// must not depend on the scale (they are checked equal at every point).
    pub fn bind<T: TreeCarrier + Real>(
        &self,
        runtime: &mut TreeRuntime<T>,
        values: &[Rational],
        scales: &[Rational],
        zero: &T,
    ) -> RusticolResult<BoundParameters<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        // Exact states under the key; plane applications touch no Symbolica
        // state.
        let _unlock = runtime
            .runs_exact_states()
            .then(crate::licence::unlock)
            .flatten();
        let program = &self.parameters;
        let points = scales.len();
        let width = program.names.len();
        require(points > 0, "a batch needs points")?;
        require(values.len() == width, "external parameter count")?;
        // A model whose shared values run with the scale takes one mu_R^2
        // per batch (`V2Runtime::evaluate` partitions by it).
        if let Some(dependent) = self.scale_dependent_shared()
            && !dependent.is_empty()
            && scales.iter().any(|scale| scale != &scales[0])
        {
            return Err(RusticolError::invalid_argument(format!(
                "loop v2 trees: {} depend on the renormalisation scale {}: \
                 a batch takes one mu_R^2 (evaluate one batch per scale)",
                dependent.join(", "),
                self.scale_parameter().unwrap_or("parameter"),
            )));
        }
        let czero = Complex::new(zero.clone(), zero.clone());
        let shared: Vec<T> = values.iter().map(|value| cast(zero, value)).collect();
        let mut rows_in = Vec::with_capacity(points);
        for scale in scales {
            let mut point = shared.clone();
            if let Some(index) = program.scale {
                require(
                    !scale.is_zero() && !scale.is_negative(),
                    "mu_R^2 must be positive",
                )?;
                point[index] = cast(zero, scale).sqrt();
            }
            rows_in.push(point);
        }
        let inputs: Vec<Complex<T>> = rows_in
            .iter()
            .flatten()
            .map(|value| Complex::new(value.clone(), zero.clone()))
            .collect();
        let mut outputs = vec![czero; points * program.outputs];
        runtime
            .parameters
            .evaluate_batch(&inputs, &mut outputs, points)
            .map_err(|e| invalid(format!("parameter program: {e}")))?;
        let rows: Vec<&[Complex<T>]> = outputs.chunks_exact(program.outputs).collect();
        // Values a batch shares (Born couplings, kernel constants, masses):
        // equal at every point unless they depend on the scale, which a
        // batch of mixed scales of such a model was refused for above (or,
        // in an artifact written before the scale record, here, naming the
        // outputs that differ).
        let shared = |indices: &[usize], what: &str| -> RusticolResult<Vec<Complex<T>>> {
            let first: Vec<Complex<T>> = indices.iter().map(|&o| rows[0][o].clone()).collect();
            let differ: Vec<usize> = indices
                .iter()
                .zip(&first)
                .filter(|&(&o, value)| rows[1..].iter().any(|row| row[o] != *value))
                .map(|(&o, _)| o)
                .collect();
            if !differ.is_empty() {
                return Err(RusticolError::invalid_argument(format!(
                    "loop v2 trees: the points of one batch need equal {what}; {} \
                     differ between the batch's scales (they depend on the \
                     renormalisation scale: evaluate one batch per scale)",
                    self.output_names(&differ).join(", "),
                )));
            }
            Ok(first)
        };
        for row in &rows {
            if let Some((o, _)) = &program.photons {
                // Its square root scales the photons' sources (`scale_photons`).
                let value = &row[*o];
                require(
                    value.im == *zero && value.re > *zero && value.re.is_finite(),
                    "the on-shell external photons' alpha(0)/alpha is not real, \
                     finite and positive: set the model's 1/alpha(0) parameter \
                     (aEW0M1 of MG5_aMC's G_mu loop models, e.g. 137.035999)",
                )?;
            }
            for &o in &program.positive {
                let value = &row[o];
                require(
                    value.im == *zero && value.re > *zero,
                    "a base a fractional power was distributed over is not positive",
                )?;
            }
            for &o in &program.finite {
                let value = &row[o];
                require(
                    value.re.is_finite() && value.im.is_finite(),
                    "a counterterm coupling's diagnostic is not finite \
                     (a model helper function was evaluated outside its domain)",
                )?;
            }
        }
        let counterterm_couplings = rows
            .iter()
            .map(|row| {
                program
                    .counterterm_couplings
                    .iter()
                    .map(|&o| row[o].clone())
                    .collect()
            })
            .collect();
        let couplings = shared(&program.couplings, "Born couplings")?;
        let constants = shared(&program.constants, "kernel constants")?;
        let values_of = |sources: &[ModelSource]| -> Vec<Vec<T>> {
            rows_in
                .iter()
                .zip(&rows)
                .map(|(point, row)| {
                    sources
                        .iter()
                        .map(|source| match *source {
                            ModelSource::Input(index) => point[index].clone(),
                            ModelSource::Output { index, imaginary } => {
                                if imaginary {
                                    row[index].im.clone()
                                } else {
                                    row[index].re.clone()
                                }
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let model = values_of(&program.model);
        let counterterm_model = values_of(&program.counterterm_model);
        // Real and positive (`positive`, checked above).
        let photon_scales = program.photons.as_ref().map_or_else(Vec::new, |(o, _)| {
            rows.iter().map(|row| row[*o].re.clone().sqrt()).collect()
        });
        let masses: BTreeMap<i32, Rational> = program
            .masses
            .iter()
            .map(|&(pdg, index)| (pdg, values[index].clone()))
            .collect();
        let derived: Vec<usize> = program.derived_masses.iter().map(|&(_, o)| o).collect();
        let derived_masses: BTreeMap<i32, T> = program
            .derived_masses
            .iter()
            .zip(shared(&derived, "external masses")?)
            .map(|(&(pdg, _), value)| (pdg, value.re))
            .collect();
        // A derived mass is bound like an exact one: its carrier value's
        // exact rational, so its legs' invariants and its propagators take
        // the square of the value its sources take (DESIGN §11.1).
        let mut exact_masses = masses.clone();
        for (&pdg, value) in &derived_masses {
            exact_masses.insert(pdg, value.exact()?);
        }
        let outputs: Vec<usize> = program
            .plan_masses
            .iter()
            .filter_map(|mass| match mass {
                Some(MassSource::Output(o) | MassSource::Complex(o)) => Some(*o),
                _ => None,
            })
            .collect();
        let mut derived_plan = shared(&outputs, "loop masses")?.into_iter();
        let complex_masses = program
            .plan_masses
            .iter()
            .any(|mass| matches!(mass, Some(MassSource::Complex(_))));
        let mut plan_masses = Vec::with_capacity(program.plan_masses.len());
        let mut masses_imaginary = Vec::new();
        for (index, mass) in program.plan_masses.iter().enumerate() {
            let (re, im) = match mass {
                Some(MassSource::Input(index)) => {
                    (&values[*index] * &values[*index], Rational::from(0))
                }
                Some(MassSource::Output(_)) => {
                    let value = derived_plan
                        .next()
                        .expect("one value per output plan mass")
                        .re
                        .exact()?;
                    (&value * &value, Rational::from(0))
                }
                // A complex pole mass M: the square of its carrier value's
                // exact complex rational (as a real derived mass's, DESIGN
                // §11.1), the M² the tree propagators' carrier value squares.
                Some(MassSource::Complex(_)) => {
                    let value = derived_plan.next().expect("one value per output plan mass");
                    let (re, im) = (value.re.exact()?, value.im.exact()?);
                    let squared = (&re * &re - &im * &im, Rational::from(2) * &re * &im);
                    if T::COMPLEX_MASS_MASTER_GUARD {
                        require_complex_mass_master(index, &squared.0, &squared.1)?;
                    }
                    squared
                }
                None => (Rational::from(0), Rational::from(0)),
            };
            plan_masses.push(re);
            if complex_masses {
                masses_imaginary.push(im);
            }
        }
        Ok(BoundParameters {
            leg_masses: self
                .sources
                .leg_masses(self.legs, &exact_masses)?
                .into_iter()
                .map(Some)
                .collect(),
            trees: TreeBatch {
                model,
                masses,
                derived_masses,
                couplings,
                counterterm_couplings,
                counterterm_model,
                photon_scales,
            },
            constants,
            masses: plan_masses,
            masses_imaginary,
        })
    }

    /// Number of coupling slots a batch supplies.
    pub fn couplings(&self) -> usize {
        self.couplings
    }

    /// Number of model parameters a batch supplies.
    pub fn model_count(&self) -> usize {
        self.model_count
    }

    /// The program in carrier `T` (kernel evaluators and exact weights
    /// cast): binary64 loads the plane applications when the program has
    /// them, other carriers decode the exact states (`TreeCarrier::kernel`).
    pub fn runtime<T: TreeCarrier + Real>(&self, zero: &T) -> RusticolResult<TreeRuntime<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let complex = |re: &Rational, im: &Rational| Complex::new(cast(zero, re), cast(zero, im));
        let two = cast(zero, &Rational::from(2));
        let root = two.sqrt();
        let program = &self.parameters;
        let parameters = T::kernel(
            &program.saved,
            program.plane.as_deref(),
            (program.names.len(), program.outputs),
            zero,
        )?;
        let kernels_of = |program: &Interactions| {
            program
                .saved
                .iter()
                .zip(&program.shapes)
                .enumerate()
                .map(|(k, (saved, &shape))| {
                    let plane = program.planes.as_ref().map(|planes| planes[k].as_slice());
                    T::kernel(saved, plane, shape, zero)
                })
                .collect::<RusticolResult<Vec<_>>>()
        };
        let weights_of = |program: &Interactions| -> Vec<Vec<Vec<Complex<T>>>> {
            program
                .steps
                .iter()
                .map(|step| {
                    step.groups
                        .iter()
                        .map(|(_kernel, edges)| {
                            edges
                                .iter()
                                .map(|edge| complex(&edge.weight[0], &edge.weight[1]))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };
        let kernels = kernels_of(&self.interactions)?;
        let edge_weights = weights_of(&self.interactions);
        let (counterterm_kernels, counterterm_edge_weights) = match &self.counterterm_trees {
            Some(ct) => (kernels_of(&ct.interactions)?, weights_of(&ct.interactions)),
            None => (Vec::new(), Vec::new()),
        };
        let weights = self
            .weights
            .iter()
            .map(|([re, im], power)| {
                let mut scale = cast(zero, &Rational::from(1));
                for _ in 0..power.unsigned_abs() {
                    scale = if *power > 0 {
                        scale / root.clone()
                    } else {
                        scale * root.clone()
                    };
                }
                let w = complex(re, im);
                Complex::new(w.re * scale.clone(), w.im * scale)
            })
            .collect();
        let interference = self
            .counterterms
            .as_ref()
            .map(|ct| ct.interference.runtime(zero));
        let metric = self.metric.as_ref().map(|entries| {
            entries
                .iter()
                .map(|(_s, _l, _r, [re, im])| complex(re, im))
                .collect()
        });
        let columns = self
            .columns
            .as_ref()
            .map(|columns| {
                columns
                    .programs
                    .iter()
                    .map(|program| program.runtime(zero))
                    .collect()
            })
            .unwrap_or_default();
        Ok(TreeRuntime {
            parameters,
            kernels,
            edge_weights,
            counterterm_kernels,
            counterterm_edge_weights,
            weights,
            metric,
            average: cast(zero, &self.average),
            interference,
            columns,
        })
    }

    /// Set the observable columns (`loops/v2/native_observables.py`; JSON
    /// `ColumnsFile`): each point then also forms every column's Born and
    /// counterterm/field interferences (`TreeInputs::columns`). Needs the
    /// counterterm components, whose interference layout the columns share.
    /// Build the runtime after setting them.
    pub fn set_columns(&mut self, json: &[u8]) -> RusticolResult<()> {
        let mut file: ColumnsFile =
            serde_json::from_slice(json).map_err(|e| invalid(format!("columns: {e}")))?;
        // Each metric is loaded once (the columns share them by index).
        let mut metrics: Vec<Option<MetricCells>> = Vec::with_capacity(file.cells.len());
        for cells in std::mem::take(&mut file.cells) {
            metrics.push(cells.map(load_cells).transpose()?);
        }
        // Columns need the counterterm interference layout; a program
        // without counterterms (the certified-scaleless companion's) takes
        // the spin sources only.
        let ct = self.counterterms.as_ref();
        require(
            ct.is_some() || file.columns.is_empty(),
            "observable columns need the counterterm components",
        )?;
        let basis = ct.map_or(0, |ct| ct.layout.basis);
        let flags = |kept: &Option<Vec<usize>>, count: usize, what: &str| {
            kept.as_ref()
                .map(|kept| {
                    let mut flags = vec![false; count];
                    for &k in kept {
                        if k >= count {
                            return Err(invalid(format!("a column's {what} index")));
                        }
                        flags[k] = true;
                    }
                    Ok(flags)
                })
                .transpose()
        };
        // One interference per distinct (program, metric) pair.
        let mut index: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut programs = Vec::new();
        let mut interference = |program: usize, cells: usize| -> RusticolResult<usize> {
            if let Some(&found) = index.get(&(program, cells)) {
                return Ok(found);
            }
            let ct = ct.expect("columns need counterterms");
            let layout = &ct.layout;
            let record = file
                .programs
                .get(program)
                .ok_or_else(|| invalid("a column's program index"))?;
            let metric = match metrics
                .get(cells)
                .ok_or_else(|| invalid("a column's metric index"))?
            {
                Some(metric) => metric.clone(),
                // The Born metric: the counterterm interference's own cells.
                None => ct.interference.cells().clone(),
            };
            programs.push(Interference::new(
                layout.basis,
                layout.orders,
                layout.tensors,
                metric,
                &layout.born,
                &layout.ket,
                layout.vectors,
                load_entries(&record.entries)?,
                record.outputs.clone(),
                ct.couplings,
                layout.depth,
            )?);
            index.insert((program, cells), programs.len() - 1);
            Ok(programs.len() - 1)
        };
        let mut columns = Vec::with_capacity(file.columns.len());
        for column in &file.columns {
            let record = file
                .programs
                .get(column.program)
                .ok_or_else(|| invalid("a column's program index"))?;
            require(
                record.born_outputs <= record.outputs.len(),
                "a column's Born outputs",
            )?;
            columns.push(Column {
                direct: interference(column.program, column.cells)?,
                swapped: column
                    .swapped
                    .map(|cells| interference(column.program, cells))
                    .transpose()?,
                born_outputs: record.born_outputs,
                rows: flags(&column.rows, self.rows, "row")?,
                basis: flags(&column.basis, basis, "basis element")?,
            });
        }
        let slots = |records: &[(usize, usize, String, String)], size: usize| {
            let mut out: SourceSlots = BTreeMap::new();
            for (leg, start, re, im) in records {
                require(
                    *leg >= 1 && *leg <= self.legs && start + 4 <= size,
                    "a spin-source slot",
                )?;
                out.entry(*leg)
                    .or_default()
                    .push((*start, [exact(re)?, exact(im)?]));
            }
            Ok::<_, RusticolError>(out)
        };
        let counterterm_size = self
            .counterterm_trees
            .as_ref()
            .map_or(0, |ct| ct.layout.size);
        let restricts = !columns.is_empty();
        self.columns = Some(Columns {
            programs,
            columns,
            trees: slots(&file.trees, self.state_size)?,
            counterterms: slots(&file.counterterms, counterterm_size)?,
        });
        // Observable columns of a plan that keeps some Born rows: the
        // program evaluates what their kept rows read (`rows`).
        if restricts && let Some(restriction) = self.restriction.take() {
            self.restrict_rows(restriction);
        }
        Ok(())
    }

    /// Whether the program forms the counterterm and external-field
    /// components.
    pub fn has_counterterms(&self) -> bool {
        self.counterterms.is_some()
    }

    /// The native inputs of a batch: `momenta` are the plan's all-incoming
    /// momenta per point (initial legs as given, final legs negated) and
    /// `constants` the plan's kernel constants. Without `poles` the
    /// counterterm outputs of a nonzero epsilon power stay zero and the
    /// pole-only coupling slots are not evaluated.
    #[allow(clippy::too_many_arguments)]
    pub fn inputs<T: TreeCarrier>(
        &self,
        runtime: &mut TreeRuntime<T>,
        plan: &LoopProgramV4,
        batch: &TreeBatch<T>,
        momenta: &[Vec<Rational>],
        constants: &[Complex<T>],
        poles: bool,
        zero: &T,
    ) -> RusticolResult<TreeInputs<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        self.inputs_with(runtime, plan, batch, momenta, constants, poles, None, zero)
    }

    /// `inputs` with per point spin sources (`sources`; the observable
    /// columns' spin-source slots, `set_columns`): each listed leg's
    /// wavefunction is the given vector times its source's crossing phase in
    /// the trees and the counterterm program, and the point's Born
    /// components vanish outside its kept rows.
    #[allow(clippy::too_many_arguments)]
    pub fn inputs_with<T: TreeCarrier>(
        &self,
        runtime: &mut TreeRuntime<T>,
        plan: &LoopProgramV4,
        batch: &TreeBatch<T>,
        momenta: &[Vec<Rational>],
        constants: &[Complex<T>],
        poles: bool,
        sources: Option<&[PointSources<T>]>,
        zero: &T,
    ) -> RusticolResult<TreeInputs<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        self.inputs_loop_side(
            runtime, plan, batch, momenta, constants, poles, sources, None, zero,
        )
    }

    /// `inputs_with` with per point loop-side sources (`loop_sources`, one
    /// `LoopSources` per point; the asymmetric Ward mode): each listed vector
    /// leg's wavefunction is the given vector times its source's crossing
    /// phase (every helicity source of the leg, as a spin source) in a second
    /// run of the trees and in the counterterm program, on top of the spin
    /// sources. The physical run gives the plan's Born components (the seeds'
    /// bra), every Born component (`amplitudes`) and `|A0|²`; the loop-side
    /// run gives the block rows, the counterterm kets and the external-field
    /// insertions' Born ket, and `loop_born` the Born interference of the
    /// physical bra with the loop-side ket. The vectors need the full tree
    /// kernels (the caller's slice generation; the transverse Yang-Mills
    /// kernel assumes conserved currents). Observable columns are refused
    /// with loop-side sources (they run the inclusive outputs).
    #[allow(clippy::too_many_arguments)]
    pub fn inputs_loop_side<T: TreeCarrier>(
        &self,
        runtime: &mut TreeRuntime<T>,
        plan: &LoopProgramV4,
        batch: &TreeBatch<T>,
        momenta: &[Vec<Rational>],
        constants: &[Complex<T>],
        poles: bool,
        sources: Option<&[PointSources<T>]>,
        loop_sources: Option<&[LoopSources<T>]>,
        zero: &T,
    ) -> RusticolResult<TreeInputs<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        if let Some(loop_sources) = loop_sources {
            require(
                loop_sources.len() == momenta.len(),
                "loop-side sources per point",
            )?;
            require(
                self.columns
                    .as_ref()
                    .is_none_or(|columns| columns.columns.is_empty()),
                "loop-side sources run the inclusive outputs, not observable columns",
            )?;
        }
        if let Some(sources) = sources {
            require(
                sources.len() == momenta.len() && self.columns.is_some(),
                "spin sources per point need the observable columns",
            )?;
            for point in sources {
                require(
                    point
                        .rows
                        .as_ref()
                        .is_none_or(|rows| rows.len() == self.rows),
                    "a point's kept rows",
                )?;
            }
        }
        // Exact states under the key; plane applications touch no Symbolica
        // state.
        let _unlock = runtime
            .runs_exact_states()
            .then(crate::licence::unlock)
            .flatten();
        let points = momenta.len();
        require(
            batch.model.len() == points
                && batch
                    .model
                    .iter()
                    .all(|model| model.len() == self.model_count),
            "model parameter count",
        )?;
        require(batch.couplings.len() == self.couplings, "coupling count")?;
        if let Some(ct) = &self.counterterms {
            require(
                batch.counterterm_couplings.len() == points
                    && batch
                        .counterterm_couplings
                        .iter()
                        .all(|couplings| couplings.len() == ct.couplings),
                "counterterm coupling count",
            )?;
        }
        if let Some(ct) = &self.counterterm_trees {
            require(
                batch.counterterm_model.len() == points
                    && batch
                        .counterterm_model
                        .iter()
                        .all(|model| model.len() == ct.layout.model_block),
                "counterterm program model parameter count",
            )?;
        }
        require(constants.len() == plan.constants, "kernel constant count")?;
        let czero = Complex::new(zero.clone(), zero.clone());
        // Incoming momenta per point and leg, in the carrier.
        let mut incoming = Vec::with_capacity(points);
        for point in momenta {
            require(point.len() == 4 * self.legs, "momentum components")?;
            incoming.push(
                point
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| p.each_ref().map(|x| cast(zero, x)))
                    .collect::<Vec<[T; 4]>>(),
            );
        }
        let masses: BTreeMap<i32, T> = batch
            .masses
            .iter()
            .map(|(&pdg, mass)| (pdg, cast(zero, mass)))
            .chain(
                batch
                    .derived_masses
                    .iter()
                    .map(|(&pdg, mass)| (pdg, mass.clone())),
            )
            .collect();
        // The tree state of every point.
        let layout = StateLayout {
            size: self.state_size,
            model_offset: self.model_offset,
            model_block: self.model_count,
        };
        let started = std::time::Instant::now();
        let mut states = Self::states(
            layout,
            &self.sources,
            &self.momentum_slots,
            &self.initial,
            &incoming,
            &masses,
            &batch.model,
            zero,
        )?;
        if let (Some(sources), Some(columns)) = (sources, &self.columns) {
            Self::override_sources(&mut states, layout.size, &columns.trees, sources, zero)?;
        }
        // The loop side's states: the physical ones with each point's
        // loop-side vectors in their legs' vector source slots.
        let mut loop_states = match loop_sources {
            Some(loop_sources) => {
                let mut loop_states = states.clone();
                Self::override_loop_sources(
                    &mut loop_states,
                    layout.size,
                    &self.sources,
                    loop_sources,
                    zero,
                )?;
                Some(loop_states)
            }
            None => None,
        };
        for states in std::iter::once(&mut states).chain(loop_states.as_mut()) {
            self.scale_photons(states, layout.size, &self.sources, batch)?;
        }
        for states in std::iter::once(&mut states).chain(loop_states.as_mut()) {
            run_interactions(
                &self.interactions,
                layout,
                &mut runtime.kernels,
                &runtime.edge_weights,
                states,
                points,
                true,
                &czero,
            )?;
        }
        // The counterterm program's state of every point (empty without it).
        let counterterm_states = match &self.counterterm_trees {
            Some(ct) => {
                let mut ct_states = Self::states(
                    ct.layout,
                    &ct.sources,
                    &ct.momentum_slots,
                    &self.initial,
                    &incoming,
                    &masses,
                    &batch.counterterm_model,
                    zero,
                )?;
                if let (Some(sources), Some(columns)) = (sources, &self.columns) {
                    Self::override_sources(
                        &mut ct_states,
                        ct.layout.size,
                        &columns.counterterms,
                        sources,
                        zero,
                    )?;
                }
                // Counterterm currents are loop side only.
                if let Some(loop_sources) = loop_sources {
                    Self::override_loop_sources(
                        &mut ct_states,
                        ct.layout.size,
                        &ct.sources,
                        loop_sources,
                        zero,
                    )?;
                }
                self.scale_photons(&mut ct_states, ct.layout.size, &ct.sources, batch)?;
                run_interactions(
                    &ct.interactions,
                    ct.layout,
                    &mut runtime.counterterm_kernels,
                    &runtime.counterterm_edge_weights,
                    &mut ct_states,
                    points,
                    poles,
                    &czero,
                )?;
                ct_states
            }
            None => Vec::new(),
        };
        let counterterm_size = self
            .counterterm_trees
            .as_ref()
            .map_or(0, |ct| ct.layout.size);
        let mut out = TreeInputs {
            block_rows: Vec::with_capacity(points),
            born: Vec::with_capacity(points),
            amplitudes: Vec::with_capacity(points),
            born_squared: runtime.metric.is_some().then(|| Vec::with_capacity(points)),
            loop_born: (runtime.metric.is_some() && loop_states.is_some())
                .then(|| Vec::with_capacity(points)),
            counterterms: self
                .counterterms
                .as_ref()
                .map(|_| Vec::with_capacity(points)),
            columns: self.columns.as_ref().map(|_| Vec::with_capacity(points)),
            seconds: Vec::with_capacity(3),
        };
        out.seconds
            .push(("trees: programs", started.elapsed().as_secs_f64()));
        let started = std::time::Instant::now();
        // The root closures of every point, lanes of points at a time.
        let momenta: Vec<Vec<[T; 4]>> = incoming
            .iter()
            .map(|point| self.mask_momenta(point, zero))
            .collect();
        let states = (states.as_slice(), self.state_size);
        // The loop side's trees: the loop-side run, else the physical one.
        let loop_side = loop_states.as_ref().map_or(states, |loop_states| {
            (loop_states.as_slice(), self.state_size)
        });
        let counterterm_states = (counterterm_states.as_slice(), counterterm_size);
        let born_closure = |states| {
            self.close(
                &self.terms,
                &runtime.weights,
                &vec![batch.couplings.as_slice(); points],
                None,
                self.components,
                plan,
                &momenta,
                states,
                counterterm_states,
                constants,
                zero,
            )
        };
        let mut amplitudes = born_closure(states);
        // The loop side's Born amplitudes (the external-field insertions'
        // ket), with loop-side sources.
        let mut loop_amplitudes = loop_states.as_ref().map(|_| born_closure(loop_side));
        let kets = match (&self.counterterms, &runtime.interference) {
            (Some(ct), Some(_)) => self.close(
                &ct.terms,
                &runtime.weights,
                &batch
                    .counterterm_couplings
                    .iter()
                    .map(Vec::as_slice)
                    .collect::<Vec<_>>(),
                (!poles).then_some(ct.finite_slot.as_slice()),
                ct.components,
                plan,
                &momenta,
                loop_side,
                counterterm_states,
                constants,
                zero,
            ),
            _ => Vec::new(),
        };
        out.seconds
            .push(("trees: closures", started.elapsed().as_secs_f64()));
        let started = std::time::Instant::now();
        for (index, state) in loop_side.0.chunks_exact(self.state_size).enumerate() {
            let mut amplitudes = std::mem::take(&mut amplitudes[index]);
            let mut loop_amplitudes = loop_amplitudes
                .as_mut()
                .map(|loop_amplitudes| std::mem::take(&mut loop_amplitudes[index]));
            let kept_rows = sources.and_then(|sources| sources[index].rows.as_deref());
            if let Some(kept) = kept_rows {
                // A source leg keeps its fixed row only: the other rows'
                // Born components vanish (no seeds, no interference).
                for amplitudes in std::iter::once(&mut amplitudes).chain(loop_amplitudes.as_mut()) {
                    for (row, keep) in kept.iter().enumerate() {
                        if !keep {
                            for value in &mut amplitudes[row * self.components..][..self.components]
                            {
                                *value = czero.clone();
                            }
                        }
                    }
                }
            }
            if let (Some(columns), Some(out)) = (
                self.columns.as_ref().filter(|c| !c.columns.is_empty()),
                &mut out.columns,
            ) {
                let ct = self
                    .counterterms
                    .as_ref()
                    .expect("columns need counterterms");
                let couplings = &batch.counterterm_couplings[index];
                let every = vec![true; self.rows];
                let mut values = Vec::with_capacity(columns.columns.len());
                for column in &columns.columns {
                    let keep: Vec<bool> = (0..self.rows)
                        .map(|row| {
                            column.rows.as_ref().is_none_or(|rows| rows[row])
                                && kept_rows.is_none_or(|kept| kept[row])
                        })
                        .collect();
                    let keep = if keep.iter().all(|k| *k) {
                        &every
                    } else {
                        &keep
                    };
                    let evaluate = |program: usize| {
                        columns.programs[program].evaluate_complex(
                            &runtime.columns[program],
                            &amplitudes,
                            None,
                            self.components,
                            &kets[index],
                            ct.components,
                            keep,
                            column.basis.as_deref(),
                            couplings,
                            poles,
                            zero,
                        )
                    };
                    let mut column_values = evaluate(column.direct);
                    let other = column.swapped.map(evaluate);
                    for (k, value) in column_values.iter_mut().enumerate() {
                        if k < column.born_outputs {
                            // The Born: `observables._Assembly.result`
                            // divides it by the average.
                            continue;
                        }
                        // Python's `value + other.conjugate()`, divided by
                        // the average as `_Assembly._local` does.
                        let conjugate = match &other {
                            Some(other) => other[k].conj(),
                            None => value.conj(),
                        };
                        *value += conjugate;
                        *value = Complex::new(
                            value.re.clone() / runtime.average.clone(),
                            value.im.clone() / runtime.average.clone(),
                        );
                    }
                    values.push(column_values);
                }
                out.push(values);
            }
            let mut rows = Vec::new();
            for &(start, dimension, sign) in &self.block_rows {
                for value in &state[start..start + dimension] {
                    rows.push(if sign < 0 {
                        -value.clone()
                    } else {
                        value.clone()
                    });
                }
            }
            for &(row, start, sign) in &self.block_row_terms {
                let dimension = self.block_rows[row].1;
                let offset = self.block_row_offsets[row];
                for (target, value) in rows[offset..offset + dimension]
                    .iter_mut()
                    .zip(&state[start..start + dimension])
                {
                    *target += if sign < 0 {
                        -value.clone()
                    } else {
                        value.clone()
                    };
                }
            }
            out.block_rows.push(rows);
            if let (Some(ct), Some(interference), Some(out)) = (
                &self.counterterms,
                &runtime.interference,
                &mut out.counterterms,
            ) {
                let couplings = &batch.counterterm_couplings[index];
                let values = ct.interference.evaluate(
                    interference,
                    &amplitudes,
                    loop_amplitudes.as_deref(),
                    self.components,
                    &kets[index],
                    ct.components,
                    self.rows,
                    couplings,
                    poles,
                    zero,
                );
                out.push(
                    values
                        .into_iter()
                        .map(|value| value / runtime.average.clone())
                        .collect(),
                );
            }
            let mut born = Vec::with_capacity(self.rows * self.plan_components.len());
            for row in amplitudes.chunks_exact(self.components) {
                for component in &self.plan_components {
                    born.push(component.map_or_else(|| czero.clone(), |j| row[j].clone()));
                }
            }
            out.born.push(born);
            if let Some(squared) = &mut out.born_squared {
                let metric = runtime.metric.as_ref().expect("a runtime Born metric");
                squared.push(self.squared(runtime, metric, &amplitudes, &amplitudes, zero));
            }
            if let (Some(loop_born), Some(ket)) = (&mut out.loop_born, &loop_amplitudes) {
                let metric = runtime.metric.as_ref().expect("a runtime Born metric");
                loop_born.push(self.squared(runtime, metric, &amplitudes, ket, zero));
            }
            out.amplitudes.push(amplitudes);
        }
        out.seconds
            .push(("trees: interferences", started.elapsed().as_secs_f64()));
        Ok(out)
    }

    /// The states of a batch's points for one interaction program: the
    /// external wavefunctions, the all-outgoing momentum of every momentum
    /// slot and the model parameter values (`models`, per point).
    #[allow(clippy::too_many_arguments)]
    fn states<T: TreeCarrier>(
        layout: StateLayout,
        sources: &SchemaSources,
        momentum_slots: &[(usize, u64)],
        initial: &[bool],
        incoming: &[Vec<[T; 4]>],
        masses: &BTreeMap<i32, T>,
        models: &[Vec<T>],
        zero: &T,
    ) -> RusticolResult<Vec<Complex<T>>> {
        let czero = Complex::new(zero.clone(), zero.clone());
        let mut states = vec![czero; incoming.len() * layout.size];
        for ((point, state), model) in incoming
            .iter()
            .zip(states.chunks_exact_mut(layout.size))
            .zip(models)
        {
            let physical: Vec<[T; 4]> = point
                .iter()
                .zip(initial)
                .map(|(p, &initial)| {
                    if initial {
                        p.clone()
                    } else {
                        p.clone().map(|x| -x)
                    }
                })
                .collect();
            T::fill_sources(sources, masses, &physical, state)?;
            for &(start, mask) in momentum_slots {
                // All-outgoing momentum: minus the incoming sum.
                for mu in 0..4 {
                    let mut total = zero.clone();
                    for (leg, p) in point.iter().enumerate() {
                        if mask >> leg & 1 == 1 {
                            total -= p[mu].clone();
                        }
                    }
                    state[start + mu] = Complex::new(total, zero.clone());
                }
            }
            for (index, value) in model.iter().enumerate() {
                state[layout.model_offset + index] = Complex::new(value.clone(), zero.clone());
            }
        }
        Ok(states)
    }

    /// Multiply every source of the on-shell external photons
    /// (`ParameterFile::photons`) by its point's sqrt(alpha(0) / alpha),
    /// after the spin and loop-side sources replaced theirs: one interaction
    /// program's states, `sources` its source fill. Every amplitude is
    /// linear in each leg's source, so each output takes
    /// (alpha(0) / alpha)^n, as the Python lane's scaled average
    /// (`assemble._PointEvaluation.average`) gives it.
    fn scale_photons<T: TreeCarrier>(
        &self,
        states: &mut [Complex<T>],
        size: usize,
        sources: &SchemaSources,
        batch: &TreeBatch<T>,
    ) -> RusticolResult<()> {
        let Some((_, legs)) = &self.parameters.photons else {
            return Ok(());
        };
        if size == 0 {
            return Ok(());
        }
        require(
            batch.photon_scales.len() * size == states.len(),
            "a photon source scale per point",
        )?;
        let slots: Vec<(usize, usize)> = legs
            .iter()
            .flat_map(|&leg| sources.leg_slots(leg))
            .collect();
        require(
            slots.iter().all(|&(start, stop)| start <= stop && stop <= size),
            "a photon source slot outside the state",
        )?;
        for (state, scale) in states.chunks_exact_mut(size).zip(&batch.photon_scales) {
            for &(start, stop) in &slots {
                for value in &mut state[start..stop] {
                    *value = Complex::new(
                        value.re.clone() * scale.clone(),
                        value.im.clone() * scale.clone(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Write each point's spin-source vectors (times each slot's crossing
    /// phase) into the source slots of its legs.
    fn override_sources<T: TreeCarrier>(
        states: &mut [Complex<T>],
        size: usize,
        slots: &SourceSlots,
        sources: &[PointSources<T>],
        zero: &T,
    ) -> RusticolResult<()> {
        Self::write_vectors(
            states,
            size,
            slots,
            sources.iter().map(|point| point.vectors.as_slice()),
            zero,
        )
    }

    /// Write each point's loop-side vectors (times each slot's crossing
    /// phase) into the vector source slots of its legs in one interaction
    /// program's state; `sources` is that program's source fill, whose
    /// vector sources (`SchemaSources::vector_slots`) are the targets.
    fn override_loop_sources<T: TreeCarrier>(
        states: &mut [Complex<T>],
        size: usize,
        sources: &SchemaSources,
        loop_sources: &[LoopSources<T>],
        zero: &T,
    ) -> RusticolResult<()> {
        let mut slots: SourceSlots = BTreeMap::new();
        for (leg, start, [re, im]) in sources.vector_slots() {
            require(start + 4 <= size, "a vector source slot outside the state")?;
            // The phase as the Python lane reads it (`Decimal(str(x))`).
            let phase = [exact(&format!("{re:?}"))?, exact(&format!("{im:?}"))?];
            slots.entry(leg).or_default().push((start, phase));
        }
        Self::write_vectors(
            states,
            size,
            &slots,
            loop_sources.iter().map(Vec::as_slice),
            zero,
        )
    }

    /// Write per point its (leg, vector) pairs, times each slot's crossing
    /// phase, into the source slots of the legs (`slots`).
    fn write_vectors<'a, T: TreeCarrier + 'a>(
        states: &mut [Complex<T>],
        size: usize,
        slots: &SourceSlots,
        points: impl Iterator<Item = &'a [(usize, [Complex<T>; 4])]>,
        zero: &T,
    ) -> RusticolResult<()> {
        for (state, vectors) in states.chunks_exact_mut(size.max(1)).zip(points) {
            for (leg, vector) in vectors {
                let found = slots
                    .get(leg)
                    .ok_or_else(|| invalid(format!("leg {leg} has no vector source slot")))?;
                for (start, [re, im]) in found {
                    let phase = Complex::new(cast(zero, re), cast(zero, im));
                    for (mu, component) in vector.iter().enumerate() {
                        state[start + mu] = component.clone() * phase.clone();
                    }
                }
            }
        }
        Ok(())
    }

    /// The incoming momentum sum of each closure slot's leg mask.
    fn mask_momenta<T: Carrier>(&self, incoming: &[[T; 4]], zero: &T) -> Vec<[T; 4]> {
        self.masks
            .iter()
            .map(|&mask| {
                [0, 1, 2, 3].map(|mu| {
                    let mut total = zero.clone();
                    for (leg, p) in incoming.iter().enumerate() {
                        if mask >> leg & 1 == 1 {
                            total += p[mu].clone();
                        }
                    }
                    total
                })
            })
            .collect()
    }

    /// The root closures of a batch's points: per point and Born row, every
    /// closure term's kernel on its blocks' rows (in the trees' or the
    /// counterterm program's state, each `(states, size)` with one state per
    /// point) and momenta (`momenta`: per point and leg mask), times its sign
    /// and per contribution its coupling (`couplings`: per point) and colour
    /// weight, accumulated into `width` components per row. `active`: the
    /// coupling slots to evaluate (none: all).
    ///
    /// Points run in lanes (`Closure::lanes`): every lane performs one point's
    /// operations in the scalar order, so a point's values do not depend on
    /// its batch; lanes and Born rows run in parallel. The batch runs in
    /// lanes of 8 points, its last `points % 8` in one chunk of 1, 2, 4 or
    /// 8 lanes (`lane_runs`).
    #[allow(clippy::too_many_arguments)]
    fn close<T: Carrier>(
        &self,
        terms: &[Term],
        weights: &[Complex<T>],
        couplings: &[&[Complex<T>]],
        active: Option<&[bool]>,
        width: usize,
        plan: &LoopProgramV4,
        momenta: &[Vec<[T; 4]>],
        states: (&[Complex<T>], usize),
        counterterm_states: (&[Complex<T>], usize),
        constants: &[Complex<T>],
        zero: &T,
    ) -> Vec<Vec<Complex<T>>> {
        let closure = Closure {
            program: self,
            terms,
            weights,
            couplings,
            active,
            width,
            plan,
            momenta,
            states,
            counterterm_states,
            constants,
        };
        let mut out = Vec::with_capacity(momenta.len());
        for (first, count, lanes) in lane_runs(momenta.len()) {
            out.extend(match lanes {
                8 => closure.lanes::<8>(first, count, zero),
                4 => closure.lanes::<4>(first, count, zero),
                2 => closure.lanes::<2>(first, count, zero),
                _ => closure.lanes::<1>(first, count, zero),
            });
        }
        out
    }

    /// `Re <bra|M|ket>` per squared Born order of one point (divided by the
    /// average): the metric over pairs of Born components, `|A0|²` for
    /// `bra = ket` (the loop-side Born interference otherwise).
    fn squared<T: Carrier>(
        &self,
        runtime: &TreeRuntime<T>,
        metric: &[Complex<T>],
        bra: &[Complex<T>],
        ket: &[Complex<T>],
        zero: &T,
    ) -> Vec<T> {
        let entries = self.metric.as_ref().expect("a runtime metric has entries");
        let mut out = vec![zero.clone(); self.squared.len()];
        for (row, ket) in bra
            .chunks_exact(self.components)
            .zip(ket.chunks_exact(self.components))
        {
            for ((squared, left, right, _), value) in entries.iter().zip(metric) {
                out[*squared] += (row[*left].conj() * value.clone() * ket[*right].clone()).re;
            }
        }
        out.into_iter()
            .map(|value| value / runtime.average.clone())
            .collect()
    }
}

/// The runs (first point, points, lane width) a batch of `points` points
/// closes in: lanes of 8 points, then the remaining `points % 8` in one
/// chunk of the narrowest width that holds them (1, 2, 4 or 8; a chunk's
/// unused lanes repeat its run's last point and are dropped). A chunk
/// costs about its instruction count at any width (g g > g g g g full
/// colour, binary64 closures per chunk of width 8 / 4 / 2 / 1: 487 / 449 /
/// 350 / 249 ms), so the batch runs in the fewest chunks, its last one as
/// narrow as it can be.
fn lane_runs(points: usize) -> Vec<(usize, usize, usize)> {
    let full = points / 8 * 8;
    let rest = points - full;
    let mut runs = Vec::with_capacity(2);
    if full > 0 {
        runs.push((0, full, 8));
    }
    if rest > 0 {
        runs.push((full, rest, rest.next_power_of_two()));
    }
    runs
}

/// The root closure of a batch (`TreeProgram::close`).
struct Closure<'a, T> {
    program: &'a TreeProgram,
    terms: &'a [Term],
    weights: &'a [Complex<T>],
    couplings: &'a [&'a [Complex<T>]],
    active: Option<&'a [bool]>,
    width: usize,
    plan: &'a LoopProgramV4,
    momenta: &'a [Vec<[T; 4]>],
    states: (&'a [Complex<T>], usize),
    counterterm_states: (&'a [Complex<T>], usize),
    constants: &'a [Complex<T>],
}

/// `(x_re, x_im) *= y` on complex lanes (`y`: `L` real parts, then `L`
/// imaginary parts), per lane as `Complex<T>`'s product. The lanes are
/// arrays, so a product chain stays in registers.
#[inline(always)]
fn lane_mul<T: Carrier, const L: usize>(x_re: &mut [T; L], x_im: &mut [T; L], y: &[T]) {
    let (y_re, y_im) = y[..2 * L].split_at(L);
    for l in 0..L {
        let re = x_re[l].clone() * &y_re[l] - x_im[l].clone() * &y_im[l];
        let im = x_re[l].clone() * &y_im[l] + x_im[l].clone() * &y_re[l];
        x_re[l] = re;
        x_im[l] = im;
    }
}

/// `Σ_terms constant × Π input^power` of a closure kernel on complex lanes
/// (`x`: per kernel input `L` real parts, then `L` imaginary parts), per
/// lane in the scalar order of `Complex<T>`: each term's product from its
/// constant, the terms summed from zero.
#[inline(always)]
fn kernel_lanes<T: Carrier, const L: usize>(
    kernel: &[super::plan::KernelTerm],
    constants: &[Complex<T>],
    x: &[T],
    zero: &T,
) -> ([T; L], [T; L]) {
    let v = 2 * L;
    let mut value_re: [T; L] = std::array::from_fn(|_| zero.clone());
    let mut value_im: [T; L] = std::array::from_fn(|_| zero.clone());
    for entry in kernel {
        let constant = &constants[entry.constant];
        let mut re: [T; L] = std::array::from_fn(|_| constant.re.clone());
        let mut im: [T; L] = std::array::from_fn(|_| constant.im.clone());
        for &(input, exponent) in &entry.monomial {
            let factor = &x[input * v..(input + 1) * v];
            for _ in 0..exponent {
                lane_mul::<T, L>(&mut re, &mut im, factor);
            }
        }
        for l in 0..L {
            value_re[l] += &re[l];
            value_im[l] += &im[l];
        }
    }
    (value_re, value_im)
}

/// `kernel_lanes` compiled for AVX-512 and AVX2 (wider vectors for the same
/// operations: the products and sums of a lane are those of `Complex<T>`,
/// never fused, so the values are bitwise the baseline's).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn kernel_lanes_avx512<T: Carrier, const L: usize>(
    kernel: &[super::plan::KernelTerm],
    constants: &[Complex<T>],
    x: &[T],
    zero: &T,
) -> ([T; L], [T; L]) {
    kernel_lanes::<T, L>(kernel, constants, x, zero)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn kernel_lanes_avx2<T: Carrier, const L: usize>(
    kernel: &[super::plan::KernelTerm],
    constants: &[Complex<T>],
    x: &[T],
    zero: &T,
) -> ([T; L], [T; L]) {
    kernel_lanes::<T, L>(kernel, constants, x, zero)
}

/// The vector width the closure kernels run in: binary64 lanes take the
/// widest the running CPU reports (never a build-host feature), the other
/// carriers the baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wide {
    Baseline,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// Caps the closure kernels' vector width (`Wide::detect`) at `baseline`
/// or `avx2`, so that one AVX-512 host runs the code paths of an AVX2-only
/// or a baseline x86-64 host (the bitwise check of
/// `tests/unit/test_loop_v2_native_trees.py`); any other value, or none,
/// caps nothing. It never enables a feature the CPU lacks.
const CLOSURE_SIMD_ENV: &str = "RUSTICOL_TEST_CLOSURE_SIMD";

impl Wide {
    fn detect<T: 'static>() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            static CAP: std::sync::OnceLock<Wide> = std::sync::OnceLock::new();
            let cap =
                *CAP.get_or_init(|| Self::cap(std::env::var(CLOSURE_SIMD_ENV).ok().as_deref()));
            if cap == Self::Avx512 && std::arch::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if cap != Self::Baseline && std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        Self::Baseline
    }

    /// The widest width `CLOSURE_SIMD_ENV`'s value allows.
    #[cfg(target_arch = "x86_64")]
    fn cap(value: Option<&str>) -> Self {
        match value {
            Some("baseline") => Self::Baseline,
            Some("avx2") => Self::Avx2,
            _ => Self::Avx512,
        }
    }
}

impl<T: Carrier> Closure<'_, T> {
    /// The closures of points `first .. first + points` in lanes of `L`
    /// points (the unused lanes of a last chunk repeat the run's last point
    /// and are dropped): per lane
    /// chunk and Born row, the terms in order, each lane in the scalar
    /// operations of `Complex<T>` (the kernel's terms `constant × Π
    /// input^power` summed from zero, the sign, then per contribution
    /// `(coupling × weight) × value` added to its component).
    fn lanes<const L: usize>(&self, first: usize, points: usize, zero: &T) -> Vec<Vec<Complex<T>>> {
        let program = self.program;
        let rows = program.rows;
        let width = self.width;
        let czero = Complex::new(zero.clone(), zero.clone());
        let mut out = vec![vec![czero; rows * width]; points];
        if width == 0 || rows == 0 {
            return out;
        }
        let v = 2 * L;
        let chunks = points.div_ceil(L);
        let inputs = program.layouts.iter().map(Vec::len).max().unwrap_or(0);
        let (state, size) = self.states;
        let (ct_state, ct_size) = self.counterterm_states;
        let wide = Wide::detect::<T>();
        // Per (lane chunk, row): `width` complex lane vectors.
        let mut sums = vec![zero.clone(); chunks * rows * width * v];
        sums.par_chunks_mut(width * v).enumerate().for_each_init(
            || (vec![zero.clone(); inputs * v], Vec::new()),
            |(x, starts), (task, target)| {
                let (chunk, row) = (task / rows, task % rows);
                // A row the program's restriction drops stays zero (`rows`).
                if program.closure_rows.as_ref().is_some_and(|kept| !kept[row]) {
                    return;
                }
                let lanes: [usize; L] =
                    std::array::from_fn(|l| first + (chunk * L + l).min(points - 1));
                for term in self.terms {
                    if self
                        .active
                        .is_some_and(|active| term.contributions.iter().all(|(c, ..)| !active[*c]))
                    {
                        continue;
                    }
                    starts.clear();
                    for &block in &term.blocks {
                        match program.blocks[block][row] {
                            Some(start) => starts.push(start),
                            None => break,
                        }
                    }
                    if starts.len() != term.blocks.len() {
                        continue;
                    }
                    for (input, lane) in program.layouts[term.layout]
                        .iter()
                        .zip(x.chunks_exact_mut(v))
                    {
                        let (re, im) = lane.split_at_mut(L);
                        match *input {
                            TermInput::Current { slot, component } => {
                                let index = starts[slot] + component;
                                let (values, stride, index) = if index < size {
                                    (state, size, index)
                                } else {
                                    (ct_state, ct_size, index - size)
                                };
                                for (l, &point) in lanes.iter().enumerate() {
                                    let value = &values[point * stride + index];
                                    re[l] = value.re.clone();
                                    im[l] = value.im.clone();
                                }
                            }
                            TermInput::Momentum { slot, mu } => {
                                let mask = term.masks[slot];
                                for (l, &point) in lanes.iter().enumerate() {
                                    re[l] = self.momenta[point][mask][mu].clone();
                                    im[l] = zero.clone();
                                }
                            }
                        }
                    }
                    let kernel = &self.plan.kernels[term.kernel].outputs[0];
                    let (mut value_re, mut value_im) = match wide {
                        // SAFETY: `wide` holds only the features this CPU
                        // reports (`Wide::detect`).
                        #[cfg(target_arch = "x86_64")]
                        Wide::Avx512 => unsafe {
                            kernel_lanes_avx512::<T, L>(kernel, self.constants, x, zero)
                        },
                        #[cfg(target_arch = "x86_64")]
                        Wide::Avx2 => unsafe {
                            kernel_lanes_avx2::<T, L>(kernel, self.constants, x, zero)
                        },
                        Wide::Baseline => kernel_lanes::<T, L>(kernel, self.constants, x, zero),
                    };
                    if term.sign < 0 {
                        for l in 0..L {
                            value_re[l] = -value_re[l].clone();
                            value_im[l] = -value_im[l].clone();
                        }
                    }
                    for &(coupling, component, weight) in &term.contributions {
                        if self.active.is_some_and(|active| !active[coupling]) {
                            continue;
                        }
                        let weight = &self.weights[weight];
                        let (re, im) = target[component * v..(component + 1) * v].split_at_mut(L);
                        for (l, &point) in lanes.iter().enumerate() {
                            let factor = self.couplings[point][coupling].clone() * weight;
                            re[l] +=
                                factor.re.clone() * &value_re[l] - factor.im.clone() * &value_im[l];
                            im[l] +=
                                factor.re.clone() * &value_im[l] + factor.im.clone() * &value_re[l];
                        }
                    }
                }
            },
        );
        for (task, target) in sums.chunks_exact(width * v).enumerate() {
            let (chunk, row) = (task / rows, task % rows);
            for l in 0..L.min(points - chunk * L) {
                let values = &mut out[chunk * L + l][row * width..(row + 1) * width];
                for (component, item) in values.iter_mut().enumerate() {
                    *item = Complex::new(
                        target[component * v + l].clone(),
                        target[component * v + L + l].clone(),
                    );
                }
            }
        }
        out
    }
}

/// The interaction program on the states of a batch: per step, each
/// vertex kernel runs batched over its interactions and the points,
/// weighted sums accumulate per output current, then each output slot
/// copies its sum or applies its propagator kernel.
#[allow(clippy::too_many_arguments)]
fn run_interactions<T: Carrier>(
    program: &Interactions,
    layout: StateLayout,
    kernels: &mut [Kernel<T>],
    edge_weights: &[Vec<Vec<Complex<T>>>],
    states: &mut [Complex<T>],
    points: usize,
    poles: bool,
    czero: &Complex<T>,
) -> RusticolResult<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    // Rows (interaction x point) per kernel call. Without `poles` the
    // interactions and outputs of pole-only currents are skipped; no other
    // current reads them, and kernel rows are independent, so every other
    // value is bitwise unchanged.
    const ROWS: usize = 4096;
    let model = |kernel: usize| {
        program.models[kernel]
            .iter()
            .map(move |&index| layout.model_offset + index)
    };
    let mut parameters = Vec::new();
    let mut outputs = Vec::new();
    // The weights follow the program's interactions (a runtime built before
    // a row restriction, `rows`, would not).
    require(
        edge_weights.len() == program.steps.len()
            && program
                .steps
                .iter()
                .zip(edge_weights)
                .all(|(step, weights)| {
                    step.groups.len() == weights.len()
                        && step
                            .groups
                            .iter()
                            .zip(weights)
                            .all(|((_, edges), weights)| edges.len() == weights.len())
                }),
        "a tree runtime of another interaction program",
    )?;
    for (step, weights) in program.steps.iter().zip(edge_weights) {
        let mut sums = vec![czero.clone(); points * step.width];
        for ((kernel, edges), weights) in step.groups.iter().zip(weights) {
            let active: Vec<(&Edge, &Complex<T>)> = edges
                .iter()
                .zip(weights)
                .filter(|(edge, _)| poles || !edge.pole)
                .collect();
            if active.is_empty() {
                continue;
            }
            let evaluator = &mut kernels[*kernel];
            if let Kernel::Plane(plane) = evaluator {
                let models = &program.models[*kernel];
                plane_interactions(
                    plane.as_mut(),
                    layout,
                    models,
                    &active,
                    states,
                    step,
                    &mut sums,
                )?;
                continue;
            }
            let width = evaluator.output_len();
            let per_call = (ROWS / points.max(1)).max(1);
            for chunk in active.chunks(per_call) {
                let rows = chunk.len() * points;
                parameters.clear();
                for (edge, _) in chunk {
                    for state in states.chunks_exact(layout.size) {
                        parameters
                            .extend_from_slice(&state[edge.left..edge.left + edge.left_dimension]);
                        parameters.extend_from_slice(
                            &state[edge.right..edge.right + edge.right_dimension],
                        );
                        parameters
                            .extend_from_slice(&state[edge.left_momentum..edge.left_momentum + 4]);
                        parameters.extend_from_slice(
                            &state[edge.right_momentum..edge.right_momentum + 4],
                        );
                        parameters.extend(model(*kernel).map(|index| state[index].clone()));
                    }
                }
                outputs.clear();
                outputs.resize(rows * width, czero.clone());
                evaluator
                    .evaluate_batch(&parameters, &mut outputs, rows)
                    .map_err(|e| invalid(format!("interaction kernel: {e}")))?;
                let mut row = outputs.chunks_exact(width);
                for &(edge, weight) in chunk {
                    for point in 0..points {
                        let values = row.next().expect("a kernel row per point");
                        let target = &mut sums[point * step.width + edge.accumulator..];
                        for (sum, value) in target.iter_mut().zip(values) {
                            *sum += weight.clone() * value.clone();
                        }
                    }
                }
            }
        }
        for &(accumulator, propagator, momentum, start, dimension, pole) in &step.outputs {
            if pole && !poles {
                continue;
            }
            let Some(kernel) = propagator else {
                for (point, state) in states.chunks_exact_mut(layout.size).enumerate() {
                    state[start..start + dimension]
                        .clone_from_slice(&sums[point * step.width + accumulator..][..dimension]);
                }
                continue;
            };
            let evaluator = &mut kernels[kernel];
            if let Kernel::Plane(plane) = evaluator {
                let sums = (sums.as_slice(), step.width, accumulator);
                plane_propagator(
                    plane.as_mut(),
                    layout,
                    &program.models[kernel],
                    sums,
                    (momentum, start, dimension),
                    states,
                )?;
                continue;
            }
            let width = evaluator.output_len();
            let inputs = evaluator.input_len();
            let value = inputs - 4 - program.models[kernel].len();
            parameters.clear();
            for (point, state) in states.chunks_exact(layout.size).enumerate() {
                parameters.extend_from_slice(&sums[point * step.width + accumulator..][..value]);
                parameters.extend_from_slice(&state[momentum..momentum + 4]);
                parameters.extend(model(kernel).map(|index| state[index].clone()));
            }
            outputs.clear();
            outputs.resize(points * width, czero.clone());
            evaluator
                .evaluate_batch(&parameters, &mut outputs, points)
                .map_err(|e| invalid(format!("propagator kernel: {e}")))?;
            for (state, row) in states
                .chunks_exact_mut(layout.size)
                .zip(outputs.chunks_exact(width))
            {
                state[start..start + dimension].clone_from_slice(&row[..dimension]);
            }
        }
    }
    Ok(())
}

/// Rows (interaction x point) per plane-kernel call: a call's planes stay in
/// the core's L2 cache.
const PLANE_ROWS: usize = 1024;

/// Fill input planes `first..first + count` of a plane call: row (edge,
/// point) of input `first + k` is the point's state value at
/// `start(edge) + k`. One plane at a time: contiguous stores.
#[allow(clippy::too_many_arguments)]
fn fill_planes<T: Clone>(
    planes: &mut [T],
    stride: usize,
    first: usize,
    count: usize,
    edges: &[(&Edge, &Complex<T>)],
    states: &[Complex<T>],
    size: usize,
    start: impl Fn(&Edge) -> usize,
) {
    for k in 0..count {
        let (re, im) = planes[2 * (first + k) * stride..][..2 * stride].split_at_mut(stride);
        let mut row = 0;
        for (edge, _) in edges {
            let at = start(edge) + k;
            for state in states.chunks_exact(size) {
                re[row] = state[at].re.clone();
                im[row] = state[at].im.clone();
                row += 1;
            }
        }
    }
}

/// One kernel group of a step on a plane kernel (`run_interactions`): the
/// input planes filled from the states, without row-major rows (the model
/// inputs are the kernel's `models`, state components `model_offset +
/// index`); the outputs accumulated in the row-major path's order, so the
/// sums are bitwise the same.
fn plane_interactions<T: Carrier>(
    plane: &mut dyn PlaneRows<T>,
    layout: StateLayout,
    models: &[usize],
    active: &[(&Edge, &Complex<T>)],
    states: &[Complex<T>],
    step: &Step,
    sums: &mut [Complex<T>],
) -> RusticolResult<()>
where
    Complex<T>: Real + EvaluationDomain,
{
    let points = states.len() / layout.size.max(1);
    let Some(&(first, _)) = active.first() else {
        return Ok(());
    };
    if points == 0 {
        return Ok(());
    }
    let (left, right) = (first.left_dimension, first.right_dimension);
    let inputs = left + right + 8 + models.len();
    require(
        plane.input_len() == inputs
            && active
                .iter()
                .all(|(edge, _)| edge.left_dimension == left && edge.right_dimension == right),
        "interaction kernel inputs",
    )?;
    let width = plane.output_len();
    let size = layout.size;
    for chunk in active.chunks((PLANE_ROWS / points).max(1)) {
        let rows = chunk.len() * points;
        let (planes, stride) = plane.planes(rows);
        fill_planes(planes, stride, 0, left, chunk, states, size, |edge| {
            edge.left
        });
        fill_planes(planes, stride, left, right, chunk, states, size, |edge| {
            edge.right
        });
        let momenta = left + right;
        fill_planes(planes, stride, momenta, 4, chunk, states, size, |edge| {
            edge.left_momentum
        });
        fill_planes(
            planes,
            stride,
            momenta + 4,
            4,
            chunk,
            states,
            size,
            |edge| edge.right_momentum,
        );
        for (k, &index) in models.iter().enumerate() {
            fill_planes(
                planes,
                stride,
                momenta + 8 + k,
                1,
                chunk,
                states,
                size,
                |_| layout.model_offset + index,
            );
        }
        let planes = plane
            .run(rows)
            .map_err(|e| invalid(format!("interaction kernel: {}", e.message())))?;
        let base = 2 * inputs * stride;
        let mut row = 0;
        for &(edge, weight) in chunk {
            for point in 0..points {
                let target = &mut sums[point * step.width + edge.accumulator..];
                for (output, sum) in (0..width).zip(target.iter_mut()) {
                    let at = base + 2 * output * stride + row;
                    let value = Complex::new(planes[at].clone(), planes[at + stride].clone());
                    *sum += weight.clone() * value;
                }
                row += 1;
            }
        }
    }
    Ok(())
}

/// A propagator output on a plane kernel (`run_interactions`): its input
/// planes filled from the step's sums (`(sums, width, accumulator)`), the
/// momentum at `momentum` and the kernel's `models` (state components
/// `model_offset + index`), its first `dimension` outputs written to the
/// states at `start`.
fn plane_propagator<T: Clone>(
    plane: &mut dyn PlaneRows<T>,
    layout: StateLayout,
    models: &[usize],
    (sums, sum_width, accumulator): (&[Complex<T>], usize, usize),
    (momentum, start, dimension): (usize, usize, usize),
    states: &mut [Complex<T>],
) -> RusticolResult<()> {
    let points = states.len() / layout.size.max(1);
    if points == 0 {
        return Ok(());
    }
    let inputs = plane.input_len();
    let value = inputs
        .checked_sub(4 + models.len())
        .ok_or_else(|| invalid("propagator kernel inputs"))?;
    require(dimension <= plane.output_len(), "propagator kernel outputs")?;
    let (planes, stride) = plane.planes(points);
    for k in 0..value {
        let (re, im) = planes[2 * k * stride..][..2 * stride].split_at_mut(stride);
        for point in 0..points {
            let sum = &sums[point * sum_width + accumulator + k];
            re[point] = sum.re.clone();
            im[point] = sum.im.clone();
        }
    }
    let state_inputs = (0..4)
        .map(|mu| momentum + mu)
        .chain(models.iter().map(|&index| layout.model_offset + index));
    for (k, at) in state_inputs.enumerate() {
        let (re, im) = planes[2 * (value + k) * stride..][..2 * stride].split_at_mut(stride);
        for (point, state) in states.chunks_exact(layout.size).enumerate() {
            re[point] = state[at].re.clone();
            im[point] = state[at].im.clone();
        }
    }
    let planes = plane
        .run(points)
        .map_err(|e| invalid(format!("propagator kernel: {}", e.message())))?;
    let base = 2 * inputs * stride;
    for (point, state) in states.chunks_exact_mut(layout.size).enumerate() {
        for (output, slot) in state[start..start + dimension].iter_mut().enumerate() {
            let at = base + 2 * output * stride + point;
            *slot = Complex::new(planes[at].clone(), planes[at + stride].clone());
        }
    }
    Ok(())
}

/// A carrier with source wavefunctions: binary64, DoubleFloat or MPFR (the
/// exact carrier has none, wavefunctions being irrational). The external
/// masses are carrier values (binary64 ones in binary64, as the tree
/// runtime's own source records).
pub trait TreeCarrier: Carrier + PartialOrd {
    /// Whether `TreeProgram::bind` refuses a complex pole mass with
    /// 0 < |Im m²| < 1e-150 Re m² (`require_complex_mass_master`): the
    /// carrier whose masters failed there (double-double).
    const COMPLEX_MASS_MASTER_GUARD: bool = false;

    /// The carrier value's exact rational (a finite value).
    fn exact(&self) -> RusticolResult<Rational>;

    fn fill_sources(
        sources: &SchemaSources,
        masses: &BTreeMap<i32, Self>,
        point: &[[Self; 4]],
        state: &mut [Complex<Self>],
    ) -> RusticolResult<()>;

    /// A kernel of `shape` (inputs, outputs) from its saved exact evaluator
    /// and, when stored, its binary64 plane application: binary64 runs the
    /// plane application, the other carriers the exact state.
    fn kernel(
        saved: &[u8],
        plane: Option<&[u8]>,
        shape: (usize, usize),
        zero: &Self,
    ) -> RusticolResult<Kernel<Self>>;
}

impl TreeCarrier for f64 {
    fn exact(&self) -> RusticolResult<Rational> {
        Rational::try_from(*self).map_err(|_| invalid("a carrier value is not finite"))
    }

    fn fill_sources(
        sources: &SchemaSources,
        masses: &BTreeMap<i32, Self>,
        point: &[[Self; 4]],
        state: &mut [Complex<Self>],
    ) -> RusticolResult<()> {
        sources.fill_f64(masses, point, state)
    }

    fn kernel(
        saved: &[u8],
        plane: Option<&[u8]>,
        shape: (usize, usize),
        zero: &Self,
    ) -> RusticolResult<Kernel<Self>> {
        match plane {
            // On an x86-64 CPU without AVX and FMA3, where SymJIT cannot
            // compile (`symjit_plane::require_symjit_host_cpu`), binary64
            // runs the exact states (under the library key, as the extended
            // carriers do).
            Some(bytes) if crate::engine::symjit_host_cpu_supported() => {
                Ok(Kernel::Plane(Box::new(BinaryPlane::load(bytes, shape)?)))
            }
            _ => exact_kernel(saved, zero),
        }
    }
}

impl TreeCarrier for DoubleFloat {
    const COMPLEX_MASS_MASTER_GUARD: bool = true;

    fn exact(&self) -> RusticolResult<Rational> {
        // The unevaluated sum of its two binary64 parts.
        let parts = self.into_inner();
        Ok(&parts.hi().exact()? + &parts.lo().exact()?)
    }

    fn fill_sources(
        sources: &SchemaSources,
        masses: &BTreeMap<i32, Self>,
        point: &[[Self; 4]],
        state: &mut [Complex<Self>],
    ) -> RusticolResult<()> {
        sources.fill_double_float(masses, point, state)
    }

    fn kernel(
        _saved: &[u8],
        _plane: Option<&[u8]>,
        _shape: (usize, usize),
        zero: &Self,
    ) -> RusticolResult<Kernel<Self>> {
        exact_kernel(_saved, zero)
    }
}

impl TreeCarrier for Float {
    fn exact(&self) -> RusticolResult<Rational> {
        if !self.is_finite() {
            return Err(invalid("a carrier value is not finite"));
        }
        Ok(self.to_rational())
    }

    fn fill_sources(
        sources: &SchemaSources,
        masses: &BTreeMap<i32, Self>,
        point: &[[Self; 4]],
        state: &mut [Complex<Self>],
    ) -> RusticolResult<()> {
        sources.fill_float(masses, point, state)
    }

    fn kernel(
        _saved: &[u8],
        _plane: Option<&[u8]>,
        _shape: (usize, usize),
        zero: &Self,
    ) -> RusticolResult<Kernel<Self>> {
        exact_kernel(_saved, zero)
    }
}

#[path = "trees_rows.rs"]
mod rows;

#[cfg(test)]
#[path = "trees_tests.rs"]
mod tests;
