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
//! carrier, under pyAmpliCol's Symbolica library key (`crate::licence`).

use super::dress::{Carrier, cast, parse_rational};
use super::interference::{Entry, Interference, InterferenceRuntime, Ket};
use super::plan::LoopProgramV4;
use crate::engine::{PlaneDescriptor, SchemaSources, SymjitPlaneKernel, SymjitPlaneLayout};
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use symbolica::domains::float::{Complex, DoubleFloat, Float, Real};
use symbolica::domains::rational::Rational;
use symbolica::evaluate::{BatchExpressionEvaluator, EvaluationDomain};

/// The tree-program schema this executor runs.
pub const TREE_SCHEMA: u32 = 2;

/// Generation: the binary64 SymJIT plane application of a saved exact
/// evaluator of a tree program (a kernel or the parameter program), with its
/// input and output counts. The evaluator is the one binary64 would run
/// without it (its coefficients, registered constants included, cast as
/// [`TreeProgram::runtime`] casts them), compiled at `opt_level` into
/// storage-v4 IR (`evaluator::symjit_plane`).
#[cfg(feature = "python-generation-bridge")]
pub fn compile_plane_application(
    saved: &[u8],
    opt_level: u8,
    compress: bool,
) -> RusticolResult<(Vec<u8>, usize, usize)> {
    let _unlock = crate::licence::unlock();
    let zero = 0.0_f64;
    let complex = |re: &Rational, im: &Rational| Complex::new(cast(&zero, re), cast(&zero, im));
    let evaluator = crate::engine::decode_exact_evaluator_state(saved)?.map_coeff_with_prec(
        &|r| complex(&r.re, &r.im),
        symbolica::domains::float::FloatLike::get_precision(&zero),
    );
    let (plane, inputs, outputs) =
        crate::engine::compile_symbolica_evaluator_to_plane_application_bytes(
            &evaluator, opt_level, compress,
        )?;
    verify_plane(&plane, evaluator.into_batch_evaluator(), (inputs, outputs))?;
    Ok((plane, inputs, outputs))
}

/// Refuse a plane application that disagrees with its evaluator at two fixed
/// real points (inputs in (0.5, 1.5)): a translation error (a constant or a
/// function read wrongly) changes an output at O(1), while the two
/// evaluations' rounding differs by orders of magnitude less (the criterion
/// of the tree engine's exporter, `symbolica_helpers._verify_symjit_plane_
/// constants`).
#[cfg(feature = "python-generation-bridge")]
fn verify_plane(
    plane: &[u8],
    mut evaluator: BatchExpressionEvaluator<Complex<f64>>,
    shape: (usize, usize),
) -> RusticolResult<()> {
    const ROWS: usize = 2;
    let (inputs, outputs) = shape;
    // A low-discrepancy sequence: deterministic, distinct per input.
    let values: Vec<Complex<f64>> = (0..ROWS * inputs)
        .map(|k| {
            Complex::new(
                0.5 + (0.618_033_988_749_894_8 * (k + 1) as f64).fract(),
                0.0,
            )
        })
        .collect();
    let mut expected = vec![Complex::new(0.0, 0.0); ROWS * outputs];
    evaluator
        .evaluate_batch(&values, &mut expected, ROWS)
        .map_err(|e| invalid(format!("plane check: {e}")))?;
    let got = evaluate_plane_application(plane, shape, &values, ROWS)?;
    let parts = |values: &[Complex<f64>]| -> Vec<f64> {
        values.iter().flat_map(|z| [z.re, z.im]).collect()
    };
    let (expected, got) = (parts(&expected), parts(&got));
    let scale = expected
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    for (index, (&e, &g)) in expected.iter().zip(&got).enumerate() {
        let agree = e.to_bits() == g.to_bits()
            || (e.is_nan() && g.is_nan())
            || (e - g).abs() <= 1e-6 * e.abs()
            || (e - g).abs() <= 1e-12 * scale;
        if !agree {
            return Err(invalid(format!(
                "the SymJIT plane application disagrees with its exact evaluator \
                 (value {index}: {g:e} against {e:e}); keep this program on the \
                 exact evaluator"
            )));
        }
    }
    Ok(())
}

/// The binary64 rows of a plane application (`compile_plane_application`)
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
    /// Per plan mass: the external index of its mass parameter (squared),
    /// or none (massless).
    plan_masses: Vec<Option<usize>>,
    /// The renormalisation scale's parameter, if the program reads it:
    /// sqrt(mu_R^2) of each point.
    scale: Option<usize>,
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
    interactions: InteractionFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InteractionFile {
    /// (inputs, outputs) per vertex kernel: inputs `[left, right, p_left,
    /// p_right, model]`.
    kernels: Vec<(usize, usize)>,
    /// (inputs, outputs) per propagator kernel: inputs `[value, p, model]`.
    propagators: Vec<(usize, usize)>,
    steps: Vec<StepFile>,
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
    /// Interactions and outputs of currents only epsilon-pole terms read
    /// (indices; a counterterm program's pole slots): finite-only
    /// evaluations skip them.
    #[serde(default)]
    pole_interactions: Vec<usize>,
    #[serde(default)]
    pole_outputs: Vec<usize>,
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
    model_offset: usize,
    model_count: usize,
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
    plan_masses: Vec<Option<usize>>,
    scale: Option<usize>,
}

/// The counterterm closure of a program and its interferences.
struct Counterterms {
    couplings: usize,
    /// Per coupling slot: whether it is the epsilon^0 coefficient.
    finite_slot: Vec<bool>,
    components: usize,
    terms: Vec<Term>,
    interference: Interference,
}

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
    /// The leg masks of the closure terms' slots.
    masks: Vec<u64>,
    /// The closure terms' kernel input layouts.
    layouts: Vec<Vec<TermInput>>,
    /// The closure terms' colour weights and their sqrt(2) powers.
    weights: Vec<([Rational; 2], i32)>,
}

/// The parameter-only inputs of one batch.
pub struct TreeBatch<T> {
    /// Per point the model parameter values in the schema's order (real
    /// carrier values; the renormalisation scale may change per point).
    pub model: Vec<Vec<T>>,
    /// Exact masses of massive external particles by PDG code.
    pub masses: BTreeMap<i32, Rational>,
    /// Couplings per (name, N_c grade) slot.
    pub couplings: Vec<Complex<T>>,
    /// Per point the counterterm and field couplings per (name, epsilon
    /// power, N_c grade) slot.
    pub counterterm_couplings: Vec<Vec<Complex<T>>>,
    /// Per point the counterterm program's model parameter values.
    pub counterterm_model: Vec<Vec<T>>,
}

/// The values a batch's external parameters bind (`TreeProgram::bind`).
pub struct BoundParameters<T> {
    /// The tree program's per-batch inputs.
    pub trees: TreeBatch<T>,
    /// The plan's kernel constants.
    pub constants: Vec<Complex<T>>,
    /// The plan's squared masses per mass id.
    pub masses: Vec<Rational>,
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
    /// The counterterm and external-field outputs (component, sector,
    /// epsilon power), divided by the average (with counterterms).
    pub counterterms: Option<Vec<Vec<T>>>,
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
}

/// One kernel (or the parameter program) in carrier `T`: its exact
/// evaluator in the carrier, or in binary64 its SymJIT plane application.
pub enum Kernel<T> {
    Exact(BatchExpressionEvaluator<Complex<T>>),
    Plane(Box<dyn PlaneRows<T> + Send>),
}

/// Row-major batch evaluation of a plane application (binary64 only).
pub trait PlaneRows<T> {
    fn evaluate(
        &mut self,
        inputs: &[Complex<T>],
        outputs: &mut [Complex<T>],
        rows: usize,
    ) -> RusticolResult<()>;
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
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

/// A binary64 plane application, evaluated on row-major complex rows
/// (transposed into its split-complex planes: inputs re, im, then outputs).
struct BinaryPlane {
    kernel: SymjitPlaneKernel,
    inputs: usize,
    outputs: usize,
    planes: Vec<f64>,
}

impl BinaryPlane {
    fn load(bytes: &[u8], (inputs, outputs): (usize, usize)) -> RusticolResult<Self> {
        let layout = SymjitPlaneLayout::complex(inputs, outputs);
        Ok(Self {
            kernel: SymjitPlaneKernel::load_bytes(bytes, "<loop v2 tree kernel>", layout)?,
            inputs,
            outputs,
            planes: Vec::new(),
        })
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
        let planes = 2 * (self.inputs + self.outputs);
        self.planes.clear();
        self.planes.resize(planes * rows, 0.0);
        for (row, values) in inputs.chunks_exact(self.inputs.max(1)).enumerate() {
            for (input, value) in values.iter().enumerate() {
                self.planes[2 * input * rows + row] = value.re;
                self.planes[(2 * input + 1) * rows + row] = value.im;
            }
        }
        {
            let descriptors = self
                .planes
                .chunks_exact_mut(rows)
                .map(PlaneDescriptor::from_slice)
                .collect();
            let mut table = self.kernel.build_raw_table(descriptors)?;
            self.kernel.execute_table(&mut table, 0, rows)?;
        }
        let base = 2 * self.inputs;
        for (row, values) in outputs.chunks_exact_mut(self.outputs.max(1)).enumerate() {
            for (output, value) in values.iter_mut().enumerate() {
                *value = Complex::new(
                    self.planes[(base + 2 * output) * rows + row],
                    self.planes[(base + 2 * output + 1) * rows + row],
                );
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
}

/// Validate and group an interaction program (`InteractionFile`); `saved`
/// holds its kernels' evaluators, vertex kernels first.
fn load_interactions(
    program: InteractionFile,
    saved: &[Vec<u8>],
    planes: Option<&[Vec<u8>]>,
    state_size: usize,
    model_count: usize,
) -> RusticolResult<Interactions> {
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
    let shapes = program
        .kernels
        .iter()
        .chain(&program.propagators)
        .copied()
        .collect();
    let within = |start: usize, length: usize| start + length <= state_size;
    let mut steps = Vec::with_capacity(program.steps.len());
    for step in program.steps {
        let mut offsets = Vec::with_capacity(step.accumulators.len());
        let mut width = 0;
        for &dimension in &step.accumulators {
            offsets.push(width);
            width += dimension;
        }
        require(
            step.pole_interactions
                .iter()
                .all(|&i| i < step.interactions.len())
                && step.pole_outputs.iter().all(|&i| i < step.outputs.len()),
            "pole interaction or output index",
        )?;
        let pole_interactions: std::collections::BTreeSet<usize> =
            step.pole_interactions.iter().copied().collect();
        let pole_outputs: std::collections::BTreeSet<usize> =
            step.pole_outputs.iter().copied().collect();
        let mut groups: BTreeMap<usize, Vec<Edge>> = BTreeMap::new();
        for (index, (kernel, left, ld, right, rd, pl, pr, accumulator, re, im)) in
            step.interactions.into_iter().enumerate()
        {
            let &(inputs, outputs) = program
                .kernels
                .get(kernel)
                .ok_or_else(|| invalid("interaction kernel index"))?;
            require(
                accumulator < step.accumulators.len()
                    && outputs == step.accumulators[accumulator]
                    && inputs == ld + rd + 8 + model_count
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
                pole: pole_interactions.contains(&index),
            });
        }
        let mut outputs = Vec::with_capacity(step.outputs.len());
        for (index, (accumulator, propagator, momentum, start, dimension)) in
            step.outputs.into_iter().enumerate()
        {
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
                let &(inputs, outputs) = program
                    .propagators
                    .get(index)
                    .ok_or_else(|| invalid("propagator index"))?;
                require(
                    inputs == step.accumulators[accumulator] + 4 + model_count
                        && outputs == dimension
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
                pole_outputs.contains(&index),
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
    require(
        file.couplings
            .iter()
            .chain(&file.counterterm_couplings)
            .chain(&file.finite)
            .chain(&file.constants)
            .chain(&file.positive)
            .all(|&o| output(o))
            && file.masses.iter().all(|&(_, i)| i < inputs)
            && file.plan_masses.iter().flatten().all(|&i| i < inputs)
            && file.scale.is_none_or(|i| i < inputs)
            && file
                .plan_masses
                .iter()
                .flatten()
                .chain(file.masses.iter().map(|(_, i)| i))
                .all(|&i| Some(i) != file.scale),
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
        plan_masses: file.plan_masses,
        scale: file.scale,
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
        cells,
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
        let file: TreeFile = serde_json::from_slice(json).map_err(|e| invalid(e.to_string()))?;
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
                Some(CountertermTrees {
                    layout: StateLayout {
                        size: ct.state_size,
                        model_offset: ct.model_offset,
                        model_count: ct.model_count,
                    },
                    momentum_slots: ct.momentum_slots,
                    sources: SchemaSources::from_value(&ct.source_fill)?,
                    interactions: load_interactions(
                        ct.interactions,
                        counterterm_kernels,
                        counterterm_planes,
                        ct.state_size,
                        ct.model_count,
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
                .map_or(0, |ct| ct.layout.model_count),
            file.born.couplings,
            file.counterterms.as_ref().map_or(0, |ct| ct.couplings),
            plan,
        )?;
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
            sources: SchemaSources::from_value(&file.source_fill)?,
            interactions,
            block_rows: file.block_rows,
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
        // Exact kernels (DoubleFloat, arbitrary precision) under the key.
        let _unlock = crate::licence::unlock();
        let program = &self.parameters;
        let points = scales.len();
        let width = program.names.len();
        require(points > 0, "a batch needs points")?;
        require(values.len() == width, "external parameter count")?;
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
        let shared = |indices: &[usize], what: &str| -> RusticolResult<Vec<Complex<T>>> {
            let first: Vec<Complex<T>> = indices.iter().map(|&o| rows[0][o].clone()).collect();
            for row in &rows[1..] {
                require(
                    indices
                        .iter()
                        .zip(&first)
                        .all(|(&o, value)| row[o] == *value),
                    what,
                )?;
            }
            Ok(first)
        };
        for row in &rows {
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
        let couplings = shared(
            &program.couplings,
            "the points of one batch need equal Born couplings",
        )?;
        let constants = shared(
            &program.constants,
            "the points of one batch need equal kernel constants",
        )?;
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
        Ok(BoundParameters {
            trees: TreeBatch {
                model,
                masses: program
                    .masses
                    .iter()
                    .map(|&(pdg, index)| (pdg, values[index].clone()))
                    .collect(),
                couplings,
                counterterm_couplings,
                counterterm_model,
            },
            constants,
            masses: program
                .plan_masses
                .iter()
                .map(|mass| match mass {
                    Some(index) => &values[*index] * &values[*index],
                    None => Rational::from(0),
                })
                .collect(),
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
        })
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
        // Exact kernels (DoubleFloat, arbitrary precision) under the key.
        let _unlock = crate::licence::unlock();
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
                        .all(|model| model.len() == ct.layout.model_count),
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
            .collect();
        // The tree state of every point.
        let layout = StateLayout {
            size: self.state_size,
            model_offset: self.model_offset,
            model_count: self.model_count,
        };
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
        run_interactions(
            &self.interactions,
            layout,
            &mut runtime.kernels,
            &runtime.edge_weights,
            &mut states,
            points,
            true,
            &czero,
        )?;
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
            counterterms: self
                .counterterms
                .as_ref()
                .map(|_| Vec::with_capacity(points)),
        };
        for (index, (point, state)) in incoming
            .iter()
            .zip(states.chunks_exact(self.state_size))
            .enumerate()
        {
            let ct_state =
                &counterterm_states[index * counterterm_size..(index + 1) * counterterm_size];
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
            out.block_rows.push(rows);
            let momenta = self.mask_momenta(point, zero);
            let amplitudes = self.close(
                &self.terms,
                &runtime.weights,
                &batch.couplings,
                None,
                self.components,
                plan,
                &momenta,
                (state, ct_state),
                constants,
                zero,
            );
            if let (Some(ct), Some(interference), Some(out)) = (
                &self.counterterms,
                &runtime.interference,
                &mut out.counterterms,
            ) {
                let couplings = &batch.counterterm_couplings[index];
                let ket = self.close(
                    &ct.terms,
                    &runtime.weights,
                    couplings,
                    (!poles).then_some(ct.finite_slot.as_slice()),
                    ct.components,
                    plan,
                    &momenta,
                    (state, ct_state),
                    constants,
                    zero,
                );
                let values = ct.interference.evaluate(
                    interference,
                    &amplitudes,
                    self.components,
                    &ket,
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
                squared.push(self.squared(runtime, metric, &amplitudes, zero));
            }
            out.amplitudes.push(amplitudes);
        }
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

    /// The root closure of one point: per Born row, every closure term's
    /// kernel on its blocks' rows (in the trees' or the counterterm
    /// program's state) and momenta (`momenta`: per leg mask),
    /// times its sign and per contribution its coupling and colour weight,
    /// accumulated into `width` components. `active`: the coupling slots
    /// to evaluate (none: all). Rows run in parallel.
    #[allow(clippy::too_many_arguments)]
    fn close<T: Carrier>(
        &self,
        terms: &[Term],
        weights: &[Complex<T>],
        couplings: &[Complex<T>],
        active: Option<&[bool]>,
        width: usize,
        plan: &LoopProgramV4,
        momenta: &[[T; 4]],
        (state, ct_state): (&[Complex<T>], &[Complex<T>]),
        constants: &[Complex<T>],
        zero: &T,
    ) -> Vec<Complex<T>> {
        let czero = Complex::new(zero.clone(), zero.clone());
        let mut out = vec![czero.clone(); self.rows * width];
        // Closure block starts index the trees' state, then the counterterm
        // program's.
        let read = |index: usize| {
            if index < state.len() {
                state[index].clone()
            } else {
                ct_state[index - state.len()].clone()
            }
        };
        if width == 0 {
            return out;
        }
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, target)| {
                let mut x = Vec::new();
                let mut starts = Vec::new();
                for term in terms {
                    if active
                        .is_some_and(|active| term.contributions.iter().all(|(c, ..)| !active[*c]))
                    {
                        continue;
                    }
                    starts.clear();
                    for &block in &term.blocks {
                        match self.blocks[block][row] {
                            Some(start) => starts.push(start),
                            None => break,
                        }
                    }
                    if starts.len() != term.blocks.len() {
                        continue;
                    }
                    x.clear();
                    for input in &self.layouts[term.layout] {
                        x.push(match *input {
                            TermInput::Current { slot, component } => {
                                read(starts[slot] + component)
                            }
                            TermInput::Momentum { slot, mu } => {
                                Complex::new(momenta[term.masks[slot]][mu].clone(), zero.clone())
                            }
                        });
                    }
                    let mut value = czero.clone();
                    for entry in &plan.kernels[term.kernel].outputs[0] {
                        let mut product = constants[entry.constant].clone();
                        for &(input, exponent) in &entry.monomial {
                            for _ in 0..exponent {
                                product *= x[input].clone();
                            }
                        }
                        value += product;
                    }
                    if term.sign < 0 {
                        value = -value;
                    }
                    for &(coupling, component, weight) in &term.contributions {
                        if active.is_some_and(|active| !active[coupling]) {
                            continue;
                        }
                        target[component] +=
                            couplings[coupling].clone() * weights[weight].clone() * value.clone();
                    }
                }
            });
        out
    }

    /// `|A0|²` per squared Born order of one point (divided by the average):
    /// the metric over pairs of Born components.
    fn squared<T: Carrier>(
        &self,
        runtime: &TreeRuntime<T>,
        metric: &[Complex<T>],
        amplitudes: &[Complex<T>],
        zero: &T,
    ) -> Vec<T> {
        let entries = self.metric.as_ref().expect("a runtime metric has entries");
        let mut out = vec![zero.clone(); self.squared.len()];
        for row in amplitudes.chunks_exact(self.components) {
            for ((squared, left, right, _), value) in entries.iter().zip(metric) {
                out[*squared] += (row[*left].conj() * value.clone() * row[*right].clone()).re;
            }
        }
        out.into_iter()
            .map(|value| value / runtime.average.clone())
            .collect()
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
    let model = layout.model_offset..layout.model_offset + layout.model_count;
    let mut parameters = Vec::new();
    let mut outputs = Vec::new();
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
                        parameters.extend_from_slice(&state[model.clone()]);
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
            let width = evaluator.output_len();
            let inputs = evaluator.input_len();
            let value = inputs - 4 - layout.model_count;
            parameters.clear();
            for (point, state) in states.chunks_exact(layout.size).enumerate() {
                parameters.extend_from_slice(&sums[point * step.width + accumulator..][..value]);
                parameters.extend_from_slice(&state[momentum..momentum + 4]);
                parameters.extend_from_slice(&state[model.clone()]);
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

/// A carrier with source wavefunctions: binary64, DoubleFloat or MPFR (the
/// exact carrier has none, wavefunctions being irrational). The external
/// masses are carrier values (binary64 ones in binary64, as the tree
/// runtime's own source records).
pub trait TreeCarrier: Carrier + PartialOrd {
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
            Some(bytes) => Ok(Kernel::Plane(Box::new(BinaryPlane::load(bytes, shape)?))),
            None => exact_kernel(saved, zero),
        }
    }
}

impl TreeCarrier for DoubleFloat {
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
