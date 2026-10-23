// SPDX-License-Identifier: 0BSD

//! Native per-point inputs of the v2 executor (WP3.1 I3/I4,
//! `workspace/reports/engine-v2/wp31/PLAN.md`): the block rows of the
//! loop-request tree program and the Born amplitude components of its root
//! closure, without Python per point.
//!
//! **Trees.** The slice's tree program (`loops/v2/assemble.TreeRows`: the
//! stage program of the blocks DAG) is replayed on one complex state per
//! point: the external wavefunctions (the tree runtime's own source records
//! and formulas, `engine::SchemaSources`), the all-outgoing momenta of every
//! momentum slot and the model parameters fill the state, then each stage
//! gathers its inputs, runs its saved exact Symbolica evaluator in the
//! carrier and scatters its outputs. A block row is a signed state slice
//! (its Wick sign, `loops/v2/blocks.Block.wick_sign`).
//!
//! **Born.** The Born is the recursion's last step: every root-closure term
//! (`assemble._born_terms`) closes an ordinary vertex holding leg 1 on the
//! block rows of its attached blocks with a plan kernel, times its sign, its
//! coupling per N_c grade and the exact colour weights of its literal
//! tensors, accumulated per Born helicity row into the components
//! `A0[b, Born order, grade]`. The plan's Born components (the seeds' inputs)
//! are a subset. With a colour metric (full colour), `|A0|²` per squared
//! Born order is formed as well.
//!
//! Every value derives from the stored program; the per-batch inputs are the
//! parameter-only values (model parameters, external masses, couplings and
//! the plan's kernel constants).

use super::dress::{Carrier, cast, parse_rational};
use super::plan::LoopProgramV4;
use crate::engine::SchemaSources;
use crate::{RusticolError, RusticolResult};
use serde::Deserialize;
use std::collections::BTreeMap;
use symbolica::domains::float::{Complex, DoubleFloat, Float, Real};
use symbolica::domains::rational::Rational;
use symbolica::evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator};

/// The tree-program schema this executor runs.
pub const TREE_SCHEMA: u32 = 1;

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
    stages: Vec<StageFile>,
    /// (state component start, dimension, Wick sign) per plan block row.
    block_rows: Vec<(usize, usize, i32)>,
    born: BornFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageFile {
    parameters: usize,
    /// (parameter index, state component).
    inputs: Vec<(usize, usize)>,
    /// (output start, length, state component start).
    outputs: Vec<(usize, usize, usize)>,
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
    /// The exact colour metric <b|b'> at the request's accuracy, sparse.
    gram: Option<Vec<(usize, usize, String, String)>>,
    average: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TermFile {
    kernel: usize,
    /// Per kernel input: ("current", slot, component) or ("momentum", slot, mu).
    inputs: Vec<(String, usize, usize)>,
    /// Closure block of each slot.
    blocks: Vec<usize>,
    /// Leg mask of each slot.
    masks: Vec<u64>,
    sign: i32,
    /// (coupling, component, weight re, weight im, sqrt(2) power p): the
    /// term adds coupling * weight * 2^(-p/2) * value to the component.
    contributions: Vec<(usize, usize, String, String, i32)>,
}

// ------------------------------------------------------------ program form

#[derive(Clone, Copy, Debug)]
enum TermInput {
    Current { slot: usize, component: usize },
    Momentum { slot: usize, mu: usize },
}

struct Stage {
    exact: ExpressionEvaluator<Complex<Rational>>,
    parameters: usize,
    inputs: Vec<(usize, usize)>,
    outputs: Vec<(usize, usize, usize)>,
}

struct Term {
    kernel: usize,
    inputs: Vec<TermInput>,
    blocks: Vec<usize>,
    masks: Vec<u64>,
    sign: i32,
    contributions: Vec<(usize, usize, [Rational; 2], i32)>,
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
    stages: Vec<Stage>,
    block_rows: Vec<(usize, usize, i32)>,
    rows: usize,
    components: usize,
    plan_components: Vec<Option<usize>>,
    blocks: Vec<Vec<Option<usize>>>,
    couplings: usize,
    terms: Vec<Term>,
    amplitudes: Vec<Vec<(usize, Rational)>>,
    orders: usize,
    squared: Vec<Vec<(usize, usize)>>,
    gram: Option<Vec<(usize, usize, [Rational; 2])>>,
    average: Rational,
}

/// The parameter-only inputs of one batch.
pub struct TreeBatch<T> {
    /// Per point the model parameter values in the schema's order (real;
    /// the renormalisation scale may change per point).
    pub model: Vec<Vec<Rational>>,
    /// Exact masses of massive external particles by PDG code.
    pub masses: BTreeMap<i32, Rational>,
    /// Couplings per (name, N_c grade) slot.
    pub couplings: Vec<Complex<T>>,
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
}

/// The carrier form of a program: its stage evaluators and exact weights.
pub struct TreeRuntime<T> {
    evaluators: Vec<BatchExpressionEvaluator<Complex<T>>>,
    weights: Vec<Vec<Complex<T>>>,
    factors: Vec<Vec<T>>,
    gram: Option<Vec<Complex<T>>>,
    average: T,
}

impl TreeProgram {
    /// Load a program exported by `loops/v2/native_trees.py` for `plan`.
    /// `stages` holds each stage's saved exact evaluator.
    pub fn load(json: &[u8], stages: &[Vec<u8>], plan: &LoopProgramV4) -> RusticolResult<Self> {
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
        require(stages.len() == file.stages.len(), "stage evaluator count")?;
        for &(start, _mask) in &file.momentum_slots {
            require(
                start + 4 <= file.state_size,
                "momentum slot outside the state",
            )?;
        }
        let mut loaded = Vec::with_capacity(stages.len());
        for (bytes, stage) in stages.iter().zip(file.stages) {
            let exact = crate::engine::decode_exact_evaluator_state(bytes)?;
            for &(parameter, component) in &stage.inputs {
                require(
                    parameter < stage.parameters && component < file.state_size,
                    "stage input",
                )?;
            }
            for &(start, length, component) in &stage.outputs {
                require(component + length <= file.state_size, "stage output")?;
                let _ = start;
            }
            loaded.push(Stage {
                exact,
                parameters: stage.parameters,
                inputs: stage.inputs,
                outputs: stage.outputs,
            });
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
                block.iter().flatten().all(|&start| start < file.state_size),
                "closure block row outside the state",
            )?;
        }
        let mut terms = Vec::with_capacity(born.terms.len());
        for term in born.terms {
            let kernel = plan
                .kernels
                .get(term.kernel)
                .ok_or_else(|| invalid("Born term kernel index"))?;
            require(
                kernel.outputs.len() == 1 && kernel.inputs == term.inputs.len(),
                "a Born closure kernel has one output and the term's inputs",
            )?;
            require(
                term.blocks.len() == term.masks.len()
                    && term.blocks.iter().all(|&block| block < born.blocks.len())
                    && term.sign.abs() == 1,
                "Born term blocks",
            )?;
            let inputs = term
                .inputs
                .iter()
                .map(|(kind, slot, component)| {
                    require(*slot < term.blocks.len(), "Born term input slot")?;
                    match kind.as_str() {
                        "current" => Ok(TermInput::Current {
                            slot: *slot,
                            component: *component,
                        }),
                        "momentum" if *component < 4 => Ok(TermInput::Momentum {
                            slot: *slot,
                            mu: *component,
                        }),
                        _ => Err(invalid(format!("Born term input {kind:?}"))),
                    }
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let contributions = term
                .contributions
                .iter()
                .map(|(coupling, component, re, im, power)| {
                    require(
                        *coupling < born.couplings && *component < born.components,
                        "Born term contribution",
                    )?;
                    Ok((*coupling, *component, [exact(re)?, exact(im)?], *power))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            terms.push(Term {
                kernel: term.kernel,
                inputs,
                blocks: term.blocks,
                masks: term.masks,
                sign: term.sign,
                contributions,
            });
        }
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
        let gram = born
            .gram
            .map(|entries| {
                entries
                    .iter()
                    .map(|(left, right, re, im)| {
                        require(*left < basis && *right < basis, "colour metric entry")?;
                        Ok((*left, *right, [exact(re)?, exact(im)?]))
                    })
                    .collect::<RusticolResult<Vec<_>>>()
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
            stages: loaded,
            block_rows: file.block_rows,
            rows: born.rows,
            components: born.components,
            plan_components: born.plan_components,
            blocks: born.blocks,
            couplings: born.couplings,
            terms,
            amplitudes,
            orders: born.orders,
            squared: born.squared,
            gram,
            average: exact(&born.average)?,
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

    /// The program in carrier `T` (stage evaluators and exact weights cast).
    pub fn runtime<T: Carrier + Real>(&self, zero: &T) -> TreeRuntime<T>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let complex = |re: &Rational, im: &Rational| Complex::new(cast(zero, re), cast(zero, im));
        let two = cast(zero, &Rational::from(2));
        let root = two.sqrt();
        let evaluators = self
            .stages
            .iter()
            .map(|stage| {
                stage
                    .exact
                    .clone()
                    .map_coeff_with_prec(&|r| complex(&r.re, &r.im), zero.get_precision())
                    .into_batch_evaluator()
            })
            .collect();
        let weights = self
            .terms
            .iter()
            .map(|term| {
                term.contributions
                    .iter()
                    .map(|(_c, _j, [re, im], power)| {
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
                    .collect()
            })
            .collect();
        let factors = self
            .amplitudes
            .iter()
            .map(|row| row.iter().map(|(_j, f)| cast(zero, f)).collect())
            .collect();
        let gram = self.gram.as_ref().map(|entries| {
            entries
                .iter()
                .map(|(_l, _r, [re, im])| complex(re, im))
                .collect()
        });
        TreeRuntime {
            evaluators,
            weights,
            factors,
            gram,
            average: cast(zero, &self.average),
        }
    }

    /// The native inputs of a batch: `momenta` are the plan's all-incoming
    /// momenta per point (initial legs as given, final legs negated) and
    /// `constants` the plan's kernel constants.
    pub fn inputs<T: TreeCarrier>(
        &self,
        runtime: &mut TreeRuntime<T>,
        plan: &LoopProgramV4,
        batch: &TreeBatch<T>,
        momenta: &[Vec<Rational>],
        constants: &[Complex<T>],
        zero: &T,
    ) -> RusticolResult<TreeInputs<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
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
        let mut states = vec![czero.clone(); points * self.state_size];
        for ((point, state), model) in incoming
            .iter()
            .zip(states.chunks_exact_mut(self.state_size))
            .zip(&batch.model)
        {
            let physical: Vec<[T; 4]> = point
                .iter()
                .zip(&self.initial)
                .map(|(p, &initial)| {
                    if initial {
                        p.clone()
                    } else {
                        p.clone().map(|x| -x)
                    }
                })
                .collect();
            T::fill_sources(&self.sources, &masses, &physical, state)?;
            for &(start, mask) in &self.momentum_slots {
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
                state[self.model_offset + index] = Complex::new(cast(zero, value), zero.clone());
            }
        }
        for (stage, evaluator) in self.stages.iter().zip(&mut runtime.evaluators) {
            let mut parameters = vec![czero.clone(); points * stage.parameters];
            for (state, row) in states
                .chunks_exact(self.state_size)
                .zip(parameters.chunks_exact_mut(stage.parameters))
            {
                for &(parameter, component) in &stage.inputs {
                    row[parameter] = state[component].clone();
                }
            }
            let width = evaluator.get_output_len();
            let mut outputs = vec![czero.clone(); points * width];
            evaluator
                .evaluate_batch(&parameters, &mut outputs, points)
                .map_err(|e| invalid(format!("stage evaluation: {e}")))?;
            for (state, row) in states
                .chunks_exact_mut(self.state_size)
                .zip(outputs.chunks_exact(width))
            {
                for &(start, length, component) in &stage.outputs {
                    state[component..component + length]
                        .clone_from_slice(&row[start..start + length]);
                }
            }
        }
        let mut out = TreeInputs {
            block_rows: Vec::with_capacity(points),
            born: Vec::with_capacity(points),
            amplitudes: Vec::with_capacity(points),
            born_squared: runtime.gram.as_ref().map(|_| Vec::with_capacity(points)),
        };
        for (point, state) in incoming.iter().zip(states.chunks_exact(self.state_size)) {
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
            let amplitudes = self.born(runtime, plan, batch, point, state, constants, zero)?;
            let mut born = Vec::with_capacity(self.rows * self.plan_components.len());
            for row in amplitudes.chunks_exact(self.components) {
                for component in &self.plan_components {
                    born.push(component.map_or_else(|| czero.clone(), |j| row[j].clone()));
                }
            }
            out.born.push(born);
            if let (Some(gram), Some(squared)) = (&runtime.gram, &mut out.born_squared) {
                squared.push(self.squared(runtime, gram, &amplitudes, zero));
            }
            out.amplitudes.push(amplitudes);
        }
        Ok(out)
    }

    /// `A0` of one point: per Born row every component.
    #[allow(clippy::too_many_arguments)]
    fn born<T: Carrier>(
        &self,
        runtime: &TreeRuntime<T>,
        plan: &LoopProgramV4,
        batch: &TreeBatch<T>,
        incoming: &[[T; 4]],
        state: &[Complex<T>],
        constants: &[Complex<T>],
        zero: &T,
    ) -> RusticolResult<Vec<Complex<T>>> {
        let czero = Complex::new(zero.clone(), zero.clone());
        let mut out = vec![czero.clone(); self.rows * self.components];
        let mut x = Vec::new();
        for (term, weights) in self.terms.iter().zip(&runtime.weights) {
            let kernel = &plan.kernels[term.kernel];
            let momenta: Vec<[T; 4]> = term
                .masks
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
                .collect();
            for row in 0..self.rows {
                let mut starts = Vec::with_capacity(term.blocks.len());
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
                for input in &term.inputs {
                    x.push(match *input {
                        TermInput::Current { slot, component } => {
                            state[starts[slot] + component].clone()
                        }
                        TermInput::Momentum { slot, mu } => {
                            Complex::new(momenta[slot][mu].clone(), zero.clone())
                        }
                    });
                }
                let mut value = czero.clone();
                for term_entry in &kernel.outputs[0] {
                    let mut product = constants[term_entry.constant].clone();
                    for &(input, exponent) in &term_entry.monomial {
                        for _ in 0..exponent {
                            product *= x[input].clone();
                        }
                    }
                    value += product;
                }
                if term.sign < 0 {
                    value = -value;
                }
                let target = &mut out[row * self.components..][..self.components];
                for ((coupling, component, _, _), weight) in term.contributions.iter().zip(weights)
                {
                    target[*component] +=
                        batch.couplings[*coupling].clone() * weight.clone() * value.clone();
                }
            }
        }
        Ok(out)
    }

    /// `|A0|²` per squared Born order of one point (divided by the average).
    fn squared<T: Carrier>(
        &self,
        runtime: &TreeRuntime<T>,
        gram: &[Complex<T>],
        amplitudes: &[Complex<T>],
        zero: &T,
    ) -> Vec<T> {
        let czero = Complex::new(zero.clone(), zero.clone());
        let entries = self.gram.as_ref().expect("a runtime gram has entries");
        let mut out = vec![zero.clone(); self.squared.len()];
        for row in amplitudes.chunks_exact(self.components) {
            // a[b * orders + o] = sum_g A0[(b, o, g)] N_c^g.
            let a: Vec<Complex<T>> = self
                .amplitudes
                .iter()
                .zip(&runtime.factors)
                .map(|(terms, factors)| {
                    let mut total = czero.clone();
                    for ((component, _), factor) in terms.iter().zip(factors) {
                        let value = &row[*component];
                        total += Complex::new(
                            value.re.clone() * factor.clone(),
                            value.im.clone() * factor.clone(),
                        );
                    }
                    total
                })
                .collect();
            for (target, pairs) in out.iter_mut().zip(&self.squared) {
                let mut total = czero.clone();
                for &(left, right) in pairs {
                    for ((b, b2, _), metric) in entries.iter().zip(gram) {
                        let bra = &a[b * self.orders + left];
                        let ket = &a[b2 * self.orders + right];
                        total += bra.conj() * metric.clone() * ket.clone();
                    }
                }
                *target += total.re;
            }
        }
        out.into_iter()
            .map(|value| value / runtime.average.clone())
            .collect()
    }
}

/// A carrier with source wavefunctions: binary64, DoubleFloat or MPFR (the
/// exact carrier has none, wavefunctions being irrational). The external
/// masses are carrier values (binary64 ones in binary64, as the tree
/// runtime's own source records).
pub trait TreeCarrier: Carrier {
    fn fill_sources(
        sources: &SchemaSources,
        masses: &BTreeMap<i32, Self>,
        point: &[[Self; 4]],
        state: &mut [Complex<Self>],
    ) -> RusticolResult<()>;
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
}
