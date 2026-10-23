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
use symbolica::evaluate::{BatchExpressionEvaluator, EvaluationDomain};

/// The tree-program schema this executor runs.
pub const TREE_SCHEMA: u32 = 2;

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
    /// The exact full-colour metric <b|b'> at N_c = 3, sparse.
    gram: Option<Vec<(usize, usize, String, String)>>,
    /// Leading and next-to-leading colour: per squared Born order the metric
    /// over Born components, (squared order, component, component, re, im),
    /// each cell's N_c powers kept by the projection (sparse; no `gram`).
    #[serde(default)]
    metric: Option<Vec<(usize, usize, usize, String, String)>>,
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
}

/// One runtime stage of the interaction program.
struct Step {
    /// Per vertex kernel the interactions it evaluates.
    groups: Vec<(usize, Vec<Edge>)>,
    /// Accumulator components per point.
    width: usize,
    /// (accumulator offset, propagator, momentum start, state start,
    /// dimension).
    outputs: Vec<(usize, Option<usize>, usize, usize, usize)>,
}

struct Interactions {
    /// Saved exact evaluators: vertex kernels, then propagator kernels.
    saved: Vec<Vec<u8>>,
    steps: Vec<Step>,
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
    interactions: Interactions,
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
    metric: Option<Vec<(usize, usize, usize, [Rational; 2])>>,
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

/// The carrier form of a program: its kernel evaluators and exact weights.
pub struct TreeRuntime<T> {
    /// The interaction program's kernels and, per step and kernel group,
    /// its interactions' weights.
    kernels: Vec<BatchExpressionEvaluator<Complex<T>>>,
    edge_weights: Vec<Vec<Vec<Complex<T>>>>,
    weights: Vec<Vec<Complex<T>>>,
    factors: Vec<Vec<T>>,
    gram: Option<Vec<Complex<T>>>,
    metric: Option<Vec<Complex<T>>>,
    average: T,
}

/// Validate and group an interaction program (`InteractionFile`); `saved`
/// holds its kernels' evaluators, vertex kernels first.
fn load_interactions(
    program: InteractionFile,
    saved: &[Vec<u8>],
    state_size: usize,
    model_count: usize,
) -> RusticolResult<Interactions> {
    let kernels = program.kernels.len();
    require(
        saved.len() == kernels + program.propagators.len(),
        "interaction kernel evaluator count",
    )?;
    for bytes in saved {
        drop(crate::engine::decode_exact_evaluator_state(bytes)?);
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
        let mut groups: BTreeMap<usize, Vec<Edge>> = BTreeMap::new();
        for (kernel, left, ld, right, rd, pl, pr, accumulator, re, im) in step.interactions {
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
            outputs.push((offsets[accumulator], propagator, momentum, start, dimension));
        }
        steps.push(Step {
            groups: groups.into_iter().collect(),
            width,
            outputs,
        });
    }
    Ok(Interactions {
        saved: saved.to_vec(),
        steps,
    })
}

impl TreeProgram {
    /// Load a program exported by `loops/v2/native_trees.py` for `plan`.
    /// `kernels` holds the interaction program's saved exact evaluators.
    pub fn load(json: &[u8], kernels: &[Vec<u8>], plan: &LoopProgramV4) -> RusticolResult<Self> {
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
        let interactions = load_interactions(
            file.interactions,
            kernels,
            file.state_size,
            file.model_count,
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
        require(
            gram.is_none() || metric.is_none(),
            "a Born metric is either full colour or per component",
        )?;
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
            amplitudes,
            orders: born.orders,
            squared: born.squared,
            gram,
            metric,
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

    /// The program in carrier `T` (kernel evaluators and exact weights
    /// cast).
    pub fn runtime<T: Carrier + Real>(&self, zero: &T) -> RusticolResult<TreeRuntime<T>>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        let complex = |re: &Rational, im: &Rational| Complex::new(cast(zero, re), cast(zero, im));
        let two = cast(zero, &Rational::from(2));
        let root = two.sqrt();
        let program = &self.interactions;
        let kernels = program
            .saved
            .iter()
            .map(|saved| {
                Ok(crate::engine::decode_exact_evaluator_state(saved)?
                    .map_coeff_with_prec(&|r| complex(&r.re, &r.im), zero.get_precision())
                    .into_batch_evaluator())
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let edge_weights = program
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
        let metric = self.metric.as_ref().map(|entries| {
            entries
                .iter()
                .map(|(_s, _l, _r, [re, im])| complex(re, im))
                .collect()
        });
        Ok(TreeRuntime {
            kernels,
            edge_weights,
            weights,
            factors,
            gram,
            metric,
            average: cast(zero, &self.average),
        })
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
        self.run_interactions(&self.interactions, runtime, &mut states, points, &czero)?;
        let mut out = TreeInputs {
            block_rows: Vec::with_capacity(points),
            born: Vec::with_capacity(points),
            amplitudes: Vec::with_capacity(points),
            born_squared: (runtime.gram.is_some() || runtime.metric.is_some())
                .then(|| Vec::with_capacity(points)),
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
            if let Some(squared) = &mut out.born_squared {
                squared.push(match &runtime.metric {
                    Some(metric) => self.squared_by_component(runtime, metric, &amplitudes, zero),
                    None => self.squared(
                        runtime,
                        runtime.gram.as_ref().expect("a runtime Born metric"),
                        &amplitudes,
                        zero,
                    ),
                });
            }
            out.amplitudes.push(amplitudes);
        }
        Ok(out)
    }

    /// The interaction program on the states of a batch: per step, each
    /// vertex kernel runs batched over its interactions and the points,
    /// weighted sums accumulate per output current, then each output slot
    /// copies its sum or applies its propagator kernel.
    fn run_interactions<T: Carrier>(
        &self,
        program: &Interactions,
        runtime: &mut TreeRuntime<T>,
        states: &mut [Complex<T>],
        points: usize,
        czero: &Complex<T>,
    ) -> RusticolResult<()>
    where
        Complex<T>: Real + EvaluationDomain,
    {
        // Rows (interaction x point) per kernel call.
        const ROWS: usize = 4096;
        let model = self.model_offset..self.model_offset + self.model_count;
        let mut parameters = Vec::new();
        let mut outputs = Vec::new();
        for (step, weights) in program.steps.iter().zip(&runtime.edge_weights) {
            let mut sums = vec![czero.clone(); points * step.width];
            for ((kernel, edges), weights) in step.groups.iter().zip(weights) {
                let evaluator = &mut runtime.kernels[*kernel];
                let width = evaluator.get_output_len();
                let per_call = (ROWS / points.max(1)).max(1);
                for (chunk, chunk_weights) in edges.chunks(per_call).zip(weights.chunks(per_call))
                {
                    let rows = chunk.len() * points;
                    parameters.clear();
                    for edge in chunk {
                        for state in states.chunks_exact(self.state_size) {
                            parameters.extend_from_slice(
                                &state[edge.left..edge.left + edge.left_dimension],
                            );
                            parameters.extend_from_slice(
                                &state[edge.right..edge.right + edge.right_dimension],
                            );
                            parameters.extend_from_slice(
                                &state[edge.left_momentum..edge.left_momentum + 4],
                            );
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
                    for (edge, weight) in chunk.iter().zip(chunk_weights) {
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
            for &(accumulator, propagator, momentum, start, dimension) in &step.outputs {
                let Some(kernel) = propagator else {
                    for (point, state) in states.chunks_exact_mut(self.state_size).enumerate() {
                        state[start..start + dimension].clone_from_slice(
                            &sums[point * step.width + accumulator..][..dimension],
                        );
                    }
                    continue;
                };
                let evaluator = &mut runtime.kernels[kernel];
                let width = evaluator.get_output_len();
                let inputs = evaluator.get_input_len();
                let value = inputs - 4 - self.model_count;
                parameters.clear();
                for (point, state) in states.chunks_exact(self.state_size).enumerate() {
                    parameters
                        .extend_from_slice(&sums[point * step.width + accumulator..][..value]);
                    parameters.extend_from_slice(&state[momentum..momentum + 4]);
                    parameters.extend_from_slice(&state[model.clone()]);
                }
                outputs.clear();
                outputs.resize(points * width, czero.clone());
                evaluator
                    .evaluate_batch(&parameters, &mut outputs, points)
                    .map_err(|e| invalid(format!("propagator kernel: {e}")))?;
                for (state, row) in states
                    .chunks_exact_mut(self.state_size)
                    .zip(outputs.chunks_exact(width))
                {
                    state[start..start + dimension].clone_from_slice(&row[..dimension]);
                }
            }
        }
        Ok(())
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

    /// `|A0|²` per squared Born order of one point (divided by the average)
    /// with a per-component metric (leading and next-to-leading colour).
    fn squared_by_component<T: Carrier>(
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
