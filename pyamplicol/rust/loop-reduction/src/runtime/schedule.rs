// SPDX-License-Identifier: 0BSD

//! One tiled local-coefficient executor shared by native callers and Python.
use super::coefficients::batch_evaluator;
use super::*;
use std::collections::HashMap;
use symbolica::{
    domains::{
        float::{DoubleFloat, Float, Real},
        rational::Rational,
    },
    evaluate::{
        BatchEvaluator, BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator,
        JITCompilationSettings, JITCompiledEvaluator,
    },
};

trait ScheduledKernel<T: Carrier> {
    fn evaluate(
        &mut self,
        parameters: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<(), String>;
}
impl<T: Carrier> ScheduledKernel<T> for BatchExpressionEvaluator<Complex<T>>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn evaluate(
        &mut self,
        parameters: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<(), String> {
        self.evaluate_batch(parameters, output, rows)
            .map_err(|failure| failure.to_string())
    }
}
impl ScheduledKernel<f64> for JITCompiledEvaluator<Complex<f64>> {
    fn evaluate(
        &mut self,
        parameters: &[Complex<f64>],
        output: &mut [Complex<f64>],
        rows: usize,
    ) -> Result<(), String> {
        // The dependency's batch trait divides by rows, so empty batches must
        // be handled before entering its matrix evaluator.
        if rows == 0 {
            return Ok(());
        }
        BatchEvaluator::evaluate_batch(self, rows, parameters, output)
    }
}

// Loop coefficient rows carry the full local tensor/polynomial arena. A
// 16-row tile reduced memory and runtime in the complete gg control relative
// to 64/128 rows; the constructor retains an explicit diagnostic override.
const DEFAULT_COEFFICIENT_TILE_ROWS: usize = 16;

#[derive(Clone)]
struct KernelTemplate<E> {
    evaluator: E,
    inputs: usize,
    outputs: usize,
}

struct KernelCache<E> {
    templates: HashMap<Vec<u8>, KernelTemplate<E>>,
    hits: usize,
}

impl<E> Default for KernelCache<E> {
    fn default() -> Self {
        Self {
            templates: HashMap::new(),
            hits: 0,
        }
    }
}

fn kernel_template<T, E: Clone>(
    state: &[u8],
    prototype: &T,
    mut cache: Option<&mut KernelCache<E>>,
    create: impl FnOnce(ExpressionEvaluator<Complex<Rational>>, &T) -> ExecutionResult<E>,
) -> ExecutionResult<KernelTemplate<E>> {
    if let Some(cache) = cache.as_deref_mut() {
        if let Some(template) = cache.templates.get(state) {
            let template = template.clone();
            // Never evaluate the cached template: dependency Clone gives each
            // schedule independent mutable workspace and shared immutable code.
            cache.hits += 1;
            return Ok(template);
        }
    }
    let exact = exact_state(state)?;
    let template = KernelTemplate {
        inputs: exact.get_input_len(),
        outputs: exact.get_output_len(),
        evaluator: create(exact, prototype)?,
    };
    if let Some(cache) = cache {
        cache.templates.insert(state.to_vec(), template.clone());
    }
    Ok(template)
}

struct Node<E> {
    evaluator: E,
    inputs: Vec<usize>,
    output_offset: usize,
    output_count: usize,
}
struct Schedule<T: Carrier, E = BatchExpressionEvaluator<Complex<T>>> {
    nodes: Vec<Node<E>>,
    inputs: usize,
    width: usize,
    outputs: Vec<usize>,
    arena: Vec<Complex<T>>,
    gathered: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
    tile_rows: usize,
    zero: Complex<T>,
}
impl<T: BatchCarrier, E: ScheduledKernel<T>> Schedule<T, E>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(
        states: &[Vec<u8>],
        input_slots: &[Vec<usize>],
        inputs: usize,
        outputs: Vec<usize>,
        prototype: T,
        tile_rows: usize,
        mut create: impl FnMut(&[u8], &T) -> ExecutionResult<KernelTemplate<E>>,
    ) -> ExecutionResult<Self> {
        if states.len() != input_slots.len() {
            return Err(error(
                "coefficient schedule node and input-map counts differ",
            ));
        }
        let mut nodes = Vec::with_capacity(states.len());
        let mut width = inputs;
        for (index, (state, slots)) in states.iter().zip(input_slots).enumerate() {
            let template = create(state, &prototype)
                .map_err(|failure| error(format!("coefficient node {index}: {failure}")))?;
            if template.inputs != slots.len() || slots.iter().any(|&i| i >= width) {
                return Err(error(
                    "coefficient schedule input map has wrong shape or a forward reference",
                ));
            }
            let output_offset = width;
            let output_count = template.outputs;
            width = width
                .checked_add(output_count)
                .ok_or_else(|| error("coefficient arena dimensions overflow"))?;
            nodes.push(Node {
                evaluator: template.evaluator,
                inputs: slots.clone(),
                output_offset,
                output_count,
            });
        }
        if outputs.iter().any(|&i| i >= width) {
            return Err(error("coefficient schedule output slot out of range"));
        }
        Ok(Self {
            nodes,
            inputs,
            width,
            outputs,
            arena: Vec::new(),
            gathered: Vec::new(),
            output: Vec::new(),
            tile_rows,
            zero: Complex::new(prototype.zero(), prototype.zero()),
        })
    }
    fn evaluate(&mut self, parameters: &[Complex<T>], input: &Batch) -> ExecutionResult<Batch> {
        if input.columns() != self.inputs {
            return Err(error("coefficient schedule input dimensions differ"));
        }
        let mut values = Vec::with_capacity(shape(input.rows(), self.outputs.len())?);
        // Rows are independent evaluator lanes. Bound both our arena and the
        // dependency evaluators' internal batch buffers without changing their
        // arithmetic, instruction order or the public batch/identity contract.
        for start in (0..input.rows()).step_by(self.tile_rows) {
            let rows = (input.rows() - start).min(self.tile_rows);
            let arena_len = shape(rows, self.width)?;
            if self.arena.len() < arena_len {
                self.arena.resize(arena_len, self.zero.clone());
            }
            for row in 0..rows {
                self.arena[row * self.width..row * self.width + self.inputs].clone_from_slice(
                    &parameters[(start + row) * self.inputs..(start + row + 1) * self.inputs],
                );
            }
            for node in &mut self.nodes {
                let count = node.inputs.len();
                let output_count = node.output_count;
                // Existing evaluators overwrite all active output entries.
                // These scratch buffers can therefore be reused by successive
                // nodes, including nodes with different shapes and constants.
                let gathered_len = shape(rows, count)?;
                let output_len = shape(rows, output_count)?;
                // Keep inactive entries alive so arbitrary-precision storage
                // survives smaller nodes and the final partial tile.
                if self.gathered.len() < gathered_len {
                    self.gathered.resize(gathered_len, self.zero.clone());
                }
                if self.output.len() < output_len {
                    self.output.resize(output_len, self.zero.clone());
                }
                for row in 0..rows {
                    for (column, &slot) in node.inputs.iter().enumerate() {
                        self.gathered[row * count + column]
                            .clone_from(&self.arena[row * self.width + slot]);
                    }
                }
                node.evaluator
                    .evaluate(
                        &self.gathered[..gathered_len],
                        &mut self.output[..output_len],
                        rows,
                    )
                    .map_err(error)?;
                for row in 0..rows {
                    self.arena[row * self.width + node.output_offset
                        ..row * self.width + node.output_offset + output_count]
                        .clone_from_slice(
                            &self.output[row * output_count..(row + 1) * output_count],
                        );
                }
            }
            for row in 0..rows {
                values.extend(
                    self.outputs
                        .iter()
                        .map(|&slot| self.arena[row * self.width + slot].clone()),
                );
            }
        }
        Batch::typed(
            values,
            input.ids().to_vec(),
            self.outputs.len(),
            input.precision_bits(),
        )
    }

    fn workspace_capacity(&self) -> (usize, usize, usize) {
        (
            self.arena.capacity(),
            self.gathered.capacity(),
            self.output.capacity(),
        )
    }
}

enum KernelCaches {
    Double(KernelCache<BatchExpressionEvaluator<Complex<f64>>>),
    Jit(KernelCache<JITCompiledEvaluator<Complex<f64>>>),
    DoubleDouble(KernelCache<BatchExpressionEvaluator<Complex<DoubleFloat>>>),
    Arbitrary(KernelCache<BatchExpressionEvaluator<Complex<Float>>>),
}

/// Process-scoped cold evaluator templates; no event values are cached.
pub struct CoefficientKernelCache {
    templates: KernelCaches,
    arithmetic: String,
    precision_bits: u32,
    engine: String,
}

impl CoefficientKernelCache {
    pub fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
    pub fn engine(&self) -> &str {
        &self.engine
    }
    pub fn new(arithmetic: &str, precision: Option<u32>, engine: &str) -> ExecutionResult<Self> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if !matches!(engine, "interpreted" | "jit") {
            return Err(error("coefficient engine must be interpreted or jit"));
        }
        if engine == "jit" && !matches!(mode, Arithmetic::Double) {
            return Err(error("coefficient JIT requires double arithmetic"));
        }
        let templates = match mode {
            Arithmetic::Double if engine == "jit" => KernelCaches::Jit(KernelCache::default()),
            Arithmetic::Double => KernelCaches::Double(KernelCache::default()),
            Arithmetic::DoubleDouble => KernelCaches::DoubleDouble(KernelCache::default()),
            Arithmetic::Arbitrary => KernelCaches::Arbitrary(KernelCache::default()),
        };
        Ok(Self {
            templates,
            arithmetic: arithmetic.into(),
            precision_bits: bits,
            engine: engine.into(),
        })
    }

    pub fn entries(&self) -> usize {
        match &self.templates {
            KernelCaches::Double(c) => c.templates.len(),
            KernelCaches::Jit(c) => c.templates.len(),
            KernelCaches::DoubleDouble(c) => c.templates.len(),
            KernelCaches::Arbitrary(c) => c.templates.len(),
        }
    }

    pub fn hits(&self) -> usize {
        match &self.templates {
            KernelCaches::Double(c) => c.hits,
            KernelCaches::Jit(c) => c.hits,
            KernelCaches::DoubleDouble(c) => c.hits,
            KernelCaches::Arbitrary(c) => c.hits,
        }
    }

    /// Settings are fixed by this adapter and part of the cache's identity.
    pub fn settings(&self) -> &'static str {
        if self.engine == "jit" {
            "use_threads=false;use_simd=true"
        } else {
            "dependency-batch-interpreter"
        }
    }
}

enum Schedules {
    Double(Schedule<f64>),
    Jit(Schedule<f64, JITCompiledEvaluator<Complex<f64>>>),
    DoubleDouble(Schedule<DoubleFloat>),
    Arbitrary(Schedule<Float>),
}
pub struct CoefficientPlan {
    schedule: Schedules,
    precision_bits: u32,
    engine: String,
    tile_rows: usize,
}
impl CoefficientPlan {
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
    pub fn engine(&self) -> &str {
        &self.engine
    }
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
    pub fn new(
        states: Vec<Vec<u8>>,
        input_slots: Vec<Vec<usize>>,
        input_count: usize,
        output_slots: Vec<usize>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
        tile_rows: Option<usize>,
        mut kernel_cache: Option<&mut CoefficientKernelCache>,
    ) -> ExecutionResult<Self> {
        let tile_rows = tile_rows.unwrap_or(DEFAULT_COEFFICIENT_TILE_ROWS);
        if tile_rows == 0 {
            return Err(error("coefficient tile size must be positive"));
        }
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if !matches!(engine, "interpreted" | "jit") {
            return Err(error("coefficient engine must be interpreted or jit"));
        }
        if engine == "jit" && !matches!(mode, Arithmetic::Double) {
            return Err(error("coefficient JIT requires double arithmetic"));
        }
        if let Some(cache) = &kernel_cache {
            if cache.arithmetic != arithmetic
                || cache.precision_bits != bits
                || cache.engine != engine
            {
                return Err(error(
                    "coefficient kernel cache arithmetic, precision or engine differs",
                ));
            }
        }
        let cache = kernel_cache
            .as_deref_mut()
            .map(|cache| &mut cache.templates);
        let schedule = match mode {
            Arithmetic::Double if engine == "jit" => {
                let mut cache = match cache {
                    Some(KernelCaches::Jit(cache)) => Some(cache),
                    None => None,
                    _ => unreachable!("validated kernel cache arithmetic and engine"),
                };
                Schedules::Jit(Schedule::new(
                    &states,
                    &input_slots,
                    input_count,
                    output_slots,
                    0_f64,
                    tile_rows,
                    |state, prototype| {
                        kernel_template(state, prototype, cache.as_deref_mut(), |exact, _| {
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                exact.jit_compile::<Complex<f64>>(
                                    JITCompilationSettings::default()
                                        .with_option("use_threads", "false")
                                        .with_option("use_simd", "true"),
                                )
                            }))
                            .map_err(|_| {
                                error("dependency JIT compiler panicked on this kernel or host")
                            })?
                            .map_err(|failure| {
                                error(format!("dependency JIT unavailable: {failure}"))
                            })
                        })
                    },
                )?)
            }
            Arithmetic::Double => {
                let mut cache = match cache {
                    Some(KernelCaches::Double(cache)) => Some(cache),
                    None => None,
                    _ => unreachable!("validated kernel cache arithmetic and engine"),
                };
                Schedules::Double(Schedule::new(
                    &states,
                    &input_slots,
                    input_count,
                    output_slots,
                    0_f64,
                    tile_rows,
                    |state, prototype| {
                        kernel_template(
                            state,
                            prototype,
                            cache.as_deref_mut(),
                            |exact, prototype| Ok(batch_evaluator(exact, prototype)),
                        )
                    },
                )?)
            }
            Arithmetic::DoubleDouble => {
                let mut cache = match cache {
                    Some(KernelCaches::DoubleDouble(cache)) => Some(cache),
                    None => None,
                    _ => unreachable!("validated kernel cache arithmetic and engine"),
                };
                Schedules::DoubleDouble(Schedule::new(
                    &states,
                    &input_slots,
                    input_count,
                    output_slots,
                    DoubleFloat::from(0.0),
                    tile_rows,
                    |state, prototype| {
                        kernel_template(
                            state,
                            prototype,
                            cache.as_deref_mut(),
                            |exact, prototype| Ok(batch_evaluator(exact, prototype)),
                        )
                    },
                )?)
            }
            Arithmetic::Arbitrary => {
                let mut cache = match cache {
                    Some(KernelCaches::Arbitrary(cache)) => Some(cache),
                    None => None,
                    _ => unreachable!("validated kernel cache arithmetic and engine"),
                };
                Schedules::Arbitrary(Schedule::new(
                    &states,
                    &input_slots,
                    input_count,
                    output_slots,
                    Float::with_val(bits, 0),
                    tile_rows,
                    |state, prototype| {
                        kernel_template(
                            state,
                            prototype,
                            cache.as_deref_mut(),
                            |exact, prototype| Ok(batch_evaluator(exact, prototype)),
                        )
                    },
                )?)
            }
        };
        Ok(Self {
            schedule,
            precision_bits: bits,
            engine: engine.into(),
            tile_rows,
        })
    }
    pub fn evaluate_batch(&mut self, input: &Batch) -> ExecutionResult<Batch> {
        if self.precision_bits != input.precision_bits() {
            return Err(error("batch arithmetic or precision differs"));
        }
        match (&mut self.schedule, input.data()) {
            (Schedules::Double(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::Jit(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::DoubleDouble(s), Data::DoubleDouble(v)) => s.evaluate(v, input),
            (Schedules::Arbitrary(s), Data::Arbitrary(v)) => s.evaluate(v, input),
            _ => Err(error("batch arithmetic or precision differs")),
        }
    }

    /// Retained schedule elements (arena, gathered inputs, node outputs).
    /// This diagnostic excludes evaluator-owned workspace and returned batches.
    pub fn workspace_capacity(&self) -> (usize, usize, usize) {
        match &self.schedule {
            Schedules::Double(s) => s.workspace_capacity(),
            Schedules::Jit(s) => s.workspace_capacity(),
            Schedules::DoubleDouble(s) => s.workspace_capacity(),
            Schedules::Arbitrary(s) => s.workspace_capacity(),
        }
    }
}
