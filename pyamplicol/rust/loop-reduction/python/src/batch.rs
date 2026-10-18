// SPDX-License-Identifier: 0BSD

//! Python ownership and argument conversion for shared native batches.
use super::*;
use rusticol_loop_reduction::runtime::Batch;
pub(super) use rusticol_loop_reduction::runtime::{BatchCarrier, Data};
use std::collections::HashMap;
use symbolica::evaluate::{BatchEvaluator, JITCompilationSettings, JITCompiledEvaluator};
fn shape(rows: usize, columns: usize) -> PyResult<usize> {
    runtime::shape(rows, columns).map_err(error)
}
#[derive(Clone)]
#[pyclass(skip_from_py_object)]
pub(super) struct NativeBatch(pub Batch);
impl std::ops::Deref for NativeBatch {
    type Target = Batch;
    fn deref(&self) -> &Batch {
        &self.0
    }
}
impl NativeBatch {
    pub fn typed<T: BatchCarrier>(
        values: Vec<Complex<T>>,
        ids: Vec<u64>,
        columns: usize,
        bits: u32,
    ) -> PyResult<Self> {
        Batch::typed(values, ids, columns, bits)
            .map(Self)
            .map_err(error)
    }
    pub fn compatible(&self, other: &Self) -> PyResult<()> {
        self.0.compatible(&other.0).map_err(error)
    }
    pub fn same_rows(&self, other: &Self) -> PyResult<()> {
        self.0.same_rows(&other.0).map_err(error)
    }
}
#[pymethods]
impl NativeBatch {
    #[staticmethod]
    #[pyo3(signature = (values, rows, columns, arithmetic="double", precision=None, row_ids=None))]
    fn from_decimal(
        values: Vec<DecimalComplex>,
        rows: usize,
        columns: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        Batch::from_decimal(values, rows, columns, arithmetic, precision, row_ids)
            .map(Self)
            .map_err(error)
    }
    #[getter(rows)]
    fn get_rows(&self) -> usize {
        self.0.rows()
    }
    #[getter(columns)]
    fn get_columns(&self) -> usize {
        self.0.columns()
    }
    #[getter(precision_bits)]
    fn get_precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn row_ids(&self) -> Vec<u64> {
        self.0.row_ids()
    }
    fn to_decimal(&self, py: Python<'_>) -> Vec<DecimalComplex> {
        py.detach(|| self.0.to_decimal())
    }
    fn nonfinite_rows(&self, py: Python<'_>) -> Vec<u64> {
        py.detach(|| self.0.nonfinite_rows())
    }
    #[pyo3(signature = (indices, row_ids=None))]
    fn take_rows(&self, indices: Vec<usize>, row_ids: Option<Vec<u64>>) -> PyResult<Self> {
        self.0.take_rows(indices, row_ids).map(Self).map_err(error)
    }
    fn take_columns(&self, indices: Vec<usize>) -> PyResult<Self> {
        self.0.take_columns(indices).map(Self).map_err(error)
    }
    fn reshape_rows(&self, rows: usize, columns: usize, row_ids: Vec<u64>) -> PyResult<Self> {
        self.0
            .reshape_rows(rows, columns, row_ids)
            .map(Self)
            .map_err(error)
    }
    #[staticmethod]
    fn concat_columns(batches: Vec<PyRef<'_, NativeBatch>>) -> PyResult<Self> {
        Batch::concat_columns(&batches.iter().map(|b| &b.0).collect::<Vec<_>>())
            .map(Self)
            .map_err(error)
    }
}

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
    create: impl FnOnce(ExpressionEvaluator<Complex<Rational>>, &T) -> PyResult<E>,
) -> PyResult<KernelTemplate<E>> {
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
        mut create: impl FnMut(&[u8], &T) -> PyResult<KernelTemplate<E>>,
    ) -> PyResult<Self> {
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
    fn evaluate(
        &mut self,
        parameters: &[Complex<T>],
        input: &NativeBatch,
    ) -> PyResult<NativeBatch> {
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
        NativeBatch::typed(
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
#[pyclass]
struct CoefficientKernelCache {
    templates: KernelCaches,
    #[pyo3(get)]
    arithmetic: String,
    #[pyo3(get)]
    precision_bits: u32,
    #[pyo3(get)]
    engine: String,
}

#[pymethods]
impl CoefficientKernelCache {
    #[new]
    #[pyo3(signature = (arithmetic="double", precision=None, engine="interpreted"))]
    fn new(arithmetic: &str, precision: Option<u32>, engine: &str) -> PyResult<Self> {
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

    #[getter]
    fn entries(&self) -> usize {
        match &self.templates {
            KernelCaches::Double(c) => c.templates.len(),
            KernelCaches::Jit(c) => c.templates.len(),
            KernelCaches::DoubleDouble(c) => c.templates.len(),
            KernelCaches::Arbitrary(c) => c.templates.len(),
        }
    }

    #[getter]
    fn hits(&self) -> usize {
        match &self.templates {
            KernelCaches::Double(c) => c.hits,
            KernelCaches::Jit(c) => c.hits,
            KernelCaches::DoubleDouble(c) => c.hits,
            KernelCaches::Arbitrary(c) => c.hits,
        }
    }

    /// Settings are fixed by this adapter and part of the cache's identity.
    #[getter]
    fn settings(&self) -> &'static str {
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
#[pyclass]
pub(super) struct CoefficientPlan {
    schedule: Schedules,
    #[pyo3(get)]
    precision_bits: u32,
    #[pyo3(get)]
    engine: String,
    #[pyo3(get)]
    tile_rows: usize,
}
#[pymethods]
impl CoefficientPlan {
    #[new]
    #[pyo3(signature = (states, input_slots, input_count, output_slots, arithmetic="double", precision=None, engine="interpreted", tile_rows=None, kernel_cache=None))]
    fn new(
        states: Vec<Vec<u8>>,
        input_slots: Vec<Vec<usize>>,
        input_count: usize,
        output_slots: Vec<usize>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
        tile_rows: Option<usize>,
        mut kernel_cache: Option<PyRefMut<'_, CoefficientKernelCache>>,
    ) -> PyResult<Self> {
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
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        if self.precision_bits != input.precision_bits() {
            return Err(error("batch arithmetic or precision differs"));
        }
        py.detach(|| match (&mut self.schedule, input.data()) {
            (Schedules::Double(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::Jit(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::DoubleDouble(s), Data::DoubleDouble(v)) => s.evaluate(v, input),
            (Schedules::Arbitrary(s), Data::Arbitrary(v)) => s.evaluate(v, input),
            _ => Err(error("batch arithmetic or precision differs")),
        })
    }

    /// Retained schedule elements (arena, gathered inputs, node outputs).
    /// This diagnostic excludes evaluator-owned workspace and returned batches.
    #[getter]
    fn workspace_capacity(&self) -> (usize, usize, usize) {
        match &self.schedule {
            Schedules::Double(s) => s.workspace_capacity(),
            Schedules::Jit(s) => s.workspace_capacity(),
            Schedules::DoubleDouble(s) => s.workspace_capacity(),
            Schedules::Arbitrary(s) => s.workspace_capacity(),
        }
    }
}

/// All-lane diagnostics plus compact successful values; failed numbers stay hidden.
#[pyclass]
struct ReductionBatch {
    values: NativeBatch,
    #[pyo3(get)]
    statuses: Vec<(u64, String)>,
}
#[pymethods]
impl ReductionBatch {
    #[getter]
    fn values(&self) -> PyResult<NativeBatch> {
        if self.statuses.len() != self.values.rows() {
            return Err(error(
                "reduction contains failed lanes; inspect statuses and successful_values",
            ));
        }
        Ok(self.values.clone())
    }
    fn successful_values(&self) -> NativeBatch {
        self.values.clone()
    }
}

fn reduce<T: BatchCarrier>(
    plan: &mut Reductions<T>,
    shifts: &[Complex<T>],
    masses: &[Complex<T>],
    scales: &[Complex<T>],
    parameters: &[Complex<T>],
    matrix: Option<&[Complex<T>]>,
    input: &NativeBatch,
) -> PyResult<ReductionBatch>
where
    Complex<T>: Real + EvaluationDomain,
{
    let shifts: Vec<_> = shifts
        .chunks_exact(4)
        .map(|p| FourMomentum::from([p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone()]))
        .collect();
    let lanes = plan.evaluate_native(input.ids(), &shifts, masses, scales, parameters, matrix)?;
    reduced_batch(lanes, input.precision_bits())
}
fn reduced_batch<T: BatchCarrier>(
    lanes: Vec<Lane<Complex<T>, String>>,
    bits: u32,
) -> PyResult<ReductionBatch> {
    let mut values = Vec::with_capacity(shape(lanes.len(), 3)?);
    let mut ids = Vec::with_capacity(lanes.len());
    let mut statuses = Vec::with_capacity(lanes.len());
    for lane in lanes {
        statuses.push((lane.id, lane.status));
        if let Some(value) = lane.value {
            ids.push(lane.id);
            values.extend([value.finite, value.single_pole, value.double_pole]);
        }
    }
    Ok(ReductionBatch {
        values: NativeBatch::typed(values, ids, 3, bits)?,
        statuses,
    })
}

#[pymethods]
impl PolynomialPlan {
    #[pyo3(signature = (shifts, masses_squared, scales_squared, parameters, s_matrix=None))]
    fn evaluate_batch(
        &mut self,
        py: Python<'_>,
        shifts: &NativeBatch,
        masses_squared: &NativeBatch,
        scales_squared: &NativeBatch,
        parameters: &NativeBatch,
        s_matrix: Option<&NativeBatch>,
    ) -> PyResult<ReductionBatch> {
        for batch in [shifts, masses_squared, scales_squared] {
            parameters.same_rows(batch)?;
        }
        if let Some(matrix) = s_matrix {
            parameters.same_rows(matrix)?;
        }
        if parameters.precision_bits() != self.precision_bits {
            return Err(error("batch arithmetic or precision differs"));
        }
        macro_rules! run {
            ($plan:expr, $variant:ident) => {{
                let p = $plan;
                if shifts.columns() != p.propagators * 4
                    || masses_squared.columns() != p.propagators
                    || scales_squared.columns() != 1
                    || parameters.columns() != p.parameter_count
                    || s_matrix.is_some_and(|s| s.columns() != p.propagators * p.propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (
                    Data::$variant(shifts),
                    Data::$variant(masses),
                    Data::$variant(scales),
                    Data::$variant(values),
                ) = (
                    shifts.data(),
                    masses_squared.data(),
                    scales_squared.data(),
                    parameters.data(),
                )
                else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::$variant(v) = s.data() else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                reduce(p, shifts, masses, scales, values, matrix, parameters)
            }};
        }
        py.detach(|| match &mut self.plan {
            Plans::Fast(p) => {
                if shifts.columns() != p.propagators * 4
                    || masses_squared.columns() != p.propagators
                    || scales_squared.columns() != 1
                    || parameters.columns() != p.parameter_count
                    || s_matrix.is_some_and(|s| s.columns() != p.propagators * p.propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (Data::Double(s), Data::Double(m), Data::Double(r), Data::Double(c)) = (
                    shifts.data(),
                    masses_squared.data(),
                    scales_squared.data(),
                    parameters.data(),
                ) else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::Double(v) = s.data() else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                let shifts: Vec<_> = s
                    .chunks_exact(4)
                    .map(|v| FourMomentum::from([v[0], v[1], v[2], v[3]]))
                    .collect();
                reduced_batch(
                    p.evaluate_native(parameters.ids(), &shifts, m, r, c, matrix)?,
                    53,
                )
            }
            Plans::Double(p) => run!(p, Double),
            Plans::DoubleDouble(p) => run!(p, DoubleDouble),
            Plans::Arbitrary(p) => run!(p, Arbitrary),
        })
    }
}

fn spinor_buffer<T: BatchCarrier>(
    values: &[Complex<T>],
    input: &NativeBatch,
    helicities: &[i32],
    kinds: &[String],
) -> PyResult<NativeBatch> {
    if input.columns() != 4 || input.rows() != helicities.len() || input.rows() != kinds.len() {
        return Err(error("massless state batch dimensions differ"));
    }
    let antiquarks = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "u" | "u_bar" => Ok(false),
            "v" | "v_bar" => Ok(true),
            _ => Err(error("massless state kind must be u, v, u_bar, or v_bar")),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let momenta = values
        .chunks_exact(4)
        .map(|p| {
            if p.iter().any(|x| !x.im.is_zero() || !x.re.is_finite()) || p[0].re <= p[0].re.zero() {
                return Err(error(
                    "massless states require finite real physical momenta with positive energy",
                ));
            }
            Ok([
                p[0].re.clone(),
                p[1].re.clone(),
                p[2].re.clone(),
                p[3].re.clone(),
            ])
        })
        .collect::<PyResult<Vec<_>>>()?;
    let states = physical_spinors(&momenta, helicities, kinds, &antiquarks)?;
    NativeBatch::typed(
        states.into_iter().flatten().collect(),
        input.ids().to_vec(),
        4,
        input.precision_bits(),
    )
}

#[pyfunction]
fn massless_spinors_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
) -> PyResult<NativeBatch> {
    py.detach(|| match momenta.data() {
        Data::Double(v) => spinor_buffer(v, momenta, &helicities, &kinds),
        Data::DoubleDouble(v) => spinor_buffer(v, momenta, &helicities, &kinds),
        Data::Arbitrary(v) => spinor_buffer(v, momenta, &helicities, &kinds),
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBatch>()?;
    module.add_class::<CoefficientPlan>()?;
    module.add_class::<CoefficientKernelCache>()?;
    module.add_class::<ReductionBatch>()?;
    module.add_function(wrap_pyfunction!(massless_spinors_batch, module)?)?;
    Ok(())
}
