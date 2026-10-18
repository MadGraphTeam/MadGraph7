// SPDX-License-Identifier: 0BSD

//! Typed immutable batch buffers and a validated native coefficient schedule.

use super::*;
use std::sync::Arc;
use symbolica::evaluate::{BatchEvaluator, JITCompilationSettings, JITCompiledEvaluator};

#[derive(Clone)]
pub(super) enum Data {
    Double(Arc<Vec<Complex<f64>>>),
    DoubleDouble(Arc<Vec<Complex<DoubleFloat>>>),
    Arbitrary(Arc<Vec<Complex<Float>>>),
}

pub(super) trait BatchCarrier: Carrier {
    fn data(values: Vec<Complex<Self>>) -> Data;
}
impl BatchCarrier for f64 {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::Double(Arc::new(values))
    }
}
impl BatchCarrier for DoubleFloat {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::DoubleDouble(Arc::new(values))
    }
}
impl BatchCarrier for Float {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::Arbitrary(Arc::new(values))
    }
}

fn shape(rows: usize, columns: usize) -> PyResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| error("batch dimensions overflow"))
}

#[derive(Clone)]
#[pyclass(skip_from_py_object)]
pub(super) struct NativeBatch {
    pub data: Data,
    pub ids: Vec<u64>,
    #[pyo3(get)]
    pub rows: usize,
    #[pyo3(get)]
    pub columns: usize,
    #[pyo3(get)]
    pub precision_bits: u32,
}

impl NativeBatch {
    pub fn typed<T: BatchCarrier>(
        values: Vec<Complex<T>>,
        ids: Vec<u64>,
        columns: usize,
        bits: u32,
    ) -> PyResult<Self> {
        if values.len() != shape(ids.len(), columns)? {
            return Err(error("batch values do not match dimensions"));
        }
        Ok(Self {
            data: T::data(values),
            rows: ids.len(),
            ids,
            columns,
            precision_bits: bits,
        })
    }
    pub fn compatible(&self, other: &Self) -> PyResult<()> {
        if self.precision_bits != other.precision_bits
            || std::mem::discriminant(&self.data) != std::mem::discriminant(&other.data)
        {
            return Err(error("batch arithmetic or precision differs"));
        }
        Ok(())
    }
    pub fn same_rows(&self, other: &Self) -> PyResult<()> {
        self.compatible(other)?;
        if self.ids != other.ids {
            return Err(error("batch row identities differ"));
        }
        Ok(())
    }
}

fn parse_values<T: Carrier>(values: &[DecimalComplex], bits: u32) -> PyResult<Vec<Complex<T>>> {
    values.iter().map(|v| parse_complex(v, bits)).collect()
}
fn decimals<T: Carrier>(values: &[Complex<T>], bits: u32) -> Vec<DecimalComplex> {
    values
        .iter()
        .cloned()
        .map(|v| decimal_complex(v, bits))
        .collect()
}
fn nonfinite<T: Carrier>(values: &[Complex<T>], columns: usize, ids: &[u64]) -> Vec<u64> {
    ids.iter()
        .enumerate()
        .filter_map(|(row, &id)| {
            values[row * columns..(row + 1) * columns]
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
                .then_some(id)
        })
        .collect()
}
fn take<T: BatchCarrier>(
    values: &[Complex<T>],
    source_columns: usize,
    rows: &[usize],
    columns: &[usize],
    ids: Vec<u64>,
    bits: u32,
) -> PyResult<NativeBatch> {
    let mut out = Vec::with_capacity(shape(rows.len(), columns.len())?);
    for row in rows {
        for column in columns {
            out.push(values[row * source_columns + column].clone());
        }
    }
    NativeBatch::typed(out, ids, columns.len(), bits)
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
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if values.len() != shape(rows, columns)? {
            return Err(error("batch values do not match dimensions"));
        }
        let ids = row_ids.unwrap_or_else(|| (0..rows as u64).collect());
        if ids.len() != rows {
            return Err(error("batch row identities do not match dimensions"));
        }
        match mode {
            Arithmetic::Double => {
                Self::typed(parse_values::<f64>(&values, bits)?, ids, columns, bits)
            }
            Arithmetic::DoubleDouble => Self::typed(
                parse_values::<DoubleFloat>(&values, bits)?,
                ids,
                columns,
                bits,
            ),
            Arithmetic::Arbitrary => {
                Self::typed(parse_values::<Float>(&values, bits)?, ids, columns, bits)
            }
        }
    }
    #[getter]
    fn row_ids(&self) -> Vec<u64> {
        self.ids.clone()
    }
    fn to_decimal(&self, py: Python<'_>) -> Vec<DecimalComplex> {
        py.detach(|| match &self.data {
            Data::Double(v) => decimals(v, self.precision_bits),
            Data::DoubleDouble(v) => decimals(v, self.precision_bits),
            Data::Arbitrary(v) => decimals(v, self.precision_bits),
        })
    }
    fn nonfinite_rows(&self, py: Python<'_>) -> Vec<u64> {
        py.detach(|| match &self.data {
            Data::Double(v) => nonfinite(v, self.columns, &self.ids),
            Data::DoubleDouble(v) => nonfinite(v, self.columns, &self.ids),
            Data::Arbitrary(v) => nonfinite(v, self.columns, &self.ids),
        })
    }
    #[pyo3(signature = (indices, row_ids=None))]
    fn take_rows(&self, indices: Vec<usize>, row_ids: Option<Vec<u64>>) -> PyResult<Self> {
        if indices.iter().any(|&i| i >= self.rows) {
            return Err(error("batch row index out of range"));
        }
        let ids = row_ids.unwrap_or_else(|| indices.iter().map(|&i| self.ids[i]).collect());
        if ids.len() != indices.len() {
            return Err(error("batch row identities do not match dimensions"));
        }
        let columns: Vec<_> = (0..self.columns).collect();
        match &self.data {
            Data::Double(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
            Data::DoubleDouble(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
            Data::Arbitrary(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
        }
    }
    fn take_columns(&self, indices: Vec<usize>) -> PyResult<Self> {
        if indices.iter().any(|&i| i >= self.columns) {
            return Err(error("batch column index out of range"));
        }
        let rows: Vec<_> = (0..self.rows).collect();
        match &self.data {
            Data::Double(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
            Data::DoubleDouble(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
            Data::Arbitrary(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
        }
    }
    fn reshape_rows(&self, rows: usize, columns: usize, row_ids: Vec<u64>) -> PyResult<Self> {
        if shape(rows, columns)? != shape(self.rows, self.columns)? || row_ids.len() != rows {
            return Err(error("reshape dimensions or row identities differ"));
        }
        Ok(Self {
            data: self.data.clone(),
            ids: row_ids,
            rows,
            columns,
            precision_bits: self.precision_bits,
        })
    }
    #[staticmethod]
    fn concat_columns(batches: Vec<PyRef<'_, NativeBatch>>) -> PyResult<Self> {
        let first = batches
            .first()
            .ok_or_else(|| error("concatenation needs at least one batch"))?;
        for batch in &batches {
            first.same_rows(batch)?;
        }
        let columns = batches.iter().try_fold(0_usize, |n, b| {
            n.checked_add(b.columns)
                .ok_or_else(|| error("batch dimensions overflow"))
        })?;
        macro_rules! concat {
            ($variant:ident) => {{
                let mut values = Vec::with_capacity(shape(first.rows, columns)?);
                for row in 0..first.rows {
                    for batch in &batches {
                        let Data::$variant(source) = &batch.data else {
                            unreachable!()
                        };
                        values.extend_from_slice(
                            &source[row * batch.columns..(row + 1) * batch.columns],
                        );
                    }
                }
                Self::typed(values, first.ids.clone(), columns, first.precision_bits)
            }};
        }
        match &first.data {
            Data::Double(_) => concat!(Double),
            Data::DoubleDouble(_) => concat!(DoubleDouble),
            Data::Arbitrary(_) => concat!(Arbitrary),
        }
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

struct Node<T: Carrier, E> {
    evaluator: E,
    inputs: Vec<usize>,
    output_offset: usize,
    output_count: usize,
    gathered: Vec<Complex<T>>,
    output: Vec<Complex<T>>,
}
struct Schedule<T: Carrier, E = BatchExpressionEvaluator<Complex<T>>> {
    nodes: Vec<Node<T, E>>,
    inputs: usize,
    width: usize,
    outputs: Vec<usize>,
    arena: Vec<Complex<T>>,
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
        create: impl Fn(ExpressionEvaluator<Complex<Rational>>, &T) -> PyResult<E>,
    ) -> PyResult<Self> {
        if states.len() != input_slots.len() {
            return Err(error(
                "coefficient schedule node and input-map counts differ",
            ));
        }
        let mut nodes = Vec::with_capacity(states.len());
        let mut width = inputs;
        for (index, (state, slots)) in states.iter().zip(input_slots).enumerate() {
            let exact = exact_state(state)?;
            if exact.get_input_len() != slots.len() || slots.iter().any(|&i| i >= width) {
                return Err(error(
                    "coefficient schedule input map has wrong shape or a forward reference",
                ));
            }
            let output_offset = width;
            let output_count = exact.get_output_len();
            width = width
                .checked_add(output_count)
                .ok_or_else(|| error("coefficient arena dimensions overflow"))?;
            nodes.push(Node {
                evaluator: create(exact, &prototype)
                    .map_err(|failure| error(format!("coefficient node {index}: {failure}")))?,
                inputs: slots.clone(),
                output_offset,
                output_count,
                gathered: Vec::new(),
                output: Vec::new(),
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
            zero: Complex::new(prototype.zero(), prototype.zero()),
        })
    }
    fn evaluate(
        &mut self,
        parameters: &[Complex<T>],
        input: &NativeBatch,
    ) -> PyResult<NativeBatch> {
        if input.columns != self.inputs {
            return Err(error("coefficient schedule input dimensions differ"));
        }
        self.arena
            .resize(shape(input.rows, self.width)?, self.zero.clone());
        for row in 0..input.rows {
            self.arena[row * self.width..row * self.width + self.inputs]
                .clone_from_slice(&parameters[row * self.inputs..(row + 1) * self.inputs]);
        }
        for node in &mut self.nodes {
            let count = node.inputs.len();
            node.gathered
                .resize(shape(input.rows, count)?, self.zero.clone());
            let output_count = node.output_count;
            node.output
                .resize(shape(input.rows, output_count)?, self.zero.clone());
            for row in 0..input.rows {
                for (column, &slot) in node.inputs.iter().enumerate() {
                    node.gathered[row * count + column]
                        .clone_from(&self.arena[row * self.width + slot]);
                }
            }
            node.evaluator
                .evaluate(&node.gathered, &mut node.output, input.rows)
                .map_err(error)?;
            for row in 0..input.rows {
                self.arena[row * self.width + node.output_offset
                    ..row * self.width + node.output_offset + output_count]
                    .clone_from_slice(&node.output[row * output_count..(row + 1) * output_count]);
            }
        }
        take(
            &self.arena,
            self.width,
            &(0..input.rows).collect::<Vec<_>>(),
            &self.outputs,
            input.ids.clone(),
            input.precision_bits,
        )
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
}
#[pymethods]
impl CoefficientPlan {
    #[new]
    #[pyo3(signature = (states, input_slots, input_count, output_slots, arithmetic="double", precision=None, engine="interpreted"))]
    fn new(
        states: Vec<Vec<u8>>,
        input_slots: Vec<Vec<usize>>,
        input_count: usize,
        output_slots: Vec<usize>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
    ) -> PyResult<Self> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if !matches!(engine, "interpreted" | "jit") {
            return Err(error("coefficient engine must be interpreted or jit"));
        }
        if engine == "jit" && !matches!(mode, Arithmetic::Double) {
            return Err(error("coefficient JIT requires double arithmetic"));
        }
        let schedule = match mode {
            Arithmetic::Double if engine == "jit" => Schedules::Jit(Schedule::new(
                &states,
                &input_slots,
                input_count,
                output_slots,
                0_f64,
                |exact, _| {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        exact.jit_compile::<Complex<f64>>(
                            JITCompilationSettings::default()
                                .with_option("use_threads", "false")
                                .with_option("use_simd", "true"),
                        )
                    }))
                    .map_err(|_| error("dependency JIT compiler panicked on this kernel or host"))?
                    .map_err(|failure| error(format!("dependency JIT unavailable: {failure}")))
                },
            )?),
            Arithmetic::Double => Schedules::Double(Schedule::new(
                &states,
                &input_slots,
                input_count,
                output_slots,
                0_f64,
                |exact, prototype| Ok(batch_evaluator(exact, prototype)),
            )?),
            Arithmetic::DoubleDouble => Schedules::DoubleDouble(Schedule::new(
                &states,
                &input_slots,
                input_count,
                output_slots,
                DoubleFloat::from(0.0),
                |exact, prototype| Ok(batch_evaluator(exact, prototype)),
            )?),
            Arithmetic::Arbitrary => Schedules::Arbitrary(Schedule::new(
                &states,
                &input_slots,
                input_count,
                output_slots,
                Float::with_val(bits, 0),
                |exact, prototype| Ok(batch_evaluator(exact, prototype)),
            )?),
        };
        Ok(Self {
            schedule,
            precision_bits: bits,
            engine: engine.into(),
        })
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        if self.precision_bits != input.precision_bits {
            return Err(error("batch arithmetic or precision differs"));
        }
        py.detach(|| match (&mut self.schedule, &input.data) {
            (Schedules::Double(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::Jit(s), Data::Double(v)) => s.evaluate(v, input),
            (Schedules::DoubleDouble(s), Data::DoubleDouble(v)) => s.evaluate(v, input),
            (Schedules::Arbitrary(s), Data::Arbitrary(v)) => s.evaluate(v, input),
            _ => Err(error("batch arithmetic or precision differs")),
        })
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
        if self.statuses.len() != self.values.rows {
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
    let lanes = plan.evaluate_native(&input.ids, &shifts, masses, scales, parameters, matrix)?;
    reduced_batch(lanes, input.precision_bits)
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
        if parameters.precision_bits != self.precision_bits {
            return Err(error("batch arithmetic or precision differs"));
        }
        macro_rules! run {
            ($plan:expr, $variant:ident) => {{
                let p = $plan;
                if shifts.columns != p.propagators * 4
                    || masses_squared.columns != p.propagators
                    || scales_squared.columns != 1
                    || parameters.columns != p.parameter_count
                    || s_matrix.is_some_and(|s| s.columns != p.propagators * p.propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (
                    Data::$variant(shifts),
                    Data::$variant(masses),
                    Data::$variant(scales),
                    Data::$variant(values),
                ) = (
                    &shifts.data,
                    &masses_squared.data,
                    &scales_squared.data,
                    &parameters.data,
                )
                else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::$variant(v) = &s.data else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                reduce(p, shifts, masses, scales, values, matrix, parameters)
            }};
        }
        py.detach(|| match &mut self.plan {
            Plans::Fast(p) => {
                if shifts.columns != p.propagators * 4
                    || masses_squared.columns != p.propagators
                    || scales_squared.columns != 1
                    || parameters.columns != p.parameter_count
                    || s_matrix.is_some_and(|s| s.columns != p.propagators * p.propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (Data::Double(s), Data::Double(m), Data::Double(r), Data::Double(c)) = (
                    &shifts.data,
                    &masses_squared.data,
                    &scales_squared.data,
                    &parameters.data,
                ) else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::Double(v) = &s.data else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                let shifts: Vec<_> = s
                    .chunks_exact(4)
                    .map(|v| FourMomentum::from([v[0], v[1], v[2], v[3]]))
                    .collect();
                reduced_batch(
                    p.evaluate_native(&parameters.ids, &shifts, m, r, c, matrix)?,
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
    if input.columns != 4 || input.rows != helicities.len() || input.rows != kinds.len() {
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
        input.ids.clone(),
        4,
        input.precision_bits,
    )
}

#[pyfunction]
fn massless_spinors_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
) -> PyResult<NativeBatch> {
    py.detach(|| match &momenta.data {
        Data::Double(v) => spinor_buffer(v, momenta, &helicities, &kinds),
        Data::DoubleDouble(v) => spinor_buffer(v, momenta, &helicities, &kinds),
        Data::Arbitrary(v) => spinor_buffer(v, momenta, &helicities, &kinds),
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBatch>()?;
    module.add_class::<CoefficientPlan>()?;
    module.add_class::<ReductionBatch>()?;
    module.add_function(wrap_pyfunction!(massless_spinors_batch, module)?)?;
    Ok(())
}
