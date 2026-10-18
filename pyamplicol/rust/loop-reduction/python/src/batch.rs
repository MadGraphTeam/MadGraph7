// SPDX-License-Identifier: 0BSD

//! Python ownership and argument conversion for shared native batches.
use super::*;
use rusticol_loop_reduction::runtime::Batch;
pub(super) use rusticol_loop_reduction::runtime::{BatchCarrier, Data};
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

/// Python ownership of cold dependency evaluator templates.
#[pyclass]
struct CoefficientKernelCache(runtime::CoefficientKernelCache);
#[pymethods]
impl CoefficientKernelCache {
    #[new]
    #[pyo3(signature = (arithmetic="double", precision=None, engine="interpreted"))]
    fn new(arithmetic: &str, precision: Option<u32>, engine: &str) -> PyResult<Self> {
        runtime::CoefficientKernelCache::new(arithmetic, precision, engine)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn arithmetic(&self) -> &str {
        self.0.arithmetic()
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn engine(&self) -> &str {
        self.0.engine()
    }
    #[getter]
    fn entries(&self) -> usize {
        self.0.entries()
    }
    #[getter]
    fn hits(&self) -> usize {
        self.0.hits()
    }
    #[getter]
    fn settings(&self) -> &'static str {
        self.0.settings()
    }
}

#[pyclass]
pub(super) struct CoefficientPlan(runtime::CoefficientPlan);
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
        runtime::CoefficientPlan::new(
            states,
            input_slots,
            input_count,
            output_slots,
            arithmetic,
            precision,
            engine,
            tile_rows,
            kernel_cache.as_deref_mut().map(|c| &mut c.0),
        )
        .map(Self)
        .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn engine(&self) -> &str {
        self.0.engine()
    }
    #[getter]
    fn tile_rows(&self) -> usize {
        self.0.tile_rows()
    }
    #[getter]
    fn workspace_capacity(&self) -> (usize, usize, usize) {
        self.0.workspace_capacity()
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        py.detach(|| self.0.evaluate_batch(&input.0))
            .map(NativeBatch)
            .map_err(error)
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
