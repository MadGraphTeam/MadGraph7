// SPDX-License-Identifier: 0BSD

//! Python ownership and argument conversion for shared native batches.
use super::*;
use rusticol_loop_reduction::runtime::Batch;
pub(super) use rusticol_loop_reduction::runtime::{BatchCarrier, Data};
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

/// Python ownership of all-lane diagnostics and compact successful values.
#[pyclass]
pub(super) struct ReductionBatch(pub runtime::ReductionBatch);
#[pymethods]
impl ReductionBatch {
    #[getter]
    fn statuses(&self) -> Vec<(u64, String)> {
        self.0.statuses().to_vec()
    }
    #[getter]
    fn values(&self) -> PyResult<NativeBatch> {
        self.0.values().map(NativeBatch).map_err(error)
    }
    fn successful_values(&self) -> NativeBatch {
        NativeBatch(self.0.successful_values())
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
