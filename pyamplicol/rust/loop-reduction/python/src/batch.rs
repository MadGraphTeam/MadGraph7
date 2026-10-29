// SPDX-License-Identifier: 0BSD

//! Python ownership and argument conversion for shared native batches.
use super::*;
use rusticol_loop_reduction::runtime::Batch;

#[derive(Clone)]
#[pyclass(skip_from_py_object)]
pub(super) struct NativeBatch(pub Batch);
impl std::ops::Deref for NativeBatch {
    type Target = Batch;
    fn deref(&self) -> &Batch {
        &self.0
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeBatch>()?;
    module.add_class::<ReductionBatch>()?;
    Ok(())
}
