// SPDX-License-Identifier: 0BSD

//! Python ownership of shared exact decimal ingress and rational evaluation.
use super::*;

pub(super) fn decimal_rational(value: &str) -> PyResult<Rational> {
    runtime::decimal_rational(value).map_err(error)
}

#[pymethods]
impl NativeBatch {
    #[staticmethod]
    #[pyo3(signature = (values, rows, columns, arithmetic="double", precision=None, row_ids=None))]
    fn from_exact_decimal(
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
        columns: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        py.detach(|| {
            runtime::Batch::from_exact_decimal(
                values, rows, columns, arithmetic, precision, row_ids,
            )
        })
        .map(Self)
        .map_err(error)
    }
}

#[pyclass]
pub(super) struct ExactPolynomialEvaluator(runtime::ExactPolynomialEvaluator);
#[pymethods]
impl ExactPolynomialEvaluator {
    #[new]
    fn new(state: &[u8]) -> PyResult<Self> {
        runtime::ExactPolynomialEvaluator::new(state)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn input_count(&self) -> usize {
        self.0.input_count()
    }
    #[getter]
    fn output_count(&self) -> usize {
        self.0.output_count()
    }
    #[pyo3(signature = (values, rows, arithmetic="double", precision=None, row_ids=None))]
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> PyResult<NativeBatch> {
        py.detach(|| {
            self.0
                .evaluate_decimal(values, rows, arithmetic, precision, row_ids)
        })
        .map(NativeBatch)
        .map_err(error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ExactPolynomialEvaluator>()?;
    Ok(())
}
