// SPDX-License-Identifier: 0BSD
//! Prepared Python ownership of the same dual operation used by saved runtimes.
use crate::{batch::NativeBatch, error};
use pyo3::prelude::*;
use rusticol_core::loop_runtime::RectangularDualWeights;
use rusticol_loop_reduction::runtime;

#[pyclass]
struct DualBornWeights(runtime::DualBornWeights);

#[pymethods]
impl DualBornWeights {
    #[new]
    #[pyo3(signature = (state, descriptor, constants, columns, arithmetic="double", precision=None))]
    fn new(
        state: &[u8],
        descriptor: &str,
        constants: &[u8],
        columns: Vec<[usize; 3]>,
        arithmetic: &str,
        precision: Option<u32>,
    ) -> PyResult<Self> {
        let rectangle: RectangularDualWeights = serde_json::from_str(descriptor).map_err(error)?;
        runtime::DualBornWeights::new(
            state,
            Some(&rectangle),
            Some(constants),
            &columns,
            arithmetic,
            precision,
        )
        .map(Self)
        .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn algorithm(&self) -> &'static str {
        self.0.algorithm()
    }
    #[getter]
    fn fallback_reason(&self) -> Option<&'static str> {
        self.0.fallback_reason()
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        py.detach(|| self.0.evaluate_batch(&input.0))
            .map(NativeBatch)
            .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DualBornWeights>()
}
