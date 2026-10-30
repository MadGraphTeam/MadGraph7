// SPDX-License-Identifier: 0BSD

//! The native loop module of `pyamplicol._rusticol`: the v2 engine's
//! generation helpers, executor, reducer and seed contraction, and the shared
//! batch/evaluator transport (the legacy per-diagram runtime was removed).

mod batch;
mod model_proofs;
mod stability;
mod v2;
mod v2_catalogue;
mod v2_colour;
mod v2_dag;
mod v2_executor;
mod v2_generate;
mod v2_lower;
mod v2_seeds;
use batch::NativeBatch;

use pyo3::{exceptions::PyValueError, prelude::*};
use rusticol_loop_reduction::Complex;
use symbolica::{
    domains::float::{DoubleFloat, Float},
    domains::rational::Rational,
};

use rusticol_loop_reduction::runtime::{self, DecimalComplex, decimal_complex};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

/// Python ownership of the shared native saved evaluator.
#[pyclass]
struct CoefficientEvaluator(runtime::CoefficientEvaluator);
#[pymethods]
impl CoefficientEvaluator {
    #[new]
    #[pyo3(signature = (state, arithmetic="double", precision=None))]
    fn new(state: &[u8], arithmetic: &str, precision: Option<u32>) -> PyResult<Self> {
        runtime::CoefficientEvaluator::new(state, arithmetic, precision)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
    ) -> PyResult<Vec<DecimalComplex>> {
        py.detach(|| self.0.evaluate_decimal(values, rows))
            .map_err(error)
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        py.detach(|| self.0.evaluate_batch(&input.0))
            .map(NativeBatch)
            .map_err(error)
    }
    fn evaluate_f64(
        &mut self,
        py: Python<'_>,
        values: Vec<[f64; 2]>,
        rows: usize,
    ) -> PyResult<Vec<[f64; 2]>> {
        py.detach(|| self.0.evaluate_f64(values, rows))
            .map_err(error)
    }
}

#[pymodule]
fn _loop_reduction(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}

/// Register the same native loop core in the ordinary pyAmpliCol extension.
///
/// Importing the extension touches no Symbolica state: the source-scalar
/// primitives are installed by the paths that decode saved evaluators or
/// parse expressions (`runtime::register_scalar_primitives`). Without a
/// licence key in the environment, Symbolica's first state access takes the
/// user's single restricted-thread permit, so an import-time registration made
/// every second unlicensed pyAmpliCol process abort, tree-only ones included
/// (requirements addendum G; docs/user/symbolica-and-licensing.md).
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    batch::register(module)?;
    module.add_class::<CoefficientEvaluator>()?;
    v2::register(module)?;
    v2_catalogue::register(module)?;
    v2_dag::register(module)?;
    v2_executor::register(module)?;
    v2_generate::register(module)?;
    v2_colour::register(module)?;
    v2_lower::register(module)?;
    v2_seeds::register(module)?;
    model_proofs::register(module)?;
    Ok(())
}
