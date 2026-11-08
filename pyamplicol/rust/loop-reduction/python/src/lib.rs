// SPDX-License-Identifier: 0BSD

//! The native loop module of `pyamplicol._rusticol`: the v2 engine's
//! generation helpers, executor, reducer and seed contraction, and the shared
//! batch/evaluator transport (the legacy per-diagram runtime was removed).

mod batch;
mod model_proofs;
mod stability;
mod v2;
mod v2_catalogue;
mod v2_closure;
mod v2_colour;
mod v2_dag;
mod v2_executor;
mod v2_export;
mod v2_generate;
mod v2_lower;
mod v2_seeds;
use batch::NativeBatch;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
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

/// `saved` (a Python `Evaluator.save()`) in its canonical stored form: only
/// the Symbolica state its evaluator references, numbered canonically;
/// `saved` itself when it already is in it
/// (`rusticol_core::__private::canonical_saved_evaluator`).
#[pyfunction]
fn canonical_saved_evaluator<'py>(
    py: Python<'py>,
    saved: Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = saved.as_bytes();
    match py
        .detach(|| rusticol_core::__private::canonical_saved_evaluator(bytes))
        .map_err(error)?
    {
        Some(canonical) => Ok(PyBytes::new(py, &canonical)),
        None => Ok(saved),
    }
}

/// Whether this native extension links TIRust, the tensor-integral reduction
/// backend (the default cargo feature `tir`; `rusticol_loop_reduction::tir`).
#[pyfunction]
fn tirust_available() -> bool {
    rusticol_loop_reduction::tir::available()
}

/// The reduction backends of this native extension's v2 executor, as JSON
/// `{"ninjars": [carriers], "tirust": [carriers]}` (TIRust only in a build
/// with it): the capability query `loops.v2.reduction.native_backends`
/// reads (`runtime::backend_carriers`).
#[pyfunction]
fn reduction_backends() -> String {
    let record: serde_json::Map<String, serde_json::Value> =
        rusticol_loop_reduction::runtime::backend_carriers()
            .into_iter()
            .map(|(name, carriers)| (name.to_owned(), serde_json::json!(carriers)))
            .collect();
    serde_json::Value::Object(record).to_string()
}

/// TIRust's own structural bounds at a carrier of `bits` bits, as JSON
/// `{"max_propagators": n, "max_rank": r}` (`tir::capabilities`), or None
/// in a build without TIRust: the numbers its capability query
/// (`reducer_rank_refusal`) decides on.
#[pyfunction]
#[pyo3(signature = (bits=53))]
fn tirust_capabilities(bits: u32) -> Option<String> {
    rusticol_loop_reduction::tir::capabilities(bits).map(|(propagators, rank)| {
        serde_json::json!({"max_propagators": propagators, "max_rank": rank}).to_string()
    })
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
/// (requirements addendum G; docs/user/symbolica-and-licensing.md). Symbolica's
/// diagnostics are routed to standard error here (no Symbolica state either),
/// so they never interleave with the program's standard output.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    rusticol_core::diagnostics::symbolica_diagnostics_on_stderr();
    batch::register(module)?;
    module.add_class::<CoefficientEvaluator>()?;
    module.add_function(wrap_pyfunction!(canonical_saved_evaluator, module)?)?;
    module.add_function(wrap_pyfunction!(tirust_available, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_backends, module)?)?;
    module.add_function(wrap_pyfunction!(tirust_capabilities, module)?)?;
    v2::register(module)?;
    v2_catalogue::register(module)?;
    v2_closure::register(module)?;
    v2_dag::register(module)?;
    v2_executor::register(module)?;
    v2_export::register(module)?;
    v2_generate::register(module)?;
    v2_colour::register(module)?;
    v2_lower::register(module)?;
    v2_seeds::register(module)?;
    model_proofs::register(module)?;
    Ok(())
}
