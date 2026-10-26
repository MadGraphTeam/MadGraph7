// SPDX-License-Identifier: 0BSD

//! Experimental native consumer, separate until NinjaRS is self-contained.
//! Raw reduction primitives return integrals; NativeLoopProgram wraps the shared
//! complete Born/loop-interference coordinator for validated internal plans.

mod batch;
mod dual_weights;
mod exact;
mod physical;
mod portable;
mod stability;
mod states;
mod v2;
mod v2_colour;
mod v2_dag;
mod v2_executor;
mod v2_generate;
mod v2_lower;
mod virtual_runtime;
use batch::NativeBatch;

use pyo3::{exceptions::PyValueError, prelude::*};
use rusticol_loop_reduction::Complex;
use symbolica::{
    domains::float::{DoubleFloat, Float},
    domains::rational::Rational,
};

use runtime::{DecimalMomentum, RawLane};
use rusticol_loop_reduction::runtime::{
    self, Arithmetic, Carrier, DecimalComplex, decimal_complex,
};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn arithmetic_mode(arithmetic: &str, precision: Option<u32>) -> PyResult<(Arithmetic, u32)> {
    runtime::arithmetic_mode(arithmetic, precision).map_err(error)
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

/// Python ownership of native polynomial reduction and epsilon convolution.
#[pyclass]
struct PolynomialPlan(runtime::PolynomialPlan);
#[pymethods]
impl PolynomialPlan {
    #[new]
    #[pyo3(signature = (propagators, epsilon_sectors, parameters, arithmetic="double", precision=None, engine="generic"))]
    fn new(
        propagators: usize,
        epsilon_sectors: [String; 3],
        parameters: Vec<String>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
    ) -> PyResult<Self> {
        runtime::PolynomialPlan::new(
            propagators,
            epsilon_sectors,
            parameters,
            arithmetic,
            precision,
            engine,
        )
        .map(Self)
        .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    fn set_profiling(&mut self, enabled: bool) -> PyResult<()> {
        self.0.set_profiling(enabled).map_err(error)
    }
    fn take_profile(&mut self) -> PyResult<std::collections::BTreeMap<String, f64>> {
        self.0.take_profile().map_err(error)
    }
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        ids: Vec<u64>,
        shifts: Vec<DecimalMomentum>,
        masses_squared: Vec<DecimalComplex>,
        scales_squared: Vec<DecimalComplex>,
        parameters: Vec<DecimalComplex>,
    ) -> PyResult<Vec<RawLane>> {
        py.detach(|| {
            self.0
                .evaluate_decimal(ids, shifts, masses_squared, scales_squared, parameters)
        })
        .map_err(error)
    }
    #[pyo3(signature = (shifts, masses_squared, scales_squared, parameters, s_matrix=None))]
    fn evaluate_batch(
        &mut self,
        py: Python<'_>,
        shifts: &NativeBatch,
        masses_squared: &NativeBatch,
        scales_squared: &NativeBatch,
        parameters: &NativeBatch,
        s_matrix: Option<&NativeBatch>,
    ) -> PyResult<batch::ReductionBatch> {
        py.detach(|| {
            self.0.evaluate_batch(
                &shifts.0,
                &masses_squared.0,
                &scales_squared.0,
                &parameters.0,
                s_matrix.map(|b| &b.0),
            )
        })
        .map(batch::ReductionBatch)
        .map_err(error)
    }
}

fn spinors<T: Carrier>(
    momenta: &[[String; 4]],
    helicities: &[i32],
    kinds: &[String],
    bits: u32,
) -> PyResult<Vec<[DecimalComplex; 4]>> {
    if momenta.len() != kinds.len() || momenta.len() != helicities.len() {
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
    let momenta = momenta
        .iter()
        .map(|p| {
            let p = p
                .iter()
                .map(|x| T::parse_decimal(x, bits).map_err(error))
                .collect::<PyResult<Vec<_>>>()?;
            if p.iter().any(|x| !x.is_finite()) || p[0] <= p[0].zero() {
                return Err(error(
                    "massless states require finite physical momenta with positive energy",
                ));
            }
            Ok(p.try_into().unwrap_or_else(|_| unreachable!()))
        })
        .collect::<PyResult<Vec<[T; 4]>>>()?;
    Ok(physical_spinors(&momenta, helicities, kinds, &antiquarks)?
        .into_iter()
        .map(|row| row.map(|component| decimal_complex(component, bits)))
        .collect())
}

fn physical_spinors<T: Carrier>(
    momenta: &[[T; 4]],
    helicities: &[i32],
    kinds: &[String],
    antiquarks: &[bool],
) -> PyResult<Vec<[Complex<T>; 4]>> {
    let states = rusticol_core::__private::massless_dirac_batch(momenta, helicities, antiquarks)
        .map_err(error)?;
    Ok(states
        .into_iter()
        .zip(kinds)
        .map(|(x, kind)| {
            if kind == "u" || kind == "v_bar" {
                // Chiral Dirac adjoint: conjugation followed by gamma^0.
                [x[2].conj(), x[3].conj(), x[0].conj(), x[1].conj()]
            } else {
                x
            }
        })
        .collect())
}

/// Existing massless tree states for positive on-shell physical momenta.
/// Rows may mix helicities and kinds; all rows execute in one native call.
#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, arithmetic="double", precision=None))]
fn massless_spinors_decimal(
    py: Python<'_>,
    momenta: Vec<[String; 4]>,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    arithmetic: &str,
    precision: Option<u32>,
) -> PyResult<Vec<[DecimalComplex; 4]>> {
    let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
    py.detach(|| match arithmetic {
        Arithmetic::Double => spinors::<f64>(&momenta, &helicities, &kinds, bits),
        Arithmetic::DoubleDouble => spinors::<DoubleFloat>(&momenta, &helicities, &kinds, bits),
        Arithmetic::Arbitrary => spinors::<Float>(&momenta, &helicities, &kinds, bits),
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
/// (requirements addendum G; docs/user/symbolica-and-licensing.md).
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    portable::register(module)?;
    physical::register(module)?;
    virtual_runtime::register(module)?;
    batch::register(module)?;
    exact::register(module)?;
    states::register(module)?;
    module.add_class::<CoefficientEvaluator>()?;
    dual_weights::register(module)?;
    module.add_class::<PolynomialPlan>()?;
    v2::register(module)?;
    v2_dag::register(module)?;
    v2_executor::register(module)?;
    v2_generate::register(module)?;
    v2_colour::register(module)?;
    v2_lower::register(module)?;
    module.add_function(wrap_pyfunction!(massless_spinors_decimal, module)?)?;
    Ok(())
}
