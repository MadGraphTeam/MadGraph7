// SPDX-License-Identifier: 0BSD

//! The native loop module of `pyamplicol._rusticol`: the v2 engine's
//! generation helpers, executor, reducer and seed contraction, and the shared
//! batch/evaluator transport (the legacy per-diagram runtime was removed).

mod batch;
mod kinematics;
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
mod v2_induced;
mod v2_lower;
mod v2_seeds;
mod v2_stability;
use batch::NativeBatch;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use rusticol_loop_reduction::Complex;
use std::collections::{BTreeMap, BTreeSet};
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
        let _unlock = pyamplicol::unlock();
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
        let _unlock = pyamplicol::unlock();
        py.detach(|| self.0.evaluate_decimal(values, rows))
            .map_err(error)
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        let _unlock = pyamplicol::unlock();
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
        let _unlock = pyamplicol::unlock();
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
    let _unlock = pyamplicol::unlock();
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
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::tir::available()
}

/// The reduction backends of this native extension's v2 executor, as JSON
/// `{"ninjars": [carriers], "tirust": [carriers]}` (TIRust only in a build
/// with it): the capability query `loops.v2.reduction.native_backends`
/// reads (`runtime::backend_carriers`).
#[pyfunction]
fn reduction_backends() -> String {
    let _unlock = pyamplicol::unlock();
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
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::tir::capabilities(bits).map(|(propagators, rank)| {
        serde_json::json!({"max_propagators": propagators, "max_rank": rank}).to_string()
    })
}

/// The reduction-backend policy, the one definition of `"auto"` (m-lit
/// DECISIONS 2026-11-07 21:50 R7; `rusticol_loop_reduction::reduction_policy`)
/// as JSON: `{"backends", "carriers", "defaults": {kind: backend}, "routes":
/// {key: {"serves", "backends", "auto"}}}`, which `loops.v2.reduction` reads.
#[pyfunction]
fn reduction_policy() -> String {
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::reduction_policy::policy_json().to_string()
}

/// `reduction_policy::resolve` of a call: `requested` ("auto", "ninjars",
/// "tirust") for `contribution` at `carrier`, the native extension's served
/// backends `served` (backend -> carriers), the artifact's `admitted`
/// backends (None: every one) and the route (`serves`, the backends its
/// executor runs, those `"auto"` may take: None, every one it runs).
/// Returns `("ok", requested, effective, reason, fallback)` (`fallback`: the
/// per-point fallback backend, or None), or `("unavailable", "", "",
/// message, None)` (the typed `ReductionUnavailable`) or `("invalid", "",
/// "", message, None)` (a `ValueError`).
#[pyfunction]
#[pyo3(signature = (requested, contribution, carrier, served, admitted, serves, backends, auto))]
#[allow(clippy::too_many_arguments)]
fn reduction_resolve(
    requested: &str,
    contribution: &str,
    carrier: &str,
    served: BTreeMap<String, BTreeSet<String>>,
    admitted: Option<BTreeSet<String>>,
    serves: String,
    backends: BTreeSet<String>,
    auto: Option<BTreeSet<String>>,
) -> (&'static str, String, String, String, Option<String>) {
    let _unlock = pyamplicol::unlock();
    use rusticol_loop_reduction::reduction_policy::{Refusal, Route, resolve};
    let route = Route {
        serves,
        backends,
        auto,
    };
    match resolve(
        requested,
        contribution,
        carrier,
        &served,
        admitted.as_ref(),
        &route,
    ) {
        Ok(choice) => (
            "ok",
            choice.requested,
            choice.effective,
            choice.reason,
            choice.fallback,
        ),
        Err(Refusal::Unavailable(message)) => {
            ("unavailable", String::new(), String::new(), message, None)
        }
        Err(Refusal::Invalid(message)) => ("invalid", String::new(), String::new(), message, None),
    }
}

/// `reduction_policy::auto_default`: the backend `"auto"` takes by default
/// for `contribution` on an artifact admitting `admitted` (None: every
/// backend), or None for an unknown contribution.
#[pyfunction]
#[pyo3(signature = (contribution, admitted=None))]
fn reduction_auto_default(
    contribution: &str,
    admitted: Option<BTreeSet<String>>,
) -> Option<&'static str> {
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::reduction_policy::auto_default(contribution, admitted.as_ref())
}

/// TIRust's method-choice acceptance at the target accuracy `target`
/// (`v2_tir::acceptance_of`, the one calibrated mapping; a `ValueError`
/// unless `target` is finite and positive), or None in a build without
/// TIRust.
#[pyfunction]
fn tir_acceptance(target: f64) -> PyResult<Option<f64>> {
    let _unlock = pyamplicol::unlock();
    let target = v2::reduction_target(Some(target))?;
    #[cfg(feature = "tir")]
    {
        Ok(Some(rusticol_loop_reduction::v2_tir::acceptance_of(target)))
    }
    #[cfg(not(feature = "tir"))]
    {
        let _ = target;
        Ok(None)
    }
}

/// The texts of the per-point fallback and of the cross-backend check's
/// refusal (`reduction_policy::fallback_served`, `fallback_failed`,
/// `fallback_rejected`, `cross_refusal`): `kind` "served" (`a`, `b`,
/// `text`: the primary backend, the fallback and the primary's failure),
/// "failed" (`a`: the fallback, `text`: its reason), "rejected" (as
/// "served", with the call's `carrier`, the fallback guard's `estimate` and
/// the call's `target`, whose limit it names: the point the guard rejected,
/// m-lit lit-fallback-guard) or "cross" (`text`: the refusal).
#[pyfunction]
#[pyo3(signature = (kind, a, b, text, *, carrier="", estimate=f64::NAN, target=f64::NAN))]
fn reduction_text(
    kind: &str,
    a: &str,
    b: &str,
    text: &str,
    carrier: &str,
    estimate: f64,
    target: f64,
) -> PyResult<String> {
    let _unlock = pyamplicol::unlock();
    use rusticol_loop_reduction::reduction_policy as policy;
    match kind {
        "served" => Ok(policy::fallback_served(a, b, text)),
        "failed" => Ok(policy::fallback_failed(a, text)),
        "rejected" => Ok(policy::fallback_rejected(
            a, b, text, carrier, estimate, target,
        )),
        "cross" => Ok(policy::cross_refusal(text)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown reduction text {other:?}"
        ))),
    }
}

/// Whether the per-point fallback's guard accepts a served point whose
/// reduction-direction estimate is `estimate` at the call's `target`
/// (`reduction_policy::fallback_accepted`, the one rule: a number at most
/// the guard's limit, the target never below `FALLBACK_FLOOR`; m-lit
/// lit-fallback-guard).
#[pyfunction]
fn reduction_fallback_accepted(estimate: f64, target: f64) -> bool {
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::reduction_policy::fallback_accepted(estimate, target)
}

/// The per-point fallback guard's estimate of one point from the outputs
/// the call returns (`reduction_policy::fallback_estimate`, the one rule:
/// each output against its own magnitude, a vanishing one against the
/// largest returned output; m-lit lit-fallback-guard, DECISIONS 2026-11-09
/// 14:30 Q7): `outputs[o][s]` = (magnitude, error, scale) of sector s of
/// output o, `roundoff` the carrier's unit roundoff.
#[pyfunction]
fn reduction_fallback_estimate(outputs: Vec<Vec<(f64, f64, f64)>>, roundoff: f64) -> f64 {
    let _unlock = pyamplicol::unlock();
    use rusticol_loop_reduction::reduction_policy::{GuardSector, fallback_estimate};
    let outputs: Vec<Vec<GuardSector>> = outputs
        .into_iter()
        .map(|sectors| {
            sectors
                .into_iter()
                .map(|(magnitude, error, scale)| GuardSector {
                    magnitude,
                    error,
                    scale,
                })
                .collect()
        })
        .collect();
    fallback_estimate(&outputs, roundoff)
}

/// Whether the per-point fallback's guard runs in the carrier `carrier`
/// (`reduction_policy::fallback_guarded`: the extended carriers only; m-lit
/// lit-fallback-guard, USER 2026-11-09 ~11:53).
#[pyfunction]
fn reduction_fallback_guarded(carrier: &str) -> bool {
    let _unlock = pyamplicol::unlock();
    rusticol_loop_reduction::reduction_policy::fallback_guarded(carrier)
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
    module.add_function(wrap_pyfunction!(reduction_policy, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_resolve, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_auto_default, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_text, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_fallback_accepted, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_fallback_guarded, module)?)?;
    module.add_function(wrap_pyfunction!(reduction_fallback_estimate, module)?)?;
    module.add_function(wrap_pyfunction!(tir_acceptance, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::kinematics_policy, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::kinematics_applies, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::kinematics_reaches, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::kinematics_off_record, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::kinematics_notes, module)?)?;
    module.add_function(wrap_pyfunction!(kinematics::improve_kinematics, module)?)?;
    v2::register(module)?;
    v2_catalogue::register(module)?;
    v2_closure::register(module)?;
    v2_dag::register(module)?;
    v2_executor::register(module)?;
    v2_induced::register(module)?;
    v2_export::register(module)?;
    v2_generate::register(module)?;
    v2_colour::register(module)?;
    v2_lower::register(module)?;
    v2_seeds::register(module)?;
    v2_stability::register(module)?;
    model_proofs::register(module)?;
    Ok(())
}
