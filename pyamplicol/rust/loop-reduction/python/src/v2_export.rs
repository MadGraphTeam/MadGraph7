// SPDX-License-Identifier: 0BSD

//! Python entry points of the export's native parse stages (M-FINAL
//! MF-RUST): the exact kernels' term reading
//! (`rusticol_core::loop_v2::generate::kernel_terms`, the twin of
//! `reference_executor._kernel_terms`, which stays the oracle) and the
//! symbol renaming of a Symbolica export
//! (`rusticol_core::loop_v2::generate::state_rename`, the parameter
//! program's canonical stage; `assemble._canonical_stage` stays the oracle).
use super::*;
use rusticol_core::loop_v2::generate::kernel_terms::{
    Base, KernelTermsJob, TermReading, kernel_term_readings,
};
use rusticol_core::loop_v2::generate::state_rename::rename_export;

/// (numerator, denominator) of an exact rational, decimal.
type RationalTuple = (String, String);
/// A model factor: ("param", name, []), ("prime", p, []), ("pi", "", []) or
/// ("expr", canonical string, (symbol, parameter name) pairs).
type BaseTuple = (&'static str, String, Vec<(String, String)>);
/// ((real, imaginary) coefficient, (factor, exponent) powers, positive
/// factors, (input position, power) monomial).
type TermTuple = (
    (RationalTuple, RationalTuple),
    Vec<(BaseTuple, RationalTuple)>,
    Vec<BaseTuple>,
    Vec<(usize, i64)>,
);
/// A kernel: (output canonical strings, (input symbol, position) of the
/// kinematic inputs, (input symbol, parameter name) of the model inputs).
type KernelItem = (Vec<String>, Vec<(String, usize)>, Vec<(String, String)>);

fn rational(value: &Rational) -> RationalTuple {
    (
        value.numerator().to_string(),
        value.denominator().to_string(),
    )
}

fn base(value: Base) -> BaseTuple {
    match value {
        Base::Param(name) => ("param", name, Vec::new()),
        Base::Prime(prime) => ("prime", prime.to_string(), Vec::new()),
        Base::Pi => ("pi", String::new(), Vec::new()),
        Base::Expr(text, pairs) => ("expr", text, pairs),
    }
}

fn term(reading: TermReading) -> TermTuple {
    (
        (
            rational(&reading.coefficient.0),
            rational(&reading.coefficient.1),
        ),
        reading
            .powers
            .into_iter()
            .map(|(b, exponent)| (base(b), rational(&exponent)))
            .collect(),
        reading.positive.into_iter().map(base).collect(),
        reading.monomial,
    )
}

/// Per kernel, the term readings of every output (None with the oracle's
/// `NotImplementedError` message otherwise). The GIL is released while they
/// run.
#[pyfunction]
#[pyo3(signature = (kernels, threads=None))]
fn loop_kernel_terms(
    py: Python<'_>,
    kernels: Vec<KernelItem>,
    threads: Option<usize>,
) -> PyResult<Vec<(Option<Vec<Vec<TermTuple>>>, Option<String>)>> {
    let jobs: Vec<KernelTermsJob> = kernels
        .into_iter()
        .map(|(expressions, positions, parameters)| KernelTermsJob {
            expressions,
            positions,
            parameters,
        })
        .collect();
    let results = py
        .detach(|| kernel_term_readings(&jobs, threads))
        .map_err(error)?;
    Ok(results
        .into_iter()
        .map(|result| match result {
            Ok(outputs) => (
                Some(
                    outputs
                        .into_iter()
                        .map(|terms| terms.into_iter().map(term).collect())
                        .collect(),
                ),
                None,
            ),
            Err(message) => (None, Some(message)),
        })
        .collect())
}

/// Rewrite the Symbolica export file `source` (`Expression.save`) into
/// `target` with the symbols of `renames` ((old, new) fully qualified names)
/// renamed in its symbol table; `Expression.load(target)` then reads the
/// renamed expression. Returns (renamed entries, entries). The GIL is
/// released.
#[pyfunction]
#[pyo3(signature = (source, target, renames, quality=0))]
fn loop_rename_export(
    py: Python<'_>,
    source: std::path::PathBuf,
    target: std::path::PathBuf,
    renames: Vec<(String, String)>,
    quality: u32,
) -> PyResult<(usize, usize)> {
    py.detach(|| rename_export(&source, &target, &renames, quality))
        .map(|counts| (counts.renamed, counts.symbols))
        .map_err(error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loop_kernel_terms, module)?)?;
    module.add_function(wrap_pyfunction!(loop_rename_export, module)?)
}
