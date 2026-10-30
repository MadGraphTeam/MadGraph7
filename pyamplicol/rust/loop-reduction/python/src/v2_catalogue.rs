// SPDX-License-Identifier: 0BSD

//! Python entry point of the loop-kernel catalogue's polynomial stage
//! (`rusticol_core::loop_v2::generate::catalogue`, M-SM MS-GEN): the twin of
//! `loops/v2/catalogue.py::_polynomial_candidate` without its signature,
//! which stays the oracle.
use super::*;
use rusticol_core::loop_v2::generate::catalogue::{
    CandidateError, PolynomialJob, polynomial_candidates,
};

/// (loop_in, out, (q0, q1, q2, q3) exponents, block indices).
type EntryTuple = (i64, i64, (i64, i64, i64, i64), Vec<i64>);
/// (entries, exact canonical strings, input positions).
type CandidateTuple = (Vec<EntryTuple>, Vec<String>, Vec<usize>);
/// (refusal kind, detail): `not-linear`, `unexpected-monomial` (the key),
/// `not-finite` (the output index), `undeclared` (the names, `, `-joined).
type RefusalTuple = (String, String);
/// A job: (outputs, rows, per variant the blocks, own available inputs),
/// every expression a canonical string.
type JobItem = (Vec<String>, Vec<String>, Vec<Vec<Vec<String>>>, Vec<String>);

/// Per job, per variant, the lowered candidate (None with no refusal: the
/// variant lowers to zero) or the oracle's refusal. `q` are the four
/// loop-momentum symbols and `shared` the available inputs that follow each
/// job's own. The GIL is released while they run.
#[pyfunction]
#[pyo3(signature = (jobs, q, shared, threads=None))]
fn loop_catalogue_polynomials(
    py: Python<'_>,
    jobs: Vec<JobItem>,
    q: Vec<String>,
    shared: Vec<String>,
    threads: Option<usize>,
) -> PyResult<Vec<Vec<(Option<CandidateTuple>, Option<RefusalTuple>)>>> {
    let jobs: Vec<PolynomialJob> = jobs
        .into_iter()
        .map(|(outputs, rows, variants, available)| PolynomialJob {
            outputs,
            rows,
            variants,
            available,
        })
        .collect();
    let results = py
        .detach(|| polynomial_candidates(&jobs, &q, &shared, threads))
        .map_err(error)?;
    Ok(results
        .into_iter()
        .map(|variants| {
            variants
                .into_iter()
                .map(|result| match result {
                    Ok(None) => (None, None),
                    Ok(Some(candidate)) => (
                        Some((
                            candidate
                                .entries
                                .into_iter()
                                .map(|entry| {
                                    let [a, b, c, d] = entry.exponents;
                                    (entry.loop_in, entry.out, (a, b, c, d), entry.blocks)
                                })
                                .collect(),
                            candidate.exact,
                            candidate.inputs,
                        )),
                        None,
                    ),
                    Err(refusal) => (
                        None,
                        Some(match refusal {
                            CandidateError::NotLinear => ("not-linear".into(), String::new()),
                            CandidateError::UnexpectedMonomial(key) => {
                                ("unexpected-monomial".into(), key)
                            }
                            CandidateError::NotFinite(index) => {
                                ("not-finite".into(), index.to_string())
                            }
                            CandidateError::Undeclared(names) => {
                                ("undeclared".into(), names.join(", "))
                            }
                        }),
                    ),
                })
                .collect()
        })
        .collect())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loop_catalogue_polynomials, module)?)
}
