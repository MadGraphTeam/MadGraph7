// SPDX-License-Identifier: 0BSD

//! Python entry point of the loop-kernel catalogue's polynomial stage
//! (`rusticol_core::loop_v2::generate::catalogue`, M-SM MS-GEN): the twin of
//! `loops/v2/catalogue.py::_polynomial_candidate` with the candidate's
//! signature, which stays the oracle.
use super::*;
use rusticol_core::loop_v2::generate::catalogue::{
    CandidateError, PolynomialJob, SignatureHeader, polynomial_candidates,
};
use rusticol_core::loop_v2::generate::rename::rename_symbols;

/// (loop_in, out, (q0, q1, q2, q3) exponents, block indices).
type EntryTuple = (i64, i64, (i64, i64, i64, i64), Vec<i64>);
/// (entries, exact canonical strings, input positions, canonical signature).
type CandidateTuple = (Vec<EntryTuple>, Vec<String>, Vec<usize>, String);
/// (refusal kind, detail): `not-linear`, `unexpected-monomial` (the key),
/// `not-finite` (the output index), `undeclared` (the names, `, `-joined).
type RefusalTuple = (String, String);
/// An available input: (canonical symbol, canonical JSON of its descriptor).
type InputItem = (String, String);
/// A job: (outputs as Symbolica text with namespaces, rows, per variant the
/// blocks, own available inputs), the variables canonical strings.
type JobItem = (
    Vec<String>,
    Vec<String>,
    Vec<Vec<Vec<String>>>,
    Vec<InputItem>,
);

/// Per job, per variant, the lowered candidate (None with no refusal: the
/// variant lowers to zero) or the oracle's refusal. `q` are the four
/// loop-momentum symbols, `shared` the available inputs that follow each
/// job's own and `header` the canonical JSON of the signatures' ABI and
/// contract kind. The GIL is released while they run.
#[pyfunction]
#[pyo3(signature = (jobs, q, shared, header, threads=None))]
fn loop_catalogue_polynomials(
    py: Python<'_>,
    jobs: Vec<JobItem>,
    q: Vec<String>,
    shared: Vec<InputItem>,
    header: (String, String),
    threads: Option<usize>,
) -> PyResult<Vec<Vec<(Option<CandidateTuple>, Option<RefusalTuple>)>>> {
    let header = SignatureHeader {
        abi: header.0,
        contract_kind: header.1,
    };
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
        .detach(|| polynomial_candidates(&jobs, &q, &shared, &header, threads))
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
                            candidate.signature,
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

/// `texts` (Symbolica's portable text) with the old names of `variables`
/// and of the called `heads` renamed, (old, new) pairs; the native half of
/// `assemble._canonical_stage`'s text renaming. The GIL is released.
#[pyfunction]
#[pyo3(signature = (texts, variables, heads, threads=None))]
fn loop_rename_symbols(
    py: Python<'_>,
    texts: Vec<String>,
    variables: Vec<(String, String)>,
    heads: Vec<(String, String)>,
    threads: Option<usize>,
) -> PyResult<Vec<String>> {
    py.detach(|| rename_symbols(&texts, &variables, &heads, threads))
        .map_err(error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loop_catalogue_polynomials, module)?)?;
    module.add_function(wrap_pyfunction!(loop_rename_symbols, module)?)
}
