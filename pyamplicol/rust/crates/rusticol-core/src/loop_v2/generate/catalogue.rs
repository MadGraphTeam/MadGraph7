// SPDX-License-Identifier: 0BSD

//! The polynomial stage of the loop-kernel catalogue in parallel (M-SM
//! MS-GEN, HANDOFF §0.4).
//!
//! The twin of `loops/v2/catalogue.py`'s `_polynomial_candidate` minus the
//! candidate's JSON signature: `_polynomial_coefficients` (the monomial
//! coefficients of a map linear in the open-loop current and in each block),
//! `canonical_expressions` (the expanded coefficients' canonical strings and
//! their finiteness check), `used_input_descriptors` (the inputs the
//! unexpanded coefficients read) and the undeclared-input check of
//! `prepared_catalog_builder._candidate_from_payload`. The Python functions
//! stay the oracle (`catalogue.native_catalogue_polynomials(False)`).
//!
//! The expressions cross as Symbolica canonical strings and this crate builds
//! the Symbolica revision of the Python wheel, so every operation below is
//! the oracle's own, in the oracle's order (`(0 + output).expand()`, the
//! coefficient list in `i16` exponents over the current, block and loop
//! momentum variables, the per-entry sums `0 + c1 + c2 + ...` in coefficient
//! order, their expansion and canonical strings), and every string is the
//! oracle's byte for byte. Jobs are independent: they run on a rayon pool
//! whose workers hold pyAmpliCol's Symbolica key, and results are returned in
//! job order, so they do not depend on the thread count.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use symbolica::atom::{Atom, AtomCore, AtomView, Symbol};
use symbolica::coefficient::CoefficientView;
use symbolica::parser::ParseSettings;

use crate::{RusticolError, RusticolResult};

/// One `_polynomial_candidate` call per variant on shared outputs: a loop
/// segment's fused and split forms read the same outputs with different
/// block variables.
#[derive(Clone, Debug, Default)]
pub struct PolynomialJob {
    /// Canonical strings of the kernel's output components.
    pub outputs: Vec<String>,
    /// Canonical strings of the open-loop current components (`rows`).
    pub rows: Vec<String>,
    /// Per variant, the canonical strings of each block's components (no
    /// block for a fused segment or a propagator numerator).
    pub variants: Vec<Vec<Vec<String>>>,
    /// Canonical symbols of the job's own available inputs; the shared
    /// inputs of the call follow them.
    pub available: Vec<String>,
}

/// `LoopMonomialEntry`: (loop_in, out, exponents of `q0..q3`, blocks).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MonomialEntry {
    pub loop_in: i64,
    pub out: i64,
    pub exponents: [i64; 4],
    pub blocks: Vec<i64>,
}

/// A lowered variant: its entries in sorted order, the canonical strings of
/// their coefficients and the positions (in the job's available inputs
/// followed by the shared ones) of the inputs the coefficients read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialCandidate {
    pub entries: Vec<MonomialEntry>,
    pub exact: Vec<String>,
    pub inputs: Vec<usize>,
}

/// The oracle's refusals, raised by the caller with the oracle's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateError {
    /// `loop kernel is not linear in the open-loop current and blocks`.
    NotLinear,
    /// `unexpected monomial {key}`.
    UnexpectedMonomial(String),
    /// Output `index` is not a finite exact expression.
    NotFinite(usize),
    /// The sorted names of the undeclared inputs.
    Undeclared(Vec<String>),
}

/// `None`: the variant lowers to zero.
pub type VariantResult = Result<Option<PolynomialCandidate>, CandidateError>;

fn parse(text: &str) -> RusticolResult<Atom> {
    // Canonical strings name every symbol with its namespace; the default
    // namespace is the Python wheel's (`E`).
    Atom::parse(text, "python", ParseSettings::default()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "loop catalogue: cannot parse a canonical expression: {error}"
        ))
    })
}

fn variable(text: &str) -> RusticolResult<Atom> {
    let atom = parse(text)?;
    if matches!(atom, Atom::Var(_)) {
        Ok(atom)
    } else {
        Err(RusticolError::invalid_argument(format!(
            "loop catalogue: {text} is not a variable"
        )))
    }
}

/// The exponents of a coefficient-list key over `variables` (by symbol), or
/// `None` when the key is not a monomial in them with natural exponents (the
/// oracle's `to_polynomial` then finds a variable outside the list).
fn degrees(key: AtomView<'_>, position: &HashMap<Symbol, usize>, count: usize) -> Option<Vec<i64>> {
    let mut out = vec![0i64; count];
    let factor = |factor: AtomView<'_>, out: &mut Vec<i64>| -> bool {
        match factor {
            AtomView::Var(v) => match position.get(&v.get_symbol()) {
                Some(&index) => {
                    out[index] += 1;
                    true
                }
                None => false,
            },
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                let (AtomView::Var(v), AtomView::Num(n)) = (base, exponent) else {
                    return false;
                };
                let CoefficientView::Natural(power, 1, 0, 1) = n.get_coeff_view() else {
                    return false;
                };
                match position.get(&v.get_symbol()) {
                    Some(&index) if power > 0 => {
                        out[index] += power;
                        true
                    }
                    _ => false,
                }
            }
            AtomView::Num(n) => n.get_coeff_view() == CoefficientView::Natural(1, 1, 0, 1),
            _ => false,
        }
    };
    let ok = match key {
        AtomView::Mul(m) => m.iter().all(|item| factor(item, &mut out)),
        other => factor(other, &mut out),
    };
    ok.then_some(out)
}

/// `_single_linear_index`: the position of the single 1 of degrees that
/// are linear (largest 1, sum 1).
fn single_linear_index(degrees: &[i64]) -> Option<i64> {
    if degrees.iter().copied().max() != Some(1) || degrees.iter().sum::<i64>() != 1 {
        return None;
    }
    degrees.iter().position(|&d| d == 1).map(|p| p as i64)
}

fn canonical_symbol(symbol: Symbol) -> String {
    Atom::var(symbol).to_canonical_string()
}

/// One variant of a job (`_polynomial_candidate` without its signature).
fn variant(
    outputs: &[Atom],
    rows: &[Atom],
    blocks: &[Vec<Atom>],
    q: &[Atom],
    available: &[&str],
) -> VariantResult {
    let variables: Vec<Atom> = rows
        .iter()
        .chain(blocks.iter().flatten())
        .chain(q)
        .cloned()
        .collect();
    let mut position = HashMap::with_capacity(variables.len());
    for (index, var) in variables.iter().enumerate() {
        if let Atom::Var(v) = var {
            position.entry(v.get_symbol()).or_insert(index);
        }
    }
    let mut slices = Vec::with_capacity(blocks.len());
    let mut start = rows.len();
    for block in blocks {
        slices.push(start..start + block.len());
        start += block.len();
    }
    let count = variables.len();
    let mut order: HashMap<MonomialEntry, usize> = HashMap::new();
    let mut sums: Vec<(MonomialEntry, Atom)> = Vec::new();
    for (out, expanded) in outputs.iter().enumerate() {
        for (key, coefficient) in expanded.coefficient_list::<i16>(&variables) {
            if coefficient.is_zero() {
                continue;
            }
            let Some(degrees) = degrees(key.as_view(), &position, count) else {
                return Err(CandidateError::UnexpectedMonomial(key.to_string()));
            };
            let row = single_linear_index(&degrees[..rows.len()]);
            let block_indices: Option<Vec<i64>> = slices
                .iter()
                .map(|slice| single_linear_index(&degrees[slice.clone()]))
                .collect();
            let (Some(row), Some(block_indices)) = (row, block_indices) else {
                return Err(CandidateError::NotLinear);
            };
            let exponents = [
                degrees[count - 4],
                degrees[count - 3],
                degrees[count - 2],
                degrees[count - 1],
            ];
            let entry = MonomialEntry {
                loop_in: row,
                out: out as i64,
                exponents,
                blocks: block_indices,
            };
            match order.get(&entry) {
                Some(&index) => {
                    let sum = &sums[index].1 + &coefficient;
                    sums[index].1 = sum;
                }
                None => {
                    order.insert(entry.clone(), sums.len());
                    sums.push((entry, Atom::num(0) + &coefficient));
                }
            }
        }
    }
    // The oracle drops the entries whose expansion vanishes and canonicalises
    // the rest by the same expansion (`value.expand()` both times).
    let mut kept: Vec<(MonomialEntry, Atom, Atom)> = sums
        .into_iter()
        .filter_map(|(entry, value)| {
            let expanded = value.expand();
            (!expanded.is_zero()).then_some((entry, value, expanded))
        })
        .collect();
    if kept.is_empty() {
        return Ok(None);
    }
    kept.sort_by(|a, b| a.0.cmp(&b.0));
    let mut exact = Vec::with_capacity(kept.len());
    for (index, (_, _, expanded)) in kept.iter().enumerate() {
        let canonical = expanded.to_canonical_string();
        let lowered = canonical.to_lowercase();
        if canonical.is_empty()
            || ["indeterminate", "nan", "infinity", "complexinf"]
                .iter()
                .any(|marker| lowered.contains(marker))
        {
            return Err(CandidateError::NotFinite(index));
        }
        exact.push(canonical);
    }
    // `used_input_descriptors` on the unexpanded sums.
    let mut used = HashSet::new();
    for (_, value, _) in &kept {
        used.extend(value.get_all_symbols(false));
    }
    let used: HashSet<String> = used.into_iter().map(canonical_symbol).collect();
    let inputs: Vec<usize> = available
        .iter()
        .enumerate()
        .filter(|(_, symbol)| used.contains(**symbol))
        .map(|(index, _)| index)
        .collect();
    // `_candidate_from_payload`: the exact outputs' non-constant symbols are
    // declared inputs.
    let declared: HashSet<&str> = inputs.iter().map(|&index| available[index]).collect();
    let mut symbols = HashSet::new();
    for (_, _, expanded) in &kept {
        symbols.extend(expanded.get_all_symbols(false));
    }
    let mut unbound: Vec<String> = symbols
        .into_iter()
        .filter(|&symbol| !Atom::var(symbol).is_constant())
        .map(canonical_symbol)
        .filter(|name| !declared.contains(name.as_str()))
        .collect();
    if !unbound.is_empty() {
        unbound.sort();
        return Err(CandidateError::Undeclared(unbound));
    }
    Ok(Some(PolynomialCandidate {
        entries: kept.into_iter().map(|(entry, _, _)| entry).collect(),
        exact,
        inputs,
    }))
}

fn job(
    input: &PolynomialJob,
    q: &[Atom],
    shared: &[String],
) -> RusticolResult<Vec<VariantResult>> {
    let outputs = input
        .outputs
        .iter()
        .map(|text| Ok((Atom::num(0) + &parse(text)?).expand()))
        .collect::<RusticolResult<Vec<_>>>()?;
    let rows = input
        .rows
        .iter()
        .map(|text| variable(text))
        .collect::<RusticolResult<Vec<_>>>()?;
    let available: Vec<&str> = input
        .available
        .iter()
        .chain(shared)
        .map(String::as_str)
        .collect();
    input
        .variants
        .iter()
        .map(|blocks| {
            let blocks = blocks
                .iter()
                .map(|block| block.iter().map(|text| variable(text)).collect())
                .collect::<RusticolResult<Vec<Vec<Atom>>>>()?;
            Ok(variant(&outputs, &rows, &blocks, q, &available))
        })
        .collect()
}

fn pool(threads: Option<usize>) -> RusticolResult<rayon::ThreadPool> {
    // Symbolica state is touched on the workers: they hold the key.
    let mut builder =
        rayon::ThreadPoolBuilder::new().start_handler(|_| crate::licence::unlock_thread());
    if let Some(threads) = threads {
        builder = builder.num_threads(threads.max(1));
    }
    builder
        .build()
        .map_err(|error| RusticolError::internal(error.to_string()))
}

/// Per job, per variant, the lowered candidate (`None`: zero) or the
/// oracle's refusal. `q` are the loop-momentum variables (the last four
/// variables of every variant) and `shared` the available inputs that follow
/// every job's own.
pub fn polynomial_candidates(
    jobs: &[PolynomialJob],
    q: &[String],
    shared: &[String],
    threads: Option<usize>,
) -> RusticolResult<Vec<Vec<VariantResult>>> {
    if q.len() != 4 {
        return Err(RusticolError::invalid_argument(
            "loop catalogue: four loop-momentum variables expected",
        ));
    }
    pool(threads)?.install(|| {
        jobs.par_iter()
            .map(|input| {
                let q = q
                    .iter()
                    .map(|text| variable(text))
                    .collect::<RusticolResult<Vec<_>>>()?;
                job(input, &q, shared)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn linear_map_coefficients_follow_the_oracle() {
        let _guard = crate::licence::unlock();
        let q = strings(&["msgen::q0", "msgen::q1", "msgen::q2", "msgen::q3"]);
        let job = PolynomialJob {
            outputs: strings(&[
                "msgen::g*msgen::r0*msgen::b0*msgen::q0+msgen::g*msgen::r1*msgen::b1",
                "msgen::r0*msgen::b1*(msgen::q1+msgen::m)-msgen::r0*msgen::b1*msgen::q1",
            ]),
            rows: strings(&["msgen::r0", "msgen::r1"]),
            variants: vec![vec![], vec![strings(&["msgen::b0", "msgen::b1"])]],
            available: strings(&["msgen::b0", "msgen::b1", "msgen::z"]),
        };
        let shared = strings(&["msgen::g", "msgen::m"]);
        let results = polynomial_candidates(&[job], &q, &shared, Some(2)).unwrap();
        let fused = results[0][0].clone().unwrap().unwrap();
        assert_eq!(fused.entries.len(), 3);
        // sorted (loop_in, out, exponents): r0 q0 of output 0 first
        assert_eq!(fused.entries[0].exponents, [1, 0, 0, 0]);
        assert_eq!((fused.entries[0].loop_in, fused.entries[0].out), (0, 0));
        assert_eq!((fused.entries[1].loop_in, fused.entries[1].out), (0, 1));
        assert_eq!(fused.exact[1], "msgen::b1*msgen::m");
        // b0, b1 (own inputs), g and m (shared); z is not read
        assert_eq!(fused.inputs, vec![0, 1, 3, 4]);
        let split = results[0][1].clone().unwrap().unwrap();
        assert!(split.entries.iter().all(|entry| entry.blocks.len() == 1));
        assert_eq!(split.inputs, vec![3, 4]);
        let zero = PolynomialJob {
            outputs: strings(&["msgen::r0*msgen::q0-msgen::q0*msgen::r0"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[zero], &q, &[], None).unwrap();
        assert_eq!(results[0][0], Ok(None));
        let quadratic = PolynomialJob {
            outputs: strings(&["msgen::r0^2"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[quadratic], &q, &[], None).unwrap();
        assert_eq!(results[0][0], Err(CandidateError::NotLinear));
        let undeclared = PolynomialJob {
            outputs: strings(&["msgen::r0*msgen::y"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[undeclared], &q, &[], None).unwrap();
        assert_eq!(
            results[0][0],
            Err(CandidateError::Undeclared(vec!["msgen::y".to_owned()]))
        );
    }
}
