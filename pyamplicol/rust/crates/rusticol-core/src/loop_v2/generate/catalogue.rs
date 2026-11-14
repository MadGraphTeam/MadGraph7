// SPDX-License-Identifier: 0BSD

//! The polynomial stage of the loop-kernel catalogue in parallel (M-SM
//! MS-GEN, HANDOFF §0.4).
//!
//! The twin of `loops/v2/catalogue.py`'s `_polynomial_candidate`:
//! `_polynomial_coefficients` (the monomial coefficients of a map linear in
//! the open-loop current and in each block), `canonical_expressions` (the
//! expanded coefficients' canonical strings and their finiteness check),
//! `used_input_descriptors` (the inputs the unexpanded coefficients read)
//! and `prepared_catalog_builder._candidate_from_payload` (the
//! undeclared-input check and the candidate's signature, the sha256 of its
//! canonical JSON). The Python functions stay the oracle
//! (`catalogue.native_catalogue_polynomials(False)`).
//!
//! The outputs cross as Symbolica's portable text (`repr`, every symbol with
//! its namespace) and the variables and inputs as canonical strings; this
//! crate builds the Symbolica revision of the Python wheel, so every
//! operation below is the oracle's own, in the oracle's order
//! (`(0 + output).expand()`, the
//! coefficient list in `i16` exponents over the current, block and loop
//! momentum variables, the per-entry sums `0 + c1 + c2 + ...` in coefficient
//! order, their expansion and canonical strings), and every string is the
//! oracle's byte for byte. Jobs are independent: they run on a rayon pool
//! whose workers hold pyAmpliCol's Symbolica key, and results are returned in
//! job order, so they do not depend on the thread count.

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use symbolica::atom::{Atom, AtomCore, AtomView, Symbol};
use symbolica::coefficient::CoefficientView;
use symbolica::parser::ParseSettings;

use crate::model_proofs::json_string;
use crate::{RusticolError, RusticolResult};

/// One `_polynomial_candidate` call per variant on shared outputs: a loop
/// segment's fused and split forms read the same outputs with different
/// block variables.
#[derive(Clone, Debug, Default)]
pub struct PolynomialJob {
    /// The kernel's output components (Symbolica text with namespaces).
    pub outputs: Vec<String>,
    /// Canonical strings of the open-loop current components (`rows`).
    pub rows: Vec<String>,
    /// Per variant, the canonical strings of each block's components (no
    /// block for a fused segment or a propagator numerator).
    pub variants: Vec<Vec<Vec<String>>>,
    /// The job's own available inputs (canonical symbol, the canonical JSON
    /// of the descriptor); the shared inputs of the call follow them.
    pub available: Vec<(String, String)>,
}

/// The fixed fields of the candidates' signature payload: the canonical
/// JSON of the ABI and of the contract kind (`json.dumps` of the strings).
#[derive(Clone, Debug, Default)]
pub struct SignatureHeader {
    pub abi: String,
    pub contract_kind: String,
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
/// their coefficients, the positions (in the job's available inputs
/// followed by the shared ones) of the inputs the coefficients read and the
/// candidate's canonical signature (sha256 hex).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialCandidate {
    pub entries: Vec<MonomialEntry>,
    pub exact: Vec<String>,
    pub inputs: Vec<usize>,
    pub signature: String,
}

impl MonomialEntry {
    /// `LoopMonomialEntry.label`.
    pub fn label(&self) -> String {
        let mut out = format!("in{}", self.loop_in);
        if !self.blocks.is_empty() {
            out.push_str(":b");
            for (index, block) in self.blocks.iter().enumerate() {
                if index > 0 {
                    out.push('.');
                }
                let _ = write!(out, "{block}");
            }
        }
        let [a, b, c, d] = self.exponents;
        let _ = write!(out, ":out{}:q{a}.{b}.{c}.{d}", self.out);
        out
    }
}

/// `_candidate_from_payload`'s signature: the sha256 of
/// `canonical_json({"abi", "contract_kind", "inputs", "outputs",
/// "output_layout"})` (sorted keys, no spaces, ASCII escapes).
fn signature(
    header: &SignatureHeader,
    inputs: &[&str],
    layout: &[String],
    exact: &[String],
) -> String {
    let size = exact.iter().map(String::len).sum::<usize>()
        + inputs.iter().map(|item| item.len()).sum::<usize>()
        + 64 * layout.len()
        + 256;
    let mut text = String::with_capacity(size);
    text.push_str("{\"abi\":");
    text.push_str(&header.abi);
    text.push_str(",\"contract_kind\":");
    text.push_str(&header.contract_kind);
    text.push_str(",\"inputs\":[");
    for (index, input) in inputs.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        text.push_str(input);
    }
    text.push_str("],\"output_layout\":[");
    for (index, label) in layout.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        json_string(label, &mut text);
    }
    text.push_str("],\"outputs\":[");
    for (index, value) in exact.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        json_string(value, &mut text);
    }
    text.push_str("]}");
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
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
    available: &[(&str, &str)],
    header: &SignatureHeader,
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
        .filter(|(_, (symbol, _))| used.contains(*symbol))
        .map(|(index, _)| index)
        .collect();
    // `_candidate_from_payload`: the exact outputs' non-constant symbols are
    // declared inputs.
    let declared: HashSet<&str> = inputs.iter().map(|&index| available[index].0).collect();
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
    let entries: Vec<MonomialEntry> = kept.into_iter().map(|(entry, _, _)| entry).collect();
    let layout: Vec<String> = entries.iter().map(MonomialEntry::label).collect();
    let fragments: Vec<&str> = inputs.iter().map(|&index| available[index].1).collect();
    let signature = signature(header, &fragments, &layout, &exact);
    Ok(Some(PolynomialCandidate {
        entries,
        exact,
        inputs,
        signature,
    }))
}

fn job(
    input: &PolynomialJob,
    q: &[Atom],
    shared: &[(String, String)],
    header: &SignatureHeader,
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
    let available: Vec<(&str, &str)> = input
        .available
        .iter()
        .chain(shared)
        .map(|(symbol, json)| (symbol.as_str(), json.as_str()))
        .collect();
    input
        .variants
        .iter()
        .map(|blocks| {
            let blocks = blocks
                .iter()
                .map(|block| block.iter().map(|text| variable(text)).collect())
                .collect::<RusticolResult<Vec<Vec<Atom>>>>()?;
            Ok(variant(&outputs, &rows, &blocks, q, &available, header))
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
/// variables of every variant), `shared` the available inputs that follow
/// every job's own and `header` the signatures' fixed fields.
pub fn polynomial_candidates(
    jobs: &[PolynomialJob],
    q: &[String],
    shared: &[(String, String)],
    header: &SignatureHeader,
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
                job(input, &q, shared, header)
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

    /// Available inputs by their canonical strings (`descriptor.symbol`).
    fn inputs(items: &[&str]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|item| {
                let symbol = parse(item).unwrap().to_canonical_string();
                let json = format!("{{\"symbol\":\"{symbol}\"}}");
                (symbol, json)
            })
            .collect()
    }

    #[test]
    fn labels_and_signatures_follow_the_oracle() {
        let entry = MonomialEntry {
            loop_in: 2,
            out: 1,
            exponents: [1, 0, 0, 3],
            blocks: vec![0, 4],
        };
        assert_eq!(entry.label(), "in2:b0.4:out1:q1.0.0.3");
        let plain = MonomialEntry {
            blocks: vec![],
            ..entry
        };
        assert_eq!(plain.label(), "in2:out1:q1.0.0.3");
        // hashlib.sha256(b'{"abi":"a","contract_kind":"k","inputs":[{"x":1}],
        // "output_layout":["l"],"outputs":["\\u00e9"]}').hexdigest()
        let header = SignatureHeader {
            abi: "\"a\"".into(),
            contract_kind: "\"k\"".into(),
        };
        let text = "{\"abi\":\"a\",\"contract_kind\":\"k\",\"inputs\":[{\"x\":1}],\"output_layout\":[\"l\"],\"outputs\":[\"\\u00e9\"]}";
        let digest = Sha256::digest(text.as_bytes());
        let expected: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            signature(
                &header,
                &["{\"x\":1}"],
                &["l".to_owned()],
                &["é".to_owned()]
            ),
            expected
        );
    }

    #[test]
    fn linear_map_coefficients_follow_the_oracle() {
        let _guard = crate::licence::unlock();
        let header = SignatureHeader::default();
        let q = strings(&["msgen::q0", "msgen::q1", "msgen::q2", "msgen::q3"]);
        let job = PolynomialJob {
            outputs: strings(&[
                "msgen::g*msgen::r0*msgen::b0*msgen::q0+msgen::g*msgen::r1*msgen::b1",
                "msgen::r0*msgen::b1*(msgen::q1+msgen::m)-msgen::r0*msgen::b1*msgen::q1",
            ]),
            rows: strings(&["msgen::r0", "msgen::r1"]),
            variants: vec![vec![], vec![strings(&["msgen::b0", "msgen::b1"])]],
            available: inputs(&["msgen::b0", "msgen::b1", "msgen::z"]),
        };
        let shared = inputs(&["msgen::g", "msgen::m"]);
        let results = polynomial_candidates(&[job], &q, &shared, &header, Some(2)).unwrap();
        let fused = results[0][0].clone().unwrap().unwrap();
        assert_eq!(fused.entries.len(), 3);
        // sorted (loop_in, out, exponents): r0 q0 of output 0 first
        assert_eq!(fused.entries[0].exponents, [1, 0, 0, 0]);
        assert_eq!((fused.entries[0].loop_in, fused.entries[0].out), (0, 0));
        assert_eq!((fused.entries[1].loop_in, fused.entries[1].out), (0, 1));
        assert_eq!(
            fused.exact[1],
            parse("msgen::b1*msgen::m").unwrap().to_canonical_string()
        );
        // b0, b1 (own inputs), g and m (shared); z is not read
        assert_eq!(fused.inputs, vec![0, 1, 3, 4]);
        let split = results[0][1].clone().unwrap().unwrap();
        assert!(split.entries.iter().all(|entry| entry.blocks.len() == 1));
        assert_eq!(split.inputs, vec![3, 4]);
        assert_ne!(split.signature, fused.signature);
        let zero = PolynomialJob {
            outputs: strings(&["msgen::r0*msgen::q0-msgen::q0*msgen::r0"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[zero], &q, &[], &header, None).unwrap();
        assert_eq!(results[0][0], Ok(None));
        let quadratic = PolynomialJob {
            outputs: strings(&["msgen::r0^2"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[quadratic], &q, &[], &header, None).unwrap();
        assert_eq!(results[0][0], Err(CandidateError::NotLinear));
        let undeclared = PolynomialJob {
            outputs: strings(&["msgen::r0*msgen::y"]),
            rows: strings(&["msgen::r0"]),
            variants: vec![vec![]],
            available: vec![],
        };
        let results = polynomial_candidates(&[undeclared], &q, &[], &header, None).unwrap();
        assert_eq!(
            results[0][0],
            Err(CandidateError::Undeclared(vec![
                parse("msgen::y").unwrap().to_canonical_string()
            ]))
        );
    }
}
