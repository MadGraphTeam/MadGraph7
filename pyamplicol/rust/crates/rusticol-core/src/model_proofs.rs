// SPDX-License-Identifier: 0BSD

//! Kernel-equivalence proofs of the model compiler in parallel (WP3.2
//! EWI-3R, HANDOFF §0.4).
//!
//! The twin of `models/compiler_entry.py::_kernel_equivalence_proof`, which
//! stays its oracle: for one oriented kernel, given the canonical strings of
//! its coupling and of its components per input order (`(0, 1)` and the
//! swapped `(1, 0)`) and the component dimensions of its two inputs and its
//! output, the class digest (the sha256 of the least signature over the input
//! orders and overall signs), the sign and input order of that signature, and
//! the input-exchange factor when the inputs have equal dimensions and the
//! swapped components equal the direct ones up to a sign.
//!
//! The expressions cross as Symbolica canonical strings and this crate builds
//! the Symbolica revision of the Python wheel, so every canonical string, the
//! JSON signature (`json.dumps(sort_keys=True, separators=(",", ":"))` with
//! `ensure_ascii`'s escapes) and the digest are the oracle's byte for byte.
//! The proofs are independent: they run on a rayon pool whose workers hold
//! pyAmpliCol's Symbolica key (`crate::licence`), and the results are
//! returned in input order, so they do not depend on the thread count.
//!
//! `kernel_equivalence` also forms the proofs' keys (the twin of
//! `compiler_entry._kernel_keys_and_proofs`, the oracle): per kernel its
//! coupling with the derived couplings resolved and the colour-flow
//! normalisation, its components in the canonical kernel symbols (kind 0,
//! the sides exchanged for the input order `(1, 0)`) with the derived
//! couplings resolved, and the sha256 of their JSON; the proofs of the
//! distinct keys not already known follow.

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use symbolica::atom::{Atom, AtomCore};
use symbolica::id::Replacement;
use symbolica::parser::ParseSettings;

use crate::{RusticolError, RusticolResult};

/// One kernel's proof input (`_kernel_equivalence_proof`'s arguments).
#[derive(Clone, Debug)]
pub struct KernelProofInput {
    /// Component dimensions of the left input, the right input and the output.
    pub dimensions: [i64; 3],
    /// Canonical string of the kernel's resolved coupling.
    pub coupling: String,
    /// Canonical strings of the components per input order `(0, 1)`, `(1, 0)`.
    pub components: [Vec<String>; 2],
}

/// One kernel's proof (`_kernel_equivalence_proof`'s result).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelProof {
    /// sha256 (hex) of the least signature.
    pub class_digest: String,
    /// Overall sign of the least signature (the factor's real part).
    pub sign: i32,
    /// Input order of the least signature.
    pub input_order: [u8; 2],
    /// Sign relating the swapped components to the direct ones, if any.
    pub exchange: Option<i32>,
}

const ORDERS: [[u8; 2]; 2] = [[0, 1], [1, 0]];

fn parse(text: &str) -> RusticolResult<Atom> {
    // Canonical strings name every symbol with its namespace; the default
    // namespace is the Python wheel's (`E`).
    Atom::parse(text, "python", ParseSettings::default()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "kernel proof: cannot parse a canonical expression: {error}"
        ))
    })
}

/// `json.dumps(text)` (``ensure_ascii``): the quoted string with `"` and `\`
/// escaped, the short escapes of `\b \f \n \r \t`, every other character
/// outside printable ASCII as `\uXXXX` (lower-case hex), surrogate pairs above
/// the basic plane.
pub(crate) fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(character),
            _ => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

fn json_token(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    json_string(text, &mut out);
    out
}

/// The expanded oriented components (`(component * coupling).expand()`) and
/// their signed canonical strings, by the component's canonical string,
/// shared by the proofs of one cluster (`kernel_equivalence`: kernels with
/// the same coupling whose component lists are equal or each other's input
/// exchange, e.g. the two input orders of one vertex orientation), so each
/// is formed once per cluster.
#[derive(Default)]
struct Shared<'a> {
    oriented: HashMap<&'a str, Atom>,
    // (component, sign index) -> canonical string.
    strings: HashMap<(&'a str, usize), String>,
}

struct Proof<'a, 'm> {
    input: &'a KernelProofInput,
    coupling: Atom,
    // Formed lazily: the oracle forms every oriented component, but only the
    // candidates' compared components and, for inputs of equal dimensions,
    // the exchange test read them.
    shared: &'m mut Shared<'a>,
}

const SIGNS: [i64; 2] = [1, -1];

impl<'a> Proof<'a, '_> {
    fn component(&self, order: usize, index: usize) -> &'a str {
        let input: &'a KernelProofInput = self.input;
        input.components[order][index].as_str()
    }

    fn oriented(&mut self, order: usize, index: usize) -> RusticolResult<&Atom> {
        let component = self.component(order, index);
        if !self.shared.oriented.contains_key(component) {
            let expanded = (parse(component)? * &self.coupling).expand();
            self.shared.oriented.insert(component, expanded);
        }
        Ok(&self.shared.oriented[component])
    }

    fn component_string(
        &mut self,
        order: usize,
        sign: usize,
        index: usize,
    ) -> RusticolResult<&str> {
        let component = self.component(order, index);
        if !self.shared.strings.contains_key(&(component, sign)) {
            // The oracle's `(sign * oriented).expand()`, for the sign 1 too:
            // Symbolica's expansion is not idempotent (a second one can
            // expand a sum the first left in a factor).
            let text = (Atom::num(SIGNS[sign]) * self.oriented(order, index)?)
                .expand()
                .to_canonical_string();
            self.shared.strings.insert((component, sign), text);
        }
        Ok(self.shared.strings[&(component, sign)].as_str())
    }

    fn signature(&mut self, order: usize, sign: usize) -> RusticolResult<String> {
        let dimensions = self.input.dimensions;
        let input = ORDERS[order];
        let count = self.input.components[order].len();
        let mut out = String::from("{\"components\":[");
        for index in 0..count {
            if index > 0 {
                out.push(',');
            }
            let text = self.component_string(order, sign, index)?.to_owned();
            json_string(&text, &mut out);
        }
        let _ = write!(
            out,
            "],\"input_dimensions\":[{},{}],\"output_dimension\":{}}}",
            dimensions[input[0] as usize], dimensions[input[1] as usize], dimensions[2]
        );
        Ok(out)
    }

    /// Whether every swapped component equals `sign` times the direct one
    /// (`(swapped - sign * direct).expand() == 0`, component by component,
    /// stopping at the first that is not; equal atoms are zero at once).
    fn exchanges(&mut self, sign: i64) -> RusticolResult<bool> {
        for index in 0..self.input.components[0].len() {
            let direct = Atom::num(sign) * self.oriented(0, index)?;
            let swapped = self.oriented(1, index)?;
            if *swapped != direct && !(swapped - &direct).expand().is_zero() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn prove<'a>(input: &'a KernelProofInput, shared: &mut Shared<'a>) -> RusticolResult<KernelProof> {
    let mut proof = Proof {
        input,
        coupling: parse(&input.coupling)?,
        shared,
    };
    // Candidates in the oracle's order: per input order, the signs 1, -1.
    let candidates: Vec<(usize, usize)> = (0..2)
        .flat_map(|order| (0..2).map(move |sign| (order, sign)))
        .collect();
    let mut tied = candidates;
    let first = input.components[0].len();
    if input.components[1].len() == first {
        for index in 0..first {
            if tied.len() == 1 {
                break;
            }
            let mut tokens = Vec::with_capacity(tied.len());
            for &(order, sign) in &tied {
                tokens.push(json_token(proof.component_string(order, sign, index)?));
            }
            let least = tokens.iter().min().cloned().unwrap_or_default();
            tied = tied
                .iter()
                .zip(&tokens)
                .filter(|(_, token)| **token == least)
                .map(|(candidate, _)| *candidate)
                .collect();
        }
    }
    // `min(tied, key=signature)`: the first least signature.
    let mut winner: Option<((usize, usize), String)> = None;
    for &(order, sign) in &tied {
        let signature = proof.signature(order, sign)?;
        if winner.as_ref().is_none_or(|(_, best)| signature < *best) {
            winner = Some(((order, sign), signature));
        }
    }
    let ((order, sign), signature) =
        winner.ok_or_else(|| RusticolError::internal("kernel proof without candidates"))?;
    let digest = Sha256::digest(signature.as_bytes());
    let mut class_digest = String::with_capacity(64);
    for byte in digest {
        let _ = write!(class_digest, "{byte:02x}");
    }
    let mut exchange = None;
    if input.dimensions[0] == input.dimensions[1] && input.components[1].len() == first {
        for exchange_sign in SIGNS {
            if proof.exchanges(exchange_sign)? {
                exchange = Some(exchange_sign as i32);
                break;
            }
        }
    }
    Ok(KernelProof {
        class_digest,
        sign: SIGNS[sign] as i32,
        input_order: ORDERS[order],
        exchange,
    })
}

/// One kernel's key input (`_annotate_oriented_kernel_evaluation_equivalence`).
#[derive(Clone, Debug)]
pub struct KernelKeyInput {
    /// The kernel's kind (its input symbols' number).
    pub kind: i64,
    /// Canonical string of the kernel's coupling.
    pub coupling: String,
    /// Power of the colour-flow normalisation `2^(-power/2)` (0: none).
    pub normalization_power: i64,
    /// Canonical strings of the kernel's components.
    pub components: Vec<String>,
    /// (derived coupling symbol name, canonical string of its coupling), in
    /// the order of the kernel's runtime parameters.
    pub derived: Vec<(String, String)>,
    /// Component dimensions of the left input, the right input and the output.
    pub dimensions: [i64; 3],
}

fn tagged(name: &str) -> Option<(i64, bool, bool, i64)> {
    // `kernel_<kind>_<left|right>_[momentum_]<index>` (the tail of a name).
    let tail = name.rsplit("::").next()?;
    let rest = tail.strip_prefix("kernel_")?;
    let (kind, rest) = rest.split_once('_')?;
    let (left, rest) = if let Some(rest) = rest.strip_prefix("left_") {
        (true, rest)
    } else {
        (false, rest.strip_prefix("right_")?)
    };
    let (momentum, index) = match rest.strip_prefix("momentum_") {
        Some(index) => (true, index),
        None => (false, rest),
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    if !digits(kind) || !digits(index) {
        return None;
    }
    Some((kind.parse().ok()?, left, momentum, index.parse().ok()?))
}

fn kernel_symbol(prefix: &str, kind: i64, left: bool, momentum: bool, index: i64) -> String {
    let side = if left { "left" } else { "right" };
    if momentum {
        format!("{prefix}kernel_{kind}_{side}_momentum_{index}")
    } else {
        format!("{prefix}kernel_{kind}_{side}_{index}")
    }
}

/// `compiler_kernels._remap_kernel_symbols`: the input symbols of kernel
/// `kind` (named `prefix` + tail, the model's registry) renamed to kind 0,
/// the sides exchanged with `swap`. As in Python, a tail found under another
/// namespace names the registry symbol, which is then absent (no change).
fn remap(expression: &Atom, kind: i64, prefix: &str, swap: bool) -> RusticolResult<Atom> {
    let mut found = BTreeSet::new();
    for symbol in expression.get_all_symbols(false) {
        if let Some((symbol_kind, left, momentum, index)) = tagged(symbol.get_name())
            && symbol_kind == kind
        {
            found.insert((left, momentum, index));
        }
    }
    if found.is_empty() {
        return Ok(expression.clone());
    }
    let mut replacements = Vec::with_capacity(found.len());
    for (left, momentum, index) in found {
        let old = parse(&kernel_symbol(prefix, kind, left, momentum, index))?;
        let new = parse(&kernel_symbol(prefix, 0, left != swap, momentum, index))?;
        replacements.push(Replacement::new(old.to_pattern(), new.to_pattern()));
    }
    Ok(expression.replace_multiple(&replacements))
}

/// `compiler_kernels._replace_expression_symbols`: one replacement after the
/// other, in order.
fn resolve(expression: Atom, derived: &[(Atom, Atom)]) -> Atom {
    let mut result = expression;
    for (symbol, value) in derived {
        result = result.replace(symbol.to_pattern()).with(value.to_pattern());
    }
    result
}

fn key_of(input: &KernelKeyInput, prefix: &str) -> RusticolResult<(String, KernelProofInput)> {
    let derived = input
        .derived
        .iter()
        .map(|(symbol, value)| Ok((parse(symbol)?, parse(value)?)))
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut coupling = resolve(parse(&input.coupling)?, &derived);
    if input.normalization_power != 0 {
        coupling = coupling * parse(&format!("2^(-{}/2)", input.normalization_power))?;
    }
    let mut components: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    let parsed = input
        .components
        .iter()
        .map(|component| parse(component))
        .collect::<RusticolResult<Vec<_>>>()?;
    for (order, swap) in [(0, false), (1, true)] {
        for component in &parsed {
            components[order].push(
                resolve(remap(component, input.kind, prefix, swap)?, &derived)
                    .to_canonical_string(),
            );
        }
    }
    let coupling = coupling.to_canonical_string();
    // json.dumps([dimensions, coupling, *components], separators=(",", ":")).
    let [left, right, output] = input.dimensions;
    let mut json = format!("[[{left},{right},{output}],");
    json_string(&coupling, &mut json);
    for strings in &components {
        json.push_str(",[");
        for (index, text) in strings.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            json_string(text, &mut json);
        }
        json.push(']');
    }
    json.push(']');
    let digest = Sha256::digest(json.as_bytes());
    let mut key = String::with_capacity(64);
    for byte in digest {
        let _ = write!(key, "{byte:02x}");
    }
    Ok((
        key,
        KernelProofInput {
            dimensions: input.dimensions,
            coupling,
            components,
        },
    ))
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

/// Per kernel its proof key, and the proofs of the distinct keys of
/// `inputs` not in `known`, in first-occurrence order (`prefix`: the model's
/// kernel symbol namespace, e.g. `model_sm::compiler::`).
pub fn kernel_equivalence(
    inputs: &[KernelKeyInput],
    prefix: &str,
    known: &HashSet<String>,
    threads: Option<usize>,
) -> RusticolResult<(Vec<String>, Vec<(String, KernelProof)>)> {
    let pool = pool(threads)?;
    let keyed = pool.install(|| {
        inputs
            .par_iter()
            .map(|input| key_of(input, prefix))
            .collect::<RusticolResult<Vec<_>>>()
    })?;
    let mut seen = HashSet::new();
    let mut missing = Vec::new();
    let mut keys = Vec::with_capacity(keyed.len());
    for (key, item) in keyed {
        if !known.contains(&key) && seen.insert(key.clone()) {
            missing.push((key.clone(), item));
        }
        keys.push(key);
    }
    // Clusters of the missing proofs whose kernels have the same coupling and
    // equal or exchanged component lists (the two input orders of a vertex
    // orientation): one task each, sharing their expanded components and
    // canonical strings (`Shared`); the proofs are returned in first-occurrence
    // order, so they depend neither on the clusters nor on the thread count.
    let mut cluster_of: HashMap<(&str, &[String], &[String]), usize> = HashMap::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for (position, (_, item)) in missing.iter().enumerate() {
        let [direct, exchanged] = &item.components;
        let (low, high) = if direct <= exchanged {
            (direct, exchanged)
        } else {
            (exchanged, direct)
        };
        let next = clusters.len();
        let cluster = *cluster_of
            .entry((item.coupling.as_str(), low.as_slice(), high.as_slice()))
            .or_insert(next);
        if cluster == next {
            clusters.push(Vec::new());
        }
        clusters[cluster].push(position);
    }
    let proved = pool.install(|| {
        clusters
            .par_iter()
            .map(|members| {
                let mut shared = Shared::default();
                members
                    .iter()
                    .map(|&position| {
                        prove(&missing[position].1, &mut shared).map(|proof| (position, proof))
                    })
                    .collect::<RusticolResult<Vec<_>>>()
            })
            .collect::<RusticolResult<Vec<_>>>()
    })?;
    let mut proofs: Vec<Option<KernelProof>> = vec![None; missing.len()];
    for (position, proof) in proved.into_iter().flatten() {
        proofs[position] = Some(proof);
    }
    let proofs = missing
        .into_iter()
        .zip(proofs)
        .map(|((key, _), proof)| (key, proof.expect("every cluster member is proved")))
        .collect();
    Ok((keys, proofs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_match_python_ensure_ascii() {
        assert_eq!(json_token("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_token("\n\t\u{1}\u{7f}"), "\"\\n\\t\\u0001\\u007f\"");
        assert_eq!(json_token("é𝑖"), "\"\\u00e9\\ud835\\udc56\"");
    }

    #[test]
    fn kernel_symbol_tails() {
        assert_eq!(
            tagged("model_m::compiler::kernel_12_left_momentum_3"),
            Some((12, true, true, 3))
        );
        assert_eq!(
            tagged("model_m::compiler::kernel_0_right_7"),
            Some((0, false, false, 7))
        );
        assert_eq!(tagged("model_m::compiler::kernel_1_up_7"), None);
        assert_eq!(tagged("model_m::compiler::kernel_x_left_7"), None);
        assert_eq!(tagged("model_m::compiler::kernel_1_left_"), None);
    }
}
