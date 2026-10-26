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

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use symbolica::atom::{Atom, AtomCore};
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
    // Canonical strings name every symbol with its namespace: the default
    // namespace is never used.
    Atom::parse(text, "pyamplicol_model_proofs", ParseSettings::default()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "kernel proof: cannot parse a canonical expression: {error}"
        ))
    })
}

/// `json.dumps(text)` (``ensure_ascii``): the quoted string with `"` and `\`
/// escaped, the short escapes of `\b \f \n \r \t`, every other character
/// outside printable ASCII as `\uXXXX` (lower-case hex), surrogate pairs above
/// the basic plane.
fn json_string(text: &str, out: &mut String) {
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

struct Proof<'a> {
    input: &'a KernelProofInput,
    oriented: [Vec<Atom>; 2],
    // (order, sign index, component) -> canonical string, formed lazily.
    strings: std::collections::HashMap<(usize, usize, usize), String>,
}

const SIGNS: [i64; 2] = [1, -1];

impl Proof<'_> {
    fn component_string(&mut self, order: usize, sign: usize, index: usize) -> &str {
        let oriented = &self.oriented;
        self.strings
            .entry((order, sign, index))
            .or_insert_with(|| {
                (Atom::num(SIGNS[sign]) * &oriented[order][index])
                    .expand()
                    .to_canonical_string()
            })
            .as_str()
    }

    fn signature(&mut self, order: usize, sign: usize) -> String {
        let dimensions = self.input.dimensions;
        let input = ORDERS[order];
        let count = self.oriented[order].len();
        let mut out = String::from("{\"components\":[");
        for index in 0..count {
            if index > 0 {
                out.push(',');
            }
            let text = self.component_string(order, sign, index).to_owned();
            json_string(&text, &mut out);
        }
        let _ = write!(
            out,
            "],\"input_dimensions\":[{},{}],\"output_dimension\":{}}}",
            dimensions[input[0] as usize], dimensions[input[1] as usize], dimensions[2]
        );
        out
    }
}

fn prove(input: &KernelProofInput) -> RusticolResult<KernelProof> {
    let coupling = parse(&input.coupling)?;
    let mut oriented: [Vec<Atom>; 2] = [Vec::new(), Vec::new()];
    for (order, components) in input.components.iter().enumerate() {
        for component in components {
            oriented[order].push((parse(component)? * &coupling).expand());
        }
    }
    let mut proof = Proof {
        input,
        oriented,
        strings: Default::default(),
    };
    // Candidates in the oracle's order: per input order, the signs 1, -1.
    let candidates: Vec<(usize, usize)> = (0..2)
        .flat_map(|order| (0..2).map(move |sign| (order, sign)))
        .collect();
    let mut tied = candidates;
    let first = proof.oriented[0].len();
    if proof.oriented[1].len() == first {
        for index in 0..first {
            if tied.len() == 1 {
                break;
            }
            let tokens: Vec<String> = tied
                .iter()
                .map(|&(order, sign)| json_token(proof.component_string(order, sign, index)))
                .collect();
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
        let signature = proof.signature(order, sign);
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
    if input.dimensions[0] == input.dimensions[1] {
        let [direct, swapped] = &proof.oriented;
        for exchange_sign in SIGNS {
            if direct.len() == swapped.len()
                && direct.iter().zip(swapped).all(|(direct, swapped)| {
                    (swapped - &(Atom::num(exchange_sign) * direct))
                        .expand()
                        .is_zero()
                })
            {
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

/// The proofs of `inputs`, in input order, on `threads` workers (`None`: the
/// rayon default, `RAYON_NUM_THREADS` or the machine's parallelism).
pub fn kernel_equivalence_proofs(
    inputs: &[KernelProofInput],
    threads: Option<usize>,
) -> RusticolResult<Vec<KernelProof>> {
    for input in inputs {
        if input.components[0].len() != input.components[1].len() {
            return Err(RusticolError::invalid_argument(
                "kernel proof: the input orders have different component counts",
            ));
        }
    }
    // Symbolica state is touched on the workers: they hold the key.
    let mut builder =
        rayon::ThreadPoolBuilder::new().start_handler(|_| crate::licence::unlock_thread());
    if let Some(threads) = threads {
        builder = builder.num_threads(threads.max(1));
    }
    let pool = builder
        .build()
        .map_err(|error| RusticolError::internal(error.to_string()))?;
    pool.install(|| inputs.par_iter().map(prove).collect())
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
}
