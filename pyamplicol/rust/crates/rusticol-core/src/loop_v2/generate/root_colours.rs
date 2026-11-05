// SPDX-License-Identifier: 0BSD

//! Closed root colours of the counterterm and Born closures (M-FINAL
//! mf-8g-1b, A6 generation side): the native twin of `ct_sites.SiteClosure.
//! _closed`, which closes a vertex colour network on the colour words of the
//! blocks attached to its legs and reduces the product exactly
//! (`colour.reduce_network`).
//!
//! A closure asks for many closed colours that share their factors: one
//! vertex network per (UFO colour, attached fields) and one block network
//! per (block word, field colour, leg slot). They arrive as a
//! [`FactoredSpec`] (`native_seeds._Networks.network` encoding, one local
//! numbering over every factor); each content's factors are multiplied in
//! list order ([`networks::multiply`]: the vertex network, then the blocks'
//! words, as `_closed` does) and reduced on the process legs. Contents reduce
//! in parallel, each on its thread's [`Reducer`] (its Fierz memo), so every
//! result depends on its content only and equals the oracle's tensor.
//!
//! No float and no Symbolica licence call.

use super::networks::{self, Network, Reducer};
use super::seeds::{FactoredSpec, Tensor, parse_network};
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("root colours: {}", message.into()))
}

/// The reduced tensor of every content of `spec` (in its order) on the
/// process legs `legs`.
pub fn reduce_factored(spec: &FactoredSpec, legs: &[i64]) -> RusticolResult<Vec<Tensor>> {
    let factors = spec
        .factors
        .iter()
        .map(parse_network)
        .collect::<RusticolResult<Vec<Network>>>()?;
    spec.contents
        .par_iter()
        .map_init(
            || Reducer::new(legs),
            |reducer, (_key, ids)| {
                let parts = ids
                    .iter()
                    .map(|&f| {
                        factors
                            .get(f as usize)
                            .ok_or_else(|| invalid("a content refers to an unknown factor"))
                    })
                    .collect::<RusticolResult<Vec<&Network>>>()?;
                reducer.reduce(&networks::multiply(parts)).map_err(invalid)
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(text: &str) -> FactoredSpec {
        serde_json::from_str(text).expect("spec")
    }

    /// delta^{ab} (a two-letter trace) closed on blocks tr(1 a) and tr(2 b):
    /// equal to the reduction of the expanded product, for every content.
    #[test]
    fn factored_contents_equal_their_expanded_reduction() {
        let text = r#"{"factors": [
            [["1", "0", 0, 0, [], [[-3, -4]]]],
            [["1", "0", 0, 0, [], [[1, -3]]]],
            [["1", "0", 0, 0, [], [[2, -4]]]],
            [["0", "-1", 0, 1, [], [[-3, -4, -5]]], ["0", "1", 0, 1, [], [[-3, -5, -4]]]],
            [["1", "0", 0, 0, [], [[3, -5]]]]
        ], "contents": [[0, [0, 1, 2]], [1, [3, 1, 2, 4]]]}"#;
        let parsed = spec(text);
        let out = reduce_factored(&parsed, &[1, 2, 3]).expect("reduce");
        assert_eq!(out.len(), 2);
        let factors: Vec<Network> = parsed
            .factors
            .iter()
            .map(|f| parse_network(f).expect("factor"))
            .collect();
        for ((_key, ids), tensor) in parsed.contents.iter().zip(&out) {
            let network = networks::multiply(ids.iter().map(|&f| &factors[f as usize]));
            let expected = Reducer::new(&[1, 2, 3]).reduce(&network).expect("oracle");
            assert_eq!(tensor.normalization, expected.normalization);
            assert_eq!(tensor.terms, expected.terms);
            assert!(!tensor.terms.is_empty());
        }
    }

    #[test]
    fn unknown_factor_is_refused() {
        let text = r#"{"factors": [], "contents": [[0, [3]]]}"#;
        assert!(reduce_factored(&spec(text), &[1]).is_err());
    }
}
