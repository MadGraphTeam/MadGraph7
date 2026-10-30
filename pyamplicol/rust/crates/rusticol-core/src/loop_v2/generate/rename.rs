// SPDX-License-Identifier: 0BSD

//! Symbol renaming on Symbolica's portable text, in parallel (M-SM MS-GEN).
//!
//! The native half of `loops/v2/assemble.py::_canonical_stage` for a stage
//! with function definitions: its outputs and definition bodies, printed by
//! the Python wheel (`repr`, every namespace but the default one), have the
//! stage's parameters, function arguments and function heads renamed onto
//! canonical symbols, and Python parses them back. A token is a maximal run
//! of name characters (alphanumerics, `_`, `:`, `{`, `}`); it is renamed
//! when it is exactly an old name, as a head when a `(` follows it and as a
//! variable otherwise. The `replace_multiple` passes of `_canonical_stage`
//! stay the oracle (`assemble.text_canonical_stage(False)`). No Symbolica
//! state is touched.

use rayon::prelude::*;
use std::collections::HashMap;

use crate::{RusticolError, RusticolResult};

fn is_name(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | ':' | '{' | '}')
}

fn rename(text: &str, variables: &HashMap<&str, &str>, heads: &HashMap<&str, &str>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(is_name) {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let end = rest.find(|c: char| !is_name(c)).unwrap_or(rest.len());
        let token = &rest[..end];
        let call = rest[end..].starts_with('(');
        let renamed = if call {
            heads.get(token)
        } else {
            variables.get(token)
        };
        out.push_str(renamed.copied().unwrap_or(token));
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// `texts` with the tokens of `variables` and `heads` (old name, new name)
/// renamed, in input order. Every name must be a run of name characters.
pub fn rename_symbols(
    texts: &[String],
    variables: &[(String, String)],
    heads: &[(String, String)],
    threads: Option<usize>,
) -> RusticolResult<Vec<String>> {
    for (old, new) in variables.iter().chain(heads) {
        if old.is_empty() || !old.chars().all(is_name) || !new.chars().all(is_name) {
            return Err(RusticolError::invalid_argument(format!(
                "symbol rename: {old:?} -> {new:?} is not a name token"
            )));
        }
    }
    let variables: HashMap<&str, &str> = variables
        .iter()
        .map(|(old, new)| (old.as_str(), new.as_str()))
        .collect();
    let heads: HashMap<&str, &str> = heads
        .iter()
        .map(|(old, new)| (old.as_str(), new.as_str()))
        .collect();
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = threads {
        builder = builder.num_threads(threads.max(1));
    }
    let pool = builder
        .build()
        .map_err(|error| RusticolError::internal(error.to_string()))?;
    Ok(pool.install(|| {
        texts
            .par_iter()
            .map(|text| rename(text, &variables, &heads))
            .collect()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items
            .iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect()
    }

    #[test]
    fn whole_tokens_heads_and_variables() {
        let variables = pairs(&[("ns::e1", "c::c0"), ("x", "c::c3")]);
        let heads = pairs(&[("ns::f", "c::c1"), ("ns::e12", "c::c9")]);
        let texts = vec![
            "ns::f(ns::e1,ns::f(ns::e12,2))*ns::e12+ns::e123-x^2*𝑖".to_owned(),
            "UFO::{}::x+ns::e1(3)+ns::ff(x)".to_owned(),
        ];
        let out = rename_symbols(&texts, &variables, &heads, Some(2)).unwrap();
        // e12 is a head only where it is called; e123 and ns::ff are other names
        assert_eq!(
            out[0],
            "c::c1(c::c0,c::c1(ns::e12,2))*ns::e12+ns::e123-c::c3^2*𝑖"
        );
        // a run with braces is one token; a variable called is not renamed
        assert_eq!(out[1], "UFO::{}::x+ns::e1(3)+ns::ff(c::c3)");
        assert!(rename_symbols(&texts, &pairs(&[("a.b", "c")]), &[], None).is_err());
    }
}
