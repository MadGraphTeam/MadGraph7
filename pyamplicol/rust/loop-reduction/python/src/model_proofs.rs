// SPDX-License-Identifier: 0BSD

//! Python entry point of the model compiler's parallel kernel-equivalence
//! keys and proofs (`rusticol_core::model_proofs`, WP3.2 EWI-3R): the twin of
//! `models/compiler_entry.py::_kernel_keys_and_proofs`, which stays the
//! oracle.
use super::*;
use rusticol_core::model_proofs::{KernelKeyInput, kernel_equivalence};
use std::collections::HashSet;

/// (class digest, sign, input order, input-exchange sign or None).
type ProofResult = (String, i32, (u8, u8), Option<i32>);

/// A kernel's key input: (kind, coupling, colour-flow normalisation power,
/// components, [(derived coupling symbol name, its coupling)], dimensions),
/// every expression a canonical string.
type KeyItem = (
    i64,
    String,
    i64,
    Vec<String>,
    Vec<(String, String)>,
    (i64, i64, i64),
);

/// Per kernel of `items` (`KeyItem`) its proof key, and the proofs
/// (`ProofResult`) of the distinct keys not in `known` by key, in
/// first-occurrence order; `prefix` is the model's kernel symbol namespace
/// (e.g. `model_sm::compiler::`). The GIL is released while they run.
#[pyfunction]
#[pyo3(signature = (items, prefix, known, threads=None))]
fn model_kernel_equivalence(
    py: Python<'_>,
    items: Vec<KeyItem>,
    prefix: String,
    known: HashSet<String>,
    threads: Option<usize>,
) -> PyResult<(Vec<String>, Vec<(String, ProofResult)>)> {
    let inputs: Vec<KernelKeyInput> = items
        .into_iter()
        .map(
            |(kind, coupling, normalization_power, components, derived, (left, right, output))| {
                KernelKeyInput {
                    kind,
                    coupling,
                    normalization_power,
                    components,
                    derived,
                    dimensions: [left, right, output],
                }
            },
        )
        .collect();
    let (keys, proofs) = py
        .detach(|| kernel_equivalence(&inputs, &prefix, &known, threads))
        .map_err(error)?;
    Ok((
        keys,
        proofs
            .into_iter()
            .map(|(key, proof)| {
                (
                    key,
                    (
                        proof.class_digest,
                        proof.sign,
                        (proof.input_order[0], proof.input_order[1]),
                        proof.exchange,
                    ),
                )
            })
            .collect(),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model_kernel_equivalence, module)?)
}
