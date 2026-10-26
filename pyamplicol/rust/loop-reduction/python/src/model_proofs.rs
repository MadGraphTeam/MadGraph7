// SPDX-License-Identifier: 0BSD

//! Python entry point of the model compiler's parallel kernel-equivalence
//! proofs (`rusticol_core::model_proofs`, WP3.2 EWI-3R): the twin of
//! `models/compiler_entry.py::_kernel_equivalence_proof`, which stays the
//! oracle.
use super::*;
use rusticol_core::model_proofs::{KernelProofInput, kernel_equivalence_proofs};

/// A kernel's proof input: (dimensions, coupling, components of the input
/// order (0, 1), components of (1, 0)), every expression a canonical string.
type ProofItem = ((i64, i64, i64), String, Vec<String>, Vec<String>);
/// (class digest, sign, input order, input-exchange sign or None).
type ProofResult = (String, i32, (u8, u8), Option<i32>);

/// The proofs of `items` (`ProofItem`), in order, on `threads` workers
/// (`None`: rayon's default). The GIL is released while they run.
#[pyfunction]
#[pyo3(signature = (items, threads=None))]
fn model_kernel_proofs(
    py: Python<'_>,
    items: Vec<ProofItem>,
    threads: Option<usize>,
) -> PyResult<Vec<ProofResult>> {
    let inputs: Vec<KernelProofInput> = items
        .into_iter()
        .map(|((left, right, output), coupling, direct, swapped)| KernelProofInput {
            dimensions: [left, right, output],
            coupling,
            components: [direct, swapped],
        })
        .collect();
    let proofs = py
        .detach(|| kernel_equivalence_proofs(&inputs, threads))
        .map_err(error)?;
    Ok(proofs
        .into_iter()
        .map(|proof| {
            (
                proof.class_digest,
                proof.sign,
                (proof.input_order[0], proof.input_order[1]),
                proof.exchange,
            )
        })
        .collect())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(model_kernel_proofs, module)?)
}
