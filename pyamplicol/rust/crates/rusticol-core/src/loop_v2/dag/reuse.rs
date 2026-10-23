// SPDX-License-Identifier: 0BSD

//! Evaluation reuse of loop-request blocks
//! (`dag_equivalence.assign_recursive_current_evaluation_reuse`).
//!
//! When no two currents share an evaluation contract
//! (`_current_evaluation_contract`), the oracle's value-equivalence
//! derivation classifies every current as its own representative with
//! factor 1, so the rewrite reduces to the model-certified kernel symmetry:
//! group key (class, canonical inputs, result particle, result chirality,
//! runtime coupling identity), first seen. The loop blocks of the plain
//! request model have unique contracts (RUST-MIGRATION §1 item 2); a DAG
//! with shared contracts needs the derivation itself (A3) and is refused.

use super::input::{DagInput, Vertex};
use super::recursion::{Dag, FxMap, complex_mul};
use super::{DagResult, unsupported};

/// `_canonical_zero` on both components.
fn canonical(value: [f64; 2]) -> [f64; 2] {
    [
        if value[0] == 0.0 { 0.0 } else { value[0] },
        if value[1] == 0.0 { 0.0 } else { value[1] },
    ]
}

fn finite(value: [f64; 2]) -> bool {
    value[0].is_finite() && value[1].is_finite()
}

/// `_canonical_kernel_evaluation`: canonical representative inputs and the
/// concrete kernel's factor.
pub fn canonical_kernel_evaluation(vertex: &Vertex, left: u32, right: u32) -> ((u32, u32), [f64; 2]) {
    let mut inputs = (left, right);
    if vertex.swap {
        inputs = (inputs.1, inputs.0);
    }
    let mut factor = vertex.factor;
    if let Some(exchange) = vertex.exchange
        && inputs.1 < inputs.0
    {
        inputs = (inputs.1, inputs.0);
        factor = complex_mul(factor, exchange);
    }
    (inputs, factor)
}

/// Python float equality of two pairs (`0.0 == -0.0`, NaN unequal).
fn pair_eq(left: [f64; 2], right: [f64; 2]) -> bool {
    left[0] == right[0] && left[1] == right[1]
}

/// `assign_recursive_current_evaluation_reuse` for a DAG whose currents all
/// have their own evaluation contract. The compile-time groups and factors
/// stay when the rewrite equals them (the oracle then returns its input).
pub fn assign_identity_reuse(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    if dag.interactions.is_empty() {
        return Ok(());
    }
    // `_current_evaluation_contract` without the colour state; dimension,
    // auxiliary kind, quantum numbers and momentum are functions of these.
    let mut contracts = FxMap::default();
    for (id, current) in dag.currents.iter().enumerate() {
        let key = &current.key;
        let contract = (
            key.particle,
            key.mask,
            key.word,
            key.ancestry,
            key.chirality,
            key.spin,
            key.flavour,
            key.orders,
            current.is_source,
        );
        if let Some(other) = contracts.insert(contract, id) {
            return unsupported(format!(
                "currents {other} and {id} share an evaluation contract; their reuse needs \
                 the value-equivalence derivation (native-dag A3)"
            ));
        }
    }
    let mut groups: FxMap<(u32, u32, u32, u32, i8, u32), i64> = FxMap::default();
    let mut rewritten = Vec::with_capacity(dag.interactions.len());
    for interaction in &dag.interactions {
        let vertex = &input.vertices[interaction.vertex as usize];
        let (inputs, mut factor) =
            canonical_kernel_evaluation(vertex, interaction.left, interaction.right);
        // `_canonical_interaction_evaluation` with identity parents: the
        // exact product of the kernel factor and 1 (an exact unit), or the
        // kernel evaluation itself when that factor is not finite.
        if finite(factor) {
            factor = canonical(complex_mul(factor, [1.0, 0.0]));
        }
        let result = &dag.currents[interaction.result as usize].key;
        let next = groups.len() as i64;
        let group = *groups
            .entry((
                vertex.class,
                inputs.0,
                inputs.1,
                result.particle,
                result.chirality,
                interaction.coupling_identity,
            ))
            .or_insert(next);
        rewritten.push((group, factor));
    }
    if rewritten
        .iter()
        .zip(&dag.interactions)
        .all(|((group, factor), old)| *group == old.group && pair_eq(*factor, old.factor))
    {
        return Ok(());
    }
    for ((group, factor), interaction) in rewritten.into_iter().zip(&mut dag.interactions) {
        interaction.group = group;
        interaction.factor = factor;
    }
    Ok(())
}
