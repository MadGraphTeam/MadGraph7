// SPDX-License-Identifier: 0BSD

//! External-fermion ordering signs in the colour-flow convention
//! (`generation/fermion_ordering.apply_fermion_ordering_signs` with
//! `coloured_lines=False`): only colour-singlet fermions own exterior signs;
//! each merge multiplies its colour weight by the Koszul sign of its
//! supports and the orientation sign of a two-fermion kernel.
//!
//! `recurrence/fermion_ordering.rs` holds the Rust recurrence builder's
//! version of the same rule for the Wick convention of tree programs; its
//! inputs are the recurrence template's, so the merge sign is restated here
//! on the DAG's own tables.
//!
//! [`apply_connection_signs`] is
//! `generation/fermion_ordering.apply_colour_connection_signs`: a block
//! whose trees pair its coloured fermion endpoints in more than one way can
//! have derivations of both colour-connection parities; the other parity's
//! `-1` goes into those derivations' colour weights, so every block has one
//! connection sign ([`connection_parities`], shared with the block grouping).

use super::input::{DagInput, Particle, SignContract};
use super::recursion::{CurrentKey, Dag};
use super::{DagError, DagResult, invalid};
use std::collections::HashMap;
use std::hash::Hash;

/// `_fermion_input_orientation_sign`.
fn orientation_sign(left: &SignContract, right: &SignContract) -> DagResult<f64> {
    let (left_fermion, left_orientation, left_owned) = left;
    let (right_fermion, right_orientation, right_owned) = right;
    if !(left_fermion & right_fermion) {
        return Ok(1.0);
    }
    if left_owned != right_owned {
        return invalid("external-fermion ordering encountered mixed color-sign ownership");
    }
    if !left_owned {
        return Ok(1.0);
    }
    let supported = |orientation: &str| matches!(orientation, "particle" | "antiparticle");
    if !supported(left_orientation) || !supported(right_orientation) {
        return invalid("external-fermion ordering requires non-self-conjugate model orientations");
    }
    match (left_orientation.as_str(), right_orientation.as_str()) {
        ("antiparticle", "particle") => Ok(1.0),
        ("particle", "antiparticle") => Ok(-1.0),
        _ => invalid("external-fermion ordering requires opposite Dirac orientations"),
    }
}

pub fn apply_colour_flow_signs(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    if dag.interactions.is_empty() {
        return Ok(());
    }
    // `FermionOrderingSign.__init__`: source ranks of the owned legs.
    let mut ranks = [None::<usize>; 65];
    let mut active = false;
    for leg in &input.legs {
        let rank = leg.sign_rank.clone().map_err(DagError::Invalid)?;
        if let Some(rank) = rank {
            ranks[leg.label as usize] = Some(rank);
            active = true;
        }
    }
    if !active {
        return Ok(());
    }
    let contract = |particle: u32| -> DagResult<&SignContract> {
        input.particles[particle as usize]
            .sign_contract
            .as_ref()
            .map_err(|error| DagError::Invalid(error.clone()))
    };
    let owned = |mask: u64| -> Vec<usize> {
        (0..64u8)
            .filter(|bit| mask >> bit & 1 == 1)
            .filter_map(|bit| ranks[bit as usize + 1])
            .collect()
    };
    for interaction in &mut dag.interactions {
        let left = &dag.currents[interaction.left as usize].key;
        let right = &dag.currents[interaction.right as usize].key;
        let left_contract = contract(left.particle)?;
        let right_contract = contract(right.particle)?;
        let left_ranks = owned(left.mask);
        let right_ranks = owned(right.mask);
        let inversions = left_ranks
            .iter()
            .map(|l| right_ranks.iter().filter(|r| l > *r).count())
            .sum::<usize>();
        let mut sign = if inversions % 2 == 1 { -1.0 } else { 1.0 };
        sign *= orientation_sign(left_contract, right_contract)?;
        if sign < 0.0 {
            interaction.colour_weight =
                [-interaction.colour_weight[0], -interaction.colour_weight[1]];
        }
    }
    Ok(())
}

/// The colour-connection parity rule
/// (`generation/fermion_ordering.colour_connection_parities`): `merges` are
/// the interactions `(left, right, result, connects)` in DAG order, `size`
/// the leg count of a current, `block` its block key (the current without
/// its helicity ancestry). Sources have parity 0; every current of a block
/// takes the parity of the block's first derivation (merges by result
/// size, in DAG order). Returns the parities and the indices of the merges
/// whose derivation has the other parity, in increasing order.
pub fn connection_parities<K: Eq + Hash>(
    current_count: usize,
    is_source: impl Fn(usize) -> bool,
    merges: &[(usize, usize, usize, bool)],
    size: impl Fn(usize) -> u32,
    block: impl Fn(usize) -> K,
) -> DagResult<(Vec<Option<bool>>, Vec<usize>)> {
    let mut parity: Vec<Option<bool>> = (0..current_count)
        .map(|id| is_source(id).then_some(false))
        .collect();
    let mut order: Vec<usize> = (0..merges.len()).collect();
    order.sort_by_key(|k| size(merges[*k].2));
    let mut blocks: HashMap<K, bool> = HashMap::new();
    let mut deviating = Vec::new();
    for k in order {
        let (left, right, result, connects) = merges[k];
        let (Some(left_parity), Some(right_parity)) = (parity[left], parity[right]) else {
            return invalid(format!(
                "interaction {k} reads a current without a derivation"
            ));
        };
        let value = left_parity ^ right_parity ^ connects;
        let known = *blocks.entry(block(result)).or_insert(value);
        parity[result] = Some(known);
        if known != value {
            deviating.push(k);
        }
    }
    deviating.sort_unstable();
    Ok((parity, deviating))
}

/// `apply_colour_connection_signs` on the builder's DAG: the leg sets and
/// representations of `input` decide which merges connect two coloured
/// fermion lines (`loops/v2/blocks.py`, `_connection_parities`).
pub fn apply_connection_signs(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    if dag.interactions.is_empty() {
        return Ok(());
    }
    let mut endpoints = 0u64;
    for leg in &input.legs {
        let particle = &input.particles[leg.particle as usize];
        if leg.rep.is_some_and(|rep| rep.abs() == 3) && particle.block_fermion == Some(true) {
            endpoints |= 1u64 << (leg.label - 1);
        }
    }
    let u1 = input.u1_subtraction_kind.as_str();
    let particle = |current: usize| -> &Particle {
        &input.particles[dag.currents[current].key.particle as usize]
    };
    let rep = |current: usize| -> DagResult<i32> {
        let particle = particle(current);
        particle
            .block_rep
            .ok_or_else(|| DagError::Invalid(format!("particle not in model: {}", particle.pdg)))
    };
    let u1_auxiliary = |current: usize| particle(current).auxiliary.as_deref() == Some(u1);
    let octet =
        |current: usize| -> DagResult<bool> { Ok(rep(current)? == 8 || u1_auxiliary(current)) };
    let singlet =
        |current: usize| -> DagResult<bool> { Ok(rep(current)? == 1 || u1_auxiliary(current)) };
    let coloured_fermion = |current: usize| -> DagResult<bool> {
        let particle = particle(current);
        let fermion = particle
            .block_fermion
            .ok_or_else(|| DagError::Invalid(format!("particle not in model: {}", particle.pdg)))?;
        Ok(fermion && rep(current)?.abs() == 3)
    };
    let ends = |current: usize| {
        dag.words
            .get(dag.currents[current].key.word)
            .iter()
            .any(|label| endpoints >> (label - 1) & 1 == 1)
    };
    let mut merges = Vec::with_capacity(dag.interactions.len());
    for interaction in &dag.interactions {
        let (left, right, result) = (
            interaction.left as usize,
            interaction.right as usize,
            interaction.result as usize,
        );
        // The oracle's short-circuit order (a representation is read only
        // where the rule needs it).
        let connects = ends(left)
            && ends(right)
            && (octet(left)? || octet(right)?)
            && (coloured_fermion(result)? || !(singlet(left)? || singlet(right)?));
        merges.push((left, right, result, connects));
    }
    let block = |current: usize| -> CurrentKey {
        CurrentKey {
            ancestry: 0,
            ..dag.currents[current].key
        }
    };
    let (_, deviating) = connection_parities(
        dag.currents.len(),
        |id| dag.currents[id].is_source,
        &merges,
        |id| dag.currents[id].key.mask.count_ones(),
        block,
    )?;
    for k in deviating {
        let weight = &mut dag.interactions[k].colour_weight;
        *weight = [-weight[0], -weight[1]];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::connection_parities;

    /// Currents 0-3 are sources; 4 and 5 are two helicity rows of one block
    /// (key 10), each derived twice: through a connection (parity 1) and
    /// without one (parity 0), in opposite DAG orders. The block takes the
    /// parity of its first derivation (row 4's connecting merge), the other
    /// derivations of that parity stay, and the two of parity 0 deviate.
    /// Current 6 reads row 5 and inherits the block's parity.
    #[test]
    fn a_block_takes_the_parity_of_its_first_derivation() {
        let merges = [
            (0, 1, 4, true),
            (2, 3, 4, false),
            (2, 3, 5, false),
            (0, 1, 5, true),
            (5, 0, 6, false),
        ];
        let size = |id: usize| match id {
            0..=3 => 1,
            4 | 5 => 2,
            _ => 3,
        };
        let block = |id: usize| if id == 4 || id == 5 { 10 } else { id };
        let (parity, deviating) =
            connection_parities(7, |id| id < 4, &merges, size, block).expect("parities");
        assert_eq!(
            parity,
            [false, false, false, false, true, true, true]
                .map(Some)
                .to_vec()
        );
        assert_eq!(deviating, vec![1, 2]);
    }

    #[test]
    fn a_merge_of_a_current_without_derivation_is_refused() {
        let merges = [(4, 0, 5, false)];
        assert!(connection_parities(6, |id| id < 4, &merges, |_| 1, |id| id).is_err());
    }
}
