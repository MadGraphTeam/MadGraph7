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

use super::input::DagInput;
use super::recursion::Dag;
use super::{DagError, DagResult, invalid};

/// `_fermion_input_orientation_sign`.
fn orientation_sign(
    left: &(bool, String, bool),
    right: &(bool, String, bool),
) -> DagResult<f64> {
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
    let contract = |particle: u32| -> DagResult<&(bool, String, bool)> {
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
            interaction.colour_weight = [-interaction.colour_weight[0], -interaction.colour_weight[1]];
        }
    }
    Ok(())
}
