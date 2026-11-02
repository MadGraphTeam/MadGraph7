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

use super::input::{DagInput, SignContract};
use super::recursion::{CurrentKey, Dag, FxMap};
use super::{DagError, DagResult, FERMION_LINE_REPRESENTATION, invalid, refused};

/// `_fermion_input_orientation_sign`. A merge of fermions of different sign
/// ownership (a coloured and a colour-singlet one: a fermion line changing
/// colour representation) is the listed refusal
/// [`FERMION_LINE_REPRESENTATION`].
fn orientation_sign(left: &SignContract, right: &SignContract) -> DagResult<f64> {
    let (left_fermion, left_orientation, left_owned) = left;
    let (right_fermion, right_orientation, right_owned) = right;
    if !(left_fermion & right_fermion) {
        return Ok(1.0);
    }
    if left_owned != right_owned {
        return refused(
            FERMION_LINE_REPRESENTATION,
            "a merge of a coloured and a colour-singlet fermion current (one \
             fermion line changing colour representation): the colour flow signs \
             coloured fermion lines and the Wick order colour-singlet ones",
        );
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
        sign *= orientation_sign(left_contract, right_contract).map_err(|error| match error {
            DagError::Unsupported(message) => {
                let vertex = &input.vertices[interaction.vertex as usize];
                DagError::Unsupported(format!(
                    "{message} (vertex kind {}, particles {:?})",
                    vertex.kind, vertex.pdgs
                ))
            }
            other => other,
        })?;
        if sign < 0.0 {
            interaction.colour_weight =
                [-interaction.colour_weight[0], -interaction.colour_weight[1]];
        }
    }
    Ok(())
}

/// The merges in increasing result size, in DAG order within a size (the
/// oracle's stable sort); `None` when the DAG order already is that order
/// (the recursion appends merges by result size).
pub(super) fn size_order(count: usize, size: impl Fn(usize) -> u32) -> Option<Vec<usize>> {
    if (1..count).all(|k| size(k - 1) <= size(k)) {
        return None;
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|k| size(*k));
    Some(order)
}

/// The colour-connection parity rule
/// (`generation/fermion_ordering.colour_connection_parities`). `block_of`
/// is the dense block id of every current (the current without its
/// helicity ancestry), below `block_count`; `merge(k)` is interaction `k`
/// as `(left, right, result, connects)`, visited in `order` (`size_order`;
/// `None`: DAG order). Sources have parity 0; every current of a block
/// takes the parity of the block's first derivation. Returns the parities
/// of the currents and the merges whose derivation has the other parity,
/// in increasing order.
pub fn connection_parities(
    is_source: impl Fn(usize) -> bool,
    block_of: &[u32],
    block_count: usize,
    merge_count: usize,
    merge: impl Fn(usize) -> (usize, usize, usize, bool),
    order: Option<&[usize]>,
) -> DagResult<(Vec<Option<bool>>, Vec<usize>)> {
    let mut parity: Vec<Option<bool>> = (0..block_of.len())
        .map(|id| is_source(id).then_some(false))
        .collect();
    let mut blocks: Vec<Option<bool>> = vec![None; block_count];
    let mut deviating = Vec::new();
    let mut visit = |k: usize| -> DagResult<()> {
        let (left, right, result, connects) = merge(k);
        let (Some(left_parity), Some(right_parity)) = (parity[left], parity[right]) else {
            return invalid(format!(
                "interaction {k} reads a current without a derivation"
            ));
        };
        let value = left_parity ^ right_parity ^ connects;
        let known = *blocks[block_of[result] as usize].get_or_insert(value);
        parity[result] = Some(known);
        if known != value {
            deviating.push(k);
        }
        Ok(())
    };
    match order {
        None => (0..merge_count).try_for_each(&mut visit)?,
        Some(order) => order.iter().try_for_each(|k| visit(*k))?,
    }
    deviating.sort_unstable();
    Ok((parity, deviating))
}

/// `apply_colour_connection_signs` on the builder's DAG: the leg sets and
/// representations of `input` decide which merges connect two coloured
/// fermion lines (`loops/v2/blocks.py`, `_connection_parities`). Without a
/// connecting merge every parity is 0 and nothing is folded: the blocks are
/// grouped (one map entry per current) only when a merge connects.
pub fn apply_connection_signs(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    let mut endpoints = 0u64;
    for leg in &input.legs {
        let particle = &input.particles[leg.particle as usize];
        if leg.rep.is_some_and(|rep| rep.abs() == 3) && particle.block_fermion == Some(true) {
            endpoints |= 1u64 << (leg.label - 1);
        }
    }
    if dag.interactions.is_empty() || endpoints == 0 {
        return Ok(());
    }
    // Whether each interned word holds an endpoint, and whether each
    // particle is the U(1)-subtraction auxiliary.
    let ends: Vec<bool> = dag
        .words
        .values
        .iter()
        .map(|word| word.iter().any(|label| endpoints >> (label - 1) & 1 == 1))
        .collect();
    let u1 = input.u1_subtraction_kind.as_str();
    let u1_auxiliary: Vec<bool> = input
        .particles
        .iter()
        .map(|particle| particle.auxiliary.as_deref() == Some(u1))
        .collect();
    let rep = |particle: u32| -> DagResult<i32> {
        let particle = &input.particles[particle as usize];
        particle
            .block_rep
            .ok_or_else(|| DagError::Invalid(format!("particle not in model: {}", particle.pdg)))
    };
    let octet = |particle: u32| -> DagResult<bool> {
        Ok(rep(particle)? == 8 || u1_auxiliary[particle as usize])
    };
    let singlet = |particle: u32| -> DagResult<bool> {
        Ok(rep(particle)? == 1 || u1_auxiliary[particle as usize])
    };
    let coloured_fermion = |particle: u32| -> DagResult<bool> {
        let fermion = input.particles[particle as usize]
            .block_fermion
            .ok_or_else(|| {
                DagError::Invalid(format!(
                    "particle not in model: {}",
                    input.particles[particle as usize].pdg
                ))
            })?;
        Ok(fermion && rep(particle)?.abs() == 3)
    };
    let mut connects = Vec::with_capacity(dag.interactions.len());
    let mut any = false;
    for interaction in &dag.interactions {
        let left = &dag.currents[interaction.left as usize].key;
        let right = &dag.currents[interaction.right as usize].key;
        let result = &dag.currents[interaction.result as usize].key;
        // The oracle's short-circuit order (a representation is read only
        // where the rule needs it).
        let value = ends[left.word as usize]
            && ends[right.word as usize]
            && (octet(left.particle)? || octet(right.particle)?)
            && (coloured_fermion(result.particle)?
                || !(singlet(left.particle)? || singlet(right.particle)?));
        any |= value;
        connects.push(value);
    }
    if !any {
        return Ok(());
    }
    let mut blocks: FxMap<CurrentKey, u32> = FxMap::default();
    let block_of: Vec<u32> = dag
        .currents
        .iter()
        .map(|current| {
            let next = blocks.len() as u32;
            *blocks
                .entry(CurrentKey {
                    ancestry: 0,
                    ..current.key
                })
                .or_insert(next)
        })
        .collect();
    let order = size_order(dag.interactions.len(), |k| {
        dag.currents[dag.interactions[k].result as usize]
            .key
            .mask
            .count_ones()
    });
    let (_, deviating) = connection_parities(
        |id| dag.currents[id].is_source,
        &block_of,
        blocks.len(),
        dag.interactions.len(),
        |k| {
            let interaction = &dag.interactions[k];
            (
                interaction.left as usize,
                interaction.right as usize,
                interaction.result as usize,
                connects[k],
            )
        },
        order.as_deref(),
    )?;
    for k in deviating {
        let weight = &mut dag.interactions[k].colour_weight;
        *weight = [-weight[0], -weight[1]];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DagError, FERMION_LINE_REPRESENTATION, connection_parities, orientation_sign, size_order,
    };

    #[test]
    fn mixed_sign_ownership_is_the_listed_refusal() {
        let quark = (true, "particle".to_string(), false);
        let lepton = (true, "antiparticle".to_string(), true);
        let gluon = (false, "self-conjugate".to_string(), false);
        for (left, right) in [(&quark, &lepton), (&lepton, &quark)] {
            match orientation_sign(left, right) {
                Err(DagError::Unsupported(message)) => {
                    assert!(
                        message.starts_with(&format!("{FERMION_LINE_REPRESENTATION}: ")),
                        "{message}"
                    )
                }
                other => panic!("expected the listed refusal, got {other:?}"),
            }
        }
        // A fermion with a boson, and fermions of one ownership, are signed.
        assert_eq!(orientation_sign(&quark, &gluon).unwrap(), 1.0);
        let antilepton = (true, "particle".to_string(), true);
        assert_eq!(orientation_sign(&lepton, &antilepton).unwrap(), 1.0);
        assert_eq!(orientation_sign(&antilepton, &lepton).unwrap(), -1.0);
    }

    type Merge = (usize, usize, usize, bool);

    fn parities(
        current_count: usize,
        merges: &[Merge],
        size: impl Fn(usize) -> u32,
        block: impl Fn(usize) -> u32,
    ) -> super::DagResult<(Vec<Option<bool>>, Vec<usize>)> {
        // Dense block ids in first-seen order, as the callers number them.
        let mut ids: Vec<u32> = Vec::new();
        let block_of: Vec<u32> = (0..current_count)
            .map(|id| {
                let key = block(id);
                match ids.iter().position(|seen| *seen == key) {
                    Some(dense) => dense as u32,
                    None => {
                        ids.push(key);
                        (ids.len() - 1) as u32
                    }
                }
            })
            .collect();
        let order = size_order(merges.len(), |k| size(merges[k].2));
        connection_parities(
            |id| id < 4,
            &block_of,
            ids.len(),
            merges.len(),
            |k| merges[k],
            order.as_deref(),
        )
    }

    fn size(id: usize) -> u32 {
        match id {
            0..=3 => 1,
            4 | 5 => 2,
            _ => 3,
        }
    }

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
        let block = |id: usize| if id == 4 || id == 5 { 10 } else { id as u32 };
        let (parity, deviating) = parities(7, &merges, size, block).expect("parities");
        assert_eq!(
            parity,
            [false, false, false, false, true, true, true]
                .map(Some)
                .to_vec()
        );
        assert_eq!(deviating, vec![1, 2]);
    }

    /// The same merges with the size-3 merge first in DAG order: the merges
    /// are visited by result size (stable), so the parities and the
    /// deviating merges (now indices 2 and 3) are those of the sorted DAG.
    #[test]
    fn merges_are_visited_by_result_size() {
        let merges = [
            (5, 0, 6, false),
            (0, 1, 4, true),
            (2, 3, 4, false),
            (2, 3, 5, false),
            (0, 1, 5, true),
        ];
        assert!(size_order(merges.len(), |k| size(merges[k].2)).is_some());
        let block = |id: usize| if id == 4 || id == 5 { 10 } else { id as u32 };
        let (parity, deviating) = parities(7, &merges, size, block).expect("parities");
        assert_eq!(
            parity,
            [false, false, false, false, true, true, true]
                .map(Some)
                .to_vec()
        );
        assert_eq!(deviating, vec![2, 3]);
    }

    #[test]
    fn a_merge_of_a_current_without_derivation_is_refused() {
        let merges = [(4, 0, 5, false)];
        assert!(parities(6, &merges, |_| 1, |id| id as u32).is_err());
    }
}
