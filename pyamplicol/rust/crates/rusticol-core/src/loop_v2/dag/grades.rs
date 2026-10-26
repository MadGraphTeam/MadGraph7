// SPDX-License-Identifier: 0BSD

//! N_c grades of loop-request currents (`generation/dag_grades.py`, design
//! §3.2 item 4 and §3.3).
//!
//! A current whose trees produce k U(1)-subtraction auxiliaries has the N_c
//! grade `-k`. A current derived with several grades (identical-flavour
//! lines reaching one colour word through a gluon and through the U(1)
//! piece of the Fermi-exchanged pairing) is not exact in N_c, so
//! [`split_u1_grades`] makes it one current per grade, each fed by the
//! derivations of that grade, and so its consumers (a current fed by a mixed
//! one mixes as well). The copies take the current's place by decreasing
//! grade and carry it in their colour state (`ColourState::nc_grade`); the
//! interactions of a split result are emitted in its place, one per grade
//! of each split input (decreasing, left before right). A DAG without mixed
//! currents is left as it is, and one without U(1)-subtraction auxiliaries
//! is not even scanned.

use super::input::DagInput;
use super::recursion::{Current, Dag, Interaction};
use super::{ColourState, DagResult, invalid};

/// The grade set of every current as a bit set of `-grade` (bit k: grade
/// -k), in DAG order; `None` without a U(1)-subtraction auxiliary.
///
/// A grade is minus a number of auxiliaries, each of which joins two
/// external fermion legs of the at most 64 (leg labels are at most 64), so
/// k < 64 always fits; a larger k would be a malformed DAG.
fn grade_sets(dag: &Dag, u1: &[bool]) -> DagResult<Vec<u64>> {
    let mut sets: Vec<u64> = dag
        .currents
        .iter()
        .map(|current| current.is_source as u64)
        .collect();
    for interaction in &dag.interactions {
        let left = sets[interaction.left as usize];
        let right = sets[interaction.right as usize];
        if left == 0 || right == 0 {
            return invalid("a loop-request DAG interaction reads a current derived after it");
        }
        let shift = u1[dag.currents[interaction.result as usize].key.particle as usize] as u32;
        let mut out = 0u64;
        let mut rest = left;
        while rest != 0 {
            let k = rest.trailing_zeros() + shift;
            rest &= rest - 1;
            if k >= 64 || (right << k) >> k != right {
                return invalid("a current with 64 or more U(1)-subtraction pieces");
            }
            out |= right << k;
        }
        sets[interaction.result as usize] |= out;
    }
    Ok(sets)
}

/// `split_u1_grades` (module docs).
pub fn split_u1_grades(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    let kind = input.u1_subtraction_kind.as_str();
    let u1: Vec<bool> = input
        .particles
        .iter()
        .map(|particle| particle.auxiliary.as_deref() == Some(kind))
        .collect();
    split_by_grade(dag, &u1)
}

/// The split on particle flags: `u1[particle]` whether the particle is the
/// U(1)-subtraction auxiliary.
fn split_by_grade(dag: &mut Dag, u1: &[bool]) -> DagResult<()> {
    if !dag
        .currents
        .iter()
        .any(|current| u1[current.key.particle as usize])
    {
        return Ok(());
    }
    let sets = grade_sets(dag, u1)?;
    if sets.iter().all(|set| set.count_ones() == 1) {
        return Ok(());
    }
    // First new id of every current; a grade's copy follows at the rank of
    // its bit.
    let mut first = Vec::with_capacity(dag.currents.len());
    let mut currents: Vec<Current> = Vec::with_capacity(dag.currents.len());
    for (current, set) in dag.currents.iter().zip(&sets) {
        first.push(currents.len() as u32);
        if set.count_ones() == 1 {
            currents.push(*current);
            continue;
        }
        let state = dag.colours.get(current.key.colour).clone();
        let mut rest = *set;
        while rest != 0 {
            let k = rest.trailing_zeros();
            rest &= rest - 1;
            let mut copy = *current;
            copy.key.colour = dag.colours.intern(ColourState {
                nc_grade: Some(-(k as i64)),
                ..state.clone()
            });
            currents.push(copy);
        }
    }
    let copy_id = |current: u32, k: u32| -> u32 {
        let set = sets[current as usize];
        first[current as usize] + (set & ((1u64 << k) - 1)).count_ones()
    };
    let mut interactions: Vec<Interaction> = Vec::with_capacity(dag.interactions.len());
    for interaction in &dag.interactions {
        let shift = u1[dag.currents[interaction.result as usize].key.particle as usize] as u32;
        let mut left_rest = sets[interaction.left as usize];
        while left_rest != 0 {
            let left_k = left_rest.trailing_zeros();
            left_rest &= left_rest - 1;
            let mut right_rest = sets[interaction.right as usize];
            while right_rest != 0 {
                let right_k = right_rest.trailing_zeros();
                right_rest &= right_rest - 1;
                interactions.push(Interaction {
                    left: copy_id(interaction.left, left_k),
                    right: copy_id(interaction.right, right_k),
                    result: copy_id(interaction.result, left_k + right_k + shift),
                    ..*interaction
                });
            }
        }
    }
    dag.sources = dag.sources.iter().map(|id| first[*id as usize]).collect();
    dag.currents = currents;
    dag.interactions = interactions;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_v2::dag::recursion::CurrentKey;

    fn current(particle: u32, mask: u64, is_source: bool) -> Current {
        Current {
            key: CurrentKey {
                particle,
                mask,
                ancestry: mask,
                chirality: 0,
                spin: 0,
                flavour: 0,
                colour: 0,
                orders: 0,
                word: 0,
            },
            is_source,
            leg: -1,
            helicity: 0,
        }
    }

    fn interaction(left: u32, right: u32, result: u32) -> Interaction {
        Interaction {
            vertex: 0,
            left,
            right,
            result,
            coupling: [1.0, 0.0],
            coupling_identity: 0,
            colour_weight: [1.0, 0.0],
            group: result as i64,
            factor: [1.0, 0.0],
        }
    }

    #[test]
    fn mixed_currents_and_their_consumers_split_per_grade() {
        // Particles: 0 a quark, 1 a gluon, 2 the U(1)-subtraction auxiliary.
        let u1 = [false, false, true];
        let mut dag = Dag::default();
        dag.colours.intern(ColourState {
            accuracy: "full".into(),
            sector_id: 0,
            line_groups: vec![],
            basis_key: vec![],
            traces: vec![],
            lines: vec![],
            nc_grade: None,
        });
        // 0, 1, 2, 3: sources; 4 = gluon(0, 1), 5 = U(1)(0, 1);
        // 6 = quark from 4 + 2 (grade 0) and from 5 + 2 (grade -1): mixed;
        // 7 = quark from 6 + 3: mixed through its input.
        dag.currents = vec![
            current(0, 1, true),
            current(0, 2, true),
            current(0, 4, true),
            current(1, 8, true),
            current(1, 3, false),
            current(2, 3, false),
            current(0, 7, false),
            current(0, 15, false),
        ];
        dag.sources = vec![0, 1, 2, 3];
        dag.interactions = vec![
            interaction(0, 1, 4),
            interaction(0, 1, 5),
            interaction(4, 2, 6),
            interaction(5, 2, 6),
            interaction(6, 3, 7),
        ];
        split_by_grade(&mut dag, &u1).expect("split");
        let grades: Vec<Option<i64>> = dag
            .currents
            .iter()
            .map(|current| dag.colours.get(current.key.colour).nc_grade)
            .collect();
        assert_eq!(
            grades,
            vec![None, None, None, None, None, None, Some(0), Some(-1), Some(0), Some(-1)]
        );
        let edges: Vec<(u32, u32, u32)> = dag
            .interactions
            .iter()
            .map(|i| (i.left, i.right, i.result))
            .collect();
        assert_eq!(
            edges,
            vec![(0, 1, 4), (0, 1, 5), (4, 2, 6), (5, 2, 7), (6, 3, 8), (7, 3, 9)]
        );
        assert_eq!(dag.sources, vec![0, 1, 2, 3]);
        // Nothing mixes any more: a second pass changes nothing.
        let before: Vec<u32> = dag.currents.iter().map(|c| c.key.colour).collect();
        split_by_grade(&mut dag, &u1).expect("split");
        let after: Vec<u32> = dag.currents.iter().map(|c| c.key.colour).collect();
        assert_eq!(before, after);
        assert_eq!(dag.interactions.len(), 6);
    }
}
