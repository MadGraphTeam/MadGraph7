// SPDX-License-Identifier: 0BSD

//! The rows a tree program evaluates for a plan whose observable columns
//! keep some Born helicity rows only (MQ-SELECT, DESIGN.md §8.5).
//!
//! The plan's loader keeps the schedule-B row links that a kept Born row
//! reaches and builds its fused factors and split currents on the rows
//! those links read (`plan/live.rs`, [`LoopProgramV4::read_rows`]). A tree
//! program with observable columns (`TreeProgram::set_columns`, whose
//! outputs the columns, the block rows and the seeds' Born components are)
//! then evaluates only what those read: the plan block rows of the read
//! factor and current rows (each with its further terms, `block_row_terms`,
//! where the tree engine splits the block current into several trees by
//! flavour flow), and the root closures (Born and counterterm) of the kept
//! Born rows. The interactions and output slots of both interaction
//! programs follow by backward liveness over their states
//! (an output is kept when a later read or a closure reads its slot, an
//! interaction when an output reads its accumulator); the closures skip the
//! other rows, whose Born and counterterm components stay zero, as a spin
//! source's dropped rows do. The kept interactions run in their order on
//! the same values and every kernel row is independent, so every value read
//! is bitwise the full program's. Without such columns, or when the columns
//! keep every row, the program runs whole.

use super::{Interactions, Term, TermInput, TreeProgram};
use crate::loop_v2::plan::LoopProgramV4;

/// The Born rows a plan's columns keep and the plan block rows its kept
/// links read.
pub(super) struct Restriction {
    rows: Vec<bool>,
    blocks: Vec<bool>,
}

impl Restriction {
    /// The restriction of `plan`, when its columns leave some Born row
    /// unseeded and its loader restricted the rows its inputs are built on
    /// (`LoopProgramV4::read_rows`).
    pub(super) fn of(plan: &LoopProgramV4) -> Option<Self> {
        if !plan.column_rows.restricts() {
            return None;
        }
        let read = plan.read_rows.as_ref()?;
        let rows: Vec<bool> = (0..plan.rows)
            .map(|row| (0..plan.columns).any(|column| plan.column_rows.keeps(row, column)))
            .collect();
        if rows.iter().all(|&kept| kept) {
            return None;
        }
        let mut blocks = vec![false; plan.block_rows.len()];
        let mut mark = |ids: &[usize]| ids.iter().for_each(|&id| blocks[id] = true);
        for (factor, &(content, _shift)) in plan.factors.iter().enumerate() {
            for term in &plan.contents[content].terms {
                for (row, ids) in &term.rows {
                    if read.factors[factor][*row] {
                        mark(ids);
                    }
                }
            }
        }
        for (current, &(content, term)) in plan.currents.iter().enumerate() {
            let rows = &plan.contents[content].terms[term].rows;
            for ((_row, ids), &read) in rows.iter().zip(&read.currents[current]) {
                if read {
                    mark(ids);
                }
            }
        }
        Some(Self { rows, blocks })
    }
}

impl TreeProgram {
    /// Evaluate only what `restriction` reads (module docs): the kept rows'
    /// closures and the interactions and outputs they and the read block
    /// rows need. Before the program's runtimes are built (their weights
    /// follow its interactions).
    pub(super) fn restrict_rows(&mut self, restriction: Restriction) {
        let size = self.state_size;
        let counterterm_size = self
            .counterterm_trees
            .as_ref()
            .map_or(0, |ct| ct.layout.size);
        let mut trees = block_slots(
            &self.block_rows,
            &self.block_row_terms,
            &restriction.blocks,
            size,
        );
        let mut counterterms = vec![false; counterterm_size];
        // The state components the closure terms read on the kept rows: a
        // term input below the trees' state size is a tree slot, any other
        // a counterterm-program slot.
        let mut closures = |terms: &[Term]| {
            for term in terms {
                for row in (0..restriction.rows.len()).filter(|&row| restriction.rows[row]) {
                    let starts: Option<Vec<usize>> = term
                        .blocks
                        .iter()
                        .map(|&block| self.blocks[block][row])
                        .collect();
                    let Some(starts) = starts else {
                        continue;
                    };
                    for input in &self.layouts[term.layout] {
                        if let TermInput::Current { slot, component } = *input {
                            let index = starts[slot] + component;
                            if index < size {
                                trees[index] = true;
                            } else {
                                counterterms[index - size] = true;
                            }
                        }
                    }
                }
            }
        };
        closures(&self.terms);
        if let Some(ct) = &self.counterterms {
            closures(&ct.terms);
        }
        prune(&mut self.interactions, &mut trees);
        if let Some(ct) = &mut self.counterterm_trees {
            prune(&mut ct.interactions, &mut counterterms);
        }
        self.closure_rows = Some(restriction.rows);
    }
}

/// The tree state components the read plan block rows `read` take: each
/// read row's slot (`block_rows`: start, dimension, Wick sign) and the slots
/// of its further terms (`block_row_terms`: row, start, Wick sign; a block
/// current the tree engine splits into several trees by flavour flow), on a
/// state of `size` components.
fn block_slots(
    block_rows: &[(usize, usize, i32)],
    block_row_terms: &[(usize, usize, i32)],
    read: &[bool],
    size: usize,
) -> Vec<bool> {
    let mut slots = vec![false; size];
    for (&(start, dimension, _sign), &read) in block_rows.iter().zip(read) {
        if read {
            slots[start..start + dimension].fill(true);
        }
    }
    for &(row, start, _sign) in block_row_terms {
        if read[row] {
            slots[start..start + block_rows[row].1].fill(true);
        }
    }
    slots
}

/// Keep the interactions and outputs of `interactions` that the state
/// components `needed` (after the program) depend on: backward over the
/// steps, an output is kept when it writes a needed component (which it
/// then defines), an interaction when a kept output reads its accumulator
/// (its inputs are then needed).
fn prune(interactions: &mut Interactions, needed: &mut [bool]) {
    for step in interactions.steps.iter_mut().rev() {
        let mut read = vec![false; step.width];
        step.outputs.retain(
            |&(accumulator, _propagator, _momentum, start, dimension, _pole)| {
                let kept = needed[start..start + dimension].iter().any(|&n| n);
                if kept {
                    read[accumulator] = true;
                }
                kept
            },
        );
        for &(_accumulator, _propagator, _momentum, start, dimension, _pole) in &step.outputs {
            needed[start..start + dimension].fill(false);
        }
        for (_kernel, edges) in &mut step.groups {
            edges.retain(|edge| read[edge.accumulator]);
            for edge in edges.iter() {
                needed[edge.left..edge.left + edge.left_dimension].fill(true);
                needed[edge.right..edge.right + edge.right_dimension].fill(true);
            }
        }
        step.groups.retain(|(_kernel, edges)| !edges.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Edge, Interactions, Step};
    use super::{block_slots, prune};
    use symbolica::domains::rational::Rational;

    fn edge(left: usize, right: usize, accumulator: usize) -> Edge {
        Edge {
            left,
            left_dimension: 1,
            right,
            right_dimension: 1,
            left_momentum: 10,
            right_momentum: 10,
            accumulator,
            weight: [Rational::from(1), Rational::from(0)],
            pole: false,
        }
    }

    /// Two steps on a state of 14 components: step 0 forms slot 5 from
    /// slots 0 and 1 (kernel 0) and slot 6 from 2 and 3 (kernel 1); step 1
    /// forms slot 8 from 5 and 0, and slot 9 from 6 and 4 (kernel 0).
    fn program() -> Interactions {
        let output = |accumulator, start| (accumulator, None, 10, start, 1, false);
        Interactions {
            saved: Vec::new(),
            models: Vec::new(),
            planes: None,
            shapes: Vec::new(),
            steps: vec![
                Step {
                    groups: vec![(0, vec![edge(0, 1, 0)]), (1, vec![edge(2, 3, 1)])],
                    width: 2,
                    outputs: vec![output(0, 5), output(1, 6)],
                },
                Step {
                    groups: vec![(0, vec![edge(5, 0, 0), edge(6, 4, 1)])],
                    width: 2,
                    outputs: vec![output(0, 8), output(1, 9)],
                },
            ],
        }
    }

    /// Per step, per kept interaction (kernel, left), and the kept output
    /// slots.
    fn kept(program: &Interactions) -> Vec<(Vec<(usize, usize)>, Vec<usize>)> {
        program
            .steps
            .iter()
            .map(|step| {
                let edges = step
                    .groups
                    .iter()
                    .flat_map(|(kernel, edges)| edges.iter().map(move |e| (*kernel, e.left)))
                    .collect();
                let outputs = step.outputs.iter().map(|output| output.3).collect();
                (edges, outputs)
            })
            .collect()
    }

    /// A read block row takes its slot and its further terms' slots (a
    /// split block current); an unread row takes none of them.
    #[test]
    fn a_read_block_row_takes_its_further_terms_slots() {
        // Rows 0 (slot 0..2) and 1 (slot 2..4); row 0 adds slots 6..8,
        // row 1 adds 8..10 and 10..12.
        let rows = [(0, 2, 1), (2, 2, -1)];
        let terms = [(0, 6, 1), (1, 8, -1), (1, 10, 1)];
        let taken = |read: &[bool]| -> Vec<usize> {
            block_slots(&rows, &terms, read, 14)
                .iter()
                .enumerate()
                .filter_map(|(slot, &taken)| taken.then_some(slot))
                .collect()
        };
        assert_eq!(taken(&[true, false]), [0, 1, 6, 7]);
        assert_eq!(taken(&[false, true]), [2, 3, 8, 9, 10, 11]);
        assert_eq!(taken(&[false, false]), Vec::<usize>::new());
    }

    #[test]
    fn interactions_keep_what_the_needed_slots_depend_on() {
        let needed_of = |slots: &[usize]| {
            let mut needed = vec![false; 14];
            slots.iter().for_each(|&slot| needed[slot] = true);
            needed
        };
        // Slot 8 reads slot 5 (step 0's kernel-0 interaction) and slot 0.
        let mut one = program();
        let mut needed = needed_of(&[8]);
        prune(&mut one, &mut needed);
        assert_eq!(
            kept(&one),
            [(vec![(0, 0)], vec![5]), (vec![(0, 5)], vec![8])]
        );
        // What the program reads from outside: slots 0 and 1.
        assert_eq!(needed, needed_of(&[0, 1]));
        // Slots 9 and 1: slot 1 is an input, slot 9 reads 6 (kernel 1).
        let mut other = program();
        let mut needed = needed_of(&[9, 1]);
        prune(&mut other, &mut needed);
        assert_eq!(
            kept(&other),
            [(vec![(1, 2)], vec![6]), (vec![(0, 6)], vec![9])]
        );
        assert_eq!(needed, needed_of(&[1, 2, 3, 4]));
        // Every output read: the program is whole.
        let mut whole = program();
        prune(&mut whole, &mut needed_of(&[8, 9]));
        assert_eq!(kept(&whole), kept(&program()));
        // Nothing read: no interaction is left.
        let mut none = program();
        prune(&mut none, &mut needed_of(&[]));
        assert!(
            kept(&none)
                .iter()
                .all(|(e, o)| e.is_empty() && o.is_empty())
        );
    }
}
