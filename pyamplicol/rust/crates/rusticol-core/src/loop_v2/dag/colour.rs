// SPDX-License-Identifier: 0BSD

//! Loop-request colour engine: `generation/dag_color.ColorEngine` with
//! `loop_blocks=True` (its `combine`, `ordered_combination_labels` and the
//! `_loop_block_*` rules): the shared-ordering branch the oracle takes for
//! processes with coloured external legs, and its colour-sector branch for
//! processes without.
//!
//! Words are label sequences; a colour state is the oracle's `ColorState`.
//! Every rule below is the Python rule with the same name; their docstrings
//! hold the physics.

use super::ColourState;
use super::input::{DagInput, Vertex};
use std::collections::BTreeSet;

/// Open colour-flow ends `(incoming, outgoing)` by representation
/// (`_LOOP_BLOCK_OPEN_ENDS`).
fn open_ends_of(representation: Option<i32>) -> Option<(bool, bool)> {
    match representation {
        Some(1) => Some((false, false)),
        Some(3) => Some((false, true)),
        Some(-3) => Some((true, false)),
        Some(8) => Some((true, true)),
        _ => None,
    }
}

pub struct ColourEngine<'a> {
    input: &'a DagInput,
}

impl<'a> ColourEngine<'a> {
    pub fn new(input: &'a DagInput) -> Self {
        Self { input }
    }

    fn bit(label: u8) -> u64 {
        1u64 << (label - 1)
    }

    fn is_coloured(&self, label: u8) -> bool {
        self.input.coloured_labels & Self::bit(label) != 0
    }

    fn is_fundamental(&self, label: u8) -> bool {
        self.input.fundamental_labels & Self::bit(label) != 0
    }

    fn is_antifundamental(&self, label: u8) -> bool {
        self.input.antifundamental_labels & Self::bit(label) != 0
    }

    fn is_singlet(&self, label: u8) -> bool {
        self.input.singlet_labels & Self::bit(label) != 0
    }

    pub fn coloured(&self, word: &[u8]) -> Vec<u8> {
        word.iter()
            .copied()
            .filter(|label| self.is_coloured(*label))
            .collect()
    }

    /// `loop_block_word_open_ends`.
    pub fn word_open_ends(&self, coloured: &[u8]) -> Option<(bool, bool)> {
        for pair in coloured.windows(2) {
            if self.is_antifundamental(pair[0]) != self.is_fundamental(pair[1]) {
                return None;
            }
        }
        match (coloured.first(), coloured.last()) {
            (Some(first), Some(last)) => Some((
                !self.is_fundamental(*first),
                !self.is_antifundamental(*last),
            )),
            _ => Some((false, false)),
        }
    }

    /// `_lc_combined_line_groups` with its defaults (no closed groups, no
    /// sector), the only form loop-request blocks use.
    pub fn line_groups(
        &self,
        left: &ColourState,
        right: &ColourState,
        vertex: &Vertex,
    ) -> Option<Vec<i64>> {
        let particles = &self.input.particles;
        let left_coloured = particles[vertex.particles[0] as usize].has_colour;
        let right_coloured = particles[vertex.particles[1] as usize].has_colour;
        let result_coloured = particles[vertex.particles[2] as usize].has_colour;
        let left_groups: BTreeSet<i64> = left.line_groups.iter().copied().collect();
        let right_groups: BTreeSet<i64> = right.line_groups.iter().copied().collect();
        let groups: Vec<i64> = left_groups.union(&right_groups).copied().collect();
        if left_coloured != right_coloured {
            let (coloured, singlet) = if left_coloured {
                (&left_groups, &right_groups)
            } else {
                (&right_groups, &left_groups)
            };
            if !singlet.is_empty() && !coloured.is_empty() && !singlet.is_subset(coloured) {
                return None;
            }
            return Some(groups);
        }
        if !(left_coloured && result_coloured)
            && !left_groups.is_empty()
            && !right_groups.is_empty()
            && left_groups != right_groups
        {
            // ``groups.issubset(closed_groups)`` with no closed groups.
            return None;
        }
        Some(groups)
    }

    /// `_loop_block_split_singlet_strings`.
    fn split_singlet_strings(
        &self,
        coloured: &[u8],
        sides: [bool; 2],
    ) -> Option<(Vec<Vec<u8>>, Vec<u8>)> {
        let word = coloured;
        let mut strings = Vec::new();
        if sides[0] {
            let mut start = 0usize;
            while start < word.len() && self.is_fundamental(word[start]) {
                let mut stop = start;
                while !self.is_antifundamental(word[stop]) {
                    stop += 1;
                    if stop == word.len() {
                        return None;
                    }
                }
                strings.push(word[start..=stop].to_vec());
                start = stop + 1;
            }
            Some((strings, word[start..].to_vec()))
        } else {
            let mut stop = word.len();
            while stop > 0 && self.is_antifundamental(word[stop - 1]) {
                let mut start = stop as isize - 1;
                while !self.is_fundamental(word[start as usize]) {
                    start -= 1;
                    if start < 0 {
                        return None;
                    }
                }
                strings.push(word[start as usize..stop].to_vec());
                stop = start as usize;
            }
            Some((strings, word[..stop].to_vec()))
        }
    }

    /// `ColorEngine.combine` in loop-request mode (`_loop_block_flows`):
    /// the merged state, or `None` when the colour flow forbids the merge.
    pub fn combine(
        &self,
        left: &ColourState,
        right: &ColourState,
        vertex: &Vertex,
        ordered: &[u8],
    ) -> Option<ColourState> {
        if left.accuracy != right.accuracy {
            return None;
        }
        let groups = self.line_groups(left, right, vertex)?;
        let particles = &self.input.particles;
        let representation = particles[vertex.particles[2] as usize].loop_rep;
        let mut coloured = self.coloured(ordered);
        let mut traces: BTreeSet<Vec<u8>> = left
            .traces
            .iter()
            .chain(right.traces.iter())
            .cloned()
            .collect();
        let mut lines: BTreeSet<Vec<u8>> = left
            .lines
            .iter()
            .chain(right.lines.iter())
            .cloned()
            .collect();
        if let Some(sides) = vertex.absorbing {
            let (strings, word) = self.split_singlet_strings(&coloured, sides)?;
            lines.extend(strings);
            coloured = word;
            if coloured.is_empty() {
                return None;
            }
            let coloured_input = if !sides[0] { 0 } else { 1 };
            if representation == Some(1)
                && !coloured.iter().any(|label| self.is_fundamental(*label))
                && particles[vertex.particles[coloured_input] as usize].physical_adjoint
            {
                return None;
            }
        }
        let open_ends = self.word_open_ends(&coloured);
        let closing = open_ends == Some((true, true))
            && representation == Some(1)
            && (vertex.absorbing.is_some()
                || vertex.particles[..2]
                    .iter()
                    .all(|p| particles[*p as usize].rep == Some(8)));
        if closing {
            let starts: Vec<usize> = coloured
                .iter()
                .enumerate()
                .filter(|(_, label)| self.is_fundamental(**label))
                .map(|(index, _)| index)
                .collect();
            if let Some(last) = starts.last() {
                let mut rotated = coloured[*last..].to_vec();
                rotated.extend_from_slice(&coloured[..*last]);
                let mut closed: Vec<Vec<u8>> = Vec::new();
                for label in rotated {
                    if self.is_fundamental(label) {
                        closed.push(Vec::new());
                    }
                    closed
                        .last_mut()
                        .expect("rotated starts at a fundamental")
                        .push(label);
                }
                lines.extend(closed);
            } else {
                let word: Vec<u8> = coloured.iter().rev().copied().collect();
                let smallest = *word.iter().min().expect("closing word is not empty");
                let start = word
                    .iter()
                    .position(|label| *label == smallest)
                    .expect("min");
                let mut trace = word[start..].to_vec();
                trace.extend_from_slice(&word[..start]);
                traces.insert(trace);
            }
        } else if open_ends.is_none() || open_ends != open_ends_of(representation) {
            return None;
        }
        Some(ColourState {
            accuracy: left.accuracy.clone(),
            sector_id: 0,
            line_groups: groups,
            basis_key: Vec::new(),
            traces: traces.into_iter().collect(),
            lines: lines.into_iter().collect(),
            nc_grade: None,
        })
    }

    /// `_singlet_order_allowed` on the external label sets (masks).
    fn singlet_order_allowed(&self, left_mask: u64, right_mask: u64) -> bool {
        let all_singlet = |mask: u64| mask & !self.input.singlet_labels == 0;
        let left_singlet = all_singlet(left_mask);
        let right_singlet = all_singlet(right_mask);
        if left_singlet && !right_singlet {
            return false;
        }
        if left_singlet && right_singlet {
            return 63 - left_mask.leading_zeros() < 63 - right_mask.leading_zeros();
        }
        true
    }

    /// `_loop_block_trace_orientation_allowed`.
    fn trace_orientation_allowed(
        &self,
        left_word: &[u8],
        right_word: &[u8],
        left_state: &ColourState,
        vertex: &Vertex,
    ) -> bool {
        let left = self.coloured(left_word);
        let right = self.coloured(right_word);
        if (!left_state.traces.is_empty() || !left_state.lines.is_empty())
            && left.is_empty()
            && !right.is_empty()
        {
            return false;
        }
        let particles = &self.input.particles;
        let result = &particles[vertex.particles[2] as usize];
        let min_less = || match (left.iter().min(), right.iter().min()) {
            (Some(a), Some(b)) => a < b,
            _ => false,
        };
        if let Some(sides) = vertex.absorbing {
            let singlet = &particles[vertex.particles[if sides[0] { 0 } else { 1 }] as usize];
            if !singlet.u1_singlet {
                return !sides[0];
            }
            if sides[1]
                && !result.fla
                && result.loop_rep == Some(1)
                && particles[vertex.particles[0] as usize].physical_adjoint
            {
                return min_less();
            }
            return true;
        }
        if !left.is_empty()
            && !right.is_empty()
            && !result.fla
            && result.rep == Some(1)
            && vertex.particles[..2]
                .iter()
                .all(|p| particles[*p as usize].rep == Some(8))
        {
            return min_less();
        }
        true
    }

    /// The sector branch of `ordered_combination_labels`, which loop-request
    /// mode takes for a process without coloured legs (its shared orderings
    /// are off): the first intermediate word of the left input's sector with
    /// an ordered segment (`_ordered_combination_segment`) whose extras are
    /// singlets of the sector (`_line_local_singlet_extras_allowed`); the
    /// proposed labels when the sector is not in the plan.
    fn sector_ordered_labels(
        &self,
        left_word: &[u8],
        right_word: &[u8],
        left_state: &ColourState,
        proposed: &[u8],
    ) -> Option<Vec<u8>> {
        let Some(sector) = self.input.sectors.get(&left_state.sector_id) else {
            return Some(proposed.to_vec());
        };
        for word in &sector.words {
            let Some(segment) = ordered_combination_segment(left_word, right_word, word) else {
                continue;
            };
            let mut extras: Vec<u8> = proposed
                .iter()
                .copied()
                .filter(|label| !word.contains(label))
                .collect();
            extras.sort_unstable();
            if extras
                .iter()
                .any(|label| sector.singlet_labels & Self::bit(*label) == 0)
            {
                continue;
            }
            let mut labels = segment;
            labels.extend(extras);
            return Some(labels);
        }
        None
    }

    /// `ColorEngine.ordered_combination_labels` in loop-request mode,
    /// uncached (the recursion keeps the oracle's cache).
    pub fn ordered_combination_labels(
        &self,
        left_mask: u64,
        left_word: &[u8],
        left_state: &ColourState,
        right_mask: u64,
        right_word: &[u8],
        vertex: &Vertex,
    ) -> Option<Vec<u8>> {
        if !self.singlet_order_allowed(left_mask, right_mask) {
            return None;
        }
        if !self.trace_orientation_allowed(left_word, right_word, left_state, vertex) {
            return None;
        }
        let proposed: Vec<u8> = left_word.iter().chain(right_word.iter()).copied().collect();
        if vertex.absorbing.is_some() {
            if proposed
                .iter()
                .any(|label| !self.is_coloured(*label) && !self.is_singlet(*label))
            {
                return None;
            }
            let mut labels = self.coloured(&proposed);
            let mut extras: Vec<u8> = proposed
                .iter()
                .copied()
                .filter(|label| !self.is_coloured(*label))
                .collect();
            extras.sort_unstable();
            extras.dedup();
            labels.extend(extras);
            return Some(labels);
        }
        if self.input.coloured_labels == 0 {
            return self.sector_ordered_labels(left_word, right_word, left_state, &proposed);
        }
        // `_shared_lc_ordered_proposed_labels` (loop-request branch).
        let segment = self.coloured(&proposed);
        let mut extras: Vec<u8> = proposed
            .iter()
            .copied()
            .filter(|label| !self.is_coloured(*label))
            .collect();
        extras.sort_unstable();
        if extras.iter().any(|label| !self.is_singlet(*label)) {
            return None;
        }
        if segment.is_empty() {
            return Some(extras);
        }
        self.word_open_ends(&segment)?;
        let mut labels = segment;
        labels.extend(extras);
        Some(labels)
    }
}

/// `_line_positions`: positions in `word` of the labels it holds (the last
/// one of a repeated label, as the oracle's position map keeps it).
fn line_positions(labels: &[u8], word: &[u8]) -> Vec<usize> {
    labels
        .iter()
        .filter_map(|label| word.iter().rposition(|other| other == label))
        .collect()
}

/// `_positions_contiguous`.
fn contiguous(positions: &[usize]) -> bool {
    positions
        .iter()
        .enumerate()
        .all(|(offset, position)| *position == positions[0] + offset)
}

/// `_ordered_combination_segment`.
fn ordered_combination_segment(left: &[u8], right: &[u8], word: &[u8]) -> Option<Vec<u8>> {
    let left_positions = line_positions(left, word);
    let right_positions = line_positions(right, word);
    if left_positions.is_empty() && right_positions.is_empty() {
        return Some(Vec::new());
    }
    if left_positions.is_empty() || right_positions.is_empty() {
        let positions = if left_positions.is_empty() {
            &right_positions
        } else {
            &left_positions
        };
        if !contiguous(positions) {
            return None;
        }
        return Some(word[positions[0]..=positions[positions.len() - 1]].to_vec());
    }
    let mut union: Vec<usize> = left_positions
        .iter()
        .chain(&right_positions)
        .copied()
        .collect();
    union.sort_unstable();
    if contiguous(&left_positions)
        && contiguous(&right_positions)
        && contiguous(&union)
        && left_positions.iter().max() < right_positions.iter().min()
    {
        return Some(word[union[0]..=union[union.len() - 1]].to_vec());
    }
    None
}
