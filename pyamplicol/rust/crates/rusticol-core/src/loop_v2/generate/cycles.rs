// SPDX-License-Identifier: 0BSD

//! SU(N_c) overlaps of products of adjoint traces by cycle counting (WP3.1
//! I10, the overlap engine of the seed tables, the Born metric and the IR
//! dipoles).
//!
//! The literal overlap `<left|right>` of two colour tensors is a product of
//! traces of generators τ^a with `Tr(τ^a τ^b) = δ^{ab}`, every adjoint label
//! occurring exactly twice (`color/connections.overlap_traces`). Per label,
//! `τ^a_ij τ^a_kl = δ_il δ_kj − δ_ij δ_kl / N_c`. Expanding the second term
//! over the subsets S of labels that take it, the product is
//!
//! ```text
//! Σ_S (−1)^|S| N_c^(F(S) − |S|)
//! ```
//!
//! with F(S) the index loops of the U(N_c) gluing of the remaining labels
//! once the labels of S are deleted from their traces (an empty trace is one
//! loop): the cycles of "partner occurrence, then next in its trace", as
//! `colour.face_count`. This equals the trace algebra of
//! `color/contraction_trace.py` exactly, which stays its oracle.

use crate::{RusticolError, RusticolResult};
use std::collections::BTreeMap;

/// Labels beyond which the 2^L subset sum is refused (the caller keeps its
/// own trace algebra for such products).
pub const MAX_LABELS: usize = 24;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("trace overlap: {}", message.into()))
}

/// The Laurent coefficients `(power of N_c, integer coefficient)` of a
/// product of traces, nonzero ones in ascending power.
pub fn trace_overlap(traces: &[Vec<i64>]) -> RusticolResult<Vec<(i32, i64)>> {
    // Flattened positions: their trace span and label index.
    let mut spans = Vec::with_capacity(traces.len());
    let mut labels: BTreeMap<i64, usize> = BTreeMap::new();
    let mut label_of = Vec::new();
    let mut occurrences: Vec<[usize; 2]> = Vec::new();
    let mut counts: Vec<u8> = Vec::new();
    for trace in traces {
        spans.push((label_of.len(), trace.len()));
        for &label in trace {
            let index = *labels.entry(label).or_insert_with(|| {
                occurrences.push([usize::MAX; 2]);
                counts.push(0);
                occurrences.len() - 1
            });
            if counts[index] >= 2 {
                return Err(invalid(format!("label {label} occurs more than twice")));
            }
            occurrences[index][counts[index] as usize] = label_of.len();
            counts[index] += 1;
            label_of.push(index);
        }
    }
    if counts.iter().any(|&count| count != 2) {
        return Err(invalid("a label occurs once"));
    }
    let labels = occurrences.len();
    if labels > MAX_LABELS {
        return Err(invalid(format!("{labels} labels exceed {MAX_LABELS}")));
    }
    let total = label_of.len();
    let mut coefficients: BTreeMap<i32, i64> = BTreeMap::new();
    let mut following = vec![0usize; total];
    let mut visited = vec![false; total];
    let mut kept = Vec::with_capacity(total);
    for subset in 0u64..(1u64 << labels) {
        let removed = |position: usize| subset >> label_of[position] & 1 == 1;
        // The next kept position within each trace; a trace left empty (or
        // empty from the start) is one index loop.
        let mut loops = 0i32;
        for &(start, len) in &spans {
            kept.clear();
            kept.extend((start..start + len).filter(|&p| !removed(p)));
            if kept.is_empty() {
                loops += 1;
                continue;
            }
            for (k, &position) in kept.iter().enumerate() {
                following[position] = kept[(k + 1) % kept.len()];
            }
        }
        for (position, seen) in visited.iter_mut().enumerate() {
            *seen = removed(position);
        }
        for position in 0..total {
            if visited[position] {
                continue;
            }
            loops += 1;
            let mut current = position;
            while !visited[current] {
                visited[current] = true;
                let [a, b] = occurrences[label_of[current]];
                current = following[if a == current { b } else { a }];
            }
        }
        let size = subset.count_ones() as i32;
        *coefficients.entry(loops - size).or_insert(0) += if size % 2 == 0 { 1 } else { -1 };
    }
    Ok(coefficients
        .into_iter()
        .filter(|&(_, coefficient)| coefficient != 0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlap(traces: &[&[i64]]) -> Vec<(i32, i64)> {
        trace_overlap(&traces.iter().map(|t| t.to_vec()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn known_traces() {
        // Tr(1) = N.
        assert_eq!(overlap(&[&[]]), [(1, 1)]);
        // Tr(τ^a τ^a) = N^2 - 1.
        assert_eq!(overlap(&[&[1, 1]]), [(0, -1), (2, 1)]);
        // Tr(τ^a) Tr(τ^a) = 0.
        assert!(overlap(&[&[1], &[1]]).is_empty());
        // Tr(τ^a τ^b) Tr(τ^a τ^b) = N^2 - 1 (δ^{ab} δ^{ab}).
        assert_eq!(overlap(&[&[1, 2], &[1, 2]]), [(0, -1), (2, 1)]);
        // Tr(τ^a τ^b τ^a τ^b) = Tr(-τ^b τ^b / N) = -N + 1/N.
        assert_eq!(overlap(&[&[1, 2, 1, 2]]), [(-1, 1), (1, -1)]);
    }

    #[test]
    fn refuses_unpaired_labels() {
        assert!(trace_overlap(&[vec![1, 2, 2]]).is_err());
        assert!(trace_overlap(&[vec![1, 1, 1]]).is_err());
    }
}
