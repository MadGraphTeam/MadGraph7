// SPDX-License-Identifier: 0BSD

//! Schedule B: the per-family suffix-trie executor (DESIGN.md §2.2, §5.1).
//!
//! Steps run in plan order, children before parents. A seeded step dresses
//! leaf objects `w_s U0` (one lane vector per row, the seeded layout); every
//! other step dresses the output of its child steps, which is freed (back to
//! the family's buffer pool) as soon as it is consumed, so only the open
//! objects of the current trie paths are live. Trace steps close the cut into
//! the family accumulator of their loop orders. Families are independent:
//! [`run_family`] owns its workspace, so families run on separate threads.

use super::closure::{FamilyNumerator, trace_into};
use super::dress::{Carrier, FactorStore, LaneData, Pool, dress_edge};
use super::kernels::lane_vector_len;
use super::plan::{LoopProgramV4, Source};
use super::seeds::{Seeds, leaf_object};
use crate::{RusticolError, RusticolResult};

/// A family's numerators and the peak of its live open-loop objects (reals).
pub struct FamilyRun<T> {
    pub numerator: FamilyNumerator<T>,
    pub peak_reals: usize,
}

/// Run schedule B for family `index` of `plan` on one chunk of lanes.
pub fn run_family<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    index: usize,
    factors: &FactorStore<T>,
    seeds: &Seeds<T>,
    data: &LaneData<T, L>,
) -> RusticolResult<FamilyRun<T>> {
    let family = &plan.families[index];
    let columns = plan.columns;
    let v = lane_vector_len(L);
    let zero = &data.zero;
    let mut pool = Pool::new(zero);
    let mut numerator = FamilyNumerator::zeros::<L>(index, family, columns, zero);
    let mut outputs: Vec<Option<Vec<T>>> = (0..family.steps.len()).map(|_| None).collect();
    let weights: Vec<T> = family
        .leaves
        .iter()
        .map(|leaf| data.cast(&leaf.weight))
        .collect();
    for (s, step) in family.steps.iter().enumerate() {
        let mut out = pool.take(step.rows * columns * step.out_row() * v);
        for edge in &step.edges {
            let content = &plan.contents[edge.content];
            let factor = factors.get(edge.factor);
            let products = &plan.products[edge.products];
            match edge.source {
                Source::Leaf(leaf) => {
                    let seed = &seeds.values[family.leaves[leaf].seed];
                    let mut input = pool.take(seed.len());
                    leaf_object::<T, L>(seed, &weights[leaf], &mut input);
                    dress_edge::<T, L>(
                        content,
                        true,
                        step.d_alpha,
                        products,
                        step.rank_out,
                        &input,
                        factor,
                        &edge.links,
                        &mut out,
                    );
                    pool.give(input);
                }
                Source::Step(child) => {
                    let input = outputs[child].take().ok_or_else(|| {
                        RusticolError::internal("a trie object is consumed twice")
                    })?;
                    dress_edge::<T, L>(
                        content,
                        false,
                        step.d_alpha,
                        products,
                        step.rank_out,
                        &input,
                        factor,
                        &edge.links,
                        &mut out,
                    );
                    pool.give(input);
                }
            }
        }
        match step.trace {
            Some(accumulator) => {
                trace_into::<T, L>(
                    step,
                    &out,
                    &mut numerator.accumulators[accumulator],
                    columns,
                    family.rank,
                    zero,
                )?;
                pool.give(out);
            }
            None => outputs[s] = Some(out),
        }
    }
    Ok(FamilyRun {
        numerator,
        peak_reals: pool.peak,
    })
}
