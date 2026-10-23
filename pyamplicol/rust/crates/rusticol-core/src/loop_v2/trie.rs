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
//!
//! Each step runs in the kernel form the plan chose for it (gate G0's rule,
//! DESIGN.md §5.4): fused, `O' += O · X(J)` per edge; or split, `T += O ⊗ J`
//! per edge and routing term into one accumulator per vertex tensor, then
//! `O' += T · V` once per group and target row.

use super::closure::{FamilyNumerator, trace_into};
use super::dress::{Carrier, ChunkInputs, LaneData, Pool, dress_edge};
use super::kernels::{Segment, lane_vector_len, split_edge, split_vertex};
use super::plan::{Edge, LoopProgramV4, Source, SplitStep, Step};
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
    inputs: &ChunkInputs<T>,
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
        // The source object of an edge: a weighted leaf seed, or a child's
        // output (freed after this edge).
        let mut source = |edge: &Edge, pool: &mut Pool<T>| -> RusticolResult<Vec<T>> {
            match edge.source {
                Source::Leaf(leaf) => {
                    let seed = &seeds.values[family.leaves[leaf].seed];
                    let mut input = pool.take(seed.len());
                    leaf_object::<T, L>(seed, &weights[leaf], &mut input);
                    Ok(input)
                }
                Source::Step(child) => outputs[child]
                    .take()
                    .ok_or_else(|| RusticolError::internal("a trie object is consumed twice")),
            }
        };
        match &step.split {
            None => {
                for edge in &step.edges {
                    let input = source(edge, &mut pool)?;
                    dress_edge::<T, L>(
                        &plan.contents[edge.content],
                        step.seeded,
                        step.d_alpha,
                        &plan.products[edge.products],
                        step.rank_out,
                        &input,
                        inputs.factors.get(edge.factor),
                        &edge.links,
                        &mut out,
                    );
                    pool.give(input);
                }
            }
            Some(split) => {
                let mut accumulators: Vec<Vec<T>> = split
                    .groups
                    .iter()
                    .map(|group| {
                        let row = split_segment(plan, step, group.vertex, group.products)
                            .accumulator_row_len();
                        pool.take(step.rows * columns * row * v)
                    })
                    .collect();
                for (edge, parts) in step.edges.iter().zip(&split.parts) {
                    let input = source(edge, &mut pool)?;
                    for part in parts {
                        let group = &split.groups[part.group];
                        let segment = split_segment(plan, step, group.vertex, edge.products);
                        split_edge::<T, L>(
                            &segment,
                            &input,
                            inputs.currents.get(part.current),
                            &part.links,
                            &mut accumulators[part.group],
                        );
                    }
                    pool.give(input);
                }
                apply_vertices::<T, L>(plan, step, split, inputs, &accumulators, &mut out);
                for accumulator in accumulators {
                    pool.give(accumulator);
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
                );
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

/// The split-form segment of `step` for `vertex`, with products table
/// `products`: a seeded step reads `δ_αβ` seeds (cut and open end of
/// dimension 1, output end `α × β'`).
pub(super) fn split_segment<'a>(
    plan: &'a LoopProgramV4,
    step: &Step,
    vertex: usize,
    products: usize,
) -> Segment<'a> {
    let vertex = &plan.vertices[vertex];
    let products = &plan.products[products];
    if step.seeded {
        Segment::new(
            1,
            1,
            vertex.width,
            vertex.d_in * vertex.d_out,
            products,
            step.rank_out,
        )
    } else {
        Segment::new(
            step.d_alpha,
            vertex.d_in,
            vertex.width,
            step.d_out,
            products,
            step.rank_out,
        )
    }
}

/// `O' += T · V` for every vertex group of a split step, once per target row.
pub(super) fn apply_vertices<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    step: &Step,
    split: &SplitStep,
    inputs: &ChunkInputs<T>,
    accumulators: &[Vec<T>],
    out: &mut [T],
) {
    let v = lane_vector_len(L);
    for (group, accumulator) in split.groups.iter().zip(accumulators) {
        let vertex = &plan.vertices[group.vertex];
        let segment = split_segment(plan, step, group.vertex, group.products);
        let layout = if step.seeded {
            &vertex.seeded
        } else {
            &vertex.open
        };
        let values = inputs.vertices.get(group.instance);
        let (from, to) = (
            segment.accumulator_row_len() * v,
            segment.output_row_len() * v,
        );
        for &row in &group.targets {
            let row = row as usize;
            split_vertex::<T, L>(
                &segment,
                layout,
                values,
                1,
                &accumulator[row * from..][..from],
                &mut out[row * to..][..to],
            );
        }
    }
}
