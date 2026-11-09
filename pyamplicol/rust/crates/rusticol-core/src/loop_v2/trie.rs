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

use super::closure::{FamilyNumerator, close_column, close_row, trace_into, trace_rows_into};
use super::dress::{Carrier, ChunkInputs, LaneData, Pool, Workspaces, dress_edge};
use super::kernels::{
    DiagonalSchedule, Segment, TraceTerm, lane_vector_len, split_edge, split_vertex, trace_row,
};
use super::plan::{Edge, Family, LoopProgramV4, RowTrace, Source, SplitStep, Step};
use super::seeds::Seeds;
use crate::{RusticolError, RusticolResult};
use ninjars::fast::tensor_layout::ncoeff;

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
    let workspaces = Workspaces::new(1, &data.zero);
    run_family_on(
        plan,
        index,
        &plan.families[index],
        inputs,
        seeds,
        data,
        &workspaces,
    )
}

/// The source object of an edge: a weighted leaf seed, or a child's output
/// (taken: the caller frees it once consumed).
#[allow(clippy::too_many_arguments)]
fn source<T: Carrier, const L: usize>(
    edge: &Edge,
    family: &Family,
    seeds: &Seeds<T>,
    weights: &[T],
    outputs: &mut [Option<Vec<T>>],
    pool: &mut Pool<T>,
) -> RusticolResult<Vec<T>> {
    match edge.source {
        Source::Leaf(leaf) => {
            let mut input = pool.take(seeds.len());
            seeds.leaf_into::<L>(
                family.leaves[leaf].seed as usize,
                &weights[leaf],
                &mut input,
            );
            Ok(input)
        }
        Source::Step(child) => outputs[child]
            .take()
            .ok_or_else(|| RusticolError::internal("a trie object is consumed twice")),
    }
}

/// [`run_family`] on `family`: plan family `index` or its copy pruned to an
/// evaluation's live rows (`plan::LiveRows`).
/// The walk takes its buffers from the calling thread's pool of
/// `workspaces` and its accumulators from their numerator pool (lit-dress
/// lever E), and gives the open objects back when it ends.
pub fn run_family_on<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    index: usize,
    family: &Family,
    inputs: &ChunkInputs<T>,
    seeds: &Seeds<T>,
    data: &LaneData<T, L>,
    workspaces: &Workspaces<T>,
) -> RusticolResult<FamilyRun<T>> {
    // Objects carry `object_columns` columns per row (one with dressed
    // rows); the accumulators every observable column.
    let columns = plan.object_columns();
    let v = lane_vector_len(L);
    let zero = &data.zero;
    let mut numerator = {
        let mut numerators = workspaces
            .numerators
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        FamilyNumerator::zeros_in::<L>(
            index,
            family,
            plan.columns,
            family.support.clone(),
            &mut numerators,
        )
    };
    let mut guard = workspaces.objects();
    let pool = &mut *guard;
    pool.begin();
    let mut outputs: Vec<Option<Vec<T>>> = (0..family.steps.len()).map(|_| None).collect();
    let weights: Vec<T> = family
        .leaves
        .iter()
        .map(|leaf| data.cast(&plan.leaf_weights[leaf.weight as usize]))
        .collect();
    // A split part's row links, unpacked (`plan::PackedLinks`).
    let mut links = Vec::new();
    for (s, step) in family.steps.iter().enumerate() {
        // A fused trace step forms only the diagonal its trace reads
        // (lit-dress lever A; decided per step from the plan).
        if let (Some(accumulator), None) = (step.trace, &step.split)
            && let Some(diagonals) = diagonals(plan, step)
        {
            let sources = step
                .edges
                .iter()
                .map(|edge| source::<T, L>(edge, family, seeds, &weights, &mut outputs, pool))
                .collect::<RusticolResult<Vec<_>>>()?;
            let mut trace = pool.take(ncoeff(step.rank_out) * v);
            let row_trace = family.row_traces.get(s).and_then(Option::as_ref);
            close_diagonal::<T, L>(
                plan,
                step,
                &diagonals,
                &sources,
                inputs,
                row_trace,
                &mut numerator.accumulators[accumulator],
                slots(family),
                family.rank,
                &mut trace,
                data,
            );
            pool.give(trace);
            for input in sources {
                pool.give(input);
            }
            continue;
        }
        let mut out = pool.take(step.rows * columns * step.out_row() * v);
        match &step.split {
            None => {
                for edge in &step.edges {
                    let input = source::<T, L>(edge, family, seeds, &weights, &mut outputs, pool)?;
                    dress_edge::<T, L>(
                        &plan.contents[edge.content],
                        step.seeded,
                        step.d_alpha,
                        &plan.products[edge.products],
                        plan.gather(edge.content, edge.products, step.seeded),
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
                    let input = source::<T, L>(edge, family, seeds, &weights, &mut outputs, pool)?;
                    for part in parts {
                        let group = &split.groups[part.group];
                        let segment = split_segment(plan, step, group.vertex, edge.products);
                        split_edge::<T, L>(
                            &segment,
                            &input,
                            inputs.currents.get(part.current),
                            part.links.unpack_into(&mut links),
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
        match (step.trace, family.row_traces.get(s)) {
            (Some(accumulator), Some(Some(row_trace))) => {
                // Dressed rows: the per-row closure with the amplitude
                // vector of the step's seed key (decided per step).
                let vector = key_vector(plan, row_trace, data);
                trace_rows_into::<T, L>(
                    step,
                    &out,
                    &mut numerator.accumulators[accumulator],
                    &row_trace.rows,
                    &plan.key_columns[row_trace.key as usize],
                    slots(family),
                    &vector,
                    family.rank,
                    zero,
                );
                pool.give(out);
            }
            (Some(accumulator), _) => {
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
            (None, _) => outputs[s] = Some(out),
        }
    }
    for out in outputs.into_iter().flatten() {
        pool.give(out);
    }
    Ok(FamilyRun {
        numerator,
        peak_reals: pool.peak,
    })
}

/// The accumulator slot of each observable column of `family` (empty: every
/// column stored in order; [`super::plan::Support`]).
fn slots(family: &Family) -> &[u32] {
    family
        .support
        .as_deref()
        .map_or(&[], |support| support.slots.as_slice())
}

/// The amplitude vector of a dressed trace step's seed key in the carrier.
fn key_vector<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    row_trace: &RowTrace,
    data: &LaneData<T, L>,
) -> Vec<(T, T)> {
    plan.key_vectors[row_trace.key as usize]
        .iter()
        .map(|[re, im]| (data.cast(re), data.cast(im)))
        .collect()
}

/// The diagonal schedules of a fused trace step's edges (lit-dress lever
/// A), or `None` where an edge has none: the step then forms its whole
/// output.
fn diagonals<'a>(plan: &'a LoopProgramV4, step: &Step) -> Option<Vec<&'a DiagonalSchedule>> {
    step.edges
        .iter()
        .map(|edge| {
            (plan.contents[edge.content].d_out == step.d_out)
                .then(|| plan.diagonal(edge.content, edge.products, step.seeded, step.d_alpha))
                .flatten()
        })
        .collect()
}

/// Close a fused trace step from the diagonal of its output (lit-dress
/// lever A): per traced object row, in row order, its trace
/// ([`trace_row`]: the diagonal coefficients accumulated over the row's
/// links in edge and link order, each from zero, then summed over the cut
/// index in order) and its closure into the accumulator (dressed rows:
/// [`close_row`] with the row's Born columns; seeded: [`close_column`]),
/// the operations [`dress_edge`] into a zeroed output followed by
/// [`trace_rows_into`] or [`trace_into`] perform on what the closure
/// reads, so the accumulator is bitwise theirs. The traced rows are the row
/// closure's rows (dressed rows) or the object columns (seeded plans, as
/// `trace_into`); a link into another row is not formed (no closure reads
/// it), nor is a row no link reaches closed (it would add exact zeros).
#[allow(clippy::too_many_arguments)]
fn close_diagonal<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    step: &Step,
    diagonals: &[&DiagonalSchedule],
    sources: &[Vec<T>],
    inputs: &ChunkInputs<T>,
    row_trace: Option<&RowTrace>,
    accumulator: &mut [T],
    slots: &[u32],
    rank: usize,
    trace: &mut [T],
    data: &LaneData<T, L>,
) {
    let v = lane_vector_len(L);
    let traced = row_trace.map_or(plan.object_columns(), |r| r.rows.len());
    // Per edge: its input row and factor row lengths.
    let shapes: Vec<(usize, usize)> = step
        .edges
        .iter()
        .map(|edge| {
            let content = &plan.contents[edge.content];
            let products = &plan.products[edge.products];
            let input_row = if step.seeded {
                Segment::new(
                    1,
                    1,
                    1,
                    content.d_in * content.d_out,
                    products,
                    step.rank_out,
                )
            } else {
                Segment::new(
                    step.d_alpha,
                    content.d_in,
                    1,
                    content.d_out,
                    products,
                    step.rank_out,
                )
            }
            .input_row_len();
            (input_row * v, content.factor_row() * v)
        })
        .collect();
    // The links into each traced row, in edge then link order.
    let mut starts = vec![0usize; traced + 1];
    for edge in &step.edges {
        for link in &edge.links {
            if let Some(count) = starts.get_mut(link.output as usize + 1) {
                *count += 1;
            }
        }
    }
    for r in 0..traced {
        starts[r + 1] += starts[r];
    }
    let mut next = starts.clone();
    let mut order = vec![(0u32, 0u32); starts[traced]];
    for (e, edge) in step.edges.iter().enumerate() {
        for (l, link) in edge.links.iter().enumerate() {
            let row = link.output as usize;
            if row < traced {
                order[next[row]] = (e as u32, l as u32);
                next[row] += 1;
            }
        }
    }
    let vector = row_trace.map(|r| key_vector(plan, r, data));
    let mut term = vec![data.zero.clone(); v];
    let mut contributions: Vec<TraceTerm<'_, T>> = Vec::new();
    for r in 0..traced {
        let links = &order[starts[r]..starts[r + 1]];
        if links.is_empty() {
            continue;
        }
        contributions.clear();
        for &(e, l) in links {
            let edge = &step.edges[e as usize];
            let link = edge.links[l as usize];
            let (input_row, factor_row) = shapes[e as usize];
            contributions.push(TraceTerm {
                input: &sources[e as usize][link.input as usize * input_row..][..input_row],
                factor: &inputs.factors.get(edge.factor)[link.block as usize * factor_row..]
                    [..factor_row],
                schedule: diagonals[e as usize],
            });
        }
        trace_row::<T, L>(&contributions, trace);
        match (row_trace, &vector) {
            (Some(row_trace), Some(vector)) => close_row::<T, L>(
                trace,
                accumulator,
                plan.key_columns[row_trace.key as usize].row(row_trace.rows[r] as usize),
                slots,
                vector,
                rank,
                &mut term,
            ),
            _ => close_column::<T, L>(trace, accumulator, r, rank),
        }
    }
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
