// SPDX-License-Identifier: 0BSD

//! Schedule M: the merged, primitive-seeded executor (DESIGN.md §2.2, §3.5,
//! §5.1, WP2.8).
//!
//! The plan's merged section (`loops.v2.merged`) is a planar state graph:
//! every certified skeleton is split into its colour-ordered terms, each
//! seeded at the cut by the primitive class of its cut word, and states that
//! agree in their dressed (leg set, species) positions and in the rest of the
//! cut word are shared across compositions and families. A state is one
//! dressing step whose edges read entries (`w_entry × U0[seed]`, the seeded
//! layout) or earlier states (open-loop objects); each edge may carry an
//! exact complex weight (the coefficient of its position's vertex
//! decomposition term times its propagator sign), applied to the edge's
//! input. Steps run in the kernel form the plan chose (gate G0's rule), with
//! the same fused and split kernels as schedule B.
//!
//! **Chunks.** States are ordered chunk by chunk (a cut species and the
//! first dressed position) and no edge crosses a chunk, so a chunk owns its
//! live objects. Within a chunk the seeded states run first, in parallel;
//! the other states form groups ([`super::plan::ChunkLayout`]) that read only
//! seeded states and their own group, and run in parallel on the seeded
//! outputs, each releasing a state's output after its last reader. Chunks
//! of a part run in parallel too. Closures are summed per (family, loop
//! orders, frame map) within a group and the sums are added in chunk and
//! group order, so results do not depend on the thread count.
//!
//! **Closures and frames.** A full state is traced into its family's
//! accumulator of its frame map. Paths read from the family's trie reading
//! close in the trie frame itself (the default); paths read from another
//! reading close in its frame, so each family keeps one sum per distinct
//! frame map (at most `2N`) and maps each into the trie frame once, when the
//! family is complete ([`Accumulators::finish`]): frame maps cost per family,
//! not per closure.
//!
//! **Streaming.** Chunks run in parts of consecutive chunks; the inputs of a
//! part (fused factors, split vertex tensors and currents) are built before it
//! and kept only while a later part reads them, and every family whose last
//! closure lies in a finished part is handed to the caller. A family's chunks
//! lie in the run of chunks of its first leg set (chunks are ordered by it),
//! so families stream into the reduction part by part.

use super::closure::{FamilyNumerator, frame_shift, substitute, trace_into};
use super::dress::{Carrier, ChunkInputs, LaneData, Pool, Reads, cadd, dress_edge};
use super::kernels::{lane_vector_len, split_edge};
use super::plan::{LoopProgramV4, Merged, Source};
use super::seeds::Seeds;
use super::trie::{apply_vertices, split_segment};
use crate::{RusticolError, RusticolResult};
use ninjars::fast::tensor_layout::ncoeff;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;

/// Default number of chunks per streamed part.
pub const DEFAULT_PART_CHUNKS: usize = 64;

/// A closure target: (family, accumulator, frame-map variant).
type Target = (usize, usize, usize);

/// Closures of a group (or a chunk's seeded states) summed per target, in
/// first-closure order, and the group's peak live reals.
struct Sums<T> {
    targets: Vec<Target>,
    sums: Vec<Vec<T>>,
    peak_reals: usize,
}

impl<T> Default for Sums<T> {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            sums: Vec::new(),
            peak_reals: 0,
        }
    }
}

/// The M accumulators of every family: per (accumulator, variant), `columns
/// × C(R+4, 4)` lane vectors in the closures' frame.
pub struct Accumulators<T> {
    columns: usize,
    sums: Vec<Option<Vec<Vec<T>>>>,
}

impl<T: Carrier> Accumulators<T> {
    pub fn new(plan: &LoopProgramV4) -> Self {
        Self {
            columns: plan.columns,
            sums: (0..plan.families.len()).map(|_| None).collect(),
        }
    }

    /// Reals held by the families' sums.
    pub fn reals(&self) -> usize {
        self.sums.iter().flatten().flatten().map(Vec::len).sum()
    }

    fn add<const L: usize>(
        &mut self,
        plan: &LoopProgramV4,
        merged: &Merged,
        target: Target,
        values: &[T],
        zero: &T,
    ) {
        let (family, accumulator, variant) = target;
        let variants = merged.variants[family].len();
        let len = self.columns * ncoeff(plan.families[family].rank) * lane_vector_len(L);
        let sums = self.sums[family].get_or_insert_with(|| {
            (0..plan.families[family].orders.len() * variants)
                .map(|_| vec![zero.clone(); len])
                .collect()
        });
        let sum = &mut sums[accumulator * variants + variant];
        let v = lane_vector_len(L);
        for (s, x) in sum.chunks_exact_mut(v).zip(values.chunks_exact(v)) {
            cadd::<T, L>(s, x);
        }
    }

    /// The numerator of a complete family in its trie frame: each frame
    /// map's sum substituted once and added.
    pub fn finish<const L: usize>(
        &mut self,
        plan: &LoopProgramV4,
        merged: &Merged,
        family: usize,
        data: &LaneData<T, L>,
    ) -> FamilyNumerator<T> {
        let zero = &data.zero;
        let plan_family = &plan.families[family];
        let mut numerator = FamilyNumerator::zeros::<L>(family, plan_family, self.columns, zero);
        let Some(sums) = self.sums[family].take() else {
            return numerator;
        };
        let v = lane_vector_len(L);
        let rank = plan_family.rank;
        let width = ncoeff(rank) * v;
        let variants = &merged.variants[family];
        for (index, sum) in sums.into_iter().enumerate() {
            let (accumulator, variant) = (index / variants.len(), index % variants.len());
            let target = &mut numerator.accumulators[accumulator];
            match variants[variant] {
                None => {
                    for (t, x) in target.chunks_exact_mut(v).zip(sum.chunks_exact(v)) {
                        cadd::<T, L>(t, x);
                    }
                }
                Some((sign, mask)) => {
                    let shift = frame_shift(data, mask);
                    for (column, values) in
                        target.chunks_exact_mut(width).zip(sum.chunks_exact(width))
                    {
                        let mapped = substitute::<T, L>(values, rank, sign, &shift, zero);
                        for (t, x) in column.chunks_exact_mut(v).zip(mapped.chunks_exact(v)) {
                            cadd::<T, L>(t, x);
                        }
                    }
                }
            }
        }
        numerator
    }
}

/// `out = w × input` for complex lane vectors and a complex weight.
fn scaled<T: Carrier, const L: usize>(weight: &[T; 2], input: &[T], out: &mut [T]) {
    let [a, b] = weight;
    for (target, source) in out
        .chunks_exact_mut(lane_vector_len(L))
        .zip(input.chunks_exact(lane_vector_len(L)))
    {
        let (re, im) = source.split_at(L);
        let (target_re, target_im) = target.split_at_mut(L);
        for l in 0..L {
            target_re[l] = a.clone() * &re[l] - b.clone() * &im[l];
            target_im[l] = a.clone() * &im[l] + b.clone() * &re[l];
        }
    }
}

fn product<T: Carrier>(x: &[T; 2], y: &[T; 2]) -> [T; 2] {
    [
        x[0].clone() * &y[0] - x[1].clone() * &y[1],
        x[0].clone() * &y[1] + x[1].clone() * &y[0],
    ]
}

/// Exact weights of one plan cast into the carrier of a lane chunk.
pub struct Weights<T> {
    entries: Vec<[T; 2]>,
    edges: Vec<[T; 2]>,
}

impl<T: Carrier> Weights<T> {
    pub fn new<const L: usize>(merged: &Merged, data: &LaneData<T, L>) -> Self {
        let cast =
            |[re, im]: &[symbolica::domains::rational::Rational; 2]| [data.cast(re), data.cast(im)];
        Self {
            entries: merged.entries.iter().map(|e| cast(&e.weight)).collect(),
            edges: merged.weights.iter().map(cast).collect(),
        }
    }
}

/// Everything a state reads besides its sources' objects.
struct Context<'a, T, const L: usize> {
    plan: &'a LoopProgramV4,
    merged: &'a Merged,
    inputs: &'a ChunkInputs<T>,
    seeds: &'a Seeds<T>,
    weights: &'a Weights<T>,
    data: &'a LaneData<T, L>,
}

impl<T: Carrier, const L: usize> Context<'_, T, L> {
    /// The output of state `s`: every edge dressed from its input, an entry
    /// object (entry and edge weights folded into the seed) or a source's
    /// output (`read`), scaled into a buffer when the edge has a weight.
    fn run_state<'b>(
        &self,
        s: usize,
        read: impl Fn(usize) -> Option<&'b [T]>,
        pool: &mut Pool<T>,
    ) -> RusticolResult<Vec<T>>
    where
        T: 'b,
    {
        let (plan, merged) = (self.plan, self.merged);
        let state = &merged.states[s];
        let step = &state.step;
        let v = lane_vector_len(L);
        let mut out = pool.take(step.rows * plan.columns * step.out_row() * v);
        // The input of edge `e`: owned (an entry, or a weighted source) or
        // the source's output in place.
        let input = |e: usize, pool: &mut Pool<T>| -> RusticolResult<Input<'b, T>> {
            let weight = state.weights[e].map(|w| &self.weights.edges[w]);
            match step.edges[e].source {
                Source::Leaf(entry) => {
                    let seed = &self.seeds.values[merged.entries[entry].seed];
                    let w = match weight {
                        Some(w) => product(&self.weights.entries[entry], w),
                        None => self.weights.entries[entry].clone(),
                    };
                    let mut object = pool.take(seed.len());
                    scaled::<T, L>(&w, seed, &mut object);
                    Ok(Input::Owned(object))
                }
                Source::Step(parent) => {
                    let source = read(parent)
                        .ok_or_else(|| RusticolError::internal("a released M state is read"))?;
                    Ok(match weight {
                        Some(w) => {
                            let mut object = pool.take(source.len());
                            scaled::<T, L>(w, source, &mut object);
                            Input::Owned(object)
                        }
                        None => Input::Shared(source),
                    })
                }
            }
        };
        match &step.split {
            None => {
                for (e, edge) in step.edges.iter().enumerate() {
                    let source = input(e, pool)?;
                    dress_edge::<T, L>(
                        &plan.contents[edge.content],
                        step.seeded,
                        step.d_alpha,
                        &plan.products[edge.products],
                        step.rank_out,
                        source.values(),
                        self.inputs.factors.get(edge.factor),
                        &edge.links,
                        &mut out,
                    );
                    source.release(pool);
                }
            }
            Some(split) => {
                let mut accumulators: Vec<Vec<T>> = split
                    .groups
                    .iter()
                    .map(|group| {
                        let row = split_segment(plan, step, group.vertex, group.products)
                            .accumulator_row_len();
                        pool.take(step.rows * plan.columns * row * v)
                    })
                    .collect();
                for (e, (edge, parts)) in step.edges.iter().zip(&split.parts).enumerate() {
                    let source = input(e, pool)?;
                    for part in parts {
                        let group = &split.groups[part.group];
                        let segment = split_segment(plan, step, group.vertex, edge.products);
                        split_edge::<T, L>(
                            &segment,
                            source.values(),
                            self.inputs.currents.get(part.current),
                            &part.links,
                            &mut accumulators[part.group],
                        );
                    }
                    source.release(pool);
                }
                apply_vertices::<T, L>(plan, step, split, self.inputs, &accumulators, &mut out);
                for accumulator in accumulators {
                    pool.give(accumulator);
                }
            }
        }
        Ok(out)
    }

    /// Trace a full state into `sums` (its target's slot).
    fn close(&self, s: usize, out: &[T], sums: &mut Sums<T>) {
        let (plan, state) = (self.plan, &self.merged.states[s]);
        let closure = state.closure.expect("a closing state");
        let target = (closure.family, closure.accumulator, closure.variant);
        let rank = plan.families[closure.family].rank;
        let zero = &self.data.zero;
        let slot = match sums.targets.iter().position(|t| *t == target) {
            Some(slot) => slot,
            None => {
                sums.targets.push(target);
                sums.sums.push(vec![
                    zero.clone();
                    plan.columns * ncoeff(rank) * lane_vector_len(L)
                ]);
                sums.sums.len() - 1
            }
        };
        trace_into::<T, L>(
            &state.step,
            out,
            &mut sums.sums[slot],
            plan.columns,
            rank,
            zero,
        );
    }

    /// Run one group of a chunk on the chunk's seeded outputs `head`
    /// (indexed from the chunk's first state `start`).
    fn run_group(
        &self,
        group: &[usize],
        head: &[Option<Vec<T>>],
        start: usize,
    ) -> RusticolResult<Sums<T>> {
        let merged = self.merged;
        let mut pool = Pool::new(&self.data.zero);
        let mut sums = Sums::default();
        let mut outputs: HashMap<usize, Vec<T>> = HashMap::new();
        let mut readers: HashMap<usize, usize> = group
            .iter()
            .map(|&s| (s, merged.states[s].readers))
            .collect();
        for &s in group {
            let out = self.run_state(
                s,
                |p| {
                    outputs
                        .get(&p)
                        .or(head[p - start].as_ref())
                        .map(Vec::as_slice)
                },
                &mut pool,
            )?;
            for edge in &merged.states[s].step.edges {
                if let Source::Step(parent) = edge.source
                    && let Some(left) = readers.get_mut(&parent)
                {
                    *left -= 1;
                    if *left == 0 {
                        pool.give(outputs.remove(&parent).expect("live parent"));
                    }
                }
            }
            if merged.states[s].closure.is_some() {
                self.close(s, &out, &mut sums);
                pool.give(out);
            } else {
                outputs.insert(s, out);
            }
        }
        sums.peak_reals = pool.peak;
        Ok(sums)
    }

    /// Run chunk `k`: its seeded states in parallel, then its groups in
    /// parallel on their outputs. Returns the closure sums of the seeded
    /// states and of every group in group order (thread-count independent).
    fn run_chunk(&self, k: usize) -> RusticolResult<(Vec<Sums<T>>, usize)> {
        let merged = self.merged;
        let range = merged.chunk(k);
        let start = range.start;
        let layout = &merged.layout[k];
        let zero = &self.data.zero;
        let head: Vec<Vec<T>> = layout
            .head
            .par_iter()
            .map_init(
                || Pool::new(zero),
                |pool, &s| self.run_state(s, |_| None, pool),
            )
            .collect::<RusticolResult<_>>()?;
        let mut sums = vec![Sums::default()];
        let mut outputs: Vec<Option<Vec<T>>> = (0..range.len()).map(|_| None).collect();
        let mut head_reals = 0;
        for (&s, out) in layout.head.iter().zip(head) {
            if merged.states[s].closure.is_some() {
                self.close(s, &out, &mut sums[0]);
            } else {
                head_reals += out.len();
                outputs[s - start] = Some(out);
            }
        }
        let groups = layout
            .groups
            .par_iter()
            .map(|group| self.run_group(group, &outputs, start))
            .collect::<RusticolResult<Vec<_>>>()?;
        let peak = head_reals + groups.iter().map(|g| g.peak_reals).max().unwrap_or(0);
        sums.extend(groups);
        Ok((sums, peak))
    }
}

/// An edge's input: a buffer of its own, or a source's output in place.
enum Input<'a, T> {
    Owned(Vec<T>),
    Shared(&'a [T]),
}

impl<T: Carrier> Input<'_, T> {
    fn values(&self) -> &[T] {
        match self {
            Input::Owned(object) => object,
            Input::Shared(object) => object,
        }
    }

    fn release(self, pool: &mut Pool<T>) {
        if let Input::Owned(object) = self {
            pool.give(object);
        }
    }
}

/// A part of consecutive chunks with the inputs it reads and those still
/// read by it or a later part.
pub struct Part {
    pub chunks: Range<usize>,
    pub reads: Reads,
    pub live: Reads,
}

/// Parts of at most `size` chunks.
pub fn parts(merged: &Merged, size: usize, plan: &LoopProgramV4) -> Vec<Part> {
    let count = merged.chunks.len();
    let ranges: Vec<Range<usize>> = (0..count)
        .step_by(size.max(1))
        .map(|start| start..(start + size.max(1)).min(count))
        .collect();
    let reads: Vec<Reads> = ranges
        .iter()
        .map(|range| {
            let (factors, vertices, currents) = merged.reads(range.clone());
            Reads {
                factors,
                vertices,
                currents,
            }
        })
        .collect();
    let last = |len: usize, ids: &dyn Fn(&Reads) -> &[usize]| {
        let mut last = vec![0; len];
        for (k, part) in reads.iter().enumerate() {
            for &id in ids(part) {
                last[id] = k;
            }
        }
        last
    };
    let factors = last(plan.factors.len(), &|r| &r.factors);
    let vertices = last(plan.vertex_instances.len(), &|r| &r.vertices);
    let currents = last(plan.currents.len(), &|r| &r.currents);
    let alive = |last: &[usize], k: usize| -> Vec<usize> {
        (0..last.len()).filter(|&id| last[id] >= k).collect()
    };
    ranges
        .into_iter()
        .zip(reads)
        .enumerate()
        .map(|(k, (chunks, reads))| Part {
            chunks,
            reads,
            live: Reads {
                factors: alive(&factors, k),
                vertices: alive(&vertices, k),
                currents: alive(&currents, k),
            },
        })
        .collect()
}

/// Measurements of one schedule-M run (reals per lane are the caller's).
#[derive(Clone, Debug, Default)]
pub struct MergedStats {
    pub states: u64,
    pub split_states: u64,
    pub chunks: u64,
    /// Peak reals of the chunk inputs (all lanes).
    pub peak_inputs: usize,
    /// Peak reals of the live objects of one chunk (all lanes).
    pub peak_objects: usize,
    /// Peak reals of the family accumulators (all lanes).
    pub peak_accumulators: usize,
    pub seconds_inputs: f64,
    pub seconds_dressing: f64,
}

/// Run schedule M on one chunk of lanes, part by part; `done` receives the
/// numerators of the families completed by each part (in family order).
#[allow(clippy::too_many_arguments)]
pub fn run_merged<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    parts: &[Part],
    inputs: &mut ChunkInputs<T>,
    seeds: &Seeds<T>,
    data: &LaneData<T, L>,
    mut done: impl FnMut(Vec<FamilyNumerator<T>>) -> RusticolResult<()>,
) -> RusticolResult<MergedStats> {
    let merged = plan
        .merged
        .as_ref()
        .ok_or_else(|| RusticolError::invalid_argument("the plan has no schedule M"))?;
    let weights = Weights::new(merged, data);
    let mut accumulators = Accumulators::new(plan);
    let mut stats = MergedStats::default();
    for part in parts {
        let start = std::time::Instant::now();
        inputs.prepare(plan, data, &part.reads, &part.live);
        stats.seconds_inputs += start.elapsed().as_secs_f64();
        stats.peak_inputs = stats.peak_inputs.max(inputs.reals());
        let start = std::time::Instant::now();
        let context = Context {
            plan,
            merged,
            inputs: &*inputs,
            seeds,
            weights: &weights,
            data,
        };
        let runs = part
            .chunks
            .clone()
            .into_par_iter()
            .map(|k| context.run_chunk(k))
            .collect::<RusticolResult<Vec<_>>>()?;
        for (run, peak) in runs {
            stats.peak_objects = stats.peak_objects.max(peak);
            for sums in &run {
                for (target, values) in sums.targets.iter().zip(&sums.sums) {
                    accumulators.add::<L>(plan, merged, *target, values, &data.zero);
                }
            }
        }
        stats.seconds_dressing += start.elapsed().as_secs_f64();
        stats.peak_accumulators = stats.peak_accumulators.max(accumulators.reals());
        for k in part.chunks.clone() {
            let states = &merged.states[merged.chunk(k)];
            stats.states += states.len() as u64;
            stats.split_states += states.iter().filter(|s| s.step.split.is_some()).count() as u64;
        }
        stats.chunks += part.chunks.len() as u64;
        let last = part.chunks.end - 1;
        let complete: Vec<FamilyNumerator<T>> = merged
            .families
            .iter()
            .filter(|&&family| {
                merged.last_chunk[family].is_some_and(|k| k <= last && k >= part.chunks.start)
            })
            .map(|&family| accumulators.finish(plan, merged, family, data))
            .collect();
        if !complete.is_empty() {
            done(complete)?;
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_v2::dress::V2Batch;
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::run_family;
    use symbolica::domains::float::Complex;
    use symbolica::domains::rational::Rational;

    /// Two scalar bubble families of a two-leg "process" (constant factors
    /// 2 and 3), each B family a seeded step and a trace step, leaf weight
    /// 3. Schedule M: family 0 merges two entries (weights 1 and 2i) in one
    /// state, the second through an edge of weight -i; family 1 is one path
    /// in a chunk of its own with entry weight -i and edge weight 3i.
    const PLAN: &str = r#"{"schema":4,"legs":2,"rows":1,"helicities":[[0],[0]],"columns":1,"average":1,
        "born_components":1,"constants":2,"masses":1,"block_rows":[],
        "kernels":[{"inputs":[],"outputs":[[[0,[]]]]},{"inputs":[],"outputs":[[[1,[]]]]}],
        "contents":[CONTENT0,CONTENT1],
        "factors":[[0,0],[1,1]],
        "seed_classes":[{"columns":[[[0,"1","0"]]]}],"seeds":[{"classes":[[0,"1","0"]]}],
        "families":[FAMILY,FAMILY],"forward":null,
        "merged":{"families":[0,1],
            "entries":[{"seed":0,"weight":["1","0"]},{"seed":0,"weight":["0","2"]},
                       {"seed":0,"weight":["0","-1"]}],
            "states":[
                {"seeded":true,"rows":1,"d_alpha":1,"d_out":1,"degree":0,"rank_in":0,
                 "edges":[{"entry":0,"parent":-1,"factor":0,"weight":null,"links":[0,0,0]}],
                 "closure":null},
                {"seeded":true,"rows":1,"d_alpha":1,"d_out":1,"degree":0,"rank_in":0,
                 "edges":[{"entry":1,"parent":-1,"factor":0,"weight":null,"links":[0,0,0]}],
                 "closure":null},
                {"seeded":false,"rows":1,"d_alpha":1,"d_out":1,"degree":0,"rank_in":0,
                 "edges":[{"entry":-1,"parent":0,"factor":1,"weight":null,"links":[0,0,0]},
                          {"entry":-1,"parent":1,"factor":1,"weight":["0","-1"],"links":[0,0,0]}],
                 "closure":{"family":0,"accumulator":0,"relabel":null}},
                {"seeded":true,"rows":1,"d_alpha":1,"d_out":1,"degree":0,"rank_in":0,
                 "edges":[{"entry":2,"parent":-1,"factor":0,"weight":null,"links":[0,0,0]}],
                 "closure":null},
                {"seeded":false,"rows":1,"d_alpha":1,"d_out":1,"degree":0,"rank_in":0,
                 "edges":[{"entry":-1,"parent":3,"factor":1,"weight":["0","3"],"links":[0,0,0]}],
                 "closure":{"family":1,"accumulator":0,"relabel":null}}],
            "chunks":[3,5]}}"#;
    const CONTENT: &str = r#"{"d_in":1,"d_out":1,"degree":0,"rows":1,"mask":0,"row_helicities":[[]],
        "fused":[[0,0,0,0,0,0]],
        "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
            "bindings":[{"kernel":KERNEL,"inputs":[],"entries":[0]}]}]}"#;
    const FAMILY: &str = r#"{"sequence":[[1,0],[2,0]],"rank":0,"orders":[0],
        "leaves":[{"seed":0,"weight":"3"}],
        "steps":[{"seeded":true,"trace":-1,"out":2,"rows":1,"d_alpha":1,"d_out":1,"degree":0,
                  "rank_in":0,"edges":[{"leaf":0,"child":-1,"factor":0,"links":[0,0,0]}]},
                 {"seeded":false,"trace":0,"out":0,"rows":1,"d_alpha":1,"d_out":1,"degree":0,
                  "rank_in":0,"edges":[{"leaf":-1,"child":0,"factor":1,"links":[0,0,0]}]}]}"#;

    fn plan() -> LoopProgramV4 {
        let text = PLAN
            .replace("CONTENT0", &CONTENT.replace("KERNEL", "0"))
            .replace("CONTENT1", &CONTENT.replace("KERNEL", "1"))
            .replace("FAMILY", FAMILY);
        LoopProgramV4::load(text.as_bytes()).unwrap()
    }

    fn batch() -> V2Batch<Rational> {
        let momentum =
            |e: i64| -> Vec<Rational> { [e, 0, 0, e, e, 0, 0, -e].map(Rational::from).to_vec() };
        let complex = |re: i64, im: i64| Complex::new(Rational::from(re), Rational::from(im));
        V2Batch {
            momenta: vec![momentum(1), momentum(2)],
            scales: vec![Rational::from(1); 2],
            masses: vec![Rational::from(0)],
            masses_imaginary: Vec::new(),
            leg_masses: Vec::new(),
            constants: vec![complex(2, 0), complex(0, 3)],
            block_rows: vec![Vec::new(); 2],
            born: vec![vec![complex(1, 1)], vec![complex(3, -2)]],
        }
    }

    #[test]
    fn plans_load_their_merged_graph() {
        let plan = plan();
        let merged = plan.merged.as_ref().unwrap();
        assert_eq!(merged.families, [0, 1]);
        assert_eq!((merged.chunk(0), merged.chunk(1)), (0..3, 3..5));
        assert_eq!(merged.last_chunk, [Some(0), Some(1)]);
        assert_eq!(merged.weights.len(), 2);
        let readers: Vec<usize> = merged.states.iter().map(|s| s.readers).collect();
        assert_eq!(readers, [1, 1, 0, 1, 0]);
        // Seeded states first, then the groups that read them.
        let layout: Vec<_> = merged
            .layout
            .iter()
            .map(|l| (l.head.clone(), l.groups.clone()))
            .collect();
        assert_eq!(
            layout,
            [(vec![0, 1], vec![vec![2]]), (vec![3], vec![vec![4]])]
        );
        // An edge into a later chunk's state is refused.
        let crossing = PLAN.replace(r#""chunks":[3,5]"#, r#""chunks":[1,5]"#);
        let text = crossing
            .replace("CONTENT0", &CONTENT.replace("KERNEL", "0"))
            .replace("CONTENT1", &CONTENT.replace("KERNEL", "1"))
            .replace("FAMILY", FAMILY);
        let error = LoopProgramV4::load(text.as_bytes()).unwrap_err();
        assert!(error.message().contains("crosses a chunk"), "{error}");
    }

    /// Closures above their family's rank are refused when the plan loads
    /// (a closure never truncates a coefficient while it evaluates).
    #[test]
    fn plans_refuse_closures_above_the_family_rank() {
        let load = |plan: &str, family: &str| {
            let text = plan
                .replace("CONTENT0", &CONTENT.replace("KERNEL", "0"))
                .replace("CONTENT1", &CONTENT.replace("KERNEL", "1"))
                .replace("FAMILY", family);
            LoopProgramV4::load(text.as_bytes())
        };
        let state = "\"rank_in\":0,\n                 \"edges\":[{\"entry\":-1,\"parent\":3";
        assert_eq!(PLAN.matches(state).count(), 1);
        let error = load(&PLAN.replace(state, &state.replace(":0,", ":1,")), FAMILY).unwrap_err();
        assert!(error.message().contains("merged closure"), "{error}");
        let trace = "\"trace\":0,\"out\":0,\"rows\":1,\"d_alpha\":1,\"d_out\":1,\"degree\":0,";
        assert_eq!(FAMILY.matches(trace).count(), 1);
        let family = FAMILY.replace(trace, &trace.replace("\"degree\":0", "\"degree\":1"));
        let error = load(PLAN, &family).unwrap_err();
        assert!(error.message().contains("above the family rank"), "{error}");
    }

    /// Exact numerators of M equal B's, family by family; each part hands
    /// over the families it completed.
    #[test]
    fn merged_numerators_equal_the_trie_numerators() {
        const L: usize = 2;
        let plan = plan();
        let batch = batch();
        let zero = Rational::from(0);
        let data = LaneData::<Rational, L>::new(&plan, &batch, &[0, 1], &zero).unwrap();
        let seeds = seeds(&plan, &data);
        let mut inputs = ChunkInputs::new(&plan);
        let all: Vec<usize> = (0..plan.families.len()).collect();
        let reads = Reads::of(&plan, &all);
        inputs.prepare(&plan, &data, &reads, &reads);
        let b: Vec<_> = all
            .iter()
            .map(|&f| run_family::<Rational, L>(&plan, f, &inputs, &seeds, &data).unwrap())
            .collect();
        let merged = plan.merged.as_ref().unwrap();
        let mut handed: Vec<Vec<usize>> = Vec::new();
        let mut m = Vec::new();
        let stats = run_merged(
            &plan,
            &parts(merged, 1, &plan),
            &mut inputs,
            &seeds,
            &data,
            |done| {
                handed.push(done.iter().map(|n| n.family).collect());
                m.extend(done);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(handed, [vec![0], vec![1]]);
        assert_eq!((stats.states, stats.chunks), (5, 2));
        for (mine, theirs) in m.iter().zip(&b) {
            assert_eq!(mine.family, theirs.numerator.family);
            assert_eq!(mine.accumulators, theirs.numerator.accumulators);
        }
        // Family 0: 3 × 2 × 3i × conj(A0) at lane 0 (A0 = 1 + i).
        let value = m[0].lane::<L>(0, 0, 0);
        assert_eq!(value[0].re, Rational::from(18));
        assert_eq!(value[0].im, Rational::from(18));
    }
}
