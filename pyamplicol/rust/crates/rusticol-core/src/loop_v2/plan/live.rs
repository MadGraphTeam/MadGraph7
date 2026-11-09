// SPDX-License-Identifier: 0BSD

//! Live object rows of the schedule-B steps (MQ-SELECT, DESIGN.md §8.5).
//!
//! A plan whose observable columns keep some Born helicity rows only
//! (`column_rows`: helicity-resolved columns and leading-colour helicity
//! selections) seeds zeros on every other row ([`super::super::seeds`]).
//! Every object row of a step is a sum over row links of a source row times
//! a block row, so the rows no kept seed row reaches hold exact zeros, and
//! the rows no trace reads are never used. The loader drops the row links
//! of both (forward: a link whose source row is dead; backward: a link
//! whose output row no later link or trace reads), the split groups' target
//! rows no kept part link writes, the parts, groups, edges and steps left
//! without one, and recounts each step's modelled work and each family's
//! inputs (fused factors, vertex tensors, split currents): the cost of a
//! step follows its kept rows. The fused factors and split currents are
//! built on the rows the kept links read only ([`super::ReadRows`]; not
//! with a schedule-F or schedule-M program, which reads them too).
//!
//! A split part's link writes a target row of its group, so a row is live in
//! a split step when a part link from a live row writes it, and a part link
//! is kept when its source row is live and its output row is read; a target
//! row is kept when a kept part link of its group writes it.
//!
//! The kept links run in the same order on the same values, so every kept
//! sum is the one the full plan forms less additions of exact zeros: the
//! numerators are equal bitwise up to the sign of a zero (and a non-finite
//! block value on a dropped row no longer reaches a kept one). A plan whose
//! columns keep every row is left as it is. Schedules F and M keep their own
//! programs.
//!
//! **Dressed rows** (route R2, [`LoopProgramV4::dressed`]): the leaves seed
//! one constant row, so nothing dies forward; a trace step closes each
//! object row into the columns of its Born row ([`super::RowTrace`]), so the
//! backward pass keeps a trace step's object rows whose Born row is kept and
//! read by a column (`row_columns`): at load the rows no column reads (the
//! exact helicity-zero filter's drops), per call also the rows a helicity
//! mask drops (a loop-induced single-helicity selection). The pruned copy
//! keeps its trace steps' row closures.

use super::super::kernels::RowLink;
use super::{
    Content, Edge, Family, LinkInterner, LoopProgramV4, PackedLinks, ReadRows, Source, SplitGroup,
    SplitPart, SplitStep, Step, StepCost, Vertex, fused_cost, step_shape,
};
use crate::RusticolResult;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

impl LoopProgramV4 {
    /// Drop the schedule-B row links, edges and steps that only carry the
    /// zeros of the Born rows no column keeps (module docs); the split
    /// parts' shortened link lists are interned in `links`.
    pub(super) fn prune_dead_rows(&mut self, links: &mut LinkInterner) {
        let (seeded, columns, traced) = if self.dressed {
            let Some(traced) = self.traced_rows(None) else {
                return;
            };
            (vec![true], self.object_columns(), Some(traced))
        } else {
            if !self.column_rows.restricts() {
                return;
            }
            let columns = self.columns;
            let seeded: Vec<bool> = (0..self.rows * columns)
                .map(|row| self.column_rows.keeps(row / columns, row % columns))
                .collect();
            (seeded, columns, None)
        };
        for family in &mut self.families {
            (*family, _) = prune_family(
                family,
                &seeded,
                columns,
                &self.contents,
                &self.vertices,
                links,
                traced.as_deref(),
            );
        }
        // The factor and current rows the kept links read (schedules F and
        // M read the plan's chunk inputs too, with links of their own).
        if self.forward.is_none() && self.merged.is_none() {
            self.read_rows = Some(self.rows_read());
        }
    }

    /// The Born rows a dressed-row plan's trace steps close: those read by
    /// a column (`row_columns`) and kept by `mask` (one flag per Born row;
    /// None keeps all); `None` when that is every row a column reads
    /// (nothing to prune).
    fn traced_rows(&self, mask: Option<&[bool]>) -> Option<Vec<bool>> {
        let read = |h: usize| !self.row_columns[h].is_empty();
        let traced: Vec<bool> = (0..self.rows)
            .map(|h| read(h) && mask.is_none_or(|kept| kept[h]))
            .collect();
        if traced.iter().all(|&kept| kept) {
            return None;
        }
        Some(traced)
    }

    /// The content rows of each fused factor and the term rows of each
    /// split current that the kept links of schedule B read.
    fn rows_read(&self) -> ReadRows {
        self.rows_read_by(self.families.iter())
    }

    /// [`Self::rows_read`] by the steps of `families` only.
    fn rows_read_by<'a>(&self, families: impl Iterator<Item = &'a Family>) -> ReadRows {
        let mut factors: Vec<Vec<bool>> = self
            .factors
            .iter()
            .map(|&(content, _)| vec![false; self.contents[content].rows])
            .collect();
        let mut currents: Vec<Vec<bool>> = self
            .currents
            .iter()
            .map(|&(content, term)| vec![false; self.contents[content].terms[term].rows.len()])
            .collect();
        for step in families.flat_map(|family| &family.steps) {
            match &step.split {
                None => {
                    for edge in &step.edges {
                        for link in &edge.links {
                            factors[edge.factor][link.block as usize] = true;
                        }
                    }
                }
                Some(split) => {
                    for part in split.parts.iter().flatten() {
                        for link in part.links.iter() {
                            currents[part.current][link.block as usize] = true;
                        }
                    }
                }
            }
        }
        ReadRows { factors, currents }
    }
}

/// Per-call row liveness of schedule B (MF-MASKS-ROWS, DESIGN.md §8.5).
///
/// An evaluation that keeps some Born rows only (an evaluation-time
/// selection's `rows` mask on the template plan of a stored process, whose
/// columns keep every row) seeds zeros on the other rows, as a plan whose
/// columns keep those rows only does. Per call, each family it dresses is
/// pruned as the loader prunes such a plan ([`prune_family`]: the same
/// forward and backward link liveness, the edges, groups and steps left
/// without a link dropped, the work and the family's inputs recounted), and
/// the fused factor and split current rows the dressed links read are
/// recorded ([`Self::read`]), so the call builds, dresses and (its tree
/// program, `trees_rows.rs`) evaluates what that plan would: the same kept
/// sums in the same order, less additions of exact zeros, equal bitwise up
/// to the sign of a zero. A family left without a step has a zero numerator
/// and is not dressed (its lanes are absent, as under a family mask).
///
/// Memory: a pruned copy is built fresh at the lengths it keeps (a split
/// part's list keeping every link stays shared with the template, a shorter
/// one is packed and interned). In a seeded plan it is kept only when it runs
/// at most [`KEPT_SHARE`] of its family's row links (it then saves at least
/// the rest of its dressing): a family keeping more runs whole on the masked
/// seeds (the template's steps, the unpruned path's values). In a
/// dressed-row plan (a loop-induced helicity selection, lit-dress) a copy is
/// kept whenever it drops a link, so the call dresses, closes, scatters and
/// reduces exactly the restriction of its kept rows (a copy is a subset of
/// its template, the memory a call adds); a family whose every link is
/// reached runs its template. A copy derives its walk data (`plan::Walk`:
/// written rows, closed trace rows) on its first walk, as a loaded family
/// does. A family left without a step is never
/// dressed. In a seeded plan, if the kept copies together would hold more
/// heap than the dressed families themselves (their own lists, steps,
/// edges, fused links, split groups and parts, and the part lists they
/// read, each once), every family runs whole. A dressed-row plan keeps its
/// copies regardless (USER 2026-11-08 ~22:34: a selection evaluates only
/// its rows): each copy keeps a subset of its template's steps, edges and
/// links, so the copies add at most the sum of the listed families' own
/// heaps (the report's "row-pruned family bytes" against "row-liveness
/// dressed bytes"; the call's peak is measured). The families are pruned
/// on the calling thread with one link interner, so the copies and the
/// pruning's scratch sit in one allocator arena and equal shortened lists
/// are shared across families (a parallel prune spread them over the
/// worker threads' arenas: about 50 MB more peak on u u~ > g g g g g at
/// eight threads for 17 MB of copies). The executor keeps one liveness and
/// drops it before it forms another (`V2Executor::live_rows`).
pub struct LiveRows {
    /// The kept Born rows, one flag per plan row.
    pub rows: Vec<bool>,
    /// The families the call dresses (ascending) and their pruned copies
    /// (`None`: the family runs whole).
    listed: Vec<usize>,
    pruned: Vec<Option<Family>>,
    /// The listed families with a step left (ascending).
    live: Vec<usize>,
    /// Row links the listed families run (pruned or whole).
    links: usize,
    /// The heap of the pruned copies (their own lists, `steps_bytes` and
    /// shortened part lists), and of the listed template families (the
    /// same, with the part lists they read, each once).
    bytes: usize,
    dressed: usize,
    /// The factor and current rows the dressed links read.
    pub read: Arc<ReadRows>,
}

/// The largest share (numerator, denominator) of its family's row links a
/// kept pruned copy runs ([`LiveRows`]): a copy costs memory and saves the
/// dressing of the links it drops, so it must at least halve its family's.
pub const KEPT_SHARE: (usize, usize) = (1, 2);

/// The heap of a family's own short lists: sequence, orders, leaves and
/// inputs.
fn own_bytes(family: &Family) -> usize {
    use std::mem::size_of;
    family.sequence.len() * size_of::<(u64, usize)>()
        + family.orders.len() * size_of::<usize>()
        + family.leaves.len() * size_of::<super::Leaf>()
        + (family.factors.len() + family.vertices.len() + family.currents.len())
            * size_of::<usize>()
        + family.support.as_ref().map_or(0, |support| {
            (support.columns.len() + support.slots.len()) * size_of::<u32>()
        })
}

/// The heap of a family's steps that a pruned copy allocates anew: steps,
/// edges, fused links, split groups (with their targets), part lists and
/// parts (the parts' link lists are packed and shared, `plan/packed.rs`).
fn steps_bytes(family: &Family) -> usize {
    use std::mem::size_of;
    family
        .steps
        .iter()
        .map(|step| {
            let edges: usize = step
                .edges
                .iter()
                .map(|edge| size_of::<Edge>() + edge.links.len() * size_of::<RowLink>())
                .sum();
            let split = step.split.as_ref().map_or(0, |split| {
                let groups: usize = split
                    .groups
                    .iter()
                    .map(|group| size_of::<SplitGroup>() + group.targets.len() * size_of::<u32>())
                    .sum();
                let parts: usize = split
                    .parts
                    .iter()
                    .map(|parts| size_of::<Vec<SplitPart>>() + parts.len() * size_of::<SplitPart>())
                    .sum();
                groups + parts
            });
            size_of::<Step>() + edges + split
        })
        .sum::<usize>()
}

/// The heap of the distinct split-part lists of `family` not yet in
/// `seen` (which then holds them).
fn list_bytes(family: &Family, seen: &mut HashSet<*const PackedLinks>) -> usize {
    family
        .steps
        .iter()
        .filter_map(|step| step.split.as_ref())
        .flat_map(|split| split.parts.iter().flatten())
        .filter(|part| seen.insert(Arc::as_ptr(&part.links)))
        .map(|part| part.links.shared_bytes())
        .sum()
}

/// Row links of a family's steps (fused edges and split parts).
fn family_links(family: &Family) -> usize {
    family
        .steps
        .iter()
        .map(|step| -> usize {
            match &step.split {
                None => step.edges.iter().map(|edge| edge.links.len()).sum(),
                Some(split) => split.parts.iter().flatten().map(|p| p.links.len()).sum(),
            }
        })
        .sum()
}

impl LiveRows {
    /// The families the liveness was formed for.
    pub fn listed(&self) -> &[usize] {
        &self.listed
    }

    /// The listed families with a step left.
    pub fn families(&self) -> &[usize] {
        &self.live
    }

    /// Family `f`'s pruned copy (`None`: not listed, or it runs whole).
    pub fn family(&self, f: usize) -> Option<&Family> {
        self.listed
            .binary_search(&f)
            .ok()
            .and_then(|index| self.pruned[index].as_ref())
    }

    /// Row links the listed families run (pruned copies or whole families).
    pub fn links(&self) -> usize {
        self.links
    }

    /// The heap of the pruned copies in bytes (their steps, edges, fused
    /// links, groups, parts and shortened part lists).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The heap of the listed template families in bytes (with the part
    /// lists they read, each once): what the copies may hold at most.
    pub fn dressed_bytes(&self) -> usize {
        self.dressed
    }

    /// The listed families that run a pruned copy.
    pub fn pruned(&self) -> usize {
        self.pruned.iter().flatten().count()
    }
}

impl LoopProgramV4 {
    /// The row liveness of an evaluation keeping the Born rows `rows` (one
    /// flag per plan row) and dressing `families` (ascending) by schedule B
    /// ([`LiveRows`]): a seeded plan keeps a family's pruned copy up to
    /// [`KEPT_SHARE`] of its links (and none when the copies would hold
    /// more heap than their families); a dressed-row plan keeps every copy
    /// that drops a link (the selection's row restriction). `None` when
    /// `rows` drops no row a column seeds (nothing to prune). The families
    /// are pruned on the calling thread.
    pub fn live_rows(&self, rows: &[bool], families: &[usize]) -> RusticolResult<Option<LiveRows>> {
        self.live_rows_keeping(rows, families, KEPT_SHARE)
    }

    /// [`Self::live_rows`] keeping a seeded plan's pruned copy that runs at
    /// most `share` (numerator, denominator) of its family's row links;
    /// `share` does not apply to a dressed-row plan, whose copies run
    /// whenever they drop a link.
    pub fn live_rows_keeping(
        &self,
        rows: &[bool],
        families: &[usize],
        share: (usize, usize),
    ) -> RusticolResult<Option<LiveRows>> {
        super::require(rows.len() == self.rows, "one kept flag per Born row")?;
        super::require(
            families.windows(2).all(|pair| pair[0] < pair[1])
                && families.last().is_none_or(|&f| f < self.families.len()),
            "ascending plan families",
        )?;
        let (seeded, columns, traced) = if self.dressed {
            // Dressed rows: the trace steps close the kept Born rows a
            // column reads (`traced_rows`; the load already dropped the
            // rows no column reads).
            let traced = self.traced_rows(Some(rows));
            let unchanged = match &traced {
                None => true,
                Some(traced) => {
                    (0..self.rows).all(|h| traced[h] == !self.row_columns[h].is_empty())
                }
            };
            if unchanged {
                return Ok(None);
            }
            (vec![true], self.object_columns(), traced)
        } else {
            let columns = self.columns;
            let keeps = |row: usize| self.column_rows.keeps(row / columns, row % columns);
            let seeded: Vec<bool> = (0..self.rows * columns)
                .map(|row| rows[row / columns] && keeps(row))
                .collect();
            if seeded
                .iter()
                .enumerate()
                .all(|(row, &seed)| seed == keeps(row))
            {
                return Ok(None);
            }
            (seeded, columns, None)
        };
        // One link interner: the parts' shortened lists are shared across
        // the families (a list keeping every link stays the template's).
        // Per listed family: the copy (None: it runs whole), the links it
        // runs, and the heap the copy adds.
        let mut interner = LinkInterner::default();
        let mut pruned: Vec<(Option<Family>, usize, usize)> = families
            .iter()
            .map(|&f| {
                let template = &self.families[f];
                let (copy, lists) = prune_family(
                    template,
                    &seeded,
                    columns,
                    &self.contents,
                    &self.vertices,
                    &mut interner,
                    traced.as_deref(),
                );
                let mut copy = copy;
                copy.support = self.support_of(&copy);
                let (links, whole) = (family_links(&copy), family_links(template));
                // Dressed rows (loop-induced helicity selections): the call
                // evaluates exactly the restriction of the kept rows, so a
                // family dressing a link outside it runs its copy; one whose
                // every link is reached runs its template (the same links).
                // Seeded plans keep a copy up to `share` of its family's
                // links (the masked seeds' zeros run otherwise).
                let kept = if self.dressed {
                    links < whole
                } else {
                    links * share.1 <= whole * share.0
                };
                if copy.steps.is_empty() || kept {
                    let bytes = own_bytes(&copy) + steps_bytes(&copy) + lists;
                    (Some(copy), links, bytes)
                } else {
                    (None, whole, 0)
                }
            })
            .collect();
        // Seeded plans: the copies of the call together hold no more than
        // the families they replace, else every family with a step runs
        // whole. Dressed rows keep their copies (each a subset of its
        // template: the selection's restriction is the requirement).
        let mut seen = HashSet::new();
        let dressed: usize = families
            .iter()
            .map(|&f| {
                let family = &self.families[f];
                own_bytes(family) + steps_bytes(family) + list_bytes(family, &mut seen)
            })
            .sum();
        let copies: usize = pruned.iter().map(|&(.., bytes)| bytes).sum();
        if !self.dressed && copies > dressed {
            for (&f, entry) in families.iter().zip(&mut pruned) {
                if entry.0.as_ref().is_some_and(|copy| !copy.steps.is_empty()) {
                    *entry = (None, family_links(&self.families[f]), 0);
                }
            }
        }
        fn dressed_family<'a>(
            template: &'a [Family],
            (&f, (copy, ..)): (&usize, &'a (Option<Family>, usize, usize)),
        ) -> &'a Family {
            copy.as_ref().unwrap_or(&template[f])
        }
        let live = families
            .iter()
            .zip(&pruned)
            .filter(|&pair| !dressed_family(&self.families, pair).steps.is_empty())
            .map(|(&f, _)| f)
            .collect();
        let read = Arc::new(
            self.rows_read_by(
                families
                    .iter()
                    .zip(&pruned)
                    .map(|pair| dressed_family(&self.families, pair)),
            ),
        );
        let links = pruned.iter().map(|&(_, links, _)| links).sum();
        let bytes = pruned.iter().map(|&(.., bytes)| bytes).sum();
        Ok(Some(LiveRows {
            rows: rows.to_vec(),
            listed: families.to_vec(),
            pruned: pruned.into_iter().map(|(copy, ..)| copy).collect(),
            live,
            links,
            bytes,
            dressed,
            read,
        }))
    }
}

/// The live rows of each edge's source: the kept seed rows of a leaf, or
/// the live output rows of a child step.
fn sources<'a>(step: &Step, seeded: &'a [bool], live: &'a [Vec<bool>]) -> Vec<&'a [bool]> {
    step.edges
        .iter()
        .map(|edge| match edge.source {
            Source::Leaf(_) => seeded,
            Source::Step(child) => live[child].as_slice(),
        })
        .collect()
}

/// `family` pruned to the object rows `seeded` reaches and the traces read
/// (module docs), built fresh: every list of the result is allocated at the
/// length it keeps, so a pruned family holds its kept links only (a split
/// part's list that keeps every link stays shared with `family`; a shorter
/// one is interned in `interner`). The template is read only. Also returns
/// the heap of the shortened part lists it added to `interner`.
fn prune_family(
    family: &Family,
    seeded: &[bool],
    columns: usize,
    contents: &[Content],
    vertices: &[Vertex],
    interner: &mut LinkInterner,
    traced: Option<&[bool]>,
) -> (Family, usize) {
    let mut lists = 0;
    let steps = &family.steps;
    // Forward: the object rows a kept seed row reaches.
    let mut live: Vec<Vec<bool>> = Vec::with_capacity(steps.len());
    for step in steps {
        let inputs = sources(step, seeded, &live);
        let mut out = vec![false; step.rows * columns];
        let mut mark = |links: &mut dyn Iterator<Item = RowLink>, input: &[bool]| {
            for link in links {
                if input[link.input as usize] {
                    out[link.output as usize] = true;
                }
            }
        };
        match &step.split {
            None => {
                for (edge, input) in step.edges.iter().zip(&inputs) {
                    mark(&mut edge.links.iter().copied(), input);
                }
            }
            Some(split) => {
                for (parts, input) in split.parts.iter().zip(&inputs) {
                    for part in parts {
                        mark(&mut part.links.iter(), input);
                    }
                }
            }
        }
        live.push(out);
    }
    // Backward: keep the links from a live row into a row that is read.
    // Every open object feeds exactly one edge, of a later step, so a step's
    // read rows are complete when the reverse walk reaches it. Each step is
    // built with its kept edges, parts and groups (sources still in the
    // template's numbering), or dropped without an edge left.
    let mut read: Vec<Vec<bool>> = steps
        .iter()
        .map(|step| vec![false; step.rows * columns])
        .collect();
    let mut kept: Vec<Option<Step>> = Vec::with_capacity(steps.len());
    kept.resize_with(steps.len(), || None);
    for s in (0..steps.len()).rev() {
        let step = &steps[s];
        let wanted = match (step.trace, traced, family.row_traces.get(s)) {
            // A dressed trace step closes the object rows of its traced
            // Born rows only.
            (Some(_), Some(traced), Some(Some(trace))) => live[s]
                .iter()
                .zip(&trace.rows)
                .map(|(&live, &row)| live && traced[row as usize])
                .collect(),
            (Some(_), ..) => live[s].clone(),
            (None, ..) => std::mem::take(&mut read[s]),
        };
        let inputs = sources(step, seeded, &live);
        let keep = |link: &RowLink, input: &[bool]| {
            input[link.input as usize] && wanted[link.output as usize]
        };
        let edge_with = |edge: &Edge, links: Vec<RowLink>| Edge {
            source: edge.source,
            factor: edge.factor,
            content: edge.content,
            source_rank: edge.source_rank,
            products: edge.products,
            links,
        };
        let mut edges = Vec::new();
        let split = match &step.split {
            None => {
                for (edge, input) in step.edges.iter().zip(&inputs) {
                    let links = exact(edge.links.iter().copied().filter(|l| keep(l, input)));
                    if links.is_empty() {
                        continue;
                    }
                    if let Source::Step(child) = edge.source {
                        for link in &links {
                            read[child][link.input as usize] = true;
                        }
                    }
                    edges.push(edge_with(edge, links));
                }
                None
            }
            Some(split) => {
                let mut parts = Vec::new();
                for ((edge, edge_parts), input) in step.edges.iter().zip(&split.parts).zip(&inputs)
                {
                    let mut kept_parts = Vec::new();
                    for part in edge_parts {
                        let mut links = Arc::clone(&part.links);
                        let interned = interner.len();
                        interner.retain(&mut links, |link| keep(link, input));
                        if interner.len() > interned {
                            lists += links.shared_bytes();
                        }
                        if links.is_empty() {
                            continue;
                        }
                        if let Source::Step(child) = edge.source {
                            for link in links.iter() {
                                read[child][link.input as usize] = true;
                            }
                        }
                        kept_parts.push(SplitPart {
                            content: part.content,
                            term: part.term,
                            current: part.current,
                            group: part.group,
                            links,
                        });
                    }
                    if !kept_parts.is_empty() {
                        kept_parts.shrink_to_fit();
                        edges.push(edge_with(edge, Vec::new()));
                        parts.push(kept_parts);
                    }
                }
                // A group's target rows are those its kept part links write;
                // a group without one goes, the parts' groups renumbered.
                let mut members: Vec<Vec<(usize, usize)>> = vec![Vec::new(); split.groups.len()];
                for (e, edge_parts) in parts.iter().enumerate() {
                    for (p, part) in edge_parts.iter().enumerate() {
                        members[part.group].push((e, p));
                    }
                }
                let mut written = vec![false; step.rows * columns];
                let mut renumbered: Vec<Option<usize>> = vec![None; split.groups.len()];
                let mut groups = Vec::new();
                for (g, (group, members)) in split.groups.iter().zip(&members).enumerate() {
                    let links = || members.iter().flat_map(|&(e, p)| parts[e][p].links.iter());
                    links().for_each(|link| written[link.output as usize] = true);
                    let targets = exact(
                        group
                            .targets
                            .iter()
                            .copied()
                            .filter(|&t| written[t as usize]),
                    );
                    links().for_each(|link| written[link.output as usize] = false);
                    if !targets.is_empty() {
                        renumbered[g] = Some(groups.len());
                        groups.push(SplitGroup {
                            vertex: group.vertex,
                            shift: group.shift,
                            instance: group.instance,
                            products: group.products,
                            targets,
                        });
                    }
                }
                for part in parts.iter_mut().flatten() {
                    part.group = renumbered[part.group].expect("a kept part's group has a target");
                }
                groups.shrink_to_fit();
                parts.shrink_to_fit();
                Some(SplitStep { groups, parts })
            }
        };
        if edges.is_empty() {
            continue;
        }
        edges.shrink_to_fit();
        kept[s] = Some(Step {
            seeded: step.seeded,
            trace: step.trace,
            rows: step.rows,
            d_alpha: step.d_alpha,
            d_out: step.d_out,
            degree: step.degree,
            rank_in: step.rank_in,
            rank_out: step.rank_out,
            edges,
            split,
            costs: step.costs,
        });
    }
    drop(live);
    // Renumber the kept steps' sources and recount the modelled work and
    // the family's inputs. An edge with a link reads its child step, which
    // then has a link left too (every open object feeds one edge).
    let mut index: Vec<Option<usize>> = vec![None; steps.len()];
    let mut pruned: Vec<Step> = Vec::with_capacity(kept.iter().flatten().count());
    // A dressed-row family's row closures, by kept step (empty for seeded
    // plans).
    let mut row_traces = Vec::new();
    for (s, step) in kept.into_iter().enumerate() {
        let Some(mut step) = step else {
            continue;
        };
        if !family.row_traces.is_empty() {
            row_traces.push(family.row_traces[s].clone());
        }
        for edge in &mut step.edges {
            if let Source::Step(child) = edge.source {
                let child = index[child].expect("the child of a kept edge is kept");
                edge.source = Source::Step(child);
            }
        }
        step.costs = costs(&step, contents, vertices);
        index[s] = Some(pruned.len());
        pruned.push(step);
    }
    let mut factors = BTreeSet::new();
    let mut instances = BTreeSet::new();
    let mut currents = BTreeSet::new();
    for step in &pruned {
        match &step.split {
            None => factors.extend(step.edges.iter().map(|edge| edge.factor)),
            Some(split) => {
                instances.extend(split.groups.iter().map(|group| group.instance));
                currents.extend(split.parts.iter().flatten().map(|part| part.current));
            }
        }
    }
    let family = Family {
        sequence: family.sequence.clone(),
        rank: family.rank,
        orders: family.orders.clone(),
        leaves: family.leaves.clone(),
        steps: pruned,
        factors: factors.into_iter().collect(),
        vertices: instances.into_iter().collect(),
        currents: currents.into_iter().collect(),
        row_traces,
        support: None,
        walk: std::sync::OnceLock::new(),
    };
    (family, lists)
}

/// The items of `items` in a list of their own length.
fn exact<T>(items: impl Iterator<Item = T>) -> Vec<T> {
    let mut list: Vec<T> = items.collect();
    list.shrink_to_fit();
    list
}

/// The modelled work of a pruned step (as the loader counts it,
/// `fused_cost` and `split_candidate`): the fused form's of a fused step,
/// and the split form's of a step that runs split (a fused step's split
/// alternative is not recounted, nor a split step's fused one: its edges
/// keep no links, `release_split_links`).
fn costs(step: &Step, contents: &[Content], vertices: &[Vertex]) -> (StepCost, Option<StepCost>) {
    let Some(split) = &step.split else {
        return (
            fused_cost(step.seeded, step.d_alpha, &step.edges, contents),
            None,
        );
    };
    let fused = step.costs.0;
    let rank_in = if step.seeded { 0 } else { step.rank_in };
    let (cut, m_in) = step_shape(step.seeded, step.d_alpha, rank_in);
    let mut iterations = 0;
    for part in split.parts.iter().flatten() {
        let incoming = if step.seeded {
            1
        } else {
            contents[part.content].d_in
        };
        let width = vertices[split.groups[part.group].vertex].width;
        iterations += part.links.len() * cut * incoming * width;
    }
    for group in &split.groups {
        iterations += group.targets.len() * cut * vertices[group.vertex].entries.len();
    }
    let split_cost = StepCost {
        work: (iterations * m_in) as u64,
        iterations: iterations as u64,
    };
    (fused, Some(split_cost))
}

#[cfg(test)]
mod tests {
    use crate::loop_v2::dress::{ChunkInputs, LaneData, Reads, V2Batch, Workspaces};
    use crate::loop_v2::plan::{LoopProgramV4, Source};
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::{run_family, run_family_on};
    use symbolica::domains::float::Complex;
    use symbolica::domains::rational::Rational;

    /// Two scalar bubble families of a two-leg "process" whose leg 0 has two
    /// helicities (Born rows 0 and 1), each a seeded step on leg 0 and a
    /// trace step on leg 1. Family 0's leg-0 content has both helicity rows
    /// (factor 2 on row 0, the content's constant times 5 on row 1: one
    /// kernel per row); family 1's has helicity +1 only (row 1).
    const PLAN: &str = r#"{"schema":4,"legs":2,"rows":2,"helicities":[[-1,1],[0]],"columns":COLUMNS,
        "average":1,"born_components":1,"constants":3,"masses":1,"block_rows":[],
        "kernels":[{"inputs":[],"outputs":[[[0,[]]]]},{"inputs":[],"outputs":[[[1,[]]]]},
                   {"inputs":[],"outputs":[[[2,[]]]]}],
        "contents":[
            {"d_in":1,"d_out":1,"degree":0,"rows":2,"mask":1,"row_helicities":[[-1],[1]],
             "fused":[[0,0,0,0,0,0]],
             "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
                       "bindings":[{"kernel":0,"inputs":[],"entries":[0]}]},
                      {"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[1,[]]],
                       "bindings":[{"kernel":2,"inputs":[],"entries":[0]}]}]},
            {"d_in":1,"d_out":1,"degree":0,"rows":1,"mask":2,"row_helicities":[[0]],
             "fused":[[0,0,0,0,0,0]],
             "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
                       "bindings":[{"kernel":1,"inputs":[],"entries":[0]}]}]},
            {"d_in":1,"d_out":1,"degree":0,"rows":1,"mask":1,"row_helicities":[[1]],
             "fused":[[0,0,0,0,0,0]],
             "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
                       "bindings":[{"kernel":2,"inputs":[],"entries":[0]}]}]}],
        "factors":[[0,0],[1,1],[2,0]],
        "seed_classes":[{"columns":[WEIGHTS]}],"seeds":[{"classes":[[0,"1","0"]]}],
        "families":[FAMILY0,FAMILY1],"forward":null,"merged":null,"column_rows":ROWS}"#;
    const FAMILY: &str = r#"{"sequence":[[1,0],[2,0]],"rank":0,"orders":[0],
        "leaves":[{"seed":0,"weight":"3"}],
        "steps":[{"seeded":true,"trace":-1,"out":2,"d_alpha":1,"d_out":1,"degree":0,
                  "rank_in":0,"edges":[{"leaf":0,"child":-1,"factor":FACTOR}]},
                 {"seeded":false,"trace":0,"out":0,"d_alpha":1,"d_out":1,"degree":0,
                  "rank_in":0,"edges":[{"leaf":-1,"child":0,"factor":1}]}]}"#;

    /// The plan with `column_rows` = `rows` (JSON) over `columns` columns
    /// (one seed-weight list each).
    fn plan(rows: &str, columns: usize) -> LoopProgramV4 {
        let weights = vec![r#"[[0,"1","0"]]"#; columns].join(",");
        let text = PLAN
            .replace("COLUMNS", &columns.to_string())
            .replace("WEIGHTS", &weights)
            .replace("ROWS", rows)
            .replace("FAMILY0", &FAMILY.replace("FACTOR", "0"))
            .replace("FAMILY1", &FAMILY.replace("FACTOR", "2"));
        LoopProgramV4::load(text.as_bytes()).unwrap()
    }

    /// Two points; Born row 1 of both is zero when `kept_row_only`.
    fn batch(kept_row_only: bool) -> V2Batch<Rational> {
        let momentum =
            |e: i64| -> Vec<Rational> { [e, 0, 0, e, e, 0, 0, -e].map(Rational::from).to_vec() };
        let complex = |re: i64, im: i64| Complex::new(Rational::from(re), Rational::from(im));
        let other = |re, im| {
            if kept_row_only {
                complex(0, 0)
            } else {
                complex(re, im)
            }
        };
        V2Batch {
            momenta: vec![momentum(1), momentum(2)],
            scales: vec![Rational::from(1); 2],
            masses: vec![Rational::from(0)],
            masses_imaginary: Vec::new(),
            leg_masses: Vec::new(),
            constants: vec![complex(2, 0), complex(0, 3), complex(5, 0)],
            block_rows: vec![Vec::new(); 2],
            born: vec![
                vec![complex(1, 1), other(7, -1)],
                vec![complex(3, -2), other(-2, 4)],
            ],
        }
    }

    /// Per family, the accumulators of schedule B on both points.
    fn numerators(plan: &LoopProgramV4, batch: &V2Batch<Rational>) -> Vec<Vec<Vec<Rational>>> {
        const L: usize = 2;
        let zero = Rational::from(0);
        let data = LaneData::<Rational, L>::new(plan, batch, &[0, 1], &zero).unwrap();
        let seeds = seeds(plan, &data);
        let mut inputs = ChunkInputs::new(plan);
        let all: Vec<usize> = (0..plan.families.len()).collect();
        let reads = Reads::of(plan, &all);
        inputs.prepare(plan, &data, &reads, &reads);
        all.iter()
            .map(|&f| {
                run_family::<Rational, L>(plan, f, &inputs, &seeds, &data)
                    .unwrap()
                    .numerator
                    .accumulators
            })
            .collect()
    }

    #[test]
    fn plans_keeping_every_row_are_left_as_they_are() {
        for rows in ["null", "[null]"] {
            let plan = plan(rows, 1);
            assert!(plan.read_rows.is_none());
            let links: Vec<usize> = plan.families[0]
                .steps
                .iter()
                .map(|step| step.edges[0].links.len())
                .collect();
            assert_eq!(links, [2, 1]);
            assert_eq!(plan.families[1].steps.len(), 2);
        }
    }

    /// Keeping Born row 0 drops family 0's link from row 1, the factor row
    /// it read, and family 1 (whose leg-0 content has row 1 only) entirely;
    /// the numerators equal the full plan's on Born components that vanish
    /// on row 1, and differ from them where row 1 is not zero.
    #[test]
    fn steps_keep_the_rows_their_kept_born_rows_reach() {
        let pruned = plan("[[0]]", 1);
        let full = plan("null", 1);
        let step = &pruned.families[0].steps[0];
        let links: Vec<_> = step.edges[0]
            .links
            .iter()
            .map(|l| (l.output, l.input, l.block))
            .collect();
        assert_eq!(links, [(0, 0, 0)]);
        assert_eq!(step.costs.0.iterations, 1);
        assert_eq!(full.families[0].steps[0].costs.0.iterations, 2);
        assert!(pruned.families[1].steps.is_empty());
        assert!(pruned.families[1].factors.is_empty());
        assert_eq!(pruned.families[0].factors, [0, 1]);
        let read = pruned.read_rows.as_ref().unwrap();
        assert_eq!(read.factors, [vec![true, false], vec![true], vec![false]]);
        assert_eq!(
            numerators(&pruned, &batch(false)),
            numerators(&full, &batch(true))
        );
        assert_ne!(
            numerators(&pruned, &batch(false)),
            numerators(&full, &batch(false))
        );
        // 3 (leaf) x 2 x 3i x conj(1 + i) at lane 0 of family 0.
        let value = &numerators(&pruned, &batch(false))[0][0];
        assert_eq!(
            (&value[0], &value[2]),
            (&Rational::from(18), &Rational::from(18))
        );
    }

    /// Per column: a column keeping every row keeps every link of its own,
    /// and the other column's dead row only; the children are renumbered.
    #[test]
    fn rows_are_kept_per_observable_column() {
        let pruned = plan("[null,[0]]", 2);
        let step = &pruned.families[0].steps[0];
        let links: Vec<_> = step.edges[0]
            .links
            .iter()
            .map(|l| (l.output, l.input))
            .collect();
        // Object rows `row x 2 + column`: column 1 of Born row 1 is dropped.
        assert_eq!(links, [(0, 0), (1, 1), (0, 2)]);
        // Family 1 reads Born row 1, which column 0 keeps.
        assert_eq!(pruned.families[1].steps.len(), 2);
        let trace = &pruned.families[1].steps[1];
        assert_eq!(trace.edges[0].source, Source::Step(0));
        assert_eq!(trace.edges[0].links.len(), 1);
        let read = pruned.read_rows.as_ref().unwrap();
        assert_eq!(read.factors, [vec![true, true], vec![true], vec![true]]);
    }

    /// The links of every step's first edge, per family.
    fn links_of(families: &[&crate::loop_v2::plan::Family]) -> Vec<Vec<Vec<(u32, u32, u32)>>> {
        families
            .iter()
            .map(|family| {
                family
                    .steps
                    .iter()
                    .map(|step| {
                        step.edges[0]
                            .links
                            .iter()
                            .map(|l| (l.output, l.input, l.block))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// A call's row liveness on a plan whose columns keep every row (an
    /// evaluation-time selection on the template) prunes the families as
    /// the loader prunes a plan whose columns keep those rows: the same
    /// links, factor rows and costs; the family without a step left is not
    /// live; its numerators equal the row-restricted plan's bitwise; a mask
    /// keeping every row prunes nothing.
    #[test]
    fn a_call_keeping_some_rows_prunes_as_the_row_restricted_plan() {
        let template = plan("null", 1);
        let restricted = plan("[[0]]", 1);
        assert!(
            template
                .live_rows(&[true, true], &[0, 1])
                .unwrap()
                .is_none()
        );
        assert!(template.live_rows(&[true], &[0, 1]).is_err());
        assert!(template.live_rows(&[true, false], &[1, 0]).is_err());
        let live = template
            .live_rows_keeping(&[true, false], &[0, 1], (1, 1))
            .unwrap()
            .unwrap();
        assert_eq!(live.listed(), [0, 1]);
        assert_eq!(live.families(), [0]);
        let pruned: Vec<_> = (0..2).map(|f| live.family(f).unwrap()).collect();
        let reference: Vec<_> = restricted.families.iter().collect();
        assert_eq!(links_of(&pruned), links_of(&reference));
        for (a, b) in pruned.iter().zip(&reference) {
            assert_eq!(a.factors, b.factors);
            let costs = |f: &crate::loop_v2::plan::Family| -> Vec<u64> {
                f.steps.iter().map(|s| s.costs.0.iterations).collect()
            };
            assert_eq!(costs(a), costs(b));
        }
        let read = restricted.read_rows.as_ref().unwrap();
        assert_eq!(live.read.factors, read.factors);
        assert_eq!(live.read.currents, read.currents);
        assert_eq!(live.links(), 2);
        // Family 0's numerators on the call's liveness (Born row 1 zero, as
        // the call's sources make it) = the restricted plan's.
        const L: usize = 2;
        let zero = Rational::from(0);
        let batch = batch(true);
        let data = LaneData::<Rational, L>::new(&template, &batch, &[0, 1], &zero).unwrap();
        let seeds = seeds(&template, &data);
        let mut inputs = ChunkInputs::new(&template).with_rows(Some(live.read.clone()));
        let reads = Reads::of_families(std::iter::once(live.family(0).unwrap()));
        inputs.prepare(&template, &data, &reads, &reads);
        let run = run_family_on::<Rational, L>(
            &template,
            0,
            live.family(0).unwrap(),
            &inputs,
            &seeds,
            &data,
            &Workspaces::new(1, &zero),
        )
        .unwrap();
        assert_eq!(
            run.numerator.accumulators,
            numerators(&restricted, &batch)[0]
        );
        assert_eq!(run.numerator.accumulators, numerators(&template, &batch)[0]);
        // Per column: column 1 keeps row 0 only and the call row 1 only, so
        // only object row 2 (row 1, column 0) is seeded.
        let per_column = plan("[null,[0]]", 2);
        let live = per_column
            .live_rows_keeping(&[false, true], &[0, 1], (1, 1))
            .unwrap()
            .unwrap();
        let pairs = |links: &[crate::loop_v2::kernels::RowLink]| -> Vec<(u32, u32)> {
            links.iter().map(|l| (l.output, l.input)).collect()
        };
        let full = &per_column.families[0].steps[0].edges[0].links;
        let expected: Vec<_> = pairs(full)
            .into_iter()
            .filter(|&(_, input)| input == 2)
            .collect();
        assert_eq!(expected.len(), 1);
        assert_eq!(
            pairs(&live.family(0).unwrap().steps[0].edges[0].links),
            expected
        );
        // The copies are allocated at the lengths they keep.
        let copy = live.family(0).unwrap();
        assert_eq!(copy.steps.capacity(), copy.steps.len());
        for step in &copy.steps {
            assert_eq!(step.edges.capacity(), step.edges.len());
            for edge in &step.edges {
                assert_eq!(edge.links.capacity(), edge.links.len());
            }
        }
    }

    /// A family whose pruned copy would keep more than [`KEPT_SHARE`] of
    /// its row links runs whole on the masked seeds: no copy, its links and the
    /// rows they read counted; values bitwise the row-restricted plan's.
    #[test]
    fn a_family_keeping_most_of_its_links_runs_whole() {
        let template = plan("null", 1);
        let restricted = plan("[[0]]", 1);
        // Family 0 keeps both steps and 2 of its 3 links (whole); family 1
        // nothing (pruned to no step, not live).
        let live = template
            .live_rows(&[true, false], &[0, 1])
            .unwrap()
            .unwrap();
        assert!(live.family(0).is_none());
        assert!(live.family(1).unwrap().steps.is_empty());
        assert_eq!(
            (live.families(), live.links(), live.pruned()),
            (&[0][..], 3, 1)
        );
        assert_eq!(
            live.read.factors,
            [vec![true, true], vec![true], vec![false]]
        );
        const L: usize = 2;
        let zero = Rational::from(0);
        let batch = batch(true);
        let data = LaneData::<Rational, L>::new(&template, &batch, &[0, 1], &zero).unwrap();
        let seeds = seeds(&template, &data);
        let mut inputs = ChunkInputs::new(&template).with_rows(Some(live.read.clone()));
        let family = &template.families[0];
        let reads = Reads::of_families(std::iter::once(family));
        inputs.prepare(&template, &data, &reads, &reads);
        let run = run_family_on::<Rational, L>(
            &template,
            0,
            family,
            &inputs,
            &seeds,
            &data,
            &Workspaces::new(1, &zero),
        )
        .unwrap();
        assert_eq!(
            run.numerator.accumulators,
            numerators(&restricted, &batch)[0]
        );
    }

    /// The executor keeps its latest row liveness and returns it again for
    /// the same rows and family mask; other rows or masks form a new one
    /// (the latest dropped first), and a mask of the wrong length is
    /// refused.
    #[test]
    fn the_executor_reuses_its_latest_row_liveness() {
        use crate::loop_execution::v2::{Schedule, V2Executor};
        use std::sync::Arc;
        let executor = V2Executor::new(Arc::new(plan("null", 1)), 2, Some(1)).unwrap();
        let first = executor
            .live_rows(&[true, false], None, Schedule::B)
            .unwrap()
            .unwrap();
        let again = executor
            .live_rows(&[true, false], None, Schedule::B)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        // `first`, `again` and the executor's one.
        assert_eq!(Arc::strong_count(&first), 3);
        let masked = executor
            .live_rows(&[true, false], Some(&[true, false]), Schedule::B)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &masked));
        // A miss dropped the executor's before forming the new one.
        assert_eq!(Arc::strong_count(&first), 2);
        assert_eq!(masked.listed(), [0]);
        let other = executor
            .live_rows(&[false, true], None, Schedule::B)
            .unwrap()
            .unwrap();
        assert_eq!(other.families(), [0, 1]);
        assert!(
            executor
                .live_rows(&[true, true], None, Schedule::B)
                .unwrap()
                .is_none()
        );
        assert!(
            executor
                .live_rows(&[true, false], Some(&[true]), Schedule::B)
                .is_err()
        );
    }

    /// Only schedule B takes a row liveness: schedules M and F dress their
    /// own programs whole, so a rows mask with them forms none (a liveness
    /// passed to them is refused); the mask itself still holds.
    #[test]
    fn only_schedule_b_forms_a_row_liveness() {
        use crate::loop_execution::v2::{Schedule, V2Executor};
        use std::sync::Arc;
        let executor = V2Executor::new(Arc::new(plan("null", 1)), 2, Some(1)).unwrap();
        for schedule in [Schedule::M, Schedule::F] {
            let live = executor.live_rows(&[true, false], None, schedule).unwrap();
            assert!(live.is_none(), "{schedule:?}");
        }
        let live = executor
            .live_rows(&[true, false], None, Schedule::B)
            .unwrap();
        assert_eq!(live.unwrap().families(), [0]);
        assert!(
            executor
                .live_rows(&[true, false], Some(&[true]), Schedule::M)
                .is_err()
        );
    }
}
