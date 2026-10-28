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
//! step follows its kept rows.
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

use super::{
    Content, Family, LoopProgramV4, Source, Step, StepCost, Vertex, fused_cost, step_shape,
};
use std::collections::BTreeSet;

impl LoopProgramV4 {
    /// Drop the schedule-B row links, edges and steps that only carry the
    /// zeros of the Born rows no column keeps (module docs).
    pub(super) fn prune_dead_rows(&mut self) {
        if !self.column_rows.restricts() {
            return;
        }
        let columns = self.columns;
        let seeded: Vec<bool> = (0..self.rows * columns)
            .map(|row| self.column_rows.keeps(row / columns, row % columns))
            .collect();
        for family in &mut self.families {
            prune_family(family, &seeded, columns, &self.contents, &self.vertices);
        }
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

fn prune_family(
    family: &mut Family,
    seeded: &[bool],
    columns: usize,
    contents: &[Content],
    vertices: &[Vertex],
) {
    let steps = &mut family.steps;
    // Forward: the object rows a kept seed row reaches.
    let mut live: Vec<Vec<bool>> = Vec::with_capacity(steps.len());
    for step in steps.iter() {
        let inputs = sources(step, seeded, &live);
        let mut out = vec![false; step.rows * columns];
        let mut mark = |links: &[super::super::kernels::RowLink], input: &[bool]| {
            for link in links {
                if input[link.input as usize] {
                    out[link.output as usize] = true;
                }
            }
        };
        match &step.split {
            None => {
                for (edge, input) in step.edges.iter().zip(&inputs) {
                    mark(&edge.links, input);
                }
            }
            Some(split) => {
                for (parts, input) in split.parts.iter().zip(&inputs) {
                    for part in parts {
                        mark(&part.links, input);
                    }
                }
            }
        }
        live.push(out);
    }
    // Backward: keep the links from a live row into a row that is read.
    // Every open object feeds exactly one edge, of a later step, so a step's
    // read rows are complete when the reverse walk reaches it.
    let mut read: Vec<Vec<bool>> = steps
        .iter()
        .map(|step| vec![false; step.rows * columns])
        .collect();
    for s in (0..steps.len()).rev() {
        let wanted = if steps[s].trace.is_some() {
            live[s].clone()
        } else {
            std::mem::take(&mut read[s])
        };
        let step = &mut steps[s];
        let inputs = sources(step, seeded, &live);
        let kept = |link: &super::super::kernels::RowLink, input: &[bool]| {
            input[link.input as usize] && wanted[link.output as usize]
        };
        for (edge, input) in step.edges.iter_mut().zip(&inputs) {
            edge.links.retain(|link| kept(link, input));
        }
        match step.split.as_mut() {
            None => {
                for edge in &step.edges {
                    if let Source::Step(child) = edge.source {
                        for link in &edge.links {
                            read[child][link.input as usize] = true;
                        }
                    }
                }
            }
            Some(split) => {
                let mut members: Vec<Vec<(usize, usize)>> = vec![Vec::new(); split.groups.len()];
                for (e, (parts, input)) in split.parts.iter_mut().zip(&inputs).enumerate() {
                    for (p, part) in parts.iter_mut().enumerate() {
                        part.links.retain(|link| kept(link, input));
                        members[part.group].push((e, p));
                        if let Source::Step(child) = step.edges[e].source {
                            for link in &part.links {
                                read[child][link.input as usize] = true;
                            }
                        }
                    }
                }
                // A group's target rows are those its kept part links write.
                let mut written = vec![false; step.rows * columns];
                for (group, members) in split.groups.iter_mut().zip(&members) {
                    let links = || {
                        members
                            .iter()
                            .flat_map(|&(e, p)| split.parts[e][p].links.iter())
                    };
                    links().for_each(|link| written[link.output as usize] = true);
                    group.targets.retain(|&target| written[target as usize]);
                    links().for_each(|link| written[link.output as usize] = false);
                }
            }
        }
    }
    remove_empty(family, contents, vertices);
}

/// Whether edge `e` of `step` has a link left.
fn edge_runs(step: &Step, e: usize) -> bool {
    match &step.split {
        None => !step.edges[e].links.is_empty(),
        Some(split) => split.parts[e].iter().any(|part| !part.links.is_empty()),
    }
}

/// Drop the edges without a link (with their split parts), the steps left
/// without an edge, the split groups without a target row and the parts
/// without a link; renumber the steps' sources and the parts' groups;
/// recount the modelled work and the family's inputs.
///
/// An edge without a link reads no row of its source: a child step's rows
/// are read by that edge only (every open object feeds one edge), so the
/// child has no link left either and goes with it.
fn remove_empty(family: &mut Family, contents: &[Content], vertices: &[Vertex]) {
    let old = std::mem::take(&mut family.steps);
    let mut index: Vec<Option<usize>> = vec![None; old.len()];
    let mut steps: Vec<Step> = Vec::with_capacity(old.len());
    for (s, mut step) in old.into_iter().enumerate() {
        let runs: Vec<bool> = (0..step.edges.len()).map(|e| edge_runs(&step, e)).collect();
        if !runs.iter().any(|&r| r) {
            continue;
        }
        let mut flags = runs.iter();
        step.edges
            .retain(|_| *flags.next().expect("one flag per edge"));
        for edge in &mut step.edges {
            if let Source::Step(child) = edge.source {
                let child = index[child].expect("the child of a kept edge is kept");
                edge.source = Source::Step(child);
            }
        }
        if let Some(split) = step.split.as_mut() {
            let mut flags = runs.iter();
            split
                .parts
                .retain(|_| *flags.next().expect("one flag per edge"));
            let mut renumbered: Vec<Option<usize>> = vec![None; split.groups.len()];
            let mut groups = Vec::with_capacity(split.groups.len());
            for (g, group) in std::mem::take(&mut split.groups).into_iter().enumerate() {
                if !group.targets.is_empty() {
                    renumbered[g] = Some(groups.len());
                    groups.push(group);
                }
            }
            split.groups = groups;
            for parts in &mut split.parts {
                parts.retain(|part| !part.links.is_empty());
                for part in parts.iter_mut() {
                    part.group = renumbered[part.group].expect("a kept part's group has a target");
                }
            }
        }
        step.costs = costs(&step, contents, vertices);
        index[s] = Some(steps.len());
        steps.push(step);
    }
    let mut factors = BTreeSet::new();
    let mut instances = BTreeSet::new();
    let mut currents = BTreeSet::new();
    for step in &steps {
        match &step.split {
            None => factors.extend(step.edges.iter().map(|edge| edge.factor)),
            Some(split) => {
                instances.extend(split.groups.iter().map(|group| group.instance));
                currents.extend(split.parts.iter().flatten().map(|part| part.current));
            }
        }
    }
    family.steps = steps;
    family.factors = factors.into_iter().collect();
    family.vertices = instances.into_iter().collect();
    family.currents = currents.into_iter().collect();
}

/// The modelled work of a pruned step (as the loader counts it,
/// `fused_cost` and `split_candidate`): the fused form's, and the split
/// form's when the step runs split (a fused step's split alternative is not
/// recounted).
fn costs(step: &Step, contents: &[Content], vertices: &[Vertex]) -> (StepCost, Option<StepCost>) {
    let fused = fused_cost(step.seeded, step.d_alpha, &step.edges, contents);
    let Some(split) = &step.split else {
        return (fused, None);
    };
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
