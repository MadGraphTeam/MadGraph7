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

use super::{
    Content, Family, LoopProgramV4, ReadRows, Source, Step, StepCost, Vertex, fused_cost,
    step_shape,
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
        // The factor and current rows the kept links read (schedules F and
        // M read the plan's chunk inputs too, with links of their own).
        if self.forward.is_none() && self.merged.is_none() {
            self.read_rows = Some(self.rows_read());
        }
    }

    /// The content rows of each fused factor and the term rows of each
    /// split current that the kept links of schedule B read.
    fn rows_read(&self) -> ReadRows {
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
        for step in self.families.iter().flat_map(|family| &family.steps) {
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
                        for link in &part.links {
                            currents[part.current][link.block as usize] = true;
                        }
                    }
                }
            }
        }
        ReadRows { factors, currents }
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

#[cfg(test)]
mod tests {
    use crate::loop_v2::dress::{ChunkInputs, LaneData, Reads, V2Batch};
    use crate::loop_v2::plan::{LoopProgramV4, Source};
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::run_family;
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
}
