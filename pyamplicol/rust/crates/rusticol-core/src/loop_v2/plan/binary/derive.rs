// SPDX-License-Identifier: 0BSD

//! The bound of the loader's derivations for a plan in file form (WP3.1
//! package D2; RUST-MIGRATION.md §3.D, the reader's fail-closed contract).
//!
//! [`LoopProgramV4::from_file`] (`plan.rs`, shared with the JSON load)
//! derives tables whose sizes follow plan scalars, not the lengths of the
//! plan's lists: the schedule-B row links (per edge one per source row, the
//! source rows being at most the Born rows, `rows` = the product of the legs'
//! helicity counts), expanded over the observable `columns`, the candidate
//! split parts of every content term, the monomial-product tables of each
//! (rank, degree), the column-row flags (`rows × columns`) and the vertex
//! table (by vertex id). A plan of a few hundred bytes can thus ask the
//! derivations for any amount of memory (40 legs of two helicities: 2^40 Born
//! rows), and the derivations allocate infallibly.
//!
//! [`bound`] computes, from the file form alone and in time linear in it, an
//! upper bound of the bytes those derivations allocate (summed over every
//! allocation, so it bounds their peak too, at the allocator's growth
//! factors) and of the iterations of their loops whose count is a product of
//! scalars and list lengths. The derivations' copies of the plan's own lists
//! (at most about twice the decoded model) are not part of it: the reader
//! reserves them with it. The binary member must pay for it: its reader
//! refuses a member whose bound exceeds [`DERIVED_PER_BYTE`] per member byte
//! (the writer pads the member of a plan that needs more, as it does for the
//! decoded size), and reserves the memory bound fallibly before deriving, so a
//! plan whose derivations the allocator cannot grant is refused with an error
//! instead of aborting the process.
//!
//! Before bounding, [`bound`] refuses the scalars the derivations would trust
//! beyond the plan's own tables. Every one of these checks is implied by the
//! loader's validation, so it refuses only plans the loader refuses, but
//! before their first allocation: a vertex id at or beyond the number of
//! terms that carry one (the loader requires every id up to the largest to be
//! defined by a term); a family reading of more entries than legs, or a
//! family rank above its propagators + 2; a schedule-B step whose output rank
//! `rank_in + degree` exceeds its family's rank (a trace step's is checked,
//! and an open step's output feeds a later step's input rank); a schedule-M
//! state above the largest family rank (likewise, up to its closing state).

use super::super::{ContentFile, PlanFile, Vertex, invalid, require};
use super::ALLOCATION;
use crate::RusticolResult;
use crate::loop_v2::kernels::RowLink;

/// The most bytes the loader's derivations may allocate and iterate per
/// member byte (module docs). The plans written today need about 100-500
/// (six and seven legs, one observable column: 16 times less or more); a
/// plan needing more (tens of columns at seven legs) gets a padded member.
pub const DERIVED_PER_BYTE: u64 = 8192;

/// The largest rank or degree a monomial table of a valid plan can have: a
/// family rank is at most its propagators + 2, and a family has at most one
/// propagator per leg (`legs < 64`).
const RANKS: usize = 66;

/// An upper bound of what the loader's derivations make of a plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeriveBound {
    /// Bytes of every allocation (each plus the allocator's bookkeeping).
    pub memory: u64,
    /// Iterations of the scalar-driven loops.
    pub work: u64,
}

impl DeriveBound {
    /// What the member must pay for: memory and work (saturating).
    pub fn cost(&self) -> u64 {
        self.memory.saturating_add(self.work)
    }

    #[inline]
    fn alloc(&mut self, items: u64, size: usize) {
        if items > 0 {
            let bytes = items.saturating_mul(size as u64);
            self.memory = self
                .memory
                .saturating_add(bytes.saturating_add(ALLOCATION as u64));
        }
    }

    #[inline]
    fn iterate(&mut self, count: u64) {
        self.work = self.work.saturating_add(count);
    }

    /// The monomial-product table of `(rank, degree)` and its monomial lists.
    fn table(&mut self, rank: u64, degree: u64) {
        let (left, right) = (ncoeff(rank), ncoeff(degree));
        let entries = left.saturating_mul(right);
        self.alloc(entries, size_of::<usize>());
        self.alloc(left.saturating_mul(2), size_of::<[u32; 4]>());
        self.alloc(right.saturating_mul(2), size_of::<[u32; 4]>());
        self.iterate(entries);
    }
}

/// The coefficients of a rank-`rank` polynomial in four variables,
/// `C(rank + 4, 4)` (saturating).
fn ncoeff(rank: u64) -> u64 {
    let r = u128::from(rank);
    let n = (r + 1) * (r + 2) / 2 * (r + 3) / 3 * (r + 4) / 4;
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Bytes per entry of a hash map or set the derivations fill (key, value,
/// control byte, load factor and the table kept while it grows).
const MAP_ENTRY: usize = 64;
/// Bytes per entry of a B-tree set of row indices (`u32` keys in nodes at
/// least half full, with their bookkeeping) and of the list of its targets.
const TREE_ENTRY: usize = 32;
/// A helicity row restricted to legs (`links::Row`).
const ROW: usize = 16;

/// The (rank, degree) pairs whose monomial tables the derivations build.
struct Tables {
    /// Fused edge tables `(source rank, content degree)`.
    pairs: [[bool; RANKS]; RANKS],
    /// Per input rank of a candidate split step, the largest degree of its
    /// vertices (a vertex's degree is at most its content's).
    split: [Option<usize>; RANKS],
}

impl Tables {
    fn pair(&mut self, bound: &mut DeriveBound, rank: usize, degree: usize) {
        if rank < RANKS && degree < RANKS {
            self.pairs[rank][degree] = true;
        } else {
            // Beyond any valid plan (forward states only): counted per use.
            bound.table(rank as u64, degree as u64);
        }
    }

    fn split(&mut self, bound: &mut DeriveBound, rank: usize, degree: usize) {
        if rank < RANKS && degree < RANKS {
            let most = &mut self.split[rank];
            *most = Some(most.map_or(degree, |d| d.max(degree)));
        } else {
            bound.table(rank as u64, degree as u64);
        }
    }

    fn finish(&self, bound: &mut DeriveBound) {
        for (rank, degrees) in self.pairs.iter().enumerate() {
            for (degree, &used) in degrees.iter().enumerate() {
                let split = self.split[rank].is_some_and(|most| degree <= most);
                if used || split {
                    bound.table(rank as u64, degree as u64);
                }
            }
        }
    }
}

/// The plan's sizes every edge's bound reads.
struct Sizes<'a> {
    file: &'a PlanFile,
    columns: u64,
    /// Per content the rows of its terms, summed.
    term_rows: Vec<u64>,
}

impl Sizes<'_> {
    fn content(&self, factor: usize) -> Option<(usize, &ContentFile)> {
        let content = self.file.factors.get(factor)?.0;
        Some((content, self.file.contents.get(content)?))
    }

    /// An edge of `links` row links (before the column expansion) into
    /// `content`: its expanded links and the candidate split parts of every
    /// term (`split_candidate`, counted whether or not the step splits).
    fn edge(&self, bound: &mut DeriveBound, content: usize, links: u64) {
        let expanded = links.saturating_mul(self.columns);
        bound.alloc(expanded, size_of::<RowLink>());
        bound.iterate(expanded.saturating_add(links));
        let c = &self.file.contents[content];
        let terms = c.terms.len() as u64;
        let rows = c.row_helicities.len() as u64;
        // Per term: its block-row map, its filtered links (grown by
        // doubling), their targets, and its part and group.
        let mut term = DeriveBound::default();
        term.alloc(rows, size_of::<Option<u32>>());
        term.alloc(expanded.saturating_mul(2), size_of::<RowLink>());
        term.alloc(expanded, TREE_ENTRY);
        term.alloc(2, MAP_ENTRY);
        bound.memory = bound
            .memory
            .saturating_add(term.memory.saturating_mul(terms));
        let visits = rows.saturating_add(expanded).saturating_mul(terms);
        bound.iterate(visits.saturating_add(self.term_rows[content]));
    }
}

/// An empty list with room for `len` items (one per record of a decoded
/// list), or an error.
fn buffer<T>(len: usize) -> RusticolResult<Vec<T>> {
    let mut items = Vec::new();
    items
        .try_reserve_exact(len)
        .map_err(|_| invalid("the derivation bound's lists exceed what the allocator grants"))?;
    Ok(items)
}

/// The product of the helicity counts of the legs in `mask` (saturating;
/// a bit beyond the process's legs is no leg, as in `RowSpace::restrict`).
fn rows_of(counts: &[u64], mask: u64) -> u64 {
    let mut rows: u64 = 1;
    let mut bits = mask;
    while bits != 0 {
        let leg = bits.trailing_zeros() as usize;
        rows = rows.saturating_mul(counts.get(leg).copied().unwrap_or(1));
        bits &= bits - 1;
    }
    rows
}

/// The bound of the loader's derivations for `file`, after refusing the
/// scalars they would trust beyond the plan's tables (module docs).
pub fn bound(file: &PlanFile) -> RusticolResult<DeriveBound> {
    require(
        file.legs < 64 && file.helicities.len() == file.legs,
        "plan helicities per leg differ from its legs",
    )?;
    let mut bound = DeriveBound::default();
    let counts: Vec<u64> = file.helicities.iter().map(|h| h.len() as u64).collect();
    let (rows, columns) = (file.rows as u64, file.columns as u64);

    // Contents: the block-row maps, the term-row checks per binding input,
    // the fused-entry index per vertex term, the vertex table.
    let vertex_terms = file.contents.iter().flat_map(|c| &c.terms);
    let vertex_terms = vertex_terms.filter(|t| t.vertex >= 0).count();
    let mut term_rows = buffer::<u64>(file.contents.len())?;
    for c in &file.contents {
        bound.alloc(c.row_helicities.len() as u64, MAP_ENTRY);
        let mut rows_here: u64 = 0;
        for t in &c.terms {
            let inputs: u64 = t.bindings.iter().map(|b| b.inputs.len() as u64).sum();
            bound.iterate((t.rows.len() as u64).saturating_mul(inputs.saturating_add(1)));
            rows_here = rows_here.saturating_add(t.rows.len() as u64);
            if t.vertex >= 0 {
                require(
                    (t.vertex as u64) < vertex_terms as u64,
                    "a vertex id beyond the terms that carry one",
                )?;
                bound.alloc(c.fused.len() as u64, MAP_ENTRY);
                bound.iterate(c.fused.len() as u64);
            }
        }
        term_rows.push(rows_here);
    }
    bound.alloc(2 * vertex_terms as u64, size_of::<Option<Vertex>>());
    let sizes = Sizes {
        file,
        columns,
        term_rows,
    };
    let mut tables = Tables {
        pairs: [[false; RANKS]; RANKS],
        split: [None; RANKS],
    };

    // Schedule B: per family the Born rows, per step its output rows (at
    // most its source rows and the rows of the legs it leaves open), per
    // edge one link per source row.
    let mut largest_rank = 0;
    for f in &file.families {
        require(
            f.sequence.len() <= file.legs && f.rank <= f.sequence.len() + 2,
            "family rank exceeds propagators + 2",
        )?;
        largest_rank = largest_rank.max(f.rank);
        bound.alloc(rows, ROW);
        bound.iterate(rows);
        let mut outputs = buffer::<(u64, usize)>(f.steps.len())?;
        for (index, s) in f.steps.iter().enumerate() {
            let rank_out = s
                .rank_in
                .checked_add(s.degree)
                .filter(|&rank| rank <= f.rank)
                .ok_or_else(|| invalid("a step above its family's rank"))?;
            let mut targets: u64 = 0;
            for e in &s.edges {
                let (source, source_rank) = match (e.leaf, e.child) {
                    (-1, child) if child >= 0 && (child as usize) < index => {
                        outputs[child as usize]
                    }
                    _ => (rows, 0),
                };
                targets = targets.saturating_add(source);
                bound.iterate(source);
                bound.alloc(source.saturating_mul(2), 3 * size_of::<u32>());
                if let Some((content, c)) = sizes.content(e.factor) {
                    sizes.edge(&mut bound, content, source);
                    if c.degree <= s.degree {
                        tables.pair(&mut bound, source_rank, c.degree);
                        let rank = if s.seeded { 0 } else { s.rank_in };
                        tables.split(&mut bound, rank, c.degree);
                    }
                }
            }
            let out = targets.min(rows_of(&counts, s.out)).min(rows);
            bound.alloc(out.saturating_mul(2), ROW);
            bound.alloc(out, MAP_ENTRY);
            outputs.push((out, rank_out));
        }
    }

    // Schedule F: stored links (one column), tables by the states' ranks.
    if let Some(forward) = &file.forward {
        let mut ranks = buffer::<usize>(forward.states.len())?;
        for (index, s) in forward.states.iter().enumerate() {
            let parent = usize::try_from(s.parent).ok().filter(|&p| p < index);
            let source_rank = parent.map_or(0, |p| ranks[p]);
            bound.alloc((s.links.len() / 3) as u64, size_of::<RowLink>());
            tables.pair(&mut bound, source_rank, s.degree);
            ranks.push(s.rank);
        }
    }

    // Schedule M: stored links expanded over the columns, split parts.
    if let Some(merged) = &file.merged {
        let mut ranks = buffer::<usize>(merged.states.len())?;
        for (index, s) in merged.states.iter().enumerate() {
            let rank_out = s
                .rank_in
                .checked_add(s.degree)
                .filter(|&rank| rank <= largest_rank)
                .ok_or_else(|| invalid("a merged state above every family's rank"))?;
            for e in &s.edges {
                let parent = usize::try_from(e.parent).ok().filter(|&p| p < index);
                let source_rank = parent.map_or(0, |p| ranks[p]);
                if let Some((content, c)) = sizes.content(e.factor) {
                    sizes.edge(&mut bound, content, (e.links.len() / 3) as u64);
                    if c.degree <= s.degree {
                        tables.pair(&mut bound, source_rank, c.degree);
                        let rank = if s.seeded { 0 } else { s.rank_in };
                        tables.split(&mut bound, rank, c.degree);
                    }
                }
            }
            ranks.push(rank_out);
        }
    }
    tables.finish(&mut bound);

    // The column-row flags (`ColumnRows::new`).
    if let Some(column_rows) = &file.column_rows {
        let flags = rows.saturating_mul(columns);
        bound.alloc(flags, 1);
        bound.iterate(flags);
        let kept = column_rows.iter().flatten().map(|kept| kept.len() as u64);
        bound.iterate(kept.sum());
    }
    Ok(bound)
}

/// Reserves `bytes` fallibly and releases them: an error, not an abort,
/// when the allocator would refuse the derivations (an address-space limit,
/// or more than the host can map).
pub fn reserve(bytes: u64) -> RusticolResult<()> {
    let refused = || {
        invalid(format!(
            "the loader's derivations need up to {bytes} bytes, which the allocator refuses"
        ))
    };
    let bytes = usize::try_from(bytes).map_err(|_| refused())?;
    let mut probe: Vec<u8> = Vec::new();
    probe.try_reserve_exact(bytes).map_err(|_| refused())?;
    // The probe must reach the allocator (an unused allocation may be elided).
    std::hint::black_box(probe.as_mut_ptr());
    Ok(())
}
