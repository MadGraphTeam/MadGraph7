// SPDX-License-Identifier: 0BSD

//! The bound of the loader's derivations for a plan in file form (WP3.1
//! package D2; RUST-MIGRATION.md §3.D, the reader's fail-closed contract).
//!
//! [`LoopProgramV4::from_file`] (`plan.rs`, shared with the JSON load)
//! validates a plan in file form and derives the program from it, allocating
//! infallibly. Some of its tables follow plan scalars, not the lengths of the
//! plan's lists: the schedule-B row links (per edge one per source row, the
//! source rows being at most the Born rows, `rows` = the product of the legs'
//! helicity counts), expanded over the observable `columns`, the candidate
//! split parts of every content term, the monomial-product tables of each
//! (rank, degree), the column-row flags (`rows × columns`), the columns'
//! seed-weight columns (`columns`, written or the identity), the vertex table
//! (by vertex id) and the power of ten of a decimal weight (`"1e1000000000"`
//! is a billion-digit number). A plan of a few hundred bytes can thus ask the
//! derivations for any amount of memory (40 legs of two helicities: 2^40 Born
//! rows). The other tables follow the plan's lists, but at several times
//! their decoded size: per fused entry its two segment layouts and their
//! repeated-entry sets, per split entry a vertex's entry tables and layouts,
//! the parsed weights, the schedules' structs, hash maps and B-trees.
//!
//! [`bound`] computes, from the file form alone and in time linear in it,
//! upper bounds of what the derivations do ([`DeriveBound`]):
//! - the bytes they hold at their peak: every table they make, kept or not,
//!   at the capacity its growth can reach, plus the largest set of
//!   temporaries alive at once and the previous buffer of a list while it
//!   grows;
//! - the bytes of all their allocations, summed;
//! - the iterations of their loops whose count is a product of scalars and
//!   list lengths (and the cost of parsing a multi-precision weight).
//!
//! Sizes are those of the program's own types (`size_of`) under std's growth
//! rules: a list collected from an iterator of unknown length holds at most
//! `max(4, 2n)` items, and its earlier buffers sum to less than its last; a
//! hash table takes the smallest power-of-two bucket count holding its
//! entries at 7/8 load (at most 16/7 buckets per entry), a control byte per
//! bucket and one 16-byte group beyond, and while it grows its previous
//! table (half the buckets) is alive; a B-tree node (11 keys) other than the
//! root holds at least 5; a number of more than 40 digits is multi-precision,
//! at most a byte per digit. The member pays for the allocations and
//! iterations: its reader refuses a member whose bound exceeds
//! [`DERIVED_PER_BYTE`] per member byte (the writer pads the member of a plan
//! that needs more, as it does for the decoded size), and probes the peak,
//! with its [`margin`], fallibly before deriving ([`reserve`]), so a plan
//! whose derivations the allocator cannot grant is refused with an error
//! instead of aborting the process. The probe is a check, not a reservation:
//! it allocates the bytes and releases them at once. It discriminates under
//! an address-space limit (`RLIMIT_AS`); under Linux's default overcommit
//! without one, it succeeds for any size up to about the host's RAM plus
//! swap, and the per-member-byte bounds are the guard. The bound is checked
//! against the derivations' real allocations (a counting allocator over
//! crafted and exported plans, GMP's allocations counted through its memory
//! functions) and, for multi-precision weights, against the address space
//! their derivations take, C `malloc`'s placement included
//! (`tests/loop_v2_plan_derive_bound.rs`).
//!
//! Before bounding, [`bound`] refuses the values the derivations would trust
//! beyond the plan's own tables, before their first allocation. Each is
//! implied by the loader's validation, or the loader would crash or panic on
//! it, so a plan the loader loads is never refused, with one exception (the
//! second fused-entry check):
//! - a vertex id at or beyond the number of terms that carry one (the loader
//!   requires every id up to the largest to be defined by a term);
//! - a family reading of more entries than legs, or a family rank above its
//!   propagators + 2;
//! - a schedule-B step whose output rank `rank_in + degree` exceeds its
//!   family's rank (a trace step's is checked, and an open step's output
//!   feeds a later step's input rank); a schedule-M state above the largest
//!   family rank (likewise, up to its closing state);
//! - a fused entry whose q-degree (its exponents' sum) exceeds its content's
//!   degree: the loader checks the content's kernel degree only after the
//!   segment layouts it computes it in, whose monomial indices NinjaRS's
//!   `tensor_layout::monomial_index` computes recursively, one call per unit
//!   of exponent (an exponent of 2^20 overflows the stack before the check,
//!   and exponents summing to 2^32 wrap the layout's `u32` degree to 0);
//! - a fused entry of q-degree [`RANKS`] or more, whatever its content's
//!   degree: not implied for a content no step reads, whose degree nothing
//!   checks, but no plan has one (a content's degree is at most its step's,
//!   at most its family's rank, at most legs + 2 < 66), and the recursion
//!   above has no other bound;
//! - a decimal weight whose power of ten is 2^32 or more (numerica panics).

use super::super::links::Row;
use super::super::{
    Binding, ChunkLayout, Closure, Content, ContentFile, ContentTerm, Edge, EdgeFile, Entry,
    Family, Kernel, KernelInput, KernelTerm, Leaf, MergedEdgeFile, MergedState, PlanFile, Seed,
    SeedKey, SeedTerm, SplitGroup, SplitKernel, SplitPart, State, Step, Vertex, VertexEntry,
    invalid, require,
};
use super::ALLOCATION;
use crate::RusticolResult;
use crate::loop_v2::kernels::RowLink;
use crate::loop_v2::poly::{MonomialProductTable, SegmentEntry};
use std::collections::{BTreeSet, HashMap};
use symbolica::domains::rational::Rational;

/// The most bytes the loader's derivations may allocate and iterate per
/// member byte (module docs). The plans written today need a few hundred
/// (five to seven legs, one observable column); a plan needing more (tens of
/// columns at seven legs) gets a padded member.
pub const DERIVED_PER_BYTE: u64 = 8192;

/// The largest rank or degree a monomial table of a valid plan can have, and
/// the bound of a fused entry's q-degree: a family rank is at most its
/// propagators + 2, and a family has at most one propagator per leg
/// (`legs < 64`).
pub const RANKS: usize = 66;

/// GMP's allocations, summed, per byte and bit of length of a
/// multi-precision weight's numbers (`Tally::weight`; measured at most 4.9).
const GMP_ALLOCATED: u64 = 16;

/// The fixed part of a probe's margin: room for the allocator's own growth
/// and for the fixed-size allocations around the bounded ones (the thread
/// pool's per-thread state, error messages).
const SLACK: u64 = 16 << 20;

/// What the reader probes for a bound of `bytes`: an eighth more for the
/// allocator's rounding and fragmentation, and [`SLACK`].
pub fn margin(bytes: u64) -> u64 {
    bytes.saturating_add(bytes / 8).saturating_add(SLACK)
}

/// Upper bounds of what the loader's derivations make of a plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeriveBound {
    /// Bytes held at the peak: every table made, plus the largest set of
    /// temporaries alive at once and a growing list's previous buffer (each
    /// allocation plus the allocator's bookkeeping).
    pub memory: u64,
    /// Bytes of every allocation, summed.
    pub allocated: u64,
    /// Iterations of the scalar-driven loops.
    pub work: u64,
}

impl DeriveBound {
    /// What the member must pay for: allocations and iterations (saturating).
    pub fn cost(&self) -> u64 {
        self.allocated.saturating_add(self.work)
    }

    /// The bytes the reader probes before deriving: the peak with its
    /// [`margin`].
    pub fn reserved(&self) -> u64 {
        margin(self.memory)
    }
}

/// The bytes of one allocation of `items` of `size` bytes (none when empty).
fn bytes(items: u64, size: usize) -> u64 {
    if items == 0 {
        0
    } else {
        items
            .saturating_mul(size as u64)
            .saturating_add(ALLOCATION as u64)
    }
}

/// The items a list collected from `items` of unknown count can hold:
/// `max(4, 2 × items)` (std's amortised growth from its first buffer of 4).
fn capacity(items: u64) -> u64 {
    if items == 0 {
        0
    } else {
        items.saturating_mul(2).max(4)
    }
}

/// The bytes of a hash table with room for `items` entries of `entry` bytes
/// (std's SwissTable): the smallest power-of-two bucket count at 7/8 load (4
/// or 8 below eight entries), the entry slots rounded to the 16-byte group,
/// one control byte per bucket and one group beyond.
fn table(items: u64, entry: usize) -> u64 {
    if items == 0 {
        return 0;
    }
    let buckets = match items {
        0..4 => 4,
        4..8 => 8,
        _ => (items.saturating_mul(8) / 7)
            .checked_next_power_of_two()
            .unwrap_or(u64::MAX),
    };
    buckets
        .saturating_mul(entry as u64)
        .saturating_add(15)
        .saturating_add(buckets)
        .saturating_add(16 + ALLOCATION as u64)
}

/// Bytes per key of a B-tree set of keys of at most 8 bytes: a leaf (11 keys,
/// its parent link and lengths, 104 bytes and the allocator's 16) holds at
/// least 5 but at the root, an internal node (216 bytes) at least 6 children,
/// and so on up: at most 24 + 7.2 + 1.2 + … < 33 bytes per key.
const TREE_KEY: u64 = 40;
/// The bytes of the one node a B-tree may hold below half full (its root).
const TREE_ROOT: u64 = 256;

/// The bytes of a B-tree set of `items` keys of at most 8 bytes.
fn tree(items: u64) -> u64 {
    if items == 0 {
        0
    } else {
        items.saturating_mul(TREE_KEY).saturating_add(TREE_ROOT)
    }
}

/// The running bound: bytes made (kept or not), the largest temporary set
/// seen, the largest previous buffer of a growing list, all allocations, the
/// iterations.
#[derive(Default)]
struct Tally {
    kept: u64,
    peak: u64,
    growth: u64,
    allocated: u64,
    work: u64,
}

impl Tally {
    /// An allocation of `bytes` counted to the end (the program keeps it, or
    /// it is dropped later than its neighbours: counted as kept).
    #[inline]
    fn keep_bytes(&mut self, bytes: u64) {
        self.kept = self.kept.saturating_add(bytes);
        self.allocated = self.allocated.saturating_add(bytes);
    }

    /// A list of exactly `items` (an exact-size collect, a clone, a
    /// `with_capacity`).
    #[inline]
    fn keep(&mut self, items: u64, size: usize) {
        self.keep_bytes(bytes(items, size));
    }

    /// A list collected from `items` of unknown count: at its largest
    /// capacity, its earlier buffers allocated before it, its previous
    /// buffer alive while it grows.
    #[inline]
    fn keep_grown(&mut self, items: u64, size: usize) {
        let last = bytes(capacity(items), size);
        self.keep_bytes(last);
        self.allocated = self.allocated.saturating_add(last);
        self.growth = self.growth.max(last / 2);
    }

    /// A hash table of `items` entries reserved up front (`with_capacity`,
    /// a collect of known length).
    #[inline]
    fn map(&mut self, items: u64, entry: usize) {
        self.keep_bytes(table(items, entry));
    }

    /// A hash table grown by insertions to `items` entries: its earlier
    /// tables allocated before it, its previous table alive while it grows.
    #[inline]
    fn map_grown(&mut self, items: u64, entry: usize) {
        let last = table(items, entry);
        self.keep_bytes(last);
        self.allocated = self.allocated.saturating_add(last);
        self.growth = self.growth.max(last / 2);
    }

    /// `lists` lists collected from iterators of unknown count, holding
    /// `items` in all and at most `most` each: each at most `max(4, 2k)`
    /// items, so at most `4 × lists + 2 × items` together.
    fn keep_lists(&mut self, lists: u64, items: u64, most: u64, size: usize) {
        if lists == 0 {
            return;
        }
        let held = lists
            .saturating_mul(4)
            .saturating_add(items.saturating_mul(2));
        let last = held
            .saturating_mul(size as u64)
            .saturating_add(lists.saturating_mul(ALLOCATION as u64));
        self.keep_bytes(last);
        self.allocated = self.allocated.saturating_add(last);
        self.growth = self.growth.max(bytes(capacity(most), size) / 2);
    }

    /// A temporary allocation of `bytes`, alive with the others of `scope`.
    #[inline]
    fn hold_bytes(&mut self, scope: &mut u64, bytes: u64) {
        *scope = scope.saturating_add(bytes);
        self.allocated = self.allocated.saturating_add(bytes);
    }

    #[inline]
    fn hold(&mut self, scope: &mut u64, items: u64, size: usize) {
        self.hold_bytes(scope, bytes(items, size));
    }

    /// A temporary list collected from `items` of unknown count.
    #[inline]
    fn hold_grown(&mut self, scope: &mut u64, items: u64, size: usize) {
        let last = bytes(capacity(items), size);
        self.hold_bytes(scope, last.saturating_add(last / 2));
        self.allocated = self.allocated.saturating_add(last / 2);
    }

    /// A temporary hash table grown by insertions to `items` entries (with
    /// its previous table, alive while it grows).
    #[inline]
    fn hold_map_grown(&mut self, scope: &mut u64, items: u64, entry: usize) {
        let last = table(items, entry);
        self.hold_bytes(scope, last.saturating_add(last / 2));
        self.allocated = self.allocated.saturating_add(last / 2);
    }

    /// The temporaries alive at once at some point: `held` bytes.
    #[inline]
    fn alive(&mut self, held: u64) {
        self.peak = self.peak.max(held);
    }

    #[inline]
    fn iterate(&mut self, count: u64) {
        self.work = self.work.saturating_add(count);
    }

    /// The monomial-product table of `(rank, degree)` (kept) and its
    /// monomial lists (temporary).
    fn products(&mut self, rank: u64, degree: u64) {
        let (left, right) = (ncoeff(rank), ncoeff(degree));
        let entries = left.saturating_mul(right);
        self.keep(entries, size_of::<usize>());
        let mut lists = 0;
        self.hold_grown(&mut lists, left, size_of::<[u32; 4]>());
        self.hold_grown(&mut lists, right, size_of::<[u32; 4]>());
        self.alive(lists);
        self.iterate(entries);
    }

    /// The exact weight `text` parsed (`dress::parse_rational`): a fraction
    /// `p/q`, else a decimal (`source_scalar::decimal_rational`: its digits
    /// as one integer, times or over `10^|e - places|`, which takes
    /// `|e - places| × log2(10) / 8` bytes; refused from 2^32, where
    /// numerica panics). A text of more than 40 characters or a power of
    /// more than 16 bytes is multi-precision (numerica's GMP numbers, which
    /// allocate through C `malloc`), at most a byte per character. The
    /// parsed value is kept; the parse's buffers, base conversion, the
    /// power's squarings, the product and a fraction's normalising gcd and
    /// divisions are temporaries, at most four times the numbers at once,
    /// and GMP's allocations sum to at most [`GMP_ALLOCATED`] × `size` ×
    /// log2(`size`) bytes for numbers of `size` bytes (its subquadratic
    /// algorithms allocate per recursion level). Both are measured with
    /// GMP's allocations counted (`tests/loop_v2_plan_derive_bound.rs`): the
    /// peak at most 0.74 of its charge and the sum at most 4.9 × `size` ×
    /// log2(`size`), over powers of ten of 10^5 to 10^8 digits of either
    /// sign, mantissas of up to 10^7 digits and fractions of up to 3 × 10^7
    /// digits; and the address space the derivations take (`VmPeak`, C
    /// `malloc`'s placement and fragmentation included) at most 0.75 of the
    /// bound's `memory` (without the probe's margin), over powers of ten,
    /// mantissas times and over them and fractions of 10^6 to 10^8 digits.
    fn weight(&mut self, text: &str) -> RusticolResult<()> {
        let text = text.trim();
        let len = text.len() as u64;
        // Numbers of more than 40 characters are multi-precision; all of the
        // text's numbers together take at most a byte per character.
        let long = if len > 40 { bytes(len, 1) } else { 0 };
        let mut size = if len > 40 { len } else { 0 };
        let mut kept = long;
        let mut scratch = long.saturating_mul(4);
        if !text.contains('/') {
            // The decimal's digits copied into one string first.
            scratch = scratch.saturating_add(bytes(len, 1));
            if let Some(power) = decimal_power(text) {
                require(
                    power <= u64::from(u32::MAX),
                    "a decimal weight's power of ten beyond 2^32",
                )?;
                // log2(10) / 8 < 107 / 256.
                let digits = power.saturating_mul(107) / 256;
                if digits > 16 {
                    size = size.saturating_add(digits);
                }
                let digits = bytes(digits, 1);
                kept = kept.saturating_add(digits.saturating_mul(2));
                scratch = scratch.saturating_add(digits.saturating_mul(4));
                self.iterate(digits.saturating_mul(512));
            }
        }
        self.keep_bytes(kept);
        self.hold_bytes(&mut 0, scratch);
        self.alive(scratch);
        if size > 0 {
            let log = u64::from(64 - size.leading_zeros());
            self.allocated = self
                .allocated
                .saturating_add(size.saturating_mul(log).saturating_mul(GMP_ALLOCATED));
        }
        if len > 40 {
            self.iterate(len.saturating_mul(u64::from(64 - len.leading_zeros())));
        }
        Ok(())
    }

    fn finish(self) -> DeriveBound {
        DeriveBound {
            memory: self
                .kept
                .saturating_add(self.peak)
                .saturating_add(self.growth),
            allocated: self.allocated,
            work: self.work,
        }
    }
}

/// The power of ten `|exponent - decimal places|` of a decimal literal in
/// `source_scalar::decimal_rational`'s grammar (sign, digits, an optional
/// fraction, an optional `e`/`E` exponent), or `None` where that grammar
/// refuses the text before raising ten to it.
fn decimal_power(text: &str) -> Option<u64> {
    let unsigned = text
        .strip_prefix('-')
        .or_else(|| text.strip_prefix('+'))
        .unwrap_or(text);
    let mut parts = unsigned.split(['e', 'E']);
    let mantissa = parts.next().unwrap_or_default();
    let exponent = match parts.next() {
        Some(exponent) => exponent.parse::<i64>().ok()?,
        None => 0,
    };
    if parts.next().is_some() {
        return None;
    }
    let mut decimals = mantissa.split('.');
    let whole = decimals.next().unwrap_or_default();
    let fractional = decimals.next().unwrap_or_default();
    let mut digits = whole.bytes().chain(fractional.bytes());
    if decimals.next().is_some()
        || (whole.is_empty() && fractional.is_empty())
        || !digits.all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let places = i64::try_from(fractional.len()).ok()?;
    Some(exponent.checked_sub(places)?.unsigned_abs())
}

/// The coefficients of a rank-`rank` polynomial in four variables,
/// `C(rank + 4, 4)` (saturating).
fn ncoeff(rank: u64) -> u64 {
    let r = u128::from(rank);
    let n = (r + 1) * (r + 2) / 2 * (r + 3) / 3 * (r + 4) / 4;
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// A helicity row restricted to legs (`links::Row`).
const ROW: usize = size_of::<Row>();

/// The (rank, degree) pairs whose monomial tables the derivations build.
struct Tables {
    /// Fused edge tables `(source rank, content degree)`.
    pairs: [[bool; RANKS]; RANKS],
    /// Per input rank of a candidate split step, the largest degree of its
    /// vertices (a vertex's degree is at most its content's).
    split: [Option<usize>; RANKS],
    /// Tables beyond [`RANKS`], counted per use.
    beyond: u64,
}

impl Tables {
    fn pair(&mut self, tally: &mut Tally, rank: usize, degree: usize) {
        if rank < RANKS && degree < RANKS {
            self.pairs[rank][degree] = true;
        } else {
            // Beyond any valid plan (forward states only): counted per use.
            tally.products(rank as u64, degree as u64);
            self.beyond = self.beyond.saturating_add(1);
        }
    }

    fn split(&mut self, tally: &mut Tally, rank: usize, degree: usize) {
        if rank < RANKS && degree < RANKS {
            let most = &mut self.split[rank];
            *most = Some(most.map_or(degree, |d| d.max(degree)));
        } else {
            tally.products(rank as u64, degree as u64);
            self.beyond = self.beyond.saturating_add(1);
        }
    }

    /// The tables, the list holding them and their index.
    fn finish(&self, tally: &mut Tally) {
        let mut count = self.beyond;
        for (rank, degrees) in self.pairs.iter().enumerate() {
            for (degree, &used) in degrees.iter().enumerate() {
                let split = self.split[rank].is_some_and(|most| degree <= most);
                if used || split {
                    tally.products(rank as u64, degree as u64);
                    count += 1;
                }
            }
        }
        tally.keep_grown(count, size_of::<MonomialProductTable>());
        tally.map_grown(count, size_of::<((usize, usize), usize)>());
    }
}

/// The plan's sizes every edge's bound reads, and the split parts counted.
struct Sizes<'a> {
    file: &'a PlanFile,
    columns: u64,
    /// Per content the rows of its terms, summed.
    term_rows: Vec<u64>,
    /// Per content whether every term has a vertex (`split_candidate`
    /// builds a step's split form only if every edge's content has).
    vertexed: Vec<bool>,
    /// Per content the most terms naming one of its rows.
    multiplicity: Vec<u64>,
    /// The candidate split parts of every step (one per edge and term of
    /// its content).
    parts: u64,
}

impl Sizes<'_> {
    fn content(&self, factor: usize) -> Option<(usize, &ContentFile)> {
        let content = self.file.factors.get(factor)?.0;
        Some((content, self.file.contents.get(content)?))
    }

    /// Whether `split_candidate` builds the split form of a step at input
    /// rank `rank` (0 when seeded) over edges of these factors and source
    /// ranks: every source at that rank, every term of every edge's content
    /// with a vertex.
    fn splits(&self, rank: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> bool {
        edges.into_iter().all(|(factor, source_rank)| {
            source_rank == rank
                && self
                    .content(factor)
                    .is_some_and(|(content, _)| self.vertexed[content])
        })
    }

    /// An edge of `links` row links (before the column expansion): its
    /// expanded links (kept). Returns them.
    fn links(&self, tally: &mut Tally, links: u64) -> u64 {
        let expanded = links.saturating_mul(self.columns);
        tally.keep(expanded, size_of::<RowLink>());
        tally.iterate(expanded.saturating_add(links));
        expanded
    }

    /// The candidate split part of every term of an edge of `expanded`
    /// links into `content` (`split_candidate`): its filtered links
    /// (collected) and the list of their targets (kept if the step splits),
    /// the part, its block-row map and the targets' B-tree (temporaries of
    /// the step, `step`). A link's block is a content row, so the terms'
    /// filtered links hold at most `expanded` times the most terms naming
    /// one row. Returns the parts.
    fn parts(&self, tally: &mut Tally, step: &mut u64, content: usize, expanded: u64) -> u64 {
        let c = &self.file.contents[content];
        let rows = c.row_helicities.len() as u64;
        let terms = c.terms.len() as u64;
        let filtered = expanded.saturating_mul(self.multiplicity[content]);
        tally.keep_lists(terms, filtered, expanded, size_of::<RowLink>());
        tally.keep_lists(terms, filtered, expanded, size_of::<u32>());
        // One block-row map at a time.
        let map = bytes(rows, size_of::<Option<u32>>());
        tally.hold_bytes(step, map);
        tally.allocated = tally.allocated.saturating_add(map.saturating_mul(terms));
        let trees = filtered.saturating_mul(TREE_KEY);
        tally.hold_bytes(step, trees.saturating_add(terms.saturating_mul(TREE_ROOT)));
        tally.keep(terms, size_of::<SplitPart>());
        let visits = rows.saturating_add(expanded).saturating_mul(terms);
        tally.iterate(visits.saturating_add(self.term_rows[content]));
        terms
    }

    /// A step's candidate split form over `parts` parts (at most one group
    /// per part): its groups (kept), the parts' lists per edge, the groups'
    /// index and target sets (temporaries of the step).
    fn split(&mut self, tally: &mut Tally, step: &mut u64, edges: u64, parts: u64) {
        tally.keep_grown(parts, size_of::<SplitGroup>());
        tally.keep(edges, size_of::<Vec<SplitPart>>());
        tally.hold_grown(step, parts, size_of::<BTreeSet<u32>>());
        tally.hold_map_grown(step, parts, size_of::<((usize, u64), usize)>());
        self.parts = self.parts.saturating_add(parts);
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

/// One content (`from_file`'s contents loop): its fused entries refused
/// beyond its degree or [`RANKS`] (module docs), its two segment layouts
/// (entries and monomial indices kept, one repeated-entry set at a time),
/// its terms' copies, and per vertex term the vertex `term_vertex` builds:
/// its entries, their index and the content's fused-entry index
/// (temporaries), its fused entries, bindings and two layouts.
fn content(
    tally: &mut Tally,
    c: &ContentFile,
    vertex_terms: usize,
    named: &mut Vec<u64>,
) -> RusticolResult<(u64, u64)> {
    for e in &c.fused {
        let degree: u64 = e[2..].iter().map(|&x| u64::from(x)).sum();
        require(
            degree <= c.degree as u64 && degree < RANKS as u64,
            "a fused entry's q-degree above its content's",
        )?;
    }
    let fused = c.fused.len() as u64;
    let layout = |tally: &mut Tally, entries: u64| {
        tally.keep(entries, size_of::<SegmentEntry>());
        tally.keep(entries, size_of::<usize>());
        table(entries, size_of::<SegmentEntry>())
    };
    // The open and the seeded layout, one repeated-entry set at a time.
    let set = layout(tally, fused);
    layout(tally, fused);
    tally.hold_bytes(&mut 0, set.saturating_mul(2));
    tally.alive(set);
    tally.keep(c.terms.len() as u64, size_of::<ContentTerm>());
    let mut rows_here: u64 = 0;
    for t in &c.terms {
        let inputs: u64 = t.bindings.iter().map(|b| b.inputs.len() as u64).sum();
        tally.iterate((t.rows.len() as u64).saturating_mul(inputs.saturating_add(1)));
        rows_here = rows_here.saturating_add(t.rows.len() as u64);
        tally.keep_grown(t.bindings.len() as u64, size_of::<Binding>());
        for b in &t.bindings {
            tally.keep_grown(b.inputs.len() as u64, size_of::<KernelInput>());
            tally.keep(b.entries.len() as u64, size_of::<usize>());
        }
        tally.keep_grown(t.split.len() as u64, size_of::<SplitKernel>());
        for s in &t.split {
            tally.keep_grown(s.inputs.len() as u64, size_of::<KernelInput>());
            tally.keep(s.entries.len() as u64, size_of::<(usize, usize)>());
        }
        tally.keep(t.slots.len() as u64, size_of::<u64>());
        tally.keep(t.dimensions.len() as u64, size_of::<usize>());
        tally.keep(t.rows.len() as u64, size_of::<(usize, Vec<usize>)>());
        for (_, blocks) in &t.rows {
            tally.keep(blocks.len() as u64, size_of::<usize>());
        }
        if t.vertex >= 0 {
            require(
                (t.vertex as u64) < vertex_terms as u64,
                "a vertex id beyond the terms that carry one",
            )?;
            let entries: u64 = t.split.iter().map(|s| s.entries.len() as u64).sum();
            tally.keep_grown(entries, size_of::<VertexEntry>());
            let mut index = 0;
            let entry = size_of::<(VertexEntry, usize)>();
            tally.hold_bytes(&mut index, table(entries, entry));
            let entry = size_of::<((usize, usize, [u32; 4]), usize)>();
            tally.hold_bytes(&mut index, table(fused, entry));
            tally.keep_grown(entries, size_of::<usize>());
            tally.keep(t.split.len() as u64, size_of::<Binding>());
            for s in &t.split {
                tally.keep(s.inputs.len() as u64, size_of::<KernelInput>());
                tally.keep(s.entries.len() as u64, size_of::<usize>());
            }
            let set = layout(tally, entries);
            layout(tally, entries);
            tally.hold_bytes(&mut index, set);
            tally.allocated = tally.allocated.saturating_add(set);
            tally.keep(t.slots.len() as u64, size_of::<u64>());
            tally.alive(index);
            tally.iterate(fused);
        }
    }
    // The most terms naming one row (a row beyond the content's is refused
    // by the loader).
    named.clear();
    named.resize(c.row_helicities.len(), 0);
    for t in &c.terms {
        for (row, _) in &t.rows {
            if let Some(count) = named.get_mut(*row) {
                *count += 1;
            }
        }
    }
    Ok((rows_here, named.iter().copied().max().unwrap_or(0)))
}

/// The bound of the loader's derivations for `file`, after refusing the
/// values they would trust beyond the plan's tables (module docs).
pub fn bound(file: &PlanFile) -> RusticolResult<DeriveBound> {
    // Every field of the plan named (no `..`): a field added to the loader's
    // file form does not compile until the bound accounts for what the
    // derivations make of it (the fields bounded through their records'
    // lists below, or by nothing: plain scalars the loader copies).
    let PlanFile {
        schema: _,
        legs: _,
        rows: _,
        helicities: _,
        columns: _,
        average: _,
        born_components: _,
        constants: _,
        masses: _,
        block_rows: _,
        kernels: _,
        contents: _,
        factors: _,
        seed_classes: _,
        seeds: _,
        families: _,
        forward: _,
        merged: _,
        column_rows: _,
        column_weights: _,
    } = file;
    require(
        file.legs < 64 && file.helicities.len() == file.legs,
        "plan helicities per leg differ from its legs",
    )?;
    let mut tally = Tally::default();
    let counts: Vec<u64> = file.helicities.iter().map(|h| h.len() as u64).collect();
    let (rows, columns) = (file.rows as u64, file.columns as u64);

    // The kernels, copied with their terms.
    tally.keep_grown(file.kernels.len() as u64, size_of::<Kernel>());
    for k in &file.kernels {
        tally.keep_grown(k.outputs.len() as u64, size_of::<Vec<KernelTerm>>());
        for terms in &k.outputs {
            tally.keep_grown(terms.len() as u64, size_of::<KernelTerm>());
            for (_, monomial) in terms {
                tally.keep(monomial.len() as u64, size_of::<(usize, u32)>());
            }
        }
    }

    // Contents (`content`), then the vertex table: resized by id (at most
    // twice the vertex terms) and collected.
    let vertex_terms = file.contents.iter().flat_map(|c| &c.terms);
    let vertex_terms = vertex_terms.filter(|t| t.vertex >= 0).count();
    let mut term_rows = buffer::<u64>(file.contents.len())?;
    let mut multiplicity = buffer::<u64>(file.contents.len())?;
    let most_rows = file.contents.iter().map(|c| c.row_helicities.len());
    let mut named = buffer::<u64>(most_rows.max().unwrap_or(0))?;
    tally.keep(file.contents.len() as u64, size_of::<Content>());
    for c in &file.contents {
        let (rows_here, most) = content(&mut tally, c, vertex_terms, &mut named)?;
        term_rows.push(rows_here);
        multiplicity.push(most);
    }
    tally.keep_grown(vertex_terms as u64, size_of::<Option<Vertex>>());
    tally.keep_grown(vertex_terms as u64, size_of::<Vertex>());

    // Seed classes and seed keys with their parsed weights.
    tally.keep_grown(file.seed_classes.len() as u64, size_of::<Seed>());
    for s in &file.seed_classes {
        tally.keep_grown(s.columns.len() as u64, size_of::<Vec<SeedTerm>>());
        for terms in &s.columns {
            tally.keep_grown(terms.len() as u64, size_of::<SeedTerm>());
            for (_, re, im) in terms {
                tally.weight(re)?;
                tally.weight(im)?;
            }
        }
    }
    tally.keep_grown(file.seeds.len() as u64, size_of::<SeedKey>());
    for key in &file.seeds {
        let size = size_of::<(usize, [Rational; 2])>();
        tally.keep_grown(key.classes.len() as u64, size);
        for (_, re, im) in &key.classes {
            tally.weight(re)?;
            tally.weight(im)?;
        }
    }

    // The Born row space (`links::RowSpace`) and every content's row map.
    let legs = file.legs as u64;
    tally.keep(legs.saturating_mul(2), size_of::<u64>());
    tally.keep(legs, size_of::<HashMap<i32, u64>>());
    for h in &file.helicities {
        tally.map(h.len() as u64, size_of::<(i32, u64)>());
    }
    let contents = file.contents.len() as u64;
    tally.keep_grown(contents, size_of::<HashMap<Row, u32>>());
    for c in &file.contents {
        tally.map(c.row_helicities.len() as u64, size_of::<(Row, u32)>());
    }
    let vertexed = file
        .contents
        .iter()
        .map(|c| c.terms.iter().all(|t| t.vertex >= 0));
    let mut sizes = Sizes {
        file,
        columns,
        term_rows,
        vertexed: vertexed.collect(),
        multiplicity,
        parts: 0,
    };
    let mut tables = Tables {
        pairs: [[false; RANKS]; RANKS],
        split: [None; RANKS],
        beyond: 0,
    };

    // Schedule B. Per family (`links::derive`, temporary for the family):
    // the Born rows, per step its output rows (at most its source rows and
    // the rows of the legs it leaves open) and their map, per edge one link
    // per source row (collected); then per step its edges, its split
    // candidate, and the family's factor, vertex and current sets.
    tally.keep(file.families.len() as u64, size_of::<Family>());
    let mut largest_rank = 0;
    for f in &file.families {
        require(
            f.sequence.len() <= file.legs && f.rank <= f.sequence.len() + 2,
            "family rank exceeds propagators + 2",
        )?;
        largest_rank = largest_rank.max(f.rank);
        tally.keep_grown(f.leaves.len() as u64, size_of::<Leaf>());
        for leaf in &f.leaves {
            tally.weight(&leaf.weight)?;
        }
        tally.keep(f.sequence.len() as u64, size_of::<(u64, usize)>());
        tally.keep(f.orders.len() as u64, size_of::<usize>());
        let steps = f.steps.len() as u64;
        tally.keep(steps, size_of::<Step>());
        let mut family = 0;
        tally.hold(&mut family, rows, ROW);
        tally.hold(&mut family, steps, size_of::<Option<Vec<Row>>>());
        tally.hold(&mut family, steps, size_of::<usize>());
        tally.hold(&mut family, steps, size_of::<Vec<Vec<u32>>>());
        tally.hold(&mut family, steps, size_of::<bool>());
        tally.iterate(rows);
        let (mut most_map, mut most_step) = (0, 0);
        let (mut edges_here, mut parts_here) = (0u64, 0u64);
        // The live-row pruning's sizes (`live.rs`): object rows in all and
        // at most per step, parts and edges at most per step.
        let (mut object_rows, mut most_objects, mut most_parts, mut most_edges) =
            (0u64, 0u64, 0u64, 0u64);
        let mut outputs = buffer::<(u64, usize)>(f.steps.len())?;
        for (index, s) in f.steps.iter().enumerate() {
            // The loader's own message for a trace step (`from_file`).
            let rank_out = s
                .rank_in
                .checked_add(s.degree)
                .filter(|&rank| rank <= f.rank)
                .ok_or_else(|| {
                    invalid(if s.trace >= 0 {
                        "a trace step above the family rank"
                    } else {
                        "a step above the family rank"
                    })
                })?;
            let (mut targets, mut step): (u64, u64) = (0, 0);
            let edges = s.edges.len() as u64;
            tally.keep(edges, size_of::<Edge>());
            tally.hold(&mut family, edges, size_of::<Vec<u32>>());
            let source = |e: &EdgeFile| match (e.leaf, e.child) {
                (-1, child) if child >= 0 && (child as usize) < index => outputs[child as usize],
                _ => (rows, 0),
            };
            let rank = if s.seeded { 0 } else { s.rank_in };
            let splits = sizes.splits(rank, s.edges.iter().map(|e| (e.factor, source(e).1)));
            let mut parts = 0u64;
            for e in &s.edges {
                let (source, source_rank) = source(e);
                targets = targets.saturating_add(source);
                tally.iterate(source);
                tally.hold_grown(&mut family, source.saturating_mul(3), size_of::<u32>());
                if let Some((content, c)) = sizes.content(e.factor) {
                    let expanded = sizes.links(&mut tally, source);
                    if splits {
                        let terms = sizes.parts(&mut tally, &mut step, content, expanded);
                        parts = parts.saturating_add(terms);
                    }
                    if c.degree <= s.degree {
                        tables.pair(&mut tally, source_rank, c.degree);
                        if splits {
                            tables.split(&mut tally, rank, c.degree);
                        }
                    }
                }
            }
            if splits {
                sizes.split(&mut tally, &mut step, edges, parts);
            }
            edges_here = edges_here.saturating_add(edges);
            parts_here = parts_here.saturating_add(parts);
            let out = targets.min(rows_of(&counts, s.out)).min(rows);
            tally.hold_grown(&mut family, out, ROW);
            let objects = out.saturating_mul(columns);
            object_rows = object_rows.saturating_add(objects);
            most_objects = most_objects.max(objects);
            most_parts = most_parts.max(parts);
            most_edges = most_edges.max(edges);
            let mut map = 0;
            tally.hold_map_grown(&mut map, out, size_of::<(Row, u32)>());
            (most_map, most_step) = (most_map.max(map), most_step.max(step));
            outputs.push((out, rank_out));
        }
        tally.hold_bytes(&mut family, tree(edges_here));
        tally.hold_bytes(&mut family, tree(parts_here).saturating_mul(2));
        tally.keep_grown(edges_here, size_of::<usize>());
        tally.keep_grown(parts_here, size_of::<usize>());
        tally.keep_grown(parts_here, size_of::<usize>());
        tally.alive(family.saturating_add(most_map.max(most_step)));
        if file.column_rows.is_some() {
            // The live-row pruning of the family (`live.rs`, after every
            // family is derived, with the plan's flag per seeded object
            // row): two flags per object row of its steps (live and read)
            // and their lists, per step its edges' sources and flags, a
            // split step's written flags (one per object row), its groups'
            // members (one pair per part) and renumbering, and the
            // renumbered steps with their index.
            let mut pruning = 0;
            tally.hold(&mut pruning, rows.saturating_mul(columns), 1);
            tally.hold(&mut pruning, object_rows.saturating_mul(2), 1);
            tally.hold(
                &mut pruning,
                steps.saturating_mul(2),
                size_of::<Vec<bool>>(),
            );
            tally.hold(&mut pruning, most_edges, size_of::<&[bool]>() + 1);
            tally.hold(&mut pruning, most_objects, 1);
            tally.hold(&mut pruning, most_parts, size_of::<Vec<(usize, usize)>>());
            tally.hold(&mut pruning, most_parts, size_of::<(usize, usize)>());
            tally.hold(&mut pruning, most_parts, size_of::<Option<usize>>());
            tally.hold(&mut pruning, steps, size_of::<Step>());
            tally.hold(&mut pruning, steps, size_of::<Option<usize>>());
            tally.iterate(object_rows.saturating_mul(3));
            tally.alive(pruning);
        }
    }
    if file.column_rows.is_some() && file.forward.is_none() && file.merged.is_none() {
        // The rows the kept links read (`plan::ReadRows`, kept): a flag per
        // content row of each fused factor and per term row of each split
        // current (one current per content term at most), then a pass over
        // the kept links.
        let factor_rows = file.factors.iter().map(|&(content, _)| {
            file.contents
                .get(content)
                .map_or(0, |c| c.row_helicities.len() as u64)
        });
        tally.keep(factor_rows.sum(), 1);
        tally.keep(file.factors.len() as u64, size_of::<Vec<bool>>());
        tally.keep(sizes.term_rows.iter().sum(), 1);
        let terms = file.contents.iter().map(|c| c.terms.len() as u64);
        tally.keep(terms.sum(), size_of::<Vec<bool>>());
        tally.iterate(sizes.parts);
    }

    // Schedule F: stored links (one column), tables by the states' ranks,
    // closures with their weights and rows.
    if let Some(forward) = &file.forward {
        tally.keep(forward.states.len() as u64, size_of::<State>());
        let mut ranks = buffer::<usize>(forward.states.len())?;
        for (index, s) in forward.states.iter().enumerate() {
            let parent = usize::try_from(s.parent).ok().filter(|&p| p < index);
            let source_rank = parent.map_or(0, |p| ranks[p]);
            tally.keep((s.links.len() / 3) as u64, size_of::<RowLink>());
            tables.pair(&mut tally, source_rank, s.degree);
            tally.keep_grown(s.closures.len() as u64, size_of::<Closure>());
            for closure in &s.closures {
                tally.keep(closure.rows.len() as u64, size_of::<Option<usize>>());
                tally.weight(&closure.weight)?;
            }
            ranks.push(s.rank);
        }
    }

    // Schedule M: entries and edge weights (parsed, an edge's also cloned
    // into the weights' index), the per-family closure variants (each
    // closure scans its family's), stored links expanded over the columns,
    // split candidates, the chunks' layouts.
    if let Some(merged) = &file.merged {
        let families = file.families.len() as u64;
        let states = merged.states.len() as u64;
        tally.keep_grown(merged.entries.len() as u64, size_of::<Entry>());
        for entry in &merged.entries {
            tally.weight(&entry.weight[0])?;
            tally.weight(&entry.weight[1])?;
        }
        tally.keep(families, size_of::<Vec<Option<(i8, u64)>>>());
        tally.keep(families, size_of::<Option<usize>>());
        tally.keep(states, size_of::<usize>());
        tally.keep(states, size_of::<MergedState>());
        tally.keep(merged.families.len() as u64, size_of::<usize>());
        tally.keep(merged.chunks.len() as u64, size_of::<usize>());
        let mut closures = buffer::<u64>(file.families.len())?;
        closures.resize(file.families.len(), 0);
        let mut weighted = 0u64;
        let mut ranks = buffer::<usize>(merged.states.len())?;
        for (index, s) in merged.states.iter().enumerate() {
            // A closing state is above its own family's rank too: the
            // loader's refusal of that closure ("merged closure").
            let rank_out = s
                .rank_in
                .checked_add(s.degree)
                .filter(|&rank| rank <= largest_rank)
                .ok_or_else(|| {
                    invalid(if s.closure.is_some() {
                        "a merged closure above the family rank"
                    } else {
                        "a merged state above every family's rank"
                    })
                })?;
            let edges = s.edges.len() as u64;
            tally.keep(edges, size_of::<Edge>());
            tally.keep(edges, size_of::<Option<usize>>());
            if let Some(closure) = &s.closure
                && let Some(count) = closures.get_mut(closure.family)
            {
                tally.iterate(*count);
                *count += 1;
            }
            let (mut step, mut parts) = (0, 0u64);
            let source_rank = |e: &MergedEdgeFile| {
                let parent = usize::try_from(e.parent).ok().filter(|&p| p < index);
                parent.map_or(0, |p| ranks[p])
            };
            let rank = if s.seeded { 0 } else { s.rank_in };
            let splits = sizes.splits(rank, s.edges.iter().map(|e| (e.factor, source_rank(e))));
            for e in &s.edges {
                if let Some(weight) = &e.weight {
                    for _ in 0..2 {
                        tally.weight(&weight[0])?;
                        tally.weight(&weight[1])?;
                    }
                    weighted += 1;
                }
                if let Some((content, c)) = sizes.content(e.factor) {
                    let expanded = sizes.links(&mut tally, (e.links.len() / 3) as u64);
                    if splits {
                        let terms = sizes.parts(&mut tally, &mut step, content, expanded);
                        parts = parts.saturating_add(terms);
                    }
                    if c.degree <= s.degree {
                        tables.pair(&mut tally, source_rank(e), c.degree);
                        if splits {
                            tables.split(&mut tally, rank, c.degree);
                        }
                    }
                }
            }
            if splits {
                sizes.split(&mut tally, &mut step, edges, parts);
            }
            tally.alive(step);
            ranks.push(rank_out);
        }
        for &count in &closures {
            tally.keep_grown(count, size_of::<Option<(i8, u64)>>());
        }
        tally.keep_grown(weighted, size_of::<[Rational; 2]>());
        tally.map_grown(weighted, size_of::<([Rational; 2], usize)>());
        // Per chunk (`ChunkLayout::of`): its union-find and group index
        // (temporaries), its head and groups (each group's list collected).
        tally.keep(merged.chunks.len() as u64, size_of::<ChunkLayout>());
        let mut start = 0u64;
        for &end in &merged.chunks {
            let range = (end as u64).min(states).saturating_sub(start);
            start = start.max((end as u64).min(states));
            let mut chunk = 0;
            tally.hold(&mut chunk, range, size_of::<usize>());
            tally.hold_map_grown(&mut chunk, range, size_of::<(usize, usize)>());
            tally.alive(chunk);
            tally.keep_grown(range, size_of::<usize>());
            tally.keep_grown(range, size_of::<Vec<usize>>());
            // Groups of g states: capacity at most max(4, 2g) each.
            tally.keep(range.saturating_mul(6), size_of::<usize>());
            tally.keep_bytes(range.saturating_mul(ALLOCATION as u64));
        }
    }
    tables.finish(&mut tally);

    // The split inputs (vertex instances and currents: at most one per
    // part) and their ordered lists.
    let parts = sizes.parts;
    tally.map_grown(parts, size_of::<((usize, u64), usize)>());
    tally.map_grown(parts, size_of::<((usize, usize), usize)>());
    tally.keep(parts, size_of::<(usize, (usize, u64))>());
    tally.keep(parts, size_of::<(usize, u64)>());
    tally.keep(parts, size_of::<(usize, (usize, usize))>());
    tally.keep(parts, size_of::<(usize, usize)>());

    // The column-row flags (`ColumnRows::new`).
    if let Some(column_rows) = &file.column_rows {
        let flags = rows.saturating_mul(columns);
        tally.keep(flags, 1);
        tally.iterate(flags);
        let kept = column_rows.iter().flatten().map(|kept| kept.len() as u64);
        tally.iterate(kept.sum());
    }
    // The seed-weight column of each observable column (the file's
    // `column_weights` copied, or the identity), and the scan for the
    // largest.
    tally.keep(columns, size_of::<usize>());
    tally.iterate(columns);
    Ok(tally.finish())
}

/// Probes `bytes` with their [`margin`] fallibly and releases them: an
/// error, not an abort, when the allocator would refuse them (an
/// address-space limit, or more than the host can map); `need` names what
/// needs them (the error reads "`need` up to N bytes"). A check, not a
/// reservation (module docs).
pub fn reserve(need: &str, bytes: u64) -> RusticolResult<()> {
    let probe = margin(bytes);
    let refused = || {
        invalid(format!(
            "{need} up to {probe} bytes, which the allocator refuses"
        ))
    };
    let probe = usize::try_from(probe).map_err(|_| refused())?;
    let mut buffer: Vec<u8> = Vec::new();
    buffer.try_reserve_exact(probe).map_err(|_| refused())?;
    // The probe must reach the allocator (an unused allocation may be elided).
    std::hint::black_box(buffer.as_mut_ptr());
    Ok(())
}
