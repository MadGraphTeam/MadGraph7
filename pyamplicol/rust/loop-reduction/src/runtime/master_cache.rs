// SPDX-License-Identifier: 0BSD

//! Per-batch cache of binary64 master integrals in front of a NinjaRS
//! [`FastMasters`] provider (design §11.2).
//!
//! Every request row is first brought to a canonical argument order, then
//! keyed by the exact bit patterns of its arguments. Canonicalisation is
//! restricted to relabellings under which the requested outputs are invariant:
//!
//! - `C0`: the six relabellings of the triangle, acting consistently on its
//!   three invariants and squared masses;
//! - `D0`: the eight dihedral relabellings of the box, acting consistently on
//!   `(p_i², s12, s23, m_i²)`. They keep consecutive propagators adjacent, so a
//!   box reaches oneloopmaster in loop-cycle order whenever the caller's family
//!   is in loop-cycle order;
//! - `A0` needs none;
//! - bubbles are keyed by the ordered tuple `[p², m0², m1², mu²]`: their tensor
//!   coefficients are not invariant under `m0 <-> m1`
//!   (`B1(p²; m0, m1) = -B0 - B1(p²; m1, m0)`).
//!
//! The same canonical rows are sent to the provider whether or not caching is
//! enabled, so cached and uncached reductions agree bit for bit as long as the
//! provider evaluates each row independently of the other rows of its batch.
//! Only successful rows are cached; keys are exact, so entries never go stale,
//! and [`MasterCache::clear`] only bounds memory.

use ninjars::Result;
use ninjars::fast::{C64, FastMaster, FastMasters};
use std::collections::HashMap;
use std::time::Instant;

const FAMILIES: [FastMaster; 6] = [
    FastMaster::A0,
    FastMaster::BubbleRank2,
    FastMaster::BubbleRank3,
    FastMaster::BubbleRank4,
    FastMaster::C0,
    FastMaster::D0,
];
const MAX_ARITY: usize = 11;

fn family_index(family: FastMaster) -> usize {
    FAMILIES.iter().position(|f| *f == family).unwrap()
}

/// Request counts per master family since the last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MasterStatistics {
    /// Rows requested by the reduction engine.
    pub requests: [u64; 6],
    /// Rows served without a provider evaluation (cached, or repeated in one
    /// request batch).
    pub hits: [u64; 6],
    /// Rows evaluated by the provider.
    pub evaluated: [u64; 6],
    /// Wall time of the cache's `evaluate` calls (canonicalisation, cache
    /// look-ups and the provider), nanoseconds.
    pub nanoseconds: u64,
    /// Wall time inside the provider's `evaluate` calls, nanoseconds.
    pub provider_nanoseconds: u64,
}

impl MasterStatistics {
    pub const FAMILY_NAMES: [&'static str; 6] = ["A0", "B2", "B3", "B4", "C0", "D0"];

    pub fn total_requests(&self) -> u64 {
        self.requests.iter().sum()
    }
    pub fn total_hits(&self) -> u64 {
        self.hits.iter().sum()
    }
    pub fn total_evaluated(&self) -> u64 {
        self.evaluated.iter().sum()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    family: FastMaster,
    bits: [u64; 2 * MAX_ARITY],
}

impl Key {
    fn new(family: FastMaster, row: &[C64]) -> Self {
        let mut bits = [0; 2 * MAX_ARITY];
        for (slot, value) in bits.chunks_exact_mut(2).zip(row) {
            slot[0] = value.re.to_bits();
            slot[1] = value.im.to_bits();
        }
        Self { family, bits }
    }
}

fn order_key(value: &C64) -> (u64, u64) {
    (value.re.to_bits(), value.im.to_bits())
}

/// Write the canonical form of one request row into `out`.
///
/// For `n = 3, 4` cut lines the row holds the adjacent invariants
/// `(lines 0-1, 1-2, ..., (n-1)-0)`, for boxes the two diagonals `(0-2, 1-3)`,
/// then the `n` squared masses and the scale. The canonical row is the
/// lexicographic minimum (on bit patterns) over the dihedral relabellings of
/// the lines, which for `n = 3` are all permutations.
fn canonicalise(family: FastMaster, row: &[C64], out: &mut [C64]) {
    let n = match family {
        FastMaster::C0 => 3,
        FastMaster::D0 => 4,
        _ => {
            out.copy_from_slice(row);
            return;
        }
    };
    let sides = n;
    let diagonals = if n == 4 { 2 } else { 0 };
    let masses = sides + diagonals;
    // Invariant between distinct lines i and j of the original row.
    let pair = |i: usize, j: usize| {
        if (i + 1) % n == j {
            row[i]
        } else if (j + 1) % n == i {
            row[j]
        } else {
            row[sides + i.min(j) % 2]
        }
    };
    let mut candidate = [C64::ZERO; MAX_ARITY];
    let mut best = [C64::ZERO; MAX_ARITY];
    let arity = row.len();
    let mut first = true;
    for reflected in [false, true] {
        for rotation in 0..n {
            let line = |k: usize| {
                if reflected {
                    (rotation + n - k) % n
                } else {
                    (rotation + k) % n
                }
            };
            let (invariants, rest) = candidate.split_at_mut(masses);
            for (k, slot) in invariants[..sides].iter_mut().enumerate() {
                *slot = pair(line(k), line((k + 1) % n));
            }
            for (k, slot) in invariants[sides..].iter_mut().enumerate() {
                *slot = pair(line(k), line(k + 2));
            }
            for (k, slot) in rest[..n].iter_mut().enumerate() {
                *slot = row[masses + line(k)];
            }
            candidate[arity - 1] = row[arity - 1];
            if first
                || candidate[..arity]
                    .iter()
                    .map(order_key)
                    .lt(best[..arity].iter().map(order_key))
            {
                best[..arity].copy_from_slice(&candidate[..arity]);
                first = false;
            }
        }
    }
    out.copy_from_slice(&best[..arity]);
}

/// Where a request row's outputs come from.
#[derive(Clone, Copy)]
enum Source {
    Cached(usize),
    Evaluated(usize),
}

/// Canonicalising, optionally caching [`FastMasters`] wrapper.
pub struct MasterCache<M> {
    inner: M,
    enabled: bool,
    entries: HashMap<Key, usize>,
    values: Vec<[C64; 3]>,
    statistics: MasterStatistics,
    canonical: Vec<C64>,
    sources: Vec<Source>,
    pending: HashMap<Key, usize>,
    pending_keys: Vec<Key>,
    miss_args: Vec<C64>,
    miss_out: Vec<[C64; 3]>,
    miss_ok: Vec<bool>,
}

impl<M: FastMasters> MasterCache<M> {
    pub fn new(inner: M, enabled: bool) -> Self {
        Self {
            inner,
            enabled,
            entries: HashMap::new(),
            values: Vec::new(),
            statistics: MasterStatistics::default(),
            canonical: Vec::new(),
            sources: Vec::new(),
            pending: HashMap::new(),
            pending_keys: Vec::new(),
            miss_args: Vec::new(),
            miss_out: Vec::new(),
            miss_ok: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Drop every cached value (statistics are kept).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.values.clear();
    }

    /// Number of distinct cached rows.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn statistics(&self) -> MasterStatistics {
        self.statistics
    }

    pub fn take_statistics(&mut self) -> MasterStatistics {
        std::mem::take(&mut self.statistics)
    }
}

impl<M: FastMasters> FastMasters for MasterCache<M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let start = Instant::now();
        let result = self.evaluate_rows(family, args, out, ok);
        self.statistics.nanoseconds += start.elapsed().as_nanos() as u64;
        result
    }
}

impl<M: FastMasters> MasterCache<M> {
    fn provider(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let start = Instant::now();
        let result = self.inner.evaluate(family, args, out, ok);
        self.statistics.provider_nanoseconds += start.elapsed().as_nanos() as u64;
        result
    }

    fn evaluate_rows(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> Result<()> {
        let arity = family.arity();
        let width = family.outputs();
        let rows = ok.len();
        for (name, expected, actual) in [
            ("cached master arguments", rows * arity, args.len()),
            ("cached master outputs", rows * width, out.len()),
        ] {
            if expected != actual {
                return Err(ninjars::Error::Shape {
                    name,
                    expected,
                    actual,
                });
            }
        }
        let index = family_index(family);
        self.statistics.requests[index] += rows as u64;
        self.canonical.resize(args.len(), C64::ZERO);
        for (row, canonical) in args
            .chunks_exact(arity)
            .zip(self.canonical.chunks_exact_mut(arity))
        {
            canonicalise(family, row, canonical);
        }
        if !self.enabled {
            self.statistics.evaluated[index] += rows as u64;
            let canonical = std::mem::take(&mut self.canonical);
            let result = self.provider(family, &canonical, out, ok);
            self.canonical = canonical;
            return result;
        }
        self.sources.clear();
        self.pending.clear();
        self.pending_keys.clear();
        self.miss_args.clear();
        for row in self.canonical.chunks_exact(arity) {
            let key = Key::new(family, row);
            let source = if let Some(&offset) = self.entries.get(&key) {
                Source::Cached(offset)
            } else if let Some(&miss) = self.pending.get(&key) {
                Source::Evaluated(miss)
            } else {
                let miss = self.pending_keys.len();
                self.pending.insert(key, miss);
                self.pending_keys.push(key);
                self.miss_args.extend_from_slice(row);
                Source::Evaluated(miss)
            };
            self.sources.push(source);
        }
        let misses = self.pending_keys.len();
        self.statistics.evaluated[index] += misses as u64;
        self.statistics.hits[index] += (rows - misses) as u64;
        self.miss_out.clear();
        self.miss_out.resize(misses * width, [C64::ZERO; 3]);
        self.miss_ok.clear();
        self.miss_ok.resize(misses, false);
        if misses > 0 {
            let (miss_args, mut miss_out, mut miss_ok) = (
                std::mem::take(&mut self.miss_args),
                std::mem::take(&mut self.miss_out),
                std::mem::take(&mut self.miss_ok),
            );
            let result = self.provider(family, &miss_args, &mut miss_out, &mut miss_ok);
            (self.miss_args, self.miss_out, self.miss_ok) = (miss_args, miss_out, miss_ok);
            result?;
        }
        for (miss, key) in self.pending_keys.iter().enumerate() {
            if self.miss_ok[miss] {
                self.entries.insert(*key, self.values.len());
                self.values
                    .extend_from_slice(&self.miss_out[miss * width..(miss + 1) * width]);
            }
        }
        for ((source, target), status) in self
            .sources
            .iter()
            .zip(out.chunks_exact_mut(width))
            .zip(ok.iter_mut())
        {
            match *source {
                Source::Cached(offset) => {
                    target.copy_from_slice(&self.values[offset..offset + width]);
                    *status = true;
                }
                Source::Evaluated(miss) => {
                    target.copy_from_slice(&self.miss_out[miss * width..(miss + 1) * width]);
                    *status = self.miss_ok[miss];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(re: f64) -> C64 {
        C64::real(re)
    }

    /// Box row for lines with symmetric invariant table `s` and masses `m`.
    fn box_row(s: &[[f64; 4]; 4], m: [f64; 4], lines: [usize; 4]) -> Vec<C64> {
        let [a, b, c, d] = lines;
        vec![
            value(s[a][b]),
            value(s[b][c]),
            value(s[c][d]),
            value(s[d][a]),
            value(s[a][c]),
            value(s[b][d]),
            value(m[a]),
            value(m[b]),
            value(m[c]),
            value(m[d]),
            value(91.),
        ]
    }

    #[test]
    fn dihedral_relabellings_share_one_canonical_box() {
        let s = [
            [0., 1.5, -7.25, 0.],
            [1.5, 0., 0., -3.5],
            [-7.25, 0., 0., 2.125],
            [0., -3.5, 2.125, 0.],
        ];
        let m = [0., 0., 30.25, 30.25];
        let reference = box_row(&s, m, [0, 1, 2, 3]);
        let mut canonical = vec![C64::ZERO; 11];
        canonicalise(FastMaster::D0, &reference, &mut canonical);
        for lines in [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 3, 2, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [1, 0, 3, 2],
        ] {
            let mut other = vec![C64::ZERO; 11];
            canonicalise(FastMaster::D0, &box_row(&s, m, lines), &mut other);
            assert_eq!(other, canonical, "{lines:?}");
        }
        // A non-dihedral relabelling is a different box.
        let mut crossed = vec![C64::ZERO; 11];
        canonicalise(FastMaster::D0, &box_row(&s, m, [0, 2, 1, 3]), &mut crossed);
        assert_ne!(crossed, canonical);
    }

    #[test]
    fn triangles_canonicalise_under_all_permutations_and_bubbles_keep_order() {
        let s = [[0., 2., -5.], [2., 0., 11.], [-5., 11., 0.]];
        let m = [0., 3., 7.];
        let row = |[a, b, c]: [usize; 3]| {
            vec![
                value(s[a][b]),
                value(s[b][c]),
                value(s[c][a]),
                value(m[a]),
                value(m[b]),
                value(m[c]),
                value(4.),
            ]
        };
        let mut reference = vec![C64::ZERO; 7];
        canonicalise(FastMaster::C0, &row([0, 1, 2]), &mut reference);
        for lines in [[1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]] {
            let mut other = vec![C64::ZERO; 7];
            canonicalise(FastMaster::C0, &row(lines), &mut other);
            assert_eq!(other, reference);
        }
        let bubble = [value(5.), value(0.), value(9.), value(4.)];
        let mut out = [C64::ZERO; 4];
        canonicalise(FastMaster::BubbleRank2, &bubble, &mut out);
        assert_eq!(out, bubble);
    }

    /// Counts provider rows and returns a row-dependent value.
    struct Counting(usize);
    impl FastMasters for Counting {
        fn evaluate(
            &mut self,
            family: FastMaster,
            args: &[C64],
            out: &mut [[C64; 3]],
            ok: &mut [bool],
        ) -> Result<()> {
            let arity = family.arity();
            let width = family.outputs();
            for (row, (point, values)) in args
                .chunks_exact(arity)
                .zip(out.chunks_exact_mut(width))
                .enumerate()
            {
                self.0 += 1;
                let sum: f64 = point.iter().map(|z| z.re).sum();
                for (k, triple) in values.iter_mut().enumerate() {
                    *triple = [value(sum + k as f64), value(1.), value(2.)];
                }
                ok[row] = point[0].re >= 0.;
            }
            Ok(())
        }
    }

    #[test]
    fn repeated_rows_are_evaluated_once_and_failures_are_not_cached() {
        let mut cache = MasterCache::new(Counting(0), true);
        let rows = [
            [value(1.), value(2.), value(3.), value(4.)],
            [value(1.), value(2.), value(3.), value(4.)],
            [value(-1.), value(2.), value(3.), value(4.)],
        ]
        .concat();
        let mut out = vec![[C64::ZERO; 3]; 12];
        let mut ok = vec![false; 3];
        cache
            .evaluate(FastMaster::BubbleRank3, &rows, &mut out, &mut ok)
            .unwrap();
        assert_eq!(ok, [true, true, false]);
        assert_eq!(out[..4], out[4..8]);
        assert_eq!(cache.inner.0, 2);
        cache
            .evaluate(FastMaster::BubbleRank3, &rows, &mut out, &mut ok)
            .unwrap();
        // The successful row is cached; the failed one is evaluated again.
        assert_eq!(cache.inner.0, 3);
        assert_eq!(cache.len(), 1);
        let statistics = cache.statistics();
        assert_eq!(statistics.total_requests(), 6);
        assert_eq!(statistics.total_evaluated(), 3);
        assert_eq!(statistics.total_hits(), 3);
    }
}
