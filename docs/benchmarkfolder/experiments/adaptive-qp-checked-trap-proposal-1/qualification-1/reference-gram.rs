//! Small-Gram expansion of box coefficients (Denner and Dittmaier,
//! hep-ph/0509141, sect. "Reduction for small Gram determinant", eqs.
//! (recurGram1) and (recurGram2); COLLIER's `CalcDg`).
//!
//! For a box with momenta `p_1..p_3` (relative to its first member), Gram
//! matrix `Z = (2 p_i p_j)`, `Delta = det Z` and `f_i = p_i^2 - m_i^2 + m_0^2`,
//! the PV relations contracted with the adjugate `Zadj` give, with
//! `(Zadj f)_j = sum_n Zadj_jn f_n` maximal and `Zadj_kl` maximal,
//!
//!   C(0, k) = [ sum_n Zadj_jn (Shat_n(0, k) - 2 k_n C(1, k - e_n))
//!               - Delta C(0, k + e_j) ] / (Zadj f)_j,
//!
//!   4 ((l + |k|) - eps) Zadj_kl C(l, k) = - Delta C(l-1, k + e_k + e_l)
//!       + 2 Zadj_kl C0(l-1, k) + Xtilde_kl C(l-1, k)
//!       + sum_n (Zadj_nl Shat_n(l-1, k + e_k) - Zadj_kl Shat_n(l-1, k + e_n))
//!       - sum_nm Zadj2_(kn)(lm) [ f_n Shat_m(l-1, k) + 2 k_n Shat_m(l, k - e_n)
//!           - 2 f_n k_m C(l, k - e_m) - 2 f_m k_n C(l, k - e_n)
//!           - 4 (k_n k_m - delta_nm k_n) C(l+1, k - e_n - e_m) ],
//!
//! for `C(lambda, kappa)` in the engine's normalization (`lambda` metric
//! pairs, momentum multiplicities `kappa`). `Shat_n` is the triangle without
//! member `n` minus the shifted triangle without the first member (`C0`),
//! `Zadj2` the second adjugate and `Xtilde_kl = 2 m_0^2 Zadj_kl +
//! sum_nm Zadj2_(kn)(lm) f_n f_m`. Dropping the `Delta` terms gives every
//! coefficient from triangles of the same rank; each further pass puts the
//! previous pass into the `Delta` terms and gains a factor `Delta /
//! (Zadj f)_j`. With triangles of rank `r + G` the rank-`r` coefficients
//! carry `G` corrections. The scalar `D0` is the master itself. When it
//! does not converge, the expansion for small Gram and small modified Cayley
//! determinants (sect. 3.5) is tried, then the alternative PV reduction
//! through the modified Cayley matrix (sect. 3.2), which is exact where the
//! modified Cayley determinant does not cancel (it vanishes for IR-divergent
//! boxes and triangles). Where it cancels, the expansions get the larger
//! boost their observed convergence rate predicts (`predicted_passes`); the
//! lane is refused otherwise.

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use crate::direct::plan::FastMap;

use iregi_masters::NativeFloat;
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use super::{DirectExecutor, DirectMasters, GRAM_EXPANSION};
use crate::direct::{
    numeric::{
        C, L4, cadd, cinv, cmul, cnorm2, cscale, csub, czero, dimension_powers, div_dimension, div_dimension_by, lfma, lfma_slots,
        lzero, mul_dimension,
    },
    plan::{DirectPlanData, SubsetKind},
};
use crate::{KinematicsBatch, LaneStatus};

/// Expansion executors of a binary64 executor, by subset size, coefficient
/// rank, boost and whether the expansion is skipped (alternative PV only).
#[derive(Clone)]
pub(crate) struct GramRescue<T: NativeFloat> {
    executors: HashMap<(usize, u16, u16, bool), DirectExecutor<T>>,
    /// Boosts tried in order: a smaller one first (fewer high-rank triangles
    /// for its gate to refuse), the largest one only where it fails. Where
    /// the largest one's expansions converge too slowly for its passes and
    /// the alternative PV is refused, the workspace's own boxes and
    /// triangles (and those of the auxiliary executors of their last
    /// attempts) get further attempts at the boost their observed
    /// convergence predicts (`predicted_passes`), up to twice the largest.
    boosts: Vec<u16>,
    /// Alternative PV reduction only (no expansions), on the boxes and
    /// triangles whose a priori estimate (`alternative_estimate_of`) is at
    /// most `ALTERNATIVE_PV_LIMIT`.
    alternative: bool,
    /// Try the alternative PV reduction first on the boxes whose a priori
    /// estimate (`alternative_estimate_of`) is at most this, before the
    /// expansions.
    alternative_first: Option<f64>,
    /// Gate of the auxiliary executors' triangles (default: the parent's
    /// threshold squared).
    aux_threshold: Option<T>,
}

impl<T: NativeFloat> GramRescue<T> {
    pub(crate) fn new(boost: u16) -> Self {
        let mut boosts = vec![boost.min(FIRST_BOOST), boost];
        boosts.dedup();
        Self { executors: HashMap::new(), boosts, alternative: false, alternative_first: None, aux_threshold: None }
    }

    pub(crate) fn set_alternative_first(&mut self, limit: Option<f64>) {
        self.alternative_first = limit;
    }

    pub(crate) fn everywhere(boost: u16, aux_threshold: T) -> Self {
        Self { aux_threshold: Some(aux_threshold), ..Self::new(boost) }
    }

    /// The alternative PV reduction alone: triangles of the box's own rank
    /// suffice (boost 1 keeps the expansion bookkeeping valid).
    /// Bits identifying this configuration (keys of shared results).
    pub(crate) fn signature(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |x: u64| {
            hash ^= x;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        };
        for &boost in &self.boosts {
            eat(u64::from(boost));
        }
        eat(u64::from(self.alternative));
        eat(self.alternative_first.map_or(u64::MAX, f64::to_bits));
        let [threshold, threshold_low] = self.aux_threshold.as_ref().map_or([u64::MAX; 2], super::real_key_words);
        eat(threshold);
        eat(threshold_low);
        hash
    }

    pub(crate) fn alternative() -> Self {
        Self { executors: HashMap::new(), boosts: vec![1], alternative: true, alternative_first: None, aux_threshold: None }
    }
}

/// First boost tried (see `GramRescue::boosts`).
const FIRST_BOOST: u16 = 8;

type Key = (u16, [u16; 3]);

/// Index of the expansion coefficients `(lambda, kappa)`, kappa over `nm`
/// (1 to 3) momenta, graded by rank `2 lambda + |kappa|`: every key of rank
/// `r` comes after those of lower ranks, so a set truncated in rank is a
/// prefix. Within a rank, lambda ascending, then kappa as `compositions`
/// lists it. (Hash maps keyed by `Key` took most of the expansions' time.)
struct Grading {
    nm: usize,
    max_rank: u16,
    /// First index of each rank, then the total.
    rank_start: Vec<usize>,
    /// Offset of lambda within rank r: `lambda_start[r * half + lambda]`.
    lambda_start: Vec<usize>,
    half: usize,
    /// The key of every index.
    keys: Vec<Key>,
    /// Neighbours of every index (`NONE` outside the grading): `(lambda - 1,
    /// kappa)`, `(lambda + 1, kappa)`, then `(lambda, kappa + e_i)` and
    /// `(lambda, kappa - e_i)` for `i < 3`. The expansions' recursions read
    /// their terms through these instead of indexing every key.
    near: Vec<[u32; 8]>,
}

/// No index (outside the grading).
const NONE: u32 = u32::MAX;

thread_local! {
    /// Gradings with their neighbour tables by `(nm, max_rank)`, built once
    /// per thread.
    static GRADINGS: std::cell::RefCell<HashMap<(usize, u16), std::rc::Rc<Grading>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// The cached grading of `nm` momenta through `max_rank`.
fn cached_grading(nm: usize, max_rank: u16) -> std::rc::Rc<Grading> {
    GRADINGS.with(|cache| {
        cache.borrow_mut().entry((nm, max_rank)).or_insert_with(|| std::rc::Rc::new(Grading::new(nm, max_rank))).clone()
    })
}

/// Coefficients needed per order of a small-Gram expansion, by grading index.
type Need = std::rc::Rc<Vec<Vec<bool>>>;

thread_local! {
    /// [`expansion_need`] by `(nm, max_rank, target, passes, j, k, l)`.
    static NEEDS: std::cell::RefCell<HashMap<(usize, u16, u16, u16, usize, usize, usize), Need>> =
        std::cell::RefCell::new(HashMap::new());
}

/// The coefficients of orders `0..=passes` that the expansion's results (every
/// key of rank at most `target` of every order) depend on: the recursion's
/// terms walked back from those keys. The others (most of them in the highest
/// ranks, reached only to keep the rank bookkeeping of the orders simple) are
/// never read. `pairs` are the `(a, b)` of the second adjugates.
fn expansion_need(
    grading: &Grading,
    target: u16,
    passes: u16,
    (j, k, l): (usize, usize, usize),
    pairs: &[(usize, usize)],
) -> Need {
    let key = (grading.nm, grading.max_rank, target, passes, j, k, l);
    if let Some(need) = NEEDS.with(|cache| cache.borrow().get(&key).cloned()) {
        return need;
    }
    let size = grading.end(grading.max_rank);
    let orders = usize::from(passes) + 1;
    let mut need = vec![vec![false; size]; orders];
    let mut stack: Vec<(usize, u32)> = (0..orders).flat_map(|s| (0..grading.end(target)).map(move |index| (s, index as u32))).collect();
    while let Some((s, index)) = stack.pop() {
        if index == NONE || need[s][index as usize] {
            continue;
        }
        need[s][index as usize] = true;
        let (lambda, kappa) = grading.key(index as usize);
        if lambda > 0 {
            // As `expand_top`'s metric step.
            let below = grading.lower(index);
            if s > 0 {
                stack.push((s - 1, grading.plus(grading.plus(below, k), l)));
            }
            stack.push((s, below));
            for &(a, b) in pairs {
                let (ka, kb) = (usize::from(kappa[a]), usize::from(kappa[b]));
                if ka > 0 {
                    stack.push((s, grading.minus(index, a)));
                }
                if kb > 0 {
                    stack.push((s, grading.minus(index, b)));
                }
                if (if a == b { ka * ka.saturating_sub(1) } else { ka * kb }) > 0 {
                    stack.push((s, grading.higher(grading.minus(grading.minus(index, a), b))));
                }
            }
        } else if kappa.iter().any(|&x| x > 0) {
            // As its momentum step (the scalar of each order is set directly).
            for m in 0..grading.nm {
                if kappa[m] > 0 {
                    stack.push((s, grading.higher(grading.minus(index, m))));
                }
            }
            if s > 0 {
                stack.push((s - 1, grading.plus(index, j)));
            }
        }
    }
    let need = std::rc::Rc::new(need);
    NEEDS.with(|cache| cache.borrow_mut().insert(key, need.clone()));
    need
}

thread_local! {
    /// [`cayley_need`] by `(nm, max_rank, target, passes, l, k, gi, gj)`.
    static CAYLEY_NEEDS: std::cell::RefCell<HashMap<(usize, u16, u16, u16, [usize; 4]), Need>> =
        std::cell::RefCell::new(HashMap::new());
}

/// As [`expansion_need`] for the expansion for small Gram and small modified
/// Cayley determinants (`expand_top`'s `gc_metric` and `gc_momentum`): the
/// coefficients of orders `0..=passes` that every key of rank at most
/// `target` of every order depends on. Its second adjugates
/// `Zadj2_(gi n)(gj m)` exist exactly for `n != gi`, `m != gj`.
fn cayley_need(grading: &Grading, target: u16, passes: u16, (l, k, gi, gj): (usize, usize, usize, usize)) -> Need {
    let key = (grading.nm, grading.max_rank, target, passes, [l, k, gi, gj]);
    if let Some(need) = CAYLEY_NEEDS.with(|cache| cache.borrow().get(&key).cloned()) {
        return need;
    }
    let nm = grading.nm;
    let size = grading.end(grading.max_rank);
    let orders = usize::from(passes) + 1;
    let mut need = vec![vec![false; size]; orders];
    let mut stack: Vec<(usize, u32)> = (0..orders).flat_map(|s| (0..grading.end(target)).map(move |index| (s, index as u32))).collect();
    while let Some((s, index)) = stack.pop() {
        if index == NONE || need[s][index as usize] {
            continue;
        }
        need[s][index as usize] = true;
        let (lambda, kappa) = grading.key(index as usize);
        if lambda > 0 {
            // As `gc_metric` for (level, target_kappa) = (lambda, kappa).
            for m in (0..nm).filter(|&m| m != l) {
                if kappa[m] > 0 {
                    stack.push((s, grading.plus(grading.minus(index, m), l)));
                }
            }
            if s > 0 {
                let source = grading.plus(grading.lower(index), l);
                stack.push((s - 1, source));
                stack.push((s - 1, grading.plus(source, k)));
            }
        } else if kappa.iter().any(|&x| x > 0) {
            // As `gc_momentum` (the scalar of each order is set directly).
            let metric = grading.higher(index);
            stack.push((s, metric));
            for m in (0..nm).filter(|&m| m != gj) {
                if kappa[m] > 0 {
                    stack.push((s, grading.minus(metric, m)));
                }
            }
            if s > 0 {
                stack.push((s - 1, grading.plus(index, gi)));
            }
        }
    }
    let need = std::rc::Rc::new(need);
    CAYLEY_NEEDS.with(|cache| cache.borrow_mut().insert(key, need.clone()));
    need
}

impl Grading {
    fn new(nm: usize, max_rank: u16) -> Self {
        let count = |d: usize| match nm {
            1 => 1,
            2 => d + 1,
            _ => (d + 1) * (d + 2) / 2,
        };
        let top = usize::from(max_rank);
        let half = top / 2 + 1;
        let mut rank_start = Vec::with_capacity(top + 2);
        let mut lambda_start = vec![0; (top + 1) * half];
        let mut next = 0;
        for r in 0..=top {
            rank_start.push(next);
            let mut offset = 0;
            for lambda in 0..=r / 2 {
                lambda_start[r * half + lambda] = offset;
                offset += count(r - 2 * lambda);
            }
            next += offset;
        }
        rank_start.push(next);
        let mut grading = Self { nm, max_rank, rank_start, lambda_start, half, keys: Vec::with_capacity(next), near: Vec::new() };
        for r in 0..=max_rank {
            for lambda in 0..=r / 2 {
                let d = r - 2 * lambda;
                let kappas = if nm == 1 { vec![[d, 0, 0]] } else { compositions(d, nm) };
                grading.keys.extend(kappas.into_iter().map(|kappa| (lambda, kappa)));
            }
        }
        let at = |lambda: u16, kappa: [u16; 3]| grading.index(lambda, &kappa).map_or(NONE, |index| index as u32);
        let near: Vec<[u32; 8]> = grading
            .keys
            .iter()
            .map(|&(lambda, kappa)| {
                let mut out = [NONE; 8];
                if lambda > 0 {
                    out[0] = at(lambda - 1, kappa);
                }
                out[1] = at(lambda + 1, kappa);
                for i in 0..3 {
                    let mut more = kappa;
                    more[i] += 1;
                    out[2 + i] = at(lambda, more);
                    if kappa[i] > 0 {
                        let mut less = kappa;
                        less[i] -= 1;
                        out[5 + i] = at(lambda, less);
                    }
                }
                out
            })
            .collect();
        grading.near = near;
        grading
    }

    /// The key of `index`.
    fn key(&self, index: usize) -> Key {
        self.keys[index]
    }

    /// Indices of the keys `(lambda, kappa)` of rank `r`, kappa as
    /// `compositions` lists it.
    fn block(&self, r: u16, lambda: u16) -> std::ops::Range<usize> {
        let start = self.rank_start[usize::from(r)] + self.lambda_start[usize::from(r) * self.half + usize::from(lambda)];
        let d = usize::from(r - 2 * lambda);
        let count = match self.nm {
            1 => 1,
            2 => d + 1,
            _ => (d + 1) * (d + 2) / 2,
        };
        start..start + count
    }

    fn neighbour(&self, index: u32, which: usize) -> u32 {
        self.near.get(index as usize).map_or(NONE, |near| near[which])
    }

    /// `(lambda - 1, kappa)` of `index`.
    fn lower(&self, index: u32) -> u32 {
        self.neighbour(index, 0)
    }

    /// `(lambda + 1, kappa)` of `index`.
    fn higher(&self, index: u32) -> u32 {
        self.neighbour(index, 1)
    }

    /// `(lambda, kappa + e_i)` of `index`.
    fn plus(&self, index: u32, i: usize) -> u32 {
        self.neighbour(index, 2 + i)
    }

    /// `(lambda, kappa - e_i)` of `index`.
    fn minus(&self, index: u32, i: usize) -> u32 {
        self.neighbour(index, 5 + i)
    }

    /// Number of keys of rank at most `r`.
    fn end(&self, r: u16) -> usize {
        self.rank_start[usize::from(r.min(self.max_rank)) + 1]
    }

    /// Indices of the keys of rank `r`.
    fn range(&self, r: u16) -> std::ops::Range<usize> {
        if r > self.max_rank {
            return 0..0;
        }
        self.rank_start[usize::from(r)]..self.rank_start[usize::from(r) + 1]
    }

    #[inline]
    fn index(&self, lambda: u16, kappa: &[u16; 3]) -> Option<usize> {
        let [a, b, c] = kappa.map(usize::from);
        // Components beyond `nm` must vanish.
        let outside = match self.nm {
            0 => a | b | c,
            1 => b | c,
            2 => c,
            _ => 0,
        };
        let d = a + b + c;
        let r = 2 * usize::from(lambda) + d;
        if outside != 0 || r > usize::from(self.max_rank) {
            return None;
        }
        let position = match self.nm {
            1 => 0,
            2 => a,
            _ => a * (d + 1) - a * a.saturating_sub(1) / 2 + b,
        };
        Some(self.rank_start[r] + self.lambda_start[r * self.half + usize::from(lambda)] + position)
    }
}

/// Expansion coefficients stored by their `Grading` index. `present` marks
/// the stored ones and its length is the set's extent; `values` may be longer
/// (reused storage), and only present entries are ever read.
#[derive(Clone)]
struct Coefficients<T> {
    values: Vec<L4<T>>,
    present: Vec<bool>,
}

impl<T: NativeFloat> Coefficients<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self { values: Vec::with_capacity(capacity), present: Vec::with_capacity(capacity) }
    }

    fn at(&self, index: usize) -> Option<&L4<T>> {
        self.present.get(index).copied().unwrap_or(false).then(|| &self.values[index])
    }

    fn at_mut(&mut self, index: usize) -> Option<&mut L4<T>> {
        if self.present.get(index).copied().unwrap_or(false) { Some(&mut self.values[index]) } else { None }
    }

    fn get(&self, grading: &Grading, key: Key) -> Option<&L4<T>> {
        self.at(grading.index(key.0, &key.1)?)
    }

    /// The coefficient `key`, or the error "`what`: coefficient `key` missing".
    fn need(&self, grading: &Grading, key: Key, what: &str) -> Result<&L4<T>, String> {
        self.get(grading, key).ok_or_else(|| format!("{what}: coefficient {key:?} missing"))
    }

    /// The coefficient at `index` (`NONE` included), or the error "`what`:
    /// coefficient `key` missing" with the key it stands for.
    fn need_at(&self, index: u32, key: impl FnOnce() -> Key, what: &str) -> Result<&L4<T>, String> {
        self.at(index as usize).ok_or_else(|| format!("{what}: coefficient {:?} missing", key()))
    }

    fn insert(&mut self, grading: &Grading, key: Key, value: L4<T>, zero: &C<T>) -> Result<(), String> {
        let index = grading
            .index(key.0, &key.1)
            .ok_or_else(|| format!("small-Gram expansion: coefficient {key:?} outside the expansion ranks"))?;
        self.insert_at(grading, index, value, zero);
        Ok(())
    }

    /// Insert at an index of `grading`.
    #[inline]
    fn insert_at(&mut self, grading: &Grading, index: usize, value: L4<T>, zero: &C<T>) {
        if index >= self.present.len() {
            self.extend_to(grading, index, zero);
        }
        self.values[index] = value;
        self.present[index] = true;
    }

    /// Extend the set through the rank of `index` (nothing new present;
    /// storage beyond the old extent is reused as it is, else zeroed).
    #[cold]
    #[inline(never)]
    fn extend_to(&mut self, grading: &Grading, index: usize, zero: &C<T>) {
        let end = grading.end(rank_of(&grading.key(index)));
        if self.values.len() < end {
            self.values.resize(end, lzero(zero));
        }
        self.present.resize(end, false);
    }
}

/// Reused storage of [`Coefficients`] (one executor's scratch): the
/// expansions build about a dozen coefficient sets per call, and fresh
/// allocations of their size cost page faults and zero fills.
pub(super) struct CoefficientPool<T> {
    free: Vec<(Vec<L4<T>>, Vec<bool>)>,
}

impl<T> Default for CoefficientPool<T> {
    fn default() -> Self {
        Self { free: Vec::new() }
    }
}

impl<T> Clone for CoefficientPool<T> {
    /// Scratch only: a clone starts empty.
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<T: NativeFloat> Coefficients<T> {
    /// [`Coefficients::with_capacity`] on reused storage (same contents:
    /// nothing present).
    fn pooled(pool: &RefCell<CoefficientPool<T>>, capacity: usize) -> Self {
        let mut pool = pool.borrow_mut();
        // The smallest storage that holds `capacity` values, else the
        // largest (the least to extend).
        let fit = |len: usize| if len >= capacity { (1, usize::MAX - len) } else { (0, len) };
        let best = (0..pool.free.len()).max_by_key(|&i| fit(pool.free[i].0.len()));
        match best.map(|i| pool.free.swap_remove(i)) {
            Some((mut values, mut present)) => {
                // Stale values stay (never read before they are stored).
                present.clear();
                values.reserve(capacity.saturating_sub(values.len()));
                present.reserve(capacity);
                Self { values, present }
            }
            None => Self::with_capacity(capacity),
        }
    }

    /// Return the storage to `pool`.
    fn recycle(self, pool: &RefCell<CoefficientPool<T>>) {
        pool.borrow_mut().free.push((self.values, self.present));
    }

    /// The coefficients of index below `end` (a rank prefix), on reused
    /// storage.
    fn prefix_pooled(&self, pool: &RefCell<CoefficientPool<T>>, end: usize) -> Self {
        let end = end.min(self.present.len());
        let mut out = Self::pooled(pool, end);
        out.values.clear();
        out.values.extend_from_slice(&self.values[..end]);
        out.present.extend_from_slice(&self.present[..end]);
        out
    }

    /// A clone on reused storage.
    fn clone_pooled(&self, pool: &RefCell<CoefficientPool<T>>) -> Self {
        self.prefix_pooled(pool, self.present.len())
    }
}

/// A source value index of [`TopLayout`] outside the triangle's layout.
const MISSING: u32 = u32::MAX;
/// A source value index of [`TopLayout`] that is not read.
const UNUSED: u32 = u32::MAX - 1;

/// What [`DirectExecutor::expand_top`] derives from the plan alone (the top
/// subset's children and layouts), built once per plan variant instead of
/// once per lane.
pub(super) struct TopTables {
    /// The tables, or the error `expand_top` reports for this plan.
    layout: Result<TopLayout, &'static str>,
}

struct TopLayout {
    /// The triangles' (lowest) rank.
    child_rank: u16,
    /// Per grading index `(nm, child_rank + 2)` through `child_rank`: the
    /// value index of the pinch-zero triangle's coefficient that seeds the
    /// shifted `C0` (keys with `kappa_0 = 0`; `MISSING` outside its layout,
    /// `UNUSED` for the other keys).
    spz_source: Vec<u32>,
    /// Per grading index through `child_rank` and pinch `m`: the value index
    /// of the triangle without member `m + 1` at `kappa` without component
    /// `m` (keys with `kappa_m = 0`; `MISSING`, `UNUSED` as above).
    shat_source: Vec<[u32; 3]>,
    /// The top subset's layout in its map's order: value index, grading
    /// index (`NONE` outside the grading) and rank of each key; `None`
    /// without a layout.
    output: Option<Vec<(u32, u32, u16)>>,
    /// Metric keys of each degree through the grading's rank, by decreasing
    /// `kappa_l`, for each `l` (the small-Cayley variant's order).
    gc_kappas: Vec<Vec<Vec<[u16; 3]>>>,
    /// The lowest rank with a `MISSING` source (`u16::MAX`: none).
    missing_rank: u16,
}

impl TopTables {
    pub(super) fn new(plan: &DirectPlanData) -> Self {
        Self { layout: TopLayout::new(plan) }
    }
}

impl TopLayout {
    fn new(plan: &DirectPlanData) -> Result<Self, &'static str> {
        let subset = plan.subsets.last().ok_or("small-Gram expansion: no subsets")?;
        let members = &subset.members;
        if !(3..=4).contains(&members.len()) {
            return Err("small-Gram expansion: the top subset is not a PV triangle or box");
        }
        let nm = members.len() - 1;
        let origin = members[0];
        let lookup = |mask: crate::direct::plan::SubsetMask| plan.coefficient_lookup.get(&mask);
        let mask = subset.mask;
        let children: Vec<&FastMap<(u16, Vec<u16>), u32>> = (0..nm)
            .map(|i| lookup(mask & !(1 << members[i + 1])).ok_or("small-Gram expansion: missing pinch"))
            .collect::<Result<_, _>>()?;
        let zero_child = lookup(mask & !(1 << origin)).ok_or("small-Gram expansion: missing triangle")?;
        let child_rank = children
            .iter()
            .chain(std::iter::once(&zero_child))
            .map(|layout| layout.keys().map(|(lambda, kappa)| 2 * lambda + kappa.iter().sum::<u16>()).max().unwrap_or(0))
            .min()
            .unwrap_or(0);
        let grading = cached_grading(nm, child_rank + 2);
        let child_grading = cached_grading(nm - 1, child_rank);
        let dense_layout = |layout: &FastMap<(u16, Vec<u16>), u32>| -> Vec<u32> {
            let mut out = vec![u32::MAX; child_grading.end(child_rank)];
            for (&(lambda, ref kappa), &index) in layout {
                if kappa.len() > 3 {
                    continue;
                }
                let key: [u16; 3] = std::array::from_fn(|i| kappa.get(i).copied().unwrap_or(0));
                if let Some(position) = child_grading.index(lambda, &key) {
                    out[position] = index;
                }
            }
            out
        };
        let children: Vec<Vec<u32>> = children.into_iter().map(dense_layout).collect();
        let zero_child = dense_layout(zero_child);
        let value = |layout: &[u32], lambda: u16, kappa: [u16; 3]| -> u32 {
            child_grading
                .index(lambda, &kappa)
                .and_then(|position| layout.get(position).copied())
                .filter(|&index| index != u32::MAX)
                .unwrap_or(MISSING)
        };
        let end = grading.end(child_rank);
        let mut spz_source = vec![UNUSED; end];
        let mut shat_source = vec![[UNUSED; 3]; end];
        for index in 0..end {
            let (lambda, kappa) = grading.key(index);
            if kappa[0] == 0 {
                spz_source[index] = value(&zero_child, lambda, [kappa[1], kappa[2], 0]);
            }
            for m in 0..nm {
                if kappa[m] == 0 {
                    let mut reduced = [0u16; 3];
                    for (slot, i) in (0..nm).filter(|&i| i != m).enumerate() {
                        reduced[slot] = kappa[i];
                    }
                    shat_source[index][m] = value(&children[m], lambda, reduced);
                }
            }
        }
        let missing_rank = (0..end)
            .filter(|&index| spz_source[index] == MISSING || shat_source[index].contains(&MISSING))
            .map(|index| rank_of(&grading.key(index)))
            .min()
            .unwrap_or(u16::MAX);
        let output = lookup(mask).map(|layout| {
            layout
                .iter()
                .map(|(&(lambda, ref kappa), &index)| {
                    let key: Key = (lambda, std::array::from_fn(|i| kappa.get(i).copied().unwrap_or(0)));
                    let at = grading.index(key.0, &key.1).map_or(NONE, |at| at as u32);
                    (index, at, rank_of(&key))
                })
                .collect()
        });
        let gc_kappas = (0..nm)
            .map(|l| {
                (0..=grading.max_rank)
                    .map(|d| {
                        let mut kappas = compositions(d, nm);
                        kappas.sort_by_key(|kappa| std::cmp::Reverse(kappa[l]));
                        kappas
                    })
                    .collect()
            })
            .collect();
        Ok(Self { child_rank, spz_source, shat_source, output, gc_kappas, missing_rank })
    }
}

/// COLLIER-style indicator of a box or triangle: `w = max_j |(Zadj f)_j| /
/// |Delta|` (`None` if it vanishes or is not finite). Passarino-Veltman
/// rounding grows roughly like `w^r`, the alternative PV reduction's like
/// `w^ceil(r/2)`, and `1/w` is the small-Gram expansion parameter.
pub(super) fn gram_indicator<T: NativeFloat>(z: &[[C<T>; 3]; 3], f: &[C<T>; 3], nm: usize) -> Option<f64> {
    let zero = czero(&z[0][0].re);
    let adj = |i: usize, j: usize| -> C<T> {
        if nm == 2 {
            return match (i, j) {
                (0, 0) => z[1][1].clone(),
                (1, 1) => z[0][0].clone(),
                _ => csub(&zero, &z[j][i]),
            };
        }
        let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
        csub(&cmul(&z[a][c], &z[b][d]), &cmul(&z[a][d], &z[b][c]))
    };
    let mut delta = zero.clone();
    for j in 0..nm {
        delta = cadd(&delta, &cmul(&z[0][j], &adj(j, 0)));
    }
    let mut largest = 0.0f64;
    for j in 0..nm {
        let mut value = zero.clone();
        for m in 0..nm {
            value = cadd(&value, &cmul(&adj(j, m), &f[m]));
        }
        largest = largest.max(cnorm2(&value).to_f64().sqrt());
    }
    let w = largest / cnorm2(&delta).to_f64().sqrt();
    (largest > 0.0 && w.is_finite()).then_some(w)
}

/// A priori estimate of the alternative PV reduction at coefficient rank
/// `rank` from COLLIER's `w` (`gram_indicator`, its CalcDred w_pv2) and the
/// cancellation `v` of the modified Cayley determinant
/// (`cayley_cancellation`): 2^-53 (max(1, w) max(1, v))^ceil((rank + 1) / 2).
pub(super) fn alternative_pv_estimate(w: f64, v: f64, rank: u16) -> f64 {
    2f64.powi(-53) * (w.max(1.0) * v.max(1.0)).powi((i32::from(rank) + 1) / 2)
}

/// Further passes an expansion needs from its observed convergence: its last
/// pass changed it by `change` (relative) and the passes shrink by `ratio`
/// (< 1), so `ln(EXPANSION_TOLERANCE / change) / ln(ratio)` more passes reach
/// the tolerance. The rate is extrapolated over at most the `observed` passes
/// it was measured on; `None` beyond them, or when the expansion does not
/// converge.
pub(super) fn predicted_passes(change: f64, ratio: f64, observed: u16) -> Option<u16> {
    if !(change.is_finite() && change > EXPANSION_TOLERANCE && ratio > 0.0 && ratio < 1.0) {
        return None;
    }
    let passes = ((EXPANSION_TOLERANCE / change).ln() / ratio.ln()).ceil();
    (passes >= 1.0 && passes <= f64::from(observed)).then_some(passes as u16)
}

/// Cancellation in the modified Cayley determinant of a box or triangle,
/// `det X = 2 m_0^2 Delta - sum_ab f_a Zadj_ab f_b`: the ratio of the sum of
/// the terms' sizes to `|det X|` (`None` if it vanishes or is not finite).
/// `det X` is formed with relative error `~ 2^-53` times this ratio (from its
/// inputs' rounding, which every relabelled frame shares, and its own), and
/// the alternative PV reduction divides by it at every rank step, so in
/// COLLIER's worst case its rounding grows like this ratio to the power
/// `ceil(r/2)` as well (`alternative_pv_estimate`; `gram_indicator`'s `w`
/// does not see it).
pub(super) fn cayley_cancellation<T: NativeFloat>(z: &[[C<T>; 3]; 3], f: &[C<T>; 3], m02: &C<T>, nm: usize) -> Option<f64> {
    let zero = czero(&z[0][0].re);
    let size = |x: &C<T>| cnorm2(x).to_f64().sqrt();
    let adj = |i: usize, j: usize| -> C<T> {
        if nm == 2 {
            return match (i, j) {
                (0, 0) => z[1][1].clone(),
                (1, 1) => z[0][0].clone(),
                _ => csub(&zero, &z[j][i]),
            };
        }
        let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
        csub(&cmul(&z[a][c], &z[b][d]), &cmul(&z[a][d], &z[b][c]))
    };
    let mut delta = zero.clone();
    for j in 0..nm {
        delta = cadd(&delta, &cmul(&z[0][j], &adj(j, 0)));
    }
    let two = z[0][0].re.from_usize(2);
    let mut det_x = cmul(&cscale(m02, &two), &delta);
    let mut terms = size(&det_x);
    for a in 0..nm {
        for b in 0..nm {
            let term = cmul(&cmul(&f[a], &f[b]), &adj(a, b));
            terms += size(&term);
            det_x = csub(&det_x, &term);
        }
    }
    let ratio = terms / size(&det_x);
    ratio.is_finite().then_some(ratio)
}

/// COLLIER's a priori convergence factors of the small-Gram expansion and
/// of its variant for a small modified Cayley determinant (`fac_g` and
/// `fac_gy` of its CalcCgn/CalcDgn, CalcCgy/CalcDgy):
/// fac_g = |Delta| / max_j |(Zadj f)_j| * max(1, max |Xtilde| / (2 max_j |(Zadj f)_j|)),
/// fac_gy = max(max_j |(Zadj f)_j|, |Delta|) / max |Xtilde| * max(1, max |f| / max |Z|).
pub(super) fn gram_factors<T: NativeFloat>(z: &[[C<T>; 3]; 3], f: &[C<T>; 3], m02: &C<T>, nm: usize) -> Option<(f64, f64)> {
    let zero = czero(&z[0][0].re);
    let size = |x: &C<T>| cnorm2(x).to_f64().sqrt();
    let adj = |i: usize, j: usize| -> C<T> {
        if nm == 2 {
            return match (i, j) {
                (0, 0) => z[1][1].clone(),
                (1, 1) => z[0][0].clone(),
                _ => csub(&zero, &z[j][i]),
            };
        }
        let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
        csub(&cmul(&z[a][c], &z[b][d]), &cmul(&z[a][d], &z[b][c]))
    };
    let mut delta = zero.clone();
    for j in 0..nm {
        delta = cadd(&delta, &cmul(&z[0][j], &adj(j, 0)));
    }
    let mut largest = 0.0f64;
    for j in 0..nm {
        let mut value = zero.clone();
        for m in 0..nm {
            value = cadd(&value, &cmul(&adj(j, m), &f[m]));
        }
        largest = largest.max(size(&value));
    }
    let two = z[0][0].re.from_usize(2);
    let mut xtilde = 0.0f64;
    for a in 0..nm {
        for b in 0..nm {
            let mut value = cscale(&cmul(m02, &adj(a, b)), &two);
            for n1 in 0..nm {
                for m1 in 0..nm {
                    if let Some(w) = second_adjugate(z, nm, a, n1, b, m1) {
                        value = cadd(&value, &cmul(&w, &cmul(&f[n1], &f[m1])));
                    }
                }
            }
            xtilde = xtilde.max(size(&value));
        }
    }
    let max_z = (0..nm).flat_map(|i| (0..nm).map(move |j| (i, j))).map(|(i, j)| size(&z[i][j])).fold(0.0f64, f64::max);
    let max_f = (0..nm).map(|i| size(&f[i])).fold(0.0f64, f64::max);
    let fac_g = size(&delta) / largest * (xtilde / (2.0 * largest)).max(1.0);
    let fac_gy = largest.max(size(&delta)) / xtilde * (max_f / max_z).max(1.0);
    (largest > 0.0 && fac_g.is_finite() && fac_gy.is_finite()).then_some((fac_g, fac_gy))
}

/// Nested triangles (those of an auxiliary box executor) whose small-Gram
/// expansion and small-Cayley variant both stay above these bounds (log2 of
/// COLLIER's convergence factor to the power of the largest boost's passes)
/// go straight to the alternative PV reduction. On all stability-study
/// points every binary64 array, status and estimate is unchanged by the skip
/// up to (-24, -30); (-30, -30) moves two heptagon estimates.
const NESTED_HOPELESS_LOG2: (f64, f64) = (-18.0, -30.0);

// Largest a priori error estimate of the alternative PV stage (the binary64
// retries accept rotated copies within 2^-36) and largest accepted relative
// change of the last expansion pass (2^-30).
use crate::direct::{ALTERNATIVE_PV_LIMIT, EXPANSION_TOLERANCE};

fn rank_of(key: &Key) -> u16 {
    2 * key.0 + key.1.iter().sum::<u16>()
}

/// Every `kappa` with `|kappa| = total` over `nm` (2 or 3) momenta, padded
/// with zeros to three components.
fn compositions(total: u16, nm: usize) -> Vec<[u16; 3]> {
    let t = usize::from(total);
    let mut out = Vec::with_capacity(if nm == 2 { t + 1 } else { (t + 1) * (t + 2) / 2 });
    out.extend(composition_iter(total, nm));
    out
}

/// [compositions] in the same order, without a vector.
fn composition_iter(total: u16, nm: usize) -> impl Iterator<Item = [u16; 3]> {
    (0..=total).flat_map(move |a| {
        let last = if nm == 2 { 0 } else { total - a };
        (0..=last).map(move |b| if nm == 2 { [a, total - a, 0] } else { [a, b, total - a - b] })
    })
}

/// `-sum_{m,n} eps_{ikm} eps_{jln} Z_mn`: the second adjugate of a 3x3
/// matrix in the paper's convention (its relations (Zadjadjrelation0-2)
/// fix the sign; see the test below).
fn second_adjugate<T: NativeFloat>(z: &[[C<T>; 3]; 3], nm: usize, i: usize, k: usize, j: usize, l: usize) -> Option<C<T>> {
    if i == k || j == l {
        return None;
    }
    if nm == 2 {
        // The paper's N = 2 case: d_il d_kj - d_ij d_kl (i != k, j != l).
        let one = C::new(z[0][0].re.one(), z[0][0].re.zero());
        return Some(if i == l && k == j { one } else { csub(&czero(&one.re), &one) });
    }
    let sign = |a: usize, b: usize| -> bool {
        // eps_{a b c} with c the third index: +1 for cyclic (0,1),(1,2),(2,0).
        (b + 3 - a) % 3 == 1
    };
    let m = 3 - i - k;
    let n = 3 - j - l;
    let value = z[m][n].clone();
    Some(if sign(i, k) == sign(j, l) { csub(&czero(&value.re), &value) } else { value })
}

impl<T: NativeFloat> DirectExecutor<T>
where
    Complex<T>: EvaluationDomain,
{
    /// COLLIER's convergence factors (`gram_factors`) of PV subset `id` of
    /// this lane's plan.
    fn gram_factors_of(&self, id: usize) -> Option<(f64, f64)> {
        let subset = &self.plan.subsets[id];
        let nm = subset.members.len().checked_sub(1).filter(|nm| (2..=3).contains(nm))?;
        let n = self.plan.settings.propagators;
        let members = &subset.members;
        let origin = members[0];
        let zero = czero(&self.prototype);
        let dist = |i: usize, j: usize| &self.distances[i * n + j];
        let z: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i >= nm || j >= nm {
                    return zero.clone();
                }
                let (a, b) = (members[i + 1], members[j + 1]);
                csub(&cadd(dist(origin, a), dist(origin, b)), dist(a, b))
            })
        });
        let geometry = subset.pv?;
        let f: [C<T>; 3] = std::array::from_fn(|i| {
            if i < nm { self.geometry[geometry.f as usize + i].clone() } else { zero.clone() }
        });
        gram_factors(&z, &f, &self.masses[origin], nm)
    }

    /// A priori estimate of the alternative PV reduction of PV subset `id` at
    /// coefficient rank `rank` (`set_alternative_first`, `enable_alternative_pv`):
    /// COLLIER's w_pv2 with the modified Cayley determinant's cancellation `v`
    /// (`cayley_cancellation`), 2^-53 (max(1, w) max(1, v))^ceil((rank + 1) / 2),
    /// w = max_j |(Zadj f)_j| / |Delta| (`alternative_pv_estimate`). (w_pv2
    /// alone admits near-forward gg -> HH boxes whose alternative PV keeps
    /// only E/A ~1e-4 at rank 6 while the expansions refuse them; there
    /// v ~ 1e3.)
    fn alternative_estimate_of(&self, id: usize, rank: u16) -> Option<f64> {
        let subset = &self.plan.subsets[id];
        let nm = subset.members.len().checked_sub(1).filter(|nm| (2..=3).contains(nm))?;
        let n = self.plan.settings.propagators;
        let members = &subset.members;
        let origin = members[0];
        let zero = czero(&self.prototype);
        let dist = |i: usize, j: usize| &self.distances[i * n + j];
        let z: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i >= nm || j >= nm {
                    return zero.clone();
                }
                let (a, b) = (members[i + 1], members[j + 1]);
                csub(&cadd(dist(origin, a), dist(origin, b)), dist(a, b))
            })
        });
        let geometry = subset.pv?;
        let f: [C<T>; 3] = std::array::from_fn(|i| {
            if i < nm { self.geometry[geometry.f as usize + i].clone() } else { zero.clone() }
        });
        let w = gram_indicator(&z, &f, nm)?;
        let v = cayley_cancellation(&z, &f, &self.masses[origin], nm)?;
        Some(alternative_pv_estimate(w, v, rank))
    }

    /// Evaluate every expansion box of this lane by its auxiliary executor
    /// and inject its coefficients.
    pub(super) fn inject_gram(&mut self) -> Result<(), String> {
        let plan = self.plan.clone();
        let boxes = self.gram_external.clone();
        let mut gram = self.gram.take().ok_or("no small-Gram expansion")?;
        let outcome = (|| -> Result<(), String> {
            for &id in &boxes {
                let subset = &plan.subsets[id];
                let size = subset.members.len();
                let rank = self.injection_rank(size, subset.coefficient_rank.max(0) as u16);
                let count = plan.coefficient_counts[id] as usize;
                let start = subset.master.ok_or("small-Gram expansion: no master")? as usize;
                // The alternative PV alone (`enable_alternative_pv`) only
                // where its a priori estimate with det X's cancellation is
                // within `ALTERNATIVE_PV_LIMIT`; its executor checks w_pv2
                // alone, which the cancellation escapes.
                if gram.alternative
                    && let Some(estimate) = self.alternative_estimate_of(id, rank).filter(|&e| e > ALTERNATIVE_PV_LIMIT)
                {
                    return Err(format!(
                        "small-Gram expansion of {:?}: alternative PV: estimated error {estimate:e}",
                        subset.members
                    ));
                }
                // A nested triangle whose expansion cannot converge at the
                // largest boost goes straight to the alternative PV at the
                // first boost (same bubbles and gate as its expansion; a
                // refused lane still tries the larger boost's expansion).
                let largest = *gram.boosts.last().unwrap_or(&FIRST_BOOST);
                let skip = !gram.alternative
                    && self.gram_top.is_some()
                    && size == 3
                    && self.gram_factors_of(id).is_some_and(|(fac_g, fac_gy)| {
                        // Passes at the largest boost: largest - 1 of the
                        // expansion, (largest - 2) / 2 of the variant.
                        let (bound_g, bound_gy) = NESTED_HOPELESS_LOG2;
                        f64::from(largest - 1) * fac_g.log2() > bound_g
                            && f64::from((largest - 2) / 2) * fac_gy.log2() > bound_gy
                    });
                // The expansion's settings: boosts, alternative PV only, the
                // skip, the auxiliary gate and this executor's gate (its
                // default), then the alternative-first limit, the inputs'
                // precision (the auxiliary executor's lightlike tolerance) and
                // whether the data-derived further boost applies (a top-level
                // executor, or an auxiliary one of its parent's last attempt).
                let mut config: Vec<u64> = gram.boosts.iter().map(|&b| u64::from(b)).collect();
                config.push(u64::from(gram.alternative));
                config.push(u64::from(skip));
                // Both gates' binary64 bits are exact: `subset_key` keys only
                // binary64 executors (`set_shared_subsets`).
                config.push(gram.aux_threshold.as_ref().map_or(u64::MAX, |t| t.to_f64().to_bits()));
                config.push(self.stability_threshold.to_f64().to_bits());
                // The alternative PV first where its a priori estimate is
                // small enough (top-level boxes and triangles).
                let first = gram.alternative_first.filter(|_| !gram.alternative && self.gram_top.is_none()).filter(|&limit| {
                    self.alternative_estimate_of(id, rank).is_some_and(|estimate| estimate <= limit)
                });
                config.push(first.map_or(u64::MAX, f64::to_bits));
                config.push(u64::from(self.input_bits));
                config.push(u64::from(self.gram_top.is_none() || self.gram_final));
                let key = self.subset_key(2, &subset.members, rank, 0, &config);
                if let Some(shared) = self.shared_get(key.as_ref(), GRAM_EXPANSION, &subset.members) {
                    self.shared_write(start, count, &shared?)?;
                    continue;
                }
                let outcome = (|| -> Result<Vec<[f64; 8]>, String> {
                let mut full = None;
                let mut failure = String::new();
                let mut done = false;
                // Boost 0 stands for the alternative-first executor (boost 1,
                // alternative PV only).
                let mut attempts: Vec<(u16, bool)> = first
                    .map(|_| (0, false))
                    .into_iter()
                    .chain(gram.boosts.iter().enumerate().map(|(i, &boost)| (boost, skip && i == 0)))
                    .collect();
                let mut next = 0;
                while let Some(&(boost, direct)) = attempts.get(next) {
                next += 1;
                let alternative_only = gram.alternative || boost == 0;
                let executor = match gram.executors.entry((size, rank, boost, direct)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let aux = DirectPlanData::with_top_child_boost(self.subset_settings(size, rank), boost.max(1))
                            .map_err(|e| e.to_string())?;
                        let mut executor = DirectExecutor::<T>::new(
                            std::sync::Arc::new(aux),
                            self.prototype.clone(),
                            DirectMasters::Native,
                        )
                        .map_err(|e| e.to_string())?;
                        executor.set_input_precision(self.input_bits);
                        // Its triangles carry `boost` extra ranks whose errors
                        // enter only Delta-suppressed terms (COLLIER relaxes them
                        // likewise); the caller's a posteriori check decides.
                        executor.set_stability_threshold(
                            gram.aux_threshold.clone().unwrap_or_else(|| self.stability_threshold.clone() * &self.stability_threshold),
                        );
                        executor.gram_top = Some(rank);
                        executor.gram_alt_only = alternative_only;
                        executor.gram_alt_direct = direct;
                        executor.gram_final = !alternative_only && !direct && boost >= largest;
                        executor.gram_extra_ranks = if boost == 0 {
                            0
                        } else {
                            gram.boosts.iter().find(|&&next| next > boost).map_or(0, |&next| next - boost)
                        };
                        // (Bounded: the nested triangles' plans stay at rank <= 16.)
                        if let Some(threshold) = gram.aux_threshold.as_ref().filter(|_| rank + boost <= 17) {
                            // COLLIER chooses per C function too: the auxiliary
                            // box's own triangles by the same choice (nested).
                            let boost = *gram.boosts.last().unwrap_or(&FIRST_BOOST);
                            executor.enable_gram_expansion_everywhere(boost, threshold.clone());
                        }
                        entry.insert(executor)
                    }
                };
                let values = self.subset_row(&subset.members, |z| z.clone());
                // Nested expansions of the auxiliary box share too.
                executor.set_shared_subsets(self.shared.clone());
                executor.gram_wanted_ranks = None;
                let result = executor
                    .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values })
                    .map_err(|e| e.to_string())?;
                if result.statuses[0] != LaneStatus::Success {
                    failure = format!("{GRAM_EXPANSION} of {:?}: {:?}", subset.members, result.statuses[0]);
                    // The largest boost failed with expansions converging too
                    // slowly for its passes and the alternative PV refused:
                    // an attempt at the boost the observed rate predicts, and
                    // so on, up to the largest boost's passes again (the rate
                    // is extrapolated no further than the range it was
                    // measured over). The workspace's own boxes and triangles,
                    // and those of an auxiliary executor of its parent's last
                    // attempt: a lane served before is served as before.
                    if (self.gram_top.is_none() || self.gram_final)
                        && !gram.alternative
                        && !direct
                        && boost >= largest
                        && next == attempts.len()
                        && let Some(ranks) = executor.gram_wanted_ranks.take()
                        && u32::from(boost) + u32::from(ranks) <= 2 * u32::from(largest)
                    {
                        attempts.push((boost + ranks, false));
                    }
                    continue;
                }
                let coefficients = executor.top_coefficients().ok_or("small-Gram expansion: no coefficients")?;
                if coefficients.len() < count {
                    return Err("small-Gram expansion: coefficient layouts differ".into());
                }
                full = Some(Self::shared_values(&coefficients));
                done = true;
                break;
                }
                if !done {
                    return Err(failure);
                }
                full.ok_or_else(|| "small-Gram expansion: no coefficients".to_string())
                })();
                self.shared_put(key, &outcome, GRAM_EXPANSION, &subset.members);
                self.shared_write(start, count, &outcome?)?;
            }
            Ok(())
        })();
        self.gram = Some(gram);
        outcome
    }

    /// Fill the top box's coefficients up to rank `target` from its
    /// triangles by the small-Gram expansion (auxiliary executors).
    pub(super) fn expand_top(&mut self, target: u16) -> Result<(), String> {
        let tables = self.top_tables();
        let pool = RefCell::new(std::mem::take(&mut self.gram_pool));
        let outcome = self.expand_top_with(target, &tables, &pool);
        self.gram_pool = pool.into_inner();
        outcome
    }

    /// The [`TopTables`] of the current plan variant.
    fn top_tables(&mut self) -> Arc<TopTables> {
        if let Some((_, tables)) = self.gram_tables.iter().find(|(plan, _)| Arc::ptr_eq(plan, &self.plan)) {
            return tables.clone();
        }
        let tables = Arc::new(TopTables::new(&self.plan));
        self.gram_tables.push((self.plan.clone(), tables.clone()));
        tables
    }

    fn expand_top_with(&mut self, target: u16, tables: &TopTables, pool: &RefCell<CoefficientPool<T>>) -> Result<(), String> {
        let plan = self.plan.clone();
        let id = plan.subsets.len() - 1;
        let subset = &plan.subsets[id];
        if subset.kind != SubsetKind::Pv || !(3..=4).contains(&subset.members.len()) {
            return Err("small-Gram expansion: the top subset is not a PV triangle or box".into());
        }
        let nm = subset.members.len() - 1;
        let n = plan.settings.propagators;
        let members = &subset.members;
        let origin = members[0];
        let zero = czero(&self.prototype);
        let dist = |i: usize, j: usize| self.distances[i * n + j].clone();
        let z: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i >= nm || j >= nm {
                    return zero.clone();
                }
                let (a, b) = (members[i + 1], members[j + 1]);
                csub(&cadd(&dist(origin, a), &dist(origin, b)), &dist(a, b))
            })
        });
        let geometry = subset.pv.unwrap();
        let f: [C<T>; 3] = std::array::from_fn(|i| {
            if i < nm { self.geometry[geometry.f as usize + i].clone() } else { zero.clone() }
        });
        let m02 = self.masses[origin].clone();
        // Adjugate (Z symmetric) and determinant.
        let adj: [[C<T>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if nm == 2 {
                    return match (i, j) {
                        (0, 0) => z[1][1].clone(),
                        (1, 1) => z[0][0].clone(),
                        (0, 1) | (1, 0) => csub(&zero, &z[j][i]),
                        _ => zero.clone(),
                    };
                }
                let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
                csub(&cmul(&z[a][c], &z[b][d]), &cmul(&z[a][d], &z[b][c]))
            })
        });
        let mut delta = zero.clone();
        for j in 0..nm {
            delta = cadd(&delta, &cmul(&z[0][j], &adj[j][0]));
        }
        let adjf: [C<T>; 3] = std::array::from_fn(|j| {
            (0..nm).fold(zero.clone(), |acc, m| cadd(&acc, &cmul(&adj[j][m], &f[m])))
        });
        let j = (0..nm)
            .max_by(|&a, &b| cnorm2(&adjf[a]).partial_cmp(&cnorm2(&adjf[b])).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let (mut k, mut l) = (0, 0);
        for a in 0..nm {
            for b in a..nm {
                if cnorm2(&adj[a][b]) > cnorm2(&adj[k][l]) {
                    (k, l) = (a, b);
                }
            }
        }
        if cnorm2(&adjf[j]).is_zero() || cnorm2(&adj[k][l]).is_zero() {
            return Err("small-Gram expansion: vanishing adjugates".into());
        }
        let inv_adjf = cinv(&adjf[j]);
        let two = self.prototype.from_usize(2);
        let four = self.prototype.from_usize(4);
        let inv_two_adj = cinv(&cscale(&adj[k][l], &two));
        // Second adjugates Zadj2_(kn)(lm) and Xtilde_kl.
        let mut pairs: Vec<(usize, usize, C<T>)> = Vec::new();
        for a in 0..nm {
            for b in 0..nm {
                if let Some(w) = second_adjugate(&z, nm, k, a, l, b) {
                    pairs.push((a, b, w));
                }
            }
        }
        let mut xtilde = cscale(&cmul(&m02, &adj[k][l]), &two);
        for (a, b, w) in &pairs {
            xtilde = cadd(&xtilde, &cmul(w, &cmul(&f[*a], &f[*b])));
        }

        // Triangle values: the pinch of member `i + 1` and of the origin,
        // through the plan's tables.
        let layout = tables.layout.as_ref().map_err(|error| error.to_string())?;
        let child_rank = layout.child_rank;
        if child_rank < target {
            return Err("small-Gram expansion: triangles below the requested rank".into());
        }
        // Keys of the expansion (its steps reach one rank above the
        // triangles).
        let grading = cached_grading(nm, child_rank + 2);
        let values = &self.values;
        let value = |index: u32| -> Result<&L4<T>, String> {
            if index >= UNUSED {
                return Err("small-Gram expansion: triangle coefficient outside its layout".to_string());
            }
            Ok(&values[index as usize])
        };
        // Shifted pinch-zero values C0(lambda, kappa) in the box basis
        // (Denner, Eq. (2.9); as the PV program's `spz` instructions):
        // (-1)^k0 sum over k0 = a + b + c of the trinomial k0!/(a! b! c!)
        // times the pinch's C0(lambda, [k1 + b, k2 + c]), computed by the
        // trinomial recursion spz(k) = -[spz(k - e0) + spz(k - e0 + e1)
        // + spz(k - e0 + e2)] from spz(0, k1, k2) = C0(lambda, [k1, k2]).
        // The keys of a block come in increasing kappa_0 (as `compositions`
        // lists them), so every term below is already in place; the terms
        // are read through the grading's neighbour tables.
        // The alternative PV alone (the executors that skip the expansions)
        // reads them below rank `target` only.
        let built = if self.gram_alt_only || self.gram_alt_direct { target } else { child_rank };
        let mut spz: Coefficients<T> = Coefficients::pooled(pool, grading.end(built));
        for r in 0..=built {
            for lambda in 0..=r / 2 {
                for index in grading.block(r, lambda) {
                    let kappa = grading.key(index).1;
                    let shifted = if kappa[0] == 0 {
                        value(layout.spz_source[index])?.clone()
                    } else {
                        let lower = grading.minus(index as u32, 0);
                        let mut acc = lzero(&zero);
                        for i in 0..nm {
                            let at = if i > 0 { grading.plus(lower, i) } else { lower };
                            let term = spz.at(at as usize).ok_or("small-Gram expansion: shifted C0 missing")?;
                            for slot in 0..4 {
                                acc[slot] = csub(&acc[slot], &term[slot]);
                            }
                        }
                        acc
                    };
                    spz.insert_at(&grading, index, shifted, &zero);
                }
            }
        }
        // Shat_n(lambda, kappa) = triangle without member n+1 (if kappa_n = 0)
        // minus the shifted pinch zero, for every key of `spz`.
        let mut shat: Vec<Coefficients<T>> = (0..nm).map(|_| Coefficients::pooled(pool, grading.end(built))).collect();
        for r in 0..=built {
            for lambda in 0..=r / 2 {
                for index in grading.block(r, lambda) {
                    let kappa = grading.key(index).1;
                    let Some(c0) = spz.at(index) else { continue };
                    for m in 0..nm {
                        let mut out: L4<T> = std::array::from_fn(|slot| csub(&zero, &c0[slot]));
                        if kappa[m] == 0 {
                            let source = value(layout.shat_source[index][m])?;
                            for slot in 0..4 {
                                out[slot] = cadd(&out[slot], &source[slot]);
                            }
                        }
                        shat[m].insert_at(&grading, index, out, &zero);
                    }
                }
            }
        }
        // A source missing above `built` fails as building every rank would.
        if layout.missing_rank != u16::MAX && layout.missing_rank > built {
            return Err("small-Gram expansion: triangle coefficient outside its layout".into());
        }
        let shat_of = |m: usize, lambda: u16, kappa: [u16; 3]| -> Result<&L4<T>, String> {
            shat[m].get(&grading, (lambda, kappa)).ok_or_else(|| "small-Gram expansion: Shat outside the triangle ranks".to_string())
        };
        let shat_at = |m: usize, index: u32| -> Result<&L4<T>, String> {
            shat[m].at(index as usize).ok_or_else(|| "small-Gram expansion: Shat outside the triangle ranks".to_string())
        };
        let d0 = self.values[subset.master.unwrap() as usize].clone();
        let real = |x: usize| C::new(self.prototype.from_usize(x), self.prototype.zero());
        let add = |kappa: [u16; 3], i: usize| -> [u16; 3] {
            let mut out = kappa;
            out[i] += 1;
            out
        };
        let sub = |kappa: [u16; 3], i: usize| -> Option<[u16; 3]> {
            (kappa[i] > 0).then(|| {
                let mut out = kappa;
                out[i] -= 1;
                out
            })
        };

        // The expansion order by order in Delta: the coefficients are
        // D_0 + D_1 + ..., D_s = O(Delta^s). D_0 carries the triangles (C0,
        // Shat, D0) and no Delta term; D_s (s > 0) only the homogeneous terms
        // of the recursion and -Delta D_{s-1}. Step q of an order gives its
        // metric coefficients of rank q and momentum coefficients of rank
        // q - 1 and needs the previous order through step q + 1, so orders
        // 0..=S take order s through step target + 1 + S - s: their sum is
        // pass S of the fixed-point form (each pass recomputing everything,
        // the previous pass in the Delta terms) and D_S that pass's change.
        // Order by order the triangles enter once, and the first orders,
        // which show whether the expansion converges, need the lowest ranks.
        let passes = child_rank - target;
        let extra = i32::from(self.gram_extra_ranks);
        let neg_delta = csub(&zero, &delta);
        let two_adj = cscale(&adj[k][l], &two);
        let neg_adj = csub(&zero, &adj[k][l]);
        // On a lane without IR poles every value's IR slots are zero, so their
        // products (with finite factors) only add zeros: skip them, as the
        // recurrences do. In binary64 the accumulators, which start at +0,
        // stay +0 bit for bit either way.
        let full = !self.ir_free;
        let fma = |acc: &mut L4<T>, factor: &C<T>, value: &L4<T>| {
            if full { lfma(acc, factor, value) } else { lfma_slots::<T, false>(acc, factor, value) }
        };
        // The recursion's scalar factors, by multiplicity, built once by the
        // operations each key used to repeat (so the same bits).
        let top = usize::from(grading.max_rank) + 1;
        let real2: Vec<C<T>> = (0..=top).map(|x| real(2 * x)).collect();
        let two_x: Vec<T> = (0..=top).map(|x| two.clone() * &self.prototype.from_usize(x)).collect();
        let neg_f2: Vec<Vec<C<T>>> =
            (0..nm).map(|b| two_x.iter().map(|x| csub(&zero, &cscale(&f[b], x))).collect()).collect();
        let neg_adj2: Vec<Vec<C<T>>> =
            (0..nm).map(|m| two_x.iter().map(|x| csub(&zero, &cscale(&adj[j][m], x))).collect()).collect();
        let neg_four: Vec<C<T>> = (0..=top * top).map(|x| csub(&zero, &cscale(&real(x), &four))).collect();
        let neg_w: Vec<C<T>> = pairs.iter().map(|(_, _, w)| csub(&zero, w)).collect();
        // Divisor (a - 2 eps) of the metric keys, a = 2 (lambda + |kappa|) + 4 - N,
        // by lambda + |kappa| - 1 (lambda >= 1).
        let dims: Vec<[T; 3]> = (1..=top)
            .map(|s| dimension_powers(&(self.prototype.one() / &self.prototype.from_usize(2 * s + 3 - nm))))
            .collect();
        // Only the coefficients the results depend on.
        let structure: Vec<(usize, usize)> = pairs.iter().map(|&(a, b, _)| (a, b)).collect();
        let need = expansion_need(&grading, target, passes, (j, k, l), &structure);
        // Each step walks its keys by index and reads every term through the
        // grading's neighbour tables (the order of the keys and of the terms is
        // that of the recursion above).
        let what = "small-Gram expansion";
        let step = |cur: &mut Coefficients<T>, prev: Option<&Coefficients<T>>, need: &[bool], q: u16| -> Result<(), String> {
            let first = prev.is_none();
            for lambda in (1..=q / 2).rev() {
                for index in grading.block(q, lambda) {
                    if !need[index] {
                        continue;
                    }
                    let kappa = grading.key(index).1;
                    let here = index as u32;
                    // (lambda - 1, kappa)
                    let below = grading.lower(here);
                    let mut acc = lzero(&zero);
                    if let Some(prev) = prev {
                        let at = grading.plus(grading.plus(below, k), l);
                        fma(&mut acc, &neg_delta, prev.need_at(at, || (lambda - 1, add(add(kappa, k), l)), what)?);
                    } else {
                        fma(&mut acc, &two_adj, spz.at(below as usize).ok_or("small-Gram expansion: C0 outside the triangle ranks")?);
                    }
                    fma(&mut acc, &xtilde, cur.need_at(below, || (lambda - 1, kappa), what)?);
                    if first {
                        for m in 0..nm {
                            fma(&mut acc, &adj[m][l], shat_at(m, grading.plus(below, k))?);
                            fma(&mut acc, &neg_adj, shat_at(m, grading.plus(below, m))?);
                        }
                    }
                    for (&(a, b, _), neg_w) in pairs.iter().zip(&neg_w) {
                        let (ka, kb) = (usize::from(kappa[a]), usize::from(kappa[b]));
                        let mut bracket = lzero(&zero);
                        if first {
                            fma(&mut bracket, &f[a], shat_at(b, below)?);
                        }
                        if ka > 0 {
                            let less = grading.minus(here, a);
                            if first {
                                fma(&mut bracket, &real2[ka], shat_at(b, less)?);
                            }
                            fma(&mut bracket, &neg_f2[b][ka], cur.need_at(less, || (lambda, sub(kappa, a).unwrap()), what)?);
                        }
                        if kb > 0 {
                            let less = grading.minus(here, b);
                            fma(&mut bracket, &neg_f2[a][kb], cur.need_at(less, || (lambda, sub(kappa, b).unwrap()), what)?);
                        }
                        let multiplicity = if a == b { ka * ka.saturating_sub(1) } else { ka * kb };
                        if multiplicity > 0 {
                            let less = grading.higher(grading.minus(grading.minus(here, a), b));
                            fma(&mut bracket, &neg_four[multiplicity], cur.need_at(less, || (lambda + 1, sub(sub(kappa, a).unwrap(), b).unwrap()), what)?);
                        }
                        fma(&mut acc, neg_w, &bracket);
                    }
                    let mut out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_two_adj));
                    div_dimension_by(&mut out, &dims[usize::from(lambda) + kappa.iter().map(|&x| usize::from(x)).sum::<usize>() - 1]);
                    cur.insert_at(&grading, index, out, &zero);
                }
            }
            if q >= 2 {
                for index in grading.block(q - 1, 0) {
                    if !need[index] {
                        continue;
                    }
                    let kappa = grading.key(index).1;
                    let here = index as u32;
                    let mut acc = lzero(&zero);
                    for m in 0..nm {
                        if first {
                            fma(&mut acc, &adj[j][m], shat_at(m, here)?);
                        }
                        if kappa[m] > 0 {
                            let less = grading.higher(grading.minus(here, m));
                            fma(&mut acc, &neg_adj2[m][usize::from(kappa[m])], cur.need_at(less, || (1, sub(kappa, m).unwrap()), what)?);
                        }
                    }
                    if let Some(prev) = prev {
                        fma(&mut acc, &neg_delta, prev.need_at(grading.plus(here, j), || (0, add(kappa, j)), what)?);
                    }
                    let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_adjf));
                    cur.insert_at(&grading, index, out, &zero);
                }
            }
            Ok(())
        };
        let mut orders: Vec<Coefficients<T>> = Vec::with_capacity(usize::from(passes) + 1);
        // Steps completed per order (step 1 is empty).
        let mut steps: Vec<u16> = Vec::with_capacity(usize::from(passes) + 1);
        // Sum of the accepted orders, ranks <= target.
        let mut total: Coefficients<T> = Coefficients::pooled(pool, 0);
        let mut accepted = false;
        let mut last_change = f64::INFINITY;
        // Observed change ratio per order (convergence rate), for deciding
        // whether a larger boost would converge; `best` the smallest one from
        // the third order on (the first ratio is often much smaller).
        let mut ratio = 1.0f64;
        let mut best = 1.0f64;
        // Last accepted order, and whether the expansion diverged.
        let mut reached = 0u16;
        let mut diverged = false;
        for order in 0..=passes {
            if self.gram_alt_only || self.gram_alt_direct {
                // Alternative PV only: no expansion orders.
                break;
            }
            // Order `order` runs through step target + 1 + passes - order.
            let mut fresh: Coefficients<T> = Coefficients::pooled(pool, grading.end(target + 1 + passes - order));
            fresh.insert(&grading, (0, [0, 0, 0]), if order == 0 { d0.clone() } else { lzero(&zero) }, &zero)?;
            orders.push(fresh);
            steps.push(1);
            for s in 0..=usize::from(order) {
                let through = target + 1 + order - s as u16;
                let (lower, upper) = orders.split_at_mut(s);
                for q in steps[s] + 1..=through {
                    step(&mut upper[0], lower.last(), &need[s], q)?;
                }
                steps[s] = through;
            }
            let latest = &orders[usize::from(order)];
            if order == 0 {
                std::mem::replace(&mut total, latest.prefix_pooled(pool, grading.end(target))).recycle(pool);
                if passes == 0 {
                    accepted = true;
                    last_change = 0.0;
                }
                continue;
            }
            // Largest relative change per rank, ranks <= target.
            let mut next = total.clone_pooled(pool);
            let mut change = 0.0f64;
            for r in 1..=target {
                let (mut diff, mut size) = (0.0f64, 0.0f64);
                for index in grading.range(r) {
                    let Some(value) = latest.at(index) else { continue };
                    let sum = next.at_mut(index).ok_or("small-Gram expansion: coefficient missing")?;
                    for slot in 0..4 {
                        sum[slot] = cadd(&sum[slot], &value[slot]);
                        diff = diff.max(cnorm2(&value[slot]).to_f64().sqrt());
                        size = size.max(cnorm2(&sum[slot]).to_f64().sqrt());
                    }
                }
                if size > 0.0 {
                    change = change.max(diff / size);
                }
            }
            if !change.is_finite() || (order >= 2 && change > last_change) {
                // Diverging: keep the previous orders.
                diverged = true;
                break;
            }
            if last_change.is_finite() && last_change > 0.0 {
                ratio = (change / last_change).min(1.0);
                if order >= 3 {
                    best = best.min(ratio);
                }
            }
            last_change = change;
            std::mem::replace(&mut total, next).recycle(pool);
            accepted = true;
            reached = order;
            if change <= f64::EPSILON {
                break;
            }
            // Hopeless: from the fourth order on, even the best rate seen
            // since the third keeps the remaining orders and a larger boost
            // above the tolerance (by 2^4; on 27479 expansions of the
            // stability points this stopped 13017, none of which converged).
            if order >= 4 && last_change * best.powi(i32::from(passes - order) + extra) > 16.0 * EXPANSION_TOLERANCE {
                break;
            }
        }
        for order in orders {
            order.recycle(pool);
        }
        // At the largest boost: the further ranks with which the small-Gram
        // expansion's observed rate reaches the tolerance (each rank one
        // more pass; `predicted_passes`), for the caller's data-derived boost.
        let gram_ranks = (self.gram_extra_ranks == 0 && !diverged)
            .then(|| predicted_passes(last_change, ratio, reached))
            .flatten()
            .map(|more| (reached + more + target).saturating_sub(child_rank).max(1));
        let chosen = if accepted {
            Some(total)
        } else {
            total.recycle(pool);
            None
        };
        // Alternative PV reduction (hep-ph/0509141 sect. 3.2, eqs. (PValtrec1),
        // (PValtrec2); COLLIER's CalcDpv2), the last resort when neither
        // expansion converges: the modified Cayley matrix X instead of Z, so
        // metric coefficients divide by Delta once per two ranks and momentum
        // coefficients by det X (Passarino-Veltman divides by the Gram matrix at
        // every rank). Exact up to rounding; triangles of rank target - 1. With
        // r_n = Shat_n(l, k') - 2 k'_n C(l+1, k'-e_n) and Q = 2l + |k'|:
        //   2 Delta ((Q+5-N) - 2 eps) C(l+1, k') = det X C(l, k') + Delta C0(l, k')
        //       - sum_n X~_0n r_n,
        //   det X C(0, k'+e_i) = X~_i0 [2 ((Q+5-N) - 2 eps) C(1, k') - C0(0, k')]
        //       + sum_n X~_in r_n.
        let alt_pv = || -> Result<Coefficients<T>, String> {
            let mut det_x = cmul(&cscale(&m02, &two), &delta);
            for a in 0..nm {
                for b in 0..nm {
                    det_x = csub(&det_x, &cmul(&cmul(&f[a], &f[b]), &adj[a][b]));
                }
            }
            if cnorm2(&det_x).is_zero() || cnorm2(&delta).is_zero() {
                return Err("alternative PV: singular modified Cayley or Gram matrix".into());
            }
            let inv_det_x = cinv(&det_x);
            let inv_two_delta = cinv(&cscale(&delta, &two));
            let xt0 = |n: usize| csub(&zero, &adjf[n]);
            let xtm: Vec<Vec<C<T>>> = (0..nm)
                .map(|i| {
                    (0..nm)
                        .map(|j| {
                            let mut v = cscale(&cmul(&m02, &adj[i][j]), &two);
                            for n1 in 0..nm {
                                for m1 in 0..nm {
                                    if let Some(w) = second_adjugate(&z, nm, i, n1, j, m1) {
                                        v = cadd(&v, &cmul(&w, &cmul(&f[n1], &f[m1])));
                                    }
                                }
                            }
                            v
                        })
                        .collect()
                })
                .collect();
            let mut cur: Coefficients<T> = Coefficients::pooled(pool, grading.end(target + 1));
            cur.insert(&grading, (0, [0, 0, 0]), d0.clone(), &zero)?;
            let mut rhs: Vec<L4<T>> = Vec::with_capacity(nm);
            for q in 0..target {
                for lambda in 0..=q / 2 {
                    for kappa in composition_iter(q - 2 * lambda, nm) {
                        rhs.clear();
                        for n1 in 0..nm {
                            let mut value = shat_of(n1, lambda, kappa)?.clone();
                            if let Some(less) = sub(kappa, n1) {
                                fma(&mut value, &csub(&zero, &real(2 * kappa[n1] as usize)), cur.need(&grading, (lambda + 1, less), "alternative PV")?);
                            }
                            rhs.push(value);
                        }
                        let c0 = spz.get(&grading, (lambda, kappa)).ok_or("alternative PV: C0 outside the triangle ranks")?;
                        let mut acc = lzero(&zero);
                        fma(&mut acc, &det_x, cur.need(&grading, (lambda, kappa), "alternative PV")?);
                        fma(&mut acc, &delta, c0);
                        for n1 in 0..nm {
                            fma(&mut acc, &csub(&zero, &xt0(n1)), &rhs[n1]);
                        }
                        let mut metric: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_two_delta));
                        let a = self.prototype.from_usize(q as usize + 4 - nm);
                        div_dimension(&mut metric, &(self.prototype.one() / &a));
                        cur.insert(&grading, (lambda + 1, kappa), metric.clone(), &zero)?;
                        if lambda == 0 {
                            mul_dimension(&mut metric, &a);
                            let mut bracket = lzero(&zero);
                            fma(&mut bracket, &real(2), &metric);
                            fma(&mut bracket, &csub(&zero, &real(1)), c0);
                            let first = (0..nm).find(|&i| kappa[i] > 0).unwrap_or(nm);
                            for i in 0..nm.min(first + 1) {
                                let mut acc = lzero(&zero);
                                fma(&mut acc, &xt0(i), &bracket);
                                for n1 in 0..nm {
                                    fma(&mut acc, &xtm[i][n1], &rhs[n1]);
                                }
                                let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_det_x));
                                cur.insert(&grading, (0, add(kappa, i)), out, &zero)?;
                            }
                        }
                    }
                }
            }
            Ok(cur)
        };
        // The alternative PV reduction where neither expansion converges
        // (or for a hopeless nested triangle) only where det X, by which it
        // divides once per two ranks, is determined to the expansions'
        // tolerance: its relative rounding at the working precision, 2^-p v
        // with v its cancellation (`cayley_cancellation`), at most
        // `EXPANSION_TOLERANCE`. det X vanishes for IR-divergent boxes and
        // triangles (massless propagators with lightlike legs), where the
        // reduction is 0/0 at any precision. Refused otherwise (a larger boost
        // follows, if any). A relative error e of det X enters the
        // coefficients once, to first order: each rank's momentum
        // coefficients divide by det X (error e), the next rank's metric ones
        // multiply them by det X again (the error cancels). So 2^-p v, not
        // `alternative_pv_estimate`'s worst case v^ceil(r/2) for the
        // numerators' rounding, is the bound on this part, which no relabelled
        // frame sees (det X does not depend on the labelling); the rest the
        // callers' a posteriori check sees, as for the expansions. On the
        // lanes measured with this fallback (pyAmpliCol's method-choice
        // lanes, 2^-53 v up to 1.7e-8) its error stayed within 4 times 2^-53 v
        // where measured.
        let alt_pv_checked = || -> Result<Coefficients<T>, String> {
            let v = cayley_cancellation(&z, &f, &m02, nm).unwrap_or(f64::INFINITY);
            let rounding = 2f64.powi(-(self.prototype.get_precision().min(i32::MAX as u32) as i32)) * v;
            if rounding.is_nan() || rounding > EXPANSION_TOLERANCE {
                return Err(format!("alternative PV: modified Cayley determinant cancels (relative rounding {rounding:e})"));
            }
            alt_pv()
        };
        // A converged expansion whose last pass still changed it by `change`
        // (not machine precision) against the alternative PV reduction: if
        // the two differ by no more than twice `change`, the expansion's own
        // remaining error explains the difference and the alternative PV is
        // the more accurate one (COLLIER keeps the method with the smaller
        // error); otherwise the alternative PV is the one that is off.
        let prefer = |expansion: Coefficients<T>, change: f64| -> Coefficients<T> {
            if change <= 64.0 * f64::EPSILON {
                return expansion;
            }
            let Ok(alternative) = alt_pv() else { return expansion };
            let mut difference = 0.0f64;
            for r in 1..=target {
                let (mut diff, mut size) = (0.0f64, 0.0f64);
                for index in grading.range(r) {
                    let Some(value) = expansion.at(index) else { continue };
                    let Some(other) = alternative.at(index) else { return expansion.clone() };
                    for slot in 0..4 {
                        diff = diff.max(cnorm2(&csub(&value[slot], &other[slot])).to_f64().sqrt());
                        size = size.max(cnorm2(&value[slot]).to_f64().sqrt());
                    }
                }
                if size > 0.0 {
                    difference = difference.max(diff / size);
                }
            }
            if difference.is_finite() && difference <= 2.0 * change { alternative } else { expansion }
        };
        let chosen = match chosen {
            _ if self.gram_alt_only => {
                // COLLIER's a priori estimate of this reduction (CalcDred,
                // w_pv2): rounding grows like w^ceil(r/2), w = max |(Zadj f)_j|
                // / |Delta|; its rotated copy shares that error (Delta and
                // det X do not depend on the labelling), so large w is
                // refused here instead of trusted to the rotation test. The
                // parent executor counts det X's cancellation against its
                // own limit before it calls this one (`inject_gram`).
                let size = |z: &C<T>| cnorm2(z).to_f64().sqrt();
                let w = (0..nm).map(|j| size(&adjf[j])).fold(0.0f64, f64::max) / size(&delta);
                let estimate = alternative_pv_estimate(w, 1.0, target);
                if !(estimate <= ALTERNATIVE_PV_LIMIT) {
                    return Err(format!("alternative PV: estimated error {estimate:e}"));
                }
                alt_pv()?
            }
            // A hopeless nested triangle: the alternative PV, as when its
            // expansion fails.
            _ if self.gram_alt_direct => alt_pv_checked()?,
            Some(map) if last_change <= EXPANSION_TOLERANCE => prefer(map, last_change),
            _ => 'gc: {
                // Small (Zadj f)_j as well (Xtilde >> Zadj f): the expansion
                // for small Gram and small modified Cayley determinants
                // (hep-ph/0509141, eqs. (recurGramCayley1), (recurGramCayley2);
                // COLLIER's CalcDgy). Metric coefficients from
                //   2 sum_m k_m Zadj_km C(l+1, k-e_m) = sum_n Zadj_kn Shat_n(l, k)
                //       + Y_k0 C(l, k) - Delta C(l, k+e_k),
                // momentum coefficients from
                //   Xtilde_ij C(0, k) = Zadj_ij [(2(P+5-N) - 4 eps) C(1, k) - C0(0, k)]
                //       + sum_mn Zadj2_(in)(jm) f_n [Shat_m(0, k) - 2 k_m C(1, k-e_m)]
                //       + Y_0j C(0, k+e_i),
                // Y_0j = -(Zadj f)_j; each pass gains a factor Delta, Y_0j and
                // costs two ranks.
                let mut xt = [[zero.clone(), zero.clone(), zero.clone()], [zero.clone(), zero.clone(), zero.clone()], [zero.clone(), zero.clone(), zero.clone()]];
                let (mut gi, mut gj) = (0, 0);
                for a in 0..nm {
                    for b in 0..nm {
                        let mut value = cscale(&cmul(&m02, &adj[a][b]), &two);
                        for n1 in 0..nm {
                            for m1 in 0..nm {
                                if let Some(w) = second_adjugate(&z, nm, a, n1, b, m1) {
                                    value = cadd(&value, &cmul(&w, &cmul(&f[n1], &f[m1])));
                                }
                            }
                        }
                        xt[a][b] = value;
                        if cnorm2(&xt[a][b]) > cnorm2(&xt[gi][gj]) {
                            (gi, gj) = (a, b);
                        }
                    }
                }
                if cnorm2(&xt[gi][gj]).is_zero() || child_rank < target + 1 {
                    let extra = i32::from(self.gram_extra_ranks);
                    if extra > 0 && last_change * ratio.powi(extra) <= EXPANSION_TOLERANCE {
                        return Err("small-Gram expansion did not converge; a larger boost follows".into());
                    }
                    match alt_pv_checked() {
                        Ok(coefficients) => break 'gc coefficients,
                        Err(error) => {
                            self.gram_wanted_ranks = gram_ranks;
                            return Err(error);
                        }
                    }
                }
                let inv_xt = cinv(&xt[gi][gj]);
                let y0 = |j: usize| csub(&zero, &adjf[j]);
                // Order by order in (Delta, Y_0j) as the small-Gram expansion
                // above: E_0 carries the triangles, E_s (s > 0) the
                // homogeneous terms and Y_0k E_{s-1}, -Delta E_{s-1}. Metric
                // coefficients of an order need the previous order at their
                // rank, momentum coefficients of rank p its metric ones of
                // rank p + 2 and the previous order's momentum of rank p + 1,
                // so each order costs two ranks: for orders 0..=S, order s
                // runs metric ranks through target + 2 + 2 (S - s) and
                // momentum ranks through target + 2 (S - s), and E_S is pass
                // S's change.
                let gc_passes = (child_rank - 1 - target) / 2;
                let y0k = y0(k);
                let y0gj = y0(gj);
                // Factors of the recursions by multiplicity, built once by the
                // operations each key used to repeat (so the same bits).
                let gc_neg_adj: Vec<Vec<C<T>>> =
                    (0..nm).map(|m| two_x.iter().map(|x| csub(&zero, &cscale(&adj[k][m], x))).collect()).collect();
                let gc_divisor: Vec<C<T>> = two_x
                    .iter()
                    .enumerate()
                    .map(|(x, factor)| if x == 0 { zero.clone() } else { cinv(&cscale(&adj[k][l], factor)) })
                    .collect();
                let gc_wf: Vec<Vec<Option<C<T>>>> = (0..nm)
                    .map(|n1| (0..nm).map(|m1| second_adjugate(&z, nm, gi, n1, gj, m1).map(|w| cmul(&w, &f[n1]))).collect())
                    .collect();
                let gc_neg_wf: Vec<Vec<Vec<C<T>>>> = gc_wf
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|wf| match wf {
                                Some(wf) => two_x.iter().map(|x| csub(&zero, &cscale(wf, x))).collect(),
                                None => Vec::new(),
                            })
                            .collect()
                    })
                    .collect();
                let (gc_two, gc_neg_one) = (real(2), csub(&zero, &real(1)));
                // Metric keys of each degree, by decreasing kappa_l.
                let gc_kappas = &layout.gc_kappas[l];
                // Only the coefficients the results depend on.
                let gc_need = cayley_need(&grading, target, gc_passes, (l, k, gi, gj));
                let gc_metric = |cur: &mut Coefficients<T>, prev: Option<&Coefficients<T>>, need: &[bool], r: u16| -> Result<(), String> {
                    for level in 1..=r / 2 {
                        let kappas = gc_kappas.get(usize::from(r - 2 * level)).ok_or("small-Gram/Cayley expansion: rank beyond the grading")?;
                        for &target_kappa in kappas {
                            let lambda = level - 1;
                            let kappa = add(target_kappa, l);
                            // Every key below is read through the grading's
                            // neighbour tables from the target's index (the
                            // target, of rank r <= max_rank - 1, is inside).
                            let here = grading.index(level, &target_kappa).ok_or_else(|| {
                                format!("small-Gram expansion: coefficient {:?} outside the expansion ranks", (level, target_kappa))
                            })? as u32;
                            if !need[here as usize] {
                                continue;
                            }
                            let source = grading.plus(grading.lower(here), l);
                            let mut acc = lzero(&zero);
                            if let Some(prev) = prev {
                                fma(&mut acc, &y0k, prev.need_at(source, || (lambda, kappa), "small-Gram/Cayley expansion")?);
                                fma(&mut acc, &neg_delta, prev.need_at(grading.plus(source, k), || (lambda, add(kappa, k)), "small-Gram/Cayley expansion")?);
                            } else {
                                for m in 0..nm {
                                    fma(&mut acc, &adj[k][m], shat_at(m, source)?);
                                }
                            }
                            for m in (0..nm).filter(|&m| m != l) {
                                if let Some(less) = sub(kappa, m) {
                                    // (level, target - e_m + e_l), of rank r.
                                    let at = grading.plus(grading.minus(here, m), l);
                                    fma(&mut acc, &gc_neg_adj[m][usize::from(kappa[m])], cur.need_at(at, || (level, less), "small-Gram/Cayley expansion")?);
                                }
                            }
                            let divisor = &gc_divisor[usize::from(kappa[l])];
                            let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], divisor));
                            cur.insert_at(&grading, here as usize, out, &zero);
                        }
                    }
                    Ok(())
                };
                let gc_momentum = |cur: &mut Coefficients<T>, prev: Option<&Coefficients<T>>, need: &[bool], p: u16| -> Result<(), String> {
                    let dimension = self.prototype.from_usize(p as usize + 4 - nm);
                    for kappa in composition_iter(p, nm) {
                        // (1, kappa) of rank p + 2 <= max_rank - 1 and its
                        // neighbours through the grading's tables.
                        let metric_at = grading.index(1, &kappa).map_or(NONE, |index| index as u32);
                        let here = grading.lower(metric_at);
                        if here != NONE && !need[here as usize] {
                            continue;
                        }
                        let mut metric = cur.need_at(metric_at, || (1, kappa), "small-Gram/Cayley expansion")?.clone();
                        mul_dimension(&mut metric, &dimension);
                        let mut bracket = lzero(&zero);
                        fma(&mut bracket, &gc_two, &metric);
                        if prev.is_none() {
                            fma(&mut bracket, &gc_neg_one, spz.at(here as usize).ok_or("small-Gram/Cayley expansion: C0 outside the triangle ranks")?);
                        }
                        let mut acc = lzero(&zero);
                        fma(&mut acc, &adj[gi][gj], &bracket);
                        for n1 in 0..nm {
                            for m1 in 0..nm {
                                let Some(wf) = &gc_wf[n1][m1] else { continue };
                                if prev.is_none() {
                                    fma(&mut acc, wf, shat_at(m1, here)?);
                                }
                                if let Some(less) = sub(kappa, m1) {
                                    fma(&mut acc, &gc_neg_wf[n1][m1][usize::from(kappa[m1])], cur.need_at(grading.minus(metric_at, m1), || (1, less), "small-Gram/Cayley expansion")?);
                                }
                            }
                        }
                        if let Some(prev) = prev {
                            fma(&mut acc, &y0gj, prev.need_at(grading.plus(here, gi), || (0, add(kappa, gi)), "small-Gram/Cayley expansion")?);
                        }
                        let out: L4<T> = std::array::from_fn(|slot| cmul(&acc[slot], &inv_xt));
                        let here = grading.index(0, &kappa).ok_or_else(|| {
                            format!("small-Gram expansion: coefficient {:?} outside the expansion ranks", (0u16, kappa))
                        })?;
                        cur.insert_at(&grading, here, out, &zero);
                    }
                    Ok(())
                };
                let mut gc_orders: Vec<Coefficients<T>> = Vec::with_capacity(usize::from(gc_passes) + 1);
                // Metric and momentum ranks completed per order.
                let mut gc_done: Vec<(u16, u16)> = Vec::with_capacity(usize::from(gc_passes) + 1);
                let mut gc_total: Coefficients<T> = Coefficients::pooled(pool, 0);
                let mut gc_accepted = false;
                let mut gc_change = f64::INFINITY;
                let mut gc_ratio = 1.0f64;
                let mut gc_reached = 0u16;
                let mut gc_diverged = false;
                for order in 0..=gc_passes {
                    // Order `order` runs metric ranks through target + 2 + 2 (gc_passes - order).
                    let mut fresh: Coefficients<T> = Coefficients::pooled(pool, grading.end(target + 2 + 2 * (gc_passes - order)));
                    fresh.insert(&grading, (0, [0, 0, 0]), if order == 0 { d0.clone() } else { lzero(&zero) }, &zero)?;
                    gc_orders.push(fresh);
                    gc_done.push((1, 0));
                    for s in 0..=usize::from(order) {
                        let span = 2 * (order - s as u16);
                        let (lower, upper) = gc_orders.split_at_mut(s);
                        for r in gc_done[s].0 + 1..=target + 2 + span {
                            gc_metric(&mut upper[0], lower.last(), &gc_need[s], r)?;
                        }
                        for p in gc_done[s].1 + 1..=target + span {
                            gc_momentum(&mut upper[0], lower.last(), &gc_need[s], p)?;
                        }
                        gc_done[s] = (gc_done[s].0.max(target + 2 + span), gc_done[s].1.max(target + span));
                    }
                    let latest = &gc_orders[usize::from(order)];
                    if order == 0 {
                        std::mem::replace(&mut gc_total, latest.prefix_pooled(pool, grading.end(target))).recycle(pool);
                        continue;
                    }
                    let mut next = gc_total.clone_pooled(pool);
                    let mut change = 0.0f64;
                    for r in 1..=target {
                        let (mut diff, mut size) = (0.0f64, 0.0f64);
                        for index in grading.range(r) {
                            let Some(value) = latest.at(index) else { continue };
                            let Some(sum) = next.at_mut(index) else { continue };
                            for slot in 0..4 {
                                sum[slot] = cadd(&sum[slot], &value[slot]);
                                diff = diff.max(cnorm2(&value[slot]).to_f64().sqrt());
                                size = size.max(cnorm2(&sum[slot]).to_f64().sqrt());
                            }
                        }
                        if size > 0.0 {
                            change = change.max(diff / size);
                        }
                    }
                    if !change.is_finite() || (order >= 2 && change > gc_change) {
                        gc_diverged = true;
                        break;
                    }
                    if gc_change.is_finite() && gc_change > 0.0 {
                        gc_ratio = (change / gc_change).min(1.0);
                    }
                    gc_change = change;
                    std::mem::replace(&mut gc_total, next).recycle(pool);
                    gc_accepted = true;
                    gc_reached = order;
                    if change <= f64::EPSILON {
                        break;
                    }
                }
                for order in gc_orders {
                    order.recycle(pool);
                }
                let gc_chosen = if gc_accepted {
                    Some(gc_total)
                } else {
                    gc_total.recycle(pool);
                    None
                };
                // A larger boost adds `extra` passes of the small-Gram
                // expansion and `extra / 2` of the small-Cayley variant.
                let extra = i32::from(self.gram_extra_ranks);
                let retry = extra > 0
                    && (last_change * ratio.powi(extra) <= EXPANSION_TOLERANCE
                        || gc_change * gc_ratio.powi(extra / 2) <= EXPANSION_TOLERANCE);
                match gc_chosen {
                    Some(map) if gc_change <= EXPANSION_TOLERANCE => prefer(map, gc_change),
                    _ if retry => {
                        return Err(format!(
                            "small-Gram expansion did not converge (last relative change {last_change:e}; with small Cayley determinant {gc_change:e}); a larger boost follows"
                        ));
                    }
                    _ => match alt_pv_checked() {
                        Ok(coefficients) => coefficients,
                        Err(error) => {
                            // As `gram_ranks`; the variant's passes cost two
                            // ranks each.
                            let gc_ranks = (self.gram_extra_ranks == 0 && !gc_diverged)
                                .then(|| predicted_passes(gc_change, gc_ratio, gc_reached))
                                .flatten()
                                .map(|more| (target + 1 + 2 * (gc_reached + more)).saturating_sub(child_rank).max(1));
                            self.gram_wanted_ranks = gram_ranks.into_iter().chain(gc_ranks).min();
                            return Err(format!(
                                "small-Gram expansion did not converge (last relative change {last_change:e}; with small Cayley determinant {gc_change:e}); {error}"
                            ));
                        }
                    },
                }
            }
        };
        // Write the coefficients of rank <= target in the layout order.
        let output = layout.output.as_ref().ok_or("small-Gram expansion: no box layout")?;
        for &(index, at, rank) in output {
            self.values[index as usize] = if rank <= target {
                chosen.at(at as usize).cloned().ok_or("small-Gram expansion: coefficient missing")?
            } else {
                lzero(&zero)
            };
        }
        for coefficients in shat {
            coefficients.recycle(pool);
        }
        spz.recycle(pool);
        chosen.recycle(pool);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_passes_extrapolate_the_observed_rate_within_its_window() {
        // 2^-20 at a rate of 1/4 per pass: 2^-30 after five more passes.
        assert_eq!(predicted_passes(2f64.powi(-20), 0.25, 5), Some(5));
        assert_eq!(predicted_passes(2f64.powi(-20), 0.25, 4), None);
        // Converged already, not converging, or no rate.
        assert_eq!(predicted_passes(EXPANSION_TOLERANCE, 0.25, 9), None);
        assert_eq!(predicted_passes(1e-3, 1.0, 9), None);
        assert_eq!(predicted_passes(f64::INFINITY, 0.5, 9), None);
        assert_eq!(predicted_passes(1e-3, 0.0, 9), None);
    }

    #[test]
    fn expansion_need_is_the_dependency_cone_of_the_results() {
        // A box (nm = 3) at rank 5 with seven passes (triangles of rank 12):
        // every key of rank <= 5 of every order, and of the others what their
        // terms reach. The counts come from an independent model of the
        // recursion's terms; the momentum direction j and the adjugate's (k, l)
        // decide how much of the higher ranks the results reach.
        let grading = Grading::new(3, 14);
        for ((j, k, l), expected) in [((0, 0, 0), 2356), ((0, 0, 1), 3868), ((0, 1, 1), 3903), ((0, 1, 2), 4230)] {
            let pairs: Vec<(usize, usize)> =
                (0..3).flat_map(|a| (0..3).map(move |b| (a, b))).filter(|&(a, b)| a != k && b != l).collect();
            let need = expansion_need(&grading, 5, 7, (j, k, l), &pairs);
            assert_eq!(need.len(), 8);
            assert_eq!(need.iter().flatten().filter(|&&x| x).count(), expected, "(j, k, l) = {:?}", (j, k, l));
            assert!(need.iter().all(|order| order[..grading.end(5)].iter().all(|&x| x)));
        }
    }

    #[test]
    fn cayley_need_is_the_dependency_cone_of_the_results() {
        // An independent model of `gc_metric` / `gc_momentum`: walk every
        // key each order computes, record the coefficients it reads (by key
        // arithmetic, not the neighbour tables), and close the results
        // (every key of rank <= target of every order) over those reads.
        for (nm, target, child_rank) in [(3usize, 5u16, 12u16), (3, 4, 15), (2, 12, 19)] {
            let grading = Grading::new(nm, child_rank + 2);
            let passes = (child_rank - 1 - target) / 2;
            let orders = usize::from(passes) + 1;
            let at = |lambda: u16, kappa: [u16; 3]| grading.index(lambda, &kappa).unwrap();
            let mut checked = 0;
            for l in 0..nm {
                for k in 0..nm {
                    for gi in 0..nm {
                        for gj in 0..nm {
                            let mut reads: Vec<Vec<Vec<(usize, usize)>>> = vec![vec![Vec::new(); grading.end(grading.max_rank)]; orders];
                            let mut computed = vec![vec![false; grading.end(grading.max_rank)]; orders];
                            for s in 0..orders {
                                let span = 2 * (passes - s as u16);
                                for r in 1..=target + 2 + span {
                                    for level in 1..=r / 2 {
                                        for tau in compositions(r - 2 * level, nm) {
                                            let here = at(level, tau);
                                            computed[s][here] = true;
                                            for m in (0..nm).filter(|&m| m != l && tau[m] > 0) {
                                                let mut key = tau;
                                                key[m] -= 1;
                                                key[l] += 1;
                                                reads[s][here].push((s, at(level, key)));
                                            }
                                            if s > 0 {
                                                let mut source = tau;
                                                source[l] += 1;
                                                reads[s][here].push((s - 1, at(level - 1, source)));
                                                source[k] += 1;
                                                reads[s][here].push((s - 1, at(level - 1, source)));
                                            }
                                        }
                                    }
                                }
                                for p in 1..=target + span {
                                    for kappa in compositions(p, nm) {
                                        let here = at(0, kappa);
                                        computed[s][here] = true;
                                        reads[s][here].push((s, at(1, kappa)));
                                        for m in (0..nm).filter(|&m| m != gj && kappa[m] > 0) {
                                            let mut key = kappa;
                                            key[m] -= 1;
                                            reads[s][here].push((s, at(1, key)));
                                        }
                                        if s > 0 {
                                            let mut key = kappa;
                                            key[gi] += 1;
                                            reads[s][here].push((s - 1, at(0, key)));
                                        }
                                    }
                                }
                                computed[s][0] = true;
                            }
                            let mut cone = vec![vec![false; grading.end(grading.max_rank)]; orders];
                            let mut stack: Vec<(usize, usize)> =
                                (0..orders).flat_map(|s| (0..grading.end(target)).map(move |index| (s, index))).collect();
                            while let Some((s, index)) = stack.pop() {
                                if cone[s][index] {
                                    continue;
                                }
                                assert!(computed[s][index], "a read of an uncomputed key");
                                cone[s][index] = true;
                                stack.extend(reads[s][index].iter().copied());
                            }
                            let need = cayley_need(&grading, target, passes, (l, k, gi, gj));
                            assert_eq!(need.len(), orders);
                            for s in 0..orders {
                                assert_eq!(need[s], cone[s], "nm {nm} target {target} (l, k, gi, gj) {:?} order {s}", (l, k, gi, gj));
                            }
                            checked += 1;
                        }
                    }
                }
            }
            assert_eq!(checked, nm.pow(4));
        }
    }

    #[test]
    fn neighbour_tables_match_the_graded_index() {
        for nm in 1..=3 {
            let grading = Grading::new(nm, 10);
            let at = |lambda: u16, kappa: [u16; 3]| grading.index(lambda, &kappa).map_or(NONE, |index| index as u32);
            for r in 0..=10u16 {
                for lambda in 0..=r / 2 {
                    let d = r - 2 * lambda;
                    let kappas = if nm == 1 { vec![[d, 0, 0]] } else { compositions(d, nm) };
                    let block: Vec<usize> = grading.block(r, lambda).collect();
                    assert_eq!(block.len(), kappas.len());
                    for (&index, &kappa) in block.iter().zip(&kappas) {
                        assert_eq!(grading.key(index), (lambda, kappa));
                        let here = index as u32;
                        assert_eq!(grading.lower(here), if lambda > 0 { at(lambda - 1, kappa) } else { NONE });
                        assert_eq!(grading.higher(here), at(lambda + 1, kappa));
                        for i in 0..3 {
                            let mut more = kappa;
                            more[i] += 1;
                            assert_eq!(grading.plus(here, i), at(lambda, more), "nm {nm} {lambda} {kappa:?} + e{i}");
                            let less = (kappa[i] > 0).then(|| {
                                let mut less = kappa;
                                less[i] -= 1;
                                at(lambda, less)
                            });
                            assert_eq!(grading.minus(here, i), less.unwrap_or(NONE));
                        }
                    }
                }
            }
            // Outside the grading every neighbour is NONE.
            assert_eq!(grading.plus(NONE, 0), NONE);
            assert_eq!(grading.higher(grading.block(10, 5).start as u32), NONE);
        }
    }

    #[test]
    fn graded_index_numbers_every_key_once_in_rank_order() {
        for nm in 1..=3 {
            let grading = Grading::new(nm, 12);
            let mut next = 0;
            for r in 0..=12u16 {
                assert_eq!(grading.range(r).start, next);
                for lambda in 0..=r / 2 {
                    let d = r - 2 * lambda;
                    let kappas = if nm == 1 { vec![[d, 0, 0]] } else { compositions(d, nm) };
                    for kappa in kappas {
                        assert_eq!(grading.index(lambda, &kappa), Some(next), "nm {nm} lambda {lambda} kappa {kappa:?}");
                        next += 1;
                    }
                }
                assert_eq!(grading.range(r).end, next);
                assert_eq!(grading.end(r), next);
            }
            assert_eq!(grading.index(7, &[0, 0, 0]), None);
            if nm < 3 {
                let mut beyond = [0u16; 3];
                beyond[nm] = 1;
                assert_eq!(grading.index(0, &beyond), None);
            }
        }
    }

    #[test]
    fn collier_factors_of_a_triangle() {
        // Z = diag(2, 3), f = (1, 1), m0^2 = 1: Delta = 6, Zadj f = (3, 2),
        // Xtilde = [[5, 1], [1, 3]]; fac_g = 6/3 * max(1, 5/6) = 2,
        // fac_gy = max(3, 6)/5 * max(1, 1/3) = 6/5.
        let c = |x: f64| C::new(x, 0.0);
        let z = [[c(2.0), c(0.0), c(0.0)], [c(0.0), c(3.0), c(0.0)], [c(0.0), c(0.0), c(0.0)]];
        let f = [c(1.0), c(1.0), c(0.0)];
        let (fac_g, fac_gy) = gram_factors(&z, &f, &c(1.0), 2).unwrap();
        assert!((fac_g - 2.0).abs() < 1e-14, "{fac_g}");
        assert!((fac_gy - 1.2).abs() < 1e-14, "{fac_gy}");
    }

    #[test]
    fn second_adjugate_satisfies_the_paper_identities() {
        // Eq. (Zadjadjrelation0): sum_m Zadj2_(im)(jl) Z_mk = Zadj_il d_jk - Zadj_ij d_lk.
        // N = 2: the paper's d_il d_kj - d_ij d_kl with Zadj = [[z11, -z01], [-z10, z00]].
        let e2 = [[1.3, -0.4], [-0.4, 0.8]];
        let z2: [[C<f64>; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| C::new(if i < 2 && j < 2 { e2[i][j] } else { 0.0 }, 0.0))
        });
        let adj2 = |i: usize, j: usize| match (i, j) {
            (0, 0) => e2[1][1],
            (1, 1) => e2[0][0],
            _ => -e2[j][i],
        };
        for i in 0..2 {
            for j in 0..2 {
                for l in 0..2 {
                    for k in 0..2 {
                        let mut lhs = 0.0;
                        for m in 0..2 {
                            if let Some(w) = second_adjugate(&z2, 2, i, m, j, l) {
                                lhs += w.re * e2[m][k];
                            }
                        }
                        let rhs = adj2(i, l) * f64::from(u16::from(j == k)) - adj2(i, j) * f64::from(u16::from(l == k));
                        assert!((lhs - rhs).abs() < 1e-12, "N=2 i{i} j{j} l{l} k{k}: {lhs} vs {rhs}");
                    }
                }
            }
        }
        let entries = [[2.0, 0.3, -1.1], [0.3, -0.7, 0.45], [-1.1, 0.45, 1.9]];
        let z: [[C<f64>; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| C::new(entries[i][j], 0.0)));
        let adj = |i: usize, j: usize| {
            let (a, b, c, d) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
            entries[a][c] * entries[b][d] - entries[a][d] * entries[b][c]
        };
        for i in 0..3 {
            for j in 0..3 {
                for l in 0..3 {
                    for k in 0..3 {
                        let mut lhs = 0.0;
                        for m in 0..3 {
                            if let Some(w) = second_adjugate(&z, 3, i, m, j, l) {
                                lhs += w.re * entries[m][k];
                            }
                        }
                        let rhs = adj(i, l) * f64::from(u16::from(j == k)) - adj(i, j) * f64::from(u16::from(l == k));
                        assert!((lhs - rhs).abs() < 1e-12, "i{i} j{j} l{l} k{k}: {lhs} vs {rhs}");
                    }
                }
            }
        }
    }
}
