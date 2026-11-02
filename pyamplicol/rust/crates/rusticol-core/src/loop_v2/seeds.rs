// SPDX-License-Identifier: 0BSD

//! Born-weighted seeds (DESIGN.md §7.3): per seed class and loop orders
//! `U0_c[o][h] = Σ_b K[c, b] conj(A0_b(h))` per observable column `o` (its
//! weights `K` those of its seed-weight column, `LoopProgramV4::
//! column_weights`) and
//! Born helicity row `h`, from the plan's exact weights and the batch's Born
//! components (zero on the rows a helicity-resolved column does not keep,
//! [`super::observables::ColumnRows`]); per seed key `U0[k] = Σ_c x_c U0_c`;
//! and the leaf objects
//! `w_s U0[k]` that schedule B dresses (a seed times `δ_ab`, one lane vector
//! per object row).
//!
//! Only the class sums `U0_c` are held per lane chunk: schedule B forms a
//! key's `U0[k]` where a leaf reads it ([`Seeds::leaf_into`]), so a lane
//! chunk holds one value per seed class instead of one per seed key (7
//! gluons: 360-1,080 classes against ~58,000 keys of 128 rows each). The
//! value of a key is the same function of the class sums wherever it is
//! formed, so every leaf is bitwise what the materialised keys gave;
//! schedules F and M and the seed oracle read every key
//! ([`Seeds::materialise`]).

use super::dress::{Carrier, LaneData};
use super::kernels::{dispatched, lane_vector_len};
use super::plan::{LoopProgramV4, Seed};
use rayon::prelude::*;

/// Seeds of one chunk: per seed class `U0_c` and per seed key its
/// combination of classes in the carrier; per seed key, once materialised,
/// `rows × columns` lane vectors in object row order `h × columns + o`.
pub struct Seeds<T> {
    classes: Vec<Vec<T>>,
    keys: Vec<Key<T>>,
    /// Every key's value ([`Seeds::materialise`]; empty before, read
    /// through [`Seeds::values`]).
    values: Vec<Vec<T>>,
}

/// A seed key in a carrier: one class with weight one, or its classes with
/// their weights `x_c = (re, im)`.
enum Key<T> {
    Class(usize),
    Sum(Vec<(usize, T, T)>),
}

impl<T: Carrier> Seeds<T> {
    /// Lane vectors of one key's value (`rows × columns` per lane vector).
    pub fn len(&self) -> usize {
        self.classes.first().map_or(0, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `U0[key]` into `out` (zeroed by the caller for a sum, as the
    /// materialised keys start from zero).
    fn key_into<const L: usize>(&self, key: usize, out: &mut [T]) {
        match &self.keys[key] {
            Key::Class(class) => out.clone_from_slice(&self.classes[*class]),
            Key::Sum(terms) => key_sum::<T, L>(&self.classes, terms, out),
        }
    }

    /// Form every key's value (schedules F and M, the seed oracle).
    pub fn materialise<const L: usize>(&mut self, zero: &T) {
        if self.values.len() == self.keys.len() {
            return;
        }
        let len = self.len();
        let values = (0..self.keys.len())
            .into_par_iter()
            .map(|key| {
                let mut out = vec![zero.clone(); len];
                self.key_into::<L>(key, &mut out);
                out
            })
            .collect();
        self.values = values;
    }

    /// Every key's value, in key order. Panics before
    /// [`Seeds::materialise`] (schedule B never materialises: its leaves
    /// read [`Seeds::leaf_into`]).
    pub fn values(&self) -> &[Vec<T>] {
        assert_eq!(
            self.values.len(),
            self.keys.len(),
            "Seeds::values before Seeds::materialise"
        );
        &self.values
    }

    /// `U0[key]`: the materialised value, else formed.
    pub fn value<const L: usize>(&self, key: usize) -> std::borrow::Cow<'_, [T]> {
        if let Some(value) = self.values.get(key) {
            return std::borrow::Cow::Borrowed(value);
        }
        let mut out = vec![self.classes[0][0].zero(); self.len()];
        self.key_into::<L>(key, &mut out);
        std::borrow::Cow::Owned(out)
    }

    /// The leaf object `weight × U0[key]` into `out`, from the materialised
    /// value when there is one, else formed in one pass: per lane vector the
    /// key's classes summed from zero in order, then scaled (the
    /// materialised key's operations, so the same bits).
    pub fn leaf_into<const L: usize>(&self, key: usize, weight: &T, out: &mut [T]) {
        if let Some(value) = self.values.get(key) {
            leaf_object::<T, L>(value, weight, out);
            return;
        }
        match &self.keys[key] {
            Key::Class(class) => leaf_object::<T, L>(&self.classes[*class], weight, out),
            Key::Sum(terms) => leaf_sum::<T, L>(&self.classes, terms, weight, out),
        }
    }
}

/// A seed class's direct sum `U0_c[o][h] = Σ_b K[c, b] conj(A0_b(h))` into
/// `out` (zero), per Born row and observable column its weight column's
/// terms in order: `(a + ib)(c - id)` per lane, added in place.
#[inline(always)]
fn class_sum_body<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    weights: &[Vec<(usize, T, T)>],
    born: &[T],
    out: &mut [T],
) {
    let v = lane_vector_len(L);
    let columns = plan.columns;
    for h in 0..plan.rows {
        for (o, &w) in plan.column_weights.iter().enumerate() {
            if !plan.column_rows.keeps(h, o) {
                continue;
            }
            let terms = &weights[w];
            let target = &mut out[(h * columns + o) * v..][..v];
            for (component, re, im) in terms {
                let born = &born[(h * plan.born_components + component) * v..][..v];
                let (target_re, target_im) = target.split_at_mut(L);
                for l in 0..L {
                    let (c, d) = (&born[l], &born[L + l]);
                    target_re[l] += re.clone() * c + im.clone() * d;
                    target_im[l] += im.clone() * c - re.clone() * d;
                }
            }
        }
    }
}

dispatched! {
    /// [`class_sum_body`] in the widest vectors the running CPU reports for
    /// binary64 ([`super::kernels::Simd`]; the same operations per lane).
    fn class_sum<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        weights: &[Vec<(usize, T, T)>],
        born: &[T],
        out: &mut [T],
    ) => class_sum_body;
}

/// `out += Σ_c x_c U0_c` over a key's classes in order: `(a + ib)(c + id)`
/// per lane, added in place.
#[inline(always)]
fn key_sum_body<T: Carrier, const L: usize>(
    classes: &[Vec<T>],
    terms: &[(usize, T, T)],
    out: &mut [T],
) {
    let v = lane_vector_len(L);
    for (class, re, im) in terms {
        let source = &classes[*class];
        for (target, input) in out.chunks_exact_mut(v).zip(source.chunks_exact(v)) {
            let (target_re, target_im) = target.split_at_mut(L);
            let (c, d) = input.split_at(L);
            for l in 0..L {
                target_re[l] += re.clone() * &c[l] - im.clone() * &d[l];
                target_im[l] += re.clone() * &d[l] + im.clone() * &c[l];
            }
        }
    }
}

/// `out = weight × Σ_c x_c U0_c` per lane vector: the sum from zero over the
/// key's classes in order (as [`key_sum_body`] into a zero buffer), then the
/// product with `weight` (as [`leaf_object`]).
#[inline(always)]
fn leaf_sum_body<T: Carrier, const L: usize>(
    classes: &[Vec<T>],
    terms: &[(usize, T, T)],
    weight: &T,
    out: &mut [T],
) {
    let v = lane_vector_len(L);
    let zero = weight.zero();
    for (j, target) in out.chunks_exact_mut(v).enumerate() {
        let mut sum_re: [T; L] = std::array::from_fn(|_| zero.clone());
        let mut sum_im: [T; L] = std::array::from_fn(|_| zero.clone());
        for (class, re, im) in terms {
            let (c, d) = classes[*class][j * v..][..v].split_at(L);
            for l in 0..L {
                sum_re[l] += re.clone() * &c[l] - im.clone() * &d[l];
                sum_im[l] += re.clone() * &d[l] + im.clone() * &c[l];
            }
        }
        let (target_re, target_im) = target.split_at_mut(L);
        for l in 0..L {
            target_re[l] = weight.clone() * &sum_re[l];
            target_im[l] = weight.clone() * &sum_im[l];
        }
    }
}

dispatched! {
    /// [`leaf_sum_body`] in the widest vectors the running CPU reports for
    /// binary64 ([`super::kernels::Simd`]; the same operations per lane).
    fn leaf_sum<T: Carrier, const L: usize>(
        classes: &[Vec<T>],
        terms: &[(usize, T, T)],
        weight: &T,
        out: &mut [T],
    ) => leaf_sum_body;
}

dispatched! {
    /// [`key_sum_body`] in the widest vectors the running CPU reports for
    /// binary64 ([`super::kernels::Simd`]; the same operations per lane).
    fn key_sum<T: Carrier, const L: usize>(
        classes: &[Vec<T>],
        terms: &[(usize, T, T)],
        out: &mut [T],
    ) => key_sum_body;
}

/// `U0` of every seed key of the plan (classes, then keys, on parallel
/// threads), every weight contracted directly.
pub fn seeds<T: Carrier, const L: usize>(plan: &LoopProgramV4, data: &LaneData<T, L>) -> Seeds<T> {
    contract(plan, &plan.seed_classes, None, data)
}

/// Contributions added to a seed class's direct sum: per class record its
/// (Born row, weight column, lane vector) terms (the symmetric-group FFT of
/// the certified orbit pairs, [`super::fourier`]).
pub type ClassTerms<'a, T> = [Vec<(usize, usize, &'a [T])>];

/// [`seeds`] with the seed classes `classes` (the plan's, or the plan's
/// without the weights another contraction covers) and that contraction's
/// `extra` terms per class, added to every observable column of their weight
/// column that keeps their row.
pub fn contract<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    classes: &[Seed],
    extra: Option<&ClassTerms<'_, T>>,
    data: &LaneData<T, L>,
) -> Seeds<T> {
    let v = lane_vector_len(L);
    let columns = plan.columns;
    let classes: Vec<Vec<T>> = classes
        .par_iter()
        .enumerate()
        .map(|(class, seed)| {
            let weights: Vec<Vec<(usize, T, T)>> = seed
                .columns
                .iter()
                .map(|terms| {
                    terms
                        .iter()
                        .map(|t| {
                            let [re, im] = &t.weight;
                            (t.component, data.cast(re), data.cast(im))
                        })
                        .collect()
                })
                .collect();
            let mut out = vec![data.zero.clone(); plan.rows * columns * v];
            class_sum::<T, L>(plan, &weights, &data.born, &mut out);
            for &(h, w, lanes) in extra.map_or(&[][..], |extra| &extra[class]) {
                for (o, &column) in plan.column_weights.iter().enumerate() {
                    if column != w || !plan.column_rows.keeps(h, o) {
                        continue;
                    }
                    let target = &mut out[(h * columns + o) * v..][..v];
                    for (t, x) in target.iter_mut().zip(lanes) {
                        *t += x;
                    }
                }
            }
            out
        })
        .collect();
    let keys = plan
        .seeds
        .iter()
        .map(|key| {
            if let [(class, [re, im])] = key.classes.as_slice()
                && re.is_one()
                && im.is_zero()
            {
                return Key::Class(*class);
            }
            Key::Sum(
                key.classes
                    .iter()
                    .map(|(class, [re, im])| (*class, data.cast(re), data.cast(im)))
                    .collect(),
            )
        })
        .collect();
    Seeds {
        classes,
        keys,
        values: Vec::new(),
    }
}

/// The leaf object `weight × U0[seed]` into `out` (`rows × columns` lane
/// vectors).
pub fn leaf_object<T: Carrier, const L: usize>(seed: &[T], weight: &T, out: &mut [T]) {
    for (target, value) in out.iter_mut().zip(seed) {
        *target = weight.clone() * value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_objects_scale_every_component() {
        let seed = [1.0, -2.0, 0.5, 4.0];
        let mut out = [0.0; 4];
        leaf_object::<f64, 2>(&seed, &-0.5, &mut out);
        assert_eq!(out, [-0.5, 1.0, -0.25, -2.0]);
    }

    #[test]
    fn leaves_formed_in_place_equal_the_materialised_keys_bitwise() {
        const L: usize = 4;
        let len = 3 * lane_vector_len(L);
        let mut x = 0.1_f64;
        let mut next = || {
            x = (x * 7.3 + 0.37).fract();
            x * 2.0 - 1.0
        };
        let classes: Vec<Vec<f64>> = (0..3).map(|_| (0..len).map(|_| next()).collect()).collect();
        let keys = vec![
            Key::Class(1),
            Key::Sum(vec![(0, 0.3, -1.2), (2, 2.0 / 3.0, 0.5)]),
            Key::Sum(vec![(2, -1.0, 0.0), (1, 0.25, 1.0 / 7.0), (0, 1e-3, -3.5)]),
        ];
        let mut seeds = Seeds {
            classes,
            keys,
            values: Vec::new(),
        };
        let weight = -1.0 / 3.0;
        let lazy: Vec<Vec<f64>> = (0..3)
            .map(|key| {
                let mut out = vec![0.0; len];
                seeds.leaf_into::<L>(key, &weight, &mut out);
                out
            })
            .collect();
        seeds.materialise::<L>(&0.0);
        for (key, lazy) in lazy.iter().enumerate() {
            let mut out = vec![0.0; len];
            seeds.leaf_into::<L>(key, &weight, &mut out);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(lazy), bits(&out), "key {key}");
            assert_eq!(bits(&seeds.value::<L>(key)), bits(&seeds.values()[key]));
        }
    }
}
