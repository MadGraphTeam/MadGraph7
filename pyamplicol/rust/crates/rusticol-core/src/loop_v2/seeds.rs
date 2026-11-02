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
use super::kernels::lane_vector_len;
use super::plan::{LoopProgramV4, Seed};
use rayon::prelude::*;

/// Seeds of one chunk: per seed class `U0_c` and per seed key its
/// combination of classes in the carrier; per seed key, once materialised,
/// `rows × columns` lane vectors in object row order `h × columns + o`.
pub struct Seeds<T> {
    classes: Vec<Vec<T>>,
    keys: Vec<Key<T>>,
    /// Every key's value ([`Seeds::materialise`]; empty before).
    pub values: Vec<Vec<T>>,
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

    /// `U0[key]`: the materialised value, else formed.
    pub fn value<const L: usize>(&self, key: usize) -> std::borrow::Cow<'_, [T]> {
        if let Some(value) = self.values.get(key) {
            return std::borrow::Cow::Borrowed(value);
        }
        let mut out = vec![self.classes[0][0].zero(); self.len()];
        self.key_into::<L>(key, &mut out);
        std::borrow::Cow::Owned(out)
    }

    /// The leaf object `weight × U0[key]` into `out` (zeroed: a buffer of
    /// the family walk's pool), from the materialised value when there is
    /// one, else formed in place.
    pub fn leaf_into<const L: usize>(&self, key: usize, weight: &T, out: &mut [T]) {
        if let Some(value) = self.values.get(key) {
            leaf_object::<T, L>(value, weight, out);
            return;
        }
        self.key_into::<L>(key, out);
        for target in out.iter_mut() {
            let value = target.clone();
            *target = weight.clone() * &value;
        }
    }
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

/// [`key_sum_body`] in the widest vectors the running CPU reports for
/// binary64 ([`super::kernels::Simd`]; the same operations per lane).
fn key_sum<T: Carrier, const L: usize>(classes: &[Vec<T>], terms: &[(usize, T, T)], out: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    {
        use super::kernels::Simd;
        #[target_feature(enable = "avx512f")]
        fn avx512<T: Carrier, const L: usize>(
            classes: &[Vec<T>],
            terms: &[(usize, T, T)],
            out: &mut [T],
        ) {
            key_sum_body::<T, L>(classes, terms, out)
        }
        #[target_feature(enable = "avx2")]
        fn avx2<T: Carrier, const L: usize>(
            classes: &[Vec<T>],
            terms: &[(usize, T, T)],
            out: &mut [T],
        ) {
            key_sum_body::<T, L>(classes, terms, out)
        }
        match Simd::detect::<T>() {
            // SAFETY: `Simd::detect` returns a width only when the running
            // CPU reports its feature.
            Simd::Avx512 => return unsafe { avx512::<T, L>(classes, terms, out) },
            Simd::Avx2 => return unsafe { avx2::<T, L>(classes, terms, out) },
            Simd::Baseline => {}
        }
    }
    key_sum_body::<T, L>(classes, terms, out)
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
            for h in 0..plan.rows {
                for (o, &w) in plan.column_weights.iter().enumerate() {
                    if !plan.column_rows.keeps(h, o) {
                        continue;
                    }
                    let terms = &weights[w];
                    let target = &mut out[(h * columns + o) * v..][..v];
                    for (component, re, im) in terms {
                        let born = &data.born[(h * plan.born_components + component) * v..][..v];
                        // weight * conj(A0): (a + ib)(c - id), added in place.
                        let (target_re, target_im) = target.split_at_mut(L);
                        for l in 0..L {
                            let (c, d) = (&born[l], &born[L + l]);
                            target_re[l] += re.clone() * c + im.clone() * d;
                            target_im[l] += im.clone() * c - re.clone() * d;
                        }
                    }
                }
            }
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
}
