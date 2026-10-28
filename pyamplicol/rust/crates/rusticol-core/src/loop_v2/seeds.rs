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

use super::dress::{Carrier, LaneData};
use super::kernels::lane_vector_len;
use super::plan::{LoopProgramV4, Seed};
use rayon::prelude::*;

/// Seeds of one chunk: per seed key, `rows × columns` lane vectors in object
/// row order `h × columns + o`.
pub struct Seeds<T> {
    pub values: Vec<Vec<T>>,
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
    let values = plan
        .seeds
        .par_iter()
        .map(|key| {
            if let [(class, [re, im])] = key.classes.as_slice()
                && re.is_one()
                && im.is_zero()
            {
                return classes[*class].clone();
            }
            let mut out = vec![data.zero.clone(); plan.rows * columns * v];
            for (class, [re, im]) in &key.classes {
                let (re, im) = (data.cast(re), data.cast(im));
                let source = &classes[*class];
                for (target, input) in out.chunks_exact_mut(v).zip(source.chunks_exact(v)) {
                    // x * U0_c: (a + ib)(c + id), added in place.
                    let (target_re, target_im) = target.split_at_mut(L);
                    let (c, d) = input.split_at(L);
                    for l in 0..L {
                        target_re[l] += re.clone() * &c[l] - im.clone() * &d[l];
                        target_im[l] += re.clone() * &d[l] + im.clone() * &c[l];
                    }
                }
            }
            out
        })
        .collect();
    Seeds { values }
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
