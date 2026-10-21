// SPDX-License-Identifier: 0BSD

//! Born-weighted seeds (DESIGN.md §7.3): `U0[k][o][h] = Σ_b K[k, b]
//! conj(A0_b(h))` per seed key `k` (colour class, loop orders), observable
//! column `o` and Born helicity row `h`, from the plan's exact weights and the
//! batch's Born components; and the leaf objects `w_s U0[k]` that schedule B
//! dresses (a seed times `δ_ab`, one lane vector per object row).

use super::dress::{Carrier, LaneData, cadd, parse_rational};
use super::kernels::lane_vector_len;
use super::plan::LoopProgramV4;
use crate::RusticolResult;
use symbolica::domains::float::Complex;

/// Seeds of one chunk: per seed key, `rows × columns` lane vectors in object
/// row order `h × columns + o`.
pub struct Seeds<T> {
    pub values: Vec<Vec<T>>,
}

/// An exact complex literal pair cast to the carrier.
pub fn cast_complex<T: Carrier, const L: usize>(
    data: &LaneData<T, L>,
    value: &[String; 2],
) -> RusticolResult<Complex<T>> {
    Ok(Complex::new(
        data.cast(&parse_rational(&value[0])?),
        data.cast(&parse_rational(&value[1])?),
    ))
}

/// `U0` of every seed key of the plan.
pub fn seeds<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    data: &LaneData<T, L>,
) -> RusticolResult<Seeds<T>> {
    let v = lane_vector_len(L);
    let columns = plan.columns;
    let mut term = vec![data.zero.clone(); v];
    let values = plan
        .seeds
        .iter()
        .map(|seed| {
            let weights = seed
                .columns
                .iter()
                .map(|terms| {
                    terms
                        .iter()
                        .map(|t| Ok((t.component, cast_complex(data, &t.weight)?)))
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let mut out = vec![data.zero.clone(); plan.rows * columns * v];
            for h in 0..plan.rows {
                for (o, terms) in weights.iter().enumerate() {
                    let target = &mut out[(h * columns + o) * v..][..v];
                    for (component, weight) in terms {
                        let born = &data.born[(h * plan.born_components + component) * v..][..v];
                        // weight * conj(A0): (a + ib)(c - id).
                        for l in 0..L {
                            let (c, d) = (&born[l], &born[L + l]);
                            term[l] = weight.re.clone() * c + weight.im.clone() * d;
                            term[L + l] = weight.im.clone() * c - weight.re.clone() * d;
                        }
                        cadd::<T, L>(target, &term);
                    }
                }
            }
            Ok(out)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(Seeds { values })
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
