// SPDX-License-Identifier: 0BSD
//! The native Catani operator's colour input (M-FINAL MF-8G-3): the stored
//! IR dipole operators ([`DipoleOrbits`], exact) contracted with a tree
//! program's Born colour vectors (binary64 / double-double carriers) by orbit
//! traversal, without forming a dense operator.

use super::dipole_orbits::{DipoleOrbits, filled, invalid};
use super::generate::FastMap;
use crate::RusticolResult;
use crate::loop_v2::dress::{Carrier, cast};
use rayon::prelude::*;
use symbolica::domains::float::Complex;
use symbolica::domains::rational::Rational;

/// Bytes of real B × B overlaps one orbit traversal of
/// [`Correlator::evaluate`] holds (more points take more traversals).
pub const CORRELATION_BUDGET: usize = 4 << 30;

/// The native Catani operator's colour input (M-FINAL MF-8G-3): the stored
/// operators ([`DipoleOrbits`]) and the map from a tree program's Born
/// components to Born colour vectors. [`Correlator::evaluate`] contracts the
/// operators with the Born vectors of every helicity row by orbit traversal
/// ([`DipoleOrbits::expand`]): no dense operator is formed (c² B² entries,
/// ~13 GB in binary64 at eight gluons), only one real B × B overlap per
/// point and sandwich.
#[derive(Clone, Debug)]
pub struct Correlator {
    orbits: DipoleOrbits,
    /// Per Born component (the tree program's order of `TreeInputs::
    /// amplitudes` per row): basis element, Born order and its factor (N_c
    /// to the component's grade at N_c = 3).
    components: Vec<(usize, usize, Rational)>,
    orders: usize,
    /// (bra order, ket order) per sandwich.
    sandwiches: Vec<(usize, usize)>,
}

impl Correlator {
    /// `components` per Born component (basis element, Born order, factor),
    /// `orders` Born orders, `sandwiches` the (bra, ket) order pairs formed.
    pub fn new(
        orbits: DipoleOrbits,
        components: Vec<(usize, usize, Rational)>,
        orders: usize,
        sandwiches: Vec<(usize, usize)>,
    ) -> RusticolResult<Self> {
        if components
            .iter()
            .any(|&(b, o, _)| b >= orbits.basis || o >= orders)
        {
            return Err(invalid("a Born component outside the basis or the Born orders"));
        }
        if sandwiches.iter().any(|&(a, b)| a >= orders || b >= orders) {
            return Err(invalid("a sandwich outside the Born orders"));
        }
        Ok(Self {
            orbits,
            components,
            orders,
            sandwiches,
        })
    }

    /// The stored operators.
    pub fn orbits(&self) -> &DipoleOrbits {
        &self.orbits
    }

    /// Per point (its Born component values, `rows × components`, row-major
    /// as `TreeInputs::amplitudes`): per sandwich (a, b) the `c × c` values
    /// `sum_rows Re <A_a| C_ij |A_b>` over ordered coloured-leg index pairs
    /// (zero on i = j), not divided by the average.
    pub fn evaluate<T: Carrier>(
        &self,
        points: &[Vec<Complex<T>>],
        zero: &T,
    ) -> RusticolResult<Vec<Vec<Vec<T>>>> {
        let (n, width) = (self.orbits.basis, self.components.len());
        let factors: Vec<T> = self
            .components
            .iter()
            .map(|(_, _, f)| cast(zero, f))
            .collect();
        let czero = Complex::new(zero.clone(), zero.clone());
        // Born colour vectors per point and order: rows × B.
        let vectors = |values: &Vec<Complex<T>>| -> RusticolResult<Vec<Vec<Complex<T>>>> {
            if width == 0 || values.len() % width != 0 {
                return Err(invalid("Born component values of another layout"));
            }
            let rows = values.len() / width;
            let mut out = vec![filled(rows * n, czero.clone(), "Born vectors")?; self.orders];
            for (r, row) in values.chunks_exact(width).enumerate() {
                for ((b, o, _), (value, factor)) in
                    self.components.iter().zip(row.iter().zip(&factors))
                {
                    let slot = &mut out[*o][r * n + b];
                    *slot = slot.clone()
                        + Complex::new(
                            value.re.clone() * factor.clone(),
                            value.im.clone() * factor.clone(),
                        );
                }
            }
            Ok(out)
        };
        let per_point = self.sandwiches.len().max(1) * n * n * std::mem::size_of::<T>();
        let group = (CORRELATION_BUDGET / per_point.max(1)).max(1);
        let c = self.orbits.coloured.len();
        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(group) {
            let mut overlaps = Vec::with_capacity(chunk.len() * self.sandwiches.len());
            for values in chunk {
                let vectors = vectors(values)?;
                for &(a, b) in &self.sandwiches {
                    overlaps.push(overlap(&vectors[a], &vectors[b], n, zero)?);
                }
            }
            let mut sums = vec![vec![zero.clone(); c * c]; overlaps.len()];
            let mut cache: FastMap<Rational, T> = FastMap::default();
            self.orbits.expand(|i, j, b, b2, value| {
                let weight = match cache.get(value) {
                    Some(weight) => weight.clone(),
                    None => {
                        let weight = cast(zero, value);
                        cache.insert(value.clone(), weight.clone());
                        weight
                    }
                };
                let k = b * n + b2;
                for (sum, overlap) in sums.iter_mut().zip(&overlaps) {
                    sum[i * c + j] += weight.clone() * overlap[k].clone();
                }
            })?;
            let mut sums = sums.into_iter();
            for _ in chunk {
                out.push(sums.by_ref().take(self.sandwiches.len()).collect());
            }
        }
        Ok(out)
    }
}

/// `sum_r Re(conj(bra[r, b]) ket[r, b'])` over the rows, B × B row-major.
fn overlap<T: Carrier>(
    bra: &[Complex<T>],
    ket: &[Complex<T>],
    n: usize,
    zero: &T,
) -> RusticolResult<Vec<T>> {
    let mut out = filled(n * n, zero.clone(), "Born overlaps")?;
    let rows = bra.len() / n.max(1);
    out.par_chunks_mut(n.max(1)).enumerate().for_each(|(b, line)| {
        for r in 0..rows {
            let x = &bra[r * n + b];
            if x.re == *zero && x.im == *zero {
                continue;
            }
            for (b2, slot) in line.iter_mut().enumerate() {
                let y = &ket[r * n + b2];
                *slot += x.re.clone() * y.re.clone() + x.im.clone() * y.im.clone();
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_v2::dipole_orbits::tests::trace_basis;
    use crate::loop_v2::generate::cells::Leg;

    const ADJOINT: i32 = 8;

    #[test]
    fn correlations_by_orbit_traversal_equal_the_dense_operators() {
        // Five gluons (24 traces), two Born orders, three helicity rows;
        // components: every (b, order) once plus a graded duplicate of some.
        let labels = [1, 2, 3, 4, 5];
        let legs: Vec<Leg> = labels.iter().map(|&l| (l, ADJOINT)).collect();
        let orbits =
            DipoleOrbits::build(&legs, &trace_basis(&labels), &[labels.to_vec()]).unwrap();
        let (c, n) = (orbits.coloured.len(), orbits.basis);
        let mut components = Vec::new();
        for b in 0..n {
            for o in 0..2 {
                components.push((b, o, Rational::one()));
            }
        }
        for b in (0..n).step_by(5) {
            components.push((b, b % 2, Rational::from(3)));
        }
        let sandwiches = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let correlator =
            Correlator::new(orbits.clone(), components.clone(), 2, sandwiches.clone()).unwrap();
        let rows = 3;
        let value = |k: usize| {
            let x = (k as f64 * 0.7548776662466927).fract() - 0.5;
            let y = (k as f64 * 0.5698402909980532).fract() - 0.5;
            Complex::new(x, y)
        };
        let points: Vec<Vec<Complex<f64>>> = (0..2)
            .map(|p| {
                (0..rows * components.len())
                    .map(|k| value(k + 1000 * p))
                    .collect()
            })
            .collect();
        let got = correlator.evaluate(&points, &0.0).unwrap();
        let dense = orbits.expand_f64().unwrap();
        for (p, values) in points.iter().enumerate() {
            let mut vectors = vec![vec![Complex::new(0.0, 0.0); rows * n]; 2];
            for r in 0..rows {
                for (k, (b, o, f)) in components.iter().enumerate() {
                    let f = f.to_f64();
                    let z = values[r * components.len() + k];
                    vectors[*o][r * n + b] += Complex::new(z.re * f, z.im * f);
                }
            }
            for (s, &(a, b)) in sandwiches.iter().enumerate() {
                for i in 0..c {
                    for j in 0..c {
                        let mut want = 0.0;
                        for r in 0..rows {
                            for x in 0..n {
                                for y in 0..n {
                                    let entry = dense[((i * c + j) * n + x) * n + y];
                                    let (u, v) = (vectors[a][r * n + x], vectors[b][r * n + y]);
                                    want += entry * (u.re * v.re + u.im * v.im);
                                }
                            }
                        }
                        let have = got[p][s][i * c + j];
                        assert!(
                            (have - want).abs() <= 1e-12 * (1.0 + want.abs()),
                            "{p} {s} {i} {j}: {have} vs {want}"
                        );
                    }
                }
            }
        }
        // A component outside the basis or the orders is refused.
        assert!(Correlator::new(orbits.clone(), vec![(n, 0, Rational::one())], 1, vec![]).is_err());
        assert!(Correlator::new(orbits, vec![(0, 0, Rational::one())], 1, vec![(0, 1)]).is_err());
    }
}
