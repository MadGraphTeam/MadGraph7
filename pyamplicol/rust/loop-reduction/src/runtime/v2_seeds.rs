// SPDX-License-Identifier: 0BSD

//! Binary64 seed weights of the v2 loop engine on certified regular
//! symmetric-group orbits (design §7.3).
//!
//! Born colour basis elements and loop seed classes come in orbits indexed by
//! the lexicographic elements of `S_m` acting on external adjoint labels
//! (`loops.v2.colour.SeedFourier` certifies them exactly). On an orbit pair
//! the seed table is a group function, `K[(a, g), (l, h)] = k(g^-1 ∘ h)` with
//! `(p ∘ q)[i] = p[q[i]]`, so the seeds
//!
//! ```text
//! w_l(h) = sum_{a, g} conj(B_a(g)) k_{a l}(g^-1 ∘ h)
//! ```
//!
//! are one group convolution per pair. They are evaluated with the shared
//! [`SymmetricGroupFftPlan`] (real orthogonal Young representations,
//! `F_lambda = sum_g f(g) rho_lambda(g)`): `R_l = sum_a Bhat_a K_al^dagger`
//! with `K_al = F[g -> k_al(g^-1)]`, then `w_l = conj(F^-1 R_l)`
//! (`docs/development/LOOP_RECTANGULAR_FFT.md`). Kernels are the exact cells
//! after N_c projection, rounded once to binary64; entries outside the orbits
//! stay with the caller's direct contraction.
use super::*;
use crate::C64;
use rusticol_core::recurrence::{SymmetricGroupFftPlan, SymmetricGroupFftWorkspace};

/// Precomputed Fourier kernels of one projected seed table.
pub struct V2SeedFourier {
    plan: SymmetricGroupFftPlan,
    workspace: SymmetricGroupFftWorkspace,
    kernels: Vec<Vec<C64>>,
    pairs: Vec<[usize; 3]>,
    born_orbits: usize,
    loop_orbits: usize,
}

impl V2SeedFourier {
    /// `kernels[k]` lists `k(x)` at the lexicographic ranks of `x` in `S_degree`;
    /// `pairs` are `(kernel, Born orbit, loop orbit)`, each orbit pair at most once.
    pub fn new(
        degree: usize,
        kernels: &[Vec<C64>],
        pairs: &[[usize; 3]],
        born_orbits: usize,
        loop_orbits: usize,
    ) -> ExecutionResult<Self> {
        let plan = SymmetricGroupFftPlan::new(degree).map_err(error)?;
        let group = plan.order();
        let mut covered = std::collections::BTreeSet::new();
        for &[kernel, born, loop_] in pairs {
            if kernel >= kernels.len() || born >= born_orbits || loop_ >= loop_orbits {
                return Err(error("seed Fourier pair index out of range"));
            }
            if !covered.insert((born, loop_)) {
                return Err(error("seed Fourier orbit pair given twice"));
            }
        }
        let mut workspace = plan.workspace(1).map_err(error)?;
        let mut transformed = Vec::with_capacity(kernels.len());
        for values in kernels {
            if values.len() != group {
                return Err(error("seed Fourier kernel is not a function on the group"));
            }
            if values
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
            {
                return Err(error("seed Fourier kernel is not finite"));
            }
            let reversed = (0..group)
                .map(|g| {
                    let v = values[plan.inverse_lexicographic_index(g).map_err(error)?];
                    Ok((v.re, v.im))
                })
                .collect::<ExecutionResult<Vec<_>>>()?;
            let mut blocks = vec![(0., 0.); group];
            plan.forward(&reversed, &mut blocks, &mut workspace)
                .map_err(error)?;
            transformed.push(
                blocks
                    .into_iter()
                    .map(|(re, im)| C64::new(re, im))
                    .collect(),
            );
        }
        Ok(Self {
            plan,
            workspace,
            kernels: transformed,
            pairs: pairs.to_vec(),
            born_orbits,
            loop_orbits,
        })
    }

    pub fn group_order(&self) -> usize {
        self.plan.order()
    }

    /// Seed weights of a binary64 batch whose column `a * |G| + g` is
    /// `B_a(g)`; output column `l * |G| + h` is `w_l(h)`. Rows are lanes.
    pub fn weights(&mut self, input: &Batch) -> ExecutionResult<Batch> {
        let Data::Double(values) = input.data() else {
            return Err(error("seed Fourier input requires binary64"));
        };
        let group = self.plan.order();
        let width = shape(self.born_orbits, group)?;
        if input.precision_bits() != 53 || input.columns() != width {
            return Err(error("seed Fourier input dimensions/arithmetic differ"));
        }
        let lanes = input.rows();
        let out_width = shape(self.loop_orbits, group)?;
        if lanes == 0 {
            return Batch::typed::<f64>(vec![], vec![], out_width, 53);
        }
        if self.workspace.lane_capacity() < lanes {
            self.workspace = self.plan.workspace(lanes).map_err(error)?;
        }
        let size = shape(group, lanes)?;
        // Forward transform of every Born orbit, lane innermost.
        let mut born = Vec::with_capacity(self.born_orbits);
        for orbit in 0..self.born_orbits {
            let mut ordered = vec![(0., 0.); size];
            for g in 0..group {
                for lane in 0..lanes {
                    let v = &values[lane * width + orbit * group + g];
                    ordered[g * lanes + lane] = (v.re, v.im);
                }
            }
            let mut fourier = vec![(0., 0.); size];
            self.plan
                .forward_lanes(lanes, &ordered, &mut fourier, &mut self.workspace)
                .map_err(error)?;
            born.push(fourier);
        }
        let mut output = vec![Complex::new(0., 0.); shape(lanes, out_width)?];
        for loop_orbit in 0..self.loop_orbits {
            let mut weighted = vec![(0., 0.); size];
            let mut any = false;
            for &[kernel, born_orbit, _] in self.pairs.iter().filter(|p| p[2] == loop_orbit) {
                any = true;
                let kernel = &self.kernels[kernel];
                let born = &born[born_orbit];
                for block in self.plan.blocks() {
                    let d = block.dimension();
                    let offset = block.coefficient_offset();
                    for column in 0..d {
                        for row in 0..d {
                            let target = (offset + column * d + row) * lanes;
                            for inner in 0..d {
                                let k = kernel[offset + inner * d + column].conj();
                                let source = (offset + inner * d + row) * lanes;
                                for lane in 0..lanes {
                                    let b = born[source + lane];
                                    let value = C64::new(b.0, b.1) * k;
                                    let slot = &mut weighted[target + lane];
                                    slot.0 += value.re;
                                    slot.1 += value.im;
                                }
                            }
                        }
                    }
                }
            }
            if !any {
                continue;
            }
            let mut inverse = vec![(0., 0.); size];
            self.plan
                .inverse_lanes(lanes, &weighted, &mut inverse, &mut self.workspace)
                .map_err(error)?;
            for h in 0..group {
                for lane in 0..lanes {
                    let v = inverse[h * lanes + lane];
                    output[lane * out_width + loop_orbit * group + h] = Complex::new(v.0, -v.1);
                }
            }
        }
        Batch::typed(output, input.ids().to_vec(), out_width, 53)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unrank(degree: usize, mut rank: usize) -> Vec<usize> {
        let mut available: Vec<usize> = (0..degree).collect();
        let mut out = Vec::with_capacity(degree);
        for position in 0..degree {
            let block: usize = (1..degree - position).product();
            out.push(available.remove(rank / block));
            rank %= block;
        }
        out
    }

    fn rank(permutation: &[usize]) -> usize {
        let degree = permutation.len();
        let mut available: Vec<usize> = (0..degree).collect();
        let mut out = 0;
        for (position, value) in permutation.iter().enumerate() {
            let index = available.iter().position(|a| a == value).unwrap();
            out += index * (1..degree - position).product::<usize>();
            available.remove(index);
        }
        out
    }

    /// A deterministic, irregular complex sequence.
    fn sample(seed: usize) -> C64 {
        let x = (seed as f64 * 0.618_033_988_75).fract();
        let y = (seed as f64 * 0.414_213_562_37 + 0.3).fract();
        C64::new(x - 0.5, 2.0 * y - 1.0)
    }

    #[test]
    fn convolutions_equal_direct_contraction() {
        for degree in 2..=5 {
            let group: usize = (1..=degree).product();
            let (born_orbits, loop_orbits, lanes) = (2, 3, 3);
            let kernels: Vec<Vec<C64>> = (0..3)
                .map(|k| (0..group).map(|x| sample(97 * k + x + 1)).collect())
                .collect();
            // Loop orbit 2 has no kernel: it must come back zero.
            let pairs = [[0, 0, 0], [1, 1, 0], [2, 1, 1], [0, 0, 1]];
            let mut fourier =
                V2SeedFourier::new(degree, &kernels, &pairs, born_orbits, loop_orbits).unwrap();
            let width = born_orbits * group;
            let values: Vec<_> = (0..lanes * width)
                .map(|i| {
                    let v = sample(7 * i + 3);
                    Complex::new(v.re, v.im)
                })
                .collect();
            let input = Batch::typed::<f64>(values.clone(), (0..lanes as u64).collect(), width, 53)
                .unwrap();
            let output = fourier.weights(&input).unwrap();
            let Data::Double(output) = output.data() else {
                unreachable!()
            };
            let permutations: Vec<_> = (0..group).map(|r| unrank(degree, r)).collect();
            for lane in 0..lanes {
                for loop_orbit in 0..loop_orbits {
                    for h in 0..group {
                        let mut direct = C64::new(0., 0.);
                        for &[kernel, born_orbit, l] in &pairs {
                            if l != loop_orbit {
                                continue;
                            }
                            for g in 0..group {
                                let mut inverse = vec![0; degree];
                                for (i, &p) in permutations[g].iter().enumerate() {
                                    inverse[p] = i;
                                }
                                let relative: Vec<_> =
                                    permutations[h].iter().map(|&i| inverse[i]).collect();
                                let b = &values[lane * width + born_orbit * group + g];
                                direct += C64::new(b.re, -b.im) * kernels[kernel][rank(&relative)];
                            }
                        }
                        let got = &output[lane * loop_orbits * group + loop_orbit * group + h];
                        let error = (C64::new(got.re, got.im) - direct).norm_sqr().sqrt();
                        assert!(
                            error <= 1e-13 * (1.0 + direct.norm_sqr().sqrt()),
                            "degree {degree} lane {lane} orbit {loop_orbit} h {h}: {error}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_plans_are_refused() {
        let kernel = vec![vec![C64::new(1., 0.); 6]];
        assert!(V2SeedFourier::new(3, &kernel, &[[0, 0, 0], [0, 0, 0]], 1, 1).is_err());
        assert!(V2SeedFourier::new(3, &kernel, &[[1, 0, 0]], 1, 1).is_err());
        assert!(V2SeedFourier::new(4, &kernel, &[[0, 0, 0]], 1, 1).is_err());
    }
}
