// SPDX-License-Identifier: 0BSD

//! The certified symmetric-group FFT of the seed contraction (DESIGN.md §7.3;
//! M-QCD package MQ-FFT).
//!
//! `S_m` acts on the external adjoint labels. On a certified pair of a Born
//! orbit `a` (basis elements `b_a(h)`, `h ∈ S_m` by lexicographic rank) and a
//! loop orbit `l` (seed classes `c_l(g)` with signs `s_l(g)`,
//! `g.κ_l = s_l(g) κ_{c_l(g)}`), the seed weights of one weight column are a
//! group function,
//!
//! ```text
//! K[c_l(g), b_a(h)] s_l(g) = k_al(h⁻¹ ∘ g),        (p ∘ q)[i] = p[q[i]],
//! ```
//!
//! so the covered part of `U0_c = Σ_b K[c, b] conj(A0_b)` is the group
//! convolution `w_l = Σ_a f_a ⋆ k_al` of `f_a(h) = conj(A0_{b_a(h)})`, read at
//! the first element `g` of each class: `U0_{c_l(g)} += s_l(g) w_l(g)`. It is
//! evaluated with the subgroup-tower Fourier transform of the tree engine's
//! `SymmetricGroupFftPlan` (Young's orthogonal form, `F_λ = Σ_g f(g) ρ_λ(g)`,
//! `F[f ⋆ k] = F[f] F[k]`), in the carrier itself: the plan supplies the tower
//! and the integer axial distances, and the Young factors `1/r` and
//! `sqrt(1 - 1/r²)` and the inverse weights `d_λ/m!` are formed in the
//! carrier, so binary64, double-double and MPFR contract at their own
//! precision. The exact carrier, whose arithmetic has no square roots,
//! contracts directly.
//!
//! The record ([`SeedFourier::load`]; `plan_export.fourier_record` writes it
//! from `loops.v2.colour.seed_fourier`, which detects and certifies the orbits
//! structurally on the exact Laurent cells) is structural only: per job (one
//! weight column, Born order, loop order and Born N_c grade) the plan's Born
//! component of every Born-orbit element, the plan's seed-class record and
//! sign of every loop-orbit element, and the certified orbit pairs. The
//! kernels are read from the plan's own exact weights, and loading certifies
//! every pair again on them: every weight of every covered (record, component)
//! entry must equal the group function exactly, else the record is refused.
//! The covered entries then leave the direct seed classes, so the FFT and the
//! direct contraction are equal before rounding.

use super::dress::{Carrier, LaneData, cast};
use super::kernels::lane_vector_len;
use super::plan::{LoopProgramV4, Seed};
use super::seeds::{Seeds, contract};
use crate::recurrence::SymmetricGroupFftPlan;
use crate::recurrence::symmetric_group_fft::SymmetricGroupLevel;
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use symbolica::domains::float::Real;
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;

/// The schema of the seed Fourier record.
pub const FOURIER_SCHEMA: u32 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FourierFile {
    schema: u32,
    degree: usize,
    jobs: Vec<JobFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobFile {
    /// The weight column (an index into every `Seed::columns`).
    column: usize,
    /// Per Born orbit, per group element: the plan's Born component, or -1
    /// when the plan has none (no seed weight reads it: its amplitude is
    /// zero for the FFT).
    born: Vec<Vec<i64>>,
    /// Per loop orbit, per group element: (seed-class record or -1, sign).
    #[serde(rename = "loop")]
    loops: Vec<Vec<(i64, i8)>>,
    /// The certified (Born orbit, loop orbit) pairs.
    pairs: Vec<(usize, usize)>,
}

/// One certified orbit pair with its exact kernel `k(x)` per rank of `x`.
#[derive(Debug)]
struct Pair {
    born: usize,
    loop_orbit: usize,
    kernel: Vec<[Rational; 2]>,
}

#[derive(Debug)]
struct Job {
    column: usize,
    born: Vec<Vec<Option<usize>>>,
    /// Per loop orbit its outputs: (seed-class record, first group element
    /// of the record, negative sign).
    outputs: Vec<Vec<(usize, usize, bool)>>,
    pairs: Vec<Pair>,
    /// Per loop orbit the indices of its pairs, in pair order (one scan of
    /// the pairs, not one per loop orbit).
    by_loop: Vec<Vec<usize>>,
}

/// The certified seed FFT of one plan: its jobs and the plan's seed classes
/// without the entries the jobs cover.
#[derive(Debug)]
pub struct SeedFourier {
    fft: SymmetricGroupFftPlan,
    jobs: Vec<Job>,
    direct: Vec<Seed>,
    covered: usize,
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(format!("loop v2 seed Fourier record: {}", message.into()))
}

fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

/// The permutation of lexicographic rank `rank` in `S_degree`.
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

/// The lexicographic rank of a permutation of `0..len`.
fn rank(permutation: &[usize]) -> usize {
    let degree = permutation.len();
    let mut out = 0;
    for (position, &value) in permutation.iter().enumerate() {
        let smaller = permutation[position + 1..]
            .iter()
            .filter(|&&later| later < value)
            .count();
        out += smaller * (1..degree - position).product::<usize>();
    }
    out
}

/// `rank(h⁻¹ ∘ g)` for permutations given with `h`'s inverse.
fn relative(h_inverse: &[usize], g: &[usize], scratch: &mut [usize]) -> usize {
    for (slot, &x) in scratch.iter_mut().zip(g) {
        *slot = h_inverse[x];
    }
    rank(scratch)
}

impl SeedFourier {
    /// Load and certify a record against `program` (the plan it was exported
    /// with): indices in range, regular orbits of the record's degree, and
    /// every covered weight equal to its pair's group function exactly.
    pub fn load(json: &[u8], program: &LoopProgramV4) -> RusticolResult<Self> {
        let file: FourierFile =
            serde_json::from_slice(json).map_err(|e| invalid(format!("{e}")))?;
        require(file.schema == FOURIER_SCHEMA, "unsupported schema")?;
        require(file.degree >= 2, "the group degree must be at least 2")?;
        // The tower's own admission bounds the degree (its reorder storage).
        let fft = SymmetricGroupFftPlan::new(file.degree)?;
        let group = fft.order();
        let degree = file.degree;
        let elements: Vec<Vec<usize>> = (0..group).map(|r| unrank(degree, r)).collect();
        let inverses: Vec<Vec<usize>> = elements
            .iter()
            .map(|p| {
                let mut inverse = vec![0; degree];
                for (i, &x) in p.iter().enumerate() {
                    inverse[x] = i;
                }
                inverse
            })
            .collect();
        let weight_columns = program.column_weights.iter().max().map_or(0, |w| w + 1);
        let classes = program.seed_classes.len();
        // Per (record, weight column): component -> exact weight (summed).
        let mut weights: HashMap<(usize, usize), HashMap<usize, [Rational; 2]>> = HashMap::new();
        let mut load_row = |record: usize, column: usize| {
            weights.entry((record, column)).or_insert_with(|| {
                let mut out: HashMap<usize, [Rational; 2]> = HashMap::new();
                for term in &program.seed_classes[record].columns[column] {
                    let entry = out
                        .entry(term.component)
                        .or_insert_with(|| [Rational::zero(), Rational::zero()]);
                    entry[0] = &entry[0] + &term.weight[0];
                    entry[1] = &entry[1] + &term.weight[1];
                }
                out
            });
        };
        let mut rows_needed = Vec::new();
        for job in &file.jobs {
            for orbit in &job.loops {
                for &(record, _sign) in orbit {
                    if record >= 0 && (record as usize) < program.seed_classes.len() {
                        rows_needed.push((record as usize, job.column));
                    }
                }
            }
        }
        for (record, column) in rows_needed {
            if column < weight_columns {
                load_row(record, column);
            }
        }
        let weight_of = |record: usize, column: usize| -> &HashMap<usize, [Rational; 2]> {
            &weights[&(record, column)]
        };
        let zero = [Rational::zero(), Rational::zero()];
        let mut covered: BTreeMap<(usize, usize), BTreeSet<usize>> = BTreeMap::new();
        let mut jobs = Vec::with_capacity(file.jobs.len());
        let mut scratch = vec![0; degree];
        for job in &file.jobs {
            require(job.column < weight_columns, "weight column index")?;
            let born = job
                .born
                .iter()
                .map(|orbit| {
                    require(orbit.len() == group, "a Born orbit is not regular")?;
                    orbit
                        .iter()
                        .map(|&j| match j {
                            -1 => Ok(None),
                            j if j >= 0 && (j as usize) < program.born_components => {
                                Ok(Some(j as usize))
                            }
                            _ => Err(invalid("Born component index")),
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let loops = job
                .loops
                .iter()
                .map(|orbit| {
                    require(orbit.len() == group, "a loop orbit is not complete")?;
                    orbit
                        .iter()
                        .map(|&(record, sign)| {
                            require(sign == 1 || sign == -1, "a loop-orbit sign is not ±1")?;
                            match record {
                                -1 => Ok(None),
                                r if r >= 0 && (r as usize) < classes => {
                                    Ok(Some((r as usize, sign < 0)))
                                }
                                _ => Err(invalid("seed-class record index")),
                            }
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let mut seen = BTreeSet::new();
            let mut pairs = Vec::new();
            for &(a, l) in &job.pairs {
                require(a < born.len() && l < loops.len(), "orbit pair index")?;
                require(seen.insert((a, l)), "an orbit pair is given twice")?;
                let Some(g0) = loops[l].iter().position(Option::is_some) else {
                    // No class of the orbit is read by a seed key.
                    continue;
                };
                let (r0, negative0) = loops[l][g0].expect("record");
                // k(h⁻¹ ∘ g0) = s(g0) K[c(g0), b(h)].
                let mut kernel = vec![zero.clone(); group];
                {
                    let row = weight_of(r0, job.column);
                    for h in 0..group {
                        let x = relative(&inverses[h], &elements[g0], &mut scratch);
                        if let Some(w) = born[a][h].and_then(|j| row.get(&j)) {
                            kernel[x] = if negative0 {
                                [-w[0].clone(), -w[1].clone()]
                            } else {
                                w.clone()
                            };
                        }
                    }
                }
                // Certify every record of the orbit on every Born element
                // (a record recurs at every element of its stabiliser coset).
                let mut records = BTreeSet::new();
                for (g, entry) in loops[l].iter().enumerate() {
                    let Some((r, negative)) = *entry else {
                        continue;
                    };
                    records.insert(r);
                    let row = weight_of(r, job.column);
                    for h in 0..group {
                        let x = relative(&inverses[h], &elements[g], &mut scratch);
                        let w = born[a][h].and_then(|j| row.get(&j)).unwrap_or(&zero);
                        let k = &kernel[x];
                        let equal = if negative {
                            w[0] == -k[0].clone() && w[1] == -k[1].clone()
                        } else {
                            w[0] == k[0] && w[1] == k[1]
                        };
                        if !equal {
                            return Err(invalid(format!(
                                "the plan's weight of record {r} at component {:?} is not \
                                 the group function of orbit pair ({a}, {l})",
                                born[a][h]
                            )));
                        }
                    }
                }
                for r in records {
                    let entry = covered.entry((r, job.column)).or_default();
                    for j in born[a].iter().flatten() {
                        require(entry.insert(*j), "two pairs cover one seed weight")?;
                    }
                }
                pairs.push(Pair {
                    born: a,
                    loop_orbit: l,
                    kernel,
                });
            }
            let mut outputs = vec![Vec::new(); loops.len()];
            for pair in &pairs {
                let orbit = &mut outputs[pair.loop_orbit];
                if !orbit.is_empty() {
                    continue;
                }
                let mut first = BTreeSet::new();
                for (g, entry) in loops[pair.loop_orbit].iter().enumerate() {
                    if let Some((r, negative)) = *entry
                        && first.insert(r)
                    {
                        orbit.push((r, g, negative));
                    }
                }
            }
            if !pairs.is_empty() {
                let mut by_loop = vec![Vec::new(); loops.len()];
                for (index, pair) in pairs.iter().enumerate() {
                    by_loop[pair.loop_orbit].push(index);
                }
                jobs.push(Job {
                    column: job.column,
                    born,
                    outputs,
                    pairs,
                    by_loop,
                });
            }
        }
        // A record's outputs come from one loop orbit per weight column and
        // loop order: two jobs may share a record only on disjoint components
        // (checked above), so each job adds its own convolution.
        let mut direct = program.seed_classes.clone();
        let mut removed = 0;
        for ((record, column), components) in &covered {
            let terms = &mut direct[*record].columns[*column];
            let before = terms.len();
            terms.retain(|term| !components.contains(&term.component));
            removed += before - terms.len();
        }
        Ok(Self {
            fft,
            jobs,
            direct,
            covered: removed,
        })
    }

    /// Jobs (weight column, Born order, loop order, Born grade) with a
    /// certified pair.
    pub fn jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Certified orbit pairs over every job.
    pub fn pairs(&self) -> usize {
        self.jobs.iter().map(|job| job.pairs.len()).sum()
    }

    /// Plan seed weights the FFT contracts instead of the direct sum.
    pub fn covered(&self) -> usize {
        self.covered
    }

    /// The group order `m!`.
    pub fn group_order(&self) -> usize {
        self.fft.order()
    }
}

// ----------------------------------------------------------- the transform

/// Young factors of every tower level in one carrier: per level, irrep and
/// (generator, basis vector) its partner, `1/r` and `sqrt(1 - 1/r²)`; the
/// inverse weights `d_λ/m!` of the top level.
struct Factors<T> {
    levels: Vec<Vec<Vec<Action<T>>>>,
    weights: Vec<T>,
}

#[derive(Clone)]
struct Action<T> {
    partner: usize,
    /// `|r| = 1`: the basis vector is fixed, `±1` (`flip` for -1).
    fixed: bool,
    flip: bool,
    diagonal: T,
    mixing: T,
}

impl<T: Carrier + Real> Factors<T> {
    fn new(fft: &SymmetricGroupFftPlan, zero: &T) -> Self {
        let levels = fft
            .levels()
            .iter()
            .map(|level| {
                level
                    .irreps
                    .iter()
                    .map(|irrep| {
                        irrep
                            .generators
                            .iter()
                            .map(|action| {
                                let r = i64::from(action.axial);
                                let diagonal =
                                    cast(zero, &Rational::new(Integer::from(1), Integer::from(r)));
                                let mixing = if r.abs() == 1 {
                                    zero.clone()
                                } else {
                                    cast(
                                        zero,
                                        &Rational::new(
                                            Integer::from(r * r - 1),
                                            Integer::from(r * r),
                                        ),
                                    )
                                    .sqrt()
                                };
                                Action {
                                    partner: action.partner as usize,
                                    fixed: r.abs() == 1,
                                    flip: r == -1,
                                    diagonal,
                                    mixing,
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let order = fft.order() as i64;
        let weights = fft
            .blocks()
            .map(|block| {
                cast(
                    zero,
                    &Rational::new(
                        Integer::from(block.dimension() as i64),
                        Integer::from(order),
                    ),
                )
            })
            .collect();
        Self { levels, weights }
    }
}

/// Left multiplication of the `d × d` column-major block (`lanes` reals per
/// entry) by one Young generator.
fn apply_generator<T: Carrier>(
    dimension: usize,
    actions: &[Action<T>],
    generator: usize,
    block: &mut [T],
    lanes: usize,
) {
    let actions = &actions[generator * dimension..(generator + 1) * dimension];
    for first in 0..dimension {
        let action = &actions[first];
        let other = action.partner;
        if other < first {
            continue;
        }
        if other == first {
            debug_assert!(action.fixed);
            if action.flip {
                for column in 0..dimension {
                    let base = (column * dimension + first) * lanes;
                    for value in &mut block[base..base + lanes] {
                        *value = -value.clone();
                    }
                }
            }
            continue;
        }
        let partner = &actions[other];
        for column in 0..dimension {
            let first_base = (column * dimension + first) * lanes;
            let other_base = (column * dimension + other) * lanes;
            for lane in 0..lanes {
                let a = block[first_base + lane].clone();
                let b = block[other_base + lane].clone();
                block[first_base + lane] =
                    action.diagonal.clone() * &a + action.mixing.clone() * &b;
                block[other_base + lane] =
                    partner.mixing.clone() * &a + partner.diagonal.clone() * &b;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transform_level<T: Carrier>(
    level: &SymmetricGroupLevel,
    child: &SymmetricGroupLevel,
    actions: &[Vec<Action<T>>],
    input: &[T],
    output: &mut [T],
    block: &mut Vec<T>,
    lanes: usize,
    total: usize,
    zero: &T,
) {
    for group_start in (0..total).step_by(level.order) {
        output[group_start * lanes..(group_start + level.order) * lanes].fill(zero.clone());
        for coset in 0..level.degree {
            let child_start = group_start + coset * child.order;
            for (index, irrep) in level.irreps.iter().enumerate() {
                let d = irrep.dimension;
                block.clear();
                block.resize(d * d * lanes, zero.clone());
                for branch in irrep.branches.iter() {
                    let sub = &child.irreps[branch.child_irrep as usize];
                    let offset = branch.basis_offset as usize;
                    for column in 0..sub.dimension {
                        for row in 0..sub.dimension {
                            let source = (child_start
                                + sub.coefficient_offset
                                + column * sub.dimension
                                + row)
                                * lanes;
                            let target = ((offset + column) * d + offset + row) * lanes;
                            block[target..target + lanes]
                                .clone_from_slice(&input[source..source + lanes]);
                        }
                    }
                }
                // The coset representative is s_i s_(i+1) ... s_(m-1): its
                // rightmost factor acts first.
                for generator in (coset..level.degree - 1).rev() {
                    apply_generator(d, &actions[index], generator, block, lanes);
                }
                let offset = (group_start + irrep.coefficient_offset) * lanes;
                for (target, value) in output[offset..offset + d * d * lanes]
                    .iter_mut()
                    .zip(block.iter())
                {
                    *target += value;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn transpose_level<T: Carrier>(
    level: &SymmetricGroupLevel,
    child: &SymmetricGroupLevel,
    actions: &[Vec<Action<T>>],
    input: &[T],
    output: &mut [T],
    block: &mut Vec<T>,
    lanes: usize,
    total: usize,
    zero: &T,
) {
    for group_start in (0..total).step_by(level.order) {
        output[group_start * lanes..(group_start + level.order) * lanes].fill(zero.clone());
        for coset in 0..level.degree {
            let child_start = group_start + coset * child.order;
            for (index, irrep) in level.irreps.iter().enumerate() {
                let d = irrep.dimension;
                let start = (group_start + irrep.coefficient_offset) * lanes;
                block.clear();
                block.extend_from_slice(&input[start..start + d * d * lanes]);
                for generator in coset..level.degree - 1 {
                    apply_generator(d, &actions[index], generator, block, lanes);
                }
                for branch in irrep.branches.iter() {
                    let sub = &child.irreps[branch.child_irrep as usize];
                    let offset = branch.basis_offset as usize;
                    for column in 0..sub.dimension {
                        for row in 0..sub.dimension {
                            let source = ((offset + column) * d + offset + row) * lanes;
                            let target = (child_start
                                + sub.coefficient_offset
                                + column * sub.dimension
                                + row)
                                * lanes;
                            for lane in 0..lanes {
                                output[target + lane] += &block[source + lane];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Scratch of one transform.
struct Workspace<T> {
    buffer: Vec<T>,
    block: Vec<T>,
}

impl<T: Carrier> Workspace<T> {
    fn new(group: usize, lanes: usize, zero: &T) -> Self {
        Self {
            buffer: vec![zero.clone(); group * lanes],
            block: Vec::new(),
        }
    }
}

/// `F_λ = Σ_g f(g) ρ_λ(g)` of `lanes` real functions (lane innermost; values
/// at lexicographic ranks, coefficients packed as the plan's blocks).
fn forward<T: Carrier>(
    fft: &SymmetricGroupFftPlan,
    factors: &Factors<T>,
    lanes: usize,
    values: &[T],
    coefficients: &mut [T],
    work: &mut Workspace<T>,
    zero: &T,
) {
    let total = fft.order();
    for (recursive, &lexicographic) in fft.recursive_to_lexicographic().iter().enumerate() {
        let source = lexicographic as usize * lanes;
        work.buffer[recursive * lanes..(recursive + 1) * lanes]
            .clone_from_slice(&values[source..source + lanes]);
    }
    let levels = fft.levels();
    let mut in_buffer = true;
    for degree in 2..levels.len() {
        let (level, child, actions) = (
            &levels[degree],
            &levels[degree - 1],
            &factors.levels[degree],
        );
        if in_buffer {
            transform_level(
                level,
                child,
                actions,
                &work.buffer,
                coefficients,
                &mut work.block,
                lanes,
                total,
                zero,
            );
        } else {
            transform_level(
                level,
                child,
                actions,
                coefficients,
                &mut work.buffer,
                &mut work.block,
                lanes,
                total,
                zero,
            );
        }
        in_buffer = !in_buffer;
    }
    if in_buffer {
        coefficients.clone_from_slice(&work.buffer);
    }
}

/// The inverse of [`forward`]: `f(g) = Σ_λ (d_λ/m!) tr(ρ_λ(g)ᵀ F_λ)`.
fn inverse<T: Carrier>(
    fft: &SymmetricGroupFftPlan,
    factors: &Factors<T>,
    lanes: usize,
    coefficients: &[T],
    values: &mut [T],
    work: &mut Workspace<T>,
    zero: &T,
) {
    let total = fft.order();
    work.buffer.clone_from_slice(coefficients);
    for (block, weight) in fft.blocks().zip(&factors.weights) {
        let start = block.coefficient_offset() * lanes;
        for value in &mut work.buffer[start..start + block.coefficient_count() * lanes] {
            *value = value.clone() * weight;
        }
    }
    let levels = fft.levels();
    let mut in_buffer = true;
    for degree in (2..levels.len()).rev() {
        let (level, child, actions) = (
            &levels[degree],
            &levels[degree - 1],
            &factors.levels[degree],
        );
        if in_buffer {
            transpose_level(
                level,
                child,
                actions,
                &work.buffer,
                values,
                &mut work.block,
                lanes,
                total,
                zero,
            );
        } else {
            transpose_level(
                level,
                child,
                actions,
                values,
                &mut work.buffer,
                &mut work.block,
                lanes,
                total,
                zero,
            );
        }
        in_buffer = !in_buffer;
    }
    if !in_buffer {
        work.buffer.clone_from_slice(values);
    }
    for (recursive, &lexicographic) in fft.recursive_to_lexicographic().iter().enumerate() {
        let target = lexicographic as usize * lanes;
        values[target..target + lanes]
            .clone_from_slice(&work.buffer[recursive * lanes..(recursive + 1) * lanes]);
    }
}

// ------------------------------------------------------------- the seeds

/// Born rows per transform: the rows of a task are folded into the
/// transform's lane dimension (`rows × 2L` real lanes), which amortises the
/// tower walk over them and keeps every inner loop contiguous.
const ROWS_PER_TASK: usize = 16;

/// FFT outputs of one (job, block of Born rows): per output its Born row,
/// seed-class record and lane vector.
type RowOutputs<T> = Vec<(usize, usize, Vec<T>)>;

/// The covered seeds of the Born rows `rows` for one job.
fn job_rows<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    fourier: &SeedFourier,
    factors: &Factors<T>,
    job: &Job,
    kernels: &[Vec<T>],
    data: &LaneData<T, L>,
    rows: &[usize],
) -> RowOutputs<T> {
    let v = lane_vector_len(L);
    let n = rows.len() * v;
    let group = fourier.fft.order();
    let zero = &data.zero;
    let mut work = Workspace::new(group, n, zero);
    let used: BTreeSet<usize> = job.pairs.iter().map(|pair| pair.born).collect();
    let mut born: BTreeMap<usize, Vec<T>> = BTreeMap::new();
    let mut values = vec![zero.clone(); group * n];
    for &a in &used {
        // f_a(h) = conj(A0_{b_a(h)}), per row a lane vector of the lane block.
        for (h, component) in job.born[a].iter().enumerate() {
            for (r, &row) in rows.iter().enumerate() {
                let target = &mut values[h * n + r * v..][..v];
                match component {
                    Some(j) => {
                        let source = &data.born[(row * plan.born_components + j) * v..][..v];
                        target[..L].clone_from_slice(&source[..L]);
                        for (t, s) in target[L..].iter_mut().zip(&source[L..]) {
                            *t = -s.clone();
                        }
                    }
                    None => target.fill(zero.clone()),
                }
            }
        }
        let mut transformed = vec![zero.clone(); group * n];
        forward(
            &fourier.fft,
            factors,
            n,
            &values,
            &mut transformed,
            &mut work,
            zero,
        );
        born.insert(a, transformed);
    }
    let mut out = Vec::new();
    let mut product = vec![zero.clone(); group * n];
    let mut convolution = vec![zero.clone(); group * n];
    for (l, outputs) in job.outputs.iter().enumerate() {
        if outputs.is_empty() {
            continue;
        }
        product.fill(zero.clone());
        for &index in &job.by_loop[l] {
            let (pair, kernel) = (&job.pairs[index], &kernels[index]);
            let source = &born[&pair.born];
            // R(r, c) += F(r, i) K(i, c), complex, per block.
            for block in fourier.fft.blocks() {
                let d = block.dimension();
                let offset = block.coefficient_offset();
                for column in 0..d {
                    for r in 0..d {
                        let target = (offset + column * d + r) * n;
                        for inner in 0..d {
                            let k = offset + column * d + inner;
                            let (kr, ki) = (&kernel[2 * k], &kernel[2 * k + 1]);
                            if kr.is_zero() && ki.is_zero() {
                                continue;
                            }
                            let s = (offset + inner * d + r) * n;
                            for b in 0..rows.len() {
                                let (fr, fi) = source[s + b * v..][..v].split_at(L);
                                let (pr, pi) = product[target + b * v..][..v].split_at_mut(L);
                                for lane in 0..L {
                                    pr[lane] += kr.clone() * &fr[lane] - ki.clone() * &fi[lane];
                                    pi[lane] += kr.clone() * &fi[lane] + ki.clone() * &fr[lane];
                                }
                            }
                        }
                    }
                }
            }
        }
        inverse(
            &fourier.fft,
            factors,
            n,
            &product,
            &mut convolution,
            &mut work,
            zero,
        );
        for &(record, g, negative) in outputs {
            for (b, &row) in rows.iter().enumerate() {
                let lanes = &convolution[g * n + b * v..][..v];
                out.push((
                    row,
                    record,
                    if negative {
                        lanes.iter().map(|x| -x.clone()).collect()
                    } else {
                        lanes.to_vec()
                    },
                ));
            }
        }
    }
    out
}

/// `U0` of every seed key ([`super::seeds::seeds`]) with the certified pairs
/// of `fourier` contracted by the symmetric-group FFT in the carrier and the
/// other weights directly.
pub fn seeds_fourier<T: Carrier + Real, const L: usize>(
    plan: &LoopProgramV4,
    fourier: &SeedFourier,
    data: &LaneData<T, L>,
) -> Seeds<T> {
    let zero = &data.zero;
    let factors = Factors::new(&fourier.fft, zero);
    let group = fourier.fft.order();
    // Per job and pair the transformed kernel: (re, im) per coefficient.
    let kernels: Vec<Vec<Vec<T>>> = fourier
        .jobs
        .iter()
        .map(|job| {
            job.pairs
                .iter()
                .map(|pair| {
                    let mut work = Workspace::new(group, 2, zero);
                    let values: Vec<T> = pair
                        .kernel
                        .iter()
                        .flat_map(|[re, im]| [data.cast(re), data.cast(im)])
                        .collect();
                    let mut out = vec![zero.clone(); 2 * group];
                    forward(
                        &fourier.fft,
                        &factors,
                        2,
                        &values,
                        &mut out,
                        &mut work,
                        zero,
                    );
                    out
                })
                .collect()
        })
        .collect();
    // Per job its Born rows some column of its weight column keeps, in
    // blocks of `ROWS_PER_TASK`.
    let tasks: Vec<(usize, Vec<usize>)> = fourier
        .jobs
        .iter()
        .enumerate()
        .flat_map(|(k, job)| {
            let rows: Vec<usize> = (0..plan.rows)
                .filter(|&h| {
                    plan.column_weights
                        .iter()
                        .enumerate()
                        .any(|(o, &w)| w == job.column && plan.column_rows.keeps(h, o))
                })
                .collect();
            rows.chunks(ROWS_PER_TASK)
                .map(|block| (k, block.to_vec()))
                .collect::<Vec<_>>()
        })
        .collect();
    let outputs: Vec<RowOutputs<T>> = tasks
        .par_iter()
        .map(|(k, rows)| {
            job_rows(
                plan,
                fourier,
                &factors,
                &fourier.jobs[*k],
                &kernels[*k],
                data,
                rows,
            )
        })
        .collect();
    // Per record: (Born row, weight column, lanes).
    let mut extra: Vec<Vec<(usize, usize, &[T])>> = vec![Vec::new(); fourier.direct.len()];
    for ((k, _rows), outputs) in tasks.iter().zip(&outputs) {
        for (row, record, lanes) in outputs {
            extra[*record].push((*row, fourier.jobs[*k].column, lanes));
        }
    }
    contract(plan, &fourier.direct, Some(extra.as_slice()), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::float::{DoubleFloat, RealLike};

    fn sample(seed: usize) -> f64 {
        let x = (seed as f64 * 0.618_033_988_75).fract();
        x - 0.5
    }

    /// A synthetic plan on S_3: one regular Born orbit (components 0..6 at
    /// the ranks of h), one loop orbit of three classes with stabiliser
    /// {e, τ} (τ = (0 2 1), the classes are the cosets {g, g∘τ}, sign -1 on
    /// g∘τ: an adjoint ring's reflection), a fourth class outside the orbit
    /// and a seed key combining two classes. Weights K[c(g), b(h)] s(g) =
    /// k(h⁻¹ ∘ g) with k(x ∘ τ) = -k(x).
    fn synthetic(perturb: bool) -> (Vec<u8>, Vec<u8>) {
        let degree = 3;
        let group = 6;
        let tau = [0, 2, 1];
        let elements: Vec<Vec<usize>> = (0..group).map(|r| unrank(degree, r)).collect();
        let mut k = vec![(0i64, 0i64); group];
        for x in [0, 2, 4] {
            let xt: Vec<usize> = tau.iter().map(|&i| elements[x][i]).collect();
            let value = (3 * x as i64 + 1, 2 - x as i64);
            k[x] = value;
            k[rank(&xt)] = (-value.0, -value.1);
        }
        let mut scratch = vec![0; degree];
        let mut classes = Vec::new();
        for g in [0, 2, 4] {
            let mut terms = Vec::new();
            for h in 0..group {
                let mut inverse = vec![0; degree];
                for (i, &x) in elements[h].iter().enumerate() {
                    inverse[x] = i;
                }
                let (re, im) = k[relative(&inverse, &elements[g], &mut scratch)];
                let re = if perturb && g == 2 && h == 3 {
                    re + 1
                } else {
                    re
                };
                terms.push(format!("[{h},\"{re}/7\",\"{im}/5\"]"));
            }
            classes.push(format!("{{\"columns\":[[{}]]}}", terms.join(",")));
        }
        classes.push(r#"{"columns":[[[1,"2","-1"],[4,"1/3","0"]]]}"#.to_string());
        let plan = format!(
            r#"{{"schema":4,"legs":2,"rows":2,"helicities":[[0],[-1,1]],"columns":1,
            "average":1,"born_components":6,"constants":0,"masses":0,"block_rows":[],
            "kernels":[],"contents":[],"factors":[],"seed_classes":[{}],
            "seeds":[{{"classes":[[0,"1","0"]]}},{{"classes":[[1,"1","0"]]}},
                {{"classes":[[2,"-1/2","3"],[3,"1","0"]]}}],
            "families":[],"forward":null}}"#,
            classes.join(",")
        );
        let record = r#"{"schema":1,"degree":3,"jobs":[{"column":0,
            "born":[[0,1,2,3,4,5]],
            "loop":[[[0,1],[0,-1],[1,1],[1,-1],[2,1],[2,-1]]],
            "pairs":[[0,0]]}]}"#;
        (plan.into_bytes(), record.as_bytes().to_vec())
    }

    fn batch<T: Carrier>(zero: &T) -> super::super::dress::V2Batch<T> {
        use symbolica::domains::float::Complex;
        let value = |i: usize| {
            Complex::new(
                zero.from_rational(&Rational::new(
                    Integer::from((7 * i % 11) as i64 - 5),
                    Integer::from(3),
                )),
                zero.from_rational(&Rational::new(
                    Integer::from((5 * i % 13) as i64 - 6),
                    Integer::from(7),
                )),
            )
        };
        super::super::dress::V2Batch {
            momenta: vec![vec![Rational::zero(); 8]; 2],
            scales: vec![Rational::from(1); 2],
            masses: Vec::new(),
            masses_imaginary: Vec::new(),
            leg_masses: Vec::new(),
            constants: Vec::new(),
            block_rows: vec![Vec::new(); 2],
            born: (0..2)
                .map(|p| (0..12).map(|i| value(12 * p + i + 1)).collect())
                .collect(),
        }
    }

    fn assert_seeds_agree<T: Carrier + Real + RealLike>(zero: &T, tolerance: f64) {
        let (plan, record) = synthetic(false);
        let plan = LoopProgramV4::load(&plan).unwrap();
        let fourier = SeedFourier::load(&record, &plan).unwrap();
        assert_eq!(
            (fourier.jobs(), fourier.pairs(), fourier.covered()),
            (1, 1, 18)
        );
        let batch = batch(zero);
        let data = LaneData::<T, 2>::new(&plan, &batch, &[0, 1], zero).unwrap();
        let direct = super::super::seeds::seeds(&plan, &data);
        let fast = seeds_fourier(&plan, &fourier, &data);
        for (a, b) in fast.values.iter().zip(&direct.values) {
            let scale = b.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max);
            assert!(scale > 0.0);
            for (x, y) in a.iter().zip(b) {
                let error = (x.clone() - y).to_f64().abs();
                assert!(error <= tolerance * scale, "{error} vs {scale}");
            }
        }
    }

    /// FFT seeds equal the direct seeds of the same plan (binary64 and
    /// double-double, stabilisers and signs included), and a plan whose
    /// weights break the group function refuses the record.
    #[test]
    fn certified_seeds_equal_the_direct_contraction() {
        assert_seeds_agree(&0.0, 1e-14);
        assert_seeds_agree(&DoubleFloat::from(0.0), 1e-30);
        let (plan, record) = synthetic(true);
        let plan = LoopProgramV4::load(&plan).unwrap();
        let error = SeedFourier::load(&record, &plan).unwrap_err().to_string();
        assert!(error.contains("not the group function"), "{error}");
        // An orbit of the wrong size is refused.
        let (plan, _) = synthetic(false);
        let plan = LoopProgramV4::load(&plan).unwrap();
        let short = br#"{"schema":1,"degree":3,"jobs":[{"column":0,"born":[[0,1,2]],
            "loop":[[[0,1],[0,-1],[1,1],[1,-1],[2,1],[2,-1]]],"pairs":[[0,0]]}]}"#;
        assert!(SeedFourier::load(short, &plan).is_err());
    }

    #[test]
    fn ranks_invert_unranks() {
        for degree in 1..=5 {
            let order: usize = (1..=degree).product();
            for r in 0..order {
                assert_eq!(rank(&unrank(degree, r)), r);
            }
        }
    }

    /// The carrier transforms equal the tree engine's binary64 transforms,
    /// and the inverse undoes the forward one (double-double to its rounding).
    #[test]
    fn carrier_transforms_match_the_tree_engine() {
        for degree in 2..=6 {
            let fft = SymmetricGroupFftPlan::new(degree).unwrap();
            let group = fft.order();
            let lanes = 3;
            let values: Vec<f64> = (0..group * lanes).map(|i| sample(7 * i + 1)).collect();
            let factors = Factors::new(&fft, &0.0);
            let mut work = Workspace::new(group, lanes, &0.0);
            let mut ours = vec![0.0; group * lanes];
            forward(&fft, &factors, lanes, &values, &mut ours, &mut work, &0.0);
            let pairs: Vec<(f64, f64)> = values.iter().map(|&x| (x, 0.0)).collect();
            let mut theirs = vec![(0.0, 0.0); group * lanes];
            let mut workspace = fft.workspace(lanes).unwrap();
            fft.forward_lanes(lanes, &pairs, &mut theirs, &mut workspace)
                .unwrap();
            for (a, b) in ours.iter().zip(&theirs) {
                assert!((a - b.0).abs() <= 1e-13, "degree {degree}: {a} {b:?}");
            }
            let mut back = vec![0.0; group * lanes];
            inverse(&fft, &factors, lanes, &ours, &mut back, &mut work, &0.0);
            for (a, b) in back.iter().zip(&values) {
                assert!((a - b).abs() <= 1e-13, "degree {degree}");
            }
            // Double-double: the round trip to its own rounding.
            let zero = DoubleFloat::from(0.0);
            let factors = Factors::new(&fft, &zero);
            let values: Vec<DoubleFloat> = values.iter().map(|&x| DoubleFloat::from(x)).collect();
            let mut work = Workspace::new(group, lanes, &zero);
            let mut coefficients = vec![zero; group * lanes];
            forward(
                &fft,
                &factors,
                lanes,
                &values,
                &mut coefficients,
                &mut work,
                &zero,
            );
            let mut back = vec![zero; group * lanes];
            inverse(
                &fft,
                &factors,
                lanes,
                &coefficients,
                &mut back,
                &mut work,
                &zero,
            );
            for (a, b) in back.iter().zip(&values) {
                let error = (a.clone() - b).to_f64().abs();
                assert!(error <= 1e-29, "degree {degree}: {error}");
            }
        }
    }

    /// `F[f ⋆ k] = F[f] F[k]` with `(f ⋆ k)(g) = Σ_h f(h) k(h⁻¹ ∘ g)` and the
    /// block product of [`job_rows`].
    #[test]
    fn block_products_are_group_convolutions() {
        for degree in 2..=5 {
            let fft = SymmetricGroupFftPlan::new(degree).unwrap();
            let group = fft.order();
            let factors = Factors::new(&fft, &0.0);
            let elements: Vec<Vec<usize>> = (0..group).map(|r| unrank(degree, r)).collect();
            let f: Vec<f64> = (0..group).map(|i| sample(3 * i + 2)).collect();
            let k: Vec<f64> = (0..group).map(|i| sample(11 * i + 5)).collect();
            let mut work = Workspace::new(group, 1, &0.0);
            let (mut fh, mut kh) = (vec![0.0; group], vec![0.0; group]);
            forward(&fft, &factors, 1, &f, &mut fh, &mut work, &0.0);
            forward(&fft, &factors, 1, &k, &mut kh, &mut work, &0.0);
            let mut product = vec![0.0; group];
            for block in fft.blocks() {
                let (d, offset) = (block.dimension(), block.coefficient_offset());
                for column in 0..d {
                    for r in 0..d {
                        for inner in 0..d {
                            product[offset + column * d + r] +=
                                fh[offset + inner * d + r] * kh[offset + column * d + inner];
                        }
                    }
                }
            }
            let mut convolution = vec![0.0; group];
            inverse(
                &fft,
                &factors,
                1,
                &product,
                &mut convolution,
                &mut work,
                &0.0,
            );
            let mut scratch = vec![0; degree];
            for g in 0..group {
                let mut direct = 0.0;
                for h in 0..group {
                    let mut inverse_h = vec![0; degree];
                    for (i, &x) in elements[h].iter().enumerate() {
                        inverse_h[x] = i;
                    }
                    direct += f[h] * k[relative(&inverse_h, &elements[g], &mut scratch)];
                }
                assert!(
                    (convolution[g] - direct).abs() <= 1e-12,
                    "degree {degree} g {g}"
                );
            }
        }
    }
}
