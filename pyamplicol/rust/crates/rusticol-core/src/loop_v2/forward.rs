// SPDX-License-Identifier: 0BSD

//! Schedule F: the forward prefix-tree executor (DESIGN.md §2.2), baseline
//! and oracle of schedule B.
//!
//! States are dressed from the cut with helicity-explicit rows of their
//! attached legs, starting from the unit seed `δ_ab`; prefixes are shared
//! across families. Each closure contracts the state with its skeleton's seed,
//! `w Σ_h U0[o][h] Tr O[h]`, maps it into the family's trie frame with the
//! exact affine relabelling and adds it to the family accumulator, so F and B
//! produce the same numerators in the same frames.

use super::closure::{FamilyNumerator, frame_shift, substitute};
use super::dress::{Carrier, FactorStore, LaneData, Pool, cadd, cmul_into, dress_edge};
use super::kernels::lane_vector_len;
use super::plan::LoopProgramV4;
use super::seeds::Seeds;
use crate::{RusticolError, RusticolResult};
use ninjars::fast::tensor_layout::ncoeff;

/// The factors schedule F reads, ascending.
pub fn forward_factors(plan: &LoopProgramV4) -> Vec<usize> {
    let mut out: Vec<usize> = plan
        .forward
        .iter()
        .flatten()
        .map(|state| state.factor)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Run schedule F on one chunk: every family's numerators in its trie frame.
pub fn run_forward<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    factors: &FactorStore<T>,
    seeds: &Seeds<T>,
    data: &LaneData<T, L>,
) -> RusticolResult<(Vec<FamilyNumerator<T>>, usize)> {
    let states = plan
        .forward
        .as_ref()
        .ok_or_else(|| RusticolError::invalid_argument("the plan has no schedule F"))?;
    let columns = plan.columns;
    let v = lane_vector_len(L);
    let zero = &data.zero;
    let mut numerators: Vec<FamilyNumerator<T>> = plan
        .families
        .iter()
        .enumerate()
        .map(|(index, family)| FamilyNumerator::zeros::<L>(index, family, columns, zero))
        .collect();
    let mut unit = vec![zero.clone(); v];
    for x in &mut unit[..L] {
        *x = zero.one();
    }
    let mut pool = Pool::new(zero);
    let mut outputs: Vec<Option<Vec<T>>> = (0..states.len()).map(|_| None).collect();
    let mut remaining: Vec<usize> = states.iter().map(|state| state.children).collect();
    let mut factor = vec![zero.clone(); v];
    let mut term = vec![zero.clone(); v];
    for (s, state) in states.iter().enumerate() {
        let content = &plan.contents[state.content];
        let mut out = pool.take(state.rows * state.out_row() * v);
        {
            let input: &[T] = match state.parent {
                None => &unit,
                Some(parent) => outputs[parent]
                    .as_deref()
                    .ok_or_else(|| RusticolError::internal("a released F state is read"))?,
            };
            dress_edge::<T, L>(
                content,
                state.seeded,
                state.d_alpha,
                &plan.products[state.products],
                state.rank,
                input,
                factors.get(state.factor),
                &state.links,
                &mut out,
            );
        }
        if let Some(parent) = state.parent {
            remaining[parent] -= 1;
            if remaining[parent] == 0 {
                let buffer = outputs[parent].take().expect("live parent");
                pool.give(buffer);
            }
        }
        let n = ncoeff(state.rank);
        let row = state.out_row() * v;
        for closure in &state.closures {
            let weight = data.cast(&closure.weight);
            let seed = &seeds.values[closure.seed];
            let family = &plan.families[closure.family];
            let n_family = ncoeff(family.rank);
            for o in 0..columns {
                let mut column = vec![zero.clone(); n * v];
                for (r, born) in closure.rows.iter().enumerate() {
                    let Some(h) = *born else { continue };
                    for (f, x) in factor.iter_mut().zip(&seed[(h * columns + o) * v..][..v]) {
                        *f = weight.clone() * x;
                    }
                    let values = &out[r * row..][..row];
                    for a in 0..state.d_alpha {
                        let diagonal = &values[(a * state.d_out + a) * n * v..][..n * v];
                        for (c, x) in column.chunks_exact_mut(v).zip(diagonal.chunks_exact(v)) {
                            cmul_into::<T, L>(&mut term, &factor, x);
                            cadd::<T, L>(c, &term);
                        }
                    }
                }
                if let Some((sign, mask)) = closure.relabel {
                    let shift = frame_shift(data, mask);
                    column = substitute::<T, L>(&column, state.rank, sign, &shift, zero);
                }
                if column[n_family.min(n) * v..]
                    .iter()
                    .any(|x| !x.is_fully_zero())
                {
                    return Err(RusticolError::evaluation(
                        "family rank truncates a nonzero coefficient",
                    ));
                }
                let target = &mut numerators[closure.family].accumulators[closure.accumulator]
                    [o * n_family * v..][..n_family * v];
                for (t, c) in target.chunks_exact_mut(v).zip(column.chunks_exact(v)) {
                    cadd::<T, L>(t, c);
                }
            }
        }
        if state.children > 0 {
            outputs[s] = Some(out);
        } else {
            pool.give(out);
        }
    }
    Ok((numerators, pool.peak))
}
