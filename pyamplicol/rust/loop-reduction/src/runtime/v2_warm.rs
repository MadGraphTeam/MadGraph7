// SPDX-License-Identifier: 0BSD

//! The warm entry of a reduction backend (`V2Executor.warm`): the classes a
//! plan's first reductions reach ([`warm_classes`]) and the build of their
//! engines ahead of the first evaluation ([`warm`]).

use super::v2_execution::{Reduction, TypedBatch, V2Backends, plan_histogram, rank_refusal};
use super::v2_reduce::Poles;
use crate::Complex;
use rusticol_core::RusticolResult;
use rusticol_core::loop_execution::v2::{FamilyLanes, V2Backend};

/// The (propagators, rank) classes whose engines `reduction`'s first
/// reductions of `plan` build ([`warm`]): TIRust's are the plan's declared
/// classes (its plans); NinjaRS buckets a lane by its true rank (the highest
/// monomial degree with a nonzero coefficient, `true_rank`), and a family's
/// accumulator holds a polynomial of at most the largest output rank of the
/// trace steps closing into it, below the family's declared rank when no
/// such step reaches it: per family and accumulator that rank. A dressed-row
/// plan's lanes (amplitude mode) fall below it too (measured: g g > Z Z and
/// g g > h h build lower classes on their first reduction): every rank up to
/// it. A lane whose top coefficients vanish otherwise falls into a lower
/// class, built on its first use. Classes the backend refuses are left out
/// (an evaluation refuses them).
pub fn warm_classes(
    plan: &rusticol_core::loop_v2::plan::LoopProgramV4,
    reduction: Reduction,
) -> std::collections::BTreeSet<(usize, usize)> {
    let classes: std::collections::BTreeSet<(usize, usize)> = match reduction {
        Reduction::TiRust => plan_histogram(plan),
        Reduction::NinjaRs => plan
            .families
            .iter()
            .flat_map(|family| {
                let mut top: Vec<Option<usize>> = vec![None; family.orders.len()];
                for step in &family.steps {
                    if let Some(k) = step.trace {
                        top[k] = top[k].max(Some(step.rank_out));
                    }
                }
                let n = family.propagators();
                let lowest = |rank: usize| if plan.dressed { 0 } else { rank };
                top.into_iter()
                    .flatten()
                    .flat_map(move |rank| (lowest(rank)..=rank).map(move |r| (n, r)))
            })
            .collect(),
    };
    classes
        .into_iter()
        .filter(|&(n, rank)| rank_refusal(reduction, n, rank).is_none())
        .collect()
}

/// Build now what a backend's first reductions of the (propagators, rank)
/// classes `classes` ([`warm_classes`]) would build lazily, in the carrier
/// of `zero`, without reducing a point of the process (the warm entry,
/// `V2Executor.warm`): NinjaRS's engine of each class (binary64: its fast
/// amplitude and tensor numerator; the extended carriers: its generic
/// engine, whose cut expansions are built by the first evaluation), TIRust's
/// plans and workspaces. One synthetic lane per class (a fixed generic
/// massive configuration with exact rational inputs cast to the carrier,
/// unit-size coefficients) runs through the backend's own `reduce` in its
/// current mode, and its outcome is dropped; the master cache and the
/// counters it filled are reset after (`begin`, `take_statistics`,
/// `take_lane_measures`), so the next call reports and caches only its own
/// lanes. A later reduction's values do not depend on it: every lane's
/// outcome is a function of its own inputs (the engines' caches are keyed by
/// them). Not built: the binary64 fast engine's exact fallback (its lanes
/// stay below the absolute thresholds that route a lane to it).
pub fn warm<T>(backend: &mut V2Backends, classes: &[(usize, usize)], zero: &T) -> RusticolResult<()>
where
    T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
    V2Backends: V2Backend<T>,
{
    let groups = classes
        .iter()
        .map(|&(n, rank)| synthetic_lane(n, rank, zero, 0))
        .collect::<RusticolResult<Vec<_>>>()?;
    V2Backend::<T>::begin(backend);
    // One class per call: a class the backend refuses (its typed refusal)
    // is the caller's error, as in an evaluation.
    for group in &groups {
        V2Backend::<T>::reduce(backend, std::slice::from_ref(group), Poles::All, None)?;
    }
    V2Backend::<T>::begin(backend);
    let _ = V2Backend::<T>::take_statistics(backend);
    let _ = V2Backend::<T>::take_lane_measures(backend);
    Ok(())
}

/// One synthetic reduction lane of `n` propagators and rank `rank` in the
/// carrier of `zero` ([`warm`]): shifts `q_0 = 0` and `q_k` in general
/// position (`variant` moves them), distinct positive squared masses, the
/// invariants `(q_i - q_j)^2`, unit scale and `O(1)` coefficients, every
/// value an exact rational cast to the carrier (at its precision).
fn synthetic_lane<T>(
    n: usize,
    rank: usize,
    zero: &T,
    variant: i64,
) -> RusticolResult<FamilyLanes<T>>
where
    T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
{
    use ninjars::fast::tensor_layout::ncoeff;
    use rusticol_core::loop_execution::v2::LaneKey;
    use rusticol_core::loop_v2::dress::{cast, parse_rational};
    use symbolica::domains::rational::Rational;

    let exact = |numerator: i64, denominator: i64| -> RusticolResult<T> {
        Ok(cast(
            zero,
            &parse_rational(&format!("{numerator}/{denominator}"))?,
        ))
    };
    let complex = |re: T| Complex::new(re, zero.clone());
    let shift = |k: usize, mu: usize| -> (i64, i64) {
        if k == 0 {
            return (0, 1);
        }
        let k = k as i64 + variant;
        match mu {
            0 => (3 * k * k + 5 * k + 2, 7),
            1 => (2 * k + 1, 3 + k),
            2 => (5 - 3 * k, 11),
            _ => (k * k - 4, 13 + 2 * k),
        }
    };
    let mut shifts = Vec::with_capacity(4 * n);
    for k in 0..n {
        for mu in 0..4 {
            let (a, b) = shift(k, mu);
            shifts.push(complex(exact(a, b)?));
        }
    }
    let mut s_matrix = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            // (q_i - q_j)^2 exactly: a rational of the shifts.
            let mut square = Rational::from(0);
            for mu in 0..4 {
                let (a, b) = shift(i, mu);
                let (c, d) = shift(j, mu);
                let difference = parse_rational(&format!("{}/{}", a * d - c * b, b * d))?;
                let term = difference.clone() * &difference;
                square += &if mu == 0 {
                    term
                } else {
                    Rational::from(0) - term
                };
            }
            s_matrix.push(complex(cast(zero, &square)));
        }
    }
    let masses = (0..n)
        .map(|k| Ok(complex(exact(17 + 6 * (k as i64 + variant), 13)?)))
        .collect::<RusticolResult<Vec<_>>>()?;
    let coefficients = (0..ncoeff(rank))
        .map(|c| {
            let c = c as i64 + variant;
            Ok(complex(exact(1 + c % 5, 2 + c % 3)?))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(FamilyLanes {
        family: 0,
        propagators: n,
        rank,
        points: vec![0],
        keys: vec![LaneKey {
            family: 0,
            order: 0,
            column: 0,
        }],
        shifts,
        masses,
        scales: vec![complex(exact(1, 1)?)],
        s_matrix,
        coefficients,
    })
}

#[cfg(test)]
mod warm_tests {
    use super::*;
    use symbolica::domains::float::DoubleFloat;

    /// A NinjaRS backend as the runtime builds it (`V2Precision::backend`:
    /// binary64's fast engine, double-double's generic one, every master
    /// pole kept).
    fn ninjars(precision: crate::v2_runtime::V2Precision) -> V2Backends {
        let plan = rusticol_core::loop_v2::plan::LoopProgramV4::load(
            br#"{"schema":4,"legs":1,"rows":1,"helicities":[[1]],"columns":1,"average":1,
                "born_components":1,"constants":0,"masses":1,"block_rows":[],"kernels":[],
                "contents":[],"factors":[],"seed_classes":[],"seeds":[],"families":[],
                "forward":null,"merged":null}"#,
        )
        .expect("plan");
        precision
            .backend(
                Reduction::NinjaRs,
                &plan,
                crate::reduction_policy::DEFAULT_TARGET,
            )
            .expect("backend")
    }

    /// The outcomes of `lanes` reduced by `backend`, as text (exact: the
    /// carriers' `Debug` forms).
    fn reduced<T>(backend: &mut V2Backends, lanes: &[FamilyLanes<T>]) -> String
    where
        T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
        V2Backends: V2Backend<T>,
    {
        V2Backend::<T>::begin(backend);
        let out = V2Backend::<T>::reduce(backend, lanes, Poles::All, None).expect("reduce");
        format!("{:?}", out.0)
    }

    fn check<T>(
        make: impl Fn() -> V2Backends,
        zero: &T,
        classes: &[(usize, usize)],
        lower: bool,
        sensitive: bool,
    ) where
        T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
        V2Backends: V2Backend<T>,
    {
        // Other configurations than the warm lanes', declared and lower ranks.
        let lanes: Vec<FamilyLanes<T>> = classes
            .iter()
            .flat_map(|&(n, rank)| {
                let below = (lower && rank > 0).then(|| (n, rank - 1));
                std::iter::once((n, rank)).chain(below)
            })
            .map(|(n, rank)| synthetic_lane(n, rank, zero, 3).expect("lane"))
            .collect();
        let mut plain = make();
        let cold = reduced(&mut plain, &lanes);
        // A reduction fills the master cache and the counters (binary64 fast
        // engine): what the warm must leave empty.
        let (masters, statistics, _) = residue::<T>(&mut plain);
        if sensitive {
            assert!(masters > 0 && statistics > 0, "{masters} {statistics}");
        }
        let mut warmed = make();
        warm(&mut warmed, classes, zero).expect("warm");
        assert_eq!(residue::<T>(&mut warmed), (0, 0, 0), "the warm resets");
        assert_eq!(reduced(&mut warmed, &lanes), cold);
        assert!(cold.contains("Ok("), "the test lanes reduce: {cold}");
    }

    /// The cached master rows (NinjaRS), the counters and seconds taken and
    /// the lane measures taken, of a backend.
    fn residue<T>(backend: &mut V2Backends) -> (usize, usize, usize)
    where
        T: TypedBatch,
        V2Backends: V2Backend<T>,
    {
        let masters = match &*backend {
            V2Backends::NinjaRs(backend) => backend.reducer.cached_masters(),
            #[cfg(feature = "tir")]
            V2Backends::TiRust(_) => 0,
        };
        // Counters and seconds that recorded something (a backend may list
        // its counters at zero).
        let statistics = V2Backend::<T>::take_statistics(backend);
        let recorded = statistics.counts.iter().filter(|(_, n)| *n > 0).count()
            + statistics.seconds.iter().filter(|(_, s)| *s > 0.0).count();
        let measures = V2Backend::<T>::take_lane_measures(backend).len();
        (masters, recorded, measures)
    }

    #[test]
    fn a_warmed_backend_reduces_bit_identically() {
        let _unlock = pyamplicol::unlock();
        let classes = [(2, 2), (3, 3), (4, 4)];
        // NinjaRS buckets a lane by its true rank: lanes below the
        // declared ranks too.
        check(
            || ninjars(crate::v2_runtime::V2Precision::Double),
            &0.0,
            &classes,
            true,
            true,
        );
        check(
            || ninjars(crate::v2_runtime::V2Precision::DoubleDouble),
            &DoubleFloat::from(0.0),
            &classes,
            true,
            false,
        );
        #[cfg(feature = "tir")]
        {
            let histogram: std::collections::BTreeSet<_> = classes.into_iter().collect();
            let tir = || {
                V2Backends::tirust(
                    histogram.clone(),
                    53,
                    crate::reduction_policy::DEFAULT_TARGET,
                )
                .expect("tirust")
            };
            check(tir, &0.0, &classes, false, false);
        }
    }
}
