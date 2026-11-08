// SPDX-License-Identifier: 0BSD

//! The loop engine's two structural bounds, each with one source of truth.
//!
//! - External legs are one bit each of a `u64` leg mask: every leg-count
//!   check (DAG input, families, plan, tree program, dressing) admits
//!   `1..=`[`MAX_LEGS`] through [`full_mask`] / [`check_legs`].
//! - A family's numerator rank is admitted by the reduction backends
//!   themselves ([`rank_refusal`] per [`Reducer`]: NinjaRS's
//!   [`ninjars::RankClass::of`], rank <= N + 2; TIRust's capability query,
//!   `iregi_rust::DirectCapabilities`, with the cargo feature
//!   `tirust-capability`, which the loop crates' `tir` turns on). An
//!   artifact admits a family that one of them reduces ([`admission_refusal`]:
//!   the plan loader, the binary member bound and the Python generator,
//!   `_loop.reducer_rank_refusal`); a family beyond NinjaRS's rank makes the
//!   artifact TIR-only (m-lit lit-eft: [`tir_only`]), which NinjaRS refuses
//!   typed when its backend is built for it. No rank limit is stated here.

/// The most external legs a `u64` leg mask holds.
pub const MAX_LEGS: usize = u64::BITS as usize;

/// The mask of all `legs` legs, or `None` outside `1..=MAX_LEGS`.
pub const fn full_mask(legs: usize) -> Option<u64> {
    if legs == 0 || legs > MAX_LEGS {
        None
    } else if legs == MAX_LEGS {
        Some(u64::MAX)
    } else {
        Some((1u64 << legs) - 1)
    }
}

/// The refusal of a leg count outside `1..=MAX_LEGS`.
pub fn leg_count_refusal(legs: usize) -> String {
    format!(
        "a process of {legs} external legs: the loop engine's u64 leg masks hold 1 to {MAX_LEGS} legs"
    )
}

/// [`full_mask`], or [`leg_count_refusal`].
pub fn check_legs(legs: usize) -> Result<u64, String> {
    full_mask(legs).ok_or_else(|| leg_count_refusal(legs))
}

/// A reduction backend, for the rank admission of its family numerators
/// (m-lit PLAN §2.b.7: the single source of truth takes the backend), and
/// the runtime's backend choice (`rusticol_loop_reduction::runtime::
/// Reduction`). The discriminants index the runtime's backend slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reducer {
    /// NinjaRS (integrand reduction): rank <= N + 2 ([`ninjars::RankClass`]).
    NinjaRs = 0,
    /// TIRust (tensor-integral reduction): no rank <= N + 2 limit; what it
    /// reduces is its own capability query ([`rank_refusal`]), refused
    /// without the cargo feature `tirust-capability` (the loop crates' `tir`).
    TiRust = 1,
}

impl Reducer {
    /// Every reduction backend, in slot order.
    pub const ALL: [Self; 2] = [Self::NinjaRs, Self::TiRust];

    /// The backend of `name`: `"ninjars"` or `"tirust"`.
    pub fn parse(name: &str) -> crate::RusticolResult<Self> {
        match name {
            "ninjars" => Ok(Self::NinjaRs),
            "tirust" => Ok(Self::TiRust),
            other => Err(crate::RusticolError::invalid_argument(format!(
                "reduction must be ninjars or tirust, not {other}"
            ))),
        }
    }

    /// The backend's name (`parse`'s input).
    pub fn name(self) -> &'static str {
        match self {
            Self::NinjaRs => "ninjars",
            Self::TiRust => "tirust",
        }
    }
}

/// `None` if `reducer` reduces a numerator of rank `rank` over `propagators`
/// loop propagators, else the refusal naming the reducer's actual
/// capability: NinjaRS's rank classes (rank <= N + 2 over N >= 1); TIRust's
/// capability query at binary64 (`DirectCapabilities::of_representation`:
/// its subset masks and monomial indices; the TIR backend also asks it per
/// carrier with its own settings, `rusticol_loop_reduction::v2_tir::
/// capability_refusal`), or the refusal of a build without TIRust.
pub fn rank_refusal(reducer: Reducer, propagators: usize, rank: usize) -> Option<String> {
    match reducer {
        Reducer::NinjaRs
            if propagators > 0 && ninjars::RankClass::of(propagators, rank).is_some() =>
        {
            None
        }
        Reducer::NinjaRs => Some(format!(
            "numerator rank {rank} over {propagators} loop propagators: NinjaRS reduces rank <= N+2 \
             over N >= 1 propagators"
        )),
        Reducer::TiRust => tirust_refusal(propagators, rank),
    }
}

#[cfg(feature = "tirust-capability")]
fn tirust_refusal(propagators: usize, rank: usize) -> Option<String> {
    let refused = |why: String| {
        Some(format!(
            "TIRust cannot reduce {propagators} propagators at rank {rank}: {why}"
        ))
    };
    let Ok(settings_rank) = u16::try_from(rank) else {
        return refused("its settings hold a u16 rank".to_owned());
    };
    let settings = iregi_rust::ReductionSettings::new(propagators, settings_rank);
    iregi_rust::DirectCapabilities::of_representation(f64::MANTISSA_DIGITS)
        .admits(&settings)
        .err()
        .and_then(|error| refused(error.to_string()))
}

#[cfg(not(feature = "tirust-capability"))]
fn tirust_refusal(propagators: usize, rank: usize) -> Option<String> {
    Some(format!(
        "TIRust cannot reduce {propagators} propagators at rank {rank}: this build has no TIRust \
         (the loop crates' cargo feature tir)"
    ))
}

/// The artifacts' admission of a family: `None` if a reduction backend of
/// this build reduces it ([`rank_refusal`]), else every backend's refusal.
pub fn admission_refusal(propagators: usize, rank: usize) -> Option<String> {
    let mut refusals = Vec::with_capacity(Reducer::ALL.len());
    for reducer in Reducer::ALL {
        refusals.push(rank_refusal(reducer, propagators, rank)?);
    }
    Some(refusals.join("; "))
}

/// Whether an artifact of families `(propagators, rank)` is TIR-only: one of
/// them is beyond NinjaRS's rank (m-lit lit-eft; the artifacts admit it
/// through TIRust, [`admission_refusal`]). NinjaRS refuses such an artifact
/// typed ([`refused_families`]); its declarations record it.
pub fn tir_only(families: impl IntoIterator<Item = (usize, usize)>) -> bool {
    refused_families(Reducer::NinjaRs, families).is_some()
}

/// `reducer`'s refusal of the first of `families` `(propagators, rank)` it
/// does not reduce, or `None` when it reduces every one.
pub fn refused_families(
    reducer: Reducer,
    families: impl IntoIterator<Item = (usize, usize)>,
) -> Option<String> {
    families
        .into_iter()
        .find_map(|(propagators, rank)| rank_refusal(reducer, propagators, rank))
}

/// The largest numerator rank NinjaRS admits over `propagators` loop
/// propagators (`propagators + 2`), from [`rank_refusal`]; the excess above
/// the propagators is a `u8` in NinjaRS's rank classes. The binary member
/// bound takes it over a plan's legs, raised to the plan's largest family
/// rank (a TIR-only plan's).
pub fn largest_rank(propagators: usize) -> usize {
    let mut rank = propagators;
    for _ in 0..=u8::MAX {
        if rank_refusal(Reducer::NinjaRs, propagators, rank + 1).is_some() {
            break;
        }
        rank += 1;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_leg_boundary() {
        assert_eq!(full_mask(0), None);
        assert_eq!(full_mask(1), Some(1));
        assert_eq!(full_mask(63), Some(u64::MAX >> 1));
        assert_eq!(full_mask(64), Some(u64::MAX));
        assert_eq!(full_mask(65), None);
        assert!(check_legs(65).unwrap_err().contains("65 external legs"));
    }

    #[test]
    fn the_reducers_admit_the_rank() {
        for n in 1..12 {
            for rank in 0..=n + 2 {
                assert_eq!(rank_refusal(Reducer::NinjaRs, n, rank), None);
                assert_eq!(admission_refusal(n, rank), None);
            }
            assert!(rank_refusal(Reducer::NinjaRs, n, n + 3).is_some());
        }
        assert!(rank_refusal(Reducer::NinjaRs, 0, 0).is_some());
        assert!(admission_refusal(0, 0).is_some());
        assert_eq!(largest_rank(4), 6);
        assert_eq!(largest_rank(64), 66);
        let refusal = rank_refusal(Reducer::NinjaRs, 4, 7).unwrap();
        assert!(refusal.contains("NinjaRS reduces rank <= N+2"), "{refusal}");
        assert!(!tir_only([(4, 6), (5, 7), (1, 0)]));
        assert!(tir_only([(4, 6), (5, 8)]));
        let refused = refused_families(Reducer::NinjaRs, [(4, 6), (5, 8)]).unwrap();
        assert!(refused.contains("rank 8 over 5"), "{refused}");
    }

    /// With TIRust (the cargo feature `tirust-capability`) a family beyond NinjaRS's rank
    /// is admitted through TIRust's capability query, up to its
    /// representation; without it the artifacts admit NinjaRS's ranks only.
    #[test]
    fn tirust_admits_ranks_beyond_n_plus_two_by_its_capability_query() {
        let refusal = rank_refusal(Reducer::TiRust, 0, 0).unwrap();
        assert!(refusal.contains("TIRust"), "{refusal}");
        if cfg!(feature = "tirust-capability") {
            for n in 1..12 {
                for rank in 0..=n + 6 {
                    assert_eq!(rank_refusal(Reducer::TiRust, n, rank), None);
                }
            }
            assert_eq!(admission_refusal(4, 7), None);
            assert_eq!(admission_refusal(5, 8), None);
            // Beyond TIRust's own representation (and its u16 settings).
            for rank in [100_000, 1_000_000] {
                let refusal = admission_refusal(1, rank).unwrap();
                assert!(
                    refusal.contains("NinjaRS") && refusal.contains("TIRust"),
                    "{refusal}"
                );
            }
            let refusal = rank_refusal(Reducer::TiRust, 65, 0).unwrap();
            assert!(refusal.contains("subset masks"), "{refusal}");
        } else {
            let refusal = admission_refusal(4, 7).unwrap();
            assert!(refusal.contains("no TIRust"), "{refusal}");
        }
    }
}
