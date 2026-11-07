// SPDX-License-Identifier: 0BSD

//! The loop engine's two structural bounds, each with one source of truth.
//!
//! - External legs are one bit each of a `u64` leg mask: every leg-count
//!   check (DAG input, families, plan, tree program, dressing) admits
//!   `1..=`[`MAX_LEGS`] through [`full_mask`] / [`check_legs`].
//! - A family's numerator rank is admitted by the reduction backend itself
//!   ([`Reducer`]: NinjaRS's [`ninjars::RankClass::of`]); the plan loader,
//!   the binary member bound, both runtime backends and the Python generator
//!   (`_loop.reducer_rank_refusal`) all ask [`rank_refusal`].

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
/// (m-lit PLAN §2.b.7: the single source of truth takes the backend).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reducer {
    /// NinjaRS (integrand reduction): rank <= N + 2 ([`ninjars::RankClass`]).
    NinjaRs,
    /// TIRust (tensor-integral reduction). Its own capability is what its
    /// plan construction serves (the TIR backend refuses typed with
    /// TIRust's reason); until artifacts admit families beyond rank N + 2
    /// (m-lit `lit-eft`: TIR-only artifacts) its families are held to the
    /// artifacts' admission, NinjaRS's, so that every artifact serves both
    /// backends.
    TiRust,
}

impl Reducer {
    /// The backend whose admission every artifact's families satisfy (every
    /// artifact admits both backends until `lit-eft`): the plan loader, the
    /// binary member bound and the generator ask it.
    pub const ARTIFACT: Self = Self::NinjaRs;
}

/// `None` if `reducer` reduces a numerator of rank `rank` over `propagators`
/// loop propagators, else the refusal naming the reducer's actual
/// capability.
pub fn rank_refusal(reducer: Reducer, propagators: usize, rank: usize) -> Option<String> {
    match reducer {
        Reducer::NinjaRs | Reducer::TiRust
            if propagators > 0 && ninjars::RankClass::of(propagators, rank).is_some() =>
        {
            None
        }
        Reducer::NinjaRs => Some(format!(
            "numerator rank {rank} over {propagators} loop propagators: NinjaRS reduces rank <= N+2 \
             over N >= 1 propagators"
        )),
        Reducer::TiRust => Some(format!(
            "numerator rank {rank} over {propagators} loop propagators: artifacts admit rank <= N+2 \
             over N >= 1 propagators for both reduction backends (NinjaRS's capability)"
        )),
    }
}

/// The largest numerator rank `reducer` admits over `propagators` loop
/// propagators (NinjaRS: `propagators + 2`), from [`rank_refusal`]; the
/// excess above the propagators is a `u8` in NinjaRS's rank classes.
pub fn largest_rank(reducer: Reducer, propagators: usize) -> usize {
    let mut rank = propagators;
    for _ in 0..=u8::MAX {
        if rank_refusal(reducer, propagators, rank + 1).is_some() {
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
    fn the_reducer_admits_the_rank() {
        for reducer in [Reducer::NinjaRs, Reducer::TiRust] {
            for n in 1..12 {
                for rank in 0..=n + 2 {
                    assert_eq!(rank_refusal(reducer, n, rank), None);
                }
                assert!(rank_refusal(reducer, n, n + 3).is_some());
            }
            assert!(rank_refusal(reducer, 0, 0).is_some());
            assert_eq!(largest_rank(reducer, 4), 6);
            assert_eq!(largest_rank(reducer, 64), 66);
        }
        let refusal = rank_refusal(Reducer::NinjaRs, 4, 7).unwrap();
        assert!(refusal.contains("NinjaRS reduces rank <= N+2"), "{refusal}");
        let refusal = rank_refusal(Reducer::TiRust, 4, 7).unwrap();
        assert!(refusal.contains("for both reduction backends"), "{refusal}");
        assert_eq!(Reducer::ARTIFACT, Reducer::NinjaRs);
    }
}
