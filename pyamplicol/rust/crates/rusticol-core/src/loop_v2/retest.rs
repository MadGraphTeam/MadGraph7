// SPDX-License-Identifier: 0BSD

//! The reduction-direction re-test (DESIGN.md §10.2 item 2,
//! REQUIREMENTS_ADDENDUM C.2).
//!
//! Every reduction lane is reduced a second time in a dihedrally relabelled
//! frame of its propagator cycle, reusing the stored family coefficients and
//! the cached masters (the NinjaRS adapter's `ReductionFrame`; by default the
//! loop momentum reversed from the second propagator). The check is optional
//! and costs one more reduction pass; without it no estimate exists.
//!
//! The error budget of a point sums, per squared-order sector and Laurent
//! coefficient, `2 |Ω_f − Ω'_f| / (16 π² average)` over its lanes, the bound
//! of the bare part's deviation in which no two lanes cancel (the slice's
//! `_lane_budget`). The combination with rotation checks and the safety
//! factor belong to the caller (`loops/v2/stability.py`).

use symbolica::domains::float::{Complex, Real};

/// A relabelled reduction frame: propagator `j` of the frame is propagator
/// `rotation ± j` of the family, `reverse` flips the loop momentum and
/// `shift` moves the origin onto the frame's first propagator (exact
/// re-expansion).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReductionCheck {
    pub reverse: bool,
    pub rotation: usize,
    pub shift: bool,
}

impl ReductionCheck {
    /// The default check: loop direction reversed from the second
    /// propagator, coefficients reused exactly.
    pub const REVERSED: Self = Self {
        reverse: true,
        rotation: 1,
        shift: false,
    };
}

impl Default for ReductionCheck {
    fn default() -> Self {
        Self::REVERSED
    }
}

/// `|a − b|` per Laurent coefficient.
pub fn deviation<T: Real>(direct: &[Complex<T>], check: &[Complex<T>]) -> Vec<T> {
    direct
        .iter()
        .zip(check)
        .map(|(a, b)| (a.clone() - b.clone()).norm().re)
        .collect()
}

/// Adds one lane's deviation to its sector's budget:
/// `budget[k] += 2 |ΔΩ_k| / (16 π² average)`.
pub fn add_to_budget<T: Real>(budget: &mut [T], deviation: &[T], average: u64) {
    let Some(first) = deviation.first() else {
        return;
    };
    let pi = first.pi();
    let factor = first.from_usize(2)
        / (first.from_usize(16) * &pi * &pi * first.from_usize(average as usize));
    for (b, d) in budget.iter_mut().zip(deviation) {
        *b += factor.clone() * d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_sum_scaled_deviations() {
        let direct = [Complex::new(1.0, 2.0), Complex::new(-3.0, 0.0)];
        let check = [Complex::new(1.0, 2.5), Complex::new(-1.0, 0.0)];
        let d = deviation(&direct, &check);
        assert_eq!(d, [0.5, 2.0]);
        let mut budget = [0.0, 0.0];
        add_to_budget(&mut budget, &d, 4);
        let factor = 2.0 / (16.0 * std::f64::consts::PI * std::f64::consts::PI * 4.0);
        assert!((budget[0] - 0.5 * factor).abs() < 1e-18);
        assert!((budget[1] - 2.0 * factor).abs() < 1e-18);
    }
}
