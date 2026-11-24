// SPDX-License-Identifier: 0BSD

//! The rescue-precision kinematics of reduction lanes
//! ([`super::V2Options::refinement`], [`super::LaneRefinement`]): the input
//! a reduction backend's own local rescue reads (NinjaRS's cancellation
//! rescue, `FastRefinement`).
//!
//! ONE CURED POINT PER EVALUATION POINT. The point is the caller's exact
//! input ([`V2Batch::momenta`]: binary64 values read exactly, or a caller's
//! decimal or refined point) improved in the rescue carrier by the one
//! implementation ([`crate::kinematics_improvement::improve_exact`], MadLoop's
//! default chain finished by the projection), from the call's cache
//! ([`super::Cures`]) that the lane rescue reads too.
//!
//! PER LANE, from it: the shifts `p(mask)` (sums of the improved incoming
//! legs), the invariants `s[a ^ b]` (`p²`, or the bound leg mass exactly on
//! a single-leg subset or its complement, as the executor's exact
//! invariants: an on-shell invariant equals its mass in every carrier) and
//! the squared masses, each as the low part `round(cured - hi)` in the
//! evaluation's carrier against the lane's own value `hi`. The sums and
//! squares run in the rescue carrier (its rounding, never binary64's), once
//! per (point, mask) and shared by every lane and family of the point.
//!
//! COST: nothing unless the options ask for it and the backend names a
//! carrier; then, per point, the improvement (once per call, shared with the
//! lane rescue) and one sum and one square per distinct subset mask in the
//! rescue carrier; per lane, copies.

use super::*;
use crate::loop_v2::dress::onshell_leg;
use std::any::Any;
use std::collections::HashMap;
use symbolica::domains::float::{DoubleFloat, Float, RealLike};

/// The per-call store of the improved points' subset momenta and the low
/// parts already formed, per point (one per call of the executor's chunk
/// walk; a point's lanes of every family share it).
#[derive(Default)]
pub(crate) struct Lows {
    points: HashMap<usize, Box<dyn Any + Send>>,
}

/// One point in rescue carrier `R`, against evaluation carrier `T`.
struct PointLows<T, R> {
    /// The improved incoming legs, `4 × legs`, in `R`.
    legs: Vec<R>,
    momenta: HashMap<u64, [R; 4]>,
    /// Low parts of `p(mask)` and of `s[mask]` against their `hi`.
    momentum_lows: HashMap<u64, [T; 4]>,
    square_lows: HashMap<u64, T>,
}

/// `hi` of the evaluation's carrier as a value of the rescue carrier,
/// exactly.
fn lift<T: Carrier, R: Carrier>(hi: &T, high: &R) -> R {
    let any = hi as &dyn Any;
    if let (Some(x), true) = (
        any.downcast_ref::<f64>(),
        (high as &dyn Any).is::<DoubleFloat>(),
    ) {
        let value = DoubleFloat::from(*x);
        return (&value as &dyn Any)
            .downcast_ref::<R>()
            .expect("double-double")
            .clone();
    }
    cast(high, &exact_value(hi).expect("a finite lane input"))
}

/// A rescue-carrier value rounded to the evaluation's carrier.
fn lower<R: Carrier, T: Carrier>(value: &R, zero: &T) -> T {
    if (zero as &dyn Any).is::<f64>() {
        let x = RealLike::to_f64(value);
        return (&x as &dyn Any)
            .downcast_ref::<T>()
            .expect("binary64")
            .clone();
    }
    cast(zero, &exact_value(value).expect("a finite low part"))
}

impl<T: Carrier, R: Carrier> PointLows<T, R> {
    fn new(legs: &[Rational], high: &R) -> Self {
        Self {
            legs: legs.iter().map(|x| cast(high, x)).collect(),
            momenta: HashMap::new(),
            momentum_lows: HashMap::new(),
            square_lows: HashMap::new(),
        }
    }

    fn momentum(&mut self, mask: u64, high: &R) -> [R; 4] {
        if mask == 0 {
            return std::array::from_fn(|_| high.zero());
        }
        if let Some(found) = self.momenta.get(&mask) {
            return found.clone();
        }
        let leg = mask.trailing_zeros() as usize;
        let rest = self.momentum(mask & (mask - 1), high);
        let value: [R; 4] = std::array::from_fn(|mu| rest[mu].clone() + &self.legs[4 * leg + mu]);
        self.momenta.insert(mask, value.clone());
        value
    }

    fn momentum_low(&mut self, mask: u64, hi: &[T], high: &R, zero: &T) -> [T; 4] {
        if let Some(found) = self.momentum_lows.get(&mask) {
            return found.clone();
        }
        let p = self.momentum(mask, high);
        let low: [T; 4] =
            std::array::from_fn(|mu| lower(&(p[mu].clone() - &lift(&hi[mu], high)), zero));
        self.momentum_lows.insert(mask, low.clone());
        low
    }

    fn square_low(
        &mut self,
        mask: u64,
        hi: &T,
        onshell: &[Option<Rational>],
        high: &R,
        zero: &T,
    ) -> T {
        if let Some(found) = self.square_lows.get(&mask) {
            return found.clone();
        }
        let bound = if onshell.is_empty() {
            None
        } else {
            onshell_leg(mask, onshell.len()).and_then(|leg| onshell[leg].as_ref())
        };
        let cured = match bound {
            // On its leg's shell: the bound mass, exactly (as `hi`).
            Some(mass) => cast(high, mass),
            None if mask == 0 => high.zero(),
            None => {
                let p = self.momentum(mask, high);
                p[0].clone() * &p[0]
                    - p[1].clone() * &p[1]
                    - p[2].clone() * &p[2]
                    - p[3].clone() * &p[3]
            }
        };
        let low = lower(&(cured - &lift(hi, high)), zero);
        self.square_lows.insert(mask, low.clone());
        low
    }
}

/// The lazy source of the refinement over one reduction call's `groups`
/// ([`super::RefinementSource`]): a lane's low parts are formed when the
/// backend asks for them (NinjaRS asks for the lanes it flags only), from
/// the call's improved points ([`super::Cures`]) and per-point sums
/// (`lows`, kept by the executor across the chunk's family walks).
pub(crate) struct GroupsSource<'a, T> {
    pub plan: &'a LoopProgramV4,
    pub batch: &'a V2Batch<T>,
    pub groups: &'a [FamilyLanes<T>],
    pub carrier: RescueCarrier,
    pub n_initial: usize,
    pub variant: crate::kinematics_improvement::Variant,
    pub cures: &'a Cures,
    pub lows: std::sync::Mutex<&'a mut Lows>,
    pub zero: T,
    /// Lanes asked for.
    pub asked: std::sync::atomic::AtomicU64,
}

impl<T: Carrier> super::RefinementSource<T> for GroupsSource<'_, T> {
    fn carrier(&self) -> RescueCarrier {
        self.carrier
    }

    fn lane(&self, group: usize, row: usize) -> RusticolResult<LaneRefinement<T>> {
        self.asked
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let lanes = self
            .groups
            .get(group)
            .filter(|g| row < g.keys.len())
            .ok_or_else(|| RusticolError::internal("the refinement's lane is not in the call"))?;
        let mut lows = self.lows.lock().expect("the refinement's sums");
        match self.carrier {
            RescueCarrier::DoubleDouble => {
                refine_row(self, lanes, row, &mut lows, &DoubleFloat::from(0.0))
            }
            RescueCarrier::Arbitrary(_) => refine_row(
                self,
                lanes,
                row,
                &mut lows,
                &Float::with_val(self.carrier.bits(), 0),
            ),
        }
    }
}

/// One lane's low parts ([`GroupsSource`]) in rescue carrier `R`.
fn refine_row<T: Carrier, R: Carrier>(
    source: &GroupsSource<'_, T>,
    group: &FamilyLanes<T>,
    row: usize,
    lows: &mut Lows,
    high: &R,
) -> RusticolResult<LaneRefinement<T>> {
    let (batch, zero) = (source.batch, &source.zero);
    let onshell: &[Option<Rational>] = if batch.leg_masses.is_empty() {
        &[]
    } else {
        &batch.leg_masses
    };
    let family = &source.plan.families[group.family];
    let masks = family.shift_masks();
    let n = group.propagators;
    if masks.len() != n {
        return Err(RusticolError::internal(
            "the refinement's family has another propagator count",
        ));
    }
    let point = group.points[row];
    let entry = lows.points.entry(point).or_insert_with(|| {
        let legs = cured_legs(
            source.cures,
            batch,
            point,
            source.n_initial,
            source.variant,
            source.carrier,
        );
        Box::new(PointLows::<T, R>::new(&legs, high))
    });
    let state = entry
        .downcast_mut::<PointLows<T, R>>()
        .ok_or_else(|| RusticolError::internal("one rescue carrier per call"))?;
    let real = |x: T| Complex::new(x, zero.zero());
    let mut shifts = Vec::with_capacity(4 * n);
    for (k, &mask) in masks.iter().enumerate() {
        let hi: Vec<T> = group.shifts[(row * n + k) * 4..(row * n + k + 1) * 4]
            .iter()
            .map(|z| z.re.clone())
            .collect();
        for low in state.momentum_low(mask, &hi, high, zero) {
            shifts.push(real(low));
        }
    }
    let mut s_matrix = Vec::with_capacity(n * n);
    for (i, &a) in masks.iter().enumerate() {
        for (j, &b) in masks.iter().enumerate() {
            let hi = &group.s_matrix[row * n * n + i * n + j].re;
            s_matrix.push(real(state.square_low(a ^ b, hi, onshell, high, zero)));
        }
    }
    // The squared masses' low parts against their carrier values, exactly.
    let mass_low = |exact: &Rational, hi: &T| -> T {
        let hi = exact_value(hi).expect("a finite squared mass");
        cast(zero, &(exact - &hi))
    };
    let masses: Vec<Complex<T>> = family
        .sequence
        .iter()
        .enumerate()
        .map(|(k, &(_, mass))| {
            let hi = &group.masses[row * n + k];
            let im = batch
                .masses_imaginary
                .get(mass)
                .cloned()
                .unwrap_or_else(|| Rational::from(0));
            Complex::new(mass_low(&batch.masses[mass], &hi.re), mass_low(&im, &hi.im))
        })
        .collect();
    let exact = masses.iter().all(|z| z.re.is_zero() && z.im.is_zero());
    Ok(LaneRefinement {
        carrier: source.carrier,
        shifts,
        s_matrix,
        masses: (!exact).then_some(masses),
    })
}

/// The low parts of every lane of `groups` (one [`LaneRefinement`] per
/// group, its lanes concatenated in its lane order), eagerly: what the lazy
/// source gives lane by lane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refine_groups<T: Carrier>(
    plan: &LoopProgramV4,
    batch: &V2Batch<T>,
    groups: &[FamilyLanes<T>],
    carrier: RescueCarrier,
    n_initial: usize,
    cures: &Cures,
    lows: &mut Lows,
    zero: &T,
) -> RusticolResult<Vec<LaneRefinement<T>>> {
    use super::RefinementSource;
    let source = GroupsSource {
        plan,
        batch,
        groups,
        carrier,
        n_initial,
        variant: crate::kinematics_improvement::DEFAULT_VARIANT,
        cures,
        lows: std::sync::Mutex::new(lows),
        zero: zero.clone(),
        asked: Default::default(),
    };
    let mut out = Vec::with_capacity(groups.len());
    for (g, group) in groups.iter().enumerate() {
        let mut all = LaneRefinement {
            carrier,
            shifts: Vec::new(),
            s_matrix: Vec::new(),
            masses: None,
        };
        let rows = group.keys.len();
        let lanes: Vec<LaneRefinement<T>> = (0..rows)
            .map(|row| source.lane(g, row))
            .collect::<RusticolResult<_>>()?;
        let any_masses = lanes.iter().any(|l| l.masses.is_some());
        let mut masses = Vec::new();
        for (row, lane) in lanes.into_iter().enumerate() {
            all.shifts.extend(lane.shifts);
            all.s_matrix.extend(lane.s_matrix);
            if any_masses {
                match lane.masses {
                    Some(m) => masses.extend(m),
                    None => masses.extend(
                        group.masses[row * group.propagators..(row + 1) * group.propagators]
                            .iter()
                            .map(|_| Complex::new(zero.zero(), zero.zero())),
                    ),
                }
            }
        }
        all.masses = any_masses.then_some(masses);
        out.push(all);
    }
    Ok(out)
}

/// The public entry of the refinement (tests, development tools): the low
/// parts of `groups` at `batch`'s points improved in `carrier`, with a
/// fresh cache.
pub fn lane_refinements<T: Carrier>(
    plan: &LoopProgramV4,
    batch: &V2Batch<T>,
    groups: &[FamilyLanes<T>],
    carrier: RescueCarrier,
    n_initial: usize,
    zero: &T,
) -> RusticolResult<Vec<LaneRefinement<T>>> {
    let cures = Cures::default();
    refine_groups(
        plan,
        batch,
        groups,
        carrier,
        n_initial,
        &cures,
        &mut Lows::default(),
        zero,
    )
}

/// The consistency of one lane's refined inputs (`hi + lo`, exactly, in
/// MPFR at `high_bits`): the largest `|s_ij - (V_i - V_j)²|` over the lane's
/// invariant matrix relative to the lane's scale (the largest `|s_ij|` and
/// `(V_i^0 - V_j^0)²`), and that scale. An on-shell entry tests its leg's
/// shell (`p² = m²`), every other entry the invariant's consistency with
/// the shifts: the residuals NinjaRS's rescue needs at the
/// rescue precision.
pub fn refinement_residuals<T: Carrier>(
    group: &FamilyLanes<T>,
    refinement: &LaneRefinement<T>,
    row: usize,
    high_bits: u32,
) -> (f64, f64) {
    let n = group.propagators;
    let high = Float::with_val(high_bits, 0);
    let at =
        |hi: &T, lo: &T| -> Float { lift::<T, Float>(hi, &high) + &lift::<T, Float>(lo, &high) };
    let shift: Vec<[Float; 4]> = (0..n)
        .map(|k| {
            std::array::from_fn(|mu| {
                let index = (row * n + k) * 4 + mu;
                at(&group.shifts[index].re, &refinement.shifts[index].re)
            })
        })
        .collect();
    let mut scale = 0f64;
    let mut worst = 0f64;
    for i in 0..n {
        for j in 0..n {
            let index = row * n * n + i * n + j;
            let s = at(&group.s_matrix[index].re, &refinement.s_matrix[index].re);
            let d: [Float; 4] = std::array::from_fn(|mu| shift[i][mu].clone() - &shift[j][mu]);
            let square = d[0].clone() * &d[0]
                - d[1].clone() * &d[1]
                - d[2].clone() * &d[2]
                - d[3].clone() * &d[3];
            scale = scale
                .max(RealLike::to_f64(&s).abs())
                .max(RealLike::to_f64(&(d[0].clone() * &d[0])));
            worst = worst.max(RealLike::to_f64(&(s - &square)).abs());
        }
    }
    (if scale > 0.0 { worst / scale } else { worst }, scale)
}
