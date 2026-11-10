// SPDX-License-Identifier: 0BSD

//! The kinematics improvement of an input point (m-lit lit-psimprove; USER
//! 2026-11-10 ~00:56 and ~01:56; DECISIONS 2026-11-10 01:56 and 02:29).
//!
//! Before a point is evaluated in a carrier, its external momenta are moved
//! onto exact energy-momentum conservation and the external mass shells in
//! that carrier's arithmetic, by one of three VARIANTS:
//!
//! * [`Variant::ZShift`], MadLoop's `ImprovePSPoint = 2`
//!   (`ORIG_IMPROVE_PS_POINT_PRECISION` of MG5_aMC 3.7.0
//!   `madgraph/iolibs/template_files/loop/improve_ps.inc`): every final state
//!   keeps E, px, py and takes pz on its shell; the two initial states (beams
//!   along z) take the closed-form E / z shifts that restore conservation;
//! * [`Variant::Rescaling`], MadLoop's `ImprovePSPoint = 1`
//!   (`PSMC_IMPROVE_PS_POINT_PRECISION`): the last particle balances the
//!   3-momentum, every 3-momentum is scaled (particle 1 by x², the others by
//!   x, x from Newton on the energy balance) and every energy is set on its
//!   shell. As MadLoop codes it, it leaves the 3-momentum residual
//!   (x² - x) p₁ wherever x ≠ 1 (the EXACTNESS RECORD flags it);
//! * [`Variant::Projection`], the frame-independent minimal-norm projection
//!   of the 4n components onto the n shells and the 4 conservation equations
//!   (Gauss-Newton `x <- x - Jᵀ (J Jᵀ)⁻¹ c(x)`, Euclidean metric on the
//!   components), iterated in the carrier until the residual is at the
//!   carrier's rounding level or stops decreasing (no iteration cap).
//!
//! THE CHAIN of each selection ([`improve`]): MadLoop's dispatch exactly for
//! the two MadLoop variants (`IMPROVE_PS_POINT_PRECISION`: IS_PHYSICAL, then
//! the selected variant, on a refusal code the other one, both refusing ->
//! evaluated as given with a typed warning); the projection alone, with no
//! MadLoop variant after it (DECISIONS 2026-11-10 02:29). The produced
//! point's EXACTNESS RECORD measures its shell and conservation defects in
//! the carrier against the carrier's rounding level; above it the record
//! carries a typed warning naming the variant and the residual. A record,
//! never a fallback (DECISIONS 2026-11-10 01:56 (3)).
//!
//! TOLERANCES. improve_ps.inc's precision literals are quadruple-precision
//! values; here each is derived from the carrier's unit roundoff `u`
//! ([`PsReal::unit_roundoff`]) and the point:
//!
//! * `THRS_TEST` (ORIG's 'beams along z', MadLoop 1e-15 of `REF = Σ|E|`)
//!   is `sqrt(u)`: ORIG's closed form assumes the second initial state has
//!   no transverse momentum, so a transverse momentum `t REF` costs both
//!   initial shells about `t² REF²`; `t <= sqrt(u)` keeps that at the
//!   carrier's rounding;
//! * `CONVERGED` (PSMC's Newton, MadLoop 1e-26) is the rounding of the
//!   Newton step itself: `2 (n + 3) u Σ E / (|f'| |x|)` (the energy balance
//!   `f` is a sum of n square roots, each with at most `3 u E` error, summed
//!   with `n - 1` roundings);
//! * `MAXITERATIONS` (MadLoop 8) is `ceil(log2(bits)) + 1` (quadratic
//!   Newton doubles the correct bits per step; 8 at quadruple precision's
//!   113 bits, as MadLoop);
//! * `CONSISTENCY_THRES` (MadLoop 1e-25) is `(3 n + 6) u` of `Σ E`: twice
//!   the energy balance's rounding bound plus the n-term sum;
//! * the EXACTNESS bound: a shell defect `|p_k² - m_k²| <= (2 n + 5) u REF_k
//!   S` and a conservation defect `|Σ_in p^μ - Σ_out p^μ| <= 2 n u S`, with
//!   `S = Σ_j |E_j|` and `REF_k = |E_k| + |px_k| + |py_k| + |pz_k|`: forming
//!   `p² - m²` costs 5 roundings of `REF_k²`, forming an n-term sum n - 1 of
//!   `S`, and a component formed by a sum over the n legs carries up to
//!   `n u S` (its first-order effect on a shell `2 REF_k n u S`). The
//!   projection's stopping rule uses the same bound.
//!
//! IS_PHYSICAL (`THRES_FOURMOM` 1e-6, `THRES_ONSHELL` 1e-2) and IS_CLOSE
//! (`THRS_CLOSE` 1e-2) bound the INPUT's distance from a physical point and
//! the size of the move, relative to the input's own scale as in MadLoop;
//! they are independent of the carrier and overridable ([`Tolerances`];
//! DECISIONS 2026-11-10 02:29). IS_PHYSICAL tests the conservation defect's
//! magnitude (MadLoop's test is signed and passes a deficit of the final
//! states).
//!
//! The module is generic over [`PsReal`] (binary64 always; double-double
//! and MPFR with `symbolica-runtime`), so the binary64 instance builds
//! without Symbolica (the tree C ABI's). Points are physical momenta
//! `[E, px, py, pz]`, the `n_initial` initial states first (incoming), the
//! final states outgoing, in the caller's leg order: MadLoop's variants give
//! legs roles (particle 1, the last particle, the initial state with
//! pz > 0), so a permuted leg order is another improvement.

// The negated comparisons are deliberate (a NaN fails every bound test),
// and the index loops follow improve_ps.inc and the linear algebra's
// formulas term by term (the order of the roundings is part of the result).
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A variant of the improvement (a selection of the public setting).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variant {
    /// MadLoop's `ImprovePSPoint = 2` (ORIG; the default).
    ZShift,
    /// MadLoop's `ImprovePSPoint = 1` (PSMC).
    Rescaling,
    /// The frame-independent minimal-norm projection.
    Projection,
}

impl Variant {
    /// The setting's name of the variant.
    pub fn name(self) -> &'static str {
        match self {
            Variant::ZShift => "z-shift",
            Variant::Rescaling => "rescaling",
            Variant::Projection => "projection",
        }
    }

    /// The other MadLoop variant (MadLoop's fallback), none for the
    /// projection.
    pub fn madloop_fallback(self) -> Option<Variant> {
        match self {
            Variant::ZShift => Some(Variant::Rescaling),
            Variant::Rescaling => Some(Variant::ZShift),
            Variant::Projection => None,
        }
    }
}

/// The public setting's value meaning "no improvement in any carrier".
pub const OFF: &str = "off";

/// The setting's values: the three variants and [`OFF`].
pub const SETTINGS: [&str; 4] = ["z-shift", "rescaling", "projection", OFF];

/// THE DEFAULT VARIANT: MadLoop's default, `ImprovePSPoint = 2` with
/// MadLoop's own fallback to `ImprovePSPoint = 1` (USER 2026-11-10 ~01:56
/// "Use the same default as MadLoop currently does, and disabled by default
/// on double-precision evals."; DECISIONS 2026-11-10 01:56 (1)). The one
/// constant every surface reads (the Python API, the run card and CLI, the
/// native SDKs).
pub const DEFAULT_VARIANT: Variant = Variant::ZShift;

/// The binary64 switch's default: off (USER ~01:56; DECISIONS 2026-11-10
/// 01:56 (2)). What counts as a double-precision evaluation is the CARRIER
/// KIND ([`applies`]; DECISIONS 2026-11-10 02:29 (G3) key (a)).
pub const DEFAULT_BINARY64: bool = false;

/// The one refusal text of the variant setting.
pub const SETTING_REFUSAL: &str = "the kinematics improvement (kinematics_improvement) is one of \
     \"z-shift\" (MadLoop's ImprovePSPoint 2, the default), \"rescaling\" (MadLoop's \
     ImprovePSPoint 1), \"projection\" (the frame-independent minimal-norm projection) or \"off\"";

/// The one refusal text of the binary64 switch.
pub const BINARY64_REFUSAL: &str = "the binary64 kinematics improvement switch \
     (kinematics_improvement_binary64) is a boolean: true or false (1 or 0 in the C ABI)";

/// The variant setting of a text: `Some(variant)`, or `None` for [`OFF`].
pub fn setting(text: &str) -> Result<Option<Variant>, String> {
    match text {
        "z-shift" => Ok(Some(Variant::ZShift)),
        "rescaling" => Ok(Some(Variant::Rescaling)),
        "projection" => Ok(Some(Variant::Projection)),
        OFF => Ok(None),
        _ => Err(SETTING_REFUSAL.to_owned()),
    }
}

/// The binary64 switch given as an integer (the C ABI's): 1 (on) or 0.
pub fn binary64_switch(value: i64) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(BINARY64_REFUSAL.to_owned()),
    }
}

/// THE KEY of 'double-precision evaluations' (DECISIONS 2026-11-10 02:29
/// (G3) (a)): the variant an evaluation in a carrier runs, given the
/// variant setting and the binary64 switch. A binary64 carrier follows the
/// switch (off by default); double-double and every arbitrary-precision
/// carrier, at any requested digit count, follow the variant setting.
pub fn applies(
    setting: Option<Variant>,
    binary64: bool,
    carrier_is_binary64: bool,
) -> Option<Variant> {
    if !reaches(binary64, carrier_is_binary64) {
        return None;
    }
    setting
}

/// Whether the settings reach an evaluation in a carrier (the same key as
/// [`applies`]): every carrier above binary64, binary64 only under the
/// binary64 switch. A reached evaluation's results carry a record, which
/// reads "off" under the setting [`OFF`] ([`off_record_json`]); an
/// evaluation not reached carries none (its values bitwise as without the
/// improvement).
pub fn reaches(binary64: bool, carrier_is_binary64: bool) -> bool {
    !carrier_is_binary64 || binary64
}

/// The record of a point the settings reach but do not improve (the setting
/// [`OFF`]), in [`Record::to_json`]'s layout: every surface's (the Python
/// API's `kinematics.off_record`, the loop SDK's diagnostics).
pub fn off_record_json(setting: &str) -> serde_json::Value {
    serde_json::json!({
        "requested": setting,
        "fallback": true,
        "status": "off",
        "produced": null,
        "attempts": [],
        "exactness": null,
        "iterations": null,
        "scale": null,
        "moved": 0.0,
        "warning": null,
        "warning_text": null,
    })
}

/// MadLoop's input-sanity and move bounds (improve_ps.inc IS_PHYSICAL and
/// IS_CLOSE): relative to the input's own scale, independent of the
/// carrier; MadLoop's values by default, overridable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// IS_PHYSICAL's `THRES_FOURMOM`: the largest conservation defect per
    /// component, relative to `Σ_j |E_j|` (MadLoop 1e-6).
    pub fourmom: f64,
    /// IS_PHYSICAL's `THRES_ONSHELL`: the largest `sqrt(|p² - m²|)` per leg,
    /// relative to `|E| + |px| + |py| + |pz|` (MadLoop 1e-2).
    pub onshell: f64,
    /// IS_CLOSE's `THRS_CLOSE`: the largest move `Σ|P - NEWP|` relative to
    /// `Σ|P|` over every component (MadLoop 1e-2).
    pub close: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            fourmom: 1e-6,
            onshell: 1e-2,
            close: 1e-2,
        }
    }
}

/// The refusal codes beyond MadLoop's (improve_ps.inc: 1 a non-physical
/// input, 100-999 ORIG, 1000-9999 PSMC; 999 IS_CLOSE): the projection's
/// singular `J Jᵀ` (e.g. a massless leg of zero momentum).
pub const CODE_NOT_PHYSICAL: i32 = 1;
pub const CODE_NOT_CLOSE: i32 = 999;
pub const CODE_PROJECTION_SINGULAR: i32 = 10001;
/// A non-finite input component (refused before any step).
pub const CODE_NOT_FINITE: i32 = 10002;

/// A real carrier of the improvement.
pub trait PsReal:
    Clone
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The integer `value` in this carrier (at this value's precision).
    fn int(&self, value: i64) -> Self;
    /// A binary64 value in this carrier, exactly.
    fn binary64(&self, value: f64) -> Self;
    fn sqrt(&self) -> Self;
    /// The carrier's unit roundoff (2^-53 binary64, 2^-104 double-double,
    /// 2^-p for p MPFR bits), as a carrier value.
    fn unit_roundoff(&self) -> Self;
    /// Binary digits of the significand.
    fn bits(&self) -> u32;
    fn to_f64(&self) -> f64;
    fn finite(&self) -> bool;

    fn abs(&self) -> Self {
        if *self < self.int(0) {
            -self.clone()
        } else {
            self.clone()
        }
    }
}

impl PsReal for f64 {
    fn int(&self, value: i64) -> Self {
        value as f64
    }
    fn binary64(&self, value: f64) -> Self {
        value
    }
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    fn unit_roundoff(&self) -> Self {
        f64::EPSILON / 2.0
    }
    fn bits(&self) -> u32 {
        f64::MANTISSA_DIGITS
    }
    fn to_f64(&self) -> f64 {
        *self
    }
    fn finite(&self) -> bool {
        self.is_finite()
    }
}

#[cfg(feature = "symbolica-runtime")]
pub use symbolica_carriers::Mpfr;

#[cfg(feature = "symbolica-runtime")]
mod symbolica_carriers {
    use super::PsReal;
    use std::ops::{Add, Div, Mul, Neg, Sub};
    use symbolica::domains::float::{
        DoubleFloat, Float, FloatLike, Real, RealLike, RoundingDirection, SingleFloat,
    };
    use symbolica::domains::rational::Rational;

    /// An MPFR value of a fixed precision: every operation rounds to `bits`
    /// binary digits (nearest). Symbolica's `Float` arithmetic tracks
    /// significance and lowers a result's precision after a cancelling
    /// sum, which would round the later steps of a variant (e.g. MadLoop 2's
    /// z-shift of a slow particle, whose `E² - p_T² - m²` cancels) to fewer
    /// digits than the carrier keeps; the improvement needs the carrier's
    /// full precision in every step.
    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    pub struct Mpfr(pub Float);

    impl Mpfr {
        fn bits(&self) -> u32 {
            self.0.prec()
        }
    }

    impl Add for Mpfr {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let bits = self.bits().max(rhs.bits());
            Mpfr(self.0.add_round(&rhs.0, bits, RoundingDirection::Nearest))
        }
    }

    impl Sub for Mpfr {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            let bits = self.bits().max(rhs.bits());
            Mpfr(self.0.sub_round(&rhs.0, bits, RoundingDirection::Nearest))
        }
    }

    impl Mul for Mpfr {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let bits = self.bits().max(rhs.bits());
            Mpfr(self.0.mul_round(&rhs.0, bits, RoundingDirection::Nearest))
        }
    }

    impl Div for Mpfr {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let bits = self.bits().max(rhs.bits());
            Mpfr(self.0.div_round(&rhs.0, bits, RoundingDirection::Nearest))
        }
    }

    impl Neg for Mpfr {
        type Output = Self;
        fn neg(self) -> Self {
            Mpfr(-self.0)
        }
    }

    impl PsReal for Mpfr {
        fn int(&self, value: i64) -> Self {
            Mpfr(Float::with_val(self.bits(), value))
        }
        fn binary64(&self, value: f64) -> Self {
            let exact = Rational::try_from(value).expect("a finite binary64 value");
            Mpfr(Float::from_rational_round(
                &exact,
                self.bits(),
                RoundingDirection::Nearest,
            ))
        }
        fn sqrt(&self) -> Self {
            let bits = self.bits();
            let mut root = Real::sqrt(&self.0);
            root.set_prec(bits);
            Mpfr(root)
        }
        fn unit_roundoff(&self) -> Self {
            // 2^-p, exactly.
            Mpfr(
                Float::with_val(self.bits(), 2)
                    .pow(u64::from(self.bits()))
                    .inv(),
            )
        }
        fn bits(&self) -> u32 {
            self.0.prec()
        }
        fn to_f64(&self) -> f64 {
            RealLike::to_f64(&self.0)
        }
        fn finite(&self) -> bool {
            SingleFloat::is_finite(&self.0)
        }
    }

    impl PsReal for DoubleFloat {
        fn int(&self, value: i64) -> Self {
            FloatLike::from_i64(self, value)
        }
        fn binary64(&self, value: f64) -> Self {
            DoubleFloat::from(value)
        }
        fn sqrt(&self) -> Self {
            Real::sqrt(self)
        }
        fn unit_roundoff(&self) -> Self {
            // A DoubleFloat operation rounds to about 2^-104 relative, not
            // to its 106-bit significand (`assemble.Precision.unit_roundoff`).
            DoubleFloat::from(2f64.powi(-104))
        }
        fn bits(&self) -> u32 {
            FloatLike::get_precision(self)
        }
        fn to_f64(&self) -> f64 {
            RealLike::to_f64(self)
        }
        fn finite(&self) -> bool {
            SingleFloat::is_finite(self)
        }
    }
}

/// One step of the chain and its code (0: the step produced the point).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    /// `"is_physical"` (MadLoop's input test) or a variant's name.
    pub step: &'static str,
    pub code: i32,
}

/// The exactness record of a produced point: its largest defects relative
/// to the point's scale and the carrier's bounds (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Exactness {
    /// `max_k |p_k² - m_k²| / (REF_k S)`.
    pub shell: f64,
    /// `max_μ |Σ_in p^μ - Σ_out p^μ| / S`.
    pub conservation: f64,
    /// The bound of `shell`: `(2 n + 5) u`.
    pub shell_bound: f64,
    /// The bound of `conservation`: `2 n u`.
    pub conservation_bound: f64,
    /// Both defects within their bounds (decided in the carrier).
    pub exact: bool,
}

/// What became of a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// A variant produced the point.
    Improved,
    /// Already exactly physical (an exact test against the exact masses):
    /// passed unchanged.
    Exact,
    /// No variant could: evaluated as given (a typed warning).
    NotImproved,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Improved => "improved",
            Status::Exact => "exact",
            Status::NotImproved => "not-improved",
        }
    }
}

/// The record of one point's improvement.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub requested: Variant,
    /// Whether MadLoop's fallback to the other MadLoop variant ran on a
    /// refusal (the public chain), or the requested variant alone.
    pub fallback: bool,
    pub status: Status,
    /// The variant that produced the point.
    pub produced: Option<Variant>,
    /// Every step of the chain, in order, with its code.
    pub attempts: Vec<Attempt>,
    /// The produced point's exactness record.
    pub exactness: Option<Exactness>,
    /// The projection's Gauss-Newton steps.
    pub iterations: Option<u32>,
    /// PSMC's scale factor x.
    pub scale: Option<f64>,
    /// The largest component move relative to `Σ|P|` (MadLoop's IS_CLOSE
    /// measure).
    pub moved: f64,
}

/// The kind of a record's typed warning.
pub const WARNING_NOT_IMPROVED: &str = "kinematics-not-improved";
pub const WARNING_INEXACT: &str = "kinematics-inexact";

impl Record {
    /// A record of an exactly physical input, passed unchanged.
    pub fn exact(requested: Variant, fallback: bool) -> Self {
        Self {
            requested,
            fallback,
            status: Status::Exact,
            produced: None,
            attempts: Vec::new(),
            exactness: None,
            iterations: None,
            scale: None,
            moved: 0.0,
        }
    }

    /// The typed warning's kind, if any: [`WARNING_NOT_IMPROVED`] (the point
    /// is evaluated as given) or [`WARNING_INEXACT`] (the produced point's
    /// residual exceeds the carrier's rounding level; the point is used).
    pub fn warning_kind(&self) -> Option<&'static str> {
        match self.status {
            Status::NotImproved => Some(WARNING_NOT_IMPROVED),
            Status::Improved if self.exactness.as_ref().is_some_and(|e| !e.exact) => {
                Some(WARNING_INEXACT)
            }
            _ => None,
        }
    }

    /// The typed warning's one text (every surface), if any.
    pub fn warning(&self) -> Option<String> {
        let codes = || {
            self.attempts
                .iter()
                .map(|a| format!("{} {}", a.step, a.code))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self.warning_kind()? {
            WARNING_NOT_IMPROVED => Some(format!(
                "{WARNING_NOT_IMPROVED}: the kinematics improvement ({}) could not improve the \
                 point, which is evaluated as given (codes: {})",
                self.requested.name(),
                codes()
            )),
            _ => {
                let e = self
                    .exactness
                    .as_ref()
                    .expect("an inexact record has its exactness");
                Some(format!(
                    "{WARNING_INEXACT}: the point produced by the {} variant misses exactness: \
                     shell defect {:.3e} (bound {:.3e}), conservation defect {:.3e} (bound \
                     {:.3e}), relative to the point's scale; the point is used",
                    self.produced.expect("an improved record").name(),
                    e.shell,
                    e.shell_bound,
                    e.conservation,
                    e.conservation_bound
                ))
            }
        }
    }

    /// The record as JSON (the Python API's and the SDKs' diagnostics).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "requested": self.requested.name(),
            "fallback": self.fallback,
            "status": self.status.name(),
            "produced": self.produced.map(Variant::name),
            "attempts": self.attempts.iter().map(|a| serde_json::json!({"step": a.step, "code": a.code})).collect::<Vec<_>>(),
            "exactness": self.exactness.as_ref().map(|e| serde_json::json!({
                "shell": e.shell,
                "conservation": e.conservation,
                "shell_bound": e.shell_bound,
                "conservation_bound": e.conservation_bound,
                "exact": e.exact,
            })),
            "iterations": self.iterations,
            "scale": self.scale,
            "moved": self.moved,
            "warning": self.warning_kind(),
            "warning_text": self.warning(),
        })
    }
}

/// The policy record of the settings (every surface reads it).
pub fn policy_json() -> serde_json::Value {
    let tolerances = Tolerances::default();
    serde_json::json!({
        "settings": SETTINGS,
        "default": DEFAULT_VARIANT.name(),
        "off": OFF,
        "binary64_default": DEFAULT_BINARY64,
        "setting_refusal": SETTING_REFUSAL,
        "binary64_refusal": BINARY64_REFUSAL,
        "key": "carrier",
        "tolerances": {
            "fourmom": tolerances.fourmom,
            "onshell": tolerances.onshell,
            "close": tolerances.close,
        },
        "warnings": [WARNING_NOT_IMPROVED, WARNING_INEXACT],
    })
}

/// A point and its record.
#[derive(Clone, Debug)]
pub struct Improved<T> {
    /// The produced point, or the input unchanged.
    pub legs: Vec<[T; 4]>,
    pub record: Record,
}

/// Improve one point in its carrier.
///
/// `legs` are physical momenta (the `n_initial` initial states first),
/// `masses2` the target squared mass of every leg (exactly the masses the
/// evaluation's external shells use; a leg without a bound mass takes its
/// own `p²`). With `fallback` the MadLoop selections run MadLoop's dispatch
/// (the public chain); without it the requested variant runs alone (a
/// forced variant: its refusal is reported with its code).
pub fn improve<T: PsReal>(
    legs: &[[T; 4]],
    n_initial: usize,
    masses2: &[T],
    variant: Variant,
    fallback: bool,
    tolerances: &Tolerances,
) -> Improved<T> {
    assert_eq!(legs.len(), masses2.len(), "one squared mass per leg");
    assert!(
        n_initial >= 1 && n_initial < legs.len(),
        "initial and final states"
    );
    let ctx = Ctx::new(&legs[0][0], legs.len(), tolerances);
    let mut record = Record {
        requested: variant,
        fallback,
        status: Status::NotImproved,
        produced: None,
        attempts: Vec::new(),
        exactness: None,
        iterations: None,
        scale: None,
        moved: 0.0,
    };
    if legs.iter().flatten().chain(masses2).any(|x| !x.finite()) {
        record.attempts.push(Attempt {
            step: "input",
            code: CODE_NOT_FINITE,
        });
        return Improved {
            legs: legs.to_vec(),
            record,
        };
    }
    let mut produced: Option<(Vec<[T; 4]>, Variant)> = None;
    match variant {
        Variant::ZShift | Variant::Rescaling => {
            // improve_ps.inc IMPROVE_PS_POINT_PRECISION.
            if !ctx.is_physical(legs, masses2, n_initial) {
                record.attempts.push(Attempt {
                    step: "is_physical",
                    code: CODE_NOT_PHYSICAL,
                });
            } else {
                let mut chain = vec![variant];
                if fallback {
                    chain.extend(variant.madloop_fallback());
                }
                for step in chain {
                    let outcome = match step {
                        Variant::ZShift => ctx.orig(legs, masses2, n_initial),
                        _ => {
                            let (outcome, x) = ctx.psmc(legs, masses2, n_initial);
                            record.scale = x.map(|x| x.to_f64());
                            outcome
                        }
                    };
                    match outcome {
                        Ok(point) => {
                            record.attempts.push(Attempt {
                                step: step.name(),
                                code: 0,
                            });
                            produced = Some((point, step));
                            break;
                        }
                        Err(code) => record.attempts.push(Attempt {
                            step: step.name(),
                            code,
                        }),
                    }
                }
            }
        }
        Variant::Projection => {
            let (outcome, iterations) = ctx.projection(legs, masses2, n_initial);
            record.iterations = Some(iterations);
            match outcome {
                Ok(point) => {
                    record.attempts.push(Attempt {
                        step: Variant::Projection.name(),
                        code: 0,
                    });
                    produced = Some((point, Variant::Projection));
                }
                Err(code) => record.attempts.push(Attempt {
                    step: Variant::Projection.name(),
                    code,
                }),
            }
        }
    }
    match produced {
        Some((point, step)) => {
            record.status = Status::Improved;
            record.produced = Some(step);
            record.exactness = Some(ctx.exactness(&point, masses2, n_initial));
            record.moved = ctx.moved(legs, &point);
            Improved {
                legs: point,
                record,
            }
        }
        None => Improved {
            legs: legs.to_vec(),
            record,
        },
    }
}

/// The exactness record of a point in its carrier (module docs), e.g. of
/// an input evaluated as given.
pub fn exactness<T: PsReal>(legs: &[[T; 4]], n_initial: usize, masses2: &[T]) -> Exactness {
    Ctx::new(&legs[0][0], legs.len(), &Tolerances::default()).exactness(legs, masses2, n_initial)
}

/// The derived precision quantities of one carrier and multiplicity.
struct Ctx<T> {
    zero: T,
    u: T,
    n: usize,
    /// ORIG's `THRS_TEST`.
    thrs_test: T,
    /// PSMC's Newton step cap.
    newton_cap: u32,
    tolerances: Tolerances,
}

impl<T: PsReal> Ctx<T> {
    fn new(like: &T, n: usize, tolerances: &Tolerances) -> Self {
        let u = like.unit_roundoff();
        let bits = like.bits().max(2);
        Self {
            zero: like.int(0),
            thrs_test: u.sqrt(),
            u,
            n,
            newton_cap: (u32::BITS - (bits - 1).leading_zeros()) + 1,
            tolerances: *tolerances,
        }
    }

    fn int(&self, value: i64) -> T {
        self.zero.int(value)
    }

    fn sq(&self, x: &T) -> T {
        x.clone() * x.clone()
    }

    /// `Σ_j |E_j|`.
    fn energy_scale(&self, legs: &[[T; 4]]) -> T {
        legs.iter()
            .fold(self.zero.clone(), |acc, leg| acc + leg[0].abs())
    }

    /// improve_ps.inc IS_PHYSICAL, with the conservation defect's magnitude.
    fn is_physical(&self, legs: &[[T; 4]], masses2: &[T], n_initial: usize) -> bool {
        let fourmom = self.zero.binary64(self.tolerances.fourmom);
        let onshell = self.zero.binary64(self.tolerances.onshell);
        let reference = self.energy_scale(legs);
        let mut physical = true;
        for mu in 0..4 {
            let mut buff = self.zero.clone();
            for leg in &legs[..n_initial] {
                buff = buff - leg[mu].clone();
            }
            for leg in &legs[n_initial..] {
                buff = buff + leg[mu].clone();
            }
            if buff.abs() / reference.clone() > fourmom {
                physical = false;
            }
        }
        for (leg, m2) in legs.iter().zip(masses2) {
            let reference = leg.iter().fold(self.zero.clone(), |acc, x| acc + x.abs());
            let defect = (self.sq(&leg[0])
                - self.sq(&leg[1])
                - self.sq(&leg[2])
                - self.sq(&leg[3])
                - m2.clone())
            .abs()
            .sqrt();
            // (A zero reference of a massless zero momentum: 0/0, which
            // MadLoop's comparison passes.)
            if reference > self.zero {
                if defect / reference > onshell {
                    physical = false;
                }
            } else if defect > self.zero {
                physical = false;
            }
        }
        physical
    }

    /// improve_ps.inc IS_CLOSE.
    fn is_close(&self, legs: &[[T; 4]], new: &[[T; 4]]) -> bool {
        let close = self.zero.binary64(self.tolerances.close);
        let (moved, reference) = self.move_ratio(legs, new);
        !(moved / reference > close)
    }

    fn move_ratio(&self, legs: &[[T; 4]], new: &[[T; 4]]) -> (T, T) {
        let mut reference = self.zero.clone();
        let mut moved = self.zero.clone();
        for (old, new) in legs.iter().zip(new) {
            for mu in 0..4 {
                reference = reference + old[mu].abs();
                moved = moved + (old[mu].clone() - new[mu].clone()).abs();
            }
        }
        (moved, reference)
    }

    fn moved(&self, legs: &[[T; 4]], new: &[[T; 4]]) -> f64 {
        let (moved, reference) = self.move_ratio(legs, new);
        (moved / reference).to_f64()
    }

    /// improve_ps.inc ORIG_IMPROVE_PS_POINT_PRECISION (MadLoop 2): the
    /// improved point or the refusal code.
    fn orig(&self, p: &[[T; 4]], masses2: &[T], n_initial: usize) -> Result<Vec<[T; 4]>, i32> {
        let n = p.len();
        let zero = &self.zero;
        let two = self.int(2);
        let reference = self.energy_scale(p);
        // The refusals in MadLoop's order, each overwriting the previous.
        let mut code = 0;
        if n_initial != 2 {
            code = 100;
        }
        let beam =
            |leg: usize, mu: usize| (p[leg][mu].clone() / reference.clone()).abs() > self.thrs_test;
        if beam(0, 1) || beam(0, 2) || beam(1, 1) || beam(1, 2) {
            code = 200;
        }
        if masses2[0] != *zero || masses2[1] != *zero {
            code = 300;
        }
        for (i, leg) in p.iter().enumerate() {
            if leg[0] < *zero {
                code = 400 + i32::try_from(i + 1).unwrap_or(i32::MAX - 400);
            }
        }
        if code != 0 {
            return Err(code);
        }
        let mut new = p.to_vec();
        let mut pt = [zero.clone(), zero.clone(), zero.clone(), zero.clone()];
        for i in n_initial..n {
            for j in 0..4 {
                if j == 3 {
                    let value = (self.sq(&p[i][0])
                        - self.sq(&p[i][1])
                        - self.sq(&p[i][2])
                        - masses2[i].clone())
                    .abs()
                    .sqrt();
                    // Fortran SIGN(a, b): |a| with the sign of b (b = 0: +).
                    new[i][3] = if p[i][3] < *zero { -value } else { value };
                } else {
                    new[i][j] = p[i][j].clone();
                }
                pt[j] = pt[j].clone() + new[i][j].clone();
            }
        }
        let (p1, p2) = if p[0][3] > *zero {
            (0, 1)
        } else if p[1][3] > *zero {
            (1, 0)
        } else {
            return Err(500);
        };
        let [t0, t1, t2, t3] = pt.clone();
        let mut discr = -self.sq(&t0) + self.sq(&t1) + self.sq(&t2) + self.sq(&t3);
        if discr < *zero {
            discr = -discr;
        }
        let e1 = p[p1][0].clone();
        let e2 = p[p2][0].clone();
        let shift_e1 = (t0.clone()
            * (-two.clone() * e1.clone() * t0.clone()
                + self.sq(&t0)
                + self.sq(&t1)
                + self.sq(&t2))
            + (two.clone() * e1.clone() - t0.clone()) * self.sq(&t3)
            + t3.clone() * discr.clone())
            / (two.clone() * (t0.clone() - t3.clone()) * (t0.clone() + t3.clone()));
        let shift_e2 = -(t0.clone()
            * (two.clone() * e2.clone() * t0.clone() - self.sq(&t0) + self.sq(&t1) + self.sq(&t2))
            + (-two.clone() * e2.clone() + t0.clone()) * self.sq(&t3)
            + t3.clone() * discr.clone())
            / (two.clone() * (t0.clone() - t3.clone()) * (t0.clone() + t3.clone()));
        let shift_z1 = (-two.clone() * p[p1][3].clone() * (self.sq(&t0) - self.sq(&t3))
            + t3.clone() * (self.sq(&t0) + self.sq(&t1) + self.sq(&t2) - self.sq(&t3))
            + t0.clone() * discr.clone())
            / (two.clone() * (self.sq(&t0) - self.sq(&t3)));
        let shift_z2 = -(two.clone() * p[p2][3].clone() * (self.sq(&t0) - self.sq(&t3))
            + t3.clone() * (-self.sq(&t0) + self.sq(&t1) + self.sq(&t2) + self.sq(&t3))
            + t0.clone() * discr)
            / (two.clone() * (self.sq(&t0) - self.sq(&t3)));
        new[p1][0] = e1 + shift_e1;
        new[p1][3] = p[p1][3].clone() + shift_z1;
        new[p2][0] = e2 + shift_e2;
        new[p2][3] = p[p2][3].clone() + shift_z2;
        new[p2][1] = p[p2][1].clone();
        new[p2][2] = p[p2][2].clone();
        for j in 1..3 {
            let mut sum = zero.clone();
            for leg in &p[n_initial..] {
                sum = sum + leg[j].clone();
            }
            new[p1][j] = sum - p[p2][j].clone();
        }
        if !self.is_close(p, &new) {
            return Err(CODE_NOT_CLOSE);
        }
        if new.iter().flatten().any(|x| !x.finite()) {
            return Err(CODE_NOT_FINITE);
        }
        Ok(new)
    }

    /// improve_ps.inc FUNCT: the energy balance (or its derivative as
    /// MadLoop codes it, particle 1's `2 x |p|² / E`) at `x`.
    fn funct(
        &self,
        pvecsq: &[T],
        x: &T,
        derivative: bool,
        masses2: &[T],
        n_initial: usize,
    ) -> Result<T, i32> {
        let mut res = self.zero.clone();
        for (i, (pv, m2)) in pvecsq.iter().zip(masses2).enumerate() {
            let factor = if i < n_initial {
                self.int(-1)
            } else {
                self.int(1)
            };
            let x2 = x.clone() * x.clone();
            let buff = if i == 0 {
                m2.clone() + pv.clone() * (x2.clone() * x2)
            } else {
                m2.clone() + pv.clone() * x2
            };
            if buff < self.zero {
                return Err(1);
            }
            res = res
                + factor
                    * if derivative {
                        if i == 0 {
                            (self.int(2) * x.clone() * pv.clone()) / buff.sqrt()
                        } else {
                            (x.clone() * pv.clone()) / buff.sqrt()
                        }
                    } else {
                        buff.sqrt()
                    };
        }
        Ok(res)
    }

    /// improve_ps.inc FINDX: Newton from `seed`.
    fn findx(&self, p: &[[T; 4]], seed: T, masses2: &[T], n_initial: usize) -> Result<T, i32> {
        let pvecsq: Vec<T> = p
            .iter()
            .map(|leg| self.sq(&leg[1]) + self.sq(&leg[2]) + self.sq(&leg[3]))
            .collect();
        let energies = pvecsq
            .iter()
            .zip(masses2)
            .fold(self.zero.clone(), |acc, (pv, m2)| {
                acc + (m2.clone() + pv.clone()).abs().sqrt()
            });
        // The rounding of the Newton step: f's bound (n + 3) u Σ E over |f'|.
        let rounding = self.int(2) * self.int(self.n as i64 + 3) * self.u.clone() * energies;
        let mut xn = seed;
        for _ in 0..self.newton_cap {
            let f = self.funct(&pvecsq, &xn, false, masses2, n_initial)?;
            let d = self.funct(&pvecsq, &xn, true, masses2, n_initial)?;
            let xnp1 = xn.clone() - f / d.clone();
            let converged = rounding.clone() / (d.abs() * xn.abs());
            let step =
                ((xnp1.clone() - xn.clone()) * self.int(2) / (xnp1.clone() + xn.clone())).abs();
            if step < converged {
                // One last Newton step, for good measure (as MadLoop).
                let f = self.funct(&pvecsq, &xnp1, false, masses2, n_initial)?;
                let d = self.funct(&pvecsq, &xnp1, true, masses2, n_initial)?;
                return Ok(xnp1 - f / d);
            }
            xn = xnp1;
        }
        Err(9)
    }

    /// improve_ps.inc PSMC_IMPROVE_PS_POINT_PRECISION (MadLoop 1): the
    /// improved point or the refusal code, and the scale factor x found.
    fn psmc(
        &self,
        p: &[[T; 4]],
        masses2: &[T],
        n_initial: usize,
    ) -> (Result<Vec<[T; 4]>, i32>, Option<T>) {
        let n = p.len();
        let zero = &self.zero;
        let mut new = p.to_vec();
        let mut buff = [zero.clone(), zero.clone(), zero.clone(), zero.clone()];
        for leg in &new[..n_initial] {
            for j in 1..4 {
                buff[j] = buff[j].clone() + leg[j].clone();
            }
        }
        for leg in &new[n_initial..n - 1] {
            for j in 1..4 {
                buff[j] = buff[j].clone() - leg[j].clone();
            }
        }
        new[n - 1][1..4].clone_from_slice(&buff[1..4]);
        let seeds = [
            self.int(1),
            self.int(11) / self.int(10),
            self.int(9) / self.int(10),
        ];
        let mut code = 0;
        let mut scale = None;
        for (i, seed) in seeds.into_iter().enumerate() {
            match self.findx(&new, seed, masses2, n_initial) {
                Ok(x) => {
                    scale = Some(x);
                    break;
                }
                Err(error) => code += 10i32.pow(i as u32) * error,
            }
        }
        let Some(x) = scale else {
            return (Err(if code < 1000 { code + 1000 } else { code }), None);
        };
        for (i, leg) in new.iter_mut().enumerate() {
            for component in leg.iter_mut().skip(1) {
                *component = if i == 0 {
                    component.clone() * (x.clone() * x.clone())
                } else {
                    component.clone() * x.clone()
                };
            }
        }
        for (leg, m2) in new.iter_mut().zip(masses2) {
            let mut b = m2.clone();
            for j in 1..4 {
                b = b + self.sq(&leg[j]);
            }
            leg[0] = b.sqrt();
        }
        let mut balance = zero.clone();
        let mut total = zero.clone();
        for leg in &new[..n_initial] {
            balance = balance - leg[0].clone();
            total = total + leg[0].clone();
        }
        for leg in &new[n_initial..] {
            balance = balance + leg[0].clone();
            total = total + leg[0].clone();
        }
        let consistency = self.int(3 * self.n as i64 + 6) * self.u.clone();
        if !(balance.abs() / total <= consistency) {
            return (Err(1000), Some(x));
        }
        if !self.is_close(p, &new) {
            return (Err(CODE_NOT_CLOSE), Some(x));
        }
        if new.iter().flatten().any(|v| !v.finite()) {
            return (Err(CODE_NOT_FINITE), Some(x));
        }
        (Ok(new), Some(x))
    }

    /// The shell and conservation defects in the carrier.
    fn defects(&self, p: &[[T; 4]], masses2: &[T], n_initial: usize) -> (Vec<T>, [T; 4]) {
        let shell = p
            .iter()
            .zip(masses2)
            .map(|(leg, m2)| {
                self.sq(&leg[0])
                    - self.sq(&leg[1])
                    - self.sq(&leg[2])
                    - self.sq(&leg[3])
                    - m2.clone()
            })
            .collect();
        let mut conservation = [
            self.zero.clone(),
            self.zero.clone(),
            self.zero.clone(),
            self.zero.clone(),
        ];
        for (k, leg) in p.iter().enumerate() {
            for mu in 0..4 {
                conservation[mu] = if k < n_initial {
                    conservation[mu].clone() + leg[mu].clone()
                } else {
                    conservation[mu].clone() - leg[mu].clone()
                };
            }
        }
        (shell, conservation)
    }

    /// The bounds (module docs): per leg `(2 n + 5) u REF_k S`, per
    /// component `2 n u S`.
    fn bounds(&self, p: &[[T; 4]]) -> (Vec<T>, T, T) {
        let scale = self.energy_scale(p);
        let n = self.n as i64;
        let shell_factor = self.int(2 * n + 5) * self.u.clone();
        let shell = p
            .iter()
            .map(|leg| {
                let reference = leg.iter().fold(self.zero.clone(), |acc, x| acc + x.abs());
                shell_factor.clone() * reference * scale.clone()
            })
            .collect();
        let conservation = self.int(2 * n) * self.u.clone() * scale.clone();
        (shell, conservation, scale)
    }

    /// The largest defect over its bound (<= 1: exact to the carrier's
    /// rounding).
    fn scaled_residual(&self, p: &[[T; 4]], masses2: &[T], n_initial: usize) -> T {
        let (shell, conservation) = self.defects(p, masses2, n_initial);
        let (shell_bounds, conservation_bound, _) = self.bounds(p);
        let mut worst = self.zero.clone();
        for (d, b) in shell.iter().zip(&shell_bounds) {
            // (A zero defect of a zero-reference leg is exact.)
            if *d == self.zero {
                continue;
            }
            let r = d.abs() / b.clone();
            if !(r <= worst) {
                worst = r;
            }
        }
        for d in &conservation {
            let r = d.abs() / conservation_bound.clone();
            if !(r <= worst) {
                worst = r;
            }
        }
        worst
    }

    fn exactness(&self, p: &[[T; 4]], masses2: &[T], n_initial: usize) -> Exactness {
        let (shell, conservation) = self.defects(p, masses2, n_initial);
        let (_, _, scale) = self.bounds(p);
        let n = self.n as i64;
        let mut worst_shell = self.zero.clone();
        for (d, leg) in shell.iter().zip(p) {
            if *d == self.zero {
                continue;
            }
            let reference = leg.iter().fold(self.zero.clone(), |acc, x| acc + x.abs());
            let r = d.abs() / (reference * scale.clone());
            if !(r <= worst_shell) {
                worst_shell = r;
            }
        }
        let mut worst_conservation = self.zero.clone();
        for d in &conservation {
            let r = d.abs() / scale.clone();
            if !(r <= worst_conservation) {
                worst_conservation = r;
            }
        }
        let shell_bound = self.int(2 * n + 5) * self.u.clone();
        let conservation_bound = self.int(2 * n) * self.u.clone();
        let exact = worst_shell <= shell_bound && worst_conservation <= conservation_bound;
        Exactness {
            shell: worst_shell.to_f64(),
            conservation: worst_conservation.to_f64(),
            shell_bound: shell_bound.to_f64(),
            conservation_bound: conservation_bound.to_f64(),
            exact,
        }
    }

    /// The minimal-norm projection: the point or the refusal code, and the
    /// Gauss-Newton steps taken. STOPPING RULE: stop when the scaled
    /// residual is at most 1 (exact to the carrier's rounding), or when a
    /// step does not strictly decrease it (its rounding floor; the better
    /// point is kept and its exactness record states the residual). The
    /// strict decrease over the carrier's finitely many values terminates.
    fn projection(
        &self,
        p: &[[T; 4]],
        masses2: &[T],
        n_initial: usize,
    ) -> (Result<Vec<[T; 4]>, i32>, u32) {
        let n = p.len();
        let rows = n + 4;
        let mut x = p.to_vec();
        let mut residual = self.scaled_residual(&x, masses2, n_initial);
        let one = self.int(1);
        let mut iterations = 0;
        while residual > one {
            let (shell, conservation) = self.defects(&x, masses2, n_initial);
            let c: Vec<T> = shell.into_iter().chain(conservation).collect();
            // J (rows x 4n) by its nonzero structure: a shell row k has
            // 2 g_μμ x_k^μ on leg k; a conservation row μ has ±1 on every
            // leg's component μ (+ initial, - final).
            let sign = |k: usize| {
                if k < n_initial {
                    one.clone()
                } else {
                    -one.clone()
                }
            };
            let metric = |mu: usize| if mu == 0 { one.clone() } else { -one.clone() };
            let shell_row = |k: usize, mu: usize| self.int(2) * metric(mu) * x[k][mu].clone();
            // A = J Jᵀ.
            let mut a = vec![vec![self.zero.clone(); rows]; rows];
            for k in 0..n {
                let mut diag = self.zero.clone();
                for mu in 0..4 {
                    diag = diag + self.sq(&shell_row(k, mu));
                }
                a[k][k] = diag;
                for mu in 0..4 {
                    let v = shell_row(k, mu) * sign(k);
                    a[k][n + mu] = v.clone();
                    a[n + mu][k] = v;
                }
            }
            for mu in 0..4 {
                a[n + mu][n + mu] = self.int(n as i64);
            }
            let Some(lambda) = self.solve_spd(a, c) else {
                return (Err(CODE_PROJECTION_SINGULAR), iterations);
            };
            let mut y = x.clone();
            for k in 0..n {
                for mu in 0..4 {
                    let dx =
                        shell_row(k, mu) * lambda[k].clone() + sign(k) * lambda[n + mu].clone();
                    y[k][mu] = x[k][mu].clone() - dx;
                }
            }
            iterations += 1;
            if y.iter().flatten().any(|v| !v.finite()) {
                break;
            }
            let next = self.scaled_residual(&y, masses2, n_initial);
            if !(next < residual) {
                break;
            }
            x = y;
            residual = next;
        }
        if !self.is_close(p, &x) {
            return (Err(CODE_NOT_CLOSE), iterations);
        }
        (Ok(x), iterations)
    }

    /// Solve `A z = b` for a symmetric positive definite `A` by Cholesky
    /// after the Jacobi scaling `D A D` (unit diagonal); `None` when a pivot
    /// is at most `m u` (singular to the carrier's rounding).
    fn solve_spd(&self, a: Vec<Vec<T>>, b: Vec<T>) -> Option<Vec<T>> {
        let m = b.len();
        let d: Vec<T> = (0..m).map(|i| a[i][i].sqrt()).collect();
        if d.iter().any(|x| !(*x > self.zero)) {
            return None;
        }
        let s: Vec<Vec<T>> = (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| a[i][j].clone() / (d[i].clone() * d[j].clone()))
                    .collect()
            })
            .collect();
        let rhs: Vec<T> = (0..m).map(|i| b[i].clone() / d[i].clone()).collect();
        let floor = self.int(m as i64) * self.u.clone();
        let mut l = vec![vec![self.zero.clone(); m]; m];
        for i in 0..m {
            for j in 0..=i {
                let mut sum = s[i][j].clone();
                for k in 0..j {
                    sum = sum - l[i][k].clone() * l[j][k].clone();
                }
                if i == j {
                    if !(sum > floor) {
                        return None;
                    }
                    l[i][i] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j].clone();
                }
            }
        }
        let mut y = vec![self.zero.clone(); m];
        for i in 0..m {
            let mut sum = rhs[i].clone();
            for k in 0..i {
                sum = sum - l[i][k].clone() * y[k].clone();
            }
            y[i] = sum / l[i][i].clone();
        }
        let mut z = vec![self.zero.clone(); m];
        for i in (0..m).rev() {
            let mut sum = y[i].clone();
            for k in i + 1..m {
                sum = sum - l[k][i].clone() * z[k].clone();
            }
            z[i] = sum / l[i][i].clone();
        }
        Some((0..m).map(|i| z[i].clone() / d[i].clone()).collect())
    }
}

/// The warnings-channel lines of one call's records (the tree runtimes'
/// record, m-lit lit-psimprove (G4)): one summary line (the requested
/// variant and, per producer or status, its count) and every typed warning.
/// The Python tree runtime takes the same lines ([`notes_json`]).
pub fn notes(records: &[Record], carrier: &str, precision: u32) -> Vec<String> {
    format_notes(
        records.iter().map(|record| {
            (
                record.requested.name().to_owned(),
                record
                    .produced
                    .map_or_else(|| record.status.name(), Variant::name)
                    .to_owned(),
                record.warning(),
            )
        }),
        carrier,
        precision,
    )
}

/// [`notes`] of records given as their JSON ([`Record::to_json`]; the
/// Python tree runtime's higher-precision records, which reach it as JSON):
/// the same lines from the same formatter.
pub fn notes_json(records: &[serde_json::Value], carrier: &str, precision: u32) -> Vec<String> {
    let text = |value: &serde_json::Value, key: &str| value[key].as_str().map(str::to_owned);
    format_notes(
        records.iter().map(|record| {
            (
                text(record, "requested").unwrap_or_default(),
                text(record, "produced")
                    .or_else(|| text(record, "status"))
                    .unwrap_or_default(),
                text(record, "warning_text"),
            )
        }),
        carrier,
        precision,
    )
}

/// The lines of [`notes`]: (requested, producer or status, typed warning)
/// per point.
fn format_notes(
    records: impl Iterator<Item = (String, String, Option<String>)>,
    carrier: &str,
    precision: u32,
) -> Vec<String> {
    let records: Vec<_> = records.collect();
    let Some((requested, _, _)) = records.first() else {
        return Vec::new();
    };
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for (_, key, _) in &records {
        *counts.entry(key.as_str()).or_default() += 1;
    }
    let summary = counts
        .iter()
        .map(|(key, count)| format!("{key} {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut lines = vec![format!(
        "kinematics-improvement: {carrier} at precision {precision} ({requested}): {summary}"
    )];
    lines.extend(
        records
            .iter()
            .enumerate()
            .filter_map(|(index, (_, _, warning))| {
                warning
                    .as_ref()
                    .map(|text| format!("point {index}: {text}"))
            }),
    );
    lines
}

/// The binary64 entry of the tree runtimes (the binary64 switch of the tree
/// C ABI and of the Python tree runtime): binary64 points toward the
/// binary64 masses the external wavefunctions use, without Symbolica (the
/// tree C ABI builds without `symbolica-runtime`). The counterpart of
/// `improve_exact` in binary64 on binary64 inputs.
pub mod binary64 {
    use super::{Improved, Record, Tolerances, Variant, improve};

    /// `a b` as the exact pair `(p, e)` with `p + e = a b` (an FMA
    /// error-free product; exact unless `a b` underflows).
    fn two_product(a: f64, b: f64) -> [f64; 2] {
        let p = a * b;
        [p, a.mul_add(b, -p)]
    }

    /// Whether the exact sum of `terms` is zero: Shewchuk's exact partials
    /// (the algorithm of Python's `math.fsum`) hold the exact sum as
    /// non-overlapping partials, which sum to zero only if every one is
    /// zero.
    fn exact_sum_is_zero(terms: impl IntoIterator<Item = f64>) -> bool {
        let mut partials: Vec<f64> = Vec::new();
        for mut x in terms {
            let mut kept = 0;
            for index in 0..partials.len() {
                let mut y = partials[index];
                if x.abs() < y.abs() {
                    std::mem::swap(&mut x, &mut y);
                }
                let high = x + y;
                let low = y - (high - x);
                if low != 0.0 {
                    partials[kept] = low;
                    kept += 1;
                }
                x = high;
            }
            partials.truncate(kept);
            partials.push(x);
        }
        partials.iter().all(|partial| *partial == 0.0)
    }

    /// Whether a binary64 point is EXACTLY physical: every leg exactly on
    /// the shell of its binary64 mass (`E² - p² - m² = 0` in exact
    /// arithmetic on the binary64 values) and every component exactly
    /// conserved. Exact above binary64's underflow (a defect below the
    /// square of the smallest normal number is not seen).
    pub fn exactly_physical(legs: &[[f64; 4]], n_initial: usize, masses: &[f64]) -> bool {
        let shells = legs.iter().zip(masses).all(|(leg, mass)| {
            let terms = two_product(leg[0], leg[0])
                .into_iter()
                .chain((1..4).flat_map(|i| two_product(leg[i], leg[i]).map(|t| -t)))
                .chain(two_product(*mass, *mass).map(|t| -t));
            exact_sum_is_zero(terms)
        });
        shells
            && (0..4).all(|mu| {
                exact_sum_is_zero(
                    legs.iter().enumerate().map(
                        |(k, leg)| {
                            if k < n_initial { leg[mu] } else { -leg[mu] }
                        },
                    ),
                )
            })
    }

    /// Improve one binary64 point toward the binary64 masses `masses` (each
    /// leg's target the square `m²` rounded once, as `improve_exact` rounds
    /// an exact target): an exactly physical point is passed unchanged
    /// (status "exact", as `improve_exact`), otherwise [`improve`] runs in
    /// binary64 (with MadLoop's chain when `fallback`).
    pub fn improve_point(
        legs: &[[f64; 4]],
        n_initial: usize,
        masses: &[f64],
        variant: Variant,
        fallback: bool,
        tolerances: &Tolerances,
    ) -> Improved<f64> {
        assert_eq!(legs.len(), masses.len(), "one mass per leg");
        let finite = legs.iter().flatten().chain(masses).all(|x| x.is_finite());
        if finite && exactly_physical(legs, n_initial, masses) {
            return Improved {
                legs: legs.to_vec(),
                record: Record::exact(variant, fallback),
            };
        }
        let masses2: Vec<f64> = masses.iter().map(|mass| mass * mass).collect();
        improve(legs, n_initial, &masses2, variant, fallback, tolerances)
    }
}

#[cfg(feature = "symbolica-runtime")]
pub use exact::{Arithmetic, exactly_physical, improve_exact};

/// The improvement of exact (rational) points in a named carrier: the loop
/// runtimes' and the Python API's entry.
#[cfg(feature = "symbolica-runtime")]
mod exact {
    use super::*;
    use symbolica::domains::float::{DoubleFloat, Float, RealLike, RoundingDirection};
    use symbolica::domains::integer::Integer;
    use symbolica::domains::rational::Rational;

    /// The carrier of an evaluation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Arithmetic {
        Binary64,
        DoubleDouble,
        /// MPFR at `bits` bits.
        Arbitrary(u32),
    }

    impl Arithmetic {
        pub fn is_binary64(self) -> bool {
            self == Arithmetic::Binary64
        }
    }

    fn square(leg: &[Rational; 4]) -> Rational {
        &(&(&(&leg[0] * &leg[0]) - &(&leg[1] * &leg[1])) - &(&leg[2] * &leg[2]))
            - &(&leg[3] * &leg[3])
    }

    /// Whether a point is exactly on its shells (`masses2`; a leg without a
    /// bound mass is on its own) and exactly conserving.
    pub fn exactly_physical(
        legs: &[[Rational; 4]],
        n_initial: usize,
        masses2: &[Option<Rational>],
    ) -> bool {
        let shells = legs.iter().zip(masses2).all(|(leg, m2)| match m2 {
            Some(m2) => square(leg) == *m2,
            None => true,
        });
        shells
            && (0..4).all(|mu| {
                let mut sum = Rational::from(0);
                for (k, leg) in legs.iter().enumerate() {
                    sum = if k < n_initial {
                        &sum + &leg[mu]
                    } else {
                        &sum - &leg[mu]
                    };
                }
                sum.is_zero()
            })
    }

    /// The binary64 value nearest to `value` (one rounding, as
    /// `loop_v2::dress::cast`).
    fn to_binary64(value: &Rational) -> f64 {
        const EXACT: i64 = 1 << f64::MANTISSA_DIGITS;
        match (value.numerator_ref(), value.denominator_ref()) {
            (Integer::Single(n), Integer::Single(d)) if n.abs() <= EXACT && *d <= EXACT => {
                *n as f64 / *d as f64
            }
            _ => RealLike::to_f64(&value.to_multi_prec_float(f64::MANTISSA_DIGITS)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run<T: PsReal>(
        legs: &[[Rational; 4]],
        n_initial: usize,
        targets: &[Rational],
        variant: Variant,
        fallback: bool,
        tolerances: &Tolerances,
        cast: impl Fn(&Rational) -> T,
        back: impl Fn(&T) -> Rational,
    ) -> (Vec<[Rational; 4]>, Record) {
        let carried: Vec<[T; 4]> = legs.iter().map(|leg| leg.each_ref().map(&cast)).collect();
        let masses: Vec<T> = targets.iter().map(&cast).collect();
        let improved = improve(&carried, n_initial, &masses, variant, fallback, tolerances);
        if improved.record.status == Status::Improved {
            let out = improved
                .legs
                .iter()
                .map(|leg| leg.each_ref().map(&back))
                .collect();
            (out, improved.record)
        } else {
            // Evaluated as given: the exact input itself (bitwise the
            // evaluation without the improvement).
            (legs.to_vec(), improved.record)
        }
    }

    /// Improve an exact point in `arithmetic`: an exactly physical point is
    /// passed unchanged (MadLoopParams.dat: the deformation 'leaves the input
    /// PS point unchanged if it already satisfies the physical conditions');
    /// otherwise the point is cast once to the carrier, improved there and
    /// returned as the exact rationals of the carrier values. `masses2`: the
    /// executor's bound leg masses (`V2Batch::leg_masses`); a leg without one
    /// takes its own exact `p²`, cast once.
    pub fn improve_exact(
        legs: &[[Rational; 4]],
        n_initial: usize,
        masses2: &[Option<Rational>],
        arithmetic: Arithmetic,
        variant: Variant,
        fallback: bool,
        tolerances: &Tolerances,
    ) -> (Vec<[Rational; 4]>, Record) {
        assert_eq!(legs.len(), masses2.len(), "one squared mass per leg");
        if exactly_physical(legs, n_initial, masses2) {
            return (legs.to_vec(), Record::exact(variant, fallback));
        }
        let targets: Vec<Rational> = legs
            .iter()
            .zip(masses2)
            .map(|(leg, m2)| m2.clone().unwrap_or_else(|| square(leg)))
            .collect();
        match arithmetic {
            Arithmetic::Binary64 => run(
                legs,
                n_initial,
                &targets,
                variant,
                fallback,
                tolerances,
                to_binary64,
                |x: &f64| Rational::try_from(*x).expect("a finite improved component"),
            ),
            Arithmetic::DoubleDouble => run(
                legs,
                n_initial,
                &targets,
                variant,
                fallback,
                tolerances,
                |r: &Rational| -> DoubleFloat { r.into() },
                |x: &DoubleFloat| {
                    let parts = (*x).into_inner();
                    &Rational::try_from(parts.hi()).expect("finite")
                        + &Rational::try_from(parts.lo()).expect("finite")
                },
            ),
            Arithmetic::Arbitrary(bits) => run(
                legs,
                n_initial,
                &targets,
                variant,
                fallback,
                tolerances,
                |r: &Rational| {
                    Mpfr(Float::from_rational_round(
                        r,
                        bits,
                        RoundingDirection::Nearest,
                    ))
                },
                |x: &Mpfr| x.0.to_rational(),
            ),
        }
    }
}

#[cfg(test)]
#[path = "kinematics_improvement_tests.rs"]
mod tests;
