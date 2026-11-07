// SPDX-License-Identifier: 0BSD

//! The prime-field carrier of the exact helicity-zero filter (M-LIT LIT-R2,
//! RULED Q10 = (a); DECISIONS 2026-11-06 ~09:00 (2) and ~13:45): the field
//! `F_p`, `p = 2^61 − 1`, as a [`Carrier`](super::dress::Carrier) of the
//! dressing, so a plan's family numerators can be evaluated exactly modulo
//! `p` at random points (Schwartz–Zippel: a non-zero polynomial of total
//! degree `d` vanishes at a uniformly random point of `F_p^n` with
//! probability at most `d / p`).
//!
//! Complex values are pairs `(re, im)` with `i² = −1`; since `p ≡ 3 (mod 4)`
//! `−1` is not a square in `F_p`, so `F_p[i]` is the field `F_{p²}` and the
//! test holds there too. A rational casts as `n · d⁻¹ mod p` (its denominator
//! must be invertible: a multiple of `p` is an error the caller sees as a
//! non-finite value, [`Fp::is_finite`]). The carrier dresses numerators only:
//! the operations the dressing never runs on a field element (comparisons of
//! magnitude, precision) answer as for an exact carrier. The arithmetic is
//! numerica's `FiniteField<Mersenne64>` (Symbolica's domains); `Fp` adds the
//! non-finite marker and the carrier traits.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use symbolica::domains::finite_field::{FiniteField, FiniteFieldCore, Mersenne64};
use symbolica::domains::float::{FloatLike, RealLike, SingleFloat};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;
use symbolica::domains::{Field, Ring, RingOps};
use symbolica::rand::Rng;

/// The prime `2^61 − 1` (a Mersenne prime, `≡ 3 mod 4`).
pub const PRIME: u64 = Mersenne64::PRIME;

/// The field `F_p` of numerica (its elements are the residues in `[0, p)`).
#[inline(always)]
fn field() -> FiniteField<Mersenne64> {
    FiniteField::<Mersenne64>::new(Mersenne64::new())
}

/// An element of `F_p` (`p` = [`PRIME`]), or the marker of a rational whose
/// denominator `p` divides (not finite).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp(u64);

/// The value of the non-finite marker (outside `[0, p)`).
const INVALID: u64 = u64::MAX;

impl Fp {
    /// The residue `value mod p`.
    pub fn new(value: u64) -> Self {
        Self(value % PRIME)
    }

    /// The residue in `[0, p)` (or `u64::MAX` for the non-finite marker).
    pub fn residue(self) -> u64 {
        self.0
    }

    fn valid(self, other: Self) -> bool {
        self.0 != INVALID && other.0 != INVALID
    }

    fn from_integer(value: &Integer) -> Self {
        let reduced = match value {
            Integer::Single(n) => *n,
            other => match other.clone().symmetric_mod(&Integer::from(PRIME as i64)) {
                Integer::Single(n) => n,
                // A residue in (-p/2, p/2] fits in an i64.
                _ => return Self(INVALID),
            },
        };
        Self(reduced.rem_euclid(PRIME as i64) as u64)
    }

    /// `self^e`.
    fn power(self, e: u64) -> Self {
        if self.0 == INVALID {
            return self;
        }
        Self(field().pow(&self.0, e))
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({})", self.0)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == INVALID {
            f.write_str("invalid")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl fmt::LowerExp for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Neg for Fp {
    type Output = Self;
    fn neg(self) -> Self {
        match self.0 {
            INVALID => self,
            x => Self(RingOps::neg(&field(), x)),
        }
    }
}

impl Add for Fp {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        if !self.valid(other) {
            return Self(INVALID);
        }
        Self(RingOps::add(&field(), self.0, other.0))
    }
}

impl Sub for Fp {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl Mul for Fp {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        if !self.valid(other) {
            return Self(INVALID);
        }
        Self(RingOps::mul(&field(), self.0, other.0))
    }
}

impl Div for Fp {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        self * other.inverse()
    }
}

impl Fp {
    /// `self⁻¹`, the non-finite marker for zero.
    fn inverse(self) -> Self {
        match self.0 {
            0 | INVALID => Self(INVALID),
            x => Self(field().inv(&x)),
        }
    }
}

macro_rules! by_reference {
    ($($trait:ident $method:ident $assign:ident $assign_method:ident),*) => {
        $(
            impl<'a> $trait<&'a Fp> for Fp {
                type Output = Fp;
                fn $method(self, other: &'a Fp) -> Fp {
                    $trait::$method(self, *other)
                }
            }
            impl $assign for Fp {
                fn $assign_method(&mut self, other: Fp) {
                    *self = $trait::$method(*self, other);
                }
            }
            impl<'a> $assign<&'a Fp> for Fp {
                fn $assign_method(&mut self, other: &'a Fp) {
                    *self = $trait::$method(*self, *other);
                }
            }
        )*
    };
}

by_reference!(
    Add add AddAssign add_assign,
    Sub sub SubAssign sub_assign,
    Mul mul MulAssign mul_assign,
    Div div DivAssign div_assign
);

impl FloatLike for Fp {
    fn set_from(&mut self, other: &Self) {
        *self = *other;
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        *self * *a + *b
    }

    fn neg(&self) -> Self {
        -*self
    }

    fn zero(&self) -> Self {
        Self(0)
    }

    fn new_zero() -> Self {
        Self(0)
    }

    fn one(&self) -> Self {
        Self(1)
    }

    fn pow(&self, e: u64) -> Self {
        self.power(e)
    }

    fn inv(&self) -> Self {
        self.inverse()
    }

    fn from_usize(&self, a: usize) -> Self {
        Self::new(a as u64)
    }

    fn from_i64(&self, a: i64) -> Self {
        let magnitude = Self::new(a.unsigned_abs());
        if a < 0 { -magnitude } else { magnitude }
    }

    fn get_precision(&self) -> u32 {
        u32::MAX
    }

    fn set_precision(&mut self, _precision: u32) {}

    fn get_epsilon(&self) -> f64 {
        0.0
    }

    fn fixed_precision(&self) -> bool {
        true
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        Self::new(rng.random_range(0..PRIME))
    }

    fn is_fully_zero(&self) -> bool {
        self.0 == 0
    }
}

impl SingleFloat for Fp {
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }

    /// A residue is finite; the marker of a division by a multiple of `p`
    /// is not.
    fn is_finite(&self) -> bool {
        self.0 != INVALID
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        Self::from_integer(&rat.numerator()) / Self::from_integer(&rat.denominator())
    }
}

impl RealLike for Fp {
    fn to_usize_clamped(&self) -> usize {
        usize::try_from(self.0).unwrap_or(usize::MAX)
    }

    /// The residue as a float (a diagnostic; a field element has no size).
    fn to_f64(&self) -> f64 {
        self.0 as f64
    }

    fn round_to_nearest_integer(&self) -> Integer {
        Integer::from(self.0 as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_operations_hold_modulo_p() {
        let a = Fp::new(PRIME - 3);
        let b = Fp::new(12_345_678_901);
        assert_eq!((a + b).residue(), 12_345_678_898);
        assert_eq!((a - b) + b, a);
        assert_eq!((a * b) / b, a);
        assert_eq!(a * a.inv(), Fp::new(1));
        assert_eq!(-Fp::new(0), Fp::new(0));
        assert_eq!(Fp::new(2).pow(61), Fp::new(1));
        assert_eq!(Fp::new(0).from_i64(-1), Fp::new(PRIME - 1));
    }

    #[test]
    fn rationals_cast_as_their_quotient() {
        let zero = Fp::new(0);
        let half = zero.from_rational(&Rational::from((1, 2)));
        assert_eq!(half * Fp::new(2), Fp::new(1));
        let negative = zero.from_rational(&Rational::from((-7, 3)));
        assert_eq!(negative * Fp::new(3), zero.from_i64(-7));
        let big = Rational::from_unchecked(
            Integer::from(PRIME as i64) * Integer::from(5) + Integer::from(4),
            Integer::from(1),
        );
        assert_eq!(zero.from_rational(&big), Fp::new(4));
        let divided = zero.from_rational(&Rational::from_unchecked(
            Integer::from(1),
            Integer::from(PRIME as i64),
        ));
        assert!(!divided.is_finite());
        assert!(!(divided + Fp::new(1)).is_finite());
    }
}
