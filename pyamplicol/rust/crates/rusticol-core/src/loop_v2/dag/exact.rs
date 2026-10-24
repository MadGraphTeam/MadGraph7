// SPDX-License-Identifier: 0BSD

//! Bit-exact binary64 arithmetic of the value-equivalence derivation
//! (`generation/dag_equivalence.py`).
//!
//! Every function is the Python helper with the same name, operation for
//! operation, so a result is the oracle's to the bit:
//!
//! * [`fsum`] is CPython's `math.fsum` (Shewchuk partials with the
//!   half-even correction, the same special values and errors);
//! * [`python_complex_quotient`] is CPython's `_Py_c_quot` (Smith's
//!   algorithm), which `complex(a) / complex(b)` runs;
//! * [`exact_product`] and [`exact_ratio`] decide exact representability
//!   with exact rationals (Symbolica's `Rational`, a bignum with no domain
//!   cap, RUST-MIGRATION rule 4) exactly where Python uses `Fraction`.
//!
//! Rust never contracts `a * b + c` into a fused multiply-add, and the
//! oracle's CPython is built for baseline x86-64 (no FMA), so plain binary64
//! expressions round identically. Exact rationals call no Symbolica licence
//! check (they are `numerica` domains).

use super::{DagError, DagResult};
use symbolica::domains::rational::Rational;

/// A complex binary64 pair `(re, im)` (the oracle's `_ComplexWeight`).
pub type Complex = [f64; 2];

/// `_canonical_zero`.
#[inline]
pub fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// `_canonical_complex_weight`.
#[inline]
pub fn canonical(value: Complex) -> Complex {
    [canonical_zero(value[0]), canonical_zero(value[1])]
}

/// `_complex_weight_is_finite`.
#[inline]
pub fn finite(value: Complex) -> bool {
    value[0].is_finite() && value[1].is_finite()
}

/// Python tuple equality of two pairs (`0.0 == -0.0`, NaN unequal).
#[inline]
pub fn pair_eq(left: Complex, right: Complex) -> bool {
    left[0] == right[0] && left[1] == right[1]
}

/// `_complex_weight_bits_equal` (`float.hex` equality: the sign of a zero
/// counts).
#[inline]
pub fn bits_equal(left: Complex, right: Complex) -> bool {
    left[0].to_bits() == right[0].to_bits() && left[1].to_bits() == right[1].to_bits()
}

/// `_complex_weight_mul`, the oracle's operation order.
#[inline]
pub fn complex_mul(left: Complex, right: Complex) -> Complex {
    [
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    ]
}

/// A pair as a hashable key with Python's float equality (`0.0 == -0.0`).
/// Callers never key a NaN.
#[inline]
pub fn key_bits(value: Complex) -> [u64; 2] {
    let value = canonical(value);
    [value[0].to_bits(), value[1].to_bits()]
}

/// CPython's `math.fsum` (`Modules/mathmodule.c`, 3.11): the correctly
/// rounded sum, `nan` with a NaN summand, `±inf` with infinities of one sign;
/// `inf + -inf` and an intermediate overflow are the oracle's `ValueError` /
/// `OverflowError`.
pub fn fsum(values: impl IntoIterator<Item = f64>) -> DagResult<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0f64;
    let mut inf_sum = 0.0f64;
    for item in values {
        let mut x = item;
        let saved = x;
        let mut i = 0usize;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            if !x.is_finite() {
                if saved.is_finite() {
                    return Err(DagError::Invalid("intermediate overflow in fsum".into()));
                }
                if saved.is_infinite() {
                    inf_sum += saved;
                }
                special_sum += saved;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    // `special_sum != 0.0` holds for a NaN too.
    if special_sum != 0.0 || special_sum.is_nan() {
        if inf_sum.is_nan() {
            return Err(DagError::Invalid("-inf + inf in fsum".into()));
        }
        return Ok(special_sum);
    }
    let mut n = partials.len();
    let mut hi = 0.0f64;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0f64;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Ok(hi)
}

/// CPython's `_Py_c_quot` (`Objects/complexobject.c`, 3.11): Smith's
/// division, `None` where Python raises `ZeroDivisionError`.
pub fn python_complex_quotient(a: Complex, b: Complex) -> Option<Complex> {
    let abs_breal = if b[0] < 0.0 { -b[0] } else { b[0] };
    let abs_bimag = if b[1] < 0.0 { -b[1] } else { b[1] };
    if abs_breal >= abs_bimag {
        if abs_breal == 0.0 {
            return None;
        }
        let ratio = b[1] / b[0];
        let denom = b[0] + b[1] * ratio;
        Some([(a[0] + a[1] * ratio) / denom, (a[1] - a[0] * ratio) / denom])
    } else if abs_bimag >= abs_breal {
        let ratio = b[0] / b[1];
        let denom = b[0] * ratio + b[1];
        Some([(a[0] * ratio + a[1]) / denom, (a[1] * ratio - a[0]) / denom])
    } else {
        Some([f64::NAN, f64::NAN])
    }
}

/// `Fraction.from_float` of a finite float.
fn rational(value: f64) -> Rational {
    Rational::try_from(value).expect("a finite float is a rational")
}

/// `_exact_fraction_as_f64`: the value as binary64 when it is exactly one
/// (finite), else `None`.
///
/// `Rational::to_f64` returns an exactly representable value exactly (its
/// numerator and denominator are then binary64 integers, or it rounds once
/// at 53 bits), so the round trip decides representability as Python's
/// `Fraction.from_float(float(value)) == value` does.
pub fn exact_f64(value: &Rational) -> Option<f64> {
    let candidate = value.to_f64();
    if !candidate.is_finite() {
        return None;
    }
    if Rational::try_from(candidate).ok().as_ref() != Some(value) {
        return None;
    }
    Some(canonical_zero(candidate))
}

/// `±1` and `±i` (`_EXACT_UNITS`, compared with float equality).
fn exact_unit(value: Complex) -> bool {
    pair_eq(value, [1.0, 0.0])
        || pair_eq(value, [-1.0, 0.0])
        || pair_eq(value, [0.0, 1.0])
        || pair_eq(value, [0.0, -1.0])
}

/// `_exact_representable_complex_product`.
pub fn exact_product(left: Complex, right: Complex) -> Option<Complex> {
    if !finite(left) || !finite(right) {
        return None;
    }
    if exact_unit(left) || exact_unit(right) {
        // A factor ±1 or ±i only swaps and negates the other's components.
        return Some(canonical(complex_mul(left, right)));
    }
    let (lr, li) = (rational(left[0]), rational(left[1]));
    let (rr, ri) = (rational(right[0]), rational(right[1]));
    let real = &(&lr * &rr) - &(&li * &ri);
    let imaginary = &(&lr * &ri) + &(&li * &rr);
    let product = [exact_f64(&real)?, exact_f64(&imaginary)?];
    let reconstructed = canonical(complex_mul(left, right));
    bits_equal(reconstructed, product).then_some(product)
}

/// `_exact_representable_complex_ratio`.
pub fn exact_ratio(numerator: Complex, denominator: Complex) -> Option<Complex> {
    if !finite(numerator) || !finite(denominator) || pair_eq(denominator, [0.0, 0.0]) {
        return None;
    }
    let (nr, ni) = (rational(numerator[0]), rational(numerator[1]));
    let (dr, di) = (rational(denominator[0]), rational(denominator[1]));
    let norm = &(&dr * &dr) + &(&di * &di);
    if norm.is_zero() {
        return None;
    }
    let real = &(&(&nr * &dr) + &(&ni * &di)) / &norm;
    let imaginary = &(&(&ni * &dr) - &(&nr * &di)) / &norm;
    let factor = [exact_f64(&real)?, exact_f64(&imaginary)?];
    let reconstructed = canonical(complex_mul(denominator, factor));
    bits_equal(reconstructed, canonical(numerator)).then_some(factor)
}

/// `_roundtrip_complex_ratio`: a finite lookup ratio that reconstructs
/// `numerator` exactly.
pub fn roundtrip_ratio(numerator: Complex, denominator: Complex) -> Option<Complex> {
    if !finite(numerator) || !finite(denominator) || pair_eq(denominator, [0.0, 0.0]) {
        return None;
    }
    let result = canonical(python_complex_quotient(numerator, denominator)?);
    if !finite(result) {
        return None;
    }
    let reconstructed = canonical(complex_mul(denominator, result));
    bits_equal(reconstructed, numerator).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsum_is_correctly_rounded() {
        assert_eq!(fsum([0.1; 10]).unwrap(), 1.0);
        assert_eq!(fsum([1e-16, 1.0, 1e16]).unwrap(), 1.0000000000000002e16);
        assert_eq!(fsum([1e100, 1.0, -1e100, 1e-100]).unwrap(), 1.0);
        assert_eq!(fsum([-0.0]).unwrap().to_bits(), 0.0f64.to_bits());
        assert_eq!(fsum(std::iter::empty()).unwrap(), 0.0);
        assert!(fsum([f64::NAN, 1.0]).unwrap().is_nan());
        assert_eq!(fsum([f64::INFINITY, 1.0]).unwrap(), f64::INFINITY);
        assert!(fsum([f64::INFINITY, f64::NEG_INFINITY]).is_err());
        assert!(fsum([f64::MAX, f64::MAX]).is_err());
    }

    #[test]
    fn exact_products_and_ratios_fail_closed() {
        assert_eq!(exact_product([2.0, 0.0], [0.5, 0.25]), Some([1.0, 0.5]));
        // 0.1 * 3 is not exactly a binary64 value.
        assert_eq!(exact_product([0.1, 0.0], [3.0, 0.0]), None);
        assert_eq!(exact_product([0.0, -1.0], [3.0, 2.0]), Some([2.0, -3.0]));
        assert_eq!(exact_ratio([1.0, 0.0], [2.0, 0.0]), Some([0.5, 0.0]));
        assert_eq!(exact_ratio([1.0, 0.0], [3.0, 0.0]), None);
        assert_eq!(exact_ratio([0.0, 2.0], [0.0, 1.0]), Some([2.0, 0.0]));
        assert_eq!(roundtrip_ratio([1.0, 0.0], [0.0, 1.0]), Some([0.0, -1.0]));
        assert_eq!(python_complex_quotient([1.0, 1.0], [0.0, 0.0]), None);
        let tiny = f64::from_bits(1);
        assert_eq!(
            exact_ratio([tiny, 0.0], [0.5, 0.0]),
            Some([2.0 * tiny, 0.0])
        );
    }
}
