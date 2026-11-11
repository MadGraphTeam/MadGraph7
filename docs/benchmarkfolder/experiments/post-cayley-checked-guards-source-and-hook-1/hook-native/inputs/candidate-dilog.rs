//! Complex dilogarithm at arbitrary working precision.
//!
//! Algorithm and branch conventions follow Symbolica's binary64 kernel
//! `symbolica::transcendental::dilog_complex_f64`, which the binary64 native
//! evaluator uses: the principal branch; an exactly real argument `x > 1`
//! takes the imaginary part `-pi ln x`, `x = -1` a negative zero imaginary
//! part. The argument is mapped to `|z| <= 1`, `Re z <= 1/2` and the
//! Bernoulli series in `u = -ln(1 - z)`,
//! `Li2(z) = u - u^2/4 + sum_{k>=1} B_2k u^(2k+1) / (2k+1)!`,
//! is summed to the working precision (`|u| <= pi/3`, so each term gains
//! `log2(36)` bits). The coefficients are exact rationals from the tangent
//! numbers (Brent and Harvey), rounded once per precision.
//!
//! Symbolica's generic `polylog(2, z)` evaluation, used before for MPFR and
//! double-double arguments, costs about 0.35-0.5 ms per call at 106-256 bits,
//! which dominated every higher-precision box evaluation.

use std::{cell::RefCell, rc::Rc, sync::Mutex};

use symbolica::domains::{
    float::{Complex as C, Float, SingleFloat},
    integer::Integer,
    rational::Rational,
};

use super::primitives::NativeFloat;

/// Tangent numbers `T_1..=T_n` (1, 2, 16, 272, ...), cached.
fn tangent_numbers(n: usize) -> Vec<Integer> {
    static CACHE: Mutex<Vec<Integer>> = Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.len() < n {
        // Brent and Harvey, "Fast computation of Bernoulli, tangent and
        // secant numbers", Algorithm TangentNumbers.
        let mut t = vec![Integer::from(0i64); n + 1];
        t[1] = Integer::from(1i64);
        for k in 2..=n {
            t[k] = Integer::from((k - 1) as i64) * &t[k - 1];
        }
        for k in 2..=n {
            for j in k..=n {
                t[j] = Integer::from((j - k) as i64) * &t[j - 1]
                    + Integer::from((j - k + 2) as i64) * &t[j];
            }
        }
        *cache = t.split_off(1);
    }
    cache[..n].to_vec()
}

/// `B_2k / (2k+1)!` for `k = 1..=count`, exactly:
/// `B_2k = (-1)^(k-1) 2k T_k / (2^2k (2^2k - 1))`.
pub(crate) fn series_coefficients(count: usize) -> Vec<Rational> {
    tangent_numbers(count)
        .into_iter()
        .enumerate()
        .map(|(index, tangent)| {
            let k = index as u64 + 1;
            let power = Integer::from(4i64).pow(k);
            let denominator = power.clone()
                * &(power - Integer::from(1i64))
                * &Integer::factorial(2 * k as u32 + 1);
            let mut numerator = tangent * &Integer::from(2 * k as i64);
            if k % 2 == 0 {
                numerator = -numerator;
            }
            Rational::new(numerator, denominator)
        })
        .collect()
}


// Optional, scoped, thread-local evidence supplied by an embedding host.
// The probe must only inspect current floating-point controls; it must neither
// change the environment nor evaluate scientific kernels. Unknown means false.
#[cfg(feature = "checked-dilog-rounding")]
mod checked_rounding {
    use std::cell::Cell;
    std::thread_local! {
        static PROBE: Cell<Option<fn() -> bool>> = const { Cell::new(None) };
    }
    struct Restore(Option<fn() -> bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROBE.with(|slot| slot.set(self.0));
        }
    }
    /// Install a read-only, fresh-control probe only for this thread and scope.
    /// No pointer is retained after normal return or Rust panic unwinding.
    /// Returning true promises binary64 nearest rounding and no FTZ/DAZ.
    /// The host owns this promise; false/absent probes retain the old arithmetic.
    pub fn with_probe<R>(probe: fn() -> bool, action: impl FnOnce() -> R) -> R {
        let previous = PROBE.with(|slot| slot.replace(Some(probe)));
        let _restore = Restore(previous);
        action()
    }
    pub(super) fn now() -> bool {
        // No Cell access or borrow remains active while calling host code.
        // Reentry from a probe sees no probe and therefore the old path.
        let previous = PROBE.with(|slot| slot.replace(None));
        let _restore = Restore(previous);
        previous.is_some_and(|probe| probe())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn yes() -> bool { true }
        fn no() -> bool { false }
        fn reentrant() -> bool { !now() }
        fn panic_probe() -> bool { panic!("probe unwind"); }
        #[test]
        fn default_nesting_reentry_and_thread_isolation() {
            assert!(!now());
            with_probe(yes, || {
                assert!(now());
                with_probe(no, || assert!(!now()));
                assert!(now());
                std::thread::spawn(|| assert!(!now())).join().unwrap();
                with_probe(reentrant, || assert!(now()));
            });
            assert!(!now());
        }
        #[test]
        fn restores_action_and_probe_panics() {
            with_probe(yes, || {
                let result = std::panic::catch_unwind(|| with_probe(no, || panic!("action unwind")));
                assert!(result.is_err());
                assert!(now());
                let result = std::panic::catch_unwind(|| with_probe(panic_probe, now));
                assert!(result.is_err());
                assert!(now());
            });
            assert!(!now());
        }
    }
}
#[cfg(feature = "checked-dilog-rounding")]
pub use checked_rounding::with_probe as with_checked_dilog_rounding_probe;

/// Series terms needed at `bits` for `|u|^2 <= modulus` (at most `(pi/3)^2`).
pub(crate) fn series_terms(bits: u32, modulus: f64) -> usize {
    #[cfg(feature = "checked-dilog-rounding")]
    let ratio = if modulus.to_bits() > 0 && modulus.to_bits() <= 0x0030_0000_0000_0000
        && checked_rounding::now()
    {
        // Under the fresh checked contract the old binary64 ratio overflows
        // to positive infinity. Preserve the old masked coefficient count.
        // This intentionally suppresses old overflow/inexact status effects.
        f64::INFINITY
    } else {
        4.0 * std::f64::consts::PI * std::f64::consts::PI / modulus
    };
    #[cfg(not(feature = "checked-dilog-rounding"))]
    let ratio = 4.0 * std::f64::consts::PI * std::f64::consts::PI / modulus;
    ((f64::from(bits) + 6.0) * std::f64::consts::LN_2 / ratio.ln()).ceil() as usize + 1
}

/// Coefficient count for every argument at `bits` (`|u| <= pi/3`).
pub(crate) fn table_length(bits: u32) -> usize {
    series_terms(bits, std::f64::consts::PI * std::f64::consts::PI / 9.0)
}

thread_local! {
    static FLOAT_TABLES: RefCell<Vec<(u32, Rc<Vec<Float>>)>> = const { RefCell::new(Vec::new()) };
}

/// [`series_coefficients`] rounded at `bits`, cached per thread and precision.
pub(crate) fn float_coefficients(bits: u32) -> Rc<Vec<Float>> {
    FLOAT_TABLES.with(|tables| {
        if let Some((_, table)) = tables.borrow().iter().find(|(b, _)| *b == bits) {
            return table.clone();
        }
        let prototype = Float::new(bits);
        let table = Rc::new(
            series_coefficients(table_length(bits))
                .iter()
                .map(|c| prototype.from_rational(c))
                .collect::<Vec<_>>(),
        );
        tables.borrow_mut().push((bits, table.clone()));
        table
    })
}

/// [`series_coefficients`] rounded once to double-double (via 160 bits).
pub(crate) fn double_coefficients() -> &'static [symbolica::domains::float::DoubleFloat] {
    static TABLE: std::sync::OnceLock<Vec<symbolica::domains::float::DoubleFloat>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let prototype = Float::new(160);
        series_coefficients(table_length(106))
            .iter()
            .map(|c| prototype.from_rational(c).to_double_float())
            .collect()
    })
}

/// The domain's complex product (binary64 and MPFR: the plain expression;
/// double-double: the fused kernel).
fn mul<T: NativeFloat>(a: &C<T>, b: &C<T>) -> C<T> {
    T::complex_mul(a, b)
}

/// `u - u^2/4 + sum_k c_k u^(2k+1)` with `coefficients[k-1] = c_k`.
fn series<T: NativeFloat>(u: &C<T>, coefficients: &[T], bits: u32) -> C<T> {
    let v = mul(u, u);
    let modulus = v.re.to_f64().hypot(v.im.to_f64());
    let count = if modulus == 0.0 {
        0
    } else {
        series_terms(bits, modulus).min(coefficients.len())
    };
    let quarter_v = C::new(
        v.re.clone() / &v.re.from_usize(4),
        v.im.clone() / &v.im.from_usize(4),
    );
    let mut result = C::new(u.re.clone() - &quarter_v.re, u.im.clone() - &quarter_v.im);
    if count == 0 {
        return result;
    }
    // Horner in v: s = sum_k c_k v^(k-1).
    let mut sum = C::new(coefficients[count - 1].clone(), u.re.zero());
    for coefficient in coefficients[..count - 1].iter().rev() {
        sum = mul(&sum, &v);
        sum.re += coefficient;
    }
    let term = mul(&mul(u, &v), &sum);
    result.re += &term.re;
    result.im += &term.im;
    result
}

/// Principal `ln(1 + w)`, keeping the relative precision of small `w`.
fn log_one_plus<T: NativeFloat>(w: &C<T>) -> C<T> {
    let one = w.re.one();
    let real = one.clone() + &w.re;
    let small = |x: &T| x.norm() + &x.norm() < one;
    let modulus_log = if small(&w.re) && small(&w.im) {
        let two = w.re.from_usize(2);
        (w.re.clone() * &(two.clone() + &w.re) + &(w.im.clone() * &w.im)).log1p() / &two
    } else {
        NativeFloat::hypot(&real, &w.im).log()
    };
    C::new(modulus_log, w.im.atan2(&real))
}

fn log<T: NativeFloat>(z: &C<T>) -> C<T> {
    C::new(NativeFloat::hypot(&z.re, &z.im).log(), z.im.atan2(&z.re))
}

fn neg<T: NativeFloat>(z: &C<T>) -> C<T> {
    C::new(-z.re.clone(), -z.im.clone())
}

/// `Li2(z)` with `coefficients` from [`series_coefficients`] rounded at the
/// working precision `bits` (at least [`table_length`] of them).
pub(crate) fn dilog<T: NativeFloat>(z: &C<T>, coefficients: &[T], bits: u32) -> C<T> {
    let zero = z.re.zero();
    if !z.re.is_finite() || !z.im.is_finite() {
        let nan = zero.clone() / &zero;
        return C::new(nan.clone(), nan);
    }
    let pi = z.re.pi();
    let pi_squared_over_six = pi.clone() * &pi / &z.re.from_usize(6);
    if z.im.is_zero() {
        let one = z.re.one();
        if z.re == one {
            return C::new(pi_squared_over_six, zero);
        }
        // The real part does not depend on the side of the cut.
        let value = dilog_complex(&C::new(z.re.clone(), zero.clone()), coefficients, bits, &pi_squared_over_six);
        let imaginary = if z.re > one {
            -(pi * &z.re.log())
        } else if z.re == -one {
            -zero
        } else {
            zero
        };
        return C::new(value.re, imaginary);
    }
    dilog_complex(z, coefficients, bits, &pi_squared_over_six)
}

fn dilog_complex<T: NativeFloat>(z: &C<T>, coefficients: &[T], bits: u32, pi_squared_over_six: &T) -> C<T> {
    let one = z.re.one();
    let one_minus_re = one.clone() - &z.re;
    let half = one.clone() / &z.re.from_usize(2);
    if z.re > half && one_minus_re.clone() * &one_minus_re + &(z.im.clone() * &z.im) <= one {
        // |1-z| <= 1: Li2(z) = -Li2(1-z) + pi^2/6 - ln(z) ln(1-z), with the
        // series variable -ln(z) of Li2(1-z).
        let log_z = log_one_plus(&C::new(-one_minus_re.clone(), z.im.clone()));
        let log_one_minus_z = log(&C::new(one_minus_re, -z.im.clone()));
        let value = series(&neg(&log_z), coefficients, bits);
        let product = mul(&log_z, &log_one_minus_z);
        return C::new(
            -value.re + pi_squared_over_six - &product.re,
            -value.im - &product.im,
        );
    }
    if z.re.clone() * &z.re + &(z.im.clone() * &z.im) <= one {
        // |z| <= 1 and Re z <= 1/2, so |ln(1-z)| <= pi/3.
        let log = log_one_plus(&neg(z));
        return series(&neg(&log), coefficients, bits);
    }
    // |z| > 1 and |1-z| > 1: Li2(z) = -Li2(1/z) - pi^2/6 - ln(-z)^2/2.
    let inverse = T::complex_inverse(z);
    let log_inverse = log_one_plus(&neg(&inverse));
    let value = series(&neg(&log_inverse), coefficients, bits);
    let log_minus_z = log(&neg(z));
    let square = mul(&log_minus_z, &log_minus_z);
    let two = z.re.from_usize(2);
    C::new(
        -value.re - pi_squared_over_six - &(square.re / &two),
        -value.im - &(square.im / &two),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_numbers_and_bernoulli_coefficients() {
        let tangent: Vec<i64> = tangent_numbers(5).iter().map(|t| t.to_i64().unwrap()).collect();
        assert_eq!(tangent, [1, 2, 16, 272, 7936]);
        let c = series_coefficients(3);
        // B2/3! = 1/36, B4/5! = -1/3600, B6/7! = 1/211680.
        assert_eq!(c[0], Rational::new(1, 36));
        assert_eq!(c[1], Rational::new(-1, 3600));
        assert_eq!(c[2], Rational::new(1, 211680));
    }

    #[test]
    fn binary64_series_matches_symbolica_kernel() {
        let coefficients: Vec<f64> = series_coefficients(table_length(53))
            .iter()
            .map(|c| c.to_f64())
            .collect();
        for &(re, im) in &[
            (0.3, 0.2), (0.9, 0.1), (1.5, -0.4), (-3.0, 2.0), (4.0, 0.0), (1.0, 0.0),
            (-1.0, 0.0), (0.5, 0.0), (2.0, 1e-9), (1e-12, -1e-13), (-0.7, 0.9), (0.6, -0.8),
        ] {
            let z = C::new(re, im);
            let ours = dilog(&z, &coefficients, 53);
            let reference = symbolica::transcendental::dilog_complex_f64(z);
            let scale = reference.re.hypot(reference.im).max(1e-300);
            let error = (ours.re - reference.re).hypot(ours.im - reference.im) / scale;
            assert!(error < 1e-14, "z={z:?}: {ours:?} vs {reference:?}");
            assert_eq!(ours.im.is_sign_negative(), reference.im.is_sign_negative(), "z={z:?}");
        }
    }
}
