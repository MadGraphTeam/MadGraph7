// SPDX-License-Identifier: 0BSD

//! The assembly arithmetic of the Python API, for the SDK's plan v4 runtime.
//!
//! The Python API assembles a plan v4 virtual from the native executor's
//! outputs (`loops/v2/assemble._PointEvaluation._assemble`,
//! `loops/v2/observables._Assembly.result`) in its carrier's arithmetic:
//! Python floats in binary64, and `decimal.Decimal` at the evaluation's
//! working digits (`assemble.Precision.working_digits`, the active
//! `localcontext(prec=...)`, rounding half to even) in double-double and
//! arbitrary precision, the executor's carrier values entering as their
//! decimal texts (`runtime::decimal_complex`). [`Arith`] is that
//! arithmetic: [`Binary64`] (IEEE binary64, Python's `float` and `complex`)
//! and [`DecimalContext`] (Python's `Decimal`: every operation's exact
//! result rounded to the context's significant digits, half to even), so
//! the SDK forms bitwise (binary64) or value-identical (`Decimal`) results.
//! [`DecimalContext::pi`] is `assemble._decimal_pi` (Machin's formula at the
//! context's digits plus five, then rounded), term for term.

use std::cmp::Ordering;
use symbolica::domains::integer::Integer;

/// A decimal number `(-1)^neg * coefficient * 10^exponent`, exact (Python's
/// `Decimal` value; its representation's trailing zeros are not kept).
#[derive(Clone, Debug)]
pub struct Decimal {
    neg: bool,
    coefficient: Integer,
    exponent: i64,
}

impl PartialEq for Decimal {
    /// Value equality (as Python's `Decimal.__eq__`; `-0 == 0`).
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

fn ten(power: u64) -> Integer {
    Integer::from(10).pow(power)
}

/// Decimal digits of a nonnegative integer (one for zero).
fn digits(value: &Integer) -> i64 {
    value.to_string().trim_start_matches('-').len() as i64
}

impl Decimal {
    pub fn zero() -> Self {
        Self {
            neg: false,
            coefficient: Integer::zero(),
            exponent: 0,
        }
    }

    /// An exact integer.
    pub fn integer(value: i64) -> Self {
        Self {
            neg: value < 0,
            coefficient: Integer::from(value.unsigned_abs()),
            exponent: 0,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }

    /// The exact value of a decimal literal (`Decimal(text)`): a sign,
    /// digits with an optional fraction part and an optional exponent
    /// (`e`/`E`). Infinities and NaNs are refused.
    pub fn parse(text: &str) -> Result<Self, String> {
        let invalid = || format!("{text:?} is not a finite decimal number");
        let text = text.trim();
        let (neg, body) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(at) => (
                &body[..at],
                body[at + 1..].parse::<i64>().map_err(|_| invalid())?,
            ),
            None => (body, 0),
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty() && fraction.is_empty()
            || !whole
                .bytes()
                .chain(fraction.bytes())
                .all(|c| c.is_ascii_digit())
        {
            return Err(invalid());
        }
        let digits = format!("{whole}{fraction}");
        let coefficient: Integer = digits.parse().map_err(|_| invalid())?;
        Ok(Self {
            neg,
            coefficient,
            exponent: exponent - fraction.len() as i64,
        })
    }

    /// A decimal literal of the exact value (`d.ddd…E±x`), which Python's
    /// `Decimal` reads as this value.
    pub fn text(&self) -> String {
        if self.is_zero() {
            return if self.neg { "-0".into() } else { "0".into() };
        }
        let raw = self.coefficient.to_string();
        let trimmed = raw.trim_end_matches('0');
        let exponent = self.exponent + (raw.len() - trimmed.len()) as i64;
        let adjusted = exponent + trimmed.len() as i64 - 1;
        let sign = if self.neg { "-" } else { "" };
        let (head, tail) = trimmed.split_at(1);
        if tail.is_empty() {
            format!("{sign}{head}E{adjusted:+}")
        } else {
            format!("{sign}{head}.{tail}E{adjusted:+}")
        }
    }

    /// The nearest binary64 value (half to even, Python's `float(Decimal)`).
    pub fn to_f64(&self) -> f64 {
        let sign = if self.neg { "-" } else { "" };
        format!("{sign}{}e{}", self.coefficient, self.exponent)
            .parse()
            .expect("a decimal literal")
    }

    fn negated(&self) -> Self {
        Self {
            neg: !self.neg,
            ..self.clone()
        }
    }

    /// The signed coefficient at exponent `exponent <= self.exponent`.
    fn scaled(&self, exponent: i64) -> Integer {
        let value = &self.coefficient * &ten((self.exponent - exponent) as u64);
        if self.neg { -value } else { value }
    }

    fn compare(&self, other: &Self) -> Ordering {
        let exponent = self.exponent.min(other.exponent);
        self.scaled(exponent).cmp(&other.scaled(exponent))
    }
}

/// Python's `decimal` arithmetic at `digits` significant digits, rounding
/// half to even (a `localcontext(prec=digits)` of the default context).
#[derive(Clone, Debug)]
pub struct DecimalContext {
    pub digits: u32,
}

impl DecimalContext {
    pub fn new(digits: u32) -> Self {
        assert!(digits > 0, "a decimal context needs digits");
        Self { digits }
    }

    /// `(-1)^neg * (numerator / denominator) * 10^exponent` correctly rounded
    /// (`numerator >= 0`, `denominator > 0`).
    fn round(&self, neg: bool, numerator: Integer, denominator: Integer, exponent: i64) -> Decimal {
        if numerator.is_zero() {
            return Decimal {
                neg,
                coefficient: Integer::zero(),
                exponent: 0,
            };
        }
        let precision = i64::from(self.digits);
        if denominator.is_one() && digits(&numerator) <= precision {
            return Decimal {
                neg,
                coefficient: numerator,
                exponent,
            };
        }
        // Scale so that the integer quotient has more than `precision`
        // digits, then drop the excess digits with the remainder as sticky.
        let shift = (precision + 1 + digits(&denominator) - digits(&numerator)).max(0);
        let scaled = &numerator * &ten(shift as u64);
        let (quotient, remainder) = scaled.quot_rem(&denominator);
        let excess = digits(&quotient) - precision;
        debug_assert!(excess >= 1);
        let unit = ten(excess as u64);
        let (mut coefficient, tail) = quotient.quot_rem(&unit);
        let half = &unit / &Integer::from(2);
        let up = match tail.cmp(&half) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => {
                !remainder.is_zero() || {
                    let (_, parity) = coefficient.quot_rem(&Integer::from(2));
                    !parity.is_zero()
                }
            }
        };
        let mut exponent = exponent - shift + excess;
        if up {
            coefficient = coefficient + Integer::one();
            if coefficient == ten(self.digits as u64) {
                coefficient = ten(u64::from(self.digits) - 1);
                exponent += 1;
            }
        }
        Decimal {
            neg,
            coefficient,
            exponent,
        }
    }

    fn signed(&self, value: Integer, exponent: i64) -> Decimal {
        let neg = value.is_negative();
        self.round(neg, value.abs(), Integer::one(), exponent)
    }

    /// `+x`: `x` rounded to the context.
    pub fn plus(&self, x: &Decimal) -> Decimal {
        self.round(x.neg, x.coefficient.clone(), Integer::one(), x.exponent)
    }

    /// A decimal literal read and rounded (`+Decimal(text)`).
    pub fn read(&self, text: &str) -> Result<Decimal, String> {
        Decimal::parse(text).map(|x| self.plus(&x))
    }

    pub fn add(&self, a: &Decimal, b: &Decimal) -> Decimal {
        let exponent = a.exponent.min(b.exponent);
        let sum = a.scaled(exponent) + b.scaled(exponent);
        if sum.is_zero() {
            // Exact zero: negative only if both are (round half to even).
            return Decimal {
                neg: a.neg && b.neg,
                coefficient: Integer::zero(),
                exponent: 0,
            };
        }
        self.signed(sum, exponent)
    }

    pub fn sub(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.add(a, &b.negated())
    }

    pub fn mul(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.round(
            a.neg != b.neg,
            &a.coefficient * &b.coefficient,
            Integer::one(),
            a.exponent + b.exponent,
        )
    }

    pub fn div(&self, a: &Decimal, b: &Decimal) -> Decimal {
        assert!(!b.is_zero(), "decimal division by zero");
        self.round(
            a.neg != b.neg,
            a.coefficient.clone(),
            b.coefficient.clone(),
            a.exponent - b.exponent,
        )
    }

    /// pi at the context's digits (`assemble._decimal_pi`): Machin's
    /// formula at `digits + 5` digits, each series stopped at the first term
    /// that leaves its rounded sum unchanged (`_machin_pi`), then rounded.
    pub fn pi(&self) -> Decimal {
        let work = DecimalContext::new(self.digits + 5);
        let arctan_inverse = |x: i64| -> Decimal {
            let mut power = work.div(&Decimal::integer(1), &Decimal::integer(x));
            let mut total = power.clone();
            let x2 = Decimal::integer(x * x);
            let (mut n, mut sign) = (1_i64, 1_i64);
            loop {
                power = work.div(&power, &x2);
                n += 2;
                sign = -sign;
                let term = work.div(&power, &Decimal::integer(n));
                let following = work.add(&total, &work.mul(&Decimal::integer(sign), &term));
                if following == total {
                    return total;
                }
                total = following;
            }
        };
        let four = Decimal::integer(4);
        let inner = work.sub(&work.mul(&four, &arctan_inverse(5)), &arctan_inverse(239));
        self.plus(&work.mul(&four, &inner))
    }
}

/// The scalar arithmetic of an assembly: Python's binary64 floats
/// ([`Binary64`]) or its `Decimal` context ([`DecimalContext`]).
pub trait Arith {
    type N: Clone;
    fn zero(&self) -> Self::N;
    /// An exact small integer (Python mixes `int` into both carriers).
    fn int(&self, value: i64) -> Self::N;
    fn add(&self, a: &Self::N, b: &Self::N) -> Self::N;
    fn sub(&self, a: &Self::N, b: &Self::N) -> Self::N;
    fn mul(&self, a: &Self::N, b: &Self::N) -> Self::N;
    fn div(&self, a: &Self::N, b: &Self::N) -> Self::N;
    fn neg(&self, a: &Self::N) -> Self::N;
    /// The carrier's pi (`_Numbers.pi`).
    fn pi(&self) -> Self::N;
    /// `numbers.real(value)` of an integer (`float(n)`, `+Decimal(n)`).
    fn real_int(&self, value: i64) -> Self::N {
        self.int(value)
    }
    /// A complex value times a real one: Python's `complex * float` (the
    /// float as `complex(x, 0.0)`, one full complex product) or
    /// `reference_executor.Complex * Decimal` (each part times it).
    fn times_real(&self, z: &C<Self::N>, x: &Self::N) -> C<Self::N>;
}

/// A complex value of an assembly (Python's `complex` or
/// `reference_executor.Complex`).
#[derive(Clone, Debug)]
pub struct C<N> {
    pub re: N,
    pub im: N,
}

impl<N: Clone> C<N> {
    pub fn new(re: N, im: N) -> Self {
        Self { re, im }
    }
}

pub fn cadd<A: Arith>(arith: &A, a: &C<A::N>, b: &C<A::N>) -> C<A::N> {
    C::new(arith.add(&a.re, &b.re), arith.add(&a.im, &b.im))
}

/// `(a * c - b * d, a * d + b * c)`, as CPython's `_Py_c_prod` and
/// `reference_executor.Complex.__mul__`.
pub fn cmul<A: Arith>(arith: &A, x: &C<A::N>, y: &C<A::N>) -> C<A::N> {
    C::new(
        arith.sub(&arith.mul(&x.re, &y.re), &arith.mul(&x.im, &y.im)),
        arith.add(&arith.mul(&x.re, &y.im), &arith.mul(&x.im, &y.re)),
    )
}

pub fn conj<A: Arith>(arith: &A, z: &C<A::N>) -> C<A::N> {
    C::new(z.re.clone(), arith.neg(&z.im))
}

/// Python's binary64 floats.
#[derive(Clone, Copy, Debug, Default)]
pub struct Binary64;

impl Arith for Binary64 {
    type N = f64;
    fn zero(&self) -> f64 {
        0.0
    }
    fn int(&self, value: i64) -> f64 {
        value as f64
    }
    fn add(&self, a: &f64, b: &f64) -> f64 {
        a + b
    }
    fn sub(&self, a: &f64, b: &f64) -> f64 {
        a - b
    }
    fn mul(&self, a: &f64, b: &f64) -> f64 {
        a * b
    }
    fn div(&self, a: &f64, b: &f64) -> f64 {
        a / b
    }
    fn neg(&self, a: &f64) -> f64 {
        -a
    }
    fn pi(&self) -> f64 {
        std::f64::consts::PI
    }
    fn times_real(&self, z: &C<f64>, x: &f64) -> C<f64> {
        cmul(self, z, &C::new(*x, 0.0))
    }
}

/// Python's `Decimal` in a context, with its pi formed once.
#[derive(Clone, Debug)]
pub struct DecimalArith {
    pub context: DecimalContext,
    pi: Decimal,
}

impl DecimalArith {
    pub fn new(digits: u32) -> Self {
        let context = DecimalContext::new(digits);
        let pi = context.pi();
        Self { context, pi }
    }
}

impl Arith for DecimalArith {
    type N = Decimal;
    fn zero(&self) -> Decimal {
        Decimal::zero()
    }
    fn int(&self, value: i64) -> Decimal {
        Decimal::integer(value)
    }
    fn add(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.context.add(a, b)
    }
    fn sub(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.context.sub(a, b)
    }
    fn mul(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.context.mul(a, b)
    }
    fn div(&self, a: &Decimal, b: &Decimal) -> Decimal {
        self.context.div(a, b)
    }
    fn neg(&self, a: &Decimal) -> Decimal {
        // Python's unary minus rounds too (exact for context values).
        self.context.plus(&a.negated())
    }
    fn pi(&self) -> Decimal {
        self.pi.clone()
    }
    fn real_int(&self, value: i64) -> Decimal {
        self.context.plus(&Decimal::integer(value))
    }
    fn times_real(&self, z: &C<Decimal>, x: &Decimal) -> C<Decimal> {
        C::new(self.mul(&z.re, x), self.mul(&z.im, x))
    }
}

/// |A|² of loop-induced amplitude vectors per sector and Laurent
/// coefficient as the Python API's `loops/v2/induced.square` forms it in
/// double-double and arbitrary precision, operation for operation (the
/// canonical squaring order, DESIGN §7.10, of RULED Q41): per sector, per
/// slot pair `(l, l')` in order (weight 1 for `l == l'`, 2 otherwise), per
/// helicity row whose two vectors exist (`vectors[l * rows + row]`; `None`
/// is absent and skipped, as Python's `amplitudes.get`), per coefficient k
/// `Σ_{i+j=k} Σ_t conj(a_{t,i}) Σ_u G_{tu} b_{u,j}`, its real part times the
/// weight added to the sector's coefficient; each total divided by the
/// average. `metric` is `G` row-major (`size²` cells). Binary64 squares with
/// `rusticol_core::loop_v2::interference::square_amplitudes` (Python's
/// `induced.native_square`).
#[allow(clippy::too_many_arguments)]
pub fn square_amplitudes_python<A: Arith>(
    arith: &A,
    vectors: &[Option<Vec<Vec<C<A::N>>>>],
    rows: usize,
    size: usize,
    width: usize,
    metric: &[C<A::N>],
    pairs: &[Vec<(usize, usize)>],
    average: i64,
) -> Vec<Vec<A::N>> {
    let zero = C::new(arith.zero(), arith.zero());
    let real_zero = arith.real_int(0);
    let divisor = arith.int(average);
    let mut out = Vec::with_capacity(pairs.len());
    for sector in pairs {
        let mut totals = vec![real_zero.clone(); width];
        for &(left, right) in sector {
            let weight = arith.int(if left == right { 1 } else { 2 });
            for row in 0..rows {
                let (Some(a), Some(b)) = (&vectors[left * rows + row], &vectors[right * rows + row])
                else {
                    continue;
                };
                for (k, slot) in totals.iter_mut().enumerate() {
                    let mut total = zero.clone();
                    for i in 0..=k {
                        let j = k - i;
                        for (t, left_cell) in a.iter().enumerate().take(size) {
                            let mut inner = zero.clone();
                            for (u, right_cell) in b.iter().enumerate().take(size) {
                                inner = cadd(arith, &inner, &cmul(arith, &metric[t * size + u], &right_cell[j]));
                            }
                            total = cadd(arith, &total, &cmul(arith, &conj(arith, &left_cell[i]), &inner));
                        }
                    }
                    *slot = arith.add(slot, &arith.mul(&weight, &total.re));
                }
            }
        }
        out.push(totals.iter().map(|value| arith.div(value, &divisor)).collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> Decimal {
        Decimal::parse(text).unwrap()
    }

    #[test]
    fn rounding_is_half_even_at_the_context_digits() {
        let c = DecimalContext::new(3);
        assert_eq!(c.read("1.2345").unwrap().text(), "1.23E+0");
        assert_eq!(c.read("1.235").unwrap().text(), "1.24E+0");
        assert_eq!(c.read("1.245").unwrap().text(), "1.24E+0");
        assert_eq!(c.read("1.2451").unwrap().text(), "1.25E+0");
        assert_eq!(c.read("-999.6").unwrap().text(), "-1E+3");
        assert_eq!(c.read("0.0001234567").unwrap().text(), "1.23E-4");
        assert_eq!(c.add(&d("1"), &d("0.0005")).text(), "1E+0");
        assert_eq!(c.add(&d("1"), &d("0.00500001")).text(), "1.01E+0");
        assert_eq!(c.div(&d("1"), &d("3")).text(), "3.33E-1");
        assert_eq!(c.div(&d("2"), &d("3")).text(), "6.67E-1");
        assert_eq!(c.div(&d("-1"), &d("8")).text(), "-1.25E-1");
        assert_eq!(c.div(&d("1"), &d("16")).text(), "6.25E-2");
        assert_eq!(c.mul(&d("12.5"), &d("0.2")).text(), "2.5E+0");
        assert_eq!(c.sub(&d("1.23"), &d("1.23")).text(), "0");
        assert!(c.add(&d("1e-30"), &d("-1e-30")).is_zero());
        assert_eq!(d("1.50"), d("1.5"));
        assert_eq!(d("-0"), d("0"));
        assert_eq!(d("2.5e-3").to_f64(), 0.0025);
        assert_eq!(d("0.1").to_f64(), 0.1);
        assert!(Decimal::parse("inf").is_err() && Decimal::parse("1e").is_err());
    }

    #[test]
    fn pi_is_machins_at_every_context() {
        // pi to 60 digits; Machin's sum rounded at digits + 5 then to digits.
        let pi = "3.14159265358979323846264338327950288419716939937510582097494";
        for digits in [16_u32, 40, 52, 57] {
            let expected = DecimalContext::new(digits).read(pi).unwrap();
            assert_eq!(DecimalContext::new(digits).pi(), expected, "{digits}");
        }
    }
}
