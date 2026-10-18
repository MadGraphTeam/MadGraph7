// SPDX-License-Identifier: 0BSD

//! Shared exact decimal ingress and immutable version-one source scalar callbacks.
//! Callback names and semantics are retained in saved evaluators.

use symbolica::domains::{integer::Integer, rational::Rational};
use symbolica::prelude::*;

fn error(value: impl std::fmt::Display) -> String {
    value.to_string()
}

pub fn decimal_rational(value: &str) -> Result<Rational, String> {
    // Lexical decimal conversion only. Integer powers, products and rational
    // normalization all use Numerica, without a floating intermediate.
    let (negative, value) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else {
        (false, value.strip_prefix('+').unwrap_or(value))
    };
    let mut exponent_parts = value.split(['e', 'E']);
    let mantissa = exponent_parts.next().unwrap_or_default();
    let exponent = exponent_parts
        .next()
        .map(|e| e.parse::<i64>().map_err(error))
        .transpose()?
        .unwrap_or(0);
    if exponent_parts.next().is_some() {
        return Err(error("expected a finite decimal literal"));
    }
    let mut decimal_parts = mantissa.split('.');
    let whole = decimal_parts.next().unwrap_or_default();
    let fractional = decimal_parts.next().unwrap_or_default();
    if decimal_parts.next().is_some()
        || (whole.is_empty() && fractional.is_empty())
        || !whole
            .bytes()
            .chain(fractional.bytes())
            .all(|c| c.is_ascii_digit())
    {
        return Err(error("expected a finite decimal literal"));
    }
    let digits = format!("{whole}{fractional}");
    let mut numerator = digits.parse::<Integer>().map_err(error)?;
    if negative {
        numerator = -numerator;
    }
    let decimal_places = i64::try_from(fractional.len()).map_err(error)?;
    let exponent = exponent
        .checked_sub(decimal_places)
        .ok_or_else(|| error("decimal exponent overflows"))?;
    let scale = Integer::from(10).pow(exponent.unsigned_abs());
    Ok(if exponent < 0 {
        Rational::from((numerator, scale))
    } else {
        Rational::from(numerator * scale)
    })
}

fn predicate<T: Real + SingleFloat + PartialOrd>(args: &[Complex<T>], mode: u8) -> Complex<T> {
    let zero = args.first().map_or_else(T::new_zero, |x| x.re.zero());
    if args.len() == 1 && mode >= 4 {
        let component = if mode == 4 {
            args[0].re.clone()
        } else {
            args[0].im.clone()
        };
        return Complex::new(component, zero);
    }
    let arity = if mode == 0 || mode >= 4 { 1 } else { 2 };
    if args.len() != arity
        || args.iter().any(|x| !x.is_finite())
        || (mode >= 2 && args.iter().any(|x| !x.im.is_fully_zero()))
    {
        return Complex::new(zero.nan().expect("supported carriers have NaN"), zero);
    }
    let result = match mode {
        0 => !args[0].is_fully_zero(),
        1 => args[0] == args[1],
        2 => args[0].re < args[1].re,
        3 => args[0].re <= args[1].re,
        _ => unreachable!(),
    };
    Complex::new(if result { zero.one() } else { zero.clone() }, zero)
}

pub fn register_scalar_primitives() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(register_definitions);
}

fn register_definitions() {
    macro_rules! install {
        ($name:literal, $mode:literal) => {
            let _ = symbol!(
                $name,
                eval = EvaluationInfo::new()
                    .register(|x: &[Complex<f64>]| predicate(x, $mode))
                    .register(|x: &[Complex<DoubleFloat>]| predicate(x, $mode))
                    .register(|x: &[Complex<Float>]| predicate(x, $mode))
            );
        };
    }
    install!("pyamplicol_scalar_v1::truth", 0);
    install!("pyamplicol_scalar_v1::equal", 1);
    install!("pyamplicol_scalar_v1::less", 2);
    install!("pyamplicol_scalar_v1::less_equal", 3);
    install!("pyamplicol_scalar_v1::real_part", 4);
    install!("pyamplicol_scalar_v1::imaginary_part", 5);
}
