// SPDX-License-Identifier: 0BSD

//! Immutable predicates and component access for the scalar frontend, version one.
//! Serialized programs retain calls, not callback implementations. These names
//! and their meaning must therefore stay fixed for the process/cache lifetime.

use symbolica::prelude::*;

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

pub(super) fn register() {
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
