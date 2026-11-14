// SPDX-License-Identifier: 0BSD

//! pyAmpliCol's version-one scalar primitives in binary64, without Symbolica.
//!
//! Saved evaluators call them by their Symbolica symbol; `source_scalar`
//! registers the same semantics with Symbolica for every carrier, and SymJIT
//! plane applications bind this binary64 form by name (`symjit_plane`).

/// The primitives: the Symbolica symbol and its mode in
/// [`scalar_primitive_f64`] (and in `source_scalar`'s carrier-generic form).
pub const SCALAR_PRIMITIVES: [(&str, u8); 6] = [
    ("pyamplicol_scalar_v1::truth", 0),
    ("pyamplicol_scalar_v1::equal", 1),
    ("pyamplicol_scalar_v1::less", 2),
    ("pyamplicol_scalar_v1::less_equal", 3),
    ("pyamplicol_scalar_v1::real_part", 4),
    ("pyamplicol_scalar_v1::imaginary_part", 5),
];

/// One primitive on (real, imaginary) pairs: truth, equality and the two
/// orderings return 1 or 0, the parts return their component; a wrong arity,
/// a non-finite argument of a predicate, or a non-real argument of an
/// ordering returns NaN.
pub fn scalar_primitive_f64(args: &[(f64, f64)], mode: u8) -> (f64, f64) {
    if args.len() == 1 && mode >= 4 {
        let (re, im) = args[0];
        return (if mode == 4 { re } else { im }, 0.0);
    }
    let arity = if mode == 0 || mode >= 4 { 1 } else { 2 };
    if args.len() != arity
        || args
            .iter()
            .any(|(re, im)| !re.is_finite() || !im.is_finite())
        || (mode >= 2 && args.iter().any(|&(_, im)| im != 0.0))
    {
        return (f64::NAN, 0.0);
    }
    let result = match mode {
        0 => args[0].0 != 0.0 || args[0].1 != 0.0,
        1 => args[0].0 == args[1].0 && args[0].1 == args[1].1,
        2 => args[0].0 < args[1].0,
        _ => args[0].0 <= args[1].0,
    };
    (if result { 1.0 } else { 0.0 }, 0.0)
}
