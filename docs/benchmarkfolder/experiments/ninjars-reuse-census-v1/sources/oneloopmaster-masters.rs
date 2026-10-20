//! Public scalar master Symbols and their numeric evaluation hooks.
//!
//! A call has the form `oneloopmaster::M(tag, arguments...)`, with Laurent
//! tags `0`, `-1`, and `-2`; A0 and B0 also accept `1`, their O(epsilon) coefficients
//! from [`a0_epsilon`] and [`b0_epsilon`]. Numeric arguments follow the corresponding
//! lowercase constructor, with the squared renormalization scale last.
//! Native definitions supplied by [`OneLoopExpressions`] take precedence
//! over the hooks and remain visible to Symbolica's evaluator compiler.

use super::*;
use std::{
    cell::Cell,
    collections::BTreeMap,
    ops::RangeInclusive,
    sync::{Mutex, OnceLock},
};
use symbolica::{
    atom::{AtomView, EvaluationInfo},
    domains::{
        float::{Complex, DoubleFloat, Float, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::ExpressionEvaluator,
};

/// Scalar integral family. Each evaluator returns finite, simple-pole and
/// double-pole coefficients, with `mu_squared` as its final numeric argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ScalarIntegral {
    A0,
    B0,
    DB0,
    C0,
    D0,
}

type Family = ScalarIntegral;

impl ScalarIntegral {
    /// Conventional master name.
    pub fn name(self) -> &'static str {
        ["A0", "B0", "dB0", "C0", "D0"][self as usize]
    }

    /// Number of numeric arguments, including the squared scale but not the
    /// Laurent tag used by the Symbol interface.
    pub fn arity(self) -> usize {
        [2, 4, 4, 7, 11][self as usize]
    }

    /// Laurent orders whose coefficient may be nonzero. Every master accepts
    /// the tags from -2 through the upper end; A0 and B0 reach O(epsilon).
    pub fn laurent_orders(self) -> RangeInclusive<i32> {
        match self {
            Self::A0 | Self::B0 => -1..=1,
            Self::DB0 => -1..=0,
            Self::C0 | Self::D0 => -2..=0,
        }
    }

    fn series(self, a: &[Atom]) -> LaurentSeries {
        match self {
            Self::A0 => a0(&a[0], &a[1]),
            Self::B0 => b0(&a[0], &a[1], &a[2], &a[3]),
            Self::DB0 => db0(&a[0], &a[1], &a[2], &a[3]),
            Self::C0 => {
                expressions::triangle_series([&a[0], &a[1], &a[2]], [&a[3], &a[4], &a[5]], &a[6])
            }
            Self::D0 => expressions::box_series(
                [&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]],
                [&a[6], &a[7], &a[8], &a[9]],
                &a[10],
            ),
        }
    }

    fn parameters(self) -> Vec<Symbol> {
        (0..self.arity())
            .map(|index| symbolica::symbol!(format!("__olo_master_{}_{}", self.name(), index)))
            .collect()
    }

    fn check_arity(self, count: usize) {
        assert_eq!(
            count,
            self.arity(),
            "oneloopmaster::{} expects {} numeric arguments after its Laurent tag; received {}",
            self.name(),
            self.arity(),
            count
        );
    }
}

/// Coefficient index of the positive O(epsilon) tag, after the finite, simple-pole
/// and double-pole indices 0, 1 and 2.
const EPSILON_INDEX: usize = 3;

fn tag_index(family: Family, tag: i32) -> Option<usize> {
    match tag {
        1 if matches!(family, Family::A0 | Family::B0) => Some(EPSILON_INDEX),
        -2..=0 => Some(tag.unsigned_abs() as usize),
        _ => None,
    }
}

fn coefficient_index(family: Family, tags: &[AtomView<'_>]) -> usize {
    assert_eq!(
        tags.len(),
        1,
        "oneloopmaster::{} requires one Laurent tag",
        family.name()
    );
    i32::try_from(tags[0])
        .ok()
        .and_then(|tag| tag_index(family, tag))
        .unwrap_or_else(|| {
            let expected = if matches!(family, Family::A0 | Family::B0) {
                "1, 0, -1, or -2"
            } else {
                "0, -1, or -2"
            };
            panic!(
                "invalid oneloopmaster::{} Laurent tag {}; expected {expected}",
                family.name(),
                tags[0]
            )
        })
}

/// A0's O(epsilon) coefficient from its finite part `f = m²(1 - L)`: as
/// `L = (m² - f)/m²`, [`a0_epsilon`] equals `f + m²(L²/2 + π²/6)`. Taking
/// `L` first keeps binary64 finite wherever the finite part is.
fn a0_epsilon_from_finite<T: Real + SingleFloat>(mass_squared: &T, finite: &T) -> T {
    if mass_squared.is_zero() {
        return mass_squared.zero();
    }
    let logarithm = (mass_squared.clone() - finite) / mass_squared;
    let pi = mass_squared.pi();
    let series = logarithm.clone() * &logarithm / mass_squared.from_i64(2)
        + pi.clone() * &pi / mass_squared.from_i64(6);
    finite.clone() + mass_squared.clone() * &series
}

/// Select a coefficient of `[finite, simple pole, double pole]`, or A0's
/// O(epsilon) coefficient derived from its finite part.
fn select<T: Real + SingleFloat>(tag: usize, args: &[T], output: [T; 3]) -> T {
    let [finite, simple_pole, double_pole] = output;
    match tag {
        0 => finite,
        1 => simple_pole,
        2 => double_pole,
        EPSILON_INDEX => a0_epsilon_from_finite(&args[0], &finite),
        _ => unreachable!("tag_index only returns coefficient indices"),
    }
}

// A hook only evaluates native definitions, which never call a public master.
// Detect accidental recursion before acquiring any cache or OnceLock: this
// turns a future dependency cycle into a useful error rather than a deadlock.
thread_local! { static IN_MASTER: Cell<bool> = const { Cell::new(false) }; }
struct EvaluationGuard;
impl EvaluationGuard {
    fn enter() -> Self {
        IN_MASTER.with(|active| {
            assert!(!active.get(), "reentrant OneLOop master evaluation");
            active.set(true);
        });
        Self
    }
}
impl Drop for EvaluationGuard {
    fn drop(&mut self) {
        IN_MASTER.with(|active| active.set(false));
    }
}

type ExactEvaluator = ExpressionEvaluator<Complex<Rational>>;
#[cfg(feature = "generated-evaluators")]
type PrecisionEvaluator = crate::NativeEvaluator<Float>;
#[cfg(not(feature = "generated-evaluators"))]
type PrecisionEvaluator = crate::PrecisionEvaluator;

pub(crate) fn exact_evaluator(family: ScalarIntegral) -> &'static ExactEvaluator {
    initialization::ensure_symbolica_state();
    static CACHE: [OnceLock<ExactEvaluator>; 5] = [const { OnceLock::new() }; 5];
    CACHE[family as usize].get_or_init(|| {
        let args = family
            .parameters()
            .into_iter()
            .map(Symbol::to_atom)
            .collect::<Vec<_>>();
        OneLoopExpressions::new()
            .evaluator(family.series(&args).coefficients(), &args)
            .expect("compile native OneLOop master definitions")
    })
}

pub(crate) fn jit_evaluator(family: ScalarIntegral) -> Result<JitEvaluator, String> {
    let args = family
        .parameters()
        .into_iter()
        .map(Symbol::to_atom)
        .collect::<Vec<_>>();
    OneLoopExpressions::new().jit_evaluator(family.series(&args).coefficients(), &args)
}

fn b0_epsilon_evaluator() -> &'static ExactEvaluator {
    static CACHE: OnceLock<ExactEvaluator> = OnceLock::new();
    CACHE.get_or_init(|| {
        let args = Family::B0
            .parameters()
            .into_iter()
            .map(Symbol::to_atom)
            .collect::<Vec<_>>();
        OneLoopExpressions::new()
            .evaluator(&[b0_epsilon(&args[0], &args[1], &args[2], &args[3])], &args)
            .expect("compile B0 positive-epsilon coefficient")
    })
}

fn scalar_f64(family: Family, tag: usize, args: &[Complex<f64>]) -> Complex<f64> {
    family.check_arity(args.len());
    let _guard = EvaluationGuard::enter();
    let coefficient = |tag| {
        #[cfg(feature = "generated-evaluators")]
        {
            crate::backend::native_coefficient(family, tag, args)
        }
        #[cfg(not(feature = "generated-evaluators"))]
        {
            crate::backend::expression_coefficient(family, tag, args)
        }
    };
    if tag == EPSILON_INDEX && family == Family::B0 {
        crate::native::validate_point(family, args).expect("valid master arguments");
        static CACHE: OnceLock<Mutex<ExpressionEvaluator<Complex<f64>>>> = OnceLock::new();
        let mut evaluator = CACHE
            .get_or_init(|| {
                Mutex::new(
                    b0_epsilon_evaluator()
                        .clone()
                        .map_coeff(&|z| Complex::new(z.re.to_f64(), z.im.to_f64())),
                )
            })
            .lock()
            .unwrap();
        let mut output = [Complex::new(0., 0.)];
        evaluator.evaluate(args, &mut output);
        return output[0];
    }
    if tag == EPSILON_INDEX {
        return a0_epsilon_from_finite(&args[0], &coefficient(0));
    }
    coefficient(tag)
}

fn scalar_double_float(
    family: Family,
    tag: usize,
    args: &[Complex<DoubleFloat>],
) -> Complex<DoubleFloat> {
    family.check_arity(args.len());
    let _guard = EvaluationGuard::enter();
    if tag == EPSILON_INDEX && family == Family::B0 {
        crate::native::validate_point(family, args).expect("valid master arguments");
        let prototype = DoubleFloat::from(0.0);
        let mut evaluator = b0_epsilon_evaluator().clone().map_coeff(&|z| {
            Complex::new(
                prototype.from_rational(&z.re),
                prototype.from_rational(&z.im),
            )
        });
        let mut output = [Complex::new(prototype.clone(), prototype)];
        evaluator.evaluate(args, &mut output);
        return output[0];
    }
    // Validate before taking the lock: rejected input must not poison the cache.
    #[cfg(not(feature = "generated-evaluators"))]
    crate::native::validate_point(family, args).expect("valid master arguments");
    #[cfg(feature = "generated-evaluators")]
    type DoubleEvaluator = crate::NativeEvaluator<DoubleFloat>;
    #[cfg(not(feature = "generated-evaluators"))]
    type DoubleEvaluator = ExpressionEvaluator<Complex<DoubleFloat>>;
    static CACHE: [OnceLock<Mutex<DoubleEvaluator>>; 5] = [const { OnceLock::new() }; 5];
    let mut evaluator = CACHE[family as usize]
        .get_or_init(|| {
            #[cfg(feature = "generated-evaluators")]
            let evaluator =
                crate::NativeEvaluator::new(family).expect("native DoubleFloat constants");
            #[cfg(not(feature = "generated-evaluators"))]
            let evaluator = {
                let prototype = DoubleFloat::from(0.0);
                exact_evaluator(family).clone().map_coeff(&|z| {
                    Complex::new(
                        prototype.from_rational(&z.re),
                        prototype.from_rational(&z.im),
                    )
                })
            };
            Mutex::new(evaluator)
        })
        .lock()
        .expect("OneLOop DoubleFloat cache poisoned");
    let mut output = core::array::from_fn::<_, 3, _>(|_| Complex::new(0.0.into(), 0.0.into()));
    #[cfg(feature = "generated-evaluators")]
    let result = evaluator.evaluate(args, &mut output);
    #[cfg(not(feature = "generated-evaluators"))]
    let result: Result<(), String> = {
        evaluator.evaluate(args, &mut output);
        Ok(())
    };
    drop(evaluator);
    // Domain rejection must not poison a reusable shared evaluator when the
    // caller catches the numeric callback's deliberate panic.
    result.expect("valid master arguments");
    select(tag, args, output)
}

fn scalar_float(family: Family, tag: usize, args: &[Complex<Float>]) -> Complex<Float> {
    family.check_arity(args.len());
    let _guard = EvaluationGuard::enter();
    // The requested precision reaches an external function in its arguments;
    // include every real and imaginary component, including the scale.
    let precision = args
        .iter()
        .flat_map(|value| [value.re.prec(), value.im.prec()])
        .max()
        .expect("every master has numeric arguments");
    if tag == EPSILON_INDEX && family == Family::B0 {
        crate::native::validate_point(family, args).expect("valid master arguments");
        let prototype = Float::new(precision);
        let mut evaluator = b0_epsilon_evaluator().clone().map_coeff_with_prec(
            &|z| {
                Complex::new(
                    prototype.from_rational(&z.re),
                    prototype.from_rational(&z.im),
                )
            },
            precision,
        );
        let mut output = [Complex::new(prototype.clone(), prototype)];
        evaluator.evaluate(args, &mut output);
        return output[0].clone();
    }
    static CACHE: [OnceLock<Mutex<BTreeMap<u32, PrecisionEvaluator>>>; 5] =
        [const { OnceLock::new() }; 5];
    let mut cache = CACHE[family as usize]
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("OneLOop precision evaluator cache poisoned");
    let evaluator = cache.entry(precision).or_insert_with(|| {
        PrecisionEvaluator::with_binary_precision(family, precision)
            .expect("native arbitrary-precision constants")
    });
    let mut output = core::array::from_fn::<_, 3, _>(|_| {
        Complex::new(Float::new(precision), Float::new(precision))
    });
    let result = evaluator.evaluate(args, &mut output);
    drop(cache);
    result.expect("valid master arguments");
    select(tag, args, output)
}

/// Fold only explicitly approximate, complete coefficient calls. In particular,
/// an untagged inspection call or an exact rational call remains symbolic.
fn normalized_numeric_call(family: Family, view: AtomView<'_>) -> Option<Complex<Float>> {
    crate::record_usage();
    let AtomView::Fun(function) = view else {
        return None;
    };
    if function.get_nargs() != family.arity() + 1 {
        return None;
    }
    let mut arguments = function.iter();
    let tag = tag_index(family, i32::try_from(arguments.next()?).ok()?)?;
    let arguments = arguments.collect::<Vec<_>>();
    // Exact numbers acquire precision only when another argument explicitly
    // supplies it. Include imaginary components and the renormalization scale.
    let precision = arguments
        .iter()
        .filter_map(|argument| Complex::<Float>::try_from(*argument).ok())
        .flat_map(|value| [value.re.prec(), value.im.prec()])
        .max()?;
    let arguments = arguments
        .into_iter()
        .map(|argument| {
            let AtomView::Num(number) = argument else {
                return None;
            };
            number.get_coeff_view().to_float(precision).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    // Construction must remain non-throwing for points outside the numerical
    // contract. Explicit evaluation retains the existing descriptive errors.
    crate::native::validate_point(family, &arguments).ok()?;
    let value = scalar_float(family, tag, &arguments);
    (value.re.is_finite() && value.im.is_finite()).then_some(value)
}

fn info(family: Family) -> EvaluationInfo {
    // Symbolica routes a call containing only its tag through the constant
    // interface. Such a call is malformed for every master; report the same
    // arity contract there as in the numeric callbacks.
    EvaluationInfo::constant(move |tags, _precision| {
        coefficient_index(family, tags);
        Err(format!(
            "oneloopmaster::{} expects {} numeric arguments after its Laurent tag; received 0",
            family.name(),
            family.arity()
        ))
    })
    .with_tags(1)
    .register_tagged::<Complex<f64>>(move |tags| {
        let tag = coefficient_index(family, tags);
        Box::new(move |args: &[Complex<f64>]| scalar_f64(family, tag, args))
    })
    .register_tagged::<Complex<Float>>(move |tags| {
        let tag = coefficient_index(family, tags);
        Box::new(move |args: &[Complex<Float>]| scalar_float(family, tag, args))
    })
    .register_tagged::<Complex<DoubleFloat>>(move |tags| {
        let tag = coefficient_index(family, tags);
        Box::new(move |args: &[Complex<DoubleFloat>]| scalar_double_float(family, tag, args))
    })
}

/// Initializes and returns `oneloopmaster::A0(tag, m², mu²)`.
///
/// All master accessors attach numeric callbacks for `Complex<f64>`,
/// `Complex<DoubleFloat>` and `Complex<Float>`. Invalid tags, domains or numeric arities panic with a descriptive
/// message because Symbolica's numeric callback interface returns a number,
/// not a `Result`. The native map performs its own function validation.
/// Complete tagged calls containing a floating-point argument normalize via
/// the native hook at the supplied precision. Untagged, exact, incomplete, and
/// invalid-domain calls remain symbolic until explicitly evaluated.
#[allow(non_snake_case)]
pub fn A0() -> Symbol {
    initialization::ensure_symbolica_state();
    static SYMBOL: OnceLock<Symbol> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        symbolica::symbol!(
            "oneloopmaster::A0",
            eval = info(Family::A0),
            norm = |view, out| {
                if let Some(value) = normalized_numeric_call(Family::A0, view) {
                    out.to_num(value);
                }
            }
        )
    })
}

/// Initializes and returns `oneloopmaster::B0(tag, p², m0², m1², mu²)`.
#[allow(non_snake_case)]
pub fn B0() -> Symbol {
    initialization::ensure_symbolica_state();
    static SYMBOL: OnceLock<Symbol> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        symbolica::symbol!(
            "oneloopmaster::B0",
            eval = info(Family::B0),
            norm = |view, out| {
                if let Some(value) = normalized_numeric_call(Family::B0, view) {
                    out.to_num(value);
                }
            }
        )
    })
}

/// Initializes and returns `oneloopmaster::dB0(tag, p², m0², m1², mu²)`.
#[allow(non_snake_case)]
pub fn dB0() -> Symbol {
    initialization::ensure_symbolica_state();
    static SYMBOL: OnceLock<Symbol> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        symbolica::symbol!(
            "oneloopmaster::dB0",
            eval = info(Family::DB0),
            norm = |view, out| {
                if let Some(value) = normalized_numeric_call(Family::DB0, view) {
                    out.to_num(value);
                }
            }
        )
    })
}

/// Initializes and returns `oneloopmaster::C0(tag, p1², p2², p3², m1², m2², m3², mu²)`.
#[allow(non_snake_case)]
pub fn C0() -> Symbol {
    initialization::ensure_symbolica_state();
    static SYMBOL: OnceLock<Symbol> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        symbolica::symbol!(
            "oneloopmaster::C0",
            eval = info(Family::C0),
            norm = |view, out| {
                if let Some(value) = normalized_numeric_call(Family::C0, view) {
                    out.to_num(value);
                }
            }
        )
    })
}

/// Initializes and returns the box master, with the arguments of [`crate::d0`]
/// flattened as six momenta, four squared masses, and the squared scale.
#[allow(non_snake_case)]
pub fn D0() -> Symbol {
    initialization::ensure_symbolica_state();
    static SYMBOL: OnceLock<Symbol> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        symbolica::symbol!(
            "oneloopmaster::D0",
            eval = info(Family::D0),
            norm = |view, out| {
                if let Some(value) = normalized_numeric_call(Family::D0, view) {
                    out.to_num(value);
                }
            }
        )
    })
}

/// Adds the seventeen tagged native master definitions, including A0/B0's
/// O(epsilon) coefficient, to a complete native map.
/// Pure sector-call constructors keep registration independent of map creation.
pub(crate) fn register_masters(map: &mut FunctionMap) {
    for (family, master) in [
        (Family::A0, A0()),
        (Family::B0, B0()),
        (Family::DB0, dB0()),
        (Family::C0, C0()),
        (Family::D0, D0()),
    ] {
        let vars = family.parameters();
        let args = vars
            .iter()
            .copied()
            .map(Symbol::to_atom)
            .collect::<Vec<_>>();
        let epsilon = match family {
            Family::A0 => Some((1, a0_epsilon(&args[0], &args[1]))),
            Family::B0 => Some((1, b0_epsilon(&args[0], &args[1], &args[2], &args[3]))),
            _ => None,
        };
        for (tag, body) in [0, -1, -2]
            .into_iter()
            .zip(family.series(&args).into_coefficients())
            .chain(epsilon)
        {
            map.add_tagged_function_with_options(
                master,
                vec![Atom::num(tag)],
                vars.clone(),
                body,
                native_function_options(),
            )
            .expect("unique native OneLOop master definition");
        }
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::*;
    use symbolica::prelude::RealLike;

    fn approximate(value: i64, precision: u32) -> Atom {
        Atom::num(Float::with_val(precision, value))
    }

    fn assert_symbolic(expression: Atom, symbol: Symbol) {
        assert!(
            matches!(expression.as_view(), AtomView::Fun(function) if function.get_symbol() == symbol)
        );
    }

    #[test]
    fn mixed_floating_calls_fold_for_every_master() {
        // The massless simple sectors provide independent pole/derivative
        // values while exercising all five registered normalization callbacks.
        let cases = [
            (A0(), -1, vec![2, 1], 2.0),
            (B0(), -1, vec![-2, 0, 0, 1], 1.0),
            (dB0(), 0, vec![-2, 0, 0, 1], 0.5),
            (C0(), -2, vec![0, 0, -2, 0, 0, 0, 1], -0.5),
            (D0(), -2, vec![0, 0, 0, 0, -2, -3, 0, 0, 0, 0, 1], 2.0 / 3.0),
        ];
        for (symbol, tag, values, expected) in cases {
            let mut arguments = vec![Atom::num(tag)];
            arguments.extend(values.into_iter().map(Atom::num));
            *arguments.last_mut().unwrap() = approximate(1, 160);
            let result = symbol.call(&arguments);
            let value = Complex::<Float>::try_from(result.as_view()).unwrap();
            assert!(
                (value.re.to_f64() - expected).abs() < 1e-14,
                "{}: {value}",
                symbol.get_name()
            );
            assert_eq!(value.im.to_f64(), 0.0);
        }
    }

    #[test]
    fn finite_normalization_retains_supplied_precision() {
        let result = A0().call((0, approximate(2, 200), 1));
        let value = Complex::<Float>::try_from(result.as_view()).unwrap();
        assert!((value.re.to_f64() - 2.0 * (1.0 - 2.0_f64.ln())).abs() < 1e-14);
        assert!(value.re.prec() > 150);
        // Even when the only inexact input is the scale, exact complex masses
        // are converted at that precision, rather than through binary64.
        let mass = Atom::num(Complex::new(
            Rational::from((2, 3)),
            Rational::from((-1, 7)),
        ));
        let result = A0().call((-1, mass, approximate(1, 240)));
        let value = Complex::<Float>::try_from(result.as_view()).unwrap();
        assert!(value.re.prec() > 200 && value.im.prec() > 200);
        assert!((value.re.to_f64() - 2.0 / 3.0).abs() < 1e-15);
        assert!((value.im.to_f64() + 1.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn exact_untagged_symbolic_and_invalid_calls_are_preserved() {
        let x = Atom::var(symbolica::symbol!("normalization_test::x"));
        let float = approximate(2, 100);
        for call in [
            A0().call((0, 2, 1)),
            A0().call((&float, 1)),
            A0().call((0, &float, &x)),
            A0().call((3, &float, 1)),
            A0().call((0, &float, 1, 2)),
            A0().call((0, &float, 0)),
            A0().call((0, &float, -1)),
            A0().call((
                0,
                Atom::num(Complex::new(Rational::from(2), Rational::from(1))),
                &float,
            )),
        ] {
            assert_symbolic(call, A0());
        }
        assert_symbolic(
            B0().call((
                0,
                Atom::num(Complex::new(Rational::from(-2), Rational::from(1))),
                0,
                0,
                float,
            )),
            B0(),
        );
    }
}
