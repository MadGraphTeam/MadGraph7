//! Exact Symbolica expressions for one-loop integrals.
//!
//! All masses and renormalization scales passed to this crate are squared
//! quantities. Laurent coefficients are stored in the OneLOop order
//! `[epsilon^0, epsilon^-1, epsilon^-2]`.
#![forbid(unsafe_code)]

mod backend;
mod branch_selection;
mod definitions;
mod epsilon;
mod evaluators;
pub use epsilon::b0_epsilon;
mod expressions;
mod initialization;
mod inspection;
mod inspection_formulas;
mod labelling;
pub use labelling::regular_d0_labelling;
pub use native::double_double;
mod native;
mod precision;
mod poles;
#[cfg(feature = "rust-codegen")]
pub mod rust_codegen;
mod sheet_exact;
mod triangle_hv;
pub use backend::{
    DEFAULT_BACKEND, EvaluationBackend, evaluate, evaluate_batch, evaluate_batch_with_backend,
    evaluate_with_backend,
};
pub use branch_selection::{select_branch, select_branch_with};
pub use evaluators::{JitEvaluator, ScalarEvaluator, jit_settings, rebuild_cached_evaluator};
pub use expressions::OneLoopExpressions;
pub use initialization::{initialize, is_initialized};
pub use inspection::{
    ExpressionOptions, SharedLaurentSeries, get_expression, get_expression_for_family,
    get_expression_for_family_with_options, get_expression_on_branch,
    get_expression_on_branch_with_options, get_expression_shared,
    get_expression_shared_with_options, get_expression_with_options, master_arguments,
    select_branch_shared,
};
pub use native::{NativeEvaluator, NativeFloat, NativeTensorEvaluator, TensorIntegral, b0_momentum_shift};
#[doc(hidden)]
pub use native::primitives::{double_dilog_via_float, symbolica_dilog_float};
pub use precision::PrecisionEvaluator;
pub use poles::{PoleCoefficient, PoleSelection};

symbolica::initialize!(
    initialization::from_symbolica,
    "symbolica::special_functions"
);
mod box_complex;
mod box_integral;
mod cayley;
mod masters;
mod triangle;
mod two_point;

pub use box_integral::*;
use masters::register_masters;
pub use masters::{A0, B0, C0, D0, ScalarIntegral, dB0};
pub use triangle::*;
pub use two_point::*;

use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    coefficient::Coefficient,
};

use definitions::FunctionMap;

use sheet_exact::{SheetAtom, divided_difference as sheet_dilog_divided_difference};
use sheet_exact::{SheetAtom as ExactSheetAtom, divided_difference as exact_sheet_difference};

fn native_function_options() -> symbolica::evaluate::FunctionRegistrationOptions {
    symbolica::evaluate::FunctionRegistrationOptions::new()
        .inlining(symbolica::evaluate::InliningPolicy::Never)
}

/// The finite, simple-pole, and double-pole coefficients of an integral.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaurentSeries {
    coefficients: [Atom; 3],
}

/// A compact Laurent series plus the transparent native definitions needed to evaluate it.
#[derive(Clone, Debug)]
pub struct MappedLaurentSeries {
    series: LaurentSeries,
    function_map: &'static FunctionMap,
}

impl MappedLaurentSeries {
    /// Returns the compact coefficients in `[epsilon^0, epsilon^-1, epsilon^-2]` order.
    pub fn coefficients(&self) -> &[Atom; 3] {
        self.series.coefficients()
    }

    /// Returns the Symbolica function definitions referenced by the coefficients.
    pub fn function_map(&self) -> &symbolica::evaluate::FunctionMap {
        self.function_map.as_symbolica()
    }

    /// Consumes the result into its compact series and transparent function map.
    pub fn into_parts(self) -> (LaurentSeries, symbolica::evaluate::FunctionMap) {
        (self.series, self.function_map.as_symbolica().clone())
    }
}

impl LaurentSeries {
    fn new(finite: Atom, simple_pole: Atom, double_pole: Atom) -> Self {
        Self {
            coefficients: [finite, simple_pole, double_pole],
        }
    }

    /// Returns the coefficients in the order `[epsilon^0, epsilon^-1, epsilon^-2]`.
    pub fn coefficients(&self) -> &[Atom; 3] {
        &self.coefficients
    }

    /// Consumes the series and returns its coefficients.
    pub fn into_coefficients(self) -> [Atom; 3] {
        self.coefficients
    }
}

/// Error returned for tadpole tensor ranks above four.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedRank(pub usize);

fn if_nonzero(condition: &Atom, value: Atom) -> Atom {
    Symbol::IF.call((condition, value, 0))
}

fn if_nonzero_else(condition: &Atom, nonzero: Atom, zero: Atom) -> Atom {
    Symbol::IF.call((condition, nonzero, zero))
}

// Test exact zeros without squaring small values or expanding complex norms.
fn any_nonzero(conditions: impl IntoIterator<Item = Atom>) -> Atom {
    conditions
        .into_iter()
        .fold(Atom::num(0), |predicate, condition| {
            if_nonzero_else(&condition, Atom::num(1), predicate)
        })
}

/// Preserve an arithmetic subexpression when forming exact axis predicates.
///
/// Native conjugation distributes over sums and products during normalization.
/// Comparing that expanded result with a separately rounded value can turn an
/// exact real-axis test into a spurious nonzero. The native identity if(z,z,0)
/// keeps both uses grouped, without a callback or a numerical tolerance.
fn grouped(value: &Atom) -> Atom {
    Symbol::IF.call((value, value, 0))
}

fn off_real_axis(value: &Atom) -> Atom {
    let value = grouped(value);
    &value - value.conj()
}

/// Principal logarithm continued to the lower lip of the negative real axis.
///
/// This is the Feynman branch selected by squared masses with non-positive
/// imaginary parts. It is built entirely from Symbolica's native operations.
fn physical_log(value: &Atom) -> Atom {
    // Complex conjugation flips IEEE signed zero on the real axis, so the
    // nested-conjugation identity alone cannot prescribe its branch lip. Select
    // the lower lip explicitly for real arguments and retain the principal
    // logarithm away from the axis.  The IF is native and lazy, so the
    // inactive branch may contain singular expressions without affecting
    // evaluation of the active coefficient.
    let off_axis = off_real_axis(value);
    let value = grouped(value);
    let absolute = Symbol::ABS.call((&value,));
    let negative = Symbol::IF.call((&value - &absolute, Atom::num(1), Atom::num(0)));
    // ABS selects the lip, but cannot represent the analytic neighborhood:
    // its derivative discards imaginary seeds at a real primal value.
    let axis_value = (Atom::num(1) - Atom::num(2) * &negative) * &value;
    let lower_lip = axis_value.log() - Atom::num(-1).sqrt() * Symbol::PI.to_atom() * negative;
    Symbol::IF.call((&off_axis, value.log(), lower_lip))
}

/// Principal logarithm with an explicitly upper real-axis lip. IEEE signed
/// zero is not a reliable substitute for the analytic root's prescribed lip.
fn upper_log(value: &Atom) -> Atom {
    let off_axis = off_real_axis(value);
    let value = grouped(value);
    let absolute = Symbol::ABS.call((&value,));
    let negative = if_nonzero_else(&(&value - &absolute), Atom::num(1), Atom::num(0));
    let axis_value = (Atom::num(1) - Atom::num(2) * &negative) * &value;
    let upper_lip = axis_value.log() + sheet_exact::i() * Symbol::PI.to_atom() * negative;
    if_nonzero_else(&off_axis, value.log(), upper_lip)
}

fn polylog_symbol() -> Symbol {
    // Symbolica has registered this before running our initialization callback.
    // Looking it up directly also works when its public special-symbol cache is
    // what originally triggered global initialization.
    symbolica::get_symbol!("symbolica::polylog").expect("polylog is registered")
}

fn dilog_atom(value: impl AtomCore) -> Atom {
    polylog_symbol().call((2, value.as_atom_view()))
}

fn physical_sqrt(value: &Atom) -> Atom {
    let off_axis = off_real_axis(value);
    let value = grouped(value);
    let absolute = Symbol::ABS.call((&value,));
    let real_axis = if_nonzero_else(
        &(&value - &absolute),
        -sheet_exact::i() * (-&value).sqrt(),
        value.sqrt(),
    );
    if_nonzero_else(&off_axis, value.sqrt(), real_axis)
}

fn log_over_one_minus(value: &Atom) -> Atom {
    let difference: Atom = 1 - value;
    // Rust evaluates the arguments to IF eagerly. Resolve an exact removable
    // singularity before constructing the otherwise discarded 0/0 expression.
    if difference.is_zero() {
        return Atom::num(-1);
    }
    if_nonzero_else(
        &difference,
        physical_log(value) / &difference,
        Atom::num(-1),
    )
}

fn x_log_ratio(value: &Atom, mu_squared: &Atom) -> Atom {
    if value.is_zero() {
        Atom::num(0)
    } else {
        if_nonzero(value, value * physical_log(&(value / mu_squared)))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod analytic_continuation_tests;

#[cfg(test)]
mod removable_log_tests;

#[cfg(test)]
mod quotient_tests;

#[cfg(test)]
mod coincident_dilog_tests;

static CITATIONS_USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
pub fn record_usage() {
    use std::sync::atomic::Ordering;
    if !CITATIONS_USED.load(Ordering::Relaxed) {
        CITATIONS_USED.store(true, Ordering::Relaxed);
    }
}

/// Whether this package has performed an operation in this process.
pub fn was_used() -> bool {
    CITATIONS_USED.load(std::sync::atomic::Ordering::Relaxed)
}
