//! Ahead-of-time generic Rust implementations of the native master expressions.
//!
//! The generated functions contain ordinary arithmetic and structured Rust
//! branches. They neither build nor execute a Symbolica/SymJIT evaluator.
#[cfg(feature = "rust-codegen")]
pub mod batch;
#[cfg_attr(not(feature = "generated-evaluators"), allow(dead_code))]
pub mod primitives;
mod tensor;
pub use tensor::{NativeTensorEvaluator, TensorIntegral};
#[cfg(feature = "generated-evaluators")]
mod simple;
#[cfg(feature = "generated-evaluators")]
mod handwritten;
// The emitter has its own stable formatting and mechanically named SSA locals.
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_a0;
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_b0;
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_db0;
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_c0;
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_d0;
#[rustfmt::skip]
#[allow(clippy::all)]
#[cfg(feature = "generated-evaluators")]
mod generated_constants;

use crate::ScalarIntegral;
pub use primitives::NativeFloat;
use primitives::*;
#[cfg(any(feature = "generated-evaluators", feature = "rust-codegen"))]
mod constants;
#[cfg(any(feature = "generated-evaluators", feature = "rust-codegen"))]
pub use constants::{ConstantContext, ConstantSpec};

#[cfg(not(feature = "generated-evaluators"))]
mod disabled;
#[cfg(not(feature = "generated-evaluators"))]
pub use disabled::NativeEvaluator;

#[derive(Clone)]
#[cfg(feature = "generated-evaluators")]
struct Context<T: NativeFloat> {
    constants: ConstantContext<T>,
    simple: simple::Constants<T>,
    hand: handwritten::Constants<T>,
}

#[cfg(feature = "generated-evaluators")]
impl<T: NativeFloat> Context<T> {
    fn new(bits: u32) -> Result<Self, String> {
        let prototype = T::zero_at(bits);
        let constants = ConstantContext::new(generated_constants::CONSTANTS, bits)?;
        Ok(Self {
            constants,
            simple: simple::Constants::new(&prototype),
            hand: handwritten::Constants::new(&prototype),
        })
    }

    #[inline(always)]
    fn constant(&self, index: usize) -> &C<T> {
        self.constants.constant(index)
    }
}

/// A reusable direct Rust scalar master implementation.
///
/// `T` may be `f64`, Symbolica's `DoubleFloat`, or arbitrary-precision `Float`.
/// Inputs use the usual squared quantities and have `mu_squared` last; outputs
/// are `[finite, simple pole, double pole]`. This backend uses no runtime
/// evaluator or JIT. Constants are prepared once at construction.
#[derive(Clone)]
#[cfg(feature = "generated-evaluators")]
pub struct NativeEvaluator<T: NativeFloat> {
    family: ScalarIntegral,
    bits: u32,
    context: Context<T>,
    arguments: Vec<C<T>>,
    handwritten: bool,
}

#[cfg(feature = "generated-evaluators")]
impl<T: NativeFloat> NativeEvaluator<T> {
    /// Construct at the numeric type's default working precision.
    pub fn new(family: ScalarIntegral) -> Result<Self, String> {
        Self::with_binary_precision(family, T::DEFAULT_BITS)
    }

    /// Construct with an explicit bit count. Fixed-precision types require
    /// their native bit count; use `Float` for arbitrary working precision.
    pub fn with_binary_precision(family: ScalarIntegral, bits: u32) -> Result<Self, String> {
        if bits == 0 {
            return Err("binary precision must be positive".into());
        }
        if let Some(fixed) = T::FIXED_BITS
            && bits != fixed
        {
            return Err(format!(
                "this numeric type has fixed precision {fixed} bits"
            ));
        }
        Ok(Self {
            family,
            bits,
            context: Context::new(bits)?,
            arguments: (0..family.arity())
                .map(|_| C::new(T::zero_at(bits), T::zero_at(bits)))
                .collect(),
            // Binary64 uses the hand-written sector kernels. Tracked-precision
            // types keep the generated expression code: their exact-zero tests
            // and precision bookkeeping follow the expression's own grouping.
            handwritten: T::FIXED_BITS == Some(53),
        })
    }

    /// Select the hand-written sector kernels (the binary64 default) or the
    /// generated expression code for every sector. For validation only.
    #[doc(hidden)]
    pub fn set_handwritten(&mut self, enabled: bool) {
        self.handwritten = enabled;
    }

    /// Integral family represented by this implementation.
    pub fn family(&self) -> ScalarIntegral {
        self.family
    }

    /// Working precision in bits (not decimal digits).
    pub fn binary_precision(&self) -> u32 {
        self.bits
    }

    /// Evaluate one point. Shape/domain errors leave output unchanged.
    pub fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>]) -> Result<(), String> {
        self.validate_shape(input.len(), output.len(), 1)?;
        self.validate_point(input)?;
        self.evaluate_row(input, output);
        Ok(())
    }

    /// Evaluate flat row-major points. Validate every row before writing any
    /// output; empty batches are valid. No hidden worker threads are spawned.
    pub fn evaluate_batch(
        &mut self,
        input: &[C<T>],
        output: &mut [C<T>],
        rows: usize,
    ) -> Result<(), String> {
        self.validate_shape(input.len(), output.len(), rows)?;
        for point in input.chunks_exact(self.family.arity()) {
            self.validate_point(point)?;
        }
        for (point, result) in input
            .chunks_exact(self.family.arity())
            .zip(output.chunks_exact_mut(3))
        {
            self.evaluate_row(point, result);
        }
        Ok(())
    }

    fn evaluate_row(&mut self, input: &[C<T>], output: &mut [C<T>]) {
        let input = if T::FIXED_BITS.is_some() {
            input
        } else {
            for (working, value) in self.arguments.iter_mut().zip(input) {
                working.re = value.re.at_precision(self.bits);
                working.im = value.im.at_precision(self.bits);
            }
            &self.arguments
        };
        let values = match self.family {
            ScalarIntegral::A0 if !truth(&input[0]) => {
                let mut values = simple::zeros(&input[0].re);
                values[1] = input[0].clone();
                values
            }
            ScalarIntegral::A0 => generated_a0::evaluate(input, &self.context),
            ScalarIntegral::B0 if input[..3].iter().all(|z| !truth(z)) => {
                simple::zeros(&input[0].re)
            }
            ScalarIntegral::B0 => generated_b0::evaluate(input, &self.context),
            ScalarIntegral::DB0 => generated_db0::evaluate(input, &self.context),
            ScalarIntegral::C0 => simple::massless_triangle(input, &self.context.simple)
                .or_else(|| {
                    self.handwritten
                        .then(|| handwritten::c0(input, &self.context.hand))
                        .flatten()
                })
                .unwrap_or_else(|| generated_c0::evaluate(input, &self.context)),
            ScalarIntegral::D0 => simple::massless_onshell_box(input, &self.context.simple)
                .or_else(|| {
                    self.handwritten
                        .then(|| handwritten::d0(input, &self.context.hand))
                        .flatten()
                })
                .unwrap_or_else(|| generated_d0::evaluate(input, &self.context)),
        };
        for (out, value) in output.iter_mut().zip(values) {
            *out = value;
        }
    }

    fn validate_shape(&self, inputs: usize, outputs: usize, rows: usize) -> Result<(), String> {
        let expected_inputs = rows
            .checked_mul(self.family.arity())
            .ok_or("input size overflow")?;
        let expected_outputs = rows.checked_mul(3).ok_or("output size overflow")?;
        if inputs != expected_inputs || outputs != expected_outputs {
            return Err(format!(
                "expected {expected_inputs} inputs and {expected_outputs} outputs for {rows} rows; received {inputs} and {outputs}"
            ));
        }
        Ok(())
    }

    fn validate_point(&self, input: &[C<T>]) -> Result<(), String> {
        validate_point(self.family, input)
    }
}

/// Shared domain contract for the explicit manual backends. Call only after
/// checking shape, so every point has the family's complete argument list.
pub(crate) fn validate_point<T: NativeFloat>(
    family: ScalarIntegral,
    input: &[C<T>],
) -> Result<(), String> {
    if input.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err("all input components must be finite".into());
    }
    let momenta = match family {
        ScalarIntegral::A0 => 0,
        ScalarIntegral::B0 | ScalarIntegral::DB0 => 1,
        ScalarIntegral::C0 => 3,
        ScalarIntegral::D0 => 6,
    };
    if input[..momenta].iter().any(|z| !z.im.is_zero()) {
        return Err("external squared invariants must be real".into());
    }
    if input[momenta..input.len() - 1]
        .iter()
        .any(|z| z.im > z.im.zero())
    {
        return Err("squared masses require nonpositive imaginary parts".into());
    }
    let mu = &input[input.len() - 1];
    if !mu.im.is_zero() || mu.re <= mu.re.zero() {
        return Err("mu_squared must be positive and real".into());
    }
    Ok(())
}
