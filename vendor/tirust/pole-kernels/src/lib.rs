//! Scalar master residues generated from exact oneloopmaster expressions.
//!
//! A result has exactly two coefficients, `[epsilon^-1, epsilon^-2]`, so an
//! omitted finite coefficient cannot be confused with a computed zero. Inputs
//! use oneloopmaster's squared invariants and masses, with `mu_squared` last.
//! The warmed evaluator runs generated native arithmetic without a JIT or
//! symbolic evaluator. No finite coefficient is constructed or evaluated.
#![forbid(unsafe_code)]

use oneloop::rust_codegen::runtime::{C, ConstantSpec, Context, primitives};
pub use oneloop::{NativeFloat, ScalarIntegral};
pub use symbolica::domains::float::Complex;

#[rustfmt::skip]
#[allow(clippy::all)]
mod generated_a0;
#[rustfmt::skip]
#[allow(clippy::all)]
mod generated_b0;
#[rustfmt::skip]
#[allow(clippy::all)]
mod generated_c0;
#[rustfmt::skip]
#[allow(clippy::all)]
mod generated_d0;
#[rustfmt::skip]
#[allow(clippy::all)]
mod generated_constants;

/// Provenance and instruction counts emitted with the committed kernels.
pub const GENERATION_MANIFEST: &str = include_str!("generated_manifest.txt");

/// Reusable pole-only evaluator for A0, B0, C0 or D0.
///
/// Supported scalar types are native `f64`, Symbolica `DoubleFloat`, and
/// Symbolica `Float`. Fixed-precision types require their native bit count.
#[derive(Clone)]
pub struct ScalarPoleEvaluator<T: NativeFloat> {
    family: ScalarIntegral,
    bits: u32,
    context: Context<T>,
    arguments: Vec<C<T>>,
}

impl<T: NativeFloat> ScalarPoleEvaluator<T> {
    pub fn new(family: ScalarIntegral) -> Result<Self, String> {
        Self::with_binary_precision(family, T::DEFAULT_BITS)
    }

    pub fn with_binary_precision(family: ScalarIntegral, bits: u32) -> Result<Self, String> {
        if family == ScalarIntegral::DB0 {
            return Err("pole-only scalar kernels support A0, B0, C0 and D0; not dB0".into());
        }
        let context = Context::new(generated_constants::CONSTANTS, bits)?;
        Ok(Self {
            family,
            bits,
            context,
            arguments: (0..family.arity())
                .map(|_| C::new(T::zero_at(bits), T::zero_at(bits)))
                .collect(),
        })
    }

    pub fn family(&self) -> ScalarIntegral {
        self.family
    }

    pub fn binary_precision(&self) -> u32 {
        self.bits
    }

    /// Evaluate `[simple pole, double pole]`. Errors leave output unchanged.
    ///
    /// Invariants must be real, squared masses must have nonpositive imaginary
    /// parts, and `mu_squared` must be positive and real. Every input component
    /// must be finite. Nonfinite residues return an error for caller rescue.
    pub fn evaluate(&mut self, input: &[C<T>], output: &mut [C<T>]) -> Result<(), String> {
        if input.len() != self.family.arity() || output.len() != 2 {
            return Err(format!(
                "{} poles require {} inputs and 2 outputs; received {} and {}",
                self.family.name(),
                self.family.arity(),
                input.len(),
                output.len()
            ));
        }
        self.validate_point(input)?;
        // Arbitrary-precision constants and inputs must share the working
        // precision. D0 also needs the upstream public soft-box relabelling.
        let input = if T::FIXED_BITS.is_none() || self.family == ScalarIntegral::D0 {
            for (working, value) in self.arguments.iter_mut().zip(input) {
                working.re = value.re.at_precision(self.bits);
                working.im = value.im.at_precision(self.bits);
            }
            if self.family == ScalarIntegral::D0 {
                oneloop::regular_d0_labelling(&mut self.arguments);
            }
            &self.arguments
        } else {
            input
        };
        let values = match self.family {
            ScalarIntegral::A0 => generated_a0::evaluate(input, &self.context),
            ScalarIntegral::B0 => generated_b0::evaluate(input, &self.context),
            ScalarIntegral::C0 => generated_c0::evaluate(input, &self.context),
            ScalarIntegral::D0 => generated_d0::evaluate(input, &self.context),
            ScalarIntegral::DB0 => unreachable!("constructor rejects dB0"),
        };
        if values
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(format!(
                "{} pole-only kernel returned nonfinite residues",
                self.family.name()
            ));
        }
        for (target, value) in output.iter_mut().zip(values) {
            *target = value;
        }
        Ok(())
    }

    fn validate_point(&self, input: &[C<T>]) -> Result<(), String> {
        if input.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err("all input components must be finite".into());
        }
        let momenta = match self.family {
            ScalarIntegral::A0 => 0,
            ScalarIntegral::B0 => 1,
            ScalarIntegral::C0 => 3,
            ScalarIntegral::D0 => 6,
            ScalarIntegral::DB0 => unreachable!("constructor rejects dB0"),
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
}
