// SPDX-License-Identifier: 0BSD

//! A Symbolica evaluator run over row-major batches, one row at a time.

use symbolica::atom::Atom;
use symbolica::domains::float::Real;
use symbolica::evaluate::{EvaluationDomain, EvaluationError, ExpressionEvaluator};

/// An [`ExpressionEvaluator`] taking whole row-major batches: `rows` rows of
/// [`Self::get_input_len`] inputs give `rows` rows of
/// [`Self::get_output_len`] outputs. Each row is Symbolica's ordinary scalar
/// evaluation of that row (the interpreter keeps its stack between rows), so
/// a row's value does not depend on the batch it is evaluated in.
#[derive(Clone)]
pub struct RowEvaluator<T> {
    evaluator: ExpressionEvaluator<T>,
    inputs: usize,
    outputs: usize,
}

impl<T: Default> RowEvaluator<T> {
    pub fn new(evaluator: ExpressionEvaluator<T>) -> Self {
        let (inputs, outputs) = (evaluator.get_input_len(), evaluator.get_output_len());
        Self {
            evaluator,
            inputs,
            outputs,
        }
    }
}

impl<T> RowEvaluator<T> {
    /// `source` with its coefficients mapped by `f` at `binary_prec` bits
    /// ([`ExpressionEvaluator::map_coeff_with_prec`]).
    pub fn map_coeff_with_prec<S: Default, F: Fn(&S) -> T>(
        source: ExpressionEvaluator<S>,
        f: &F,
        binary_prec: u32,
    ) -> Self
    where
        T: EvaluationDomain,
    {
        let (inputs, outputs) = (source.get_input_len(), source.get_output_len());
        Self {
            evaluator: source.map_coeff_with_prec(f, binary_prec),
            inputs,
            outputs,
        }
    }

    /// Number of parameters in each input row.
    pub fn get_input_len(&self) -> usize {
        self.inputs
    }

    /// Number of results in each output row.
    pub fn get_output_len(&self) -> usize {
        self.outputs
    }
}

impl<T: Real> RowEvaluator<T> {
    /// Evaluate `rows` rows. Input and output lengths must be exactly
    /// `rows * get_input_len()` and `rows * get_output_len()`; a mismatch is
    /// an error and leaves the output unchanged. Empty batches are supported.
    pub fn evaluate_batch(
        &mut self,
        parameters: &[T],
        out: &mut [T],
        rows: usize,
    ) -> Result<(), EvaluationError> {
        let (width, height) = (self.get_input_len(), self.get_output_len());
        let overflow = || EvaluationError::EvaluationFailed {
            expression: Atom::num(0),
            reason: "Batch dimensions exceed the addressable buffer size".into(),
        };
        let inputs = rows.checked_mul(width).ok_or_else(overflow)?;
        let outputs = rows.checked_mul(height).ok_or_else(overflow)?;
        if parameters.len() != inputs {
            return Err(EvaluationError::InvalidParameterCount {
                expected: inputs,
                actual: parameters.len(),
            });
        }
        if out.len() != outputs {
            return Err(EvaluationError::InvalidOutputCount {
                expected: outputs,
                actual: out.len(),
            });
        }
        for row in 0..rows {
            self.evaluator.try_evaluate(
                &parameters[row * width..(row + 1) * width],
                &mut out[row * height..(row + 1) * height],
            )?;
        }
        Ok(())
    }
}
