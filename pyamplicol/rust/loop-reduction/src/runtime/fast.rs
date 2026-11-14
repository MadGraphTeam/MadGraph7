// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    C64, FastInput, FastPolynomialReducer, FastProfile, FastStatus, FastTensorPolynomialReducer,
};
use crate::{FourMomentum, Lane, Laurent};
use symbolica::atom::Atom;

enum FastPolynomial {
    Callback(FastPolynomialReducer),
    Tensor(FastTensorPolynomialReducer),
}
impl FastPolynomial {
    fn set_profiling(&mut self, enabled: bool) {
        match self {
            Self::Callback(plan) => plan.set_profiling(enabled),
            Self::Tensor(plan) => plan.set_profiling(enabled),
        }
    }
    fn take_profile(&mut self) -> FastProfile {
        match self {
            Self::Callback(plan) => plan.take_profile(),
            Self::Tensor(plan) => plan.take_profile(),
        }
    }
    fn evaluate(
        &mut self,
        ids: &[u64],
        input: FastInput<'_>,
        parameters: &[Complex<f64>],
    ) -> ExecutionResult<Vec<Lane<C64, FastStatus>>> {
        match self {
            Self::Callback(plan) => plan.evaluate(ids, input, parameters).map_err(error),
            Self::Tensor(plan) => plan.evaluate(ids, input, parameters).map_err(error),
        }
    }
}

pub(super) struct FastReductions {
    sectors: [FastPolynomial; 3],
    active: [bool; 3],
    pub propagators: usize,
    pub parameter_count: usize,
}
impl FastReductions {
    pub fn set_profiling(&mut self, enabled: bool) {
        for sector in &mut self.sectors {
            sector.set_profiling(enabled);
        }
    }
    pub fn take_profile(&mut self) -> FastProfile {
        let mut profile = FastProfile::default();
        for sector in &mut self.sectors {
            profile.merge(sector.take_profile());
        }
        profile
    }
    pub fn new(
        expressions: &[String; 3],
        parameters: &[String],
        n: usize,
        tensor: bool,
    ) -> ExecutionResult<Self> {
        // The expressions may call the source-scalar primitives: install them
        // before parsing (the extension's import no longer does).
        super::register_scalar_primitives();
        let parse =
            |text: &str| Atom::parse(text, "pyamplicol_loop", Default::default()).map_err(error);
        let variables = [
            parse("q0")?,
            parse("q1")?,
            parse("q2")?,
            parse("q3")?,
            parse("mu2")?,
        ];
        let parameters = parameters
            .iter()
            .map(|p| parse(p))
            .collect::<ExecutionResult<Vec<_>>>()?;
        let mut active = [false; 3];
        let mut sectors = Vec::with_capacity(3);
        for (index, expression) in expressions.iter().enumerate() {
            let expression = parse(expression)?;
            active[index] = !expression.is_zero();
            sectors.push(if tensor {
                FastPolynomial::Tensor(
                    FastTensorPolynomialReducer::new(
                        n,
                        expression,
                        variables.clone(),
                        parameters.clone(),
                    )
                    .map_err(error)?,
                )
            } else {
                FastPolynomial::Callback(
                    FastPolynomialReducer::new(
                        n,
                        expression,
                        variables.clone(),
                        parameters.clone(),
                    )
                    .map_err(error)?,
                )
            });
        }
        if !active.iter().any(|value| *value) {
            active[0] = true;
        }
        Ok(Self {
            sectors: sectors.try_into().ok().unwrap(),
            active,
            propagators: n,
            parameter_count: parameters.len(),
        })
    }
    pub fn evaluate_native(
        &mut self,
        ids: &[u64],
        shifts: &[FourMomentum<Complex<f64>>],
        masses: &[Complex<f64>],
        scales: &[Complex<f64>],
        parameters: &[Complex<f64>],
        s_matrix: Option<&[Complex<f64>]>,
    ) -> ExecutionResult<Vec<Lane<Complex<f64>, String>>> {
        let real = |value: &Complex<f64>| {
            if value.im != 0. {
                return Err(error(
                    "fast engine requires real shifts, scales and invariant matrix",
                ));
            }
            Ok(value.re)
        };
        let shifts = shifts
            .iter()
            .map(|p| {
                let v = p.components();
                Ok([real(v[0])?, real(v[1])?, real(v[2])?, real(v[3])?])
            })
            .collect::<ExecutionResult<Vec<_>>>()?;
        let masses: Vec<_> = masses.iter().map(|z| C64::new(z.re, z.im)).collect();
        let scales = scales
            .iter()
            .map(real)
            .collect::<ExecutionResult<Vec<_>>>()?;
        let matrix = s_matrix
            .map(|s| s.iter().map(real).collect::<ExecutionResult<Vec<_>>>())
            .transpose()?;
        let input = FastInput {
            lanes: ids.len(),
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: matrix.as_deref(),
        };
        let mut results = Vec::with_capacity(3);
        for (index, sector) in self.sectors.iter_mut().enumerate() {
            if self.active[index] {
                results.push(sector.evaluate(ids, input, parameters)?);
            } else {
                results.push(
                    ids.iter()
                        .map(|&id| Lane {
                            id,
                            status: FastStatus::Success,
                            value: Some(Laurent::from([C64::ZERO; 3])),
                        })
                        .collect(),
                );
            }
        }
        let mut sectors = results.into_iter();
        let zero = sectors.next().unwrap();
        let one = sectors.next().unwrap();
        let two = sectors.next().unwrap();
        Ok(zero
            .into_iter()
            .zip(one)
            .zip(two)
            .map(|((zero, one), two)| {
                let status = [zero.status, one.status, two.status]
                    .iter()
                    .enumerate()
                    .find(|(_, s)| **s != FastStatus::Success)
                    .map(|(i, s)| format!("epsilon sector {i}: {s:?}"))
                    .unwrap_or_else(|| "success".into());
                let value = match (zero.value, one.value, two.value) {
                    (Some(a), Some(b), Some(c)) => {
                        let v = Laurent::epsilon_sectors(a, b, c);
                        let convert = |z: C64| Complex::new(z.re, z.im);
                        Some(Laurent {
                            finite: convert(v.finite),
                            single_pole: convert(v.single_pole),
                            double_pole: convert(v.double_pole),
                        })
                    }
                    _ => None,
                };
                Lane {
                    id: zero.id,
                    status,
                    value,
                }
            })
            .collect())
    }
}
