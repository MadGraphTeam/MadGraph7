//! Map the MadLoop ABI selector to the upstream master-provider option.
//!
//! Numerical UV/IR separation belongs to oneloopmaster and is forwarded by
//! NinjaRS's master providers. Keep only ABI validation in this module.

pub(crate) use ninjars_symbolica::PoleSelection;

pub(crate) fn from_mode(mode: i32) -> std::result::Result<PoleSelection, String> {
    match mode {
        0 => Ok(PoleSelection::BOTH),
        1 => Ok(PoleSelection::UV),
        2 => Ok(PoleSelection::IR),
        3 => Ok(PoleSelection::NONE),
        _ => Err(format!(
            "invalid loop pole mode {mode}: expected 0, 1, 2, or 3"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorGenericNumerator;
    use ninjars::{
        AmplitudeInput, Complex, EvaluationStatus, Float, FloatLike, FourMomentum, MasterFailure,
        MasterIntegral, MasterProvider, MasterRequest, MasterStatus,
    };
    use ninjars_symbolica::{
        SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts, SymbolicaMasters,
    };

    fn c(value: f64) -> Complex<f64> {
        Complex::new(value, 0.)
    }

    fn close(actual: Complex<f64>, expected: Complex<f64>) {
        assert!(
            (actual.re - expected.re).hypot(actual.im - expected.im)
                < 2e-11 * (1. + expected.re.hypot(expected.im)),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn scaleless_bubble_poles_survive_the_provider_in_every_tensor_rank() {
        let mut provider = SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::BOTH);
        for (rank, integral) in [
            MasterIntegral::B0,
            MasterIntegral::B1,
            MasterIntegral::B11,
            MasterIntegral::B111,
            MasterIntegral::B1111,
        ]
        .into_iter()
        .enumerate()
        {
            let residue = (-1f64).powi(rank as i32) / (rank + 1) as f64;
            for (selection, single) in [
                (PoleSelection::BOTH, 0.),
                (PoleSelection::UV, residue),
                (PoleSelection::IR, -residue),
                (PoleSelection::NONE, 0.),
                (PoleSelection::BOTH, 0.),
            ] {
                provider.set_pole_selection(selection);
                let mut output = [[c(91.); 3]];
                provider
                    .evaluate(
                        MasterRequest {
                            integral,
                            invariants: &[c(0.)],
                            masses_squared: &[c(0.); 2],
                            scales_squared: &[c(17.)],
                        },
                        &mut output,
                    )
                    .unwrap();
                for (actual, expected) in output[0].into_iter().zip([0., single, 0.]) {
                    close(actual, c(expected));
                }
            }
        }
    }

    #[test]
    fn selected_master_poles_add_to_the_total_without_changing_finite_parts() {
        let mut provider = SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::BOTH);
        for (integral, invariants, masses) in [
            (MasterIntegral::A0, vec![], vec![Complex::new(4., -0.2)]),
            (MasterIntegral::An { rank: 2 }, vec![], vec![c(4.)]),
            (MasterIntegral::An { rank: 4 }, vec![], vec![c(4.)]),
            (MasterIntegral::B0, vec![c(-11.)], vec![c(4.), c(9.)]),
            (MasterIntegral::B1, vec![c(-11.)], vec![c(4.), c(9.)]),
            (MasterIntegral::B11, vec![c(-11.)], vec![c(4.), c(9.)]),
            (MasterIntegral::B111, vec![c(-11.)], vec![c(4.), c(9.)]),
            (MasterIntegral::B1111, vec![c(-11.)], vec![c(4.), c(9.)]),
            (
                MasterIntegral::C0,
                vec![c(0.), c(0.), c(-11.)],
                vec![c(0.); 3],
            ),
            (
                MasterIntegral::D0,
                vec![c(0.), c(0.), c(0.), c(0.), c(-11.), c(-7.)],
                vec![c(0.); 4],
            ),
        ] {
            let mut results = [[[c(0.); 3]]; 4];
            for (selection, output) in [
                PoleSelection::BOTH,
                PoleSelection::UV,
                PoleSelection::IR,
                PoleSelection::NONE,
            ]
            .into_iter()
            .zip(&mut results)
            {
                provider.set_pole_selection(selection);
                provider
                    .evaluate(
                        MasterRequest {
                            integral,
                            invariants: &invariants,
                            masses_squared: &masses,
                            scales_squared: &[c(3.)],
                        },
                        output,
                    )
                    .unwrap();
            }
            let [total, uv, ir, none] = results.map(|row| row[0]);
            for selected in [uv, ir, none] {
                close(selected[0], total[0]);
            }
            close(uv[1] + ir[1], total[1]);
            close(uv[2], c(0.));
            close(ir[2], total[2]);
            close(none[1], c(0.));
            close(none[2], c(0.));
            if matches!(
                integral,
                MasterIntegral::A0
                    | MasterIntegral::An { .. }
                    | MasterIntegral::B0
                    | MasterIntegral::B1
                    | MasterIntegral::B11
                    | MasterIntegral::B111
                    | MasterIntegral::B1111
            ) {
                close(ir[1], c(0.));
            } else {
                close(uv[1], c(0.));
            }
        }
    }

    #[test]
    fn failed_master_rows_and_structural_errors_preserve_outputs() {
        let mut provider = SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::UV);
        let sentinel = [c(67.); 3];
        let mut output = [sentinel; 2];
        let mut statuses = [MasterStatus::Success; 2];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::B0,
                    invariants: &[c(-1.), c(-1.)],
                    masses_squared: &[c(1.), c(1.), Complex::new(1., 0.1), c(1.)],
                    scales_squared: &[c(1.), c(1.)],
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert_eq!(
            statuses,
            [
                MasterStatus::Success,
                MasterStatus::Failed(MasterFailure::InvalidMass)
            ]
        );
        assert_eq!(output[1], sentinel);
        close(output[0][1], c(1.));
        let before = output;
        let before_statuses = statuses;
        assert!(
            provider
                .evaluate_batch(
                    MasterRequest {
                        integral: MasterIntegral::B0,
                        invariants: &[c(-1.)],
                        masses_squared: &[c(1.), c(1.)],
                        scales_squared: &[c(1.), c(1.)],
                    },
                    &mut output,
                    &mut statuses
                )
                .is_err()
        );
        assert_eq!(output, before);
        assert_eq!(statuses, before_statuses);
        assert!(
            provider
                .evaluate(
                    MasterRequest {
                        integral: MasterIntegral::B0,
                        invariants: &[c(-1.), c(-1.)],
                        masses_squared: &[c(1.), c(1.), Complex::new(1., 0.1), c(1.)],
                        scales_squared: &[c(1.), c(1.)],
                    },
                    &mut output
                )
                .is_err()
        );
        assert_eq!(output, before);
    }

    #[test]
    fn multiprecision_scaleless_tensor_bubble_retains_exact_uv_ir_cancellation() {
        let prototype = Float::new(256);
        let z = Complex::new(prototype.clone(), prototype.clone());
        let one = Complex::new(Float::with_val(256, 1), prototype.clone());
        let mut provider = SymbolicaMasters::new(prototype).with_pole_selection(PoleSelection::UV);
        let mut results = [std::array::from_fn::<_, 3, _>(|_| z.clone())];
        let mut poles = Vec::new();
        for selection in [PoleSelection::UV, PoleSelection::IR, PoleSelection::BOTH] {
            provider.set_pole_selection(selection);
            provider
                .evaluate(
                    MasterRequest {
                        integral: MasterIntegral::B11,
                        invariants: std::slice::from_ref(&z),
                        masses_squared: &[z.clone(), z.clone()],
                        scales_squared: std::slice::from_ref(&one),
                    },
                    &mut results,
                )
                .unwrap();
            assert!(results[0][0].is_fully_zero() && results[0][2].is_fully_zero());
            poles.push(results[0][1].clone());
        }
        assert!((poles[0].clone() + &poles[1]).is_fully_zero());
        assert!(!poles[0].is_fully_zero());
        assert!(poles[2].is_fully_zero());
    }

    #[test]
    fn generic_reduction_keeps_lightlike_scaleless_bubble_coefficients() {
        use ninjars::fast::tensor_layout::{monomial_index, ncoeff};
        let mut low = SymbolicaCuts::new(0.).unwrap();
        let mut high = SymbolicaHighCuts::new(0.).unwrap();
        let mut masters = SymbolicaMasters::new(0.).with_pole_selection(PoleSelection::BOTH);
        let shifts = [
            FourMomentum::from([c(0.); 4]),
            FourMomentum::from([c(2.), c(0.), c(0.), c(2.)]),
        ];
        for rank in 0..=4 {
            let mut amplitude = SymbolicaAmplitudeKernels::new(2, rank, 0.)
                .unwrap()
                .into_amplitude(1)
                .unwrap();
            let mut numerator = TensorGenericNumerator::new(rank, 0, &0.).unwrap();
            let mut coefficients = vec![c(0.); ncoeff(rank)];
            coefficients[monomial_index([rank as u32, 0, 0, 0])] = c(1.);
            numerator.set_coefficients(coefficients, 1).unwrap();
            // For N(q)=q0^rank and null K=(2,0,0,2), all metric coefficients
            // vanish: the only UV residue is K0^rank B_{1...1}.
            let uv = (-2f64).powi(rank as i32) / (rank + 1) as f64;
            for (selection, expected) in [
                (PoleSelection::BOTH, 0.),
                (PoleSelection::UV, uv),
                (PoleSelection::IR, -uv),
            ] {
                amplitude.reset_batch(1);
                masters.set_pole_selection(selection);
                amplitude
                    .evaluate_batch_with_inputs(
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &[c(0.); 2],
                            scales_squared: &[c(1.)],
                            sampling_scales: &[c(1.)],
                            s_matrix: Some(&[c(0.); 4]),
                        },
                        &mut numerator,
                        &mut low,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                assert_eq!(
                    amplitude.statuses()[0],
                    EvaluationStatus::Success,
                    "rank {rank}"
                );
                let value = amplitude.results()[0];
                close(value[0], c(0.));
                close(value[1], c(expected));
                close(value[2], c(0.));
            }
        }
    }
}
