// SPDX-License-Identifier: 0BSD

//! Independent analytic anchors; no MadLoop model or normalization is implied.
use ninjars::fast::{C64, FastInput, FastStatus, GenericNumerator, TensorNumerator};
use ninjars::{AmplitudeInput, Complex, DoubleFloat, EvaluationStatus, Float, FourMomentum, Real};
use ninjars_symbolica::PolynomialNumerator;
use oneloop::NativeFloat;
use rusticol_loop_reduction::{FastReducer, Laurent, PolynomialReducer};
use symbolica::{atom::Atom, domains::rational::Rational, evaluate::EvaluationDomain, parse};

fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}

fn close<T: NativeFloat>(actual: &Complex<T>, expected: &Complex<T>, prototype: &T, digits: u32) {
    let tolerance = prototype.from_rational(&Rational::from((1, 10)).pow(digits.into()));
    let error = actual - expected;
    assert!(
        error.re.norm() < tolerance && error.im.norm() < tolerance,
        "actual {actual}; expected {expected}; error {error}"
    );
}

fn scalar_and_mu2<T: NativeFloat>(prototype: T, digits: u32, tiny_power: Option<u64>)
where
    Complex<T>: Real + EvaluationDomain,
{
    let c = |v| Complex::new(prototype.from_i64(v), prototype.zero());
    let mut coefficient = c(1);
    if let Some(power) = tiny_power {
        coefficient.re += prototype.from_rational(&Rational::from((1, 2)).pow(power));
        assert!(coefficient.re > prototype.one());
    }
    let mut reducer = PolynomialReducer::new(
        1,
        parse!("a"),
        variables(),
        vec![parse!("a")],
        prototype.clone(),
        None,
    )
    .unwrap();
    let shifts = vec![FourMomentum::from([c(0), c(0), c(0), c(0)]); 3];
    let masses = [
        c(1),
        c(4),
        Complex::new(prototype.from_i64(4), -prototype.one()),
    ];
    let scales = [c(1), c(4), c(4)];
    let params = [coefficient.clone(), c(2), c(-3)];
    let ids = [91, 7, 40];
    // Repetition must overwrite, not double the generic engine's accumulator.
    for _ in 0..2 {
        let lanes = reducer
            .evaluate(
                &ids,
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    sampling_scales: &[],
                    s_matrix: None,
                },
                &params,
            )
            .unwrap();
        for row in 0..3 {
            assert_eq!(lanes[row].id, ids[row]);
            assert_eq!(lanes[row].status, EvaluationStatus::Success);
            let actual = lanes[row].value.as_ref().unwrap();
            let expected_pole = &params[row] * &masses[row];
            let expected_finite = &expected_pole * (c(1) - (&masses[row] / &scales[row]).log());
            close(&actual.finite, &expected_finite, &prototype, digits);
            close(&actual.single_pole, &expected_pole, &prototype, digits);
            close(&actual.double_pole, &c(0), &prototype, digits);
        }
        if tiny_power.is_some() {
            assert!(
                lanes[0].value.as_ref().unwrap().single_pole.re > prototype.one(),
                "the reduction lost input bits beyond binary64"
            );
        }
    }
    // A compacted batch binds its parameter rows anew and retains original IDs.
    let compact = reducer
        .evaluate(
            &[40],
            AmplitudeInput {
                shifts: &shifts[..1],
                masses_squared: &masses[..1],
                scales_squared: &scales[..1],
                sampling_scales: &[],
                s_matrix: None,
            },
            &[c(5)],
        )
        .unwrap();
    assert_eq!(compact[0].id, 40);
    close(
        &compact[0].value.as_ref().unwrap().finite,
        &c(5),
        &prototype,
        digits,
    );
    let empty = reducer
        .evaluate(
            &[],
            AmplitudeInput {
                shifts: &[],
                masses_squared: &[],
                scales_squared: &[],
                sampling_scales: &[],
                s_matrix: None,
            },
            &[],
        )
        .unwrap();
    assert!(empty.is_empty());

    // Under Ninja's μ²=-q_tilde² convention, the μ² triangle is +1/2.
    // This exercises explicit μ² callbacks, reductions and masters together.
    let mut mu = PolynomialReducer::new(
        3,
        parse!("a*mu2"),
        variables(),
        vec![parse!("a")],
        prototype.clone(),
        None,
    )
    .unwrap();
    let one_triangle =
        [[0, 0, 0, 0], [2, 1, 0, 0], [3, 0, 1, 0]].map(|p| FourMomentum::from(p.map(c)));
    let shifts: Vec<_> = (0..3).flat_map(|_| one_triangle.clone()).collect();
    let masses = vec![c(1); 9];
    let scales = vec![c(1); 3];
    let actual = mu
        .evaluate(
            &ids,
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                sampling_scales: &[],
                s_matrix: None,
            },
            &params,
        )
        .unwrap();
    for row in 0..3 {
        assert_eq!(actual[row].id, ids[row]);
        assert_eq!(actual[row].status, EvaluationStatus::Success);
        let result = actual[row].value.as_ref().unwrap();
        close(&result.finite, &(&params[row] / c(2)), &prototype, digits);
        close(&result.single_pole, &c(0), &prototype, digits);
        close(&result.double_pole, &c(0), &prototype, digits);
    }
}

#[test]
fn analytic_scalar_and_mu2_batches_in_three_precisions() {
    scalar_and_mu2(0_f64, 10, None);
    scalar_and_mu2(DoubleFloat::default(), 27, Some(80));
    scalar_and_mu2(Float::new(256), 60, Some(160));
}

fn pole_anchors<T: NativeFloat>(prototype: T, digits: u32)
where
    Complex<T>: Real + EvaluationDomain,
{
    let c = |v| Complex::new(prototype.from_i64(v), prototype.zero());
    let momentum = |p: [i64; 4]| FourMomentum::from(p.map(c));
    let mut bubble =
        PolynomialReducer::new(2, parse!("1"), variables(), vec![], prototype.clone(), None)
            .unwrap();
    let one_bubble = [momentum([0; 4]), momentum([0, 1, 0, 0])];
    let shifts: Vec<_> = (0..2).flat_map(|_| one_bubble.clone()).collect();
    let scales = [c(1), c(4)];
    let result = bubble
        .evaluate(
            &[17, 8],
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &vec![c(0); 4],
                scales_squared: &scales,
                sampling_scales: &[],
                s_matrix: None,
            },
            &[],
        )
        .unwrap();
    for row in 0..2 {
        assert_eq!(result[row].status, EvaluationStatus::Success);
        let value = result[row].value.as_ref().unwrap();
        close(
            &value.finite,
            &(c(2) + scales[row].log()),
            &prototype,
            digits,
        );
        close(&value.single_pole, &c(1), &prototype, digits);
        close(&value.double_pole, &c(0), &prototype, digits);
    }
    // C0(0,0,s;0,0,0) = (1/s) [L²/2, -L, 1],
    // L=log(-s/μ_R²), in the documented rΓ-normalized spacelike convention.
    let mut triangle =
        PolynomialReducer::new(3, parse!("1"), variables(), vec![], prototype.clone(), None)
            .unwrap();
    let one_triangle = [
        momentum([0; 4]),
        momentum([1, 0, 0, 1]),
        momentum([0, 0, 0, 2]),
    ];
    let shifts: Vec<_> = (0..2).flat_map(|_| one_triangle.clone()).collect();
    let result = triangle
        .evaluate(
            &[71, 9],
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &vec![c(0); 6],
                scales_squared: &scales,
                sampling_scales: &[],
                s_matrix: None,
            },
            &[],
        )
        .unwrap();
    for row in 0..2 {
        assert_eq!(result[row].status, EvaluationStatus::Success);
        let value = result[row].value.as_ref().unwrap();
        let logarithm = (c(4) / &scales[row]).log();
        close(
            &value.finite,
            &(-(logarithm.clone() * &logarithm) / c(8)),
            &prototype,
            digits,
        );
        close(&value.single_pole, &(&logarithm / c(4)), &prototype, digits);
        close(&value.double_pole, &(c(-1) / c(4)), &prototype, digits);
        let zero = Laurent::from([c(0), c(0), c(0)]);
        let shifted = Laurent::epsilon_sectors(zero.clone(), zero, value.clone());
        close(&shifted.finite, &(c(-1) / c(4)), &prototype, digits);
        close(&shifted.single_pole, &c(0), &prototype, digits);
        close(&shifted.double_pole, &c(0), &prototype, digits);
    }
    // A four-dimensional numerator still needs auxiliary μ² expansions:
    // q_4² = D0 + μ². The canceled-denominator B0 is scaleless here, so
    // the complete result is the rational +1/2 despite no explicit μ² in N.
    let mut four_dimensional = PolynomialReducer::new(
        3,
        parse!("q0^2-q1^2-q2^2-q3^2"),
        variables(),
        vec![],
        prototype.clone(),
        None,
    )
    .unwrap();
    let result = four_dimensional
        .evaluate(
            &[71, 9],
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &vec![c(0); 6],
                scales_squared: &scales,
                sampling_scales: &[],
                s_matrix: None,
            },
            &[],
        )
        .unwrap();
    for lane in result {
        assert_eq!(lane.status, EvaluationStatus::Success);
        let value = lane.value.unwrap();
        close(&value.finite, &(c(1) / c(2)), &prototype, digits);
        close(&value.single_pole, &c(0), &prototype, digits);
        close(&value.double_pole, &c(0), &prototype, digits);
    }
}

#[test]
fn analytic_massless_bubble_and_divergent_triangle_three_precisions() {
    pole_anchors(0_f64, 10);
    pole_anchors(DoubleFloat::default(), 27);
    pole_anchors(Float::new(256), 60);
}

#[test]
fn epsilon_sector_convolution_preserves_pole_order() {
    let result = Laurent::epsilon_sectors(
        Laurent::from([2, 3, 5]),
        Laurent::from([7, 11, 13]),
        Laurent::from([17, 19, 23]),
    );
    assert_eq!(
        result,
        Laurent {
            finite: 36,
            single_pole: 16,
            double_pole: 5
        }
    );
}

#[test]
fn unsupported_topologies_and_parameter_shapes_are_rejected() {
    assert!(matches!(
        PolynomialReducer::new(0, parse!("1"), variables(), vec![], 0_f64, None),
        Err(ninjars::Error::NoPropagators)
    ));
    assert!(matches!(
        PolynomialReducer::new(
            0,
            parse!("1"),
            variables(),
            vec![],
            DoubleFloat::default(),
            None
        ),
        Err(ninjars::Error::NoPropagators)
    ));
    assert!(matches!(
        PolynomialReducer::new(0, parse!("1"), variables(), vec![], Float::new(256), None),
        Err(ninjars::Error::NoPropagators)
    ));
    assert!(matches!(
        PolynomialReducer::new(1, parse!("mu2^2"), variables(), vec![], 0_f64, None),
        Err(ninjars::Error::Rank {
            rank: 4,
            propagators: 1
        })
    ));
    let c = |v| Complex::new(v, 0.);
    let mut reducer =
        PolynomialReducer::new(1, parse!("a"), variables(), vec![parse!("a")], 0_f64, None)
            .unwrap();
    let shifts = [FourMomentum::from([c(0.); 4])];
    assert!(matches!(
        reducer.evaluate(
            &[99],
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &[c(1.)],
                scales_squared: &[c(1.)],
                sampling_scales: &[],
                s_matrix: None
            },
            &[]
        ),
        Err(ninjars::Error::Shape { .. })
    ));
}

#[test]
fn fast_tensor_overwrites_preserves_ids_and_hides_failed_values() {
    let mut reducer = FastReducer::new(1, TensorNumerator::new(0, 3)).unwrap();
    let shifts = [[0.; 4]; 3];
    let masses = [C64::real(1.), C64::real(4.), C64::new(4., -1.)];
    let scales = [1., 4., 4.];
    let ids = [50, 2, 900];
    for coefficient in [1., 2.] {
        reducer
            .numerator_mut()
            .set_coefficients(3, &[C64::real(coefficient); 3])
            .unwrap();
        let result = reducer
            .evaluate(
                &ids,
                FastInput {
                    lanes: 3,
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    s_matrix: None,
                },
            )
            .unwrap();
        for row in 0..3 {
            assert_eq!(result[row].id, ids[row]);
            assert_eq!(result[row].status, FastStatus::Success);
            let expected = masses[row] * coefficient;
            let pole = result[row].value.as_ref().unwrap().single_pole;
            assert!((pole.re - expected.re).abs() < 1e-12);
            assert!((pole.im - expected.im).abs() < 1e-12);
        }
    }
    reducer
        .numerator_mut()
        .set_coefficients(3, &[C64::real(3.), C64::real(f64::NAN), C64::real(5.)])
        .unwrap();
    let result = reducer
        .evaluate(
            &ids,
            FastInput {
                lanes: 3,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: None,
            },
        )
        .unwrap();
    assert_eq!(result[1].id, 2);
    assert_ne!(result[1].status, FastStatus::Success);
    assert!(result[1].value.is_none());
    assert!(result[0].value.is_some() && result[2].value.is_some());
    assert!(
        reducer
            .evaluate(
                &[1],
                FastInput {
                    lanes: 3,
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    s_matrix: None
                }
            )
            .is_err()
    );
    reducer.numerator_mut().set_coefficients(0, &[]).unwrap();
    assert!(
        reducer
            .evaluate(
                &[],
                FastInput {
                    lanes: 0,
                    shifts: &[],
                    masses_squared: &[],
                    scales_squared: &[],
                    s_matrix: None
                }
            )
            .unwrap()
            .is_empty()
    );
}

#[test]
fn fast_generic_adapter_reduces_explicit_mu2_batch() {
    let mut numerator =
        PolynomialNumerator::new(parse!("a*mu2"), variables(), vec![parse!("a")], 0_f64).unwrap();
    numerator
        .set_parameter_batch(&[Complex::new(2., 0.), Complex::new(-3., 0.)], 2)
        .unwrap();
    let mut reducer = FastReducer::new(3, GenericNumerator::new(numerator)).unwrap();
    let shifts = [[0., 0., 0., 0.], [2., 1., 0., 0.], [3., 0., 1., 0.]].repeat(2);
    let result = reducer
        .evaluate(
            &[123, 17],
            FastInput {
                lanes: 2,
                shifts: &shifts,
                masses_squared: &[C64::real(1.); 6],
                scales_squared: &[1.; 2],
                s_matrix: None,
            },
        )
        .unwrap();
    for (row, expected) in [1., -1.5].into_iter().enumerate() {
        assert_eq!(result[row].status, FastStatus::Success);
        let actual = result[row].value.as_ref().unwrap();
        assert!((actual.finite.re - expected).abs() < 1e-10);
        assert!(actual.finite.im.abs() < 1e-10);
        assert!(actual.single_pole.re.abs() < 1e-10 && actual.double_pole.re.abs() < 1e-10);
    }
}

#[test]
fn fast_four_dimensional_tensor_retains_rational_part() {
    let mut numerator = TensorNumerator::new(2, 2);
    for (lane, factor) in [2., -3.].into_iter().enumerate() {
        for (axis, sign) in [1., -1., -1., -1.].into_iter().enumerate() {
            let mut powers = [0; 4];
            powers[axis] = 2;
            let index = ninjars::fast::tensor_layout::monomial_index(powers);
            numerator.lane_coefficients_mut(lane)[index] = C64::real(sign * factor);
        }
    }
    let mut reducer = FastReducer::new(3, numerator).unwrap();
    let shifts = [[0.; 4], [1., 0., 0., 1.], [0., 0., 0., 2.]].repeat(2);
    let result = reducer
        .evaluate(
            &[61, 12],
            FastInput {
                lanes: 2,
                shifts: &shifts,
                masses_squared: &[C64::ZERO; 6],
                scales_squared: &[1.; 2],
                s_matrix: None,
            },
        )
        .unwrap();
    for (row, expected) in [1., -1.5].into_iter().enumerate() {
        assert_eq!(result[row].status, FastStatus::Success);
        let actual = result[row].value.as_ref().unwrap();
        assert!((actual.finite.re - expected).abs() < 1e-10);
        assert!(actual.finite.im.abs() < 1e-10);
        assert!(actual.single_pole.re.abs() < 1e-10 && actual.double_pole.re.abs() < 1e-10);
    }
}

#[test]
fn profiling_forwards_owner_rank_n2_callbacks() {
    use rusticol_loop_reduction::{FastPolynomialReducer, FastTensorPolynomialReducer};
    // Three exact denominator factors cancel a rank-six, four-propagator
    // numerator to A0(1). The new x2 callbacks must survive instrumentation;
    // this is an adapter contract, not general rank-n+2 physics certification.
    // The owner's rn2-int-complex-n4-r6 fixture supplies these screened
    // Lorentzian cut planes, here uniformly scaled by an exact power of two.
    // Purely spatial orthogonal shifts do not satisfy that real-basis domain.
    let one_point = [
        [0., 0., 0., 0.],
        [341. / 128., 86. / 128., -115. / 128., 83. / 128.],
        [180. / 128., 118. / 128., -44. / 128., 191. / 128.],
        [521. / 128., 104. / 128., 34. / 128., 257. / 128.],
    ];
    let numerator = one_point[..3].iter().fold(parse!("a"), |value, shift| {
        value
            * symbolica::parse!(&format!(
                "(q0+({}))^2-(q1+({}))^2-(q2+({}))^2-(q3+({}))^2-1-mu2",
                shift[0], shift[1], shift[2], shift[3]
            ))
    });
    let shifts = one_point.repeat(2);
    let masses = [C64::real(1.); 8];
    let scales = [1.; 2];
    let parameters = [Complex::new(1., 0.), Complex::new(2., 0.)];
    macro_rules! check {
        ($reducer:ty) => {{
            let mut reducer =
                <$reducer>::new(4, numerator.clone(), variables(), vec![parse!("a")]).unwrap();
            let mut previous = None;
            for profile in [false, true] {
                reducer.set_profiling(profile);
                let result = reducer
                    .evaluate(
                        &[19, 3],
                        FastInput {
                            lanes: 2,
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            s_matrix: None,
                        },
                        &parameters,
                    )
                    .unwrap();
                let mut values = Vec::new();
                for (index, row) in result.iter().enumerate() {
                    assert_eq!(row.status, FastStatus::Success);
                    assert_eq!(row.id, [19, 3][index]);
                    let value = row.value.as_ref().unwrap();
                    let expected = (index + 1) as f64;
                    for coefficient in [&value.finite, &value.single_pole] {
                        assert!((coefficient.re - expected).abs() < 1e-7);
                        assert!(coefficient.im.abs() < 1e-7);
                    }
                    assert!(value.double_pole.re.abs() < 1e-7);
                    assert!(value.double_pole.im.abs() < 1e-7);
                    values.push(value.clone());
                }
                if let Some(previous) = previous {
                    assert_eq!(values, previous);
                    assert!(reducer.take_profile().numerator_calls > 0);
                }
                previous = Some(values);
            }
        }};
    }
    check!(FastPolynomialReducer);
    check!(FastTensorPolynomialReducer);
}

#[test]
fn profiling_keeps_owner_per_lane_rank_lowering() {
    use rusticol_loop_reduction::FastTensorPolynomialReducer;
    // Declared rank 6 = n+2 at n = 4; lanes with a = 0 have rank 5, which
    // NinjaRS reduces with its rank <= n+1 engine (FastNumerator::lane_rank,
    // lower_rank, restore_rank), bitwise as if rank 5 were declared. The
    // profiling wrapper must forward that, or profiled lanes keep the rank n+2
    // engine. The rank-6 part is homogeneous, so no coefficient is shared.
    let sixth = parse!("q0^2*q1^2*q2^2 + 3/7*q3^6 - 2*mu2*q0^3*q2 + q0*q1^4*q3");
    let fifth =
        parse!("b*((q0*q1 - q2*q3 + 1/2)*(q0^2 + q1*q3 - 3)*(q2 + 2) + mu2*(q0 - q3) - 5/4)");
    let shifts = [
        [0., 0., 0., 0.],
        [341. / 128., 86. / 128., -115. / 128., 83. / 128.],
        [180. / 128., 118. / 128., -44. / 128., 191. / 128.],
        [521. / 128., 104. / 128., 34. / 128., 257. / 128.],
    ]
    .repeat(3);
    let masses = [C64::real(1.); 12];
    let scales = [1.; 3];
    let c = |re: f64| Complex::new(re, 0.);
    // (a, b) per lane: two rank-5 lanes and one rank-6 lane.
    let full = [c(0.), c(1.), c(0.), c(-1.5), c(1.), c(1.)];
    let bits = |value: &Laurent<C64>| {
        [&value.finite, &value.single_pole, &value.double_pole]
            .map(|z| (z.re.to_bits(), z.im.to_bits()))
    };
    let run = |reducer: &mut FastTensorPolynomialReducer, lanes: usize, parameters: &[_]| {
        reducer
            .evaluate(
                &[7, 5, 3][..lanes],
                FastInput {
                    lanes,
                    shifts: &shifts[..4 * lanes],
                    masses_squared: &masses[..4 * lanes],
                    scales_squared: &scales[..lanes],
                    s_matrix: None,
                },
                parameters,
            )
            .unwrap()
            .into_iter()
            .map(|lane| {
                assert_eq!(lane.status, FastStatus::Success, "lane {}", lane.id);
                bits(lane.value.as_ref().unwrap())
            })
            .collect::<Vec<_>>()
    };
    let mut declared_six = FastTensorPolynomialReducer::new(
        4,
        parse!("a") * &sixth + &fifth,
        variables(),
        vec![parse!("a"), parse!("b")],
    )
    .unwrap();
    let plain = run(&mut declared_six, 3, &full);
    declared_six.set_profiling(true);
    let profiled = run(&mut declared_six, 3, &full);
    assert!(declared_six.take_profile().numerator_calls > 0);
    assert_eq!(profiled, plain);
    let mut declared_five =
        FastTensorPolynomialReducer::new(4, fifth, variables(), vec![parse!("b")]).unwrap();
    for profiling in [false, true] {
        declared_five.set_profiling(profiling);
        assert_eq!(run(&mut declared_five, 2, &[c(1.), c(-1.5)]), plain[..2]);
    }
}
