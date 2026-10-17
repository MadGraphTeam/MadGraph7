// SPDX-License-Identifier: 0BSD

//! Connect the independently generated Community tensor oracle to NinjaRS.
//! This is an internal reduction fixture, not loop-induced production support.
use ninjars::{AmplitudeInput, Complex, DoubleFloat, EvaluationStatus, Float, FourMomentum, Real};
use oneloop::NativeFloat;
use rusticol_loop_reduction::{Laurent, PolynomialReducer};
use serde_json::Value;
use symbolica::{
    domains::{integer::Integer, rational::Rational},
    evaluate::EvaluationDomain,
    parse,
};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tools/loop_reference/diphoton_r2.json"
    ))
    .unwrap()
}

fn check<T: NativeFloat>(prototype: T, digits: u64)
where
    Complex<T>: Real + EvaluationDomain,
{
    let fixture = fixture();
    // Decode rational text with dependency-owned integers/rationals; no f64
    // intermediary enters shifts, masses, scales, coefficients or expectations.
    let exact = |text: &str| {
        let (numerator, denominator) = text.split_once('/').unwrap_or((text, "1"));
        let value = Rational::from((
            numerator.parse::<Integer>().unwrap(),
            denominator.parse::<Integer>().unwrap(),
        ));
        Complex::new(prototype.from_rational(&value), prototype.zero())
    };
    let variables = [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ];
    let mut reducers: Vec<_> = fixture["numerator_epsilon_sectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sector| {
            PolynomialReducer::new(
                3,
                parse!(sector.as_str().unwrap()),
                variables.clone(),
                vec![],
                prototype.clone(),
                None,
            )
            .unwrap()
        })
        .collect();
    assert_eq!(reducers.len(), 3);
    let expected = Laurent::from([
        exact(fixture["expected_laurent"]["finite"].as_str().unwrap()),
        exact(fixture["expected_laurent"]["single_pole"].as_str().unwrap()),
        exact(fixture["expected_laurent"]["double_pole"].as_str().unwrap()),
    ]);
    let one_shifts: Vec<_> = fixture["shifts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|shift| FourMomentum::from(std::array::from_fn(|i| exact(shift[i].as_str().unwrap()))))
        .collect();
    let one_masses: Vec<_> = fixture["mass_squared"]
        .as_array()
        .unwrap()
        .iter()
        .map(|mass| exact(mass.as_str().unwrap()))
        .collect();
    let tolerance = prototype.from_rational(&Rational::from((1, 10)).pow(digits));
    // Alternate sizes exercise reused storage, irregular tails and empty batches.
    for size in [1, 32, 8, 33, 128, 0, 1] {
        let ids: Vec<_> = (0..size).map(|row| 1000 + 7 * row as u64).collect();
        let shifts: Vec<_> = (0..size).flat_map(|_| one_shifts.clone()).collect();
        let masses: Vec<_> = (0..size).flat_map(|_| one_masses.clone()).collect();
        let scales = vec![exact("1"); size];
        let sectors: Vec<_> = reducers
            .iter_mut()
            .map(|reducer| {
                reducer
                    .evaluate(
                        &ids,
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            sampling_scales: &[],
                            s_matrix: None,
                        },
                        &[],
                    )
                    .unwrap()
            })
            .collect();
        for row in 0..size {
            for sector in &sectors {
                assert_eq!(sector[row].id, ids[row]);
                assert_eq!(sector[row].status, EvaluationStatus::Success);
            }
            let value = Laurent::epsilon_sectors(
                sectors[0][row].value.clone().unwrap(),
                sectors[1][row].value.clone().unwrap(),
                sectors[2][row].value.clone().unwrap(),
            );
            for (actual, expected) in [
                (&value.finite, &expected.finite),
                (&value.single_pole, &expected.single_pole),
                (&value.double_pole, &expected.double_pole),
            ] {
                let error = actual - expected;
                assert!(
                    error.re.norm() < tolerance && error.im.norm() < tolerance,
                    "batch {size}, lane {row}: actual={actual}, expected={expected}, error={error}"
                );
            }
        }
    }
}

#[test]
fn generated_evanescent_fixture_through_ninjars_all_precisions_and_batches() {
    check(0_f64, 10);
    check(DoubleFloat::default(), 27);
    check(Float::new(256), 60);
}
