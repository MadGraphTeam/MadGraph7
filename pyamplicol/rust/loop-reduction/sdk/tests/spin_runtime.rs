// SPDX-License-Identifier: 0BSD
//! Adapter controls on a saved source-owned u ubar -> g g artifact; no generation.
use rusticol_loop_sdk::{
    ArithmeticSpec, Complex64, PointStatus, Runtime, SpinSources, VerificationPolicy,
};
use std::collections::BTreeMap;

fn point() -> Vec<f64> {
    vec![
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ]
}
fn vector() -> Vec<Complex64> {
    vec![
        Complex64::default(),
        Complex64 {
            real: 0.8,
            imag: 0.,
        },
        Complex64 { real: 0., imag: 1. },
        Complex64 {
            real: -0.6,
            imag: 0.,
        },
    ]
}
fn scaled(values: &[Complex64], re: f64, im: f64) -> Vec<Complex64> {
    values
        .iter()
        .map(|v| Complex64 {
            real: re * v.real - im * v.imag,
            imag: re * v.imag + im * v.real,
        })
        .collect()
}
fn close(a: Complex64, b: Complex64) {
    let scale = a
        .real
        .abs()
        .max(a.imag.abs())
        .max(b.real.abs())
        .max(b.imag.abs())
        .max(1.);
    assert!(
        (a.real - b.real).abs() < 2e-8 * scale && (a.imag - b.imag).abs() < 2e-8 * scale,
        "{a:?} != {b:?}"
    );
}
#[test]
fn saved_joint_sources_literal_values_replay_and_ownership() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_SPIN_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_SPIN_ARTIFACT for saved source execution");
        return;
    };
    let mut runtime = Runtime::open(path, None, None, "model").unwrap();
    assert_eq!(
        runtime.spin_correlation_classes(),
        [vec![3], vec![4], vec![3, 4]]
    );
    assert_eq!(
        runtime.correlation_history_ids(),
        ["identity", "ordered", "swapped"]
    );
    let p = point();
    let v = vector();
    let legs = [3];
    let baseline = runtime
        .evaluate_correlated_with_sources(
            &p,
            &[u64::MAX],
            &[1e6],
            SpinSources {
                legs: &legs,
                values: &v,
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(baseline.spin_legs, [3]);
    assert!(
        baseline.virtual_values[3..6]
            .iter()
            .any(|x| x.imag.abs() > 1e-8)
    );
    for e in 0..3 {
        let conjugate = Complex64 {
            real: baseline.virtual_values[6 + e].real,
            imag: -baseline.virtual_values[6 + e].imag,
        };
        close(baseline.virtual_values[3 + e], conjugate);
    }
    let ids = [u64::MAX, 9, 0];
    let points = [p.clone(), p.clone(), p.clone()].concat();
    let rows = [v.clone(), scaled(&v, 2., -3.), v.clone()].concat();
    let batch = runtime
        .evaluate_correlated_with_sources(
            &points,
            &ids,
            &[1e6],
            SpinSources {
                legs: &legs,
                values: &rows,
            },
            None,
            None,
        )
        .unwrap();
    let broadcast = runtime
        .evaluate_correlated_with_sources(
            &points,
            &ids,
            &[1e6],
            SpinSources {
                legs: &legs,
                values: &v,
            },
            None,
            None,
        )
        .unwrap();
    for row in 0..3 {
        for k in 0..9 {
            close(
                batch.virtual_values[row * 9 + k],
                scaled(
                    &baseline.virtual_values,
                    if row == 1 { 13. } else { 1. },
                    0.,
                )[k],
            );
            close(
                broadcast.virtual_values[row * 9 + k],
                baseline.virtual_values[k],
            );
        }
    }
    let joint = [v.clone(), v.clone()].concat();
    let both = runtime
        .evaluate_correlated_with_sources(
            &p,
            &[9],
            &[1e6],
            SpinSources {
                legs: &[3, 4],
                values: &joint,
            },
            None,
            None,
        )
        .unwrap();
    let scaled_joint = [scaled(&v, 2., -3.), scaled(&v, 1., 2.)].concat();
    let other = runtime
        .evaluate_correlated_with_sources(
            &p,
            &[9],
            &[1e6],
            SpinSources {
                legs: &[3, 4],
                values: &scaled_joint,
            },
            None,
            None,
        )
        .unwrap();
    for (&a, &b) in both.virtual_values.iter().zip(&other.virtual_values) {
        close(b, scaled(&[a], 65., 0.)[0]);
    }
    for (bad_legs, bad_values) in [
        (&[0][..], v.clone()),
        (&[3, 3][..], joint.clone()),
        (&[1][..], v.clone()),
        (&[3][..], v[..3].to_vec()),
        (&[3][..], scaled(&v, f64::NAN, 0.)),
    ] {
        assert!(
            runtime
                .evaluate_correlated_with_sources(
                    &p,
                    &[1],
                    &[1e6],
                    SpinSources {
                        legs: bad_legs,
                        values: &bad_values
                    },
                    None,
                    None
                )
                .is_err()
        );
    }
    let empty = runtime
        .evaluate_correlated_with_sources(
            &[],
            &[],
            &[],
            SpinSources {
                legs: &[3, 4],
                values: &[],
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(empty.spin_legs, [3, 4]);
    assert!(empty.born.is_empty());
    let exact_sources: BTreeMap<_, _> = [(
        3,
        rows.iter()
            .map(|v| [v.real.to_string(), v.imag.to_string()])
            .collect(),
    )]
    .into();
    let decimal = points.iter().map(ToString::to_string).collect::<Vec<_>>();
    let scales = vec!["1000000".into(), "1e-400".into(), "1000000".into()];
    let partial = runtime
        .evaluate_correlated_decimal_with_sources(
            decimal.clone(),
            ids.to_vec(),
            scales.clone(),
            exact_sources.clone(),
            None,
            None,
        )
        .unwrap();
    assert_eq!(partial.successful_point_ids, [u64::MAX, 0]);
    assert_eq!(
        partial.statuses,
        [
            PointStatus::Success,
            PointStatus::Failed,
            PointStatus::Success
        ]
    );
    let policy = VerificationPolicy {
        higher: vec![
            ArithmeticSpec {
                arithmetic: "arbitrary".into(),
                precision: Some(60),
            },
            ArithmeticSpec {
                arithmetic: "arbitrary".into(),
                precision: Some(80),
            },
        ],
        relative_tolerance: "1e-8".into(),
        absolute_tolerance: "1e-9".into(),
        verify_successful: false,
    };
    let rescued = runtime
        .evaluate_correlated_decimal_with_sources(
            decimal,
            ids.to_vec(),
            scales,
            exact_sources,
            None,
            Some(&policy),
        )
        .unwrap();
    assert_eq!(rescued.successful_point_ids, ids);
    assert_eq!(
        rescued.statuses,
        [
            PointStatus::Success,
            PointStatus::Rescued,
            PointStatus::Success
        ]
    );
    assert_eq!(rescued.spin_legs, [3]);
    for h in 0..3 {
        close(
            rescued.virtual_values[9 + h * 3 + 2],
            batch.virtual_values[9 + h * 3 + 2],
        );
    }
    runtime.clear();
    let ordinary = runtime
        .evaluate_correlated(&p, &[0], &[1e6], None, None)
        .unwrap();
    assert!(ordinary.spin_legs.is_empty());
    let repeated = runtime
        .evaluate_correlated_with_sources(
            &p,
            &[0],
            &[1e6],
            SpinSources {
                legs: &legs,
                values: &v,
            },
            None,
            None,
        )
        .unwrap();
    for (a, b) in repeated.virtual_values.iter().zip(&baseline.virtual_values) {
        close(*a, *b);
    }
    drop(runtime);
    drop(rows);
    drop(v);
    assert_eq!(baseline.successful_point_ids, [u64::MAX]);
    assert_eq!(baseline.spin_legs_json().unwrap(), "[3]");
    assert!(baseline.born[0].real > 0.);
}
