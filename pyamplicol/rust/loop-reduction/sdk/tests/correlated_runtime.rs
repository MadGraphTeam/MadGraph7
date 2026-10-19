// SPDX-License-Identifier: 0BSD
//! Saved source integration through the ordinary standalone owner and ABI.
use rusticol_loop_sdk::{ArithmeticSpec, Complex64, PointStatus, Runtime, VerificationPolicy};
fn point() -> Vec<f64> {
    vec![
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ]
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
        (a.real - b.real).abs() <= 1e-8 * scale && (a.imag - b.imag).abs() <= 1e-8 * scale,
        "{a:?} != {b:?}"
    );
}
#[test]
fn source_grouped_complex_orders_ids_partial_rescue_and_owned_results() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_DIRECTED_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_DIRECTED_ARTIFACT for actual source SDK execution");
        return;
    };
    let mut runtime = Runtime::open(path, None, None, "model").unwrap();
    assert_eq!(
        runtime.correlation_history_ids(),
        ["identity", "casimir", "ordered", "swapped"]
    );
    let shape = runtime.correlated_shape();
    assert_eq!(
        (
            shape.histories,
            shape.born_orders,
            shape.virtual_orders,
            shape.epsilon_count
        ),
        (4, 1, 1, 3)
    );
    let ordinary = runtime
        .evaluate(&point(), &[u64::MAX], &[1e6], false, None, None)
        .unwrap();
    let result = runtime
        .evaluate_correlated(&point(), &[u64::MAX], &[1e6], None, None)
        .unwrap();
    assert_eq!(result.successful_point_ids, [u64::MAX]);
    close(
        result.born[0],
        Complex64 {
            real: ordinary.born[0],
            imag: 0.,
        },
    );
    assert!(
        result.virtual_values[6..9]
            .iter()
            .any(|x| x.imag.abs() > 1e-8)
    );
    for e in 0..3 {
        close(
            result.virtual_values[e],
            Complex64 {
                real: ordinary.virtual_values[e],
                imag: 0.,
            },
        );
        close(
            result.virtual_values[3 + e],
            Complex64 {
                real: 4. / 3. * ordinary.virtual_values[e],
                imag: 0.,
            },
        );
        close(
            result.virtual_values[6 + e],
            Complex64 {
                real: result.virtual_values[9 + e].real,
                imag: -result.virtual_values[9 + e].imag,
            },
        );
    }
    let ids = vec![u64::MAX, 9, 0];
    let momenta: Vec<_> = [point(), point(), point()]
        .concat()
        .iter()
        .map(ToString::to_string)
        .collect();
    let scales = vec!["1000000".into(), "1e-400".into(), "1000000".into()];
    let partial = runtime
        .evaluate_correlated_decimal(momenta.clone(), ids.clone(), scales.clone(), None, None)
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
    assert_eq!(partial.born.len(), 8);
    assert_eq!(partial.virtual_values.len(), 24);
    let failed = runtime
        .evaluate_correlated_decimal(
            point().iter().map(ToString::to_string).collect(),
            vec![9],
            vec!["1e-400".into()],
            None,
            None,
        )
        .unwrap();
    assert!(failed.born.is_empty() && failed.virtual_values.is_empty());
    assert_eq!(failed.statuses, [PointStatus::Failed]);
    let empty = runtime
        .evaluate_correlated(&[], &[], &[], None, None)
        .unwrap();
    assert!(empty.born.is_empty());
    assert_eq!(empty.history_ids, result.history_ids);
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
        .evaluate_correlated_decimal(momenta, ids.clone(), scales, None, Some(&policy))
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
    let diagnostics: serde_json::Value = serde_json::from_str(&rescued.diagnostics_json).unwrap();
    assert_eq!(
        diagnostics["verification"][1]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    // At the changed scale the finite/single pole change, but the leading pole
    // and Born values are scale independent. Every complex coefficient remains finite.
    for h in 0..4 {
        close(rescued.born[4 + h], result.born[h]);
        close(
            rescued.virtual_values[12 + h * 3 + 2],
            result.virtual_values[h * 3 + 2],
        );
    }
    assert!(
        rescued
            .virtual_values
            .iter()
            .all(|x| x.real.is_finite() && x.imag.is_finite())
    );
    runtime.clear();
    drop(runtime);
    assert_eq!(
        result.history_ids,
        ["identity", "casimir", "ordered", "swapped"]
    );
    close(
        result.born[0],
        Complex64 {
            real: ordinary.born[0],
            imag: 0.,
        },
    );
}
