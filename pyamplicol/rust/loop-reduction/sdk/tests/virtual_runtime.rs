// SPDX-License-Identifier: 0BSD
//! Native saved-source controls; set PYAMPLICOL_VIRTUAL_ARTIFACT to a resolved
//! gg->gg source artifact. This executable does not start or link Python.
use rusticol_loop_sdk::{
    ArithmeticSpec, PointStatus, Runtime, SourceOverrides, VerificationPolicy,
};
use std::f64::consts::PI;

fn point() -> Vec<f64> {
    vec![
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ]
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1e-20),
        "{a} != {b}"
    );
}

#[test]
fn saved_source_orders_resolved_updates_and_explicit_rescue() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_ARTIFACT for native source execution");
        return;
    };
    check_saved_gluon_source(&path, false);
}

#[test]
fn lc_source_anchors_preserve_statuses_and_subset_rescue() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_LC_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_LC_ARTIFACT for signed source anchors");
        return;
    };
    check_saved_gluon_source(&path, true);
}

#[test]
fn opaque_ids_preserve_sdk_resolved_partial_axes() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_LC_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_LC_ARTIFACT for signed source anchors");
        return;
    };
    let mut runtime = Runtime::open(&path, Some("gg"), None, "model").unwrap();
    let shape = runtime.shape();
    assert!(shape.virtual_color_components > 1);
    assert_eq!((shape.born_orders, shape.virtual_orders), (1, 1));
    let ids = vec![u64::MAX, u64::MAX - 1, 0];
    let exact: Vec<_> = [point(), point(), point()]
        .concat()
        .iter()
        .map(ToString::to_string)
        .collect();
    let scales = vec!["1000000".into(), "1e-400".into(), "1000000".into()];
    let inclusive = runtime
        .evaluate_decimal(
            exact.clone(),
            ids.clone(),
            scales.clone(),
            false,
            None,
            None,
        )
        .unwrap();
    let resolved = runtime
        .evaluate_decimal(exact, ids.clone(), scales, true, None, None)
        .unwrap();
    assert_eq!(resolved.requested_point_ids, ids);
    assert_eq!(resolved.successful_point_ids, [u64::MAX, 0]);
    assert_eq!(
        resolved.statuses,
        [
            PointStatus::Success,
            PointStatus::Failed,
            PointStatus::Success
        ]
    );
    let rows = shape.helicities * shape.virtual_color_components;
    assert_eq!(resolved.virtual_values.len(), 2 * rows * 3);
    for point in 0..2 {
        for pole in 0..3 {
            close(
                (0..rows)
                    .map(|row| resolved.virtual_values[(point * rows + row) * 3 + pole])
                    .sum(),
                inclusive.virtual_values[point * 3 + pole],
            );
        }
    }
    let diagnostics: serde_json::Value = serde_json::from_str(&resolved.diagnostics_json).unwrap();
    let failures = diagnostics["failures"].as_array().unwrap();
    assert!(!failures.is_empty());
    assert!(
        failures
            .iter()
            .all(|f| f["point_id"].as_u64() == Some(u64::MAX - 1))
    );
}

fn check_saved_gluon_source(path: &str, lc: bool) {
    let mut runtime = Runtime::open(path, Some("gg"), None, "model").unwrap();
    let shape = runtime.shape();
    assert_eq!(
        (
            shape.external_count,
            shape.born_orders,
            shape.virtual_orders,
            shape.helicities,
            shape.epsilon_count,
            shape.resolved_available
        ),
        (4, 1, 1, 16, 3, 1)
    );
    let metadata: serde_json::Value =
        serde_json::from_str(&runtime.metadata_json().unwrap()).unwrap();
    assert_eq!(
        metadata["physics"]["orders"]["virtual"][0]["order"][0][1],
        6
    );
    let inclusive = runtime
        .evaluate(&point(), &[u64::MAX], &[1e6], false, None, None)
        .unwrap();
    let alpha = 0.118_f64;
    let s = 1e6_f64;
    let t = -1e5_f64;
    let u = -9e5_f64;
    let born = (if lc { 9. / 8. } else { 1. }) * 9. / 4.
        * (4. * PI * alpha).powi(2)
        * (3. - u * t / (s * s) - u * s / (t * t) - s * t / (u * u));
    close(inclusive.born[0], born);
    close(inclusive.virtual_values[2], -12. * alpha / (2. * PI) * born);
    assert_eq!(inclusive.statuses, [PointStatus::Success]);
    let components = shape.virtual_color_components;
    assert_eq!(shape.born_color_components, components);
    assert!(if lc { components > 1 } else { components == 1 });
    let capabilities: serde_json::Value =
        serde_json::from_str(&runtime.capabilities_json().unwrap()).unwrap();
    assert_eq!(
        capabilities["color_accuracy"],
        if lc { "lc" } else { "full" }
    );
    assert_eq!(
        capabilities["color_components"].as_array().unwrap().len(),
        components
    );
    let resolved = runtime
        .evaluate(&point(), &[u64::MAX], &[1e6], true, None, None)
        .unwrap();
    assert_eq!(resolved.successful_point_ids, [u64::MAX]);
    close(resolved.born.iter().sum(), inclusive.born[0]);
    for pole in 0..3 {
        close(
            (0..16 * components)
                .map(|r| resolved.virtual_values[r * 3 + pole])
                .sum(),
            inclusive.virtual_values[pole],
        );
    }
    let momenta = [point(), point(), point()].concat();
    let exact: Vec<_> = momenta.iter().map(ToString::to_string).collect();
    let scales = vec!["1000000".into(), "1e-400".into(), "1000000".into()];
    let partial = runtime
        .evaluate_decimal(
            exact.clone(),
            vec![u64::MAX, u64::MAX - 1, 0],
            scales.clone(),
            true,
            None,
            None,
        )
        .unwrap();
    assert_eq!(partial.requested_point_ids, [u64::MAX, u64::MAX - 1, 0]);
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
        .evaluate_decimal(
            exact,
            vec![u64::MAX, u64::MAX - 1, 0],
            scales,
            true,
            None,
            Some(&policy),
        )
        .unwrap();
    assert_eq!(rescued.successful_point_ids, [u64::MAX, u64::MAX - 1, 0]);
    assert_eq!(
        rescued.statuses,
        [
            PointStatus::Success,
            PointStatus::Rescued,
            PointStatus::Success
        ]
    );
    let history: serde_json::Value = serde_json::from_str(&rescued.diagnostics_json).unwrap();
    assert_eq!(
        history["verification"][1]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        history["verification"][1]["attempts"][2]["conversion_agreement"],
        true
    );
    let all_failed = runtime
        .evaluate_decimal(
            point().iter().map(ToString::to_string).collect(),
            vec![70],
            vec!["1e-400".into()],
            false,
            None,
            None,
        )
        .unwrap();
    assert!(all_failed.born.is_empty() && all_failed.virtual_values.is_empty());
    assert_eq!(all_failed.statuses, [PointStatus::Failed]);
    for value in ["0.12", "0.118"] {
        runtime
            .update_parameters(&SourceOverrides::from([(
                "aS".into(),
                [value.into(), "0".into()],
            )]))
            .unwrap();
        let updated = runtime
            .evaluate(&point(), &[4], &[1e6], false, None, None)
            .unwrap();
        close(
            updated.born[0],
            born * (value.parse::<f64>().unwrap() / alpha).powi(2),
        );
    }
    runtime.clear();
    close(
        runtime
            .evaluate(&point(), &[70], &[1e6], false, None, None)
            .unwrap()
            .virtual_values[0],
        inclusive.virtual_values[0],
    );
    assert!(
        runtime
            .evaluate(&[], &[], &[], true, None, None)
            .unwrap()
            .born
            .is_empty()
    );
    assert!(
        runtime
            .evaluate(&[], &[], &[], false, Some("direct"), None)
            .is_err()
    );
    assert!(
        runtime
            .evaluate(&point(), &[u64::MAX], &[0.], false, None, None)
            .is_err()
    );
    assert!(Runtime::open(&path, Some("absent"), None, "model").is_err());
}

#[test]
fn mixed_source_orders_keep_resolved_axes_and_match_provider() {
    let Ok(path) = std::env::var("PYAMPLICOL_VIRTUAL_MIXED_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_VIRTUAL_MIXED_ARTIFACT for the full-CKM source control");
        return;
    };
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tools/loop_reference/uudd_mixed_sm_qcd.json"
    ))
    .unwrap();
    let mut runtime = Runtime::open(&path, Some("uudd"), None, "model").unwrap();
    let shape = runtime.shape();
    assert_eq!(
        (shape.born_orders, shape.virtual_orders, shape.helicities),
        (3, 3, 16)
    );
    let metadata: serde_json::Value =
        serde_json::from_str(&runtime.metadata_json().unwrap()).unwrap();
    let points: Vec<String> = reference["points"][0]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|leg| leg.as_array().unwrap())
        .map(|x| {
            x.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| x.to_string())
        })
        .collect();
    let scales = vec![reference["mu_r2"].as_str().unwrap().to_owned()];
    let inclusive = runtime
        .evaluate_decimal(points.clone(), vec![91], scales.clone(), false, None, None)
        .unwrap();
    let resolved = runtime
        .evaluate_decimal(points, vec![91], scales, true, None, None)
        .unwrap();
    assert_eq!(resolved.successful_point_ids, [91]);
    for (kind, width, reference_key, inclusive_values, resolved_values) in [
        (
            "born",
            1,
            "born_squared_orders",
            &inclusive.born,
            &resolved.born,
        ),
        (
            "virtual",
            3,
            "virtual_squared_orders",
            &inclusive.virtual_values,
            &resolved.virtual_values,
        ),
    ] {
        for (order, sector) in metadata["physics"]["orders"][kind]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let order_map: std::collections::BTreeMap<String, u64> = sector["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| (x[0].as_str().unwrap().to_owned(), x[1].as_u64().unwrap()))
                .collect();
            let key = reference[reference_key]
                .as_object()
                .unwrap()
                .iter()
                .find(|(_, value)| {
                    value
                        .as_object()
                        .unwrap()
                        .iter()
                        .filter(|(_, v)| v.as_u64() != Some(0))
                        .map(|(k, v)| (k.clone(), v.as_u64().unwrap()))
                        .collect::<std::collections::BTreeMap<_, _>>()
                        == order_map
                })
                .unwrap()
                .0;
            for pole in 0..width {
                let expected = if kind == "born" {
                    reference["rows"][0]["born"][key].as_f64().unwrap()
                } else {
                    reference["rows"][0]["full"][key]
                        [["finite", "single_pole", "double_pole"][pole]]
                        .as_f64()
                        .unwrap()
                };
                close(inclusive_values[order * width + pole], expected);
                close(
                    (0..16)
                        .map(|h| resolved_values[(order * 16 + h) * width + pole])
                        .sum(),
                    inclusive_values[order * width + pole],
                );
            }
        }
    }
}
