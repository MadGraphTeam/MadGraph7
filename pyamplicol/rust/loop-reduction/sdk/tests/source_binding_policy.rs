// SPDX-License-Identifier: 0BSD
//! Guard-only saved controls leave every physical instruction unchanged.
use rusticol_loop_sdk::{
    ArithmeticSpec, PointStatus, Runtime, SourceOverrides, VerificationPolicy,
};
use std::path::Path;

#[test]
fn retained_policy_precedes_binding_and_returns_owned_rescue_diagnostics() {
    let Ok(root) = std::env::var("PYAMPLICOL_SOURCE_BINDING_RESCUE_FIXTURES") else {
        eprintln!("set PYAMPLICOL_SOURCE_BINDING_RESCUE_FIXTURES for saved guard controls");
        return;
    };
    let root = Path::new(&root);
    let path = root.join("resolved-artifact");
    assert!(Runtime::open(&path, Some("gg"), None, "model").is_err());
    let policy = VerificationPolicy {
        higher: [60, 80]
            .map(|precision| ArithmeticSpec {
                arithmetic: "arbitrary".into(),
                precision: Some(precision),
            })
            .to_vec(),
        relative_tolerance: "1e-8".into(),
        absolute_tolerance: "1e-9".into(),
        verify_successful: false,
    };
    let parameters = SourceOverrides::from([(
        "aS".into(),
        [
            "0.11800000000000000000000000000000000001".into(),
            "0".into(),
        ],
    )]);
    let mut runtime =
        Runtime::open_with_policy(&path, Some("gg"), Some(&parameters), "model", Some(&policy))
            .unwrap();
    assert_eq!(runtime.verification_policy().unwrap().higher.len(), 2);
    let capabilities = runtime.capabilities_json().unwrap();
    assert!(capabilities.contains("retained_verification_policy"));
    let mut baseline = Runtime::open(
        root.parent()
            .unwrap()
            .join("source-colour-components/public-artifact"),
        Some("gg"),
        Some(&parameters),
        "model",
    )
    .unwrap();
    let point = [
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ];
    let result = runtime
        .evaluate(&point, &[u64::MAX], &[1e6], false, None, None)
        .unwrap();
    let reference = baseline
        .evaluate(&point, &[u64::MAX], &[1e6], false, None, None)
        .unwrap();
    assert_eq!(result.requested_point_ids, [u64::MAX]);
    assert_eq!(result.successful_point_ids, [u64::MAX]);
    assert_eq!(result.statuses, [PointStatus::Rescued]);
    for (actual, expected) in result
        .born
        .iter()
        .chain(&result.virtual_values)
        .zip(reference.born.iter().chain(&reference.virtual_values))
    {
        assert!((actual - expected).abs() <= 2e-8 * actual.abs().max(expected.abs()) + 1e-9);
    }
    assert!(result.diagnostics_json.contains("real model input type"));
    assert!(
        result
            .diagnostics_json
            .contains("\"conversion_agreement\":true")
    );
    let invalid = SourceOverrides::from([("MT".into(), ["173".into(), "1e-1000".into()])]);
    assert!(runtime.update_parameters(&invalid).is_err());
    assert_eq!(
        runtime
            .update_parameters(&SourceOverrides::from([(
                "aS".into(),
                ["0.12".into(), "0".into()]
            ),]))
            .unwrap(),
        1
    );
    runtime.clear();
    assert!(
        runtime
            .evaluate(&[], &[], &[], true, None, None)
            .unwrap()
            .statuses
            .is_empty()
    );
    drop(runtime);
    assert_eq!(result.virtual_values.len(), 3);
    assert_eq!(result.statuses, [PointStatus::Rescued]);
}
