// SPDX-License-Identifier: 0BSD
//! Actual certified full/NLC artifacts through the standalone binary64 SDK.
use rusticol_loop_sdk::{ArithmeticSpec, PointStatus, Runtime, VerificationPolicy};
use std::path::Path;

#[test]
fn certified_weights_preserve_owned_outputs_and_attempt_diagnostics() {
    let Ok(root) = std::env::var("PYAMPLICOL_LOOP_FFT_FIXTURES") else {
        eprintln!("set PYAMPLICOL_LOOP_FFT_FIXTURES for retained physical controls");
        return;
    };
    let root = Path::new(&root);
    let point = [
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ];
    for accuracy in ["nlc", "full"] {
        let mut actual = Runtime::open(
            root.join(accuracy).join("fft-artifact"),
            Some("gg"),
            None,
            "model",
        )
        .unwrap();
        let mut expected = Runtime::open(
            root.join(accuracy).join("direct-artifact"),
            Some("gg"),
            None,
            "model",
        )
        .unwrap();
        let capabilities = actual.capabilities_json().unwrap();
        assert!(capabilities.contains("bare-loop dual Born weights"));
        assert!(capabilities.contains("\"algorithm\":\"symmetric-group-fft\""));
        for resolved in [false, true] {
            let policy = VerificationPolicy {
                higher: vec![ArithmeticSpec {
                    arithmetic: "double-double".into(),
                    precision: None,
                }],
                relative_tolerance: "1e-8".into(),
                absolute_tolerance: "1e-9".into(),
                verify_successful: true,
            };
            let left = actual
                .evaluate(&point, &[u64::MAX], &[1e6], resolved, None, Some(&policy))
                .unwrap();
            let right = expected
                .evaluate(&point, &[u64::MAX], &[1e6], resolved, None, None)
                .unwrap();
            assert_eq!(left.successful_point_ids, [u64::MAX]);
            assert_eq!(left.statuses, [PointStatus::Verified]);
            assert_eq!(left.shape, right.shape);
            assert_eq!(left.born.len(), right.born.len());
            assert_eq!(left.virtual_values.len(), right.virtual_values.len());
            for (a, b) in left
                .born
                .iter()
                .chain(&left.virtual_values)
                .zip(right.born.iter().chain(&right.virtual_values))
            {
                assert!((a - b).abs() <= 3e-9 * a.abs().max(b.abs()) + 1e-9);
            }
            assert!(left.diagnostics_json.contains("primary_color_contraction"));
            assert!(
                left.diagnostics_json
                    .contains("\"algorithm\":\"symmetric-group-fft\"")
            );
            assert!(
                left.diagnostics_json
                    .contains("exact higher-precision direct contraction")
            );
            actual.clear();
            assert!(!left.virtual_values.is_empty());
            assert!(
                actual
                    .evaluate(&[], &[], &[], resolved, None, None)
                    .unwrap()
                    .statuses
                    .is_empty()
            );
        }
    }
}
