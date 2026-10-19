// SPDX-License-Identifier: 0BSD
// rustc --edition=2021 client.rs -l static=rusticol_capi -L /path/to/sdk/lib ...
// Run with the retained mixed-order source artifact; no Python runtime is used.
#[path = "../../../../src/pyamplicol/_sdk/rust/rusticol.rs"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::{PointStatus, Runtime};

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1e-20),
        "{a} != {b}"
    );
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact = std::env::args()
        .nth(1)
        .expect("mixed-order source artifact");
    let mut runtime = Runtime::open(artifact, Some("uudd"), None, Some("model"))?;
    let axes = runtime.shape();
    assert_eq!(
        (axes.external_count, axes.born_orders, axes.virtual_orders),
        (4, 3, 3)
    );
    assert_eq!((axes.epsilon_count, axes.resolved_available), (3, 1));
    assert!(runtime
        .metadata_json()?
        .contains("\"epsilon_powers\":[0,-1,-2]"));
    assert!(!runtime.capabilities_json()?.is_empty());
    let point = [
        500., 0., 0., 500., 500., 0., 0., -500., 500., 300., 0., 400., 500., -300., 0., -400.,
    ];
    let mut momenta = point.repeat(3);
    let ids = [91, 4, 70];
    let inclusive =
        runtime.evaluate(&momenta, 3, Some(&ids), &[1e6, 9e5, 1e6], false, None, None)?;
    let resolved = runtime.evaluate(&momenta, 3, Some(&ids), &[1e6, 9e5, 1e6], true, None, None)?;
    assert_eq!(inclusive.successful_point_ids, ids);
    assert_eq!(resolved.requested_point_ids, ids);
    assert_eq!(inclusive.statuses, [PointStatus::Success; 3]);
    for p in 0..3 {
        for o in 0..axes.born_orders {
            let n = axes.helicities * axes.born_color_components;
            close(
                resolved.born[(p * axes.born_orders + o) * n..(p * axes.born_orders + o + 1) * n]
                    .iter()
                    .sum(),
                inclusive.born[p * axes.born_orders + o],
            );
        }
        for o in 0..axes.virtual_orders {
            for e in 0..3 {
                let n = axes.helicities * axes.virtual_color_components;
                close(
                    (0..n)
                        .map(|hc| {
                            resolved.virtual_values
                                [((p * axes.virtual_orders + o) * n + hc) * 3 + e]
                        })
                        .sum(),
                    inclusive.virtual_values[(p * axes.virtual_orders + o) * 3 + e],
                );
            }
        }
    }
    // Independent MadLoop full-CKM fixture in the saved catalogue order.
    let born = [-2.0322878122576093, 0.8013367806451908, 3.0512301482522957];
    let poles = [
        [-3.018755956775497, -1.1476469718544264, -0.3056158216076459],
        [1.313041990604555, 0.7811563156803705, 0.20355701776284274],
        [
            -0.19094989612756078,
            -0.28421984016919477,
            -0.08026310265110026,
        ],
    ];
    for o in 0..3 {
        close(inclusive.born[o], born[o]);
        for e in 0..3 {
            close(inclusive.virtual_values[o * 3 + e], poles[o][e]);
        }
    }
    for i in 0..16 {
        momenta[16 + i] = point[i] * 1e200;
    }
    let partial = runtime.evaluate(&momenta, 3, Some(&ids), &[1e6], true, None, None)?;
    assert_eq!(partial.successful_point_ids, [91, 70]);
    assert_eq!(
        partial.statuses,
        [
            PointStatus::Success,
            PointStatus::Failed,
            PointStatus::Success
        ]
    );
    assert_eq!(partial.shape.successful_points, 2);
    assert!(!partial.diagnostics_json.is_empty());
    let policy = r#"{"higher":[{"arithmetic":"double-double","precision":null}],"relative_tolerance":"1e-8","absolute_tolerance":"1e-9","verify_successful":true}"#;
    let verified = runtime.evaluate(&point, 1, Some(&[91]), &[1e6], true, None, Some(policy))?;
    assert_eq!(verified.statuses, [PointStatus::Verified]);
    let empty = runtime.evaluate(&[], 0, None, &[], true, None, None)?;
    assert!(empty.born.is_empty() && empty.virtual_values.is_empty() && empty.statuses.is_empty());
    assert!(runtime
        .evaluate(&point, 1, Some(&ids), &[1e6], false, None, None)
        .is_err());
    let error = runtime
        .evaluate(&point, 1, None, &[0.], false, None, None)
        .err()
        .unwrap();
    assert!(!error.to_string().is_empty());
    assert!(runtime.update_parameters_json(r#"{"aS":["0.12","0"]}"#)? > 0);
    runtime.clear()?;
    drop(runtime);
    assert_eq!(inclusive.born.len(), 9); // owned values survive runtime destruction
    println!(
        "native safe Rust mixed-order/resolved/failures/verification/lifetime controls passed"
    );
    Ok(())
}
