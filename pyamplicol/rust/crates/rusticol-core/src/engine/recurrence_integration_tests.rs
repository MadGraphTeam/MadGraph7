// SPDX-License-Identifier: 0BSD

//! Opt-in generated-artifact smoke tests for the recurrence execution lane.

use super::*;

fn assert_close(actual: f64, expected: f64, context: &str) {
    let tolerance = 1.0e-12 * actual.abs().max(expected.abs()).max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: {actual:.17e} != {expected:.17e}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn generated_recurrence_artifact_loads_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_RECURRENCE_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
        .expect("load generated recurrence artifact through NativeRuntime");
    assert_eq!(runtime.metadata().execution_mode, "recurrence");

    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value = serde_json::from_slice(
        &fs::read(&validation_path).expect("read recurrence validation momenta"),
    )
    .expect("parse recurrence validation momenta");
    let momenta = validation["points"][0]
        .as_array()
        .expect("one recurrence validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("four momentum components")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("decimal momentum string")
                        .parse::<f64>()
                        .expect("f64 validation momentum")
                })
        })
        .collect::<Vec<_>>();

    let values = runtime
        .evaluate_f64(&momenta, 1)
        .expect("evaluate generated recurrence artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
    let mut direct_values = [f64::NAN];
    runtime
        .evaluate_f64_into(&momenta, 1, &mut direct_values)
        .expect("evaluate generated recurrence artifact into caller storage");
    assert_close(
        direct_values[0],
        values[0],
        "recurrence direct-output total",
    );

    let resolved = runtime
        .evaluate_resolved_f64(&momenta, 1, None, None)
        .expect("resolve generated recurrence artifact");
    assert_eq!(resolved.point_count, 1);
    assert_close(resolved.totals()[0], values[0], "recurrence resolved sum");

    let selected_color = runtime
        .color_ids()
        .expect("recurrence color metadata")
        .into_iter()
        .next()
        .expect("one recurrence color component");
    let selected = runtime
        .evaluate_resolved_f64(
            &momenta,
            1,
            None,
            Some(std::slice::from_ref(&selected_color)),
        )
        .expect("select recurrence color component");
    assert_eq!(selected.color_ids, [selected_color]);
    assert!(selected.values.iter().all(|value| value.is_finite()));
    let selected_total = runtime
        .evaluate_f64_with_selectors(&momenta, 1, None, Some(&selected.color_ids), None, None)
        .expect("evaluate selected recurrence color component");
    assert_close(
        selected_total[0],
        selected.totals()[0],
        "recurrence selected color total",
    );
    let mut selected_direct = [f64::NAN];
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            1,
            None,
            Some(&selected.color_ids),
            None,
            None,
            &mut selected_direct,
        )
        .expect("evaluate selected recurrence component into caller storage");
    assert_close(
        selected_direct[0],
        selected_total[0],
        "recurrence selected direct-output total",
    );
}
