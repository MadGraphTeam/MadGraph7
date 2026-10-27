// SPDX-License-Identifier: 0BSD
//! The safe Rust SDK on a plan v4 artifact: model parameters at parity with
//! the Python API (second genericity audit, G2-M2). Set
//! PYAMPLICOL_V4_GGTT_ARTIFACT to a g g > t t~ plan v4 artifact generated
//! with WT fixed to zero (`tests/unit/test_loop_v2_sdk_parity.py`'s `ggtt`;
//! that test checks the same refusals and values through the C ABI, which
//! wraps this `Runtime`). This executable does not start or link Python.
use rusticol_loop_sdk::{Runtime, SourceOverrides};

/// A point exact in binary64 and on shell at MT = 173 (loop_sm's default).
fn point() -> Vec<f64> {
    let (e, y, z) = (180.91015625, 31.74609375, 42.328125);
    vec![e, 0., 0., e, e, 0., 0., -e, e, 0., y, z, e, 0., -y, -z]
}

fn overrides(values: &[(&str, &str)]) -> SourceOverrides {
    values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), [(*value).to_owned(), "0".to_owned()]))
        .collect()
}

fn values(runtime: &mut Runtime) -> (Vec<f64>, Vec<f64>) {
    let result = runtime
        .evaluate(&point(), &[0], &[1.0e6], false, None, None)
        .unwrap();
    (result.born, result.virtual_values)
}

#[test]
fn v4_parameter_updates_follow_the_python_api() {
    let Ok(path) = std::env::var("PYAMPLICOL_V4_GGTT_ARTIFACT") else {
        eprintln!("set PYAMPLICOL_V4_GGTT_ARTIFACT for the plan v4 parameter controls");
        return;
    };
    let refused = [
        (
            vec![("WT", "1.4915")],
            "parameter 'WT' was fixed to 0 at generation ([loop.preparation] \
             fixed_parameters); regenerate the artifact to change it",
        ),
        (
            vec![("MU_R", "100")],
            "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
        ),
        (vec![("WX", "1")], "unknown external parameter 'WX'"),
        (
            vec![("MT", "150"), ("WX", "1")],
            "unknown external parameter 'WX'",
        ),
    ];
    // Open with the default R2 strategy ("auto": the artifact's model R2).
    let mut runtime = Runtime::open(&path, Some("ggtt"), None, "auto").unwrap();
    assert!(Runtime::open(&path, Some("ggtt"), None, "direct").is_err());
    let before = values(&mut runtime);
    for (parameters, message) in &refused {
        let at_open = Runtime::open(&path, Some("ggtt"), Some(&overrides(parameters)), "model");
        assert_eq!(at_open.err().unwrap().message(), *message);
        let error = runtime.update_parameters(&overrides(parameters)).unwrap_err();
        assert_eq!(error.message(), *message);
        // A refused update changes nothing.
        assert_eq!(values(&mut runtime), before);
    }
    // The fixed WT at its fixed value and a parameter the process does not
    // read change nothing; aS does.
    runtime
        .update_parameters(&overrides(&[("WT", "0.0"), ("MH", "130")]))
        .unwrap();
    assert_eq!(values(&mut runtime), before);
    runtime.update_parameters(&overrides(&[("aS", "0.13")])).unwrap();
    assert_ne!(values(&mut runtime), before);
}
