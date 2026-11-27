// SPDX-License-Identifier: 0BSD
//! Loop support must not make optional exact tree payloads mandatory.
#![cfg(all(feature = "symbolica-runtime", feature = "f64-symjit"))]

use rusticol_core::{NativeRuntime, RusticolErrorKind};
use std::path::Path;

#[test]
fn binary64_tree_artifact_remains_usable_with_symbolica_enabled() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../src/pyamplicol/assets/selftest/portable-64le/artifact");
    let mut runtime = NativeRuntime::load(&root, Some("d_dbar_to_z"), None).unwrap();
    let point = [
        45.594, 0., 0., 45.594, 45.594, 0., 0., -45.594, 91.188, 0., 0., 0.,
    ];
    let first = runtime.evaluate_f64(&point, 1).unwrap();
    assert_eq!(first.len(), 1);
    assert!(first[0].is_finite() && first[0] > 0.);
    let exact: Vec<_> = point.iter().map(ToString::to_string).collect();
    let error = runtime.evaluate_with_precision(&exact, 1, 60).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::UnsupportedPrecision);
    assert_eq!(runtime.evaluate_f64(&point, 1).unwrap(), first);
}
