// SPDX-License-Identifier: 0BSD
//! A particle mass that is a derived model parameter follows the external inputs after
//! `set_model_parameters`. In a G_mu-scheme UFO, MW is derived from aEWM1, Gf and MZ; the W
//! wavefunctions and propagators must use the refreshed MW, not the value derived from the artifact
//! defaults. Then |M|² of u d~ > W+ H, with the W on the refreshed mass shell, is invariant under a
//! boost along z; with a stale MW the polarisation sum is taken on another mass shell and the
//! result changes with the frame (by about 5e-5 at the point below, for MW 80.369 against 80.419).
//!
//! Supply the artifact with
//! `pyamplicol generate 'u d~ > W+ H' DIR --model models/json/sm/sm.json --restriction no_b_mass
//! --execution-mode compiled` (QED = 2) and `RUSTICOL_DERIVED_MASS_ARTIFACT=DIR`.

use rusticol_core::NativeRuntime;
use std::{collections::BTreeMap, path::PathBuf};

/// u(p1) d~(p2) > W+(p3) H(p4) at sqrt(s) = 500 GeV, cos(theta) = 3/10, in a frame boosted along z
/// with rapidity `y` (exact on-shell light-cone construction in binary64).
fn momenta(mw: f64, mh: f64, y: f64) -> Vec<f64> {
    let root_s = 500.0_f64;
    let s = root_s * root_s;
    let e3 = (s + mw * mw - mh * mh) / (2.0 * root_s);
    let p = (e3 * e3 - mw * mw).sqrt();
    let (cos, sin) = (0.3_f64, (1.0_f64 - 0.09).sqrt());
    let half = root_s / 2.0;
    let rest = [
        [half, 0.0, 0.0, half],
        [half, 0.0, 0.0, -half],
        [e3, p * sin, 0.0, p * cos],
        [root_s - e3, -p * sin, 0.0, -p * cos],
    ];
    let masses = [0.0, 0.0, mw, mh];
    rest.iter()
        .zip(masses)
        .flat_map(|(q, m)| {
            let m_t2 = m * m + q[1] * q[1] + q[2] * q[2];
            let (plus, minus) = if q[3] >= 0.0 {
                (q[0] + q[3], m_t2 / (q[0] + q[3]))
            } else {
                (m_t2 / (q[0] - q[3]), q[0] - q[3])
            };
            let (plus, minus) = (plus * y.exp(), minus * (-y).exp());
            [(plus + minus) / 2.0, q[1], q[2], (plus - minus) / 2.0]
        })
        .collect()
}

#[test]
fn derived_particle_mass_follows_the_external_inputs() {
    let Some(path) = std::env::var_os("RUSTICOL_DERIVED_MASS_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(path), Some("u d~ > W+ H"), None).unwrap();
    let default_mw = runtime.model_parameter_value("MW").unwrap().0;
    // G_mu inputs for MW = 80.3692 GeV (instead of the default 80.419 GeV).
    runtime
        .set_model_parameters(&BTreeMap::from([
            ("aEWM1".to_string(), (132.1001485837534, 0.0)),
            ("Gf".to_string(), (1.1663788e-5, 0.0)),
        ]))
        .unwrap();
    let mw = runtime.model_parameter_value("MW").unwrap().0;
    let mh = runtime.model_parameter_value("MH").unwrap().0;
    assert!((mw - default_mw).abs() > 1e-2, "the inputs must move MW ({default_mw} -> {mw})");
    let at = |runtime: &mut NativeRuntime, y: f64| runtime.evaluate_f64(&momenta(mw, mh, y), 1).unwrap()[0];
    let rest = at(&mut runtime, 0.0);
    for y in [0.5, -1.0] {
        let boosted = at(&mut runtime, y);
        let violation = (boosted / rest - 1.0).abs();
        assert!(violation < 1e-12, "|M|^2 changes by {violation:e} under a boost with rapidity {y}");
    }
}
