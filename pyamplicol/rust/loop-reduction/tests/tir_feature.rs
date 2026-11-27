// SPDX-License-Identifier: 0BSD

//! The `tir` cargo feature (TIRust, on by default; m-lit PLAN §2.c.2 item 3)
//! links TIRust and leaves NinjaRS's numerics unchanged. The NinjaRS values
//! below must not depend on the feature: with `PYAMPLICOL_NINJARS_BITS=<path>`
//! the test writes their bits to `<path>`, and the CI `tir` job (the
//! non-default configuration) compares the files of a default and a
//! `--no-default-features` run byte for byte.

use ninjars::fast::{C64, FastInput, FastStatus, TensorNumerator};
use rusticol_loop_reduction::{FastReducer, Laurent};

#[test]
fn tirust_is_linked_exactly_with_the_tir_feature() {
    assert_eq!(
        rusticol_loop_reduction::tir::available(),
        cfg!(feature = "tir")
    );
}

/// A dense rank-`rank` numerator over `lanes` lanes with distinct exact
/// dyadic coefficients.
fn numerator(rank: usize, lanes: usize) -> TensorNumerator {
    let mut numerator = TensorNumerator::new(rank, lanes);
    for lane in 0..lanes {
        for (index, value) in numerator.lane_coefficients_mut(lane).iter_mut().enumerate() {
            let k = (index * 7 + lane * 13 + 3) % 17;
            *value = C64::new((k as f64 - 8.) / 4., ((k * 5) % 11) as f64 / 8. - 0.5);
        }
    }
    numerator
}

fn bits(value: &Laurent<C64>) -> [u64; 6] {
    let [a, b, c] = [&value.finite, &value.single_pole, &value.double_pole];
    [a.re, a.im, b.re, b.im, c.re, c.im].map(f64::to_bits)
}

#[test]
fn ninjars_lanes_do_not_depend_on_the_tir_feature() {
    // A massive box at rank 4 (one complex mass) and a massless triangle
    // at rank 3 (soft and collinear poles), two lanes each, generic shifts.
    let cases: [(usize, Vec<[f64; 4]>, Vec<C64>); 2] = [
        (
            4,
            vec![
                [0., 0., 0., 0.],
                [341. / 128., 86. / 128., -115. / 128., 83. / 128.],
                [180. / 128., 118. / 128., -44. / 128., 191. / 128.],
                [521. / 128., 104. / 128., 34. / 128., 257. / 128.],
            ],
            vec![
                C64::real(1.),
                C64::new(0.75, -0.125),
                C64::real(1.),
                C64::real(0.25),
            ],
        ),
        (
            3,
            vec![[0., 0., 0., 0.], [1., 0., 0., 1.], [3., 1., 0., 0.]],
            vec![C64::ZERO; 3],
        ),
    ];
    let mut lines = Vec::new();
    for (propagators, shifts, masses) in cases {
        let lanes = 2;
        let mut reducer = FastReducer::new(propagators, numerator(propagators, lanes)).unwrap();
        let result = reducer
            .evaluate(
                &[11, 12],
                FastInput {
                    lanes,
                    shifts: &shifts.repeat(lanes),
                    masses_squared: &masses.repeat(lanes),
                    scales_squared: &[1.5; 2],
                    s_matrix: None,
                },
            )
            .unwrap();
        for row in &result {
            assert_eq!(row.status, FastStatus::Success);
            let value = row.value.as_ref().unwrap();
            let words = bits(value);
            assert!(words.iter().all(|word| f64::from_bits(*word).is_finite()));
            lines.push(
                words
                    .iter()
                    .map(|word| format!("{word:016x}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }
    if let Some(path) = std::env::var_os("PYAMPLICOL_NINJARS_BITS") {
        std::fs::write(path, lines.join("\n") + "\n").unwrap();
    }
}
