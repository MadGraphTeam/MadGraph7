// SPDX-License-Identifier: 0BSD

//! Native exact preprocessing and sparse dependency states share the Python path.

use rusticol_loop_reduction::runtime::{
    Batch, ExactPolynomialEvaluator, wavefunctions_partial_batch,
};

#[test]
fn native_exact_inputs_retain_cancellation_and_carrier_failures() {
    let python = std::env::var_os("PYAMPLICOL_TEST_PYTHON").unwrap_or_else(|| "python3".into());
    let producer = std::process::Command::new(python)
        .env("SYMBOLICA_HIDE_BANNER", "1")
        .args(["-c", "import sys\nfrom symbolica import S\nx,y=S('native_exact_x','native_exact_y')\nsys.stdout.buffer.write((x-y).evaluator([x,y],n_cores=1,jit_compile=False).save())"])
        .output().expect("launch existing Python Symbolica producer");
    assert!(
        producer.status.success(),
        "{}",
        String::from_utf8_lossy(&producer.stderr)
    );
    let mut exact = ExactPolynomialEvaluator::new(&producer.stdout).unwrap();
    let large = format!("1{}", "0".repeat(400));
    let larger = format!("1{}1", "0".repeat(399));
    for (arithmetic, precision) in [
        ("double", None),
        ("double-double", None),
        ("arbitrary", Some(80)),
    ] {
        let output = exact
            .evaluate_decimal(
                vec![[larger.clone(), "0".into()], [large.clone(), "0".into()]],
                1,
                arithmetic,
                precision,
                Some(vec![77]),
            )
            .unwrap();
        assert_eq!(output.ids(), [77]);
        assert_eq!(output.to_decimal()[0][0].parse::<f64>().unwrap(), 1.);
        let momenta = Batch::from_exact_decimal(
            ["5", "3", "0", "4", "1e400", "0", "0", "1e400"]
                .map(|v| [v.into(), "0".into()])
                .to_vec(),
            2,
            4,
            arithmetic,
            precision,
            Some(vec![9, 82]),
        )
        .unwrap();
        let (states, statuses) =
            wavefunctions_partial_batch(&momenta, vec![1, -1], vec!["u".into(), "u".into()], None)
                .unwrap();
        assert_eq!(statuses[0], (9, "success".into()));
        assert!(states.nonfinite_rows().is_empty());
        if arithmetic == "arbitrary" {
            assert_eq!(statuses[1], (82, "success".into()));
            assert_eq!(states.ids(), [9, 82]);
        } else {
            assert_ne!(statuses[1].1, "success");
            assert_eq!(states.ids(), [9]);
        }
    }
}
