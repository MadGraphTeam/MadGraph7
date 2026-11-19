// SPDX-License-Identifier: 0BSD

//! Saved instructions execute from Rust without importing the Python consumer.

use rusticol_loop_reduction::runtime::{Batch, CoefficientEvaluator};

#[test]
fn standalone_decoder_rebinds_fixed_scalar_predicates() {
    let python = std::env::var_os("PYAMPLICOL_TEST_PYTHON").unwrap_or_else(|| "python3".into());
    let producer = std::process::Command::new(python)
        .env("SYMBOLICA_HIDE_BANNER", "1")
        .args(["-c", "import sys\nfrom symbolica import S\nx=S('native_predicate_x')\nf=S('pyamplicol_scalar_v1::truth',eval={'complex': lambda x: complex(x[0]!=0)})\nsys.stdout.buffer.write(f(x).evaluator([x],n_cores=1,jit_compile=False).save())"])
        .output()
        .expect("launch existing Python Symbolica producer");
    assert!(
        producer.status.success(),
        "{}",
        String::from_utf8_lossy(&producer.stderr)
    );
    for (arithmetic, precision) in [
        ("double", None),
        ("double-double", None),
        ("arbitrary", Some(80)),
    ] {
        let mut evaluator =
            CoefficientEvaluator::new(&producer.stdout, arithmetic, precision).unwrap();
        let input = Batch::from_decimal(
            ["0", "1e-200", "-1e200"]
                .map(|v| [v.into(), "0".into()])
                .to_vec(),
            3,
            1,
            arithmetic,
            precision,
            Some(vec![21, 4, 90]),
        )
        .unwrap();
        let output = evaluator.evaluate_batch(&input).unwrap();
        assert_eq!(output.ids(), input.ids());
        let values = output
            .to_decimal()
            .into_iter()
            .map(|v| v[0].parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values, [0., 1., 1.]);
    }
}
