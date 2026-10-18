// SPDX-License-Identifier: 0BSD

//! Saved instructions execute from Rust without importing the Python consumer.

use rusticol_loop_reduction::runtime::{
    Batch, CoefficientEvaluator, CoefficientKernelCache, CoefficientPlan,
};

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

#[test]
fn standalone_saved_evaluator_and_schedule_share_carriers_and_cold_templates() {
    let python = std::env::var_os("PYAMPLICOL_TEST_PYTHON").unwrap_or_else(|| "python3".into());
    let producer = std::process::Command::new(python)
        .env("SYMBOLICA_HIDE_BANNER", "1")
        .args(["-c", "import sys\nfrom symbolica import S\nx=S('native_runtime_x')\nsys.stdout.buffer.write((x*x+1).evaluator([x],n_cores=1,jit_compile=False).save())"])
        .output()
        .expect("launch existing Python Symbolica producer");
    assert!(
        producer.status.success(),
        "{}",
        String::from_utf8_lossy(&producer.stderr)
    );
    let state = producer.stdout;
    for (arithmetic, precision, engine) in [
        ("double", None, "interpreted"),
        ("double", None, "jit"),
        ("double-double", None, "interpreted"),
        ("arbitrary", Some(80), "interpreted"),
    ] {
        let mut scalar = CoefficientEvaluator::new(&state, arithmetic, precision).unwrap();
        let mut cache = CoefficientKernelCache::new(arithmetic, precision, engine).unwrap();
        let mut bind = || {
            CoefficientPlan::new(
                vec![state.clone(), state.clone()],
                vec![vec![0], vec![1]],
                1,
                vec![2],
                arithmetic,
                precision,
                engine,
                Some(16),
                Some(&mut cache),
            )
            .unwrap()
        };
        let mut first = bind();
        let mut second = bind();
        assert_eq!(cache.entries(), 1);
        assert_eq!(cache.hits(), 3);
        for rows in [17, 1, 0, 3] {
            let input = Batch::from_decimal(
                vec![["2".into(), "0".into()]; rows],
                rows,
                1,
                arithmetic,
                precision,
                Some((0..rows as u64).map(|id| 100 - id).collect()),
            )
            .unwrap();
            let values = first.evaluate_batch(&input).unwrap();
            let retained = values.to_decimal();
            assert_eq!(values.ids(), input.ids());
            assert_eq!(
                retained,
                second.evaluate_batch(&input).unwrap().to_decimal()
            );
            assert!(
                retained
                    .iter()
                    .all(|pair| pair[0].parse::<f64>().unwrap() == 26.)
            );
            let simple = scalar.evaluate_batch(&input).unwrap().to_decimal();
            assert!(
                simple
                    .iter()
                    .all(|pair| pair[0].parse::<f64>().unwrap() == 5.)
            );
            // A different subsequent batch must not mutate a retained result.
            first
                .evaluate_batch(
                    &Batch::from_decimal(
                        vec![["3".into(), "0".into()]],
                        1,
                        1,
                        arithmetic,
                        precision,
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
            assert_eq!(values.to_decimal(), retained);
        }
    }
}
