// SPDX-License-Identifier: 0BSD

//! Real scalar masters and epsilon assembly are available without PyO3.
use rusticol_loop_reduction::runtime::{Batch, PolynomialPlan};

#[test]
fn standalone_reduction_preserves_epsilon_order_and_sparse_results() {
    for (arithmetic, precision, engine) in [
        ("double", None, "generic"),
        ("double", None, "fast"),
        ("double", None, "fast-tensor"),
        ("double-double", None, "generic"),
        ("arbitrary", Some(80), "generic"),
    ] {
        let mut plan = PolynomialPlan::new(
            1,
            ["c".into(), "2".into(), "0".into()],
            vec!["c".into()],
            arithmetic,
            precision,
            engine,
        )
        .unwrap();
        let batch = |values: Vec<&str>, columns| {
            Batch::from_decimal(
                values.into_iter().map(|v| [v.into(), "0".into()]).collect(),
                2,
                columns,
                arithmetic,
                precision,
                Some(vec![91, 4]),
            )
            .unwrap()
        };
        let shifts = batch(vec!["0"; 8], 4);
        let masses = batch(vec!["1"; 2], 1);
        let parameters = batch(vec!["1", "3"], 1);
        let result = plan
            .evaluate_batch(&shifts, &masses, &masses, &parameters, None)
            .unwrap();
        assert_eq!(
            result.statuses(),
            [(91, "success".into()), (4, "success".into())]
        );
        let retained = result.values().unwrap();
        let values = retained
            .to_decimal()
            .into_iter()
            .map(|v| v[0].parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        // A0(1)=1+1/epsilon; N0=c and N1=2 yield (c+2,c,0).
        assert_eq!(values, [3., 1., 0., 5., 3., 0.]);
        // The dense adapter's existing nonfinite coefficient-head error is
        // global, before reducer lane statuses. It remains outside sparse
        // physical rescue; this extraction does not change that domain.
        if engine != "fast-tensor" {
            let failed = plan
                .evaluate_batch(&shifts, &masses, &masses, &batch(vec!["1", "NaN"], 1), None)
                .unwrap();
            assert_eq!(failed.statuses()[0], (91, "success".into()));
            assert_ne!(failed.statuses()[1].1, "success");
            assert!(failed.values().is_err());
            assert_eq!(failed.successful_values().ids(), [91]);
        }
        let retry = plan
            .evaluate_batch(
                &shifts.take_rows(vec![1], None).unwrap(),
                &masses.take_rows(vec![1], None).unwrap(),
                &masses.take_rows(vec![1], None).unwrap(),
                &parameters.take_rows(vec![1], None).unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(retry.values().unwrap().ids(), [4]);
        assert_eq!(
            retained
                .to_decimal()
                .into_iter()
                .map(|v| v[0].parse::<f64>().unwrap())
                .collect::<Vec<_>>(),
            values
        );
    }
}
