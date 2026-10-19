// SPDX-License-Identifier: 0BSD
//! Explicit generated-fixture integration: the complete coordinator has no
//! Python dependency, including during execution or precision rebinding.
use rusticol_core::loop_execution::{LoopExecutor, LoopInputs};
use rusticol_core::loop_runtime::load_loop_runtime_pacbin;
use rusticol_loop_reduction::runtime::physical::PhysicalBackend;
use std::sync::Arc;

#[test]
#[ignore = "requires PYAMPLICOL_NATIVE_UUDD_PACBIN from the portable fixture"]
fn standalone_complete_uudd_three_carriers() {
    let loaded = Arc::new(
        load_loop_runtime_pacbin(
            std::env::var("PYAMPLICOL_NATIVE_UUDD_PACBIN").expect("generated uudd PACBIN"),
        )
        .unwrap(),
    );
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tools/loop_reference/uudd_qcd.json")).unwrap();
    let momenta = fixture["points"][0]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|leg| {
            leg.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap().to_string())
        })
        .collect::<Vec<_>>();
    eprintln!(
        "native workspace bytes: reducer={}, scalar={}, schedule={}",
        std::mem::size_of::<rusticol_loop_reduction::runtime::PolynomialPlan>(),
        std::mem::size_of::<rusticol_loop_reduction::runtime::CoefficientEvaluator>(),
        std::mem::size_of::<rusticol_loop_reduction::runtime::CoefficientPlan>()
    );
    let mut reference: Option<Vec<f64>> = None;
    for (arithmetic, precision, engine) in [
        ("double", None, "fast"),
        ("double-double", None, "generic"),
        ("arbitrary", Some(70), "generic"),
    ] {
        let backend = PhysicalBackend::new(arithmetic, precision, "interpreted", engine).unwrap();
        eprintln!("binding {arithmetic}");
        let mut program = LoopExecutor::new(loaded.clone(), backend).unwrap();
        eprintln!("evaluating {arithmetic}");
        let output = program
            .evaluate(LoopInputs {
                spin_sources: Default::default(),
                momenta: momenta.clone(),
                scales_squared: vec!["1000000".into()],
                model_values: vec!["0.118".into(), "173".into()],
                point_ids: vec![73],
            })
            .unwrap();
        assert_eq!(output.r2_strategy, "model");
        assert_eq!(output.born.ids(), [73]);
        assert!(
            output.statuses[0]
                .iter()
                .all(|(id, status)| *id == 73 && status == "success")
        );
        let mut all = vec![output.born];
        all.extend(output.components);
        all.push(output.totals);
        let numbers = all
            .iter()
            .flat_map(|batch| batch.to_decimal())
            .flat_map(|value| value.map(|x| x.parse::<f64>().unwrap()))
            .collect::<Vec<_>>();
        assert!(numbers.iter().all(|x| x.is_finite()));
        assert!(numbers[0] > 0.);
        if let Some(old) = &reference {
            for (a, b) in numbers.iter().zip(old) {
                assert!((a - b).abs() < 1e-11 * (1. + a.abs()));
            }
        } else {
            reference = Some(numbers);
        }
    }
}

#[test]
#[ignore = "requires PYAMPLICOL_NATIVE_UUDD_PACBIN; isolates existing reducer stack"]
fn standalone_existing_reducer_constructor() {
    use rusticol_loop_reduction::runtime::PolynomialPlan;
    let loaded = load_loop_runtime_pacbin(
        std::env::var("PYAMPLICOL_NATIVE_UUDD_PACBIN").expect("generated uudd PACBIN"),
    )
    .unwrap();
    for family in &loaded.plan().families {
        eprintln!("existing primitive family {}", family.index);
        let reducer = PolynomialPlan::new(
            family.propagators,
            family.numerators.clone(),
            (0..family.powers.len()).map(|i| format!("c{i}")).collect(),
            "double-double",
            None,
            "generic",
        )
        .unwrap();
        assert_eq!(reducer.precision_bits(), 106);
    }
}
