//! WP-B crossing prototype (test only). A template program evaluated at
//! caller-signed momenta `q[r] = sigma_r * p[pi_r]` must equal the crossed
//! member's own program. The hook below appends an assignment with an explicit,
//! possibly side-crossing permutation; production artifacts never carry one.
//!
//! The cases come from `RUSTICOL_CROSSING_PROTOTYPE_CASES` (JSON written by the
//! prototype driver); per-helicity/flow template values are written to its
//! `output` path and compared there against separately generated artifacts.

use super::native_runtime::crossing_negative_mask;
use super::*;

/// Append a union assignment with an explicit permutation and signed pattern subset.
fn push_test_assignment(
    runtime: &mut NativeRuntime,
    pdgs: Vec<i32>,
    permutation: Vec<usize>,
    patterns: Vec<u32>,
    signs: Vec<i32>,
) -> u32 {
    let assignments = runtime
        .flavour_union_assignments
        .as_ref()
        .expect("a pattern-resolved template");
    let pattern_count = assignments[0].signs.len();
    let crossing = runtime
        .flavour_union_relabelling
        .as_mut()
        .expect("template crossing table")
        .intern_crossing("crossing-prototype", &permutation, false, None)
        .expect("prototype crossing");
    let mut assignment = NativeUnionAssignment::new(
        "crossing-prototype".to_owned(),
        "crossing prototype".to_owned(),
        pdgs,
        permutation,
        NativeRationalFactor {
            numerator: 1,
            denominator: 1,
        },
        false,
        signs,
        patterns,
        pattern_count,
    )
    .expect("prototype assignment");
    assignment.crossing = crossing;
    let mut grown = assignments.to_vec();
    grown.push(assignment);
    let id = u32::try_from(grown.len() - 1).unwrap();
    runtime.flavour_union_assignments = Some(grown.into());
    id
}

fn usize_list(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn crossing_prototype_evaluates_signed_template_momenta_when_cases_are_supplied() {
    let Some(path) = std::env::var_os("RUSTICOL_CROSSING_PROTOTYPE_CASES") else {
        return;
    };
    let cases: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut results = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut runtime = NativeRuntime::load(
            case["artifact"].as_str().unwrap(),
            Some(case["template"].as_str().unwrap()),
            None,
        )
        .unwrap_or_else(|error| panic!("{name}: load template: {error}"));
        let pdgs = case["member_pdgs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let permutation = usize_list(&case["permutation"]);
        let n = permutation.len();
        // Patterns whose fermion lines join equal flavours (process_grouping's rule).
        let pairs = runtime
            .flavour_union_pattern_pairs()
            .expect("template pattern table")
            .to_vec();
        let patterns = pairs
            .iter()
            .enumerate()
            .filter(|(_, pattern)| {
                pattern.iter().all(|[f, a]| {
                    pdgs[permutation[*f as usize]].abs() == pdgs[permutation[*a as usize]].abs()
                })
            })
            .map(|(index, _)| index as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            patterns.len() as u64,
            case["expected_pattern_count"].as_u64().unwrap(),
            "{name}: pattern subset {patterns:?} of {pairs:?}"
        );
        let initial = runtime
            .flavour_union_relabelling
            .as_ref()
            .unwrap()
            .initial_count;
        let mask = crossing_negative_mask(&permutation, initial).unwrap();
        let expected_mask = case["sigma"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.as_i64() == Some(-1))
            .fold(0_u64, |m, (r, _)| m | (1 << r));
        assert_eq!(mask, expected_mask, "{name}: sigma");
        // Optional sign control (the driver's --flip-signs); +1 per pattern otherwise.
        let signs = case.get("pattern_signs").map_or_else(
            || vec![1; patterns.len()],
            |signs| {
                signs
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect()
            },
        );
        let id = push_test_assignment(&mut runtime, pdgs, permutation, patterns.clone(), signs);
        let momenta = case["momenta"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|point| point.as_array().unwrap().iter())
            .flat_map(|leg| leg.as_array().unwrap().iter())
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let points = momenta.len() / (4 * n);
        let offsets = (0..=points).collect::<Vec<_>>();
        let ids = vec![id; points];
        let helicities = runtime.helicities().unwrap();
        let colors = runtime.color_components().unwrap();
        let width = helicities.len() * colors.len();
        let mut values = vec![f64::NAN; points * width];
        runtime
            .evaluate_assignments_f64_into(
                &momenta,
                points,
                &offsets,
                &ids,
                None,
                None,
                AssignmentOutputMode::PerHelicityFlow,
                &mut values,
            )
            .unwrap_or_else(|error| panic!("{name}: evaluate crossed assignment: {error}"));
        let mut totals = vec![f64::NAN; points];
        runtime
            .evaluate_assignments_f64_into(
                &momenta,
                points,
                &offsets,
                &ids,
                None,
                None,
                AssignmentOutputMode::Total,
                &mut totals,
            )
            .unwrap();
        // The identity assignment (sigma = +1) refuses the signed momenta (WP-A2).
        let mut canary = vec![7.0; points];
        let error = runtime
            .evaluate_assignments_f64_into(
                &momenta,
                points,
                &offsets,
                &vec![0; points],
                None,
                None,
                AssignmentOutputMode::Total,
                &mut canary,
            )
            .expect_err("identity assignment at signed momenta");
        assert!(error.to_string().contains("energy"), "{name}: {error}");
        assert!(canary.iter().all(|v| *v == 7.0));
        // Mixed crossings in one point are refused, the output untouched.
        let mut mixed_canary = [7.0, 7.0];
        let mixed = runtime.evaluate_assignments_f64_into(
            &momenta[..4 * n],
            1,
            &[0, 2],
            &[0, id],
            None,
            None,
            AssignmentOutputMode::Total,
            &mut mixed_canary,
        );
        assert!(
            mixed
                .expect_err("mixed crossings")
                .to_string()
                .contains("different crossings")
        );
        assert_eq!(mixed_canary, [7.0, 7.0]);
        results.push(serde_json::json!({
            "name": name,
            "patterns": patterns,
            "helicities": helicities.iter().map(|h| h.helicities.clone()).collect::<Vec<_>>(),
            "colors": colors.iter().map(|c| serde_json::json!({"id": c.id, "word": c.word})).collect::<Vec<_>>(),
            "values": values,
            "totals": totals,
        }));
    }
    let output = cases["output"].as_str().unwrap();
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({ "results": results })).unwrap(),
    )
    .unwrap();
    eprintln!(
        "crossing prototype: {} cases written to {output}",
        results.len()
    );
}
