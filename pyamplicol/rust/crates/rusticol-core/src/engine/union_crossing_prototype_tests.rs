// SPDX-License-Identifier: 0BSD

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

/// Massless 2 -> n-2 point in member leg order: beams along z, RAMBO final state.
fn massless_point(n: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut state = seed;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let sqrt_s = 1000.0;
    let mut legs = vec![
        [sqrt_s / 2.0, 0.0, 0.0, sqrt_s / 2.0],
        [sqrt_s / 2.0, 0.0, 0.0, -sqrt_s / 2.0],
    ];
    let raw = (2..n)
        .map(|_| {
            let cos = 2.0 * uniform() - 1.0;
            let phi = std::f64::consts::TAU * uniform();
            let energy = -(uniform() * uniform()).ln();
            let sin = (1.0 - cos * cos).sqrt();
            [
                energy,
                energy * sin * phi.cos(),
                energy * sin * phi.sin(),
                energy * cos,
            ]
        })
        .collect::<Vec<_>>();
    let total = (0..4)
        .map(|c| raw.iter().map(|q| q[c]).sum::<f64>())
        .collect::<Vec<_>>();
    let mass = (total[0] * total[0] - total[1..].iter().map(|v| v * v).sum::<f64>()).sqrt();
    let boost = [-total[1] / mass, -total[2] / mass, -total[3] / mass];
    let (gamma, scale) = (total[0] / mass, sqrt_s / mass);
    let a = 1.0 / (1.0 + gamma);
    for q in raw {
        let bq = boost[0] * q[1] + boost[1] * q[2] + boost[2] * q[3];
        legs.push([
            scale * (gamma * q[0] + bq),
            scale * (q[1] + boost[0] * (q[0] + a * bq)),
            scale * (q[2] + boost[1] * (q[0] + a * bq)),
            scale * (q[3] + boost[2] * (q[0] + a * bq)),
        ]);
    }
    legs
}

/// Per-point slot lists: a point's rows equal the full evaluation (every assignment
/// at every point) bitwise and do not depend on requests of points in its partition
/// (same crossing and mask); moving a neighbour to another partition is not covered.
/// `RUSTICOL_UNION_ASSIGNMENT_ARTIFACTS`: LC union artifacts (path list).
#[test]
fn per_point_assignment_slots_equal_the_full_evaluation_when_artifacts_are_supplied() {
    let Some(roots) = std::env::var_os("RUSTICOL_UNION_ASSIGNMENT_ARTIFACTS") else {
        return;
    };
    let mut checked = 0;
    for root in std::env::split_paths(&roots) {
        let groups: Value =
            serde_json::from_slice(&std::fs::read(root.join("process-groups.json")).unwrap())
                .unwrap();
        for group in groups["groups"].as_array().unwrap() {
            let template = group["representative"]["process_id"].as_str().unwrap();
            let mut runtime =
                NativeRuntime::load(root.to_str().unwrap(), Some(template), None).unwrap();
            let Some(assignments) = runtime.flavour_union_assignments.clone() else {
                continue;
            };
            let width =
                runtime.helicities().unwrap().len() * runtime.color_components().unwrap().len();
            let mut classes = BTreeMap::<(u32, Option<u64>), Vec<u32>>::new();
            for (id, assignment) in assignments.iter().enumerate() {
                classes
                    .entry((assignment.crossing, assignment.mask))
                    .or_default()
                    .push(u32::try_from(id).unwrap());
            }
            for ((crossing, _), ids) in classes.into_iter().filter(|(_, ids)| ids.len() >= 2) {
                let negative = runtime
                    .flavour_union_relabelling
                    .as_ref()
                    .unwrap()
                    .crossing(crossing)
                    .unwrap()
                    .negative_mask();
                let permutation = assignments[ids[0] as usize].permutation.clone();
                let n = permutation.len();
                let points = 150;
                let momenta = (0..points)
                    .flat_map(|point| {
                        let legs = massless_point(n, 1 + point as u64);
                        let permutation = permutation.clone();
                        (0..n).flat_map(move |slot| {
                            let sign = if negative >> slot & 1 == 1 { -1.0 } else { 1.0 };
                            legs[permutation[slot]].map(|component| sign * component)
                        })
                    })
                    .collect::<Vec<_>>();
                let k = ids.len();
                for (mode, row) in [
                    (AssignmentOutputMode::Total, 1),
                    (AssignmentOutputMode::PerHelicityFlow, width),
                ] {
                    let mut evaluate = |momenta: &[f64], requests: &[Vec<u32>]| {
                        let mut offsets = vec![0];
                        for request in requests {
                            offsets.push(offsets.last().unwrap() + request.len());
                        }
                        let flat = requests.concat();
                        let mut output = vec![f64::NAN; flat.len() * row];
                        runtime
                            .evaluate_assignments_f64_into(
                                momenta,
                                requests.len(),
                                &offsets,
                                &flat,
                                None,
                                None,
                                mode,
                                &mut output,
                            )
                            .unwrap();
                        requests
                            .iter()
                            .enumerate()
                            .map(|(point, request)| {
                                (0..request.len())
                                    .map(|j| {
                                        let start = (offsets[point] + j) * row;
                                        output[start..start + row]
                                            .iter()
                                            .map(|value| value.to_bits())
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    };
                    let own = |point: usize| ids[point % k];
                    let full = evaluate(&momenta, &vec![ids.clone(); points]);
                    let single = evaluate(
                        &momenta,
                        &(0..points).map(|p| vec![own(p)]).collect::<Vec<_>>(),
                    );
                    // Odd points switch assignment; even points keep theirs.
                    let shifted = evaluate(
                        &momenta,
                        &(0..points)
                            .map(|p| vec![if p % 2 == 1 { ids[(p + 1) % k] } else { own(p) }])
                            .collect::<Vec<_>>(),
                    );
                    let duplicated = evaluate(
                        &momenta,
                        &(0..points)
                            .map(|p| vec![own(p), ids[(p + 1) % k], own(p)])
                            .collect::<Vec<_>>(),
                    );
                    for point in 0..points {
                        let reference = &full[point];
                        assert_eq!(
                            single[point][0],
                            reference[point % k],
                            "{template} point {point}"
                        );
                        if point % 2 == 0 {
                            assert_eq!(
                                shifted[point][0], single[point][0],
                                "{template} point {point}"
                            );
                        }
                        assert_eq!(duplicated[point][0], reference[point % k]);
                        assert_eq!(duplicated[point][1], reference[(point + 1) % k]);
                        assert_eq!(duplicated[point][2], reference[point % k]);
                    }
                    // Diagnostic only: the template program's tile position (alone vs in a batch).
                    let alone = (0..points)
                        .filter(|point| {
                            let at = &momenta[point * 4 * n..(point + 1) * 4 * n];
                            evaluate(at, &[vec![own(*point)]])[0][0] != single[*point][0]
                        })
                        .count();
                    eprintln!(
                        "{template} crossing {crossing} ({k} assignments, {mode:?}): \
                         {alone}/{points} points differ alone vs in the batch"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no assignment class with two or more members");
}
