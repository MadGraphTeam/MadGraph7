// SPDX-License-Identifier: 0BSD

//! Cross-side relabelling (`rusticol.flavour-union-relabelling.v2`): a template slot
//! symmetry may map a crossable slot to the opposite side with the same all-outgoing
//! flavour. The relabelled assignment is crossed and describes the same member event.

use super::*;

const W: NativeSlotCrossing = NativeSlotCrossing::WeylFermion;
const V: NativeSlotCrossing = NativeSlotCrossing::MasslessVector;

/// Template d g > d d d~ g: fundamentals at 2, 3; antifundamentals at 0 (d in), 4 (d~ out).
fn dg_relabelling(slot_crossings: Vec<NativeSlotCrossing>) -> NativeUnionRelabelling {
    NativeUnionRelabelling::new(
        vec![1, 21, 1, 1, -1, 21],
        2,
        Some(vec![vec![[2, 0], [3, 4]], vec![[2, 4], [3, 0]]]),
        2,
        slot_crossings.into(),
    )
    .unwrap()
}

fn unit_assignment(
    id: &str,
    pdgs: Vec<i32>,
    permutation: Vec<usize>,
    conjugated: bool,
    patterns: Vec<u32>,
) -> NativeUnionAssignment {
    let signs = vec![1; patterns.len()];
    NativeUnionAssignment::new(
        id.to_owned(),
        id.to_owned(),
        pdgs,
        permutation,
        NativeRationalFactor {
            numerator: 1,
            denominator: 1,
        },
        conjugated,
        signs,
        patterns,
        2,
    )
    .unwrap()
}

fn compiled_runtime(
    relabelling: NativeUnionRelabelling,
    assignments: Vec<NativeUnionAssignment>,
) -> NativeRuntime {
    NativeRuntime {
        root: PathBuf::new(),
        artifact_id: "0".repeat(64),
        runtime: tests::empty_generic_runtime(),
        execution_lane: NativeExecutionLane::Compiled,
        process: "d g > d d d~ g".to_string(),
        process_key: "p0".to_string(),
        representative_process_id: "p0".to_string(),
        external_permutation: (0..6).collect(),
        input_crossing_map: None,
        permutation_alias_of: None,
        final_state_permutation_alias_of: None,
        physics_v1: native_runtime::LazyProcessPhysicsV1::loaded(tests::alias_test_physics("lc")),
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
        correlated: None,
        flavour_union_assignments: Some(assignments.into()),
        flavour_union_relabelling: Some(Box::new(relabelling)),
        process_group: None,
    }
}

#[test]
fn cross_side_symmetries_need_equal_outgoing_flavour_on_crossable_slots() {
    let relabelling = dg_relabelling(vec![W, V, W, W, W, V]);
    // d in <-> d~ out (outgoing d~ both), g in <-> g out, and both together.
    for permutation in [
        [4, 1, 2, 3, 0, 5],
        [0, 5, 2, 3, 4, 1],
        [4, 5, 3, 2, 0, 1],
        [0, 1, 3, 2, 4, 5],
    ] {
        relabelling.validate_symmetry(&permutation).unwrap();
    }
    // d in (outgoing d~) <-> d out: different all-outgoing flavour.
    let error = relabelling
        .validate_symmetry(&[2, 1, 0, 3, 4, 5])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    // Same side, different PDG stays refused.
    let error = relabelling
        .validate_symmetry(&[0, 1, 2, 4, 3, 5])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    // Quark -> gluon across sides is a wrong flavour, not an uncrossable slot.
    let error = relabelling
        .validate_symmetry(&[5, 1, 2, 3, 4, 0])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    // A wrong flavour wins over an uncrossable slot at any slot order.
    let error = dg_relabelling(vec![W, V, W, W, NativeSlotCrossing::Refused, V])
        .validate_symmetry(&[4, 1, 0, 3, 2, 5])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    // A full-colour lane refuses cross-side images but keeps same-side ones.
    let full_colour = dg_relabelling(vec![W, V, W, W, W, V]).without_crossed_lane();
    full_colour.validate_symmetry(&[0, 1, 3, 2, 4, 5]).unwrap();
    let error = full_colour
        .validate_symmetry(&[4, 1, 2, 3, 0, 5])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("full-colour"), "{error}");
    // A slot that may not cross, or no crossing table.
    for crossings in [vec![W, V, W, W, NativeSlotCrossing::Refused, V], Vec::new()] {
        let error = dg_relabelling(crossings)
            .validate_symmetry(&[4, 1, 2, 3, 0, 5])
            .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    }
    let error = dg_relabelling(vec![W, NativeSlotCrossing::Refused, W, W, W, V])
        .validate_symmetry(&[0, 5, 2, 3, 4, 1])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    // Pure-vector template g g > g g: same-side swaps only.
    let gluons = NativeUnionRelabelling::new(vec![21; 4], 2, None, 1, vec![V; 4].into()).unwrap();
    gluons.validate_symmetry(&[1, 0, 3, 2]).unwrap();
    let error = gluons.validate_symmetry(&[2, 1, 0, 3]).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
}

#[test]
fn cross_side_relabelling_crosses_the_member_and_refuses_conjugated_bases() {
    let pdgs = vec![1, 21, 1, 1, -1, 21];
    // Member d g > d s s~ g: d pairs with d (pattern 0 joins 2-0 and 3-4).
    let member = vec![1, 21, 1, 3, -3, 21];
    let mut runtime = compiled_runtime(
        dg_relabelling(vec![W, V, W, W, W, V]),
        vec![
            unit_assignment(
                "template",
                pdgs.clone(),
                (0..6).collect(),
                false,
                vec![0, 1],
            ),
            unit_assignment("member", member.clone(), (0..6).collect(), false, vec![0]),
            unit_assignment("conjugate", member, (0..6).collect(), true, vec![0]),
        ],
    );
    // Swap d in (slot 0) with s~ out (slot 4): both crossed, pattern 0 -> 1.
    let crossed = runtime.relabel_assignment(1, &[4, 1, 2, 3, 0, 5]).unwrap();
    assert_eq!(crossed, 3);
    let entry = &runtime.flavour_union_assignments().unwrap()[3];
    assert_eq!(entry.permutation, [4, 1, 2, 3, 0, 5]);
    assert_eq!(entry.patterns, [1]);
    assert!(!entry.conjugated);
    let crossing = runtime.flavour_union_crossing(3).unwrap();
    assert_eq!(&*crossing.momentum_signs, &[-1, 1, 1, 1, -1, 1]);
    assert_eq!(&*crossing.helicity_signs, &[-1, 1, 1, 1, -1, 1]);
    // Interned, and the inverse relabelling of the crossed id returns to the base.
    assert_eq!(
        runtime.relabel_assignment(1, &[4, 1, 2, 3, 0, 5]).unwrap(),
        3
    );
    assert_eq!(
        runtime.relabel_assignment(3, &[4, 1, 2, 3, 0, 5]).unwrap(),
        4
    );
    let back = &runtime.flavour_union_assignments().unwrap()[4];
    assert_eq!(back.permutation, (0..6).collect::<Vec<_>>());
    assert_eq!(back.crossing, 0);
    assert_eq!(back.patterns, [0]);
    // Gluon in <-> gluon out keeps every pattern but crosses both gluons.
    let gluons = runtime.relabel_assignment(0, &[0, 5, 2, 3, 4, 1]).unwrap();
    let crossing = runtime.flavour_union_crossing(gluons as usize).unwrap();
    assert_eq!(&*crossing.momentum_signs, &[1, -1, 1, 1, 1, -1]);
    assert_eq!(&*crossing.helicity_signs, &[1, 1, 1, 1, 1, 1]);
    // A conjugated base may not be crossed; a same-side swap of it is fine.
    let error = runtime
        .relabel_assignment(2, &[4, 1, 2, 3, 0, 5])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(
        error.to_string().contains("crossed and conjugated"),
        "{error}"
    );
    runtime.relabel_assignment(2, &[0, 1, 3, 2, 4, 5]).unwrap();
    // Non-crossable images are refused before any id is interned.
    let count = runtime.flavour_union_assignments().unwrap().len();
    assert!(runtime.relabel_assignment(1, &[2, 1, 0, 3, 4, 5]).is_err());
    assert_eq!(runtime.flavour_union_assignments().unwrap().len(), count);
}

#[test]
fn cross_side_capability_is_advertised_with_v1() {
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    {
        let supported = supported_runtime_capabilities();
        assert!(supported.contains(&FLAVOUR_UNION_RELABELLING_RUNTIME_CAPABILITY));
        assert!(supported.contains(&FLAVOUR_UNION_CROSS_SIDE_RELABELLING_RUNTIME_CAPABILITY));
        assert!(runtime_capability_is_supported(
            FLAVOUR_UNION_CROSS_SIDE_RELABELLING_RUNTIME_CAPABILITY
        ));
    }
}

/// Massless 2 -> n point in the member's leg order (RAMBO, fixed LCG seed).
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn member_point(external_count: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let sqrt_s = 1000.0;
    let final_count = external_count - 2;
    let mut qs = Vec::with_capacity(final_count);
    for _ in 0..final_count {
        let cos = 2.0 * uniform() - 1.0;
        let phi = 2.0 * std::f64::consts::PI * uniform();
        let energy = -(uniform() * uniform()).ln();
        let sin = (1.0 - cos * cos).sqrt();
        qs.push([
            energy,
            energy * sin * phi.cos(),
            energy * sin * phi.sin(),
            energy * cos,
        ]);
    }
    let total = (0..4)
        .map(|i| qs.iter().map(|q| q[i]).sum::<f64>())
        .collect::<Vec<_>>();
    let mass =
        (total[0] * total[0] - total[1] * total[1] - total[2] * total[2] - total[3] * total[3])
            .sqrt();
    let b = [-total[1] / mass, -total[2] / mass, -total[3] / mass];
    let (x, gamma) = (sqrt_s / mass, total[0] / mass);
    let a = 1.0 / (1.0 + gamma);
    let half = sqrt_s / 2.0;
    let mut point = vec![[half, 0.0, 0.0, half], [half, 0.0, 0.0, -half]];
    for q in qs {
        let bq = b[0] * q[1] + b[1] * q[2] + b[2] * q[3];
        point.push([
            x * (gamma * q[0] + bq),
            x * (q[1] + b[0] * q[0] + a * bq * b[0]),
            x * (q[2] + b[1] * q[0] + a * bq * b[1]),
            x * (q[3] + b[2] * q[0] + a * bq * b[2]),
        ]);
    }
    point
}

/// Template-order signed momenta of `assignment` at member point `point`.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn signed_momenta(runtime: &NativeRuntime, assignment: u32, point: &[[f64; 4]]) -> Vec<f64> {
    let entry = &runtime.flavour_union_assignments().unwrap()[assignment as usize];
    let crossing = runtime.flavour_union_crossing(assignment as usize).unwrap();
    entry
        .permutation
        .iter()
        .zip(crossing.momentum_signs.iter())
        .flat_map(|(leg, sign)| point[*leg].map(|value| f64::from(*sign) * value))
        .collect()
}

/// Per-helicity values of `assignment` at member point `point`, keyed by member helicity.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn assignment_helicities(
    runtime: &mut NativeRuntime,
    assignment: u32,
    point: &[[f64; 4]],
) -> BTreeMap<Vec<i32>, f64> {
    let momenta = signed_momenta(runtime, assignment, point);
    let count = runtime.helicity_ids().unwrap().len();
    let mut values = vec![0.0; count];
    runtime
        .evaluate_assignments_f64_into(
            &momenta,
            1,
            &[0, 1],
            &[assignment],
            None,
            None,
            AssignmentOutputMode::PerHelicity,
            &mut values,
        )
        .unwrap();
    values
        .into_iter()
        .enumerate()
        .map(|(helicity, value)| {
            let labels = runtime
                .flavour_union_assignment_helicity(assignment as usize, helicity)
                .unwrap();
            (labels, value)
        })
        .collect()
}

/// Every non-identity template symmetry of `0..count`: with at least one cross-side
/// image (`cross`) or with none.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn symmetries(relabelling: &NativeUnionRelabelling, count: usize, cross: bool) -> Vec<Vec<usize>> {
    fn visit(
        prefix: &mut Vec<usize>,
        used: &mut [bool],
        out: &mut Vec<Vec<usize>>,
        relabelling: &NativeUnionRelabelling,
        cross: bool,
    ) {
        let count = used.len();
        if prefix.len() == count {
            let initial = relabelling.initial_count;
            let crosses = prefix
                .iter()
                .enumerate()
                .any(|(slot, image)| (slot < initial) != (*image < initial));
            let identity = prefix
                .iter()
                .enumerate()
                .all(|(slot, image)| slot == *image);
            if crosses == cross && !identity && relabelling.validate_symmetry(prefix).is_ok() {
                out.push(prefix.clone());
            }
            return;
        }
        let slot = prefix.len();
        for image in 0..count {
            let pdg = relabelling.template_pdgs[slot];
            let other = relabelling.template_pdgs[image];
            if used[image] || (pdg != other && pdg != -other) {
                continue;
            }
            used[image] = true;
            prefix.push(image);
            visit(prefix, used, out, relabelling, cross);
            prefix.pop();
            used[image] = false;
        }
    }
    let mut out = Vec::new();
    visit(
        &mut Vec::new(),
        &mut vec![false; count],
        &mut out,
        relabelling,
        cross,
    );
    out
}

/// `RUSTICOL_CROSS_SIDE_RELABELLING_ARTIFACTS`: LC flavour-union artifacts (path list).
/// Every cross-side relabelling of every unconjugated assignment of every template equals
/// the base at the same member point, in total and per member helicity, at roundoff
/// (99% within 1e-12, all within 1e-9); conjugated bases are refused. Same-side (v1)
/// relabellings at the same points give the roundoff reference.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn cross_side_relabelled_values_equal_their_base_when_artifacts_are_supplied() {
    let Some(roots) = std::env::var_os("RUSTICOL_CROSS_SIDE_RELABELLING_ARTIFACTS") else {
        return;
    };
    let read = |path: std::path::PathBuf| -> Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    };
    let (mut refused, mut templates) = (0usize, 0usize);
    // relative differences: [same side, cross side]; per helicity relative to the total
    let mut differences = [Vec::<f64>::new(), Vec::<f64>::new()];
    let mut helicity_differences = [Vec::<f64>::new(), Vec::<f64>::new()];
    for root in std::env::split_paths(&roots) {
        let root = root.to_str().unwrap().to_owned();
        for process in read(std::path::Path::new(&root).join("artifact.json"))["processes"]
            .as_array()
            .unwrap()
        {
            let id = process["id"].as_str().unwrap();
            let Ok(mut runtime) = NativeRuntime::load(&root, Some(id), None) else {
                continue;
            };
            if !runtime.flavour_union_pattern_resolved()
                || runtime
                    .flavour_union_relabelling
                    .as_ref()
                    .is_none_or(|table| table.pattern_pairs.is_none())
            {
                continue;
            }
            templates += 1;
            let count = runtime.external_count();
            let table = runtime.flavour_union_relabelling.as_ref().unwrap();
            let kinds = [
                symmetries(table, count, false),
                symmetries(table, count, true),
            ];
            let bases = runtime.flavour_union_assignments().unwrap().len() as u32;
            for base in 0..bases {
                let conjugated =
                    runtime.flavour_union_assignments().unwrap()[base as usize].conjugated;
                for (cross, permutations) in kinds.iter().enumerate() {
                    for (index, permutation) in permutations.iter().enumerate() {
                        if runtime
                            .flavour_union_relabelling
                            .as_ref()
                            .unwrap()
                            .derived
                            .len()
                            + 2
                            >= MAX_RELABELLED_ASSIGNMENTS
                        {
                            runtime = NativeRuntime::load(&root, Some(id), None).unwrap();
                        }
                        let base_crossed = runtime
                            .flavour_union_crossing(base as usize)
                            .unwrap()
                            .negative_mask()
                            != 0;
                        let relabelled = match runtime.relabel_assignment(base, permutation) {
                            Ok(relabelled) => {
                                assert!(cross == 0 || !conjugated, "{id} {base} {permutation:?}");
                                relabelled
                            }
                            Err(error) => {
                                // Only crossing refusals: a conjugated base, or a crossed
                                // base moved onto a slot that may not cross.
                                assert!(
                                    cross == 1 && (conjugated || base_crossed),
                                    "{id} {base} {permutation:?}: {error}"
                                );
                                assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
                                refused += 1;
                                continue;
                            }
                        };
                        let seed = (u64::from(base) << 20) | ((cross as u64) << 19) | index as u64;
                        let point = member_point(count, seed);
                        let own = assignment_helicities(&mut runtime, base, &point);
                        let other = assignment_helicities(&mut runtime, relabelled, &point);
                        let direct = own.values().sum::<f64>();
                        let value = other.values().sum::<f64>();
                        let scale = direct.abs().max(value.abs());
                        let relative = (value - direct).abs() / scale;
                        assert!(
                            relative <= 1e-9 && direct != 0.0,
                            "{id} assignment {base} by {permutation:?}: {value:e} vs {direct:e} ({relative:e})"
                        );
                        differences[cross].push(relative);
                        assert_eq!(
                            own.keys().collect::<Vec<_>>(),
                            other.keys().collect::<Vec<_>>(),
                            "{id} assignment {base} by {permutation:?}: member helicities differ"
                        );
                        let worst = own
                            .values()
                            .zip(other.values())
                            .map(|(left, right)| (left - right).abs() / scale)
                            .fold(0.0, f64::max);
                        assert!(
                            worst <= 1e-9,
                            "{id} assignment {base} by {permutation:?}: helicity difference {worst:e}"
                        );
                        helicity_differences[cross].push(worst);
                        if relative > 1e-11 {
                            eprintln!(
                                "  {} {id} assignment {base} by {permutation:?}: {relative:e}",
                                ["same-side", "cross-side"][cross]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        !differences[1].is_empty(),
        "no cross-side relabelling was compared"
    );
    let labels = [
        "same-side",
        "cross-side",
        "same-side per helicity",
        "cross-side per helicity",
    ];
    let [same, cross] = differences;
    let [same_helicity, cross_helicity] = helicity_differences;
    for (label, mut values) in labels
        .into_iter()
        .zip([same, cross, same_helicity, cross_helicity])
    {
        values.sort_by(f64::total_cmp);
        let quantile = |q: f64| {
            values
                .get(((values.len() as f64 - 1.0) * q) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        eprintln!(
            "{label} relabellings: templates {templates}, compared {}, median {:e}, 99% {:e}, max {:e}, \
             above 1e-12 {}",
            values.len(),
            quantile(0.5),
            quantile(0.99),
            values.last().copied().unwrap_or(0.0),
            values.iter().filter(|value| **value > 1e-12).count()
        );
        // Rare badly conditioned points: a wrong sign or pattern is O(1) everywhere.
        assert!(quantile(0.99) <= 1e-12, "{label}: 99% {:e}", quantile(0.99));
    }
    eprintln!("cross-side relabellings refused: {refused}");
}

/// `RUSTICOL_SELECTOR_ONLY_ARTIFACTS`: full-colour lanes (selector-only, contracted) refuse
/// a cross-side relabelling at relabel time; on the selector-only lane a same-side one
/// equals its base summed over helicities at the same member point.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn selector_only_lane_refuses_cross_side_relabelling_when_artifacts_are_supplied() {
    let Some(roots) = std::env::var_os("RUSTICOL_SELECTOR_ONLY_ARTIFACTS") else {
        return;
    };
    let read = |path: std::path::PathBuf| -> Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    };
    let (mut compared, mut refused) = (0usize, 0usize);
    for root in std::env::split_paths(&roots) {
        let root = root.to_str().unwrap().to_owned();
        for process in read(std::path::Path::new(&root).join("artifact.json"))["processes"]
            .as_array()
            .unwrap()
        {
            let id = process["id"].as_str().unwrap();
            let Ok(mut runtime) = NativeRuntime::load(&root, Some(id), None) else {
                continue;
            };
            let selector = matches!(
                runtime.execution_lane,
                NativeExecutionLane::RecurrenceSelector(_)
            );
            let contracted = matches!(
                &runtime.execution_lane,
                NativeExecutionLane::Recurrence(lane) if lane.is_contracted()
            );
            if !(selector || contracted)
                || runtime
                    .flavour_union_relabelling
                    .as_ref()
                    .is_none_or(|table| table.pattern_pairs.is_none())
            {
                continue;
            }
            let table = runtime.flavour_union_relabelling.as_ref().unwrap();
            let (pdgs, initial) = (table.template_pdgs.clone(), table.initial_count);
            let count = pdgs.len();
            // A cross-side candidate: an incoming slot to an outgoing one of the same
            // all-outgoing flavour (quark in -> antiquark out, gluon in -> gluon out).
            let candidate = (0..initial).find_map(|slot| {
                let outgoing = if pdgs[slot] == 21 { 21 } else { -pdgs[slot] };
                (initial..count)
                    .find(|image| pdgs[*image] == outgoing)
                    .map(|image| {
                        let mut permutation = (0..count).collect::<Vec<_>>();
                        permutation.swap(slot, image);
                        permutation
                    })
            });
            if let Some(permutation) = candidate {
                let known = runtime.flavour_union_assignments().unwrap().len();
                let error = runtime.relabel_assignment(0, &permutation).unwrap_err();
                assert_eq!(
                    error.kind(),
                    crate::RusticolErrorKind::Compatibility,
                    "{error}"
                );
                assert!(
                    error
                        .to_string()
                        .contains("full-colour crossing is unsupported"),
                    "{error}"
                );
                assert_eq!(runtime.flavour_union_assignments().unwrap().len(), known);
                refused += 1;
            }
            if !selector {
                continue;
            }
            // A same-side swap of two equal final slots, applied to every assignment.
            let Some((left, right)) = (initial..count).find_map(|left| {
                (left + 1..count)
                    .find(|right| pdgs[*right] == pdgs[left])
                    .map(|right| (left, right))
            }) else {
                continue;
            };
            let mut permutation = (0..count).collect::<Vec<_>>();
            permutation.swap(left, right);
            let helicity_count = runtime.helicity_ids().unwrap().len();
            let helicities = (0..helicity_count as u32).collect::<Vec<_>>();
            let summed = |runtime: &mut NativeRuntime, assignment: u32, point: &[[f64; 4]]| {
                let momenta = signed_momenta(runtime, assignment, point);
                let momenta = momenta.repeat(helicity_count);
                let offsets = (0..=helicity_count).collect::<Vec<_>>();
                let mut values = vec![0.0; helicity_count];
                runtime
                    .evaluate_assignments_f64_into(
                        &momenta,
                        helicity_count,
                        &offsets,
                        &vec![assignment; helicity_count],
                        Some(&helicities),
                        None,
                        AssignmentOutputMode::Total,
                        &mut values,
                    )
                    .unwrap();
                values.iter().sum::<f64>()
            };
            let bases = runtime.flavour_union_assignments().unwrap().len() as u32;
            for base in 0..bases {
                if runtime.flavour_union_assignments_crossed(&[base]) {
                    continue;
                }
                let relabelled = runtime.relabel_assignment(base, &permutation).unwrap();
                let point = member_point(count, 7919 * u64::from(base) + 1);
                let direct = summed(&mut runtime, base, &point);
                let value = summed(&mut runtime, relabelled, &point);
                let relative = (value - direct).abs() / direct.abs().max(value.abs());
                assert!(
                    relative <= 1e-12 && direct != 0.0,
                    "{id} assignment {base}: {value:e} vs {direct:e} ({relative:e})"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared > 0,
        "no selector-only union template was relabelled"
    );
    eprintln!("selector-only relabellings: compared {compared}, cross-side refused {refused}");
}
