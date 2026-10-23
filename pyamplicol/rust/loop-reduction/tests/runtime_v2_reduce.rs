// SPDX-License-Identifier: 0BSD

//! v2 family reductions: true-rank buckets, cached masters and frames.
use rusticol_loop_reduction::runtime::{
    Batch, FamilyGroup, MasterCache, PoleSelection, Poles, PolynomialPlan, ReduceOptions,
    ReductionFrame, V2Reducer, tensor_monomials,
};
use symbolica::domains::float::{Float, RealLike};

/// Squared mass of the heavy outgoing pair; with integer three-momenta of
/// length 400 at energy 500 every invariant is exact in binary64.
const TOP: f64 = 90000.0;

/// Massless incoming and massive outgoing momenta of one 2 -> 2 point.
fn momenta([x, y, z]: [f64; 3]) -> [[f64; 4]; 4] {
    [
        [500., 0., 0., 500.],
        [500., 0., 0., -500.],
        [500., x, y, z],
        [500., -x, -y, -z],
    ]
}

/// One family in loop-cycle order: legs attached before each propagator.
struct Family {
    legs: Vec<Vec<usize>>,
    masses: Vec<f64>,
    rank: usize,
}

fn families() -> Vec<Family> {
    vec![
        // Box with a top line between the two top legs.
        Family {
            legs: vec![vec![0], vec![1], vec![2], vec![3]],
            masses: vec![0., 0., TOP, 0.],
            rank: 3,
        },
        // Unequal-mass (gluon, top) bubbles and a top triangle.
        Family {
            legs: vec![vec![2], vec![0, 1, 3]],
            masses: vec![TOP, 0.],
            rank: 2,
        },
        Family {
            legs: vec![vec![0, 1], vec![2], vec![3]],
            masses: vec![0., TOP, 0.],
            rank: 2,
        },
        Family {
            legs: vec![vec![0, 1], vec![2, 3]],
            masses: vec![TOP, TOP],
            rank: 2,
        },
    ]
}

fn decimal(values: &[f64]) -> Vec<[String; 2]> {
    values
        .iter()
        .map(|x| [format!("{x:e}"), "0".into()])
        .collect()
}

fn complex_decimal(values: &[[f64; 2]]) -> Vec<[String; 2]> {
    values
        .iter()
        .map(|[re, im]| [format!("{re:e}"), format!("{im:e}")])
        .collect()
}

struct Lanes {
    n: usize,
    rank: usize,
    shifts: Batch,
    masses: Batch,
    scales: Batch,
    matrix: Batch,
    coefficients: Batch,
}

fn lanes(family: &Family, points: &[[f64; 3]], ids: &[u64], arithmetic: &str) -> Lanes {
    let n = family.legs.len();
    let width = tensor_monomials(family.rank).len();
    let (mut shifts, mut matrix, mut coefficients) = (vec![], vec![], vec![]);
    for (point, &direction) in points.iter().enumerate() {
        let p = momenta(direction);
        let mut v = [0.0_f64; 4];
        let mut vectors = vec![];
        for legs in &family.legs {
            for &leg in legs {
                let sign = if leg < 2 { 1. } else { -1. };
                for mu in 0..4 {
                    v[mu] += sign * p[leg][mu];
                }
            }
            vectors.push(v);
        }
        // Propagator i follows position i; the loop momentum flows through
        // the last one, whose shift is zero.
        vectors.rotate_right(1);
        vectors[0] = [0.; 4];
        for vector in &vectors {
            shifts.extend_from_slice(vector);
        }
        for a in &vectors {
            for b in &vectors {
                let d: Vec<_> = (0..4).map(|mu| a[mu] - b[mu]).collect();
                matrix.push(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]);
            }
        }
        for k in 0..width {
            let x = (k * 7 + 3 * point + 1) as f64;
            coefficients.push([(x * 0.37).sin() * 10., (x * 0.11).cos()]);
        }
    }
    let rows = points.len();
    let batch = |values: Vec<[String; 2]>, columns| {
        Batch::from_decimal(values, rows, columns, arithmetic, None, Some(ids.to_vec())).unwrap()
    };
    Lanes {
        n,
        rank: family.rank,
        shifts: batch(decimal(&shifts), 4 * n),
        masses: batch(decimal(&family.masses.repeat(rows)), n),
        scales: batch(decimal(&vec![8.0e5; rows]), 1),
        matrix: batch(decimal(&matrix), n * n),
        coefficients: batch(complex_decimal(&coefficients), width),
    }
}

fn group(lanes: &Lanes) -> FamilyGroup<'_> {
    FamilyGroup {
        propagators: lanes.n,
        shifts: &lanes.shifts,
        masses_squared: &lanes.masses,
        scales_squared: &lanes.scales,
        s_matrix: &lanes.matrix,
        coefficients: &lanes.coefficients,
    }
}

fn values(batch: &Batch) -> Vec<[f64; 2]> {
    batch
        .to_decimal()
        .into_iter()
        .map(|[re, im]| [re.parse().unwrap(), im.parse().unwrap()])
        .collect()
}

/// Largest |a - b| per lane, relative to the lane's largest |b| component.
fn relative(a: &[[f64; 2]], b: &[[f64; 2]], columns: usize) -> f64 {
    let norm = |z: &[f64; 2]| z[0].hypot(z[1]);
    a.chunks(columns)
        .zip(b.chunks(columns))
        .map(|(x, y)| {
            let scale = y.iter().map(norm).fold(0., f64::max);
            x.iter()
                .zip(y)
                .map(|(x, y)| norm(&[x[0] - y[0], x[1] - y[1]]))
                .fold(0., f64::max)
                / scale
        })
        .fold(0., f64::max)
}

/// [`relative`] on decimals at 256 bits, for extended-precision results.
fn relative_decimal(a: &Batch, b: &Batch, columns: usize) -> f64 {
    let parse = |batch: &Batch| -> Vec<[Float; 2]> {
        batch
            .to_decimal()
            .iter()
            .map(|z| z.clone().map(|x| Float::parse(&x, Some(256)).unwrap()))
            .collect()
    };
    let norm = |re: &Float, im: &Float| re.to_f64().hypot(im.to_f64());
    let (a, b) = (parse(a), parse(b));
    a.chunks(columns)
        .zip(b.chunks(columns))
        .map(|(x, y)| {
            let scale = y.iter().map(|z| norm(&z[0], &z[1])).fold(0., f64::max);
            x.iter()
                .zip(y)
                .map(|(x, y)| norm(&(x[0].clone() - &y[0]), &(x[1].clone() - &y[1])))
                .fold(0., f64::max)
                / scale
        })
        .fold(0., f64::max)
}

const POINTS: [[f64; 3]; 3] = [[240., 0., 320.], [144., 192., 320.], [320., 0., -240.]];

fn all_lanes(arithmetic: &str) -> Vec<Lanes> {
    families()
        .iter()
        .enumerate()
        .map(|(f, family)| {
            let ids: Vec<u64> = (0..POINTS.len() as u64)
                .map(|p| 10 * p + f as u64)
                .collect();
            lanes(family, &POINTS, &ids, arithmetic)
        })
        .collect()
}

#[test]
fn fast_reductions_match_the_polynomial_plan_and_are_cache_independent() {
    let all = all_lanes("double");
    let groups: Vec<_> = all.iter().map(group).collect();
    let options = ReduceOptions {
        check: Some(ReductionFrame::REVERSED),
        ..ReduceOptions::default()
    };
    let mut cached = V2Reducer::new("double", None, "fast", true).unwrap();
    let mut uncached = V2Reducer::new("double", None, "fast", false).unwrap();
    let output = cached.reduce(&groups, options).unwrap();
    let plain = uncached.reduce(&groups, options).unwrap();
    let statistics = cached.take_master_statistics().unwrap();
    assert!(statistics.total_hits() > 0);
    assert_eq!(
        statistics.total_requests(),
        uncached.take_master_statistics().unwrap().total_evaluated()
    );
    let direct = output.direct;
    let check = output.check.unwrap();
    for (lanes, (index, result)) in all.iter().zip(direct.iter().enumerate()) {
        let result = result.values().unwrap();
        // Bitwise: cached = uncached, and one family alone = all batched.
        assert_eq!(
            values(&result),
            values(&plain.direct[index].values().unwrap())
        );
        assert_eq!(
            values(&check[index].values().unwrap()),
            values(&plain.check.as_ref().unwrap()[index].values().unwrap())
        );
        let alone = V2Reducer::new("double", None, "fast", true)
            .unwrap()
            .reduce(&[group(lanes)], options)
            .unwrap();
        assert_eq!(values(&result), values(&alone.direct[0].values().unwrap()));
        // The reversed frame agrees to reducer accuracy.
        let reversed = values(&check[index].values().unwrap());
        assert!(relative(&reversed, &values(&result), 3) < 1e-12);
        // The existing polynomial path on the same numerator.
        let monomials = tensor_monomials(lanes.rank);
        let names: Vec<_> = (0..monomials.len()).map(|k| format!("c{k}")).collect();
        let numerator = monomials
            .iter()
            .zip(&names)
            .map(|(alpha, name)| {
                let mut term = name.clone();
                for (variable, power) in ["q0", "q1", "q2", "q3"].iter().zip(alpha) {
                    term += &format!("*{variable}^{power}");
                }
                term
            })
            .collect::<Vec<_>>()
            .join("+");
        for engine in ["fast-tensor", "generic"] {
            let mut plan = PolynomialPlan::new(
                lanes.n,
                [numerator.clone(), "0".into(), "0".into()],
                names.clone(),
                "double",
                None,
                engine,
            )
            .unwrap();
            let legacy = plan
                .evaluate_batch(
                    &lanes.shifts,
                    &lanes.masses,
                    &lanes.scales,
                    &lanes.coefficients,
                    Some(&lanes.matrix),
                )
                .unwrap();
            let legacy = values(&legacy.values().unwrap());
            assert!(relative(&values(&result), &legacy, 3) < 1e-12, "{engine}");
        }
    }
}

#[test]
fn frames_true_rank_and_finite_only_mode() {
    let all = all_lanes("double");
    let groups: Vec<_> = all.iter().map(group).collect();
    let mut reducer = V2Reducer::new("double", None, "fast", true).unwrap();
    let direct = reducer.reduce(&groups, ReduceOptions::default()).unwrap();
    assert!(direct.check.is_none());
    for frame in [
        ReductionFrame {
            reverse: false,
            rotation: 1,
            shift: true,
            rescale: false,
        },
        ReductionFrame {
            reverse: true,
            rotation: 2,
            shift: true,
            rescale: true,
        },
        ReductionFrame::REVERSED,
        ReductionFrame::RESCALED,
    ] {
        let output = reducer
            .reduce(
                &groups,
                ReduceOptions {
                    poles: Poles::None,
                    check: Some(frame),
                    ..ReduceOptions::default()
                },
            )
            .unwrap();
        for ((full, finite), check) in direct
            .direct
            .iter()
            .zip(&output.direct)
            .zip(output.check.as_ref().unwrap())
        {
            let full = values(&full.values().unwrap());
            let finite = values(&finite.values().unwrap());
            let finite_from_full: Vec<_> = full.iter().step_by(3).copied().collect();
            assert_eq!(finite, finite_from_full);
            let check = values(&check.values().unwrap());
            assert!(relative(&check, &finite, 1) < 1e-12, "{frame:?}");
        }
    }
    // Zero leading blocks: a rank-3 declaration reduces at the true rank 2.
    let family = &families()[2];
    let lower = lanes(family, &POINTS, &[0, 1, 2], "double");
    let width = tensor_monomials(2).len();
    let padded = tensor_monomials(3).len();
    let mut coefficients = lower.coefficients.to_decimal();
    for row in (0..POINTS.len()).rev() {
        let at = (row + 1) * width;
        coefficients.splice(at..at, vec![["0".into(), "0".into()]; padded - width]);
    }
    let declared = Batch::from_decimal(coefficients, 3, padded, "double", None, None).unwrap();
    let padded_group = FamilyGroup {
        coefficients: &declared,
        ..group(&lower)
    };
    let a = reducer
        .reduce(&[group(&lower)], ReduceOptions::default())
        .unwrap();
    let b = reducer
        .reduce(&[padded_group], ReduceOptions::default())
        .unwrap();
    assert_eq!(
        values(&a.direct[0].values().unwrap()),
        values(&b.direct[0].values().unwrap())
    );
}

#[test]
fn generic_double_double_agrees_with_the_fast_engine() {
    let fast_lanes = all_lanes("double");
    let fast = V2Reducer::new("double", None, "fast", true)
        .unwrap()
        .reduce(
            &fast_lanes.iter().map(group).collect::<Vec<_>>(),
            ReduceOptions::default(),
        )
        .unwrap();
    let dd_lanes = all_lanes("double-double");
    let mut reducer = V2Reducer::new("double-double", None, "generic", true).unwrap();
    assert!(reducer.take_master_statistics().is_none());
    // The rescaled frame re-evaluates every master at scaled inputs and
    // scales the integrals back: all three Laurent coefficients agree.
    for frame in [ReductionFrame::REVERSED, ReductionFrame::RESCALED] {
        let output = reducer
            .reduce(
                &dd_lanes.iter().map(group).collect::<Vec<_>>(),
                ReduceOptions {
                    check: Some(frame),
                    ..ReduceOptions::default()
                },
            )
            .unwrap();
        for ((dd, check), fast) in output
            .direct
            .iter()
            .zip(output.check.as_ref().unwrap())
            .zip(&fast.direct)
        {
            let dd = dd.values().unwrap();
            assert!(relative(&values(&fast.values().unwrap()), &values(&dd), 3) < 1e-12);
            assert!(
                relative_decimal(&check.values().unwrap(), &dd, 3) < 1e-28,
                "{frame:?}"
            );
        }
    }
    assert!(V2Reducer::new("double-double", None, "fast", true).is_err());
}

/// Massless families with scaleless sub-bubbles (p0² = p1² = 0): the
/// fast engine skips those bubbles, whose UV and IR residues cancel.
fn massless_families() -> Vec<Family> {
    vec![
        Family {
            legs: vec![vec![0], vec![1], vec![2, 3]],
            masses: vec![0., 0., 0.],
            rank: 3,
        },
        Family {
            legs: vec![vec![0], vec![1], vec![2], vec![3]],
            masses: vec![0., 0., 0., 0.],
            rank: 4,
        },
        Family {
            legs: vec![vec![0], vec![1, 2, 3]],
            masses: vec![0., 0.],
            rank: 2,
        },
        Family {
            legs: vec![vec![0, 1, 2, 3]],
            masses: vec![TOP],
            rank: 3,
        },
    ]
}

/// Largest deviation of column `a` of `x` from column `b` of `y`, per lane
/// relative to the lane's largest value in `x` or `y` (a scaleless lane's
/// combined Laurent vanishes, its UV and IR residues do not).
fn column_deviation(
    x: &[[f64; 2]],
    a: usize,
    wx: usize,
    y: &[[f64; 2]],
    b: usize,
    wy: usize,
) -> f64 {
    let norm = |z: &[f64; 2]| z[0].hypot(z[1]);
    x.chunks(wx)
        .zip(y.chunks(wy))
        .map(|(x, y)| {
            let scale = x.iter().chain(y).map(norm).fold(0., f64::max);
            norm(&[x[a][0] - y[b][0], x[a][1] - y[b][1]]) / scale
        })
        .fold(0., f64::max)
}

#[test]
fn split_poles_keep_the_combined_columns_and_match_selected_masters() {
    let mut all = all_lanes("double");
    for (f, family) in massless_families().iter().enumerate() {
        let ids: Vec<u64> = (0..POINTS.len() as u64)
            .map(|p| 100 + 10 * p + f as u64)
            .collect();
        all.push(lanes(family, &POINTS, &ids, "double"));
    }
    let groups: Vec<_> = all.iter().map(group).collect();
    let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
    let combined = fast.reduce(&groups, ReduceOptions::default()).unwrap();
    let split = fast
        .reduce(
            &groups,
            ReduceOptions {
                poles: Poles::Split,
                check: Some(ReductionFrame::RESCALED),
                ..ReduceOptions::default()
            },
        )
        .unwrap();
    let mut generic = V2Reducer::new("double", None, "generic", true).unwrap();
    let selected = |generic: &mut V2Reducer, masters| {
        generic
            .reduce(
                &groups,
                ReduceOptions {
                    masters,
                    ..ReduceOptions::default()
                },
            )
            .unwrap()
            .direct
    };
    let uv = selected(&mut generic, PoleSelection::UV);
    let ir = selected(&mut generic, PoleSelection::IR);
    let both = selected(&mut generic, PoleSelection::BOTH);
    let mut largest_uv = 0f64;
    for index in 0..groups.len() {
        let all3 = values(&combined.direct[index].values().unwrap());
        let four = values(&split.direct[index].values().unwrap());
        let check = values(&split.check.as_ref().unwrap()[index].values().unwrap());
        // The combined columns are those of `Poles::All`, bit for bit.
        let firsts: Vec<_> = four.chunks(4).flat_map(|z| z[..3].to_vec()).collect();
        assert_eq!(firsts, all3);
        // The residue against the generic engine's UV- and IR-selected masters.
        let uv = values(&uv[index].values().unwrap());
        let ir = values(&ir[index].values().unwrap());
        let both = values(&both[index].values().unwrap());
        assert!(
            column_deviation(&four, 3, 4, &uv, 1, 3) < 1e-10,
            "group {index} UV"
        );
        let infrared: Vec<_> = four
            .chunks(4)
            .flat_map(|z| [z[0], [z[1][0] - z[3][0], z[1][1] - z[3][1]], z[2]])
            .collect();
        assert!(
            column_deviation(&infrared, 1, 3, &ir, 1, 3) < 1e-10,
            "group {index} IR"
        );
        for row in uv.chunks(3) {
            assert_eq!(row[2], [0., 0.], "the double pole is infrared");
        }
        // Master selection leaves the generic engine's finite part bitwise.
        for ((u, i), b) in uv.chunks(3).zip(ir.chunks(3)).zip(both.chunks(3)) {
            assert_eq!(u[0], b[0]);
            assert_eq!(i[0], b[0]);
        }
        // The rescaled check frame reproduces the residue.
        assert!(
            column_deviation(&check, 3, 4, &four, 3, 4) < 1e-12,
            "group {index} check"
        );
        largest_uv = four
            .chunks(4)
            .map(|z| z[3][0].hypot(z[3][1]))
            .fold(largest_uv, f64::max);
    }
    assert!(largest_uv > 0.);
    // Selected masters need the generic engine; the fast engine refuses them.
    for masters in [PoleSelection::UV, PoleSelection::IR, PoleSelection::NONE] {
        assert!(
            fast.reduce(
                &groups,
                ReduceOptions {
                    masters,
                    ..ReduceOptions::default()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn split_residues_in_double_double_match_selected_masters() {
    let mut all = all_lanes("double-double");
    for (f, family) in massless_families().iter().enumerate() {
        let ids: Vec<u64> = (0..POINTS.len() as u64)
            .map(|p| 100 + 10 * p + f as u64)
            .collect();
        all.push(lanes(family, &POINTS, &ids, "double-double"));
    }
    let groups: Vec<_> = all.iter().map(group).collect();
    let mut reducer = V2Reducer::new("double-double", None, "generic", true).unwrap();
    let split = reducer
        .reduce(
            &groups,
            ReduceOptions {
                poles: Poles::Split,
                ..ReduceOptions::default()
            },
        )
        .unwrap();
    let uv = reducer
        .reduce(
            &groups,
            ReduceOptions {
                masters: PoleSelection::UV,
                ..ReduceOptions::default()
            },
        )
        .unwrap();
    let parse = |batch: &Batch| -> Vec<[Float; 2]> {
        batch
            .to_decimal()
            .iter()
            .map(|z| z.clone().map(|x| Float::parse(&x, Some(256)).unwrap()))
            .collect()
    };
    for (split, uv) in split.direct.iter().zip(&uv.direct) {
        let split = parse(&split.values().unwrap());
        let uv = parse(&uv.values().unwrap());
        for (four, three) in split.chunks(4).zip(uv.chunks(3)) {
            let scale = four
                .iter()
                .chain(three)
                .map(|z| z[0].to_f64().hypot(z[1].to_f64()))
                .fold(0., f64::max);
            let re = (four[3][0].clone() - &three[1][0]).to_f64();
            let im = (four[3][1].clone() - &three[1][1]).to_f64();
            assert!(re.hypot(im) <= 1e-26 * scale, "{re} {im} {scale}");
        }
    }
}

#[test]
fn the_master_cache_forwards_a_selected_providers_capability() {
    use ninjars::fast::{
        C64, FastAmplitude, FastInput, FastMasters, FastOptions, FastResult, FastStatus,
        TensorNumerator,
    };
    use ninjars_symbolica::OneLoopNativeMasters;
    let selected = OneLoopNativeMasters::new()
        .unwrap()
        .with_pole_selection(PoleSelection::UV);
    let mut cache = MasterCache::new(selected, true);
    assert!(cache.requires_scaleless_bubbles());
    let mut amplitude = FastAmplitude::new(2, 0, FastOptions::default()).unwrap();
    let mut numerator = TensorNumerator::new(0, 1);
    numerator.coefficients_mut()[0] = C64::real(1.);
    let mut results = [FastResult::default()];
    let mut statuses = [FastStatus::NotEvaluated];
    let refused = amplitude.evaluate(
        &FastInput {
            lanes: 1,
            shifts: &[[0.; 4], [2., 0., 0., 2.]],
            masses_squared: &[C64::ZERO; 2],
            scales_squared: &[7.],
            s_matrix: Some(&[0.; 4]),
        },
        &mut numerator,
        &mut cache,
        &mut results,
        &mut statuses,
    );
    assert!(refused.is_err());
    assert!(cache.is_empty());
}
