// SPDX-License-Identifier: 0BSD

//! v2 family reductions: true-rank buckets, cached masters and frames.
use rusticol_loop_reduction::runtime::{
    Batch, FamilyGroup, MasterCache, NATIVE_ONSHELL_THRESHOLD, PoleSelection, Poles,
    PolynomialPlan, ReduceOptions, ReductionFrame, V2Output, V2Reducer, fast_engine_threshold,
    reaches_fast_threshold, tensor_monomials,
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
    let points: Vec<_> = points.iter().map(|&direction| momenta(direction)).collect();
    lanes_at(family, &points, ids, arithmetic)
}

/// [`lanes`] at the given external momenta (legs 0 and 1 incoming).
fn lanes_at(family: &Family, points: &[[[f64; 4]; 4]], ids: &[u64], arithmetic: &str) -> Lanes {
    let n = family.legs.len();
    let width = tensor_monomials(family.rank).len();
    let (mut shifts, mut matrix, mut coefficients) = (vec![], vec![], vec![]);
    for (point, &p) in points.iter().enumerate() {
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

// ---------------------------------------------------------------------------
// The fast engine's absolute thresholds (genericity audit 2, G2-M3): lanes
// whose inputs reach NinjaRS 41a4c00's absolute `1e-9` (the masters'
// on-shell snap, the engine's `infrared_eps`) take the generic binary64
// engine; every other lane keeps the fast engine's bits.

/// A keV mass squared in GeV².
const LIGHT: f64 = 1e-12;

/// Families with light internal lines: collinear-singular triangles of
/// light lines between massless legs, a box with a light line between
/// massless legs, and the light bubble of the massless leg 0 (`B0(0; m², m²)
/// = 1/ε - log(m²/μ²)`, scaleless when massless).
fn light_families() -> Vec<Family> {
    vec![
        Family {
            legs: vec![vec![0], vec![1], vec![2, 3]],
            masses: vec![LIGHT; 3],
            rank: 3,
        },
        Family {
            legs: vec![vec![0], vec![1], vec![2, 3]],
            masses: vec![LIGHT, 0., LIGHT],
            rank: 2,
        },
        Family {
            legs: vec![vec![0], vec![1], vec![2], vec![3]],
            masses: vec![0., 0., 0., LIGHT],
            rank: 3,
        },
        Family {
            legs: vec![vec![0], vec![1, 2, 3]],
            masses: vec![LIGHT, LIGHT],
            rank: 2,
        },
    ]
}

/// A box with a light line beside a massive leg: oneloopmaster's binary64
/// C0 of its sub-triangles (`m²/p² ~ 1e-17`) is not finite, in the fast and
/// the generic engine alike; DoubleFloat evaluates it.
fn binary64_failing_family() -> Family {
    Family {
        legs: vec![vec![0], vec![1], vec![2], vec![3]],
        masses: vec![LIGHT, 0., 0., 0.],
        rank: 3,
    }
}

fn family_lanes(families: &[Family], base: u64, arithmetic: &str) -> Vec<Lanes> {
    families
        .iter()
        .enumerate()
        .map(|(f, family)| {
            let ids: Vec<u64> = (0..POINTS.len() as u64)
                .map(|p| base + 10 * p + f as u64)
                .collect();
            lanes(family, &POINTS, &ids, arithmetic)
        })
        .collect()
}

/// `lanes` in units `x` times GeV: every dimensionful input of the exact GeV
/// lanes scaled once (shifts by `x`; squared masses, scale and invariants by
/// `x²`; degree-`d` coefficients by `x^-d`), so that an invariant equal to a
/// mass stays equal to it. The integrals scale as `x^(4 - 2N)`.
fn in_units(lanes: &Lanes, x: f64) -> Lanes {
    let rows = lanes.coefficients.rows();
    let ids = lanes.coefficients.ids().to_vec();
    let scaled = |batch: &Batch, factor: &dyn Fn(usize) -> f64| {
        let columns = batch.columns();
        let values: Vec<[String; 2]> = values(batch)
            .iter()
            .enumerate()
            .map(|(k, [re, im])| {
                let f = factor(k % columns);
                [format!("{:e}", re * f), format!("{:e}", im * f)]
            })
            .collect();
        Batch::from_decimal(values, rows, columns, "double", None, Some(ids.clone())).unwrap()
    };
    let degrees: Vec<i32> = tensor_monomials(lanes.rank)
        .iter()
        .map(|alpha| alpha.iter().sum::<u32>() as i32)
        .collect();
    Lanes {
        n: lanes.n,
        rank: lanes.rank,
        shifts: scaled(&lanes.shifts, &|_| x),
        masses: scaled(&lanes.masses, &|_| x * x),
        scales: scaled(&lanes.scales, &|_| x * x),
        matrix: scaled(&lanes.matrix, &|_| x * x),
        coefficients: scaled(&lanes.coefficients, &|k| x.powi(-degrees[k])),
    }
}

fn reduce(reducer: &mut V2Reducer, all: &[Lanes], check: ReductionFrame) -> V2Output {
    reducer
        .reduce(
            &all.iter().map(group).collect::<Vec<_>>(),
            ReduceOptions {
                check: Some(check),
                ..ReduceOptions::default()
            },
        )
        .unwrap()
}

fn statuses(batches: &[rusticol_loop_reduction::runtime::ReductionBatch]) -> Vec<Vec<String>> {
    batches
        .iter()
        .map(|b| b.statuses().iter().map(|(_, s)| s.clone()).collect())
        .collect()
}

#[test]
fn the_pinned_native_masters_snap_at_the_routing_threshold() {
    use ninjars::fast::{C64, FastMaster, FastMasters, FastOptions};
    use ninjars_symbolica::OneLoopNativeMasters;
    // The routing threshold is the largest absolute constant of the fast path.
    assert_eq!(NATIVE_ONSHELL_THRESHOLD, 1e-9);
    assert_eq!(
        FastOptions::default().infrared_eps,
        NATIVE_ONSHELL_THRESHOLD
    );
    assert_eq!(fast_engine_threshold(), NATIVE_ONSHELL_THRESHOLD);
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut a0 = |m2: f64| {
        let (mut out, mut ok) = ([[C64::ZERO; 3]], [false]);
        masters
            .evaluate(
                FastMaster::A0,
                &[C64::real(m2), C64::real(1e4)],
                &mut out,
                &mut ok,
            )
            .unwrap();
        assert!(ok[0]);
        out[0]
    };
    // Below the threshold the provider evaluates the massless tadpole (0),
    // at and above it the given mass.
    let zero = a0(0.);
    assert_eq!(a0(0.999e-9), zero);
    assert_ne!(a0(1e-9), zero);
    assert_ne!(a0(LIGHT * 1e6), zero);
}

#[test]
fn the_routing_rule_compares_the_given_inputs_exactly() {
    let c = |re: f64, im: f64| rusticol_loop_reduction::Complex::new(re, im);
    let t = NATIVE_ONSHELL_THRESHOLD;
    let s = |entries: [f64; 4]| entries.map(|x| c(x, 0.)).to_vec();
    let route = |masses: &[f64; 2], matrix: [f64; 4]| {
        reaches_fast_threshold(2, &masses.map(|m| c(m, 0.)), &s(matrix), t)
    };
    // Exact zeros, invariants equal to a mass and generic values never route.
    assert!(!route(&[0., 0.], [0., 0., 0., 0.]));
    assert!(!route(&[TOP, 0.], [0., TOP, TOP, 0.]));
    assert!(!route(&[TOP, 1e-9], [0., 2.5e5, 2.5e5, 0.]));
    // A light mass, a light invariant, an invariant near a mass.
    assert!(route(&[LIGHT, 0.], [0., 2.5e5, 2.5e5, 0.]));
    assert!(route(&[0., 0.], [0., 5e-10, 5e-10, 0.]));
    assert!(route(&[TOP, 0.], [0., TOP + 1e-11, TOP + 1e-11, 0.]));
    // The diagonal is not an invariant; a width alone is a squared mass.
    assert!(!route(&[0., 0.], [5e-10, 0., 0., 5e-10]));
    assert!(reaches_fast_threshold(
        2,
        &[c(0., -5e-10), c(0., 0.)],
        &s([0., 0., 0., 0.]),
        t
    ));
    // A complex mass is never an invariant's on-shell value.
    assert!(!reaches_fast_threshold(
        2,
        &[c(TOP, -1.), c(0., 0.)],
        &s([0., TOP + 1e-11, TOP + 1e-11, 0.]),
        t
    ));
}

#[test]
fn light_internal_masses_take_the_exact_engine_and_agree_with_double_double() {
    let light = family_lanes(&light_families(), 300, "double");
    let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
    let routed = reduce(&mut fast, &light, ReductionFrame::RESCALED);
    // Every light lane, in both frames.
    assert_eq!(
        fast.take_exact_lanes(),
        2 * (light.len() * POINTS.len()) as u64
    );
    assert_eq!(fast.take_exact_lanes(), 0);
    let generic = reduce(
        &mut V2Reducer::new("double", None, "generic", true).unwrap(),
        &light,
        ReductionFrame::RESCALED,
    );
    let dd_lanes = family_lanes(&light_families(), 300, "double-double");
    let dd = reduce(
        &mut V2Reducer::new("double-double", None, "generic", true).unwrap(),
        &dd_lanes,
        ReductionFrame::RESCALED,
    );
    // The same engine's bits, and DoubleFloat to binary64 accuracy.
    for index in 0..light.len() {
        let mine = values(&routed.direct[index].values().unwrap());
        assert_eq!(mine, values(&generic.direct[index].values().unwrap()));
        let check = values(&routed.check.as_ref().unwrap()[index].values().unwrap());
        assert_eq!(
            check,
            values(&generic.check.as_ref().unwrap()[index].values().unwrap())
        );
        let reference = values(&dd.direct[index].values().unwrap());
        let error = relative(&mine, &reference, 3);
        assert!(error < 1e-12, "family {index}: {error:e}");
    }
    // What the snap computed: the massless lines' integrals, an O(1) change
    // where the light lines regulate a mass singularity (all but the box,
    // whose light line joins the two massive legs).
    let singular = [true, true, false, true];
    let massless: Vec<Family> = light_families()
        .into_iter()
        .map(|f| Family {
            masses: f
                .masses
                .iter()
                .map(|&m| if m == LIGHT { 0. } else { m })
                .collect(),
            ..f
        })
        .collect();
    let snapped = reduce(
        &mut fast,
        &family_lanes(&massless, 300, "double"),
        ReductionFrame::RESCALED,
    );
    assert_eq!(fast.take_exact_lanes(), 0);
    for index in (0..light.len()).filter(|&index| singular[index]) {
        let reference = values(&dd.direct[index].values().unwrap());
        let wrong = values(&snapped.direct[index].values().unwrap());
        assert!(relative(&wrong, &reference, 3) > 1e-3, "family {index}");
    }
}

#[test]
fn fast_lanes_keep_their_bits_beside_routed_lanes() {
    let heavy = all_lanes("double");
    let mut alone = V2Reducer::new("double", None, "fast", true).unwrap();
    let reference = reduce(&mut alone, &heavy, ReductionFrame::REVERSED);
    assert_eq!(alone.take_exact_lanes(), 0);
    let mut mixed_lanes = all_lanes("double");
    mixed_lanes.extend(family_lanes(&light_families(), 300, "double"));
    let mut mixed = V2Reducer::new("double", None, "fast", true).unwrap();
    let output = reduce(&mut mixed, &mixed_lanes, ReductionFrame::REVERSED);
    assert!(mixed.take_exact_lanes() > 0);
    for index in 0..heavy.len() {
        assert_eq!(
            values(&output.direct[index].values().unwrap()),
            values(&reference.direct[index].values().unwrap())
        );
        assert_eq!(
            values(&output.check.as_ref().unwrap()[index].values().unwrap()),
            values(&reference.check.as_ref().unwrap()[index].values().unwrap())
        );
        assert_eq!(
            output.direct[index].statuses(),
            reference.direct[index].statuses()
        );
    }
}

/// Unit scaling: the same lanes in TeV, GeV and MeV (every dimensionful input
/// scaled by `x = 1e-3, 1, 1e3`) give the same statuses and integrals scaled
/// by `x^(4 - 2N)`, in both check frames. The light lanes route in TeV and
/// GeV (`m² = 1e-18, 1e-12`) and take the fast engine in MeV (`1e-6`): the
/// absolute thresholds no longer change the integrals.
#[test]
fn integrals_and_statuses_scale_with_the_units() {
    let mut gev = all_lanes("double");
    gev.extend(family_lanes(&massless_families(), 100, "double"));
    gev.extend(family_lanes(&light_families(), 300, "double"));
    let light = (gev.len() - light_families().len()..gev.len()).collect::<Vec<_>>();
    for frame in [ReductionFrame::REVERSED, ReductionFrame::RESCALED] {
        let mut reducer = V2Reducer::new("double", None, "fast", true).unwrap();
        let reference = reduce(&mut reducer, &gev, frame);
        let routed_in_gev = reducer.take_exact_lanes();
        assert_eq!(routed_in_gev, 2 * (light.len() * POINTS.len()) as u64);
        for (x, routed) in [(1e-3, routed_in_gev), (1e3, 0)] {
            let units: Vec<Lanes> = gev.iter().map(|l| in_units(l, x)).collect();
            let output = reduce(&mut reducer, &units, frame);
            assert_eq!(reducer.take_exact_lanes(), routed, "x = {x}");
            assert_eq!(statuses(&output.direct), statuses(&reference.direct));
            for (index, lanes) in gev.iter().enumerate() {
                let back = x.powi(2 * lanes.n as i32 - 4);
                for (got, want) in [
                    (&output.direct[index], &reference.direct[index]),
                    (
                        &output.check.as_ref().unwrap()[index],
                        &reference.check.as_ref().unwrap()[index],
                    ),
                ] {
                    let got: Vec<[f64; 2]> = values(&got.values().unwrap())
                        .iter()
                        .map(|[re, im]| [re * back, im * back])
                        .collect();
                    let error = relative(&got, &values(&want.values().unwrap()), 3);
                    assert!(
                        error < 1e-11,
                        "x = {x}, {frame:?}, group {index}: {error:e}"
                    );
                }
            }
        }
    }
}

/// Unit scaling with exactly scaled inputs: in units of a power of two
/// (`x = 2^-10`, `2^10`) every dimensionful input scales without rounding,
/// so the routing and every status are compared on exactly the GeV inputs
/// times `x` (the decisions scale exactly), and the integrals times
/// `x^(2N - 4)` measure what the reduction itself does with the units. They
/// are not bitwise: the binary64 reduction commutes with the scaling only up
/// to rounding (measured on the pinned NinjaRS and oneloopmaster: 43 of the 96
/// comparisons bitwise, at most 2.3e-13 relative, on the massless rank-4
/// box), and no absolute constant is crossed (that would be an O(1) change,
/// as the massless integrals of the light lanes are). `2^-10` keeps the light lanes
/// routed (`m² ≈ 1e-18`); `2^10` takes them off the threshold (`m² ≈ 1e-6`)
/// onto the fast engine, which then agrees with the generic one.
#[test]
fn power_of_two_units_scale_the_decisions_exactly() {
    let mut gev = all_lanes("double");
    gev.extend(family_lanes(&massless_families(), 100, "double"));
    gev.extend(family_lanes(&light_families(), 300, "double"));
    let first_light = gev.len() - light_families().len();
    for frame in [ReductionFrame::REVERSED, ReductionFrame::RESCALED] {
        let mut reducer = V2Reducer::new("double", None, "fast", true).unwrap();
        let reference = reduce(&mut reducer, &gev, frame);
        let routed_in_gev = reducer.take_exact_lanes();
        assert_eq!(
            routed_in_gev,
            2 * ((gev.len() - first_light) * POINTS.len()) as u64
        );
        for (exponent, routed) in [(-10, routed_in_gev), (10, 0)] {
            let x = 2f64.powi(exponent);
            let units: Vec<Lanes> = gev.iter().map(|l| in_units(l, x)).collect();
            let output = reduce(&mut reducer, &units, frame);
            assert_eq!(reducer.take_exact_lanes(), routed, "x = 2^{exponent}");
            assert_eq!(statuses(&output.direct), statuses(&reference.direct));
            assert_eq!(
                statuses(output.check.as_ref().unwrap()),
                statuses(reference.check.as_ref().unwrap())
            );
            for (index, lanes) in gev.iter().enumerate() {
                let back = x.powi(2 * lanes.n as i32 - 4);
                for (got, want) in [
                    (&output.direct[index], &reference.direct[index]),
                    (
                        &output.check.as_ref().unwrap()[index],
                        &reference.check.as_ref().unwrap()[index],
                    ),
                ] {
                    let got: Vec<[f64; 2]> = values(&got.values().unwrap())
                        .iter()
                        .map(|[re, im]| [re * back, im * back])
                        .collect();
                    let error = relative(&got, &values(&want.values().unwrap()), 3);
                    assert!(
                        error < 1e-12,
                        "x = 2^{exponent}, {frame:?}, group {index}: {error:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_routed_lane_fails_as_the_generic_engine_does() {
    let family = binary64_failing_family();
    let statuses_of = |engine: &str, arithmetic: &str| {
        let lanes = lanes(&family, &POINTS, &[0, 1, 2], arithmetic);
        let mut reducer = V2Reducer::new(arithmetic, None, engine, true).unwrap();
        let out = reduce(&mut reducer, &[lanes], ReductionFrame::RESCALED);
        (statuses(&out.direct), statuses(out.check.as_ref().unwrap()))
    };
    let routed = statuses_of("fast", "double");
    assert_eq!(routed, statuses_of("generic", "double"));
    assert!(
        routed.0[0].iter().all(|s| s.contains("NonFiniteResult")),
        "{routed:?}"
    );
    let extended = statuses_of("generic", "double-double");
    assert!(
        extended.0[0]
            .iter()
            .chain(&extended.1[0])
            .all(|s| s == "success")
    );
}

/// The tensor numerator `Σ_k c_k q^α_k` of rank `rank` and its parameters.
fn tensor_numerator(rank: usize) -> ([String; 3], Vec<String>) {
    let monomials = tensor_monomials(rank);
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
    ([numerator, "0".into(), "0".into()], names)
}

fn legacy_plan(lanes: &Lanes, engine: &str) -> PolynomialPlan {
    let (numerator, names) = tensor_numerator(lanes.rank);
    PolynomialPlan::new(lanes.n, numerator, names, "double", None, engine).unwrap()
}

fn legacy_reduce(
    plan: &mut PolynomialPlan,
    lanes: &Lanes,
    matrix: bool,
) -> rusticol_loop_reduction::runtime::ReductionBatch {
    plan.evaluate_batch(
        &lanes.shifts,
        &lanes.masses,
        &lanes.scales,
        &lanes.coefficients,
        matrix.then_some(&lanes.matrix),
    )
    .unwrap()
}

/// Two lane sets of one `(n, rank)` as one call, `a`'s lanes first.
fn concatenated(a: &Lanes, b: &Lanes) -> Lanes {
    assert_eq!((a.n, a.rank), (b.n, b.rank));
    let join = |x: &Batch, y: &Batch| {
        let mut ids = x.ids().to_vec();
        ids.extend_from_slice(y.ids());
        let mut entries = complex_decimal(&values(x));
        entries.extend(complex_decimal(&values(y)));
        Batch::from_decimal(
            entries,
            x.rows() + y.rows(),
            x.columns(),
            "double",
            None,
            Some(ids),
        )
        .unwrap()
    };
    Lanes {
        n: a.n,
        rank: a.rank,
        shifts: join(&a.shifts, &b.shifts),
        masses: join(&a.masses, &b.masses),
        scales: join(&a.scales, &b.scales),
        matrix: join(&a.matrix, &b.matrix),
        coefficients: join(&a.coefficients, &b.coefficients),
    }
}

/// The legacy engine's binary64 fast plans (`PolynomialPlan` `fast` and
/// `fast-tensor`, kept as this crate's test oracle after the legacy engine's
/// removal) route by the same rule: light
/// lanes give the legacy generic binary64 plan's bits with the invariant
/// matrix, not the massless lines' integrals the snap computed, and are
/// refused without it (`light_mass_needs_invariants`; the next test); the
/// lanes of an ordinary family keep the fast plan's bits beside them in one
/// call, and a call without a light lane routes nothing.
#[test]
fn the_legacy_fast_plans_route_light_lanes_to_their_generic_plan() {
    use rusticol_loop_reduction::runtime::light_mass_needs_invariants;
    // The refusal names the threshold the rule compares with.
    let refused = light_mass_needs_invariants(fast_engine_threshold());
    assert!(
        refused.contains("absolute threshold 1e-9 needs"),
        "{refused}"
    );
    let light = family_lanes(&light_families(), 300, "double");
    let massless: Vec<Family> = light_families()
        .into_iter()
        .map(|f| Family {
            masses: f
                .masses
                .iter()
                .map(|&m| if m == LIGHT { 0. } else { m })
                .collect(),
            ..f
        })
        .collect();
    let massless = family_lanes(&massless, 300, "double");
    let singular = [true, true, false, true];
    let dd = family_lanes(&light_families(), 300, "double-double");
    let mut reference = V2Reducer::new("double-double", None, "generic", true).unwrap();
    let dd = reduce(&mut reference, &dd, ReductionFrame::REVERSED);
    for (index, lanes) in light.iter().enumerate() {
        let want = values(&dd.direct[index].values().unwrap());
        let generic = values(
            &legacy_reduce(&mut legacy_plan(lanes, "generic"), lanes, true)
                .values()
                .unwrap(),
        );
        for matrix in [true, false] {
            for engine in ["fast", "fast-tensor"] {
                let mut plan = legacy_plan(lanes, engine);
                let reduced = legacy_reduce(&mut plan, lanes, matrix);
                if matrix {
                    let got = values(&reduced.values().unwrap());
                    assert_eq!(plan.take_exact_lanes(), POINTS.len() as u64, "{engine}");
                    assert_eq!(got, generic, "family {index}, {engine}");
                    let error = relative(&got, &want, 3);
                    assert!(error < 1e-12, "family {index}, {engine}: {error:e}");
                } else {
                    // Without the invariants a light lane is refused.
                    assert_eq!(plan.take_exact_lanes(), 0, "{engine}");
                    assert!(
                        reduced
                            .statuses()
                            .iter()
                            .all(|(_, status)| *status == refused),
                        "family {index}, {engine}"
                    );
                    assert_eq!(reduced.successful_values().rows(), 0);
                }
                if singular[index] {
                    let snapped = values(
                        &legacy_reduce(&mut plan, &massless[index], matrix)
                            .values()
                            .unwrap(),
                    );
                    assert_eq!(plan.take_exact_lanes(), 0);
                    assert!(relative(&snapped, &want, 3) > 1e-3, "family {index}");
                }
            }
        }
    }
    // An ordinary box beside the light box in one call.
    let heavy = &all_lanes("double")[0];
    let mixed = concatenated(heavy, &light[2]);
    for engine in ["fast", "fast-tensor"] {
        let mut alone = legacy_plan(heavy, engine);
        let reference = values(&legacy_reduce(&mut alone, heavy, true).values().unwrap());
        assert_eq!(alone.take_exact_lanes(), 0);
        let mut plan = legacy_plan(&mixed, engine);
        let got = legacy_reduce(&mut plan, &mixed, true);
        assert_eq!(plan.take_exact_lanes(), POINTS.len() as u64);
        let ids: Vec<u64> = got.statuses().iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, mixed.coefficients.ids());
        let got = values(&got.values().unwrap());
        assert_eq!(got[..reference.len()], reference[..], "{engine}");
        let light_generic = values(
            &legacy_reduce(&mut legacy_plan(&light[2], "generic"), &light[2], true)
                .values()
                .unwrap(),
        );
        assert_eq!(got[reference.len()..], light_generic[..], "{engine}");
    }
}

/// The legacy fast plans without an invariant matrix (`evaluate_batch` with
/// `s_matrix = None`, `evaluate_decimal`) form the invariants from the
/// binary64 shifts: a massless leg at a generic angle gives a rounding
/// `p²` in `(0, 1e-9)` (here `-7.3e-11`, `E = 500`), which the fast path's
/// snap sets to zero. Such calls route on the squared masses alone: the
/// massless box keeps the fast plan (no lane routed) and equals the generic
/// plan given the exact invariants (the massless legs on their shells); the
/// generic plan given the rounded `p²` instead computes a slightly off-shell
/// box (it fails or is O(1) off). A light internal line is refused
/// (`light_mass_needs_invariants`): rounded invariants cannot resolve it;
/// given the exact invariants it is routed and gives the generic plan's
/// outcome (here its binary64 C0's failure), and beside it in one call the
/// massless lane keeps its bits.
#[test]
fn the_legacy_fast_plans_without_invariants_route_on_the_squared_masses() {
    use rusticol_loop_reduction::runtime::light_mass_needs_invariants;
    let energy = 500_f64;
    let (theta, phi) = (1.1_f64, 0.4_f64);
    let p3 = [
        energy,
        energy * theta.sin() * phi.cos(),
        energy * theta.sin() * phi.sin(),
        energy * theta.cos(),
    ];
    let point = [
        [energy, 0., 0., energy],
        [energy, 0., 0., -energy],
        p3,
        [energy, -p3[1], -p3[2], -p3[3]],
    ];
    let lanes_of = |masses: Vec<f64>, id: u64| {
        let family = Family {
            legs: vec![vec![0], vec![1], vec![2], vec![3]],
            masses,
            rank: 3,
        };
        let mut lanes = lanes_at(&family, &[point], &[id], "double");
        let n = family.legs.len();
        let rounded = values(&lanes.matrix);
        // The precondition: an invariant formed from the shifts is rounding
        // inside the thresholds.
        assert!(rounded.iter().any(|[re, _]| *re != 0. && re.abs() < 1e-9));
        let mut exact = rounded.clone();
        for i in 0..n {
            for j in [(i + 1) % n, (i + n - 1) % n] {
                exact[i * n + j] = [0., 0.];
            }
        }
        let rounded_lanes = Batch::from_decimal(
            complex_decimal(&rounded),
            1,
            n * n,
            "double",
            None,
            Some(vec![id]),
        )
        .unwrap();
        lanes.matrix = Batch::from_decimal(
            complex_decimal(&exact),
            1,
            n * n,
            "double",
            None,
            Some(vec![id]),
        )
        .unwrap();
        (lanes, rounded_lanes)
    };
    let (massless, rounded) = lanes_of(vec![0.; 4], 11);
    let exact_generic = values(
        &legacy_reduce(&mut legacy_plan(&massless, "generic"), &massless, true)
            .values()
            .unwrap(),
    );
    // What routing on the formed invariants did (the re-audit of
    // 2026-10-28): the generic plan at the rounded p² fails or is O(1) off.
    let off_shell = legacy_plan(&massless, "generic")
        .evaluate_batch(
            &massless.shifts,
            &massless.masses,
            &massless.scales,
            &massless.coefficients,
            Some(&rounded),
        )
        .unwrap();
    eprintln!(
        "the generic plan at the rounded invariants: {:?}",
        off_shell.statuses()
    );
    if let Ok(off_shell) = off_shell.values() {
        let off = relative(&values(&off_shell), &exact_generic, 3);
        eprintln!("{off:e} off");
        assert!(off > 1e-2, "{off:e}");
    }
    for engine in ["fast", "fast-tensor"] {
        let mut plan = legacy_plan(&massless, engine);
        let given = values(&legacy_reduce(&mut plan, &massless, true).values().unwrap());
        let formed = legacy_reduce(&mut plan, &massless, false);
        assert_eq!(plan.take_exact_lanes(), 0, "{engine}");
        assert_eq!(formed.statuses(), [(11, "success".to_string())], "{engine}");
        let formed = values(&formed.values().unwrap());
        let error = relative(&formed, &exact_generic, 3);
        assert!(error < 1e-12, "{engine}: {error:e}");
        assert!(relative(&given, &exact_generic, 3) < 1e-12, "{engine}");
    }
    let (light, _) = lanes_of(vec![0., 0., 0., LIGHT], 12);
    // The generic plan's outcome, a failure of its binary64 C0 here
    // (`MasterFailed`), which the routed lane reproduces.
    let generic = legacy_reduce(&mut legacy_plan(&light, "generic"), &light, true);
    for engine in ["fast", "fast-tensor"] {
        let mut plan = legacy_plan(&light, engine);
        let refused = legacy_reduce(&mut plan, &light, false);
        assert_eq!(plan.take_exact_lanes(), 0, "{engine}");
        assert_eq!(
            refused.statuses(),
            [(12, light_mass_needs_invariants(fast_engine_threshold()))],
            "{engine}"
        );
        assert_eq!(refused.successful_values().rows(), 0);
        let given = legacy_reduce(&mut plan, &light, true);
        assert_eq!(plan.take_exact_lanes(), 1, "{engine}");
        assert_eq!(given.statuses(), generic.statuses(), "{engine}");
        assert_eq!(
            values(&given.successful_values()),
            values(&generic.successful_values()),
            "{engine}"
        );
        // In one call beside the massless lane, which keeps its bits.
        let both = concatenated(&massless, &light);
        let mut plan = legacy_plan(&both, engine);
        let alone = values(
            &legacy_reduce(&mut legacy_plan(&massless, engine), &massless, false)
                .values()
                .unwrap(),
        );
        let mixed = legacy_reduce(&mut plan, &both, false);
        assert_eq!(plan.take_exact_lanes(), 0, "{engine}");
        assert_eq!(
            mixed.statuses(),
            [
                (11, "success".to_string()),
                (12, light_mass_needs_invariants(fast_engine_threshold()))
            ],
            "{engine}"
        );
        assert_eq!(values(&mixed.successful_values()), alone, "{engine}");
    }
}

/// Coefficients of `Π_j (q·a_j + b_j)` (Minkowski product, rank = the number
/// of factors) in the tensor layout of `tensor_monomials(rank)`.
fn product_numerator(factors: &[([f64; 4], f64)]) -> Vec<[f64; 2]> {
    use std::collections::HashMap;
    let mut polynomial: HashMap<[u32; 4], f64> = HashMap::from([([0; 4], 1.)]);
    for (a, b) in factors {
        let mut next: HashMap<[u32; 4], f64> = HashMap::new();
        for (alpha, c) in &polynomial {
            *next.entry(*alpha).or_default() += c * b;
            for mu in 0..4 {
                let mut beta = *alpha;
                beta[mu] += 1;
                let metric = if mu == 0 { 1. } else { -1. };
                *next.entry(beta).or_default() += c * metric * a[mu];
            }
        }
        polynomial = next;
    }
    tensor_monomials(factors.len())
        .iter()
        .map(|alpha| [polynomial.get(alpha).copied().unwrap_or(0.), 0.])
        .collect()
}

/// A rotation of the three-vector part (about x by 0.3, then about y by 0.7).
fn rotated(p: &[f64; 4]) -> [f64; 4] {
    let (c, s) = (0.3_f64.cos(), 0.3_f64.sin());
    let (y, z) = (c * p[2] - s * p[3], s * p[2] + c * p[3]);
    let (c, s) = (0.7_f64.cos(), 0.7_f64.sin());
    let (x, z) = (c * p[1] + s * z, -s * p[1] + c * z);
    [p[0], x, y, z]
}

/// Ninja's light-like spinors (`ninjars fast/geometry.rs:62-104` at 41a4c00;
/// DESIGN §11.2 'Other thresholds on the binary64 path'). The fast engine
/// builds each cut basis' complex vectors from the spinors of two light-like
/// vectors: within `infrared_eps` (absolute, `1e-9` in the model's units) of
/// `p⁰ = ±p³` it takes the beam axis' spinor and drops the transverse
/// momentum, and outside it divides by `√(p⁰ + p³)`, formed by cancellation
/// near the `-z` axis (relative error ≈ `2.2e-16/θ²` at an angle `θ` from
/// it). Neither is routed: no mass or invariant is near a threshold. A
/// massless rank-3 box with beams along `x` and the outgoing legs at `θ` from
/// the `z` axis (`s = 4E²`, `t ≈ u ≈ -2E²`, `E = 500`; the branch for
/// `θ < 2e-6`) against the same lanes rotated (momenta and the numerator's
/// four-vectors): the generic engine agrees with DoubleFloat in both frames,
/// the fast engine only in the rotated one, and its error is silent (the lane
/// succeeds) and as large as the two frames' difference, so a rotation check
/// sees it; the reduction-direction check differs from the direct frame by
/// more than half the error. Pins the pinned NinjaRS's behaviour (the upstream note
/// `workspace/reports/engine-v2/upstream/ninjars-lightlike-spinor/`).
#[test]
fn a_rotation_check_sees_ninjas_lightlike_spinor_errors() {
    let energy = 500_f64;
    let factors = [
        ([1.3, 0.2, -0.7, 0.4], 2.0e2),
        ([0.5, -1.1, 0.3, 0.9], -1.5e2),
        ([0.8, 0.6, 1.2, -0.3], 0.7e2),
    ];
    let family = Family {
        legs: vec![vec![0], vec![1], vec![2], vec![3]],
        masses: vec![0.; 4],
        rank: 3,
    };
    // (θ, the fast engine's least error): the cancellation (1e-3, 1e-5) and
    // the branch (1e-6, 1e-8; measured 7.1e-10, 6.2e-6, 8.8e-6, 8.8e-8).
    for (angle, least) in [(1e-3, 1e-10), (1e-5, 1e-6), (1e-6, 1e-6), (1e-8, 1e-8)] {
        let point = [
            [energy, energy, 0., 0.],
            [energy, -energy, 0., 0.],
            [
                energy,
                energy * f64::sin(angle),
                0.,
                energy * f64::cos(angle),
            ],
            [
                energy,
                -energy * f64::sin(angle),
                0.,
                -energy * f64::cos(angle),
            ],
        ];
        let frame = |rotate: bool, arithmetic: &str| {
            let r = |p: &[f64; 4]| if rotate { rotated(p) } else { *p };
            let momenta = point.map(|p| r(&p));
            let mut lanes = lanes_at(&family, &[momenta], &[7], arithmetic);
            // The massless legs on their shells exactly, as the executor
            // binds them (DESIGN §11.1): no invariant near a threshold.
            let n = family.legs.len();
            let mut matrix = values(&lanes.matrix);
            for i in 0..n {
                for j in [(i + 1) % n, (i + n - 1) % n] {
                    matrix[i * n + j] = [0., 0.];
                }
            }
            let batch = |entries: &[[f64; 2]]| {
                Batch::from_decimal(
                    complex_decimal(entries),
                    1,
                    entries.len(),
                    arithmetic,
                    None,
                    Some(vec![7]),
                )
                .unwrap()
            };
            lanes.matrix = batch(&matrix);
            lanes.coefficients = batch(&product_numerator(&factors.map(|(a, b)| (r(&a), b))));
            lanes
        };
        let run = |engine: &str, arithmetic: &str, rotate: bool| {
            let lanes = frame(rotate, arithmetic);
            let mut reducer = V2Reducer::new(arithmetic, None, engine, true).unwrap();
            let out = reduce(&mut reducer, &[lanes], ReductionFrame::REVERSED);
            assert_eq!(reducer.take_exact_lanes(), 0);
            assert_eq!(
                statuses(&out.direct),
                [["success"]],
                "{engine} {arithmetic}"
            );
            (
                values(&out.direct[0].values().unwrap()),
                values(&out.check.as_ref().unwrap()[0].values().unwrap()),
            )
        };
        let (reference, _) = run("generic", "double-double", false);
        let error = |x: &[[f64; 2]]| relative(x, &reference, 3);
        for rotate in [false, true] {
            let (generic, _) = run("generic", "double", rotate);
            assert!(
                error(&generic) < 1e-13,
                "θ = {angle:e}: {:e}",
                error(&generic)
            );
        }
        let (fast, reversed) = run("fast", "double", false);
        let (fast_rotated, _) = run("fast", "double", true);
        let (wrong, seen) = (error(&fast), relative(&fast, &fast_rotated, 3));
        assert!(error(&fast_rotated) < 1e-13, "θ = {angle:e}");
        assert!(wrong > least, "θ = {angle:e}: {wrong:e}");
        assert!(seen > 0.5 * wrong && seen < 2. * wrong, "θ = {angle:e}");
        // The reduction-direction check sees it too (measured 1.29 to 2.26
        // times the error).
        let direction = relative(&fast, &reversed, 3);
        assert!(direction > 0.5 * wrong, "θ = {angle:e}: {direction:e}");
        eprintln!(
            "θ = {angle:e}: fast {wrong:e}, rotation check {seen:e}, \
             reduction-direction check {direction:e}"
        );
    }
}
