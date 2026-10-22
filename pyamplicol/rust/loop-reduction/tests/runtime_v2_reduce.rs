// SPDX-License-Identifier: 0BSD

//! v2 family reductions: true-rank buckets, cached masters and frames.
use rusticol_loop_reduction::runtime::{
    Batch, FamilyGroup, PolynomialPlan, ReduceOptions, ReductionFrame, V2Reducer, tensor_monomials,
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
        poles: true,
        check: Some(ReductionFrame::REVERSED),
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
                    poles: false,
                    check: Some(frame),
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
                    poles: true,
                    check: Some(frame),
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
