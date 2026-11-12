// SPDX-License-Identifier: 0BSD

//! v2 family reductions at eight propagators (M-FINAL MF-8G-0 (iii)): the
//! adapter at the loop size of g g > 6g, which NinjaRS reduces at any N
//! (NinjaRS PR #13, `crates/ninjars-symbolica/tests/n_point.rs`).
//!
//! An 8-propagator massless family on massless 2 -> 6 phase-space points
//! (every leg on one vertex, light-like invariants exactly zero as the
//! executor passes them) with a dense rank-8 numerator: the fast engine, its
//! reversed propagator frame and the generic DoubleFloat engine agree, and
//! the pinch identity holds (an 8-point numerator `D_j(q) T(q)` reduces to
//! the 7-point reduction of `T` without line `j`; `T` has rank 6, so the
//! four-dimensional `D_j` differs from the D-dimensional one by `mu^2`, whose
//! integral `I_8[mu^2 T]` has no rational part below rank 10).
//! `generic_chunk_memory_at_eight_propagators` (ignored; run it alone with
//! `--ignored --nocapture --test-threads=1`) measures the generic engines'
//! workspace per lane, which sizes `GENERIC_CHUNK_LANES`.
use rusticol_loop_reduction::runtime::{
    Batch, FamilyGroup, ReduceOptions, ReductionFrame, V2Reducer, tensor_monomials,
};
use std::collections::HashMap;

/// Deterministic xorshift64* stream in [0, 1).
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
}

type Momentum = [f64; 4];

/// RAMBO: `outgoing` massless momenta at total energy `energy` in the
/// centre-of-mass frame, after the two incoming beams along z (legs 0, 1).
fn phase_space(outgoing: usize, energy: f64, stream: &mut Stream) -> Vec<Momentum> {
    let mut q: Vec<Momentum> = (0..outgoing)
        .map(|_| {
            let c = 2. * stream.next() - 1.;
            let s = (1. - c * c).sqrt();
            let phi = 2. * std::f64::consts::PI * stream.next();
            let e = -(stream.next() * stream.next()).ln();
            [e, e * s * phi.cos(), e * s * phi.sin(), e * c]
        })
        .collect();
    let mut total = [0.; 4];
    for p in &q {
        for mu in 0..4 {
            total[mu] += p[mu];
        }
    }
    let mass = (total[0] * total[0] - total[1..].iter().map(|x| x * x).sum::<f64>()).sqrt();
    let b: Vec<f64> = total[1..].iter().map(|x| -x / mass).collect();
    let (gamma, a, x) = (total[0] / mass, 1. / (1. + total[0] / mass), energy / mass);
    for p in &mut q {
        let bq = b[0] * p[1] + b[1] * p[2] + b[2] * p[3];
        let e = x * (gamma * p[0] + bq);
        let spatial: Vec<f64> = (0..3)
            .map(|i| x * (p[i + 1] + b[i] * p[0] + a * bq * b[i]))
            .collect();
        *p = [e, spatial[0], spatial[1], spatial[2]];
    }
    let half = energy / 2.;
    let mut out = vec![[half, 0., 0., half], [half, 0., 0., -half]];
    out.extend(q);
    out
}

fn minkowski(a: &Momentum, b: &Momentum) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

/// One massless family in loop-cycle order: `order[i]` is the leg attached
/// before propagator `i`; propagator `i` carries `q + V_i` with
/// `V_i = sum_{k<i} p_order[k]` (incoming legs counted positive).
#[derive(Clone)]
struct Lane {
    shifts: Vec<Momentum>,
    /// `S[a][b] = (V_a - V_b)^2`; one leg between `a` and `b` is exactly 0.
    matrix: Vec<f64>,
}

fn lane(points: &[Momentum], order: &[usize]) -> Lane {
    let sign = |leg: usize| if leg < 2 { 1. } else { -1. };
    let mut shifts = vec![[0.; 4]];
    for &leg in &order[..order.len() - 1] {
        let last = *shifts.last().unwrap();
        shifts.push(std::array::from_fn(|mu| {
            last[mu] + sign(leg) * points[leg][mu]
        }));
    }
    let n = shifts.len();
    let mut matrix = vec![0.; n * n];
    for a in 0..n {
        for b in 0..n {
            let d: Momentum = std::array::from_fn(|mu| shifts[a][mu] - shifts[b][mu]);
            let adjacent = (a + 1) % n == b || (b + 1) % n == a;
            matrix[a * n + b] = if a == b || adjacent {
                0.
            } else {
                minkowski(&d, &d)
            };
        }
    }
    Lane { shifts, matrix }
}

/// The lane without propagator `j` (the pinch's 7-point family).
fn pinched(lane: &Lane, j: usize) -> Lane {
    let n = lane.shifts.len();
    let keep: Vec<usize> = (0..n).filter(|&i| i != j).collect();
    Lane {
        shifts: keep.iter().map(|&i| lane.shifts[i]).collect(),
        matrix: keep
            .iter()
            .flat_map(|&a| keep.iter().map(move |&b| lane.matrix[a * n + b]))
            .collect(),
    }
}

type Coefficients = Vec<[f64; 2]>;

fn dense(rank: usize, stream: &mut Stream) -> Coefficients {
    (0..tensor_monomials(rank).len())
        .map(|_| [2. * stream.next() - 1., 2. * stream.next() - 1.])
        .collect()
}

/// `D(q) T(q)` with `D(q) = (q + V)^2 = q.q + 2 q.V + V^2` (massless line),
/// `q^mu` the contravariant components the tensor layout's `q0..q3` are.
fn times_propagator(tensor: &Coefficients, rank: usize, shift: &Momentum) -> Coefficients {
    let index = |r: usize| -> HashMap<[u32; 4], usize> {
        tensor_monomials(r)
            .into_iter()
            .enumerate()
            .map(|(k, alpha)| (alpha, k))
            .collect()
    };
    let target = index(rank + 2);
    let metric = [1., -1., -1., -1.];
    let unit = |mu: usize| -> [u32; 4] { std::array::from_fn(|nu| u32::from(nu == mu)) };
    let mut factor: Vec<([u32; 4], f64)> = vec![([0; 4], minkowski(shift, shift))];
    for mu in 0..4 {
        factor.push((unit(mu), 2. * metric[mu] * shift[mu]));
        factor.push((unit(mu).map(|e| 2 * e), metric[mu]));
    }
    let mut out = vec![[0.; 2]; target.len()];
    for (alpha, c) in tensor_monomials(rank).into_iter().zip(tensor) {
        for (beta, f) in &factor {
            let k = target[&std::array::from_fn(|mu| alpha[mu] + beta[mu])];
            out[k][0] += f * c[0];
            out[k][1] += f * c[1];
        }
    }
    out
}

fn decimal(values: impl IntoIterator<Item = [f64; 2]>) -> Vec<[String; 2]> {
    values
        .into_iter()
        .map(|[re, im]| [format!("{re:e}"), format!("{im:e}")])
        .collect()
}

/// The typed batches of one family group: one row per lane.
struct Group {
    n: usize,
    shifts: Batch,
    masses: Batch,
    scales: Batch,
    matrix: Batch,
    coefficients: Batch,
}

fn batches(lanes: &[Lane], coefficients: &[Coefficients], arithmetic: &str) -> Group {
    let n = lanes[0].shifts.len();
    let rows = lanes.len();
    let width = coefficients[0].len();
    let batch = |values: Vec<[String; 2]>, columns| {
        Batch::from_decimal(values, rows, columns, arithmetic, None, None).unwrap()
    };
    let real = |x: f64| [x, 0.];
    Group {
        n,
        shifts: batch(
            decimal(
                lanes
                    .iter()
                    .flat_map(|l| l.shifts.iter().flatten().map(|&x| real(x))),
            ),
            4 * n,
        ),
        masses: batch(decimal(vec![[0., 0.]; rows * n]), n),
        scales: batch(decimal(vec![real(1.0e6); rows]), 1),
        matrix: batch(
            decimal(lanes.iter().flat_map(|l| l.matrix.iter().map(|&x| real(x)))),
            n * n,
        ),
        coefficients: batch(decimal(coefficients.iter().flatten().copied()), width),
    }
}

impl Group {
    fn family(&self) -> FamilyGroup<'_> {
        FamilyGroup {
            propagators: self.n,
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            s_matrix: &self.matrix,
            coefficients: &self.coefficients,
        }
    }
}

fn values(batch: &Batch) -> Vec<[f64; 2]> {
    batch
        .to_decimal()
        .into_iter()
        .map(|[re, im]| [re.parse().unwrap(), im.parse().unwrap()])
        .collect()
}

/// Largest |a - b| per lane over the three Laurent coefficients, relative
/// to the lane's largest |b| (as NinjaRS's `n_point.rs` measures).
fn relative(a: &[[f64; 2]], b: &[[f64; 2]]) -> Vec<f64> {
    let norm = |z: &[f64; 2]| z[0].hypot(z[1]);
    a.chunks(3)
        .zip(b.chunks(3))
        .map(|(x, y)| {
            let scale = y.iter().map(norm).fold(0., f64::max);
            x.iter()
                .zip(y)
                .map(|(x, y)| norm(&[x[0] - y[0], x[1] - y[1]]))
                .fold(0., f64::max)
                / scale
        })
        .collect()
}

fn show(errors: &[f64]) -> String {
    errors
        .iter()
        .map(|e| format!("{e:.1e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Three massless 2 -> 6 points at sqrt(s) = 1 TeV, each on its own cyclic
/// leg order (a different family shape per lane: adjacent incoming legs, one
/// outgoing leg between them, and three between them).
fn lanes() -> Vec<Lane> {
    let orders = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 2, 1, 3, 4, 5, 6, 7],
        [0, 3, 4, 5, 1, 2, 6, 7],
    ];
    let mut stream = Stream(0x8a5c_d789_635d_2dff);
    orders
        .iter()
        .map(|order| lane(&phase_space(6, 1000., &mut stream), order))
        .collect()
}

fn reduce(
    arithmetic: &str,
    engine: &str,
    group: &Group,
    check: bool,
) -> (Coefficients, Option<Coefficients>) {
    let mut reducer = V2Reducer::new(arithmetic, None, engine, true).unwrap();
    let options = ReduceOptions {
        check: check.then_some(ReductionFrame::REVERSED),
        ..ReduceOptions::default()
    };
    let output = reducer.reduce(&[group.family()], options).unwrap();
    for status in output.direct[0].statuses() {
        assert_eq!(status.1, "success", "{arithmetic}/{engine}: {status:?}");
    }
    assert_eq!(
        reducer.take_exact_lanes(),
        0,
        "no lane may route off the fast engine"
    );
    let direct = values(&output.direct[0].values().unwrap());
    let check = output.check.map(|c| values(&c[0].values().unwrap()));
    (direct, check)
}

#[test]
fn dense_rank_eight_fast_reversed_and_double_double_agree() {
    let lanes = lanes();
    let mut stream = Stream(0x1d8e_4e27_c47d_124f);
    let tensors: Vec<_> = lanes.iter().map(|_| dense(8, &mut stream)).collect();
    let (fast, reversed) = reduce("double", "fast", &batches(&lanes, &tensors, "double"), true);
    let (dd, _) = reduce(
        "double-double",
        "generic",
        &batches(&lanes, &tensors, "double-double"),
        false,
    );
    let fast_error = relative(&fast, &dd);
    let reversed_error = relative(&reversed.unwrap(), &dd);
    eprintln!(
        "N = 8, R = 8: fast vs DD [{}], reversed vs DD [{}]",
        show(&fast_error),
        show(&reversed_error)
    );
    // Every lane has double and single poles (massless, light-like legs).
    for lane in dd.chunks(3) {
        assert!(lane.iter().all(|z| z[0].is_finite() && z[1].is_finite()));
        assert!(lane[2][0].hypot(lane[2][1]) > 0.);
    }
    assert!(fast_error.iter().all(|&e| e <= 1e-7), "{fast_error:?}");
    assert!(
        reversed_error.iter().all(|&e| e <= 1e-6),
        "{reversed_error:?}"
    );
}

#[test]
fn pinch_identity_eight_to_seven_propagators() {
    let lanes = lanes();
    let mut stream = Stream(0x5bd1_e995_9e37_79b9);
    for j in [7, 3] {
        let tensors: Vec<_> = lanes.iter().map(|_| dense(6, &mut stream)).collect();
        let pinching: Vec<_> = lanes
            .iter()
            .zip(&tensors)
            .map(|(lane, t)| times_propagator(t, 6, &lane.shifts[j]))
            .collect();
        let seven: Vec<_> = lanes.iter().map(|lane| pinched(lane, j)).collect();
        let (eight, _) = reduce(
            "double",
            "fast",
            &batches(&lanes, &pinching, "double"),
            false,
        );
        let (reference, _) = reduce(
            "double",
            "fast",
            &batches(&seven, &tensors, "double"),
            false,
        );
        let error = relative(&eight, &reference);
        eprintln!(
            "pinch of line {j} (N = 8, R = 8 -> N = 7, R = 6): [{}]",
            show(&error)
        );
        assert!(error.iter().all(|&e| e <= 1e-6), "line {j}: {error:?}");
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn rss(field: &str) -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with(field)).unwrap();
    line.split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<f64>()
        .unwrap()
        / 1024.
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    /// glibc: return free heap memory to the system.
    fn malloc_trim(pad: usize) -> i32;
}

/// Peak resident growth (MB) of one call, from `/proc/self/clear_refs`,
/// after returning the earlier calls' freed memory to the system (so that
/// none of it is reused unseen).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn peak_of(call: impl FnOnce()) -> f64 {
    // SAFETY: malloc_trim only releases free memory of the system allocator.
    unsafe { malloc_trim(0) };
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
    let before = rss("VmRSS:");
    call();
    rss("VmHWM:") - before
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
#[ignore = "measurement: run alone with --ignored --nocapture --test-threads=1"]
fn generic_chunk_memory_at_eight_propagators() {
    let mut stream = Stream(0x2545_f491_4f6c_dd1d);
    let points: Vec<_> = (0..16)
        .map(|_| phase_space(6, 1000., &mut stream))
        .collect();
    let order = [0, 2, 1, 3, 4, 5, 6, 7];
    for (arithmetic, digits) in [("double-double", None), ("arbitrary", Some(77))] {
        for n in [7, 8] {
            // An n-propagator family of the same points (legs past n merged
            // into the last vertex) at rank n.
            let lanes: Vec<_> = points
                .iter()
                .map(|p| {
                    let full = lane(p, &order);
                    if n == 8 { full } else { pinched(&full, 7) }
                })
                .collect();
            let tensors: Vec<_> = lanes.iter().map(|_| dense(n, &mut stream)).collect();
            let group = |count: usize| {
                // Both arithmetics of the probe are generic-engine carriers:
                // convert the binary64 group's inputs.
                let g = batches(&lanes[..count], &tensors[..count], "double");
                let convert = |b: &Batch, columns| {
                    Batch::from_decimal(b.to_decimal(), count, columns, arithmetic, digits, None)
                        .unwrap()
                };
                Group {
                    n,
                    shifts: convert(&g.shifts, 4 * n),
                    masses: convert(&g.masses, n),
                    scales: convert(&g.scales, 1),
                    matrix: convert(&g.matrix, n * n),
                    coefficients: convert(
                        &g.coefficients,
                        g.coefficients.to_decimal().len() / count,
                    ),
                }
            };
            let mut reducer = V2Reducer::new(arithmetic, digits, "generic", true).unwrap();
            let mut run = |count: usize| {
                let g = group(count);
                let start = std::time::Instant::now();
                let peak = peak_of(|| {
                    reducer
                        .reduce(&[g.family()], ReduceOptions::default())
                        .unwrap();
                });
                (peak, start.elapsed().as_secs_f64())
            };
            let (first, first_s) = run(1);
            let (one, one_s) = run(1);
            let (sixteen, sixteen_s) = run(16);
            eprintln!(
                "{arithmetic} ({} bits) N = R = {n}: first call {first:.0} MB ({first_s:.2} s, \
                 template + 1 lane); 1 lane {one:.0} MB ({one_s:.2} s); 16 lanes {sixteen:.0} MB \
                 ({sixteen_s:.2} s); {:.1} MB per lane",
                reducer.precision_bits(),
                (sixteen - one) / 15.
            );
        }
    }
}
