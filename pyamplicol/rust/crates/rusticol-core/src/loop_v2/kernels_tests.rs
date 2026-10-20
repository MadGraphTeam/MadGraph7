// SPDX-License-Identifier: 0BSD

use super::*;
use symbolica::domains::{
    float::{DoubleFloat, RealLike, SingleFloat},
    rational::Rational,
};

const L: usize = 2;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn rational(&mut self) -> Rational {
        let num = (self.next() % 19) as i64 - 9;
        let den = (self.next() % 7) as i64 + 1;
        Rational::from((num, den))
    }

    fn lane_vectors(&mut self, count: usize) -> Vec<Rational> {
        (0..count * lane_vector_len(L))
            .map(|_| self.rational())
            .collect()
    }
}

fn zeros(count: usize) -> Vec<Rational> {
    vec![Rational::from(0); count * lane_vector_len(L)]
}

/// Complex lane vector `a * b`.
fn cmul(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let mut out = zeros(1);
    cmac::<Rational, L>(&mut out, a, b);
    out
}

fn cadd(acc: &mut [Rational], x: &[Rational]) {
    for (a, x) in acc.iter_mut().zip(x) {
        *a += x;
    }
}

fn lane(values: &[Rational], index: usize) -> &[Rational] {
    &values[index * lane_vector_len(L)..][..lane_vector_len(L)]
}

/// Real monomial `q^exponents` as a complex lane vector.
fn monomial_value(q: &[Rational; 4], exponents: [u32; 4]) -> Vec<Rational> {
    let mut value = Rational::from(1);
    for (component, &power) in q.iter().zip(&exponents) {
        for _ in 0..power {
            value = value * component;
        }
    }
    let mut out = zeros(1);
    for re in &mut out[..L] {
        *re = value.clone();
    }
    out
}

/// `sum_m coefficients[m] q^m` for a polynomial stored as `ncoeff(rank)` lane vectors.
fn evaluate(coefficients: &[Rational], rank: usize, q: &[Rational; 4]) -> Vec<Rational> {
    let mut out = zeros(1);
    for (m, exponents) in monomials(rank).into_iter().enumerate() {
        cadd(
            &mut out,
            &cmul(lane(coefficients, m), &monomial_value(q, exponents)),
        );
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Case {
    cut: usize,
    incoming: usize,
    current: usize,
    outgoing: usize,
    rank: usize,
    degree: usize,
}

struct Edge {
    input: Vec<Rational>,
    current: Vec<Rational>,
    /// Dense fused factor `X[block][n][b][b']`.
    factor: Vec<Rational>,
    links: Vec<RowLink>,
}

const ROWS_OUT: usize = 3;
const BLOCK_ROWS: usize = 2;

fn run_case(case: Case, rng: &mut Lcg) {
    let products = MonomialProductTable::new(case.rank, case.degree);
    let segment = Segment::new(
        case.cut,
        case.incoming,
        case.current,
        case.outgoing,
        &products,
        case.rank + case.degree,
    );
    let n_factor = ncoeff(case.degree);

    // A structurally sparse vertex: roughly every third entry is zero.
    let dense_vertex = segment.dense_vertex_layout();
    let mut vertex_dense_values = zeros(dense_vertex.entries().len());
    let mut sparse_entries = Vec::new();
    let mut sparse_values = Vec::new();
    for (k, entry) in dense_vertex.entries().iter().enumerate() {
        if rng.next() % 3 == 0 {
            continue;
        }
        let value = rng.lane_vectors(1);
        vertex_dense_values[k * lane_vector_len(L)..][..lane_vector_len(L)]
            .clone_from_slice(&value);
        sparse_entries.push(*entry);
        sparse_values.extend(value);
    }
    let sparse_vertex = SegmentLayout::new(
        dense_vertex.loop_in_dimension(),
        case.outgoing,
        sparse_entries,
    )
    .unwrap();
    let vertex_at = |n: usize, b: usize, nu: usize, bp: usize| {
        let k = ((n * case.incoming + b) * case.current + nu) * case.outgoing + bp;
        lane(&vertex_dense_values, k).to_vec()
    };

    let edges: Vec<Edge> = (0..2)
        .map(|e| {
            let rows_in = ROWS_OUT * BLOCK_ROWS + e;
            let input = rng.lane_vectors(rows_in * segment.input_row_len());
            let current = rng.lane_vectors(BLOCK_ROWS * case.current);
            let mut factor = Vec::new();
            for block in 0..BLOCK_ROWS {
                for n in 0..n_factor {
                    for b in 0..case.incoming {
                        for bp in 0..case.outgoing {
                            let mut x = zeros(1);
                            for nu in 0..case.current {
                                cadd(
                                    &mut x,
                                    &cmul(
                                        &vertex_at(n, b, nu, bp),
                                        lane(&current, block * case.current + nu),
                                    ),
                                );
                            }
                            factor.extend(x);
                        }
                    }
                }
            }
            // Regular helicity links plus one irregular gather.
            let mut links: Vec<RowLink> = (0..ROWS_OUT)
                .flat_map(|h| {
                    (0..BLOCK_ROWS).map(move |hb| RowLink {
                        output: h as u32,
                        input: (h * BLOCK_ROWS + hb) as u32,
                        block: hb as u32,
                    })
                })
                .collect();
            links.push(RowLink {
                output: 0,
                input: (rows_in - 1) as u32,
                block: 1,
            });
            Edge {
                input,
                current,
                factor,
                links,
            }
        })
        .collect();

    let output_len = ROWS_OUT * segment.output_row_len();

    let split = |layout: &SegmentLayout, values: &[Rational]| {
        let mut accumulator = zeros(ROWS_OUT * segment.accumulator_row_len());
        for edge in &edges {
            split_edge::<Rational, L>(
                &segment,
                &edge.input,
                &edge.current,
                &edge.links,
                &mut accumulator,
            );
        }
        let mut output = zeros(output_len);
        split_vertex::<Rational, L>(
            &segment,
            layout,
            values,
            ROWS_OUT,
            &accumulator,
            &mut output,
        );
        output
    };
    let split_sparse = split(&sparse_vertex, &sparse_values);
    let split_dense = split(&dense_vertex, &vertex_dense_values);

    let dense_fused = segment.dense_fused_layout();
    let mut fused_dense = zeros(output_len);
    for edge in &edges {
        fused_edge::<Rational, L>(
            &segment,
            &dense_fused,
            &edge.input,
            &edge.factor,
            &edge.links,
            &mut fused_dense,
        );
    }

    // Structured fused factor: keep only the entries the sparse vertex can reach.
    let reachable: Vec<bool> = dense_fused
        .entries()
        .iter()
        .map(|f| {
            sparse_vertex.entries().iter().any(|v| {
                (v.exponents, v.loop_in / case.current, v.out) == (f.exponents, f.loop_in, f.out)
            })
        })
        .collect();
    let structured_fused = SegmentLayout::new(
        case.incoming,
        case.outgoing,
        dense_fused
            .entries()
            .iter()
            .zip(&reachable)
            .filter_map(|(entry, &keep)| keep.then_some(*entry))
            .collect(),
    )
    .unwrap();
    let mut fused_structured = zeros(output_len);
    for edge in &edges {
        let factor: Vec<Rational> = edge
            .factor
            .chunks_exact(dense_fused.entries().len() * lane_vector_len(L))
            .flat_map(|row| {
                row.chunks_exact(lane_vector_len(L))
                    .zip(&reachable)
                    .filter(|(_, keep)| **keep)
                    .flat_map(|(x, _)| x.to_vec())
            })
            .collect();
        fused_edge::<Rational, L>(
            &segment,
            &structured_fused,
            &edge.input,
            &factor,
            &edge.links,
            &mut fused_structured,
        );
    }

    let mut blocked = zeros(output_len);
    for edge in &edges {
        let args = (&segment, &edge.input[..], &edge.factor[..], &edge.links[..]);
        match (case.incoming, case.outgoing) {
            (4, 4) => fused_edge_blocked::<Rational, L, 4, 4>(
                args.0,
                args.1,
                args.2,
                args.3,
                &mut blocked,
            ),
            (1, 1) => fused_edge_blocked::<Rational, L, 1, 1>(
                args.0,
                args.1,
                args.2,
                args.3,
                &mut blocked,
            ),
            (4, 1) => fused_edge_blocked::<Rational, L, 4, 1>(
                args.0,
                args.1,
                args.2,
                args.3,
                &mut blocked,
            ),
            other => panic!("no blocked instantiation for {other:?}"),
        }
    }

    assert_eq!(
        split_sparse, split_dense,
        "{case:?}: split structured/dense"
    );
    assert_eq!(split_sparse, fused_dense, "{case:?}: split/fused");
    assert_eq!(split_sparse, fused_structured, "{case:?}: fused structured");
    assert_eq!(split_sparse, blocked, "{case:?}: register-blocked");

    // Direct evaluation at a random q: O'(q) = sum_links O(q) V(q) J.
    let q: [Rational; 4] = std::array::from_fn(|_| rng.rational());
    let m_in = ncoeff(case.rank);
    let m_out = ncoeff(segment.output_rank());
    let degree_monomials = monomials(case.degree);
    for h in 0..ROWS_OUT {
        for a in 0..case.cut {
            for bp in 0..case.outgoing {
                let mut expected = zeros(1);
                for edge in &edges {
                    for link in edge.links.iter().filter(|link| link.output as usize == h) {
                        for b in 0..case.incoming {
                            let row = (link.input as usize * case.cut + a) * case.incoming + b;
                            let o_q = evaluate(
                                &edge.input[row * m_in * lane_vector_len(L)..]
                                    [..m_in * lane_vector_len(L)],
                                case.rank,
                                &q,
                            );
                            for nu in 0..case.current {
                                let mut v_q = zeros(1);
                                for (n, &exponents) in degree_monomials.iter().enumerate() {
                                    cadd(
                                        &mut v_q,
                                        &cmul(
                                            &vertex_at(n, b, nu, bp),
                                            &monomial_value(&q, exponents),
                                        ),
                                    );
                                }
                                let j =
                                    lane(&edge.current, link.block as usize * case.current + nu);
                                cadd(&mut expected, &cmul(&cmul(&o_q, &v_q), j));
                            }
                        }
                    }
                }
                let row = (h * case.cut + a) * case.outgoing + bp;
                let actual = evaluate(
                    &split_sparse[row * m_out * lane_vector_len(L)..][..m_out * lane_vector_len(L)],
                    segment.output_rank(),
                    &q,
                );
                assert_eq!(
                    actual, expected,
                    "{case:?}: direct evaluation h={h} a={a} b'={bp}"
                );
            }
        }
    }
}

#[test]
fn dense_layouts_follow_the_blocked_order() {
    let products = MonomialProductTable::new(1, 2);
    let segment = Segment::new(4, 4, 3, 2, &products, 3);
    let layout = segment.dense_vertex_layout();
    assert_eq!(layout.entries().len(), ncoeff(2) * 4 * 3 * 2);
    for (k, (entry, &n)) in layout.entries().iter().zip(layout.monomials()).enumerate() {
        let (b, nu) = (entry.loop_in / 3, entry.loop_in % 3);
        assert_eq!(entry.loop_in, segment.vertex_column(b, nu));
        assert_eq!(k, ((n * 4 + b) * 3 + nu) * 2 + entry.out);
        assert_eq!(monomials(2)[n], entry.exponents);
    }
}

#[test]
#[should_panic(expected = "does not fit")]
fn kernels_reject_a_layout_of_another_segment() {
    let products = MonomialProductTable::new(0, 1);
    let segment = Segment::new(1, 4, 4, 4, &products, 1);
    let wrong = dense_layout(4, 1, 1);
    let input = zeros(segment.input_row_len());
    let factor = zeros(wrong.entries().len());
    let mut output = zeros(segment.output_row_len());
    let links = [RowLink {
        output: 0,
        input: 0,
        block: 0,
    }];
    fused_edge::<Rational, L>(&segment, &wrong, &input, &factor, &links, &mut output);
}

#[test]
fn kernel_forms_agree_exactly_and_match_direct_evaluation() {
    let mut rng = Lcg(0x5eed);
    let cases = [
        // Gluon or quark loop line, early and late steps.
        Case {
            cut: 4,
            incoming: 4,
            current: 4,
            outgoing: 4,
            rank: 0,
            degree: 1,
        },
        Case {
            cut: 4,
            incoming: 4,
            current: 4,
            outgoing: 4,
            rank: 2,
            degree: 1,
        },
        // Ghost line attaching a gluon block.
        Case {
            cut: 1,
            incoming: 1,
            current: 4,
            outgoing: 1,
            rank: 1,
            degree: 1,
        },
        // Vector to scalar transition, scalar block, a degree-2 segment.
        Case {
            cut: 4,
            incoming: 4,
            current: 1,
            outgoing: 1,
            rank: 1,
            degree: 2,
        },
        // q-independent segment.
        Case {
            cut: 1,
            incoming: 4,
            current: 4,
            outgoing: 4,
            rank: 3,
            degree: 0,
        },
    ];
    for case in cases {
        run_case(case, &mut rng);
    }
}

#[test]
fn binary64_blocked_matches_pattern_kernel() {
    const LANES: usize = 8;
    let products = MonomialProductTable::new(3, 1);
    let segment = Segment::new(4, 4, 4, 4, &products, 4);
    let pattern = segment.dense_fused_layout();
    let mut seed = 1u64;
    let mut sample = |count: usize| -> Vec<f64> {
        (0..count * lane_vector_len(LANES))
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    };
    let input = sample(4 * segment.input_row_len());
    let factor = sample(2 * pattern.entries().len());
    let links: Vec<RowLink> = (0..4)
        .map(|r| RowLink {
            output: (r / 2) as u32,
            input: r as u32,
            block: (r % 2) as u32,
        })
        .collect();
    let mut reference = vec![0.0; 2 * segment.output_row_len() * lane_vector_len(LANES)];
    let mut blocked = reference.clone();
    fused_edge::<f64, LANES>(&segment, &pattern, &input, &factor, &links, &mut reference);
    fused_edge_blocked::<f64, LANES, 4, 4>(&segment, &input, &factor, &links, &mut blocked);
    let scale = reference.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (r, b) in reference.iter().zip(&blocked) {
        assert!((r - b).abs() <= 1e-14 * scale, "{r} vs {b}");
    }
}

/// Runs the fused and split forms of one segment in carrier `T`.
fn fused_and_split<T: FloatLike>(
    segment: &Segment<'_>,
    data: &[Vec<Rational>; 4],
    convert: impl Fn(&Rational) -> T,
) -> [Vec<T>; 2] {
    let [input, current, factor, vertex] = data
        .each_ref()
        .map(|values| values.iter().map(&convert).collect::<Vec<T>>());
    let links: Vec<RowLink> = (0..4)
        .map(|r| RowLink {
            output: (r / 2) as u32,
            input: r as u32,
            block: (r % 2) as u32,
        })
        .collect();
    let zero = convert(&Rational::from(0));
    let mut fused = vec![zero.clone(); 2 * segment.output_row_len() * lane_vector_len(L)];
    fused_edge::<T, L>(
        segment,
        &segment.dense_fused_layout(),
        &input,
        &factor,
        &links,
        &mut fused,
    );
    let mut accumulator =
        vec![zero.clone(); 2 * segment.accumulator_row_len() * lane_vector_len(L)];
    split_edge::<T, L>(segment, &input, &current, &links, &mut accumulator);
    let mut split = vec![zero; 2 * segment.output_row_len() * lane_vector_len(L)];
    split_vertex::<T, L>(
        segment,
        &segment.dense_vertex_layout(),
        &vertex,
        2,
        &accumulator,
        &mut split,
    );
    [fused, split]
}

/// Largest error of `values` against the exact results, relative to the
/// largest exact magnitude; the exact results are rounded into the carrier.
fn relative_error<T: SingleFloat + RealLike>(values: &[T], exact: &[Rational]) -> f64 {
    let scale = exact.iter().fold(0.0f64, |m, x| m.max(x.to_f64().abs()));
    assert!(scale > 0.0);
    values
        .iter()
        .zip(exact)
        .map(|(value, exact)| (value.clone() - value.from_rational(exact)).to_f64().abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn extended_precision_carriers_keep_their_precision() {
    // Inputs with denominators up to 7 are not dyadic, so every carrier
    // rounds them at its own precision; computing in binary64 would leave
    // errors near 1e-16 and fail both bounds below.
    let mut rng = Lcg(0xcafe);
    let products = MonomialProductTable::new(2, 1);
    let segment = Segment::new(4, 4, 4, 4, &products, 3);
    let data = [
        rng.lane_vectors(4 * segment.input_row_len()),
        rng.lane_vectors(2 * segment.current()),
        rng.lane_vectors(2 * segment.dense_fused_layout().entries().len()),
        rng.lane_vectors(segment.dense_vertex_layout().entries().len()),
    ];
    let exact = fused_and_split(&segment, &data, Rational::clone);
    let binary64 = fused_and_split(&segment, &data, Rational::to_f64);
    let double_float = fused_and_split(&segment, &data, |r| DoubleFloat::from(r));
    let float = fused_and_split(&segment, &data, |r| r.to_multi_prec_float(200));
    for form in 0..2 {
        assert!(relative_error(&binary64[form], &exact[form]) > 1e-20);
        assert!(relative_error(&double_float[form], &exact[form]) <= 1e-29);
        assert!(relative_error(&float[form], &exact[form]) <= 1e-57);
    }
}
