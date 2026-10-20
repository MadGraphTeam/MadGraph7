// SPDX-License-Identifier: 0BSD

//! Gate G0 benchmark of the loop-v2 segment kernels (design §5.4, §14).
//!
//! Runs whole schedule-B trie nodes (every edge with its own child object,
//! regular helicity links, a fresh output object) in every kernel form on
//! synthetic node shapes spanning a 7-leg trie, for 8, 16 and 32 SIMD lanes,
//! and measures the fixed per-node and per-edge-call overheads. Work is
//! counted as the complex multiply-adds per lane the kernel performs, the
//! unit of the design's cost model.
//!
//! ```text
//! cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_kernels -- \
//!     [--json results.json] [--lanes 8,16,32] [--target-seconds 0.15]
//! ```
//!
//! `tools/loop_v2_costmodel.py` reads the JSON output.

use std::hint::black_box;
use std::time::Instant;

use rusticol_core::loop_v2::kernels::{
    FusedEntry, FusedPattern, MonomialProducts, RowLink, Segment, VertexEntry, VertexPattern,
    clear, fused_edge, fused_edge_blocked, lane_vector_len, split_edge, split_vertex,
};
use serde_json::{Value, json};

/// One trie node: `rows_out` output rows; `edges` edges, each attaching a
/// block with `block_rows` helicity rows to its own child object of rank
/// `rank_in` with `rows_out * block_rows` rows.
#[derive(Clone, Copy, Debug)]
struct Shape {
    line: Line,
    rows_out: usize,
    edges: usize,
    block_rows: usize,
    rank_in: usize,
}

/// Loop-line species of the segment, which fixes its dimensions and the
/// structural zeros of its vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Line {
    /// Gluon line attaching a gluon block: three-gluon vertex times the
    /// Feynman-gauge propagator numerator.
    Gluon,
    /// Ghost line attaching a gluon block.
    Ghost,
}

impl Line {
    fn name(self) -> &'static str {
        match self {
            Line::Gluon => "gluon",
            Line::Ghost => "ghost",
        }
    }

    /// (cut, incoming, current, outgoing).
    fn dims(self) -> (usize, usize, usize, usize) {
        match self {
            Line::Gluon => (4, 4, 4, 4),
            Line::Ghost => (1, 1, 4, 1),
        }
    }

    /// Structural non-zeros of `V[n][b][nu][b']` for `n` over the degree-1
    /// monomials `1, q0, q1, q2, q3`.
    fn vertex_nonzero(self, n: usize, b: usize, nu: usize, bp: usize) -> bool {
        match (self, n) {
            // g^{b nu} k^{b'} + g^{nu b'} k^{b} + g^{b' b} k^{nu}
            (Line::Gluon, 0) => b == nu || nu == bp || b == bp,
            (Line::Gluon, n) => {
                let mu = n - 1;
                (b == nu && bp == mu) || (nu == bp && b == mu) || (b == bp && nu == mu)
            }
            // (q + P)^{nu}
            (Line::Ghost, 0) => true,
            (Line::Ghost, n) => nu == n - 1,
        }
    }

    fn vertex_pattern(self, segment: &Segment<'_>) -> VertexPattern {
        VertexPattern::new(
            VertexPattern::dense(segment)
                .entries()
                .iter()
                .copied()
                .filter(|e: &VertexEntry| {
                    self.vertex_nonzero(
                        e.monomial as usize,
                        e.incoming as usize,
                        e.current as usize,
                        e.outgoing as usize,
                    )
                })
                .collect(),
        )
    }

    /// Non-zeros of `X(J) = V J` for a generic block current.
    fn fused_pattern(self, segment: &Segment<'_>) -> FusedPattern {
        FusedPattern::new(
            FusedPattern::dense(segment)
                .entries()
                .iter()
                .copied()
                .filter(|e: &FusedEntry| {
                    (0..segment.current).any(|nu| {
                        self.vertex_nonzero(
                            e.monomial as usize,
                            e.incoming as usize,
                            nu,
                            e.outgoing as usize,
                        )
                    })
                })
                .collect(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    SplitDense,
    SplitStructured,
    FusedDense,
    FusedStructured,
    FusedBlocked,
}

impl Form {
    const ALL: [Form; 5] = [
        Form::SplitDense,
        Form::SplitStructured,
        Form::FusedDense,
        Form::FusedStructured,
        Form::FusedBlocked,
    ];

    fn name(self) -> &'static str {
        match self {
            Form::SplitDense => "split-dense",
            Form::SplitStructured => "split-structured",
            Form::FusedDense => "fused-dense",
            Form::FusedStructured => "fused-structured",
            Form::FusedBlocked => "fused-blocked",
        }
    }
}

/// The probe's node shapes (design §5.4 table): early steps with many rows
/// and low rank to late steps with few rows and high rank.
fn trie_shapes() -> Vec<Shape> {
    let gluon = |rows_out, edges, block_rows, rank_in| Shape {
        line: Line::Gluon,
        rows_out,
        edges,
        block_rows,
        rank_in,
    };
    let ghost = |rows_out, edges, block_rows, rank_in| Shape {
        line: Line::Ghost,
        rows_out,
        edges,
        block_rows,
        rank_in,
    };
    vec![
        gluon(64, 2, 2, 0),
        gluon(32, 3, 4, 1),
        gluon(16, 4, 4, 2),
        gluon(8, 4, 8, 3),
        gluon(4, 6, 8, 4),
        gluon(2, 8, 8, 5),
        gluon(1, 8, 8, 6),
        ghost(16, 4, 4, 2),
        ghost(2, 8, 8, 5),
    ]
}

struct Rng(u64);

impl Rng {
    fn fill(&mut self, count: usize) -> Vec<f64> {
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }
}

/// Seconds per call: best of three timed batches of about `target` seconds each.
fn time_per_call(target: f64, mut f: impl FnMut()) -> f64 {
    f();
    let start = Instant::now();
    f();
    let once = start.elapsed().as_secs_f64().max(1e-9);
    let reps = ((target / once) as usize).max(1);
    (0..3)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..reps {
                f();
            }
            start.elapsed().as_secs_f64() / reps as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// A node instance with all buffers for every form.
struct Node {
    shape: Shape,
    products: MonomialProducts,
    dense_vertex_pattern: VertexPattern,
    structured_vertex_pattern: VertexPattern,
    dense_fused_pattern: FusedPattern,
    structured_fused_pattern: FusedPattern,
    inputs: Vec<Vec<f64>>,
    currents: Vec<Vec<f64>>,
    dense_factors: Vec<Vec<f64>>,
    structured_factors: Vec<Vec<f64>>,
    dense_vertex: Vec<f64>,
    structured_vertex: Vec<f64>,
    links: Vec<RowLink>,
    output: Vec<f64>,
    accumulator: Vec<f64>,
}

impl Node {
    fn new<const L: usize>(shape: Shape, rng: &mut Rng) -> Self {
        let v = lane_vector_len(L);
        let products = MonomialProducts::new(shape.rank_in, 1);
        let segment = segment(&shape, &products);
        let dense_vertex_pattern = VertexPattern::dense(&segment);
        let structured_vertex_pattern = shape.line.vertex_pattern(&segment);
        let dense_fused_pattern = FusedPattern::dense(&segment);
        let structured_fused_pattern = shape.line.fused_pattern(&segment);
        let dense_fused = dense_fused_pattern.entries().len();
        let structured_fused = structured_fused_pattern.entries().len();
        let rows_in = shape.rows_out * shape.block_rows;
        let links = (0..shape.rows_out)
            .flat_map(|h| {
                (0..shape.block_rows).map(move |hb| RowLink {
                    output: h as u32,
                    input: (h * shape.block_rows + hb) as u32,
                    block: hb as u32,
                })
            })
            .collect();
        Self {
            inputs: (0..shape.edges)
                .map(|_| rng.fill(rows_in * segment.input_row_len() * v))
                .collect(),
            currents: (0..shape.edges)
                .map(|_| rng.fill(shape.block_rows * segment.current * v))
                .collect(),
            dense_factors: (0..shape.edges)
                .map(|_| rng.fill(shape.block_rows * dense_fused * v))
                .collect(),
            structured_factors: (0..shape.edges)
                .map(|_| rng.fill(shape.block_rows * structured_fused * v))
                .collect(),
            dense_vertex: rng.fill(dense_vertex_pattern.entries().len() * v),
            structured_vertex: rng.fill(structured_vertex_pattern.entries().len() * v),
            links,
            output: vec![0.0; shape.rows_out * segment.output_row_len() * v],
            accumulator: vec![0.0; shape.rows_out * segment.accumulator_row_len() * v],
            shape,
            products,
            dense_vertex_pattern,
            structured_vertex_pattern,
            dense_fused_pattern,
            structured_fused_pattern,
        }
    }
}

fn segment<'a>(shape: &Shape, products: &'a MonomialProducts) -> Segment<'a> {
    let (cut, incoming, current, outgoing) = shape.line.dims();
    Segment::new(
        cut,
        incoming,
        current,
        outgoing,
        products,
        products.rank() + 1,
    )
}

/// Runs one whole node in `form` and returns its counted complex MACs per lane.
fn run_node<const L: usize>(node: &mut Node, form: Form, links: &[RowLink]) -> usize {
    let segment = segment(&node.shape, &node.products);
    let m_in = node.products.input_len();
    let rows = node.shape.rows_out;
    clear(&mut node.output);
    match form {
        Form::SplitDense | Form::SplitStructured => {
            let (pattern, vertex) = if form == Form::SplitDense {
                (&node.dense_vertex_pattern, &node.dense_vertex)
            } else {
                (&node.structured_vertex_pattern, &node.structured_vertex)
            };
            clear(&mut node.accumulator);
            for (input, current) in node.inputs.iter().zip(&node.currents) {
                split_edge::<f64, L>(&segment, input, current, links, &mut node.accumulator);
            }
            split_vertex::<f64, L>(
                &segment,
                pattern,
                vertex,
                rows,
                &node.accumulator,
                &mut node.output,
            );
            node.inputs.len() * links.len() * segment.accumulator_row_len()
                + rows * segment.cut * pattern.entries().len() * m_in
        }
        Form::FusedDense | Form::FusedStructured => {
            let (pattern, factors) = if form == Form::FusedDense {
                (&node.dense_fused_pattern, &node.dense_factors)
            } else {
                (&node.structured_fused_pattern, &node.structured_factors)
            };
            for (input, factor) in node.inputs.iter().zip(factors) {
                fused_edge::<f64, L>(&segment, pattern, input, factor, links, &mut node.output);
            }
            node.inputs.len() * links.len() * segment.cut * pattern.entries().len() * m_in
        }
        Form::FusedBlocked => {
            for (input, factor) in node.inputs.iter().zip(&node.dense_factors) {
                match (segment.incoming, segment.outgoing) {
                    (4, 4) => fused_edge_blocked::<f64, L, 4, 4>(
                        &segment,
                        input,
                        factor,
                        links,
                        &mut node.output,
                    ),
                    (1, 1) => fused_edge_blocked::<f64, L, 1, 1>(
                        &segment,
                        input,
                        factor,
                        links,
                        &mut node.output,
                    ),
                    other => unreachable!("no blocked instantiation for {other:?}"),
                }
            }
            node.inputs.len()
                * links.len()
                * segment.cut
                * segment.incoming
                * segment.outgoing
                * node.products.factor_len()
                * m_in
        }
    }
}

fn shape_json(shape: &Shape) -> Value {
    let (cut, incoming, current, outgoing) = shape.line.dims();
    json!({
        "line": shape.line.name(),
        "rows_out": shape.rows_out,
        "edges": shape.edges,
        "block_rows": shape.block_rows,
        "x": shape.edges * shape.block_rows,
        "rank_in": shape.rank_in,
        "cut": cut, "incoming": incoming, "current": current, "outgoing": outgoing,
    })
}

fn run_lanes<const L: usize>(target: f64, records: &mut Vec<Value>) {
    let mut rng = Rng(7);
    println!("lanes={L}: G complex MAC/s per core (counted work per lane x lanes / time)");
    print!("{:>6} {:>4} {:>3} {:>4} |", "line", "rows", "X", "r_in");
    for form in Form::ALL {
        print!(" {:>16}", form.name());
    }
    println!();
    for shape in trie_shapes() {
        let mut node = Node::new::<L>(shape, &mut rng);
        let links = node.links.clone();
        print!(
            "{:>6} {:>4} {:>3} {:>4} |",
            shape.line.name(),
            shape.rows_out,
            shape.edges * shape.block_rows,
            shape.rank_in
        );
        for form in Form::ALL {
            let work = run_node::<L>(&mut node, form, &links);
            let seconds = time_per_call(target, || {
                run_node::<L>(&mut node, form, &links);
                black_box(&node.output);
            });
            let rate = (work * L) as f64 / seconds;
            print!(" {:>16.3}", rate / 1e9);
            records.push(json!({
                "kind": "node",
                "lanes": L,
                "form": form.name(),
                "shape": shape_json(&shape),
                "work_cmac_per_lane": work,
                "seconds_per_node": seconds,
                "gcmac_per_s": rate / 1e9,
            }));
        }
        println!();
    }

    // Fixed overheads, which the counted work does not cover:
    // - per node: a node without output rows (calls and setup only);
    // - per edge call: the marginal cost of an edge call without links;
    // - fit: least-squares intercept and slope of the time of a one-row,
    //   rank-0 node over 1..=8 single-link edges (both include work: the
    //   output clear and the split vertex in the intercept, the link's
    //   products in the slope).
    for form in Form::ALL {
        let single_row = |edges| Shape {
            line: Line::Gluon,
            rows_out: 1,
            edges,
            block_rows: 1,
            rank_in: 0,
        };
        let no_rows = |edges| Shape {
            rows_out: 0,
            ..single_row(edges)
        };
        let mut points = Vec::new();
        for edges in 1..=8 {
            let mut node = Node::new::<L>(single_row(edges), &mut rng);
            let links = node.links.clone();
            let seconds = time_per_call(target / 4.0, || {
                run_node::<L>(&mut node, form, &links);
                black_box(&node.output);
            });
            points.push((edges as f64, seconds));
        }
        let n = points.len() as f64;
        let (sx, sy) = points
            .iter()
            .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
        let (sxx, sxy) = points
            .iter()
            .fold((0.0, 0.0), |(a, b), (x, y)| (a + x * x, b + x * y));
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;

        let mut empty = |edges| {
            let mut node = Node::new::<L>(no_rows(edges), &mut rng);
            time_per_call(target / 4.0, || {
                run_node::<L>(&mut node, form, &[]);
                black_box(&node.output);
            })
        };
        let per_node = empty(1);
        let per_edge_call = ((empty(8) - per_node) / 7.0).max(0.0);
        println!(
            "  dispatch {:>16}: per node {:6.1} ns, per edge call {:5.1} ns; fit {:6.1} ns + {:6.1} ns/edge",
            form.name(),
            per_node * 1e9,
            per_edge_call * 1e9,
            intercept * 1e9,
            slope * 1e9,
        );
        records.push(json!({
            "kind": "dispatch",
            "lanes": L,
            "form": form.name(),
            "per_node_seconds": per_node,
            "per_edge_call_seconds": per_edge_call,
            "fit_intercept_seconds": intercept,
            "fit_slope_seconds": slope,
        }));
    }
}

fn main() {
    let mut json_path = None;
    let mut lanes = vec![8, 16, 32];
    let mut target = 0.15;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json_path = args.next(),
            "--lanes" => {
                lanes = args
                    .next()
                    .expect("--lanes needs a value")
                    .split(',')
                    .map(|l| l.parse().expect("lane count"))
                    .collect()
            }
            "--target-seconds" => {
                target = args
                    .next()
                    .expect("--target-seconds needs a value")
                    .parse()
                    .expect("seconds")
            }
            _ => {} // cargo bench passes --bench
        }
    }
    let mut records = Vec::new();
    for l in lanes {
        match l {
            8 => run_lanes::<8>(target, &mut records),
            16 => run_lanes::<16>(target, &mut records),
            32 => run_lanes::<32>(target, &mut records),
            other => panic!("lane count {other} is not instantiated (8, 16, 32)"),
        }
    }
    if let Some(path) = json_path {
        let report = json!({
            "benchmark": "loop_v2_kernels",
            "target_features": {
                "sse2": cfg!(target_feature = "sse2"),
                "avx2": cfg!(target_feature = "avx2"),
                "fma": cfg!(target_feature = "fma"),
                "avx512f": cfg!(target_feature = "avx512f"),
            },
            "records": records,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
            .expect("write the JSON report");
        println!("wrote {path}");
    }
}
