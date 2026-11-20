// SPDX-License-Identifier: 0BSD

//! Gate G0 benchmark of the loop-v2 segment kernels (design §5.4, §14).
//!
//! Runs whole schedule-B trie nodes (every edge with its own child object,
//! regular helicity links, a fresh output object) in every kernel form on
//! synthetic node shapes spanning a 7-leg trie plus nodes with at most two
//! attached block rows, for 8, 16 and 32 SIMD lanes. It also times the same
//! fused step in `poly.rs`'s `Complex` lane storage and lower bounds of the
//! fixed per-node and per-edge-call overheads. Work is counted as the complex
//! multiply-adds per lane the kernel performs, the unit of the design's cost
//! model.
//!
//! With `--program`, it also runs real schedule-B programs exported by
//! `tools/loop_v2_costmodel.py --export-programs` (see "real tries" below):
//! every kernel form, a walk without kernels (dispatch), the per-step
//! fastest form (oracle) and the fitted selection rule.
//!
//! ```text
//! cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_kernels -- \
//!     [--json results.json] [--lanes 8,16,32] [--target-seconds 0.15] \
//!     [--program gg_gggg-FC.json ...] [--program-reps 3] [--programs-only]
//! ```
//!
//! `tools/loop_v2_costmodel.py` reads the JSON output.

use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::time::Instant;

use ninjars::fast::tensor_layout::{monomial_index, ncoeff};
use rusticol_core::loop_v2::kernels::{
    RowLink, Segment, clear, dense_layout, fused_edge, fused_edge_blocked, lane_vector_len,
    split_edge, split_vertex,
};
use rusticol_core::loop_v2::poly::{
    MonomialProductTable, OpenLoopPolynomial, SegmentEntry, SegmentLayout, apply_segment,
};
use serde::Deserialize;
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

/// One trie node: `rows_out` output rows; `edges` edges, each attaching a
/// block with `block_rows` helicity rows to its own child object of rank
/// `rank_in` with `rows_out * block_rows` rows.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// `probe`: the design probe's shapes, which calibrate the cost model;
    /// `small-x`: nodes with at most two attached block rows, which only
    /// inform the per-node choice of kernel form.
    set: &'static str,
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

    /// The structured split-form vertex: the non-zeros of the dense one.
    fn vertex_layout(self, segment: &Segment<'_>) -> SegmentLayout {
        let dense = segment.dense_vertex_layout();
        let current = segment.current();
        let entries = dense
            .entries()
            .iter()
            .zip(dense.monomials())
            .filter(|(e, n)| {
                self.vertex_nonzero(**n, e.loop_in / current, e.loop_in % current, e.out)
            })
            .map(|(e, _)| *e)
            .collect();
        SegmentLayout::new(dense.loop_in_dimension(), dense.out_dimension(), entries).unwrap()
    }

    /// Non-zeros of `X(J) = V J` for a generic block current.
    fn fused_layout(self, segment: &Segment<'_>) -> SegmentLayout {
        let dense = segment.dense_fused_layout();
        let entries = dense
            .entries()
            .iter()
            .zip(dense.monomials())
            .filter(|(e, n)| {
                (0..segment.current()).any(|nu| self.vertex_nonzero(**n, e.loop_in, nu, e.out))
            })
            .map(|(e, _)| *e)
            .collect();
        SegmentLayout::new(dense.loop_in_dimension(), dense.out_dimension(), entries).unwrap()
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
        set: "probe",
        line: Line::Gluon,
        rows_out,
        edges,
        block_rows,
        rank_in,
    };
    let ghost = |rows_out, edges, block_rows, rank_in| Shape {
        set: "probe",
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
        Shape {
            set: "small-x",
            ..gluon(64, 1, 1, 0)
        },
        Shape {
            set: "small-x",
            ..gluon(32, 2, 1, 1)
        },
        Shape {
            set: "small-x",
            ..gluon(16, 1, 2, 2)
        },
        Shape {
            set: "small-x",
            ..gluon(4, 2, 1, 4)
        },
        Shape {
            set: "small-x",
            ..gluon(1, 1, 2, 6)
        },
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
    products: MonomialProductTable,
    dense_vertex_layout: SegmentLayout,
    structured_vertex_layout: SegmentLayout,
    dense_fused_layout: SegmentLayout,
    structured_fused_layout: SegmentLayout,
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
        let products = MonomialProductTable::new(shape.rank_in, 1);
        let segment = segment(&shape, &products);
        let dense_vertex_layout = segment.dense_vertex_layout();
        let structured_vertex_layout = shape.line.vertex_layout(&segment);
        let dense_fused_layout = segment.dense_fused_layout();
        let structured_fused_layout = shape.line.fused_layout(&segment);
        let dense_fused = dense_fused_layout.entries().len();
        let structured_fused = structured_fused_layout.entries().len();
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
                .map(|_| rng.fill(shape.block_rows * segment.current() * v))
                .collect(),
            dense_factors: (0..shape.edges)
                .map(|_| rng.fill(shape.block_rows * dense_fused * v))
                .collect(),
            structured_factors: (0..shape.edges)
                .map(|_| rng.fill(shape.block_rows * structured_fused * v))
                .collect(),
            dense_vertex: rng.fill(dense_vertex_layout.entries().len() * v),
            structured_vertex: rng.fill(structured_vertex_layout.entries().len() * v),
            links,
            output: vec![0.0; shape.rows_out * segment.output_row_len() * v],
            accumulator: vec![0.0; shape.rows_out * segment.accumulator_row_len() * v],
            shape,
            products,
            dense_vertex_layout,
            structured_vertex_layout,
            dense_fused_layout,
            structured_fused_layout,
        }
    }
}

fn segment<'a>(shape: &Shape, products: &'a MonomialProductTable) -> Segment<'a> {
    let (cut, incoming, current, outgoing) = shape.line.dims();
    Segment::new(
        cut,
        incoming,
        current,
        outgoing,
        products,
        products.output_rank(),
    )
}

/// Runs one whole node in `form` and returns its counted complex MACs per lane.
fn run_node<const L: usize>(node: &mut Node, form: Form, links: &[RowLink]) -> usize {
    let segment = segment(&node.shape, &node.products);
    let m_in = ncoeff(node.shape.rank_in);
    let rows = node.shape.rows_out;
    clear(&mut node.output);
    match form {
        Form::SplitDense | Form::SplitStructured => {
            let (layout, vertex) = if form == Form::SplitDense {
                (&node.dense_vertex_layout, &node.dense_vertex)
            } else {
                (&node.structured_vertex_layout, &node.structured_vertex)
            };
            clear(&mut node.accumulator);
            for (input, current) in node.inputs.iter().zip(&node.currents) {
                split_edge::<f64, L>(&segment, input, current, links, &mut node.accumulator);
            }
            split_vertex::<f64, L>(
                &segment,
                layout,
                vertex,
                rows,
                &node.accumulator,
                &mut node.output,
            );
            node.inputs.len() * links.len() * segment.accumulator_row_len()
                + rows * segment.cut() * layout.entries().len() * m_in
        }
        Form::FusedDense | Form::FusedStructured => {
            let (layout, factors) = if form == Form::FusedDense {
                (&node.dense_fused_layout, &node.dense_factors)
            } else {
                (&node.structured_fused_layout, &node.structured_factors)
            };
            for (input, factor) in node.inputs.iter().zip(factors) {
                fused_edge::<f64, L>(&segment, layout, input, factor, links, &mut node.output);
            }
            node.inputs.len() * links.len() * segment.cut() * layout.entries().len() * m_in
        }
        Form::FusedBlocked => {
            for (input, factor) in node.inputs.iter().zip(&node.dense_factors) {
                match (segment.incoming(), segment.outgoing()) {
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
                * segment.cut()
                * segment.incoming()
                * segment.outgoing()
                * ncoeff(segment.degree())
                * m_in
        }
    }
}

fn shape_json(shape: &Shape) -> Value {
    let (cut, incoming, current, outgoing) = shape.line.dims();
    json!({
        "set": shape.set,
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
    print!(
        "{:>7} {:>6} {:>4} {:>3} {:>4} |",
        "set", "line", "rows", "X", "r_in"
    );
    for form in Form::ALL {
        print!(" {:>16}", form.name());
    }
    println!();
    for shape in trie_shapes() {
        let mut node = Node::new::<L>(shape, &mut rng);
        let links = node.links.clone();
        print!(
            "{:>7} {:>6} {:>4} {:>3} {:>4} |",
            shape.set,
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

    poly_layout::<L>(target, &mut rng, records);

    // Fixed overheads, which the counted work does not cover. Both are lower
    // bounds of executor dispatch: they time empty calls, not the irregular
    // row gathers or cache misses of real tries.
    // - per node: a node without output rows (calls and setup only);
    // - per edge call: the marginal cost of an edge call without links;
    // - fit: least-squares intercept and slope of the time of a one-row,
    //   rank-0 node over 1..=8 single-link edges (both include work: the
    //   output clear and the split vertex in the intercept, the link's
    //   products in the slope).
    for form in Form::ALL {
        let single_row = |edges| Shape {
            set: "dispatch",
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
            "  dispatch (lower bound) {:>16}: per node {:6.1} ns, per edge call {:5.1} ns; fit {:6.1} ns + {:6.1} ns/edge",
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

/// The same fused step in `poly.rs`'s storage, `Complex<f64>` per lane, by
/// [`apply_segment`]: for each gluon probe shape, every edge dresses an
/// object of `rows_out * block_rows * cut` rows with the structured fused
/// layout. `apply_segment` has no row links (all rows share one factor), so
/// this compares the lane storage of the two layouts, not whole nodes; the
/// output accumulates across repetitions.
fn poly_layout<const L: usize>(target: f64, rng: &mut Rng, records: &mut Vec<Value>) {
    println!("  poly.rs apply_segment (Complex<f64> lanes), structured fused layout:");
    for shape in trie_shapes() {
        if shape.set != "probe" || shape.line != Line::Gluon {
            continue;
        }
        let products = MonomialProductTable::new(shape.rank_in, 1);
        let segment = segment(&shape, &products);
        let layout = shape.line.fused_layout(&segment);
        let rows = shape.rows_out * shape.block_rows * segment.cut();
        let zero = Complex::new(0.0, 0.0);
        let complex = |rng: &mut Rng, count: usize| -> Vec<Complex<f64>> {
            rng.fill(2 * count)
                .chunks_exact(2)
                .map(|pair| Complex::new(pair[0], pair[1]))
                .collect()
        };
        let inputs: Vec<OpenLoopPolynomial<Complex<f64>>> = (0..shape.edges)
            .map(|_| {
                let mut object =
                    OpenLoopPolynomial::zeros(rows, segment.incoming(), shape.rank_in, L, &zero);
                for row in 0..rows {
                    for column in 0..segment.incoming() {
                        for monomial in 0..ncoeff(shape.rank_in) {
                            object
                                .coefficient_mut(row, column, monomial)
                                .clone_from_slice(&complex(rng, L));
                        }
                    }
                }
                object
            })
            .collect();
        let coefficients: Vec<Vec<Complex<f64>>> = (0..shape.edges)
            .map(|_| complex(rng, layout.entries().len() * L))
            .collect();
        let mut output =
            OpenLoopPolynomial::zeros(rows, segment.outgoing(), products.output_rank(), L, &zero);
        let mut run = || {
            for (input, coefficients) in inputs.iter().zip(&coefficients) {
                apply_segment(input, &layout, coefficients, &products, &mut output).unwrap();
            }
            black_box(&output);
        };
        let seconds = time_per_call(target, &mut run);
        let work = shape.edges * rows * layout.entries().len() * ncoeff(shape.rank_in);
        let rate = (work * L) as f64 / seconds;
        println!(
            "  {:>6} {:>4} {:>3} {:>4} | {:>16.3}",
            shape.line.name(),
            shape.rows_out,
            shape.edges * shape.block_rows,
            shape.rank_in,
            rate / 1e9
        );
        records.push(json!({
            "kind": "poly-layout",
            "lanes": L,
            "form": "poly-apply-segment",
            "shape": shape_json(&shape),
            "work_cmac_per_lane": work,
            "seconds": seconds,
            "gcmac_per_s": rate / 1e9,
        }));
    }
}

// ------------------------------------------------------------------ real tries
//
// Gate G0 on real tries: the schedule-B programs that
// `tools/loop_v2_costmodel.py --export-programs` writes from the reference
// executor's walk (`ReferenceExecutor.run_b`: the families, schedules and
// kernel catalogue of `loops/v2`). A program lists, per step (a dressed trie
// node, or one cut species of a family accumulator), its edges with the child
// step or leaf seed, the content (the position's segment with its catalogue
// entries) and the helicity row links, and the split-form vertex groups.
//
// The runner executes a program as the executor will: families in order, each
// trie depth-first, every child dressed into its parent as soon as it is
// complete, so live objects are one output (and, in split form, one set of
// vertex-group accumulators) per depth. Fused factors X(J) are built per
// (content, frame position) from the block currents and vertex tensors before
// the walk and are part of the timed run; block currents, vertex tensors and
// seeds are per-point inputs (random values).
//
// A step dressing leaf seeds reads a seed s[h] times the identity in
// (alpha, beta), so it runs the same kernels on a collapsed segment: cut and
// open end of dimension 1, output end alpha x beta'. Dense vertex tensors hold
// the structured values, so every form computes the same family numerators;
// `check_program` asserts it before any timing.

/// A program file of `tools/loop_v2_costmodel.py --export-programs`.
#[derive(Deserialize)]
struct ProgramFile {
    process: String,
    colour: String,
    families: usize,
    skeletons: usize,
    trie_nodes: usize,
    work: BTreeMap<String, u64>,
    contents: Vec<ContentFile>,
    groups: Vec<GroupFile>,
    currents: Vec<(usize, usize)>,
    factors: Vec<(usize, Vec<usize>)>,
    vertices: Vec<usize>,
    seeds: Vec<usize>,
    steps: Vec<StepFile>,
}

#[derive(Deserialize)]
struct ContentFile {
    d_in: usize,
    d_out: usize,
    degree: usize,
    rows: usize,
    fused: Vec<[u32; 6]>,
    split: Vec<(usize, Vec<usize>)>,
}

#[derive(Deserialize)]
struct GroupFile {
    width: usize,
    d_in: usize,
    d_out: usize,
    entries: Vec<[u32; 7]>,
}

#[derive(Deserialize)]
struct StepFile {
    depth: usize,
    seeded: bool,
    trace: bool,
    rows: usize,
    d_alpha: usize,
    d_out: usize,
    degree: usize,
    storage_rank_in: usize,
    work: StepWorkFile,
    edges: Vec<EdgeFile>,
    groups: Vec<StepGroupFile>,
}

#[derive(Deserialize)]
struct StepWorkFile {
    fused: u64,
    split: u64,
}

#[derive(Deserialize)]
struct EdgeFile {
    seed: i64,
    child: i64,
    content: usize,
    factor: usize,
    links: Vec<u32>,
    split: Vec<SplitFile>,
}

#[derive(Deserialize)]
struct SplitFile {
    current: usize,
    group: usize,
    links: Vec<u32>,
}

#[derive(Deserialize)]
struct StepGroupFile {
    group: usize,
    vertex: usize,
    targets: Vec<usize>,
}

/// Kernel structure of a content (one trie position) in both step kinds.
struct Content {
    d_in: usize,
    rows: usize,
    /// `[open-loop input, seeded]` structured fused layouts (same entry order).
    fused: [SegmentLayout; 2],
    /// `[open-loop input, seeded]` dense fused layouts (same entry order).
    dense: [SegmentLayout; 2],
    /// Per split part: vertex group and the content rows it covers.
    split: Vec<(usize, Vec<usize>)>,
    /// Per split part: (vertex entry, current component, fused entry, dense entry).
    build: Vec<Vec<(usize, usize, usize, usize)>>,
}

struct Group {
    width: usize,
    d_in: usize,
    entries: usize,
    /// `[open-loop input, seeded]` structured and dense vertex layouts.
    structured: [SegmentLayout; 2],
    dense: [SegmentLayout; 2],
    /// `[open-loop input, seeded]` position of each structured entry in the
    /// dense layout, so both forms can hold one vertex tensor.
    dense_index: [Vec<usize>; 2],
}

enum Input {
    Seed(usize),
    Step(usize),
}

struct Part {
    current: usize,
    group: usize,
    slot: usize,
    links: Vec<RowLink>,
}

struct Edge {
    input: Input,
    content: usize,
    factor: usize,
    /// Products table of (input rank, content degree).
    products: usize,
    links: Vec<RowLink>,
    parts: Vec<Part>,
}

struct StepGroup {
    group: usize,
    vertex: usize,
    targets: Vec<usize>,
    all_rows: bool,
    /// Lane vectors per accumulator row.
    row: usize,
}

struct Step {
    depth: usize,
    seeded: bool,
    trace: bool,
    rows: usize,
    d_alpha: usize,
    d_out: usize,
    rank_in: usize,
    rank_out: usize,
    /// Products table of (input rank, step degree), for the split form.
    products: usize,
    edges: Vec<Edge>,
    groups: Vec<StepGroup>,
    /// Work the reference executor counts: (fused, split).
    reference: (u64, u64),
}

impl Step {
    /// Lane vectors per output row.
    fn out_row(&self) -> usize {
        self.d_alpha * self.d_out * ncoeff(self.rank_out)
    }

    fn links(&self) -> usize {
        self.edges.iter().map(|edge| edge.links.len()).sum()
    }
}

struct Program {
    process: String,
    colour: String,
    families: usize,
    skeletons: usize,
    trie_nodes: usize,
    reference_work: BTreeMap<String, u64>,
    contents: Vec<Content>,
    groups: Vec<Group>,
    /// Per block current: (content, split part).
    currents: Vec<(usize, usize)>,
    /// Per fused factor: content, and the vertex tensor and block current of
    /// each split part.
    factors: Vec<(usize, Vec<usize>, Vec<usize>)>,
    vertices: Vec<usize>,
    /// Whether each vertex tensor is read by seeded steps (frame position 0).
    vertex_seeded: Vec<bool>,
    seeds: Vec<usize>,
    steps: Vec<Step>,
    roots: Vec<usize>,
    products: Vec<MonomialProductTable>,
    depth: usize,
}

fn links(flat: &[u32]) -> Vec<RowLink> {
    flat.chunks_exact(3)
        .map(|link| RowLink {
            output: link[0],
            input: link[1],
            block: link[2],
        })
        .collect()
}

fn entry(loop_in: usize, out: usize, exponents: &[u32]) -> SegmentEntry {
    SegmentEntry {
        loop_in,
        out,
        exponents: [exponents[0], exponents[1], exponents[2], exponents[3]],
    }
}

fn layout(loop_in: usize, out: usize, entries: Vec<SegmentEntry>) -> SegmentLayout {
    SegmentLayout::new(loop_in, out, entries).expect("a consistent program layout")
}

impl Program {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("read the program file");
        let file: ProgramFile = serde_json::from_str(&text).expect("parse the program file");
        let contents: Vec<Content> = file
            .contents
            .iter()
            .map(|c| {
                let fused = |seeded: bool| {
                    let entries = c.fused.iter().map(|e| {
                        let (b, bp) = (e[0] as usize, e[1] as usize);
                        if seeded {
                            entry(0, b * c.d_out + bp, &e[2..])
                        } else {
                            entry(b, bp, &e[2..])
                        }
                    });
                    let (loop_in, out) = if seeded {
                        (1, c.d_in * c.d_out)
                    } else {
                        (c.d_in, c.d_out)
                    };
                    layout(loop_in, out, entries.collect())
                };
                let index: HashMap<SegmentEntry, usize> = fused(false)
                    .entries()
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (*e, i))
                    .collect();
                let build = c
                    .split
                    .iter()
                    .map(|(group, _rows)| {
                        let group = &file.groups[*group];
                        group
                            .entries
                            .iter()
                            .enumerate()
                            .map(|(i, e)| {
                                let (b, nu, bp) = (e[0] as usize, e[1] as usize, e[2] as usize);
                                let target = entry(b, bp, &e[3..]);
                                let n = monomial_index(target.exponents);
                                (i, nu, index[&target], (n * c.d_in + b) * c.d_out + bp)
                            })
                            .collect()
                    })
                    .collect();
                Content {
                    d_in: c.d_in,
                    rows: c.rows,
                    fused: [fused(false), fused(true)],
                    dense: [
                        dense_layout(c.d_in, c.d_out, c.degree),
                        dense_layout(1, c.d_in * c.d_out, c.degree),
                    ],
                    split: c.split.clone(),
                    build,
                }
            })
            .collect();
        let groups = file
            .groups
            .iter()
            .map(|g| {
                let degree = g
                    .entries
                    .iter()
                    .map(|e| e[3..].iter().sum::<u32>() as usize)
                    .max()
                    .unwrap_or(0);
                let vertex = |seeded: bool| {
                    let entries = g.entries.iter().map(|e| {
                        let (b, nu, bp) = (e[0] as usize, e[1] as usize, e[2] as usize);
                        if seeded {
                            entry(nu, b * g.d_out + bp, &e[3..])
                        } else {
                            entry(b * g.width + nu, bp, &e[3..])
                        }
                    });
                    let (loop_in, out) = if seeded {
                        (g.width, g.d_in * g.d_out)
                    } else {
                        (g.d_in * g.width, g.d_out)
                    };
                    layout(loop_in, out, entries.collect())
                };
                let dense_index = |seeded: bool| {
                    g.entries
                        .iter()
                        .map(|e| {
                            let (b, nu, bp) = (e[0] as usize, e[1] as usize, e[2] as usize);
                            let n = monomial_index([e[3], e[4], e[5], e[6]]);
                            if seeded {
                                (n * g.width + nu) * g.d_in * g.d_out + b * g.d_out + bp
                            } else {
                                (n * g.d_in * g.width + b * g.width + nu) * g.d_out + bp
                            }
                        })
                        .collect()
                };
                Group {
                    width: g.width,
                    d_in: g.d_in,
                    entries: g.entries.len(),
                    structured: [vertex(false), vertex(true)],
                    dense: [
                        dense_layout(g.d_in * g.width, g.d_out, degree),
                        dense_layout(g.width, g.d_in * g.d_out, degree),
                    ],
                    dense_index: [dense_index(false), dense_index(true)],
                }
            })
            .collect();
        // Output rank of a step: the input rank its consumer stores objects at.
        let mut rank_out: Vec<usize> = file
            .steps
            .iter()
            .map(|s| s.storage_rank_in + s.degree)
            .collect();
        let mut consumers = vec![0usize; file.steps.len()];
        for step in &file.steps {
            for edge in &step.edges {
                if edge.child >= 0 {
                    rank_out[edge.child as usize] = step.storage_rank_in;
                    consumers[edge.child as usize] += 1;
                }
            }
        }
        for (s, step) in file.steps.iter().enumerate() {
            assert_eq!(
                consumers[s],
                usize::from(!step.trace),
                "one consumer per object"
            );
        }
        let mut tables: HashMap<(usize, usize), usize> = HashMap::new();
        let mut products = Vec::new();
        let mut table = |rank: usize, degree: usize| {
            *tables.entry((rank, degree)).or_insert_with(|| {
                products.push(MonomialProductTable::new(rank, degree));
                products.len() - 1
            })
        };
        let steps: Vec<Step> = file
            .steps
            .iter()
            .enumerate()
            .map(|(s, step)| {
                let rank_in = if step.seeded { 0 } else { step.storage_rank_in };
                let edges = step
                    .edges
                    .iter()
                    .map(|edge| Edge {
                        input: if edge.seed >= 0 {
                            Input::Seed(edge.seed as usize)
                        } else {
                            Input::Step(edge.child as usize)
                        },
                        content: edge.content,
                        factor: edge.factor,
                        products: table(rank_in, file.contents[edge.content].degree),
                        links: links(&edge.links),
                        parts: edge
                            .split
                            .iter()
                            .map(|part| {
                                let (content, k) = file.currents[part.current];
                                Part {
                                    current: part.current,
                                    group: file.contents[content].split[k].0,
                                    slot: part.group,
                                    links: links(&part.links),
                                }
                            })
                            .collect(),
                    })
                    .collect();
                let groups = step
                    .groups
                    .iter()
                    .map(|g| {
                        let group = &file.groups[g.group];
                        StepGroup {
                            group: g.group,
                            vertex: g.vertex,
                            all_rows: g.targets.len() == step.rows,
                            targets: g.targets.clone(),
                            row: if step.seeded {
                                group.width
                            } else {
                                step.d_alpha * group.d_in * group.width * ncoeff(rank_in)
                            },
                        }
                    })
                    .collect();
                Step {
                    depth: step.depth,
                    seeded: step.seeded,
                    trace: step.trace,
                    rows: step.rows,
                    d_alpha: step.d_alpha,
                    d_out: step.d_out,
                    rank_in,
                    rank_out: rank_out[s],
                    products: table(rank_in, step.degree),
                    edges,
                    groups,
                    reference: (step.work.fused, step.work.split),
                }
            })
            .collect();
        let roots = (0..steps.len()).filter(|&s| steps[s].trace).collect();
        let mut vertex_seeded = vec![None; file.vertices.len()];
        for step in &steps {
            for g in &step.groups {
                let seen = vertex_seeded[g.vertex].replace(step.seeded);
                assert!(
                    seen.is_none_or(|seeded| seeded == step.seeded),
                    "a vertex tensor read by seeded and open-loop steps"
                );
            }
        }
        let current_of: HashMap<(usize, usize), usize> = file
            .currents
            .iter()
            .enumerate()
            .map(|(index, key)| (*key, index))
            .collect();
        let factors = file
            .factors
            .into_iter()
            .map(|(content, vertices)| {
                let currents = (0..vertices.len())
                    .map(|k| current_of[&(content, k)])
                    .collect();
                (content, vertices, currents)
            })
            .collect();
        let depth = steps.iter().map(|s| s.depth).max().unwrap_or(0);
        Self {
            process: file.process,
            colour: file.colour,
            families: file.families,
            skeletons: file.skeletons,
            trie_nodes: file.trie_nodes,
            reference_work: file.work,
            contents,
            groups,
            currents: file.currents,
            factors,
            vertices: file.vertices,
            vertex_seeded: vertex_seeded
                .into_iter()
                .map(|seeded| seeded.unwrap_or(false))
                .collect(),
            seeds: file.seeds,
            steps,
            roots,
            products,
            depth,
        }
    }

    fn segment(&self, step: &Step, d_in: usize, current: usize, products: usize) -> Segment<'_> {
        let products = &self.products[products];
        if step.seeded {
            Segment::new(1, 1, current, d_in * step.d_out, products, step.rank_out)
        } else {
            Segment::new(
                step.d_alpha,
                d_in,
                current,
                step.d_out,
                products,
                step.rank_out,
            )
        }
    }

    /// Complex MACs per lane of one step in `form`, as the kernels perform
    /// them, and the kernels' inner-loop iterations (one per input row piece
    /// and entry: the work without the factor of input coefficients).
    fn step_cost(&self, step: &Step, form: Form) -> Cost {
        let (cut, m_in) = if step.seeded {
            (1, 1)
        } else {
            (step.d_alpha, ncoeff(step.rank_in))
        };
        let seeded = usize::from(step.seeded);
        let mut iterations = 0;
        match form {
            Form::FusedStructured | Form::FusedDense | Form::FusedBlocked => {
                for edge in &step.edges {
                    let content = &self.contents[edge.content];
                    let entries = if form == Form::FusedStructured {
                        content.fused[seeded].entries().len()
                    } else {
                        content.dense[seeded].entries().len()
                    };
                    iterations += edge.links.len() * cut * entries;
                }
            }
            Form::SplitStructured | Form::SplitDense => {
                for edge in &step.edges {
                    let incoming = if step.seeded {
                        1
                    } else {
                        self.contents[edge.content].d_in
                    };
                    for part in &edge.parts {
                        iterations +=
                            part.links.len() * cut * incoming * self.groups[part.group].width;
                    }
                }
                for g in &step.groups {
                    let group = &self.groups[g.group];
                    let entries = if form == Form::SplitStructured {
                        group.structured[seeded].entries().len()
                    } else {
                        group.dense[seeded].entries().len()
                    };
                    iterations += g.targets.len() * cut * entries;
                }
            }
        }
        let mut cost = Cost {
            work: (iterations * m_in) as u64,
            iterations: iterations as u64,
        };
        if step.trace {
            cost.work += (step.d_alpha * ncoeff(step.rank_out)) as u64;
            cost.iterations += step.d_alpha as u64;
        }
        cost
    }

    /// Complex MACs per lane of building the fused factors used by `fused`.
    fn build_work(&self, used: &[bool]) -> u64 {
        let mut work = 0;
        for (factor, (content, _vertices, _currents)) in self.factors.iter().enumerate() {
            if !used[factor] {
                continue;
            }
            let content = &self.contents[*content];
            for ((_group, rows), build) in content.split.iter().zip(&content.build) {
                work += rows.len() * build.len();
            }
        }
        work as u64
    }
}

/// Kernel work of one step in one form.
#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    /// Complex MACs per lane.
    work: u64,
    /// Inner-loop iterations (work without the input-coefficient factor).
    iterations: u64,
}

/// Per-point inputs and buffers of a program at `L` lanes.
struct Data {
    seeds: Vec<Vec<f64>>,
    currents: Vec<Vec<f64>>,
    vertices: Vec<Vec<f64>>,
    dense_vertices: Vec<Vec<f64>>,
    fused: Vec<Vec<f64>>,
    dense: Vec<Vec<f64>>,
}

struct Level {
    out: Vec<f64>,
    accumulators: Vec<Vec<f64>>,
}

impl Data {
    fn new<const L: usize>(program: &Program, rng: &mut Rng) -> Self {
        let v = lane_vector_len(L);
        let fill = |rng: &mut Rng, count: usize| rng.fill(count * v);
        let mut data = Self {
            seeds: program.seeds.iter().map(|&rows| fill(rng, rows)).collect(),
            currents: program
                .currents
                .iter()
                .map(|&(content, k)| {
                    let content = &program.contents[content];
                    fill(rng, content.rows * program.groups[content.split[k].0].width)
                })
                .collect(),
            vertices: program
                .vertices
                .iter()
                .map(|&g| fill(rng, program.groups[g].entries))
                .collect(),
            dense_vertices: Vec::new(),
            fused: program
                .factors
                .iter()
                .map(|(content, _, _)| {
                    let content = &program.contents[*content];
                    vec![0.0; content.rows * content.fused[0].entries().len() * v]
                })
                .collect(),
            dense: program
                .factors
                .iter()
                .map(|(content, _, _)| {
                    let content = &program.contents[*content];
                    vec![0.0; content.rows * content.dense[0].entries().len() * v]
                })
                .collect(),
        };
        // The dense vertex tensors hold the structured values (zero elsewhere),
        // so every form computes the same numerators (`check_program`).
        data.dense_vertices = program
            .vertices
            .iter()
            .zip(&program.vertex_seeded)
            .zip(&data.vertices)
            .map(|((&g, &seeded), values)| {
                let group = &program.groups[g];
                let seeded = usize::from(seeded);
                let mut dense = vec![0.0; group.dense[seeded].entries().len() * v];
                for (i, &slot) in group.dense_index[seeded].iter().enumerate() {
                    dense[slot * v..][..v].copy_from_slice(&values[i * v..][..v]);
                }
                dense
            })
            .collect();
        data
    }
}

fn levels<const L: usize>(program: &Program) -> Vec<Level> {
    let v = lane_vector_len(L);
    (0..=program.depth)
        .map(|depth| {
            let steps = || program.steps.iter().filter(move |s| s.depth == depth);
            let out = steps().map(|s| s.rows * s.out_row()).max().unwrap_or(0);
            let slots = steps().map(|s| s.groups.len()).max().unwrap_or(0);
            let accumulator = steps()
                .flat_map(|s| s.groups.iter().map(move |g| s.rows * g.row))
                .max()
                .unwrap_or(0);
            Level {
                out: vec![0.0; out * v],
                accumulators: vec![vec![0.0; accumulator * v]; slots],
            }
        })
        .collect()
}

#[inline(always)]
fn lane_cmac<const L: usize>(acc: &mut [f64], x: &[f64], y: &[f64]) {
    let (acc_re, acc_im) = acc[..2 * L].split_at_mut(L);
    let (x_re, x_im) = x[..2 * L].split_at(L);
    let (y_re, y_im) = y[..2 * L].split_at(L);
    for l in 0..L {
        acc_re[l] += x_re[l] * y_re[l] - x_im[l] * y_im[l];
        acc_im[l] += x_re[l] * y_im[l] + x_im[l] * y_re[l];
    }
}

/// X(J) of every used factor: sum over split parts of the vertex tensor at
/// the factor's frame position times the content's block currents.
fn build_factors<const L: usize>(program: &Program, data: &mut Data, used: &[bool], dense: bool) {
    let v = lane_vector_len(L);
    let Data {
        currents,
        vertices,
        fused,
        dense: dense_factors,
        ..
    } = data;
    for (factor, (content, vertex_of, current_of)) in program.factors.iter().enumerate() {
        if !used[factor] {
            continue;
        }
        let content = &program.contents[*content];
        let target = if dense {
            &mut dense_factors[factor]
        } else {
            &mut fused[factor]
        };
        clear(target);
        let row_len = if dense {
            content.dense[0].entries().len()
        } else {
            content.fused[0].entries().len()
        } * v;
        for (k, ((group, rows), build)) in content.split.iter().zip(&content.build).enumerate() {
            let width = program.groups[*group].width;
            let vertex = &vertices[vertex_of[k]];
            let current = &currents[current_of[k]];
            for &row in rows {
                let x = &mut target[row * row_len..][..row_len];
                let j = &current[row * width * v..][..width * v];
                for &(i, nu, structured, full) in build {
                    let slot = if dense { full } else { structured };
                    lane_cmac::<L>(&mut x[slot * v..], &vertex[i * v..], &j[nu * v..]);
                }
            }
        }
    }
}

/// What one program run executes per step.
struct Run<'a> {
    program: &'a Program,
    data: &'a Data,
    /// Kernel form per step; `None` walks the program without kernels.
    policy: &'a [Option<Form>],
}

/// Dresses step `s` (after its children) into its level's output; returns
/// the elapsed time of the step and its subtree when `own` records times.
fn run_step<const L: usize>(
    run: &Run<'_>,
    s: usize,
    levels: &mut [Level],
    numerator: &mut [f64],
    mut own: Option<&mut [f64]>,
) -> f64 {
    let start = own.as_ref().map(|_| Instant::now());
    let mut children = 0.0;
    let program = run.program;
    let data = run.data;
    let step = &program.steps[s];
    let v = lane_vector_len(L);
    let seeded = usize::from(step.seeded);
    let out_row = step.out_row() * v;
    let (lower, upper) = levels.split_at_mut(step.depth);
    let Level { out, accumulators } = &mut upper[0];
    let out = &mut out[..step.rows * out_row];
    clear(out);
    let form = run.policy[s];
    let split = matches!(form, Some(Form::SplitStructured | Form::SplitDense));
    if split {
        for (g, accumulator) in step.groups.iter().zip(accumulators.iter_mut()) {
            clear(&mut accumulator[..step.rows * g.row * v]);
        }
    }
    for edge in &step.edges {
        let input: &[f64] = match edge.input {
            Input::Seed(seed) => &data.seeds[seed],
            Input::Step(child) => {
                children += run_step::<L>(run, child, lower, numerator, own.as_deref_mut());
                let c = &program.steps[child];
                &lower[c.depth].out[..c.rows * c.out_row() * v]
            }
        };
        let content = &program.contents[edge.content];
        match form {
            None => {}
            Some(Form::FusedStructured) => {
                let segment = program.segment(step, content.d_in, 1, edge.products);
                let factor = &data.fused[edge.factor];
                fused_edge::<f64, L>(
                    &segment,
                    &content.fused[seeded],
                    input,
                    factor,
                    &edge.links,
                    out,
                );
            }
            Some(Form::FusedDense) => {
                let segment = program.segment(step, content.d_in, 1, edge.products);
                let factor = &data.dense[edge.factor];
                fused_edge::<f64, L>(
                    &segment,
                    &content.dense[seeded],
                    input,
                    factor,
                    &edge.links,
                    out,
                );
            }
            Some(Form::FusedBlocked) => {
                let segment = program.segment(step, content.d_in, 1, edge.products);
                let factor = &data.dense[edge.factor];
                let links = &edge.links;
                match (segment.incoming(), segment.outgoing()) {
                    (4, 4) => {
                        fused_edge_blocked::<f64, L, 4, 4>(&segment, input, factor, links, out)
                    }
                    (1, 1) => {
                        fused_edge_blocked::<f64, L, 1, 1>(&segment, input, factor, links, out)
                    }
                    (1, 16) => {
                        fused_edge_blocked::<f64, L, 1, 16>(&segment, input, factor, links, out)
                    }
                    _ => fused_edge::<f64, L>(
                        &segment,
                        &content.dense[seeded],
                        input,
                        factor,
                        links,
                        out,
                    ),
                }
            }
            Some(Form::SplitStructured | Form::SplitDense) => {
                for part in &edge.parts {
                    let group = &program.groups[part.group];
                    let segment = program.segment(step, content.d_in, group.width, step.products);
                    let g = &step.groups[part.slot];
                    split_edge::<f64, L>(
                        &segment,
                        input,
                        &data.currents[part.current],
                        &part.links,
                        &mut accumulators[part.slot][..step.rows * g.row * v],
                    );
                }
            }
        }
    }
    if split {
        let dense = form == Some(Form::SplitDense);
        for (g, accumulator) in step.groups.iter().zip(accumulators.iter()) {
            let group = &program.groups[g.group];
            let segment = program.segment(step, group.d_in, group.width, step.products);
            let (layout, vertex) = if dense {
                (&group.dense[seeded], &data.dense_vertices[g.vertex])
            } else {
                (&group.structured[seeded], &data.vertices[g.vertex])
            };
            let row = g.row * v;
            if g.all_rows {
                split_vertex::<f64, L>(
                    &segment,
                    layout,
                    vertex,
                    step.rows,
                    &accumulator[..step.rows * row],
                    out,
                );
            } else {
                for &t in &g.targets {
                    split_vertex::<f64, L>(
                        &segment,
                        layout,
                        vertex,
                        1,
                        &accumulator[t * row..][..row],
                        &mut out[t * out_row..][..out_row],
                    );
                }
            }
        }
    }
    if step.trace && form.is_some() {
        // Close the cut: trace over alpha = beta of the single row.
        let m = ncoeff(step.rank_out) * v;
        for a in 0..step.d_alpha {
            let source = &out[(a * step.d_out + a) * m..][..m];
            for (n, x) in numerator[..m].iter_mut().zip(source) {
                *n += x;
            }
        }
    }
    match (start, own) {
        (Some(start), Some(own)) => {
            let total = start.elapsed().as_secs_f64();
            own[s] = total - children;
            total
        }
        _ => 0.0,
    }
}

/// Factors the fused-form edges of `policy` read.
fn used_factors(program: &Program, policy: &[Option<Form>], dense: bool) -> Vec<bool> {
    let mut used = vec![false; program.factors.len()];
    for (step, form) in program.steps.iter().zip(policy) {
        let wanted = match form {
            Some(Form::FusedStructured) => !dense,
            Some(Form::FusedDense | Form::FusedBlocked) => dense,
            _ => false,
        };
        if wanted {
            for edge in &step.edges {
                used[edge.factor] = true;
            }
        }
    }
    used
}

/// Seconds per batch of one program run (factor build, walk), best of `reps`
/// after a warm-up; with `own`, the best own time per step.
struct Timing {
    build: f64,
    walk: f64,
    own: Vec<f64>,
}

/// One run of a program: build the factors `policy` reads, then dress every
/// family (one numerator per root step); returns (build, walk) seconds.
fn execute<const L: usize>(
    program: &Program,
    data: &mut Data,
    policy: &[Option<Form>],
    levels: &mut [Level],
    numerators: &mut [Vec<f64>],
    mut own: Option<&mut [f64]>,
) -> (f64, f64) {
    let start = Instant::now();
    build_factors::<L>(program, data, &used_factors(program, policy, false), false);
    build_factors::<L>(program, data, &used_factors(program, policy, true), true);
    let build = start.elapsed().as_secs_f64();
    let run = Run {
        program,
        data,
        policy,
    };
    let start = Instant::now();
    for (&root, numerator) in program.roots.iter().zip(numerators.iter_mut()) {
        clear(numerator);
        run_step::<L>(&run, root, levels, numerator, own.as_deref_mut());
    }
    let walk = start.elapsed().as_secs_f64();
    black_box(&numerators);
    (build, walk)
}

fn numerators<const L: usize>(program: &Program) -> Vec<Vec<f64>> {
    program
        .roots
        .iter()
        .map(|&root| vec![0.0; ncoeff(program.steps[root].rank_out) * lane_vector_len(L)])
        .collect()
}

fn time_program<const L: usize>(
    program: &Program,
    data: &mut Data,
    policy: &[Option<Form>],
    reps: usize,
    per_step: bool,
) -> Timing {
    let mut levels = levels::<L>(program);
    let mut numerators = numerators::<L>(program);
    let mut best = Timing {
        build: f64::INFINITY,
        walk: f64::INFINITY,
        own: vec![f64::INFINITY; if per_step { program.steps.len() } else { 0 }],
    };
    let mut own = vec![0.0; best.own.len()];
    for rep in 0..=reps {
        let (build, walk) = execute::<L>(
            program,
            data,
            policy,
            &mut levels,
            &mut numerators,
            per_step.then_some(&mut own[..]),
        );
        if rep == 0 {
            continue; // warm-up
        }
        best.build = best.build.min(build);
        best.walk = best.walk.min(walk);
        for (b, o) in best.own.iter_mut().zip(&own) {
            *b = b.min(*o);
        }
    }
    best
}

/// Every kernel form must give the same family numerators (the dense forms
/// hold the structured tensors), so the program is interpreted consistently.
/// Returns the largest deviation from the fused-structured numerators,
/// relative to their largest component.
fn check_program<const L: usize>(program: &Program, data: &mut Data) -> f64 {
    let mut levels = levels::<L>(program);
    let mut run = |form: Form| {
        let mut numerators = numerators::<L>(program);
        let policy = vec![Some(form); program.steps.len()];
        execute::<L>(program, data, &policy, &mut levels, &mut numerators, None);
        numerators
    };
    let reference = run(Form::FusedStructured);
    let scale = reference
        .iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(
        scale > 0.0,
        "{} {}: zero numerators",
        program.process,
        program.colour
    );
    let mut worst = 0.0f64;
    for form in Form::ALL {
        let numerators = run(form);
        for (a, b) in numerators.iter().flatten().zip(reference.iter().flatten()) {
            worst = worst.max((a - b).abs() / scale);
        }
    }
    worst
}

/// Per-step time model of one kernel form: `t = per_step + per_iteration *
/// iterations + work * lanes / rate`, a non-negative least-squares fit of the
/// own step times (a coefficient that would be negative is dropped).
#[derive(Clone, Copy, Debug)]
struct StepModel {
    per_step: f64,
    per_iteration: f64,
    per_cmac: f64,
}

impl StepModel {
    fn fit(program: &Program, form: Form, own: &[f64], lanes: usize) -> Self {
        let rows: Vec<([f64; 3], f64)> = program
            .steps
            .iter()
            .zip(own)
            .map(|(step, &t)| {
                let cost = program.step_cost(step, form);
                (
                    [
                        1.0,
                        cost.iterations as f64,
                        (cost.work * lanes as u64) as f64,
                    ],
                    t,
                )
            })
            .collect();
        let mut active = vec![0usize, 1, 2];
        loop {
            let x = least_squares(&rows, &active);
            match active.iter().position(|&k| x[k] < 0.0) {
                Some(drop) => {
                    active.remove(drop);
                }
                None => {
                    return Self {
                        per_step: x[0],
                        per_iteration: x[1],
                        per_cmac: x[2],
                    };
                }
            }
        }
    }

    fn seconds(&self, cost: Cost, lanes: usize) -> f64 {
        self.per_step
            + self.per_iteration * cost.iterations as f64
            + self.per_cmac * (cost.work * lanes as u64) as f64
    }

    fn json(&self) -> Value {
        json!({
            "per_step_seconds": self.per_step,
            "per_iteration_seconds": self.per_iteration,
            "gcmac_per_s": if self.per_cmac > 0.0 { 1.0 / self.per_cmac / 1e9 } else { f64::INFINITY },
        })
    }
}

/// Least squares over the `active` columns of `rows` (others are zero).
fn least_squares(rows: &[([f64; 3], f64)], active: &[usize]) -> [f64; 3] {
    let n = active.len();
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];
    for (x, t) in rows {
        for (i, &ki) in active.iter().enumerate() {
            for (j, &kj) in active.iter().enumerate() {
                a[i][j] += x[ki] * x[kj];
            }
            b[i] += x[ki] * t;
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut solution = vec![0.0f64; n];
    for row in (0..n).rev() {
        let rest: f64 = (row + 1..n).map(|k| a[row][k] * solution[k]).sum();
        solution[row] = (b[row] - rest) / a[row][row];
    }
    let mut x = [0.0f64; 3];
    for (&k, value) in active.iter().zip(solution) {
        x[k] = value;
    }
    x
}

fn policy_json(program: &Program, policy: &[Option<Form>]) -> Value {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for form in policy.iter().flatten() {
        *counts.entry(form.name()).or_default() += 1;
    }
    let work: u64 = program
        .steps
        .iter()
        .zip(policy)
        .map(|(step, form)| form.map_or(0, |f| program.step_cost(step, f).work))
        .sum();
    json!({"steps_per_form": counts, "work_cmac_per_lane": work})
}

fn run_program<const L: usize>(program: &Program, reps: usize, records: &mut Vec<Value>) {
    let mut rng = Rng(11);
    let mut data = Data::new::<L>(program, &mut rng);
    let steps = program.steps.len();
    let edges: usize = program.steps.iter().map(|s| s.edges.len()).sum();
    let links: usize = program.steps.iter().map(Step::links).sum();
    let reference_best = program
        .reference_work
        .get("best/seed")
        .copied()
        .unwrap_or(0)
        + program
            .reference_work
            .get("best/step")
            .copied()
            .unwrap_or(0);
    println!(
        "program {} {} at {L} lanes: {} families, {} skeletons, {} trie nodes, {steps} steps, {edges} edges, {links} links; reference best work {reference_best} CMAC/pt",
        program.process, program.colour, program.families, program.skeletons, program.trie_nodes,
    );
    let header = json!({
        "process": program.process,
        "colour": program.colour,
        "lanes": L,
        "steps": steps,
        "edges": edges,
        "links": links,
        "families": program.families,
        "skeletons": program.skeletons,
        "trie_nodes": program.trie_nodes,
        "reference_work": program.reference_work,
    });
    let mut record = |kind: &str, policy: &str, timing: &Timing, details: Value| {
        let seconds = timing.build + timing.walk;
        let mut value = json!({
            "kind": kind,
            "policy": policy,
            "build_seconds": timing.build,
            "walk_seconds": timing.walk,
            "seconds_per_batch": seconds,
            "seconds_per_point": seconds / L as f64,
            "reference_best_gcmac_per_s": (reference_best * L as u64) as f64 / seconds / 1e9,
            "details": details,
        });
        let object = value.as_object_mut().unwrap();
        for (key, entry) in header.as_object().unwrap() {
            object.insert(key.clone(), entry.clone());
        }
        records.push(value);
    };

    let deviation = check_program::<L>(program, &mut data);
    println!("  every kernel form gives the same numerators: max deviation {deviation:.1e}");
    assert!(
        deviation < 1e-12,
        "{} {}: kernel forms disagree by {deviation:e}",
        program.process,
        program.colour
    );

    // Walk without kernels: traversal, segment set-up, output and accumulator
    // clears, factor bookkeeping (the executor's per-step dispatch).
    let walk = vec![None; steps];
    let timing = time_program::<L>(program, &mut data, &walk, reps, false);
    println!(
        "  walk without kernels: {:8.3} ms/batch ({:.1} ns/step, {:.1} ns/edge)",
        timing.walk * 1e3,
        timing.walk / steps as f64 * 1e9,
        timing.walk / edges as f64 * 1e9
    );
    record("program-walk", "walk", &timing, json!({}));

    let mut own: Vec<(Form, Vec<f64>)> = Vec::new();
    for form in Form::ALL {
        let policy = vec![Some(form); steps];
        let timing = time_program::<L>(program, &mut data, &policy, reps, false);
        let per_step = time_program::<L>(program, &mut data, &policy, reps.min(2), true);
        let work = policy_json(program, &policy);
        let build = program.build_work(&used_factors(program, &policy, false))
            + program.build_work(&used_factors(program, &policy, true));
        println!(
            "  {:>16}: {:9.3} ms/batch = {:8.4} ms/pt (build {:7.3} ms); {:5.2} G kernel-CMAC/s, {:5.2} G reference-best-CMAC/s",
            form.name(),
            (timing.build + timing.walk) * 1e3,
            (timing.build + timing.walk) / L as f64 * 1e3,
            timing.build * 1e3,
            (work["work_cmac_per_lane"].as_u64().unwrap() * L as u64) as f64 / timing.walk / 1e9,
            (reference_best * L as u64) as f64 / (timing.build + timing.walk) / 1e9,
        );
        let model = StepModel::fit(program, form, &per_step.own, L);
        record(
            "program",
            form.name(),
            &timing,
            json!({
                "work": work,
                "build_work_cmac_per_lane": build,
                "own_seconds_sum": per_step.own.iter().sum::<f64>(),
                "model": model.json(),
            }),
        );
        own.push((form, per_step.own));
    }
    // Per-step shapes, costs and own times of every form (for the cost model).
    let costs = |form: Form| -> Vec<[u64; 2]> {
        program
            .steps
            .iter()
            .map(|step| {
                let cost = program.step_cost(step, form);
                [cost.work, cost.iterations]
            })
            .collect()
    };
    let steps_record = json!({
        "kind": "program-steps",
        "process": program.process,
        "colour": program.colour,
        "lanes": L,
        "fields": ["seeded", "rows", "edges", "links", "rank_in", "d_alpha", "d_out",
                   "reference_fused", "reference_split"],
        "steps": program.steps.iter().map(|step| json!([
            u8::from(step.seeded), step.rows, step.edges.len(), step.links(), step.rank_in,
            step.d_alpha, step.d_out, step.reference.0, step.reference.1,
        ])).collect::<Vec<_>>(),
        "forms": own.iter().map(|(form, times)| (form.name().to_string(), json!({
            "cost": costs(*form),
            "own_seconds": times,
        }))).collect::<serde_json::Map<_, _>>(),
    });

    // Oracle: the fastest measured form per step (own time; fused forms pay
    // for their factors at the program level, outside the step).
    let oracle: Vec<Option<Form>> = (0..steps)
        .map(|s| {
            own.iter()
                .min_by(|a, b| a.1[s].total_cmp(&b.1[s]))
                .map(|(form, _)| *form)
        })
        .collect();
    // Rule: the cheapest structured or blocked form by the fitted per-form
    // model of this program, lane count and code generation.
    let fits: Vec<(Form, StepModel)> = [
        Form::FusedStructured,
        Form::SplitStructured,
        Form::FusedBlocked,
    ]
    .into_iter()
    .map(|form| {
        let times = &own.iter().find(|(f, _)| *f == form).unwrap().1;
        (form, StepModel::fit(program, form, times, L))
    })
    .collect();
    let rule: Vec<Option<Form>> = program
        .steps
        .iter()
        .map(|step| {
            fits.iter()
                .map(|(form, model)| (*form, model.seconds(program.step_cost(step, *form), L)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(form, _)| form)
        })
        .collect();
    // Count rule: the form with less reference-counted work (split on ties).
    let counted: Vec<Option<Form>> = program
        .steps
        .iter()
        .map(|step| {
            Some(if step.reference.1 <= step.reference.0 {
                Form::SplitStructured
            } else {
                Form::FusedStructured
            })
        })
        .collect();
    for (name, policy) in [("oracle", &oracle), ("rule", &rule), ("counted", &counted)] {
        let timing = time_program::<L>(program, &mut data, policy, reps, false);
        let agree = policy.iter().zip(&oracle).filter(|(a, b)| a == b).count();
        println!(
            "  {:>16}: {:9.3} ms/batch = {:8.4} ms/pt (build {:7.3} ms); {:5.2} G reference-best-CMAC/s; agrees with oracle on {agree}/{steps} steps",
            name,
            (timing.build + timing.walk) * 1e3,
            (timing.build + timing.walk) / L as f64 * 1e3,
            timing.build * 1e3,
            (reference_best * L as u64) as f64 / (timing.build + timing.walk) / 1e9,
        );
        let mut details = policy_json(program, policy);
        details["oracle_agreement"] = json!(agree as f64 / steps as f64);
        if name == "rule" {
            details["fits"] = json!(
                fits.iter()
                    .map(|(form, model)| {
                        let mut value = model.json();
                        value["form"] = json!(form.name());
                        value
                    })
                    .collect::<Vec<_>>()
            );
        }
        record("program", name, &timing, details);
    }
    records.push(steps_record);
}

fn run_all<const L: usize>(
    synthetic: bool,
    target: f64,
    programs: &[Program],
    reps: usize,
    records: &mut Vec<Value>,
) {
    if synthetic {
        run_lanes::<L>(target, records);
    }
    for program in programs {
        run_program::<L>(program, reps, records);
    }
}

fn main() {
    let mut json_path = None;
    let mut lanes = vec![8, 16, 32];
    let mut target = 0.15;
    let mut programs = Vec::new();
    let mut synthetic = true;
    let mut reps = 3;
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
            "--program" => {
                programs.push(Program::load(&args.next().expect("--program needs a file")))
            }
            "--programs-only" => synthetic = false,
            "--program-reps" => {
                reps = args
                    .next()
                    .expect("--program-reps needs a value")
                    .parse()
                    .expect("repetitions")
            }
            _ => {} // cargo bench passes --bench
        }
    }
    let mut records = Vec::new();
    for l in lanes {
        match l {
            8 => run_all::<8>(synthetic, target, &programs, reps, &mut records),
            16 => run_all::<16>(synthetic, target, &programs, reps, &mut records),
            32 => run_all::<32>(synthetic, target, &programs, reps, &mut records),
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
