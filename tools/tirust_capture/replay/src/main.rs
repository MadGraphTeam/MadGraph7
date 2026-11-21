//! Replay a MadLoop -> TIRust bridge call trace (recorded by ../shim.c with
//! TIRUST_TRACE_INPUTS=1) against a bridge build linked as a library, and compare
//! every DP tensor with the recorded one.
//!
//! Adapted from the scratch replay written on 2026-11-13 by the Claude session
//! "Speed up TIRust in MadLoop and its initialization"; this version adds
//! per-call JSON lines and a JSON summary for the double-double gate study.
//!
//! usage: replay <trace> [--max-segments N] [--skip-compare] [--repeat K]
//!               [--calls-jsonl FILE] [--report-json FILE]
//!
//! A segment is the run of calls between two cache clears (one MadLoop
//! phase-space point). Per-call lines record the route counter deltas of the
//! replaying bridge ([Direct, MethodChoice, Tier106, Tier256, Tier512, Refused,
//! unreported, attempted106, attempted256, attempted512] of the call's block), so
//! a production replay labels which recorded calls went through which tier.
use madloop_tirust::{
    C64, Context, ml_tirust_cache_statistics, ml_tirust_clear_cache, ml_tirust_context_create,
    ml_tirust_context_destroy, ml_tirust_dp_route_statistics, ml_tirust_evaluate_dp, ml_tirust_register_plan,
};
use std::{collections::BTreeMap, io::Write, time::Instant};

enum Rec {
    Dp { n: i32, rank: i32, poles: i32, finite: i32, check: i32, cache: i32, st: i32, p: Vec<f64>, s: Vec<f64>, m: Vec<C64>, mu2: f64, out: Vec<C64> },
    Mp,
    Register { n: i32, rank: i32 },
    Clear,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}
impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> &[u8] {
        let b = &self.data[self.at..self.at + n];
        self.at += n;
        b
    }
    fn u8(&mut self) -> u8 {
        self.bytes(1)[0]
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
    fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.f64()).collect()
    }
    fn c64s(&mut self, n: usize) -> Vec<C64> {
        (0..n).map(|_| C64 { re: self.f64(), im: self.f64() }).collect()
    }
}

fn read(path: &str) -> Vec<Rec> {
    let data = std::fs::read(path).expect("trace");
    let mut r = Reader { data: &data, at: 0 };
    let mut out = Vec::new();
    while r.at < data.len() {
        match r.u8() {
            k @ (1 | 2 | 5) => {
                let (n, rank, poles, finite, check, cache, st) = (r.i32(), r.i32(), r.i32(), r.i32(), r.i32(), r.i32(), r.i32());
                let _dt = r.f64();
                if k == 2 {
                    out.push(Rec::Mp);
                    continue;
                }
                if k == 5 {
                    let lens: Vec<usize> = (0..4).map(|_| r.u64() as usize).collect();
                    for l in lens {
                        r.bytes(l);
                    }
                    let ol = r.u64() as usize;
                    r.bytes(ol);
                    out.push(Rec::Mp);
                    continue;
                }
                assert_eq!(r.u8(), 1, "trace recorded without inputs (set TIRUST_TRACE_INPUTS=1)");
                let nu = n as usize;
                let p = r.f64s(4 * nu);
                let s = r.f64s(nu * nu);
                let m = r.c64s(nu);
                let mu2 = r.f64();
                let ol = r.u64() as usize;
                let out_values = r.c64s(ol);
                out.push(Rec::Dp { n, rank, poles, finite, check, cache, st, p, s, m, mu2, out: out_values });
            }
            3 => {
                let (n, rank, _st) = (r.i32(), r.i32(), r.i32());
                let _dt = r.f64();
                out.push(Rec::Register { n, rank });
            }
            4 => {
                let _dt = r.f64();
                out.push(Rec::Clear);
            }
            k => panic!("bad record kind {k}"),
        }
    }
    out
}

fn coefficient_count(rank: i32) -> usize {
    let r = rank as usize;
    (r + 1) * (r + 2) * (r + 3) * (r + 4) / 24
}

/// Tensor degree of coefficient `m` in IREGI ordering (blocks of increasing degree).
fn degree(m: usize) -> usize {
    let (mut d, mut start) = (0usize, 0usize);
    loop {
        let size = (d + 1) * (d + 2) * (d + 3) / 6;
        if m < start + size {
            return d;
        }
        start += size;
        d += 1;
    }
}

fn json_f64(x: f64) -> String {
    if x.is_finite() { format!("{x:e}") } else { "null".to_string() }
}

#[derive(Default)]
struct Cmp {
    compared: u64,
    identical: u64,
    status_mismatch: u64,
    max_rel: f64,
    max_comp_rel: f64,
    worst: String,
}

const NAMES: [&str; 7] = ["Direct", "MethodChoice", "Tier106", "Tier256", "Tier512", "Refused", "unreported"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let mut max_segments = usize::MAX;
    let mut compare = true;
    let mut repeat = 1;
    let mut calls_path: Option<String> = None;
    let mut report_path: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--max-segments" => {
                max_segments = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--skip-compare" => compare = false,
            "--repeat" => {
                repeat = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--calls-jsonl" => {
                calls_path = Some(args[i + 1].clone());
                i += 1;
            }
            "--report-json" => {
                report_path = Some(args[i + 1].clone());
                i += 1;
            }
            a => panic!("unknown argument {a}"),
        }
        i += 1;
    }
    let t_read = Instant::now();
    let recs = read(path);
    eprintln!("read {} records in {:.1} s", recs.len(), t_read.elapsed().as_secs_f64());

    let mut calls = calls_path.as_ref().map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("calls-jsonl")));
    let keep = std::env::var("REPLAY_KEEP_CONTEXT").is_ok();
    let mut ctx: *mut Context = std::ptr::null_mut();
    let mut report = String::new();
    for rep in 0..repeat {
        if rep == 0 || !keep {
            assert_eq!(unsafe { ml_tirust_context_create(1, &mut ctx) }, 0);
        }
        let mut seg_times: Vec<(f64, u64, u64)> = vec![(0.0, 0, 0)];
        let mut by_route: BTreeMap<(i32, i32, i32, &str), (f64, u64, u64, f64)> = BTreeMap::new();
        let mut prev_routes = [0u64; 20];
        let mut cmp = Cmp::default();
        let mut out = vec![C64::default(); 4 * coefficient_count(6)];
        let (mut seg_idx, mut seen_in_seg, mut call_idx) = (0usize, false, 0u64);
        let t_all = Instant::now();
        for rec in &recs {
            if seg_times.len() > max_segments {
                break;
            }
            match rec {
                Rec::Register { n, rank } => {
                    assert_eq!(unsafe { ml_tirust_register_plan(ctx, *n, *rank) }, 0);
                }
                Rec::Clear => {
                    unsafe { ml_tirust_clear_cache(ctx) };
                    if seen_in_seg {
                        seg_idx += 1;
                        seen_in_seg = false;
                    }
                    if seg_times.last().unwrap().1 > 0 {
                        seg_times.push((0.0, 0, 0));
                    }
                }
                Rec::Mp => {}
                Rec::Dp { n, rank, poles, finite, check, cache, st, p, s, m, mu2, out: recorded } => {
                    seen_in_seg = true;
                    let nc = coefficient_count(*rank);
                    let olen = 4 * nc;
                    let t0 = Instant::now();
                    let status = unsafe {
                        ml_tirust_evaluate_dp(
                            ctx, *n, *rank, p.as_ptr(), p.len(), s.as_ptr(), s.len(), m.as_ptr(), m.len(), *mu2,
                            out.as_mut_ptr(), olen, *poles, *finite, *check, *cache,
                        )
                    };
                    let dt = t0.elapsed().as_secs_f64();
                    let mut routes = [0u64; 20];
                    unsafe { ml_tirust_dp_route_statistics(ctx, routes.as_mut_ptr(), 20) };
                    let off = if *check != 0 { 10 } else { 0 };
                    let delta: Vec<u64> = (0..10).map(|k| routes[off + k] - prev_routes[off + k]).collect();
                    // The highest-precision successful route decides the label.
                    let route = [4usize, 3, 2, 1, 0, 5, 6].into_iter().find(|&k| delta[k] > 0).map_or("cachehit", |k| NAMES[k]);
                    prev_routes = routes;
                    let seg = seg_times.last_mut().unwrap();
                    seg.0 += dt;
                    seg.1 += 1;
                    if status != 0 {
                        seg.2 += 1;
                    }
                    let (mut identical, mut call_rel, mut call_comp) = (true, 0.0f64, 0.0f64);
                    if compare {
                        cmp.compared += 1;
                        if status != *st {
                            cmp.status_mismatch += 1;
                            identical = false;
                            call_rel = f64::INFINITY;
                            call_comp = f64::INFINITY;
                        } else if status == 0 {
                            let got = &out[..olen];
                            if got.iter().zip(recorded).all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()) {
                                cmp.identical += 1;
                            } else {
                                identical = false;
                                // Tensor scale: largest component magnitude. Degree scale: largest |value| of the
                                // same tensor degree over all four Laurent slots (one mass dimension). A
                                // vanishing slot (e.g. the double pole of a massive integral) is judged
                                // against the finite part of that degree, as MadLoop judges small poles.
                                let scale = recorded.iter().map(|z| z.re.abs().max(z.im.abs())).fold(0.0, f64::max).max(1e-300);
                                let mut block = std::collections::HashMap::new();
                                for (j, b) in recorded.iter().enumerate() {
                                    let e = block.entry(degree(j % nc)).or_insert(0.0f64);
                                    *e = e.max((b.re * b.re + b.im * b.im).sqrt());
                                }
                                for (j, (a, b)) in got.iter().zip(recorded).enumerate() {
                                    let d = (a.re - b.re).abs().max((a.im - b.im).abs()) / scale;
                                    let d = if d.is_nan() { f64::INFINITY } else { d };
                                    call_rel = call_rel.max(d);
                                    let mag = block[&degree(j % nc)];
                                    if mag > 0.0 {
                                        let dc = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt() / mag;
                                        let dc = if dc.is_nan() { f64::INFINITY } else { dc };
                                        if dc > call_comp {
                                            call_comp = dc;
                                        }
                                        if dc > cmp.max_comp_rel {
                                            cmp.max_comp_rel = dc;
                                            cmp.worst = format!("call={call_idx} seg={seg_idx} n={n} rank={rank} check={check} slot={} comp={} got={:?} rec={:?}", j / nc, j % nc, (a.re, a.im), (b.re, b.im));
                                        }
                                    }
                                }
                                cmp.max_rel = cmp.max_rel.max(call_rel);
                            }
                        } else {
                            cmp.identical += 1;
                        }
                    }
                    let e = by_route.entry((*n, *rank, *check, route)).or_default();
                    e.0 += dt;
                    e.1 += 1;
                    e.2 += !identical as u64;
                    e.3 = e.3.max(call_comp);
                    if let Some(w) = calls.as_mut().filter(|_| rep == 0) {
                        writeln!(
                            w,
                            "{{\"i\":{call_idx},\"seg\":{seg_idx},\"n\":{n},\"rank\":{rank},\"check\":{check},\"poles\":{poles},\"finite\":{finite},\"cache\":{cache},\"rec_status\":{st},\"status\":{status},\"route\":\"{route}\",\"delta\":{delta:?},\"identical\":{identical},\"max_rel\":{},\"max_comp_rel\":{}}}",
                            json_f64(call_rel),
                            json_f64(call_comp)
                        )
                        .unwrap();
                    }
                    call_idx += 1;
                }
            }
        }
        let total = t_all.elapsed().as_secs_f64();
        let mut stats = [0u64; 9];
        unsafe { ml_tirust_cache_statistics(ctx, stats.as_mut_ptr(), 9) };
        let mut routes = [0u64; 20];
        unsafe { ml_tirust_dp_route_statistics(ctx, routes.as_mut_ptr(), 20) };
        if !keep || rep + 1 == repeat {
            unsafe { ml_tirust_context_destroy(ctx) };
        }
        let segs: Vec<_> = seg_times.iter().filter(|s| s.1 > 0).collect();
        println!("== repeat {rep}: total {total:.3} s over {} segments, {} DP calls", segs.len(), call_idx);
        println!("per (n, rank, check, route): total s, calls, non-identical, max componentwise rel diff");
        for (k, v) in &by_route {
            println!("  {k:?}: {:9.3} {:8} {:8} {:.3e}", v.0, v.1, v.2, v.3);
        }
        println!("dp routes primary [Direct, MethodChoice, T106, T256, T512, Refused, err, att106, att256, att512]: {:?}", &routes[..10]);
        println!("dp routes check: {:?}", &routes[10..]);
        if compare {
            println!(
                "compare: {} calls, {} bit-identical, {} status mismatches, max rel diff of the others {:.3e} (per tensor scale), {:.3e} (componentwise)",
                cmp.compared, cmp.identical, cmp.status_mismatch, cmp.max_rel, cmp.max_comp_rel
            );
            if !cmp.worst.is_empty() {
                println!("worst component: {}", cmp.worst);
            }
        }
        if rep == 0 {
            let groups: Vec<String> = by_route
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{{\"n\":{},\"rank\":{},\"check\":{},\"route\":\"{}\",\"seconds\":{},\"calls\":{},\"non_identical\":{},\"max_comp_rel\":{}}}",
                        k.0, k.1, k.2, k.3, json_f64(v.0), v.1, v.2, json_f64(v.3)
                    )
                })
                .collect();
            report = format!(
                "{{\"schema\":\"tirust-replay-v1\",\"trace\":{:?},\"segments\":{},\"dp_calls\":{},\"seconds\":{},\"compared\":{},\"identical\":{},\"status_mismatch\":{},\"max_rel\":{},\"max_comp_rel\":{},\"worst\":{:?},\"routes_primary\":{:?},\"routes_check\":{:?},\"cache_statistics\":{:?},\"groups\":[{}]}}\n",
                path,
                segs.len(),
                call_idx,
                json_f64(total),
                cmp.compared,
                cmp.identical,
                cmp.status_mismatch,
                json_f64(cmp.max_rel),
                json_f64(cmp.max_comp_rel),
                cmp.worst,
                &routes[..10],
                &routes[10..],
                stats,
                groups.join(",")
            );
        }
    }
    if let Some(mut w) = calls {
        w.flush().unwrap();
    }
    if let Some(p) = report_path {
        std::fs::write(p, report).expect("report-json");
    }
}
