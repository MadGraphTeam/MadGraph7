// SPDX-License-Identifier: 0BSD

//! Schedule B against schedule M on exported plans (WP2.8; data for gate G1).
//!
//! Each `--plan` is a plan v4 with a schedule-M section, as written by
//! `tools/loop_v2_schedules.py export`. The bench fills one batch of
//! synthetic inputs (random block rows, Born components and kernel constants
//! in binary64, random exact momenta) and runs the native evaluation of both
//! schedules with a reduction backend that returns at once: the reduction is
//! the same NinjaRS work for both schedules, so the times are dressing,
//! fused factors and split inputs, seeds and the scatter into reduction
//! lanes. Per schedule it reports the best of `--reps` runs per point and
//! stage, the reals per lane of every live store, and the modelled kernel
//! work (complex multiply-adds per lane) of the forms the plan chose, the
//! unit of the design's cost model.
//!
//! ```text
//! cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_schedules -- \
//!     --plan gg_gggg-FC.json [--plan ...] [--points 16] [--lanes 16] [--reps 3] \
//!     [--threads 1] [--part-chunks 64] [--json results.json]
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rusticol_core::RusticolResult;
use rusticol_core::loop_execution::v2::{
    FamilyLanes, LaneOutcome, Poles, Reduced, Schedule, V2Backend, V2Executor, V2Options,
};
use rusticol_core::loop_v2::dress::V2Batch;
use rusticol_core::loop_v2::plan::{LoopProgramV4, Step};
use rusticol_core::loop_v2::retest::ReductionCheck;
use serde_json::{Value, json};
use symbolica::domains::float::Complex;
use symbolica::domains::rational::Rational;

/// A reduction that succeeds at once with zero integrals.
struct Instant0;

impl V2Backend<f64> for Instant0 {
    fn begin(&mut self) {}

    fn reduce(
        &mut self,
        groups: &[FamilyLanes<f64>],
        poles: Poles,
        _check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<f64>> {
        let width = poles.width();
        let direct = groups
            .iter()
            .map(|group| {
                group
                    .keys
                    .iter()
                    .map(|_| -> LaneOutcome<f64> { Ok(vec![Complex::new(0.0, 0.0); width]) })
                    .collect()
            })
            .collect();
        Ok((direct, None))
    }
}

/// xorshift64*: reproducible inputs without a dependency.
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn complex(&mut self) -> Complex<f64> {
        Complex::new(self.next(), self.next())
    }

    fn rational(&mut self) -> Rational {
        Rational::from(((self.next() * 1000.0).round() as i64, 7))
    }
}

fn batch(plan: &LoopProgramV4, points: usize) -> V2Batch<f64> {
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let block: usize = plan.block_rows.iter().sum();
    V2Batch {
        momenta: (0..points)
            .map(|_| (0..4 * plan.legs).map(|_| random.rational()).collect())
            .collect(),
        scales: vec![Rational::from(1); points],
        masses: (0..plan.masses).map(|_| Rational::from(0)).collect(),
        masses_imaginary: Vec::new(),
        leg_masses: Vec::new(),
        constants: (0..plan.constants).map(|_| random.complex()).collect(),
        block_rows: (0..points)
            .map(|_| (0..block).map(|_| random.complex()).collect())
            .collect(),
        born: (0..points)
            .map(|_| {
                (0..plan.rows * plan.born_components)
                    .map(|_| random.complex())
                    .collect()
            })
            .collect(),
    }
}

/// Modelled kernel work of the chosen forms (complex MAC per lane): all
/// steps, and the seeded ones (the seed injection).
fn work<'a>(steps: impl Iterator<Item = &'a Step>) -> (u64, u64) {
    steps.fold((0, 0), |(all, seeded), step| {
        let work = match (step.split.is_some(), step.costs) {
            (true, (_, Some(split))) => split.work,
            (_, (fused, _)) => fused.work,
        };
        (all + work, seeded + if step.seeded { work } else { 0 })
    })
}

struct Args {
    plans: Vec<String>,
    points: usize,
    lanes: usize,
    reps: usize,
    threads: usize,
    part_chunks: Option<usize>,
    json: Option<String>,
}

fn args() -> Args {
    let mut out = Args {
        plans: Vec::new(),
        points: 16,
        lanes: 16,
        reps: 3,
        threads: 1,
        part_chunks: None,
        json: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = || iter.next().expect("a value after the flag");
        match arg.as_str() {
            "--plan" => out.plans.push(value()),
            "--points" => out.points = value().parse().expect("points"),
            "--lanes" => out.lanes = value().parse().expect("lanes"),
            "--reps" => out.reps = value().parse().expect("reps"),
            "--threads" => out.threads = value().parse().expect("threads"),
            "--part-chunks" => out.part_chunks = Some(value().parse().expect("part chunks")),
            "--json" => out.json = Some(value()),
            "--bench" => {}
            other => panic!("unknown argument {other}"),
        }
    }
    out
}

fn main() {
    let args = args();
    let mut results = Vec::new();
    for path in &args.plans {
        let text = std::fs::read(path).expect("plan file");
        let start = Instant::now();
        let plan = Arc::new(LoopProgramV4::load(&text).expect("plan v4"));
        let load = start.elapsed().as_secs_f64();
        let merged = plan.merged.as_ref().expect("a plan with schedule M");
        let in_m = |f: usize| merged.families.binary_search(&f).is_ok();
        let b_work = work(plan.families.iter().flat_map(|f| &f.steps));
        let b_rest = work(
            plan.families
                .iter()
                .enumerate()
                .filter(|(f, _)| !in_m(*f))
                .flat_map(|(_, f)| &f.steps),
        );
        let m_own = work(merged.states.iter().map(|s| &s.step));
        let m_work = (m_own.0 + b_rest.0, m_own.1 + b_rest.1);
        let b_steps: usize = plan.families.iter().map(|f| f.steps.len()).sum();
        let batch = batch(&plan, args.points);
        let executor = V2Executor::new(plan.clone(), args.lanes, Some(args.threads)).unwrap();
        let mut record = json!({
            "plan": path,
            "families": plan.families.len(),
            "merged_families": merged.families.len(),
            "b_steps": b_steps,
            "m_states": merged.states.len(),
            "m_chunks": merged.chunks.len(),
            "m_entries": merged.entries.len(),
            "load_seconds": load,
            "points": args.points,
            "lanes": args.lanes,
            "threads": args.threads,
            "work_per_lane": {"B": b_work.0, "M": m_work.0},
            "seeded_work_per_lane": {"B": b_work.1, "M": m_work.1},
        });
        for (name, schedule) in [("B", Schedule::B), ("M", Schedule::M)] {
            let mut options = V2Options {
                poles: Poles::None,
                schedule,
                ..V2Options::default()
            };
            if let Some(chunks) = args.part_chunks {
                options.part_chunks = chunks;
            }
            let mut best: Option<(f64, BTreeMap<&'static str, f64>)> = None;
            let mut report = None;
            for _ in 0..args.reps {
                let start = Instant::now();
                let (_results, run) = executor
                    .evaluate(&batch, &0.0, &mut Instant0, options)
                    .expect("evaluation");
                let seconds = start.elapsed().as_secs_f64() / args.points as f64;
                if best.as_ref().is_none_or(|(b, _)| seconds < *b) {
                    let stages = run
                        .seconds
                        .iter()
                        .map(|(k, v)| (*k, v / args.points as f64))
                        .collect();
                    best = Some((seconds, stages));
                }
                report = Some(run);
            }
            let (seconds, stages) = best.expect("at least one repetition");
            let report = report.expect("a report");
            record[name] = json!({
                "seconds_per_point": seconds,
                "stages_per_point": stages,
                "reals_per_lane": report.reals_per_lane,
                "counts": report.counts,
                "real_bytes": report.real_bytes,
            });
            println!(
                "{path} {name}: {:.3} ms/pt (dressing {:.3}, factors {:.3}), work {:.3e} CMAC/lane",
                seconds * 1e3,
                record[name]["stages_per_point"]["dressing"]
                    .as_f64()
                    .unwrap_or(0.0)
                    * 1e3,
                record[name]["stages_per_point"]["factors"]
                    .as_f64()
                    .unwrap_or(0.0)
                    * 1e3,
                record["work_per_lane"][name].as_u64().unwrap_or(0) as f64,
            );
        }
        results.push(record);
    }
    if let Some(path) = args.json {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&Value::Array(results)).unwrap(),
        )
        .expect("write the results");
    }
}
