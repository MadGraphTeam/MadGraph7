// SPDX-License-Identifier: 0BSD

//! The native generation stages of one process, timed (M-FINAL mf-8g-1c).
//!
//! Inputs are a generation's family input and one schedule payload, as
//! `loops/v2/generator_input.py` sends them (`NativeGeneration`'s JSON;
//! `KEEP_REPLAY_INPUTS` keeps the former). The bench reports the best of
//! `--reps` runs of the family recursion, the skeletons alone and the whole
//! retained schedule build (skeletons, view and tries), and a digest of the
//! build's tries (equal digests: equal schedules).
//!
//! ```text
//! cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_generation -- \
//!     --families families.json --schedules schedules.json [--reps 3]
//! ```

use std::time::Instant;

use rusticol_core::loop_v2::generate::arena::build_retained;
use rusticol_core::loop_v2::generate::families::{FamilyInput, generate_families};
use rusticol_core::loop_v2::generate::schedules::{ScheduleInput, build_skeletons};
use sha2::{Digest, Sha256};

fn best<T>(reps: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let mut seconds = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps {
        let start = Instant::now();
        let value = run();
        seconds = seconds.min(start.elapsed().as_secs_f64());
        out = Some(value);
    }
    (seconds, out.expect("at least one rep"))
}

fn main() {
    let mut families = None;
    let mut schedules = None;
    let mut reps = 3usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--families" => families = args.next(),
            "--schedules" => schedules = args.next(),
            "--reps" => reps = args.next().and_then(|r| r.parse().ok()).unwrap_or(3),
            "--bench" => {}
            other => panic!("unknown argument {other}"),
        }
    }
    let read = |path: Option<String>, what: &str| -> String {
        std::fs::read_to_string(path.unwrap_or_else(|| panic!("--{what} is required")))
            .expect("readable input")
    };
    let input: FamilyInput = serde_json::from_str(&read(families, "families")).expect("families");
    let payload: ScheduleInput =
        serde_json::from_str(&read(schedules, "schedules")).expect("schedules");
    let (t_families, output) = best(reps, || generate_families(&input).expect("families"));
    let (t_skeletons, skeletons) = best(reps, || {
        build_skeletons(&input, &output, &payload).expect("skeletons")
    });
    let count: usize = skeletons.2.iter().map(Vec::len).sum();
    let (t_build, build) = best(reps, || build_retained(&input, &output, &payload).expect("build"));
    let mut digest = Sha256::new();
    for schedule in build.schedules() {
        digest.update(format!(
            "{} {:?} {:?} {:?} {}\n",
            schedule.family, schedule.viewed, schedule.frame, schedule.leaves, schedule.rank
        ));
        for node in build.nodes(schedule) {
            digest.update(format!(
                "{} {} {} {:?} {:?}\n",
                node.depth, node.orders, node.rank_in, node.suffix, node.edges
            ));
        }
    }
    digest.update(format!("{:?}\n", build.labels()));
    let leaves: usize = build.schedules().iter().map(|s| s.leaves.len()).sum();
    let nodes: usize = build.schedules().iter().map(|s| s.nodes.len()).sum();
    println!(
        "families {t_families:.3} s; skeletons {t_skeletons:.3} s ({count}); \
         build {t_build:.3} s (view + tries {:.3} s; {leaves} leaves, {nodes} nodes); \
         digest {:x}",
        t_build - t_skeletons,
        digest.finalize()
    );
}
