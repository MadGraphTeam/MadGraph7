// SPDX-License-Identifier: 0BSD

//! The native loop-request DAG build of one input, timed per stage
//! (M-FINAL mf-8g-1).
//!
//! The input is `dag_input.dag_input`'s JSON, as `V2Dag.build` receives it
//! (the counterterm program's: `dag_input.native_counterterm_blocks`). The
//! bench times the stages of `loop_v2::dag::build` one by one (recursion,
//! prune, N_c grade split, connection signs, reuse, colour-flow signs,
//! columns), with each stage's memory under `PYAMPLICOL_LOOP_STAGE_MEMORY`,
//! and prints the evaluation group counts and the digest of the columns,
//! which equals `V2Dag.digest()` of the same input (equal digests: equal
//! DAGs).
//!
//! ```text
//! cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_dag -- \
//!     --input dag-input.json [--reps 1]
//! ```

use std::time::Instant;

use rusticol_core::loop_v2::dag::input::DagInput;
use rusticol_core::loop_v2::dag::{grades, recursion, reuse, signs, stage_memory};

fn main() {
    let mut path = None;
    let mut reps = 1usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => path = args.next(),
            "--reps" => reps = args.next().and_then(|r| r.parse().ok()).unwrap_or(1),
            "--bench" => {}
            other => panic!("unknown argument {other}"),
        }
    }
    let text = std::fs::read_to_string(path.expect("--input is required")).expect("input");
    let input = DagInput::from_json(&text).expect("DAG input");
    for _ in 0..reps {
        let mut stages: Vec<(&str, f64, usize)> = Vec::new();
        let total = Instant::now();
        let mut clock = Instant::now();
        // With `PYAMPLICOL_LOOP_STAGE_MEMORY` set, each stage's resident and
        // peak memory too (`dag::stage_memory`, on stderr).
        let mut lap = |stage: &'static str, count: usize, stages: &mut Vec<(&str, f64, usize)>| {
            stages.push((stage, clock.elapsed().as_secs_f64(), count));
            stage_memory(stage, count);
            clock = Instant::now();
        };
        let mut dag = recursion::compile(&input).expect("compile");
        lap("compile", dag.interactions.len(), &mut stages);
        let groups = |dag: &recursion::Dag| {
            dag.interactions
                .iter()
                .map(|i| i.group)
                .max()
                .map_or(0, |g| g + 1)
        };
        let compile_groups = groups(&dag);
        if dag.finish {
            recursion::prune_to_admitted(&input, &mut dag);
            lap("prune", dag.interactions.len(), &mut stages);
            if input.u1_grades {
                grades::split_u1_grades(&input, &mut dag).expect("grades");
                lap("grades", dag.interactions.len(), &mut stages);
            }
            signs::apply_connection_signs(&input, &mut dag).expect("connection signs");
            lap("connection signs", dag.interactions.len(), &mut stages);
            reuse::assign_reuse(&input, &mut dag).expect("reuse");
            lap("reuse", dag.interactions.len(), &mut stages);
            println!("groups: compile {compile_groups}, reuse {}", groups(&dag));
            signs::apply_colour_flow_signs(&input, &mut dag).expect("colour-flow signs");
            lap("colour-flow signs", dag.interactions.len(), &mut stages);
        }
        let currents = dag.currents.len();
        let columns = dag.into_columns(&input).expect("columns");
        lap("columns", columns.interaction_count(), &mut stages);
        let seconds = total.elapsed().as_secs_f64();
        for (stage, time, count) in &stages {
            println!("{stage:>18}: {time:8.3} s  ({count} interactions)");
        }
        println!(
            "{:>18}: {seconds:8.3} s  ({currents} currents, {} interactions)",
            "total",
            columns.interaction_count()
        );
        println!("digest {}", columns.digest());
    }
}
