//! Additional ignored test coverage and a separately authorized request timer.
//! This module is a child of the unchanged original native-study harness.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "explicit MP handoff; returned tensor-shape errors retain policy/entries"]
fn returned_shape_error_retains_entries_and_late_policy() {
    large_thread(|| {
        let mut output = stream();
        let mut failed = 0usize;
        for bits in [256, 384] {
            for policy in StudyKernelPolicy::ALL {
                let mut state = MpState::with_study_policy(bits, policy).unwrap();
                let inputs = [fixture(1, "2", &["4", "-0.125"], "9"),
                              fixture(2, "-3", &["0", "-0", "4", "-0"], "16")];
                let first = observe(&mut state, &request(&inputs[0]));
                assert!(successful(&first));
                for (index, input) in inputs.iter().enumerate() {
                    let key = (input["n"].as_u64().unwrap() as usize, 0);
                    let before = state.entries.get(&key).map(|entry| entry as *const MpEntry as usize);
                    assert_eq!(before.is_some(), index == 0);
                    let old_count = state.entries.len();
                    let mut wrong = request(input);
                    // Two complete complex coefficients for a rank-zero tensor:
                    // pairs() succeeds and set_coefficients returns the error.
                    wrong.coefficients.extend(pack(&json!(["1", "-0"])));
                    let rejected = observe(&mut state, &wrong);
                    let after_error = state.entries.get(&key).map(|entry| entry as *const MpEntry as usize);
                    let retained_after_error = after_error.is_some()
                        && before.is_none_or(|address| Some(address) == after_error)
                        && state.entries.len() == old_count + usize::from(before.is_none())
                        && state.study_policy == policy && state.precision == bits;
                    let recovered = observe(&mut state, &request(input));
                    let after_recovery = state.entries.get(&key).map(|entry| entry as *const MpEntry as usize);
                    let fresh = observe(&mut MpState::with_study_policy(bits, policy).unwrap(), &request(input));
                    let good = rejected["status"] == ERROR && rejected["panic"].is_null()
                        && rejected["error"].is_string() && rejected["output"] == json!([])
                        && rejected["raw"].is_null() && rejected["input_unchanged"] == true
                        && retained_after_error && after_recovery == after_error
                        && recovered == fresh && successful(&recovered);
                    failed += usize::from(!good);
                    emit(&mut output, &json!({"kind": "returned_shape_error",
                        "policy": format!("{policy:?}"), "working_bits": bits, "input": input,
                        "late_entry": before.is_none(), "entry_retained": retained_after_error,
                        "same_entry_after_recovery": after_recovery == after_error,
                        "rejected": rejected, "recovered": recovered, "fresh": fresh, "passed": good}));
                }
            }
        }
        emit(&mut output, &json!({"kind": "summary", "complete": true,
            "controls": 12, "failed_controls": failed, "performance_claim": false}));
        assert_eq!(failed, 0);
    });
}

fn timed_evaluate(state: &mut MpState, request: &MpRequest) -> std::result::Result<(i32, Vec<u8>), String> {
    match catch_unwind(AssertUnwindSafe(|| state.evaluate(request))) {
        Ok(result) => result,
        Err(payload) => Err(payload.downcast_ref::<String>().cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_else(|| "non-string panic payload".to_owned())),
    }
}

#[test]
#[ignore = "explicit quiet+MP handoff; fixed7triplets, private request scope only"]
fn fixed_request_screening_seven_triplets() {
    large_thread(|| {
        let mut output = stream();
        let records = input_records();
        assert_eq!(records.len(), 1392);
        let replay = std::fs::read_to_string(std::env::var("NINJARS_NATIVE_STUDY_REPLAY").unwrap()).unwrap();
        let mut expected = Vec::with_capacity(records.len());
        for line in replay.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            if row["kind"] == "request" {
                assert_eq!(row["sequence"], expected.len());
                assert_eq!(row["input"], records[expected.len()]);
                expected.push(row["arms"].clone());
            } else { assert_eq!(row["kind"], "summary"); assert_eq!(row["complete"], true); }
        }
        assert_eq!(expected.len(), records.len());
        let requests: Vec<_> = records.iter().map(request).collect();
        let input_copies: Vec<_> = requests.iter().map(input_bytes).collect();
        let mut states = StudyKernelPolicy::ALL.map(|p| MpState::with_study_policy(256, p).unwrap());
        let mut failed = 0usize;
        // Complete warmup for every arm and every original request; retain its
        // results too, with serialization after the warmup interval.
        for arm in 0..3 {
            let mut results = Vec::with_capacity(requests.len());
            let start = Instant::now();
            for r in &requests { results.push(timed_evaluate(&mut states[arm], r)); }
            let elapsed = start.elapsed().as_nanos().to_string();
            for (sequence, result) in results.into_iter().enumerate() {
                let value = screening_result(result, &expected[sequence][arm]);
                failed += usize::from(value["passed"] != true);
                emit(&mut output, &json!({"kind": "warmup_output", "arm": arm,
                    "sequence": sequence, "result": value}));
            }
            emit(&mut output, &json!({"kind": "warmup", "arm": arm,
                "policy": format!("{:?}", StudyKernelPolicy::ALL[arm]), "elapsed_ns": elapsed,
                "performance_sample": false, "requests": requests.len()}));
        }
        let orders = [[0,1,2],[1,2,0],[2,0,1],[0,2,1],[2,1,0],[1,0,2],[0,1,2]];
        for (triplet, order) in orders.iter().enumerate() {
            for (position, &arm) in order.iter().enumerate() {
                let mut results = Vec::with_capacity(requests.len());
                let start = Instant::now();
                for r in &requests { results.push(timed_evaluate(&mut states[arm], r)); }
                let elapsed = start.elapsed().as_nanos().to_string();
                // All validation, JSON serialization and output destruction
                // follow the timer; every status/output remains in evidence.
                for (sequence, result) in results.into_iter().enumerate() {
                    let value = screening_result(result, &expected[sequence][arm]);
                    failed += usize::from(value["passed"] != true);
                    emit(&mut output, &json!({"kind": "sample_output", "triplet": triplet,
                        "position": position, "arm": arm, "sequence": sequence, "result": value}));
                }
                emit(&mut output, &json!({"kind": "sample", "triplet": triplet, "position": position,
                    "arm": arm, "policy": format!("{:?}", StudyKernelPolicy::ALL[arm]),
                    "elapsed_ns": elapsed, "requests": requests.len(),
                    "scope": "private MpState::evaluate only; excludes Fortran and manager IPC"}));
            }
        }
        let unchanged = requests.iter().map(input_bytes).eq(input_copies);
        emit(&mut output, &json!({"kind": "summary", "complete": true, "triplets": 7,
            "samples": 21, "requests_per_sample": requests.len(), "warmup_streams": 3,
            "input_unchanged": unchanged, "failed_outputs": failed,
            "full_MadLoop_performance_claim": false}));
        assert!(unchanged);
        assert_eq!(failed, 0, "all failed samples/outputs retained without replacements");
    });
}

fn screening_result(result: std::result::Result<(i32, Vec<u8>), String>, expected: &Value) -> Value {
    match result {
        Ok((status, output)) => {
            let fields: Vec<_> = output.chunks_exact(RECORD)
                .map(|field| std::str::from_utf8(field).unwrap().trim().to_owned()).collect();
            let passed = status == SUCCESS && json!(fields) == expected["output"]
                && expected["status"] == SUCCESS && output.len() == 8*RECORD;
            json!({"status": status, "output": fields, "output_bytes": output.len(), "passed": passed})
        }
        Err(error) => json!({"status": ERROR, "output": [], "error": error, "passed": false}),
    }
}
