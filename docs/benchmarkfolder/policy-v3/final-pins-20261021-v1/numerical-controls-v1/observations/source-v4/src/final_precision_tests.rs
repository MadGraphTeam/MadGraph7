//! Final-source single-backend MP controls, adapted from the retained study.
//! This module contains no backend selector and never changes production policy.
use super::*;
use serde_json::{Value, json};
use std::io::Write;

fn pack(values: &Value) -> Vec<u8> {
    values.as_array().unwrap().iter().flat_map(|value| {
        let text = value.as_str().unwrap().as_bytes();
        assert!(text.len() <= RECORD);
        let mut record = [b' '; RECORD];
        record[..text.len()].copy_from_slice(text);
        record
    }).collect()
}

fn request(input: &Value) -> MpRequest {
    let (reply, _) = mpsc::sync_channel(1);
    MpRequest {
        context_id: 0,
        n: input["n"].as_u64().unwrap() as usize,
        rank: input["rank"].as_u64().unwrap() as usize,
        coefficients: pack(&input["coefficients"]),
        s_matrix: pack(&input["s_matrix"]),
        momenta: pack(&input["momenta"]),
        masses: pack(&input["masses"]),
        scale: pack(&input["scale"]), reply,
    }
}

fn input_bytes(r: &MpRequest) -> Vec<Vec<u8>> {
    [&r.coefficients, &r.s_matrix, &r.momenta, &r.masses, &r.scale]
        .into_iter().cloned().collect()
}

fn represented(value: &Float) -> Value {
    if !value.is_finite() {
        return json!({"finite": false, "display": format!("{value}"),
            "precision": value.prec(), "negative": value.is_sign_negative()});
    }
    let rational = value.to_rational();
    json!({"finite": true, "numerator": rational.numerator().to_string(),
        "denominator": rational.denominator().to_string(),
        "precision": value.prec(), "negative": value.is_sign_negative()})
}

struct Observation {
    json: Value,
    raw: Option<Vec<Float>>,
}

fn observe(state: &mut MpState, r: &MpRequest) -> Observation {
    let before = input_bytes(r);
    let precision = state.precision;
    let result = catch_unwind(AssertUnwindSafe(|| state.evaluate(r)));
    let mut raw = None;
    let mut value = match result {
        Ok(Ok((status, output))) => {
            let fields: Vec<_> = output.chunks_exact(RECORD)
                .map(|field| std::str::from_utf8(field).unwrap().trim().to_owned()).collect();
            if status == SUCCESS {
                raw = state.entries.get(&(r.n, r.rank)).map(|entry|
                    entry.amplitude.results()[0].iter().flat_map(|z| [&z.re, &z.im])
                        .cloned().collect::<Vec<_>>());
            }
            json!({"status": status, "output": fields, "output_bytes": output.len(),
                "raw": raw.as_ref().map(|values| values.iter().map(represented).collect::<Vec<_>>())})
        }
        Ok(Err(error)) => json!({"status": ERROR, "error": error, "output": [], "raw": null}),
        Err(payload) => {
            let message = payload.downcast_ref::<String>().cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                .unwrap_or_else(|| "non-string panic payload".to_owned());
            // Retain the panic explicitly before replacing damaged private test
            // state. Returned-error controls below forbid this replacement.
            *state = MpState::with_precision(precision).unwrap();
            json!({"status": ERROR, "panic": true, "panic_message": message,
                "output": [], "raw": null})
        }
    };
    value["working_bits"] = json!(precision);
    value["input_unchanged"] = json!(input_bytes(r) == before);
    Observation { json: value, raw }
}

fn successful(value: &Observation) -> bool {
    value.json["status"] == SUCCESS && value.json["input_unchanged"] == true
        && value.json["output"].as_array().is_some_and(|fields| fields.len() == 8)
        && value.raw.as_ref().is_some_and(|raw| raw.len() == 10 && raw.iter().all(Float::is_finite))
}

fn binary128_outputs_equal(actual: &Value, expected: &Value) -> bool {
    let (Some(actual), Some(expected)) = (actual.as_array(), expected.as_array()) else { return false; };
    if actual.len() != 8 || expected.len() != 8 { return false; }
    actual.iter().zip(expected).all(|(a, b)| {
        let parsed_a = parse_records(&pack(&json!([a])), 113);
        let parsed_b = parse_records(&pack(&json!([b])), 113);
        match (parsed_a, parsed_b) {
            (Ok(a), Ok(b)) => a[0] == b[0] && (!a[0].is_fully_zero()
                || a[0].is_sign_negative() == b[0].is_sign_negative()),
            _ => false,
        }
    })
}

fn emit(stream: &mut impl Write, value: &Value) {
    serde_json::to_writer(&mut *stream, value).unwrap();
    stream.write_all(b"\n").unwrap();
    stream.flush().unwrap();
}

fn large_thread(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(128 * 1024 * 1024)
        .spawn(body).unwrap().join().unwrap();
}

fn compare_precisions(low: &Observation, high: &Observation) -> Value {
    if !successful(low) || !successful(high) {
        return json!({"comparable": false, "passed": false});
    }
    // Widen represented values exactly before comparison. Float operations
    // may retain more bits than the engine's requested working precision, so
    // never shrink either observation to the nominal384-bit reference size.
    // The raw records retain their original precision/sign metadata; inputs
    // to the evaluator remain the original recovered113-bit numbers.
    let comparison_bits = low.raw.as_ref().unwrap().iter()
        .chain(high.raw.as_ref().unwrap()).map(Float::prec).max().unwrap().max(384);
    let at_reference = |value: &Float| {
        let mut result = value.clone();
        result.set_prec(comparison_bits);
        result
    };
    let a: Vec<_> = low.raw.as_ref().unwrap().iter().map(at_reference).collect();
    let b: Vec<_> = high.raw.as_ref().unwrap().iter().map(at_reference).collect();
    let norm = b[..6].iter().map(|value| value.norm())
        .fold(Float::new(comparison_bits), |largest, value| if value > largest { value } else { largest });
    let bound = Float::parse("1e-31", Some(comparison_bits)).unwrap() * &norm;
    let differences: Vec<_> = a.iter().zip(&b).map(|(x,y)| (x.clone()-y).norm()).collect();
    let split_bound = Float::parse("1e-60", Some(comparison_bits)).unwrap() * &norm;
    let splits: Vec<_> = [&a, &b].iter().map(|values|
        (0..2).map(|index| (values[index].clone() - &values[6+index] - &values[8+index]).norm())
            .collect::<Vec<_>>()).collect();
    let laurent_passed = differences[..6].iter().all(|value| value <= &bound);
    let rational_passed = differences[8..].iter().all(|value| value <= &bound);
    let split_passed = splits.iter().flatten().all(|value| value <= &split_bound);
    let changed_sign: Vec<_> = low.raw.as_ref().unwrap().iter().zip(high.raw.as_ref().unwrap())
        .enumerate().filter_map(|(i,(a,b))| (a.is_sign_negative() != b.is_sign_negative()).then_some(i)).collect();
    let changed_precision: Vec<_> = low.raw.as_ref().unwrap().iter().zip(high.raw.as_ref().unwrap())
        .enumerate().filter_map(|(i,(a,b))| (a.prec() != b.prec()).then_some(i)).collect();
    json!({"comparable": true, "passed": laurent_passed && rational_passed && split_passed,
        "comparison_bits": comparison_bits,
        "laurent_passed": laurent_passed, "rational_passed": rational_passed,
        "finite_split_passed": split_passed,
        "normalization": represented(&norm), "bound": represented(&bound),
        "absolute_differences": differences.iter().map(represented).collect::<Vec<_>>(),
        "finite_split_absolute_defects": splits.iter().map(|values|
            values.iter().map(represented).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "sign_changed_indices": changed_sign, "tracked_precision_changed_indices": changed_precision,
        "automatic_sign_or_precision_difference_acceptance": false})
}

#[test]
#[ignore = "requires exclusive MP slot and original1392request capture; no backend selector"]
fn retained_requests_replay_and_converge() {
    let input = std::fs::read_to_string(std::env::var("NINJARS_CONVERGENCE_INPUT").unwrap()).unwrap();
    let output = std::env::var("NINJARS_CONVERGENCE_OUTPUT").unwrap();
    large_thread(move || {
        let records: Vec<Value> = input.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(records.len(), 1392);
        let mut stream = std::io::BufWriter::new(std::fs::OpenOptions::new().write(true)
            .create_new(true).open(output).unwrap());
        let mut low_state = MpState::with_precision(256).unwrap();
        let mut low = Vec::with_capacity(records.len());
        let mut failed = 0usize;
        let mut ascii_changed = 0usize;
        for (sequence, input) in records.iter().enumerate() {
            assert_eq!(input["status"], SUCCESS);
            let r = request(input);
            assert_eq!(r.coefficients.len(), RECORD * 2 * ninjars::fast::tensor_layout::ncoeff(r.rank));
            let observed = observe(&mut low_state, &r);
            let binary128_equal = successful(&observed)
                && binary128_outputs_equal(&observed.json["output"], &input["output"]);
            let ascii_equal = observed.json["output"] == input["output"];
            failed += usize::from(!binary128_equal);
            ascii_changed += usize::from(!ascii_equal);
            emit(&mut stream, &json!({"kind": "replay", "sequence": sequence, "input": input,
                "observed": observed.json, "historical_ascii_equal": ascii_equal,
                "historical_binary128_values_and_zero_signs_equal": binary128_equal}));
            low.push(observed);
        }
        // Identical full input history at both precisions avoids introducing
        // a selected-subset symbol-construction history into this comparison.
        let mut high_state = MpState::with_precision(384).unwrap();
        let mut convergence_failed = 0usize;
        let mut sign_changed = 0usize;
        for (sequence, input) in records.iter().enumerate() {
            let high = observe(&mut high_state, &request(input));
            let comparison = compare_precisions(&low[sequence], &high);
            convergence_failed += usize::from(comparison["passed"] != true);
            sign_changed += usize::from(comparison["sign_changed_indices"].as_array()
                .is_none_or(|fields| !fields.is_empty()));
            emit(&mut stream, &json!({"kind": "convergence", "sequence": sequence,
                "input": input, "observed_384": high.json, "comparison": comparison}));
        }
        let summary = json!({"kind": "summary", "complete": true, "records": records.len(),
            "production_bits": 256, "reference_bits": 384, "convergence_tolerance": "1e-31",
            "fields_per_precision": records.len()*8, "failed_binary128_replay_records": failed,
            "historical_ascii_changed_records_retained": ascii_changed,
            "failed_convergence_records": convergence_failed,
            "sign_changed_convergence_records_require_review": sign_changed,
            "single_production_backend_no_selector": true, "performance_claim": false});
        emit(&mut stream, &summary);
        println!("NINJARS_CONVERGENCE_CAPTURE_SUMMARY {summary}");
        assert_eq!(failed, 0, "retain every historical output/status difference");
        assert_eq!(convergence_failed, 0, "retain every convergence failure without changing thresholds");
    });
}

fn fixture(n: usize, coefficient: &str, masses: &[&str], scale: &str) -> Value {
    let (s,p) = if n == 1 { (vec!["0"],vec!["0","0","0","0"]) }
        else { (vec!["0","-4","-4","0"],vec!["0","0","0","0","0","2","0","0"]) };
    json!({"n":n,"rank":0,"coefficients":[coefficient,"-0"],"s_matrix":s,
        "momenta":p,"masses":masses,"scale":[scale]})
}

#[test]
fn analytic_poles_reuse_and_returned_shape_errors() {
    large_thread(|| {
        let fixtures = [
            fixture(1,"1.000000000000000000000000000000788860905",&["4","-0"],"1"),
            fixture(2,"1",&["0","-0","4","-0"],"1"),
            fixture(2,"2",&["4","-0","0","-0"],"9"),
            fixture(1,"-3",&["9","-0.125"],"16"),
            fixture(2,"-3",&["0","-0","0","-0"],"4"),
        ];
        for bits in [256,384] {
            let mut state = MpState::with_precision(bits).unwrap();
            for input in &fixtures {
                let r = request(input);
                let fresh = observe(&mut MpState::with_precision(bits).unwrap(), &r);
                assert!(successful(&fresh));
                let key = (r.n,r.rank);
                let before = state.entries.get(&key).map(|entry| entry as *const MpEntry as usize);
                let mut wrong = request(input);
                wrong.coefficients.extend(pack(&json!(["1","-0"])));
                let rejected = observe(&mut state, &wrong);
                assert_eq!(rejected.json["status"], ERROR);
                assert!(rejected.json["panic"].is_null());
                assert_eq!(rejected.json["output"], json!([]));
                assert!(rejected.json["raw"].is_null());
                assert_eq!(rejected.json["input_unchanged"], true);
                let entry = state.entries.get(&key).unwrap() as *const MpEntry as usize;
                assert!(before.is_none_or(|address| address == entry));
                let recovered = observe(&mut state, &r);
                assert_eq!(state.entries.get(&key).unwrap() as *const MpEntry as usize, entry);
                assert_eq!(recovered.json, fresh.json);
                let mut invalid = input.clone();
                invalid["masses"][1] = json!("1");
                let rejected = observe(&mut state,&request(&invalid));
                assert_eq!(rejected.json["status"], ERROR);
                assert!(rejected.json["panic"].is_null());
                assert_eq!(rejected.json["output"],json!([]));
                assert!(rejected.json["raw"].is_null());
                assert_eq!(observe(&mut state,&r).json,fresh.json);
                let raw = fresh.raw.as_ref().unwrap();
                let coefficient = parse_records(&pack(&input["coefficients"]),bits).unwrap();
                let masses = parse_records(&pack(&input["masses"]),bits).unwrap();
                let expected = if r.n == 1 { [coefficient[0].clone()*&masses[0],coefficient[0].clone()*&masses[1]] }
                    else { [coefficient[0].clone(),Float::new(bits)] };
                let norm = expected[0].norm()+expected[1].norm();
                let bound = Float::parse("1e-31",Some(bits)).unwrap()*norm;
                assert!((raw[2].clone()-&expected[0]).norm()<=bound);
                assert!((raw[3].clone()-&expected[1]).norm()<=bound);
                assert!(raw[4].is_fully_zero() && raw[5].is_fully_zero());
            }
        }
    });
}

#[test]
#[ignore = "requires exclusive MP slot; seven retained physicalfixtures in allthree modes"]
fn seven_physical_fixtures_all_modes() {
    // Inputs and Ninja references are unchanged from tests/replay.py. Including
    // them in this libtest target avoids a separately linked example binary.
    const CASES: [(&str, &str); 7] = [
        (include_str!("../tests/data/cases/madloop-four-photon-top-box-0.json"),
         include_str!("../tests/data/ninja/madloop-four-photon-top-box-0.json")),
        (include_str!("../tests/data/cases/madloop-gg-hh-top-box-2.json"),
         include_str!("../tests/data/ninja/madloop-gg-hh-top-box-2.json")),
        (include_str!("../tests/data/cases/madloop-gg-hh-top-triangle-0.json"),
         include_str!("../tests/data/ninja/madloop-gg-hh-top-triangle-0.json")),
        (include_str!("../tests/data/cases/madloop-gg-hhh-top-box-8.json"),
         include_str!("../tests/data/ninja/madloop-gg-hhh-top-box-8.json")),
        (include_str!("../tests/data/cases/madloop-gg-hhh-top-pentagon-26.json"),
         include_str!("../tests/data/ninja/madloop-gg-hhh-top-pentagon-26.json")),
        (include_str!("../tests/data/cases/madloop-gg-hhh-top-triangle-0.json"),
         include_str!("../tests/data/ninja/madloop-gg-hhh-top-triangle-0.json")),
        (include_str!("../tests/data/cases/madloop-hgg-top-triangle-0.json"),
         include_str!("../tests/data/ninja/madloop-hgg-top-triangle-0.json")),
    ];
    let mut failures = 0usize;
    let mut lanes = 0usize;
    let mut calls = 0usize;
    for (case, reference) in CASES {
        let case: Value = serde_json::from_str(case).unwrap();
        let reference: Value = serde_json::from_str(reference).unwrap();
        let mut context = std::ptr::null_mut();
        assert_eq!(unsafe { ninjars_context_create(ABI_VERSION, &mut context) }, SUCCESS);
        for lane in case["lanes"].as_array().unwrap() {
            lanes += 1;
            let expected = reference["lanes"].as_array().unwrap().iter()
                .find(|row| row["id"] == lane["id"]).unwrap();
            let complex = |value: &Value| C64::new(value[0].as_f64().unwrap(), value[1].as_f64().unwrap());
            let coefficients: Vec<_> = lane["tensor"].as_array().unwrap().iter().map(complex).collect();
            let masses: Vec<_> = lane["masses_squared"].as_array().unwrap().iter().map(complex).collect();
            let flatten = |key: &str| lane[key].as_array().unwrap().iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|value| value.as_f64().unwrap())).collect::<Vec<_>>();
            let matrix = flatten("s_matrix");
            let momenta = flatten("shifts");
            for mode in 0..3 {
                calls += 1;
                let mut laurent = [C64::ZERO; 3];
                let mut rational = C64::ZERO;
                let mut ascii = Vec::<String>::new();
                let status = if mode < 2 {
                    let evaluate = if mode == 0 { ninjars_evaluate_dp } else { ninjars_evaluate_dp_compensated };
                    unsafe { evaluate(context, case["n"].as_i64().unwrap() as i32,
                        case["rank"].as_i64().unwrap() as i32, coefficients.len(),
                        coefficients.as_ptr(), matrix.as_ptr(), matrix.len(),
                        momenta.as_ptr(), momenta.len(), masses.as_ptr(), masses.len(),
                        lane["mu_squared"].as_f64().unwrap(), laurent.as_mut_ptr(),
                        laurent.len(), &mut rational, 1) }
                } else {
                    let records = |values: Vec<f64>| pack(&json!(values.into_iter()
                        .map(|value| format!("{value:.39e}")).collect::<Vec<_>>()));
                    let ct = records(coefficients.iter().flat_map(|value| [value.re,value.im]).collect());
                    let mt = records(masses.iter().flat_map(|value| [value.re,value.im]).collect());
                    let st = records(matrix.clone());
                    let pt = records(momenta.clone());
                    let scale = records(vec![lane["mu_squared"].as_f64().unwrap()]);
                    let mut output = [b' '; 8*RECORD];
                    let status = unsafe { ninjars_evaluate_mp(context,
                        case["n"].as_i64().unwrap() as i32, case["rank"].as_i64().unwrap() as i32,
                        coefficients.len(), ct.as_ptr(), ct.len(), st.as_ptr(), st.len(),
                        pt.as_ptr(), pt.len(), mt.as_ptr(), mt.len(), scale.as_ptr(), scale.len(),
                        output.as_mut_ptr(), output.len()) };
                    ascii = output.chunks_exact(RECORD).map(|field|
                        std::str::from_utf8(field).unwrap().trim().to_owned()).collect();
                    if status == SUCCESS {
                        let values: Vec<f64> = ascii.iter().map(|value| value.parse().unwrap()).collect();
                        for index in 0..3 { laurent[index] = C64::new(values[2*index],values[2*index+1]); }
                        rational = C64::new(values[6],values[7]);
                    }
                    status
                };
                let mut errors = Vec::new();
                let mut rational_difference = 0f64;
                let mut good = status == SUCCESS;
                for (index, (value,key)) in laurent.into_iter().chain([rational])
                    .zip(["eps0","eps_m1","eps_m2","rational"]).enumerate() {
                    let expected = complex(&expected[key]);
                    let error = (value-expected).abs()/expected.abs().max(1.);
                    if index == 3 { rational_difference = error; }
                    // Same existing cross-basis convention as examples/replay.rs:
                    // compare all complete Laurent terms, report MP's basis-
                    // dependent rational split independently.
                    if index != 3 || mode != 2 {
                        good &= error.is_finite() && error <= 1e-9;
                        errors.push(error);
                    }
                }
                failures += usize::from(!good);
                println!("NINJARS_FIXTURE_RECORD {}", json!({"case":case["name"],
                    "lane":lane["id"],"mode":mode,"status":status,
                    "laurent_and_rational":laurent.into_iter().chain([rational])
                        .map(|value| [value.re,value.im]).collect::<Vec<_>>(),
                    "mp_ascii":ascii,"compared_component_errors":errors,
                    "rational_compared":mode!=2,"basis_dependent_rational_scaled_difference":rational_difference,
                    "tolerance":1e-9,"passed":good}));
            }
        }
        assert_eq!(unsafe { ninjars_context_destroy(context) }, SUCCESS);
    }
    assert_eq!(lanes,21);
    assert_eq!(calls,63);
    println!("NINJARS_FIXTURE_SUMMARY {}", json!({"fixtures":7,"lanes":lanes,
        "calls":calls,"modes":["native-dp","compensated-dp","genuine256"],
        "failed_calls":failures,"passed":failures==0}));
    assert_eq!(failures,0,"all failed fixtures retained without changing inputs or thresholds");
}
