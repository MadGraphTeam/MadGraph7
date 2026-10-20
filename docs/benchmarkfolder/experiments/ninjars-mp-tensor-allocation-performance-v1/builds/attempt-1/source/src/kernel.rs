//! Pure callback replay. No amplitude, cuts, geometry or scalar masters are used.
mod original;
mod allocation;
mod common;

use common::{Q, Z, read_q, read_z, read_zs, same};
use ninjars::{ExpansionKind, ExpansionRequest, Float, FloatLike, Numerator,
              NumeratorRequest, SampleId};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const RECORDS: usize = 1392;
const OUTPUTS: usize = 86436;
const PAIRS: usize = 21;
static CURRENT_RECORD: AtomicUsize = AtomicUsize::new(usize::MAX);

struct OwnedRequest {
    id: SampleId,
    partition: Vec<usize>,
    q: Option<Q>,
    mu: Option<Z>,
    vectors: Vec<Q>,
    parameters: Vec<Z>,
}
struct Call {
    kind: Option<ExpansionKind>,
    requests: Vec<OwnedRequest>,
    expected: Vec<Z>,
}
struct Case {
    n: usize,
    rank: usize,
    coefficients: Vec<Z>,
    calls: Vec<Call>,
}
enum View<'a> {
    Value(Vec<NumeratorRequest<'a, Z>>),
    Expansion(&'a ExpansionKind, Vec<ExpansionRequest<'a, Z>>),
}
enum Tensor {
    Original(original::TensorGenericNumerator<Float>),
    Allocation(allocation::TensorGenericNumerator<Float>),
}
impl Tensor {
    fn new(rank: usize, arm: &str) -> Self {
        let prototype = Float::new(256);
        match arm {
            "original" => Self::Original(original::TensorGenericNumerator::new(rank, 0, &prototype).unwrap()),
            "allocation" => Self::Allocation(allocation::TensorGenericNumerator::new(rank, 0, &prototype).unwrap()),
            _ => panic!("invalid tensor arm"),
        }
    }
    fn bind(&mut self, coefficients: Vec<Z>) {
        match self {
            Self::Original(t) => t.set_coefficients(coefficients, 1).unwrap(),
            Self::Allocation(t) => t.set_coefficients(coefficients, 1).unwrap(),
        }
    }
    fn numerator(&mut self) -> &mut dyn Numerator<Z> {
        match self { Self::Original(t) => t, Self::Allocation(t) => t }
    }
}
struct Control;
impl Numerator<Z> for Control {
    fn rank(&self) -> usize { 0 }
    fn evaluate_batch(&mut self, requests: &[NumeratorRequest<'_, Z>], output: &mut [Z]) -> ninjars::Result<()> {
        black_box((requests, output)); Ok(())
    }
    fn expansion_batch(&mut self, kind: &ExpansionKind, requests: &[ExpansionRequest<'_, Z>], output: &mut [Z]) -> ninjars::Result<()> {
        black_box((kind, requests, output)); Ok(())
    }
}
fn usize_field(value: &Value) -> usize { usize::try_from(value.as_u64().unwrap()).unwrap() }
fn kind(value: &Value) -> Option<ExpansionKind> {
    Some(match value["name"].as_str().unwrap() {
        "value" => { assert!(value.get("powers").is_none()); return None; }
        "Mu" => ExpansionKind::Mu { powers: serde_json::from_value(value["powers"].clone()).unwrap() },
        "MuShifted" => ExpansionKind::MuShifted { powers: serde_json::from_value(value["powers"].clone()).unwrap() },
        "T3" => ExpansionKind::T3 { powers: serde_json::from_value(value["powers"].clone()).unwrap() },
        "T2" => ExpansionKind::T2 { powers: serde_json::from_value(value["powers"].clone()).unwrap() },
        _ => panic!("unknown expansion kind"),
    })
}
fn load(path: &str) -> Vec<Case> {
    assert!(std::fs::metadata(path).unwrap().len() <= 2 * 1024 * 1024 * 1024);
    let reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let mut cases = Vec::new();
    let mut outputs = 0usize;
    let mut final_fields = 0usize;
    let mut shapes = BTreeSet::new();
    for line in reader.lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        assert_eq!(usize_field(&value["bridge_record"]), cases.len());
        let n = usize_field(&value["n"]);
        let rank = usize_field(&value["rank"]);
        assert!((2..=5).contains(&n) && (1..=4).contains(&rank));
        shapes.insert((n, rank));
        let coefficients = read_zs(&value["coefficients"]);
        assert_eq!(coefficients.len(), ninjars::fast::tensor_layout::ncoeff(rank));
        let ascii = value["final_ascii"].as_array().unwrap();
        assert_eq!(ascii.len(), 8);
        assert!(ascii.iter().all(|v| v.as_str().is_some()));
        final_fields += ascii.len();
        let calls: Vec<_> = value["calls"].as_array().unwrap().iter().map(|call| {
            let kind = kind(&call["kind"]);
            let requests: Vec<_> = call["requests"].as_array().unwrap().iter().map(|r| {
                let id = SampleId { lane: usize_field(&r["lane"]), sample: usize_field(&r["sample"]) };
                assert_eq!(id.lane, 0);
                let partition = r["partition"].as_array().unwrap().iter().map(usize_field).collect();
                if kind.is_none() {
                    OwnedRequest { id, partition, q: Some(read_q(&r["q"])), mu: Some(read_z(&r["mu"])), vectors: Vec::new(), parameters: Vec::new() }
                } else {
                    OwnedRequest { id, partition, q: None, mu: None,
                        vectors: r["vectors"].as_array().unwrap().iter().map(read_q).collect(),
                        parameters: read_zs(&r["parameters"]) }
                }
            }).collect();
            let expected = read_zs(&call["output"]);
            assert_eq!(expected.len(), requests.len() * kind.as_ref().map_or(1, ExpansionKind::len));
            outputs += expected.len();
            Call { kind, requests, expected }
        }).collect();
        cases.push(Case { n, rank, coefficients, calls });
    }
    assert_eq!(cases.len(), RECORDS);
    assert_eq!(outputs, OUTPUTS);
    assert_eq!(final_fields, 11136);
    assert_eq!(shapes, BTreeSet::from([(2,1),(2,2),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,4)]));
    cases
}
fn views(case: &Case) -> Vec<View<'_>> {
    case.calls.iter().map(|call| match &call.kind {
        None => View::Value(call.requests.iter().map(|r| NumeratorRequest {
            id: r.id, partition: &r.partition, q: r.q.as_ref().unwrap(), mu_squared: r.mu.as_ref().unwrap(),
        }).collect()),
        Some(kind) => View::Expansion(kind, call.requests.iter().map(|r| ExpansionRequest {
            id: r.id, partition: &r.partition, vectors: &r.vectors, parameters: &r.parameters,
        }).collect()),
    }).collect()
}
// This single dispatch routine is shared by both arms and by the clock control.
// Its only callees are the selected Numerator implementation's two callbacks.
#[inline(never)]
fn dispatch(tensor: &mut dyn Numerator<Z>, views: &[View<'_>], outputs: &mut [Vec<Z>]) {
    for (view, output) in views.iter().zip(outputs) {
        match view {
            View::Value(requests) => tensor.evaluate_batch(requests, output).unwrap(),
            View::Expansion(kind, requests) => tensor.expansion_batch(kind, requests, output).unwrap(),
        }
    }
}
fn block(cases: &[Case], states: &mut BTreeMap<(usize, usize), Tensor>, arm: &str,
         timed: bool, control: bool) -> Value {
    let zero = Z::new(Float::new(256).zero(), Float::new(256).zero());
    let mut intervals = Vec::with_capacity(RECORDS);
    let mut checked = 0usize;
    let mut null = Control;
    for (index, case) in cases.iter().enumerate() {
        CURRENT_RECORD.store(index, Ordering::Relaxed);
        let tensor = states.entry((case.n, case.rank)).or_insert_with(|| Tensor::new(case.rank, arm));
        tensor.bind(case.coefficients.clone());
        let views = views(case);
        let mut outputs: Vec<_> = case.calls.iter().map(|call| vec![zero.clone(); call.expected.len()]).collect();
        assert_eq!(views.len(), outputs.len());
        let selected: &mut dyn Numerator<Z> = if control { &mut null } else { tensor.numerator() };
        if timed {
            let start = Instant::now();
            dispatch(selected, &views, &mut outputs);
            let elapsed = start.elapsed().as_nanos();
            intervals.push(u64::try_from(elapsed).unwrap());
        } else {
            dispatch(selected, &views, &mut outputs);
        }
        if !control {
            for (actual, call) in outputs.iter().zip(&case.calls) {
                same(actual, &call.expected);
                checked += actual.len();
            }
        }
    }
    if !control { assert_eq!(checked, OUTPUTS); }
    if timed { assert_eq!(intervals.len(), RECORDS); }
    json!({"records":RECORDS,"checked_complex_outputs":checked,"exact":!control,
        "control_only":control,"total_ns":intervals.iter().sum::<u64>(),"intervals_ns":intervals})
}
fn emit(out: &mut impl Write, value: Value) {
    serde_json::to_writer(&mut *out, &value).unwrap(); out.write_all(b"\n").unwrap(); out.flush().unwrap();
}
fn run(trace: &str, out: &mut impl Write) {
    let cases = load(trace);
    emit(out, json!({"event":"loaded","records":cases.len(),"outputs":OUTPUTS,"pairs":PAIRS,
        "timed_scope":"callback dispatch only; no amplitude/geometry/master calls"}));
    let mut original = BTreeMap::new();
    let mut allocation = BTreeMap::new();
    for (arm, states) in [("original", &mut original), ("allocation", &mut allocation)] {
        let result = block(&cases, states, arm, false, false);
        emit(out, json!({"event":"warm","arm":arm,"result":result}));
    }
    for pair in 0..PAIRS {
        let order = if pair % 2 == 0 { ["original", "allocation"] } else { ["allocation", "original"] };
        for (position, arm) in order.into_iter().enumerate() {
            emit(out, json!({"event":"block-start","pair":pair,"position":position,"arm":arm}));
            let states = if arm == "original" { &mut original } else { &mut allocation };
            let result = block(&cases, states, arm, true, false);
            emit(out, json!({"event":"block","pair":pair,"position":position,"arm":arm,"result":result}));
        }
    }
    // Twenty-one complete control blocks are separate diagnostics, never subtracted.
    for index in 0..PAIRS {
        let result = block(&cases, &mut original, "original", true, true);
        emit(out, json!({"event":"clock-control","index":index,"result":result}));
    }
    emit(out, json!({"event":"complete","pairs":PAIRS,"exact":true}));
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: tensor-kernel TRACE.jsonl RESULTS.jsonl");
    let mut out = std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[2]).unwrap());
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&args[1], &mut out)));
    if outcome.is_err() {
        emit(&mut out, json!({"event":"failed","bridge_record":CURRENT_RECORD.load(Ordering::Relaxed),
            "accepted":false,"reason":"panic/assertion; see retained stderr"}));
        std::process::exit(1);
    }
}
