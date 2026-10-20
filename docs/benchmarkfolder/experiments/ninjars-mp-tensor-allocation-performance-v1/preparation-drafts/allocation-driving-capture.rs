mod original;
mod allocation;
mod common;
use common::{float_json,z_json,zs_json,q_json};
use allocation::TensorGenericNumerator;
use original as dense_reference;
use ninjars::{AmplitudeInput,Complex,EvaluationStatus,Float,FloatLike,FourMomentum,Real,Numerator,NumeratorRequest,ExpansionRequest,ExpansionKind};
use ninjars::fast::tensor_layout::ncoeff;
use ninjars_symbolica::{SymbolicaAmplitudeKernels,SymbolicaCuts,SymbolicaHighCuts,SymbolicaMasters};
use serde_json::{Value,json};
use std::collections::{BTreeMap,HashSet};
use std::io::Write;
const MP_BITS:u32=256;
const RECORD:usize=64;
fn parse_records(data:&[u8], precision:u32)->Result<Vec<Float>,String>{
 data.chunks_exact(RECORD).map(|bytes|{
  let text=std::str::from_utf8(bytes).unwrap().trim();
  let value=Float::parse(text,Some(113)).map_err(|e|e.to_string())?;
  if !value.is_finite(){return Err("nonfinite input".into())}
  Ok(Float::new(precision).from_rational(&value.to_rational()))
 }).collect()
}
fn pairs(values:Vec<Float>)->Vec<Complex<Float>>{let mut v=values.into_iter();let mut out=Vec::new();while let Some(re)=v.next(){out.push(Complex::new(re,v.next().unwrap()));}out}
fn real(value:Float)->Complex<Float>{let zero=value.zero();Complex::new(value,zero)}
fn kind_json(kind:&ExpansionKind)->Value{match kind{
 ExpansionKind::Mu{powers}=>json!({"name":"Mu","powers":powers}),
 ExpansionKind::MuShifted{powers}=>json!({"name":"MuShifted","powers":powers}),
 ExpansionKind::T3{powers}=>json!({"name":"T3","powers":powers}),
 ExpansionKind::T2{powers}=>json!({"name":"T2","powers":powers}),
}}
struct Oracle {
    planned: TensorGenericNumerator<Float>,
    dense: dense_reference::TensorGenericNumerator<Float>,
    counts: BTreeMap<&'static str, usize>,
    calls: Vec<Value>,
}
fn equal_component(a: &Float, b: &Float) {
    assert_eq!(a.prec(), b.prec());
    assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    assert!(a.is_finite() && b.is_finite());
    assert_eq!(a.to_rational(), b.to_rational());
}
fn equal(a: &[Complex<Float>], b: &[Complex<Float>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        equal_component(&a.re, &b.re);
        equal_component(&a.im, &b.im);
    }
}
impl Numerator<Complex<Float>> for Oracle {
    fn rank(&self) -> usize {
        self.planned.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Complex<Float>>],
        output: &mut [Complex<Float>],
    ) -> ninjars::Result<()> {
        let mut reference = output.to_vec();
        self.planned.evaluate_batch(requests, output)?;
        self.dense.evaluate_batch(requests, &mut reference)?;
        equal(output, &reference);
        self.calls.push(json!({"kind":{"name":"value"},"requests":requests.iter().map(|r|json!({"lane":r.id.lane,"sample":r.id.sample,"partition":r.partition,"q":q_json(r.q),"mu":z_json(r.mu_squared)})).collect::<Vec<_>>(),"output":zs_json(output)}));
        *self.counts.entry("numerator_values").or_default() += output.len();
        Ok(())
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Complex<Float>>],
        output: &mut [Complex<Float>],
    ) -> ninjars::Result<()> {
        let mut reference = output.to_vec();
        self.planned.expansion_batch(kind, requests, output)?;
        self.dense.expansion_batch(kind, requests, &mut reference)?;
        equal(output, &reference);
        let name = match kind {
            ExpansionKind::Mu { .. } => "mu",
            ExpansionKind::MuShifted { .. } => "mu_shifted",
            ExpansionKind::T3 { .. } => "t3",
            ExpansionKind::T2 { .. } => "t2",
        };
        self.calls.push(json!({"kind":kind_json(kind),"requests":requests.iter().map(|r|json!({"lane":r.id.lane,"sample":r.id.sample,"partition":r.partition,"vectors":r.vectors.iter().map(q_json).collect::<Vec<_>>(),"parameters":zs_json(r.parameters)})).collect::<Vec<_>>(),"output":zs_json(output)}));
        *self.counts.entry(name).or_default() += output.len();
        Ok(())
    }
}
fn records(value: &Value) -> Vec<Float> {
    let mut bytes = Vec::new();
    for string in value.as_array().unwrap() {
        let text = string.as_str().unwrap().as_bytes();
        assert!(text.len() <= RECORD);
        let mut record = [b' '; RECORD];
        record[..text.len()].copy_from_slice(text);
        bytes.extend(record);
    }
    parse_records(&bytes, MP_BITS).unwrap()
}

fn main() {
    let path = std::env::var("NINJARS_TENSOR_REPLAY").expect("capture path required");
    let text = std::fs::read_to_string(path).unwrap();
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(move || {
            let prototype = Float::new(MP_BITS);
            let mut low = SymbolicaCuts::new(prototype.clone()).unwrap();
            let mut high = SymbolicaHighCuts::new(prototype.clone()).unwrap();
            let threshold = Float::with_val(MP_BITS, 1_000_000_000_000_000i64)
                / Float::with_val(MP_BITS, 2).pow(112);
            let mut masters =
                SymbolicaMasters::new(prototype.clone()).with_onshell_threshold(Some(threshold));
            let trace_path=std::env::var("NINJARS_CALLBACK_TRACE").expect("trace destination required");
            let mut trace=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(trace_path).unwrap());
            let mut trace_bytes=0usize;
            let mut total = BTreeMap::<&'static str, usize>::new();
            let mut tested = 0;
            for line in text.lines() {
                let record: Value = serde_json::from_str(line).unwrap();
                assert_eq!(record["status"],0);
                let n = record["n"].as_u64().unwrap() as usize;
                let rank = record["rank"].as_u64().unwrap() as usize;
                let coefficients = pairs(records(&record["coefficients"]));
                let trace_coefficients=zs_json(&coefficients);
                assert_eq!(coefficients.len(), ncoeff(rank));
                assert_eq!(coefficients.len(), record["count"].as_u64().unwrap() as usize);
                let shifts: Vec<_> = records(&record["momenta"])
                    .chunks_exact(4)
                    .map(|v| FourMomentum::from(std::array::from_fn(|k| real(v[k].clone()))))
                    .collect();
                let masses = pairs(records(&record["masses"]));
                let smat: Vec<_> = records(&record["s_matrix"]).into_iter().map(real).collect();
                let scales: Vec<_> = records(&record["scale"]).into_iter().map(real).collect();
                let sampling = [real(smat.iter().chain(&masses).map(|v| v.re.norm()).fold(
                    Float::with_val(MP_BITS, 1),
                    |a, b| if b > a { b } else { a },
                ))];
                let mut oracle = Oracle {
                    planned: TensorGenericNumerator::new(rank, 0, &prototype).unwrap(),
                    dense: dense_reference::TensorGenericNumerator::new(rank, 0, &prototype)
                        .unwrap(),
                    counts: BTreeMap::new(),
                    calls: Vec::new(),
                };
                oracle
                    .planned
                    .set_coefficients(coefficients.clone(), 1)
                    .unwrap();
                oracle.dense.set_coefficients(coefficients, 1).unwrap();
                let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())
                    .unwrap()
                    .into_amplitude(1)
                    .unwrap();
                amplitude.reset_batch(1);
                amplitude
                    .evaluate_batch_with_inputs(
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            sampling_scales: &sampling,
                            s_matrix: Some(&smat),
                        },
                        &mut oracle,
                        &mut low,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                assert_eq!(amplitude.statuses()[0], EvaluationStatus::Success);
                let result = &amplitude.results()[0];
                let formatted: Vec<_> = [0, 1, 2, 4]
                    .into_iter()
                    .flat_map(|i| [&result[i].re, &result[i].im])
                    .map(|v| format!("{v:.39e}"))
                    .collect();
                let expected: Vec<_> = record["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap())
                    .collect();
                assert_eq!(formatted, expected, "ABI output differs for ({n},{rank})");
                let payload=json!({"bridge_record":tested,"n":n,"rank":rank,"coefficients":trace_coefficients,"calls":oracle.calls,"final_ascii":formatted});
                let bytes=serde_json::to_vec(&payload).unwrap();trace_bytes+=bytes.len()+1;
                assert!(trace_bytes<=2*1024*1024*1024,"trace exceeds predeclared bound");
                trace.write_all(&bytes).unwrap();trace.write_all(b"\n").unwrap();
                for (kind, count) in oracle.counts {
                    *total.entry(kind).or_default() += count;
                }
                tested += 1;
                eprintln!("captured request n={n} rank={rank}: exact");
            }
            assert_eq!(tested,1392);
            trace.flush().unwrap();
            assert!(total.get("t2").copied().unwrap_or(0) > 0);
            assert!(total.get("t3").copied().unwrap_or(0) > 0);
            eprintln!("captured exact requests={tested}; expansion/value counts={total:?}");
        })
        .unwrap()
        .join()
        .unwrap();
}
