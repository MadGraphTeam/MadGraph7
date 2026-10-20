#!/usr/bin/env python3
"""Create isolated capture and pure callback binaries; never install anything."""
import hashlib,json,re,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
FROZEN=HERE.parent/'ninjars-mp-tensor-allocation-v1'
WORK=ROOT/'.benchmark-work/experiments/ninjars-mp-tensor-allocation-performance-v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def verified_frozen_inputs():
    plan=json.loads((HERE/'plan.json').read_text())
    assert sha(FROZEN/'manifest.json')==plan['correctness_manifest']['sha256']
    manifest=json.loads((FROZEN/'manifest.json').read_text())
    assert manifest['identity']==plan['correctness_manifest']['identity']
    consumed=['original/Cargo.toml','original/Cargo.lock','original/src/tensor.rs',
              'tensor.rs','tensor_capture_tests.rs']
    result={}
    for name in consumed:
        path=FROZEN/name
        actual={'sha256':sha(path),'bytes':path.stat().st_size}
        assert actual==manifest['files'][name],f'frozen source changed: {name}'
        result[name]=actual
    return result

def main():
    assert Path.cwd()==ROOT
    plan=json.loads((HERE/'plan.json').read_text())
    frozen_inputs=verified_frozen_inputs()
    WORK.mkdir(parents=True,exist_ok=False)
    crate=WORK/'crate';(crate/'src').mkdir(parents=True)
    manifest=(FROZEN/'original/Cargo.toml').read_text()
    manifest=re.sub(r'\[lib\]\n.*?(?=\[dependencies\])','',manifest,flags=re.S)
    manifest=manifest.replace('[dev-dependencies]\nserde_json = {version="1",features=["float_roundtrip"]}\n','')
    manifest=manifest.replace('[dependencies]\n','[dependencies]\nserde_json = {version="1",features=["float_roundtrip"]}\n')
    manifest+='\n[[bin]]\nname="tensor-capture"\npath="src/capture.rs"\n\n[[bin]]\nname="tensor-kernel"\npath="src/kernel.rs"\n'
    (crate/'Cargo.toml').write_text(manifest)
    shutil.copyfile(FROZEN/'original/Cargo.lock',crate/'Cargo.lock')
    shutil.copyfile(FROZEN/'original/src/tensor.rs',crate/'src/original.rs')
    shutil.copyfile(FROZEN/'tensor.rs',crate/'src/allocation.rs')
    for file in ['common.rs','kernel.rs']: shutil.copyfile(HERE/file,crate/'src'/file)
    s=(FROZEN/'tensor_capture_tests.rs').read_text()
    s=s[:s.index('\n#[test]\n#[ignore = "runs the full retained')]
    start=s.index('struct Oracle')
    s='''mod original;
mod allocation;
mod common;
use common::{float_json,z_json,zs_json,q_json};
use allocation::TensorGenericNumerator;
use original as dense_reference;
use ninjars::{AmplitudeInput,Complex,EvaluationStatus,Float,FloatLike,FourMomentum,Real,SingleFloat,Numerator,NumeratorRequest,ExpansionRequest,ExpansionKind};
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
''' + s[start:]
    s=s.replace('counts: BTreeMap<&\'static str, usize>,','counts: BTreeMap<&\'static str, usize>,\n    calls: Vec<Value>,')
    s=s.replace('counts: BTreeMap::new(),','counts: BTreeMap::new(),\n                    calls: Vec::new(),')
    s=s.replace('*self.counts.entry("numerator_values").or_default() += output.len();','''self.calls.push(json!({"kind":{"name":"value"},"requests":requests.iter().map(|r|json!({"lane":r.id.lane,"sample":r.id.sample,"partition":r.partition,"q":q_json(r.q),"mu":z_json(r.mu_squared)})).collect::<Vec<_>>(),"output":zs_json(output)}));
        *self.counts.entry("numerator_values").or_default() += output.len();''')
    s=s.replace('*self.counts.entry(name).or_default() += output.len();','''self.calls.push(json!({"kind":kind_json(kind),"requests":requests.iter().map(|r|json!({"lane":r.id.lane,"sample":r.id.sample,"partition":r.partition,"vectors":r.vectors.iter().map(q_json).collect::<Vec<_>>(),"parameters":zs_json(r.parameters)})).collect::<Vec<_>>(),"output":zs_json(output)}));
        *self.counts.entry(name).or_default() += output.len();''')
    s=s.replace('#[test]\n#[ignore = "requires an explicit retained MadLoop request capture"]\nfn actual_madloop_requests_match_dense_expansions_exactly()', 'fn main()')
    s=s.replace('let mut seen = HashSet::new();','''let trace_path=std::env::var("NINJARS_CALLBACK_TRACE").expect("trace destination required");
            let mut trace=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(trace_path).unwrap());
            let mut trace_bytes=0usize;''')
    s=s.replace('''if record["status"] != 0 {
                    continue;
                }''','''assert_eq!(record["status"],0);''')
    s=re.sub(r'\s*// Retain one successful actual request.*?if !seen.insert\(\(n, rank\)\).*?\n                }','',s,flags=re.S)
    s=s.replace('let coefficients = pairs(records(&record["coefficients"]));','let coefficients = pairs(records(&record["coefficients"]));\n                let trace_coefficients=zs_json(&coefficients);')
    s=s.replace('for (kind, count) in oracle.counts {','''let payload=json!({"bridge_record":tested,"n":n,"rank":rank,"coefficients":trace_coefficients,"calls":oracle.calls,"final_ascii":formatted});
                let bytes=serde_json::to_vec(&payload).unwrap();trace_bytes+=bytes.len()+1;
                assert!(trace_bytes<=2*1024*1024*1024,"trace exceeds predeclared bound");
                trace.write_all(&bytes).unwrap();trace.write_all(b"\\n").unwrap();
                for (kind, count) in oracle.counts {''')
    s=s.replace('assert!(tested >= 4, "expected multiple actual topology/rank shapes");','assert_eq!(tested,1392);\n            trace.flush().unwrap();')
    s=s.replace('self.planned.rank()', 'self.dense.rank()')
    s=s.replace('self.planned.evaluate_batch(requests, output)?;\n        self.dense.evaluate_batch(requests, &mut reference)?;',
                'self.dense.evaluate_batch(requests, output)?;\n        self.planned.evaluate_batch(requests, &mut reference)?;')
    s=s.replace('self.planned.expansion_batch(kind, requests, output)?;\n        self.dense.expansion_batch(kind, requests, &mut reference)?;',
                'self.dense.expansion_batch(kind, requests, output)?;\n        self.planned.expansion_batch(kind, requests, &mut reference)?;')
    (HERE/'capture.rs').write_text(s);(crate/'src/capture.rs').write_text(s)
    inputs={str(p.relative_to(crate)):sha(p) for p in sorted(crate.rglob('*')) if p.is_file()}
    (HERE/'build-inputs.json').write_text(json.dumps({'crate':str(crate.relative_to(ROOT)),'sources':inputs,
        'frozen_inputs':frozen_inputs,'plan_identity':plan['identity']},indent=2)+'\n')
    print(crate)

if __name__=='__main__':main()
