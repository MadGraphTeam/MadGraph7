//! Diagnostic only. Threshold preprocessing is copied from pinned NinjaRS
//! oneloop_native.rs (0d0d7e7), then its exact binary64 results are widened.
use ninjars::{Complex, Float, MasterIntegral, MasterProvider, MasterRequest};
use ninjars::fast::{C64, FastAmplitude, FastInput, FastMaster, FastMasters, FastOptions,
    FastResult, FastStatus, TensorNumerator};
use ninjars_symbolica::{OneLoopNativeMasters, SymbolicaMasters};
use symbolica::domains::float::RealLike;
use serde_json::{json, Value};
use std::{env, fs};

type Z = Complex<Float>;
fn c(z:C64)->Value { json!([z.re,z.im]) }
fn l(z:&[C64;3])->Value {json!(z.iter().map(|&z|c(z)).collect::<Vec<_>>())}
fn widen(z:C64)->Z {Complex::new(Float::with_val(256,z.re),Float::with_val(256,z.im))}
fn snap_onshell(args:&mut[C64],invariants:usize) {
    let masses=args.len()-invariants-1;
    let small=|z:&C64|z.re.abs()+z.im.abs()<1e-9;
    for mass in &mut args[invariants..invariants+masses] {
        if small(mass) {*mass=C64::ZERO;}
    }
    let (momenta,rest)=args.split_at_mut(invariants);
    for momentum in momenta {
        if small(momentum) {*momentum=C64::ZERO;}
        else if let Some(mass)=rest[..masses].iter()
            .find(|m|m.im==0. && (momentum.re-m.re).abs()<1e-9) {*momentum=*mass;}
    }
}
fn integrals(f:FastMaster)->Vec<MasterIntegral> {
    use MasterIntegral::*;
    match f {
        FastMaster::A0=>vec![A0], FastMaster::C0=>vec![C0], FastMaster::D0=>vec![D0],
        FastMaster::BubbleRank2=>vec![B0,B1,B11],
        FastMaster::BubbleRank3=>vec![B0,B1,B11,B111],
        FastMaster::BubbleRank4=>vec![B0,B1,B11,B111,B1111],
    }
}
struct Provider {
    native:OneLoopNativeMasters,
    high:SymbolicaMasters<Float>,
    replace:bool,
    records:Vec<Value>,
}
impl Provider {
    fn new(replace:bool)->Self {
        Self{native:OneLoopNativeMasters::new().unwrap(),
            high:SymbolicaMasters::new(Float::with_val(256,0.)).with_onshell_threshold(None),
            replace,records:vec![]}
    }
}
impl FastMasters for Provider {
    fn evaluate(&mut self,family:FastMaster,args:&[C64],out:&mut[[C64;3]],ok:&mut[bool])->ninjars::Result<()> {
        FastMasters::evaluate(&mut self.native,family,args,out,ok)?;
        let kinds=integrals(family);
        for (row,raw) in args.chunks_exact(family.arity()).enumerate() {
            let mut snapped=raw.to_vec();
            let ni=kinds[0].invariant_count();
            snap_onshell(&mut snapped,ni);
            let exact:Vec<_>=snapped.iter().copied().map(widen).collect();
            let mut rec=json!({"family":format!("{family:?}"),"row":row,
                "raw_args":raw.iter().copied().map(c).collect::<Vec<_>>(),
                "raw_arg_bits":raw.iter().map(|z|[format!("{:016x}",z.re.to_bits()),format!("{:016x}",z.im.to_bits())]).collect::<Vec<_>>(),
                "onshell_threshold":1e-9,
                "thresholded_args":snapped.iter().copied().map(c).collect::<Vec<_>>(),
                "native_ok":ok[row],"outputs":[]});
            for (column,&integral) in kinds.iter().enumerate() {
                let index=row*family.outputs()+column;
                let native=out[index];
                let mut high=[std::array::from_fn(|_|widen(C64::ZERO))];
                let result=self.high.evaluate(MasterRequest {integral,
                    invariants:&exact[..ni],masses_squared:&exact[ni..exact.len()-1],
                    scales_squared:&exact[exact.len()-1..]},&mut high);
                let mut entry=json!({"integral":format!("{integral:?}"),"native":l(&native),
                    "high_error":result.as_ref().err().map(ToString::to_string)});
                if result.is_ok() {
                    let rounded=std::array::from_fn(|i|C64::new(high[0][i].re.to_f64(),high[0][i].im.to_f64()));
                    entry["high_decimal"]=json!(high[0].iter().map(|z|[format!("{:.76e}",z.re),format!("{:.76e}",z.im)]).collect::<Vec<_>>());
                    entry["high_rounded"]=l(&rounded);
                    entry["native_minus_high_rounded"]=l(&std::array::from_fn(|i|native[i]-rounded[i]));
                    if self.replace && ok[row] {out[index]=rounded;}
                } else if self.replace {
                    ok[row]=false;
                }
                rec["outputs"].as_array_mut().unwrap().push(entry);
            }
            self.records.push(rec);
        }
        Ok(())
    }
}
fn numbers(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|x|C64::new(x[0].as_f64().unwrap(),x[1].as_f64().unwrap())).collect()}
fn run(v:&Value,p:&mut Provider)->Value {
    let n=v["n"].as_u64().unwrap() as usize;
    let rank=v["rank"].as_u64().unwrap() as usize;
    let coefficients=pairs(&v["coefficients"]);
    let masses=pairs(&v["masses_squared"]);
    let s=numbers(&v["s_matrix_column_major"]);
    let shifts:Vec<[f64;4]>=numbers(&v["momenta_column_major"]).chunks_exact(4).map(|x|x.try_into().unwrap()).collect();
    let scales=[v["scale_squared"].as_f64().unwrap()];
    let mut a=FastAmplitude::new(n,rank,FastOptions::default()).unwrap();
    let mut tensor=TensorNumerator::new(rank,1);
    tensor.coefficients_mut().copy_from_slice(&coefficients);
    let mut output=[FastResult::default()];
    let mut statuses=[FastStatus::NotEvaluated];
    let result=a.evaluate(&FastInput{lanes:1,shifts:&shifts,masses_squared:&masses,
        scales_squared:&scales,s_matrix:Some(&s)},&mut tensor,p,&mut output,&mut statuses);
    json!({"error":result.err().map(|e|e.to_string()),"status":format!("{:?}",statuses[0]),
        "laurent":l(&output[0].laurent),"rational":c(output[0].rational),
        "masters":std::mem::take(&mut p.records)})
}
fn main(){
    let argv:Vec<_>=env::args().collect();
    let inputs:Vec<Value>=serde_json::from_str(&fs::read_to_string(&argv[1]).unwrap()).unwrap();
    let mut native=Provider::new(false);
    let mut replace=Provider::new(true);
    let mut results=vec![];
    for v in inputs {
        eprintln!("loop {} direction {}",v["loop"],v["direction"]);
        let a=run(&v,&mut native);
        let b=run(&v,&mut replace);
        assert_eq!(a["status"],"Success"); assert_eq!(b["status"],"Success");
        assert_eq!(a["laurent"],v["ninjars"],"native replay mismatch");
        results.push(json!({"loop":v["loop"],"direction":v["direction"],"native":a,"substituted":b}));
        fs::write(&argv[2],serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
}
