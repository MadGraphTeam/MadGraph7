//! Probe the exact binary64 reduction weights without changing cut arithmetic.
//! Unit master values expose coefficients because integration is linear.
use ninjars::fast::{C64,FastAmplitude,FastInput,FastMaster,FastMasters,FastOptions,FastResult,FastStatus,TensorNumerator};
use serde_json::{json,Value};
use std::{env,fs};
fn c(z:C64)->Value{json!([z.re,z.im])}
fn numbers(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|x|C64::new(x[0].as_f64().unwrap(),x[1].as_f64().unwrap())).collect()}
struct Basis<'a>{records:&'a[Value], row:usize, output:usize, target:usize}
impl FastMasters for Basis<'_>{
 fn evaluate(&mut self,family:FastMaster,args:&[C64],out:&mut[[C64;3]],ok:&mut[bool])->ninjars::Result<()>{
  for (r,raw) in args.chunks_exact(family.arity()).enumerate(){
   let rec=&self.records[self.row];self.row+=1;
   assert_eq!(rec["family"],format!("{family:?}"));
   for (z,w) in raw.iter().zip(pairs(&rec["raw_args"])) {assert_eq!(z.re.to_bits(),w.re.to_bits());assert_eq!(z.im.to_bits(),w.im.to_bits());}
   ok[r]=true;
   for j in 0..family.outputs(){out[r*family.outputs()+j]=[C64::ZERO;3];if self.output==self.target {out[r*family.outputs()+j][0]=C64::real(1.);}self.output+=1;}
  }
  Ok(())
 }
}
fn run(v:&Value,p:&mut Basis<'_>)->FastResult{
 let n=v["n"].as_u64().unwrap() as usize;let rank=v["rank"].as_u64().unwrap() as usize;
 let coeff=pairs(&v["coefficients"]);let masses=pairs(&v["masses_squared"]);let s=numbers(&v["s_matrix_column_major"]);
 let shifts:Vec<[f64;4]>=numbers(&v["momenta_column_major"]).chunks_exact(4).map(|x|x.try_into().unwrap()).collect();let mu=[v["scale_squared"].as_f64().unwrap()];
 let mut a=FastAmplitude::new(n,rank,FastOptions::default()).unwrap();let mut t=TensorNumerator::new(rank,1);t.coefficients_mut().copy_from_slice(&coeff);
 let mut out=[FastResult::default()];let mut status=[FastStatus::NotEvaluated];
 a.evaluate(&FastInput{lanes:1,shifts:&shifts,masses_squared:&masses,scales_squared:&mu,s_matrix:Some(&s)},&mut t,p,&mut out,&mut status).unwrap();
 assert_eq!(status[0],FastStatus::Success);assert_eq!(p.row,p.records.len());out[0]
}
fn main(){
 let args:Vec<_>=env::args().collect();let inputs:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
 let capture:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[2]).unwrap()).unwrap();let mut result=vec![];
 for (v,row) in inputs.iter().zip(&capture){
  assert_eq!(v["loop"],row["loop"]);assert_eq!(v["direction"],row["direction"]);
  let records=row["native"]["masters"].as_array().unwrap();let count:usize=records.iter().map(|r|r["outputs"].as_array().unwrap().len()).sum();
  let zero=run(v,&mut Basis{records,row:0,output:0,target:usize::MAX});assert_eq!(zero.cut_constructible,C64::ZERO);
  let mut weights=vec![];for target in 0..count {let r=run(v,&mut Basis{records,row:0,output:0,target});assert_eq!(r.rational,zero.rational);weights.push(c(r.cut_constructible));}
  result.push(json!({"loop":v["loop"],"direction":v["direction"],"weights":weights,"rational":c(zero.rational)}));
 }
 fs::write(&args[3],serde_json::to_string_pretty(&result).unwrap()).unwrap();
}
