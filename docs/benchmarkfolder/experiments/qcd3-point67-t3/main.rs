mod t3;
use ninjars::fast::{C64,CMom,FastAmplitude,FastInput,FastMaster,FastMasters,FastOptions,FastResult,FastStatus,FastNumerator,TensorNumerator};
use serde_json::{json,Value};use std::{env,fs};
fn c(z:C64)->Value{json!([z.re,z.im])}
fn numbers(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|x|C64::new(x[0].as_f64().unwrap(),x[1].as_f64().unwrap())).collect()}
fn momenta(v:&Value)->Vec<CMom>{v.as_array().unwrap().iter().map(|v|pairs(v).try_into().unwrap()).collect()}
struct Replay<'a>{records:&'a[Value],row:usize}
impl FastMasters for Replay<'_>{fn evaluate(&mut self,f:FastMaster,args:&[C64],out:&mut[[C64;3]],ok:&mut[bool])->ninjars::Result<()>{for(r,raw)in args.chunks_exact(f.arity()).enumerate(){let rec=&self.records[self.row];self.row+=1;assert_eq!(rec["family"],format!("{f:?}"));for(z,w)in raw.iter().zip(pairs(&rec["raw_args"])){assert_eq!(z.re.to_bits(),w.re.to_bits());assert_eq!(z.im.to_bits(),w.im.to_bits());}ok[r]=rec["native_ok"].as_bool().unwrap();for j in 0..f.outputs(){out[r*f.outputs()+j]=pairs(&rec["outputs"][j]["native"]).try_into().unwrap();}}Ok(())}}
struct Tensor{native:TensorNumerator,coeff:Vec<C64>,rank:usize,mode:usize,records:Vec<Value>}
impl FastNumerator for Tensor{
 fn rank(&self)->usize{self.rank}
 fn evaluate(&mut self,cut:&[usize],lanes:&[usize],q:&[CMom],mu:&[C64],out:&mut[C64])->ninjars::Result<()>{self.native.evaluate(cut,lanes,q,mu,out)}
 fn mu_expansion(&mut self,cut:&[usize],lanes:&[usize],v0:&[CMom],v1:Option<&[CMom]>,out:&mut[C64])->ninjars::Result<()>{self.native.mu_expansion(cut,lanes,v0,v1,out)}
 fn t2_expansion(&mut self,cut:&[usize],lanes:&[usize],a0:&[CMom],a1:&[CMom],e3:&[CMom],e4:&[CMom],p:&[[C64;3]],k:usize,out:&mut[C64])->ninjars::Result<()>{self.native.t2_expansion(cut,lanes,a0,a1,e3,e4,p,k,out)}
 fn t3_expansion(&mut self,cut:&[usize],lanes:&[usize],a:&[CMom],e3:&[CMom],e4:&[CMom],p:&[C64],k:usize,out:&mut[C64])->ninjars::Result<()>{assert_eq!(k,2);assert!(lanes.iter().all(|&lane|lane==0));if self.mode==0{self.native.t3_expansion(cut,lanes,a,e3,e4,p,k,out)?;}else{for (row,out) in out.chunks_exact_mut(4).enumerate(){out.copy_from_slice(&t3::evaluate(self.rank,&self.coeff,&a[row],&e3[row],&e4[row],p[row],self.mode));}}self.records.push(json!({"cut":cut,"values":out.iter().copied().map(c).collect::<Vec<_>>()}));Ok(())}
}
fn run(v:&Value,records:&[Value],mode:usize)->Value{
 let n=v["n"].as_u64().unwrap()as usize;let rank=v["rank"].as_u64().unwrap()as usize;let coeff=pairs(&v["coefficients"]);let masses=pairs(&v["masses_squared"]);let s=numbers(&v["s_matrix_column_major"]);let shifts:Vec<[f64;4]>=numbers(&v["momenta_column_major"]).chunks_exact(4).map(|x|x.try_into().unwrap()).collect();let mu=[v["scale_squared"].as_f64().unwrap()];
 let mut a=FastAmplitude::new(n,rank,FastOptions::default()).unwrap();let mut native=TensorNumerator::new(rank,1);native.coefficients_mut().copy_from_slice(&coeff);let mut t=Tensor{native,coeff,rank,mode,records:vec![]};let mut masters=Replay{records,row:0};let mut out=[FastResult::default()];let mut status=[FastStatus::NotEvaluated];a.evaluate(&FastInput{lanes:1,shifts:&shifts,masses_squared:&masses,scales_squared:&mu,s_matrix:Some(&s)},&mut t,&mut masters,&mut out,&mut status).unwrap();assert_eq!(status[0],FastStatus::Success);assert_eq!(masters.row,records.len());let laurent=json!(out[0].laurent.iter().copied().map(c).collect::<Vec<_>>());if mode==0{assert_eq!(laurent,v["ninjars"]);}json!({"mode":mode,"status":"Success","laurent":laurent,"rational":c(out[0].rational),"callbacks":t.records})
}
fn main(){let args:Vec<_>=env::args().collect();let inputs:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();let masters:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[2]).unwrap()).unwrap();let oracle:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[3]).unwrap()).unwrap();let mut results=vec![];
 for ((v,m),reference) in inputs.iter().zip(masters).zip(oracle){assert_eq!(v["loop"],m["loop"]);assert_eq!(v["direction"],m["direction"]);let records=m["native"]["masters"].as_array().unwrap();let full:Vec<_>=(0..=6).map(|mode|run(v,records,mode)).collect();
 let mut callbacks=vec![];for r in reference["results"][0]["callbacks"].as_array().unwrap().iter().filter(|r|r["kind"]=="t3"){
  let i=&r["inputs"];let rank=v["rank"].as_u64().unwrap()as usize;let coeff=pairs(&v["coefficients"]);let a=momenta(&i["a"]);let e3=momenta(&i["e3"]);let e4=momenta(&i["e4"]);let p=pairs(&i["param"]);let values:Vec<_>=(1..=6).map(|mode|json!({"mode":mode,"values":(0..a.len()).flat_map(|row|t3::evaluate(rank,&coeff,&a[row],&e3[row],&e4[row],p[row],mode)).map(c).collect::<Vec<_>>()})).collect();callbacks.push(json!({"cut":r["cut"],"native":r["native"],"oracle":r["high_rounded"],"variants":values}));
 }results.push(json!({"loop":v["loop"],"direction":v["direction"],"full":full,"callbacks":callbacks}));}
 fs::write(&args[4],serde_json::to_string_pretty(&results).unwrap()).unwrap();}
