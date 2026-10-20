//! Diagnostic: round genuine Float256 numerator callbacks back to binary64,
//! retaining the unmodified fast engine's cut geometry and reduction algebra.
mod tensor;
use ninjars::{Complex,Float,FourMomentum,Numerator,NumeratorRequest,ExpansionKind,ExpansionRequest};
use ninjars::fast::{C64,CMom,FastAmplitude,FastInput,FastOptions,FastResult,FastStatus,FastNumerator,TensorNumerator,GenericNumerator};
use ninjars_symbolica::OneLoopNativeMasters;
use symbolica::domains::float::RealLike;
use serde_json::{json,Value};
use std::{env,fs};
type Z=Complex<Float>;
fn c(z:C64)->Value{json!([z.re,z.im])}
fn z(v:Complex<f64>)->Z{Complex::new(Float::with_val(256,v.re),Float::with_val(256,v.im))}
fn mom(v:&FourMomentum<Complex<f64>>)->FourMomentum<Z>{let q=v.components();FourMomentum::from_args(z(*q[0]),z(*q[1]),z(*q[2]),z(*q[3]))}
fn numbers(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|x|C64::new(x[0].as_f64().unwrap(),x[1].as_f64().unwrap())).collect()}
struct Widened{inner:tensor::TensorGenericNumerator<Float>}
impl Numerator<Complex<f64>> for Widened{
 fn rank(&self)->usize{self.inner.rank()}
 fn evaluate_batch(&mut self,r:&[NumeratorRequest<'_,Complex<f64>>],out:&mut[Complex<f64>])->ninjars::Result<()>{
  let q:Vec<_>=r.iter().map(|r|mom(r.q)).collect();let mu:Vec<_>=r.iter().map(|r|z(*r.mu_squared)).collect();
  let req:Vec<_>=r.iter().enumerate().map(|(i,r)|NumeratorRequest{id:r.id,q:&q[i],mu_squared:&mu[i],partition:r.partition}).collect();
  let mut values=vec![z(Complex::new(0.,0.));out.len()];self.inner.evaluate_batch(&req,&mut values)?;
  for (o,v) in out.iter_mut().zip(values){*o=Complex::new(v.re.to_f64(),v.im.to_f64());}Ok(())
 }
 fn expansion_batch(&mut self,k:&ExpansionKind,r:&[ExpansionRequest<'_,Complex<f64>>],out:&mut[Complex<f64>])->ninjars::Result<()>{
  let vectors:Vec<Vec<_>>=r.iter().map(|r|r.vectors.iter().map(mom).collect()).collect();
  let params:Vec<Vec<_>>=r.iter().map(|r|r.parameters.iter().copied().map(z).collect()).collect();
  let req:Vec<_>=r.iter().enumerate().map(|(i,r)|ExpansionRequest{id:r.id,partition:r.partition,vectors:&vectors[i],parameters:&params[i]}).collect();
  let mut values=vec![z(Complex::new(0.,0.));out.len()];self.inner.expansion_batch(k,&req,&mut values)?;
  for (o,v) in out.iter_mut().zip(values){*o=Complex::new(v.re.to_f64(),v.im.to_f64());}Ok(())
 }
}
struct Hybrid{native:TensorNumerator,high:GenericNumerator<Widened>,mode:String,records:Vec<Value>}
impl Hybrid{
 fn finish(&mut self,kind:&str,cut:&[usize],inputs:Value,original:&[C64],high:&[C64],out:&mut[C64]){
  self.records.push(json!({"kind":kind,"cut":cut,"inputs":inputs,
   "native":original.iter().copied().map(c).collect::<Vec<_>>(),"high_rounded":high.iter().copied().map(c).collect::<Vec<_>>()}));
  out.copy_from_slice(if self.mode=="all" || self.mode==kind {high}else{original});
 }
}
fn vs(v:&[CMom])->Value{json!(v.iter().map(|p|p.iter().copied().map(c).collect::<Vec<_>>()).collect::<Vec<_>>())}
fn zs(v:&[C64])->Value{json!(v.iter().copied().map(c).collect::<Vec<_>>())}
impl FastNumerator for Hybrid{
 fn rank(&self)->usize{self.native.rank()}
 fn evaluate(&mut self,cut:&[usize],lanes:&[usize],q:&[CMom],mu:&[C64],out:&mut[C64])->ninjars::Result<()>{
  let mut a=vec![C64::ZERO;out.len()];let mut b=a.clone();self.native.evaluate(cut,lanes,q,mu,&mut a)?;self.high.evaluate(cut,lanes,q,mu,&mut b)?;
  self.finish("evaluate",cut,json!({"lanes":lanes,"q":vs(q),"mu":zs(mu)}),&a,&b,out);Ok(())
 }
 fn mu_expansion(&mut self,cut:&[usize],lanes:&[usize],v0:&[CMom],v1:Option<&[CMom]>,out:&mut[C64])->ninjars::Result<()>{
  let mut a=vec![C64::ZERO;out.len()];let mut b=a.clone();self.native.mu_expansion(cut,lanes,v0,v1,&mut a)?;self.high.mu_expansion(cut,lanes,v0,v1,&mut b)?;
  self.finish("mu",cut,json!({"lanes":lanes,"v0":vs(v0),"v1":v1.map(vs)}),&a,&b,out);Ok(())
 }
 fn t3_expansion(&mut self,cut:&[usize],lanes:&[usize],a:&[CMom],e3:&[CMom],e4:&[CMom],param:&[C64],mindeg:usize,out:&mut[C64])->ninjars::Result<()>{
  let mut x=vec![C64::ZERO;out.len()];let mut y=x.clone();self.native.t3_expansion(cut,lanes,a,e3,e4,param,mindeg,&mut x)?;self.high.t3_expansion(cut,lanes,a,e3,e4,param,mindeg,&mut y)?;
  self.finish("t3",cut,json!({"lanes":lanes,"a":vs(a),"e3":vs(e3),"e4":vs(e4),"param":zs(param),"mindeg":mindeg}),&x,&y,out);Ok(())
 }
 fn t2_expansion(&mut self,cut:&[usize],lanes:&[usize],a0:&[CMom],a1:&[CMom],e3:&[CMom],e4:&[CMom],param:&[[C64;3]],mindeg:usize,out:&mut[C64])->ninjars::Result<()>{
  let mut x=vec![C64::ZERO;out.len()];let mut y=x.clone();self.native.t2_expansion(cut,lanes,a0,a1,e3,e4,param,mindeg,&mut x)?;self.high.t2_expansion(cut,lanes,a0,a1,e3,e4,param,mindeg,&mut y)?;
  self.finish("t2",cut,json!({"lanes":lanes,"a0":vs(a0),"a1":vs(a1),"e3":vs(e3),"e4":vs(e4),"param":param.iter().map(|p|zs(p)).collect::<Vec<_>>(),"mindeg":mindeg}),&x,&y,out);Ok(())
 }
}
fn run(v:&Value,mode:&str)->Value{
 let n=v["n"].as_u64().unwrap() as usize;let rank=v["rank"].as_u64().unwrap() as usize;
 let coeff=pairs(&v["coefficients"]);let masses=pairs(&v["masses_squared"]);let s=numbers(&v["s_matrix_column_major"]);
 let shifts:Vec<[f64;4]>=numbers(&v["momenta_column_major"]).chunks_exact(4).map(|x|x.try_into().unwrap()).collect();let mu=[v["scale_squared"].as_f64().unwrap()];
 let mut a=FastAmplitude::new(n,rank,FastOptions::default()).unwrap();let mut native=TensorNumerator::new(rank,1);native.coefficients_mut().copy_from_slice(&coeff);
 let mut high=tensor::TensorGenericNumerator::new(rank,0,&Float::with_val(256,0.)).unwrap();high.set_coefficients(coeff.iter().map(|c|z(Complex::new(c.re,c.im))).collect(),1).unwrap();
 let mut t=Hybrid{native,high:GenericNumerator::new(Widened{inner:high}),mode:mode.into(),records:vec![]};
 let mut masters=OneLoopNativeMasters::new().unwrap();let mut out=[FastResult::default()];let mut status=[FastStatus::NotEvaluated];
 a.evaluate(&FastInput{lanes:1,shifts:&shifts,masses_squared:&masses,scales_squared:&mu,s_matrix:Some(&s)},&mut t,&mut masters,&mut out,&mut status).unwrap();
 assert_eq!(status[0],FastStatus::Success);let laurent=json!(out[0].laurent.iter().copied().map(c).collect::<Vec<_>>());if mode=="native"{assert_eq!(laurent,v["ninjars"]);}
 json!({"mode":mode,"status":format!("{:?}",status[0]),"laurent":laurent,"rational":c(out[0].rational),"callbacks":t.records})
}
fn main(){let args:Vec<_>=env::args().collect();let inputs:Vec<Value>=serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();let mut rows=vec![];
 for v in inputs{eprintln!("loop {} direction {}",v["loop"],v["direction"]);let results:Vec<_>=["native","all","evaluate","t3","t2"].iter().map(|m|run(&v,m)).collect();rows.push(json!({"loop":v["loop"],"direction":v["direction"],"results":results}));fs::write(&args[2],serde_json::to_string_pretty(&rows).unwrap()).unwrap();}}
