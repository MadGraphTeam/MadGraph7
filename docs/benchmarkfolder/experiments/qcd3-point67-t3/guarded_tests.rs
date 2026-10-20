//! Controls for the isolated experiment; run serially because FENV is mutated.
use super::{guarded::GuardedTensor, tensor::TensorGenericNumerator};
use ninjars::{Complex, Float, FourMomentum, Numerator, NumeratorRequest,
    ExpansionKind, ExpansionRequest};
use ninjars::fast::{C64, CMom, FastNumerator, TensorNumerator, GenericNumerator};
use ninjars::fast::tensor_layout::ncoeff;
use symbolica::domains::float::RealLike;
use serde_json::{json, Value};
use std::{fs, path::PathBuf};
type Z=Complex<Float>;
fn z(v:Complex<f64>)->Z{Complex::new(Float::with_val(256,v.re),Float::with_val(256,v.im))}
fn mom(v:&FourMomentum<Complex<f64>>)->FourMomentum<Z>{let q=v.components();FourMomentum::from_args(z(*q[0]),z(*q[1]),z(*q[2]),z(*q[3]))}
struct Wide { inner: TensorGenericNumerator<Float> }
impl Numerator<Complex<f64>> for Wide {
    fn rank(&self)->usize{self.inner.rank()}
    fn evaluate_batch(&mut self,r:&[NumeratorRequest<'_,Complex<f64>>],out:&mut[Complex<f64>])->ninjars::Result<()> {
        let q:Vec<_>=r.iter().map(|r|mom(r.q)).collect();
        let mu:Vec<_>=r.iter().map(|r|z(*r.mu_squared)).collect();
        let req:Vec<_>=r.iter().enumerate().map(|(i,r)|NumeratorRequest{id:r.id,q:&q[i],mu_squared:&mu[i],partition:r.partition}).collect();
        let mut values=vec![z(Complex::new(0.,0.));out.len()];
        self.inner.evaluate_batch(&req,&mut values)?;
        for(o,v)in out.iter_mut().zip(values){*o=Complex::new(v.re.to_f64(),v.im.to_f64());}Ok(())
    }
    fn expansion_batch(&mut self,k:&ExpansionKind,r:&[ExpansionRequest<'_,Complex<f64>>],out:&mut[Complex<f64>])->ninjars::Result<()>{
        let vectors:Vec<Vec<_>>=r.iter().map(|r|r.vectors.iter().map(mom).collect()).collect();
        let params:Vec<Vec<_>>=r.iter().map(|r|r.parameters.iter().copied().map(z).collect()).collect();
        let req:Vec<_>=r.iter().enumerate().map(|(i,r)|ExpansionRequest{id:r.id,partition:r.partition,vectors:&vectors[i],parameters:&params[i]}).collect();
        let mut values=vec![z(Complex::new(0.,0.));out.len()];
        self.inner.expansion_batch(k,&req,&mut values)?;
        for(o,v)in out.iter_mut().zip(values){*o=Complex::new(v.re.to_f64(),v.im.to_f64());}Ok(())
    }
}
fn wide(rank:usize,c:&[C64])->GenericNumerator<Wide>{
    let mut inner=TensorGenericNumerator::new(rank,0,&Float::with_val(256,0.)).unwrap();
    inner.set_coefficients(c.iter().map(|c|z(Complex::new(c.re,c.im))).collect(),1).unwrap();
    GenericNumerator::new(Wide{inner})
}
fn bits(v:&[C64])->Vec<[u64;2]>{v.iter().map(|z|[z.re.to_bits(),z.im.to_bits()]).collect()}
fn len(k:usize)->usize{(0..=k).map(|j|1+j/2).sum()}
fn expand(t:&mut impl FastNumerator,k:usize,a:CMom,e:CMom,f:CMom,p:C64)->Vec<C64>{
    let mut out=vec![C64::ZERO;len(k)];
    if k==5{t.t3_expansion_x2(&[0,1,2],&[0],&[a],&[e],&[f],&[p],&mut out).unwrap();}
    else{t.t3_expansion(&[0,1,2],&[0],&[a],&[e],&[f],&[p],k,&mut out).unwrap();}
    out
}
fn root()->PathBuf{PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf()}
fn doc()->PathBuf{root().join("docs/benchmarkfolder/experiments/qcd3-point67-t3")}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|v|C64::new(v[0].as_f64().unwrap(),v[1].as_f64().unwrap())).collect()}

#[test]
fn guarded_frozen_callbacks_match_all_480_oracle_components(){
    let inputs:Vec<Value>=serde_json::from_slice(&fs::read(doc().join("../qcd3-point67-dp/oracle-inputs.json")).unwrap()).unwrap();
    let refs:Vec<Value>=serde_json::from_slice(&fs::read(doc().join("../qcd3-point67-masters/expansion-results.json")).unwrap()).unwrap();
    let mut count=0;
    for(v,r)in inputs.iter().zip(refs){
        let rank=v["rank"].as_u64().unwrap()as usize;
        let mut t=GuardedTensor::new(rank,1);t.coefficients_mut().copy_from_slice(&pairs(&v["coefficients"]));
        for cb in r["results"][0]["callbacks"].as_array().unwrap().iter().filter(|r|r["kind"]=="t3"){
            let i=&cb["inputs"];let p=pairs(&i["param"]);let mut out=Vec::new();
            for row in 0..p.len(){
                out.extend(expand(&mut t,2,pairs(&i["a"][row]).try_into().unwrap(),
                    pairs(&i["e3"][row]).try_into().unwrap(),pairs(&i["e4"][row]).try_into().unwrap(),p[row]));
            }
            assert_eq!(bits(&out),bits(&pairs(&cb["high_rounded"])));
            count+=out.len()*2;
        }
        assert_eq!(t.fallback,0);
    }
    assert_eq!(count,480);
}

fn next(seed:&mut u64)->f64{
    *seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (f64::from_bits(0x3ff0_0000_0000_0000|(*seed>>12))-1.)*2.-1.
}
fn random(seed:&mut u64)->C64{C64::new(next(seed),next(seed))}
#[test]
fn guarded_ranks_layouts_dense_sparse_and_cancellation_oracle(){
    let mut seed=0x51f4_99a7_a21c_003du64;
    let mut rows=vec![];
    for rank in 0..=8 {
        // Same retained tensor repeatedly receives completely new coefficients.
        let mut guarded=GuardedTensor::new(rank,1);
        let mut native=TensorNumerator::new(rank,1);
        for pattern in 0..4 {
            let mut coeff:Vec<_>=(0..ncoeff(rank)).map(|_|random(&mut seed)).collect();
            let mut a=std::array::from_fn(|_|random(&mut seed));
            let mut e=std::array::from_fn(|_|random(&mut seed));
            let mut f=std::array::from_fn(|_|random(&mut seed));
            let mut p=random(&mut seed);
            match pattern {
                1=>{for(i,c)in coeff.iter_mut().enumerate(){if i%7!=0{*c=C64::ZERO;}}},
                2=>{ // All-integer dyadic case: independent exact arithmetic oracle.
                    for c in &mut coeff {c.re=(c.re*3.).round();c.im=(c.im*3.).round();}
                    for x in a.iter_mut().chain(&mut e).chain(&mut f){x.re=x.re.round();x.im=x.im.round();}
                    p=C64::new(1.,-1.);
                },
                3 if rank>=2=>{
                    coeff.fill(C64::ZERO);
                    let off=ninjars::fast::tensor_layout::block_offset(rank);
                    coeff[off]=C64::real(1e16);coeff[off+1]=C64::real(1.);
                    coeff[off+2]=C64::real(-1e16);
                    a=[C64::real(1.);4];e=a;f=a;p=C64::real(1.);
                },
                _=>{}
            }
            guarded.coefficients_mut().copy_from_slice(&coeff);
            native.coefficients_mut().copy_from_slice(&coeff);
            let mut high=wide(rank,&coeff);
            for k in 0..=5 {
                let out=expand(&mut guarded,k,a,e,f,p);
                let old=expand(&mut native,k,a,e,f,p);
                let reference=expand(&mut high,k,a,e,f,p);
                if k!=2 || rank<2 { assert_eq!(bits(&out),bits(&old),"fallback rank={rank} k={k}"); }
                let mut exact=0;let mut better=0;let mut worse=0;let mut worst=0f64;
                for((o,n),h)in out.iter().zip(&old).zip(&reference){
                    for(x,y,z)in [(o.re,n.re,h.re),(o.im,n.im,h.im)]{
                        assert!(x.is_finite());
                        exact+=usize::from(x.to_bits()==z.to_bits());
                        better+=usize::from((x-z).abs()<(y-z).abs());
                        worse+=usize::from((x-z).abs()>(y-z).abs());
                        worst=worst.max((x-z).abs());
                    }
                }
                if pattern==2 {assert_eq!(bits(&out),bits(&reference),"integer case rank={rank} k={k}");}
                if pattern==3 && rank>=2 && k==2 {
                    assert_eq!(bits(&out),bits(&reference),"adversarial rank={rank}");
                }
                rows.push(json!({"rank":rank,"pattern":pattern,"mindeg":k,
                    "components":out.len()*2,"exact":exact,"better":better,"worse":worse,
                    "max_absolute_error":worst,
                    "corrected_domain":rank>=2&&k==2}));
            }
        }
    }
    fs::write(doc().join("guarded-oracle-summary.json"),serde_json::to_string_pretty(&json!({
        "seed":"0x51f499a7a21c003d","precision_bits":256,"rows":rows,
        "note":"Only rank2..8,mindeg2 is compensated. All other layouts must remain bitwise native. No general correctly-rounded claim."})).unwrap()).unwrap();
}

unsafe extern "C" {
    fn ninjars_t3_flags()->i32;
    fn ninjars_t3_clear_flags()->i32;
    fn ninjars_t3_bad_flags()->i32;
    fn ninjars_t3_round_upward()->i32;
    fn ninjars_t3_round_nearest()->i32;
    fn ninjars_t3_raise_inexact()->i32;
}
#[test]
fn guarded_unsafe_domains_and_rounding_use_native_flags_and_values(){
    let mut g=GuardedTensor::new(3,1);let mut n=TensorNumerator::new(3,1);
    for (value,vector) in [(f64::INFINITY,2.),(f64::NAN,2.),(f64::MAX,2.),
        (f64::from_bits(1),2.),(1e-250,2.),(1e250,2.),(-0.,2.),
        // Finite normal inputs whose *intermediate* products under/overflow.
        (0.25,1e-200),(0.25,1e200),(1e-250,1e-150)] {
        let v=[C64::real(vector);4];
        let mut coeff=vec![C64::real(0.25);ncoeff(3)];coeff[2]=C64::real(value);
        g.coefficients_mut().copy_from_slice(&coeff);n.coefficients_mut().copy_from_slice(&coeff);
        unsafe{ninjars_t3_clear_flags();ninjars_t3_raise_inexact();}
        let old=expand(&mut n,2,v,v,v,C64::real(2.));let old_flags=unsafe{ninjars_t3_flags()};
        unsafe{ninjars_t3_clear_flags();ninjars_t3_raise_inexact();}
        let actual=expand(&mut g,2,v,v,v,C64::real(2.));let flags=unsafe{ninjars_t3_flags()};
        assert_eq!(bits(&actual),bits(&old),"value {value}");
        assert_eq!(flags,old_flags);
    }
    // Directed rounding is not supported by compensated error-free transforms.
    let v=[C64::real(2.);4];
    g.coefficients_mut().fill(C64::real(0.25));
    n.coefficients_mut().fill(C64::real(0.25));
    unsafe{assert_eq!(ninjars_t3_round_upward(),0);}
    let before=g.corrected;
    let actual=expand(&mut g,2,v,v,v,C64::real(2.));
    let old=expand(&mut n,2,v,v,v,C64::real(2.));
    unsafe{assert_eq!(ninjars_t3_round_nearest(),0);}
    assert_eq!(g.corrected,before);assert_eq!(bits(&actual),bits(&old));
    unsafe{ninjars_t3_clear_flags();}
    expand(&mut g,2,v,v,v,C64::real(2.));
    assert_eq!(unsafe{ninjars_t3_bad_flags()},0);
}

#[test]
fn guarded_preserves_non_t3_and_explicit_mu_abi_rejection(){
    let rank=4;let mut g=GuardedTensor::new(rank,1);let mut n=TensorNumerator::new(rank,1);
    let c:Vec<_>=(0..ncoeff(rank)).map(|i|C64::new(i as f64/17.,-(i as f64)/31.)).collect();
    g.coefficients_mut().copy_from_slice(&c);n.coefficients_mut().copy_from_slice(&c);
    let a=[C64::new(0.25,-0.125);4];let b=[C64::new(0.5,0.25);4];
    for k in 0..=3 {
        let width=[1,3,7,13][k];let mut x=vec![C64::ZERO;width];let mut y=x.clone();
        g.t2_expansion(&[],&[0],&[a],&[b],&[a],&[b],&[[C64::real(1.);3]],k,&mut x).unwrap();
        n.t2_expansion(&[],&[0],&[a],&[b],&[a],&[b],&[[C64::real(1.);3]],k,&mut y).unwrap();
        assert_eq!(bits(&x),bits(&y));
    }
    for width in 1..=3 {
        let mut x=vec![C64::ZERO;width];let mut y=x.clone();
        if width==3 {
            g.mu_expansion_x2(&[],&[0],&[a],&[b],&[C64::real(1.)],&mut x).unwrap();
            n.mu_expansion_x2(&[],&[0],&[a],&[b],&[C64::real(1.)],&mut y).unwrap();
        } else {
            let bs=[b];let opt=(width==2).then_some(bs.as_slice());
            g.mu_expansion(&[],&[0],&[a],opt,&mut x).unwrap();
            n.mu_expansion(&[],&[0],&[a],opt,&mut y).unwrap();
        }
        assert_eq!(bits(&x),bits(&y));
    }
    let mut x=[C64::ZERO];let mut y=x;
    g.evaluate(&[],&[0],&[a],&[C64::real(29.)],&mut x).unwrap();
    n.evaluate(&[],&[0],&[a],&[C64::real(29.)],&mut y).unwrap();
    assert_eq!(bits(&x),bits(&y));
    // Existing public ABI has exactly ncoeff(rank) N(q) coefficients. Extra
    // explicit-mu blocks are rejected, never interpreted as N(q) coefficients.
    assert!(super::dimensions(4,rank as i32,ncoeff(rank)+ncoeff(rank-2)).is_err());
    assert_eq!(g.corrected,0);
}
