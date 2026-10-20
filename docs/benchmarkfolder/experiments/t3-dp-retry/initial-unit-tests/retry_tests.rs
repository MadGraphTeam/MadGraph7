use super::*;
use serde_json::Value;
use std::{fs,path::PathBuf};
fn doc()->PathBuf{PathBuf::from(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap()
    .join("docs/benchmarkfolder/experiments")}
fn pairs(v:&Value)->Vec<C64>{v.as_array().unwrap().iter().map(|v|C64::new(v[0].as_f64().unwrap(),v[1].as_f64().unwrap())).collect()}
fn numbers(v:&Value)->Vec<f64>{v.as_array().unwrap().iter().map(|v|v.as_f64().unwrap()).collect()}
fn bits(v:&[C64])->Vec<[u64;2]>{v.iter().map(|z|[z.re.to_bits(),z.im.to_bits()]).collect()}
fn create()->*mut Context{let mut c=std::ptr::null_mut();assert_eq!(unsafe{ninjars_context_create(2,&mut c)},SUCCESS);c}
fn call(ctx:*mut Context,comp:bool,v:&Value)->(i32,[C64;3]){
    let n=v["n"].as_i64().unwrap()as i32;let rank=v["rank"].as_i64().unwrap()as i32;
    let c=pairs(&v["coefficients"]);let s=numbers(&v["s_matrix_column_major"]);
    let p=numbers(&v["momenta_column_major"]);let m=pairs(&v["masses_squared"]);
    let mut out=[C64::new(987.,654.);3];let mut rational=C64::new(987.,654.);
    let f=if comp{ninjars_evaluate_dp_compensated}else{ninjars_evaluate_dp};
    let status=unsafe{f(ctx,n,rank,c.len(),c.as_ptr(),s.as_ptr(),s.len(),p.as_ptr(),p.len(),
        m.as_ptr(),m.len(),v["scale_squared"].as_f64().unwrap(),
        out.as_mut_ptr(),3,&mut rational,1)};
    (status,out)
}
#[test]
fn retry_native_path_is_exact_and_compensated_state_is_lazy(){
    let fixtures:Vec<Value>=serde_json::from_slice(&fs::read(doc().join(
        "qcd3-point67-dp/oracle-inputs.json")).unwrap()).unwrap();
    let corrected:Vec<Value>=serde_json::from_slice(&fs::read(doc().join(
        "qcd3-point67-t3/guarded-process-reductions.json")).unwrap()).unwrap();
    let ctx=create();
    for v in &fixtures{
        let(status,out)=call(ctx,false,v);assert_eq!(status,SUCCESS);
        assert_eq!(bits(&out),bits(&pairs(&v["ninjars"])));
    }
    unsafe{
        assert!((*ctx).compensated_dp.lock().unwrap().is_none());
        assert!((*ctx).mp.lock().unwrap().is_none());
    }
    for v in &fixtures{
        let reference=corrected.iter().find(|r|r["loop"]==v["loop"]&&r["direction"]==v["direction"]).unwrap();
        let(status,out)=call(ctx,true,v);assert_eq!(status,SUCCESS);
        assert_eq!(bits(&out),bits(&pairs(&reference["ninjars"])));
        // A subsequent ordinary call must not inherit any compensated mode.
        let(status,out)=call(ctx,false,v);assert_eq!(status,SUCCESS);
        assert_eq!(bits(&out),bits(&pairs(&v["ninjars"])));
    }
    unsafe{
        assert!((*ctx).compensated_dp.lock().unwrap().is_some());
        assert!((*ctx).mp.lock().unwrap().is_none());
        assert_eq!(ninjars_context_destroy(ctx),SUCCESS);
    }
}
#[test]
fn retry_rebinding_and_failure_preserve_explicit_call_semantics(){
    let ctx=create();
    for (coeff,mass,scale) in [(1.,2.,1.),(2.5,3.,4.),(-0.75,2.,1.)]{
        let v=serde_json::json!({"n":1,"rank":0,"coefficients":[[coeff,0.]],
            "s_matrix_column_major":[0.],"momenta_column_major":[0.,0.,0.,0.],
            "masses_squared":[[mass,0.]],"scale_squared":scale});
        let fresh=create();let a=call(ctx,true,&v);let b=call(fresh,true,&v);let c=call(ctx,false,&v);
        assert_eq!(a.0,SUCCESS);assert_eq!(b.0,SUCCESS);assert_eq!(c.0,SUCCESS);
        assert_eq!(bits(&a.1),bits(&b.1));assert_eq!(bits(&a.1),bits(&c.1));
        unsafe{ninjars_context_destroy(fresh);}
    }
    let c=[C64::real(f64::NAN)];let s=[0.];let p=[0.;4];let m=[C64::real(2.)];
    let sentinel=C64::new(987.,654.);let mut out=[sentinel;3];let mut r=sentinel;
    let code=unsafe{ninjars_evaluate_dp_compensated(ctx,1,0,1,c.as_ptr(),s.as_ptr(),1,
        p.as_ptr(),4,m.as_ptr(),1,1.,out.as_mut_ptr(),3,&mut r,1)};
    assert_eq!(code,ERROR);assert_eq!(bits(&out),bits(&[sentinel;3]));assert_eq!(bits(&[r]),bits(&[sentinel]));
    // Invalid output length fails before attempting to read the null inputs.
    let code=unsafe{ninjars_evaluate_dp_compensated(ctx,1,0,1,std::ptr::null(),
        std::ptr::null(),1,std::ptr::null(),4,std::ptr::null(),1,1.,
        out.as_mut_ptr(),2,&mut r,1)};
    assert_eq!(code,ERROR);assert_eq!(bits(&out),bits(&[sentinel;3]));
    unsafe{assert_eq!(ninjars_context_destroy(ctx),SUCCESS);}
}
