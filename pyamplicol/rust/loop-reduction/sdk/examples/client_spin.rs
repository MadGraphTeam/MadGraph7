// SPDX-License-Identifier: 0BSD
#[path = "../../../../src/pyamplicol/_sdk/rust/rusticol.rs"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::{Complex64, PointStatus, Runtime};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let path=std::env::args().nth(1).expect("saved joint-source artifact");
    let mut runtime=Runtime::open(path,None,None,Some("model"))?;
    assert_eq!(runtime.spin_correlation_classes_json()?,"[[3],[4],[3,4]]");
    let p=[500.,0.,0.,500.,500.,0.,0.,-500.,500.,300.,0.,400.,500.,-300.,0.,-400.];
    let legs=[3,4];
    let vector=vec![Complex64::new(0.,0.),Complex64::new(0.8,0.),Complex64::new(0.,1.),Complex64::new(-0.6,0.)];
    let mut sources=[vector.clone(),vector].concat();
    let baseline=runtime.evaluate_correlated_with_sources(&p,1,Some(&[u64::MAX]),&[1e6],&legs,&sources,None,None)?;
    for (i,v) in sources.iter_mut().enumerate() {
        let (r,s)=if i<4 {(2.,-3.)}else{(1.,2.)};
        *v=Complex64::new(r*v.re-s*v.im,r*v.im+s*v.re);
    }
    let scaled=runtime.evaluate_correlated_with_sources(&p,1,None,&[1e6],&legs,&sources,None,None)?;
    assert_eq!(baseline.statuses,[PointStatus::Success]);
    assert_eq!(baseline.successful_point_ids,[u64::MAX]);
    for (a,b) in baseline.virtual_values.iter().zip(&scaled.virtual_values) {
        let scale=1_f64.max(b.re.abs()).max(b.im.abs());
        assert!((b.re-65.*a.re).abs()<2e-8*scale && (b.im-65.*a.im).abs()<2e-8*scale);
    }
    drop(sources); runtime.clear()?; drop(runtime);
    assert_eq!(baseline.spin_legs_json,"[3,4]");
    assert_eq!(baseline.histories_json,"[\"identity\",\"ordered\",\"swapped\"]");
    println!("packaged Rust literal joint sources and owned labels passed");
    Ok(())
}
