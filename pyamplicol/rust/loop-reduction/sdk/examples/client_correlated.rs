// SPDX-License-Identifier: 0BSD
#[path = "../../../../src/pyamplicol/_sdk/rust/rusticol.rs"]
#[allow(dead_code)]
mod rusticol;
use rusticol::virtual_amplitudes::{PointStatus, Runtime};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let path=std::env::args().nth(1).expect("directed source artifact");
    let mut runtime=Runtime::open(path,None,None,Some("model"))?;
    let p=[500.,0.,0.,500.,500.,0.,0.,-500.,500.,300.,0.,400.,500.,-300.,0.,-400.];
    let result=runtime.evaluate_correlated(&p,1,Some(&[u64::MAX]),&[1e6],None,None)?;
    assert_eq!(result.shape.axes.histories,4);
    assert_eq!(result.statuses,[PointStatus::Success]);
    assert_eq!(result.successful_point_ids,[u64::MAX]);
    assert!(result.virtual_values[6].im.abs()>1e-8);
    for e in 0..3 {
        assert!((result.virtual_values[6+e].re-result.virtual_values[9+e].re).abs()<1e-8);
        assert!((result.virtual_values[6+e].im+result.virtual_values[9+e].im).abs()<1e-8);
    }
    drop(runtime);
    assert_eq!(result.histories_json,"[\"identity\",\"casimir\",\"ordered\",\"swapped\"]");
    println!("packaged safe Rust owned complex histories passed");
    Ok(())
}
