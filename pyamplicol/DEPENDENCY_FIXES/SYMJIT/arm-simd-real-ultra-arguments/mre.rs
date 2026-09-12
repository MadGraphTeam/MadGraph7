//! SIMD's packed-stack argument path must load, not overwrite, caller values.
use anyhow::Result;
use symjit::{Compiled, Composer, Config, PlaneDescriptor, Slot, Translator};

#[test]
fn compressed_real_stack_arguments() -> Result<()> {
    let mut config = Config::default();
    config.set_symbolica(true);
    config.set_opt_level(3);
    config.set_compress(true);
    config.set_simd(true);
    config.set_threads(false);
    config.set_direct_arena(true);
    config.set_direct_arena_identity_output(true);
    let mut t = Translator::new(config);
    t.set_num_params(9);
    for i in 0..9 {
        t.append_mul(&Slot::Temp(i), &[Slot::Param(i), Slot::Param(i)], 2)?;
    }
    for group in 0..3 {
        let start = group * 3;
        t.append_mul(&Slot::Temp(9 + group), &[Slot::Temp(start), Slot::Temp(start + 1)], 2)?;
        t.append_add(&Slot::Out(group), &[Slot::Temp(9 + group), Slot::Temp(start + 2)], 2)?;
    }
    // Keep each square materialized so all compressed-call arguments are stack
    // slots, enabling the O3 packed-offset ("ultra") call entry.
    t.append_add(&Slot::Out(3), &(0..9).map(Slot::Temp).collect::<Vec<_>>(), 9)?;
    let mut application = t.compile()?;
    assert!(format!("{:?}", application.bytecode.mir).contains("_ultra"));
    application.prepare_simd();
    let app = application.seal()?;
    let Some(machine) = app.compiled_simd.as_ref() else { return Ok(()); };
    let lanes = machine.count_lanes();
    let mut values = vec![vec![0.0; lanes]; 13];
    for p in 0..9 {
        for lane in 0..lanes { values[p][lane] = (p + lane + 1) as f64; }
    }
    let descriptors: Vec<_> = values.iter_mut().map(|row| unsafe {
        PlaneDescriptor::from_raw_parts(row.as_mut_ptr(), lanes)
    }).collect();
    let status = unsafe { app.simd_plane_kernel().unwrap()(std::ptr::null(), descriptors.as_ptr(), 0, app.params.as_ptr()) };
    eprintln!("raw SIMD status={status} (testing values independently)");
    for group in 0..3 {
        for lane in 0..lanes {
            let x = (group * 3 + lane + 1) as f64;
            assert_eq!(values[9 + group][lane], x.powi(2) * (x + 1.0).powi(2) + (x + 2.0).powi(2));
        }
    }
    Ok(())
}
