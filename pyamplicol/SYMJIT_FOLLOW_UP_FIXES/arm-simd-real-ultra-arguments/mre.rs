use anyhow::{ensure, Result};
use symjit::{Compiled, Composer, Config, PlaneDescriptor, Slot, Translator};

fn main() -> Result<()> {
    ensure!(cfg!(target_arch = "aarch64"), "this MRE requires AArch64");
    let compress = !std::env::args().any(|arg| arg == "--no-compress");
    let mut config = Config::default();
    config.set_symbolica(true);
    config.set_opt_level(3);
    config.set_compress(compress);
    config.set_simd(true);
    config.set_threads(false);
    config.set_direct_arena(true);
    config.set_direct_arena_identity_output(true);
    let mut translator = Translator::new(config);
    translator.set_num_params(9);
    for i in 0..9 {
        translator.append_mul(&Slot::Temp(i), &[Slot::Param(i), Slot::Param(i)], 2)?;
    }
    for group in 0..3 {
        let start = group * 3;
        translator.append_mul(&Slot::Temp(9 + group),
            &[Slot::Temp(start), Slot::Temp(start + 1)], 2)?;
        translator.append_add(&Slot::Out(group),
            &[Slot::Temp(9 + group), Slot::Temp(start + 2)], 2)?;
    }
    // The fourth output reuses each square. This ordinary common-subexpression
    // use keeps the squares on the stack and naturally selects the ultra call.
    translator.append_add(&Slot::Out(3), &(0..9).map(Slot::Temp).collect::<Vec<_>>(), 9)?;
    let mut application = translator.compile()?;
    if compress {
        ensure!(format!("{:?}", application.bytecode.mir).contains("_ultra"),
            "expected the compiler to select its ultra path");
    }
    application.prepare_simd();
    let app = application.seal()?;
    let machine = app.compiled_simd.as_ref().expect("AArch64 SIMD kernel");
    let lanes = machine.count_lanes();
    let mut values = vec![vec![f64::NAN; lanes]; 13];
    for p in 0..9 {
        for lane in 0..lanes { values[p][lane] = (p + lane + 1) as f64; }
    }
    let descriptors: Vec<_> = values.iter_mut().map(|row| unsafe {
        PlaneDescriptor::from_raw_parts(row.as_mut_ptr(), lanes)
    }).collect();
    let status = unsafe {
        app.simd_plane_kernel().unwrap()(std::ptr::null(), descriptors.as_ptr(), 0, app.params.as_ptr())
    };
    println!("compress={compress}, status={status}, first output={} (expected 13)", values[9][0]);
    ensure!(status == 0, "kernel returned failure");
    for group in 0..3 {
        for lane in 0..lanes {
            let x = (group * 3 + lane + 1) as f64;
            let expected = x.powi(2) * (x + 1.0).powi(2) + (x + 2.0).powi(2);
            ensure!(values[9 + group][lane] == expected,
                "group={group}, lane={lane}: {} != {expected}", values[9 + group][lane]);
        }
    }
    for lane in 0..lanes {
        let expected: f64 = (0..9).map(|p| ((p + lane + 1) as f64).powi(2)).sum();
        ensure!(values[12][lane] == expected, "sum-of-squares control failed");
    }
    Ok(())
}
