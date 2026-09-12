//! Normal compressed completion must not request scalar SIMD replay.
use anyhow::Result;
use symjit::{Compiled, Composer, Config, PlaneDescriptor, Slot, Translator};

#[test]
fn compressed_simd_returns_success() -> Result<()> {
    for complex in [false, true] {
        for opt in [2, 3] {
            let mut config = Config::default();
            config.set_symbolica(true);
            config.set_opt_level(opt);
            config.set_complex(complex);
            config.set_fast_complex(false);
            config.set_compress(true);
            config.set_simd(true);
            config.set_threads(false);
            config.set_direct_arena(true);
            config.set_direct_arena_identity_output(true);
            let mut t = Translator::new(config);
            t.set_num_params(9);
            for group in 0..3 {
                let start = group * 3;
                t.append_mul(&Slot::Temp(group), &[Slot::Param(start), Slot::Param(start + 1)], if complex { 0 } else { 2 })?;
                t.append_add(&Slot::Out(group), &[Slot::Temp(group), Slot::Param(start + 2)], if complex { 0 } else { 2 })?;
            }
            let mut application = t.compile()?;
            application.prepare_simd();
            let app = application.seal()?;
            let machine = app.compiled_simd.as_ref().expect("native SIMD kernel required");
            let lanes = machine.count_lanes();
            let width = if complex { 2 } else { 1 };
            let mut values = vec![vec![0.0; lanes]; 12 * width];
            for p in 0..9 {
                for lane in 0..lanes { values[p * width][lane] = (p + lane + 1) as f64; }
            }
            let descriptors: Vec<_> = values.iter_mut().map(|row| unsafe {
                PlaneDescriptor::from_raw_parts(row.as_mut_ptr(), lanes)
            }).collect();
            let status = unsafe { app.simd_plane_kernel().unwrap()(std::ptr::null(), descriptors.as_ptr(), 0, app.params.as_ptr()) };
            assert_eq!(status, 0, "normal completion is not lane divergence; complex={complex}, O{opt}");
            for group in 0..3 {
                for lane in 0..lanes {
                    let x = (group * 3 + lane + 1) as f64;
                    assert_eq!(values[(9 + group) * width][lane], x * (x + 1.0) + x + 2.0);
                    if complex { assert_eq!(values[(9 + group) * width + 1][lane], 0.0); }
                }
            }
        }
    }
    Ok(())
}
