//! Identity evaluation must preserve d8, even with fast-complex lowering.
#![cfg(target_arch = "aarch64")]
use anyhow::Result;
use std::arch::asm;
use symjit::{CompiledPlaneFunc, Composer, Config, PlaneDescriptor, Slot, Translator};

const D8: u64 = 0x3ff0000000000001;
const D9: u64 = 0x4000000000000002;

#[inline(never)]
unsafe fn observe(kernel: CompiledPlaneFunc<f64>, planes: *const PlaneDescriptor<f64>, params: *const f64) -> (i32, u64) {
    let status: usize;
    let after: u64;
    // Declare clobbers so Rust preserves this wrapper's own caller correctly.
    asm!(
        "fmov d8, x21", "fmov d9, x22", "blr x16", "fmov x20, d8",
        in("x16") kernel, in("x21") D8, in("x22") D9, lateout("x20") after,
        inlateout("x0") 0usize => status,
        in("x1") planes, in("x2") 0usize, in("x3") params,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        clobber_abi("C"),
    );
    (status as i32, after)
}

#[test]
fn fast_complex_register_save_slots_do_not_overlap() -> Result<()> {
    let mut observed = Vec::new();
    for compress in [false, true] {
        let mut config = Config::default();
        config.set_symbolica(true);
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_fast_complex(true);
        config.set_compress(compress);
        config.set_simd(false);
        config.set_threads(false);
        config.set_direct_arena(true);
        config.set_direct_arena_identity_output(true);
        let mut t = Translator::new(config);
        t.set_num_params(3);
        for i in 0..3 { t.append_assign(&Slot::Out(i), &Slot::Param(i))?; }
        let application = t.compile()?;
        // These identity assignments need no compressed helper or arithmetic.
        if compress { eprintln!("Identity MIR: {:?}", application.bytecode.mir); }
        let app = application.seal()?;
        let mut values = [[f64::NAN]; 12];
        for i in 0..6 { values[i][0] = if i % 2 == 0 { 1.0 } else { 0.0 }; }
        let planes: Vec<_> = values.iter_mut().map(|row| unsafe {
            PlaneDescriptor::from_raw_parts(row.as_mut_ptr(), 1)
        }).collect();
        let (status, after) = unsafe { observe(app.scalar_plane_kernel().unwrap(), planes.as_ptr(), app.params.as_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(&values[6..], &[[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]);
        eprintln!("compress={compress}: identities correct; d8 before={D8:016x}, after={after:016x}; d9 sentinel={D9:016x}");
        observed.push(after);
    }
    anyhow::ensure!(observed[0] == D8, "compression-disabled control corrupted d8");
    anyhow::ensure!(observed[1] == D8, "d8 spill was overwritten by d9 spill");
    Ok(())
}
