//! One real O3 expression: compression must not change the caller's d8.
#![cfg(target_arch = "aarch64")]
use anyhow::Result;
use std::arch::asm;
use symjit::{CompiledPlaneFunc, Composer, Config, PlaneDescriptor, Slot, Translator};

const SENTINEL: u64 = 0x3ff0000000000001;

#[inline(never)]
unsafe fn call_observing_d8(
    kernel: CompiledPlaneFunc<f64>,
    planes: *const PlaneDescriptor<f64>,
    params: *const f64,
) -> (i32, u64) {
    let status: usize;
    let after: u64;
    // Explicit clobbers preserve this wrapper's own caller, even if JIT code
    // fails to preserve its caller. AAPCS64 requires d8's low 64 bits unchanged.
    asm!(
        "fmov d8, x21", "blr x16", "fmov x20, d8",
        in("x16") kernel, in("x21") SENTINEL, lateout("x20") after,
        inlateout("x0") 0usize => status,
        in("x1") planes, in("x2") 0usize, in("x3") params,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        clobber_abi("C"),
    );
    (status as i32, after)
}

#[test]
fn compressed_real_nine_argument_call_preserves_d8() -> Result<()> {
    let mut observed = Vec::new();
    for compress in [false, true] {
        let mut config = Config::default();
        config.set_symbolica(true);
        config.set_compress(compress);
        config.set_opt_level(3);
        config.set_simd(false);
        config.set_threads(false);
        config.set_direct_arena(true);
        config.set_direct_arena_identity_output(true);
        let mut t = Translator::new(config);
        t.set_num_params(27);
        // Three copies of one nine-input expression induce a shared helper.
        // With every input equal to one, each exact output is five.
        for group in 0..3 {
            let mut value = Slot::Param(9 * group);
            for j in 1..9 {
                let dst = Slot::Temp(9 * group + j);
                let args = [value, Slot::Param(9 * group + j)];
                if j % 2 == 1 { t.append_mul(&dst, &args, 2)?; }
                else { t.append_add(&dst, &args, 2)?; }
                value = dst;
            }
            t.append_assign(&Slot::Out(group), &value)?;
        }
        let application = t.compile()?;
        if compress { eprintln!("Compressed MIR: {:?}", application.bytecode.mir); }
        let app = application.seal()?;
        let mut values = [[1.0]; 30];
        for row in &mut values[27..] { row[0] = f64::NAN; }
        let planes: Vec<_> = values.iter_mut().map(|row| unsafe {
            PlaneDescriptor::from_raw_parts(row.as_mut_ptr(), 1)
        }).collect();
        let (status, after) = unsafe {
            call_observing_d8(app.scalar_plane_kernel().unwrap(), planes.as_ptr(), app.params.as_ptr())
        };
        assert_eq!(status, 0);
        assert_eq!(&values[27..], &[[5.0], [5.0], [5.0]], "numerical outputs");
        eprintln!("compress={compress}: outputs=5,5,5; d8 before={SENTINEL:016x}, after={after:016x}");
        observed.push(after);
    }
    anyhow::ensure!(observed[0] == SENTINEL, "uncompressed control corrupted d8");
    anyhow::ensure!(observed[1] == SENTINEL, "compressed kernel corrupted d8");
    Ok(())
}
