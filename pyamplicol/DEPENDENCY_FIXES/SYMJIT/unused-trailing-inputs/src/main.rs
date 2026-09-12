//! A declared but unused trailing input must not receive the first output.
use anyhow::Result;
use symjit::{
    Compiled, CompilerType, Complex, Composer, Config, PlaneDescriptor, Slot, Translator,
};

fn main() -> Result<()> {
    let expect_bug = std::env::args().any(|value| value == "--expect-bug");
    for complex in [false, true] {
        for opt in [0, 2] {
            for all_unused in [false, true] {
                let mut config = Config::new(CompilerType::Native, 0)?;
                config.set_symbolica(true);
                config.set_opt_level(opt);
                config.set_complex(complex);
                config.set_simd(true);
                config.set_direct_arena(true);
                config.set_direct_arena_identity_output(true);
                let mut translator = Translator::new(config);
                translator.set_num_params(2);
                let source = if all_unused {
                    Slot::Const(
                        translator
                            .append_constant(Complex::new(7.0, if complex { 9.0 } else { 0.0 }))?,
                    )
                } else {
                    Slot::Param(0)
                };
                if all_unused {
                    translator.append_assign(&Slot::Out(0), &source)?;
                } else {
                    translator.append_add(
                        &Slot::Out(0),
                        &[source, source],
                        if complex { 0 } else { 2 },
                    )?;
                }
                let mut application = translator.compile()?;
                application.prepare_simd();
                let app = application.seal()?;
                let lanes = app.compiled_simd.as_ref().map_or(1, |v| v.count_lanes());
                let width = if complex { 2 } else { 1 };
                // All input/output planes are disjoint. The unused input is a sentinel.
                let mut planes: Vec<Vec<f64>> = (0..3 * width)
                    .map(|i| {
                        vec![
                            if i < width {
                                3.0 + i as f64
                            } else if i < 2 * width {
                                17.0 + i as f64
                            } else {
                                f64::NAN
                            };
                            lanes
                        ]
                    })
                    .collect();
                let table: Vec<_> = planes
                    .iter_mut()
                    .map(|plane| unsafe {
                        PlaneDescriptor::from_raw_parts(plane.as_mut_ptr(), lanes)
                    })
                    .collect();
                for (mode, kernel) in [
                    ("scalar", app.scalar_plane_kernel()),
                    ("SIMD", app.simd_plane_kernel()),
                ] {
                    let kernel = kernel.expect("this reproducer requires scalar and SIMD kernels");
                    for i in 0..width {
                        planes[i].fill(3.0 + i as f64);
                    }
                    for i in width..2 * width {
                        planes[i].fill(17.0 + i as f64);
                    }
                    for plane in &mut planes[2 * width..] {
                        plane.fill(f64::NAN);
                    }
                    let status =
                        unsafe { kernel(std::ptr::null(), table.as_ptr(), 0, app.params.as_ptr()) };
                    assert_eq!(status, 0);
                    let overwritten = (0..2 * width).any(|i| {
                        planes[i].iter().any(|value| {
                            *value
                                != if i < width {
                                    3.0 + i as f64
                                } else {
                                    17.0 + i as f64
                                }
                        })
                    });
                    if !expect_bug {
                        assert!(!overwritten, "input modified: complex={complex}, O{opt}, {mode}, constant={all_unused}");
                    }
                    let mut mismatches = 0;
                    for i in 0..width {
                        let expected = if all_unused {
                            7.0 + 2.0 * i as f64
                        } else {
                            6.0 + 2.0 * i as f64
                        };
                        for lane in 0..if mode == "scalar" { 1 } else { lanes } {
                            let missing = planes[2 * width + i][lane] != expected;
                            mismatches += usize::from(missing);
                            if !expect_bug {
                                assert!(!missing, "output: complex={complex}, O{opt}, {mode}, constant={all_unused}, lane={lane}");
                            }
                        }
                    }
                    if expect_bug {
                        println!("observed complex={complex} O{opt} {mode} constant={all_unused}: {planes:?}");
                        assert!(mismatches > 0, "expected a wrong output: complex={complex} O{opt} {mode} constant={all_unused}");
                    }
                }
                println!(
                    "unused trailing input: complex={complex}, O{opt}, constant={all_unused}: {}",
                    if expect_bug {
                        "BUG REPRODUCED (scalar and SIMD)"
                    } else {
                        "PASS (scalar and SIMD)"
                    }
                );
            }
        }
    }
    Ok(())
}
