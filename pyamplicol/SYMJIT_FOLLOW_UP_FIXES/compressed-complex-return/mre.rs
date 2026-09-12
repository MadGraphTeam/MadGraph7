//! Put this file at src/symjit/followup_return_mre.rs in SymJIT, and add
//! `#[cfg(test)] mod followup_return_mre;` at the END of src/symjit/mod.rs.
//! Run: cargo test --lib followup_return_mre -- --nocapture
//!
//! Test the complex-lowering call boundary directly, with readable MIR.
//! This is not a claim that these instructions are a complete frontend input.
use std::collections::HashSet;

use super::{
    config::Config,
    mir::Mir,
    model::{CellModel, Program},
    runnable::Application,
    symbol::Loc,
    utils::{Compiled, PlaneDescriptor, Reg},
};

fn evaluate(real_return: bool, called: bool) -> anyhow::Result<Vec<[f64; 2]>> {
    let mut config = Config::default();
    config.set_complex(true);
    config.set_fast_complex(false);
    config.set_simd(true);
    config.set_threads(false);
    config.set_direct_arena(true);
    config.set_direct_arena_identity_output(true);
    let mut program = Program::new(&CellModel::new(), config.clone())?;
    program.count_params = 2; // One complex parameter, declared real: 7+0i.
    program.count_states = 2; // One complex runtime input: 3+4i.
    program.count_obs = 2;

    let mut mir = Mir::new(config);
    // Leave the opposite real/complex type in the return temporary first.
    if real_return { mir.load_mem(Reg::Ret, 0); }
    else { mir.load_param(Reg::Ret, 0); }
    if called {
        mir.call("helper", 0)?;
        mir.save_mem(Reg::Ret, 2);
        mir.branch("finished");
        mir.set_label("helper");
    }
    if real_return { mir.load_param(Reg::Ret, 0); }
    else { mir.load_mem(Reg::Ret, 0); }
    if called {
        mir.branch(".ret");
        mir.set_label("finished");
    } else {
        mir.save_mem(Reg::Ret, 2);
    }

    // Parameter zero has identical coordinates with either marker convention.
    let mut application = Application::with_mir(
        program, HashSet::from([Loc::Param(0)]), mir,
    )?;
    application.params[0] = 7.0;
    application.prepare_simd();
    let app = application.seal()?;
    anyhow::ensure!(app.scalar_plane_kernel().is_some(), "native scalar kernel required");
    let lanes = app.compiled_simd.as_ref().map_or(1, |m| m.count_lanes());
    let mut values: Vec<Vec<f64>> = [3.0, 4.0, f64::NAN, f64::NAN]
        .into_iter().map(|v| vec![v; lanes]).collect();
    let planes: Vec<_> = values.iter_mut().map(|v| unsafe {
        PlaneDescriptor::from_raw_parts(v.as_mut_ptr(), lanes)
    }).collect();
    let mut results = Vec::new();
    for kernel in [app.scalar_plane_kernel(), app.simd_plane_kernel()].into_iter().flatten() {
        values[2].fill(f64::NAN);
        values[3].fill(f64::NAN);
        let status = unsafe { kernel(std::ptr::null(), planes.as_ptr(), 0, app.params.as_ptr()) };
        assert_eq!(status, 0);
        results.push([values[2][0], values[3][0]]);
    }
    Ok(results)
}

#[test]
fn helper_return_type_does_not_depend_on_the_previous_temporary() -> anyhow::Result<()> {
    let mut failures = Vec::new();
    for real in [false, true] {
        let expected = if real { [7.0, 0.0] } else { [3.0, 4.0] };
        for called in [false, true] {
            let results = evaluate(real, called)?;
            eprintln!("real_return={real}, called={called}: {results:?}; expected={expected:?}");
            if results.iter().any(|x| *x != expected) { failures.push((real, called)); }
        }
    }
    anyhow::ensure!(failures.is_empty(), "wrong results: {failures:?}");
    Ok(())
}
