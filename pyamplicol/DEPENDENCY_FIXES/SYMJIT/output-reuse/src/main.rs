//! Standalone reproduction: no Symbolica or pyAmpliCol dependency is needed.
use anyhow::{bail, Result};
use symjit::{CompilerType, Complex, Composer, Config, Slot, Translator};

fn run(case: &str) -> Result<()> {
    let mut config = Config::new(CompilerType::Native, 0)?;
    config.set_symbolica(true);
    config.set_opt_level(2);
    config.set_complex(case == "complex");
    let mut translator = Translator::new(config);

    if case == "copy" {
        // The minimal failing program: the first output must be both read and
        // published, even though it occurs only once on instruction RHSs.
        translator.set_num_params(1);
        translator.append_assign(&Slot::Out(0), &Slot::Param(0))?;
        translator.append_assign(&Slot::Out(1), &Slot::Out(0))?;
        let app = translator.compile()?;
        for x in [2.5, -7.0] {
            let mut output = [f64::NAN; 2];
            app.evaluate(&[x], &mut output);
            assert_eq!(output, [x, x]);
        }
    } else {
        if !["real", "complex", "overwrite"].contains(&case) {
            bail!("case must be copy, real, complex, overwrite, or all");
        }
        translator.set_num_params(2);
        let num_reals = if case == "complex" { 0 } else { 2 };
        translator.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], num_reals)?;
        translator.append_add(&Slot::Out(1), &[Slot::Out(0), Slot::Param(0)], num_reals)?;
        if case == "overwrite" {
            // The previous Out(0) is used once, but must no longer be published.
            translator.append_mul(&Slot::Out(0), &[Slot::Param(1), Slot::Param(1)], 2)?;
        }
        let app = translator.compile()?;

        if case == "complex" {
            for args in [
                [Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)],
                [Complex::new(-2.0, 1.0), Complex::new(1.0, 4.0)],
            ] {
                let mut output = [Complex::new(f64::NAN, f64::NAN); 2];
                app.evaluate(&args, &mut output);
                assert_eq!(output, [args[0] * args[1], args[0] * args[1] + args[0]]);
            }
        } else {
            for args in [[2.0, 3.0], [-1.0, 4.0]] {
                let first = if case == "overwrite" {
                    args[1] * args[1]
                } else {
                    args[0] * args[1]
                };
                let mut output = [f64::NAN; 2];
                app.evaluate(&args, &mut output);
                assert_eq!(output, [first, args[0] * args[1] + args[0]]);
            }
        }
    }
    println!("{case}: PASS (both input points)");
    Ok(())
}

fn main() -> Result<()> {
    let case = std::env::args().nth(1).unwrap_or_else(|| "copy".into());
    if case == "all" {
        for case in ["copy", "real", "complex", "overwrite"] {
            run(case)?;
        }
        Ok(())
    } else {
        run(&case)
    }
}
