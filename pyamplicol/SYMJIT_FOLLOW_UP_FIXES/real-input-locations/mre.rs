//! Two independent squares: x1 is real; x2 must retain its imaginary part.
use anyhow::{bail, Result};
use symjit::{CompilerType, Complex, Composer, Config, Slot, Translator};

fn main() -> Result<()> {
    let input = [
        Complex::new(1.0, 2.0), // unused x0 makes the coordinate error visible
        Complex::new(3.0, 0.0), // x1: declared real
        Complex::new(5.0, 7.0), // x2: not declared real
    ];
    let expected = [Complex::new(9.0, 0.0), Complex::new(-24.0, 70.0)];
    let mut failed = false;
    // Print the passing direct control before the failing indirect case.
    for direct in [true, false] {
        let mut config = Config::new(CompilerType::Native, 0)?;
        config.set_symbolica(true);
        config.set_complex(true);
        config.set_fast_complex(false);
        config.set_opt_level(2);
        config.set_compress(false);
        config.set_simd(false);
        config.set_threads(false);
        config.set_direct(direct);
        let mut translator = Translator::new(config);
        translator.set_num_params(input.len());
        // The final argument counts real arguments, not complex input slots.
        translator.append_mul(&Slot::Out(0), &[Slot::Param(1), Slot::Param(1)], 2)?;
        translator.append_mul(&Slot::Out(1), &[Slot::Param(2), Slot::Param(2)], 0)?;
        let app = translator.compile()?.seal()?;
        let mut output = [Complex::new(f64::NAN, f64::NAN); 2];
        app.evaluate_matrix(&input, &mut output, 1);
        println!("direct={direct}: actual={output:?}; expected={expected:?}");
        failed |= output != expected;
    }
    if failed {
        bail!("a real-input declaration changed a different complex input");
    }
    Ok(())
}
