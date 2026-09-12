//! Copy to tests/real_input_mre.rs in SymJIT; run cargo test --test real_input_mre.
use anyhow::Result;
use symjit::{CompilerType, Complex, Composer, Config, Slot, Translator};

#[test]
fn declaring_x1_real_must_not_discard_the_imaginary_part_of_x2() -> Result<()> {
    for direct in [false, true] {
        let mut config = Config::new(CompilerType::Native, 0)?;
        config.set_symbolica(true);
        config.set_complex(true);
        config.set_opt_level(2);
        config.set_direct(direct);
        let mut t = Translator::new(config);
        t.set_num_params(3);
        t.append_mul(&Slot::Out(0), &[Slot::Param(1), Slot::Param(1)], 2)?;
        t.append_mul(&Slot::Out(1), &[Slot::Param(2), Slot::Param(2)], 0)?;
        let app = t.compile()?.seal()?;
        let inputs = [Complex::new(1.0, 2.0), Complex::new(3.0, 0.0), Complex::new(5.0, 7.0)];
        let mut output = [Complex::new(f64::NAN, f64::NAN); 2];
        app.evaluate_matrix(&inputs, &mut output, 1);
        assert_eq!(output, [Complex::new(9.0, 0.0), Complex::new(-24.0, 70.0)],
                   "direct={direct}");
    }
    Ok(())
}
