use anyhow::{ensure, Result};
use symjit::{Complex, Composer, Config, Slot, Translator};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map_or("64", String::as_str).parse()?;
    ensure!(n > 0, "arity must be positive");
    let compress = !args.iter().any(|arg| arg == "--no-compress");
    let mut config = Config::default();
    config.set_symbolica(true);
    config.set_complex(true);
    config.set_opt_level(2);
    config.set_compress(compress);
    config.set_threads(false);
    let mut translator = Translator::new(config);
    translator.set_num_params(3 * n);

    // Three identical expression shapes, with different inputs. Alternating
    // multiplication/addition keeps a skew tree with register pressure two.
    for group in 0..3 {
        let mut value = Slot::Param(group * n);
        for j in 1..n {
            let next = Slot::Temp(group * n + j);
            let operands = [value, Slot::Param(group * n + j)];
            if j % 2 == 1 {
                translator.append_mul(&next, &operands, 0)?;
            } else {
                translator.append_add(&next, &operands, 0)?;
            }
            value = next;
        }
        translator.append_assign(&Slot::Out(group), &value)?;
    }

    let app = translator.compile()?.seal()?;
    let mut output = [Complex::new(f64::NAN, f64::NAN); 3];
    app.evaluate_matrix(&vec![Complex::new(1.0, 0.0); 3 * n], &mut output, 1);
    let expected = [Complex::new(((n + 1) / 2) as f64, 0.0); 3];
    println!("arity={n}, compress={compress}, feature={}: {output:?}; expected={expected:?}",
        cfg!(feature = "symbolica"));
    ensure!(output == expected, "incorrect result");
    Ok(())
}
