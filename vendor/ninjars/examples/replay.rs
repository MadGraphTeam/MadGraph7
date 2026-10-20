//! Replay an archived MadLoop tensor call against its original-Ninja result.
//! Usage: cargo run --release --example replay -- CASE.json REFERENCE.json [--mp]
use madloop_ninjars::{
    ninjars_context_create, ninjars_context_destroy, ninjars_evaluate_dp, ninjars_evaluate_mp,
    ninjars_last_error,
};
use ninjars::fast::C64;
use serde_json::{Value, json};

fn number(v: &Value) -> f64 {
    v.as_f64().expect("numeric fixture")
}
fn complex(v: &Value) -> C64 {
    C64::new(number(&v[0]), number(&v[1]))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "--mp"),
        "expected CASE.json REFERENCE.json [--mp]"
    );
    let mp = args.len() == 4;
    let case: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let mut worst: f64 = 0.;
    let mut rational_split_difference: f64 = 0.;
    let mut context = std::ptr::null_mut();
    assert_eq!(unsafe { ninjars_context_create(2, &mut context) }, 0);
    let lanes = case["lanes"].as_array().unwrap();
    for lane in lanes {
        let expected = reference["lanes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == lane["id"])
            .expect("reference lane");
        let tensor: Vec<_> = lane["tensor"]
            .as_array()
            .unwrap()
            .iter()
            .map(complex)
            .collect();
        let masses: Vec<_> = lane["masses_squared"]
            .as_array()
            .unwrap()
            .iter()
            .map(complex)
            .collect();
        let flatten = |key| {
            lane[key]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap().iter().map(number))
                .collect::<Vec<_>>()
        };
        let momenta = flatten("shifts");
        let matrix = flatten("s_matrix");
        let mut result = [C64::ZERO; 3];
        let mut rational = C64::ZERO;
        let status = if mp {
            let records = |values: &[f64]| {
                values
                    .iter()
                    .flat_map(|value| {
                        let mut record = [b' '; 64];
                        let text = format!("{value:.39e}");
                        record[..text.len()].copy_from_slice(text.as_bytes());
                        record
                    })
                    .collect::<Vec<_>>()
            };
            let ct = records(&tensor.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>());
            let mt = records(&masses.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>());
            let st = records(&matrix);
            let pt = records(&momenta);
            let scale = records(&[number(&lane["mu_squared"])]);
            let mut output = [b' '; 512];
            let status = unsafe {
                ninjars_evaluate_mp(
                    context,
                    case["n"].as_i64().unwrap() as i32,
                    case["rank"].as_i64().unwrap() as i32,
                    tensor.len(),
                    ct.as_ptr(),
                    ct.len(),
                    st.as_ptr(),
                    st.len(),
                    pt.as_ptr(),
                    pt.len(),
                    mt.as_ptr(),
                    mt.len(),
                    scale.as_ptr(),
                    scale.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            if status == 0 {
                let component = |i| {
                    std::str::from_utf8(&output[i * 64..(i + 1) * 64])
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap()
                };
                for i in 0..3 {
                    result[i] = C64::new(component(2 * i), component(2 * i + 1));
                }
                rational = C64::new(component(6), component(7));
            }
            status
        } else {
            unsafe {
                ninjars_evaluate_dp(
                    context,
                    case["n"].as_i64().unwrap() as i32,
                    case["rank"].as_i64().unwrap() as i32,
                    tensor.len(),
                    tensor.as_ptr(),
                    matrix.as_ptr(),
                    matrix.len(),
                    momenta.as_ptr(),
                    momenta.len(),
                    masses.as_ptr(),
                    masses.len(),
                    number(&lane["mu_squared"]),
                    result.as_mut_ptr(),
                    result.len(),
                    &mut rational,
                    1,
                )
            }
        };
        if status != 0 {
            let mut error = [0; 1024];
            unsafe {
                ninjars_last_error(error.as_mut_ptr(), error.len());
            }
            panic!(
                "lane {} status {status}: {}",
                lane["id"],
                String::from_utf8_lossy(&error)
            );
        }
        for (value, key) in result
            .into_iter()
            .chain([rational])
            .zip(["eps0", "eps_m1", "eps_m2", "rational"])
        {
            let expected = complex(&expected[key]);
            let error = (value - expected).abs() / expected.abs().max(1.);
            if key == "rational" {
                rational_split_difference = rational_split_difference.max(error);
                // The generic engine conditions its bubble basis differently.
                // Its CC/rational split is basis dependent; full Laurent
                // coefficients are invariant (upstream fast_parity.rs).
                if mp {
                    continue;
                }
            }
            worst = worst.max(error);
            assert!(error <= 1e-9, "lane {} {key}: error {error}", lane["id"]);
        }
    }
    assert_eq!(unsafe { ninjars_context_destroy(context) }, 0);
    println!(
        "{}",
        json!({"case": case["name"], "precision": if mp {"Float(256)"} else {"f64"}, "lanes": lanes.len(), "max_scaled_error": worst, "rational_compared": !mp, "rational_split_max_scaled_difference": rational_split_difference, "passed": true})
    );
}
