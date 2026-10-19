// SPDX-License-Identifier: 0BSD
// Built by the adjacent Makefile against the installed, dependency-free SDK.
#[allow(dead_code)]
mod rusticol {
    include!(env!("RUSTICOL_RUST_SOURCE"));
}
use rusticol::virtual_amplitudes::Runtime;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) || (args.len() == 4 && args[3] != "resolved") {
        return Err("usage: check_standalone PROCESS MU_R2 [resolved]".into());
    }
    let scale: f64 = args[2].parse()?;
    if !scale.is_finite() || scale <= 0. {
        return Err("MU_R2 must be positive and finite".into());
    }
    let mut runtime = Runtime::open(".", Some(&args[1]), None, None)?;
    let axes = runtime.shape();
    let resolved = args.len() == 4;
    let text = std::fs::read_to_string("API/validation_points.dat")?;
    let mut lines = text.lines();
    if lines.next() != Some("RUSTICOL_VALIDATION_POINTS_V1") {
        return Err("invalid validation file".into());
    }
    let width = axes
        .external_count
        .checked_mul(4)
        .ok_or("momentum size overflow")?;
    let mut momenta = Vec::new();
    let mut count = 0usize;
    for line in lines {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.first() != Some(&args[1].as_str()) {
            continue;
        }
        if fields.len() != width.checked_add(2).ok_or("point size overflow")?
            || fields[1].parse::<usize>()? != axes.external_count
        {
            return Err("invalid validation dimensions".into());
        }
        for text in &fields[2..] {
            let value: f64 = text.parse()?;
            if !value.is_finite() {
                return Err("momenta must be finite".into());
            }
            momenta.push(value);
        }
        count = count.checked_add(1).ok_or("point count overflow")?;
    }
    if count == 0 {
        return Err("no validation points for the selected process".into());
    }
    let result = runtime.evaluate(&momenta, count, None, &[scale], resolved, None, None)?;
    println!("{}", runtime.metadata_json()?);
    println!("shape {:?}", result.shape);
    println!(
        "requested_ids {:?}\nsuccessful_ids {:?}\nstatuses {:?}",
        result.requested_point_ids, result.successful_point_ids, result.statuses
    );
    println!(
        "born {:?}\nvirtual {:?}",
        result.born, result.virtual_values
    );
    println!("{}", result.diagnostics_json);
    Ok(())
}
