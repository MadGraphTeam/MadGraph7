fn parse_record_at_precision(text: &[u8], precision: u32) -> Result<Float> {
    let text = std::str::from_utf8(text)
        .map_err(|_| "MP input is not ASCII")?
        .trim();
    let mut input = Float::parse(text, Some(113)).map_err(|e| format!("invalid MP value: {e}"))?;
    if !input.is_finite() {
        return Err("nonfinite MP input".into());
    }
    // Forty significant decimal digits round-trip binary128. First recover that
    // value at 113 bits, then widen the represented value itself. A rational
    // round-trip would discard the sign of zero, which selects complex branch lips.
    input.set_prec(precision);
    Ok(input)
}
