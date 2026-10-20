fn parse_record_at_precision(text: &[u8], precision: u32) -> Result<Float> {
    let text = std::str::from_utf8(text)
        .map_err(|_| "MP input is not ASCII")?
        .trim();
    let input = Float::parse(text, Some(113)).map_err(|e| format!("invalid MP value: {e}"))?;
    if !input.is_finite() {
        return Err("nonfinite MP input".into());
    }
    // Forty significant decimal digits round-trip binary128. First recover that
    // value at 113 bits, then widen its exact represented rational, not the
    // decimal approximation, to the rescue working precision.
    Ok(Float::new(precision).from_rational(&input.to_rational()))
}
