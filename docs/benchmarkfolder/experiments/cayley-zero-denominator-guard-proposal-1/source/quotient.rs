// Extracted final quotient/filter only; size is an identity of a precomputed binary64 denominator.
#[inline(never)]
fn baseline(terms: f64, denominator: f64) -> Option<f64> {
    let det_x = denominator;
    let size = |x: &f64| *x;
    let ratio = terms / size(&det_x);
    ratio.is_finite().then_some(ratio)
}

#[inline(never)]
fn candidate(terms: f64, denominator: f64) -> Option<f64> {
    let det_x = denominator;
    let size = |x: &f64| *x;
    let denominator = size(&det_x);
    if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
        return None;
    }
    let ratio = terms / denominator;
    ratio.is_finite().then_some(ratio)
}

