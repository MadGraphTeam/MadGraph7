// SPDX-License-Identifier: 0BSD

use super::*;
use crate::loop_v2::plan::KernelTerm;

/// A kernel of `terms` monomials of degree up to 5 over `inputs` lanes,
/// with constants and inputs from a fixed xorshift stream (signed,
/// spread over a few binades).
fn case(terms: usize, inputs: usize) -> (Vec<KernelTerm>, Vec<Complex<f64>>, Vec<f64>) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let value = |next: &mut dyn FnMut() -> u64| {
        let r = next();
        let mantissa = (r >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        mantissa * f64::powi(2.0, (r % 7) as i32 - 3)
    };
    let constants = (0..terms)
        .map(|_| Complex::new(value(&mut next), value(&mut next)))
        .collect();
    let kernel = (0..terms)
        .map(|t| KernelTerm {
            constant: t,
            monomial: (0..1 + next() % 4)
                .map(|_| ((next() % inputs as u64) as usize, 1 + (next() % 2) as u32))
                .collect(),
        })
        .collect();
    let x = (0..inputs * 16).map(|_| value(&mut next)).collect();
    (kernel, constants, x)
}

#[test]
fn wide_closure_kernels_are_bitwise_the_baseline() {
    let (kernel, constants, x) = case(72, 24);
    let baseline = kernel_lanes::<f64, 8>(&kernel, &constants, &x, &0.0);
    let bits = |(re, im): ([f64; 8], [f64; 8])| (re.map(f64::to_bits), im.map(f64::to_bits));
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2.
            let wide = unsafe { kernel_lanes_avx2::<f64, 8>(&kernel, &constants, &x, &0.0) };
            assert_eq!(bits(wide), bits(baseline));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU reports AVX-512F.
            let wide = unsafe { kernel_lanes_avx512::<f64, 8>(&kernel, &constants, &x, &0.0) };
            assert_eq!(bits(wide), bits(baseline));
        }
    }
    // Each lane is the scalar evaluation of `Complex<f64>`.
    for lane in 0..8 {
        let mut value = Complex::new(0.0, 0.0);
        for term in &kernel {
            let mut product = constants[term.constant];
            for &(input, power) in &term.monomial {
                for _ in 0..power {
                    product *= Complex::new(x[input * 16 + lane], x[input * 16 + 8 + lane]);
                }
            }
            value += product;
        }
        assert_eq!(baseline.0[lane].to_bits(), value.re.to_bits());
        assert_eq!(baseline.1[lane].to_bits(), value.im.to_bits());
    }
}

#[test]
fn lane_runs_cover_the_batch_without_padding() {
    for points in 0..40 {
        let runs = lane_runs(points);
        let mut next = 0;
        for &(first, count, lanes) in &runs {
            assert_eq!(first, next);
            assert!(count > 0 && count % lanes == 0, "{points}: {runs:?}");
            next += count;
        }
        assert_eq!(next, points);
        // Lanes of 8 first, then at most one run of 4, 2 and 1 each.
        let widths: Vec<usize> = runs.iter().map(|run| run.2).collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] > pair[1]),
            "{widths:?}"
        );
    }
    assert_eq!(lane_runs(9), vec![(0, 8, 8), (8, 1, 1)]);
    assert_eq!(lane_runs(7), vec![(0, 4, 4), (4, 2, 2), (6, 1, 1)]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_closure_simd_cap_only_lowers_the_width() {
    assert_eq!(Wide::cap(Some("baseline")), Wide::Baseline);
    assert_eq!(Wide::cap(Some("avx2")), Wide::Avx2);
    assert_eq!(Wide::cap(Some("avx512")), Wide::Avx512);
    assert_eq!(Wide::cap(Some("AVX2")), Wide::Avx512);
    assert_eq!(Wide::cap(None), Wide::Avx512);
    // Other carriers take the baseline whatever the cap.
    assert_eq!(Wide::detect::<f32>(), Wide::Baseline);
}
