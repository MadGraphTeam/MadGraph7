//! Scalar-only warm-call benchmark; this is not an end-to-end MadLoop timing.
//! cargo run --release --manifest-path vendor/tirust/pole-kernels/maintenance/Cargo.toml --example benchmark_poles
use madloop_pole_kernels::{Complex, ScalarIntegral, ScalarPoleEvaluator};
use oneloop::NativeEvaluator;
use std::{hint::black_box, time::Instant};

fn median_ns(mut operation: impl FnMut(), iterations: usize) -> f64 {
    for _ in 0..100 {
        operation();
    }
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .map(|x| x.parse().expect("iterations must be a positive integer"))
        .unwrap_or(20_000);
    assert!(iterations > 0);
    println!("family,case,iterations,full_ns,poles_ns,speedup");
    for (family, label, arguments) in [
        (ScalarIntegral::A0, "massive", vec![3.0, 2.0]),
        (ScalarIntegral::B0, "massive", vec![-3.0, 2.0, 4.0, 2.0]),
        (
            ScalarIntegral::C0,
            "massless",
            vec![0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0],
        ),
        (
            ScalarIntegral::C0,
            "massive",
            vec![-1.0, -2.0, -3.0, 1.0, 2.0, 3.0, 2.0],
        ),
        (
            ScalarIntegral::D0,
            "massless",
            vec![0.0, 0.0, 0.0, 0.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ),
        (
            ScalarIntegral::D0,
            "soft",
            vec![
                29929.0, 29929.0, 6464.0, 6464.0, -40000.0, 250000.0, 0.0, 29929.0, 0.0, 6464.0,
                1.0,
            ],
        ),
    ] {
        let input = arguments
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect::<Vec<_>>();
        let mut full = NativeEvaluator::<f64>::new(family).unwrap();
        let mut poles = ScalarPoleEvaluator::<f64>::new(family).unwrap();
        let mut full_result = [Complex::new(0.0, 0.0); 3];
        let mut pole_result = [Complex::new(0.0, 0.0); 2];
        let full_ns = median_ns(
            || {
                full.evaluate(black_box(&input), black_box(&mut full_result))
                    .unwrap();
                black_box(&full_result);
            },
            iterations,
        );
        let poles_ns = median_ns(
            || {
                poles
                    .evaluate(black_box(&input), black_box(&mut pole_result))
                    .unwrap();
                black_box(&pole_result);
            },
            iterations,
        );
        println!(
            "{},{label},{iterations},{full_ns:.3},{poles_ns:.3},{:.3}",
            family.name(),
            full_ns / poles_ns
        );
    }
}
