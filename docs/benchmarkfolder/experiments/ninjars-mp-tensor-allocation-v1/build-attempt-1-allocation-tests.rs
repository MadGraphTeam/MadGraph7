//! Numerical oracle is the frozen production dense source, not an algebraic
//! reimplementation. Compare full stored values, precision and signs.
use super::*;
use ninjars::{Float, SampleId};

fn equal_component(a: &Float, b: &Float) {
    assert_eq!(a.prec(), b.prec(), "tracked precision differs");
    assert_eq!(a.is_sign_negative(), b.is_sign_negative(), "sign differs");
    assert_eq!(a.is_finite(), b.is_finite());
    if a.is_finite() {
        assert_eq!(
            a.to_rational(),
            b.to_rational(),
            "exact stored value differs"
        );
    } else {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}

fn equal_rows(a: &[Complex<Float>], b: &[Complex<Float>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        equal_component(&a.re, &b.re);
        equal_component(&a.im, &b.im);
    }
}

fn number(bits: u32, seed: usize) -> Complex<Float> {
    // Different tracked precisions, signs, rounded fractions and signed zeros.
    let p = [bits, bits.saturating_sub(17).max(2), bits / 2][seed % 3];
    let re = if seed % 7 == 0 {
        Float::with_val(p, if seed % 2 == 0 { -0.0 } else { 0.0 })
    } else {
        Float::with_val(p, seed as i64 - 11) / Float::with_val(p, 7)
    };
    let im = if seed % 5 == 0 {
        Float::with_val(p, -0.0)
    } else {
        Float::with_val(p, 3 - seed as i64) / Float::with_val(p, 13)
    };
    Complex::new(re, im)
}

fn kinds(rank: usize) -> Vec<ExpansionKind> {
    let r = rank as i32;
    let mut t3 = Vec::new();
    let mut t2 = Vec::new();
    for p in (-1..=r).rev() {
        for l in 0..=(r - p) / 2 {
            t3.push((p, l as u16));
        }
        for a in 0..=r - p {
            for l in 0..=(r - p - a) / 2 {
                t2.push((p, a as u16, l as u16));
            }
        }
    }
    // Duplicate/reordered output columns and a deliberately empty contraction.
    t3.extend([(r, 0), (r + 2, 0), (-1, 0)]);
    t2.extend([(r, 0, 0), (r + 2, 0, 0), (-1, 0, 0)]);
    vec![
        ExpansionKind::Mu {
            powers: vec![r, 0, r - 1, -1, r],
        },
        ExpansionKind::MuShifted {
            powers: vec![r, 0, r - 1, -1, r],
        },
        ExpansionKind::T3 { powers: t3 },
        ExpansionKind::T2 { powers: t2 },
    ]
}

#[test]
fn allocation_tensor_matches_dense_values_precision_and_signed_zero() {
    for bits in [113, 256, 384] {
        for rank in 0..=6 {
            for mu_powers in 0..=2.min(rank / 2) {
                let prototype = Float::new(bits);
                let mut planned = TensorGenericNumerator::new(rank, mu_powers, &prototype).unwrap();
                let mut dense =
                    dense_reference::TensorGenericNumerator::new(rank, mu_powers, &prototype)
                        .unwrap();
                let coefficients: Vec<_> = (0..2 * planned.width)
                    .map(|i| number(bits, i + 1))
                    .collect();
                planned.set_coefficients(coefficients.clone(), 2).unwrap();
                dense.set_coefficients(coefficients.clone(), 2).unwrap();
                for kind in kinds(rank) {
                    let (vectors, scalars) = match kind {
                        ExpansionKind::Mu { .. } => (1, 0),
                        ExpansionKind::MuShifted { .. } => (2, 1),
                        ExpansionKind::T3 { .. } => (3, 1),
                        ExpansionKind::T2 { .. } => (4, 3),
                    };
                    for lane in 0..2 {
                        let v: Vec<_> = (0..vectors)
                            .map(|i| {
                                FourMomentum::from(std::array::from_fn(|c| {
                                    number(bits, i * 4 + c + lane * 9 + 1)
                                }))
                            })
                            .collect();
                        let p: Vec<_> = (0..scalars).map(|i| number(bits, i + lane + 2)).collect();
                        let request = ExpansionRequest {
                            id: SampleId { lane, sample: 0 },
                            partition: &[],
                            vectors: &v,
                            parameters: &p,
                        };
                        let mut a = vec![number(bits, 99); kind.len()];
                        let mut b = a.clone();
                        planned.expansion_batch(&kind, &[request], &mut a).unwrap();
                        // Construct the second request without requiring Clone on its type.
                        let request = ExpansionRequest {
                            id: SampleId { lane, sample: 0 },
                            partition: &[],
                            vectors: &v,
                            parameters: &p,
                        };
                        dense.expansion_batch(&kind, &[request], &mut b).unwrap();
                        equal_rows(&a, &b);
                    }
                }
                // New coefficients reuse existing allocated scratch.
                let rebound: Vec<_> = coefficients.into_iter().rev().collect();
                planned.set_coefficients(rebound.clone(), 2).unwrap();
                dense.set_coefficients(rebound, 2).unwrap();
                let kind = kinds(rank).pop().unwrap();
                let v0: Vec<_> = (0..4)
                    .map(|i| {
                        FourMomentum::from(std::array::from_fn(|c| number(bits, i * 4 + c + 31)))
                    })
                    .collect();
                let v1: Vec<_> = (0..4)
                    .map(|i| {
                        FourMomentum::from(std::array::from_fn(|c| number(bits, i * 4 + c + 57)))
                    })
                    .collect();
                let p0: Vec<_> = (0..3).map(|i| number(bits, i + 7)).collect();
                let p1: Vec<_> = (0..3).map(|i| number(bits, i + 21)).collect();
                let requests = || {
                    [
                        ExpansionRequest {
                            id: SampleId { lane: 1, sample: 0 },
                            partition: &[],
                            vectors: &v0,
                            parameters: &p0,
                        },
                        ExpansionRequest {
                            id: SampleId { lane: 0, sample: 1 },
                            partition: &[],
                            vectors: &v1,
                            parameters: &p1,
                        },
                    ]
                };
                let mut a = vec![number(bits, 99); 2 * kind.len()];
                let mut b = a.clone();
                planned.expansion_batch(&kind, &requests(), &mut a).unwrap();
                dense.expansion_batch(&kind, &requests(), &mut b).unwrap();
                equal_rows(&a, &b);
            }
        }
    }
}

#[test]
fn allocation_tensor_nonfinite_inputs_match_original() {
    let bits = 113;
    let prototype = Float::new(bits);
    let mut planned = TensorGenericNumerator::new(2, 0, &prototype).unwrap();
    let mut dense = dense_reference::TensorGenericNumerator::new(2, 0, &prototype).unwrap();
    let coefficients: Vec<_> = (0..planned.width).map(|i| number(bits, i)).collect();
    planned.set_coefficients(coefficients.clone(), 1).unwrap();
    dense.set_coefficients(coefficients, 1).unwrap();
    let mut v = vec![FourMomentum::from(std::array::from_fn(|i| {
        number(bits, i + 1)
    }))];
    let kind = ExpansionKind::Mu {
        powers: vec![2, 1, 0],
    };
    v[0] = FourMomentum::from([
        Complex::new(Float::with_val(bits, f64::INFINITY), Float::new(bits)),
        number(bits, 2),
        number(bits, 3),
        number(bits, 4),
    ]);
    let mut a = vec![number(bits, 8); kind.len()];
    let mut b = a.clone();
    planned
        .expansion_batch(
            &kind,
            &[ExpansionRequest {
                id: SampleId { lane: 0, sample: 0 },
                partition: &[],
                vectors: &v,
                parameters: &[],
            }],
            &mut a,
        )
        .unwrap();
    dense
        .expansion_batch(
            &kind,
            &[ExpansionRequest {
                id: SampleId { lane: 0, sample: 0 },
                partition: &[],
                vectors: &v,
                parameters: &[],
            }],
            &mut b,
        )
        .unwrap();
    equal_rows(&a, &b);
}

#[test]
fn allocation_evaluation_rebinds_scratch_and_lanes_exactly() {
    for bits in [113, 256, 384] {
        for rank in 0..=6 {
            for mu_powers in 0..=rank / 2 {
                let prototype = Float::new(bits);
                let mut planned = TensorGenericNumerator::new(rank, mu_powers, &prototype).unwrap();
                let mut dense = dense_reference::TensorGenericNumerator::new(rank, mu_powers, &prototype).unwrap();
                for round in 0..3 {
                    let coefficients: Vec<_> = (0..2 * planned.width).map(|i| number(bits, i + 17 * round)).collect();
                    planned.set_coefficients(coefficients.clone(), 2).unwrap();
                    dense.set_coefficients(coefficients, 2).unwrap();
                    let momenta: Vec<_> = (0..4).map(|i| FourMomentum::from(std::array::from_fn(|j| number(bits, 13 * round + i * 4 + j)))).collect();
                    let mu: Vec<_> = (0..4).map(|i| number(bits, round * 11 + i)).collect();
                    let requests = || (0..4).map(|i| NumeratorRequest {
                        id: SampleId { lane: i % 2, sample: i }, q: &momenta[i], mu_squared: &mu[i],
                    }).collect::<Vec<_>>();
                    let mut a = vec![number(bits, 99); 4];
                    let mut b = a.clone();
                    planned.evaluate_batch(&requests(), &mut a).unwrap();
                    dense.evaluate_batch(&requests(), &mut b).unwrap();
                    equal_rows(&a, &b);
                }
                planned.evaluate_batch(&[], &mut []).unwrap();
                dense.evaluate_batch(&[], &mut []).unwrap();
                let mut a = [number(bits, 99)];
                let mut b = a.clone();
                assert_eq!(format!("{:?}", planned.evaluate_batch(&[], &mut a)), format!("{:?}", dense.evaluate_batch(&[], &mut b)));
                equal_rows(&a, &b);
            }
        }
    }
}

#[test]
fn borrowed_product_and_slot_assignment_preserve_all_float_state() {
    for bits in [2, 53, 113, 256, 384] {
        let mut values: Vec<_> = (0..24).map(|i| number(bits, i)).collect();
        for exceptional in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0, 0.0] {
            values.push(Complex::new(Float::with_val(bits, exceptional), Float::with_val(bits, -0.0)));
        }
        for a in &values {
            for b in &values {
                equal_rows(&[borrowed_product(a, b)], &[a.clone() * b]);
                let mut slot = a.clone();
                slot.set_from(b);
                equal_rows(&[slot], &[b.clone()]);
            }
        }
    }
}
