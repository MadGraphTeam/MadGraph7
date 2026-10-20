//! Replay actual MadLoop binary128 requests, comparing every generic expansion.
use super::*;
use crate::{
    AmplitudeInput, EvaluationStatus, Float, MP_BITS, RECORD, SymbolicaAmplitudeKernels,
    SymbolicaCuts, SymbolicaHighCuts, SymbolicaMasters, pairs, parse_records, real,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

struct Oracle {
    planned: TensorGenericNumerator<Float>,
    dense: dense_reference::TensorGenericNumerator<Float>,
    counts: BTreeMap<&'static str, usize>,
}
fn equal_component(a: &Float, b: &Float) {
    assert_eq!(a.prec(), b.prec());
    assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    assert!(a.is_finite() && b.is_finite());
    assert_eq!(a.to_rational(), b.to_rational());
}
fn equal(a: &[Complex<Float>], b: &[Complex<Float>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        equal_component(&a.re, &b.re);
        equal_component(&a.im, &b.im);
    }
}
impl Numerator<Complex<Float>> for Oracle {
    fn rank(&self) -> usize {
        self.planned.rank()
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Complex<Float>>],
        output: &mut [Complex<Float>],
    ) -> ninjars::Result<()> {
        let mut reference = output.to_vec();
        self.planned.evaluate_batch(requests, output)?;
        self.dense.evaluate_batch(requests, &mut reference)?;
        equal(output, &reference);
        *self.counts.entry("numerator_values").or_default() += output.len();
        Ok(())
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Complex<Float>>],
        output: &mut [Complex<Float>],
    ) -> ninjars::Result<()> {
        let mut reference = output.to_vec();
        self.planned.expansion_batch(kind, requests, output)?;
        self.dense.expansion_batch(kind, requests, &mut reference)?;
        equal(output, &reference);
        let name = match kind {
            ExpansionKind::Mu { .. } => "mu",
            ExpansionKind::MuShifted { .. } => "mu_shifted",
            ExpansionKind::T3 { .. } => "t3",
            ExpansionKind::T2 { .. } => "t2",
        };
        *self.counts.entry(name).or_default() += output.len();
        Ok(())
    }
}
fn records(value: &Value) -> Vec<Float> {
    let mut bytes = Vec::new();
    for string in value.as_array().unwrap() {
        let text = string.as_str().unwrap().as_bytes();
        assert!(text.len() <= RECORD);
        let mut record = [b' '; RECORD];
        record[..text.len()].copy_from_slice(text);
        bytes.extend(record);
    }
    parse_records(&bytes, MP_BITS).unwrap()
}

#[test]
#[ignore = "requires an explicit retained MadLoop request capture"]
fn actual_madloop_requests_match_dense_expansions_exactly() {
    let path = std::env::var("NINJARS_TENSOR_REPLAY").expect("capture path required");
    let text = std::fs::read_to_string(path).unwrap();
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(move || {
            let prototype = Float::new(MP_BITS);
            let mut low = SymbolicaCuts::new(prototype.clone()).unwrap();
            let mut high = SymbolicaHighCuts::new(prototype.clone()).unwrap();
            let threshold = Float::with_val(MP_BITS, 1_000_000_000_000_000i64)
                / Float::with_val(MP_BITS, 2).pow(112);
            let mut masters =
                SymbolicaMasters::new(prototype.clone()).with_onshell_threshold(Some(threshold));
            let mut seen = HashSet::new();
            let mut total = BTreeMap::<&'static str, usize>::new();
            let mut tested = 0;
            for line in text.lines() {
                let record: Value = serde_json::from_str(line).unwrap();
                if record["status"] != 0 {
                    continue;
                }
                let n = record["n"].as_u64().unwrap() as usize;
                let rank = record["rank"].as_u64().unwrap() as usize;
                // Retain one successful actual request for every topology/rank.
                // The complete capture remains available for wider replay.
                if !seen.insert((n, rank)) {
                    continue;
                }
                let coefficients = pairs(records(&record["coefficients"]));
                let shifts: Vec<_> = records(&record["momenta"])
                    .chunks_exact(4)
                    .map(|v| FourMomentum::from(std::array::from_fn(|k| real(v[k].clone()))))
                    .collect();
                let masses = pairs(records(&record["masses"]));
                let smat: Vec<_> = records(&record["s_matrix"]).into_iter().map(real).collect();
                let scales: Vec<_> = records(&record["scale"]).into_iter().map(real).collect();
                let sampling = [real(smat.iter().chain(&masses).map(|v| v.re.norm()).fold(
                    Float::with_val(MP_BITS, 1),
                    |a, b| if b > a { b } else { a },
                ))];
                let mut oracle = Oracle {
                    planned: TensorGenericNumerator::new(rank, 0, &prototype).unwrap(),
                    dense: dense_reference::TensorGenericNumerator::new(rank, 0, &prototype)
                        .unwrap(),
                    counts: BTreeMap::new(),
                };
                oracle
                    .planned
                    .set_coefficients(coefficients.clone(), 1)
                    .unwrap();
                oracle.dense.set_coefficients(coefficients, 1).unwrap();
                let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())
                    .unwrap()
                    .into_amplitude(1)
                    .unwrap();
                amplitude.reset_batch(1);
                amplitude
                    .evaluate_batch_with_inputs(
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            sampling_scales: &sampling,
                            s_matrix: Some(&smat),
                        },
                        &mut oracle,
                        &mut low,
                        &mut high,
                        &mut masters,
                    )
                    .unwrap();
                assert_eq!(amplitude.statuses()[0], EvaluationStatus::Success);
                let result = &amplitude.results()[0];
                let formatted: Vec<_> = [0, 1, 2, 4]
                    .into_iter()
                    .flat_map(|i| [&result[i].re, &result[i].im])
                    .map(|v| format!("{v:.39e}"))
                    .collect();
                let expected: Vec<_> = record["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap())
                    .collect();
                assert_eq!(formatted, expected, "ABI output differs for ({n},{rank})");
                for (kind, count) in oracle.counts {
                    *total.entry(kind).or_default() += count;
                }
                tested += 1;
                eprintln!("captured request n={n} rank={rank}: exact");
            }
            assert!(tested >= 4, "expected multiple actual topology/rank shapes");
            assert!(total.get("t2").copied().unwrap_or(0) > 0);
            assert!(total.get("t3").copied().unwrap_or(0) > 0);
            eprintln!("captured exact requests={tested}; expansion/value counts={total:?}");
        })
        .unwrap()
        .join()
        .unwrap();
}
