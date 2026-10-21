//! Final upstream master-provider controls at MadLoop's rescue threshold.
//!
//! The soft box and its independent COLLIER/OneLOop references are from
//! NinjaRS aa6a6307 `tests/soft_box_labelling.rs`. The planar sentinels are
//! oneloopmaster 8ec256fc `STATUS.md`, ns9210 and ns11857. These controls
//! deliberately preserve upstream failures rather than accepting a finite
//! value merely because its tracked precision is large.
use super::*;
use ninjars::{Laurent, MasterFailure, MasterIntegral, MasterProvider, MasterRequest, MasterStatus};
use ninjars_symbolica::tracked_bits;
use serde_json::json;

fn run(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(128 * 1024 * 1024)
        .spawn(body).unwrap().join().unwrap();
}

fn provider(bits: u32) -> SymbolicaMasters<Float> {
    // Exactly the production bridge threshold, independent of working bits.
    let threshold = Float::with_val(bits, 1_000_000_000_000_000i64)
        / Float::with_val(bits, 2).pow(112);
    SymbolicaMasters::new(Float::new(bits)).with_onshell_threshold(Some(threshold))
}

fn recovered(text: &str, bits: u32) -> Float {
    let mut record = [b' '; RECORD];
    record[..text.len()].copy_from_slice(text.as_bytes());
    parse_record_at_precision(&record, bits).unwrap()
}

fn dyadic(value: f64, bits: u32) -> Float {
    // The STATUS inputs are binary64 values. The same fixed-width transport
    // used by MadLoop preserves them exactly through binary128 recovery.
    recovered(&format!("{value:.39e}"), bits)
}

fn rows(provider: &mut SymbolicaMasters<Float>, invariant_rows: &[[f64; 6]],
        mass_rows: &[[(f64, f64); 4]], scales: &[f64], bits: u32)
        -> (Vec<Laurent<Z>>, Vec<MasterStatus>) {
    let invariants: Vec<_> = invariant_rows.iter().flatten()
        .map(|value| real(dyadic(*value, bits))).collect();
    let masses: Vec<_> = mass_rows.iter().flatten().map(|(re, im)|
        Complex::new(dyadic(*re, bits), dyadic(*im, bits))).collect();
    let scales: Vec<_> = scales.iter().map(|value| real(dyadic(*value, bits))).collect();
    let sentinel = || real(Float::with_val(bits, 991));
    let mut output = vec![std::array::from_fn(|_| sentinel()); scales.len()];
    let mut statuses = vec![MasterStatus::Success; scales.len()];
    provider.evaluate_batch(MasterRequest {integral: MasterIntegral::D0,
        invariants: &invariants, masses_squared: &masses, scales_squared: &scales},
        &mut output, &mut statuses).unwrap();
    (output, statuses)
}

fn soft_rows() -> (Vec<[f64; 6]>, Vec<[(f64, f64); 4]>) {
    // d_ij = (q_i-q_j)^2. Reorder the propagators, then read the four
    // external invariants and the two channels in OneLOop's documented order.
    let (h,m,a,b,s) = (15625.,29929.,1299773111931./1562500.,
        976955119487./3125000.,1756052./5.);
    let distance = [[0.,h,s,a],[h,0.,m,b],[s,m,0.,m],[a,b,m,0.]];
    let masses = [m,m,0.,m];
    let mut invariant_rows = Vec::new();
    let mut mass_rows = Vec::new();
    for i in 0..4 { for j in 0..4 { for k in 0..4 { for l in 0..4 {
        if i == j || i == k || i == l || j == k || j == l || k == l { continue; }
        invariant_rows.push([distance[i][j],distance[j][k],distance[k][l],
            distance[l][i],distance[i][k],distance[j][l]]);
        mass_rows.push([i,j,k,l].map(|index| (masses[index],-0.)));
    }}}}
    assert_eq!(invariant_rows.len(),24);
    (invariant_rows,mass_rows)
}

fn represented(values: &Laurent<Z>) -> Vec<Vec<String>> {
    values.iter().map(|z| vec![format!("{:.39e}",z.re),format!("{:.39e}",z.im)]).collect()
}

#[test]
#[ignore = "requires exclusive MP slot; final-provider soft D0 permutations and failed-row recovery"]
fn soft_box_permutations_and_failure_recovery() {
    run(|| {
        let (invariants,masses) = soft_rows();
        let references = [(-2.0688070109990276e-10,4.1246220036002915e-11),
            (-2.6860952042523525e-11,3.9817595477722426e-11),(0.,0.)];
        for bits in [256,384] {
            let mut state = provider(bits);
            let (batch,statuses) = rows(&mut state,&invariants,&masses,&[1e6;24],bits);
            assert!(statuses.iter().all(|status| *status == MasterStatus::Success),"{statuses:?}");
            for (index,values) in batch.iter().enumerate() {
                for (value,(re,im)) in values.iter().zip(references) {
                    let error = (value.re.to_f64()-re).hypot(value.im.to_f64()-im);
                    assert!(error <= 1e-13*references[0].0.hypot(references[0].1),
                        "labelling{index} at{bits}bits: {value:?}");
                }
                for (actual,expected) in values.iter().zip(&batch[0]) {
                    let delta = (actual.re.clone()-&expected.re).norm()
                        +(actual.im.clone()-&expected.im).norm();
                    assert!(delta <= Float::parse("1e-31",Some(bits)).unwrap()
                        * (batch[0][0].re.norm()+batch[0][0].im.norm()));
                }
                println!("NINJARS_UPSTREAM_SOFT_BOX {}",json!({"working_bits":bits,
                    "permutation":index,"status":format!("{:?}",statuses[index]),
                    "output":represented(values),"laurent_and_permutation_passed":true,
                    "input_precision_bits":113,"threshold":"1e15*2^-112"}));
            }
            let mut invalid = masses[0]; invalid[0].1=1.;
            let (rejected,status) = rows(&mut state,&[invariants[0]],&[invalid],&[1e6],bits);
            assert_eq!(status,[MasterStatus::Failed(MasterFailure::InvalidMass)]);
            assert!(rejected[0].iter().all(|value| value == &real(Float::with_val(bits,991))));
            let (recovered,status) = rows(&mut state,&[invariants[0]],&[masses[0]],&[1e6],bits);
            assert_eq!(status,[MasterStatus::Success]);
            assert_eq!(recovered[0],batch[0]);
        }
    });
}

#[test]
#[ignore = "requires exclusive MP slot; upstream known-tail diagnostic, success is not certification"]
fn planar_box_sentinels_retain_unconverged_successes() {
    run(|| {
        let cases = [
            ("ns9210",1.,[-285.2138671875,13.123821258544922,0.,-780.9905891418457,
                -340.4909248352051,-104.8216438293457],
                [(6.957667603777042,-4.856459427327334),
                 (6.957667603777042,-4.856459427327334),
                 (6.957667603777042,-4.856459427327334),(0.,0.)]),
            ("ns11857",0.00390625,[34.35824579000473,-29.8271062374115,0.,
                0.5082151293754578,0.2645418047904968,-88.81898474693298],
                [(0.,0.),(1.97227437591852e-5,-8.531542965385371e-15),
                 (4.155059391982788e-5,0.),(0.5082151293754578,0.)]),
        ];
        let mut unconverged_successes = 0;
        for (name,scale,invariants,masses) in cases {
          // First reproduce STATUS's positive-zero records exactly, then
          // check MadLoop's negative-zero physical branch lip separately.
          for negative_zero in [false,true] {
            let masses = masses.map(|(re,im)|
                (re,if im==0. && negative_zero {-0.} else {im}));
            let mut observations = Vec::new();
            for bits in [256,512,1024] {
                let (output,status) = rows(&mut provider(bits),&[invariants],&[masses],&[scale],bits);
                println!("NINJARS_UPSTREAM_PLANAR_BOX {}",json!({"case":name,
                    "working_bits":bits,"status":format!("{:?}",status[0]),
                    "output":represented(&output[0]),"input_invariants":invariants,
                    "input_masses_squared":masses,"scale_squared":scale,
                    "zero_mass_imaginary_sign":if negative_zero {"negative"} else {"positive"},
                    "tracked_bits":output[0].iter().map(tracked_bits).collect::<Vec<_>>(),
                    "threshold":"1e15*2^-112","performance_claim":false}));
                observations.push((output[0].clone(),status[0]));
            }
            assert_eq!(observations[1].1,MasterStatus::Success,"512-bit reference {name}");
            assert_eq!(observations[2].1,MasterStatus::Success,"1024-bit reference {name}");
            let distance = |left: &Laurent<Z>,right: &Laurent<Z>| {
                let widen = |value: &Float| {let mut value=value.clone();value.set_prec(1024);value};
                let norm = right.iter().map(|z| widen(&z.re).norm()+widen(&z.im).norm())
                    .fold(Float::new(1024),|a,b| if a>b {a} else {b});
                left.iter().zip(right).map(|(a,b)|
                    ((widen(&a.re)-widen(&b.re)).norm()+(widen(&a.im)-widen(&b.im)).norm())/&norm)
                    .fold(Float::new(1024),|a,b| if a>b {a} else {b})
            };
            let reference_error = distance(&observations[1].0,&observations[2].0);
            assert!(reference_error < Float::parse("1e-31",Some(1024)).unwrap(),
                "512/1024 reference does not converge: {name}: {reference_error}");
            let error = (observations[0].1==MasterStatus::Success)
                .then(|| distance(&observations[0].0,&observations[2].0));
            let passed = error.as_ref().is_none_or(|value|
                value <= &Float::parse("1e-31",Some(1024)).unwrap());
            unconverged_successes += usize::from(!passed);
            println!("NINJARS_UPSTREAM_PLANAR_COMPARISON {}",json!({"case":name,
                "production_bits":256,"reference_bits":[512,1024],"tolerance":"1e-31",
                "zero_mass_imaginary_sign":if negative_zero {"negative"} else {"positive"},
                "production_status":format!("{:?}",observations[0].1),
                "maximum_scaled_difference":error.map(|value|format!("{value:.39e}")),
                "reference_maximum_scaled_difference":format!("{reference_error:.39e}"),
                "accepted_or_rejected_safely":passed,"performance_claim":false}));
          }
        }
        assert_eq!(unconverged_successes,0,
            "upstream finite master successes fail convergence; retain as an outstanding diagnostic");
    });
}
