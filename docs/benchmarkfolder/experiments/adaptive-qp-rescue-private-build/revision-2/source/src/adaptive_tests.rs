//! Dispatch tests use an injected outcome, not a fake numerical reference.
//! Tests whose names start `native_` additionally evaluate real scalar kernels.
use super::*;

fn inputs(n: usize, mass: &str) -> [Vec<u8>; 4] {
    fn records(values: &[&str]) -> Vec<u8> {
        let mut out = vec![b' '; values.len() * RECORD];
        for (i, value) in values.iter().enumerate() {
            out[i * RECORD..i * RECORD + value.len()].copy_from_slice(value.as_bytes());
        }
        out
    }
    let mut masses = Vec::new();
    for _ in 0..n { masses.extend([mass, "-0"]); }
    [records(&vec!["0"; 4*n]), records(&vec!["0"; n*n]), records(&masses), records(&["4"])]
}
fn refs(data: &[Vec<u8>; 4]) -> [&[u8]; 4] {
    [&data[0], &data[1], &data[2], &data[3]]
}
fn mode(check: bool, cache: bool, finite: bool) -> Mode {
    Mode { poles: 0, finite, check, cache }
}
fn refusal() -> LaneStatus {
    LaneStatus::Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 1.0e-19".into())
}
fn failed() -> MpAttempt { MpAttempt::failed(&refusal(), Some(1e-76), 4) }

#[test]
fn only_current_positive_known_gram_refusal_is_eligible() {
    assert!(gram_retry_eligible(&refusal(), Some(1e-76), 4));
    for condition in [None, Some(0.), Some(-1.), Some(f64::NAN), Some(f64::INFINITY)] {
        assert!(!gram_retry_eligible(&refusal(), condition, 4));
    }
    for status in [
        LaneStatus::Success,
        LaneStatus::MasterFailure("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 1e-19".into()),
        LaneStatus::InvalidInput("bad geometry".into()),
        LaneStatus::Unstable("singular Gram matrix for [0, 1, 2, 3]".into()),
        LaneStatus::Unstable("ill-conditioned Cayley matrix for [0, 1, 2, 3]: relative determinant 1e-76".into()),
        LaneStatus::Unstable("nonfinite direct-engine coefficient".into()),
        LaneStatus::Unstable("ill-conditioned Gram matrix for [0, 1, 1, 3]: conditioning 1e-19".into()),
        LaneStatus::Unstable("ill-conditioned Gram matrix for [0, 1, 2, 4]: conditioning 1e-19".into()),
        LaneStatus::Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning NaN".into()),
        LaneStatus::Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 1e-19 extra".into()),
    ] { assert!(!gram_retry_eligible(&status, Some(1e-76), 4), "{status:?}"); }
}

#[test]
fn ordinary_success_and_all_noneligible_failures_never_retry() {
    let data = inputs(4, "4");
    let mut state = State::new();
    let mut calls = Vec::new();
    let result = state.mp_records_with((4,1), refs(&data), mode(false,false,true), |_,_,_,_,bits| {
        calls.push(bits); Ok(MpAttempt::success(b"original-256-output".to_vec()))
    }).unwrap();
    assert_eq!(result, (SUCCESS, b"original-256-output".to_vec()));
    assert_eq!(calls, [256]);
    assert_eq!(state.qp_precision.evaluations_512, 0);
    for status in [
        LaneStatus::MasterFailure("scalar refusal".into()),
        LaneStatus::InvalidInput("invalid input".into()),
        LaneStatus::Unstable("singular Gram matrix for [0, 1, 2, 3]".into()),
        LaneStatus::Unstable("nonfinite direct-engine coefficient".into()),
    ] {
        let mut state = State::new(); let mut count = 0;
        let result = state.mp_records_with((4,1), refs(&data), mode(false,true,true), |_,_,_,_,bits| {
            count += 1; assert_eq!(bits,256); Ok(MpAttempt::failed(&status, Some(1e-76), 4))
        }).unwrap();
        assert_ne!(result.0,SUCCESS); assert!(result.1.is_empty());
        assert_eq!(count,1); assert_eq!(state.qp_precision.evaluations_512,0);
        assert!(state.qp_cache.entries.is_empty());
    }
    let mut state = State::new(); let mut count = 0;
    let result = state.mp_records_with((4,1), refs(&data), mode(false,true,true), |_,_,_,_,_| {
        count += 1; Err("nonfinite QP result".into())
    });
    assert!(result.is_err()); assert_eq!(count,1);
    assert_eq!(state.qp_precision.last_bits,0);
}

#[test]
fn one_retry_preserves_113_bit_inputs_and_never_retries_high_failure() {
    let value = Float::with_val(113,1) + Float::with_val(113,2).pow(112).inv();
    let data = inputs(4, &format!("{value:.39e}"));
    let expected = value.to_rational();
    let mut state = State::new(); let mut calls = Vec::new();
    let result = state.mp_records_with((4,1), refs(&data), mode(false,false,true), |_,_,batch,_,bits| {
        calls.push(bits);
        let mass = &batch.values[16+6];
        assert_eq!(mass.re.get_precision(), bits);
        assert_eq!(mass.re.to_rational(),expected);
        assert!(mass.im.is_sign_negative());
        if bits==256 { Ok(failed()) } else { Ok(MpAttempt::success(b"high".to_vec())) }
    }).unwrap();
    assert_eq!(calls,[256,512]); assert_eq!(result,(SUCCESS,b"high".to_vec()));
    assert_eq!(state.qp_precision.statistics()[..10],[1,1,1,1,0,0,0,0,0,512]);
    assert_eq!(f64::from_bits(state.qp_precision.last_condition_bits),1e-76);
    assert!(LAST_ERROR.with(|e|e.borrow().is_empty()));
    for high_error in [false,true] {
        let mut state=State::new(); let mut calls=Vec::new();
        let outcome=state.mp_records_with((4,1),refs(&data),mode(false,true,true),|_,_,_,_,bits| {
            calls.push(bits);
            if bits==512 && high_error { Err("512 formatting failure".into()) } else { Ok(failed()) }
        });
        assert_eq!(calls,[256,512]);
        assert!(!matches!(outcome,Ok((SUCCESS,_))));
        assert_eq!(state.qp_precision.failures_512,1);
        assert_eq!(state.qp_precision.last_bits,0);
        assert!(state.qp_cache.entries.is_empty());
    }
}

#[test]
fn tier_cache_requires_fresh_refusal_and_check_bypasses_both() {
    let data=inputs(4,"4"); let mut state=State::new();
    let mut evaluate=|_:&mut State,_,_:KinematicsBatch<Float>,_:Mode,bits| {
        if bits==256 { Ok(failed()) } else { Ok(MpAttempt::success(b"512-result".to_vec())) }
    };
    let first=state.mp_records_with((4,1),refs(&data),mode(false,true,true),&mut evaluate).unwrap();
    assert_eq!(state.qp_cache.entries.len(),1);
    assert!(state.qp_cache.entries.contains_key(&qp_cache_key((4,1),refs(&data),mode(false,true,true),512)));
    let mut calls=Vec::new();
    assert_eq!(state.mp_records_with((4,1),refs(&data),mode(false,true,true),|_,_,_,_,bits| {
        calls.push(bits); assert_eq!(bits,256); Ok(failed())
    }).unwrap(),first);
    assert_eq!(calls,[256]); assert_eq!(state.qp_precision.cache_512,1);
    for check_cache in [(true,true),(false,false)] {
        let mut calls=Vec::new();
        let result=state.mp_records_with((4,1),refs(&data),mode(check_cache.0,check_cache.1,true),|_,_,_,_,bits| {
            calls.push(bits);
            if bits==256 { Ok(failed()) } else { Ok(MpAttempt::success(b"uncached-high".to_vec())) }
        }).unwrap();
        assert_eq!(calls,[256,512]); assert_eq!(result.1,b"uncached-high");
    }
    assert_eq!(state.qp_precision.check_256,1);
    assert_eq!(state.qp_precision.check_512,1);
    // Failure of an independent check invalidates matching primary entries.
    let result=state.mp_records_with((4,1),refs(&data),mode(true,true,true),|_,_,_,_,_| Ok(failed())).unwrap();
    assert_ne!(result.0,SUCCESS); assert!(state.qp_cache.entries.is_empty());
    assert_eq!(state.qp_cache.bytes,0);
}

#[test]
fn cache_budget_input_mode_and_precision_never_alias() {
    let data=inputs(4,"4"); let changed=inputs(4,"9");
    let mut state=State::new();
    let evaluate=|_:&mut State,_,_:KinematicsBatch<Float>,_:Mode,bits| {
        if bits==256 { Ok(failed()) } else { Ok(MpAttempt::success(b"high".to_vec())) }
    };
    for (input,finite) in [(&data,true),(&data,false),(&changed,true)] {
        assert_eq!(state.mp_records_with((4,1),refs(input),mode(false,true,finite),evaluate).unwrap().0,SUCCESS);
    }
    assert_eq!(state.qp_cache.entries.len(),3);
    let used=state.qp_cache.bytes; assert!(used<=DEFAULT_BUDGET/2);
    let low_key=qp_cache_key((4,1),refs(&data),mode(false,true,true),256);
    state.qp_cache.insert(low_key,b"low");
    assert_eq!(state.mp_records_with((4,1),refs(&data),mode(false,true,true),|_,_,_,_,_| panic!("256 hit must return")).unwrap().1,b"low");
    assert_eq!(state.qp_precision.last_bits,256);
    state.qp_cache=QpCache::new(0);
    assert_eq!(state.mp_records_with((4,1),refs(&data),mode(false,true,true),evaluate).unwrap().0,SUCCESS);
    assert!(state.qp_cache.entries.is_empty()); assert_eq!(state.qp_cache.bytes,0);
    assert_eq!(state.qp_cache.not_stored,1);
}

#[test]
fn native_ordinary_256_output_matches_direct_workspace_without_retry() {
    let data=inputs(1,"4"); let mut state=State::new(); let m=mode(false,false,true);
    let batch=mp_batch(1,&data[0],&data[1],&data[2],&data[3]).unwrap();
    let reference=state.mp_at((1,0),batch,m,256).unwrap();
    assert_eq!(reference.status,SUCCESS);
    let actual=state.mp_records((1,0),refs(&data),m).unwrap();
    assert_eq!(actual,(SUCCESS,reference.values));
    assert_eq!(state.qp_precision.evaluations_512,0);
    assert!(state.mp.keys().all(|k|k.2==256));
}

#[cfg(feature="poles-only")]
#[test]
fn native_full_pole_check_tier_workspaces_and_failure_cleanup_are_separate() {
    let data=inputs(1,"4"); let mut state=State::new();
    for finite in [true,false] { for check in [false,true] { for bits in [256,512] {
        let batch=promote_batch(mp_batch(1,&data[0],&data[1],&data[2],&data[3]).unwrap(),bits);
        assert_eq!(state.mp_at((1,0),batch,mode(check,false,finite),bits).unwrap().status,SUCCESS);
    }}}
    assert_eq!(state.mp.len(),4); assert_eq!(state.pole_mp.len(),4);
    let plan=&state.plans[&(1,0)] as *const DirectReductionPlan;
    let outcome=state.mp_records_with((1,0),refs(&data),mode(true,true,true),|_,_,_,_,_| {
        Ok(MpAttempt::failed(&LaneStatus::Unstable("nonfinite direct-engine coefficient".into()),None,1))
    }).unwrap();
    assert_ne!(outcome.0,SUCCESS); assert!(state.mp.is_empty());
    assert_eq!(state.pole_mp.len(),4); assert_eq!(&state.plans[&(1,0)] as *const _,plan);
    assert_eq!(state.mp_records((1,0),refs(&data),mode(false,false,true)).unwrap().0,SUCCESS);
    assert_eq!(state.mp.len(),1); assert_eq!(state.pole_mp.len(),4);
}

#[test]
fn panic_poison_is_not_a_precision_retry() {
    let context=Box::into_raw(Box::new(Context { state: std::sync::Mutex::new(State::new()) }));
    let data=inputs(4,"4");
    let status=guard(|| {
        let mut state=unsafe { &*context }.state.lock().map_err(|_|"poison")?;
        state.mp_records_with((4,1),refs(&data),mode(false,true,true),|_,_,_,_,_| {
            panic!("deliberate private dispatch poison control")
        })?;
        Ok(SUCCESS)
    });
    assert_eq!(status,ERROR);
    assert!(unsafe { &*context }.state.is_poisoned());
    assert_eq!(unsafe { ml_tirust_clear_cache(context) },ERROR);
    assert_eq!(unsafe { ml_tirust_context_destroy(context) },SUCCESS);
}

#[cfg(feature="poles-only")]
#[test]
fn native_recorded_gram_refusal_retries_once_in_full_and_pole_modes() {
    let data: [&[u8];4]=[
        include_bytes!("../fixtures/group10-p.dat"),
        include_bytes!("../fixtures/group10-s.dat"),
        include_bytes!("../fixtures/group10-m.dat"),
        include_bytes!("../fixtures/group10-scale.dat"),
    ];
    let key=(4,3); let mut state=State::new();
    let mut full=None;
    for finite in [true,false] {
        let m=mode(false,true,finite);
        let low=state.mp_at(key,mp_batch(4,data[0],data[1],data[2],data[3]).unwrap(),m,256).unwrap();
        assert_eq!(low.status,UNSTABLE); assert!(low.eligible);
        let high_batch=promote_batch(mp_batch(4,data[0],data[1],data[2],data[3]).unwrap(),512);
        let high=state.mp_at(key,high_batch,m,512).unwrap();
        assert_eq!(high.status,SUCCESS);
        let before=state.qp_precision.statistics();
        let result=state.mp_records(key,data,m).unwrap();
        assert_eq!(result,(SUCCESS,high.values));
        assert_eq!(state.qp_precision.evaluations_256,before[0]+1);
        assert_eq!(state.qp_precision.evaluations_512,before[2]+1);
        assert_eq!(state.qp_precision.last_bits,512);
        // A check pass evaluates both tiers independently despite the primary hit.
        let before=state.qp_precision.statistics();
        let checked=state.mp_records(key,data,mode(true,true,finite)).unwrap();
        assert_eq!(checked,result);
        assert_eq!(state.qp_precision.check_256,before[7]+1);
        assert_eq!(state.qp_precision.check_512,before[8]+1);
        let values=parse_records(&result.1).unwrap();
        if finite { full=Some(values); } else {
            let full=full.as_ref().unwrap(); let pole_values=6*35;
            let tolerance=Float::with_val(256,2).pow(100).inv();
            let magnitude=|v:Float| if v<Float::new(256) {-v} else {v};
            for (a,b) in values[..pole_values].iter().zip(&full[..pole_values]) {
                let difference=magnitude(a.clone()-b);
                let scale=magnitude(a.clone())+magnitude(b.clone())+Float::with_val(256,1);
                assert!(difference <= tolerance.clone()*scale);
            }
            assert!(values[pole_values..].iter().all(|v|v.is_zero()));
        }
    }
    assert_eq!(state.qp_cache.entries.len(),2);
    assert!(state.mp.keys().all(|k|k.2==512));
    assert!(state.pole_mp.keys().all(|k|k.2==512));
}
