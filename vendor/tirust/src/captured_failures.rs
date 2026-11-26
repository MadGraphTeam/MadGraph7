use super::*;
#[test]
fn saved_zggg_failures_replay_recover_and_report_actual_tiers() {
    {
        let key = (5usize, 4u16);
        let check = false;
        let p = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xbd40000000000000u64, 0x407f3ffffffffffeu64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc07f3ffffffffffeu64, 0x406fb9ec476765ffu64, 0x4069542416eddb8cu64, 0x4033463cf2673db0u64, 0xc078bc4efdd54638u64, 0xc0660b3fbc8fc5a5u64, 0xc0600601aa2ed868u64, 0xc06a1468e2d00ac2u64, 0xc0812c925278c7efu64, 0xc06d0bd07fa4dd5eu64, 0xc05720353b51e8e6u64, 0xc06f1b1c7b5a94b2u64, 0xc080aea34bfa3626u64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x0000000000000000u64, 0xc100505f8748eee5u64, 0xc114322aee481f06u64, 0xc1126652221131aau64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40c03da02c0a4a09u64, 0x41181c60f9ddfaf5u64, 0x411c5d899ca70028u64, 0xc100505f8748eee5u64, 0x40c03da02c0a4a09u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0xc114322aee481f06u64, 0x41181c60f9ddfaf5u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc1126652221131aau64, 0x411c5d899ca70028u64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 0 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (6usize, 5u16);
        let check = false;
        let p = [0xbd20000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xbd20000000000000u64, 0x407f400000000002u64, 0xbd20000000000000u64, 0x3d10000000000000u64, 0x407f400000000002u64, 0x408f400000000000u64, 0xbd20000000000000u64, 0x3d10000000000000u64, 0x3d50000000000000u64, 0x4086e2f41fe93756u64, 0x405720353b51e8e6u64, 0x406f1b1c7b5a94b2u64, 0x4040ea34bfa36266u64, 0x408522cfef23f168u64, 0x40600601aa2ed868u64, 0x406a1468e2d00ac2u64, 0x4048c925278c7efcu64, 0x406ec613b89899fcu64, 0xc069542416eddb8cu64, 0xc033463cf2673db0u64, 0xc05a0ec408aae71cu64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x0000000000000000u64, 0x412e848000000000u64, 0x411c5d899ca70028u64, 0x41181c60f9ddfaf5u64, 0x40c03da02c0a4a06u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc10c8a48828f9c57u64, 0xc10c4a1eb840a418u64, 0xc114e1657bdfea4du64, 0x412e848000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40e4ab25ddcc9fbau64, 0x411f7d7dbf1b4befu64, 0x411c5d899ca70028u64, 0xc10c8a48828f9c57u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x41181c60f9ddfaf5u64, 0xc10c4a1eb840a418u64, 0x40e4ab25ddcc9fbau64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40c03da02c0a4a06u64, 0xc114e1657bdfea4du64, 0x411f7d7dbf1b4befu64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 1 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (4usize, 4u16);
        let check = false;
        let p = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xbd20000000000000u64, 0xc06ec613b89899fdu64, 0x4069542416eddb8cu64, 0x4033463cf2673db0u64, 0x405a0ec408aae71au64, 0xc08522cfef23f168u64, 0xc0600601aa2ed868u64, 0xc06a1468e2d00ac2u64, 0xc048c925278c7f00u64, 0xc086e2f41fe93756u64, 0xc05720353b51e8e6u64, 0xc06f1b1c7b5a94b2u64, 0xc040ea34bfa3626au64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x40c03da02c0a4a02u64, 0x41181c60f9ddfaf5u64, 0x411c5d899ca70028u64, 0x40c03da02c0a4a02u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x41181c60f9ddfaf5u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x411c5d899ca70028u64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 2 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (5usize, 5u16);
        let check = false;
        let p = [0x0000000000000000u64, 0x0000000000000000u64, 0x3d20000000000000u64, 0xbd28000000000000u64, 0xc06ec613b89899fdu64, 0x4069542416eddb8cu64, 0x4033463cf2673db0u64, 0x405a0ec408aae71au64, 0xc0800e90ce3cef29u64, 0x4072721f5a4b6800u64, 0x4070c1f20cd3be34u64, 0x406141ef343e4c1eu64, 0xc08d7fdbcf3aba12u64, 0xc041d79c32178fd0u64, 0x40441ace622a27c0u64, 0xc02f7bc19fa472f0u64, 0xc08f400000000000u64, 0x3d20000000000000u64, 0x3d10000000000000u64, 0xbd53000000000000u64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x40c03da02c0a4a01u64, 0x40f4fc68e94b8bf8u64, 0x412b193950beb36bu64, 0x412e848000000000u64, 0x40c03da02c0a4a01u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x411958e46eacdb32u64, 0x411f7d7dbf1b4befu64, 0x40f4fc68e94b8bf8u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x412b193950beb36bu64, 0x411958e46eacdb32u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x412e848000000000u64, 0x411f7d7dbf1b4befu64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 3 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (5usize, 4u16);
        let check = true;
        let p = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x3d40000000000000u64, 0xc06d0bd07fa4dd5eu64, 0xc05720353b51e8e6u64, 0xc06f1b1c7b5a94b2u64, 0xc080aea34bfa3625u64, 0x402ba4338f3bc9f0u64, 0xc072721f5a4b6800u64, 0xc070c1f20cd3be34u64, 0xc083f07bcd0f9308u64, 0x405175a7f811a8b1u64, 0xc074ad12e08e59fau64, 0xc06c7d30811cf278u64, 0xc0846e6ad38e24d1u64, 0x407f3ffffffffffeu64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc07f3ffffffffffcu64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0xc1126652221131adu64, 0xc1214545327f45a6u64, 0xc121d2379190a81cu64, 0x0000000000000000u64, 0xc1126652221131adu64, 0x0000000000000000u64, 0x40c03da02c0a4a01u64, 0x40d33e10f747abe7u64, 0x411c5d899ca70028u64, 0xc1214545327f45a6u64, 0x40c03da02c0a4a01u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0xc121d2379190a81cu64, 0x40d33e10f747abe7u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x411c5d899ca70028u64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 4 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (6usize, 5u16);
        let check = true;
        let p = [0x0000000000000000u64, 0x0000000000000000u64, 0x3d10000000000000u64, 0x0000000000000000u64, 0x406ec613b89899fdu64, 0xc069542416eddb8cu64, 0xc033463cf2673db0u64, 0xc05a0ec408aae71au64, 0x408522cfef23f168u64, 0x40600601aa2ed868u64, 0x406a1468e2d00ac2u64, 0x4048c925278c7f00u64, 0x4086e2f41fe93756u64, 0x405720353b51e8e6u64, 0x406f1b1c7b5a94b2u64, 0x4040ea34bfa3626au64, 0x408f400000000000u64, 0xbd20000000000000u64, 0x3d20000000000000u64, 0x3d52000000000000u64, 0x407f400000000002u64, 0xbd20000000000000u64, 0x3d20000000000000u64, 0x407f400000000002u64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x40c03da02c0a4a06u64, 0x41181c60f9ddfaf5u64, 0x411c5d899ca70028u64, 0x412e848000000000u64, 0x0000000000000000u64, 0x40c03da02c0a4a06u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x411f7d7dbf1b4befu64, 0xc114e1657bdfea4du64, 0x41181c60f9ddfaf5u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40e4ab25ddcc9fbcu64, 0xc10c4a1eb840a417u64, 0x411c5d899ca70028u64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc10c8a48828f9c56u64, 0x412e848000000000u64, 0x411f7d7dbf1b4befu64, 0x40e4ab25ddcc9fbcu64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0xc114e1657bdfea4du64, 0xc10c4a1eb840a417u64, 0xc10c8a48828f9c56u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 5 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (4usize, 4u16);
        let check = true;
        let p = [0x3d20000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x3d20000000000000u64, 0xc086e2f41fe93756u64, 0xc05720353b51e8e6u64, 0xc06f1b1c7b5a94b2u64, 0xc040ea34bfa36266u64, 0xc08522cfef23f168u64, 0xc0600601aa2ed868u64, 0xc06a1468e2d00ac2u64, 0xc048c925278c7efcu64, 0xc06ec613b89899fcu64, 0x4069542416eddb8cu64, 0x4033463cf2673db0u64, 0x405a0ec408aae71cu64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x411c5d899ca70028u64, 0x41181c60f9ddfaf5u64, 0x40c03da02c0a4a06u64, 0x411c5d899ca70028u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x41181c60f9ddfaf5u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40c03da02c0a4a06u64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 6 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
    {
        let key = (5usize, 5u16);
        let check = true;
        let p = [0xbd20000000000000u64, 0x0000000000000000u64, 0x3d20000000000000u64, 0x3d28000000000000u64, 0xc08f400000000000u64, 0x3d20000000000000u64, 0xbd10000000000000u64, 0xbd50000000000000u64, 0xc08d7fdbcf3aba12u64, 0xc041d79c32178fd0u64, 0x40441ace622a27bcu64, 0xc02f7bc19fa472d8u64, 0xc0800e90ce3cef29u64, 0x4072721f5a4b6800u64, 0x4070c1f20cd3be34u64, 0x406141ef343e4c20u64, 0xc06ec613b89899feu64, 0x4069542416eddb8cu64, 0x4033463cf2673db8u64, 0x405a0ec408aae71du64].map(f64::from_bits);
        let s = [0x0000000000000000u64, 0x412e848000000000u64, 0x412b193950beb36bu64, 0x40f4fc68e94b8bfau64, 0x40c03da02c0a4a06u64, 0x412e848000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40ec79a4a5a6e624u64, 0x411f7d7dbf1b4bedu64, 0x412b193950beb36bu64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x411958e46eacdb30u64, 0x40f4fc68e94b8bfau64, 0x40ec79a4a5a6e624u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x40c03da02c0a4a06u64, 0x411f7d7dbf1b4bedu64, 0x411958e46eacdb30u64, 0x0000000000000000u64, 0x0000000000000000u64].map(f64::from_bits);
        let m_bits = [0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64, 0x0000000000000000u64];
        let masses: Vec<_> = m_bits.chunks_exact(2).map(|v| C64 { re: f64::from_bits(v[0]), im: f64::from_bits(v[1]) }).collect();
        let mu2 = f64::from_bits(0x412e848000000000);
        let mut state = State::new();
        let mode = Mode { poles: 0, finite: true, check, cache: true };
        let batch = || dp_batch(key.0, &p, &s, &masses, mu2).unwrap();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        println!("captured fixture 7 N={} rank={} check={} status={} counts={:?}", key.0, key.1, check, first.0, state.dp_routes);
        assert_eq!(first.0, SUCCESS);
        assert!(first.1.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        let offset = usize::from(check) * DP_ROUTE_FIELDS;
        assert_eq!(state.dp_routes[offset + 2..offset + 5].iter().sum::<u64>(), 1);
        assert!(state.dp_routes[offset + 7..offset + 10].iter().sum::<u64>() >= 1);
        // A rescued success keeps the store: its entries are exact pure functions of their keys.
        let retained = state.store.statistics().entries;
        let other = usize::from(!check) * DP_ROUTE_FIELDS;
        assert!(state.dp_routes[other..other+DP_ROUTE_FIELDS].iter().all(|&x| x==0));
        let second = state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap();
        assert_eq!(second, first);
        assert_eq!(state.store.statistics().entries, retained);
    }
}
#[test]
fn route_counters_require_finite_returned_success_and_separate_checks() {
    let mut state = State::new();
    let key=(1,0);
    let evaluate=|state: &mut State, check, mass| state.dp_checked(key,
        dp_batch(1, &[0.;4], &[0.], &[C64{re:mass,im:0.}], 4.).unwrap(),
        Mode { poles:0, finite:true, check, cache:true }).unwrap();
    assert_eq!(evaluate(&mut state,false,4.).0,SUCCESS);
    assert_eq!(state.dp_routes[0],1);
    assert_eq!(evaluate(&mut state,true,4.).0,SUCCESS);
    assert_eq!(state.dp_routes[10],1);
    let before=state.dp_routes;
    assert_eq!(evaluate(&mut state,false,1e308).0,UNSTABLE);
    assert_eq!(state.dp_routes[5],before[5]+1);
    assert_eq!(&state.dp_routes[..5],&before[..5]);
    assert_eq!(&state.dp_routes[10..],&before[10..]);
    assert_eq!(state.store.statistics().entries,0);
    assert_eq!(evaluate(&mut state,false,4.).0,SUCCESS);
}

#[test]
fn public_route_statistics_validate_count_and_preserve_on_cache_clear() {
    let mut c=std::ptr::null_mut();
    assert_eq!(unsafe { ml_tirust_context_create(ABI_VERSION,&mut c) },SUCCESS);
    let mut counts=[77;DP_ROUTE_STATS_COUNT];
    assert_eq!(unsafe { ml_tirust_dp_route_statistics(c,counts.as_mut_ptr(),DP_ROUTE_STATS_COUNT-1) },ERROR);
    assert_eq!(counts,[77;DP_ROUTE_STATS_COUNT]);
    let mut out=[C64::default();4];
    assert_eq!(unsafe { ml_tirust_evaluate_dp(c,1,0,[0.;4].as_ptr(),4,[0.].as_ptr(),1,
        [C64{re:4.,im:0.}].as_ptr(),1,4.,out.as_mut_ptr(),4,0,1,0,1) },SUCCESS);
    assert_eq!(unsafe { ml_tirust_dp_route_statistics(c,counts.as_mut_ptr(),DP_ROUTE_STATS_COUNT) },SUCCESS);
    assert_eq!(counts[0],1);
    let before=counts;
    assert_eq!(unsafe { ml_tirust_clear_cache(c) },SUCCESS);
    assert_eq!(unsafe { ml_tirust_dp_route_statistics(c,counts.as_mut_ptr(),DP_ROUTE_STATS_COUNT) },SUCCESS);
    assert_eq!(counts,before);
    assert_eq!(unsafe { ml_tirust_context_destroy(c) },SUCCESS);
}
