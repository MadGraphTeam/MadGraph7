//! Linux x86_64 glibc, binary64 quotient/filter fixture only.
//! The harness resets state BETWEEN measurements; neither extracted function
//! contains any floating-point environment call.
include!("quotient.rs");
use std::{hint::black_box, io::{self, BufRead, Write}};

#[link(name = "m")]
unsafe extern "C" {
    fn feclearexcept(excepts: i32) -> i32;
    fn feraiseexcept(excepts: i32) -> i32;
    fn fetestexcept(excepts: i32) -> i32;
    fn fegetround() -> i32;
    fn fesetround(rounding: i32) -> i32;
    fn feenableexcept(excepts: i32) -> i32;
    fn fedisableexcept(excepts: i32) -> i32;
    fn fegetexcept() -> i32;
}
const ALL: i32 = 61;

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 8);
        let id = fields[0];
        let mode = fields[1];
        assert!(mode == "baseline" || mode == "candidate");
        let nb = u64::from_str_radix(fields[2], 16).unwrap();
        let db = u64::from_str_radix(fields[3], 16).unwrap();
        let seed: i32 = fields[4].parse().unwrap();
        let rounding: i32 = fields[5].parse().unwrap();
        let traps: i32 = fields[6].parse().unwrap();
        let enter: i32 = fields[7].parse().unwrap();
        unsafe {
            assert!(fedisableexcept(ALL) >= 0);
            assert_eq!(feclearexcept(ALL), 0);
            assert_eq!(fesetround(rounding), 0);
            if seed != 0 { assert_eq!(feraiseexcept(seed), 0); }
            if traps != 0 { assert!(feenableexcept(traps) >= 0); }
        }
        let before = unsafe { fetestexcept(ALL) };
        let before_round = unsafe { fegetround() };
        let before_traps = unsafe { fegetexcept() };
        if enter != 0 {
            println!("ENTER {id} {mode} {nb:016x} {db:016x} {before} {before_round} {before_traps}");
            io::stdout().flush().unwrap();
        }
        let n = black_box(f64::from_bits(nb));
        let d = black_box(f64::from_bits(db));
        let result = black_box(if mode == "baseline" { baseline(n, d) } else { candidate(n, d) });
        let after = unsafe { fetestexcept(ALL) };
        let after_round = unsafe { fegetround() };
        let after_traps = unsafe { fegetexcept() };
        unsafe { assert!(fedisableexcept(ALL) >= 0); }
        let (tag, bits) = match result { Some(v) => ("some", v.to_bits()), None => ("none", 0) };
        println!("ROW {id} {mode} {nb:016x} {db:016x} {seed} {rounding} {traps} {before} {after} {before_round} {after_round} {before_traps} {after_traps} {tag} {bits:016x}");
    }
}
