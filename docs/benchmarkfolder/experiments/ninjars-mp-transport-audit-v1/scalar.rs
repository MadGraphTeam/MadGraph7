use ninjars::{Float, Complex, FloatLike, Real, SingleFloat};
use symbolica::domains::rational::Rational;
use std::io::Write;
type Result<T> = std::result::Result<T,String>;
const RECORD:usize=64;
fn parse_old(text: &[u8], precision: u32) -> Result<Float> {
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
fn parse_proposed(text: &[u8], precision: u32) -> Result<Float> {
    let text = std::str::from_utf8(text)
        .map_err(|_| "MP input is not ASCII")?
        .trim();
    let mut input = Float::parse(text, Some(113)).map_err(|e| format!("invalid MP value: {e}"))?;
    if !input.is_finite() {
        return Err("nonfinite MP input".into());
    }
    // Forty significant decimal digits round-trip binary128. First recover that
    // value at 113 bits, then widen the represented value itself. A rational
    // round-trip would discard the sign of zero, which selects complex branch lips.
    input.set_prec(precision);
    Ok(input)
}
const CASES:&[(&str,&str,i128,i32,bool,&str,&str)]=&[
    ("+zero", "0.000000000000000000000000000000000000000E+39", 0_i128, 0, false, "0000000000000000", "0000000000000000"),
    ("-zero", "-0.000000000000000000000000000000000000000E+39", 0_i128, 0, true, "0000000000000000", "8000000000000000"),
    ("+min_subnormal", "6.475175119438025110924438958227646552500E-4966", 1_i128, -16494, false, "0000000000000001", "0000000000000000"),
    ("-min_subnormal", "-6.475175119438025110924438958227646552500E-4966", 1_i128, -16494, true, "0000000000000001", "8000000000000000"),
    ("+max_subnormal", "3.362103143112093506262677817321751955081E-4932", 5192296858534827628530496329220095_i128, -16494, false, "FFFFFFFFFFFFFFFF", "0000FFFFFFFFFFFF"),
    ("-max_subnormal", "-3.362103143112093506262677817321751955081E-4932", 5192296858534827628530496329220095_i128, -16494, true, "FFFFFFFFFFFFFFFF", "8000FFFFFFFFFFFF"),
    ("+min_normal", "3.362103143112093506262677817321752602598E-4932", 1_i128, -16382, false, "0000000000000000", "0001000000000000"),
    ("-min_normal", "-3.362103143112093506262677817321752602598E-4932", 1_i128, -16382, true, "0000000000000000", "8001000000000000"),
    ("+max_finite", "1.189731495357231765085759326628007016196E+4932", 10384593717069655257060992658440191_i128, 16271, false, "FFFFFFFFFFFFFFFF", "7FFEFFFFFFFFFFFF"),
    ("-max_finite", "-1.189731495357231765085759326628007016196E+4932", 10384593717069655257060992658440191_i128, 16271, true, "FFFFFFFFFFFFFFFF", "FFFEFFFFFFFFFFFF"),
    ("+beyond_dp", "1.000000000000000000000000000000788860905E+0", 1267650600228229401496703205377_i128, -100, false, "0000000000001000", "3FFF000000000000"),
    ("-beyond_dp", "-1.000000000000000000000000000000788860905E+0", 1267650600228229401496703205377_i128, -100, true, "0000000000001000", "BFFF000000000000"),
    ("+last_binary128_bit", "1.000000000000000000000000000000000192593E+0", 5192296858534827628530496329220097_i128, -112, false, "0000000000000001", "3FFF000000000000"),
    ("-last_binary128_bit", "-1.000000000000000000000000000000000192593E+0", 5192296858534827628530496329220097_i128, -112, true, "0000000000000001", "BFFF000000000000"),
];

fn record(text:&str)->[u8;RECORD] { let mut r=[b' ';RECORD]; assert!(text.len()<=RECORD);r[..text.len()].copy_from_slice(text.as_bytes());r }
fn main() {
    let args:Vec<String>=std::env::args().collect();
    assert_eq!(args.len(),3,"usage: scalar old|proposed OUTPUT-RECORDS");
    let parse:fn(&[u8],u32)->Result<Float>=match args[1].as_str() { "old"=>parse_old, "proposed"=>parse_proposed,_=>panic!("unknown arm") };
    let mut output=std::fs::OpenOptions::new().write(true).create_new(true).open(&args[2]).unwrap();
    let mut value_fail=0;let mut sign_fail=0;let mut precision_fail=0;let mut format_fail=0;let mut branch_fail=0;let mut reject_fail=0;
    let mut tests=0;
    for (index,&(name,text,significand,exponent,negative,low,high)) in CASES.iter().enumerate() {
        let scale=Rational::from(2).pow(u64::from(exponent.unsigned_abs()));
        let mut expected=Rational::from(significand);
        expected=if exponent<0 {expected/scale} else {expected*scale};
        if negative {expected=-expected;}
        for precision in [256,384] {
            let value=parse(&record(text),precision).unwrap();
            let value_ok=value.to_rational()==expected;
            let sign_ok=value.is_sign_negative()==negative;
            let precision_ok=value.prec()==precision;
            let formatted=format!("{value:.39e}");
            let back=Float::parse(&formatted,Some(113)).unwrap();
            let format_ok=formatted.len()<=RECORD && back.to_rational()==expected && back.is_sign_negative()==negative;
            value_fail+=usize::from(!value_ok);sign_fail+=usize::from(!sign_ok);precision_fail+=usize::from(!precision_ok);format_fail+=usize::from(!format_ok);
            println!("CASE {name} precision={precision} exact_value={value_ok} sign={sign_ok} precision_match={precision_ok} output_roundtrip={format_ok}");
            if precision==256 { writeln!(output,"{} {} {}",index+1,low,high).unwrap();writeln!(output,"{formatted:64}").unwrap(); }
            tests+=1;
        }
    }
    for precision in [256,384] {
        for text in ["NaN","nan","+inf","-inf","not-a-number"] { reject_fail+=usize::from(parse(&record(text),precision).is_ok()); }
        let re=parse(&record("-1"),precision).unwrap();
        let upper=Complex::new(re.clone(),parse(&record("0"),precision).unwrap());
        let lower=Complex::new(re,parse(&record("-0"),precision).unwrap());
        let branch_ok=upper.sqrt().im==Float::with_val(precision,1) && lower.sqrt().im==Float::with_val(precision,-1)
            && !upper.log().im.is_sign_negative() && lower.log().im.is_sign_negative();
        branch_fail+=usize::from(!branch_ok);
        println!("BRANCH precision={precision} preserved={branch_ok}");
    }
    println!("SUMMARY finite_cases={tests} value_fail={value_fail} sign_fail={sign_fail} precision_fail={precision_fail} format_fail={format_fail} branch_fail={branch_fail} reject_fail={reject_fail}");
    if value_fail+sign_fail+precision_fail+format_fail+branch_fail+reject_fail!=0 {std::process::exit(1)}
}
