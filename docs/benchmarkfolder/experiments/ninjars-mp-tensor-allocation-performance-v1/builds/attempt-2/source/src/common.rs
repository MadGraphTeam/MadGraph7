use ninjars::{Complex, Float, FloatLike, FourMomentum};
use serde_json::{Value, json};
pub type Z = Complex<Float>;
pub type Q = FourMomentum<Z>;

pub fn same_float(a: &Float, b: &Float) {
    assert_eq!(a.prec(), b.prec());
    assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    assert!(a.is_finite() && b.is_finite());
    assert_eq!(a.to_rational(), b.to_rational());
}
pub fn same(a: &[Z], b: &[Z]) {
    assert_eq!(a.len(), b.len());
    for (a,b) in a.iter().zip(b) { same_float(&a.re,&b.re); same_float(&a.im,&b.im); }
}
pub fn float_json(a: &Float) -> Value {
    assert!(a.is_finite());
    let digits = (a.prec() as usize * 30103).div_ceil(100000) + 4;
    let decimal = format!("{a:.digits$e}");
    let restored = Float::parse(&decimal, Some(a.prec())).unwrap();
    same_float(a,&restored);
    json!({"precision":a.prec(), "negative":a.is_sign_negative(), "decimal":decimal})
}
pub fn z_json(a: &Z) -> Value { json!([float_json(&a.re),float_json(&a.im)]) }
pub fn zs_json(a: &[Z]) -> Value { Value::Array(a.iter().map(z_json).collect()) }
pub fn q_json(q: &Q) -> Value { Value::Array(q.components().iter().map(|v| z_json(v)).collect()) }
pub fn read_float(v: &Value) -> Float {
    let precision = u32::try_from(v["precision"].as_u64().unwrap()).unwrap();
    assert!((1..=4096).contains(&precision));
    let a=Float::parse(v["decimal"].as_str().unwrap(), Some(precision)).unwrap();
    assert!(a.is_finite()); assert_eq!(a.prec(),precision);
    assert_eq!(a.is_sign_negative(),v["negative"].as_bool().unwrap()); a
}
pub fn read_z(v: &Value) -> Z { assert_eq!(v.as_array().unwrap().len(),2); Z::new(read_float(&v[0]),read_float(&v[1])) }
pub fn read_zs(v: &Value) -> Vec<Z> { v.as_array().unwrap().iter().map(read_z).collect() }
pub fn read_q(v: &Value) -> Q { assert_eq!(v.as_array().unwrap().len(),4); Q::from(std::array::from_fn(|i|read_z(&v[i]))) }
