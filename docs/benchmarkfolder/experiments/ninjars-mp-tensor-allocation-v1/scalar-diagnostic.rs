use ninjars::{Complex, Float, FloatLike};

fn same(a: &Float, b: &Float) -> bool {
    a.prec() == b.prec() && a.is_sign_negative() == b.is_sign_negative()
        && a.is_finite() == b.is_finite()
        && if a.is_finite() { a.to_rational() == b.to_rational() } else { format!("{a:?}") == format!("{b:?}") }
}

fn product(a: &Complex<Float>, b: &Complex<Float>) -> Complex<Float> {
    Complex::new(a.re.clone()*&b.re-a.im.clone()*&b.im,
                 a.re.clone()*&b.im+a.im.clone()*&b.re)
}

fn main() {
    let mut products = 0;
    let mut resets = 0;
    let mut arbitrary_nan_mismatches = 0;
    for bits in [2,53,113,256,384] {
        let values: Vec<_> = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0, 0.0, -3.0, 4.0].into_iter().map(|v| Complex::new(Float::with_val(bits,v),Float::with_val(bits,-0.0))).collect();
        for a in &values {
            for b in &values {
                let x=product(a,b);let y=a.clone()*b;
                assert!(same(&x.re,&y.re)&&same(&x.im,&y.im));products+=1;
                let mut slot=a.clone();slot.set_from(b);let cloned=b.clone();
                if !(same(&slot.re,&cloned.re)&&same(&slot.im,&cloned.im)) {
                    assert!(!b.re.is_finite() && format!("{:?}",b.re).to_lowercase().contains("nan"));
                    arbitrary_nan_mismatches+=1;
                    println!("arbitrary NaN source: bits={bits}, previous={a:?}, set_from sign={}, clone sign={}",slot.re.is_sign_negative(),cloned.re.is_sign_negative());
                }
            }
            for reset in [Complex::new(Float::new(bits),Float::new(bits)),Complex::new(Float::with_val(bits,1),Float::new(bits))] {
                let mut slot=a.clone();slot.set_from(&reset);let cloned=reset.clone();
                assert!(same(&slot.re,&cloned.re)&&same(&slot.im,&cloned.im));resets+=1;
            }
        }
    }
    assert!(arbitrary_nan_mismatches>0);
    println!("exact_products={products} exact_actual_zero_one_resets={resets} unrelated_arbitrary_nan_copy_mismatches={arbitrary_nan_mismatches}");
}
