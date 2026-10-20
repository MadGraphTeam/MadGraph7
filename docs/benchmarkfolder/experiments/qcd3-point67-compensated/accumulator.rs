// Diagnostic compensated accumulation; kept outside the production bridge.
#[derive(Clone, Copy, Default)]
struct AccurateSum { sum: f64, correction: f64 }
impl AccurateSum {
    fn add(&mut self, x: f64) {
        let next = self.sum + x;
        if !next.is_finite() {
            self.sum = next;
            self.correction = 0.;
            return;
        }
        self.correction += if self.sum.abs() >= x.abs() {
            (self.sum - next) + x
        } else {
            (x - next) + self.sum
        };
        self.sum = next;
    }
    fn product(&mut self, a: f64, b: f64, sign: f64) {
        let hi = a * b;
        let lo = a.mul_add(b, -hi);
        self.add(sign * hi);
        self.add(sign * lo);
    }
    fn value(self) -> f64 { self.sum + self.correction }
}
#[derive(Clone, Copy, Default)]
struct AccurateComplex { re: AccurateSum, im: AccurateSum }
impl AccurateComplex {
    fn add_product(&mut self, a: C64, b: C64, products: bool) {
        if products {
            self.re.product(a.re, b.re, 1.);
            self.re.product(a.im, b.im, -1.);
            self.im.product(a.re, b.im, 1.);
            self.im.product(a.im, b.re, 1.);
        } else { *self += a * b; }
    }
    fn value(self) -> C64 { C64::new(self.re.value(), self.im.value()) }
}
impl std::ops::AddAssign<C64> for AccurateComplex {
    fn add_assign(&mut self, z: C64) { self.re.add(z.re); self.im.add(z.im); }
}
