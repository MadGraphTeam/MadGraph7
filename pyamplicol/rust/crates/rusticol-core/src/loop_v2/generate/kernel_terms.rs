// SPDX-License-Identifier: 0BSD

//! The exact kernels' term reading in parallel (M-FINAL MF-RUST, HANDOFF
//! §0.4).
//!
//! The twin of `loops/v2/reference_executor.py`'s `_kernel_terms` without
//! its final key assembly: each canonical string of a prepared kernel's
//! `exact_expressions` is parsed, expanded and read term by term into an
//! exact complex coefficient, the powers of its model factors (parameters,
//! primes, pi and parameter-only sums or functions kept whole), the bases a
//! fractional power was distributed over and the monomial in the kernel's
//! kinematic inputs (`_TermReading.read`). Python assembles each term's key
//! from these readings with the oracle's own code (`_term_key`), so the
//! oracle (`reference_executor.native_kernel_terms(False)`) and this stage
//! share every rule past the reading.
//!
//! This crate builds the Symbolica revision of the Python wheel, so the
//! parse, the expansion, the powers `atom^(p/q)` and every canonical string
//! are the oracle's own operations; only the order in which a term's factors
//! are visited may differ (it follows each process's symbol ids), which no
//! output depends on (powers and monomials are summed, the coefficient is a
//! product). Every output of every kernel is an independent task on a rayon
//! pool whose workers hold pyAmpliCol's Symbolica key; results come back in
//! input order. The opaque factors of an output (a coupling's sum recurs in
//! many of its terms) are read once.

use rayon::prelude::*;
use std::collections::HashMap;
use symbolica::atom::{Atom, AtomCore, AtomView, Symbol};
use symbolica::coefficient::Coefficient;
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;
use symbolica::parser::ParseSettings;

use crate::{RusticolError, RusticolResult};

/// A model factor of a kernel coefficient (`reference_executor.Factor`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    /// `("param", name)`: a model parameter.
    Param(String),
    /// `("prime", p)`: a prime of a fractional power of a rational.
    Prime(Integer),
    /// `("pi", None)`.
    Pi,
    /// `("expr", (text, pairs))`: a parameter-only sum or function kept
    /// whole, its canonical string and its (symbol, parameter name) pairs.
    Expr(String, Vec<(String, String)>),
}

/// One product term of a kernel output (`_TermReading` after `read`).
#[derive(Clone, Debug, PartialEq)]
pub struct TermReading {
    /// The exact complex coefficient (real, imaginary).
    pub coefficient: (Rational, Rational),
    /// The summed exponent of every model factor met, in first-met order
    /// (zero sums included, as the oracle's `defaultdict`).
    pub powers: Vec<(Base, Rational)>,
    /// The bases a fractional power was distributed over, first-met order.
    pub positive: Vec<Base>,
    /// (input position, power), first-met order.
    pub monomial: Vec<(usize, i64)>,
}

/// One kernel: its outputs' canonical strings, its kinematic inputs
/// (canonical symbol, input position) and its model-parameter inputs
/// (canonical symbol, parameter name).
#[derive(Clone, Debug, Default)]
pub struct KernelTermsJob {
    pub expressions: Vec<String>,
    pub positions: Vec<(String, usize)>,
    pub parameters: Vec<(String, String)>,
}

/// Per kernel: the term readings of every output, or the oracle's refusal
/// (`NotImplementedError` and its message).
pub type KernelTermsResult = Result<Vec<Vec<TermReading>>, String>;

fn parse(text: &str) -> RusticolResult<Atom> {
    // Canonical strings name every symbol with its namespace; the default
    // namespace is the Python wheel's (`E`).
    Atom::parse(text, "python", ParseSettings::default()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "kernel terms: cannot parse a canonical expression: {error}"
        ))
    })
}

/// What a variable symbol is to a kernel.
#[derive(Clone)]
enum Role {
    Position(usize),
    Parameter(String),
}

struct Kernel {
    roles: HashMap<Symbol, Role>,
    /// Canonical strings of the kernel's input symbols (`_opaque`'s checks).
    positions: HashMap<String, usize>,
    parameters: HashMap<String, String>,
}

impl Kernel {
    /// `_TermReading._opaque` up to the powers: the factor's canonical
    /// string and its (symbol, parameter name) pairs, or its refusal.
    fn opaque(&self, atom: AtomView<'_>) -> Opaque {
        let mut symbols: Vec<String> = atom
            .get_all_symbols(false)
            .into_iter()
            .filter(|&symbol| symbol != Symbol::PI)
            .map(|symbol| Atom::var(symbol).to_canonical_string())
            .collect();
        symbols.sort();
        if symbols.iter().any(|s| self.positions.contains_key(s)) {
            return Err(format!("kernel not polynomial in {atom}"));
        }
        let unknown: Vec<&String> = symbols
            .iter()
            .filter(|s| !self.parameters.contains_key(*s))
            .collect();
        if !unknown.is_empty() {
            // Python's list repr of the names.
            let names: Vec<String> = unknown.iter().map(|s| format!("'{s}'")).collect();
            return Err(format!(
                "unknown symbols [{}] in a kernel",
                names.join(", ")
            ));
        }
        let pairs = symbols
            .iter()
            .map(|s| (s.clone(), self.parameters[s].clone()))
            .collect();
        Ok((atom.to_canonical_string(), pairs))
    }

    fn new(job: &KernelTermsJob) -> RusticolResult<Self> {
        let mut roles = HashMap::new();
        let mut positions = HashMap::new();
        let mut parameters = HashMap::new();
        // The oracle's dicts: a later entry of a repeated symbol wins.
        for (text, position) in &job.positions {
            positions.insert(text.clone(), *position);
        }
        for (text, name) in &job.parameters {
            parameters.insert(text.clone(), name.clone());
        }
        // A symbol's canonical string is its full name: it identifies it.
        for (text, position) in &positions {
            if let Atom::Var(v) = parse(text)? {
                roles.insert(v.get_symbol(), Role::Position(*position));
            }
        }
        for (text, name) in &parameters {
            if positions.contains_key(text) {
                continue; // the oracle tests the positions first
            }
            if let Atom::Var(v) = parse(text)? {
                roles.insert(v.get_symbol(), Role::Parameter(name.clone()));
            }
        }
        Ok(Self {
            roles,
            positions,
            parameters,
        })
    }
}

/// An opaque factor's canonical string and (symbol, parameter) pairs, or
/// its refusal.
type Opaque = Result<(String, Vec<(String, String)>), String>;

/// The opaque factors met by the terms of one output (a coupling's sum
/// recurs in many terms): their readings and fractional powers.
#[derive(Default)]
struct OpaqueCache {
    factors: HashMap<Atom, Opaque>,
    raised: HashMap<(Atom, String), String>,
}

struct Reading<'k, 'c> {
    kernel: &'k Kernel,
    cache: &'c mut OpaqueCache,
    re: Rational,
    im: Rational,
    powers: Vec<(Base, Rational)>,
    power_index: HashMap<Base, usize>,
    positive: Vec<Base>,
    monomial: Vec<(usize, i64)>,
}

fn exact_number(view: AtomView<'_>) -> Result<(Rational, Rational), String> {
    let AtomView::Num(n) = view else {
        unreachable!("a number atom");
    };
    match n.get_coeff_view().to_owned() {
        Coefficient::Complex(c) => Ok((c.re, c.im)),
        _ => Err(format!(
            "kernel coefficient {view} is not an exact complex rational"
        )),
    }
}

impl<'k, 'c> Reading<'k, 'c> {
    fn new(kernel: &'k Kernel, cache: &'c mut OpaqueCache) -> Self {
        Self {
            kernel,
            cache,
            re: Rational::one(),
            im: Rational::zero(),
            powers: Vec::new(),
            power_index: HashMap::new(),
            positive: Vec::new(),
            monomial: Vec::new(),
        }
    }

    fn add_power(&mut self, base: Base, exponent: &Rational) {
        match self.power_index.get(&base) {
            Some(&index) => {
                let sum = &self.powers[index].1 + exponent;
                self.powers[index].1 = sum;
            }
            None => {
                self.power_index.insert(base.clone(), self.powers.len());
                self.powers.push((base, exponent.clone()));
            }
        }
    }

    fn add_positive(&mut self, base: Base) {
        if !self.positive.contains(&base) {
            self.positive.push(base);
        }
    }

    /// `_TermReading.read`: add `atom^exponent`; `distributed` marks a
    /// fractional power already pushed through a product or a power.
    fn read(
        &mut self,
        atom: AtomView<'_>,
        exponent: &Rational,
        distributed: bool,
    ) -> Result<(), String> {
        match atom {
            AtomView::Mul(m) => {
                let fractional = !exponent.is_integer();
                for factor in m.iter() {
                    self.read(factor, exponent, distributed || fractional)?;
                }
                Ok(())
            }
            AtomView::Pow(p) => {
                let (base, power) = p.get_base_exp();
                if !matches!(power, AtomView::Num(_)) {
                    return Err(format!("symbolic exponent in {atom}"));
                }
                let (re, im) = exact_number(power)?;
                if !im.is_zero() {
                    return Err(format!("complex exponent in {atom}"));
                }
                let fractional = !exponent.is_integer();
                self.read(base, &(exponent * &re), distributed || fractional)
            }
            AtomView::Num(_) => {
                let number = exact_number(atom)?;
                self.number(number, exponent, atom)
            }
            AtomView::Var(v) => {
                let symbol = v.get_symbol();
                match self.kernel.roles.get(&symbol) {
                    Some(Role::Position(position)) => {
                        if distributed || !exponent.is_integer() || exponent < &Rational::one() {
                            return Err(format!(
                                "kernel not polynomial in {}",
                                atom.to_canonical_string()
                            ));
                        }
                        let power = exponent
                            .numerator()
                            .to_i64()
                            .ok_or_else(|| format!("power {exponent} of an input is too large"))?;
                        match self.monomial.iter_mut().find(|(p, _)| p == position) {
                            Some(entry) => entry.1 += power,
                            None => self.monomial.push((*position, power)),
                        }
                        Ok(())
                    }
                    Some(Role::Parameter(name)) => {
                        let base = Base::Param(name.clone());
                        self.add_power(base.clone(), exponent);
                        if distributed {
                            self.add_positive(base);
                        }
                        Ok(())
                    }
                    None if symbol == Symbol::PI => {
                        self.add_power(Base::Pi, exponent);
                        Ok(())
                    }
                    None => Err(format!(
                        "unknown symbol {} in a kernel",
                        atom.to_canonical_string()
                    )),
                }
            }
            _ => self.opaque(atom, exponent, distributed),
        }
    }

    /// `_TermReading._opaque`: a parameter-only sum or function, kept whole.
    fn opaque(
        &mut self,
        atom: AtomView<'_>,
        exponent: &Rational,
        distributed: bool,
    ) -> Result<(), String> {
        let owned = atom.to_owned();
        if !self.cache.factors.contains_key(&owned) {
            let found = self.kernel.opaque(atom);
            self.cache.factors.insert(owned.clone(), found);
        }
        let (text, pairs) = self.cache.factors[&owned].clone()?;
        if distributed {
            self.add_positive(Base::Expr(text.clone(), pairs.clone()));
        }
        if exponent.is_integer() {
            self.add_power(Base::Expr(text, pairs), exponent);
        } else {
            let power = format!("{}/{}", exponent.numerator(), exponent.denominator());
            let key = (owned, power);
            let raised = match self.cache.raised.get(&key) {
                Some(raised) => raised.clone(),
                None => {
                    let power = parse(&key.1).map_err(|e| e.to_string())?;
                    let raised = key.0.pow(&power).to_canonical_string();
                    self.cache.raised.insert(key, raised.clone());
                    raised
                }
            };
            self.add_power(Base::Expr(raised, pairs), &Rational::one());
        }
        Ok(())
    }

    /// `_TermReading._number`.
    fn number(
        &mut self,
        (re, im): (Rational, Rational),
        exponent: &Rational,
        atom: AtomView<'_>,
    ) -> Result<(), String> {
        if exponent.is_integer() {
            let (mut fr, mut fi) = (re, im);
            if exponent.is_negative() {
                let norm = &(&fr * &fr) + &(&fi * &fi);
                if norm.is_zero() {
                    return Err(format!("negative power of zero in {atom}"));
                }
                fr = &fr / &norm;
                fi = -(&fi / &norm);
            }
            let count = exponent
                .numerator()
                .abs()
                .to_i64()
                .ok_or_else(|| format!("power {exponent} of a number is too large"))?;
            for _ in 0..count {
                let real = &(&self.re * &fr) - &(&self.im * &fi);
                let imag = &(&self.re * &fi) + &(&self.im * &fr);
                self.re = real;
                self.im = imag;
            }
            return Ok(());
        }
        if !im.is_zero() || re.is_negative() || re.is_zero() {
            return Err(format!("fractional power of {atom}"));
        }
        for (value, sign) in [(re.numerator(), 1i64), (re.denominator(), -1i64)] {
            if value.is_one() {
                continue;
            }
            for (prime, count) in value.factor() {
                let step = exponent * &Rational::from(count) * &Rational::from(sign);
                self.add_power(Base::Prime(prime), &step);
            }
        }
        Ok(())
    }

    fn finish(self) -> TermReading {
        TermReading {
            coefficient: (self.re, self.im),
            powers: self.powers,
            positive: self.positive,
            monomial: self.monomial,
        }
    }
}

/// The term readings of one output, or the refusal of its first refused term.
fn output_terms(kernel: &Kernel, text: &str) -> RusticolResult<Result<Vec<TermReading>, String>> {
    let expanded = parse(text)?.expand();
    let terms: Vec<AtomView<'_>> = match expanded.as_view() {
        AtomView::Add(a) => a.iter().collect(),
        other => vec![other],
    };
    let mut readings = Vec::with_capacity(terms.len());
    let mut cache = OpaqueCache::default();
    for term in terms {
        let mut reading = Reading::new(kernel, &mut cache);
        if let Err(message) = reading.read(term, &Rational::one(), false) {
            return Ok(Err(message));
        }
        readings.push(reading.finish());
    }
    Ok(Ok(readings))
}

fn pool(threads: Option<usize>) -> RusticolResult<rayon::ThreadPool> {
    // Symbolica state is touched on the workers: they hold the key.
    let mut builder =
        rayon::ThreadPoolBuilder::new().start_handler(|_| crate::licence::unlock_thread());
    if let Some(threads) = threads {
        builder = builder.num_threads(threads.max(1));
    }
    builder
        .build()
        .map_err(|error| RusticolError::internal(error.to_string()))
}

/// The term readings of every kernel of `jobs`, in job order; every output
/// is an independent task. A kernel with a refused output takes the refusal
/// of its first such output (the oracle reads the outputs in order).
pub fn kernel_term_readings(
    jobs: &[KernelTermsJob],
    threads: Option<usize>,
) -> RusticolResult<Vec<KernelTermsResult>> {
    // The kernels parse Symbolica atoms on this thread (the workers hold the
    // key from their start handler).
    let _guard = crate::licence::unlock();
    let kernels = jobs
        .iter()
        .map(Kernel::new)
        .collect::<RusticolResult<Vec<_>>>()?;
    let tasks: Vec<(usize, &str)> = jobs
        .iter()
        .enumerate()
        .flat_map(|(k, job)| job.expressions.iter().map(move |text| (k, text.as_str())))
        .collect();
    let read = |&(k, text): &(usize, &str)| output_terms(&kernels[k], text);
    let outputs: Vec<Result<Vec<TermReading>, String>> = if tasks.len() <= 1 {
        tasks.iter().map(read).collect::<RusticolResult<_>>()?
    } else {
        pool(threads)?.install(|| tasks.par_iter().map(read).collect::<RusticolResult<_>>())?
    };
    let mut results: Vec<KernelTermsResult> = jobs
        .iter()
        .map(|job| Ok(Vec::with_capacity(job.expressions.len())))
        .collect();
    for ((k, _), output) in tasks.iter().zip(outputs) {
        if let Ok(done) = &mut results[*k] {
            match output {
                Ok(terms) => done.push(terms),
                Err(message) => results[*k] = Err(message),
            }
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical(text: &str) -> String {
        parse(text).unwrap().to_canonical_string()
    }

    fn rational(text: &str) -> Rational {
        let (n, d) = text.split_once('/').unwrap_or((text, "1"));
        Rational::new(n.parse::<i64>().unwrap(), d.parse::<i64>().unwrap())
    }

    fn job(expressions: &[&str]) -> KernelTermsJob {
        KernelTermsJob {
            expressions: expressions.iter().map(|e| canonical(e)).collect(),
            positions: vec![(canonical("mfrust::x"), 0), (canonical("mfrust::y"), 1)],
            parameters: vec![
                (canonical("mfrust::g"), "G".to_owned()),
                (canonical("mfrust::m"), "M".to_owned()),
            ],
        }
    }

    #[test]
    fn readings_follow_the_oracle() {
        let _guard = crate::licence::unlock();
        let results = kernel_term_readings(
            &[job(&[
                "3/2*mfrust::g^2*mfrust::x*mfrust::y^2",
                "(2*mfrust::m)^(1/2)*mfrust::x + 1𝑖*𝜋*mfrust::y",
                "(mfrust::g+mfrust::m)^(-1/3)*mfrust::x",
            ])],
            Some(2),
        )
        .unwrap();
        let outputs = results[0].clone().unwrap();
        assert_eq!(outputs.len(), 3);
        let term = &outputs[0][0];
        assert_eq!(term.coefficient, (rational("3/2"), rational("0")));
        assert_eq!(term.powers, vec![(Base::Param("G".into()), rational("2"))]);
        let mut monomial = term.monomial.clone();
        monomial.sort();
        assert_eq!(monomial, vec![(0, 1), (1, 2)]);
        // sqrt(2 m) x: m^(1/2) distributed, 2^(1/2) a prime; pi y: i pi.
        let mut sqrt = None;
        let mut pi = None;
        for term in &outputs[1] {
            if term.monomial == vec![(0, 1)] {
                sqrt = Some(term.clone());
            } else {
                pi = Some(term.clone());
            }
        }
        let sqrt = sqrt.unwrap();
        assert!(sqrt.positive.contains(&Base::Param("M".into())));
        assert!(
            sqrt.powers
                .contains(&(Base::Prime(Integer::from(2)), rational("1/2")))
        );
        assert!(
            sqrt.powers
                .contains(&(Base::Param("M".into()), rational("1/2")))
        );
        let pi = pi.unwrap();
        assert_eq!(pi.coefficient, (rational("0"), rational("1")));
        assert_eq!(pi.powers, vec![(Base::Pi, rational("1"))]);
        // (g + m)^(-1/3): kept whole with its fractional power.
        let term = &outputs[2][0];
        let expected = parse("(mfrust::g+mfrust::m)^(-1/3)")
            .unwrap()
            .to_canonical_string();
        let pairs = vec![
            (canonical("mfrust::g"), "G".to_owned()),
            (canonical("mfrust::m"), "M".to_owned()),
        ];
        assert_eq!(
            term.powers,
            vec![(Base::Expr(expected, pairs), rational("1"))]
        );
    }

    #[test]
    fn refusals_follow_the_oracle() {
        let _guard = crate::licence::unlock();
        let results = kernel_term_readings(
            &[
                job(&["mfrust::x^(1/2)"]),
                job(&["mfrust::z*mfrust::x"]),
                job(&["mfrust::x^mfrust::g"]),
            ],
            None,
        )
        .unwrap();
        assert_eq!(
            results[0],
            Err(format!(
                "kernel not polynomial in {}",
                canonical("mfrust::x")
            ))
        );
        assert_eq!(
            results[1],
            Err(format!(
                "unknown symbol {} in a kernel",
                canonical("mfrust::z")
            ))
        );
        assert!(
            results[2]
                .clone()
                .unwrap_err()
                .starts_with("symbolic exponent in")
        );
    }
}
