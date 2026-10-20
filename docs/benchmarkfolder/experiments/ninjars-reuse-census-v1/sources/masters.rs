//! One-loop master integrals from oneloopmaster.
//!
//! For the fixed-precision carriers (binary64 and DoubleFloat), scalar
//! families (A0, B0, dB0, C0, D0) use oneloopmaster's native direct-Rust
//! implementation generated from its exact expressions. Its exact branch
//! predicates need analytically real products such as `z conj(z)` to be
//! exactly real. The compensated DoubleFloat product rounds differently in
//! each operand order, so oneloopmaster forms its native DoubleFloat products
//! symmetrically; before it did, a four-mass D0 lost its real part
//! (`tests/box_masters.rs`).
//!
//! The bubble tensor coefficients B1, B11, B111 and B1111 use oneloopmaster's
//! native tensor families (`TensorIntegral::B11`, `B111`, `B1111`; OneLOop's
//! `bub11`, `bub111` and `bub1111` with the corrected mass-swap sign) for every
//! carrier, arbitrary-precision `Float` included. The exact Passarino-Veltman
//! forms divide by powers of `p²`; B1111 divides by `(p²)⁴` and loses about
//! `5-6 log10(m²/|p²|)` digits at small `|p²|` (relative error 5.5e-6 in
//! binary64 at `p² = m²/256`, 1.2e-4 in `Float(256)` at `|p²|/m² = 4e-13`),
//! while its tracked `Float` precision overstates the digits that remain. For
//! `Float`, each row is evaluated at the precision that its inputs determine
//! (at most the working precision, see [`SymbolicaMasters`]), so the native
//! route does not pad inputs beyond the digits they determine relative to the
//! row's scale.
//!
//! Otherwise arbitrary-precision `Float` keeps Symbolica's interpreter of the
//! exact expressions for the scalar families, which tracks the significant
//! precision of every input component instead of padding inputs to the
//! working precision. The tensor family An and all Dual (derivative)
//! evaluations of every family, the bubble tensors included, use the same
//! exact formulas through the interpreter for every carrier, whose DoubleFloat
//! products keep their operand-order rounding.

use std::collections::{BTreeMap, btree_map::Entry};

use ninjars::{
    BatchShape, Complex, Error, FloatLike, Laurent, MasterFailure, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Result,
};
use oneloop::{
    NativeEvaluator, NativeFloat, NativeTensorEvaluator, OneLoopExpressions, ScalarIntegral,
    TensorIntegral,
};
use symbolica::{
    domains::{
        dual::{DualNumber, DualNumberStructure, HyperDual},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
};

type ExactEvaluator = ExpressionEvaluator<Complex<Rational>>;

/// Native tensor evaluators that `SymbolicaMasters<Float>` keeps for rows
/// below its working precision. Each holds its series coefficients at its own
/// precision (about a millisecond to build at 256 bits), so the cache is
/// cleared, keeping the working-precision evaluator, when a new precision
/// would exceed this count.
const ROW_PRECISION_EVALUATORS: usize = 16;

// These views are formed only after the complete request has passed structural
// validation. Row identities remain those of the caller after compaction.
fn row_request<'a, T>(request: &MasterRequest<'a, T>, row: usize) -> MasterRequest<'a, T> {
    let invariants = request.integral.invariant_count();
    let masses = request.integral.mass_count();
    MasterRequest {
        integral: request.integral,
        invariants: &request.invariants[row * invariants..(row + 1) * invariants],
        masses_squared: &request.masses_squared[row * masses..(row + 1) * masses],
        scales_squared: &request.scales_squared[row..row + 1],
    }
}

fn strict_status(statuses: &[MasterStatus]) -> Result<()> {
    if let Some((row, MasterStatus::Failed(failure))) = statuses
        .iter()
        .enumerate()
        .find(|(_, status)| matches!(status, MasterStatus::Failed(_)))
    {
        return Err(Error::Unsupported(format!(
            "master row {row} failed: {failure:?}"
        )));
    }
    Ok(())
}

/// Reusable master provider for `f64`, `DoubleFloat`, and arbitrary-precision `Float`.
///
/// Constants are mapped directly from rationals at the representative's working
/// precision. No input is converted through binary64. Each instruction and
/// shared helper processes all active lanes; Symbolica retains lazy branches.
/// Working precision specifies constant preparation, not a required precision
/// for every input or output component. Symbolica `Float` tracks significant
/// bits through arithmetic, so cancellation can lower a component's precision
/// even when its absolute error is small. Inputs and computed outputs retain
/// that information; this provider never pads their precision. A successful
/// status reports a finite result in the supported domain, not attainment of
/// the working precision. Accuracy requires reconstruction or convergence checks.
/// `MasterProvider::evaluate_batch` compacts eligible rows, evaluates the family
/// once, and scatters only finite results. Failed rows retain their old output;
/// `evaluate` retains its strict all-or-nothing contract.
/// Expression preparation uses the calling thread, which must provide the stack
/// required by oneloopmaster (128 MiB is its documented regression-test setting).
///
/// For fixed-precision carriers the provider applies OneLOop's on-shell
/// threshold, as original Ninja configures it (`olo_onshell(INFRARED_EPS)`):
/// invariants and squared masses whose magnitude is below the threshold are
/// set to zero, and real invariants within the threshold of a real squared
/// mass are set equal to it. Rounded external momenta otherwise turn exact
/// light-like or on-shell configurations into tiny nonzero values, which the
/// exact branch selection of oneloopmaster evaluates literally with severe
/// cancellation. The default threshold is Ninja's `1e-9` in binary64 and
/// `1e15` ulp in DoubleFloat; see [`Self::with_onshell_threshold`].
/// Arbitrary-precision `Float` has no threshold: supply exact invariants (an
/// S-matrix) for such configurations. A light-like invariant differenced from
/// rounded momenta (e.g. `+2^-114` with 2 tracked bits at 128 bits) can make
/// the four-mass D0 wrong by orders of magnitude while it reports 8 bits.
///
/// B1, B11, B111 and B1111 come from oneloopmaster's native tensor families
/// for every carrier.
/// The native code rounds every input to one working precision, so for `Float`
/// that precision is chosen per row, never above the provider's working
/// precision: the least, over the row's inputs, of the bits that each input
/// determines relative to the row's scale. A nonzero component `x` of `p²`,
/// `m0²` or `m1²` with tracked precision `b` is uncertain by about
/// `|x| 2^-b`, i.e. it determines `b + log2(S/|x|)` bits relative to the
/// largest component magnitude `S` of the three, the scale that bubble
/// functions depend on (they are homogeneous in it apart from
/// `log(m²/mu²)`); `mu²` determines its own tracked precision. Inputs above
/// that precision are rounded to it, and a component that is small against
/// `S` (such as an invariant whose tracked precision reflects cancellation in
/// its own computation) is not allowed to lower the precision of the whole row.
/// The computed coefficients then carry Numerica's tracked precision from that
/// evaluation. Rows with exact inputs at the working precision are evaluated at
/// the working precision.
pub struct SymbolicaMasters<T: NativeFloat> {
    representative: T,
    onshell_threshold: Option<T>,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    native: BTreeMap<MasterIntegral, NativeEvaluator<T>>,
    /// Native tensor families (the bubble tensors), by family and working bits: the
    /// fixed precision of binary64 and DoubleFloat, or for `Float` each row's
    /// precision ([`Self::tensor_row_bits`]).
    native_tensor: BTreeMap<(MasterIntegral, u32), NativeTensorEvaluator<T>>,
    /// Every coefficient of a native tensor family, before the requested
    /// one is selected.
    tensor_results: Vec<Complex<T>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
    active: Vec<usize>,
    statuses: Vec<MasterStatus>,
}

impl<T: NativeFloat> SymbolicaMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(representative: T) -> Self {
        // Original Ninja: INFRARED_EPS = 1e-9 in double and 1e15 machine
        // epsilons in quadruple precision.
        let onshell_threshold = match T::FIXED_BITS {
            Some(53) => Some(representative.from_rational(&Rational::from((1, 1_000_000_000)))),
            Some(bits) => {
                let epsilon = representative
                    .from_rational(&Rational::from((1, 2)))
                    .powf(&representative.from_i64(i64::from(bits) - 1));
                Some(epsilon * representative.from_i64(1_000_000_000_000_000))
            }
            None => None,
        };
        Self {
            representative,
            onshell_threshold,
            evaluators: BTreeMap::new(),
            native: BTreeMap::new(),
            native_tensor: BTreeMap::new(),
            tensor_results: Vec::new(),
            arguments: Vec::new(),
            results: Vec::new(),
            active: Vec::new(),
            statuses: Vec::new(),
        }
    }

    /// Set (or with `None` disable) the absolute on-shell threshold used on the
    /// fixed-precision path.
    pub fn with_onshell_threshold(mut self, threshold: Option<T>) -> Self {
        self.onshell_threshold = threshold;
        self
    }

    /// Apply OneLOop's on-shell threshold to one row of native arguments
    /// (`invariants`, then squared masses, then the squared scale).
    fn snap_onshell(threshold: &T, row: &mut [Complex<T>], invariants: usize) {
        let zero = threshold.zero();
        let small = |value: &Complex<T>| value.re.norm() + value.im.norm() < *threshold;
        let masses = row.len() - invariants - 1;
        for value in &mut row[invariants..invariants + masses] {
            if small(value) {
                *value = Complex::new(zero.clone(), zero.clone());
            }
        }
        let (momenta, rest) = row.split_at_mut(invariants);
        for momentum in momenta {
            if small(momentum) {
                *momentum = Complex::new(zero.clone(), zero.clone());
                continue;
            }
            if let Some(mass) = rest[..masses].iter().find(|mass| {
                mass.im.is_fully_zero() && (momentum.re.clone() - &mass.re).norm() < *threshold
            }) {
                *momentum = mass.clone();
            }
        }
    }

    /// The oneloopmaster family with a native implementation, if any.
    fn native_family(integral: MasterIntegral) -> Option<ScalarIntegral> {
        match integral {
            MasterIntegral::A0 => Some(ScalarIntegral::A0),
            MasterIntegral::B0 => Some(ScalarIntegral::B0),
            MasterIntegral::DB0 => Some(ScalarIntegral::DB0),
            MasterIntegral::C0 => Some(ScalarIntegral::C0),
            MasterIntegral::D0 => Some(ScalarIntegral::D0),
            _ => None,
        }
    }

    fn native_evaluator(
        &mut self,
        integral: MasterIntegral,
    ) -> Result<Option<&mut NativeEvaluator<T>>> {
        // The native backend converts every input to the working precision;
        // only use it where that precision is fixed by the carrier type.
        let (Some(family), Some(_)) = (Self::native_family(integral), T::FIXED_BITS) else {
            return Ok(None);
        };
        if !self.native.contains_key(&integral) {
            let evaluator = NativeEvaluator::new(family).map_err(Error::Symbolic)?;
            self.native.insert(integral, evaluator);
        }
        Ok(self.native.get_mut(&integral))
    }

    /// The oneloopmaster tensor family serving `integral` natively, and the
    /// position of `integral` among its coefficients (`[B0, B1, B11]`,
    /// `[B0, .., B111]`, `[B0, .., B1111]`). The exact Passarino-Veltman B1,
    /// B11 and B111 divide by powers of `p²` and lose about 2, 3 and 4 digits
    /// per decade of `m²/|p²|` (DoubleFloat B111: 3.6e-7 at `p² = 2^-24`).
    fn native_tensor_family(integral: MasterIntegral) -> Option<(TensorIntegral, usize)> {
        match integral {
            MasterIntegral::B1 => Some((TensorIntegral::B11, 1)),
            MasterIntegral::B11 => Some((TensorIntegral::B11, 2)),
            MasterIntegral::B111 => Some((TensorIntegral::B111, 3)),
            MasterIntegral::B1111 => Some((TensorIntegral::B1111, 4)),
            _ => None,
        }
    }

    /// The native evaluator of the tensor family `family` (serving
    /// `integral`) at `bits` working bits, built on first use. Fixed-precision
    /// carriers only use their own precision.
    fn tensor_evaluator(
        evaluators: &mut BTreeMap<(MasterIntegral, u32), NativeTensorEvaluator<T>>,
        integral: MasterIntegral,
        family: TensorIntegral,
        bits: u32,
    ) -> Result<&mut NativeTensorEvaluator<T>> {
        Ok(match evaluators.entry((integral, bits)) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(
                NativeTensorEvaluator::with_binary_precision(family, bits)
                    .map_err(Error::Symbolic)?,
            ),
        })
    }

    /// Working bits of the native tensor family for one `Float` row
    /// `[invariants.., masses.., mu²]`: at most `working`, and at most the bits
    /// that each nonzero input component determines, `b + log2(S/|x|)` for a
    /// component `x` of an invariant or mass with tracked precision `b` (`S`
    /// the largest such magnitude in the row), and the tracked precision of
    /// `mu²`. See [`SymbolicaMasters`].
    fn tensor_row_bits(working: u32, row: &[Complex<T>]) -> u32 {
        let (kinematics, scale) = row.split_at(row.len() - 1);
        let components = || {
            kinematics
                .iter()
                .flat_map(|value| [&value.re, &value.im])
                .filter(|component| !component.is_fully_zero())
        };
        let mut bits = working.min(scale[0].re.get_precision());
        let Some(largest) = components()
            .map(|component| component.norm())
            .reduce(|a, b| if b > a { b } else { a })
        else {
            return bits.max(1);
        };
        for component in components() {
            // Only the binary order of magnitude matters: the quotient is
            // within a factor of two even at one tracked bit. A quotient
            // beyond binary64 range imposes no limit.
            let headroom = (largest.clone() / &component.norm()).to_f64().log2();
            let determined = f64::from(component.get_precision()) + headroom.max(0.).floor();
            if determined < f64::from(bits) {
                bits = determined as u32;
            }
        }
        bits.max(1)
    }

    /// Working bits used to prepare constants; computed component precision may differ.
    pub fn precision(&self) -> u32 {
        self.representative.get_precision()
    }

    /// Compile a family and reserve its reusable workspace before timing calls.
    /// For `Float`, bubble tensor rows below the working precision build their native
    /// evaluator on first use.
    pub fn prepare(&mut self, integral: MasterIntegral, rows: usize) -> Result<()> {
        if self.native_evaluator(integral)?.is_some() {
            return Ok(());
        }
        if let Some((family, _)) = Self::native_tensor_family(integral) {
            let bits = T::FIXED_BITS.unwrap_or_else(|| self.precision());
            Self::tensor_evaluator(&mut self.native_tensor, integral, family, bits)?;
            let size = rows.checked_mul(family.outputs()).ok_or_else(|| {
                Error::Unsupported("master output dimensions overflow usize".into())
            })?;
            let zero = self.representative.zero();
            self.tensor_results
                .resize_with(size, || Complex::new(zero.clone(), zero.clone()));
            return Ok(());
        }
        if !self.evaluators.contains_key(&integral) {
            let exact = Self::exact_evaluator(integral)?;
            let evaluator = Self::numeric_evaluator(exact, &self.representative);
            self.evaluators.insert(integral, evaluator);
        }
        self.evaluators
            .get_mut(&integral)
            .unwrap()
            .reserve(rows)
            .map_err(|error| Error::Symbolic(error.to_string()))
    }

    fn numeric_evaluator(
        exact: ExactEvaluator,
        representative: &T,
    ) -> BatchExpressionEvaluator<Complex<T>> {
        exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        representative.from_rational(&value.re),
                        representative.from_rational(&value.im),
                    )
                },
                representative.get_precision(),
            )
            .into_batch_evaluator()
    }

    fn exact_evaluator(integral: MasterIntegral) -> Result<ExactEvaluator> {
        let args = (0..integral.argument_count())
            .map(|index| symbolica::symbol!(format!("ninjars::master_argument_{index}")).to_atom())
            .collect::<Vec<_>>();
        let expressions = OneLoopExpressions::new();
        let series = match integral {
            MasterIntegral::A0 => oneloop::a0(&args[0], &args[1]),
            MasterIntegral::B0 => oneloop::b0(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::DB0 => oneloop::db0(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B1 => oneloop::b1(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B11 => oneloop::b11(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::B111 => oneloop::b111(&args[0], &args[1], &args[2], &args[3]),
            // Exact Passarino-Veltman expression with oneloopmaster's
            // corrected zero-momentum mass swap, for Dual carriers only. It
            // loses about 5-6 log10(m²/|p²|) digits at small p² (relative
            // error 5.5e-6 in binary64 at p² = m²/256, 1.2e-4 in Float(256)
            // at |p²|/m² = 4e-13), so `SymbolicaMasters` evaluates B1111 with
            // the native family for every carrier (`native_tensor_family`).
            MasterIntegral::B1111 => oneloop::b1111(&args[0], &args[1], &args[2], &args[3]),
            MasterIntegral::An { rank } => oneloop::an(rank, &args[0], &args[1])
                .map_err(|_| Error::Unsupported("tadpole tensor rank exceeds four".into()))?
                .pop()
                .expect("oneloopmaster includes A0 at every supported rank"),
            MasterIntegral::C0 => expressions.c0(
                [&args[0], &args[1], &args[2]],
                [&args[3], &args[4], &args[5]],
                &args[6],
            ),
            MasterIntegral::D0 => expressions.d0(
                [&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]],
                [&args[6], &args[7], &args[8], &args[9]],
                &args[10],
            ),
        };
        expressions
            .evaluator(series.coefficients(), &args)
            .map_err(|error| Error::Symbolic(error.to_string()))
    }

    fn validate_values(
        &self,
        request: &MasterRequest<'_, Complex<T>>,
    ) -> std::result::Result<(), MasterFailure> {
        let zero = self.representative.zero();
        for value in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(MasterFailure::NonFiniteInput);
            }
        }
        if request
            .invariants
            .iter()
            .any(|value| !value.im.is_fully_zero())
        {
            return Err(MasterFailure::NonRealInvariant);
        }
        if request.masses_squared.iter().any(|value| value.im > zero) {
            return Err(MasterFailure::InvalidMass);
        }
        if request
            .scales_squared
            .iter()
            .any(|value| !value.im.is_fully_zero() || value.re <= zero)
        {
            return Err(MasterFailure::InvalidScale);
        }
        Ok(())
    }

    fn stage(&mut self, request: &MasterRequest<'_, Complex<T>>, strict: bool) -> Result<()> {
        self.active.clear();
        self.arguments.clear();
        self.statuses.clear();
        for row in 0..request.rows() {
            let point = row_request(request, row);
            match self.validate_values(&point) {
                Err(failure) => self.statuses.push(MasterStatus::Failed(failure)),
                Ok(()) => {
                    self.statuses.push(MasterStatus::Success);
                    self.active.push(row);
                    let start = self.arguments.len();
                    self.arguments.extend_from_slice(point.invariants);
                    self.arguments.extend_from_slice(point.masses_squared);
                    self.arguments.extend_from_slice(point.scales_squared);
                    if let Some(threshold) = &self.onshell_threshold {
                        Self::snap_onshell(
                            threshold,
                            &mut self.arguments[start..],
                            request.integral.invariant_count(),
                        );
                    }
                }
            }
        }
        if strict {
            strict_status(&self.statuses)?;
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let rows = self.active.len();
        self.prepare(request.integral, rows)?;
        let size = rows
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master output dimensions overflow usize".into()))?;
        self.results.resize_with(size, || {
            Complex::new(self.representative.zero(), self.representative.zero())
        });
        let results = &mut self.results[..size];
        if let Some(native) = self.native.get_mut(&request.integral) {
            native
                .evaluate_batch(&self.arguments, results, rows)
                .map_err(Error::Symbolic)?;
        } else if let Some((family, slot)) = Self::native_tensor_family(request.integral) {
            let width = family.outputs();
            let all = &mut self.tensor_results[..rows * width];
            if let Some(bits) = T::FIXED_BITS {
                Self::tensor_evaluator(&mut self.native_tensor, request.integral, family, bits)?
                    .evaluate_batch(&self.arguments, all, rows)
                    .map_err(Error::Symbolic)?;
            } else {
                // Arbitrary precision: every row at its own working bits.
                let working = self.representative.get_precision();
                for (point, values) in self
                    .arguments
                    .chunks_exact(family.arity())
                    .zip(all.chunks_exact_mut(width))
                {
                    let bits = Self::tensor_row_bits(working, point);
                    if !self.native_tensor.contains_key(&(request.integral, bits))
                        && self.native_tensor.len() > ROW_PRECISION_EVALUATORS
                    {
                        self.native_tensor.retain(|&(_, kept), _| kept == working);
                    }
                    Self::tensor_evaluator(
                        &mut self.native_tensor,
                        request.integral,
                        family,
                        bits,
                    )?
                    .evaluate(point, values)
                    .map_err(Error::Symbolic)?;
                }
            }
            for (result, values) in results
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(all.chunks_exact(width))
            {
                result.clone_from_slice(&values[3 * slot..3 * slot + 3]);
            }
        } else {
            self.evaluators
                .get_mut(&request.integral)
                .unwrap()
                .evaluate_batch(&self.arguments, results, rows)
                .map_err(|error| Error::Symbolic(error.to_string()))?;
        }
        for (&row, values) in self.active.iter().zip(self.results.as_chunks::<3>().0) {
            if values
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                self.statuses[row] = MasterStatus::Failed(MasterFailure::NonFiniteResult);
            }
        }
        Ok(())
    }

    fn commit(&self, output: &mut [Laurent<Complex<T>>]) {
        for (&row, values) in self.active.iter().zip(self.results.as_chunks::<3>().0) {
            if self.statuses[row] == MasterStatus::Success {
                output[row].clone_from_slice(values);
            }
        }
        if !self.active.is_empty() {
            oneloop::record_usage();
        }
    }
}

/// Batched higher/mixed derivatives using Symbolica's static or dynamic Dual carriers.
///
/// Coefficients follow the supplied ancestor-closed Symbolica shape, including
/// its Taylor convention: component `alpha` equals the derivative divided by
/// `alpha!`. Derivatives
/// are currently supported and tested for positive real masses and scales in tadpoles and
/// strictly spacelike bubbles, with real parameter seeds. Other domains return
/// an explicit failure status (or an error through strict `evaluate`), including
/// removable zero-momentum limits whose specialized value formulas do not
/// determine their neighborhood derivatives. Shape, domain and finiteness checks
/// cover every component, before compaction or arithmetic. As for
/// [`SymbolicaMasters`], constants use the representative's working precision;
/// each input and computed component keeps its own tracked significant precision.
/// Success does not certify that all derivatives achieve the working precision.
pub struct SymbolicaDualMasters<T: NativeFloat> {
    representative: T,
    shape: Vec<Vec<usize>>,
    evaluators: BTreeMap<MasterIntegral, BatchExpressionEvaluator<Complex<T>>>,
    arguments: Vec<Complex<T>>,
    results: Vec<Complex<T>>,
    active: Vec<usize>,
    statuses: Vec<MasterStatus>,
}

impl<T: NativeFloat> SymbolicaDualMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    /// Use the shape of any Symbolica static or dynamic Dual instance.
    pub fn new(representative: T, structure: &impl DualNumberStructure) -> Self {
        Self {
            representative,
            shape: structure
                .get_shape()
                .into_iter()
                .map(<[usize]>::to_vec)
                .collect(),
            evaluators: BTreeMap::new(),
            arguments: Vec::new(),
            results: Vec::new(),
            active: Vec::new(),
            statuses: Vec::new(),
        }
    }

    fn valid_shape<D: DualNumber<Complex<T>>>(&self, value: &D) -> bool {
        !self.shape.is_empty()
            && value.values().len() == self.shape.len()
            && value
                .get_shape()
                .into_iter()
                .eq(self.shape.iter().map(Vec::as_slice))
    }

    fn validate<D: DualNumber<Complex<T>>>(
        &self,
        request: &MasterRequest<'_, D>,
    ) -> std::result::Result<(), MasterFailure> {
        if matches!(request.integral, MasterIntegral::C0 | MasterIntegral::D0) {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        let zero = self.representative.zero();
        for dual in request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
        {
            if !self.valid_shape(dual) {
                return Err(MasterFailure::DerivativeShape);
            }
            for component in dual.values() {
                if !component.re.is_finite() || !component.im.is_finite() {
                    return Err(MasterFailure::NonFiniteInput);
                }
                if !component.im.is_fully_zero() {
                    return Err(MasterFailure::UnsupportedDerivativeDomain);
                }
            }
        }
        if request
            .invariants
            .iter()
            .any(|value| value.values()[0].re >= zero)
        {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        if request
            .masses_squared
            .iter()
            .any(|value| value.values()[0].re <= zero)
        {
            return Err(MasterFailure::UnsupportedDerivativeDomain);
        }
        if request
            .scales_squared
            .iter()
            .any(|value| value.values()[0].re <= zero)
        {
            return Err(MasterFailure::InvalidScale);
        }
        Ok(())
    }

    fn stage<D: DualNumber<Complex<T>>>(
        &mut self,
        request: &MasterRequest<'_, D>,
        strict: bool,
    ) -> Result<()> {
        self.active.clear();
        self.arguments.clear();
        self.statuses.clear();
        for row in 0..request.rows() {
            let point = row_request(request, row);
            match self.validate(&point) {
                Err(failure) => self.statuses.push(MasterStatus::Failed(failure)),
                Ok(()) => {
                    self.statuses.push(MasterStatus::Success);
                    self.active.push(row);
                    for dual in point
                        .invariants
                        .iter()
                        .chain(point.masses_squared)
                        .chain(point.scales_squared)
                    {
                        self.arguments.extend_from_slice(dual.values());
                    }
                }
            }
        }
        if strict {
            strict_status(&self.statuses)?;
        }
        if self.active.is_empty() {
            return Ok(());
        }
        let rows = self.active.len();
        let components = self.shape.len();
        if !self.evaluators.contains_key(&request.integral) {
            let exact = SymbolicaMasters::<T>::exact_evaluator(request.integral)?;
            let structure = HyperDual::<Complex<Rational>>::new(self.shape.clone());
            let dual = exact
                .vectorize(&Dualizer::new(structure, Vec::new()))
                .map_err(|error| Error::Symbolic(error.to_string()))?;
            self.evaluators.insert(
                request.integral,
                SymbolicaMasters::numeric_evaluator(dual, &self.representative),
            );
        }
        let columns = components
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master Dual dimensions overflow usize".into()))?;
        let size = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::Unsupported("master Dual dimensions overflow usize".into()))?;
        self.results.resize_with(size, || {
            Complex::new(self.representative.zero(), self.representative.zero())
        });
        self.evaluators
            .get_mut(&request.integral)
            .unwrap()
            .evaluate_batch(&self.arguments, &mut self.results, rows)
            .map_err(|error| Error::Symbolic(error.to_string()))?;
        for (&row, values) in self.active.iter().zip(self.results.chunks_exact(columns)) {
            if values
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                self.statuses[row] = MasterStatus::Failed(MasterFailure::NonFiniteResult);
            }
        }
        Ok(())
    }

    fn commit<D: FloatLike + DualNumber<Complex<T>>>(
        &self,
        request: &MasterRequest<'_, D>,
        output: &mut [Laurent<D>],
    ) {
        if self.active.is_empty() {
            return;
        }
        let components = self.shape.len();
        for (&row, values) in self
            .active
            .iter()
            .zip(self.results.chunks_exact(3 * components))
        {
            if self.statuses[row] != MasterStatus::Success {
                continue;
            }
            for (coefficient, values) in output[row].iter_mut().zip(values.chunks_exact(components))
            {
                if !self.valid_shape(coefficient) {
                    *coefficient = request.scales_squared[row].zero();
                }
                coefficient.values_mut().clone_from_slice(values);
            }
        }
        oneloop::record_usage();
    }
}

impl<T: NativeFloat> MasterProvider<Complex<T>> for SymbolicaMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> Result<()> {
        request.validate(output.len())?;
        self.stage(&request, true)?;
        strict_status(&self.statuses)?;
        self.commit(output);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        BatchShape {
            rows: request.rows(),
            columns: 1,
        }
        .validate("master statuses", statuses.len())?;
        self.stage(&request, false)?;
        self.commit(output);
        statuses.copy_from_slice(&self.statuses);
        Ok(())
    }
}

impl<T: NativeFloat, D: FloatLike + DualNumber<Complex<T>>> MasterProvider<D>
    for SymbolicaDualMasters<T>
where
    Complex<T>: EvaluationDomain,
{
    fn evaluate(&mut self, request: MasterRequest<'_, D>, output: &mut [Laurent<D>]) -> Result<()> {
        request.validate(output.len())?;
        self.stage(&request, true)?;
        strict_status(&self.statuses)?;
        self.commit(&request, output);
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, D>,
        output: &mut [Laurent<D>],
        statuses: &mut [MasterStatus],
    ) -> Result<()> {
        request.validate(output.len())?;
        BatchShape {
            rows: request.rows(),
            columns: 1,
        }
        .validate("master statuses", statuses.len())?;
        self.stage(&request, false)?;
        self.commit(&request, output);
        statuses.copy_from_slice(&self.statuses);
        Ok(())
    }
}
