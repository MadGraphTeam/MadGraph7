// SPDX-License-Identifier: 0BSD

//! Existing epsilon-sector reduction facade shared with non-Python callers.
use super::fast::FastReductions;
use super::v2_reduce::{
    GENERIC_CHUNK_LANES, fast_engine_threshold, reaches_fast_threshold,
    squared_mass_reaches_fast_threshold,
};
use super::*;
use crate::{
    AmplitudeInput, EvaluationStatus, FourMomentum, Lane, Laurent, PolynomialReducer,
    SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus, SymbolicaSamplingScales,
};
use symbolica::{atom::Atom, domains::float::Real, evaluate::EvaluationDomain};
pub type DecimalMomentum = [DecimalComplex; 4];
pub type RawLane = (u64, String, Option<[DecimalComplex; 3]>);

pub(super) struct Reductions<T: Carrier>
where
    Complex<T>: Real + EvaluationDomain,
{
    sectors: [PolynomialReducer<T>; 3],
    active: [bool; 3],
    zero: Complex<T>,
    propagators: usize,
    parameter_count: usize,
    sampling: Option<SymbolicaSamplingScales<T>>,
    sampling_values: Vec<Complex<T>>,
    sampling_statuses: Vec<SamplingScaleStatus>,
}

impl<T: Carrier> Reductions<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub(super) fn new(
        expressions: &[String; 3],
        parameters: &[String],
        n: usize,
        prototype: T,
    ) -> ExecutionResult<Self> {
        // The expressions may call the source-scalar primitives: install them
        // before parsing (the extension's import no longer does).
        super::register_scalar_primitives();
        let parse =
            |text: &str| Atom::parse(text, "pyamplicol_loop", Default::default()).map_err(error);
        let variables = [
            parse("q0")?,
            parse("q1")?,
            parse("q2")?,
            parse("q3")?,
            parse("mu2")?,
        ];
        let parameters = parameters
            .iter()
            .map(|p| parse(p))
            .collect::<ExecutionResult<Vec<_>>>()?;
        let mut sectors = Vec::with_capacity(3);
        let mut active = [false; 3];
        for (index, expression) in expressions.iter().enumerate() {
            let expression = parse(expression)?;
            active[index] = !expression.is_zero();
            sectors.push(
                PolynomialReducer::new(
                    n,
                    expression,
                    variables.clone(),
                    parameters.clone(),
                    prototype.clone(),
                    None,
                )
                .map_err(error)?,
            );
        }
        // Still execute one zero sector for authoritative input validation.
        if !active.iter().any(|value| *value) {
            active[0] = true;
        }
        Ok(Self {
            active,
            zero: Complex::new(prototype.zero(), prototype.zero()),
            sectors: sectors.try_into().ok().unwrap(),
            propagators: n,
            parameter_count: parameters.len(),
            sampling: if n >= 4 {
                Some(SymbolicaSamplingScales::new(n, prototype).map_err(error)?)
            } else {
                None
            },
            sampling_values: Vec::new(),
            sampling_statuses: Vec::new(),
        })
    }

    /// Pole residues the masters of every epsilon sector keep.
    pub(super) fn set_pole_selection(&mut self, selection: crate::PoleSelection) {
        for sector in &mut self.sectors {
            sector.set_pole_selection(selection);
        }
    }

    fn evaluate(
        &mut self,
        ids: &[u64],
        shifts: &[DecimalMomentum],
        masses: &[DecimalComplex],
        scales: &[DecimalComplex],
        parameters: &[DecimalComplex],
        bits: u32,
    ) -> ExecutionResult<Vec<RawLane>> {
        let parse_all = |values: &[DecimalComplex]| {
            values
                .iter()
                .map(|v| parse_complex::<T>(v, bits))
                .collect::<ExecutionResult<Vec<_>>>()
        };
        let shifts = shifts
            .iter()
            .map(|p| {
                Ok(FourMomentum::from([
                    parse_complex::<T>(&p[0], bits)?,
                    parse_complex::<T>(&p[1], bits)?,
                    parse_complex::<T>(&p[2], bits)?,
                    parse_complex::<T>(&p[3], bits)?,
                ]))
            })
            .collect::<ExecutionResult<Vec<_>>>()?;
        let masses = parse_all(masses)?;
        let scales = parse_all(scales)?;
        let parameters = parse_all(parameters)?;
        Ok(self
            .evaluate_native(ids, &shifts, &masses, &scales, &parameters, None)?
            .into_iter()
            .map(|lane| {
                (
                    lane.id,
                    lane.status,
                    lane.value.map(|value| {
                        [
                            decimal_complex(value.finite, bits),
                            decimal_complex(value.single_pole, bits),
                            decimal_complex(value.double_pole, bits),
                        ]
                    }),
                )
            })
            .collect())
    }

    pub(super) fn evaluate_native(
        &mut self,
        ids: &[u64],
        shifts: &[FourMomentum<Complex<T>>],
        masses: &[Complex<T>],
        scales: &[Complex<T>],
        parameters: &[Complex<T>],
        s_matrix: Option<&[Complex<T>]>,
    ) -> ExecutionResult<Vec<Lane<Complex<T>, String>>> {
        if let Some(sampling) = &mut self.sampling {
            self.sampling_values.resize(ids.len(), self.zero.clone());
            self.sampling_values.fill(self.zero.clone());
            self.sampling_statuses.resize(
                ids.len(),
                SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback),
            );
            sampling
                .evaluate_batch(
                    shifts,
                    masses,
                    ids.len(),
                    &mut self.sampling_values,
                    &mut self.sampling_statuses,
                )
                .map_err(error)?;
        }
        let mut results = Vec::with_capacity(3);
        for (index, sector) in self.sectors.iter_mut().enumerate() {
            if !self.active[index] {
                results.push(
                    ids.iter()
                        .map(|&id| Lane {
                            id,
                            status: EvaluationStatus::Success,
                            value: Some(Laurent::from([
                                self.zero.clone(),
                                self.zero.clone(),
                                self.zero.clone(),
                            ])),
                        })
                        .collect(),
                );
                continue;
            }
            results.push(
                sector
                    .evaluate(
                        ids,
                        AmplitudeInput {
                            shifts,
                            masses_squared: masses,
                            scales_squared: scales,
                            sampling_scales: &self.sampling_values,
                            s_matrix,
                        },
                        parameters,
                    )
                    .map_err(error)?,
            );
        }
        let mut sectors = results.into_iter();
        let zero = sectors.next().unwrap();
        let one = sectors.next().unwrap();
        let two = sectors.next().unwrap();
        Ok(zero
            .into_iter()
            .zip(one)
            .zip(two)
            .enumerate()
            .map(|(row, ((zero, one), two))| {
                if let Some(SamplingScaleStatus::InvalidInput(failure)) =
                    self.sampling_statuses.get(row)
                {
                    return Lane {
                        id: zero.id,
                        status: format!("sampling scale: {failure:?}"),
                        value: None,
                    };
                }
                let status = [zero.status, one.status, two.status]
                    .iter()
                    .enumerate()
                    .find(|(_, status)| **status != EvaluationStatus::Success)
                    .map(|(sector, status)| format!("epsilon sector {sector}: {status:?}"));
                let value = match (zero.value, one.value, two.value) {
                    (Some(a), Some(b), Some(c)) => Some(Laurent::epsilon_sectors(a, b, c)),
                    _ => None,
                };
                Lane {
                    id: zero.id,
                    status: status.unwrap_or_else(|| "success".into()),
                    value,
                }
            })
            .collect())
    }
}

enum Plans {
    Double(Reductions<f64>),
    Fast(Box<FastPlan>),
    DoubleDouble(Reductions<DoubleFloat>),
    Arbitrary(Reductions<Float>),
}

/// The binary64 fast plan of the legacy engine and its exact fallback
/// (genericity audit 2, G2-M3; `v2_reduce` module docs, 'Absolute
/// thresholds'). On the pinned NinjaRS the fast path compares its inputs with
/// absolute constants in the model's units (the native masters' on-shell
/// snap and the engine's `infrared_eps`, both `1e-9`). Lanes are routed on
/// the given inputs ([`Route`]): with an invariant matrix, every lane whose
/// squared masses or invariants reach the thresholds
/// ([`reaches_fast_threshold`], the v2 engine's rule) is reduced instead by
/// the generic binary64 reduction of the same numerator, which has none,
/// built on first use under pyAmpliCol's library key. Without one the
/// invariants are formed from the binary64 shifts, which are not given
/// inputs: below the thresholds they are indistinguishable from rounding (a
/// massless leg at a generic angle gives `|p²| ~ 1e-11` at `E = 500`), which
/// the fast path's snap sets to zero as before, and an external leg's
/// squared mass is not given at all. Such a call routes on the given squared
/// masses alone: a lane with a light squared mass is refused with
/// [`light_mass_needs_invariants`] (rounded invariants cannot resolve so
/// light a mass: a formed `p² ≈ 0` may stand for a massless leg or for one
/// of that mass), every other lane keeps the fast path. Every lane neither
/// routed nor refused, and every call without such a lane, keeps the fast
/// reduction's code path and bits.
struct FastPlan {
    fast: FastReductions,
    expressions: [String; 3],
    parameters: Vec<String>,
    exact: Option<Reductions<f64>>,
    threshold: f64,
    /// Lanes reduced by `exact` since the last
    /// [`PolynomialPlan::take_exact_lanes`].
    exact_lanes: u64,
}

/// The status of a lane of a legacy binary64 fast plan that carries a squared
/// mass in `(0, threshold)` while the call gives no invariant matrix
/// ([`FastPlan`]; `threshold` is the plan's, [`fast_engine_threshold`]):
/// binary64 invariants formed from rounded shifts cannot resolve so light a
/// mass. Pass the exact invariants (`s_matrix`), or use double-double or
/// arbitrary precision. The message names the threshold the rule compared
/// with (`1e-9` on the pinned NinjaRS).
pub fn light_mass_needs_invariants(threshold: f64) -> String {
    format!(
        "InvalidInput(a squared mass below the binary64 fast path's absolute threshold \
         {threshold:e} needs the exact invariant matrix, which invariants formed from the rounded \
         shifts cannot replace; pass s_matrix, or use double-double or arbitrary precision)"
    )
}

/// Where a lane of [`FastPlan`] is reduced.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    /// The fast reduction.
    Fast,
    /// The generic binary64 reduction (`FastPlan::exact`).
    Exact,
    /// Not reduced: [`light_mass_needs_invariants`].
    Refused,
}

/// The rows of `values` (`width` entries per lane) whose route is `wanted`.
fn kept<V: Clone>(values: &[V], width: usize, routes: &[Route], wanted: Route) -> Vec<V> {
    values
        .chunks_exact(width.max(1))
        .zip(routes)
        .filter(|&(_, &route)| route == wanted)
        .flat_map(|(row, _)| row.iter().cloned())
        .collect()
}

impl FastPlan {
    /// The route of each lane ([`FastPlan`]); empty for inputs the fast path
    /// refuses (shape, complex shifts, scales or invariants), which keep its
    /// refusal.
    fn routes(
        &self,
        lanes: usize,
        shifts: &[FourMomentum<Complex<f64>>],
        masses: &[Complex<f64>],
        scales: &[Complex<f64>],
        parameters: &[Complex<f64>],
        s_matrix: Option<&[Complex<f64>]>,
    ) -> Vec<Route> {
        let (n, width) = (self.fast.propagators, self.fast.parameter_count);
        let real = |values: &[Complex<f64>]| values.iter().all(|z| z.im == 0.);
        let admissible = shifts.len() == lanes * n
            && masses.len() == lanes * n
            && scales.len() == lanes
            && parameters.len() == lanes * width
            && s_matrix.is_none_or(|s| s.len() == lanes * n * n && real(s))
            && real(scales)
            && shifts
                .iter()
                .all(|p| p.components().iter().all(|z| z.im == 0.));
        if !admissible {
            return Vec::new();
        }
        (0..lanes)
            .map(|lane| {
                let masses = &masses[lane * n..(lane + 1) * n];
                match s_matrix {
                    Some(matrix) => {
                        let invariants = &matrix[lane * n * n..(lane + 1) * n * n];
                        if reaches_fast_threshold(n, masses, invariants, self.threshold) {
                            Route::Exact
                        } else {
                            Route::Fast
                        }
                    }
                    None if squared_mass_reaches_fast_threshold(masses, self.threshold) => {
                        Route::Refused
                    }
                    None => Route::Fast,
                }
            })
            .collect()
    }

    fn evaluate_native(
        &mut self,
        ids: &[u64],
        shifts: &[FourMomentum<Complex<f64>>],
        masses: &[Complex<f64>],
        scales: &[Complex<f64>],
        parameters: &[Complex<f64>],
        s_matrix: Option<&[Complex<f64>]>,
    ) -> ExecutionResult<Vec<Lane<Complex<f64>, String>>> {
        let (n, width) = (self.fast.propagators, self.fast.parameter_count);
        let routes = self.routes(ids.len(), shifts, masses, scales, parameters, s_matrix);
        if routes.iter().all(|&route| route == Route::Fast) {
            return self
                .fast
                .evaluate_native(ids, shifts, masses, scales, parameters, s_matrix);
        }
        let count = |wanted: Route| routes.iter().filter(|&&route| route == wanted).count();
        self.exact_lanes += count(Route::Exact) as u64;
        let threshold = self.threshold;
        let part = |wanted: Route| {
            (
                kept(ids, 1, &routes, wanted),
                kept(shifts, n, &routes, wanted),
                kept(masses, n, &routes, wanted),
                kept(scales, 1, &routes, wanted),
                kept(parameters, width, &routes, wanted),
                s_matrix.map(|s| kept(s, n * n, &routes, wanted)),
            )
        };
        let mut fast = if count(Route::Fast) > 0 {
            let (ids, shifts, masses, scales, parameters, matrix) = part(Route::Fast);
            self.fast.evaluate_native(
                &ids,
                &shifts,
                &masses,
                &scales,
                &parameters,
                matrix.as_deref(),
            )?
        } else {
            Vec::new()
        }
        .into_iter();
        let mut exact = if count(Route::Exact) > 0 {
            // The generic reduction parses and evaluates Symbolica
            // expressions.
            let _unlock = pyamplicol::unlock();
            let exact = match &mut self.exact {
                Some(exact) => exact,
                None => self.exact.insert(Reductions::new(
                    &self.expressions,
                    &self.parameters,
                    n,
                    0_f64,
                )?),
            };
            let (ids, shifts, masses, scales, parameters, matrix) = part(Route::Exact);
            let mut lanes = Vec::with_capacity(ids.len());
            // In chunks, as the v2 generic engine (values unchanged).
            for start in (0..ids.len()).step_by(GENERIC_CHUNK_LANES) {
                let end = (start + GENERIC_CHUNK_LANES).min(ids.len());
                lanes.extend(exact.evaluate_native(
                    &ids[start..end],
                    &shifts[start * n..end * n],
                    &masses[start * n..end * n],
                    &scales[start..end],
                    &parameters[start * width..end * width],
                    matrix.as_deref().map(|s| &s[start * n * n..end * n * n]),
                )?);
            }
            lanes
        } else {
            Vec::new()
        }
        .into_iter();
        ids.iter()
            .zip(&routes)
            .map(|(&id, &route)| {
                match route {
                    Route::Fast => fast.next(),
                    Route::Exact => exact.next(),
                    Route::Refused => Some(Lane {
                        id,
                        status: light_mass_needs_invariants(threshold),
                        value: None,
                    }),
                }
                .ok_or_else(|| error("a reduction engine lost a lane"))
            })
            .collect()
    }

    fn evaluate_decimal(
        &mut self,
        ids: &[u64],
        shifts: &[DecimalMomentum],
        masses: &[DecimalComplex],
        scales: &[DecimalComplex],
        parameters: &[DecimalComplex],
    ) -> ExecutionResult<Vec<RawLane>> {
        let parse_all = |values: &[DecimalComplex]| {
            values
                .iter()
                .map(|z| parse_complex::<f64>(z, 53))
                .collect::<ExecutionResult<Vec<_>>>()
        };
        let shifts = shifts
            .iter()
            .map(|p| {
                Ok(FourMomentum::from([
                    parse_complex::<f64>(&p[0], 53)?,
                    parse_complex::<f64>(&p[1], 53)?,
                    parse_complex::<f64>(&p[2], 53)?,
                    parse_complex::<f64>(&p[3], 53)?,
                ]))
            })
            .collect::<ExecutionResult<Vec<_>>>()?;
        Ok(self
            .evaluate_native(
                ids,
                &shifts,
                &parse_all(masses)?,
                &parse_all(scales)?,
                &parse_all(parameters)?,
                None,
            )?
            .into_iter()
            .map(|lane| {
                (
                    lane.id,
                    lane.status,
                    lane.value.map(|v| {
                        [
                            decimal_complex(v.finite, 53),
                            decimal_complex(v.single_pole, 53),
                            decimal_complex(v.double_pole, 53),
                        ]
                    }),
                )
            })
            .collect())
    }
}

/// Native polynomial numerator reduction with explicit epsilon sectors.
/// The output is raw [finite, single pole, double pole], without model factors.
pub struct PolynomialPlan {
    plan: Plans,
    precision_bits: u32,
}

impl PolynomialPlan {
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
    /// Enable optional fast-engine/master-provider wall clocks, resetting counts.
    /// Ordinary calls leave profiling disabled and execute no timing callbacks.
    pub fn set_profiling(&mut self, enabled: bool) -> ExecutionResult<()> {
        match &mut self.plan {
            Plans::Fast(plan) => {
                plan.fast.set_profiling(enabled);
                Ok(())
            }
            _ => Err(error(
                "profiling currently requires fast or fast-tensor engine",
            )),
        }
    }

    /// Read and reset instrumented totals across all active epsilon sectors.
    /// Engine-exclusive time excludes delegated masters and numerator callbacks.
    /// Numerator preparation occurs outside the engine and is reported separately.
    pub fn take_profile(&mut self) -> ExecutionResult<std::collections::BTreeMap<String, f64>> {
        let profile = match &mut self.plan {
            Plans::Fast(plan) => plan.fast.take_profile(),
            _ => {
                return Err(error(
                    "profiling currently requires fast or fast-tensor engine",
                ));
            }
        };
        Ok([
            ("engine_seconds", profile.engine_time.as_secs_f64()),
            ("masters_seconds", profile.masters_time.as_secs_f64()),
            (
                "numerator_callback_seconds",
                profile.numerator_callback_time.as_secs_f64(),
            ),
            (
                "numerator_preparation_seconds",
                profile.numerator_preparation_time.as_secs_f64(),
            ),
            (
                "engine_excluding_masters_seconds",
                profile
                    .engine_time
                    .saturating_sub(profile.masters_time)
                    .as_secs_f64(),
            ),
            (
                "reduction_exclusive_seconds",
                profile
                    .engine_time
                    .saturating_sub(profile.masters_time + profile.numerator_callback_time)
                    .as_secs_f64(),
            ),
            ("engine_calls", profile.engine_calls as f64),
            ("master_calls", profile.master_calls as f64),
            ("master_rows", profile.master_rows as f64),
            ("numerator_calls", profile.numerator_calls as f64),
            ("numerator_rows", profile.numerator_rows as f64),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect())
    }

    /// Lanes of the binary64 fast plan reduced by the generic binary64
    /// reduction since the last call because their inputs reach one of the
    /// fast path's absolute thresholds (`FastPlan`); 0 for the generic plans.
    /// A test hook: the legacy Python binding and `runtime::physical` do not
    /// report it (the v2 executor reports its own count,
    /// `V2Reducer::take_exact_lanes`).
    pub fn take_exact_lanes(&mut self) -> u64 {
        match &mut self.plan {
            Plans::Fast(plan) => std::mem::take(&mut plan.exact_lanes),
            _ => 0,
        }
    }

    pub fn new(
        propagators: usize,
        epsilon_sectors: [String; 3],
        parameters: Vec<String>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
    ) -> ExecutionResult<Self> {
        let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
        if !matches!(engine, "generic" | "fast" | "fast-tensor") {
            return Err(error("engine must be generic, fast or fast-tensor"));
        }
        if engine != "generic" {
            if !matches!(arithmetic, Arithmetic::Double) {
                return Err(error("fast engine supports only double arithmetic"));
            }
            return Ok(Self {
                plan: Plans::Fast(Box::new(FastPlan {
                    fast: FastReductions::new(
                        &epsilon_sectors,
                        &parameters,
                        propagators,
                        engine == "fast-tensor",
                    )?,
                    expressions: epsilon_sectors,
                    parameters,
                    exact: None,
                    threshold: fast_engine_threshold(),
                    exact_lanes: 0,
                })),
                precision_bits: bits,
            });
        }
        let plan = match arithmetic {
            Arithmetic::Double => Plans::Double(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                0_f64,
            )?),
            Arithmetic::DoubleDouble => Plans::DoubleDouble(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                DoubleFloat::from(0.0),
            )?),
            Arithmetic::Arbitrary => Plans::Arbitrary(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                Float::with_val(bits, 0),
            )?),
        };
        Ok(Self {
            plan,
            precision_bits: bits,
        })
    }

    /// Every buffer is lane-major. The caller's original lane IDs survive
    /// compaction and resizing; failed lanes return None, never stale values.
    pub fn evaluate_decimal(
        &mut self,
        ids: Vec<u64>,
        shifts: Vec<DecimalMomentum>,
        masses_squared: Vec<DecimalComplex>,
        scales_squared: Vec<DecimalComplex>,
        parameters: Vec<DecimalComplex>,
    ) -> ExecutionResult<Vec<RawLane>> {
        match &mut self.plan {
            Plans::Fast(p) => {
                p.evaluate_decimal(&ids, &shifts, &masses_squared, &scales_squared, &parameters)
            }
            Plans::Double(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
            Plans::DoubleDouble(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
            Plans::Arbitrary(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
        }
    }
}

/// All-lane diagnostics plus compact successful values; failed numbers stay hidden.
pub struct ReductionBatch {
    pub(super) values: Batch,
    pub(super) statuses: Vec<(u64, String)>,
}
impl ReductionBatch {
    pub fn statuses(&self) -> &[(u64, String)] {
        &self.statuses
    }
    pub fn values(&self) -> ExecutionResult<Batch> {
        if self.statuses.len() != self.values.rows() {
            return Err(error(
                "reduction contains failed lanes; inspect statuses and successful_values",
            ));
        }
        Ok(self.values.clone())
    }
    pub fn successful_values(&self) -> Batch {
        self.values.clone()
    }
}

fn reduce<T: BatchCarrier>(
    plan: &mut Reductions<T>,
    shifts: &[Complex<T>],
    masses: &[Complex<T>],
    scales: &[Complex<T>],
    parameters: &[Complex<T>],
    matrix: Option<&[Complex<T>]>,
    input: &Batch,
) -> ExecutionResult<ReductionBatch>
where
    Complex<T>: Real + EvaluationDomain,
{
    let shifts: Vec<_> = shifts
        .chunks_exact(4)
        .map(|p| FourMomentum::from([p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone()]))
        .collect();
    let lanes = plan.evaluate_native(input.ids(), &shifts, masses, scales, parameters, matrix)?;
    reduced_batch(lanes, input.precision_bits())
}
fn reduced_batch<T: BatchCarrier>(
    lanes: Vec<Lane<Complex<T>, String>>,
    bits: u32,
) -> ExecutionResult<ReductionBatch> {
    let mut values = Vec::with_capacity(shape(lanes.len(), 3)?);
    let mut ids = Vec::with_capacity(lanes.len());
    let mut statuses = Vec::with_capacity(lanes.len());
    for lane in lanes {
        statuses.push((lane.id, lane.status));
        if let Some(value) = lane.value {
            ids.push(lane.id);
            values.extend([value.finite, value.single_pole, value.double_pole]);
        }
    }
    Ok(ReductionBatch {
        values: Batch::typed(values, ids, 3, bits)?,
        statuses,
    })
}

impl PolynomialPlan {
    pub fn evaluate_batch(
        &mut self,
        shifts: &Batch,
        masses_squared: &Batch,
        scales_squared: &Batch,
        parameters: &Batch,
        s_matrix: Option<&Batch>,
    ) -> ExecutionResult<ReductionBatch> {
        for batch in [shifts, masses_squared, scales_squared] {
            parameters.same_rows(batch)?;
        }
        if let Some(matrix) = s_matrix {
            parameters.same_rows(matrix)?;
        }
        if parameters.precision_bits() != self.precision_bits {
            return Err(error("batch arithmetic or precision differs"));
        }
        macro_rules! run {
            ($plan:expr, $variant:ident) => {{
                let p = $plan;
                if shifts.columns() != p.propagators * 4
                    || masses_squared.columns() != p.propagators
                    || scales_squared.columns() != 1
                    || parameters.columns() != p.parameter_count
                    || s_matrix.is_some_and(|s| s.columns() != p.propagators * p.propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (
                    Data::$variant(shifts),
                    Data::$variant(masses),
                    Data::$variant(scales),
                    Data::$variant(values),
                ) = (
                    shifts.data(),
                    masses_squared.data(),
                    scales_squared.data(),
                    parameters.data(),
                )
                else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::$variant(v) = s.data() else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                reduce(p, shifts, masses, scales, values, matrix, parameters)
            }};
        }
        match &mut self.plan {
            Plans::Fast(p) => {
                let (propagators, parameter_count) = (p.fast.propagators, p.fast.parameter_count);
                if shifts.columns() != propagators * 4
                    || masses_squared.columns() != propagators
                    || scales_squared.columns() != 1
                    || parameters.columns() != parameter_count
                    || s_matrix.is_some_and(|s| s.columns() != propagators * propagators)
                {
                    return Err(error("reduction batch dimensions differ"));
                }
                let (Data::Double(s), Data::Double(m), Data::Double(r), Data::Double(c)) = (
                    shifts.data(),
                    masses_squared.data(),
                    scales_squared.data(),
                    parameters.data(),
                ) else {
                    return Err(error("batch arithmetic or precision differs"));
                };
                let matrix = s_matrix.map(|s| {
                    let Data::Double(v) = s.data() else {
                        unreachable!()
                    };
                    v.as_slice()
                });
                let shifts: Vec<_> = s
                    .chunks_exact(4)
                    .map(|v| FourMomentum::from([v[0], v[1], v[2], v[3]]))
                    .collect();
                reduced_batch(
                    p.evaluate_native(parameters.ids(), &shifts, m, r, c, matrix)?,
                    53,
                )
            }
            Plans::Double(p) => run!(p, Double),
            Plans::DoubleDouble(p) => run!(p, DoubleDouble),
            Plans::Arbitrary(p) => run!(p, Arbitrary),
        }
    }
}
