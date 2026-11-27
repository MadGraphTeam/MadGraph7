// SPDX-License-Identifier: 0BSD
//! Binary64 stability monitoring of the native runtime (m-lit PLAN §3.5, the
//! revision-2c ruling): the Python API's `NativeExecutor.evaluate` with
//! stability checks (`plan_export.NativeExecutor._runs`: the reference
//! evaluation re-tested in the reduction check's frame, one evaluation per
//! rotation at the exactly rotated points, the cross-backend evaluation on
//! the other backend), each point's estimate by the one implementation the
//! Python API's binary64 path calls (`crate::v2_stability`). A point a
//! rotated evaluation fails fails with the rotation named, as the Python
//! API's; a point the other backend fails or refuses keeps the estimate of
//! the other checks, the cross-backend check in its `not_run` record.

use super::induced::Induced;
use super::induced_outputs::{
    GuardForm, GuardPoint, GuardSink, InducedCall, InducedCollected, Monitoring, rows_of,
};
use super::*;
use crate::reduction_policy;
use crate::v2_decimal::DecimalContext;
use crate::v2_stability::{
    self as stability, AmplitudeCells, Budget, CrossRun, ErrorCells, Estimate, Frame, InducedPoint,
    InducedShape, Keyed, LaneShape, PointLanes, StabilityChecks,
};

/// A collected lane: its key, its sector and its Laurent values.
pub(crate) type CollectedLane<N> = (LaneKey, usize, Vec<C<N>>);

/// What a monitored batch collects per successful point
/// ([`V2Runtime::evaluate_batch`] with `COLLECT`): its reduction lanes and
/// re-test lanes (key, sector, Laurent values), its local components (per
/// component `r2`, `uv_local`, fields, per record sector the Laurent values)
/// and its reference values (the renormalised virtual per record sector).
pub(crate) struct Collected<N> {
    pub lanes: Vec<CollectedLane<N>>,
    pub retest: Option<Vec<CollectedLane<N>>>,
    pub local: Vec<Vec<Vec<N>>>,
    pub reference: Vec<Vec<N>>,
}

fn lanes64(lanes: &[CollectedLane<f64>]) -> Vec<stability::Lane<LaneKey>> {
    lanes
        .iter()
        .map(|(key, sector, values)| (*key, *sector, values.iter().map(|z| (z.re, z.im)).collect()))
        .collect()
}

impl Collected<f64> {
    fn point(&self) -> PointLanes<LaneKey> {
        PointLanes {
            lanes: lanes64(&self.lanes),
            local: self.local.clone(),
        }
    }
}

/// One monitored point: its values (or failures), for a successful point its
/// stability estimate and, for a loop-induced point with poles, its
/// finiteness ratio (`VirtualLaneDiagnostic.finiteness`).
pub type Monitored = (V2Point, Option<Estimate>, Option<f64>);

/// The name of a reduction backend in the cross-backend check's records
/// (`reduction.NAMES`).
fn backend_name(reduction: Reduction) -> &'static str {
    match reduction {
        Reduction::NinjaRs => "NinjaRS",
        Reduction::TiRust => "TIRust",
    }
}

/// The other reduction backend (`reduction.cross`).
pub fn cross_reduction(reduction: Reduction) -> Reduction {
    match reduction {
        Reduction::NinjaRs => Reduction::TiRust,
        Reduction::TiRust => Reduction::NinjaRs,
    }
}

impl V2Runtime {
    /// A binary64 batch with the stability checks `checks` (the Python API's
    /// `Runtime.evaluate(..., stability_checks=...)` in binary64): per point
    /// its values or failures, as [`V2Runtime::evaluate`], and its estimate.
    /// Without an enabled check this is [`V2Runtime::evaluate`] (no
    /// estimate, nothing else runs). The checks serve Born-interfered
    /// processes in the pole selections "all" and "none".
    pub fn evaluate_checked(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        checks: &StabilityChecks,
    ) -> RusticolResult<Vec<Monitored>> {
        if !checks.enabled() {
            return Ok(self
                .evaluate(momenta, scales, mode)?
                .into_iter()
                .map(|point| (point, None, None))
                .collect());
        }
        if let Some(induced) = self.induced.clone() {
            self.check_batch(momenta, scales)?;
            self.check_poles(mode)?;
            return self.partitioned(momenta, scales, mode, |runtime, momenta, scales| {
                runtime.induced_checked_batch(&induced, momenta, scales, mode, checks)
            });
        }
        if mode.split() {
            return Err(RusticolError::unsupported_runtime_capability(
                "stability checks",
                "the native stability checks serve the pole selections \"all\" and \"none\"; \
                 the Python API estimates the \"uv\" and \"ir\" selections",
            ));
        }
        self.partitioned(momenta, scales, mode, |runtime, momenta, scales| {
            runtime.checked_batch(momenta, scales, mode, checks)
        })
    }

    /// One batch of one scale partition with its checks
    /// (`NativeExecutor._runs` and `point_estimate`).
    fn checked_batch(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        checks: &StabilityChecks,
    ) -> RusticolResult<Vec<Monitored>> {
        let frame = checks.frame().map(|frame| match frame {
            Frame::Reversed => ReductionCheck::REVERSED,
            Frame::Rescaled => ReductionCheck::RESCALED,
        });
        let mut collected = Vec::with_capacity(momenta.len());
        let points = self.evaluate_batch::<f64, Binary64, true>(
            momenta,
            scales,
            mode,
            V2Precision::Double,
            &0.0,
            &Binary64,
            frame,
            &mut collected,
        )?;
        let mut failures: Vec<PointFailure> = points
            .iter()
            .map(|point| point.as_ref().err().cloned().unwrap_or_default())
            .collect();
        let mut rotated = Vec::with_capacity(checks.rotations().len());
        for (k, rotation) in checks.rotations().iter().enumerate() {
            let moved = rotate(rotation, momenta)?;
            let mut runs = Vec::with_capacity(momenta.len());
            let turned = self.evaluate_batch::<f64, Binary64, true>(
                &moved,
                scales,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
                None,
                &mut runs,
            )?;
            for (index, point) in turned.iter().enumerate() {
                if let Err(reasons) = point {
                    failures[index].extend(
                        reasons
                            .iter()
                            .map(|reason| format!("{}: {reason}", stability::rotation_name(k))),
                    );
                    failures[index]
                        .measures
                        .extend(reasons.measures.iter().copied());
                }
            }
            rotated.push(runs);
        }
        let cross = if checks.cross_backend() {
            Some(self.cross_runs(momenta, scales, mode)?)
        } else {
            None
        };
        let shape = self.lane_shape(mode);
        let roundoff = 2.0_f64.powi(-53);
        let mut out = Vec::with_capacity(points.len());
        for (index, point) in points.into_iter().enumerate() {
            if !failures[index].is_empty() {
                out.push((Err(std::mem::take(&mut failures[index])), None, None));
                continue;
            }
            let run = |runs: &[Option<Collected<f64>>]| {
                runs[index]
                    .as_ref()
                    .map(Collected::point)
                    .ok_or_else(|| invalid("a successful point without its collected lanes"))
            };
            let direct = collected[index]
                .as_ref()
                .ok_or_else(|| invalid("a successful point without its collected lanes"))?;
            let rotated = rotated
                .iter()
                .map(|runs| run(runs))
                .collect::<RusticolResult<Vec<_>>>()?;
            let retest = match checks.reduction() {
                Some(name) => Some((
                    name,
                    lanes64(
                        direct
                            .retest
                            .as_ref()
                            .ok_or_else(|| invalid("the reduction check returned no re-test"))?,
                    ),
                )),
                None => None,
            };
            let estimate = stability::born_interfered_estimate(
                shape,
                roundoff,
                &direct.reference,
                &direct.point(),
                &rotated,
                retest
                    .as_ref()
                    .map(|(name, lanes)| (*name, lanes.as_slice())),
                cross.as_ref().map(|runs| &runs[index]),
            )?;
            out.push((point, Some(estimate), None));
        }
        Ok(out)
    }

    /// One loop-induced batch of one scale partition with its checks
    /// (`NativeExecutor._runs` of an amplitude-mode plan: the reference
    /// evaluation with the re-test frame and the term scale, the rotated and
    /// cross-backend evaluations without; `point_estimate`).
    fn induced_checked_batch(
        &mut self,
        induced: &Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        checks: &StabilityChecks,
    ) -> RusticolResult<Vec<Monitored>> {
        let frame = checks.frame().map(|frame| match frame {
            Frame::Reversed => ReductionCheck::REVERSED,
            Frame::Rescaled => ReductionCheck::RESCALED,
        });
        let gram = induced.carrier(&Binary64, induced.metric())?;
        let rows = induced.plan_rows();
        let average = self.record.average;
        let mut collected = Vec::with_capacity(momenta.len());
        let points = self.induced_points_with(
            induced,
            momenta,
            scales,
            mode,
            V2Precision::Double,
            &0.0,
            &Binary64,
            Monitoring {
                check: frame,
                term_scale: true,
                collected: Some(&mut collected),
            },
            None,
            &mut |_, vectors, width| {
                induced.square_vectors(&Binary64, vectors, &rows, &gram, width, average)
            },
        )?;
        let mut failures: Vec<PointFailure> = points
            .iter()
            .map(|point| point.as_ref().err().cloned().unwrap_or_default())
            .collect();
        let mut rotated = Vec::with_capacity(checks.rotations().len());
        for (k, rotation) in checks.rotations().iter().enumerate() {
            let moved = rotate(rotation, momenta)?;
            let mut runs = Vec::with_capacity(momenta.len());
            let turned = self.induced_points_with(
                induced,
                &moved,
                scales,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
                Monitoring {
                    check: None,
                    term_scale: false,
                    collected: Some(&mut runs),
                },
                None,
                &mut |_, _, _| Ok(()),
            )?;
            for (index, point) in turned.iter().enumerate() {
                if let Err(reasons) = point {
                    failures[index].extend(
                        reasons
                            .iter()
                            .map(|reason| format!("{}: {reason}", stability::rotation_name(k))),
                    );
                    failures[index]
                        .measures
                        .extend(reasons.measures.iter().copied());
                }
            }
            rotated.push(runs);
        }
        let cross = if checks.cross_backend() {
            Some(self.induced_cross_runs(induced, momenta, scales, mode)?)
        } else {
            None
        };
        // The key tables: loop orders (the squaring's slots first) and plan
        // rows by id.
        let mut orders: Vec<Orders> = induced.slots().to_vec();
        let shape = induced_shape(
            induced,
            gram.iter().map(|z| (z.re, z.im)).collect(),
            rows.len(),
            mode,
            average,
        );
        let mut convert =
            |found: &InducedCollected<f64>| induced_point(induced, &rows, &mut orders, found);
        let mut out = Vec::with_capacity(points.len());
        for (index, point) in points.into_iter().enumerate() {
            if !failures[index].is_empty() {
                out.push((Err(std::mem::take(&mut failures[index])), None, None));
                continue;
            }
            let squared = point.map_err(|_| invalid("a successful point without values"))?;
            let missing = || invalid("a successful point without its collected vectors");
            let direct = convert(collected[index].as_ref().ok_or_else(missing)?)?;
            let turned = rotated
                .iter()
                .map(|runs: &Vec<Option<InducedCollected<f64>>>| {
                    convert(runs[index].as_ref().ok_or_else(missing)?)
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let other = match cross.as_ref().map(|runs| &runs[index]) {
                Some(CrossRun::Run(found)) => Some(CrossRun::Run(convert(found)?)),
                Some(CrossRun::NotRun(why)) => Some(CrossRun::NotRun(why.clone())),
                None => None,
            };
            let estimate = stability::induced_estimate(
                &shape,
                2.0_f64.powi(-53),
                &squared,
                &direct,
                &turned,
                checks.reduction(),
                other.as_ref(),
            )?;
            let finiteness = shape.finiteness(&direct)?;
            let values = self
                .sector_index
                .iter()
                .map(|&s| squared[s].clone())
                .collect();
            out.push((Ok((Vec::new(), values)), Some(estimate), finiteness));
        }
        Ok(out)
    }

    /// [`V2Runtime::cross_runs`] of a loop-induced batch.
    fn induced_cross_runs(
        &mut self,
        induced: &Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
    ) -> RusticolResult<Vec<CrossRun<InducedCollected<f64>>>> {
        let other = cross_reduction(self.reduction);
        let name = backend_name(other);
        self.swap_cross(other);
        let result: RusticolResult<Vec<CrossRun<InducedCollected<f64>>>> = (|| {
            if let Err(refusal) = self.backend(V2Precision::Double) {
                let why = format!(
                    "{name} refused the call: reduction '{}' cannot serve this artifact: {refusal}",
                    other.name()
                );
                return Ok((0..momenta.len())
                    .map(|_| CrossRun::NotRun(why.clone()))
                    .collect());
            }
            let mut runs = Vec::with_capacity(momenta.len());
            let points = self.induced_points_with(
                induced,
                momenta,
                scales,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
                Monitoring {
                    check: None,
                    term_scale: false,
                    collected: Some(&mut runs),
                },
                None,
                &mut |_, _, _| Ok(()),
            )?;
            points
                .iter()
                .zip(runs)
                .map(|(point, run)| match (point, run) {
                    (Err(reasons), _) => Ok(CrossRun::NotRun(format!(
                        "{name} failed the point: {}",
                        reasons.join("; ")
                    ))),
                    (Ok(_), Some(run)) => Ok(CrossRun::Run(run)),
                    (Ok(_), None) => Err(invalid("a successful point without its vectors")),
                })
                .collect()
        })();
        self.swap_cross(other);
        result
    }

    /// The cross-backend check's evaluations (`NativeExecutor._cross_runs`):
    /// every point again on the other backend, at the same inputs, without
    /// re-test; a point it fails, or every point when it refuses the
    /// artifact (its plans), keeps the reason instead.
    fn cross_runs(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
    ) -> RusticolResult<Vec<CrossRun<PointLanes<LaneKey>>>> {
        let other = cross_reduction(self.reduction);
        let name = backend_name(other);
        self.swap_cross(other);
        let result: RusticolResult<Vec<CrossRun<PointLanes<LaneKey>>>> = (|| {
            if let Err(refusal) = self.backend(V2Precision::Double) {
                let why = format!(
                    "{name} refused the call: reduction '{}' cannot serve this artifact: {refusal}",
                    other.name()
                );
                return Ok((0..momenta.len())
                    .map(|_| CrossRun::NotRun(why.clone()))
                    .collect());
            }
            let mut runs = Vec::with_capacity(momenta.len());
            let points = self.evaluate_batch::<f64, Binary64, true>(
                momenta,
                scales,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
                None,
                &mut runs,
            )?;
            points
                .iter()
                .zip(runs)
                .map(|(point, run)| match (point, run) {
                    (Err(reasons), _) => Ok(CrossRun::NotRun(format!(
                        "{name} failed the point: {}",
                        reasons.join("; ")
                    ))),
                    (Ok(_), Some(run)) => Ok(CrossRun::Run(run.point())),
                    (Ok(_), None) => Err(invalid("a successful point without its lanes")),
                })
                .collect()
        })();
        self.swap_cross(other);
        result
    }

    /// [`V2Runtime::evaluate_extended`] of a loop-induced process with the
    /// per-point fallback's GUARD (m-lit lit-fallback-guard,
    /// `reduction_policy::FALLBACK_CHECK`; the Python API's
    /// `loops.v2.reduction.with_fallback`; the summed calls): per point its
    /// values, as `evaluate_extended` returns them (the same executor run and
    /// assembly, with the term scale collected beside it), the guard's
    /// rotation re-test (the points rotated exactly by the guard's rotation
    /// and evaluated again, [`V2Runtime::guard_rotated`]; a point whose
    /// rotated evaluation fails is failed with its reasons), and, for a
    /// successful point, the guard's estimate of its summed value
    /// (`reduction_policy::fallback_estimate` of the one output) in the
    /// extended carrier `precision` (the guard runs in double-double and
    /// arbitrary precision only, USER 2026-11-09 ~11:53; never escalated),
    /// on the runtime's current backend (the fallback's, inside
    /// [`V2Runtime::on_other`]). The monitored run is read in binary64 into
    /// the shared estimate function (`v2_stability::induced_estimate`, that
    /// of the binary64 monitoring), as the Python API's assembly reads the
    /// executor's budgets (`float(x)`), at the carrier's unit roundoff
    /// ([`V2Precision::unit_roundoff`]). A Born-interfered process has no
    /// per-point fallback (`reduction_policy::resolve`), so no guard.
    pub fn evaluate_extended_guarded(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<(V2Point<Decimal>, Option<f64>)>> {
        let induced = self.guarded_induced(precision)?;
        self.check_batch(momenta, scales)?;
        self.check_poles(mode)?;
        if mode.split() {
            return Err(RusticolError::unsupported_runtime_capability(
                "stability checks",
                "the native stability checks serve the pole selections \"all\" and \"none\"",
            ));
        }
        let arith = DecimalArith::new(precision.working_digits());
        let mut batch = |runtime: &mut Self,
                         momenta: &[Vec<Rational>],
                         scales: &[Rational]|
         -> RusticolResult<Vec<(V2Point<Decimal>, Option<f64>)>> {
            match precision {
                V2Precision::DoubleDouble => runtime.induced_guarded(
                    &induced,
                    momenta,
                    scales,
                    mode,
                    precision,
                    &DoubleFloat::from(0.0),
                    &arith,
                ),
                _ => runtime.induced_guarded(
                    &induced,
                    momenta,
                    scales,
                    mode,
                    precision,
                    &Float::with_val(precision.bits()?, 0),
                    &arith,
                ),
            }
        };
        self.partitioned(momenta, scales, mode, &mut batch)
    }

    /// The outputs of a loop-induced `call` in an extended carrier
    /// ([`V2Runtime::evaluate_induced_call_extended`]; `spins` per point its
    /// spin source assignment, correlated calls only) with the per-point
    /// fallback's GUARD (m-lit lit-fallback-guard, DECISIONS 2026-11-09 14:30
    /// RULING Q7; the resolved, correlated and selected calls): per point its
    /// outputs, as that call returns them (the same executor run of the
    /// call's rows only, with the term scale collected beside it), the call
    /// again at the points and spin sources rotated exactly by the guard's
    /// rotation (its check since DECISIONS 2026-11-10 19:20 (c); a point
    /// whose rotated call fails is failed with its reasons), and, for a
    /// successful point, the guard's estimate of THOSE outputs
    /// (`reduction_policy::fallback_estimate`: each output's rotation
    /// estimate against its own magnitude, a vanishing one against the
    /// largest returned output), each read in binary64 as the Python API's
    /// `InducedColumns.evaluate` estimates it with the guard's rotation alone
    /// (`point_estimate` for the inclusive |A|^2 of every row without a spin
    /// assignment, `PointOutputs.estimate` otherwise;
    /// [`call_guard_estimate`]), at the carrier's unit roundoff.
    #[allow(clippy::type_complexity)]
    pub fn evaluate_induced_call_guarded(
        &mut self,
        call: &InducedCall,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<(ColumnPoint<Decimal>, Option<f64>)>> {
        let induced = self.guarded_induced(precision)?;
        self.check_batch(momenta, scales)?;
        self.check_poles(mode)?;
        let arith = DecimalArith::new(precision.working_digits());
        // The guard's rotation re-test (`reduction_policy::FALLBACK_CHECK`):
        // the call again at the points and spin sources rotated exactly.
        let rotation = guard_rotation()?;
        let moved = rotate(&rotation, momenta)?;
        let turned_spins = rotate_spins(&rotation, spins)?;
        let mut sink = GuardSink::default();
        let mut turned_sink = GuardSink::default();
        let (points, turned) = match precision {
            V2Precision::DoubleDouble => {
                let zero = DoubleFloat::from(0.0);
                let points = self.induced_call_with(
                    call,
                    momenta,
                    scales,
                    spins,
                    mode,
                    precision,
                    &zero,
                    &arith,
                    Some(&mut sink),
                )?;
                let turned = self.induced_call_with(
                    call,
                    &moved,
                    scales,
                    &turned_spins,
                    mode,
                    precision,
                    &zero,
                    &arith,
                    Some(&mut turned_sink),
                )?;
                (points, turned)
            }
            _ => {
                let zero = Float::with_val(precision.bits()?, 0);
                let points = self.induced_call_with(
                    call,
                    momenta,
                    scales,
                    spins,
                    mode,
                    precision,
                    &zero,
                    &arith,
                    Some(&mut sink),
                )?;
                let turned = self.induced_call_with(
                    call,
                    &moved,
                    scales,
                    &turned_spins,
                    mode,
                    precision,
                    &zero,
                    &arith,
                    Some(&mut turned_sink),
                )?;
                (points, turned)
            }
        };
        let average = self.record.average;
        let roundoff = precision.unit_roundoff();
        points
            .into_iter()
            .zip(turned)
            .enumerate()
            .map(|(index, (point, turned))| {
                if let Err(mut reasons) = point {
                    // A failed point keeps the fallback's own failures, then
                    // its rotation's, as the Python API's guarded evaluation
                    // reports them (the guard's check runs with it).
                    if let Err(more) = &turned {
                        reasons.absorb(guard_rotation_failure(more));
                    }
                    return Ok((Err(reasons), None));
                }
                if let Err(reasons) = turned {
                    // The guard's check fails the point: the fallback
                    // failed it.
                    return Ok((Err(guard_rotation_failure(&reasons)), None));
                }
                let missing = || invalid("a successful point without its guard's record");
                let found = sink.points.get(&index).ok_or_else(missing)?;
                let other = turned_sink.points.get(&index).ok_or_else(missing)?;
                let estimate = call_guard_estimate(
                    &induced,
                    average,
                    mode,
                    roundoff,
                    &arith.context,
                    &sink.forms,
                    found,
                    other,
                )?;
                Ok((point, Some(estimate)))
            })
            .collect()
    }

    /// [`V2Runtime::evaluate_induced_call_guarded`] of a stored column set
    /// (`kind`, [`V2Runtime::evaluate_columns_extended`]'s loop-induced
    /// calls: `correlated` with the points' spin sources, `resolved`).
    #[allow(clippy::type_complexity)]
    pub fn evaluate_columns_guarded(
        &mut self,
        kind: &str,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<(ColumnPoint<Decimal>, Option<f64>)>> {
        self.guarded_induced(precision)?;
        let call = Self::column_call(kind)?;
        self.evaluate_induced_call_guarded(&call, momenta, scales, spins, mode, precision)
    }

    /// The loop-induced process of a guarded evaluation in the extended
    /// carrier `precision`: the guard runs in double-double and arbitrary
    /// precision only, and only a loop-induced process has a per-point
    /// fallback (`reduction_policy::resolve`: a Born-interfered call's
    /// default is NinjaRS, with no fallback).
    fn guarded_induced(&self, precision: V2Precision) -> RusticolResult<Arc<Induced>> {
        if !reduction_policy::fallback_guarded(precision.name()) {
            return Err(invalid(
                "the per-point fallback's guard runs in the extended carriers only",
            ));
        }
        self.induced.clone().ok_or_else(|| {
            invalid(
                "the per-point fallback's guard serves loop-induced processes (a \
                 Born-interfered call has no per-point fallback)",
            )
        })
    }

    /// [`V2Runtime::evaluate_extended_guarded`] of one loop-induced batch of
    /// one scale partition in an extended carrier: `evaluate_induced`'s
    /// assembly (the colour metric in the carrier, every plan row) on
    /// `induced_checked_batch`'s reference evaluation (the re-test frame
    /// and the term scale).
    #[allow(clippy::too_many_arguments)]
    fn induced_guarded<T: Native>(
        &mut self,
        induced: &Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &DecimalArith,
    ) -> RusticolResult<Vec<(V2Point<Decimal>, Option<f64>)>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let gram = induced.carrier(arith, induced.metric())?;
        let rows = induced.plan_rows();
        let average = self.record.average;
        let sector_index = self.sector_index.clone();
        let mut collected = Vec::with_capacity(momenta.len());
        let points = self.induced_points_with(
            induced,
            momenta,
            scales,
            mode,
            precision,
            zero,
            arith,
            Monitoring {
                check: None,
                term_scale: true,
                collected: Some(&mut collected),
            },
            None,
            &mut |_, vectors, width| {
                induced.square_vectors(arith, vectors, &rows, &gram, width, average)
            },
        )?;
        // The guard's rotation re-test (`reduction_policy::FALLBACK_CHECK`):
        // the same points rotated exactly, evaluated again in the carrier.
        let (turned, mut runs) =
            self.guard_rotated(induced, momenta, scales, mode, precision, zero, arith)?;
        // The estimate reads binary64 (the Python assembly's floats): the
        // colour metric, the values and the collected vectors.
        let metric = induced.carrier(&Binary64, induced.metric())?;
        let shape = induced_shape(
            induced,
            metric.iter().map(|z| (z.re, z.im)).collect(),
            rows.len(),
            mode,
            average,
        );
        let mut orders: Vec<Orders> = induced.slots().to_vec();
        let mut out = Vec::with_capacity(points.len());
        for (index, point) in points.into_iter().enumerate() {
            let squared = match point {
                Ok(squared) => squared,
                Err(mut reasons) => {
                    // The fallback's own failures, then its rotation's (the
                    // Python API's guarded evaluation).
                    if let Some(more) = turned[index].as_ref().err() {
                        reasons.absorb(guard_rotation_failure(more));
                    }
                    out.push((Err(reasons), None));
                    continue;
                }
            };
            if let Some(reasons) = turned[index].as_ref().err() {
                // The guard's check fails the point: the fallback failed it.
                out.push((Err(guard_rotation_failure(reasons)), None));
                continue;
            }
            let missing = || invalid("a successful point without its collected vectors");
            let found = collected[index].as_ref().ok_or_else(missing)?;
            let direct = induced_point_of(induced, &rows, &mut orders, found, Decimal::to_f64)?;
            let turned = runs[index].take().ok_or_else(missing)?;
            let reference: Vec<Vec<f64>> = squared
                .iter()
                .map(|laurent| laurent.iter().map(Decimal::to_f64).collect())
                .collect();
            let estimate = guard_rotation_estimate(
                &shape,
                &arith.context,
                precision.unit_roundoff(),
                &reference,
                (&rows, &mut orders),
                (&direct, found),
                &turned,
            )?;
            let values = sector_index.iter().map(|&s| squared[s].clone()).collect();
            let estimate = reduction_policy::fallback_estimate(
                &[guard_sectors(&reference, &estimate)],
                precision.unit_roundoff(),
            );
            out.push((Ok((Vec::new(), values)), Some(estimate)));
        }
        Ok(out)
    }

    /// The per-point fallback guard's rotated evaluation of a loop-induced
    /// batch in an extended carrier (`reduction_policy::FALLBACK_CHECK`): the
    /// points rotated exactly by the guard's rotation
    /// ([`guard_rotation`]) and evaluated again on the current backend (the
    /// fallback's), per point its outcome and its collected run (the Python
    /// API's rotation check of `VirtualStabilityChecks(rotations=1)`).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn guard_rotated<T: Native>(
        &mut self,
        induced: &Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &DecimalArith,
    ) -> RusticolResult<(
        Vec<Result<(), PointFailure>>,
        Vec<Option<InducedCollected<Decimal>>>,
    )>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let moved = rotate(&guard_rotation()?, momenta)?;
        let mut runs = Vec::with_capacity(momenta.len());
        let turned = self.induced_points_with(
            induced,
            &moved,
            scales,
            mode,
            precision,
            zero,
            arith,
            Monitoring {
                check: None,
                term_scale: false,
                collected: Some(&mut runs),
            },
            None,
            &mut |_, _, _| Ok(()),
        )?;
        Ok((turned, runs))
    }

    /// The lane shape of a Born-interfered monitored batch in the pole
    /// selection `mode` (its sectors, Laurent width and lane factor).
    fn lane_shape(&self, mode: V2Poles) -> LaneShape {
        LaneShape {
            sectors: self.record.sectors.len(),
            width: if mode.poles() { 3 } else { 1 },
            factor: stability::lane_factor(self.record.average as f64),
        }
    }

    /// Run `evaluate` on the other reduction backend ([`cross_reduction`]),
    /// then return to this one: the evaluations of `"auto"`'s per-point
    /// fallback (`reduction_policy::Choice::fallback`, m-lit F3 (c)) at the
    /// carriers `evaluate` uses, on the cross-backend check's backends (built
    /// on first use and kept with the runtime, so a fallback rebuilds
    /// nothing). NinjaRS on a TIR-only process is refused typed.
    pub fn on_other<R>(
        &mut self,
        evaluate: impl FnOnce(&mut Self) -> RusticolResult<R>,
    ) -> RusticolResult<R> {
        let other = cross_reduction(self.reduction);
        if other == Reduction::NinjaRs {
            crate::runtime::ninjars_refusal(&self.process.plan)?;
        }
        if other == Reduction::TiRust && !crate::tir::available() {
            return Err(RusticolError::unsupported_runtime_capability(
                "tirust",
                crate::runtime::TIRUST_NOT_BUILT,
            ));
        }
        self.swap_cross(other);
        let result = evaluate(self);
        self.swap_cross(other);
        result
    }

    /// Exchange the evaluation backends with the cross-backend check's
    /// (`other`'s, built on first use and kept with the runtime).
    fn swap_cross(&mut self, other: Reduction) {
        std::mem::swap(&mut self.backends, &mut self.cross_backends);
        self.reduction = if self.reduction == other {
            cross_reduction(other)
        } else {
            other
        };
    }
}

/// Every point's momenta rotated exactly (`Rotation.apply` of each point,
/// whose legs are its four-component chunks).
fn rotate(
    rotation: &stability::Rotation,
    momenta: &[Vec<Rational>],
) -> RusticolResult<Vec<Vec<Rational>>> {
    momenta
        .iter()
        .map(|point| {
            let legs: Vec<Vec<Rational>> =
                point.chunks_exact(4).map(<[Rational]>::to_vec).collect();
            Ok(rotation.apply(&legs)?.concat())
        })
        .collect()
}

/// The per-point fallback guard's rotation (`reduction_policy::
/// FALLBACK_ROTATION` of `v2_stability::default_rotation`; the Python API's
/// `loops.v2.reduction.guard_rotation`).
fn guard_rotation() -> RusticolResult<stability::Rotation> {
    stability::default_rotation(reduction_policy::FALLBACK_ROTATION)
}

/// The reasons of a point whose guard rotation failed (the rotation check's
/// own failure text, `stability::rotation_name` 0: the Python API's).
fn guard_rotation_failure(reasons: &PointFailure) -> PointFailure {
    PointFailure::new(
        reasons
            .iter()
            .map(|reason| format!("{}: {reason}", stability::rotation_name(0)))
            .collect(),
        reasons.measures.clone(),
    )
}

/// Every point's spin source vectors rotated exactly with its momenta
/// (real and imaginary parts, `observables._rotated_spin`).
#[allow(clippy::type_complexity)]
fn rotate_spins(
    rotation: &stability::Rotation,
    spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
) -> RusticolResult<Vec<BTreeMap<usize, [(Rational, Rational); 4]>>> {
    spins
        .iter()
        .map(|spin| {
            spin.iter()
                .map(|(&leg, vector)| {
                    let parts = rotation.apply(&[
                        vector.iter().map(|(re, _)| re.clone()).collect(),
                        vector.iter().map(|(_, im)| im.clone()).collect(),
                    ])?;
                    let rotated: [(Rational, Rational); 4] =
                        std::array::from_fn(|mu| (parts[0][mu].clone(), parts[1][mu].clone()));
                    Ok((leg, rotated))
                })
                .collect()
        })
        .collect()
}

/// The guard's rotation estimate of a loop-induced point in an extended
/// carrier (`induced.rotation_budget` as the Python API forms it there):
/// the little-group phase per row (`induced._row_phases`: sum conj(a') a
/// over the row's loop orders and t at k = 0, its modulus
/// sqrt(re^2 + im^2) and 1 / modulus in the carrier's context `context`)
/// and the deviations (`induced._rotation_deviations` of a native point:
/// per key of the counterterm kets, then of the bare vectors, |y phase - x|
/// formed in the context and read as `abs(complex(.))` in binary64), then
/// `quadratic_budget` and the estimate of that one check on the binary64
/// point `direct` (`v2_stability::induced_estimate` of a rotation, whose
/// binary64 deviations would floor the estimate at the binary64 rounding
/// of the vectors). `rows` / `orders` key the vectors as
/// [`induced_point_of`] did for `direct`.
#[allow(clippy::too_many_arguments)]
fn guard_rotation_estimate(
    shape: &InducedShape,
    context: &DecimalContext,
    roundoff: f64,
    reference: &Budget,
    (rows, orders): (&[induced::Row], &mut Vec<Orders>),
    (direct, found): (&InducedPoint, &InducedCollected<Decimal>),
    turned: &InducedCollected<Decimal>,
) -> RusticolResult<Estimate> {
    let row_of = |row: &induced::Row| -> RusticolResult<usize> {
        rows.iter()
            .position(|r| r == row)
            .ok_or_else(|| invalid("a vector of a row outside the plan's rows"))
    };
    let negate = |x: &Decimal| context.plus(&x.negated());
    let times = |(a, b): (&Decimal, &Decimal), (c, d): (&Decimal, &Decimal)| {
        (
            context.sub(&context.mul(a, c), &context.mul(b, d)),
            context.add(&context.mul(a, d), &context.mul(b, c)),
        )
    };
    let mut phases: Vec<(usize, (Decimal, Decimal))> = Vec::with_capacity(shape.rows.len());
    for &r in &shape.rows {
        let (mut re, mut im) = (Decimal::zero(), Decimal::zero());
        for (key, vector) in found.parts.total.iter() {
            if row_of(&key.1)? != r {
                continue;
            }
            let Some(other) = turned.parts.total.get(key) else {
                continue;
            };
            for t in 0..shape.size {
                let (y, x) = (&other[t][0], &vector[t][0]);
                let (pr, pi) = times((&y.re, &negate(&y.im)), (&x.re, &x.im));
                re = context.add(&re, &pr);
                im = context.add(&im, &pi);
            }
        }
        let magnitude = context.sqrt(&context.add(&context.mul(&re, &re), &context.mul(&im, &im)));
        let phase = if magnitude.is_zero() {
            (Decimal::integer(1), Decimal::zero())
        } else {
            let inverse = context.div(&Decimal::integer(1), &magnitude);
            (context.mul(&re, &inverse), context.mul(&im, &inverse))
        };
        phases.push((r, phase));
    }
    let mut errors: Keyed<ErrorCells> = Vec::new();
    for (mine, theirs) in [
        (&found.parts.kets, &turned.parts.kets),
        (&found.parts.bare, &turned.parts.bare),
    ] {
        let mut keys: Vec<&(Orders, induced::Row)> = Vec::new();
        for (key, _) in mine.iter().chain(theirs.iter()) {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        for key in keys {
            let (a, b) = (mine.get(key), theirs.get(key));
            let r = row_of(&key.1)?;
            let (pr, pi) = &phases
                .iter()
                .find(|(row, _)| *row == r)
                .ok_or_else(|| invalid("a vector of a row outside the plan's rows"))?
                .1;
            let o = match orders.iter().position(|o| *o == key.0) {
                Some(o) => o,
                None => {
                    orders.push(key.0.clone());
                    orders.len() - 1
                }
            };
            let at = match errors.iter().position(|(k, _)| *k == (o, r)) {
                Some(at) => at,
                None => {
                    errors.push(((o, r), vec![vec![0.0; shape.width]; shape.size]));
                    errors.len() - 1
                }
            };
            let zero = Decimal::zero();
            for t in 0..shape.size {
                for k in 0..shape.width {
                    let (xr, xi) = a.map_or((&zero, &zero), |v| (&v[t][k].re, &v[t][k].im));
                    let (yr, yi) = b.map_or((&zero, &zero), |v| (&v[t][k].re, &v[t][k].im));
                    let (zr, zi) = times((yr, yi), (pr, pi));
                    let (dr, di) = (context.sub(&zr, xr), context.sub(&zi, xi));
                    errors[at].1[t][k] += stability::modulus(dr.to_f64(), di.to_f64());
                }
            }
        }
    }
    let budget = shape.quadratic_budget(direct, &errors);
    Estimate::from_checks(
        reference,
        vec![(stability::rotation_name(0), budget)],
        shape.scale(direct)?,
        roundoff,
        Vec::new(),
    )
}

/// A collected loop-induced run as the shared estimate reads it
/// (`InducedPoint`): keys by (loop order id in `orders`, plan row id), the
/// executor's per-slot re-test budget and term scale per (plan order, row)
/// as `induced.read_native` reads them.
fn induced_point(
    induced: &Induced,
    rows: &[induced::Row],
    orders: &mut Vec<Orders>,
    found: &InducedCollected<f64>,
) -> RusticolResult<InducedPoint> {
    induced_point_of(induced, rows, orders, found, |x: &f64| *x)
}

/// One output's sectors as the per-point fallback's guard reads its estimate
/// (`reduction_policy::GuardSector`; the Python API's
/// `loops.v2.reduction._guard_sectors`): per record sector the largest
/// |reference| (the `"relative"` metric's scale), the largest absolute
/// estimate E and the largest term scale S; NaN where a value is not a
/// number (a NaN budget is no estimate, and the largest ratio would pass over
/// it).
fn guard_sectors(
    reference: &[Vec<f64>],
    estimate: &Estimate,
) -> Vec<reduction_policy::GuardSector> {
    fn largest(values: &[f64]) -> f64 {
        let mut worst = 0.0_f64;
        for value in values {
            let value = value.abs();
            if value.is_nan() {
                return f64::NAN;
            }
            if value > worst {
                worst = value;
            }
        }
        worst
    }
    reference
        .iter()
        .enumerate()
        .map(|(s, laurent)| reduction_policy::GuardSector {
            magnitude: largest(laurent),
            error: estimate.absolute.get(s).map_or(f64::NAN, |e| largest(e)),
            scale: estimate.scale.get(s).map_or(f64::NAN, |e| largest(e)),
        })
        .collect()
}

/// Absolute cells per (loop order, row), in insertion order (the Python
/// API's dicts of floats, `induced_observables._absolute`).
type RowCells = Vec<((Orders, induced::Row), ErrorCells)>;

/// `induced_observables._spin_errors` without coefficient changes: the
/// absolute errors (or floors) of the amplitudes moved onto a spin
/// assignment per (loop order, remaining row), from the rows' cells e, sum
/// prod_j |c_j| e (a row whose spin-leg helicity has no coefficient, or a
/// zero weight, adds nothing); the cells themselves without an assignment.
fn spin_moved(cells: RowCells, magnitudes: Option<&BTreeMap<usize, Vec<(i32, f64)>>>) -> RowCells {
    let Some(magnitudes) = magnitudes else {
        return cells;
    };
    let mut out: RowCells = Vec::new();
    for ((order, row), values) in cells {
        let mut weight = 1.0_f64;
        let mut rest: induced::Row = Vec::new();
        for &(leg, h) in &row {
            match magnitudes.get(&leg) {
                None => rest.push((leg, h)),
                Some(found) => {
                    weight *= found
                        .iter()
                        .find(|(helicity, _)| *helicity == h)
                        .map_or(0.0, |(_, size)| *size);
                }
            }
        }
        if weight == 0.0 {
            continue;
        }
        let key = (order, rest);
        let at = match out.iter().position(|(k, _)| *k == key) {
            Some(at) => at,
            None => {
                out.push((
                    key,
                    values.iter().map(|cell| vec![0.0; cell.len()]).collect(),
                ));
                out.len() - 1
            }
        };
        let target = &mut out[at].1;
        for (t, cell) in values.iter().enumerate() {
            for (k, value) in cell.iter().enumerate() {
                target[t][k] += weight * value;
            }
        }
    }
    out
}

/// The per-point fallback guard's estimate of one point of a guarded
/// loop-induced call ([`V2Runtime::evaluate_induced_call_guarded`]; m-lit
/// lit-fallback-guard, DECISIONS 2026-11-09 14:30 Q7; its rotation re-test,
/// DECISIONS 2026-11-10 19:20 (c)): per output the estimate of the guard's
/// rotation alone that the Python API's `InducedColumns.evaluate` forms
/// (the inclusive |A|^2 of every row without a spin assignment:
/// `point_estimate`, `v2_stability::induced_estimate` of the rotated run;
/// any other output: `PointOutputs.budgets`, |rotated output - output| per
/// Laurent coefficient in the carrier's assembly arithmetic, over the scale
/// |M| form of the term floors, the spin legs' coefficient magnitudes
/// moving them), read in binary64 as the Python assembly reads the
/// executor's values (`float(x)`), then `reduction_policy::
/// fallback_estimate` of the outputs. `turned` is the point's guarded call
/// at the rotated point (its outputs, in the call's order).
#[allow(clippy::too_many_arguments)]
fn call_guard_estimate(
    induced: &Induced,
    average: i64,
    mode: V2Poles,
    roundoff: f64,
    context: &DecimalContext,
    forms: &[GuardForm],
    point: &GuardPoint<Decimal>,
    turned: &GuardPoint<Decimal>,
) -> RusticolResult<f64> {
    let missing = || invalid("a successful point without its collected vectors");
    let found = point.collected.as_ref().ok_or_else(missing)?;
    let plan_rows = induced.plan_rows();
    let mut orders: Vec<Orders> = induced.slots().to_vec();
    let direct = induced_point_of(induced, &plan_rows, &mut orders, found, Decimal::to_f64)?;
    let metric = induced.carrier(&Binary64, induced.metric())?;
    let gram: Vec<(f64, f64)> = metric.iter().map(|z| (z.re, z.im)).collect();
    let size = induced.size();
    let width = if mode.poles() { 3 } else { 1 };
    let modulus = |cell: &(f64, f64)| cell.0.hypot(cell.1);
    let decimal = |z: &C<Decimal>| z.re.to_f64().hypot(z.im.to_f64());
    let keyed = |cells: &Keyed<ErrorCells>| -> RowCells {
        cells
            .iter()
            .map(|((o, r), values)| ((orders[*o].clone(), plan_rows[*r].clone()), values.clone()))
            .collect()
    };
    let absolute = |cells: &Keyed<AmplitudeCells>| -> RowCells {
        cells
            .iter()
            .map(|((o, r), values)| {
                (
                    (orders[*o].clone(), plan_rows[*r].clone()),
                    values
                        .iter()
                        .map(|cell| cell.iter().map(modulus).collect())
                        .collect(),
                )
            })
            .collect()
    };
    // The spin legs' coefficient magnitudes (abs(complex(c))).
    let magnitudes: Option<BTreeMap<usize, Vec<(i32, f64)>>> =
        point.coefficients.as_ref().map(|coefficients| {
            coefficients
                .iter()
                .map(|(&leg, values)| (leg, values.iter().map(|(h, c)| (*h, decimal(c))).collect()))
                .collect()
        });
    // The rows the outputs read.
    let rows: Vec<induced::Row> = match &point.moved {
        None => plan_rows.clone(),
        Some(moved) => rows_of(moved),
    };
    // The term floors (`PointOutputs.scale`): per key of the term scale,
    // then of the kets, per t the term values plus the kets' magnitudes,
    // floored at the finite coefficient's.
    let terms = keyed(
        direct
            .terms
            .as_ref()
            .ok_or_else(|| invalid("a native loop-induced point has no term scale"))?,
    );
    let kets = absolute(&direct.kets);
    let mut floors: RowCells = Vec::new();
    for (key, _) in terms.iter().chain(kets.iter()) {
        if floors.iter().any(|(k, _)| k == key) {
            continue;
        }
        let sources: Vec<&ErrorCells> = [&terms, &kets]
            .into_iter()
            .filter_map(|part| part.iter().find(|(k, _)| k == key).map(|(_, v)| v))
            .collect();
        let cells = (0..size)
            .map(|t| {
                let mut values = vec![0.0_f64; width];
                for source in &sources {
                    for (k, value) in values.iter_mut().enumerate() {
                        *value += source[t][k];
                    }
                }
                let floor = values[0];
                values
                    .into_iter()
                    .map(|value| if floor > value { floor } else { value })
                    .collect()
            })
            .collect();
        floors.push((key.clone(), cells));
    }
    let floors = spin_moved(floors, magnitudes.as_ref());
    // Keys of the shape: (loop order slot, row of `rows`).
    let slots = induced.slots();
    let ids = |cells: &RowCells| -> Keyed<ErrorCells> {
        cells
            .iter()
            .filter_map(|((order, row), values)| {
                let a = slots.iter().position(|o| o == order)?;
                let b = rows.iter().position(|r| r == row)?;
                Some(((a, b), values.clone()))
            })
            .collect()
    };
    let floors = ids(&floors);
    let rotation = stability::rotation_name(0);
    if turned.outputs.len() != point.outputs.len() {
        return Err(invalid("the rotated call returned other outputs"));
    }
    let mut outputs = Vec::with_capacity(point.outputs.len());
    for ((output, form), other) in point.outputs.iter().zip(forms).zip(&turned.outputs) {
        let reference: Vec<Vec<f64>> = match &output.imaginary {
            None => output
                .real
                .iter()
                .map(|laurent| laurent.iter().map(Decimal::to_f64).collect())
                .collect(),
            // A directed history's complex output: its moduli.
            Some(imaginary) => output
                .real
                .iter()
                .zip(imaginary)
                .map(|(x, y)| {
                    x.iter()
                        .zip(y)
                        .map(|(x, y)| x.to_f64().hypot(y.to_f64()))
                        .collect()
                })
                .collect(),
        };
        let estimate = if form.inclusive && point.moved.is_none() {
            let shape = induced_shape(induced, gram.clone(), plan_rows.len(), mode, average);
            let other = turned.collected.as_ref().ok_or_else(missing)?;
            guard_rotation_estimate(
                &shape,
                context,
                roundoff,
                &reference,
                (&plan_rows, &mut orders),
                (&direct, found),
                other,
            )?
        } else {
            let shape = InducedShape {
                rows: output
                    .rows
                    .iter()
                    .filter_map(|row| rows.iter().position(|r| r == row))
                    .collect(),
                ..induced_shape(induced, gram.clone(), 0, mode, average)
            };
            // |y - x| per Laurent coefficient in the assembly arithmetic
            // (`float(abs(y - x))`; a complex output's modulus).
            let deviation = |x: &[Vec<Decimal>], y: &[Vec<Decimal>]| -> Vec<Vec<Decimal>> {
                x.iter()
                    .zip(y)
                    .map(|(x, y)| x.iter().zip(y).map(|(x, y)| context.sub(y, x)).collect())
                    .collect()
            };
            let real = deviation(&output.real, &other.real);
            let budget: Budget = match (&output.imaginary, &other.imaginary) {
                (Some(x), Some(y)) => {
                    let imaginary = deviation(x, y);
                    real.iter()
                        .zip(&imaginary)
                        .map(|(a, b)| {
                            a.iter()
                                .zip(b)
                                .map(|(a, b)| {
                                    let square =
                                        context.add(&context.mul(a, a), &context.mul(b, b));
                                    context.sqrt(&square).to_f64()
                                })
                                .collect()
                        })
                        .collect()
                }
                _ => real
                    .iter()
                    .map(|laurent| laurent.iter().map(|d| context.abs(d).to_f64()).collect())
                    .collect(),
            };
            let scale = shape.magnitude_form(&form.magnitude, &floors, &floors);
            Estimate::from_checks(
                &reference,
                vec![(rotation.clone(), budget)],
                scale,
                roundoff,
                Vec::new(),
            )?
        };
        outputs.push(guard_sectors(&reference, &estimate));
    }
    Ok(reduction_policy::fallback_estimate(&outputs, roundoff))
}

/// The loop-induced shape of a monitored batch: the colour metric `gram`
/// (binary64, as the Python assembly reads it), every plan row (`rows`),
/// the Laurent width of `mode`, the squared-order sectors and the average.
fn induced_shape(
    induced: &Induced,
    gram: Vec<(f64, f64)>,
    rows: usize,
    mode: V2Poles,
    average: i64,
) -> InducedShape {
    InducedShape {
        size: induced.size(),
        width: if mode.poles() { 3 } else { 1 },
        gram,
        rows: (0..rows).collect(),
        sectors: induced.pairs().to_vec(),
        average: average as f64,
    }
}

/// [`induced_point`] of a run collected in any carrier, its values read in
/// binary64 by `real` (an extended carrier's, as the Python API's
/// `induced.read_native` reads the executor's budgets: `float(x)`).
fn induced_point_of<N>(
    induced: &Induced,
    rows: &[induced::Row],
    orders: &mut Vec<Orders>,
    found: &InducedCollected<N>,
    real: impl Fn(&N) -> f64,
) -> RusticolResult<InducedPoint> {
    let mut key = |order: &Orders, row: &induced::Row| -> RusticolResult<(usize, usize)> {
        let o = match orders.iter().position(|o| o == order) {
            Some(o) => o,
            None => {
                orders.push(order.clone());
                orders.len() - 1
            }
        };
        let r = rows
            .iter()
            .position(|r| r == row)
            .ok_or_else(|| invalid("a vector of a row outside the plan's rows"))?;
        Ok((o, r))
    };
    let mut cells = |vectors: &induced::Vectors<N>| -> RusticolResult<_> {
        vectors
            .iter()
            .map(|((order, row), vector)| {
                Ok((
                    key(order, row)?,
                    vector
                        .iter()
                        .map(|cell| cell.iter().map(|z| (real(&z.re), real(&z.im))).collect())
                        .collect(),
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()
    };
    let amplitudes = cells(&found.parts.total)?;
    let bare = cells(&found.parts.bare)?;
    let kets = cells(&found.parts.kets)?;
    let size = induced.size();
    let columns = rows.len() * size;
    let mut slots = |values: &Option<Vec<Vec<N>>>| -> RusticolResult<Option<_>> {
        let Some(values) = values else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for (o, order) in induced.plan_orders().iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                let base = o * columns + r * size;
                let vector = (0..size)
                    .map(|t| {
                        values
                            .get(base + t)
                            .map(|slot| slot.iter().map(&real).collect::<Vec<f64>>())
                            .ok_or_else(|| invalid("the executor's slots differ from the plan's"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                out.push((key(order, row)?, vector));
            }
        }
        Ok(Some(out))
    };
    let budget = slots(&found.budget)?;
    let terms = slots(&found.terms)?;
    Ok(InducedPoint {
        amplitudes,
        bare,
        kets,
        budget,
        terms,
    })
}
