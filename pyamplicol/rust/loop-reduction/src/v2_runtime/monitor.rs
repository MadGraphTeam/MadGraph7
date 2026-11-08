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

use super::*;
use crate::v2_stability::{
    self as stability, CrossRun, Estimate, Frame, LaneShape, PointLanes, StabilityChecks,
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

/// One monitored point: its values (or failures) and, for a successful
/// point, its stability estimate.
pub type Monitored = (V2Point, Option<Estimate>);

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
                .map(|point| (point, None))
                .collect());
        }
        if self.loop_induced() {
            return Err(RusticolError::unsupported_runtime_capability(
                "stability checks",
                "the native stability checks of a loop-induced process are not built yet; \
                 the Python API estimates them",
            ));
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
        let mut failures: Vec<Vec<String>> = points
            .iter()
            .map(|point| point.as_ref().err().cloned().unwrap_or_default())
            .collect();
        let mut rotated = Vec::with_capacity(checks.rotations().len());
        for (k, rotation) in checks.rotations().iter().enumerate() {
            let moved = momenta
                .iter()
                .map(|point| {
                    let legs: Vec<Vec<Rational>> =
                        point.chunks_exact(4).map(<[Rational]>::to_vec).collect();
                    Ok(rotation.apply(&legs)?.concat())
                })
                .collect::<RusticolResult<Vec<_>>>()?;
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
                }
            }
            rotated.push(runs);
        }
        let cross = if checks.cross_backend() {
            Some(self.cross_runs(momenta, scales, mode)?)
        } else {
            None
        };
        let shape = LaneShape {
            sectors: self.record.sectors.len(),
            width: if mode.poles() { 3 } else { 1 },
            factor: stability::lane_factor(self.record.average as f64),
        };
        let roundoff = 2.0_f64.powi(-53);
        let mut out = Vec::with_capacity(points.len());
        for (index, point) in points.into_iter().enumerate() {
            if !failures[index].is_empty() {
                out.push((Err(std::mem::take(&mut failures[index])), None));
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
            out.push((point, Some(estimate)));
        }
        Ok(out)
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
    ) -> RusticolResult<Vec<CrossRun<LaneKey>>> {
        let other = cross_reduction(self.reduction);
        let name = backend_name(other);
        self.swap_cross(other);
        let result: RusticolResult<Vec<CrossRun<LaneKey>>> = (|| {
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
