// SPDX-License-Identifier: 0BSD
//! The failure policy of an evaluation (m-lit ux-besteffort; USER 2026-11-11
//! items 8-9), the Python API's `on_failure=`.
//!
//! A point no configured alternative accepts (the call's backend with its
//! own acceptance checks, `"auto"`'s per-point fallback) is, by default
//! ([`OnFailure::BestEffort`]), evaluated again from its original exact
//! inputs WITHOUT the backends' own acceptance and refusal checks (the
//! unchecked reduction, `Runtime::set_reduction_unchecked`): where that gives
//! a value, the point returns it with status `BestEffort` (never a success),
//! its refusal and an uncertainty (the call's own binary64 stability estimate
//! when the call runs checks, else one default rotation re-test of that point
//! alone; none outside the binary64 monitoring, said so explicitly). The
//! call still returns OK: nothing is printed, logged or raised; the result's
//! statuses, `uncertainties` and its diagnostics (`"best_effort"`,
//! `"warnings"`) carry it. [`OnFailure::Return`] keeps such points failed
//! without computing a best-effort value; [`OnFailure::Raise`] returns an
//! evaluation error for a call with such a point (the opt-in strict mode).
//! The points every configured alternative accepts are evaluated exactly as
//! before, at the same cost.

use crate::{CorrelatedEvaluation, Evaluation, PointStatus, Runtime, Selection};
use rusticol_core::{RusticolError, RusticolResult};
use rusticol_loop_reduction::v2_stability::StabilityChecks;
use std::collections::BTreeMap;

/// The failure policies (`config.models.ON_FAILURE`), the default first.
pub const ON_FAILURE: [&str; 3] = ["best-effort", "return", "raise"];

/// The failure policy of later evaluations (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnFailure {
    #[default]
    BestEffort,
    Return,
    Raise,
}

impl OnFailure {
    /// One of [`ON_FAILURE`]; another text is an invalid argument.
    pub fn parse(text: &str) -> RusticolResult<Self> {
        match text {
            "best-effort" => Ok(Self::BestEffort),
            "return" => Ok(Self::Return),
            "raise" => Ok(Self::Raise),
            other => Err(RusticolError::invalid_argument(format!(
                "on_failure must be 'best-effort', 'return' or 'raise', not {other:?}"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::BestEffort => "best-effort",
            Self::Return => "return",
            Self::Raise => "raise",
        }
    }
}

/// `reasons` (one failed point's, one message per failed lane or check)
/// with one message per distinct reason CLASS (the message with its numbers
/// masked): the first message of each class in first-seen order, with the
/// count of the others appended (`" (+N more like this)"`); the Python
/// API's `plan_export.bounded_reasons` (m-lit ux-besteffort F6: a refused
/// g g > z g g point had ~11700 lane messages, ~8.8 MB).
pub fn bounded_reasons(reasons: &[String]) -> Vec<String> {
    let mut order: Vec<String> = Vec::new();
    let mut first: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for reason in reasons {
        let key = mask_numbers(reason);
        match first.get_mut(&key) {
            Some((_, more)) => *more += 1,
            None => {
                order.push(key.clone());
                first.insert(key, (reason.clone(), 0));
            }
        }
    }
    order
        .iter()
        .map(|key| {
            let (message, more) = &first[key];
            if *more == 0 {
                message.clone()
            } else {
                format!("{message} (+{more} more like this)")
            }
        })
        .collect()
}

/// `text` with every decimal number (sign, digits, fraction, exponent; the
/// Python `_NUMBER` pattern) replaced by `#`.
fn mask_numbers(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let mut j = i;
        if matches!(bytes[j], b'+' | b'-') {
            j += 1;
        }
        let digits = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        let mut seen = j > digits;
        if j < bytes.len() && bytes[j] == b'.' {
            let dot = j + 1;
            let mut k = dot;
            while k < bytes.len() && bytes[k].is_ascii_digit() {
                k += 1;
            }
            if seen || k > dot {
                seen = true;
                j = k;
            }
        }
        if seen {
            if j < bytes.len() && matches!(bytes[j], b'e' | b'E') {
                let mut k = j + 1;
                if k < bytes.len() && matches!(bytes[k], b'+' | b'-') {
                    k += 1;
                }
                let exponent = k;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                if k > exponent {
                    j = k;
                }
            }
            out.push('#');
            i = j;
        } else {
            let c = text[start..].chars().next().expect("a character");
            out.push(c);
            i = start + c.len_utf8();
        }
    }
    out
}

/// The parts of a result the failure policy reads and merges (the ordinary,
/// resolved and selected [`Evaluation`] and the [`CorrelatedEvaluation`]).
pub(crate) trait PolicyResult: Sized {
    fn requested(&self) -> &[u64];
    fn successful(&self) -> &[u64];
    fn statuses_mut(&mut self) -> &mut Vec<PointStatus>;
    fn uncertainties_mut(&mut self) -> &mut Vec<f64>;
    fn diagnostics_mut(&mut self) -> &mut String;
    /// Insert the rows of `extra`'s points `served` (ids) at their
    /// positions: the successful ids become `order`.
    fn merge_rows(&mut self, extra: &Self, served: &[u64], order: &[u64]) -> RusticolResult<()>;
}

/// `rows` of the points `ids` (each `stride` values) merged with the rows of
/// the points `served` of `extra` (ids `extra_ids`), in the order `order`.
fn interleave<T: Clone>(
    rows: &[T],
    ids: &[u64],
    extra: &[T],
    extra_ids: &[u64],
    served: &[u64],
    order: &[u64],
) -> RusticolResult<Vec<T>> {
    let stride = if !ids.is_empty() {
        rows.len() / ids.len()
    } else if !extra_ids.is_empty() {
        extra.len() / extra_ids.len()
    } else {
        0
    };
    if stride * ids.len() != rows.len() || stride * extra_ids.len() != extra.len() {
        return Err(RusticolError::internal(
            "best-effort rows: a result's rows are not one stride per point",
        ));
    }
    let position = |list: &[u64], id: u64| list.iter().position(|&x| x == id);
    let mut out = Vec::with_capacity(stride * order.len());
    for &id in order {
        let row = if let Some(p) = position(ids, id) {
            &rows[p * stride..(p + 1) * stride]
        } else if served.contains(&id)
            && let Some(p) = position(extra_ids, id)
        {
            &extra[p * stride..(p + 1) * stride]
        } else {
            return Err(RusticolError::internal(format!(
                "best-effort rows: no row for point {id}"
            )));
        };
        out.extend_from_slice(row);
    }
    Ok(out)
}

impl PolicyResult for Evaluation {
    fn requested(&self) -> &[u64] {
        &self.requested_point_ids
    }
    fn successful(&self) -> &[u64] {
        &self.successful_point_ids
    }
    fn statuses_mut(&mut self) -> &mut Vec<PointStatus> {
        &mut self.statuses
    }
    fn uncertainties_mut(&mut self) -> &mut Vec<f64> {
        &mut self.uncertainties
    }
    fn diagnostics_mut(&mut self) -> &mut String {
        &mut self.diagnostics_json
    }
    fn merge_rows(&mut self, extra: &Self, served: &[u64], order: &[u64]) -> RusticolResult<()> {
        let (ids, extra_ids) = (&self.successful_point_ids, &extra.successful_point_ids);
        self.born = interleave(&self.born, ids, &extra.born, extra_ids, served, order)?;
        self.virtual_values = interleave(
            &self.virtual_values,
            ids,
            &extra.virtual_values,
            extra_ids,
            served,
            order,
        )?;
        if let (Some(mine), Some(theirs)) = (&mut self.decimal, &extra.decimal) {
            mine.born = interleave(&mine.born, ids, &theirs.born, extra_ids, served, order)?;
            mine.virtual_values = interleave(
                &mine.virtual_values,
                ids,
                &theirs.virtual_values,
                extra_ids,
                served,
                order,
            )?;
        }
        self.successful_point_ids = order.to_vec();
        Ok(())
    }
}

impl PolicyResult for CorrelatedEvaluation {
    fn requested(&self) -> &[u64] {
        &self.requested_point_ids
    }
    fn successful(&self) -> &[u64] {
        &self.successful_point_ids
    }
    fn statuses_mut(&mut self) -> &mut Vec<PointStatus> {
        &mut self.statuses
    }
    fn uncertainties_mut(&mut self) -> &mut Vec<f64> {
        &mut self.uncertainties
    }
    fn diagnostics_mut(&mut self) -> &mut String {
        &mut self.diagnostics_json
    }
    fn merge_rows(&mut self, extra: &Self, served: &[u64], order: &[u64]) -> RusticolResult<()> {
        let (ids, extra_ids) = (&self.successful_point_ids, &extra.successful_point_ids);
        self.born = interleave(&self.born, ids, &extra.born, extra_ids, served, order)?;
        self.virtual_values = interleave(
            &self.virtual_values,
            ids,
            &extra.virtual_values,
            extra_ids,
            served,
            order,
        )?;
        if let (Some(mine), Some(theirs)) = (&mut self.decimal, &extra.decimal) {
            mine.born = interleave(&mine.born, ids, &theirs.born, extra_ids, served, order)?;
            mine.virtual_values = interleave(
                &mine.virtual_values,
                ids,
                &theirs.virtual_values,
                extra_ids,
                served,
                order,
            )?;
        }
        self.successful_point_ids = order.to_vec();
        Ok(())
    }
}

/// The raw inputs of a call, subset to some of its points for the
/// best-effort re-evaluation.
pub(crate) struct CallInputs<'a> {
    pub momenta: &'a [String],
    pub point_ids: &'a [u64],
    pub scales: &'a [String],
    pub selection: Option<&'a Selection>,
    pub sources: Option<&'a BTreeMap<usize, Vec<[String; 2]>>>,
}

/// The subset of a call's inputs at `indices` (point positions).
pub(crate) struct Subset {
    pub momenta: Vec<String>,
    pub point_ids: Vec<u64>,
    pub scales: Vec<String>,
    pub selection: Option<Selection>,
    pub sources: Option<BTreeMap<usize, Vec<[String; 2]>>>,
}

impl CallInputs<'_> {
    fn subset(&self, indices: &[usize]) -> Subset {
        let points = self.point_ids.len().max(1);
        let width = self.momenta.len() / points;
        let pick = |values: &[u32]| indices.iter().map(|&i| values[i]).collect::<Vec<_>>();
        Subset {
            momenta: indices
                .iter()
                .flat_map(|&i| self.momenta[i * width..(i + 1) * width].iter().cloned())
                .collect(),
            point_ids: indices.iter().map(|&i| self.point_ids[i]).collect(),
            scales: if self.scales.len() == 1 {
                self.scales.to_vec()
            } else {
                indices.iter().map(|&i| self.scales[i].clone()).collect()
            },
            selection: self.selection.map(|selection| Selection {
                helicity_by_point: selection.helicity_by_point.as_deref().map(pick),
                color_flow_by_point: selection.color_flow_by_point.as_deref().map(pick),
                ..selection.clone()
            }),
            sources: self.sources.map(|sources| {
                sources
                    .iter()
                    .map(|(&leg, values)| {
                        (
                            leg,
                            indices
                                .iter()
                                .flat_map(|&i| values[4 * i..4 * (i + 1)].iter().cloned())
                                .collect(),
                        )
                    })
                    .collect()
            }),
        }
    }
}

/// The bounded reasons of point `id`, joined.
fn joined(reasons: &BTreeMap<u64, Vec<String>>, id: u64) -> String {
    reasons
        .get(&id)
        .map(|list| list.join("; "))
        .unwrap_or_else(|| "no value".to_owned())
}

/// The relative estimate of the `position`-th successful point of a
/// result's diagnostics (`"stability"."estimates"[position]."relative"`),
/// when the binary64 monitoring formed one and it is a finite number.
fn estimate_of(diagnostics: &serde_json::Value, position: usize) -> Option<f64> {
    diagnostics
        .get("stability")?
        .get("estimates")?
        .get(position)?
        .get("relative")?
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
}

impl Runtime {
    /// The failure policy of later evaluations (module docs; default
    /// [`OnFailure::BestEffort`]).
    pub fn set_on_failure(&mut self, on_failure: OnFailure) {
        self.on_failure = on_failure;
    }
    pub fn on_failure(&self) -> OnFailure {
        self.on_failure
    }

    /// `base` (a call's result on `inputs`) under the failure policy:
    /// `rerun` evaluates a subset of the inputs as the call did.
    pub(crate) fn with_policy<R: PolicyResult>(
        &mut self,
        mut base: R,
        inputs: CallInputs<'_>,
        rerun: impl FnOnce(&mut Self, Subset) -> RusticolResult<R>,
    ) -> RusticolResult<R> {
        let requested = base.requested().to_vec();
        let failed: Vec<usize> = (0..requested.len())
            .filter(|&i| base.statuses_mut()[i] == PointStatus::Failed)
            .collect();
        *base.uncertainties_mut() = vec![f64::NAN; requested.len()];
        let mut diagnostics: serde_json::Value =
            serde_json::from_str(base.diagnostics_mut()).map_err(crate::serialization)?;
        // Bounded failure reasons (F6), in every policy.
        let mut reasons: BTreeMap<u64, Vec<String>> = BTreeMap::new();
        if let Some(failures) = diagnostics.get("failures").and_then(|f| f.as_object()) {
            for (id, list) in failures {
                let list: Vec<String> = list
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .map(|item| {
                                item.as_str()
                                    .map_or_else(|| item.to_string(), str::to_owned)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if let Ok(id) = id.parse::<u64>() {
                    reasons.insert(id, bounded_reasons(&list));
                }
            }
        }
        if self.on_failure == OnFailure::Raise
            && let Some(&first) = failed.first()
        {
            let id = requested[first];
            return Err(RusticolError::evaluation(format!(
                "v2 virtual evaluation failed at point {id}: {} (on_failure \"raise\")",
                joined(&reasons, id)
            )));
        }
        let mut best: BTreeMap<u64, serde_json::Value> = BTreeMap::new();
        if self.on_failure == OnFailure::BestEffort && !failed.is_empty() && !self.unchecked {
            let rotation = !self.checks.enabled();
            let saved = (self.unchecked, self.checks.clone());
            self.unchecked = true;
            if rotation {
                self.checks = StabilityChecks::with_default_rotations(1, false, false, false)?;
            }
            let extra = rerun(self, inputs.subset(&failed));
            (self.unchecked, self.checks) = saved;
            let mut extra = extra?;
            let extra_diagnostics: serde_json::Value =
                serde_json::from_str(extra.diagnostics_mut()).map_err(crate::serialization)?;
            let extra_ids = extra.successful().to_vec();
            let mut served = Vec::new();
            for (position, &id) in extra_ids.iter().enumerate() {
                let uncertainty = estimate_of(&extra_diagnostics, position);
                let source = match (uncertainty, rotation) {
                    (None, _) => "none",
                    (Some(_), true) => "rotation",
                    (Some(_), false) => "stability-checks",
                };
                let index = requested
                    .iter()
                    .position(|&x| x == id)
                    .expect("a requested id");
                base.statuses_mut()[index] = PointStatus::BestEffort;
                base.uncertainties_mut()[index] = uncertainty.unwrap_or(f64::NAN);
                best.insert(
                    id,
                    serde_json::json!({
                        "refusal": joined(&reasons, id),
                        "uncertainty": uncertainty,
                        "source": source,
                        "detail": match uncertainty {
                            None => "no uncertainty could be estimated (the unchecked value \
                                     has no binary64 stability estimate)",
                            Some(_) => "the estimated relative uncertainty of the unchecked \
                                        reduction's value",
                        },
                        "reduction": extra_diagnostics.get("reduction"),
                    }),
                );
                served.push(id);
            }
            if !served.is_empty() {
                let order: Vec<u64> = requested
                    .iter()
                    .zip(base.statuses_mut().iter())
                    .filter(|(_, status)| **status != PointStatus::Failed)
                    .map(|(&id, _)| id)
                    .collect();
                base.merge_rows(&extra, &served, &order)?;
                for id in &served {
                    reasons.remove(id);
                }
            }
        }
        let statuses = base.statuses_mut().clone();
        let mut warnings = Vec::new();
        for (kind, status) in [
            ("best-effort", PointStatus::BestEffort),
            ("failed", PointStatus::Failed),
        ] {
            let points: Vec<u64> = requested
                .iter()
                .zip(&statuses)
                .filter(|(_, s)| **s == status)
                .map(|(&id, _)| id)
                .collect();
            if let Some(&first) = points.first() {
                let message = match status {
                    PointStatus::BestEffort => best[&first]["refusal"].clone(),
                    _ => serde_json::Value::String(joined(&reasons, first)),
                };
                warnings.push(serde_json::json!({
                    "source": "numerical",
                    "kind": kind,
                    "message": message,
                    "count": points.len(),
                    "where": "evaluate",
                    "points": points,
                }));
            }
        }
        let object = diagnostics
            .as_object_mut()
            .ok_or_else(|| RusticolError::internal("diagnostics are a JSON object"))?;
        object.insert("failures".to_owned(), serde_json::json!(reasons));
        object.insert(
            "successful_point_ids".to_owned(),
            serde_json::json!(base.successful()),
        );
        object.insert("best_effort".to_owned(), serde_json::json!(best));
        object.insert("warnings".to_owned(), serde_json::Value::Array(warnings));
        object.insert(
            "on_failure".to_owned(),
            serde_json::Value::String(self.on_failure.name().to_owned()),
        );
        *base.diagnostics_mut() =
            serde_json::to_string(&diagnostics).map_err(crate::serialization)?;
        Ok(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_are_bounded_per_class() {
        let reasons: Vec<String> = (0..1000)
            .map(|i| format!("lane {i}: TIRust Unstable: deviation {}e-3", i + 1))
            .chain(["NinjaRS refused".to_owned()])
            .collect();
        assert_eq!(
            bounded_reasons(&reasons),
            vec![
                "lane 0: TIRust Unstable: deviation 1e-3 (+999 more like this)".to_owned(),
                "NinjaRS refused".to_owned(),
            ]
        );
        assert_eq!(mask_numbers("a -1.5e-3 b .5 c 7."), "a # b # c #");
    }

    #[test]
    fn the_policy_parses_its_names() {
        assert_eq!(OnFailure::default(), OnFailure::BestEffort);
        for name in ON_FAILURE {
            assert_eq!(OnFailure::parse(name).unwrap().name(), name);
        }
        assert!(OnFailure::parse("warn").is_err());
    }

    #[test]
    fn rows_interleave_in_the_requested_order() {
        let rows = interleave(
            &[1, 1, 3, 3],
            &[1, 3],
            &[2, 2, 9, 9],
            &[2, 9],
            &[2],
            &[1, 2, 3],
        )
        .unwrap();
        assert_eq!(rows, vec![1, 1, 2, 2, 3, 3]);
    }
}
