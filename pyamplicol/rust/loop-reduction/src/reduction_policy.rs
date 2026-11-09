// SPDX-License-Identifier: 0BSD

//! The reduction-backend policy of a v2 one-loop evaluation (m-lit PLAN
//! §2.b.4, RULED Q17 / Q18): the ONE definition of `"auto"` (DECISIONS
//! 2026-11-07 21:50 R7). The Python API (`loops/v2/reduction.py`, through
//! the PyO3 functions `reduction_policy` and `reduction_resolve`) and the
//! native SDKs (`rusticol_virtual_set_reduction`) read it; nothing else
//! holds a routing table.
//!
//! A call requests `"auto"`, `"ninjars"` or `"tirust"`. [`resolve`] chooses
//! the effective backend once per call and carrier, never per point:
//!
//! * `"auto"` takes the per-type default ([`auto_default`]: TIRust for a
//!   loop-induced process, NinjaRS for a Born-interfered one; TIRust for both
//!   on a TIR-only artifact, `admitted` = TIRust, m-lit lit-eft) when it
//!   serves the carrier on the call's route, else NinjaRS, with the reason
//!   recorded. Where `"auto"` takes TIRust and NinjaRS serves the call too,
//!   NinjaRS is the call's PER-POINT FALLBACK ([`Choice::fallback`], m-lit
//!   F3 (c), USER 2026-11-08 ~15:40, DECISIONS 15:19 / 15:32): a point TIRust
//!   refuses or fails (also in the enabled binary64 stability monitoring's
//!   own evaluations) is evaluated again on NinjaRS from its original exact
//!   inputs, at the call's own carrier, after the TIRust call; the backend
//!   that served each point and the fallback's reason are recorded per point
//!   ([`fallback_served`]; a point NinjaRS fails too fails with both
//!   reasons, [`fallback_failed`]). In the extended carriers every point
//!   the fallback serves is GUARDED (m-lit lit-fallback-guard, DECISIONS
//!   2026-11-09 05:56 (1) and 11:53, [`fallback_guarded`]): NinjaRS can
//!   return finite, badly wrong values as success at near-planar 2 -> 3
//!   points of rank-n pentagon families, so its value runs the
//!   reduction-direction re-reduction ([`FALLBACK_CHECK`], cached masters)
//!   at the call's carrier, its relative estimate is recorded per point,
//!   and a point whose estimate exceeds the guard's limit (the call's
//!   target, never below [`FALLBACK_FLOOR`]: [`fallback_limit`],
//!   [`fallback_accepted`]) fails with the one reason
//!   [`fallback_rejected`]; a binary64 fallback serves unguarded. A
//!   TIR-only artifact has no fallback;
//! * an explicit backend that cannot serve the call is refused typed
//!   ([`Refusal::Unavailable`], the Python API's `ReductionUnavailable`):
//!   never a silent switch, and an explicit backend has no fallback (its
//!   refusals and failures stay failed points).
//!
//! TIRust serves at a TARGET ACCURACY ([`DEFAULT_TARGET`], set per call or
//! runtime, validated by [`target`]): the relative accuracy of each point's
//! evaluated result (every squared-order sector against its own size, the
//! rescue's `"relative"` metric), from which TIRust's method-choice
//! acceptance is calibrated (`v2_tir::acceptance_of`, feature `tir`).
//!
//! The UNCHECKED setting ([`DEFAULT_UNCHECKED`] off, validated by
//! [`unchecked`], m-lit lit-unchecked) drops the backends' own acceptance
//! and refusal checks for a call: each backend in its unchecked form, the
//! same choice of backend. Under it `"auto"` keeps its rule; its per-point
//! NinjaRS fallback (NinjaRS unchecked, at the call's carrier) acts only on
//! what the unchecked TIRust still fails (the rows without a finite value
//! in any scheme), so where nothing fails `"auto"` is a pure TIRust call.
//!
//! What a backend serves is the native capability
//! (`runtime::backend_carriers`), passed in by the caller; an artifact may
//! admit fewer backends (`admitted`). A route ([`Route`]) is what a call
//! evaluates and the backends its executor runs, and `auto` there takes
//! only the backends of its `auto` set: both backends on the artifact
//! runtime's routes ([`SUMMED`], [`RESOLVED`]), so `"auto"` takes TIRust for
//! loop-induced processes on both (m-lit F3, USER 2026-11-08 ~14:40 / ~15:40,
//! superseding DECISIONS 2026-11-08 11:17: the per-point NinjaRS fallback
//! serves the points TIRust fails closed on).

use std::collections::{BTreeMap, BTreeSet};

/// The backends, by name (`config.ReductionBackend` less `"auto"`).
pub const BACKENDS: [&str; 2] = ["ninjars", "tirust"];
/// The carriers, by name.
pub const CARRIERS: [&str; 3] = ["double", "double-double", "arbitrary"];
/// The contribution kinds, by name.
pub const CONTRIBUTIONS: [&str; 2] = ["virtual", "loop-induced"];

/// The backend `"auto"` falls back to: for the whole call when the
/// per-type default cannot serve it, per point when it serves the call
/// ([`Choice::fallback`]).
const FALLBACK: &str = "ninjars";

/// The default target accuracy of a call (m-lit F3 (a), USER 2026-11-08
/// ~14:40: '1e-5 is fine for many applications'): the relative accuracy of
/// each point's evaluated result, every squared-order sector against its
/// own size (the rescue's `"relative"` metric). TIRust's method-choice
/// acceptance follows it through its calibration (`v2_tir::acceptance_of`);
/// NinjaRS has no setting it drives. The one definition (Python reads it
/// from [`policy_json`], the native SDKs from here).
pub const DEFAULT_TARGET: f64 = 1e-5;

/// The refusal of a target accuracy that is not a finite positive number:
/// one text for every surface (the Python API's `config.models.
/// reduction_target` reads it from [`policy_json`]).
pub const TARGET_REFUSAL: &str =
    "the reduction target is a relative accuracy: a finite positive number";

/// A target accuracy (the one validator of the native SDKs; the Python
/// API's states the same rule with [`TARGET_REFUSAL`]): a finite positive
/// number.
pub fn target(value: f64) -> Result<f64, Refusal> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(Refusal::Invalid(TARGET_REFUSAL.to_owned()))
    }
}

/// The default of the UNCHECKED reduction setting (m-lit lit-unchecked, USER
/// 2026-11-08 ~21:14 / ~22:34, DECISIONS 22:34 (1)): off. A call (or
/// runtime) that sets it on reduces without the reduction backends' own
/// acceptance and refusal checks ([`UNCHECKED_RECORD`]): TIRust keeps its a
/// priori conditioning gates and method choice as METHOD SELECTION, never
/// computes pyAmpliCol's a posteriori acceptance pass of the method-choice
/// lanes (so the target accuracy drives nothing on TIRust), and serves the
/// rows the method choice (or, where it is off, the gates) refuses with
/// plain PV at gate 0, its last resort (`v2_tir::TirOptions::unchecked`);
/// NinjaRS runs with zero stability thresholds (`runtime::V2Reducer::
/// set_unchecked`). Only rows without a finite value in any scheme still
/// fail (an exactly singular pivot, a non-finite output, TIRust's
/// `InvalidInput` / `MasterFailure`, NinjaRS's `NonFinite` / `MasterFailed`
/// and a NaN lane). pyAmpliCol's own stability checks, monitoring and
/// rescue stay their own settings. The one definition: Python reads it from
/// [`policy_json`], the native SDKs from here.
pub const DEFAULT_UNCHECKED: bool = false;

/// What the unchecked setting does, as the policy record states it (one
/// text for every surface: the Python API's and the SDKs' inspect records).
pub const UNCHECKED_RECORD: &str = "reduction_unchecked: TIRust keeps its conditioning gates and \
     method choice as method selection, without pyAmpliCol's acceptance check, and serves what \
     they refuse by plain PV at gate 0 (its last resort); NinjaRS runs with zero stability \
     thresholds; only exactly singular or non-finite rows fail";

/// The refusal of an unchecked setting that is not a boolean: one text for
/// every surface (the Python API's `config.models.reduction_unchecked` reads
/// it from [`policy_json`]).
pub const UNCHECKED_REFUSAL: &str = "the unchecked reduction setting (reduction_unchecked) is a boolean: true or false \
     (1 or 0 in the C ABI)";

/// An unchecked setting given as an integer (the one validator of the
/// native SDKs' C ABI, `rusticol_virtual_set_reduction_unchecked`; the
/// Python API's is the language's boolean with [`UNCHECKED_REFUSAL`]): 1
/// (on) or 0 (off).
pub fn unchecked(value: i64) -> Result<bool, Refusal> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Refusal::Invalid(UNCHECKED_REFUSAL.to_owned())),
    }
}

/// The default of `"auto"` for a `contribution` on an artifact admitting
/// `admitted` (`None`: every backend): TIRust on a TIR-only artifact (its
/// only backend, m-lit lit-eft), else the per-type default ([`default_of`]).
/// What [`resolve`], `V2Runtime::open`, the SDK capabilities and the Python
/// API's inspect record read.
pub fn auto_default(
    contribution: &str,
    admitted: Option<&BTreeSet<String>>,
) -> Option<&'static str> {
    if tir_only(admitted) {
        Some("tirust")
    } else {
        default_of(contribution)
    }
}

/// Whether an artifact admitting `admitted` is TIR-only.
fn tir_only(admitted: Option<&BTreeSet<String>>) -> bool {
    admitted.is_some_and(|set| set.len() == 1 && set.contains("tirust"))
}

/// The reason of a point the per-point fallback served: `primary` (the
/// call's effective backend) refused or failed it (`failure`, its reasons),
/// and `fallback` evaluated it again.
pub fn fallback_served(primary: &str, fallback: &str, failure: &str) -> String {
    format!(
        "auto: {}, the per-point fallback: {} refused or failed the point ({failure})",
        name_of(fallback),
        name_of(primary)
    )
}

/// A reason of a point the per-point fallback failed too (after the
/// primary backend's own reasons).
pub fn fallback_failed(fallback: &str, reason: &str) -> String {
    format!("{} fallback: {reason}", name_of(fallback))
}

/// The check the per-point fallback's GUARD runs on every point the
/// fallback served in a guarded carrier ([`fallback_guarded`]; m-lit
/// lit-fallback-guard, DECISIONS 2026-11-09 05:56 (1) and 11:53): the
/// reduction-direction re-reduction (every loop numerator reduced a second
/// time in the reversed frame, its coefficients and the cached master
/// integrals reused; the Python API's `VirtualStabilityChecks(
/// reduction_direction=True)` without `rescaled`, the SDKs' re-test frame
/// `reversed`), at the call's own carrier (never escalated, USER 2026-11-08
/// ~20:55 'Same precision as call'). Its estimate, in the `"relative"`
/// metric (each squared-order sector against its largest Laurent
/// coefficient, the metric of [`DEFAULT_TARGET`]), is recorded per point;
/// a point whose estimate exceeds the guard's limit ([`fallback_limit`]) or
/// is not a number fails typed ([`fallback_rejected`]), a point within it
/// keeps the fallback's value bitwise.
pub const FALLBACK_CHECK: &str = "reduction-direction";

/// The guard's floor: its limit is never stricter than this relative
/// accuracy, whatever the call's target (USER 2026-11-09 ~11:53, 'it's fine
/// to relax the target precision to 1e-5 or so'; DECISIONS 11:53: the limit
/// is max(call target, 1e-5)).
pub const FALLBACK_FLOOR: f64 = 1e-5;

/// Whether the per-point fallback's guard runs in the carrier `carrier`
/// ([`CARRIERS`]): in the extended carriers only, double-double and
/// arbitrary precision (USER 2026-11-09 ~11:53, 'Double-double+ only';
/// DECISIONS 11:53). In binary64 the protection is NinjaRS's own
/// conditioning bound in its fast engine (m-lit lit-ninja-planar): a
/// binary64 fallback point is served as the fallback returns it, with no
/// estimate recorded.
pub fn fallback_guarded(carrier: &str) -> bool {
    carrier != "double"
}

/// The guard's limit at the call's target `target`: the larger of the
/// target and [`FALLBACK_FLOOR`].
pub fn fallback_limit(target: f64) -> f64 {
    target.max(FALLBACK_FLOOR)
}

/// Whether the per-point fallback's guard accepts a served point whose
/// check estimate is `estimate` at the call's target `target`: the
/// estimate is a number at most the guard's limit ([`fallback_limit`]; a
/// NaN estimate is rejected).
pub fn fallback_accepted(estimate: f64, target: f64) -> bool {
    estimate <= fallback_limit(target)
}

/// The ONE reason of a point the per-point fallback served but its guard
/// rejected ([`FALLBACK_CHECK`]): `primary` (the call's effective backend)
/// refused or failed it (`failure`, its reasons), `fallback` evaluated it
/// again at the call's carrier `carrier`, and the fallback's value failed
/// its reduction-direction check with the relative estimate `estimate`
/// above the guard's limit at the call's target `target`
/// ([`fallback_limit`]). Python (`loops.v2.reduction.text`) and the native
/// SDKs read it from here.
pub fn fallback_rejected(
    primary: &str,
    fallback: &str,
    failure: &str,
    carrier: &str,
    estimate: f64,
    target: f64,
) -> String {
    let limit = format!(
        "{:e} = max(target {target:e}, {FALLBACK_FLOOR:e})",
        fallback_limit(target)
    );
    let verdict = if estimate.is_nan() {
        format!("estimate NaN, not within {limit}")
    } else {
        format!("estimate {estimate:.1e} > {limit}")
    };
    format!(
        "auto: {} refused or failed the point ({failure}); the per-point {} fallback's value \
         failed its {FALLBACK_CHECK} check: {verdict} ({carrier})",
        name_of(primary),
        name_of(fallback)
    )
}

/// The refusal of a call's cross-backend check whose other backend cannot
/// serve the call (`message`, its refusal).
pub fn cross_refusal(message: &str) -> String {
    format!(
        "the cross-backend check (stability_checks cross_backend) compares with the \
         other reduction backend: {message}"
    )
}

/// The per-type default of `"auto"` (RULED Q17 / Q40): NinjaRS for
/// Born-interfered processes, TIRust for loop-induced ones.
pub fn default_of(contribution: &str) -> Option<&'static str> {
    match contribution {
        "virtual" => Some("ninjars"),
        "loop-induced" => Some("tirust"),
        _ => None,
    }
}

fn name_of(backend: &str) -> &str {
    match backend {
        "ninjars" => "NinjaRS",
        "tirust" => "TIRust",
        other => other,
    }
}

fn kind_of(contribution: &str) -> &str {
    match contribution {
        "virtual" => "Born-interfered",
        other => other,
    }
}

/// A route of the artifact runtime: what it evaluates (for messages), the
/// backends its executor runs and those `"auto"` may take there (`auto`;
/// `None`: every backend it runs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub serves: String,
    pub backends: BTreeSet<String>,
    pub auto: Option<BTreeSet<String>>,
}

impl Route {
    /// Whether `"auto"` may take `backend` on this route.
    pub fn automatic(&self, backend: &str) -> bool {
        self.auto
            .as_ref()
            .unwrap_or(&self.backends)
            .contains(backend)
    }
}

/// A route's static definition: (key, what it serves, backends its executor
/// runs, the backends `"auto"` takes there).
type RouteDefinition = (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);

/// The summed virtual through the native executor.
pub const SUMMED: &str = "summed";
/// The colour-, spin- and helicity-resolved outputs and selections.
pub const RESOLVED: &str = "resolved";

/// The artifact runtime's routes. Both executors run both backends (m-lit
/// lit-tir), and `"auto"` takes either on both (m-lit F3: TIRust for
/// loop-induced processes, with the per-point NinjaRS fallback).
const ROUTES: [RouteDefinition; 2] = [
    (
        SUMMED,
        "the summed virtual",
        &["ninjars", "tirust"],
        &["ninjars", "tirust"],
    ),
    (
        RESOLVED,
        "the colour-, spin- and helicity-resolved outputs and selections",
        &["ninjars", "tirust"],
        &["ninjars", "tirust"],
    ),
];

/// The artifact runtime's route `key` ([`SUMMED`], [`RESOLVED`]).
pub fn route(key: &str) -> Option<Route> {
    ROUTES
        .iter()
        .find(|(name, ..)| *name == key)
        .map(|(_, serves, backends, auto)| Route {
            serves: (*serves).to_owned(),
            backends: backends.iter().map(|b| (*b).to_owned()).collect(),
            auto: Some(auto.iter().map(|b| (*b).to_owned()).collect()),
        })
}

/// The policy as JSON (what the Python API reads, `reduction.DEFAULTS`, its
/// routes, the default target and the unchecked setting): `{"backends",
/// "carriers", "defaults": {kind: backend}, "fallback", "target":
/// {"default", "refusal"}, "unchecked": {"default", "refusal", "record"},
/// "routes": {key: {"serves", "backends", "auto"}}}`.
pub fn policy_json() -> serde_json::Value {
    serde_json::json!({
        "backends": BACKENDS,
        "carriers": CARRIERS,
        "fallback": FALLBACK,
        "target": {"default": DEFAULT_TARGET, "refusal": TARGET_REFUSAL},
        "unchecked": {
            "default": DEFAULT_UNCHECKED,
            "refusal": UNCHECKED_REFUSAL,
            "record": UNCHECKED_RECORD,
        },
        "defaults": CONTRIBUTIONS
            .iter()
            .map(|kind| ((*kind).to_owned(), serde_json::json!(default_of(kind))))
            .collect::<serde_json::Map<_, _>>(),
        "routes": ROUTES
            .iter()
            .map(|(key, serves, backends, auto)| {
                (
                    (*key).to_owned(),
                    serde_json::json!({"serves": serves, "backends": backends, "auto": auto}),
                )
            })
            .collect::<serde_json::Map<_, _>>(),
    })
}

/// The chosen backend of a call: the request, the effective backend and the
/// reason (the Python API's `VirtualReduction`), and the per-point fallback
/// (`"auto"` on TIRust where NinjaRS serves the call too: NinjaRS; none
/// otherwise, an explicit request included). `unchecked` is the call's
/// unchecked setting ([`DEFAULT_UNCHECKED`] from [`resolve`], which chooses
/// the backend whatever the setting; the caller sets the call's), recorded
/// with the choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub requested: String,
    pub effective: String,
    pub reason: String,
    pub fallback: Option<String>,
    pub unchecked: bool,
}

/// Why a request is not served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No backend serves the request (the Python API's typed
    /// `ReductionUnavailable`).
    Unavailable(String),
    /// An unknown backend, carrier or contribution (a `ValueError`).
    Invalid(String),
}

/// The served backends of a native build ([`crate::runtime::backend_carriers`])
/// as the map [`resolve`] reads.
pub fn served() -> BTreeMap<String, BTreeSet<String>> {
    crate::runtime::backend_carriers()
        .into_iter()
        .map(|(name, carriers)| {
            (
                name.to_owned(),
                carriers.into_iter().map(str::to_owned).collect(),
            )
        })
        .collect()
}

fn listed(names: &BTreeSet<String>) -> String {
    let quoted: Vec<String> = names.iter().map(|b| format!("'{}'", name_of(b))).collect();
    format!("[{}]", quoted.join(", "))
}

/// Why `backend` cannot serve `carrier` on `route` (None: it can).
fn unavailable(
    backend: &str,
    carrier: &str,
    served: &BTreeMap<String, BTreeSet<String>>,
    admitted: &BTreeSet<String>,
    route: &Route,
) -> Option<String> {
    let name = name_of(backend);
    if !admitted.contains(backend) {
        return Some(format!(
            "the artifact admits {} only, not {name}",
            listed(admitted)
        ));
    }
    let Some(carriers) = served.get(backend) else {
        let feature = if backend == "tirust" {
            " (the 'tir' feature)"
        } else {
            ""
        };
        return Some(format!(
            "{name} is not built into this native extension{feature}"
        ));
    };
    if !carriers.contains(carrier) {
        return Some(format!(
            "{name} does not serve the {carrier} carrier in this native extension"
        ));
    }
    if !route.backends.contains(backend) {
        return Some(format!(
            "{name} is not wired to the executor of {} in this build",
            route.serves
        ));
    }
    None
}

/// The [`Choice`] of a call on `route` (module docstring): `requested`
/// (`"auto"`, `"ninjars"` or `"tirust"`) for a `contribution` (`"virtual"`
/// or `"loop-induced"`) at `carrier`, with the backends the native build
/// serves (`served`, backend -> carriers) and those the artifact admits
/// (`admitted`; `None`: every backend). An explicit request that cannot be
/// served is [`Refusal::Unavailable`]; `"auto"` without any backend for the
/// carrier as well.
pub fn resolve(
    requested: &str,
    contribution: &str,
    carrier: &str,
    served: &BTreeMap<String, BTreeSet<String>>,
    admitted: Option<&BTreeSet<String>>,
    route: &Route,
) -> Result<Choice, Refusal> {
    if requested != "auto" && !BACKENDS.contains(&requested) {
        return Err(Refusal::Invalid(format!(
            "unknown reduction backend '{requested}'"
        )));
    }
    let Some(type_default) = default_of(contribution) else {
        return Err(Refusal::Invalid(format!(
            "unknown contribution '{contribution}'"
        )));
    };
    if !CARRIERS.contains(&carrier) {
        return Err(Refusal::Invalid(format!("unknown carrier '{carrier}'")));
    }
    let every: BTreeSet<String> = BACKENDS.iter().map(|b| (*b).to_owned()).collect();
    let allowed = admitted.unwrap_or(&every);
    let choice = |effective: &str, reason: String| Choice {
        requested: requested.to_owned(),
        effective: effective.to_owned(),
        reason,
        fallback: None,
        unchecked: DEFAULT_UNCHECKED,
    };
    if requested != "auto" {
        if let Some(why) = unavailable(requested, carrier, served, allowed, route) {
            return Err(Refusal::Unavailable(format!(
                "reduction '{requested}' was requested, but {why}; request \
                 'auto' or the other backend"
            )));
        }
        return Ok(choice(requested, "requested".to_owned()));
    }
    let tir_only = tir_only(Some(allowed));
    // A TIR-only artifact: no NinjaRS fallback exists (m-lit Q18, Q24).
    let default = auto_default(contribution, Some(allowed)).unwrap_or(type_default);
    let kind = kind_of(contribution);
    let why = unavailable(default, carrier, served, allowed, route);
    let fallback_serves = unavailable(FALLBACK, carrier, served, allowed, route).is_none();
    if why.is_none() && !route.automatic(default) && !tir_only && fallback_serves {
        // Served on explicit request, not by "auto" on this route
        // (`Route::auto`).
        return Ok(choice(
            FALLBACK,
            format!(
                "auto: NinjaRS; {}, the {kind} default, serves {} on explicit \
                 request only",
                name_of(default),
                route.serves
            ),
        ));
    }
    let Some(why) = why else {
        // A TIR-only artifact's reason names its one backend for both
        // process types (m-lit lit-eft), the loop-induced default included;
        // it has no fallback.
        if tir_only {
            return Ok(choice(
                default,
                format!(
                    "auto: {}, the only backend the artifact admits",
                    name_of(default)
                ),
            ));
        }
        if default != FALLBACK && fallback_serves {
            // The per-point fallback (m-lit F3 (c)).
            return Ok(Choice {
                fallback: Some(FALLBACK.to_owned()),
                ..choice(
                    default,
                    format!(
                        "auto: {}, the {kind} default; a point it refuses or fails is \
                         evaluated again on {} (the per-point fallback, recorded per point)",
                        name_of(default),
                        name_of(FALLBACK)
                    ),
                )
            });
        }
        return Ok(choice(
            default,
            format!("auto: {}, the {kind} default", name_of(default)),
        ));
    };
    let fallback = FALLBACK;
    let other = unavailable(fallback, carrier, served, allowed, route);
    if fallback == default || other.is_some() {
        let tail = other.map(|o| format!("; {o}")).unwrap_or_default();
        return Err(Refusal::Unavailable(format!(
            "no reduction backend serves this {kind} call at the {carrier} \
             carrier: {why}{tail}"
        )));
    }
    Ok(choice(
        "ninjars",
        format!(
            "auto: {}, the {kind} default, is unavailable ({why}); NinjaRS \
             serves the call",
            name_of(default)
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    fn served_of(entries: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        entries
            .iter()
            .map(|(name, carriers)| ((*name).to_owned(), set(carriers)))
            .collect()
    }

    #[test]
    fn auto_takes_tirust_for_loop_induced_with_the_per_point_fallback() {
        let both = served_of(&[("ninjars", &CARRIERS), ("tirust", &CARRIERS)]);
        for key in [SUMMED, RESOLVED] {
            let route = route(key).unwrap();
            assert_eq!(route.backends, set(&BACKENDS));
            for carrier in CARRIERS {
                let found = resolve("auto", "virtual", carrier, &both, None, &route).unwrap();
                assert_eq!(found.effective, "ninjars");
                assert_eq!(found.fallback, None);
                assert_eq!(found.reason, "auto: NinjaRS, the Born-interfered default");
                let found = resolve("auto", "loop-induced", carrier, &both, None, &route).unwrap();
                assert_eq!(found.effective, "tirust", "{key} {carrier}");
                assert_eq!(found.fallback.as_deref(), Some("ninjars"));
                assert_eq!(
                    found.reason,
                    "auto: TIRust, the loop-induced default; a point it refuses or fails \
                     is evaluated again on NinjaRS (the per-point fallback, recorded per point)"
                );
                // Explicit backends: never switched, no fallback.
                for explicit in BACKENDS {
                    for kind in CONTRIBUTIONS {
                        let found = resolve(explicit, kind, carrier, &both, None, &route).unwrap();
                        assert_eq!(
                            (found.requested.as_str(), found.effective.as_str()),
                            (explicit, explicit)
                        );
                        assert_eq!(found.reason, "requested");
                        assert_eq!(found.fallback, None);
                    }
                }
            }
        }
    }

    #[test]
    fn one_auto_default_helper() {
        let only = set(&["tirust"]);
        assert_eq!(auto_default("loop-induced", None), Some("tirust"));
        assert_eq!(auto_default("virtual", None), Some("ninjars"));
        assert_eq!(auto_default("virtual", Some(&only)), Some("tirust"));
        assert_eq!(auto_default("loop-induced", Some(&only)), Some("tirust"));
        assert_eq!(
            auto_default("virtual", Some(&set(&BACKENDS))),
            Some("ninjars")
        );
        assert_eq!(auto_default("tree", None), None);
    }

    #[test]
    fn the_target_is_a_finite_positive_number() {
        assert_eq!(target(DEFAULT_TARGET), Ok(1e-5));
        assert_eq!(target(1e-12), Ok(1e-12));
        for bad in [0.0, -1e-5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                target(bad),
                Err(Refusal::Invalid(TARGET_REFUSAL.to_owned()))
            );
        }
    }

    #[test]
    fn the_unchecked_setting_is_off_by_default_and_a_boolean() {
        const { assert!(!DEFAULT_UNCHECKED) };
        assert_eq!(unchecked(0), Ok(false));
        assert_eq!(unchecked(1), Ok(true));
        for bad in [-1, 2, i64::MAX, i64::MIN] {
            assert_eq!(
                unchecked(bad),
                Err(Refusal::Invalid(UNCHECKED_REFUSAL.to_owned()))
            );
        }
        let record = policy_json();
        assert_eq!(record["unchecked"]["default"], false);
        assert_eq!(record["unchecked"]["refusal"], UNCHECKED_REFUSAL);
        assert_eq!(record["unchecked"]["record"], UNCHECKED_RECORD);
    }

    #[test]
    fn fallback_texts() {
        assert_eq!(
            fallback_served("tirust", "ninjars", "family 2: refused"),
            "auto: NinjaRS, the per-point fallback: TIRust refused or failed the point \
             (family 2: refused)"
        );
        assert_eq!(
            fallback_failed("ninjars", "ImpreciseResult"),
            "NinjaRS fallback: ImpreciseResult"
        );
        assert!(cross_refusal("x").ends_with("other reduction backend: x"));
    }

    #[test]
    fn the_fallback_guard_has_one_reason_and_one_threshold() {
        // The one reason of a guard-rejected point: the fallback, the
        // primary backend's failure, the check, the estimate, the limit (the
        // call's target, never below the floor) and the carrier (m-lit
        // lit-fallback-guard, DECISIONS 2026-11-09 05:56 (1) and 11:53).
        assert_eq!(
            fallback_rejected(
                "tirust",
                "ninjars",
                "TIRust refused the row",
                "double-double",
                142.37,
                1e-5
            ),
            "auto: TIRust refused or failed the point (TIRust refused the row); the per-point \
             NinjaRS fallback's value failed its reduction-direction check: estimate 1.4e2 > \
             1e-5 = max(target 1e-5, 1e-5) (double-double)"
        );
        assert!(
            fallback_rejected("tirust", "ninjars", "x", "arbitrary", f64::NAN, 3e-7).ends_with(
                "check: estimate NaN, not within 1e-5 = max(target 3e-7, 1e-5) (arbitrary)"
            )
        );
        assert!(
            fallback_rejected("tirust", "ninjars", "x", "arbitrary", 2e-3, 1e-3)
                .ends_with("estimate 2.0e-3 > 1e-3 = max(target 1e-3, 1e-5) (arbitrary)")
        );
        // The only threshold is the call's target, never below the floor
        // (USER ~11:53); a NaN never passes.
        assert_eq!(FALLBACK_FLOOR, 1e-5);
        assert_eq!(fallback_limit(1e-8), 1e-5);
        assert_eq!(fallback_limit(1e-3), 1e-3);
        assert!(fallback_accepted(1e-5, 1e-5));
        assert!(fallback_accepted(0.0, 1e-5));
        assert!(fallback_accepted(9e-6, 1e-8));
        assert!(fallback_accepted(1e-3, 1e-3));
        assert!(!fallback_accepted(1.0000001e-5, 1e-5));
        assert!(!fallback_accepted(1.0000001e-5, 1e-8));
        assert!(!fallback_accepted(f64::NAN, 1e-5));
        assert!(!fallback_accepted(f64::INFINITY, 1e-5));
        assert_eq!(FALLBACK_CHECK, "reduction-direction");
        // The extended carriers only (USER ~11:53, 'Double-double+ only').
        assert_eq!(
            CARRIERS.map(fallback_guarded),
            [false, true, true],
            "{CARRIERS:?}"
        );
    }

    #[test]
    fn unavailable_backends_fall_back_or_are_refused_typed() {
        let every = Route {
            serves: "a test route".to_owned(),
            backends: set(&BACKENDS),
            auto: None,
        };
        let ninja = served_of(&[("ninjars", &CARRIERS)]);
        let found = resolve("auto", "loop-induced", "double", &ninja, None, &every).unwrap();
        assert_eq!(found.effective, "ninjars");
        assert_eq!(
            found.reason,
            "auto: TIRust, the loop-induced default, is unavailable (TIRust is \
             not built into this native extension (the 'tir' feature)); NinjaRS \
             serves the call"
        );
        assert!(matches!(
            resolve("tirust", "virtual", "double", &ninja, None, &every),
            Err(Refusal::Unavailable(message)) if message.contains("not built into")
        ));
        let tir = served_of(&[("ninjars", &CARRIERS), ("tirust", &["double"])]);
        let found = resolve("auto", "loop-induced", "double", &tir, None, &every).unwrap();
        assert_eq!(found.effective, "tirust");
        assert_eq!(found.fallback.as_deref(), Some("ninjars"));
        let found = resolve("auto", "loop-induced", "double-double", &tir, None, &every).unwrap();
        assert!(
            found
                .reason
                .contains("does not serve the double-double carrier")
        );
        let only = set(&["tirust"]);
        let found = resolve("auto", "virtual", "double", &tir, Some(&only), &every).unwrap();
        assert_eq!(found.effective, "tirust");
        assert_eq!(
            found.reason,
            "auto: TIRust, the only backend the artifact admits"
        );
        let found = resolve("auto", "loop-induced", "double", &tir, Some(&only), &every).unwrap();
        assert_eq!(found.effective, "tirust");
        assert_eq!(found.fallback, None);
        assert_eq!(
            found.reason,
            "auto: TIRust, the only backend the artifact admits"
        );
        assert!(matches!(
            resolve("ninjars", "virtual", "double", &tir, Some(&only), &every),
            Err(Refusal::Unavailable(message)) if message.contains("the artifact admits ['TIRust'] only")
        ));
        assert!(matches!(
            resolve("auto", "loop-induced", "double-double", &tir, Some(&only), &every),
            Err(Refusal::Unavailable(message)) if message.starts_with("no reduction backend serves")
        ));
        assert!(matches!(
            resolve("fast", "virtual", "double", &tir, None, &every),
            Err(Refusal::Invalid(_))
        ));
        let wired = Route {
            serves: "a NinjaRS route".to_owned(),
            backends: set(&["ninjars"]),
            auto: None,
        };
        let found = resolve("auto", "loop-induced", "double", &tir, None, &wired).unwrap();
        assert!(
            found
                .reason
                .contains("not wired to the executor of a NinjaRS route")
        );
    }

    #[test]
    fn the_policy_record_lists_the_defaults_and_routes() {
        let record = policy_json();
        assert_eq!(record["defaults"]["loop-induced"], "tirust");
        assert_eq!(record["defaults"]["virtual"], "ninjars");
        assert_eq!(
            record["routes"][SUMMED]["auto"],
            serde_json::json!(["ninjars", "tirust"])
        );
        assert_eq!(record["fallback"], "ninjars");
        assert_eq!(record["target"]["default"], DEFAULT_TARGET);
        assert_eq!(record["target"]["refusal"], TARGET_REFUSAL);
        assert_eq!(
            record["routes"][RESOLVED]["backends"],
            serde_json::json!(["ninjars", "tirust"])
        );
    }
}
