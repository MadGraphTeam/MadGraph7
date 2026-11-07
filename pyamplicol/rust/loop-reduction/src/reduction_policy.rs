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
//! * `"auto"` takes the per-type default ([`default_of`]) when it serves the
//!   carrier on the call's route, else NinjaRS, with the reason recorded;
//! * an explicit backend that cannot serve the call is refused typed
//!   ([`Refusal::Unavailable`], the Python API's `ReductionUnavailable`):
//!   never a silent switch.
//!
//! What a backend serves is the native capability
//! (`runtime::backend_carriers`), passed in by the caller; an artifact may
//! admit fewer backends (`admitted`). A route ([`Route`]) is what a call
//! evaluates and the backends its executor runs, and `auto` there takes
//! only the backends of its `auto` set: NinjaRS on the artifact runtime's
//! routes ([`SUMMED`], [`RESOLVED`]) until the loop-induced agreement of the
//! two backends is established (m-lit lit-cache, which adds TIRust to their
//! `auto` sets here), so `"auto"` keeps NinjaRS for loop-induced processes
//! too, with the reason recorded, and TIRust serves explicit requests.

use std::collections::{BTreeMap, BTreeSet};

/// The backends, by name (`config.ReductionBackend` less `"auto"`).
pub const BACKENDS: [&str; 2] = ["ninjars", "tirust"];
/// The carriers, by name.
pub const CARRIERS: [&str; 3] = ["double", "double-double", "arbitrary"];
/// The contribution kinds, by name.
pub const CONTRIBUTIONS: [&str; 2] = ["virtual", "loop-induced"];

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
/// lit-tir); `"auto"` stays on NinjaRS until the loop-induced agreement of
/// the two backends is checked (m-lit lit-cache adds `"tirust"` to the
/// `auto` sets here, R3).
const ROUTES: [RouteDefinition; 2] = [
    (
        SUMMED,
        "the summed virtual",
        &["ninjars", "tirust"],
        &["ninjars"],
    ),
    (
        RESOLVED,
        "the colour-, spin- and helicity-resolved outputs and selections",
        &["ninjars", "tirust"],
        &["ninjars"],
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

/// The policy as JSON (what the Python API reads, `reduction.DEFAULTS` and
/// its routes): `{"backends", "carriers", "defaults": {kind: backend},
/// "routes": {key: {"serves", "backends", "auto"}}}`.
pub fn policy_json() -> serde_json::Value {
    serde_json::json!({
        "backends": BACKENDS,
        "carriers": CARRIERS,
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
/// reason (the Python API's `VirtualReduction`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub requested: String,
    pub effective: String,
    pub reason: String,
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
    let tir_only = allowed.len() == 1 && allowed.contains("tirust");
    // A TIR-only artifact: no NinjaRS fallback exists (m-lit Q18, Q24).
    let default = if tir_only { "tirust" } else { type_default };
    let kind = kind_of(contribution);
    let why = unavailable(default, carrier, served, allowed, route);
    if why.is_none()
        && !route.automatic(default)
        && !tir_only
        && unavailable("ninjars", carrier, served, allowed, route).is_none()
    {
        // Served on explicit request, not yet by "auto" (`Route::auto`).
        return Ok(choice(
            "ninjars",
            format!(
                "auto: NinjaRS; {}, the {kind} default, serves {} on explicit \
                 request until its agreement with NinjaRS on loop-induced \
                 processes is established",
                name_of(default),
                route.serves
            ),
        ));
    }
    let Some(why) = why else {
        let reason = if default == type_default {
            format!("auto: {}, the {kind} default", name_of(default))
        } else {
            format!(
                "auto: {}, the only backend the artifact admits",
                name_of(default)
            )
        };
        return Ok(choice(default, reason));
    };
    let fallback = "ninjars";
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
    fn auto_keeps_ninjars_on_the_artifact_routes_with_its_reason() {
        let both = served_of(&[("ninjars", &CARRIERS), ("tirust", &["double"])]);
        for key in [SUMMED, RESOLVED] {
            let route = route(key).unwrap();
            assert_eq!(route.backends, set(&BACKENDS));
            let found = resolve("auto", "virtual", "double", &both, None, &route).unwrap();
            assert_eq!(found.effective, "ninjars");
            assert_eq!(found.reason, "auto: NinjaRS, the Born-interfered default");
            let found = resolve("auto", "loop-induced", "double", &both, None, &route).unwrap();
            assert_eq!(found.effective, "ninjars");
            assert_eq!(
                found.reason,
                format!(
                    "auto: NinjaRS; TIRust, the loop-induced default, serves {} on \
                     explicit request until its agreement with NinjaRS on \
                     loop-induced processes is established",
                    route.serves
                )
            );
            let found = resolve("tirust", "loop-induced", "double", &both, None, &route).unwrap();
            assert_eq!(
                (found.requested.as_str(), found.effective.as_str()),
                ("tirust", "tirust")
            );
            assert_eq!(found.reason, "requested");
        }
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
        assert_eq!(found.reason, "auto: TIRust, the loop-induced default");
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
            serde_json::json!(["ninjars"])
        );
        assert_eq!(
            record["routes"][RESOLVED]["backends"],
            serde_json::json!(["ninjars", "tirust"])
        );
    }
}
