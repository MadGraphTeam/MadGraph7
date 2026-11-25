// SPDX-License-Identifier: 0BSD

//! The runtime options of a loaded loop runtime, read from a TOML text: the
//! native SDKs' reader of the run card's `[evaluation]` table (USER
//! 2026-11-19: 'many other runtime settings (simpler to support) should be
//! done in a toml-able API function that the SDK can call'; the stability
//! rescue stack stays the Python API's alone).
//!
//! A runtime-options file is a TOML with an `[evaluation]` table; a full run
//! card is accepted too: only `[evaluation]` is read, every other table is
//! ignored. Inside `[evaluation]` the runtime options ([`KEYS`]) are read,
//! the run card's other keys ([`OTHER_KEYS`]: the inputs of a command-line
//! run, the removed rescue settings) are accepted and not read, and any other
//! key is refused by name ([`unknown_key`], the run card's text). Each key
//! takes the values the Python configuration takes
//! (`config.models.EvaluationConfig`), with its refusal texts; the
//! parity test `tests/unit/test_runtime_options.py` holds them equal key by
//! key. An option that is absent leaves the runtime's setting as it is.
//!
//! Adding or removing a runtime option is one line of [`KEYS`] (and its
//! field of [`RuntimeOptions`]).

use crate::reduction_policy;
use rusticol_core::kinematics_improvement;

/// The table a runtime-options TOML is read from.
pub const TABLE: &str = "evaluation";

/// The settings of a runtime-options TOML: each `None` where the TOML does
/// not give it (the runtime's current setting then stays).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeOptions {
    /// `"double"`, `"double-double"` or `"arbitrary"` (with
    /// [`Self::precision`]: [`Self::carrier`]).
    pub arithmetic: Option<String>,
    /// Decimal digits (positive).
    pub precision: Option<u32>,
    /// `"all"`, `"none"`, `"uv"` or `"ir"`.
    pub poles: Option<String>,
    /// `"best-effort"`, `"return"` or `"raise"`.
    pub on_failure: Option<String>,
    /// `"auto"`, `"ninjars"` or `"tirust"`.
    pub reduction: Option<String>,
    /// A finite positive relative accuracy ([`reduction_policy::target`]).
    pub reduction_target: Option<f64>,
    pub reduction_unchecked: Option<bool>,
    pub lane_rescue: Option<bool>,
    /// `"z-shift"`, `"rescaling"`, `"projection"` or `"off"`.
    pub kinematics_improvement: Option<String>,
    pub kinematics_improvement_binary64: Option<bool>,
    /// `[evaluation.stability_checks]`, when the TOML has the table.
    pub stability_checks: Option<StabilityChecksOptions>,
    /// The native threads of the runtime's loads and evaluations (positive).
    pub threads: Option<usize>,
}

/// `[evaluation.stability_checks]`: its keys at their defaults where absent
/// (the table given replaces the runtime's checks as a whole).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StabilityChecksOptions {
    pub rotations: Vec<String>,
    pub rotation_count: u64,
    pub reduction_direction: bool,
    pub rescaled: bool,
}

type Reader = fn(&mut RuntimeOptions, &str, &toml::Value) -> Result<(), String>;

/// THE RUNTIME OPTIONS: each `[evaluation]` key (a key of a sub-table is
/// dotted) and its reader. One line per option.
pub const KEYS: &[(&str, Reader)] = &[
    ("arithmetic", |o, p, v| {
        o.arithmetic = Some(choice(p, v, &reduction_policy::CARRIERS)?);
        Ok(())
    }),
    ("precision", |o, p, v| {
        o.precision = Some(positive(p, v, u32::MAX as u64)? as u32);
        Ok(())
    }),
    ("poles", |o, p, v| {
        o.poles = Some(choice(p, v, &POLES)?);
        Ok(())
    }),
    ("on_failure", |o, p, v| {
        o.on_failure = Some(choice(p, v, &ON_FAILURE)?);
        Ok(())
    }),
    ("reduction", |o, p, v| {
        o.reduction = Some(choice(p, v, &REDUCTIONS)?);
        Ok(())
    }),
    ("reduction_target", |o, p, v| {
        let value = match v {
            toml::Value::Float(value) => *value,
            toml::Value::Integer(value) => *value as f64,
            _ => return Err(format!("{p} must be a number")),
        };
        o.reduction_target = Some(
            reduction_policy::target(value)
                .map_err(|_| format!("{p}: {}", reduction_policy::TARGET_REFUSAL))?,
        );
        Ok(())
    }),
    ("reduction_unchecked", |o, p, v| {
        o.reduction_unchecked = Some(boolean(p, v)?);
        Ok(())
    }),
    ("lane_rescue", |o, p, v| {
        o.lane_rescue = Some(boolean(p, v)?);
        Ok(())
    }),
    ("kinematics_improvement", |o, p, v| {
        let text = string(p, v)?;
        kinematics_improvement::setting(&text).map_err(|refusal| format!("{p}: {refusal}"))?;
        o.kinematics_improvement = Some(text);
        Ok(())
    }),
    ("kinematics_improvement_binary64", |o, p, v| {
        o.kinematics_improvement_binary64 = Some(boolean(p, v)?);
        Ok(())
    }),
    ("stability_checks.rotations", |o, p, v| {
        let toml::Value::Array(items) = v else {
            return Err(format!("{p} must be a list of strings"));
        };
        let mut rotations = Vec::with_capacity(items.len());
        for item in items {
            match item {
                toml::Value::String(text) if !text.is_empty() => rotations.push(text.clone()),
                toml::Value::String(_) => {
                    return Err(format!("{p} must contain non-empty strings"));
                }
                _ => return Err(format!("{p} must be a list of strings")),
            }
        }
        o.stability_checks
            .get_or_insert_with(Default::default)
            .rotations = rotations;
        Ok(())
    }),
    ("stability_checks.rotation_count", |o, p, v| {
        let toml::Value::Integer(count) = v else {
            return Err(format!("{p} must be an integer"));
        };
        let count = u64::try_from(*count).map_err(|_| format!("{p} must be an integer >= 0"))?;
        o.stability_checks
            .get_or_insert_with(Default::default)
            .rotation_count = count;
        Ok(())
    }),
    ("stability_checks.reduction_direction", |o, p, v| {
        o.stability_checks
            .get_or_insert_with(Default::default)
            .reduction_direction = boolean(p, v)?;
        Ok(())
    }),
    ("stability_checks.rescaled", |o, p, v| {
        o.stability_checks
            .get_or_insert_with(Default::default)
            .rescaled = boolean(p, v)?;
        Ok(())
    }),
    ("threads", |o, p, v| {
        o.threads = Some(positive(p, v, usize::MAX as u64)? as usize);
        Ok(())
    }),
];

/// The run card's other `[evaluation]` keys: accepted in a runtime-options
/// TOML (a full run card) and not read: the inputs of a command-line run
/// and the removed rescue settings (`[evaluation.stability]`).
pub const OTHER_KEYS: &[&str] = &[
    "artifact",
    "process",
    "mu_r2",
    "mu_r2_file",
    "resolved",
    "orders",
    "correlated_requests",
    "helicity_ids",
    "color_flow_ids",
    "model_parameters",
    "momenta",
    "stability.higher",
    "stability.relative_tolerance",
    "stability.absolute_tolerance",
    "stability.verify_successful",
    "phase_space.points",
    "phase_space.seed",
    "phase_space.sqrt_s",
];

/// The sub-tables of `[evaluation]`.
const SECTIONS: [&str; 3] = ["stability", "stability_checks", "phase_space"];

const POLES: [&str; 4] = ["all", "none", "uv", "ir"];
const ON_FAILURE: [&str; 3] = ["best-effort", "return", "raise"];
const REDUCTIONS: [&str; 3] = ["auto", "ninjars", "tirust"];

impl RuntimeOptions {
    /// The options of a TOML text: its `[evaluation]` table (none: no
    /// option), every key read and checked; the refusals of every bad key,
    /// as the run card states them (one message, or `"N configuration
    /// errors:"` and one line per refusal).
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let document: toml::Table =
            toml::from_str(text).map_err(|error| format!("invalid TOML: {error}"))?;
        let mut options = Self::default();
        let mut errors = Vec::new();
        match document.get(TABLE) {
            None => {}
            Some(toml::Value::Table(table)) => options.read(table, "", &mut errors),
            Some(_) => errors.push(format!("configuration section '{TABLE}' must be a table")),
        }
        if let (Some(checks), true) = (&options.stability_checks, errors.is_empty()) {
            if !checks.rotations.is_empty() && checks.rotation_count != 0 {
                errors.push(format!(
                    "{TABLE}.stability_checks takes rotations or rotation_count"
                ));
            } else if let Err(error) = checks.checks() {
                errors.push(format!("{TABLE}.stability_checks: {error}"));
            }
        }
        match errors.len() {
            0 => Ok(options),
            1 => Err(errors.remove(0)),
            n => Err(format!(
                "{n} configuration errors:\n{}",
                errors
                    .iter()
                    .map(|error| format!("- {error}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
        }
    }

    fn read(&mut self, table: &toml::Table, prefix: &str, errors: &mut Vec<String>) {
        for (key, value) in table {
            let name = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            let path = format!("{TABLE}.{name}");
            if prefix.is_empty() && SECTIONS.contains(&key.as_str()) {
                match value {
                    toml::Value::Table(inner) => self.read(inner, key, errors),
                    _ => errors.push(format!("configuration section '{path}' must be a table")),
                }
                if key == "stability_checks" {
                    self.stability_checks.get_or_insert_with(Default::default);
                }
                continue;
            }
            if let Some((_, reader)) = KEYS.iter().find(|(known, _)| *known == name) {
                if let Err(error) = reader(self, &path, value) {
                    errors.push(error);
                }
            } else if !OTHER_KEYS.contains(&name.as_str()) {
                errors.push(unknown_key(&path));
            }
        }
    }

    /// The runtime options given, by their run-card names (the
    /// `[evaluation.stability_checks]` table as `"stability_checks"`).
    pub fn given(&self) -> Vec<&'static str> {
        [
            ("arithmetic", self.arithmetic.is_some()),
            ("precision", self.precision.is_some()),
            ("poles", self.poles.is_some()),
            ("on_failure", self.on_failure.is_some()),
            ("reduction", self.reduction.is_some()),
            ("reduction_target", self.reduction_target.is_some()),
            ("reduction_unchecked", self.reduction_unchecked.is_some()),
            ("lane_rescue", self.lane_rescue.is_some()),
            (
                "kinematics_improvement",
                self.kinematics_improvement.is_some(),
            ),
            (
                "kinematics_improvement_binary64",
                self.kinematics_improvement_binary64.is_some(),
            ),
            ("stability_checks", self.stability_checks.is_some()),
            ("threads", self.threads.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, given)| given.then_some(name))
        .collect()
    }

    /// The carrier of [`Self::arithmetic`] and [`Self::precision`], as the
    /// Python API selects it (`V2VirtualRuntime._carrier`): an arithmetic
    /// given is that carrier ("arbitrary" at the digits given, else at the
    /// default arbitrary digits, [`reduction_policy::DEFAULT_ARBITRARY_DIGITS`];
    /// the other carriers take no digits), else 16 digits are binary64 and
    /// other digits arbitrary precision; neither: None (unchanged).
    pub fn carrier(&self) -> Option<(&'static str, Option<u32>)> {
        match (self.arithmetic.as_deref(), self.precision) {
            (None, None) => None,
            (Some("double") | None, Some(16)) | (Some("double"), _) => Some(("double", None)),
            (Some("double-double"), _) => Some(("double-double", None)),
            (Some(_), digits) | (None, digits) => Some((
                "arbitrary",
                Some(digits.unwrap_or(reduction_policy::DEFAULT_ARBITRARY_DIGITS)),
            )),
        }
    }
}

impl StabilityChecksOptions {
    /// The checks (`v2_stability::StabilityChecks`, validated as the Python
    /// API's `VirtualStabilityChecks`; no cross-backend check: the run card
    /// has none).
    pub fn checks(&self) -> rusticol_core::RusticolResult<crate::v2_stability::StabilityChecks> {
        use crate::v2_stability::{Rotation, StabilityChecks};
        if self.rotations.is_empty() {
            StabilityChecks::with_default_rotations(
                self.rotation_count,
                self.reduction_direction,
                self.rescaled,
                false,
            )
        } else {
            let rotations = self
                .rotations
                .iter()
                .map(|text| Rotation::parse(text))
                .collect::<rusticol_core::RusticolResult<Vec<_>>>()?;
            StabilityChecks::new(rotations, self.reduction_direction, self.rescaled, false)
        }
    }
}

/// Every `[evaluation]` key a runtime-options TOML accepts (runtime options
/// and the run card's other keys), dotted, as the run card names them.
pub fn known_keys() -> Vec<String> {
    KEYS.iter()
        .map(|(key, _)| *key)
        .chain(OTHER_KEYS.iter().copied())
        .map(|key| format!("{TABLE}.{key}"))
        .collect()
}

/// The runtime options' keys ([`KEYS`]), dotted.
pub fn runtime_keys() -> Vec<String> {
    KEYS.iter()
        .map(|(key, _)| format!("{TABLE}.{key}"))
        .collect()
}

/// The refusal of an unknown `[evaluation]` key, the run card's text
/// (`config.resolver._unknown_field_message`): the closest known key by
/// Python's `difflib.get_close_matches(path, keys, n=1, cutoff=0.55)`.
pub fn unknown_key(path: &str) -> String {
    let suffix = close_match(path, &known_keys())
        .map(|found| format!("; did you mean '{found}'?"))
        .unwrap_or_default();
    format!("unknown configuration field '{path}'{suffix}")
}

fn choice(path: &str, value: &toml::Value, choices: &[&str]) -> Result<String, String> {
    let text = string(path, value)?;
    if choices.contains(&text.as_str()) {
        return Ok(text);
    }
    let allowed = choices
        .iter()
        .map(|item| format!("'{item}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "{path} must be one of {allowed}; got {}",
        python_repr(&text)
    ))
}

fn string(path: &str, value: &toml::Value) -> Result<String, String> {
    match value {
        toml::Value::String(text) => Ok(text.clone()),
        _ => Err(format!("{path} must be a string")),
    }
}

fn boolean(path: &str, value: &toml::Value) -> Result<bool, String> {
    match value {
        toml::Value::Boolean(value) => Ok(*value),
        _ => Err(format!("{path} must be true or false")),
    }
}

/// A positive integer (the run card's `int` kind, then `_integer`'s
/// `minimum=1`), at most `largest`.
fn positive(path: &str, value: &toml::Value, largest: u64) -> Result<u64, String> {
    let toml::Value::Integer(value) = value else {
        return Err(format!("{path} must be an integer"));
    };
    match u64::try_from(*value) {
        Ok(value @ 1..) if value <= largest => Ok(value),
        Ok(value @ 1..) => Err(format!("{path} must be at most {largest}; got {value}")),
        _ => Err(format!("{path} must be a positive integer")),
    }
}

/// Python's `repr` of a string (single quotes unless it holds one and no
/// double quote).
fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `difflib.get_close_matches(word, candidates, n=1, cutoff=0.55)`: the
/// candidate of the highest `SequenceMatcher(None, candidate, word).ratio()`
/// at least the cutoff (ties: the larger candidate), as Python computes it
/// (no junk; texts this short never meet the autojunk heuristic).
fn close_match<'a>(word: &str, candidates: &'a [String]) -> Option<&'a str> {
    let b: Vec<char> = word.chars().collect();
    let mut best: Option<(f64, &str)> = None;
    for candidate in candidates {
        let a: Vec<char> = candidate.chars().collect();
        let total = a.len() + b.len();
        if total == 0 {
            continue;
        }
        let ratio = 2.0 * matches(&a, &b) as f64 / total as f64;
        if ratio >= 0.55 {
            let better = match best {
                None => true,
                Some((score, name)) => (ratio, candidate.as_str()) > (score, name),
            };
            if better {
                best = Some((ratio, candidate));
            }
        }
    }
    best.map(|(_, name)| name)
}

/// The matched characters of `SequenceMatcher(None, a, b)`: the sum of its
/// matching blocks (`get_matching_blocks`, `find_longest_match` without
/// junk).
fn matches(a: &[char], b: &[char]) -> usize {
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut total = 0;
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest_match(a, b, alo, ahi, blo, bhi);
        if k > 0 {
            total += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    total
}

/// `SequenceMatcher.find_longest_match` without junk: the longest block
/// a[i..i+k] == b[j..j+k] in the ranges, the earliest in `a`, then in `b`.
fn longest_match(
    a: &[char],
    b: &[char],
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
    let mut previous = vec![0usize; b.len() + 1];
    for i in alo..ahi {
        let mut current = vec![0usize; b.len() + 1];
        for j in blo..bhi {
            if a[i] == b[j] {
                let k = if j > 0 { previous[j - 1] } else { 0 } + 1;
                current[j] = k;
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        previous = current;
    }
    (besti, bestj, bestsize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_reads_and_an_absent_one_stays_unset() {
        let options = RuntimeOptions::from_toml(
            "[model]\nsource = 'x'\n[evaluation]\nartifact = 'a'\nreduction = 'tirust'\n\
             reduction_target = 1e-7\nreduction_unchecked = true\nlane_rescue = false\n\
             arithmetic = 'arbitrary'\nprecision = 60\npoles = 'none'\non_failure = 'return'\n\
             kinematics_improvement = 'projection'\nkinematics_improvement_binary64 = true\n\
             threads = 3\n[evaluation.stability_checks]\nrotation_count = 2\n\
             reduction_direction = true\nrescaled = true\n",
        )
        .unwrap();
        assert_eq!(options.reduction.as_deref(), Some("tirust"));
        assert_eq!(options.reduction_target, Some(1e-7));
        assert_eq!(options.carrier(), Some(("arbitrary", Some(60))));
        assert_eq!(options.threads, Some(3));
        assert_eq!(options.stability_checks.as_ref().unwrap().rotation_count, 2);
        assert_eq!(
            RuntimeOptions::from_toml("").unwrap(),
            RuntimeOptions::default()
        );
        assert_eq!(
            RuntimeOptions::from_toml("[evaluation.stability_checks]\n")
                .unwrap()
                .stability_checks,
            Some(StabilityChecksOptions::default())
        );
    }

    #[test]
    fn the_carrier_follows_the_python_api() {
        let carrier = |text: &str| RuntimeOptions::from_toml(text).unwrap().carrier();
        assert_eq!(
            carrier("[evaluation]\nprecision = 16"),
            Some(("double", None))
        );
        assert_eq!(
            carrier("[evaluation]\nprecision = 30"),
            Some(("arbitrary", Some(30)))
        );
        assert_eq!(
            carrier("[evaluation]\narithmetic = 'arbitrary'"),
            Some((
                "arbitrary",
                Some(reduction_policy::DEFAULT_ARBITRARY_DIGITS)
            ))
        );
        assert_eq!(
            carrier("[evaluation]\narithmetic = 'double'\nprecision = 30"),
            Some(("double", None))
        );
        assert_eq!(
            carrier("[evaluation]\narithmetic = 'double-double'"),
            Some(("double-double", None))
        );
    }

    #[test]
    fn refusals_name_the_key_in_the_run_cards_words() {
        let refusal = |text: &str| RuntimeOptions::from_toml(text).unwrap_err();
        assert_eq!(
            refusal("[evaluation]\nredution = 'auto'"),
            "unknown configuration field 'evaluation.redution'; did you mean 'evaluation.reduction'?"
        );
        assert_eq!(
            refusal("[evaluation]\npoles = 'x'"),
            "evaluation.poles must be one of 'all', 'none', 'uv', 'ir'; got 'x'"
        );
        assert_eq!(
            refusal("[evaluation]\nthreads = 0"),
            "evaluation.threads must be a positive integer"
        );
        assert_eq!(
            refusal("[evaluation.stability_checks]\nrescaled = true"),
            "evaluation.stability_checks: rescaled is a variant of the reduction-direction check"
        );
        assert!(
            refusal("[evaluation]\npoles = 1\nthreads = 'x'")
                .starts_with("2 configuration errors:\n- ")
        );
        // Python's quoting (`float("q")`, `Fraction("a")`).
        assert_eq!(
            refusal("[evaluation.stability_checks]\nrotations = ['1,0,0:q']"),
            "evaluation.stability_checks: could not convert string to float: 'q'"
        );
        assert_eq!(
            refusal("[evaluation.stability_checks]\nrotations = ['x:a:b']"),
            "evaluation.stability_checks: Invalid literal for Fraction: 'a'"
        );
    }

    #[test]
    fn the_close_match_is_difflibs() {
        let keys: Vec<String> = ["abcd", "bcde", "xyz"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(close_match("bcd", &keys), Some("bcde"));
        assert_eq!(close_match("qqq", &keys), None);
    }
}
