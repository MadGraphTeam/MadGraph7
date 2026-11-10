// SPDX-License-Identifier: 0BSD

//! The kinematics improvement's Python binding (m-lit lit-psimprove): the
//! one Rust implementation (`rusticol_core::kinematics_improvement`) that
//! every Python caller runs (the loop runtimes, `check`, the tree exact
//! executors); Python never re-implements it.

use super::error;
use pyo3::prelude::*;
use rusticol_core::kinematics_improvement::{self as ki, Arithmetic, Tolerances};
use rusticol_core::loop_v2::dress::parse_rational;
use symbolica::domains::float::Float;
use symbolica::domains::rational::Rational;

/// The settings' policy record as JSON (`kinematics_improvement::
/// policy_json`: the values, the default variant, the binary64 switch's
/// default, the refusal texts, the key and MadLoop's tolerances).
#[pyfunction]
pub fn kinematics_policy() -> String {
    ki::policy_json().to_string()
}

/// The variant an evaluation in `carrier` ("double", "double-double",
/// "arbitrary") runs under the variant `setting` and the binary64 switch
/// (`kinematics_improvement::applies`, the one key: the carrier kind), or
/// None; a `ValueError` with the setting's refusal text for another value.
#[pyfunction]
pub fn kinematics_applies(
    setting: &str,
    binary64: bool,
    carrier: &str,
) -> PyResult<Option<&'static str>> {
    let setting = ki::setting(setting).map_err(error)?;
    Ok(ki::applies(setting, binary64, carrier == "double").map(ki::Variant::name))
}

/// The carrier of `arithmetic`: "double", "double-double", or "arbitrary"
/// at `digits` decimal digits (the executor's own bits, `Float::
/// decimal_digits_to_bits`).
pub fn arithmetic_of(arithmetic: &str, digits: Option<u32>) -> PyResult<Arithmetic> {
    match (arithmetic, digits) {
        ("double", _) => Ok(Arithmetic::Binary64),
        ("double-double", _) => Ok(Arithmetic::DoubleDouble),
        ("arbitrary", Some(digits)) if digits > 0 => Ok(Arithmetic::Arbitrary(
            Float::decimal_digits_to_bits(f64::from(digits)).map_err(error)?,
        )),
        _ => Err(error(
            "the kinematics improvement's arithmetic is double, double-double or arbitrary with digits",
        )),
    }
}

/// MadLoop's input-sanity and move bounds, its defaults unless overridden.
pub fn tolerances(
    fourmom: Option<f64>,
    onshell: Option<f64>,
    close: Option<f64>,
) -> PyResult<Tolerances> {
    let defaults = Tolerances::default();
    let check = |value: Option<f64>, default: f64, name: &str| match value {
        None => Ok(default),
        Some(v) if v.is_finite() && v > 0.0 => Ok(v),
        Some(_) => Err(error(format!(
            "the kinematics improvement's {name} tolerance is a finite positive number"
        ))),
    };
    Ok(Tolerances {
        fourmom: check(fourmom, defaults.fourmom, "fourmom")?,
        onshell: check(onshell, defaults.onshell, "onshell")?,
        close: check(close, defaults.close, "close")?,
    })
}

type Point = Vec<[String; 4]>;

/// Improve `points` (physical momenta, the `n_initial` initial states first,
/// each component an exact decimal or fraction text) in `arithmetic` with
/// `variant` and MadLoop's fallback (`fallback`, the public chain; False:
/// the variant alone, forced), toward the legs' squared masses `masses2`
/// (exact texts; None: a leg without a bound mass keeps its own p²). Returns
/// the improved points as exact fraction texts of the carrier values (a
/// point no variant improves, or an exactly physical one, unchanged) and
/// each point's record as JSON (`kinematics_improvement::Record::to_json`).
#[pyfunction]
#[pyo3(signature = (points, n_initial, masses2, arithmetic, digits=None, variant="z-shift", fallback=true, fourmom=None, onshell=None, close=None))]
#[allow(clippy::too_many_arguments)]
pub fn improve_kinematics(
    py: Python<'_>,
    points: Vec<Vec<Vec<String>>>,
    n_initial: usize,
    masses2: Vec<Option<String>>,
    arithmetic: &str,
    digits: Option<u32>,
    variant: &str,
    fallback: bool,
    fourmom: Option<f64>,
    onshell: Option<f64>,
    close: Option<f64>,
) -> PyResult<(Vec<Point>, Vec<String>)> {
    let arithmetic = arithmetic_of(arithmetic, digits)?;
    let variant = ki::setting(variant)
        .map_err(error)?
        .ok_or_else(|| error("the kinematics improvement 'off' improves nothing"))?;
    let tolerances = tolerances(fourmom, onshell, close)?;
    let masses: Vec<Option<Rational>> = masses2
        .iter()
        .map(|m| m.as_deref().map(parse_rational).transpose().map_err(error))
        .collect::<PyResult<_>>()?;
    let legs = masses.len();
    if n_initial == 0 || n_initial >= legs {
        return Err(error("a point has initial and final states"));
    }
    let parsed: Vec<Vec<[Rational; 4]>> = points
        .iter()
        .map(|point| {
            if point.len() != legs {
                return Err(error("one squared mass per leg of every point"));
            }
            point
                .iter()
                .map(|leg| {
                    let leg: [String; 4] = leg
                        .clone()
                        .try_into()
                        .map_err(|_| error("a momentum has four components"))?;
                    let mut out = std::array::from_fn(|_| Rational::from(0));
                    for (slot, text) in out.iter_mut().zip(&leg) {
                        *slot = parse_rational(text).map_err(error)?;
                    }
                    Ok(out)
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<_>>()?;
    let out = py.detach(|| {
        parsed
            .iter()
            .map(|legs| {
                ki::improve_exact(
                    legs,
                    n_initial,
                    &masses,
                    arithmetic,
                    variant,
                    fallback,
                    &tolerances,
                )
            })
            .collect::<Vec<_>>()
    });
    Ok(out
        .into_iter()
        .map(|(legs, record)| {
            (
                legs.iter()
                    .map(|leg| leg.each_ref().map(ToString::to_string))
                    .collect(),
                record.to_json().to_string(),
            )
        })
        .unzip())
}
