// SPDX-License-Identifier: 0BSD

//! The binary64 observables of a loop-induced process for the Python API
//! (`loops/v2/induced_observables`): the spin coefficients, the
//! spin-projected amplitude vectors and the carrier forms (a history's, the
//! amplitude-basis anchors') of the one Rust
//! implementation the native SDK runs
//! (`rusticol_loop_reduction::v2_runtime::induced_outputs`; m-lit LIT-SDK,
//! DECISIONS 2026-11-07 21:50 R7). The Python `Decimal` code stays the
//! oracle and the double-double / arbitrary-precision path.

use super::*;
use rusticol_loop_reduction::v2_decimal::{Binary64, C, Decimal};
use rusticol_loop_reduction::v2_runtime::induced_outputs::{self as outputs, Row, Vectors};
use std::collections::BTreeMap;

type Pair = (f64, f64);
type Key = (Vec<(String, i64)>, Vec<(usize, i32)>);

fn decimal(text: &str) -> PyResult<Decimal> {
    Decimal::parse(text).map_err(PyValueError::new_err)
}

fn exact_cells(matrix: &[Vec<(String, String)>]) -> PyResult<Vec<(Rational, Rational)>> {
    use rusticol_core::loop_v2::dress::parse_rational;
    matrix
        .iter()
        .flatten()
        .map(|(re, im)| {
            Ok((
                parse_rational(re).map_err(error)?,
                parse_rational(im).map_err(error)?,
            ))
        })
        .collect()
}

/// The binary64 carrier forms of an exact `size x size` matrix (rows of
/// `[re, im]` rational texts): a colour history's Hermitian parts (H, K;
/// K None for a Hermitian matrix) when `hermitian`, else the matrix itself
/// (`induced_observables.Forms`).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn induced_form(
    matrix: Vec<Vec<(String, String)>>,
    hermitian: bool,
) -> PyResult<(Vec<Vec<Pair>>, Option<Vec<Vec<Pair>>>)> {
    let _unlock = pyamplicol::unlock();
    let size = matrix.len();
    let (h, k) = outputs::binary64_form(&exact_cells(&matrix)?, size, hermitian).map_err(error)?;
    let rows = |cells: Vec<C<f64>>| -> Vec<Vec<Pair>> {
        cells
            .chunks(size.max(1))
            .map(|row| row.iter().map(|z| (z.re, z.im)).collect())
            .collect()
    };
    Ok((rows(h), k.map(rows)))
}

/// The binary64 carrier form of the amplitude-basis anchor `t` of the exact
/// colour metric `gram` (rows of `[re, im]` rational texts): the anchor's
/// metric (E_t G + G E_t)/2 (`induced_outputs::anchor_metric`, the SDK's)
/// in binary64 (`induced_observables.Forms` of an `("anchor", t)` metric).
#[pyfunction]
fn induced_anchor_form(gram: Vec<Vec<(String, String)>>, t: usize) -> PyResult<Vec<Vec<Pair>>> {
    let _unlock = pyamplicol::unlock();
    let size = gram.len();
    let cells = outputs::binary64_anchor_form(&exact_cells(&gram)?, size, t).map_err(error)?;
    Ok(cells
        .chunks(size.max(1))
        .map(|row| row.iter().map(|z| (z.re, z.im)).collect())
        .collect())
}

/// The binary64 spin coefficients c_h of vector leg `leg` (its label)
/// (`induced_observables.PointOutputs`): `sources` the tree program's
/// source records (JSON), `momentum` the leg's four momentum components and
/// `vector` the spin vector's four (re, im) components as exact decimal
/// texts, `working_digits` the carrier's. Per helicity (h, re, im).
#[pyfunction]
fn induced_spin_coefficients(
    sources: &str,
    leg: usize,
    momentum: Vec<String>,
    vector: Vec<(String, String)>,
    working_digits: u32,
) -> PyResult<Vec<(i32, f64, f64)>> {
    let _unlock = pyamplicol::unlock();
    let sources: Vec<serde_json::Value> =
        serde_json::from_str(sources).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if momentum.len() != 4 || vector.len() != 4 {
        return Err(PyValueError::new_err(
            "a momentum and a spin vector have four components",
        ));
    }
    let p = [
        decimal(&momentum[0])?,
        decimal(&momentum[1])?,
        decimal(&momentum[2])?,
        decimal(&momentum[3])?,
    ];
    let component = |k: usize| -> PyResult<(Decimal, Decimal)> {
        Ok((decimal(&vector[k].0)?, decimal(&vector[k].1)?))
    };
    let v = [component(0)?, component(1)?, component(2)?, component(3)?];
    let found = outputs::binary64_spin_coefficients(&sources, leg, &p, &v, working_digits)
        .map_err(|e| PyValueError::new_err(e.message().to_owned()))?;
    Ok(found.into_iter().map(|(h, z)| (h, z.re, z.im)).collect())
}

/// A(v) per (loop order, remaining row) of binary64 amplitude vectors
/// (`induced_observables.spin_amplitudes`): `vectors` in the Python dict's
/// order as ((order, row), [t][k] (re, im)); `coefficients` per 0-based spin
/// leg its (helicity, re, im). Returns the projected vectors in their
/// insertion order.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn induced_spin_amplitudes(
    vectors: Vec<(Key, Vec<Vec<Pair>>)>,
    coefficients: BTreeMap<usize, Vec<(i32, f64, f64)>>,
    width: usize,
    size: usize,
) -> PyResult<Vec<(Key, Vec<Vec<Pair>>)>> {
    let _unlock = pyamplicol::unlock();
    let mut amplitudes: Vectors<f64> = Vectors::default();
    for ((order, row), vector) in vectors {
        if vector.len() != size || vector.iter().any(|cell| cell.len() != width) {
            return Err(PyValueError::new_err("an amplitude vector is size x width"));
        }
        let row: Row = row;
        amplitudes.insert(
            (order, row),
            vector
                .into_iter()
                .map(|cell| cell.into_iter().map(|(re, im)| C::new(re, im)).collect())
                .collect(),
        );
    }
    let coefficients = coefficients
        .into_iter()
        .map(|(leg, values)| {
            (
                leg,
                values
                    .into_iter()
                    .map(|(h, re, im)| (h, C::new(re, im)))
                    .collect(),
            )
        })
        .collect();
    let moved = outputs::spin_amplitudes(&Binary64, &amplitudes, &coefficients, width, size);
    Ok(moved
        .iter()
        .map(|(key, vector)| {
            (
                key.clone(),
                vector
                    .iter()
                    .map(|cell| cell.iter().map(|z| (z.re, z.im)).collect())
                    .collect(),
            )
        })
        .collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(induced_form, module)?)?;
    module.add_function(wrap_pyfunction!(induced_anchor_form, module)?)?;
    module.add_function(wrap_pyfunction!(induced_spin_coefficients, module)?)?;
    module.add_function(wrap_pyfunction!(induced_spin_amplitudes, module)?)?;
    Ok(())
}
