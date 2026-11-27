// SPDX-License-Identifier: 0BSD

//! The binary64 stability estimate of the Python API (`loops/v2/stability`,
//! `plan_export.point_estimate`) through the one Rust implementation the
//! native SDKs run (`rusticol_loop_reduction::v2_stability`; m-lit LIT-SDK,
//! PLAN §3.5). The Python assembly stays as the oracle.

use super::*;
use rusticol_loop_reduction::v2_stability::{
    self as stability, Budget, CrossRun, Estimate, InducedPoint, InducedShape, LaneShape,
    PointLanes, Rotation,
};

/// A lane as Python passes it: (key, sector index, (re, im) per Laurent
/// coefficient).
type PyLane = (usize, usize, Vec<(f64, f64)>);
/// An evaluation's lanes and local components (per component, per sector,
/// the Laurent values).
type PyPoint = (Vec<PyLane>, Vec<Budget>);
/// The estimate's fields: relative, per coefficient, per check its relative
/// budget, E per sector, per check its absolute budget, S per sector, the
/// checks that did not run.
type PyEstimate = (
    f64,
    Vec<f64>,
    Vec<(String, f64)>,
    Budget,
    Vec<(String, Budget)>,
    Budget,
    Vec<(String, String)>,
);

fn point((lanes, local): PyPoint) -> PointLanes<usize> {
    PointLanes { lanes, local }
}

fn matrix(rotation: &Rotation) -> Vec<Vec<String>> {
    rotation
        .matrix()
        .iter()
        .map(|row| row.iter().map(|x| x.to_string()).collect())
        .collect()
}

/// The `index`-th default rotation check (`stability.default_rotation`) as
/// rows of exact rational texts (`Fraction` reads them).
#[pyfunction]
fn stability_default_rotation(index: u64) -> PyResult<Vec<Vec<String>>> {
    let _unlock = pyamplicol::unlock();
    Ok(matrix(&stability::default_rotation(index).map_err(error)?))
}

/// One rotation check from its text (`virtual_results.parse_rotation`) as
/// rows of exact rational texts.
#[pyfunction]
fn stability_rotation(specification: &str) -> PyResult<Vec<Vec<String>>> {
    let _unlock = pyamplicol::unlock();
    Ok(matrix(&Rotation::parse(specification).map_err(error)?))
}

/// The binary64 stability estimate of one Born-interfered point
/// (`plan_export.point_estimate`): `shape` (sectors, Laurent width, the
/// slice's average), the unit roundoff, the reference Laurent values per
/// sector, the point's lanes and local components, those of each rotated
/// evaluation, the reduction check's (name, re-test lanes), and the
/// cross-backend check's evaluation or the reason it did not run.
#[pyfunction]
#[pyo3(signature = (shape, roundoff, reference, direct, rotated, retest=None, cross=None, cross_reason=None))]
#[allow(clippy::too_many_arguments)]
fn stability_born_interfered(
    shape: (usize, usize, f64),
    roundoff: f64,
    reference: Budget,
    direct: PyPoint,
    rotated: Vec<PyPoint>,
    retest: Option<(String, Vec<PyLane>)>,
    cross: Option<PyPoint>,
    cross_reason: Option<String>,
) -> PyResult<PyEstimate> {
    let _unlock = pyamplicol::unlock();
    let (sectors, width, average) = shape;
    let shape = LaneShape {
        sectors,
        width,
        factor: stability::lane_factor(average),
    };
    let direct = point(direct);
    let rotated: Vec<_> = rotated.into_iter().map(point).collect();
    let cross = match (cross, cross_reason) {
        (Some(run), None) => Some(CrossRun::Run(point(run))),
        (None, Some(why)) => Some(CrossRun::NotRun(why)),
        (None, None) => None,
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "the cross-backend check either ran or did not",
            ));
        }
    };
    let estimate: Estimate = stability::born_interfered_estimate(
        shape,
        roundoff,
        &reference,
        &direct,
        &rotated,
        retest
            .as_ref()
            .map(|(name, lanes)| (name.as_str(), lanes.as_slice())),
        cross.as_ref(),
    )
    .map_err(error)?;
    Ok(fields(estimate))
}

/// A loop-induced evaluation as Python passes it (`InducedPoint`): per key
/// ((loop order id, row id), insertion order) the amplitude vectors, the bare
/// vectors and the counterterm kets ((re, im) per t and Laurent
/// coefficient), the re-test budget and the term scale (absolute values per
/// t and coefficient; None when not formed).
type PyCells = Vec<((usize, usize), Vec<Vec<(f64, f64)>>)>;
type PyErrors = Vec<((usize, usize), Vec<Vec<f64>>)>;
type PyInduced = (
    PyCells,
    PyCells,
    PyCells,
    Option<PyErrors>,
    Option<PyErrors>,
);

fn induced_point((amplitudes, bare, kets, budget, terms): PyInduced) -> InducedPoint {
    InducedPoint {
        amplitudes,
        bare,
        kets,
        budget,
        terms,
    }
}

fn fields(estimate: Estimate) -> PyEstimate {
    (
        estimate.relative,
        estimate.coefficients,
        estimate.checks,
        estimate.absolute,
        estimate.budgets,
        estimate.scale,
        estimate.not_run,
    )
}

/// The binary64 stability estimate of one loop-induced point
/// (`plan_export.point_estimate` of an amplitude-mode point): `shape` (basis
/// size, Laurent width, the colour metric's cells row-major, the plan's row
/// ids, per record sector its sorted (loop order id, loop order id) pairs,
/// the average), the unit roundoff, |A|^2 per sector, the point's
/// evaluation, each rotated evaluation's, the reduction check's name, and
/// the cross-backend check's evaluation or the reason it did not run.
#[pyfunction]
#[pyo3(signature = (shape, roundoff, reference, direct, rotated, reduction=None, cross=None, cross_reason=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn stability_loop_induced(
    shape: (
        usize,
        usize,
        Vec<(f64, f64)>,
        Vec<usize>,
        Vec<Vec<(usize, usize)>>,
        f64,
    ),
    roundoff: f64,
    reference: Budget,
    direct: PyInduced,
    rotated: Vec<PyInduced>,
    reduction: Option<String>,
    cross: Option<PyInduced>,
    cross_reason: Option<String>,
) -> PyResult<PyEstimate> {
    let _unlock = pyamplicol::unlock();
    let (size, width, gram, rows, sectors, average) = shape;
    let shape = InducedShape {
        size,
        width,
        gram,
        rows,
        sectors,
        average,
    };
    let cross = match (cross, cross_reason) {
        (Some(run), None) => Some(CrossRun::Run(induced_point(run))),
        (None, Some(why)) => Some(CrossRun::NotRun(why)),
        (None, None) => None,
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "the cross-backend check either ran or did not",
            ));
        }
    };
    let rotated: Vec<_> = rotated.into_iter().map(induced_point).collect();
    stability::induced_estimate(
        &shape,
        roundoff,
        &reference,
        &induced_point(direct),
        &rotated,
        reduction.as_deref(),
        cross.as_ref(),
    )
    .map(fields)
    .map_err(error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(stability_loop_induced, module)?)?;
    module.add_function(wrap_pyfunction!(stability_default_rotation, module)?)?;
    module.add_function(wrap_pyfunction!(stability_rotation, module)?)?;
    module.add_function(wrap_pyfunction!(stability_born_interfered, module)?)?;
    Ok(())
}
