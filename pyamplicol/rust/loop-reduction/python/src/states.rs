// SPDX-License-Identifier: 0BSD

//! Native blocks of existing precision-generic FeynKit external states.

use super::*;
use feynkit_kinematics::{
    FourMomentum as PhysicalMomentum, Helicity, KinematicScalar, ThreeMomentum, WavefunctionKind,
};

fn specifications(
    rows: usize,
    helicities: &[i32],
    kinds: &[String],
    components: Option<usize>,
) -> PyResult<(Vec<(Helicity, WavefunctionKind)>, usize)> {
    if helicities.len() != rows || kinds.len() != rows {
        return Err(error("external state row metadata dimensions differ"));
    }
    let specifications = helicities
        .iter()
        .zip(kinds)
        .map(|(&helicity, kind)| {
            Ok((
                Helicity::try_from(i8::try_from(helicity).map_err(error)?).map_err(error)?,
                kind.parse::<WavefunctionKind>().map_err(error)?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let columns = components.unwrap_or_else(|| {
        specifications
            .first()
            .map_or(4, |(_, kind)| kind.component_count())
    });
    if !matches!(columns, 1 | 4)
        || specifications
            .iter()
            .any(|(_, kind)| kind.component_count() != columns)
    {
        return Err(error(
            "external state batch must have a common component count of one or four",
        ));
    }
    Ok((specifications, columns))
}

fn real_finite<T: Carrier>(value: &Complex<T>) -> bool {
    value.im.is_fully_zero() && value.re.is_finite()
}

fn state_components<T: BatchCarrier + KinematicScalar>(
    p: &[Complex<T>],
    helicity: Helicity,
    kind: WavefunctionKind,
) -> Result<Vec<Complex<T>>, String> {
    if !p.iter().all(real_finite) {
        return Err("momenta must be finite and real".into());
    }
    PhysicalMomentum::from(std::array::from_fn::<_, 4, _>(|i| p[i].re.clone()))
        .wavefunction(kind, helicity)
        .map(|wavefunction| wavefunction.into_components())
        .map_err(|failure| failure.to_string())
}

fn evaluate<T: BatchCarrier + KinematicScalar>(
    values: &[Complex<T>],
    input: &NativeBatch,
    specifications: &[(Helicity, WavefunctionKind)],
    columns: usize,
) -> PyResult<NativeBatch> {
    let mut output = Vec::with_capacity(input.rows * columns);
    for (row, (p, &(helicity, kind))) in values.chunks_exact(4).zip(specifications).enumerate() {
        let lane_error = |failure| {
            error(format!(
                "external state lane {} (row {row}): {failure}",
                input.ids[row]
            ))
        };
        output.extend(state_components(p, helicity, kind).map_err(lane_error)?);
    }
    NativeBatch::typed(output, input.ids.clone(), columns, input.precision_bits)
}

type PartialStates = (NativeBatch, Vec<(u64, String)>);

fn evaluate_partial<T: BatchCarrier + KinematicScalar>(
    values: &[Complex<T>],
    input: &NativeBatch,
    specifications: &[(Helicity, WavefunctionKind)],
    columns: usize,
) -> PyResult<PartialStates> {
    let mut output = Vec::with_capacity(input.rows * columns);
    let mut ids = Vec::with_capacity(input.rows);
    let mut statuses = Vec::with_capacity(input.rows);
    for (p, (&id, &(helicity, kind))) in values
        .chunks_exact(4)
        .zip(input.ids.iter().zip(specifications))
    {
        match state_components(p, helicity, kind) {
            Ok(components) => {
                output.extend(components);
                ids.push(id);
                statuses.push((id, "success".into()));
            }
            Err(failure) => statuses.push((id, failure)),
        }
    }
    Ok((
        NativeBatch::typed(output, ids, columns, input.precision_bits)?,
        statuses,
    ))
}

fn on_shell<T: BatchCarrier + KinematicScalar>(
    spatial: &[Complex<T>],
    masses: &[Complex<T>],
    input: &NativeBatch,
    specifications: &[(Helicity, WavefunctionKind)],
    columns: usize,
) -> PyResult<(NativeBatch, NativeBatch)> {
    let mut momenta = Vec::with_capacity(input.rows * 4);
    for (row, (p, mass)) in spatial.chunks_exact(3).zip(masses).enumerate() {
        if !p.iter().all(real_finite) || !real_finite(mass) || mass.re < mass.re.zero() {
            return Err(error(format!(
                "on-shell construction lane {} (row {row}) requires finite real spatial momentum and nonnegative real mass",
                input.ids[row]
            )));
        }
        let momentum = ThreeMomentum::from(std::array::from_fn::<_, 3, _>(|i| p[i].re.clone()))
            .on_shell(Some(&mass.re));
        momenta.extend(
            momentum
                .into_iter()
                .map(|value| Complex::new(value.clone(), value.zero())),
        );
    }
    let states = evaluate(&momenta, input, specifications, columns)?;
    Ok((
        NativeBatch::typed(momenta, input.ids.clone(), 4, input.precision_bits)?,
        states,
    ))
}

#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, components=None))]
fn wavefunctions_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<NativeBatch> {
    if momenta.columns != 4 {
        return Err(error("external states require four momentum columns"));
    }
    let (specifications, columns) = specifications(momenta.rows, &helicities, &kinds, components)?;
    py.detach(|| match &momenta.data {
        Data::Double(v) => evaluate(v, momenta, &specifications, columns),
        Data::DoubleDouble(v) => evaluate(v, momenta, &specifications, columns),
        Data::Arbitrary(v) => evaluate(v, momenta, &specifications, columns),
    })
}

/// Numerical failures remain sparse native rows; malformed metadata still raises.
#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, components=None))]
fn wavefunctions_partial_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<PartialStates> {
    if momenta.columns != 4 {
        return Err(error("external states require four momentum columns"));
    }
    let (specifications, columns) = specifications(momenta.rows, &helicities, &kinds, components)?;
    if specifications.iter().any(|(helicity, kind)| {
        (kind.is_spinor() && helicity.is_zero())
            || (*kind == WavefunctionKind::Scalar && !helicity.is_zero())
    }) {
        return Err(error(
            "external state kind does not admit the requested helicity",
        ));
    }
    py.detach(|| match &momenta.data {
        Data::Double(v) => evaluate_partial(v, momenta, &specifications, columns),
        Data::DoubleDouble(v) => evaluate_partial(v, momenta, &specifications, columns),
        Data::Arbitrary(v) => evaluate_partial(v, momenta, &specifications, columns),
    })
}

#[pyfunction]
#[pyo3(signature = (spatial_momenta, masses, helicities, kinds, components=None))]
fn on_shell_wavefunctions_batch(
    py: Python<'_>,
    spatial_momenta: &NativeBatch,
    masses: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<(NativeBatch, NativeBatch)> {
    spatial_momenta.same_rows(masses)?;
    if spatial_momenta.columns != 3 || masses.columns != 1 {
        return Err(error(
            "on-shell construction requires three spatial columns and one mass column",
        ));
    }
    let (specifications, columns) =
        specifications(spatial_momenta.rows, &helicities, &kinds, components)?;
    py.detach(|| match (&spatial_momenta.data, &masses.data) {
        (Data::Double(p), Data::Double(m)) => {
            on_shell(p, m, spatial_momenta, &specifications, columns)
        }
        (Data::DoubleDouble(p), Data::DoubleDouble(m)) => {
            on_shell(p, m, spatial_momenta, &specifications, columns)
        }
        (Data::Arbitrary(p), Data::Arbitrary(m)) => {
            on_shell(p, m, spatial_momenta, &specifications, columns)
        }
        _ => unreachable!("same_rows checked carriers"),
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(wavefunctions_batch, module)?)?;
    module.add_function(wrap_pyfunction!(wavefunctions_partial_batch, module)?)?;
    module.add_function(wrap_pyfunction!(on_shell_wavefunctions_batch, module)?)?;
    Ok(())
}
