// SPDX-License-Identifier: 0BSD

use super::*;

impl ExecutionRuntime {
    pub(super) fn fill_sources_row(
        sources: &[GenericSourceRecordManifest],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<f64>],
        point: &[[f64; 4]],
    ) -> RusticolResult<()> {
        for source in sources {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            Self::write_source_wavefunction(
                source,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_sources_row_generic<T>(
        sources: &[GenericSourceRecordManifest],
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        row: &mut [Complex<T>],
        point: &[[T; 4]],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        for source in sources {
            let start = source.value_slot.component_start;
            let stop = source.value_slot.component_stop;
            if stop > row.len() || stop < start {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} has an invalid value-slot range",
                    source.source_id
                )));
            }
            Self::write_source_wavefunction_generic(
                source,
                external_count,
                particle_masses,
                point,
                &mut row[start..stop],
            )?;
        }
        Ok(())
    }

    pub(super) fn fill_momenta_row(
        momentum_slots: &[GenericMomentumSlotManifest],
        value_parameter_count: usize,
        external_count: usize,
        external_is_initial: &[bool],
        row: &mut [Complex<f64>],
        point: &[[f64; 4]],
    ) -> RusticolResult<()> {
        for slot in momentum_slots {
            let start = value_parameter_count + slot.component_start;
            let stop = value_parameter_count + slot.component_stop;
            if stop > row.len() || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum = [0.0; 4];
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= external_count || index >= external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                let sign = if external_is_initial[index] {
                    -1.0
                } else {
                    1.0
                };
                for (momentum_component, point_component) in momentum.iter_mut().zip(&point[index])
                {
                    *momentum_component += sign * point_component;
                }
            }
            for (output, component) in row[start..stop].iter_mut().zip(momentum) {
                *output = c64(component, 0.0);
            }
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_momenta_row_generic<T>(
        momentum_slots: &[GenericMomentumSlotManifest],
        value_parameter_count: usize,
        external_count: usize,
        external_is_initial: &[bool],
        row: &mut [Complex<T>],
        point: &[[T; 4]],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        for slot in momentum_slots {
            let start = value_parameter_count + slot.component_start;
            let stop = value_parameter_count + slot.component_stop;
            if stop > row.len() || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum: [T; 4] = std::array::from_fn(|_| T::new_zero());
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= external_count || index >= external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                for (momentum_component, point_component) in momentum.iter_mut().zip(&point[index])
                {
                    if external_is_initial[index] {
                        *momentum_component -= point_component.clone();
                    } else {
                        *momentum_component += point_component.clone();
                    }
                }
            }
            for (output, component) in row[start..stop].iter_mut().zip(momentum) {
                *output = c_generic(component, T::new_zero());
            }
        }
        Ok(())
    }

    pub(super) fn fill_model_parameters_row(
        model_parameter_start: usize,
        model_parameter_values: &[f64],
        row: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        let stop = model_parameter_start + model_parameter_values.len();
        if stop > row.len() {
            return Err(RusticolError::invalid_argument(
                "generic model-parameter block exceeds runtime row length",
            ));
        }
        for (index, value) in model_parameter_values.iter().enumerate() {
            row[model_parameter_start + index] = c64(*value, 0.0);
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn fill_model_parameters_row_generic<T>(
        model_parameter_start: usize,
        model_parameter_values: &[f64],
        row: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let stop = model_parameter_start + model_parameter_values.len();
        if stop > row.len() {
            return Err(RusticolError::invalid_argument(
                "generic model-parameter block exceeds runtime row length",
            ));
        }
        for (index, value) in model_parameter_values.iter().enumerate() {
            row[model_parameter_start + index] = c_generic(T::from(*value), T::new_zero());
        }
        Ok(())
    }

    pub(super) fn write_source_wavefunction(
        source: &GenericSourceRecordManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &[[f64; 4]],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if source.source_kind != "external-wavefunction" {
            return Err(RusticolError::invalid_argument(format!(
                "generic source kind {:?} is not implemented",
                source.source_kind
            )));
        }
        let index = source.leg_label.checked_sub(1).ok_or_else(|| {
            RusticolError::invalid_argument("generic source leg labels are one-based")
        })?;
        if index >= external_count {
            return Err(RusticolError::invalid_argument(format!(
                "generic source {} refers to unknown external label {}",
                source.source_id, source.leg_label
            )));
        }
        let momentum = if source.crossing == "negate-incoming-momentum" {
            negate(point[index])
        } else {
            point[index]
        };
        if source.dimension == 1 {
            if out.len() != 1 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 1 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            out[0] = c64(1.0, 0.0);
            return Ok(());
        }
        if source.dimension == 2 {
            if out.len() != 2 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 2 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let chirality = source.chirality;
            let wave = if source.particle_id < 0 {
                ext_antiquark_weyl_array(momentum, source.source_helicity, chirality)
            } else {
                ext_quark_weyl_array(momentum, source.source_helicity, chirality)
            };
            out.copy_from_slice(&wave);
            return Ok(());
        }
        if source.dimension == 4 {
            if out.len() != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 4 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = if source.wavefunction_kind == "fermion"
                || (source.wavefunction_kind.is_empty() && is_fermion_pdg(source.particle_id))
            {
                let mass = particle_mass_from_map(particle_masses, source.particle_id);
                if source.particle_id < 0 {
                    ext_antiquark_dirac_massive(momentum, source.source_helicity, mass)
                } else {
                    ext_quark_dirac_massive(momentum, source.source_helicity, mass)
                }
            } else if (source.wavefunction_kind == "vector"
                && particle_mass_from_map(particle_masses, source.particle_id) == 0.0)
                || (source.wavefunction_kind.is_empty()
                    && (source.particle_id.abs() == 21 || source.particle_id == 22))
            {
                ext_gluon(momentum, source.source_helicity)
            } else {
                ext_massive_vector(
                    momentum,
                    source.source_helicity,
                    particle_mass_from_map(particle_masses, source.particle_id),
                )
            };
            out.copy_from_slice(&wave);
            return Ok(());
        }
        if source.dimension == 16 && source.wavefunction_kind == "spin2" {
            if out.len() != 16 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 16 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = ext_spin2(
                momentum,
                source.source_helicity,
                particle_mass_from_map(particle_masses, source.particle_id),
            )?;
            out.copy_from_slice(&wave);
            return Ok(());
        }
        Err(RusticolError::invalid_argument(format!(
            "generic source dimension {} is not implemented",
            source.dimension
        )))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn write_source_wavefunction_generic<T>(
        source: &GenericSourceRecordManifest,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        point: &[[T; 4]],
        out: &mut [Complex<T>],
    ) -> RusticolResult<()>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if source.source_kind != "external-wavefunction" {
            return Err(RusticolError::invalid_argument(format!(
                "generic source kind {:?} is not implemented",
                source.source_kind
            )));
        }
        let index = source.leg_label.checked_sub(1).ok_or_else(|| {
            RusticolError::invalid_argument("generic source leg labels are one-based")
        })?;
        if index >= external_count {
            return Err(RusticolError::invalid_argument(format!(
                "generic source {} refers to unknown external label {}",
                source.source_id, source.leg_label
            )));
        }
        let momentum = if source.crossing == "negate-incoming-momentum" {
            negate_generic(&point[index])
        } else {
            point[index].clone()
        };
        if source.dimension == 1 {
            if out.len() != 1 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 1 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            out[0] = c_generic(T::from(1.0), T::new_zero());
            return Ok(());
        }
        if source.dimension == 2 {
            if out.len() != 2 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 2 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let chirality = source.chirality;
            let wave = if source.particle_id < 0 {
                ext_antiquark_weyl_generic(&momentum, source.source_helicity, chirality)
            } else {
                ext_quark_weyl_generic(&momentum, source.source_helicity, chirality)
            };
            out.clone_from_slice(&wave);
            return Ok(());
        }
        if source.dimension == 4 {
            if out.len() != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 4 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = if source.wavefunction_kind == "fermion"
                || (source.wavefunction_kind.is_empty() && is_fermion_pdg(source.particle_id))
            {
                let mass = particle_mass_from_map(particle_masses, source.particle_id);
                if mass != 0.0 {
                    return Err(RusticolError::invalid_argument(
                        "high-precision generic massive fermion sources are not implemented",
                    ));
                }
                if source.particle_id < 0 {
                    ext_antiquark_dirac_generic(&momentum, source.source_helicity)
                } else {
                    ext_quark_dirac_generic(&momentum, source.source_helicity)
                }
            } else if (source.wavefunction_kind == "vector"
                && particle_mass_from_map(particle_masses, source.particle_id) == 0.0)
                || (source.wavefunction_kind.is_empty()
                    && (source.particle_id.abs() == 21 || source.particle_id == 22))
            {
                ext_gluon_generic(&momentum, source.source_helicity)
            } else {
                ext_massive_vector_generic(
                    &momentum,
                    source.source_helicity,
                    T::from(particle_mass_from_map(particle_masses, source.particle_id)),
                )
            };
            out.clone_from_slice(&wave);
            return Ok(());
        }
        if source.dimension == 16 && source.wavefunction_kind == "spin2" {
            if out.len() != 16 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic source {} expected dimension 16 but slot has length {}",
                    source.source_id,
                    out.len()
                )));
            }
            let wave = ext_spin2_generic(
                &momentum,
                source.source_helicity,
                T::from(particle_mass_from_map(particle_masses, source.particle_id)),
            )?;
            out.clone_from_slice(&wave);
            return Ok(());
        }
        Err(RusticolError::invalid_argument(format!(
            "generic source dimension {} is not implemented",
            source.dimension
        )))
    }
}

fn particle_mass_from_map(particle_masses: &BTreeMap<i32, f64>, particle_id: i32) -> f64 {
    particle_masses
        .get(&particle_id)
        .copied()
        .or_else(|| particle_masses.get(&(-particle_id)).copied())
        .unwrap_or(0.0)
}

fn is_fermion_pdg(particle_id: i32) -> bool {
    let abs_id = particle_id.abs();
    (1..=6).contains(&abs_id) || (11..=16).contains(&abs_id)
}
