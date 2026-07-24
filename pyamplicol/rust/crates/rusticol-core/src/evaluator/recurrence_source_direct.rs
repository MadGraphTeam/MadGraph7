// SPDX-License-Identifier: 0BSD

//! Model-generic SourceIR execution for Direct-Arena recurrence plans.
//!
//! The loader resolves SourceIR crossing semantics once into immutable
//! dispatch domains. Momentum crossing is already represented by each direct
//! row's momentum form, and its exact factor owns the crossing phase. The hot
//! call therefore reads one crossed momentum, evaluates the established
//! Rusticol source formula, multiplies the row factor, and writes directly to
//! persistent split-complex current planes.

#[cfg(not(feature = "symbolica-runtime"))]
use num_complex::Complex;
#[cfg(feature = "symbolica-runtime")]
use symbolica::prelude::{Complex, Real, RealLike};

use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    DirectResolvedSourceSelection, DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
    DirectSourceExecutor, DirectUnionSourceDispatchHandle,
};
use crate::recurrence::{DIRECT_NONE_U32, DirectSourceRow};
use crate::{RusticolError, RusticolResult};
use std::ffi::{c_int, c_void};
use std::ptr;

// Recompile the established generic source formulas in this evaluator module
// instead of maintaining a second set of physics expressions.
#[path = "../wavefunctions.rs"]
mod source_wavefunctions;

const STATUS_INVALID_CONTEXT: c_int = 1;
const STATUS_INVALID_ARGUMENT: c_int = 2;
const STATUS_EXECUTION_FAILED: c_int = 3;
const STATUS_BOUNDS: c_int = 4;

/// Source families currently supported by the canonical SourceIR executor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectSourceWavefunctionFamily {
    Scalar,
    WeylFermion,
    DiracFermion,
    Vector,
    Spin2,
}

impl DirectSourceWavefunctionFamily {
    const fn component_count(self) -> u16 {
        match self {
            Self::Scalar => 1,
            Self::WeylFermion => 2,
            Self::DiracFermion | Self::Vector => 4,
            Self::Spin2 => 16,
        }
    }
}

/// Particle orientation needed by fermion source formulas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectSourceOrientation {
    Particle,
    Antiparticle,
    SelfConjugate,
}

/// One crossed SourceIR state ready for direct execution.
///
/// `helicity` and `chirality` are the effective values after SourceIR
/// crossing. `mass_parameter_index` addresses the prepared parameter view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectSourceTemplateSpec {
    pub(crate) spin_state_class: i32,
    pub(crate) family: DirectSourceWavefunctionFamily,
    pub(crate) orientation: DirectSourceOrientation,
    pub(crate) helicity: i32,
    pub(crate) chirality: i32,
    pub(crate) mass_parameter_index: Option<u32>,
}

impl DirectSourceTemplateSpec {
    fn validate(self) -> RusticolResult<()> {
        if matches!(
            self.family,
            DirectSourceWavefunctionFamily::WeylFermion
                | DirectSourceWavefunctionFamily::DiracFermion
        ) && self.orientation == DirectSourceOrientation::SelfConjugate
        {
            return Err(RusticolError::invalid_argument(
                "direct recurrence SourceIR fermion cannot use self-conjugate orientation",
            ));
        }
        match self.family {
            DirectSourceWavefunctionFamily::Scalar if self.helicity != 0 => {
                return Err(RusticolError::invalid_argument(
                    "direct recurrence scalar source helicity must be zero",
                ));
            }
            DirectSourceWavefunctionFamily::WeylFermion
            | DirectSourceWavefunctionFamily::DiracFermion
                if !matches!(self.helicity, -1 | 1) =>
            {
                return Err(RusticolError::invalid_argument(
                    "direct recurrence fermion source helicity must be -1 or +1",
                ));
            }
            DirectSourceWavefunctionFamily::WeylFermion if !matches!(self.chirality, -1 | 1) => {
                return Err(RusticolError::invalid_argument(
                    "direct recurrence Weyl source chirality must be -1 or +1",
                ));
            }
            DirectSourceWavefunctionFamily::Vector if !matches!(self.helicity, -1 | 0 | 1) => {
                return Err(RusticolError::invalid_argument(
                    "direct recurrence vector source helicity must be -1, 0, or +1",
                ));
            }
            DirectSourceWavefunctionFamily::Spin2
                if !matches!(self.helicity, -2 | -1 | 0 | 1 | 2) =>
            {
                return Err(RusticolError::invalid_argument(
                    "direct recurrence spin-2 source helicity must be between -2 and +2",
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

/// One row-addressable source-template or runtime-dispatch domain.
#[derive(Clone, Debug)]
pub(crate) struct DirectSourceDispatchDomainSpec {
    pub(crate) variants: Vec<DirectSourceTemplateSpec>,
}

struct DirectSourceDispatchDomain {
    variants: Box<[DirectSourceTemplateSpec]>,
}

impl DirectSourceDispatchDomain {
    fn resolve(&self, spin_state_class: i32) -> Option<DirectSourceTemplateSpec> {
        self.variants
            .binary_search_by_key(&spin_state_class, |variant| variant.spin_state_class)
            .ok()
            .map(|index| self.variants[index])
    }
}

struct DirectSourceExecutorContext {
    domains: Box<[DirectSourceDispatchDomain]>,
}

/// Context-aware source handle published into the Direct-Arena catalog.
#[derive(Clone, Copy)]
pub(crate) struct ContextDirectSourceExecutorHandle {
    pub(crate) call: DirectSourceExecutor,
    pub(crate) context: *const c_void,
}

/// Owns the immutable context addressed by a source handle.
pub(crate) struct LoadedDirectSourceExecutor {
    context: Box<DirectSourceExecutorContext>,
}

impl LoadedDirectSourceExecutor {
    /// Build and authenticate source dispatch domains at load time.
    ///
    /// Domain indices must match `DirectSourceRow::
    /// source_template_or_dispatch_domain`. Singleton domains represent fixed
    /// topology-replay templates; multi-variant domains support runtime
    /// helicity dispatch without changing the hot-call ABI.
    pub(crate) fn load(mut domains: Vec<DirectSourceDispatchDomainSpec>) -> RusticolResult<Self> {
        if domains.is_empty() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence source catalog must contain at least one dispatch domain",
            ));
        }
        let mut loaded = Vec::with_capacity(domains.len());
        for (domain_index, domain) in domains.iter_mut().enumerate() {
            if domain.variants.is_empty() {
                return Err(RusticolError::invalid_argument(format!(
                    "direct recurrence source dispatch domain {domain_index} is empty"
                )));
            }
            domain
                .variants
                .sort_unstable_by_key(|variant| variant.spin_state_class);
            for variant in &domain.variants {
                variant.validate()?;
            }
            if domain
                .variants
                .windows(2)
                .any(|pair| pair[0].spin_state_class == pair[1].spin_state_class)
            {
                return Err(RusticolError::invalid_argument(format!(
                    "direct recurrence source dispatch domain {domain_index} repeats a spin-state class"
                )));
            }
            loaded.push(DirectSourceDispatchDomain {
                variants: std::mem::take(&mut domain.variants).into_boxed_slice(),
            });
        }
        Ok(Self {
            context: Box::new(DirectSourceExecutorContext {
                domains: loaded.into_boxed_slice(),
            }),
        })
    }

    pub(crate) fn handle(&self) -> ContextDirectSourceExecutorHandle {
        ContextDirectSourceExecutorHandle {
            call: execute_direct_source_rows,
            context: ptr::from_ref(self.context.as_ref()).cast(),
        }
    }

    /// Return the all-flow-union source-dispatch handle backed by the same
    /// immutable SourceIR domain catalog.
    pub(crate) fn union_handle(&self) -> DirectUnionSourceDispatchHandle {
        DirectUnionSourceDispatchHandle {
            call: execute_direct_union_source_rows,
            context: ptr::from_ref(self.context.as_ref()).cast(),
        }
    }
}

unsafe extern "C" fn execute_direct_source_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectSourceRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if context.is_null() {
        return STATUS_INVALID_CONTEXT;
    }
    if rows.is_null()
        || row_count == 0
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
        || momenta.point_stride != arena.point_stride
        || momenta.lorentz_component_count != 4
    {
        return STATUS_INVALID_ARGUMENT;
    }
    if arena.current_re.is_null()
        || arena.current_im.is_null()
        || momenta.values.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let context = unsafe { &*context.cast::<DirectSourceExecutorContext>() };
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        let Some(domain) = context
            .domains
            .get(row.source_template_or_dispatch_domain as usize)
        else {
            return STATUS_BOUNDS;
        };
        let Some(template) = domain.resolve(row.spin_state_class) else {
            return STATUS_BOUNDS;
        };
        if row.momentum_form_id >= momenta.form_count || row.exact_factor_id >= factors.value_count
        {
            return STATUS_BOUNDS;
        }
        let Some(destination_end) = u64::from(row.destination_component_base)
            .checked_add(u64::from(template.family.component_count()))
            .and_then(|planes| planes.checked_mul(u64::from(arena.point_stride)))
        else {
            return STATUS_BOUNDS;
        };
        if destination_end > arena.current_scalar_len {
            return STATUS_BOUNDS;
        }
        let Some(momentum_end) = u64::from(row.momentum_form_id)
            .checked_mul(4)
            .and_then(|base| base.checked_add(4))
            .and_then(|planes| planes.checked_mul(u64::from(momenta.point_stride)))
        else {
            return STATUS_BOUNDS;
        };
        if momentum_end > momenta.scalar_len {
            return STATUS_BOUNDS;
        }
        let mass = match template.mass_parameter_index {
            Some(index) => {
                if index >= parameters.value_count || parameters.values_re.is_null() {
                    return STATUS_BOUNDS;
                }
                unsafe { *parameters.values_re.add(index as usize) }
            }
            None => 0.0,
        };
        let factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };

        for point in 0..point_count {
            let momentum = read_momentum(momenta, row.momentum_form_id, point);
            let status = match template.family {
                DirectSourceWavefunctionFamily::Scalar => write_wavefunction(
                    arena,
                    row.destination_component_base,
                    point,
                    factor_re,
                    factor_im,
                    [Complex::new(1.0, 0.0)],
                ),
                DirectSourceWavefunctionFamily::WeylFermion => {
                    let wave = match template.orientation {
                        DirectSourceOrientation::Particle => {
                            source_wavefunctions::ext_quark_weyl_array(
                                momentum,
                                template.helicity,
                                template.chirality,
                            )
                        }
                        DirectSourceOrientation::Antiparticle => {
                            source_wavefunctions::ext_antiquark_weyl_array(
                                momentum,
                                template.helicity,
                                template.chirality,
                            )
                        }
                        DirectSourceOrientation::SelfConjugate => {
                            return STATUS_EXECUTION_FAILED;
                        }
                    };
                    write_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        factor_re,
                        factor_im,
                        wave,
                    )
                }
                DirectSourceWavefunctionFamily::DiracFermion => {
                    let wave = match template.orientation {
                        DirectSourceOrientation::Particle => {
                            source_wavefunctions::ext_quark_dirac_massive(
                                momentum,
                                template.helicity,
                                mass,
                            )
                        }
                        DirectSourceOrientation::Antiparticle => {
                            source_wavefunctions::ext_antiquark_dirac_massive(
                                momentum,
                                template.helicity,
                                mass,
                            )
                        }
                        DirectSourceOrientation::SelfConjugate => {
                            return STATUS_EXECUTION_FAILED;
                        }
                    };
                    write_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        factor_re,
                        factor_im,
                        wave,
                    )
                }
                DirectSourceWavefunctionFamily::Vector => {
                    let wave = if mass == 0.0 {
                        source_wavefunctions::ext_gluon(momentum, template.helicity)
                    } else {
                        source_wavefunctions::ext_massive_vector(momentum, template.helicity, mass)
                    };
                    write_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        factor_re,
                        factor_im,
                        wave,
                    )
                }
                DirectSourceWavefunctionFamily::Spin2 => {
                    let Ok(wave) =
                        source_wavefunctions::ext_spin2(momentum, template.helicity, mass)
                    else {
                        return STATUS_EXECUTION_FAILED;
                    };
                    write_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        factor_re,
                        factor_im,
                        wave,
                    )
                }
            };
            if status != DIRECT_STATUS_OK {
                return status;
            }
        }
    }
    DIRECT_STATUS_OK
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn execute_direct_union_source_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectSourceRow,
    row_count: u32,
    variants: *const DirectSourceDispatchVariantDescriptor,
    variant_count: u32,
    embeddings: *const DirectSourceEmbeddingRow,
    embedding_count: u32,
    selections: *const DirectResolvedSourceSelection,
    selection_count: u32,
    point_count: u32,
) -> c_int {
    if context.is_null() {
        return STATUS_INVALID_CONTEXT;
    }
    if rows.is_null()
        || variants.is_null()
        || embeddings.is_null()
        || selections.is_null()
        || row_count == 0
        || variant_count == 0
        || embedding_count == 0
        || selection_count == 0
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
        || momenta.point_stride != arena.point_stride
        || momenta.lorentz_component_count != 4
    {
        return STATUS_INVALID_ARGUMENT;
    }
    if arena.current_re.is_null()
        || arena.current_im.is_null()
        || momenta.values.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let context = unsafe { &*context.cast::<DirectSourceExecutorContext>() };
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    let variants = unsafe { std::slice::from_raw_parts(variants, variant_count as usize) };
    let embeddings = unsafe { std::slice::from_raw_parts(embeddings, embedding_count as usize) };
    let selections = unsafe { std::slice::from_raw_parts(selections, selection_count as usize) };

    for (source_slot, selection) in selections.iter().enumerate() {
        if selection.source_slot != source_slot as u32 {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(variant) = variants.get(selection.dispatch_variant_id as usize) else {
            return STATUS_BOUNDS;
        };
        let Some(row) = rows.get(variant.source_row_id as usize) else {
            return STATUS_BOUNDS;
        };
        if row.source_slot != selection.source_slot
            || row.source_template_or_dispatch_domain != variant.dispatch_domain_id
            || row.momentum_form_id >= momenta.form_count
            || row.exact_factor_id >= factors.value_count
            || variant.crossing_exact_factor_id >= factors.value_count
        {
            return STATUS_BOUNDS;
        }
        // Union source rows are only structural destinations. Crossing and
        // embedding own the complete source factorization.
        let row_factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let row_factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        if row_factor_re != 1.0 || row_factor_im != 0.0 {
            return STATUS_EXECUTION_FAILED;
        }

        let Some(domain) = context.domains.get(variant.dispatch_domain_id as usize) else {
            return STATUS_BOUNDS;
        };
        let Some(template) = domain.resolve(variant.crossed_spin_state_class) else {
            return STATUS_BOUNDS;
        };
        if u32::from(template.family.component_count()) != variant.projection_count {
            return STATUS_EXECUTION_FAILED;
        }

        let Some(embedding_end) = variant
            .embedding_start
            .checked_add(u64::from(variant.embedding_count))
        else {
            return STATUS_BOUNDS;
        };
        if embedding_end > embedding_count as u64 {
            return STATUS_BOUNDS;
        }
        let embedding_start = variant.embedding_start as usize;
        let embedding_end = embedding_end as usize;
        let variant_embeddings = &embeddings[embedding_start..embedding_end];
        let Some(destination_end) = u64::from(row.destination_component_base)
            .checked_add(u64::from(variant.embedding_count))
            .and_then(|planes| planes.checked_mul(u64::from(arena.point_stride)))
        else {
            return STATUS_BOUNDS;
        };
        if destination_end > arena.current_scalar_len {
            return STATUS_BOUNDS;
        }
        let Some(momentum_end) = u64::from(row.momentum_form_id)
            .checked_mul(4)
            .and_then(|base| base.checked_add(4))
            .and_then(|planes| planes.checked_mul(u64::from(momenta.point_stride)))
        else {
            return STATUS_BOUNDS;
        };
        if momentum_end > momenta.scalar_len {
            return STATUS_BOUNDS;
        }
        let mass = match template.mass_parameter_index {
            Some(index) => {
                if index >= parameters.value_count || parameters.values_re.is_null() {
                    return STATUS_BOUNDS;
                }
                unsafe { *parameters.values_re.add(index as usize) }
            }
            None => 0.0,
        };
        let crossing_re = unsafe {
            *factors
                .values_re
                .add(variant.crossing_exact_factor_id as usize)
        };
        let crossing_im = unsafe {
            *factors
                .values_im
                .add(variant.crossing_exact_factor_id as usize)
        };

        for point in 0..point_count {
            let momentum = read_momentum(momenta, row.momentum_form_id, point);
            let status = match template.family {
                DirectSourceWavefunctionFamily::Scalar => write_embedded_wavefunction(
                    arena,
                    row.destination_component_base,
                    point,
                    crossing_re,
                    crossing_im,
                    [Complex::new(1.0, 0.0)],
                    variant_embeddings,
                    factors,
                ),
                DirectSourceWavefunctionFamily::WeylFermion => {
                    let wave = match template.orientation {
                        DirectSourceOrientation::Particle => {
                            source_wavefunctions::ext_quark_weyl_array(
                                momentum,
                                template.helicity,
                                template.chirality,
                            )
                        }
                        DirectSourceOrientation::Antiparticle => {
                            source_wavefunctions::ext_antiquark_weyl_array(
                                momentum,
                                template.helicity,
                                template.chirality,
                            )
                        }
                        DirectSourceOrientation::SelfConjugate => {
                            return STATUS_EXECUTION_FAILED;
                        }
                    };
                    write_embedded_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        crossing_re,
                        crossing_im,
                        wave,
                        variant_embeddings,
                        factors,
                    )
                }
                DirectSourceWavefunctionFamily::DiracFermion => {
                    let wave = match template.orientation {
                        DirectSourceOrientation::Particle => {
                            source_wavefunctions::ext_quark_dirac_massive(
                                momentum,
                                template.helicity,
                                mass,
                            )
                        }
                        DirectSourceOrientation::Antiparticle => {
                            source_wavefunctions::ext_antiquark_dirac_massive(
                                momentum,
                                template.helicity,
                                mass,
                            )
                        }
                        DirectSourceOrientation::SelfConjugate => {
                            return STATUS_EXECUTION_FAILED;
                        }
                    };
                    write_embedded_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        crossing_re,
                        crossing_im,
                        wave,
                        variant_embeddings,
                        factors,
                    )
                }
                DirectSourceWavefunctionFamily::Vector => {
                    let wave = if mass == 0.0 {
                        source_wavefunctions::ext_gluon(momentum, template.helicity)
                    } else {
                        source_wavefunctions::ext_massive_vector(momentum, template.helicity, mass)
                    };
                    write_embedded_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        crossing_re,
                        crossing_im,
                        wave,
                        variant_embeddings,
                        factors,
                    )
                }
                DirectSourceWavefunctionFamily::Spin2 => {
                    let Ok(wave) =
                        source_wavefunctions::ext_spin2(momentum, template.helicity, mass)
                    else {
                        return STATUS_EXECUTION_FAILED;
                    };
                    write_embedded_wavefunction(
                        arena,
                        row.destination_component_base,
                        point,
                        crossing_re,
                        crossing_im,
                        wave,
                        variant_embeddings,
                        factors,
                    )
                }
            };
            if status != DIRECT_STATUS_OK {
                return status;
            }
        }
    }
    DIRECT_STATUS_OK
}

fn read_momentum(momenta: DirectMomentumView, form: u32, point: u32) -> [f64; 4] {
    std::array::from_fn(|component| {
        let plane = form as usize * 4 + component;
        let offset = plane * momenta.point_stride as usize + point as usize;
        unsafe { *momenta.values.add(offset) }
    })
}

#[allow(clippy::too_many_arguments)]
fn write_wavefunction<const N: usize>(
    arena: DirectArenaView,
    destination_component_base: u32,
    point: u32,
    factor_re: f64,
    factor_im: f64,
    wave: [Complex<f64>; N],
) -> c_int {
    for (component, value) in wave.into_iter().enumerate() {
        let Some(plane) = (destination_component_base as usize).checked_add(component) else {
            return STATUS_BOUNDS;
        };
        let Some(offset) = plane
            .checked_mul(arena.point_stride as usize)
            .and_then(|base| base.checked_add(point as usize))
        else {
            return STATUS_BOUNDS;
        };
        if offset >= arena.current_scalar_len as usize {
            return STATUS_BOUNDS;
        }
        unsafe {
            *arena.current_re.add(offset) = factor_re * value.re - factor_im * value.im;
            *arena.current_im.add(offset) = factor_re * value.im + factor_im * value.re;
        }
    }
    DIRECT_STATUS_OK
}

#[allow(clippy::too_many_arguments)]
fn write_embedded_wavefunction<const N: usize>(
    arena: DirectArenaView,
    destination_component_base: u32,
    point: u32,
    crossing_re: f64,
    crossing_im: f64,
    wave: [Complex<f64>; N],
    embeddings: &[DirectSourceEmbeddingRow],
    factors: DirectFactorView,
) -> c_int {
    for (full_component, embedding) in embeddings.iter().enumerate() {
        if embedding.full_component != full_component as u32
            || embedding.exact_factor_id >= factors.value_count
        {
            return STATUS_BOUNDS;
        }
        let Some(plane) = (destination_component_base as usize).checked_add(full_component) else {
            return STATUS_BOUNDS;
        };
        let Some(offset) = plane
            .checked_mul(arena.point_stride as usize)
            .and_then(|base| base.checked_add(point as usize))
        else {
            return STATUS_BOUNDS;
        };
        if offset >= arena.current_scalar_len as usize {
            return STATUS_BOUNDS;
        }

        let (value_re, value_im) = if embedding.source_component_or_sentinel == DIRECT_NONE_U32 {
            (0.0, 0.0)
        } else {
            let Some(value) = wave.get(embedding.source_component_or_sentinel as usize) else {
                return STATUS_BOUNDS;
            };
            let embedding_re =
                unsafe { *factors.values_re.add(embedding.exact_factor_id as usize) };
            let embedding_im =
                unsafe { *factors.values_im.add(embedding.exact_factor_id as usize) };
            let scale_re = crossing_re * embedding_re - crossing_im * embedding_im;
            let scale_im = crossing_re * embedding_im + crossing_im * embedding_re;
            (
                scale_re * value.re - scale_im * value.im,
                scale_re * value.im + scale_im * value.re,
            )
        };
        unsafe {
            *arena.current_re.add(offset) = value_re;
            *arena.current_im.add(offset) = value_im;
        }
    }
    DIRECT_STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::direct_backend::{
        DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    };

    fn domain(variant: DirectSourceTemplateSpec) -> DirectSourceDispatchDomainSpec {
        DirectSourceDispatchDomainSpec {
            variants: vec![variant],
        }
    }

    fn source_row(
        domain: u32,
        spin_state_class: i32,
        destination_component_base: u32,
        momentum_form_id: u32,
        exact_factor_id: u32,
    ) -> DirectSourceRow {
        DirectSourceRow {
            source_slot: domain,
            destination_component_base,
            momentum_form_id,
            source_template_or_dispatch_domain: domain,
            spin_state_class,
            exact_factor_id,
            selector_domain_id: 0,
        }
    }

    fn invoke(
        executor: &LoadedDirectSourceExecutor,
        rows: &[DirectSourceRow],
        current_re: &mut [f64],
        current_im: &mut [f64],
        momenta: &[f64],
        momentum_forms: u32,
        parameters: &[f64],
        factors_re: &[f64],
        factors_im: &[f64],
        point_stride: u32,
        point_count: u32,
    ) -> c_int {
        let handle = executor.handle();
        let mut amplitude_re = [0.0];
        let mut amplitude_im = [0.0];
        unsafe {
            (handle.call)(
                handle.context,
                DirectArenaView {
                    current_re: current_re.as_mut_ptr(),
                    current_im: current_im.as_mut_ptr(),
                    current_scalar_len: current_re.len() as u64,
                    amplitude_re: amplitude_re.as_mut_ptr(),
                    amplitude_im: amplitude_im.as_mut_ptr(),
                    amplitude_scalar_len: 1,
                    point_stride,
                },
                DirectMomentumView {
                    values: momenta.as_ptr(),
                    scalar_len: momenta.len() as u64,
                    form_count: momentum_forms,
                    lorentz_component_count: 4,
                    point_stride,
                },
                DirectParameterView {
                    values_re: parameters.as_ptr(),
                    values_im: ptr::null(),
                    value_count: parameters.len() as u32,
                },
                DirectFactorView {
                    values_re: factors_re.as_ptr(),
                    values_im: factors_im.as_ptr(),
                    value_count: factors_re.len() as u32,
                },
                rows.as_ptr(),
                rows.len() as u32,
                point_count,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn invoke_union(
        executor: &LoadedDirectSourceExecutor,
        rows: &[DirectSourceRow],
        variants: &[DirectSourceDispatchVariantDescriptor],
        embeddings: &[DirectSourceEmbeddingRow],
        selections: &[DirectResolvedSourceSelection],
        current_re: &mut [f64],
        current_im: &mut [f64],
        momenta: &[f64],
        momentum_forms: u32,
        parameters: &[f64],
        factors_re: &[f64],
        factors_im: &[f64],
        point_stride: u32,
        point_count: u32,
    ) -> c_int {
        let handle = executor.union_handle();
        let mut amplitude_re = [0.0];
        let mut amplitude_im = [0.0];
        unsafe {
            (handle.call)(
                handle.context,
                DirectArenaView {
                    current_re: current_re.as_mut_ptr(),
                    current_im: current_im.as_mut_ptr(),
                    current_scalar_len: current_re.len() as u64,
                    amplitude_re: amplitude_re.as_mut_ptr(),
                    amplitude_im: amplitude_im.as_mut_ptr(),
                    amplitude_scalar_len: 1,
                    point_stride,
                },
                DirectMomentumView {
                    values: momenta.as_ptr(),
                    scalar_len: momenta.len() as u64,
                    form_count: momentum_forms,
                    lorentz_component_count: 4,
                    point_stride,
                },
                DirectParameterView {
                    values_re: parameters.as_ptr(),
                    values_im: ptr::null(),
                    value_count: parameters.len() as u32,
                },
                DirectFactorView {
                    values_re: factors_re.as_ptr(),
                    values_im: factors_im.as_ptr(),
                    value_count: factors_re.len() as u32,
                },
                rows.as_ptr(),
                rows.len() as u32,
                variants.as_ptr(),
                variants.len() as u32,
                embeddings.as_ptr(),
                embeddings.len() as u32,
                selections.as_ptr(),
                selections.len() as u32,
                point_count,
            )
        }
    }

    #[test]
    fn scalar_source_writes_exact_complex_factor_directly_into_arena() {
        let executor = LoadedDirectSourceExecutor::load(vec![domain(DirectSourceTemplateSpec {
            spin_state_class: 0,
            family: DirectSourceWavefunctionFamily::Scalar,
            orientation: DirectSourceOrientation::SelfConjugate,
            helicity: 0,
            chirality: 0,
            mass_parameter_index: None,
        })])
        .unwrap();
        let rows = [source_row(0, 0, 0, 0, 0)];
        let mut current_re = [99.0, 99.0];
        let mut current_im = [99.0, 99.0];
        let momenta = [10.0, 11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            invoke(
                &executor,
                &rows,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[2.0],
                &[-3.0],
                2,
                2,
            ),
            DIRECT_STATUS_OK
        );
        assert_eq!(current_re, [2.0, 2.0]);
        assert_eq!(current_im, [-3.0, -3.0]);
    }

    #[test]
    fn vector_source_matches_established_source_ir_formula_for_every_point() {
        let executor = LoadedDirectSourceExecutor::load(vec![domain(DirectSourceTemplateSpec {
            spin_state_class: 1,
            family: DirectSourceWavefunctionFamily::Vector,
            orientation: DirectSourceOrientation::SelfConjugate,
            helicity: 1,
            chirality: 0,
            mass_parameter_index: None,
        })])
        .unwrap();
        let rows = [source_row(0, 1, 0, 0, 0)];
        let points = [
            [100.0, 30.0, 40.0, 86.60254037844386],
            [-120.0, -20.0, 50.0, -107.23805294763608],
        ];
        let momenta = [
            points[0][0],
            points[1][0],
            points[0][1],
            points[1][1],
            points[0][2],
            points[1][2],
            points[0][3],
            points[1][3],
        ];
        let mut current_re = [0.0; 8];
        let mut current_im = [0.0; 8];
        assert_eq!(
            invoke(
                &executor,
                &rows,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[1.0],
                &[0.0],
                2,
                2,
            ),
            DIRECT_STATUS_OK
        );
        for (point_index, point) in points.into_iter().enumerate() {
            let expected = source_wavefunctions::ext_gluon(point, 1);
            for (component, value) in expected.into_iter().enumerate() {
                let offset = component * 2 + point_index;
                assert_eq!(current_re[offset], value.re);
                assert_eq!(current_im[offset], value.im);
            }
        }
    }

    #[test]
    fn dispatch_domains_select_crossed_fermion_state_without_packing() {
        let executor = LoadedDirectSourceExecutor::load(vec![DirectSourceDispatchDomainSpec {
            variants: vec![
                DirectSourceTemplateSpec {
                    spin_state_class: 1,
                    family: DirectSourceWavefunctionFamily::WeylFermion,
                    orientation: DirectSourceOrientation::Particle,
                    helicity: 1,
                    chirality: 1,
                    mass_parameter_index: None,
                },
                DirectSourceTemplateSpec {
                    spin_state_class: -1,
                    family: DirectSourceWavefunctionFamily::WeylFermion,
                    orientation: DirectSourceOrientation::Antiparticle,
                    helicity: -1,
                    chirality: -1,
                    mass_parameter_index: None,
                },
            ],
        }])
        .unwrap();
        let rows = [source_row(0, -1, 0, 0, 0)];
        let momentum = [-50.0, -10.0, -20.0, -44.721359549995796];
        let momenta = [momentum[0], momentum[1], momentum[2], momentum[3]];
        let mut current_re = [0.0; 2];
        let mut current_im = [0.0; 2];
        assert_eq!(
            invoke(
                &executor,
                &rows,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[0.0],
                &[1.0],
                1,
                1,
            ),
            DIRECT_STATUS_OK
        );
        let expected = source_wavefunctions::ext_antiquark_weyl_array(momentum, -1, -1);
        for (component, value) in expected.into_iter().enumerate() {
            assert_eq!(current_re[component], -value.im);
            assert_eq!(current_im[component], value.re);
        }
    }

    #[test]
    fn every_supported_source_ir_spin_family_executes_from_persistent_views() {
        let specs = [
            DirectSourceTemplateSpec {
                spin_state_class: 0,
                family: DirectSourceWavefunctionFamily::Scalar,
                orientation: DirectSourceOrientation::SelfConjugate,
                helicity: 0,
                chirality: 0,
                mass_parameter_index: None,
            },
            DirectSourceTemplateSpec {
                spin_state_class: 1,
                family: DirectSourceWavefunctionFamily::WeylFermion,
                orientation: DirectSourceOrientation::Particle,
                helicity: 1,
                chirality: 1,
                mass_parameter_index: None,
            },
            DirectSourceTemplateSpec {
                spin_state_class: 2,
                family: DirectSourceWavefunctionFamily::DiracFermion,
                orientation: DirectSourceOrientation::Antiparticle,
                helicity: -1,
                chirality: 0,
                mass_parameter_index: Some(0),
            },
            DirectSourceTemplateSpec {
                spin_state_class: 3,
                family: DirectSourceWavefunctionFamily::Vector,
                orientation: DirectSourceOrientation::SelfConjugate,
                helicity: 0,
                chirality: 0,
                mass_parameter_index: Some(1),
            },
            DirectSourceTemplateSpec {
                spin_state_class: 4,
                family: DirectSourceWavefunctionFamily::Spin2,
                orientation: DirectSourceOrientation::SelfConjugate,
                helicity: 2,
                chirality: 0,
                mass_parameter_index: Some(2),
            },
        ];
        let executor =
            LoadedDirectSourceExecutor::load(specs.into_iter().map(domain).collect()).unwrap();
        let bases = [0, 1, 3, 7, 11];
        let rows: Vec<_> = specs
            .into_iter()
            .enumerate()
            .map(|(index, spec)| {
                source_row(index as u32, spec.spin_state_class, bases[index], 0, 0)
            })
            .collect();
        let momenta = [20.0, 3.0, 4.0, 19.364916731037084];
        let mut current_re = [0.0; 27];
        let mut current_im = [0.0; 27];
        assert_eq!(
            invoke(
                &executor,
                &rows,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[2.0, 5.0, 7.0],
                &[1.0],
                &[0.0],
                1,
                1,
            ),
            DIRECT_STATUS_OK
        );
        assert!(
            current_re
                .iter()
                .chain(&current_im)
                .all(|value| value.is_finite())
        );
        assert!(
            current_re
                .iter()
                .chain(&current_im)
                .any(|value| *value != 0.0)
        );
    }

    #[test]
    fn source_handle_owns_stable_context_and_repeated_calls_allocate_no_scratch() {
        let executor = LoadedDirectSourceExecutor::load(vec![domain(DirectSourceTemplateSpec {
            spin_state_class: 0,
            family: DirectSourceWavefunctionFamily::Scalar,
            orientation: DirectSourceOrientation::SelfConjugate,
            helicity: 0,
            chirality: 0,
            mass_parameter_index: None,
        })])
        .unwrap();
        let first = executor.handle();
        let second = executor.handle();
        assert_eq!(first.context, second.context);
        assert_eq!(first.call as usize, second.call as usize);

        let rows = [source_row(0, 0, 0, 0, 0)];
        let momenta = [1.0, 0.0, 0.0, 1.0];
        let mut current_re = [0.0];
        let mut current_im = [0.0];
        for _ in 0..32 {
            assert_eq!(
                invoke(
                    &executor,
                    &rows,
                    &mut current_re,
                    &mut current_im,
                    &momenta,
                    1,
                    &[],
                    &[1.0],
                    &[0.0],
                    1,
                    1,
                ),
                DIRECT_STATUS_OK
            );
        }
        assert_eq!(current_re, [1.0]);
        assert_eq!(current_im, [0.0]);
    }

    #[test]
    fn malformed_domains_and_views_fail_before_arena_mutation() {
        let duplicate = DirectSourceTemplateSpec {
            spin_state_class: 1,
            family: DirectSourceWavefunctionFamily::Scalar,
            orientation: DirectSourceOrientation::SelfConjugate,
            helicity: 0,
            chirality: 0,
            mass_parameter_index: None,
        };
        assert!(
            LoadedDirectSourceExecutor::load(vec![DirectSourceDispatchDomainSpec {
                variants: vec![duplicate, duplicate],
            }])
            .is_err()
        );

        let executor = LoadedDirectSourceExecutor::load(vec![domain(duplicate)]).unwrap();
        let rows = [source_row(0, 99, 0, 0, 0)];
        let mut current_re = [17.0];
        let mut current_im = [19.0];
        assert_eq!(
            invoke(
                &executor,
                &rows,
                &mut current_re,
                &mut current_im,
                &[1.0, 0.0, 0.0, 1.0],
                1,
                &[],
                &[1.0],
                &[0.0],
                1,
                1,
            ),
            STATUS_BOUNDS
        );
        assert_eq!(current_re, [17.0]);
        assert_eq!(current_im, [19.0]);
    }

    #[test]
    fn union_dispatch_embeds_weyl_source_into_full_state_and_zeros_inactive_components() {
        let executor = LoadedDirectSourceExecutor::load(vec![domain(DirectSourceTemplateSpec {
            spin_state_class: 1,
            family: DirectSourceWavefunctionFamily::WeylFermion,
            orientation: DirectSourceOrientation::Particle,
            helicity: 1,
            chirality: 1,
            mass_parameter_index: None,
        })])
        .unwrap();
        let rows = [source_row(0, 0, 0, 0, 0)];
        let variants = [DirectSourceDispatchVariantDescriptor {
            embedding_start: 0,
            projection_start: 0,
            source_row_id: 0,
            dispatch_domain_id: 0,
            runtime_variant_id: 0,
            source_state_index: 0,
            source_template_id: 0,
            source_state_template_id: 0,
            crossed_state_template_id: 0,
            crossed_spin_state_class: 1,
            direct_executor_id: 0,
            crossing_exact_factor_id: 1,
            embedding_count: 4,
            projection_count: 2,
        }];
        let embeddings = [
            DirectSourceEmbeddingRow {
                full_component: 0,
                source_component_or_sentinel: 0,
                exact_factor_id: 2,
            },
            DirectSourceEmbeddingRow {
                full_component: 1,
                source_component_or_sentinel: DIRECT_NONE_U32,
                exact_factor_id: 3,
            },
            DirectSourceEmbeddingRow {
                full_component: 2,
                source_component_or_sentinel: 1,
                exact_factor_id: 0,
            },
            DirectSourceEmbeddingRow {
                full_component: 3,
                source_component_or_sentinel: DIRECT_NONE_U32,
                exact_factor_id: 3,
            },
        ];
        let selections = [DirectResolvedSourceSelection {
            source_slot: 0,
            dispatch_variant_id: 0,
        }];
        let momentum = [50.0, 10.0, 20.0, 44.721359549995796];
        let mut current_re = [99.0; 4];
        let mut current_im = [-99.0; 4];
        let factors_re = [1.0, 0.0, 2.0, 0.0];
        let factors_im = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(
            invoke_union(
                &executor,
                &rows,
                &variants,
                &embeddings,
                &selections,
                &mut current_re,
                &mut current_im,
                &momentum,
                1,
                &[],
                &factors_re,
                &factors_im,
                1,
                1,
            ),
            DIRECT_STATUS_OK
        );
        let wave = source_wavefunctions::ext_quark_weyl_array(momentum, 1, 1);
        assert_eq!(current_re[0], -2.0 * wave[0].im);
        assert_eq!(current_im[0], 2.0 * wave[0].re);
        assert_eq!((current_re[1], current_im[1]), (0.0, 0.0));
        assert_eq!(current_re[2], -wave[1].im);
        assert_eq!(current_im[2], wave[1].re);
        assert_eq!((current_re[3], current_im[3]), (0.0, 0.0));

        current_re.fill(123.0);
        current_im.fill(456.0);
        assert_eq!(
            invoke_union(
                &executor,
                &rows,
                &variants,
                &embeddings,
                &selections,
                &mut current_re,
                &mut current_im,
                &momentum,
                1,
                &[],
                &factors_re,
                &factors_im,
                1,
                1,
            ),
            DIRECT_STATUS_OK
        );
        assert_eq!((current_re[1], current_im[1]), (0.0, 0.0));
        assert_eq!((current_re[3], current_im[3]), (0.0, 0.0));
    }

    #[test]
    fn union_dispatch_rejects_malformed_selection_variant_and_embedding_ranges() {
        let executor = LoadedDirectSourceExecutor::load(vec![domain(DirectSourceTemplateSpec {
            spin_state_class: 0,
            family: DirectSourceWavefunctionFamily::Scalar,
            orientation: DirectSourceOrientation::SelfConjugate,
            helicity: 0,
            chirality: 0,
            mass_parameter_index: None,
        })])
        .unwrap();
        let rows = [source_row(0, 0, 0, 0, 0)];
        let base_variant = DirectSourceDispatchVariantDescriptor {
            embedding_start: 0,
            projection_start: 0,
            source_row_id: 0,
            dispatch_domain_id: 0,
            runtime_variant_id: 0,
            source_state_index: 0,
            source_template_id: 0,
            source_state_template_id: 0,
            crossed_state_template_id: 0,
            crossed_spin_state_class: 0,
            direct_executor_id: 0,
            crossing_exact_factor_id: 0,
            embedding_count: 1,
            projection_count: 1,
        };
        let embeddings = [DirectSourceEmbeddingRow {
            full_component: 0,
            source_component_or_sentinel: 0,
            exact_factor_id: 0,
        }];
        let mut current_re = [7.0];
        let mut current_im = [9.0];
        let momenta = [1.0, 0.0, 0.0, 1.0];

        let bad_variant = [DirectResolvedSourceSelection {
            source_slot: 0,
            dispatch_variant_id: 1,
        }];
        assert_eq!(
            invoke_union(
                &executor,
                &rows,
                &[base_variant],
                &embeddings,
                &bad_variant,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[1.0],
                &[0.0],
                1,
                1,
            ),
            STATUS_BOUNDS
        );

        let bad_order = [DirectResolvedSourceSelection {
            source_slot: 1,
            dispatch_variant_id: 0,
        }];
        assert_eq!(
            invoke_union(
                &executor,
                &rows,
                &[base_variant],
                &embeddings,
                &bad_order,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[1.0],
                &[0.0],
                1,
                1,
            ),
            STATUS_INVALID_ARGUMENT
        );

        let mut bad_embedding = base_variant;
        bad_embedding.embedding_start = 2;
        let good_selection = [DirectResolvedSourceSelection {
            source_slot: 0,
            dispatch_variant_id: 0,
        }];
        assert_eq!(
            invoke_union(
                &executor,
                &rows,
                &[bad_embedding],
                &embeddings,
                &good_selection,
                &mut current_re,
                &mut current_im,
                &momenta,
                1,
                &[],
                &[1.0],
                &[0.0],
                1,
                1,
            ),
            STATUS_BOUNDS
        );
        assert_eq!(current_re, [7.0]);
        assert_eq!(current_im, [9.0]);
    }
}
