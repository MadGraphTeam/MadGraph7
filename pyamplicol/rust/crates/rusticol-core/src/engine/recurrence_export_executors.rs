// SPDX-License-Identifier: 0BSD

//! Reconstruct export-only executor descriptions from the authenticated pack.

use std::collections::BTreeMap;

use super::eager_manifest::{
    RecurrenceDirectParameterBindingManifest as Binding,
    RecurrenceDirectPlaneProjectionManifest as Plane,
    RecurrenceDirectScalarProjectionManifest as Scalar,
};
use super::evaluator::recurrence_source_direct::{
    DirectSourceDispatchDomainSpec, DirectSourceDispatchKey, DirectSourceOrientation,
    DirectSourceWavefunctionFamily,
};
use super::recurrence_process_pack::{
    ProcessDirectExecutorBinding, ProcessDirectExecutorDescriptor,
};
use crate::recurrence::direct_export::{
    DirectExportExecutor, DirectExportSource, DirectExportSourceEmbedding, ExportComplex,
    RelativeKernelInput,
};
use crate::recurrence::{DIRECT_NONE_U32, DirectRecurrencePlan, RecurrenceStrategy};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!("recurrence export: {}", message.into()))
}

pub(super) fn export_executors(
    descriptors: &[ProcessDirectExecutorDescriptor],
    scales: &[ExportComplex],
) -> RusticolResult<BTreeMap<u32, DirectExportExecutor>> {
    descriptors.iter().map(|descriptor| {
        let executor = match &descriptor.binding {
            ProcessDirectExecutorBinding::Source => DirectExportExecutor::Source,
            ProcessDirectExecutorBinding::Intrinsic(intrinsic) => {
                let coupling = match intrinsic.scale {
                    Some(scale) => {
                        let literal = ExportComplex { re: f64::from_bits(scale.constant_real_bits), im: f64::from_bits(scale.constant_imag_bits) };
                        match scale.parameter_index {
                            Some(index) => literal.multiply(*scales.get(index as usize).ok_or_else(|| invalid("intrinsic parameter index is absent"))?),
                            None => literal,
                        }
                    }
                    None => ExportComplex::ONE,
                };
                let coupling = if descriptor.role == crate::recurrence::DirectExecutorRole::Contribution {
                    coupling.multiply(ExportComplex::ONE)
                } else { coupling };
                DirectExportExecutor::Intrinsic { template: intrinsic.runtime_template.clone(), coupling }
            }
            ProcessDirectExecutorBinding::Jit(kernel) => {
                let projection = |plane: &Plane| match *plane {
                    Plane::ParentCurrent { parent, component, imaginary } => RelativeKernelInput::Parent { parent, component, imaginary },
                    Plane::Momentum { operand, lorentz_component } => RelativeKernelInput::Momentum { operand, component: lorentz_component },
                    Plane::DestinationCurrent { component, imaginary } => RelativeKernelInput::Destination { component, imaginary, amplitude: false },
                    Plane::DestinationAmplitude { component, imaginary } => RelativeKernelInput::Destination { component, imaginary, amplitude: true },
                };
                let inputs = kernel.parameter_bindings.iter().map(|binding| {
                    Ok(match *binding {
                        Binding::Plane { index } => projection(kernel.input_plane_projections.get(index as usize).ok_or_else(|| invalid("kernel input plane is absent"))?),
                        Binding::Scalar { index } => match *kernel.scalar_projections.get(index as usize).ok_or_else(|| invalid("kernel scalar projection is absent"))? {
                            Scalar::ExactFactor { imaginary } => RelativeKernelInput::Factor { imaginary },
                            Scalar::Parameter { index, imaginary } => {
                                let scale = scales.get(index as usize).ok_or_else(|| invalid("kernel parameter index is absent"))?;
                                RelativeKernelInput::Literal(if imaginary { scale.im } else { scale.re })
                            }
                            Scalar::Literal { value } => RelativeKernelInput::Literal(value),
                            Scalar::IntrinsicScale { .. } => return Err(invalid("intrinsic scale bound to a saved kernel")),
                        },
                    })
                }).collect::<RusticolResult<Vec<_>>>()?;
                let output_projections = kernel.output_alias_inputs.iter().map(|index| {
                    kernel.input_plane_projections.get(*index as usize).map(projection).ok_or_else(|| invalid("kernel output projection is absent"))
                }).collect::<RusticolResult<Vec<_>>>()?;
                DirectExportExecutor::Kernel { sha256: kernel.source_application_sha256.iter().map(|byte| format!("{byte:02x}")).collect(), inputs, output_projections }
            }
            ProcessDirectExecutorBinding::Native(_) => return Err(RusticolError::compatibility(
                "recurrence export requires portable JIT or intrinsic executors; regenerate the artifact with the jit backend",
            )),
        };
        Ok((descriptor.direct_executor_id, executor))
    }).collect()
}

fn exact_factor(plan: &DirectRecurrencePlan, id: u32) -> RusticolResult<ExportComplex> {
    let value = plan
        .exact_factors()
        .get(id as usize)
        .ok_or_else(|| invalid("source factor is absent"))?;
    Ok(ExportComplex {
        re: value.real().numerator() as f64 / value.real().denominator() as f64,
        im: value.imag().numerator() as f64 / value.imag().denominator() as f64,
    })
}

pub(super) fn export_source_templates(
    domains: &[DirectSourceDispatchDomainSpec],
    plan: &DirectRecurrencePlan,
    scales: &[ExportComplex],
    resolved_helicity: Option<u32>,
) -> RusticolResult<BTreeMap<u32, DirectExportSource>> {
    let union = plan.strategy() == RecurrenceStrategy::AllFlowUnion;
    let selections = if union {
        let helicity = plan
            .resolved_helicities()
            .iter()
            .find(|entry| Some(entry.id) == resolved_helicity)
            .ok_or_else(|| invalid("union source helicity is absent"))?;
        let start = helicity.source_selection_start as usize;
        plan.resolved_source_selections()
            .get(start..start + helicity.source_selection_count as usize)
            .ok_or_else(|| invalid("union source selection range is absent"))?
    } else {
        &[]
    };
    let mut output = BTreeMap::new();
    for (row_id, row) in plan.sources().iter().enumerate() {
        let variant = if union {
            let selection = selections
                .iter()
                .find(|entry| entry.source_slot == row.source_slot)
                .ok_or_else(|| invalid("union source slot is absent"))?;
            Some(
                plan.source_dispatch_variants()
                    .get(selection.dispatch_variant_id as usize)
                    .ok_or_else(|| invalid("union source variant is absent"))?,
            )
        } else {
            None
        };
        let domain = domains
            .get(row.source_template_or_dispatch_domain as usize)
            .ok_or_else(|| invalid("source dispatch domain is absent"))?;
        let key = variant.map_or(
            DirectSourceDispatchKey::SpinStateClass(row.spin_state_class),
            |variant| DirectSourceDispatchKey::RuntimeVariant {
                source_row_id: variant.source_row_id,
                runtime_variant_id: variant.runtime_variant_id,
            },
        );
        let template = domain
            .variants
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.template)
            .ok_or_else(|| invalid("source template is absent"))?;
        let (family, component_count) = match template.family {
            DirectSourceWavefunctionFamily::Scalar => ("scalar", 1),
            DirectSourceWavefunctionFamily::WeylFermion => ("weyl", 2),
            DirectSourceWavefunctionFamily::DiracFermion => ("dirac", 4),
            DirectSourceWavefunctionFamily::Vector => ("vector", 4),
            DirectSourceWavefunctionFamily::Spin2 => ("spin2", 16),
        };
        let orientation = match template.orientation {
            DirectSourceOrientation::Particle => "particle",
            DirectSourceOrientation::Antiparticle => "antiparticle",
            DirectSourceOrientation::SelfConjugate => "self_conjugate",
        };
        let mass = match template.mass_parameter_index {
            Some(index) => *scales
                .get(index as usize)
                .ok_or_else(|| invalid("source mass scale is absent"))?,
            None => ExportComplex { re: 0.0, im: 0.0 },
        };
        let mut embedding = Vec::new();
        if let Some(variant) = variant {
            let crossing = exact_factor(plan, variant.crossing_exact_factor_id)?;
            let start = variant.embedding_start as usize;
            for entry in plan
                .source_embeddings()
                .get(start..start + variant.embedding_count as usize)
                .ok_or_else(|| invalid("union source embedding range is absent"))?
            {
                let source_component = (entry.source_component_or_sentinel != DIRECT_NONE_U32)
                    .then_some(entry.source_component_or_sentinel);
                let factor = if source_component.is_some() {
                    crossing.multiply(exact_factor(plan, entry.exact_factor_id)?)
                } else {
                    ExportComplex { re: 0.0, im: 0.0 }
                };
                embedding.push(DirectExportSourceEmbedding {
                    component: entry.full_component,
                    source_component,
                    factor,
                });
            }
        }
        output.insert(
            row_id as u32,
            DirectExportSource {
                current: row.destination_component_base,
                component_count,
                family: family.to_owned(),
                orientation: orientation.to_owned(),
                leg: row.source_slot,
                momentum_form: row.momentum_form_id,
                helicity: template.helicity,
                chirality: template.chirality,
                mass,
                factor: exact_factor(plan, row.exact_factor_id)?,
                embedding,
            },
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::super::evaluator::recurrence_source_direct::{
        DirectSourceDispatchVariantSpec, DirectSourceTemplateSpec,
    };
    use super::super::recurrence_process_pack::{ProcessIntrinsicExecutor, ProcessIntrinsicScale};
    use super::*;
    use crate::recurrence::DirectExecutorRole;

    #[test]
    fn recurrence_export_resolves_current_parameter_values_without_evaluation() {
        let scale = ExportComplex { re: 0.25, im: -0.5 };
        let descriptor = ProcessDirectExecutorDescriptor {
            direct_executor_id: 3,
            role: DirectExecutorRole::Contribution,
            destination_component_count: 2,
            uses_exact_factor: true,
            binding: ProcessDirectExecutorBinding::Intrinsic(ProcessIntrinsicExecutor {
                runtime_template: "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1".into(),
                scale: Some(ProcessIntrinsicScale {
                    constant_real_bits: 2.0_f64.to_bits(),
                    constant_imag_bits: 3.0_f64.to_bits(),
                    parameter_index: Some(0),
                }),
            }),
        };
        let exported = export_executors(&[descriptor], &[scale]).unwrap();
        let DirectExportExecutor::Intrinsic { coupling, .. } = exported.get(&3).unwrap() else {
            panic!("expected intrinsic")
        };
        assert_eq!(*coupling, ExportComplex { re: 2.0, im: -0.25 });
        let mut domains = vec![DirectSourceDispatchDomainSpec { variants: vec![] }; 8];
        domains[7].variants.push(DirectSourceDispatchVariantSpec {
            key: DirectSourceDispatchKey::SpinStateClass(-1),
            template: DirectSourceTemplateSpec {
                spin_state_class: -1,
                family: DirectSourceWavefunctionFamily::WeylFermion,
                orientation: DirectSourceOrientation::Antiparticle,
                helicity: -1,
                chirality: 1,
                mass_parameter_index: Some(0),
            },
        });
        let plan = crate::recurrence::valid_direct_plan_fixture();
        let sources = export_source_templates(&domains, &plan, &[scale], None).unwrap();
        let source = sources.get(&0).unwrap();
        assert_eq!(source.mass, scale);
        assert_eq!(source.orientation, "antiparticle");
        assert_eq!(source.chirality, 1);
        assert_eq!(source.current, 0);
        assert!(source.embedding.is_empty());
    }
}
