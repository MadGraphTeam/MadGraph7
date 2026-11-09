// SPDX-License-Identifier: 0BSD

use super::*;
use crate::direct_arena::{
    DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
};
use crate::recurrence::direct_export::{
    DirectExportExecutor, DirectExportInstruction, ExportComplex, export_direct_body,
};
use crate::recurrence::{
    DirectExecutorRole, DirectRecurrencePlan, valid_direct_plan_parts_fixture,
};

/// Interpret the serialized closure operands independently and compare the
/// arithmetic with the production batched callback, including signed zero.
#[test]
fn recurrence_export_closure_roundtrip_matches_native_bits() {
    let mut parts = valid_direct_plan_parts_fixture();
    parts.closures[0].parent1_component_base_or_sentinel = 0;
    parts.closures[0].parent1_momentum_form_id_or_sentinel = 0;
    let digest = |value| crate::recurrence::SemanticDigest::new([value; 32]).unwrap();
    let contribution = crate::recurrence::ClosureProofContributionV2::new(
        0,
        0,
        Some(0),
        Some(0),
        0,
        digest(0x41),
        None,
        vec![1, 0],
        vec![Some(1), Some(0)],
        vec![digest(0x42), digest(0x45)],
        vec![digest(0x43), digest(0x46)],
        vec![0, 1],
        vec![0, 1],
        vec![0, 1],
        0,
        digest(0x44),
        None,
        vec![],
        None,
        crate::recurrence::ExactComplexRational::ONE,
        1,
    )
    .unwrap();
    parts.closure_proofs = crate::recurrence::ClosureProofMetadataV2::new(
        vec![contribution],
        parts.closure_proofs.groups().to_vec(),
        vec![],
    )
    .unwrap();
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let groups = plan
        .row_groups()
        .iter()
        .copied()
        .filter(|group| group.role == DirectExecutorRole::Closure)
        .collect::<Vec<_>>();
    let executors = BTreeMap::from([(
        groups[0].direct_executor_id,
        DirectExportExecutor::Intrinsic {
            template: "rusticol.closure-reduce.v1:test".into(),
            coupling: ExportComplex::ONE,
        },
    )]);
    let body = export_direct_body(&plan, &groups, &[], &executors, &BTreeMap::new(), None).unwrap();
    let body: Vec<DirectExportInstruction> =
        serde_json::from_str(&serde_json::to_string(&body).unwrap()).unwrap();
    let points = 5usize;
    let mut current_re = vec![
        0.0,
        -0.0,
        0.9,
        -1.1,
        f64::from_bits(0x3ff0000000000001),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        1.25,
        -1.7,
        0.6,
        0.1,
    ];
    let mut current_im = vec![
        -0.0, 0.0, -1.3, 0.7, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.3, -1.9, 0.8,
    ];
    let mut actual_re = vec![0.0; points];
    let mut actual_im = vec![0.0; points];
    let factors_re = [1.0];
    let factors_im = [0.0];
    let status = unsafe {
        evaluator::recurrence_closure_direct::execute_closure_reduce_rows(
            std::ptr::null(),
            DirectArenaView {
                current_re: current_re.as_mut_ptr(),
                current_im: current_im.as_mut_ptr(),
                current_scalar_len: current_re.len() as u64,
                amplitude_re: actual_re.as_mut_ptr(),
                amplitude_im: actual_im.as_mut_ptr(),
                amplitude_scalar_len: points as u64,
                point_stride: points as u32,
            },
            DirectMomentumView {
                values: std::ptr::null(),
                scalar_len: 0,
                form_count: 0,
                lorentz_component_count: 4,
                point_stride: points as u32,
            },
            DirectParameterView {
                values_re: std::ptr::null(),
                values_im: std::ptr::null(),
                value_count: 0,
            },
            DirectFactorView {
                values_re: factors_re.as_ptr(),
                values_im: factors_im.as_ptr(),
                value_count: 1,
            },
            plan.closures().as_ptr(),
            plan.closures().len() as u32,
            points as u32,
        )
    };
    assert_eq!(status, 0);
    let mut expected_re = vec![f64::NAN; points];
    let mut expected_im = vec![f64::NAN; points];
    for instruction in body {
        match instruction {
            DirectExportInstruction::ClearAmplitude { amplitude: 0 } => {
                expected_re.fill(0.0);
                expected_im.fill(0.0);
            }
            DirectExportInstruction::Closure {
                amplitude: 0,
                left,
                right,
                component_factors,
                factor,
            } => {
                for (component, coefficient) in component_factors.iter().enumerate() {
                    let sr = coefficient.re * factor.re - coefficient.im * factor.im;
                    let si = coefficient.re * factor.im + coefficient.im * factor.re;
                    for point in 0..points {
                        let l = (left as usize + component) * points + point;
                        let r = (right as usize + component) * points + point;
                        let pr = current_re[l] * current_re[r] - current_im[l] * current_im[r];
                        let pi = current_re[l] * current_im[r] + current_im[l] * current_re[r];
                        expected_re[point] += pr * sr - pi * si;
                        expected_im[point] += pr * si + pi * sr;
                    }
                }
            }
            other => panic!("unexpected closure program instruction {other:?}"),
        }
    }
    for point in 0..points {
        assert_eq!(actual_re[point].to_bits(), expected_re[point].to_bits());
        assert_eq!(actual_im[point].to_bits(), expected_im[point].to_bits());
    }
}

/// This runner deliberately has no access to a DirectRecurrencePlan, executor
/// catalog, selector, replay binding or runtime. The only production machinery
/// reused is the named arithmetic and the saved SymJIT application evaluator.
#[cfg(feature = "f64-symjit")]
fn interpret_recurrence_export(
    program: &RecurrenceExportProgram,
    kernels: &[RecurrenceExportKernel],
    external: &[f64],
    points: usize,
) -> Vec<f64> {
    use crate::recurrence::direct_export::{DirectExportKernelInput as Input, DirectExportMode};
    use crate::recurrence::{DIRECT_NONE_U32, DirectContributionRow, DirectFinalizationRow};
    let mut momenta = vec![0.0; program.momentum_forms.len() * 4 * points];
    for (form, terms) in program.momentum_forms.iter().enumerate() {
        for component in 0..4 {
            for point in 0..points {
                for term in terms {
                    let crossing = program.crossing.as_ref().and_then(|entries| {
                        entries
                            .iter()
                            .find(|entry| entry.program_leg == term.leg as usize)
                    });
                    let leg = crossing.map_or(term.leg as usize, |entry| entry.caller_leg);
                    let sign = crossing.map_or(1.0, |entry| entry.sign);
                    let value =
                        external[(point * program.external_count + leg) * 4 + component] * sign;
                    momenta[(form * 4 + component) * points + point] += value * term.coefficient;
                }
            }
        }
    }
    let mut cr = vec![f64::NAN; program.current_component_count as usize * points];
    let mut ci = cr.clone();
    let mut ar = vec![f64::NAN; program.amplitude_count as usize * points];
    let mut ai = ar.clone();
    let mut output = vec![0.0; points * program.helicity_outputs.len()];
    for run in &program.runs {
        let body: Vec<DirectExportInstruction> =
            serde_json::from_slice(&serde_json::to_vec(&run.body).unwrap()).unwrap();
        for instruction in body {
            let arena = DirectArenaView {
                current_re: cr.as_mut_ptr(),
                current_im: ci.as_mut_ptr(),
                current_scalar_len: cr.len() as u64,
                amplitude_re: ar.as_mut_ptr(),
                amplitude_im: ai.as_mut_ptr(),
                amplitude_scalar_len: ar.len() as u64,
                point_stride: points as u32,
            };
            let momentum_view = DirectMomentumView {
                values: momenta.as_ptr(),
                scalar_len: momenta.len() as u64,
                form_count: program.momentum_forms.len() as u32,
                lorentz_component_count: 4,
                point_stride: points as u32,
            };
            match instruction {
                DirectExportInstruction::Clear {
                    current,
                    component_count,
                } => {
                    let range =
                        current as usize * points..(current + component_count) as usize * points;
                    cr[range.clone()].fill(0.0);
                    ci[range].fill(0.0);
                }
                DirectExportInstruction::ClearAmplitude { amplitude } => {
                    ar[amplitude as usize * points..(amplitude as usize + 1) * points].fill(0.0);
                    ai[amplitude as usize * points..(amplitude as usize + 1) * points].fill(0.0);
                }
                DirectExportInstruction::Source { source } => {
                    for point in 0..points {
                        let p = std::array::from_fn(|component| {
                            momenta
                                [(source.momentum_form as usize * 4 + component) * points + point]
                        });
                        let wave: Vec<(f64, f64)> = match source.family.as_str() {
                            "scalar" => vec![(1.0, 0.0)],
                            "weyl" => {
                                let wave = if source.orientation == "particle" {
                                    wavefunctions::ext_quark_weyl_array(
                                        p,
                                        source.helicity,
                                        source.chirality,
                                    )
                                } else {
                                    wavefunctions::ext_antiquark_weyl_array(
                                        p,
                                        source.helicity,
                                        source.chirality,
                                    )
                                };
                                wave.into_iter().map(|value| (value.re, value.im)).collect()
                            }
                            "dirac" => {
                                let wave = if source.orientation == "particle" {
                                    wavefunctions::ext_quark_dirac_massive(
                                        p,
                                        source.helicity,
                                        source.mass.re,
                                    )
                                } else {
                                    wavefunctions::ext_antiquark_dirac_massive(
                                        p,
                                        source.helicity,
                                        source.mass.re,
                                    )
                                };
                                wave.into_iter().map(|value| (value.re, value.im)).collect()
                            }
                            "vector" => {
                                let wave = if source.mass.re == 0.0 {
                                    wavefunctions::ext_gluon(p, source.helicity)
                                } else {
                                    wavefunctions::ext_massive_vector(
                                        p,
                                        source.helicity,
                                        source.mass.re,
                                    )
                                };
                                wave.into_iter().map(|value| (value.re, value.im)).collect()
                            }
                            "spin2" => wavefunctions::ext_spin2(p, source.helicity, source.mass.re)
                                .unwrap()
                                .into_iter()
                                .map(|value| (value.re, value.im))
                                .collect(),
                            other => panic!("unsupported exported source {other}"),
                        };
                        if source.embedding.is_empty() {
                            for (component, (re, im)) in wave.into_iter().enumerate() {
                                let index = (source.current as usize + component) * points + point;
                                cr[index] = source.factor.re * re - source.factor.im * im;
                                ci[index] = source.factor.re * im + source.factor.im * re;
                            }
                        } else {
                            for embedding in &source.embedding {
                                let index = (source.current as usize
                                    + embedding.component as usize)
                                    * points
                                    + point;
                                if let Some(component) = embedding.source_component {
                                    let (re, im) = wave[component as usize];
                                    cr[index] = embedding.factor.re * re - embedding.factor.im * im;
                                    ci[index] = embedding.factor.re * im + embedding.factor.im * re;
                                } else {
                                    cr[index] = 0.0;
                                    ci[index] = 0.0;
                                }
                            }
                        }
                    }
                }
                DirectExportInstruction::Vertex {
                    vertex,
                    current,
                    mode,
                    left,
                    right,
                    left_momentum,
                    right_momentum,
                    coupling,
                    factor,
                } => {
                    let row = DirectContributionRow {
                        parent0_component_base: left,
                        parent1_component_base_or_sentinel: right.unwrap_or(DIRECT_NONE_U32),
                        parent0_momentum_form_id: left_momentum,
                        parent1_momentum_form_id_or_sentinel: right_momentum
                            .unwrap_or(DIRECT_NONE_U32),
                        destination_component_base: current,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                        flags: if mode == DirectExportMode::Set { 1 } else { 0 },
                    };
                    unsafe {
                        evaluator::recurrence_intrinsic_direct::execute_exported_vertex_for_test(
                            &vertex,
                            row,
                            arena,
                            momentum_view,
                            (coupling.re, coupling.im),
                            (factor.re, factor.im),
                            points,
                        )
                    };
                }
                DirectExportInstruction::Propagator {
                    propagator,
                    current,
                    component_count,
                    momentum_form,
                    factor,
                } => {
                    let handle = evaluator::recurrence_intrinsic_direct::LoadedRecurrenceIntrinsicDirectExecutor::finalization_handle(&propagator).unwrap();
                    let crate::recurrence::direct_backend::DirectExecutorHandle::Finalization {
                        call,
                        context,
                    } = handle
                    else {
                        panic!("expected propagator")
                    };
                    let row = DirectFinalizationRow {
                        component_base: current,
                        component_count: component_count as u16,
                        momentum_form_id: momentum_form,
                        exact_factor_id: 0,
                        selector_domain_id: 0,
                        flags: 0,
                    };
                    let status = unsafe {
                        call(
                            context,
                            arena,
                            momentum_view,
                            DirectParameterView {
                                values_re: std::ptr::null(),
                                values_im: std::ptr::null(),
                                value_count: 0,
                            },
                            DirectFactorView {
                                values_re: &factor.re,
                                values_im: &factor.im,
                                value_count: 1,
                            },
                            &row,
                            1,
                            points as u32,
                        )
                    };
                    assert_eq!(status, 0);
                }
                DirectExportInstruction::Scale {
                    current,
                    component_count,
                    factor,
                } => {
                    for component in current..current + component_count {
                        for point in 0..points {
                            let index = component as usize * points + point;
                            let (re, im) = (cr[index], ci[index]);
                            cr[index] = factor.re * re - factor.im * im;
                            ci[index] = factor.re * im + factor.im * re;
                        }
                    }
                }
                DirectExportInstruction::Copy {
                    current,
                    source,
                    component_count,
                    factor,
                } => {
                    for component in 0..component_count {
                        for point in 0..points {
                            let destination = (current + component) as usize * points + point;
                            let index = (source + component) as usize * points + point;
                            let (re, im) = (cr[index], ci[index]);
                            if factor.re == 0.0 && factor.im == 0.0 {
                                cr[destination] = 0.0;
                                ci[destination] = 0.0;
                            } else {
                                cr[destination] = re * factor.re - im * factor.im;
                                ci[destination] = re * factor.im + im * factor.re;
                            }
                        }
                    }
                }
                DirectExportInstruction::Scatter {
                    current,
                    source,
                    component_count,
                    mode,
                    factor,
                } => {
                    for component in 0..component_count {
                        for point in 0..points {
                            let destination = (current + component) as usize * points + point;
                            let index = (source + component) as usize * points + point;
                            let (re, im) = (cr[index], ci[index]);
                            let (re, im) = if factor.im == 0.0 {
                                if factor.re == 1.0 {
                                    (re, im)
                                } else if factor.re == -1.0 {
                                    (-re, -im)
                                } else {
                                    (re * factor.re, im * factor.re)
                                }
                            } else if factor.re == 0.0 {
                                (-im * factor.im, re * factor.im)
                            } else {
                                (
                                    re * factor.re - im * factor.im,
                                    re * factor.im + im * factor.re,
                                )
                            };
                            if mode == DirectExportMode::Set {
                                cr[destination] = re;
                                ci[destination] = im;
                            } else {
                                cr[destination] += re;
                                ci[destination] += im;
                            }
                        }
                    }
                }
                DirectExportInstruction::Closure {
                    amplitude,
                    left,
                    right,
                    component_factors,
                    factor,
                } => {
                    for (component, coefficient) in component_factors.iter().enumerate() {
                        let sr = coefficient.re * factor.re - coefficient.im * factor.im;
                        let si = coefficient.re * factor.im + coefficient.im * factor.re;
                        for point in 0..points {
                            let l = (left as usize + component) * points + point;
                            let r = (right as usize + component) * points + point;
                            let pr = cr[l] * cr[r] - ci[l] * ci[r];
                            let pi = cr[l] * ci[r] + ci[l] * cr[r];
                            ar[amplitude as usize * points + point] += pr * sr - pi * si;
                            ai[amplitude as usize * points + point] += pr * si + pi * sr;
                        }
                    }
                }
                DirectExportInstruction::Kernel {
                    binding,
                    mode,
                    factor,
                    direct,
                } => {
                    use evaluator::symjit_plane::{SymjitPlaneKernel, SymjitPlaneLayout};
                    let application = kernels
                        .iter()
                        .find(|kernel| kernel.sha256 == binding.sha256)
                        .expect("export contains every referenced kernel");
                    let kernel = SymjitPlaneKernel::load_bytes(
                        &application.application,
                        PathBuf::from("exported.plane.symjit"),
                        SymjitPlaneLayout::complex(
                            binding.inputs.len() / 2,
                            binding.outputs.len() / 2,
                        ),
                    )
                    .unwrap();
                    let mut planes = binding
                        .inputs
                        .iter()
                        .map(|input| {
                            (0..points)
                                .map(|point| match *input {
                                    Input::Current {
                                        component,
                                        imaginary,
                                    } => {
                                        if imaginary {
                                            ci[component as usize * points + point]
                                        } else {
                                            cr[component as usize * points + point]
                                        }
                                    }
                                    Input::Amplitude {
                                        amplitude,
                                        imaginary,
                                    } => {
                                        if imaginary {
                                            ai[amplitude as usize * points + point]
                                        } else {
                                            ar[amplitude as usize * points + point]
                                        }
                                    }
                                    Input::Momentum { form, component } => {
                                        momenta[(form as usize * 4 + component as usize) * points
                                            + point]
                                    }
                                    Input::Literal { value } => value,
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    planes.extend(binding.outputs.iter().map(|_| vec![f64::NAN; points]));
                    {
                        let mut table = kernel
                            .build_table(planes.iter_mut().map(Vec::as_mut_slice).collect())
                            .unwrap();
                        kernel.execute_table(&mut table, 0, points).unwrap();
                    }
                    for (component, outputs) in binding.outputs.chunks_exact(2).enumerate() {
                        for point in 0..points {
                            let re = planes[binding.inputs.len() + component * 2][point];
                            let im = planes[binding.inputs.len() + component * 2 + 1][point];
                            let (re, im) = if direct {
                                (re, im)
                            } else {
                                (
                                    re * factor.re - im * factor.im,
                                    re * factor.im + im * factor.re,
                                )
                            };
                            for (output, value) in outputs.iter().zip([re, im]) {
                                let target = match (output.amplitude, output.imaginary) {
                                    (false, false) => &mut cr,
                                    (false, true) => &mut ci,
                                    (true, false) => &mut ar,
                                    (true, true) => &mut ai,
                                };
                                let target =
                                    &mut target[output.component as usize * points + point];
                                if mode == DirectExportMode::Set {
                                    *target = value;
                                } else {
                                    *target += value;
                                }
                            }
                        }
                    }
                }
            }
        }
        for reduction in &run.reduction {
            for point in 0..points {
                let index = reduction.amplitude as usize * points + point;
                let (mut re, mut im) = (ar[index], ai[index]);
                if reduction.apply_amplitude_scale {
                    (re, im) = (
                        re * reduction.amplitude_scale.re - im * reduction.amplitude_scale.im,
                        re * reduction.amplitude_scale.im + im * reduction.amplitude_scale.re,
                    );
                }
                output[point * program.helicity_outputs.len() + reduction.helicity_output] +=
                    re.mul_add(re, im * im) * reduction.weight;
            }
        }
    }
    if program.group_factor.numerator != program.group_factor.denominator {
        for value in &mut output {
            *value = *value * program.group_factor.numerator as f64
                / program.group_factor.denominator as f64;
        }
    }
    output
}

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_generated_artifact_matches_every_helicity_bit() {
    let Some(root) = std::env::var_os("RUSTICOL_RECURRENCE_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None).unwrap();
    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value = serde_json::from_slice(&fs::read(validation_path).unwrap()).unwrap();
    let base = validation["points"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|leg| {
            std::array::from_fn::<_, 4, _>(|component| {
                leg["momentum"][component]
                    .as_str()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let points = 5;
    let momenta = (0..points)
        .flat_map(|point| {
            base.iter().flat_map(move |leg| {
                leg.iter()
                    .map(move |value| value * (1.0 + point as f64 * 0.17))
            })
        })
        .collect::<Vec<_>>();
    let export = runtime.export_recurrence(None).unwrap();
    let json = serde_json::to_vec(&export).unwrap();
    assert!(!json.is_empty());
    for program in &export.programs {
        let expected = runtime
            .evaluate_resolved_f64(&momenta, points, None, Some(&[program.colour_id.clone()]))
            .unwrap();
        let actual = interpret_recurrence_export(program, &export.kernels, &momenta, points);
        assert_eq!(expected.color_ids.len(), 1);
        assert_eq!(actual.len(), expected.values.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected.values).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{} {} index {index}: {actual:?} != {expected:?}",
                program.process_key,
                program.colour_id
            );
        }
    }
}
