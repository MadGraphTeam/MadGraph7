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
    let (body, _, _) = export_direct_body(&plan, &groups, &[], &executors, None).unwrap();
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
    let body: Vec<DirectExportInstruction> =
        serde_json::from_slice(&serde_json::to_vec(&program.body).unwrap()).unwrap();
    let mut loaded_kernels = BTreeMap::new();
    for run in &program.runs {
        for instruction in body.iter().cloned() {
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
                    let source = &run.sources[source as usize];
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
                    destination,
                    amplitude,
                    left,
                    right,
                    left_momentum,
                    right_momentum,
                    mode,
                    factor,
                } => {
                    use crate::recurrence::direct_export::Operands;
                    let binding = &program.kernel_bindings[binding as usize];
                    let direct = binding.direct;
                    let operands = Operands {
                        destination,
                        left: left.unwrap_or(DIRECT_NONE_U32),
                        right: right.unwrap_or(DIRECT_NONE_U32),
                        left_momentum: left_momentum.unwrap_or(DIRECT_NONE_U32),
                        right_momentum: right_momentum.unwrap_or(DIRECT_NONE_U32),
                    };
                    let inputs = binding
                        .inputs
                        .iter()
                        .map(|input| input.resolve(&operands, factor).unwrap())
                        .collect::<Vec<_>>();
                    let outputs = binding
                        .outputs
                        .iter()
                        .map(|output| output.resolve(&operands, factor).unwrap())
                        .collect::<Vec<_>>();
                    assert!(outputs.iter().all(|output| matches!(
                        (output, amplitude),
                        (Input::Current { .. }, false) | (Input::Amplitude { .. }, true)
                    )));
                    use evaluator::symjit_plane::{SymjitPlaneKernel, SymjitPlaneLayout};
                    let application = kernels
                        .iter()
                        .find(|kernel| kernel.sha256 == binding.sha256)
                        .expect("export contains every referenced kernel");
                    let kernel =
                        loaded_kernels
                            .entry(binding.sha256.clone())
                            .or_insert_with(|| {
                                use base64::Engine as _;
                                SymjitPlaneKernel::load_bytes(
                                    &base64::engine::general_purpose::STANDARD
                                        .decode(&application.application_base64)
                                        .unwrap(),
                                    PathBuf::from("exported.plane.symjit"),
                                    SymjitPlaneLayout::complex(inputs.len() / 2, outputs.len() / 2),
                                )
                                .unwrap()
                            });
                    let mut planes = inputs
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
                    planes.extend(outputs.iter().map(|_| vec![f64::NAN; points]));
                    {
                        let mut table = kernel
                            .build_table(planes.iter_mut().map(Vec::as_mut_slice).collect())
                            .unwrap();
                        kernel.execute_table(&mut table, 0, points).unwrap();
                    }
                    for (component, outputs) in outputs.chunks_exact(2).enumerate() {
                        for point in 0..points {
                            let re = planes[inputs.len() + component * 2][point];
                            let im = planes[inputs.len() + component * 2 + 1][point];
                            let (re, im) = if direct {
                                (re, im)
                            } else {
                                (
                                    re * factor.re - im * factor.im,
                                    re * factor.im + im * factor.re,
                                )
                            };
                            for (output, value) in outputs.iter().zip([re, im]) {
                                let (target, component) = match *output {
                                    Input::Current {
                                        component,
                                        imaginary,
                                    } => (if imaginary { &mut ci } else { &mut cr }, component),
                                    Input::Amplitude {
                                        amplitude,
                                        imaginary,
                                    } => (if imaginary { &mut ai } else { &mut ar }, amplitude),
                                    _ => unreachable!("validated output arena plane"),
                                };
                                let target = &mut target[component as usize * points + point];
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
fn arithmetic_program(
    body: Vec<DirectExportInstruction>,
    sources: Vec<crate::recurrence::direct_export::DirectExportSource>,
) -> RecurrenceExportProgram {
    use super::recurrence_export::{
        RecurrenceExportHelicity, RecurrenceExportMomentumTerm, RecurrenceExportReduction,
        RecurrenceExportRun,
    };
    RecurrenceExportProgram {
        process_key: "test".into(),
        process: "test".into(),
        representative_process_key: "test".into(),
        representative_colour_id: "flow:0".into(),
        colour_id: "flow:0".into(),
        external_count: 1,
        legs: vec![],
        caller_legs: vec![],
        crossing: None,
        momentum_forms: vec![vec![RecurrenceExportMomentumTerm {
            leg: 0,
            coefficient: 1.0,
        }]],
        currents: vec![],
        current_component_count: 64,
        amplitude_count: 1,
        scales: vec![],
        factors: vec![],
        helicity_outputs: vec![RecurrenceExportHelicity {
            id: "h:0".into(),
            helicities: vec![0],
        }],
        body,
        kernel_bindings: vec![],
        runs: vec![RecurrenceExportRun {
            sources,
            reduction: vec![RecurrenceExportReduction {
                helicity_output: 0,
                amplitude: 0,
                amplitude_scale: ExportComplex::ONE,
                apply_amplitude_scale: false,
                weight: 1.0,
            }],
        }],
        group_factor: NativeRationalFactor {
            numerator: 1,
            denominator: 1,
        },
        alpha_s: None,
    }
}

#[cfg(feature = "f64-symjit")]
fn scalar_source(
    current: u32,
    factor: ExportComplex,
) -> crate::recurrence::direct_export::DirectExportSource {
    crate::recurrence::direct_export::DirectExportSource {
        current,
        component_count: 1,
        family: "scalar".into(),
        orientation: "self_conjugate".into(),
        leg: 0,
        momentum_form: 0,
        helicity: 0,
        chirality: 0,
        mass: ExportComplex { re: 0.0, im: 0.0 },
        factor,
        embedding: vec![],
    }
}

#[cfg(feature = "f64-symjit")]
fn scalar_closure(left: u32, right: u32) -> DirectExportInstruction {
    DirectExportInstruction::Closure {
        amplitude: 0,
        left,
        right,
        component_factors: vec![ExportComplex::ONE],
        factor: ExportComplex::ONE,
    }
}

/// A one-point arena driven only by RustiCol's own executors and reduction
/// helpers. The arithmetic tests take their expectations from it, so they
/// check the interpreter's documented arithmetic against the runtime rather
/// than against hand-written formulas.
#[cfg(feature = "f64-symjit")]
struct ProductionPoint {
    current_re: Vec<f64>,
    current_im: Vec<f64>,
    amplitude_re: Vec<f64>,
    amplitude_im: Vec<f64>,
    momenta: Vec<f64>,
}

#[cfg(feature = "f64-symjit")]
impl ProductionPoint {
    fn new(momentum: [f64; 4]) -> Self {
        Self {
            current_re: vec![f64::NAN; 64],
            current_im: vec![f64::NAN; 64],
            amplitude_re: vec![0.0],
            amplitude_im: vec![0.0],
            momenta: momentum.to_vec(),
        }
    }

    fn views(
        &mut self,
        factors_re: &[f64],
        factors_im: &[f64],
    ) -> (
        DirectArenaView,
        DirectMomentumView,
        DirectParameterView,
        DirectFactorView,
    ) {
        (
            DirectArenaView {
                current_re: self.current_re.as_mut_ptr(),
                current_im: self.current_im.as_mut_ptr(),
                current_scalar_len: self.current_re.len() as u64,
                amplitude_re: self.amplitude_re.as_mut_ptr(),
                amplitude_im: self.amplitude_im.as_mut_ptr(),
                amplitude_scalar_len: self.amplitude_re.len() as u64,
                point_stride: 1,
            },
            DirectMomentumView {
                values: self.momenta.as_ptr(),
                scalar_len: self.momenta.len() as u64,
                form_count: 1,
                lorentz_component_count: 4,
                point_stride: 1,
            },
            DirectParameterView {
                values_re: std::ptr::null(),
                values_im: std::ptr::null(),
                value_count: 0,
            },
            DirectFactorView {
                values_re: factors_re.as_ptr(),
                values_im: factors_im.as_ptr(),
                value_count: factors_re.len() as u32,
            },
        )
    }

    /// RustiCol's source executor, for a massless source of `family`.
    fn source(
        &mut self,
        current: u32,
        family: evaluator::recurrence_source_direct::DirectSourceWavefunctionFamily,
        helicity: i32,
        factor: ExportComplex,
    ) {
        use evaluator::recurrence_source_direct::{
            DirectSourceDispatchDomainSpec, DirectSourceDispatchKey,
            DirectSourceDispatchVariantSpec, DirectSourceOrientation, DirectSourceTemplateSpec,
            LoadedDirectSourceExecutor,
        };
        let executor = LoadedDirectSourceExecutor::load(vec![DirectSourceDispatchDomainSpec {
            variants: vec![DirectSourceDispatchVariantSpec {
                key: DirectSourceDispatchKey::SpinStateClass(helicity),
                template: DirectSourceTemplateSpec {
                    spin_state_class: helicity,
                    family,
                    orientation: DirectSourceOrientation::SelfConjugate,
                    helicity,
                    chirality: 0,
                    mass_parameter_index: None,
                },
            }],
        }])
        .unwrap();
        let handle = executor.handle();
        let row = crate::recurrence::DirectSourceRow {
            source_slot: 0,
            destination_component_base: current,
            momentum_form_id: 0,
            source_template_or_dispatch_domain: 0,
            spin_state_class: helicity,
            exact_factor_id: 0,
            selector_domain_id: 0,
        };
        let (factor_re, factor_im) = ([factor.re], [factor.im]);
        let (arena, momenta, parameters, factors) = self.views(&factor_re, &factor_im);
        let status = unsafe {
            (handle.call)(
                handle.context,
                arena,
                momenta,
                parameters,
                factors,
                &row,
                1,
                1,
            )
        };
        assert_eq!(status, 0);
    }

    /// RustiCol's identity finalization, which the export writes as `scale`.
    fn scale(&mut self, current: u32, factor: ExportComplex) {
        let row = crate::recurrence::DirectFinalizationRow {
            component_base: current,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let (factor_re, factor_im) = ([factor.re], [factor.im]);
        let (arena, momenta, parameters, factors) = self.views(&factor_re, &factor_im);
        let status = unsafe {
            evaluator::recurrence_intrinsic_direct::execute_identity_finalization_rows(
                std::ptr::null(),
                arena,
                momenta,
                parameters,
                factors,
                &row,
                1,
                1,
            )
        };
        assert_eq!(status, 0);
    }

    /// RustiCol's certified-reuse executor, which the export writes as `copy`.
    fn copy(&mut self, destination: u32, source: u32, factor: ExportComplex) {
        use crate::recurrence::{
            DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE,
            DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DIRECT_NONE_U32,
        };
        let row = crate::recurrence::DirectContributionRow {
            parent0_component_base: source,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            // Certified-reuse rows store their component count here.
            parent0_momentum_form_id: 1,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: destination,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE
                | DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
        };
        let (factor_re, factor_im) = ([factor.re], [factor.im]);
        let mut workspace = crate::recurrence::direct_backend::DirectWorkspace {
            current_re: &mut self.current_re,
            current_im: &mut self.current_im,
            amplitude_re: &mut self.amplitude_re,
            amplitude_im: &mut self.amplitude_im,
            momenta: &self.momenta,
            momentum_form_count: 1,
            lorentz_component_count: 4,
            parameters_re: &[],
            parameters_im: &[],
            factors_re: &factor_re,
            factors_im: &factor_im,
            point_stride: 1,
        };
        crate::recurrence::direct_backend::execute_certified_reuse_rows(&[row], &mut workspace, 1)
            .unwrap();
    }

    /// RustiCol's closure executor, accumulating into amplitude 0.
    fn closure(
        &mut self,
        left: u32,
        right: u32,
        component_factors: &[ExportComplex],
        factor: ExportComplex,
    ) {
        let factor_re = std::iter::once(factor.re)
            .chain(component_factors.iter().map(|value| value.re))
            .collect::<Vec<_>>();
        let factor_im = std::iter::once(factor.im)
            .chain(component_factors.iter().map(|value| value.im))
            .collect::<Vec<_>>();
        let row = crate::recurrence::DirectClosureRow {
            parent0_component_base: left,
            parent1_component_base_or_sentinel: right,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            amplitude_destination_id: 0,
            exact_factor_id: 0,
            component_factor_start: 1,
            component_count: component_factors.len() as u16,
            selector_domain_id: 0,
            flags: 0,
        };
        let (arena, momenta, parameters, factors) = self.views(&factor_re, &factor_im);
        let status = unsafe {
            evaluator::recurrence_closure_direct::execute_closure_reduce_rows(
                std::ptr::null(),
                arena,
                momenta,
                parameters,
                factors,
                &row,
                1,
                1,
            )
        };
        assert_eq!(status, 0);
    }

    /// RustiCol's replay scale, resolved reduction and group factor.
    fn reduce(
        &self,
        amplitude_scale: Option<ExportComplex>,
        weight: f64,
        group_factor: NativeRationalFactor,
    ) -> f64 {
        let (mut re, mut im) = (self.amplitude_re[0], self.amplitude_im[0]);
        if let Some(scale) = amplitude_scale {
            (re, im) =
                crate::recurrence::direct_runtime::apply_replay_scale(re, im, scale.re, scale.im);
        }
        let mut output = [0.0];
        super::recurrence_lane::accumulate_lc_diagonal_amplitude(
            1,
            weight,
            |_| (re, im),
            |_, value| output[0] += value,
        );
        super::native_runtime::apply_rational_factor(&mut output, group_factor);
        output[0]
    }
}

#[cfg(feature = "f64-symjit")]
const UNIT_GROUP_FACTOR: NativeRationalFactor = NativeRationalFactor {
    numerator: 1,
    denominator: 1,
};

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_arithmetic_copy_zero_in_place_scale_and_nonunit_copy() {
    // Read an uninitialized source: certified zero reuse must write +0
    // directly, rather than propagating NaN through multiplication by zero.
    let program = arithmetic_program(
        vec![
            DirectExportInstruction::ClearAmplitude { amplitude: 0 },
            DirectExportInstruction::Source { source: 0 },
            DirectExportInstruction::Copy {
                current: 1,
                source: 2,
                component_count: 1,
                factor: ExportComplex { re: 0.0, im: 0.0 },
            },
            scalar_closure(1, 0),
        ],
        vec![scalar_source(0, ExportComplex::ONE)],
    );
    let momentum = [3.0, 0.2, 0.3, 0.4];
    let result = interpret_recurrence_export(&program, &[], &momentum, 1);
    let mut production = ProductionPoint::new(momentum);
    production.source(0, scalar_family(), 0, ExportComplex::ONE);
    // Current 2 is never written, so RustiCol's copy reads NaN as well.
    production.copy(1, 2, ExportComplex { re: 0.0, im: 0.0 });
    production.closure(1, 0, &[ExportComplex::ONE], ExportComplex::ONE);
    let expected = production.reduce(None, 1.0, UNIT_GROUP_FACTOR);
    assert_eq!(result[0].to_bits(), expected.to_bits());
    assert_eq!(expected.to_bits(), 0.0_f64.to_bits());
    let input = ExportComplex { re: 2.0, im: -3.0 };
    let scale = ExportComplex { re: -0.25, im: 0.5 };
    let copy = ExportComplex { re: 0.5, im: -0.2 };
    let program = arithmetic_program(
        vec![
            DirectExportInstruction::ClearAmplitude { amplitude: 0 },
            DirectExportInstruction::Source { source: 0 },
            DirectExportInstruction::Source { source: 1 },
            DirectExportInstruction::Scale {
                current: 0,
                component_count: 1,
                factor: scale,
            },
            DirectExportInstruction::Copy {
                current: 1,
                source: 0,
                component_count: 1,
                factor: copy,
            },
            scalar_closure(1, 2),
        ],
        vec![
            scalar_source(0, input),
            scalar_source(2, ExportComplex::ONE),
        ],
    );
    let mut production = ProductionPoint::new(momentum);
    production.source(0, scalar_family(), 0, input);
    production.source(2, scalar_family(), 0, ExportComplex::ONE);
    production.scale(0, scale);
    production.copy(1, 0, copy);
    production.closure(1, 2, &[ExportComplex::ONE], ExportComplex::ONE);
    let expected = production.reduce(None, 1.0, UNIT_GROUP_FACTOR);
    let result = interpret_recurrence_export(&program, &[], &momentum, 1);
    assert_eq!(result[0].to_bits(), expected.to_bits());
}

#[cfg(feature = "f64-symjit")]
fn scalar_family() -> evaluator::recurrence_source_direct::DirectSourceWavefunctionFamily {
    evaluator::recurrence_source_direct::DirectSourceWavefunctionFamily::Scalar
}

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_arithmetic_imaginary_and_general_scatter_set_add() {
    use crate::recurrence::direct_backend::DirectContributionFanoutScale;
    use crate::recurrence::direct_export::DirectExportMode;
    let input = ExportComplex { re: 2.0, im: -3.0 };
    let initial = ExportComplex { re: 0.1, im: -0.5 };
    for factor in [
        ExportComplex { re: 0.0, im: 0.75 },
        ExportComplex { re: 0.7, im: -0.2 },
    ] {
        for mode in [DirectExportMode::Set, DirectExportMode::Add] {
            let program = arithmetic_program(
                vec![
                    DirectExportInstruction::ClearAmplitude { amplitude: 0 },
                    DirectExportInstruction::Source { source: 0 },
                    DirectExportInstruction::Source { source: 1 },
                    DirectExportInstruction::Source { source: 2 },
                    DirectExportInstruction::Scatter {
                        current: 1,
                        source: 0,
                        component_count: 1,
                        mode,
                        factor,
                    },
                    scalar_closure(1, 2),
                ],
                vec![
                    scalar_source(0, input),
                    scalar_source(1, initial),
                    scalar_source(2, ExportComplex::ONE),
                ],
            );
            // RustiCol's fanout scale shortcuts, then the set or add that
            // `scatter_contribution_fanout_run` applies to the destination row.
            let momentum = [3.0, 0.2, 0.3, 0.4];
            let mut production = ProductionPoint::new(momentum);
            production.source(0, scalar_family(), 0, input);
            production.source(1, scalar_family(), 0, initial);
            production.source(2, scalar_family(), 0, ExportComplex::ONE);
            let (re, im) = DirectContributionFanoutScale::new(factor.re, factor.im)
                .apply(production.current_re[0], production.current_im[0]);
            if mode == DirectExportMode::Set {
                (production.current_re[1], production.current_im[1]) = (re, im);
            } else {
                production.current_re[1] += re;
                production.current_im[1] += im;
            }
            production.closure(1, 2, &[ExportComplex::ONE], ExportComplex::ONE);
            assert_eq!(
                interpret_recurrence_export(&program, &[], &momentum, 1)[0].to_bits(),
                production.reduce(None, 1.0, UNIT_GROUP_FACTOR).to_bits()
            );
        }
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_arithmetic_spin2_source_and_replay_multiplicity() {
    let p = [3.0, 0.2, 0.3, 2.9];
    let mut source = scalar_source(0, ExportComplex::ONE);
    source.family = "spin2".into();
    source.component_count = 16;
    source.helicity = 2;
    let coefficients = (0..16)
        .map(|index| ExportComplex {
            re: (index + 1) as f64,
            im: 0.0,
        })
        .collect::<Vec<_>>();
    let program = arithmetic_program(
        vec![
            DirectExportInstruction::ClearAmplitude { amplitude: 0 },
            DirectExportInstruction::Source { source: 0 },
            DirectExportInstruction::Closure {
                amplitude: 0,
                left: 0,
                right: 0,
                component_factors: coefficients.clone(),
                factor: ExportComplex::ONE,
            },
        ],
        vec![source],
    );
    let mut production = ProductionPoint::new(p);
    production.source(
        0,
        evaluator::recurrence_source_direct::DirectSourceWavefunctionFamily::Spin2,
        2,
        ExportComplex::ONE,
    );
    production.closure(0, 0, &coefficients, ExportComplex::ONE);
    assert_eq!(
        interpret_recurrence_export(&program, &[], &p, 1)[0].to_bits(),
        production.reduce(None, 1.0, UNIT_GROUP_FACTOR).to_bits()
    );
    let mut replay = arithmetic_program(
        vec![
            DirectExportInstruction::ClearAmplitude { amplitude: 0 },
            DirectExportInstruction::Source { source: 0 },
            DirectExportInstruction::Source { source: 1 },
            scalar_closure(0, 1),
        ],
        vec![
            scalar_source(0, ExportComplex { re: 2.0, im: -3.0 }),
            scalar_source(1, ExportComplex::ONE),
        ],
    );
    // Phase i and multiplicity 3, as `DirectReplaySelectorPlan::amplitude_scale`
    // forms them.
    let amplitude_scale = ExportComplex { re: 0.0, im: 3.0 };
    let group_factor = NativeRationalFactor {
        numerator: 2,
        denominator: 3,
    };
    replay.runs[0].reduction[0].apply_amplitude_scale = true;
    replay.runs[0].reduction[0].amplitude_scale = amplitude_scale;
    replay.runs[0].reduction[0].weight = 1.25;
    replay.group_factor = group_factor;
    let mut production = ProductionPoint::new(p);
    production.source(0, scalar_family(), 0, ExportComplex { re: 2.0, im: -3.0 });
    production.source(1, scalar_family(), 0, ExportComplex::ONE);
    production.closure(0, 1, &[ExportComplex::ONE], ExportComplex::ONE);
    let expected = production.reduce(Some(amplitude_scale), 1.25, group_factor);
    assert_eq!(
        interpret_recurrence_export(&replay, &[], &p, 1)[0].to_bits(),
        expected.to_bits()
    );
    assert_eq!(expected, 97.5);
}

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_direct_kernel_ignores_factor_and_preserves_set_add() {
    use crate::recurrence::direct_export::{
        DirectExportKernelBinding, DirectExportMode, RelativeKernelInput,
    };
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let bytes = evaluator::symjit_plane::compile_symbolica_program_to_plane_application_bytes(
        "([('assign', ('out', 0), ('param', 0))], 0, [])",
        1,
        1,
        2,
        false,
    )
    .unwrap();
    let kernels = [RecurrenceExportKernel {
        sha256: "test".into(),
        application_base64: STANDARD.encode(bytes),
    }];
    for mode in [DirectExportMode::Set, DirectExportMode::Add] {
        let mut program = arithmetic_program(
            vec![
                DirectExportInstruction::ClearAmplitude { amplitude: 0 },
                DirectExportInstruction::Source { source: 0 },
                DirectExportInstruction::Source { source: 1 },
                DirectExportInstruction::Source { source: 2 },
                DirectExportInstruction::Kernel {
                    binding: 0,
                    destination: 1,
                    amplitude: false,
                    left: Some(0),
                    right: None,
                    left_momentum: None,
                    right_momentum: None,
                    mode,
                    factor: ExportComplex { re: 4.0, im: -7.0 },
                },
                scalar_closure(1, 2),
            ],
            vec![
                scalar_source(0, ExportComplex { re: 2.0, im: -3.0 }),
                scalar_source(1, ExportComplex { re: 0.1, im: -0.5 }),
                scalar_source(2, ExportComplex::ONE),
            ],
        );
        program.kernel_bindings.push(DirectExportKernelBinding {
            id: 0,
            sha256: "test".into(),
            inputs: vec![
                RelativeKernelInput::Parent {
                    parent: 0,
                    component: 0,
                    imaginary: false,
                },
                RelativeKernelInput::Parent {
                    parent: 0,
                    component: 0,
                    imaginary: true,
                },
            ],
            outputs: vec![
                RelativeKernelInput::Destination {
                    component: 0,
                    imaginary: false,
                    amplitude: false,
                },
                RelativeKernelInput::Destination {
                    component: 0,
                    imaginary: true,
                    amplitude: false,
                },
            ],
            direct: true,
        });
        let momentum = [3.0, 0.2, 0.3, 0.4];
        let accumulate = mode == DirectExportMode::Add;
        // The saved application copies its input planes, so the kernel's raw
        // outputs are the parent current's values.
        let expected = |direct: bool| {
            let mut production = ProductionPoint::new(momentum);
            production.source(0, scalar_family(), 0, ExportComplex { re: 2.0, im: -3.0 });
            production.source(1, scalar_family(), 0, ExportComplex { re: 0.1, im: -0.5 });
            production.source(2, scalar_family(), 0, ExportComplex::ONE);
            let (raw_re, raw_im) = ([production.current_re[0]], [production.current_im[0]]);
            if direct {
                // RustiCol's direct tables store raw outputs. It marks only
                // finalization rows direct, and those always set; adding raw
                // outputs is the documented consumer semantics.
                if accumulate {
                    production.current_re[1] += raw_re[0];
                    production.current_im[1] += raw_im[0];
                } else {
                    production.current_re[1] = raw_re[0];
                    production.current_im[1] = raw_im[0];
                }
            } else {
                evaluator::symjit_direct::apply_complex_scaled(
                    &mut production.current_re[1..2],
                    &mut production.current_im[1..2],
                    &raw_re,
                    &raw_im,
                    4.0,
                    -7.0,
                    accumulate,
                );
            }
            production.closure(1, 2, &[ExportComplex::ONE], ExportComplex::ONE);
            production.reduce(None, 1.0, UNIT_GROUP_FACTOR)
        };
        assert_eq!(
            interpret_recurrence_export(&program, &kernels, &momentum, 1)[0].to_bits(),
            expected(true).to_bits()
        );
        // The same row with the adapter enabled must apply its factor.
        program.kernel_bindings[0].direct = false;
        assert_eq!(
            interpret_recurrence_export(&program, &kernels, &momentum, 1)[0].to_bits(),
            expected(false).to_bits()
        );
    }
}

#[cfg(feature = "f64-symjit")]
fn assert_export_values(program: &RecurrenceExportProgram, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{} {} index {index}: {actual:?} != {expected:?}",
            program.process_key,
            program.colour_id
        );
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn recurrence_export_generated_artifact_matches_every_helicity_bit() {
    // CI generates the artifacts in its integration step and names them here;
    // see .github/workflows/tests.yml.
    let Some(roots) = std::env::var_os("RUSTICOL_RECURRENCE_ARTIFACT") else {
        eprintln!(
            "skipped: set RUSTICOL_RECURRENCE_ARTIFACT to a generated LC recurrence artifact or a path list"
        );
        return;
    };
    for root in std::env::split_paths(&roots) {
        check_every_exported_process(&root);
    }
}

#[cfg(feature = "f64-symjit")]
fn check_every_exported_process(root: &Path) {
    // Match the CLI provider traversal: stored representatives, aliases, then
    // every process-group member, preserving first occurrence order.
    let process_ids: Vec<String> = if let Ok(ids) = std::env::var("RUSTICOL_RECURRENCE_PROCESSES") {
        serde_json::from_str(&ids)
            .expect("RUSTICOL_RECURRENCE_PROCESSES must be a JSON array of CLI process IDs")
    } else {
        // Read the manifest and the declared process-group payload through
        // the artifact's authenticated reader, not files on disk.
        let artifact = VerifiedArtifact::open(root).unwrap();
        let mut ids = Vec::new();
        let mut seen = BTreeSet::new();
        let mut add = |id: &str| {
            if seen.insert(id.to_owned()) {
                ids.push(id.to_owned());
            }
        };
        for process in &artifact.manifest().processes {
            add(&process.id);
            for alias in &process.aliases {
                add(&alias.id);
            }
        }
        if let Some(payload) = artifact
            .manifest()
            .payloads
            .iter()
            .find(|payload| payload.role == PayloadRole::ProcessGroups)
        {
            let groups: Value =
                serde_json::from_slice(&artifact.read_payload(&payload.path).unwrap()).unwrap();
            for group in groups["groups"].as_array().unwrap() {
                for member in group["members"].as_array().unwrap() {
                    add(member["process_id"].as_str().expect("provider process ID"));
                }
            }
        }
        ids
    };
    assert!(!process_ids.is_empty());
    let points = 5;
    let mut tested_programs = 0;
    let mut tested_values = 0;
    let mut representatives = BTreeMap::<String, RecurrenceExport>::new();
    for process_id in process_ids {
        let mut runtime = NativeRuntime::load(root, Some(&process_id), None).unwrap();
        let export = runtime.export_recurrence(None).unwrap();
        assert_eq!(export.schema_version, 2);
        let base =
            super::recurrence_integration_tests::generated_recurrence_validation_point(&runtime);
        let mut caller = base.clone();
        if let Some(crossing) = export
            .programs
            .first()
            .and_then(|program| program.crossing.as_ref())
        {
            for entry in crossing {
                caller[entry.caller_leg] = base[entry.program_leg].map(|value| value * entry.sign);
            }
        }
        let momenta = (0..points)
            .flat_map(|point| {
                caller.iter().flat_map(move |leg| {
                    leg.iter()
                        .map(move |value| value * (1.0 + point as f64 * 0.17))
                })
            })
            .collect::<Vec<_>>();
        for program in &export.programs {
            let expected = runtime
                .evaluate_resolved_f64(
                    &momenta,
                    points,
                    None,
                    Some(std::slice::from_ref(&program.colour_id)),
                )
                .unwrap();
            let actual = interpret_recurrence_export(program, &export.kernels, &momenta, points);
            assert_eq!(expected.color_ids.len(), 1);
            assert_export_values(program, &actual, &expected.values);
            // Validate the CLI member-reference contract independently: take
            // executable data exclusively from the named representative and
            // overlay exactly the metadata fields emitted by the CLI.
            if program.process_key != program.representative_process_key {
                if !representatives.contains_key(&program.representative_process_key) {
                    let mut representative =
                        NativeRuntime::load(root, Some(&program.representative_process_key), None)
                            .unwrap();
                    representatives.insert(
                        program.representative_process_key.clone(),
                        representative.export_recurrence(None).unwrap(),
                    );
                }
                let representative = &representatives[&program.representative_process_key];
                let mut referenced = representative
                    .programs
                    .iter()
                    .find(|candidate| candidate.colour_id == program.representative_colour_id)
                    .expect("member names an existing representative colour")
                    .clone();
                referenced.process_key = program.process_key.clone();
                referenced.process = program.process.clone();
                referenced.colour_id = program.colour_id.clone();
                referenced.representative_process_key = program.representative_process_key.clone();
                referenced.representative_colour_id = program.representative_colour_id.clone();
                referenced.caller_legs = program.caller_legs.clone();
                referenced.crossing = program.crossing.clone();
                referenced.group_factor = program.group_factor;
                referenced.helicity_outputs = program.helicity_outputs.clone();
                let actual = interpret_recurrence_export(
                    &referenced,
                    &representative.kernels,
                    &momenta,
                    points,
                );
                assert_export_values(&referenced, &actual, &expected.values);
            }
            tested_programs += 1;
            tested_values += actual.len();
        }
    }
    assert!(
        tested_programs > 0,
        "{} exported no programs",
        root.display()
    );
    eprintln!(
        "recurrence export parity for {}: {tested_programs} programs, {tested_values} per-helicity values, including all requested provider members",
        root.display()
    );
}
