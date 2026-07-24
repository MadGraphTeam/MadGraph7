// SPDX-License-Identifier: 0BSD

//! Developer-only end-to-end compiled Direct-Arena engine prototype.
//!
//! The production compiled lane remains unchanged. This module consumes the
//! existing stage/chunk manifests, lowers each already-fused SymJIT O3 leaf to
//! DirectApplication v3 at load time, and executes the unchanged leaf schedule
//! against persistent component-major planes. There is deliberately one
//! physical current plane per global value component in this first prototype:
//! that preserves the current schedule and parent locality while removing all
//! per-leaf gather and scatter traffic.

#![allow(dead_code)]

use std::path::PathBuf;
use std::ptr;

use num_complex::Complex;

use super::evaluator::symjit_compiled_direct::{
    BoundSymjitCompiledDirectStage, CompiledDirectArenaPlane, CompiledDirectOutputBinding,
    CompiledDirectPlaneBinding, CompiledDirectScalarBinding, CompiledDirectSourceInputBinding,
    LoadedSymjitCompiledDirectStage,
};
use super::*;
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaWorkspace, DirectMomentumView, DirectParameterView,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompiledDirectPrototypeTraffic {
    pub(crate) packet_bytes: u64,
    pub(crate) gather_bytes: u64,
    pub(crate) scatter_bytes: u64,
    pub(crate) remap_bytes: u64,
}

struct LoadedStage {
    leaves: Vec<LoadedSymjitCompiledDirectStage>,
}

struct BoundStage {
    leaves: Vec<BoundSymjitCompiledDirectStage>,
}

/// Persistent prototype executor for one compiled schedule.
///
/// The descriptor objects contain pointers into the boxed arena and aligned
/// side buffers. Those allocations never resize or move after construction.
pub(crate) struct CompiledDirectEnginePrototype {
    // Bound descriptors must drop before the allocations they address. Rust
    // drops fields in declaration order, so keep these owners first.
    stages: Vec<BoundStage>,
    amplitude: BoundStage,
    arena: Box<DirectArenaWorkspace>,
    momenta: Box<AlignedF64Buffer>,
    parameter_re: Box<AlignedF64Buffer>,
    parameter_im: Box<AlignedF64Buffer>,
    value_component_count: usize,
    momentum_component_count: usize,
    model_parameter_count: usize,
    amplitude_component_count: usize,
    traffic: CompiledDirectPrototypeTraffic,
}

impl CompiledDirectEnginePrototype {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        stages: &[GenericSerializedStageEvaluatorManifest],
        amplitude: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
        value_component_count: usize,
        momentum_component_count: usize,
        model_parameter_count: usize,
        amplitude_component_count: usize,
        tile_capacity: u32,
    ) -> RusticolResult<Self> {
        if value_component_count == 0 || amplitude_component_count == 0 {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena prototype requires current and amplitude planes",
            ));
        }
        if !momentum_component_count.is_multiple_of(4) {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena momentum component count must be a multiple of four",
            ));
        }
        let loaded_stages = stages
            .iter()
            .map(|stage| {
                load_stage(
                    stage,
                    false,
                    payloads,
                    value_component_count,
                    momentum_component_count,
                    amplitude_component_count,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let loaded_amplitude = load_stage(
            amplitude,
            true,
            payloads,
            value_component_count,
            momentum_component_count,
            amplitude_component_count,
        )?;

        let mut arena = Box::new(DirectArenaWorkspace::new(
            u32::try_from(value_component_count).map_err(|_| {
                RusticolError::integrity("compiled Direct-Arena current plane count exceeds u32")
            })?,
            u32::try_from(amplitude_component_count).map_err(|_| {
                RusticolError::integrity("compiled Direct-Arena amplitude plane count exceeds u32")
            })?,
            tile_capacity,
        )?);
        arena.begin_tile(tile_capacity)?;
        let stride = arena.point_stride() as usize;
        let momenta_len = momentum_component_count
            .checked_mul(stride)
            .ok_or_else(|| RusticolError::integrity("compiled momentum arena length overflows"))?;
        let mut momenta = Box::new(AlignedF64Buffer::zeroed(
            momenta_len,
            "compiled prototype momenta",
        )?);
        let mut parameter_re = Box::new(AlignedF64Buffer::zeroed(
            model_parameter_count,
            "compiled prototype parameter real",
        )?);
        let mut parameter_im = Box::new(AlignedF64Buffer::zeroed(
            model_parameter_count,
            "compiled prototype parameter imaginary",
        )?);

        let arena_view = arena.view()?;
        let momentum_view = DirectMomentumView {
            values: momenta.as_mut_ptr(),
            scalar_len: momenta.len() as u64,
            form_count: u32::try_from(momentum_component_count / 4).map_err(|_| {
                RusticolError::integrity("compiled momentum form count exceeds u32")
            })?,
            lorentz_component_count: 4,
            point_stride: arena_view.point_stride,
        };
        let parameter_view = DirectParameterView {
            values_re: if parameter_re.is_empty() {
                ptr::null()
            } else {
                parameter_re.as_mut_ptr()
            },
            values_im: if parameter_im.is_empty() {
                ptr::null()
            } else {
                parameter_im.as_mut_ptr()
            },
            value_count: u32::try_from(model_parameter_count).map_err(|_| {
                RusticolError::integrity("compiled model parameter count exceeds u32")
            })?,
        };

        // SAFETY: arena and side buffers are fixed-size heap allocations owned
        // by the returned executor. No field resizes them, and evaluate methods
        // require exclusive access to the executor.
        let bind_stage = |loaded: LoadedStage| -> RusticolResult<BoundStage> {
            Ok(BoundStage {
                leaves: loaded
                    .leaves
                    .into_iter()
                    .map(|leaf| unsafe { leaf.bind(arena_view, momentum_view, parameter_view) })
                    .collect::<RusticolResult<Vec<_>>>()?,
            })
        };
        let stages = loaded_stages
            .into_iter()
            .map(bind_stage)
            .collect::<RusticolResult<Vec<_>>>()?;
        let amplitude = bind_stage(loaded_amplitude)?;

        Ok(Self {
            stages,
            amplitude,
            arena,
            momenta,
            parameter_re,
            parameter_im,
            value_component_count,
            momentum_component_count,
            model_parameter_count,
            amplitude_component_count,
            traffic: CompiledDirectPrototypeTraffic::default(),
        })
    }

    pub(crate) const fn traffic(&self) -> CompiledDirectPrototypeTraffic {
        self.traffic
    }

    /// Transpose the outer row-major state once at the engine boundary.
    ///
    /// This is not leaf gather traffic: every compiled leaf subsequently
    /// consumes and produces the authoritative arena planes directly.
    pub(crate) fn begin_tile_from_state(
        &mut self,
        point_count: usize,
        global_parameter_count: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<()> {
        if point_count == 0
            || point_count > self.arena.tile_capacity() as usize
            || state.len() != point_count.saturating_mul(global_parameter_count)
        {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena prototype state shape is invalid",
            ));
        }
        let required_parameters = self
            .value_component_count
            .checked_add(self.momentum_component_count)
            .and_then(|value| value.checked_add(self.model_parameter_count))
            .ok_or_else(|| RusticolError::integrity("compiled parameter layout overflows"))?;
        if global_parameter_count < required_parameters {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena prototype state omits required parameters",
            ));
        }

        self.arena.begin_tile(point_count as u32)?;
        let stride = self.arena.point_stride() as usize;
        {
            let (current_re, current_im, amplitude_re, amplitude_im) =
                self.arena.split_slices_mut();
            for component in 0..self.value_component_count {
                let plane = component * stride;
                for point in 0..point_count {
                    let value = state[point * global_parameter_count + component];
                    current_re[plane + point] = value.re;
                    current_im[plane + point] = value.im;
                }
            }
            for component in 0..self.amplitude_component_count {
                let plane = component * stride;
                amplitude_re[plane..plane + point_count].fill(0.0);
                amplitude_im[plane..plane + point_count].fill(0.0);
            }
        }
        let momentum_start = self.value_component_count;
        for component in 0..self.momentum_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                self.momenta.as_mut_slice()[plane + point] =
                    state[point * global_parameter_count + momentum_start + component].re;
            }
        }
        let parameter_start = momentum_start + self.momentum_component_count;
        if self.model_parameter_count != 0 {
            for index in 0..self.model_parameter_count {
                let value = state[parameter_start + index];
                self.parameter_re.as_mut_slice()[index] = value.re;
                self.parameter_im.as_mut_slice()[index] = value.im;
                for point in 1..point_count {
                    if state[point * global_parameter_count + parameter_start + index] != value {
                        return Err(RusticolError::invalid_argument(
                            "compiled Direct-Arena model parameters vary across a point tile",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the existing full schedule. No input packet, gather, output
    /// scatter, remap, or allocation occurs in this method.
    pub(crate) fn evaluate_all(&self, point_count: usize) -> RusticolResult<()> {
        let active = point_count_u32(point_count, self.arena.active_point_count())?;
        for stage in &self.stages {
            evaluate_bound_stage(stage, active, None)?;
        }
        evaluate_bound_stage(&self.amplitude, active, None)
    }

    /// Execute an unchanged selector-pruned leaf schedule.
    pub(crate) fn evaluate_schedule(
        &self,
        point_count: usize,
        active_stage_leaves: &[Vec<usize>],
        active_amplitude_leaves: &[usize],
    ) -> RusticolResult<()> {
        if active_stage_leaves.len() != self.stages.len() {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena selector schedule has the wrong stage count",
            ));
        }
        let active = point_count_u32(point_count, self.arena.active_point_count())?;
        for (stage, selected) in self.stages.iter().zip(active_stage_leaves) {
            evaluate_bound_stage(stage, active, Some(selected))?;
        }
        evaluate_bound_stage(&self.amplitude, active, Some(active_amplitude_leaves))
    }

    pub(crate) fn copy_current_to_state(
        &self,
        point_count: usize,
        global_parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if state.len() != point_count.saturating_mul(global_parameter_count) {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena output state shape is invalid",
            ));
        }
        let stride = self.arena.point_stride() as usize;
        let (current_re, current_im) = self.arena.current_slices();
        for component in 0..self.value_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                state[point * global_parameter_count + component] =
                    Complex::new(current_re[plane + point], current_im[plane + point]);
            }
        }
        Ok(())
    }

    pub(crate) fn amplitude_values(
        &self,
        point_count: usize,
        output: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if output.len() != point_count.saturating_mul(self.amplitude_component_count) {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena amplitude output shape is invalid",
            ));
        }
        let stride = self.arena.point_stride() as usize;
        let (amplitude_re, amplitude_im) = self.arena.amplitude_slices();
        for component in 0..self.amplitude_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                output[point * self.amplitude_component_count + component] =
                    Complex::new(amplitude_re[plane + point], amplitude_im[plane + point]);
            }
        }
        Ok(())
    }
}

fn point_count_u32(point_count: usize, active: u32) -> RusticolResult<u32> {
    let point_count = u32::try_from(point_count)
        .map_err(|_| RusticolError::invalid_argument("compiled point count exceeds u32"))?;
    if point_count == 0 || point_count != active {
        return Err(RusticolError::invalid_argument(
            "compiled Direct-Arena point count does not match the active tile",
        ));
    }
    Ok(point_count)
}

fn evaluate_bound_stage(
    stage: &BoundStage,
    point_count: u32,
    selected: Option<&[usize]>,
) -> RusticolResult<()> {
    if let Some(selected) = selected {
        let mut previous = None;
        for &index in selected {
            if index >= stage.leaves.len() || previous.is_some_and(|value| value >= index) {
                return Err(RusticolError::invalid_argument(
                    "compiled Direct-Arena leaf schedule is not sorted and unique",
                ));
            }
            stage.leaves[index].evaluate(0, point_count)?;
            previous = Some(index);
        }
    } else {
        for leaf in &stage.leaves {
            leaf.evaluate(0, point_count)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum OutputArena {
    Current,
    Amplitude,
}

fn load_stage(
    stage: &GenericSerializedStageEvaluatorManifest,
    is_amplitude: bool,
    payloads: &EvaluatorPayloadStore,
    value_component_count: usize,
    momentum_component_count: usize,
    amplitude_component_count: usize,
) -> RusticolResult<LoadedStage> {
    if stage.parameter_layout != "stage-local-value-momentum"
        || stage.input_components.len() != stage.parameter_count
    {
        return Err(RusticolError::compatibility(format!(
            "compiled Direct-Arena stage {:?} requires complete stage-local input metadata",
            stage.evaluator_label
        )));
    }
    let mut components = vec![None; stage.parameter_count];
    for component in &stage.input_components {
        if component.parameter_index >= components.len()
            || components[component.parameter_index]
                .replace(component)
                .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} has invalid input-component metadata",
                stage.evaluator_label
            )));
        }
    }
    if components.iter().any(Option::is_none) {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} has incomplete input-component metadata",
            stage.evaluator_label
        )));
    }

    let output_arena = if is_amplitude {
        OutputArena::Amplitude
    } else {
        OutputArena::Current
    };
    let output_limit = if is_amplitude {
        amplitude_component_count
    } else {
        value_component_count
    };
    let mut output_components = vec![None; stage.output_length];
    for slot in &stage.output_slots {
        let output_len = slot
            .output_stop
            .checked_sub(slot.output_start)
            .ok_or_else(|| RusticolError::integrity("compiled output slot range underflows"))?;
        if slot.component_stop.checked_sub(slot.component_start) != Some(output_len)
            || slot.output_stop > output_components.len()
            || slot.component_stop > output_limit
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} has an invalid output slot",
                stage.evaluator_label
            )));
        }
        for offset in 0..output_len {
            if output_components[slot.output_start + offset]
                .replace(slot.component_start + offset)
                .is_some()
            {
                return Err(RusticolError::integrity(format!(
                    "compiled Direct-Arena stage {:?} has overlapping output slots",
                    stage.evaluator_label
                )));
            }
        }
    }
    if output_components.iter().any(Option::is_none) {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} does not cover every output",
            stage.evaluator_label
        )));
    }

    let leaves = flattened_leaves(&stage.evaluator)?;
    let mut loaded = Vec::with_capacity(leaves.len());
    let mut output_cursor = 0usize;
    for leaf in leaves {
        let EvaluatorManifest::SymjitApplication {
            application_path,
            application_abi,
            input_len,
            output_len,
            optimization_level,
            required_defuns,
            ..
        } = leaf.manifest
        else {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena stage {:?} contains a non-SymJIT leaf",
                stage.evaluator_label
            )));
        };
        if *optimization_level != 3 || !required_defuns.is_empty() {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena stage {:?} requires self-contained O3 leaves",
                stage.evaluator_label
            )));
        }
        if leaf.input_indices.len() != *input_len {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} leaf input map has the wrong length",
                stage.evaluator_label
            )));
        }
        let output_stop = output_cursor.checked_add(*output_len).ok_or_else(|| {
            RusticolError::integrity("compiled Direct-Arena leaf output range overflows")
        })?;
        if output_stop > output_components.len() {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena leaf output range exceeds the stage",
            ));
        }

        let mut source_inputs = Vec::with_capacity(*input_len * 2);
        let mut plane_bindings = Vec::new();
        let mut scalar_bindings = Vec::new();
        for &parameter_index in &leaf.input_indices {
            let component = components
                .get(parameter_index)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena leaf references an absent stage input",
                    )
                })?;
            append_component_bindings(
                component,
                value_component_count,
                momentum_component_count,
                &mut source_inputs,
                &mut plane_bindings,
                &mut scalar_bindings,
            )?;
        }
        let mut outputs = Vec::with_capacity(*output_len * 2);
        for component in output_components[output_cursor..output_stop]
            .iter()
            .map(|value| value.expect("output coverage validated"))
        {
            let plane = |imaginary| match output_arena {
                OutputArena::Current => CompiledDirectArenaPlane::Current {
                    component: component as u32,
                    imaginary,
                },
                OutputArena::Amplitude => CompiledDirectArenaPlane::Amplitude {
                    component: component as u32,
                    imaginary,
                },
            };
            outputs.push(CompiledDirectOutputBinding(plane(false)));
            outputs.push(CompiledDirectOutputBinding(plane(true)));
        }
        let source = payloads.source(application_path)?;
        let bytes = source.read()?;
        loaded.push(LoadedSymjitCompiledDirectStage::load_source_bytes(
            bytes.as_ref(),
            PathBuf::from(source.display_name()),
            application_abi,
            source_inputs,
            plane_bindings,
            scalar_bindings,
            outputs,
        )?);
        output_cursor = output_stop;
    }
    if output_cursor != stage.output_length {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} leaf outputs do not cover the stage",
            stage.evaluator_label
        )));
    }
    Ok(LoadedStage { leaves: loaded })
}

fn append_component_bindings(
    component: &GenericStageInputComponentManifest,
    value_component_count: usize,
    momentum_component_count: usize,
    source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
    planes: &mut Vec<CompiledDirectPlaneBinding>,
    scalars: &mut Vec<CompiledDirectScalarBinding>,
) -> RusticolResult<()> {
    fn push_plane(
        source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
        planes: &mut Vec<CompiledDirectPlaneBinding>,
        binding: CompiledDirectPlaneBinding,
    ) -> RusticolResult<()> {
        let index = u32::try_from(planes.len()).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena plane binding count exceeds u32")
        })?;
        source_inputs.push(CompiledDirectSourceInputBinding::Plane(index));
        planes.push(binding);
        Ok(())
    }
    fn push_scalar(
        source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
        scalars: &mut Vec<CompiledDirectScalarBinding>,
        binding: CompiledDirectScalarBinding,
    ) -> RusticolResult<()> {
        let index = u32::try_from(scalars.len()).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena scalar binding count exceeds u32")
        })?;
        source_inputs.push(CompiledDirectSourceInputBinding::Scalar(index));
        scalars.push(binding);
        Ok(())
    }

    match component.kind.as_str() {
        "value" => {
            if component.global_component >= value_component_count {
                return Err(RusticolError::integrity(
                    "compiled Direct-Arena value input is outside current planes",
                ));
            }
            let current = |imaginary| {
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: component.global_component as u32,
                    imaginary,
                })
            };
            push_plane(source_inputs, planes, current(false))?;
            push_plane(source_inputs, planes, current(true))?;
        }
        "momentum" => {
            let local = component
                .global_component
                .checked_sub(value_component_count)
                .filter(|index| *index < momentum_component_count)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena momentum input is outside momentum planes",
                    )
                })?;
            push_plane(
                source_inputs,
                planes,
                CompiledDirectPlaneBinding::Momentum {
                    form: u32::try_from(local / 4).map_err(|_| {
                        RusticolError::integrity("compiled momentum form exceeds u32")
                    })?,
                    lorentz_component: (local % 4) as u16,
                },
            )?;
            push_plane(source_inputs, planes, CompiledDirectPlaneBinding::Zero)?;
        }
        "model_parameter" => {
            let index = u32::try_from(component.source_id).map_err(|_| {
                RusticolError::integrity("compiled model parameter index exceeds u32")
            })?;
            push_scalar(
                source_inputs,
                scalars,
                CompiledDirectScalarBinding::Parameter {
                    index,
                    imaginary: false,
                },
            )?;
            push_scalar(
                source_inputs,
                scalars,
                CompiledDirectScalarBinding::Parameter {
                    index,
                    imaginary: true,
                },
            )?;
        }
        kind => {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena does not support input kind {kind:?}"
            )));
        }
    }
    Ok(())
}

struct Leaf<'a> {
    manifest: &'a EvaluatorManifest,
    input_indices: Vec<usize>,
}

fn flattened_leaves(manifest: &EvaluatorManifest) -> RusticolResult<Vec<Leaf<'_>>> {
    fn append<'a>(
        manifest: &'a EvaluatorManifest,
        parent_inputs: &[usize],
        leaves: &mut Vec<Leaf<'a>>,
    ) -> RusticolResult<()> {
        match manifest {
            EvaluatorManifest::Chunked {
                input_len,
                chunk_input_indices,
                chunks,
                ..
            } => {
                if input_len.is_some_and(|count| count != parent_inputs.len()) {
                    return Err(RusticolError::integrity(
                        "compiled Direct-Arena chunk input count is inconsistent",
                    ));
                }
                match chunk_input_indices {
                    Some(mappings) => {
                        if mappings.len() != chunks.len() {
                            return Err(RusticolError::integrity(
                                "compiled Direct-Arena chunk mapping count is inconsistent",
                            ));
                        }
                        for (chunk, mapping) in chunks.iter().zip(mappings) {
                            let mapped = mapping
                                .iter()
                                .map(|index| {
                                    parent_inputs.get(*index).copied().ok_or_else(|| {
                                        RusticolError::integrity(
                                            "compiled Direct-Arena chunk input is out of bounds",
                                        )
                                    })
                                })
                                .collect::<RusticolResult<Vec<_>>>()?;
                            append(chunk, &mapped, leaves)?;
                        }
                    }
                    None => {
                        for chunk in chunks {
                            append(chunk, parent_inputs, leaves)?;
                        }
                    }
                }
            }
            _ => {
                if manifest.io_len()?.0 != parent_inputs.len() {
                    return Err(RusticolError::integrity(
                        "compiled Direct-Arena leaf input count is inconsistent",
                    ));
                }
                leaves.push(Leaf {
                    manifest,
                    input_indices: parent_inputs.to_vec(),
                });
            }
        }
        Ok(())
    }

    let root_input_count = manifest.io_len()?.0;
    let root_inputs = (0..root_input_count).collect::<Vec<_>>();
    let mut leaves = Vec::new();
    append(manifest, &root_inputs, &mut leaves)?;
    Ok(leaves)
}

#[cfg(test)]
mod tests {
    use super::super::evaluator::count_test_allocations;
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    use symjit::{Compiler, Config, Expr, Storage};

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn test_payload_directory() -> PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target"));
        target.join(format!(
            "compiled-direct-engine-test-{}-{}",
            std::process::id(),
            TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn source_application(inputs: &[Expr], outputs: &[Expr]) -> Vec<u8> {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        config.set_compress(true);
        let application = Compiler::with_config(config)
            .compile_params(&[], outputs, inputs)
            .unwrap();
        assert_eq!(application.count_states, 0);
        assert_eq!(application.count_params, inputs.len() * 2);
        assert_eq!(application.count_obs, outputs.len() * 2);
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn evaluator(path: &str, input_len: usize, output_len: usize) -> EvaluatorManifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string(),
            application_path: path.to_string(),
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI.to_string(),
            input_len,
            output_len,
            element_layout: "complex-f64".to_string(),
            batch_layout: "row-major".to_string(),
            compiler_type: "native".to_string(),
            translation_mode: "indirect".to_string(),
            optimization_level: 3,
            word_bits: 64,
            endianness: "little".to_string(),
            required_defuns: Vec::new(),
            evaluator_state_path: None,
            evaluator_state_runtime_capability: None,
        }
    }

    fn input_component(
        kind: &str,
        source_id: usize,
        component: usize,
        global_component: usize,
        parameter_index: usize,
        real_valued: bool,
    ) -> GenericStageInputComponentManifest {
        GenericStageInputComponentManifest {
            kind: kind.to_string(),
            source_id,
            component,
            global_component,
            parameter_index,
            real_valued,
        }
    }

    fn output_slot(
        component: usize,
        current_id: isize,
        output_column: usize,
    ) -> GenericStageOutputSlotManifest {
        GenericStageOutputSlotManifest {
            value_slot_id: current_id,
            current_id,
            variant: "test".to_string(),
            component_start: component,
            component_stop: component + 1,
            output_start: output_column,
            output_stop: output_column + 1,
            color_selector_domain_ids: Vec::new(),
        }
    }

    fn stage_manifest(
        label: &str,
        inputs: Vec<GenericStageInputComponentManifest>,
        output_components: &[usize],
        evaluator: EvaluatorManifest,
        amplitude: bool,
    ) -> GenericSerializedStageEvaluatorManifest {
        let value_parameter_count = inputs.iter().filter(|item| item.kind == "value").count();
        let momentum_parameter_count = inputs.iter().filter(|item| item.kind == "momentum").count();
        let model_parameter_count = inputs
            .iter()
            .filter(|item| item.kind == "model_parameter")
            .count();
        let real_valued_inputs = inputs
            .iter()
            .filter_map(|item| item.real_valued.then_some(item.parameter_index))
            .collect();
        GenericSerializedStageEvaluatorManifest {
            stage_index: usize::from(amplitude),
            stage_kind: if amplitude {
                "amplitude".to_string()
            } else {
                "interaction".to_string()
            },
            subset_size: None,
            evaluator_label: label.to_string(),
            parameter_layout: "stage-local-value-momentum".to_string(),
            output_length: output_components.len(),
            output_slots: output_components
                .iter()
                .copied()
                .enumerate()
                .map(|(column, component)| {
                    output_slot(
                        component,
                        if amplitude { -1 } else { component as isize },
                        column,
                    )
                })
                .collect(),
            input_value_slot_ids: Vec::new(),
            output_value_slot_ids: Vec::new(),
            interaction_ids: Vec::new(),
            parameter_count: inputs.len(),
            input_components: inputs,
            value_parameter_count,
            momentum_parameter_count,
            model_parameter_count,
            real_valued_inputs,
            expression_ready: true,
            blockers: Vec::new(),
            evaluator,
        }
    }

    fn assert_close(actual: Complex<f64>, expected: Complex<f64>) {
        let tolerance = 1.0e-15 + 1.0e-12 * expected.norm();
        assert!(
            (actual - expected).norm() <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance}"
        );
    }

    #[test]
    fn manifest_driven_engine_matches_row_major_compiled_stages_at_all_tail_boundaries() {
        const VALUE_COMPONENTS: usize = 4;
        const MOMENTUM_COMPONENTS: usize = 4;
        const MODEL_PARAMETERS: usize = 1;
        const GLOBAL_PARAMETERS: usize = VALUE_COMPONENTS + MOMENTUM_COMPONENTS + MODEL_PARAMETERS;

        let payload_root = test_payload_directory();
        fs::create_dir_all(&payload_root).unwrap();
        let x = Expr::var("x");
        let energy = Expr::var("energy");
        let coupling = Expr::var("coupling");
        let product = &x * &energy;
        let shifted = &x + &coupling;
        let product_bytes =
            source_application(&[x.clone(), energy.clone()], std::slice::from_ref(&product));
        let shifted_bytes = source_application(
            &[x.clone(), coupling.clone()],
            std::slice::from_ref(&shifted),
        );
        fs::write(payload_root.join("product.symjit"), &product_bytes).unwrap();
        fs::write(payload_root.join("shifted.symjit"), &shifted_bytes).unwrap();
        let left = Expr::var("left");
        let right = Expr::var("right");
        let amplitude_value = &left * &right;
        let amplitude_bytes =
            source_application(&[left, right], std::slice::from_ref(&amplitude_value));
        fs::write(payload_root.join("amplitude.symjit"), &amplitude_bytes).unwrap();

        let chunked = EvaluatorManifest::Chunked {
            required_runtime_capabilities: vec![SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()],
            input_len: Some(3),
            chunk_input_indices: Some(vec![vec![0, 1], vec![0, 2]]),
            chunks: vec![
                evaluator("product.symjit", 2, 1),
                evaluator("shifted.symjit", 2, 1),
            ],
        };
        let stage = stage_manifest(
            "real-fused-stage",
            vec![
                input_component("value", 0, 0, 0, 0, false),
                input_component("momentum", 0, 0, VALUE_COMPONENTS, 1, true),
                input_component(
                    "model_parameter",
                    0,
                    0,
                    VALUE_COMPONENTS + MOMENTUM_COMPONENTS,
                    2,
                    true,
                ),
            ],
            &[1, 2],
            chunked,
            false,
        );
        let amplitude = stage_manifest(
            "real-amplitude-stage",
            vec![
                input_component("value", 1, 0, 1, 0, false),
                input_component("value", 2, 0, 2, 1, false),
            ],
            &[0],
            evaluator("amplitude.symjit", 2, 1),
            true,
        );
        let legacy_amplitude = stage_manifest(
            "legacy-amplitude-stage",
            vec![
                input_component("value", 1, 0, 1, 0, false),
                input_component("value", 2, 0, 2, 1, false),
            ],
            &[3],
            evaluator("amplitude.symjit", 2, 1),
            false,
        );
        let payloads = EvaluatorPayloadStore::directory(&payload_root);
        let mut legacy_stage = StageRuntime::load(&stage, &payloads).unwrap();
        let mut legacy_amplitude_stage = StageRuntime::load(&legacy_amplitude, &payloads).unwrap();
        let mut direct = CompiledDirectEnginePrototype::load(
            std::slice::from_ref(&stage),
            &amplitude,
            &payloads,
            VALUE_COMPONENTS,
            MOMENTUM_COMPONENTS,
            MODEL_PARAMETERS,
            1,
            129,
        )
        .unwrap();

        for point_count in [7_usize, 127, 128, 129] {
            let mut initial = vec![Complex::new(0.0, 0.0); point_count * GLOBAL_PARAMETERS];
            for point in 0..point_count {
                let row = point * GLOBAL_PARAMETERS;
                initial[row] =
                    Complex::new(0.25 + point as f64 / 128.0, -0.5 + point as f64 / 512.0);
                initial[row + VALUE_COMPONENTS] = Complex::new(20.0 + point as f64 / 64.0, 0.0);
                initial[row + VALUE_COMPONENTS + MOMENTUM_COMPONENTS] = Complex::new(1.25, 0.0);
            }

            let mut legacy = initial.clone();
            legacy_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();
            legacy_amplitude_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();

            direct
                .begin_tile_from_state(point_count, GLOBAL_PARAMETERS, &initial)
                .unwrap();
            direct.evaluate_all(point_count).unwrap();
            let (allocation_result, allocations, allocated_bytes) =
                count_test_allocations(|| direct.evaluate_all(point_count));
            allocation_result.unwrap();
            assert_eq!(allocations, 0, "warmed direct engine call allocated");
            assert_eq!(
                allocated_bytes, 0,
                "warmed direct engine call allocated bytes"
            );
            assert_eq!(direct.traffic(), CompiledDirectPrototypeTraffic::default());

            let mut direct_state = initial.clone();
            direct
                .copy_current_to_state(point_count, GLOBAL_PARAMETERS, &mut direct_state)
                .unwrap();
            let mut direct_amplitude = vec![Complex::new(f64::NAN, f64::NAN); point_count];
            direct
                .amplitude_values(point_count, &mut direct_amplitude)
                .unwrap();
            for point in 0..point_count {
                let row = point * GLOBAL_PARAMETERS;
                assert_close(direct_state[row + 1], legacy[row + 1]);
                assert_close(direct_state[row + 2], legacy[row + 2]);
                assert_close(direct_amplitude[point], legacy[row + 3]);
            }

            if point_count == 129 {
                const SAMPLES: usize = 7;
                const REPEATS: usize = 2_000;
                let mut direct_ns = [0_u128; SAMPLES];
                let mut legacy_ns = [0_u128; SAMPLES];
                for sample in 0..SAMPLES {
                    let time_direct = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            direct.evaluate_all(point_count).unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    let mut time_legacy = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            legacy_stage
                                .evaluate_f64_into_state(
                                    point_count,
                                    GLOBAL_PARAMETERS,
                                    &mut legacy,
                                )
                                .unwrap();
                            legacy_amplitude_stage
                                .evaluate_f64_into_state(
                                    point_count,
                                    GLOBAL_PARAMETERS,
                                    &mut legacy,
                                )
                                .unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    if sample.is_multiple_of(2) {
                        direct_ns[sample] = time_direct();
                        legacy_ns[sample] = time_legacy();
                    } else {
                        legacy_ns[sample] = time_legacy();
                        direct_ns[sample] = time_direct();
                    }
                    direct
                        .amplitude_values(point_count, &mut direct_amplitude)
                        .unwrap();
                    std::hint::black_box((&legacy, &direct_amplitude));
                }
                direct_ns.sort_unstable();
                legacy_ns.sort_unstable();
                let direct_median = direct_ns[SAMPLES / 2];
                let legacy_median = legacy_ns[SAMPLES / 2];
                eprintln!(
                    "compiled-direct-engine-prototype points={point_count} samples={SAMPLES} \
                     repeats={REPEATS} direct_ns={direct_ns:?} legacy_ns={legacy_ns:?} \
                     direct_median_ns={direct_median} legacy_median_ns={legacy_median} \
                     speedup={:.6}",
                    legacy_median as f64 / direct_median as f64
                );
            }
        }

        direct
            .begin_tile_from_state(
                7,
                GLOBAL_PARAMETERS,
                &vec![Complex::new(0.0, 0.0); 7 * GLOBAL_PARAMETERS],
            )
            .unwrap();
        direct.evaluate_schedule(7, &[vec![0, 1]], &[0]).unwrap();
        assert!(direct.evaluate_schedule(7, &[vec![0, 0]], &[0]).is_err());
    }
}
