// SPDX-License-Identifier: 0BSD

//! Compiled-stage adapter for SymJIT's factor-free DirectApplication v3 ABI.
//!
//! This is deliberately narrower than the compiled execution engine. It owns
//! one already-fused O0 through O3 source application, validates its logical
//! bindings once, lowers it to the fixed identity-overwrite Direct-Arena
//! callable, and pins a descriptor bundle to persistent storage. The hot call
//! does not gather inputs, scatter outputs, rebuild descriptors, or allocate.

// Some ABI construction helpers remain test-only even though this adapter is
// now the production compiled-JIT execution path.
#![allow(dead_code)]

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;

use symjit::{
    Config, DIRECT_APPLICATION_STORAGE_ABI, DIRECT_NO_ALIAS, DIRECT_STATUS_OK, Defuns,
    DirectApplication, DirectApplicationMetadata, DirectCallable, DirectDestinationOperation,
    DirectInputBinding, DirectOutputScale, DirectPlane, DirectScalar, Storage,
};

use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    validate_direct_views,
};
use crate::engine::SYMJIT_APPLICATION_STORAGE_ABI;
use crate::{RusticolError, RusticolResult};

/// One split-complex plane in the persistent compiled-stage arena.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledDirectArenaPlane {
    Current { component: u32, imaginary: bool },
    Amplitude { component: u32, imaginary: bool },
}

/// One point-dependent input consumed by a fused compiled stage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledDirectPlaneBinding {
    Arena(CompiledDirectArenaPlane),
    Momentum {
        form: u32,
        lorentz_component: u16,
    },
    /// A persistent aligned zero plane, useful for structural-zero bindings.
    Zero,
}

/// One point-independent scalar consumed by a fused compiled stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CompiledDirectScalarBinding {
    Parameter { index: u32, imaginary: bool },
    Literal(f64),
}

/// One distinct output plane overwritten by a fused compiled stage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CompiledDirectOutputBinding(pub CompiledDirectArenaPlane);

/// Raw-input routing used while lowering an ordinary compiled application into
/// the factor-free DirectApplication ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompiledDirectSourceInputBinding {
    Plane(u32),
    Scalar(u32),
}

/// Loaded but not yet pinned compiled-stage application.
pub(crate) struct LoadedSymjitCompiledDirectStage {
    callable: DirectCallable,
    simd_lane_width: usize,
    input_bindings: Box<[CompiledDirectPlaneBinding]>,
    scalar_bindings: Box<[CompiledDirectScalarBinding]>,
    output_bindings: Box<[CompiledDirectOutputBinding]>,
    display_path: Box<Path>,
}

/// A fixed direct descriptor bundle for one persistent compiled-stage arena.
///
/// The raw descriptors borrow storage whose lifetime is guaranteed by the
/// unsafe `bind` contract. The owning execution runtime must therefore keep
/// arena, momentum, and parameter allocations fixed until this object drops.
pub(crate) struct BoundSymjitCompiledDirectStage {
    callable: DirectCallable,
    planes: Box<[DirectPlane]>,
    scalars: Box<[DirectScalar]>,
    _literal_values: Box<[f64]>,
    point_stride: u32,
    simd_lane_width: usize,
    display_path: Box<Path>,
}

impl LoadedSymjitCompiledDirectStage {
    /// Lower one ordinary complex-f64 O0 through O3 application into the fixed
    /// identity-overwrite DirectApplication callable, then load it through the
    /// same authenticated direct path as a natively-produced direct artifact.
    ///
    /// This is intentionally a cold-path bridge for the prototype. It accepts
    /// only the current source ABI and exactly preserves the already-fused
    /// application's instruction stream and output order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_source_bytes(
        bytes: &[u8],
        display_path: impl AsRef<Path>,
        source_application_abi: &str,
        source_optimization_level: u8,
        source_input_bindings: Vec<CompiledDirectSourceInputBinding>,
        input_bindings: Vec<CompiledDirectPlaneBinding>,
        scalar_bindings: Vec<CompiledDirectScalarBinding>,
        output_bindings: Vec<CompiledDirectOutputBinding>,
    ) -> RusticolResult<Self> {
        let display_path = display_path.as_ref();
        if source_application_abi != SYMJIT_APPLICATION_STORAGE_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported compiled source application ABI {source_application_abi:?}; \
                 expected {SYMJIT_APPLICATION_STORAGE_ABI:?}"
            )));
        }
        validate_static_bindings(&input_bindings, &output_bindings)?;

        let direct_inputs = source_input_bindings
            .into_iter()
            .map(|binding| match binding {
                CompiledDirectSourceInputBinding::Plane(index) => DirectInputBinding::Plane(index),
                CompiledDirectSourceInputBinding::Scalar(index) => {
                    DirectInputBinding::Scalar(index)
                }
            })
            .collect::<Vec<_>>();
        let direct_metadata = DirectApplicationMetadata::new_with_output_scale(
            DirectDestinationOperation::Initialize,
            DirectOutputScale::Identity,
            Vec::new(),
            direct_inputs,
            u32::try_from(input_bindings.len()).map_err(|_| {
                RusticolError::integrity(
                    "compiled Direct-Arena input plane count exceeds the u32 ABI",
                )
            })?,
            u32::try_from(scalar_bindings.len()).map_err(|_| {
                RusticolError::integrity(
                    "compiled Direct-Arena scalar input count exceeds the u32 ABI",
                )
            })?,
            vec![DIRECT_NO_ALIAS; output_bindings.len()],
        )
        .map_err(|error| {
            RusticolError::integrity(format!(
                "could not describe compiled Direct-Arena application {}: {error}",
                display_path.display()
            ))
        })?;
        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let direct = guard_symjit_panic(
            || DirectApplication::from_source_storage(&mut input, &loader_config, direct_metadata),
            display_path,
            "lower source for",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not lower compiled source application {} to DirectApplication: {error}",
                display_path.display()
            ))
        })?;
        if direct.source_optimization_level() != Some(source_optimization_level) {
            return Err(RusticolError::integrity(format!(
                "compiled source application {} declares optimization level \
                 {source_optimization_level} but stores optimization level {:?}",
                display_path.display(),
                direct.source_optimization_level()
            )));
        }
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "compiled source application {} has {} trailing bytes",
                display_path.display(),
                input.len()
            )));
        }
        Self::finish_application(
            direct,
            display_path,
            input_bindings,
            scalar_bindings,
            output_bindings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_bytes(
        bytes: &[u8],
        display_path: impl AsRef<Path>,
        application_abi: &str,
        input_bindings: Vec<CompiledDirectPlaneBinding>,
        scalar_bindings: Vec<CompiledDirectScalarBinding>,
        output_bindings: Vec<CompiledDirectOutputBinding>,
    ) -> RusticolResult<Self> {
        let display_path: Box<Path> = display_path.as_ref().into();
        if application_abi != DIRECT_APPLICATION_STORAGE_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported compiled Direct-Arena application ABI {application_abi:?}; expected \
                 {DIRECT_APPLICATION_STORAGE_ABI:?}"
            )));
        }

        validate_static_bindings(&input_bindings, &output_bindings)?;

        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let application = guard_symjit_panic(
            || DirectApplication::load(&mut input, &loader_config),
            &display_path,
            "load",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load compiled Direct-Arena application {}: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena application {} has {} trailing bytes",
                display_path.display(),
                input.len()
            )));
        }

        Self::finish_application(
            application,
            &display_path,
            input_bindings,
            scalar_bindings,
            output_bindings,
        )
    }

    fn finish_application(
        mut application: DirectApplication,
        display_path: &Path,
        input_bindings: Vec<CompiledDirectPlaneBinding>,
        scalar_bindings: Vec<CompiledDirectScalarBinding>,
        output_bindings: Vec<CompiledDirectOutputBinding>,
    ) -> RusticolResult<Self> {
        let metadata = application.metadata();
        if metadata.destination_operation != DirectDestinationOperation::Initialize
            || metadata.output_scale != DirectOutputScale::Identity
            || metadata
                .output_alias_inputs
                .iter()
                .any(|alias| *alias != DIRECT_NO_ALIAS)
        {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena application {} is not a factor-free identity-overwrite \
                 O3 stage",
                display_path.display()
            )));
        }
        if input_bindings.len() != metadata.input_plane_count as usize {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena input binding count {} does not match application plane \
                 count {}",
                input_bindings.len(),
                metadata.input_plane_count
            )));
        }
        if scalar_bindings.len() != metadata.scalar_input_count as usize {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena scalar binding count {} does not match application scalar \
                 count {}",
                scalar_bindings.len(),
                metadata.scalar_input_count
            )));
        }
        if output_bindings.len() != metadata.output_alias_inputs.len() {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena output binding count {} does not match application output \
                 count {}",
                output_bindings.len(),
                metadata.output_alias_inputs.len()
            )));
        }

        guard_symjit_panic(
            || application.prepare_simd(),
            display_path,
            "prepare SIMD for",
        )?;
        let applet =
            guard_symjit_panic(|| application.seal(), display_path, "seal")?.map_err(|error| {
                RusticolError::evaluation(format!(
                    "could not seal compiled Direct-Arena application {}: {error}",
                    display_path.display()
                ))
            })?;
        let simd_lane_width = applet.simd_lane_width();
        Ok(Self {
            callable: applet.into_callable(),
            simd_lane_width,
            input_bindings: input_bindings.into_boxed_slice(),
            scalar_bindings: scalar_bindings.into_boxed_slice(),
            output_bindings: output_bindings.into_boxed_slice(),
            display_path: display_path.into(),
        })
    }

    /// Pin this stage to persistent Direct-Arena storage.
    ///
    /// # Safety
    ///
    /// All allocations referenced by `arena`, `momenta`, `parameters`, and
    /// `zero_plane` must remain alive and retain their address and declared
    /// length until the returned bound stage is dropped. The shared zero plane
    /// must remain immutable. While `evaluate` is running, no other code may
    /// read or mutate the arena, and no code may mutate momentum or parameter
    /// storage. The owning runtime must enforce those exclusivity and
    /// immutability requirements even when the backing allocations are shared
    /// through another synchronized owner.
    pub(crate) unsafe fn bind(
        self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        zero_plane: &AlignedF64Buffer,
    ) -> RusticolResult<BoundSymjitCompiledDirectStage> {
        validate_direct_views(
            arena,
            momenta,
            parameters,
            DirectFactorView {
                values_re: ptr::null(),
                values_im: ptr::null(),
                value_count: 0,
            },
        )?;

        let point_stride = arena.point_stride;
        if zero_plane.len() != point_stride as usize {
            return Err(RusticolError::integrity(format!(
                "compiled shared zero plane has length {}, expected {point_stride}",
                zero_plane.len()
            )));
        }
        if zero_plane.as_slice().iter().any(|value| *value != 0.0) {
            return Err(RusticolError::integrity(
                "compiled shared zero plane contains a nonzero value",
            ));
        }
        let mut literal_values = Vec::new();
        literal_values
            .try_reserve_exact(self.scalar_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled Direct-Arena literal storage: {error}"
                ))
            })?;
        literal_values.resize(self.scalar_bindings.len(), 0.0);
        let mut literal_values = literal_values.into_boxed_slice();
        for (index, binding) in self.scalar_bindings.iter().copied().enumerate() {
            if let CompiledDirectScalarBinding::Literal(value) = binding {
                literal_values[index] = value;
            }
        }

        let descriptor_count = self
            .input_bindings
            .len()
            .checked_add(self.output_bindings.len())
            .ok_or_else(|| {
                RusticolError::integrity("compiled Direct-Arena descriptor count overflows usize")
            })?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(descriptor_count)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled Direct-Arena plane descriptors: {error}"
                ))
            })?;
        for binding in self.input_bindings.iter().copied() {
            planes.push(resolve_input_plane(binding, arena, momenta, zero_plane)?);
        }
        for binding in self.output_bindings.iter().copied() {
            planes.push(resolve_arena_plane(binding.0, arena)?);
        }

        let mut scalars = Vec::new();
        scalars
            .try_reserve_exact(self.scalar_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled Direct-Arena scalar descriptors: {error}"
                ))
            })?;
        for (index, binding) in self.scalar_bindings.iter().copied().enumerate() {
            let value = match binding {
                CompiledDirectScalarBinding::Parameter { index, imaginary } => {
                    if index >= parameters.value_count {
                        return Err(RusticolError::integrity(format!(
                            "compiled Direct-Arena parameter binding {index} is out of bounds"
                        )));
                    }
                    let base = if imaginary {
                        parameters.values_im
                    } else {
                        parameters.values_re
                    };
                    // SAFETY: the complete parameter view was validated above,
                    // and the index is within its declared range.
                    unsafe { base.add(index as usize) }
                }
                CompiledDirectScalarBinding::Literal(_) => ptr::from_ref(&literal_values[index]),
            };
            // SAFETY: parameter storage obeys the bind lifetime contract, and
            // literals live in the bound object itself.
            scalars.push(unsafe { DirectScalar::from_raw(value) });
        }

        Ok(BoundSymjitCompiledDirectStage {
            callable: self.callable,
            planes: planes.into_boxed_slice(),
            scalars: scalars.into_boxed_slice(),
            _literal_values: literal_values,
            point_stride,
            simd_lane_width: self.simd_lane_width,
            display_path: self.display_path,
        })
    }
}

impl BoundSymjitCompiledDirectStage {
    pub(crate) const fn simd_lane_width(&self) -> usize {
        self.simd_lane_width
    }

    /// Evaluate one point range directly into its canonical output planes.
    ///
    /// Descriptor counts, pointer ranges, and aliases were authenticated by
    /// `load_bytes` plus `bind`, so this invokes the unchecked SymJIT entry
    /// point. The success path performs no heap allocation.
    pub(crate) fn evaluate(&self, point_start: u32, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena point count must be positive",
            ));
        }
        let point_stop = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("compiled Direct-Arena point range overflows u32")
        })?;
        if point_stop > self.point_stride {
            return Err(RusticolError::invalid_argument(format!(
                "compiled Direct-Arena point range {point_start}..{point_stop} exceeds physical \
                 stride {}",
                self.point_stride
            )));
        }

        let status = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: `bind` authenticated this fixed descriptor bundle, and
            // the range above is nonempty and within every pitched plane.
            unsafe {
                self.callable.invoke_unchecked(
                    &self.planes,
                    &self.scalars,
                    point_start as usize,
                    point_count as usize,
                )
            }
        }))
        .map_err(|payload| {
            RusticolError::evaluation(format!(
                "SymJIT panicked while evaluating compiled Direct-Arena application {}: {}",
                self.display_path.display(),
                panic_detail(payload)
            ))
        })?;
        if status != DIRECT_STATUS_OK {
            return Err(RusticolError::evaluation(format!(
                "compiled Direct-Arena application {} returned status {status}",
                self.display_path.display()
            )));
        }
        Ok(())
    }
}

fn validate_static_bindings(
    inputs: &[CompiledDirectPlaneBinding],
    outputs: &[CompiledDirectOutputBinding],
) -> RusticolResult<()> {
    if outputs.is_empty() {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena stage must expose an output",
        ));
    }
    for (index, output) in outputs.iter().copied().enumerate() {
        if outputs[..index].contains(&output) {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena output binding {index} aliases an earlier output"
            )));
        }
        if inputs.contains(&CompiledDirectPlaneBinding::Arena(output.0)) {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena output binding {index} aliases an input plane"
            )));
        }
    }
    Ok(())
}

fn resolve_input_plane(
    binding: CompiledDirectPlaneBinding,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    zero_plane: &AlignedF64Buffer,
) -> RusticolResult<DirectPlane> {
    match binding {
        CompiledDirectPlaneBinding::Arena(binding) => resolve_arena_plane(binding, arena),
        CompiledDirectPlaneBinding::Momentum {
            form,
            lorentz_component,
        } => {
            if form >= momenta.form_count || lorentz_component >= momenta.lorentz_component_count {
                return Err(RusticolError::integrity(format!(
                    "compiled Direct-Arena momentum binding ({form}, {lorentz_component}) is out \
                     of bounds"
                )));
            }
            let component = form
                .checked_mul(u32::from(momenta.lorentz_component_count))
                .and_then(|base| base.checked_add(u32::from(lorentz_component)))
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena momentum component overflows u32",
                    )
                })?;
            let offset = usize::try_from(component)
                .ok()
                .and_then(|component| component.checked_mul(momenta.point_stride as usize))
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena momentum offset overflows usize",
                    )
                })?;
            // SAFETY: the momentum view and component bounds were validated.
            let values = unsafe { momenta.values.add(offset) }.cast_mut();
            // SAFETY: the pointer covers exactly one persistent pitched plane.
            Ok(unsafe { DirectPlane::from_raw_parts(values, momenta.point_stride as usize) })
        }
        CompiledDirectPlaneBinding::Zero => {
            // SAFETY: zero_plane lives in the bound object and has one full pitch.
            Ok(unsafe {
                DirectPlane::from_raw_parts(
                    zero_plane.as_ptr().cast_mut(),
                    arena.point_stride as usize,
                )
            })
        }
    }
}

fn resolve_arena_plane(
    binding: CompiledDirectArenaPlane,
    arena: DirectArenaView,
) -> RusticolResult<DirectPlane> {
    let (base, shape, label) = match binding {
        CompiledDirectArenaPlane::Current {
            component,
            imaginary,
        } => (
            if imaginary {
                arena.current_im
            } else {
                arena.current_re
            },
            arena.current_shape()?,
            ("current", component),
        ),
        CompiledDirectArenaPlane::Amplitude {
            component,
            imaginary,
        } => (
            if imaginary {
                arena.amplitude_im
            } else {
                arena.amplitude_re
            },
            arena.amplitude_shape()?,
            ("amplitude", component),
        ),
    };
    let offset = shape.checked_scalar_index(label.1, 0, label.0)?;
    // SAFETY: shape validation and the checked scalar index prove that one
    // complete pitched component plane starts at this pointer.
    let values = unsafe { base.add(offset) };
    Ok(unsafe { DirectPlane::from_raw_parts(values, arena.point_stride as usize) })
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    path: &Path,
    action: &str,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "SymJIT panicked while trying to {action} compiled Direct-Arena application {}: {}",
            path.display(),
            panic_detail(payload)
        ))
    })
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::symjit_direct::tests::count_allocations;
    use super::*;
    use crate::RusticolErrorKind;
    use crate::direct_arena::{DirectArenaWorkspace, DirectMomentumView};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::Instant;
    use symjit::{Compiler, DirectApplicationMetadata, DirectInputBinding, Expr, Storage};

    fn compiled_source_application_at(optimization_level: u8) -> symjit::Application {
        let mut config = Config::default();
        config.set_opt_level(optimization_level);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        config.set_compress(true);
        let instructions = r#"[[{"Mul":[{"Out":0},[{"Param":0},{"Param":1}],0]},{"Add":[{"Out":1},[{"Param":0},{"Param":1}],0]}],2,[]]"#;
        Compiler::with_config(config)
            .translate(instructions.to_owned(), 2)
            .unwrap()
    }

    fn compiled_source_application() -> symjit::Application {
        compiled_source_application_at(3)
    }

    fn compiled_application_bytes() -> Vec<u8> {
        let source = compiled_source_application();
        assert_eq!(source.count_states, 0);
        assert_eq!(source.count_params, 4);
        assert_eq!(source.count_obs, 4);
        let metadata = DirectApplicationMetadata::new_with_output_scale(
            DirectDestinationOperation::Initialize,
            DirectOutputScale::Identity,
            vec![],
            (0..source.count_params as u32)
                .map(DirectInputBinding::Plane)
                .collect(),
            source.count_params as u32,
            0,
            vec![DIRECT_NO_ALIAS; source.count_obs],
        )
        .unwrap();
        let application = DirectApplication::new(source, metadata).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn scalar_application_bytes() -> Vec<u8> {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let value = Expr::var("value");
        let source = Compiler::with_config(config)
            .compile_params(
                &[],
                std::slice::from_ref(&value),
                std::slice::from_ref(&value),
            )
            .unwrap();
        assert_eq!(source.count_params, 2);
        assert_eq!(source.count_obs, 2);
        let metadata = DirectApplicationMetadata::new_with_output_scale(
            DirectDestinationOperation::Initialize,
            DirectOutputScale::Identity,
            vec![],
            vec![DirectInputBinding::Scalar(0), DirectInputBinding::Scalar(1)],
            0,
            2,
            vec![DIRECT_NO_ALIAS; source.count_obs],
        )
        .unwrap();
        let application = DirectApplication::new(source, metadata).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn empty_parameters() -> DirectParameterView {
        DirectParameterView {
            values_re: ptr::null(),
            values_im: ptr::null(),
            value_count: 0,
        }
    }

    fn momentum_view(values: &AlignedF64Buffer, point_stride: u32) -> DirectMomentumView {
        DirectMomentumView {
            values: values.as_ptr(),
            scalar_len: values.len() as u64,
            form_count: 1,
            lorentz_component_count: 4,
            point_stride,
        }
    }

    fn compiled_bindings() -> (
        Vec<CompiledDirectPlaneBinding>,
        Vec<CompiledDirectOutputBinding>,
    ) {
        let input_bindings = vec![
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: false,
            }),
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: true,
            }),
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 1,
                imaginary: false,
            }),
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 1,
                imaginary: true,
            }),
        ];
        let output_bindings = vec![
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 2,
                imaginary: false,
            }),
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 2,
                imaginary: true,
            }),
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 3,
                imaginary: false,
            }),
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 3,
                imaginary: true,
            }),
        ];
        (input_bindings, output_bindings)
    }

    fn fill_compiled_inputs(workspace: &mut DirectArenaWorkspace, point_count: usize) {
        let stride = workspace.point_stride() as usize;
        let (current_re, current_im, _, _) = workspace.split_slices_mut();
        for point in 0..point_count {
            current_re[point] = 0.25 + point as f64 * 0.03125;
            current_im[point] = if point % 2 == 0 { -0.0 } else { 0.5 };
            current_re[stride + point] = 1.5 - point as f64 * 0.0078125;
            current_im[stride + point] = if point % 3 == 0 { 0.0 } else { -0.25 };
        }
    }

    fn pack_call_scatter(
        dense: &symjit::Applet,
        workspace: &mut DirectArenaWorkspace,
        point_count: usize,
        packed_inputs: &mut [f64],
        packed_outputs: &mut [f64],
    ) {
        let stride = workspace.point_stride() as usize;
        {
            let (current_re, current_im) = workspace.current_slices();
            for point in 0..point_count {
                let row = 4 * point;
                packed_inputs[row] = current_re[point];
                packed_inputs[row + 1] = current_im[point];
                packed_inputs[row + 2] = current_re[stride + point];
                packed_inputs[row + 3] = current_im[stride + point];
            }
        }
        dense.evaluate_matrix(packed_inputs, packed_outputs, point_count);
        let (current_re, current_im, _, _) = workspace.split_slices_mut();
        for point in 0..point_count {
            let row = 4 * point;
            current_re[2 * stride + point] = packed_outputs[row];
            current_im[2 * stride + point] = packed_outputs[row + 1];
            current_re[3 * stride + point] = packed_outputs[row + 2];
            current_im[3 * stride + point] = packed_outputs[row + 3];
        }
    }

    fn median_u128(values: &mut [u128]) -> u128 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    #[test]
    fn fused_o3_stage_preserves_odd_tails_and_warmed_calls_allocate_zero() {
        let bytes = compiled_application_bytes();
        let (input_bindings, output_bindings) = compiled_bindings();
        let loaded = LoadedSymjitCompiledDirectStage::load_bytes(
            &bytes,
            "compiled-o3-stage.symjit",
            DIRECT_APPLICATION_STORAGE_ABI,
            input_bindings,
            vec![],
            output_bindings,
        )
        .unwrap();
        assert!(loaded.simd_lane_width > 1);

        let mut workspace = DirectArenaWorkspace::new(4, 0, 129).unwrap();
        workspace.begin_tile(129).unwrap();
        let stride = workspace.point_stride() as usize;
        fill_compiled_inputs(&mut workspace, 129);
        let momenta = AlignedF64Buffer::zeroed(4 * stride, "compiled test momenta").unwrap();
        let zero_plane = AlignedF64Buffer::zeroed(stride, "compiled shared zero").unwrap();
        let arena = workspace.view().unwrap();
        let bound = unsafe {
            loaded.bind(
                arena,
                momentum_view(&momenta, arena.point_stride),
                empty_parameters(),
                &zero_plane,
            )
        }
        .unwrap();

        for (point_start, point_count) in [
            (0_u32, 1_u32),
            (0, 2),
            (0, 3),
            (0, 127),
            (1, 127),
            (0, 128),
            (0, 129),
        ] {
            {
                let (current_re, current_im, _, _) = workspace.split_slices_mut();
                current_re[2 * stride..4 * stride].fill(f64::NAN);
                current_im[2 * stride..4 * stride].fill(f64::NAN);
            }
            bound.evaluate(point_start, point_count).unwrap();
            let (current_re, current_im) = workspace.current_slices();
            let point_stop = (point_start + point_count) as usize;
            for point in point_start as usize..point_stop {
                let x_re = current_re[point];
                let x_im = current_im[point];
                let y_re = current_re[stride + point];
                let y_im = current_im[stride + point];
                let expected_product_re = x_re * y_re - x_im * y_im;
                let expected_product_im = x_re * y_im + x_im * y_re;
                assert!((current_re[2 * stride + point] - expected_product_re).abs() < 1.0e-12);
                assert!((current_im[2 * stride + point] - expected_product_im).abs() < 1.0e-12);
                assert_eq!(
                    current_re[3 * stride + point].to_bits(),
                    (x_re + y_re).to_bits()
                );
                assert_eq!(
                    current_im[3 * stride + point].to_bits(),
                    (x_im + y_im).to_bits()
                );
            }
            for point in (0..point_start as usize).chain(point_stop..stride) {
                assert!(current_re[2 * stride + point].is_nan());
                assert!(current_im[2 * stride + point].is_nan());
                assert!(current_re[3 * stride + point].is_nan());
                assert!(current_im[3 * stride + point].is_nan());
            }
        }

        bound.evaluate(0, 129).unwrap();
        let (result, allocations, bytes) = count_allocations(|| bound.evaluate(0, 129));
        result.unwrap();
        assert_eq!(allocations, 0, "warmed compiled direct call allocated");
        assert_eq!(bytes, 0, "warmed compiled direct call allocated bytes");
    }

    #[test]
    fn source_bridge_accepts_and_authenticates_jit_o0_through_o3() {
        for optimization_level in 0..=3 {
            let source = compiled_source_application_at(optimization_level);
            let mut bytes = Vec::new();
            source.save(&mut bytes).unwrap();
            let (input_bindings, output_bindings) = compiled_bindings();
            let loaded = LoadedSymjitCompiledDirectStage::load_source_bytes(
                &bytes,
                format!("compiled-o{optimization_level}-source.symjit"),
                SYMJIT_APPLICATION_STORAGE_ABI,
                optimization_level,
                (0..4)
                    .map(CompiledDirectSourceInputBinding::Plane)
                    .collect(),
                input_bindings,
                vec![],
                output_bindings,
            )
            .unwrap();
            assert!(loaded.simd_lane_width > 1);
        }

        let source = compiled_source_application_at(1);
        let mut bytes = Vec::new();
        source.save(&mut bytes).unwrap();
        let (input_bindings, output_bindings) = compiled_bindings();
        let error = match LoadedSymjitCompiledDirectStage::load_source_bytes(
            &bytes,
            "compiled-mismatched-source.symjit",
            SYMJIT_APPLICATION_STORAGE_ABI,
            2,
            (0..4)
                .map(CompiledDirectSourceInputBinding::Plane)
                .collect(),
            input_bindings,
            vec![],
            output_bindings,
        ) {
            Ok(_) => panic!("mismatched source optimization level unexpectedly loaded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
        assert!(error.message().contains("stores optimization level"));
    }

    /// Manual raw-kernel A/B. There are deliberately no timing assertions:
    /// callers should inspect the interleaved median and retain machine/host
    /// provenance with any published number.
    #[test]
    #[ignore = "manual deterministic Direct-Arena raw benchmark"]
    fn benchmark_direct_against_pack_call_scatter() {
        const POINT_COUNT: usize = 129;
        const SAMPLES: usize = 9;
        const REPEATS: usize = 10_000;

        let bytes = compiled_application_bytes();
        let (input_bindings, output_bindings) = compiled_bindings();
        let loaded = LoadedSymjitCompiledDirectStage::load_bytes(
            &bytes,
            "compiled-o3-benchmark.symjit",
            DIRECT_APPLICATION_STORAGE_ABI,
            input_bindings,
            vec![],
            output_bindings,
        )
        .unwrap();
        let mut dense_source = compiled_source_application();
        dense_source.prepare_simd();
        let dense = dense_source.seal().unwrap();

        let mut workspace = DirectArenaWorkspace::new(4, 0, POINT_COUNT as u32).unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        fill_compiled_inputs(&mut workspace, POINT_COUNT);
        let stride = workspace.point_stride() as usize;
        let momenta = AlignedF64Buffer::zeroed(4 * stride, "compiled benchmark momenta").unwrap();
        let zero_plane = AlignedF64Buffer::zeroed(stride, "compiled shared zero").unwrap();
        let arena = workspace.view().unwrap();
        let bound = unsafe {
            loaded.bind(
                arena,
                momentum_view(&momenta, arena.point_stride),
                empty_parameters(),
                &zero_plane,
            )
        }
        .unwrap();
        let mut packed_inputs = vec![0.0; POINT_COUNT * 4];
        let mut packed_outputs = vec![0.0; POINT_COUNT * 4];

        pack_call_scatter(
            &dense,
            &mut workspace,
            POINT_COUNT,
            &mut packed_inputs,
            &mut packed_outputs,
        );
        let expected = {
            let (current_re, current_im) = workspace.current_slices();
            (0..POINT_COUNT)
                .flat_map(|point| {
                    [
                        current_re[2 * stride + point],
                        current_im[2 * stride + point],
                        current_re[3 * stride + point],
                        current_im[3 * stride + point],
                    ]
                })
                .collect::<Vec<_>>()
        };
        bound.evaluate(0, POINT_COUNT as u32).unwrap();
        {
            let (current_re, current_im) = workspace.current_slices();
            for point in 0..POINT_COUNT {
                let row = 4 * point;
                assert!((current_re[2 * stride + point] - expected[row]).abs() < 1.0e-12);
                assert!((current_im[2 * stride + point] - expected[row + 1]).abs() < 1.0e-12);
                assert!((current_re[3 * stride + point] - expected[row + 2]).abs() < 1.0e-12);
                assert!((current_im[3 * stride + point] - expected[row + 3]).abs() < 1.0e-12);
            }
        }

        let (direct_result, direct_allocations, direct_bytes) =
            count_allocations(|| bound.evaluate(0, POINT_COUNT as u32));
        direct_result.unwrap();
        let (_, packed_allocations, packed_bytes) = count_allocations(|| {
            pack_call_scatter(
                &dense,
                &mut workspace,
                POINT_COUNT,
                &mut packed_inputs,
                &mut packed_outputs,
            )
        });

        let mut direct_ns = [0_u128; SAMPLES];
        let mut packed_ns = [0_u128; SAMPLES];
        for sample in 0..SAMPLES {
            let time_direct = || {
                let start = Instant::now();
                for _ in 0..REPEATS {
                    bound.evaluate(0, POINT_COUNT as u32).unwrap();
                }
                start.elapsed().as_nanos() / REPEATS as u128
            };
            let mut time_packed = || {
                let start = Instant::now();
                for _ in 0..REPEATS {
                    pack_call_scatter(
                        &dense,
                        &mut workspace,
                        POINT_COUNT,
                        &mut packed_inputs,
                        &mut packed_outputs,
                    );
                }
                start.elapsed().as_nanos() / REPEATS as u128
            };
            if sample.is_multiple_of(2) {
                direct_ns[sample] = time_direct();
                packed_ns[sample] = time_packed();
            } else {
                packed_ns[sample] = time_packed();
                direct_ns[sample] = time_direct();
            }
            std::hint::black_box(workspace.current_slices().0[2 * stride]);
        }
        let direct_median = median_u128(&mut direct_ns);
        let packed_median = median_u128(&mut packed_ns);
        eprintln!(
            "compiled-direct-raw-benchmark points={POINT_COUNT} samples={SAMPLES} \
             repeats={REPEATS} direct_ns={direct_ns:?} packed_ns={packed_ns:?} \
             direct_median_ns={direct_median} packed_median_ns={packed_median} \
             speedup={:.6} direct_allocations={direct_allocations} \
             direct_allocated_bytes={direct_bytes} packed_allocations={packed_allocations} \
             packed_allocated_bytes={packed_bytes}",
            packed_median as f64 / direct_median as f64,
        );
    }

    #[test]
    fn scalar_bindings_are_fixed_once_and_factor_free() {
        let bytes = scalar_application_bytes();
        let outputs = vec![
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: false,
            }),
            CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: true,
            }),
        ];
        let loaded = LoadedSymjitCompiledDirectStage::load_bytes(
            &bytes,
            "compiled-scalar-stage.symjit",
            DIRECT_APPLICATION_STORAGE_ABI,
            vec![],
            vec![
                CompiledDirectScalarBinding::Parameter {
                    index: 0,
                    imaginary: false,
                },
                CompiledDirectScalarBinding::Literal(-0.75),
            ],
            outputs,
        )
        .unwrap();
        let mut workspace = DirectArenaWorkspace::new(1, 0, 9).unwrap();
        workspace.begin_tile(9).unwrap();
        let stride = workspace.point_stride();
        let momenta =
            AlignedF64Buffer::zeroed(4 * stride as usize, "compiled scalar momenta").unwrap();
        let parameter_re = [0.375];
        let parameter_im = [9.0];
        let parameters = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        let zero_plane = AlignedF64Buffer::zeroed(stride as usize, "compiled shared zero").unwrap();
        let arena = workspace.view().unwrap();
        let bound = unsafe {
            loaded.bind(
                arena,
                momentum_view(&momenta, stride),
                parameters,
                &zero_plane,
            )
        }
        .unwrap();
        bound.evaluate(0, 9).unwrap();
        let (current_re, current_im) = workspace.current_slices();
        assert!(
            current_re[..9]
                .iter()
                .all(|value| value.to_bits() == 0.375_f64.to_bits())
        );
        assert!(
            current_im[..9]
                .iter()
                .all(|value| value.to_bits() == (-0.75_f64).to_bits())
        );
    }

    #[test]
    fn static_aliases_and_malformed_payloads_fail_closed_without_unwinding() {
        let bytes = compiled_application_bytes();
        let (inputs, mut outputs) = compiled_bindings();
        outputs[0] = CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
            component: 0,
            imaginary: false,
        });
        let alias_error = match LoadedSymjitCompiledDirectStage::load_bytes(
            &bytes,
            "compiled-alias.symjit",
            DIRECT_APPLICATION_STORAGE_ABI,
            inputs,
            vec![],
            outputs,
        ) {
            Ok(_) => panic!("input/output alias unexpectedly loaded"),
            Err(error) => error,
        };
        assert_eq!(alias_error.kind(), RusticolErrorKind::Integrity);
        assert!(alias_error.message().contains("aliases an input plane"));

        let (inputs, outputs) = compiled_bindings();
        let mut malformed = bytes;
        malformed.truncate(11);
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            LoadedSymjitCompiledDirectStage::load_bytes(
                &malformed,
                "compiled-malformed.symjit",
                DIRECT_APPLICATION_STORAGE_ABI,
                inputs,
                vec![],
                outputs,
            )
        }));
        let result = attempt.expect("malformed compiled direct storage must not unwind");
        let error = match result {
            Ok(_) => panic!("malformed compiled direct storage unexpectedly loaded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
    }

    #[test]
    fn plane_resolution_rejects_out_of_bounds_bindings_before_first_call() {
        let bytes = compiled_application_bytes();
        let (_, outputs) = compiled_bindings();
        let inputs = vec![
            CompiledDirectPlaneBinding::Momentum {
                form: 1,
                lorentz_component: 0,
            },
            CompiledDirectPlaneBinding::Zero,
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: false,
            }),
            CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: true,
            }),
        ];
        let loaded = LoadedSymjitCompiledDirectStage::load_bytes(
            &bytes,
            "compiled-bounds.symjit",
            DIRECT_APPLICATION_STORAGE_ABI,
            inputs,
            vec![],
            outputs,
        )
        .unwrap();
        let mut workspace = DirectArenaWorkspace::new(4, 0, 8).unwrap();
        workspace.begin_tile(8).unwrap();
        let stride = workspace.point_stride();
        let momenta =
            AlignedF64Buffer::zeroed(4 * stride as usize, "compiled bounds momenta").unwrap();
        let zero_plane = AlignedF64Buffer::zeroed(stride as usize, "compiled shared zero").unwrap();
        let arena = workspace.view().unwrap();
        let error = match unsafe {
            loaded.bind(
                arena,
                momentum_view(&momenta, stride),
                empty_parameters(),
                &zero_plane,
            )
        } {
            Ok(_) => panic!("out-of-bounds momentum binding unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
        assert!(error.message().contains("momentum binding"));
    }
}
