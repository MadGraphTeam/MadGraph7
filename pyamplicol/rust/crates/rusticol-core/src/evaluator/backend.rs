// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

impl EvaluatorGroup {
    pub(crate) fn load(manifest: &EvaluatorManifest, root: &Path) -> RusticolResult<Self> {
        ensure_evaluator_capabilities_supported(manifest)?;
        let mut evaluators = Vec::new();
        flatten_evaluators(manifest, root, &mut evaluators)?;
        let output_len = evaluators.iter().map(|e| e.output_len).sum();
        Ok(Self {
            evaluators,
            output_len,
            chunk_scratch_f64: Vec::new(),
        })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        let mut out = Vec::new();
        self.evaluate_batch_into(batch_size, params, &mut out)?;
        Ok(out)
    }

    pub(crate) fn evaluate_single_row(
        &mut self,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        self.evaluate_batch(1, params)
    }

    pub(crate) fn evaluate_batch_into(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
    ) -> RusticolResult<()> {
        let expected_output_len = batch_size * self.output_len;
        if out.len() != expected_output_len {
            out.resize(expected_output_len, c64(0.0, 0.0));
        }
        if self.evaluators.len() == 1 {
            let evaluator = &mut self.evaluators[0];
            if params.len() != batch_size * evaluator.input_len {
                return Err(RusticolError::invalid_argument(format!(
                    "parameter buffer has length {}, expected {}",
                    params.len(),
                    batch_size * evaluator.input_len
                )));
            }
            evaluator.evaluate_f64_batch(batch_size, params, out)?;
            return Ok(());
        }
        let mut output_offset = 0;
        for evaluator in &mut self.evaluators {
            if params.len() != batch_size * evaluator.input_len {
                return Err(RusticolError::invalid_argument(format!(
                    "parameter buffer has length {}, expected {}",
                    params.len(),
                    batch_size * evaluator.input_len
                )));
            }
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            evaluator.evaluate_f64_batch(batch_size, params, &mut self.chunk_scratch_f64)?;
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .copy_from_slice(&self.chunk_scratch_f64[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_batch_generic<T>(
        &mut self,
        batch_size: usize,
        params: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<Vec<Complex<T>>>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut out = vec![complex_zero::<T>(); batch_size * self.output_len];
        let mut output_offset = 0;
        for evaluator in &mut self.evaluators {
            if params.len() != batch_size * evaluator.input_len {
                return Err(RusticolError::invalid_argument(format!(
                    "parameter buffer has length {}, expected {}",
                    params.len(),
                    batch_size * evaluator.input_len
                )));
            }
            let mut chunk_out = vec![complex_zero::<T>(); batch_size * evaluator.output_len];
            for row in 0..batch_size {
                let in_start = row * evaluator.input_len;
                let out_start = row * evaluator.output_len;
                T::evaluate_loaded(
                    evaluator,
                    &params[in_start..in_start + evaluator.input_len],
                    &mut chunk_out[out_start..out_start + evaluator.output_len],
                    binary_precision,
                )?;
            }
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .clone_from_slice(&chunk_out[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(out)
    }
}

impl LoadedEvaluator {
    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn exact_evaluator(
        &mut self,
    ) -> RusticolResult<&ExpressionEvaluator<Complex<Rational>>> {
        if self.exact_eval.is_none() {
            let path = self.exact_eval_path.as_ref().ok_or_else(|| {
                RusticolError::invalid_argument(
                    "high-precision evaluation requires an evaluator-state artifact, but this process artifact has no evaluator_state_path for one or more chunks",
                )
            })?;
            self.exact_eval = Some(load_evaluator_state(path)?.1);
        }
        Ok(self
            .exact_eval
            .as_ref()
            .expect("exact evaluator initialized from its state path"))
    }

    pub(crate) fn evaluate_f64_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        match &mut self.eval {
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(eval) => {
                if params.len() != batch_size * self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "parameter buffer has length {}, expected {}",
                        params.len(),
                        batch_size * self.input_len
                    )));
                }
                if out.len() != batch_size * self.output_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "output buffer has length {}, expected {}",
                        out.len(),
                        batch_size * self.output_len
                    )));
                }
                eval.evaluate_batch(batch_size, params, out)
                    .map_err(RusticolError::evaluation)
            }
        }
    }
}

pub(crate) fn flatten_evaluators(
    manifest: &EvaluatorManifest,
    root: &Path,
    output: &mut Vec<LoadedEvaluator>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability,
            application_path,
            application_abi,
            input_len,
            output_len,
            element_layout,
            batch_layout,
            compiler_type,
            translation_mode,
            optimization_level,
            word_bits,
            endianness,
            required_defuns,
            evaluator_state_path,
            evaluator_state_runtime_capability,
        } => {
            #[cfg(feature = "f64-symjit")]
            {
                let eval = SymjitApplicationEvaluator::load(
                    &artifact_path(root, application_path)?,
                    SymjitApplicationMetadata {
                        runtime_capability,
                        application_abi,
                        input_len: *input_len,
                        output_len: *output_len,
                        element_layout,
                        batch_layout,
                        compiler_type,
                        translation_mode,
                        optimization_level: *optimization_level,
                        word_bits: *word_bits,
                        endianness,
                        required_defuns,
                    },
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_path = match evaluator_state_path {
                    Some(state_path) => {
                        if evaluator_state_runtime_capability.as_deref()
                            != Some(SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY)
                        {
                            return Err(RusticolError::compatibility(format!(
                                "SymJIT evaluator state {:?} does not declare capability {:?}",
                                state_path, SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
                            )));
                        }
                        Some(artifact_path(root, state_path)?)
                    }
                    None => None,
                };
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = (evaluator_state_path, evaluator_state_runtime_capability);
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::SymjitApplication(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_path,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-symjit"))]
            {
                let _ = (
                    application_path,
                    application_abi,
                    input_len,
                    output_len,
                    element_layout,
                    batch_layout,
                    compiler_type,
                    translation_mode,
                    optimization_level,
                    word_bits,
                    endianness,
                    required_defuns,
                    evaluator_state_path,
                    evaluator_state_runtime_capability,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Jit {
            runtime_capability,
            input_len,
            output_len,
            evaluator_state_path,
        } => {
            #[cfg(feature = "symbolica-runtime")]
            {
                let _ = runtime_capability;
                let (jit_settings, exact_eval, jit_eval) =
                    load_evaluator_state(&artifact_path(root, evaluator_state_path)?)?;
                let eval = F64Evaluator::Jit(match jit_eval {
                    Some(eval) => eval,
                    None => exact_eval
                        .jit_compile::<Complex<f64>>(jit_settings)
                        .map_err(|err| {
                            RusticolError::evaluation(format!(
                                "could not compile scalar JIT evaluator from {}: {err}",
                                evaluator_state_path
                            ))
                        })?,
                });
                output.push(LoadedEvaluator {
                    eval,
                    exact_eval: Some(exact_eval),
                    exact_eval_path: None,
                    double_eval: None,
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "symbolica-runtime"))]
            {
                let _ = (input_len, output_len, evaluator_state_path);
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::CompiledComplex {
            runtime_capability,
            function_name,
            input_len,
            output_len,
            library_path,
            evaluator_state_path,
            number_type,
        } => {
            #[cfg(feature = "f64-compiled")]
            {
                let _ = runtime_capability;
                if number_type != "complex" {
                    return Err(RusticolError::invalid_argument(format!(
                        "rusticol currently supports compiled complex evaluators, got {number_type}"
                    )));
                }
                let library = artifact_path(root, library_path)?;
                let eval = CompiledComplexF64Evaluator::load(
                    &library,
                    function_name,
                    *input_len,
                    *output_len,
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_path = evaluator_state_path
                    .as_deref()
                    .map(|state_path| artifact_path(root, state_path))
                    .transpose()?;
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = evaluator_state_path;
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::Compiled(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_path,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-compiled"))]
            {
                let _ = (
                    function_name,
                    input_len,
                    output_len,
                    library_path,
                    evaluator_state_path,
                    number_type,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Chunked { chunks, .. } => {
            for chunk in chunks {
                flatten_evaluators(chunk, root, output)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn ensure_evaluator_capabilities_supported(
    manifest: &EvaluatorManifest,
) -> RusticolResult<()> {
    let declared = evaluator_runtime_capabilities(manifest)?;
    ensure_runtime_capabilities_supported(declared.iter().map(String::as_str))
}

pub(crate) fn evaluator_runtime_capabilities(
    manifest: &EvaluatorManifest,
) -> RusticolResult<BTreeSet<String>> {
    let mut capabilities = BTreeSet::new();
    collect_evaluator_capabilities(manifest, &mut capabilities)?;
    Ok(capabilities)
}

fn collect_evaluator_capabilities(
    manifest: &EvaluatorManifest,
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMJIT_APPLICATION_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::Jit {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::CompiledComplex {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[
                SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
                SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            ],
            output,
        ),
        EvaluatorManifest::Chunked {
            required_runtime_capabilities,
            chunks,
        } => {
            let mut actual = BTreeSet::new();
            for chunk in chunks {
                collect_evaluator_capabilities(chunk, &mut actual)?;
            }
            let declared = required_runtime_capabilities
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            if declared.len() != required_runtime_capabilities.len() || declared != actual {
                return Err(RusticolError::integrity(format!(
                    "chunked evaluator capabilities {:?} do not match child capabilities {:?}",
                    required_runtime_capabilities, actual
                )));
            }
            output.extend(actual);
            Ok(())
        }
    }
}

fn validate_and_insert_capability(
    capability: &str,
    expected: &[&str],
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    if !expected.contains(&capability) {
        return Err(RusticolError::compatibility(format!(
            "evaluator declares runtime capability {capability:?}, expected one of {expected:?}"
        )));
    }
    output.insert(capability.to_string());
    Ok(())
}

#[cfg(any(
    not(feature = "f64-compiled"),
    not(feature = "f64-symjit"),
    not(feature = "symbolica-runtime")
))]
fn unsupported_capability(capability: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        capability,
        format!(
            "this Rusticol build supports {:?}",
            supported_runtime_capabilities()
        ),
    )
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn load_evaluator_state(
    path: &Path,
) -> RusticolResult<(
    JITCompilationSettings,
    ExpressionEvaluator<Complex<Rational>>,
    Option<JITCompiledEvaluator<Complex<f64>>>,
)> {
    type SavedEvaluator = (
        bool,
        JITCompilationSettings,
        ExpressionEvaluator<Complex<Rational>>,
        Option<JITCompiledEvaluator<f64>>,
        Option<JITCompiledEvaluator<Complex<f64>>>,
    );

    let bytes = fs::read(path).map_err(|err| {
        RusticolError::artifact(format!(
            "could not read evaluator state {}: {err}",
            path.display()
        ))
    })?;
    let ((_, settings, evaluator, _, jit_complex), consumed) =
        bincode::decode_from_slice::<SavedEvaluator, _>(&bytes, bincode::config::standard())
            .map_err(|error| {
                RusticolError::compatibility(format!(
                    "evaluator state {} does not match Symbolica serialization ABI {}: {error}; regenerate the schema-v3 artifact",
                    path.display(),
                    crate::SYMBOLICA_SERIALIZATION_ABI
                ))
            })?;
    ensure_evaluator_state_consumed(path, bytes.len(), consumed)?;
    Ok((settings, evaluator, jit_complex))
}

#[cfg(feature = "symbolica-runtime")]
fn ensure_evaluator_state_consumed(
    path: &Path,
    encoded_len: usize,
    consumed: usize,
) -> RusticolResult<()> {
    if consumed == encoded_len {
        return Ok(());
    }
    Err(RusticolError::integrity(format!(
        "evaluator state {} contains {} trailing bytes",
        path.display(),
        encoded_len - consumed
    )))
}
