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
            chunk_scratch_native2: Vec::new(),
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
        if !self.supports_native2() {
            return self.evaluate_batch(1, params);
        }
        let packed = params
            .iter()
            .map(|value| {
                Complex::new(
                    wide::f64x2::new([value.re, value.re]),
                    wide::f64x2::new([value.im, value.im]),
                )
            })
            .collect::<Vec<_>>();
        let mut native_output = Vec::new();
        self.evaluate_native2_into(1, &packed, &mut native_output)?;
        Ok(native_output
            .into_iter()
            .map(|value| c64(value.re.as_array()[0], value.im.as_array()[0]))
            .collect())
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

    pub(crate) fn supports_native2(&self) -> bool {
        !self.evaluators.is_empty()
            && self
                .evaluators
                .iter()
                .all(|evaluator| match &evaluator.eval {
                    #[cfg(feature = "f64-symjit")]
                    F64Evaluator::SymjitApplication(evaluator) => evaluator.supports_native2(),
                    #[cfg(feature = "symbolica-runtime")]
                    F64Evaluator::JitNative2(_) => true,
                    #[cfg(feature = "symbolica-runtime")]
                    F64Evaluator::Compiled(_) | F64Evaluator::Jit(_) => false,
                })
    }

    pub(crate) fn evaluate_native2_into(
        &mut self,
        native_rows: usize,
        params: &[Complex<wide::f64x2>],
        out: &mut Vec<Complex<wide::f64x2>>,
    ) -> RusticolResult<()> {
        let expected_output_len = native_rows * self.output_len;
        if out.len() != expected_output_len {
            out.resize(
                expected_output_len,
                Complex::new(wide::f64x2::ZERO, wide::f64x2::ZERO),
            );
        }
        if self.evaluators.len() == 1 {
            return self.evaluators[0].evaluate_native2_batch(native_rows, params, out);
        }

        let mut output_offset = 0;
        for evaluator in &mut self.evaluators {
            self.chunk_scratch_native2.resize(
                native_rows * evaluator.output_len,
                Complex::new(wide::f64x2::ZERO, wide::f64x2::ZERO),
            );
            evaluator.evaluate_native2_batch(
                native_rows,
                params,
                &mut self.chunk_scratch_native2,
            )?;
            for row in 0..native_rows {
                let source_start = row * evaluator.output_len;
                let target_start = row * self.output_len + output_offset;
                out[target_start..target_start + evaluator.output_len].copy_from_slice(
                    &self.chunk_scratch_native2[source_start..source_start + evaluator.output_len],
                );
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
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Compiled(eval) => eval
                .evaluate_batch(batch_size, params, out)
                .map_err(RusticolError::evaluation),
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
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::JitNative2(eval) => {
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

    fn evaluate_native2_batch(
        &mut self,
        native_rows: usize,
        params: &[Complex<wide::f64x2>],
        out: &mut [Complex<wide::f64x2>],
    ) -> RusticolResult<()> {
        if params.len() != native_rows * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "native parameter buffer has length {}, expected {}",
                params.len(),
                native_rows * self.input_len
            )));
        }
        if out.len() != native_rows * self.output_len {
            return Err(RusticolError::invalid_argument(format!(
                "native output buffer has length {}, expected {}",
                out.len(),
                native_rows * self.output_len
            )));
        }
        match &mut self.eval {
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => {
                eval.evaluate_native2_batch(native_rows, params, out)
            }
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::JitNative2(eval) => {
                eval.batch_evaluate(params, out, native_rows);
                Ok(())
            }
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Compiled(_) | F64Evaluator::Jit(_) => Err(RusticolError::evaluation(
                "native two-lane evaluation requested for a non-native evaluator",
            )),
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
                let (jit_settings, exact_eval, jit_eval, jit_native2) =
                    load_evaluator_state(&artifact_path(root, evaluator_state_path)?)?;
                let eval = if native_simd_jit_enabled() {
                    F64Evaluator::JitNative2(match jit_native2 {
                        Some(eval) => eval,
                        None => exact_eval
                            .jit_compile::<Complex<wide::f64x2>>(jit_settings.clone())
                            .map_err(|err| {
                                RusticolError::evaluation(format!(
                                    "could not compile native two-lane JIT evaluator from {}: {err}",
                                    evaluator_state_path
                                ))
                            })?,
                    })
                } else {
                    F64Evaluator::Jit(match jit_eval {
                        Some(eval) => eval,
                        None => exact_eval
                            .jit_compile::<Complex<f64>>(jit_settings)
                            .map_err(|err| {
                                RusticolError::evaluation(format!(
                                    "could not compile scalar JIT evaluator from {}: {err}",
                                    evaluator_state_path
                                ))
                            })?,
                    })
                };
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
            #[cfg(feature = "symbolica-runtime")]
            {
                let _ = runtime_capability;
                if number_type != "complex" {
                    return Err(RusticolError::invalid_argument(format!(
                        "rusticol currently supports compiled complex evaluators, got {number_type}"
                    )));
                }
                let library = artifact_path(root, library_path)?;
                let eval =
                    CompiledComplexEvaluator::load(&library, function_name).map_err(|err| {
                        RusticolError::evaluation(format!(
                            "could not load compiled evaluator {} from {}: {err}",
                            function_name,
                            library.display()
                        ))
                    })?;
                let exact_eval_path = evaluator_state_path
                    .as_deref()
                    .map(|state_path| artifact_path(root, state_path))
                    .transpose()?;
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::Compiled(eval),
                    exact_eval: None,
                    exact_eval_path,
                    double_eval: None,
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "symbolica-runtime"))]
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

#[cfg(any(not(feature = "f64-symjit"), not(feature = "symbolica-runtime")))]
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
    Option<JITCompiledEvaluator<Complex<wide::f64x2>>>,
)> {
    type SavedEvaluatorNative2 = (
        bool,
        JITCompilationSettings,
        ExpressionEvaluator<Complex<Rational>>,
        Option<JITCompiledEvaluator<f64>>,
        Option<JITCompiledEvaluator<Complex<f64>>>,
        Option<JITCompiledEvaluator<Complex<wide::f64x2>>>,
    );
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
    match bincode::decode_from_slice::<SavedEvaluatorNative2, _>(
        &bytes,
        bincode::config::standard(),
    ) {
        Ok(((_, settings, evaluator, _, jit_complex, jit_native2), consumed)) => {
            ensure_evaluator_state_consumed(path, bytes.len(), consumed)?;
            Ok((settings, evaluator, jit_complex, jit_native2))
        }
        Err(native2_error) => {
            let ((_, settings, evaluator, _, jit_complex), consumed) =
                bincode::decode_from_slice::<SavedEvaluator, _>(
                    &bytes,
                    bincode::config::standard(),
                )
                .map_err(|legacy_error| {
                    RusticolError::compatibility(format!(
                        "evaluator state {} does not match Symbolica serialization ABI {}: {native2_error}; {legacy_error}; regenerate the schema-v3 artifact",
                        path.display(),
                        crate::SYMBOLICA_SERIALIZATION_ABI
                    ))
                })?;
            ensure_evaluator_state_consumed(path, bytes.len(), consumed)?;
            Ok((settings, evaluator, jit_complex, None))
        }
    }
}

#[cfg(feature = "symbolica-runtime")]
fn native_simd_jit_enabled() -> bool {
    if !cfg!(target_arch = "aarch64") {
        return false;
    }
    std::env::var("RUSTICOL_NATIVE_SIMD_JIT")
        .ok()
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
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
