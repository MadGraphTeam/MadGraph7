// SPDX-License-Identifier: 0BSD

use super::super::*;
use symjit::{Applet, Application, Config, Defuns, Storage};

pub(crate) struct SymjitApplicationEvaluator {
    applet: Applet,
    input_len: usize,
    output_len: usize,
}

impl std::fmt::Debug for SymjitApplicationEvaluator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitApplicationEvaluator")
            .field("input_len", &self.input_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

pub(crate) struct SymjitApplicationMetadata<'a> {
    pub(crate) runtime_capability: &'a str,
    pub(crate) application_abi: &'a str,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) element_layout: &'a str,
    pub(crate) batch_layout: &'a str,
    pub(crate) compiler_type: &'a str,
    pub(crate) translation_mode: &'a str,
    pub(crate) optimization_level: u8,
    pub(crate) word_bits: u8,
    pub(crate) endianness: &'a str,
    pub(crate) required_defuns: &'a [String],
}

impl SymjitApplicationEvaluator {
    pub(crate) fn load(
        path: &Path,
        metadata: SymjitApplicationMetadata<'_>,
    ) -> RusticolResult<Self> {
        validate_manifest_metadata(&metadata)?;

        let bytes = fs::read(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read SymJIT application {}: {error}",
                path.display()
            ))
        })?;

        // This is the same trusted-input path used by Symbolica's
        // JITCompiledEvaluator::load.
        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let application =
            Application::load(&mut bytes.as_slice(), &loader_config).map_err(|error| {
                RusticolError::compatibility(format!(
                    "could not load SymJIT application {} with ABI {}: {error}",
                    path.display(),
                    SYMJIT_APPLICATION_STORAGE_ABI
                ))
            })?;
        validate_loaded_application(path, &application, &metadata)?;
        let applet = application.seal().map_err(|error| {
            RusticolError::evaluation(format!(
                "could not seal SymJIT application {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            applet,
            input_len: metadata.input_len,
            output_len: metadata.output_len,
        })
    }

    pub(crate) fn evaluate_batch(
        &self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        validate_batch_lengths(
            batch_size,
            self.input_len,
            self.output_len,
            params.len(),
            out.len(),
        )?;
        let params = complex_slice_as_scalars(params);
        let out = complex_slice_as_scalars_mut(out);
        self.applet.evaluate_matrix(params, out, batch_size);
        Ok(())
    }
}

fn validate_manifest_metadata(metadata: &SymjitApplicationMetadata<'_>) -> RusticolResult<()> {
    if metadata.runtime_capability != SYMJIT_APPLICATION_RUNTIME_CAPABILITY {
        return Err(RusticolError::unsupported_runtime_capability(
            metadata.runtime_capability,
            format!("a direct SymJIT evaluator requires {SYMJIT_APPLICATION_RUNTIME_CAPABILITY:?}"),
        ));
    }
    if metadata.application_abi != SYMJIT_APPLICATION_STORAGE_ABI {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application ABI {:?} is unsupported; this runtime requires {:?}",
            metadata.application_abi, SYMJIT_APPLICATION_STORAGE_ABI
        )));
    }
    if metadata.input_len == 0 || metadata.output_len == 0 {
        return Err(RusticolError::invalid_argument(
            "SymJIT application input_len and output_len must both be positive",
        ));
    }
    if metadata.element_layout != "complex-f64" || metadata.batch_layout != "row-major" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT application layout element={:?}, batch={:?}",
            metadata.element_layout, metadata.batch_layout
        )));
    }
    if metadata.compiler_type != "native" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT compiler type {:?}; expected \"native\"",
            metadata.compiler_type
        )));
    }
    if metadata.translation_mode != "indirect" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT translation {:?}; direct translation is not a stable artifact ABI",
            metadata.translation_mode
        )));
    }
    if metadata.word_bits != usize::BITS as u8 || metadata.word_bits != 64 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application requires a {}-bit runtime, but this host uses {} bits",
            metadata.word_bits,
            usize::BITS
        )));
    }
    let host_endianness = if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    };
    if metadata.endianness != host_endianness || metadata.endianness != "little" {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application endianness {:?} is incompatible with host endianness {host_endianness:?}",
            metadata.endianness
        )));
    }
    if !metadata.required_defuns.is_empty() {
        return Err(RusticolError::unsupported_runtime_capability(
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            format!(
                "external functions are not self-contained: {}",
                metadata.required_defuns.join(", ")
            ),
        ));
    }
    Ok(())
}

fn validate_loaded_application(
    path: &Path,
    application: &Application,
    metadata: &SymjitApplicationMetadata<'_>,
) -> RusticolResult<()> {
    let expected_params = metadata.input_len.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT complex input count overflows usize")
    })?;
    let expected_outputs = metadata.output_len.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT complex output count overflows usize")
    })?;
    if application.count_states != 0
        || application.count_diffs != 0
        || application.count_params != expected_params
        || application.count_obs != expected_outputs
    {
        return Err(RusticolError::integrity(format!(
            "SymJIT application {} has counts states={}, params={}, outputs={}, diffs={}; expected 0, {}, {}, 0",
            path.display(),
            application.count_states,
            application.count_params,
            application.count_obs,
            application.count_diffs,
            expected_params,
            expected_outputs
        )));
    }
    validate_application_config(path, &application.config, metadata)
}

fn validate_application_config(
    path: &Path,
    config: &Config,
    metadata: &SymjitApplicationMetadata<'_>,
) -> RusticolResult<()> {
    if !config.is_complex() || !config.symbolica() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} is not a complex Symbolica evaluator",
            path.display()
        )));
    }
    if config.direct() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} uses direct translation; regenerate it with indirect translation",
            path.display()
        )));
    }
    let host_native = if cfg!(target_arch = "aarch64") {
        config.is_arm64()
    } else if cfg!(target_arch = "x86_64") {
        config.is_amd64()
    } else if cfg!(target_arch = "riscv64") {
        config.is_riscv64()
    } else {
        false
    };
    if !host_native {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} uses compiler type {:?}, which is not native for this host",
            path.display(),
            config.compiler_type()
        )));
    }
    if config.opt_level() != metadata.optimization_level {
        return Err(RusticolError::integrity(format!(
            "SymJIT application {} declares optimization level {} but stores optimization level {}",
            path.display(),
            metadata.optimization_level,
            config.opt_level()
        )));
    }
    Ok(())
}

fn validate_batch_lengths(
    rows: usize,
    input_len: usize,
    output_len: usize,
    params_len: usize,
    output_buffer_len: usize,
) -> RusticolResult<()> {
    let expected_params = rows.checked_mul(input_len).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT parameter buffer length overflows usize")
    })?;
    let expected_outputs = rows.checked_mul(output_len).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT output buffer length overflows usize")
    })?;
    if params_len != expected_params {
        return Err(RusticolError::invalid_argument(format!(
            "parameter buffer has length {params_len}, expected {expected_params}"
        )));
    }
    if output_buffer_len != expected_outputs {
        return Err(RusticolError::invalid_argument(format!(
            "output buffer has length {output_buffer_len}, expected {expected_outputs}"
        )));
    }
    Ok(())
}

fn complex_slice_as_scalars<T>(values: &[Complex<T>]) -> &[T] {
    assert_eq!(
        std::mem::size_of::<Complex<T>>(),
        2 * std::mem::size_of::<T>(),
        "complex storage must be two adjacent scalars"
    );
    assert_eq!(
        std::mem::align_of::<Complex<T>>(),
        std::mem::align_of::<T>(),
        "complex storage must use scalar alignment"
    );
    // SAFETY: Both supported Complex implementations are repr(C) structs containing exactly
    // adjacent `re` and `im` fields. The size/alignment assertions guard this runtime contract.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len() * 2) }
}

fn complex_slice_as_scalars_mut<T>(values: &mut [Complex<T>]) -> &mut [T] {
    assert_eq!(
        std::mem::size_of::<Complex<T>>(),
        2 * std::mem::size_of::<T>(),
        "complex storage must be two adjacent scalars"
    );
    assert_eq!(
        std::mem::align_of::<Complex<T>>(),
        std::mem::align_of::<T>(),
        "complex storage must use scalar alignment"
    );
    // SAFETY: See complex_slice_as_scalars. The mutable borrow guarantees exclusive access.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), values.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use symjit::{Compiler, CompilerType, Storage};

    fn application_bytes() -> Vec<u8> {
        let mut config = Config::new(CompilerType::Native, 0).unwrap();
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_opt_level(3);
        config.set_simd(true);
        let mut compiler = Compiler::with_config(config);
        let instructions = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;
        let application = compiler.translate(instructions.to_string(), 2).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn metadata<'a>(required_defuns: &'a [String]) -> SymjitApplicationMetadata<'a> {
        SymjitApplicationMetadata {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI,
            input_len: 2,
            output_len: 1,
            element_layout: "complex-f64",
            batch_layout: "row-major",
            compiler_type: "native",
            translation_mode: "indirect",
            optimization_level: 3,
            word_bits: 64,
            endianness: "little",
            required_defuns,
        }
    }

    fn write_application(bytes: &[u8], suffix: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rusticol-symjit-{}-{suffix}.symjit",
            std::process::id()
        ));
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    fn direct_manifest(application_path: String) -> EvaluatorManifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string(),
            application_path,
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI.to_string(),
            input_len: 2,
            output_len: 1,
            element_layout: "complex-f64".to_string(),
            batch_layout: "row-major".to_string(),
            compiler_type: "native".to_string(),
            translation_mode: "indirect".to_string(),
            optimization_level: 3,
            word_bits: 64,
            endianness: "little".to_string(),
            required_defuns: Vec::new(),
            evaluator_state_path: Some("absent-exact-evaluator-state.bin".to_string()),
            evaluator_state_runtime_capability: Some(
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY.to_string(),
            ),
        }
    }

    #[test]
    fn direct_application_loads_and_evaluates_complex_batches() {
        let path = write_application(&application_bytes(), "batch");
        let evaluator = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap();
        let params = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(-2.0, 1.5),
            Complex::new(5.0, -0.5),
        ];
        let mut outputs = [Complex::new(0.0, 0.0); 2];
        evaluator.evaluate_batch(2, &params, &mut outputs).unwrap();
        assert_eq!(outputs[0], Complex::new(4.0, 6.0));
        assert_eq!(outputs[1], Complex::new(3.0, 1.0));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_f64_group_load_does_not_read_exact_fallback_state() {
        let path = write_application(&application_bytes(), "lazy-exact");
        let root = path.parent().unwrap();
        let manifest = direct_manifest(path.file_name().unwrap().to_str().unwrap().to_string());

        let capabilities = evaluator_runtime_capabilities(&manifest).unwrap();
        assert_eq!(
            capabilities,
            BTreeSet::from([SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()])
        );

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(1, &[Complex::new(2.0, 3.0), Complex::new(5.0, 7.0)])
            .unwrap();
        assert_eq!(output, vec![Complex::new(7.0, 10.0)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_f64_group_loads_without_exact_fallback_metadata() {
        let path = write_application(&application_bytes(), "no-exact-fallback");
        let root = path.parent().unwrap();
        let mut manifest = direct_manifest(path.file_name().unwrap().to_str().unwrap().to_string());
        let EvaluatorManifest::SymjitApplication {
            evaluator_state_path,
            evaluator_state_runtime_capability,
            ..
        } = &mut manifest
        else {
            unreachable!()
        };
        *evaluator_state_path = None;
        *evaluator_state_runtime_capability = None;

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(1, &[Complex::new(2.0, 3.0), Complex::new(5.0, 7.0)])
            .unwrap();

        assert_eq!(output, vec![Complex::new(7.0, 10.0)]);
        let _ = fs::remove_file(path);
    }

    #[cfg(not(feature = "symbolica-runtime"))]
    #[test]
    fn unsupported_legacy_jit_evaluator_is_rejected_before_payload_access() {
        let manifest = EvaluatorManifest::Jit {
            runtime_capability: SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY.to_string(),
            input_len: 2,
            output_len: 1,
            evaluator_state_path: "absent-legacy-jit-state.bin".to_string(),
        };

        let error = match EvaluatorGroup::load(&manifest, Path::new("/absent-artifact-root")) {
            Ok(_) => panic!("unsupported capability must win over absent payload"),
            Err(error) => error,
        };
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
    }

    #[cfg(not(feature = "f64-compiled"))]
    #[test]
    fn unsupported_compiled_evaluators_are_rejected_before_payload_access() {
        let manifests = [
            EvaluatorManifest::CompiledComplex {
                runtime_capability: SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY.to_string(),
                function_name: "evaluate".to_string(),
                input_len: 2,
                output_len: 1,
                library_path: "absent-cpp-library.so".to_string(),
                evaluator_state_path: None,
                number_type: "complex".to_string(),
            },
            EvaluatorManifest::CompiledComplex {
                runtime_capability: SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY.to_string(),
                function_name: "evaluate".to_string(),
                input_len: 2,
                output_len: 1,
                library_path: "absent-asm-library.so".to_string(),
                evaluator_state_path: None,
                number_type: "complex".to_string(),
            },
        ];

        for manifest in manifests {
            let error = match EvaluatorGroup::load(&manifest, Path::new("/absent-artifact-root")) {
                Ok(_) => panic!("unsupported capability must win over absent payload"),
                Err(error) => error,
            };
            assert_eq!(
                error.kind(),
                crate::RusticolErrorKind::UnsupportedRuntimeCapability
            );
        }
    }

    #[test]
    fn malformed_application_bytes_are_rejected() {
        let path = write_application(b"not-a-symjit-application", "malformed");
        let error = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_application_requires_complex_f64_row_major_layout() {
        let path = write_application(&application_bytes(), "layout");

        let mut wrong_element_layout = metadata(&[]);
        wrong_element_layout.element_layout = "complex-f32";
        let error = SymjitApplicationEvaluator::load(&path, wrong_element_layout).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);

        let mut wrong_batch_layout = metadata(&[]);
        wrong_batch_layout.batch_layout = "column-major";
        let error = SymjitApplicationEvaluator::load(&path, wrong_batch_layout).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn capability_counts_and_external_functions_are_validated() {
        let bytes = application_bytes();
        let path = write_application(&bytes, "metadata");

        let mut wrong_counts = metadata(&[]);
        wrong_counts.output_len = 2;
        let error = SymjitApplicationEvaluator::load(&path, wrong_counts).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);

        let functions = vec!["model_function".to_string()];
        let error = SymjitApplicationEvaluator::load(&path, metadata(&functions)).unwrap_err();
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );

        let mut wrong_capability = metadata(&[]);
        wrong_capability.runtime_capability = SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY;
        let error = SymjitApplicationEvaluator::load(&path, wrong_capability).unwrap_err();
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
        let _ = fs::remove_file(path);
    }
}
