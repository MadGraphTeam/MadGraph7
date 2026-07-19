// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{EagerKernelBackend, EagerKernelCall};

pub(super) struct PreparedEvaluatorBackend {
    kernels: BTreeMap<u32, PreparedEvaluatorKernel>,
}

struct PreparedEvaluatorKernel {
    evaluator: EvaluatorGroup,
    #[cfg(feature = "symbolica-runtime")]
    input_scratch: Vec<Complex<f64>>,
    output_scratch: Vec<Complex<f64>>,
}

impl PreparedEvaluatorBackend {
    pub(super) fn load(
        pack: &PreparedKernelPackManifest,
        payload_root: &Path,
    ) -> RusticolResult<Self> {
        let expected_capability = match pack.backend.as_str() {
            "jit" => SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            "asm" => SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            "cpp" => SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            other => {
                return Err(RusticolError::artifact(format!(
                    "unsupported prepared backend {other:?}"
                )));
            }
        };
        let mut kernels = BTreeMap::new();
        for kernel in &pack.kernels {
            if kernel.contract_kind == "model-parameter" {
                continue;
            }
            let evaluator_manifest = kernel.runtime_evaluator_manifest()?;
            let actual = evaluator_runtime_capabilities(&evaluator_manifest)?;
            if actual != BTreeSet::from([expected_capability.to_string()]) {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} backend {:?} declares evaluator capabilities {actual:?}",
                    kernel.kernel_id, pack.backend
                )));
            }
            ensure_evaluator_capabilities_supported(&evaluator_manifest)?;
            let (input_len, output_len) = evaluator_manifest.io_len()?;
            if input_len != kernel.input_arity
                || output_len != usize::try_from(kernel.output_arity).unwrap_or(usize::MAX)
            {
                return Err(RusticolError::integrity(format!(
                    "prepared kernel {} evaluator I/O ({input_len}, {output_len}) does not match ({}, {})",
                    kernel.kernel_id, kernel.input_arity, kernel.output_arity
                )));
            }
            let evaluator = EvaluatorGroup::load(&evaluator_manifest, payload_root)?;
            if kernels
                .insert(
                    kernel.kernel_id,
                    PreparedEvaluatorKernel {
                        evaluator,
                        #[cfg(feature = "symbolica-runtime")]
                        input_scratch: Vec::new(),
                        output_scratch: Vec::new(),
                    },
                )
                .is_some()
            {
                return Err(RusticolError::integrity(format!(
                    "prepared evaluator backend repeats kernel {}",
                    kernel.kernel_id
                )));
            }
        }
        Ok(Self { kernels })
    }
}

impl EagerKernelBackend for PreparedEvaluatorBackend {
    fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()> {
        let kernel = self.kernels.get_mut(&call.kernel_id).ok_or_else(|| {
            RusticolError::artifact(format!(
                "eager plan references unloaded prepared kernel {}",
                call.kernel_id
            ))
        })?;
        if call.input_component_count != kernel.evaluator.input_len
            || call.output_component_count != kernel.evaluator.output_len
        {
            return Err(RusticolError::integrity(format!(
                "eager call for kernel {} has I/O ({}, {}), expected ({}, {})",
                call.kernel_id,
                call.input_component_count,
                call.output_component_count,
                kernel.evaluator.input_len,
                kernel.evaluator.output_len,
            )));
        }
        let input_len = call
            .lane_count
            .checked_mul(call.input_component_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager input packet overflows"))?;
        let output_len = call
            .lane_count
            .checked_mul(call.output_component_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager output packet overflows"))?;
        if call.inputs.len() != input_len || call.outputs.len() != output_len {
            return Err(RusticolError::internal(
                "eager scheduler passed an inconsistent kernel packet",
            ));
        }

        #[cfg(feature = "symbolica-runtime")]
        let evaluator_inputs = {
            kernel
                .input_scratch
                .resize(input_len, Complex::new(0.0, 0.0));
            for (target, source) in kernel.input_scratch.iter_mut().zip(call.inputs.iter()) {
                *target = Complex::new(source.re, source.im);
            }
            kernel.input_scratch.as_slice()
        };
        #[cfg(not(feature = "symbolica-runtime"))]
        let evaluator_inputs = call.inputs;

        kernel.evaluator.evaluate_batch_into(
            call.lane_count,
            evaluator_inputs,
            &mut kernel.output_scratch,
        )?;
        if kernel.output_scratch.len() != output_len {
            return Err(RusticolError::internal(
                "prepared evaluator returned an inconsistent output packet",
            ));
        }
        #[cfg(feature = "symbolica-runtime")]
        for (target, source) in call.outputs.iter_mut().zip(kernel.output_scratch.iter()) {
            *target = crate::EagerComplex64::new(source.re, source.im);
        }
        #[cfg(not(feature = "symbolica-runtime"))]
        call.outputs.copy_from_slice(&kernel.output_scratch);
        Ok(())
    }
}
