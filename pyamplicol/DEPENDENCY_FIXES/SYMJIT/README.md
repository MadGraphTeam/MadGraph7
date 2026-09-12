# SymJIT fixes found while updating pyAmpliCol

Affected baseline: SymJIT 2.25.0,
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/commit/f1c193d301897149de6609f706297b0c97a4f018).
Each directory contains a minimal reproducer, a suggested source patch and
focused regression tests. The full dependency checkout is not included.

## Newly identified

| Issue | Consequence | Report |
| --- | --- | --- |
| Compressed argument-count overflow | Crashes or incorrect results at 64 or more arguments when the `symbolica` feature is enabled | [Argument counts](argument-count-overflow/README.md) |
| Compressed normal exit jumps to the error epilogue | Correct SIMD results return a false failure status, triggering scalar reevaluation | [Normal completion](compressed-success-status/README.md) |
| ARM SIMD packed-stack argument load is a store | Incorrect real-valued arithmetic with O3 compression | [SIMD argument loading](arm-simd-real-ultra-arguments/README.md) |
| ARM kernels do not preserve all callee-saved FP registers | Corruption of caller state, including with compression disabled | [ARM register preservation](arm-callee-saved-registers/README.md) |
| Inconsistent real-input location coordinates | Wrong complex values in indirect translation; lost real specialization in arena kernels | [Real-input locations](real-input-locations/README.md) |
| Compressed helpers mishandle complex return values | A returned imaginary component is discarded or left stale after a real temporary | [Complex return values](compressed-complex-return/README.md) |

## Earlier fixes still required by this setup

- [Output reuse](output-reuse/README.md): output values consumed by other
  outputs must also retain their final publication use.
- [Unused trailing inputs](unused-trailing-inputs/README.md): direct-arena
  outputs must start after all declared inputs, not only referenced ones.

These are dependency correctness issues, not downstream adaptations to changed
APIs. The individual reports identify the relevant source paths, expected and
observed behaviour, and how to run their reproducer. Tests were performed on
Apple Silicon macOS; architecture-specific reports are labelled explicitly.

## Validation and application

The earlier six patches pass a combined application check against the exact upstream
revision above. The combined focused regression tests pass locally; individual
reports record the before/after evidence and feature configurations. Suggested
application order is output reuse, unused trailing inputs, argument counts,
normal completion, SIMD argument loading, ARM register preservation, then the
real-input location fix, followed by the compressed-complex-return fix.
The additional location fix passes its focused
scalar/SIMD and serialization tests and an application check against the
upstream source.

The reports are ready for upstream review. Both local native runtimes have
been rebuilt with all eight fixes and installed. The real-location and
compressed-return reproducers pass through both installed libraries. Fresh
SM/HEFT generation and scalar/batch and higher-precision checks passed after
the real-location fix. After the compressed-return fix, the previously failing
top-pair process also passes in double precision at batches of 1, 2 and 128
points, with a higher-precision cross-check. The normal-completion report
includes the repaired whole-process timing. The paper benchmark refresh is
complete: the leading-colour comparisons through seven final-state particles
and the two full-colour families through five pass their numerical checks.
No further dependency defect was found in the remaining runs.
