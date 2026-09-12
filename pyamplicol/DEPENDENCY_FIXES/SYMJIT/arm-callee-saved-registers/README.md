# AArch64 kernels corrupt callee-saved floating-point registers

Affected revision: `f1c193d301897149de6609f706297b0c97a4f018` (SymJIT 2.25.0).

## Reproduce

Place the standalone `mre.rs` in a SymJIT checkout's `tests/arm_abi_mre.rs`
and run on AArch64:

```sh
cargo test --test arm_abi_mre -- --nocapture
cargo test --no-default-features --test arm_abi_mre -- --nocapture
```

This minimal case disables compression and uses O3 complex evaluation, isolating
this bug from the separate compressed-subroutine normal-exit issue. It freshly
generates the program with the same library that executes it;
there is no saved-program compatibility or mixed-feature assumption. It loads
distinct bit-pattern sentinels into d8–d15 before each generated-kernel call,
then checks both the returned numerical values and exact register preservation.
The wrapper preserves its own caller's registers even when the generated code
does not.

Before the fix, numerical outputs pass but the scalar/SIMD calls replace
sentinels with arithmetic inputs. The full regression test additionally exposes
failures in compressed kernels. A host Rust loop keeping floating-point
values in these registers can consequently produce nonsensical timing counters,
comparisons or other values outside the generated function.

## Cause

[AAPCS64 §6.1.2](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#612simd-and-floating-point-registers)
requires preservation of the low 64 bits of v8–v15. ARM entry code saves integer
registers, while Builder's separate FP-save heuristic compares the logical
scratch count to `count_shadows()`. This misses both physical argument registers
used by compressed calls and additional registers written after complex lowering.
The latter explains failures without compression.

## Fix

`fix.patch` computes one ARM callee-save mask from the final MIR's physical
destinations and compressed-call argument shuffles. Public entry/exit code saves
only those d8–d15 registers, outside the working stack, preserving 16-byte stack
alignment. Scalar, SIMD and complex paths, including fast entries, share this
accounting. A small default preparation hook leaves other architectures on their
existing path and avoids duplicate ARM saves in Builder.

A kernel that does not touch these registers has mask zero and emits no extra
save/restore instructions. Saving one or two registers costs one store and one
load; the maximum eight-register case uses four stores and four loads and 64
extra stack bytes. No instruction format or caller API changes are involved.

Focused tests cover 72 fresh applications (compression off/on; O0/O2/O3; real
and both complex lowering modes; arities 1/5/9/16), through scalar and SIMD
kernels with exact register and numerical checks. Unit tests also cover narrow
and empty masks and fast scalar/complex entry points. The standalone test is
the reduced reproducer; the patch includes the broader regression and focused
unit tests. Both default-feature and no-default-feature checks passed on Apple
ARM64, with each guarded dependency test build below 0.4 GiB.

The separate topology normal-exit bug can bypass Builder's old restoration, but
correcting it alone cannot fix the compression-disabled cases above. That fix
and the SIMD ultra-argument fix are intentionally excluded from this patch.
