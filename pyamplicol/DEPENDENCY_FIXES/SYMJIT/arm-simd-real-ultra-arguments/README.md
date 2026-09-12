# AArch64 SIMD packed arguments are stored instead of loaded

Upstream: [`siravan/symjit-crate`, `f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/tree/f1c193d301897149de6609f706297b0c97a4f018)
(SymJIT 2.25).

The real-vector `save_args` ultra callback emits `str q(arg)` where it
must load the caller's stack argument with `ldr q(arg)`. The scalar and
complex-vector callbacks already load correctly. This produces wrong
results in O3 compressed real kernels whose argument locations all fit
the packed-stack calling convention.

The supplied MRE retains three repeated expressions of squared inputs.
Before the fix, its first output is 81 instead of 13; after the one-line
fix all outputs and lanes agree exactly. Tested on AArch64 with
`default-features=false`. This reproducer prints rather than asserts the
return status, so the independent compressed-success-status bug does not
mask the numerical failure. The standalone project pins the affected revision.

```sh
cargo test --manifest-path DEPENDENCY_FIXES/SYMJIT/arm-simd-real-ultra-arguments/Cargo.toml
```

For an upstream checkout, copy `mre.rs` into `tests/` and apply
`suggested-fix.patch`. No Python, Symbolica, or pyAmpliCol is needed.
