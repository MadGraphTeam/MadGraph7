# Compressed kernels bypass normal completion

Upstream: [`siravan/symjit-crate`, `f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/tree/f1c193d301897149de6609f706297b0c97a4f018)
(SymJIT 2.25).

`Topology::compile` jumps directly to `@epilogue` to skip emitted
subroutines. This bypasses the normal `Builder` register-restoration path
and the SIMD success-status initialization immediately before that label.
On AArch64 a branch-free compressed SIMD kernel consequently returns a
stale nonzero status. The caller interprets that status as lane divergence
and repeats the calculation using scalar kernels. The AMD SIMD epilogues
have the same label ordering; the runtime reproduction here is on AArch64.

In a retained pyAmpliCol ZZ process, the old native/output pair takes
3.378 microseconds/sample and the new pair takes 8.306, with identical
matrix elements. The incorrect SIMD status explains the vector evaluation
followed by two scalar evaluations. This is not a request to disable
compression or change its size/performance tradeoff.

The standalone regression pins the affected upstream revision:

```sh
cargo test --manifest-path DEPENDENCY_FIXES/SYMJIT/compressed-success-status/Cargo.toml
```

For an upstream checkout, copy `mre.rs` into its `tests/` directory. It uses
three repeated branch-free expressions, checks the raw SIMD return status
and all numerical outputs in real/complex O2/O3 modes. The suggested fix
jumps to a label after the subroutine bodies, then falls through the normal
completion path. The label used by genuine divergent branches is unchanged.

Observed raw return before fixing the retained ZZ kernel: `25293344`
(a stale address-dependent value); regenerated fixed kernels all return `0`.
On the same structured programs, compressed and uncompressed fixed outputs
agree within `7.8e-14` relative error on nontrivial complex inputs. Compression
still trades execution speed for smaller code; that policy choice is separate
from this incorrect-status bug.

After installing the rebuilt runtime and regenerating the full process, the
same batched campaign protocol measures 3.733 microseconds/sample instead of
8.306. All seven regenerated compiled n=4 process families pass their numerical
checks. These measurements preceded the separate
[real-input location repair](../real-input-locations/README.md).
Residual compiler performance differences are separate from this false-status failure.

Apply `suggested-fix.patch`; it is independent of the argument-count fix.
Regenerate previously compiled compressed applications: storage-v3 saves
the incorrect MIR branch, so merely relinking their loader does not repair
them. No format compatibility code is needed.
