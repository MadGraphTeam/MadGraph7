# Compressed helpers need a consistent complex return value

Confirmed in the published SymJIT **2.25.4** crate. This is independent of the
argument-count, real-input-coordinate and ARM register-preservation fixes.

## Clear reproducer

[`mre.rs`](mre.rs) is a small test of the complex-lowering stage, with readable
MIR. A real parameter is `7`, and a complex runtime input is `3+4i`. A helper
returns one after the caller's previous temporary held the other:

| Previous temporary | New helper result | 2.25.4 returns |
|---|---|---|
| Real | `3+4i` | `3+0i` |
| Complex, imaginary part 4 | `7+0i` | `7+4i` |

The identical inline computations pass. The test checks both scalar and SIMD
execution, resetting output buffers before every call. Parameter zero avoids
the separate real-input coordinate mismatch. There are no helper arguments,
so the argument-passing defects cannot explain this result.

This is explicitly a **compiler-stage unit test**, not a claim that a frontend
produces this exact tiny program. It exercises the same lowering operations:
`Statement::compile` emits a zero-argument MIR call for compressed helpers;
`Mir::rerun` dispatches that to `Complexifier::call_funclet`. Helper exits emitted
by `Topology::compile` dispatch to `Complexifier::ret`.

In a clean extracted 2.25.4 tree:

1. Copy `mre.rs` to `src/symjit/followup_return_mre.rs`.
2. Add `#[cfg(test)] mod followup_return_mre;` at the **end** of
   `src/symjit/mod.rs`, after its module documentation.
3. Run the test, apply the fix, and run it again:

```sh
cargo test --lib followup_return_mre -- --nocapture
patch -p1 < /absolute/path/to/fix.patch
cargo test --lib followup_return_mre -- --nocapture
```

The first run fails with both values in the table; the second passes all
inline/called, real/complex-return, scalar/SIMD cases. No other source fix is
needed for this test.

## Cause and principled fix

`call_funclet` leaves `Reg::Ret` with the previous temporary's real/complex
classification. A subsequent save can therefore zero a genuinely complex
result. Conversely, a helper returning a real value does not necessarily clear
its unused imaginary component.

[`fix.patch`](fix.patch) makes the convention explicit:

- **After** a helper call, mark `Reg::Ret` complex in the compiler's metadata,
  without changing either component of the returned value.
- **Before** helper return, materialize a complete complex value, setting the
  imaginary component to zero if the computed result is real.

Clearing registers before the call is not equivalent: argument setup may have
already populated overlapping registers. This patch leaves ordinary external
function calls, argument passing, and the storage format unchanged.

## How this arose in pyAmpliCol

pyAmpliCol's complex native evaluators select `fast_complex=false` and can
enable compression; see
[`symjit_plane.rs`](../../rust/crates/rusticol-core/src/evaluator/symjit_plane.rs)
and [`symjit_eager_direct.rs`](../../rust/crates/rusticol-core/src/evaluator/symjit_eager_direct.rs).
Matrix-element expressions mix real factors with complex currents, so repeated
subexpressions can cross precisely this call/type boundary.

During the earlier pyAmpliCol investigation, a compressed top-pair evaluator
differed from its uncompressed and higher-precision references by approximately
`5.116e-4` relative. After the local repair, the compressed/uncompressed double
precision comparison agreed within `3.4e-16` for batches of 1, 2 and 128 points.
Those are the [recorded process-level observations](../../DEPENDENCY_FIXES/SYMJIT/compressed-complex-return/README.md),
not a claim of a new full pyAmpliCol build against 2.25.4. The fresh 2.25.4
verification here is the independently reproduced stage-level failure above.
