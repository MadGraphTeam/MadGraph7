# Direct-arena outputs ignore unused trailing inputs

Affected: SymJIT 2.25.0, commit
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/commit/f1c193d301897149de6609f706297b0c97a4f018).

A P-kernel with two declared inputs and `out[0] = 2 * param[0]`
must store its result after **both** inputs, even though the second is unused.
The affected translator instead locates outputs after the highest referenced
input. The declared program shape still includes all inputs, yielding incorrect
or uninitialized outputs. Constant-output programs also fail.

This produced incorrect scalar and batched tree-level matrix elements in a
freshly generated compiled process. The same process generated with SymJIT
2.22 was correct; changing Symbolica versions did not change the failure.

## Reproduce

From this directory, on a native target supporting SIMD:

```sh
cargo run --release -- --expect-bug
```

This standalone program has no Symbolica or pyAmpliCol dependency. It prints
the observed planes and verifies a wrong result in each real/complex,
O0/O2, scalar/SIMD case, including a constant output with no referenced inputs.
The incorrect values are not guaranteed to be deterministic.

## Fix and test

In `IndirectTranslator::translate`, use
`self.count_params.max(self.num_params)` for the direct-arena output base,
matching the already declared input extent. This is a generation-time layout
correction, with no extra runtime work. Non-arena layout is unchanged.

Apply `suggested-fix.patch` to the affected checkout; it also adds a focused
integration test. This patch is independent of the sibling output-reuse fix;
both fixes can be applied together.

```sh
git -C /path/to/symjit apply /path/to/unused-trailing-inputs/suggested-fix.patch
cargo run --release --config \
  'patch."https://github.com/siravan/symjit-crate.git".symjit.path="/path/to/symjit"'
cargo test --release --manifest-path /path/to/symjit/Cargo.toml \
  --test unused_trailing_inputs
```

The fixed reproducer requires every active output lane to be correct and all
input planes to remain unchanged. All eight real/complex, O0/O2 and
trailing-unused/all-unused combinations passed, with scalar and SIMD calls.
Recompiling six captured process-stage programs with the fix also restored
agreement with the original Symbolica evaluator in both execution modes
(maximum relative difference `6.69e-16`). This check used independent buffers,
outside pyAmpliCol's runtime. This fix is already applied in the current local dependency checkout and
campaign runtime. Previously serialized incorrect programs must be regenerated.
A loader-only update cannot repair the incorrect output locations already
encoded in their portable intermediate program.

This report retains the verified reproducer and fix from the earlier dependency
upgrade; the campaign refresh did not introduce this issue. Keep this patch
alongside the other required SymJIT fixes until they are available upstream.
