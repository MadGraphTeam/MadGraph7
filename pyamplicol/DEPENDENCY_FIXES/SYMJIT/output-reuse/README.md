# SymJIT 2.25 output-reuse regression

Verified upstream bug report, 11 September 2026.

A separate incorrect-output regression is documented in
[unused-trailing-inputs](../unused-trailing-inputs/README.md).

**Affected:** 2.25.0,
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/tree/f1c193d301897149de6609f706297b0c97a4f018).
The same minimal program succeeds with 2.22.0,
[`d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`](https://github.com/siravan/symjit-crate/tree/d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7).
Observed on Apple Silicon macOS; the fault is in the architecture-independent
indirect translator.

## Reproduce

From this directory:

```sh
cargo run --release -- copy
```

The standalone Cargo project pins the affected revision and needs neither
Symbolica nor pyAmpliCol. Its entire instruction stream is:

```text
out0 = param0
out1 = out0
```

**Expected:** compilation succeeds; both outputs equal `param0`.
**Actual:** compilation panics:

```text
called `Result::unwrap()` on an `Err` value: variable __Static0 not found
```

This also affected a real 322-instruction, 191-output HEFT kernel, which failed
with `__Static56 not found`; the identical preparation succeeded with 2.22.

## Cause and proposed fix

`IndirectTranslator` counts instruction reads of each SSA static, but omits
its final publication as an output. Consequently `out0` appears single-use:
its expression is inlined into `out1`, leaving no named static for the final
output-store loop.

[suggested-fix.patch](suggested-fix.patch) makes two changes:

1. Before lowering instructions, count the final store of each output's
   **last** SSA assignment as an additional use.
2. Read the final value through `expr(Static, false)`, consuming its cached
   expression when present, instead of demanding a named static.

This adds one **O(number of outputs)** compilation pass and no runtime work.
Output-only expressions still inline into their final stores. Earlier values
of overwritten outputs retain their ordinary use counts. The fix applies to
both serialized instructions and the `Composer` API.

## Apply and verify

Replace the absolute example paths below with your checkout paths:

```sh
git clone https://github.com/siravan/symjit-crate.git /absolute/path/to/symjit-fix
git -C /absolute/path/to/symjit-fix checkout f1c193d301897149de6609f706297b0c97a4f018
git -C /absolute/path/to/symjit-fix apply /absolute/path/to/output-reuse/suggested-fix.patch

cargo test --release --manifest-path /absolute/path/to/symjit-fix/Cargo.toml \
  --lib output_use_tests
cargo run --release \
  --config 'patch."https://github.com/siravan/symjit-crate.git".symjit.path="/absolute/path/to/symjit-fix"' \
  -- all
```

**Verified:** the unmodified pinned revision reproduces the panic. The patch
passes all three upstream tests, covering real O0/O2 and complex output reuse,
overwriting, self-reassignment, and SSA use counts. The standalone executable
passes its `copy`, `real`, `complex`, and `overwrite` cases, each at two input
points, and prints four `PASS` lines.

This report retains the verified reproducer and fix from the earlier dependency
upgrade; the campaign refresh did not introduce this issue. Keep this patch
alongside the other required SymJIT fixes until they are available upstream.
