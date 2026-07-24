# SymJIT ARM64 Direct-Arena Scalar Return-Status MRE

## Summary

On macOS arm64, a successful scalar Direct-Arena application can report a
nonzero machine-call status. `DirectApplet::evaluate_planes()` treats that
undefined return value as an execution failure, and
`direct_call_trampoline()` consequently returns
`DIRECT_STATUS_EXECUTION_FAILED` (`3`).

The generated arithmetic and destination stores complete successfully. The
failure is only the stale value returned in ARM64 register `x0`.

The fix is to define the successful return value in
`ArmGenerator::epilogue_indirect()`:

```diff
diff --git a/rust/arm/scalar.rs b/rust/arm/scalar.rs
@@
     fn epilogue_indirect(
         &mut self,
         cap: usize,
         count_states: usize,
         count_obs: usize,
         _count_params: usize,
     ) {
+        self.emit(arm! {eor x(0), x(0), x(0)});
         self.set_label("@epilogue");
```

This matches the existing success-return handling in the AMD64 scalar
generator (`xor rax, rax`) and ARM64 vector generator.

## Reproducer

The standalone Rust crate is:

```text
tools/developer/symjit_arm64_direct_arena_mre/
```

It depends only on SymJIT. It does not import pyAmpliCol or Rusticol.

The initial generated `x*y` expression was too small to reproduce the issue
because that machine-code path happened to leave `x0` equal to zero. The final
MRE embeds the smallest portable O2 source application found to reproduce the
failure reliably:

- source payload size: 368 bytes;
- SHA-256:
  `e13c88d775a810fd9868e23fb37a600a59660d308c60a4523972d15232f1b337`;
- source application ABI: `symjit-application-storage-v3`;
- no external functions;
- one scalar point, which bypasses the SIMD body.

The exact `DirectApplicationMetadata` is:

```text
destination_operation = Add
state_plane_indices = []
parameter_bindings = Plane(0)..Plane(11)
input_plane_count = 16
scalar_input_count = 2
output_alias_inputs = [12, 13, 14, 15]
```

Run from the pyAmpliCol recurrence worktree:

```console
CARGO_TARGET_DIR=.artifacts/symjit-arm64-mre/target \
  cargo run --release \
  --manifest-path tools/developer/symjit_arm64_direct_arena_mre/Cargo.toml
```

The manifest currently points to:

```text
../../../dependencies/checkouts/symjit
```

For use in a standalone SymJIT checkout, change only that path dependency to
the checkout under test.

## Observed Output

Host:

```text
macOS 15.0 (24A335)
arm64
rustc 1.89.0 (29483883e 2025-08-04)
SymJIT 2.20.2 plus the Direct-Arena implementation
```

With the old ARM64 scalar epilogue, tested from an isolated copy with only the
`eor x(0), x(0), x(0)` instruction removed:

```text
target_arch=aarch64
status=3
destination=[1.70305086527633187e0, 1.98689862838483400e0, 8.98959235991435124e-1, 1.29594408408978490e0]
Error: "Direct-Arena scalar call returned status 3"
```

With the patch:

```text
target_arch=aarch64
status=0
destination=[1.70305086527633187e0, 1.98689862838483400e0, 8.98959235991435124e-1, 1.29594408408978490e0]
```

The identical destination values demonstrate that the old code executed the
kernel successfully and only returned an undefined status.

SymJIT 2.21.1 includes the corrected AArch64 scalar return path upstream.
pyAmpliCol's contributor patch now contains only the still-unreleased
Direct-Arena application API.

## Root Cause

SymJIT's indirect machine function has the effective return ABI:

```rust
fn(
    memory: *const f64,
    planes: *const &mut [f64],
    point_index: usize,
    parameters: *const f64,
) -> i32
```

The Direct-Arena caller interprets zero as success. The ARM64 scalar generator
restored its frame and returned without assigning `x0`, so the last temporary
pointer or integer value held in that register escaped as the function result.
Whether the bug appeared therefore depended on the generated instruction
sequence. Larger prepared kernels reliably exposed it while very small
expressions could mask it by coincidentally retaining zero.

Zeroing `x0` at the successful scalar epilogue makes the generated function
obey its declared ABI. It adds one instruction outside the arithmetic body and
has no effect on results or the hot computation.

## Suggested Regression Test

Run this MRE as an arm64-only test, or embed its 368-byte source payload and
metadata in SymJIT's Direct-Arena tests. Assert both:

1. a one-point `DirectCallableHandle::invoke()` returns `DIRECT_STATUS_OK`;
2. the four aliased destination planes match the values printed above.

The one-point requirement is important because it guarantees the ARM64 scalar
callable is exercised even when SIMD support is enabled.
