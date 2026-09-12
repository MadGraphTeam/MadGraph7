# Two remaining AArch64 register-preservation failures

Baseline: published **SymJIT 2.25.4**, source commit
`23caa835c355080560b3d7fdca5c02bcbc93dcde` from its `.cargo_vcs_info.json`,
on Apple AArch64. These are remaining
control-flow and spill-width issues, not the already-fixed register-use
accounting issue.

## Minimal reproducer

From this directory, using the included manifest pinned to the published crate:

```sh
cargo test --test normal_return -- --nocapture
cargo test --test fast_complex_save_slots -- --nocapture
```

Both tests use only the public API and freshly generated programs.

## A. Normal return skips restoration

`mre.rs` uses the public `Translator` API to compile three copies of a real,
nine-input expression at O3, first without and then with compression. Both must
return exactly `(5, 5, 5)`. A small assembly wrapper independently checks that
the low 64 bits of `d8` retain a distinctive sentinel, as required by AAPCS64.
The wrapper declares its clobbers so its own caller remains protected. No
handwritten MIR, serialized program, private register mapping, SIMD, or complex
arithmetic is needed. The generated compressed MIR is printed for inspection.

On 2.25.4 the reduced test passes the numerical checks but
observes `d8 = 0x3ff0000000000000` (the input `1.0`) instead of the sentinel
`0x3ff0000000000001` when compression is enabled. Compression disabled passes.
The generated MIR confirms a nine-argument helper. The control-flow-only patch
makes this reduced test pass; the broader regression then retains only the
independent fast-complex spill failures described below.

### Cause and suggested fix

`Topology::compile` places the helper bodies after the main computation and
ends the main computation with a branch to `@success`. That label is inside
the generated epilogue. However, `Builder` emits `restore_registers` **before**
calling the epilogue generator. The normal compressed return therefore jumps
over all restoration instructions. In 2.25.4 the used-register mask already
includes every register when compression is enabled; expanding it again
cannot fix this missing execution of the restore block.

The first part of `fix.patch` routes normal completion through a dedicated `@normal_return`
label immediately before restoration, for both ordinary and fast compilation.
The uncompressed fallthrough and compressed branch then share the same restore
block and existing success epilogue. This preserves the existing distinction
between normal success and exceptional SIMD replay exits. It adds no runtime
instructions to uncompressed kernels and no extra save/restore operations;
only the compressed normal branch's destination changes.

This patch addresses the reproduced normal-return path. It is not a claim
that every exceptional exit or every ABI on every architecture has been audited.

## B. Fast-complex ABI spills overwrite neighbouring registers

`fast_complex_mre.rs` evaluates three complex identity expressions with fast
complex lowering. No compressed helper is needed. Enabling compression makes
2.25.4 conservatively save all floating-point registers; the wrapper gives `d8`
and `d9` different sentinels. Numerical identities remain correct, but the
reduced test observes `d8` restored as `0x4000000000000002`, the original
value of `d9`, instead of `0x3ff0000000000001`. Compression disabled passes.
The emitted MIR contains only six identity-copy instructions and no helper
call, isolating this failure from part A.

`ArmComplexGenerator::save_used_registers` passes physical register numbers
to `save_stack`. That helper stores a **128-bit complex value** at slot `idx/2`:
registers 8 and 9 therefore spill to the same address, as do 10/11, 12/13 and
14/15. The matching reload duplicates the neighbouring register's value.

The second part of `fix.patch` uses direct 64-bit stores and loads for these
ABI spills. Only the low 64 bits are callee-preserved by AAPCS64; distinct
8-byte slots at offsets 64, 72, ..., 120 fit within the existing 128-byte
reserved ABI area. Complex evaluation's ordinary value-storage helpers remain
unchanged. There is no extra allocation or additional runtime store/load.

## Verification

Both reduced tests reproduce independently against the published crate and
pass with the combined patch. The control-flow-only patch fixes part A while
leaving the part B failures. With both changes, the broader public-API check
also passes all 72 generated applications (144 scalar/SIMD calls): compression
off/on, O0/O2/O3, real and both complex lowering modes, arities 1/5/9/16.
These are register-preservation and numerical-correctness checks, not timings.

Apply the patch in a clean 2.25.4 source checkout and test that source through
the same standalone manifest:

```sh
patch -d /path/to/symjit -p1 < /path/to/this/folder/fix.patch
cargo test --manifest-path /path/to/this/folder/Cargo.toml \
  --config 'patch.crates-io.symjit.path="/path/to/symjit"' -- --nocapture
```

## Relevance to pyAmpliCol

pyAmpliCol and Symbolica call these generated kernels from native code. A kernel
can return the correct matrix element yet corrupt floating-point values held
by its caller across the call, including surrounding runtime or profiling
calculations. The underlying C-ABI obligation is the same regardless of whether
the outer caller uses Python or a native SDK; this is not evidence of a separate
observed failure in every SDK.

The refreshed campaign normally disables compression, but the compiled
top-pair `n=5` selected result uses compression and therefore needs the corrected
path. No performance rerun is needed to establish this register-preservation
failure. The focused regression should check both the numerical outputs and
the saved registers, independently.
