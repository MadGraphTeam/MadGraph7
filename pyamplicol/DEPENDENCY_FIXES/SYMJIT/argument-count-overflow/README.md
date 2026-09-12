# Compressed-expression argument-count overflow

SymJIT revision: `f1c193d301897149de6609f706297b0c97a4f018` (2.25.0).

## Bug

With Cargo feature `symbolica`, `SLICE_CAP` is 256. Repeated-expression
compression accepts that many arguments, but MIR `LoadArgs` / `SaveArgs`
encode their count in only six bits: `0x80` and `0x40` in the same byte encode
the complex/ultra flags. At 64 arguments, the reader decodes zero arguments
with an extra flag; argument-location bytes corrupt the following instructions.

The Python reproducer aborts in the allocator. The standalone Rust test also
demonstrated silent wrong results for a real scalar expression at arity 64
before the fix. Our actual generated evaluator
instead lost the later subroutine definitions and aborted with
`label XX*XX*XX*XX*++- not found`. The named eight-argument expression was not
itself invalid. `MirIterator::next` treats decoding errors as end-of-stream,
which obscures the original corruption.

## Reproduce

With the affected Symbolica Python development build using SymJIT:

```sh
python mre.py --arity 63                 # passes
python mre.py --arity 64                 # aborts before the fix
python mre.py --arity 65                 # aborts before the fix
python mre.py --arity 64 --no-compress   # passes
```

The Rust integration test is independent of Symbolica and is included in
`fix.patch`. Apply the patch to a checkout of the revision above, then run:

```sh
git -C /path/to/symjit apply /path/to/argument-count-overflow/fix.patch
cargo test --manifest-path /path/to/symjit/Cargo.toml --test compression_arity
cargo test --manifest-path /path/to/symjit/Cargo.toml --no-default-features \
  --test compression_arity
```

## Suggested fix

`fix.patch` caps repeated-expression compression at `min(SLICE_CAP, 63)`;
larger expressions stay inline and smaller repeated subexpressions can still
be compressed. This preserves the existing MIR format and leaves the
default-features-disabled capacity of 32 unchanged.

All argument-load/save producers were checked: the MIR instructions in question
are generated only by topology compression. External-function
`Statement::LoadArgs` emits individual argument stores instead, so its larger
slice capacity need not be reduced. Allocation, compaction and complexification
only forward or translate the existing MIR instructions.

Tests cover selection at 32/63/64/65 arguments and numerical real/complex evaluation
through scalar and available SIMD kernels at O2/O3, including both complex
lowering paths. The patch contains only this fix and its tests; unrelated local
SymJIT changes are excluded.

Local validation: both focused tests pass after the fix on Apple ARM64, with
both default features and `--no-default-features`;
the numerical test checks 24 generated applications, each through its scalar
and available SIMD kernels. Before the fix it fails at the first arity-64 case.
The rebuilt and installed Symbolica Python extension also passes the arity-64
reproducer and recompiles the captured 454-input, 60-output evaluator that
originally failed with the missing-label error.
