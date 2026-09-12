# Real SIMD ultra arguments still fail on AArch64 in SymJIT 2.25.4

Tested on Apple Silicon against the **published, unmodified
`symjit = "=2.25.4"` crate**, source revision
`23caa835c355080560b3d7fdca5c02bcbc93dcde`.
The MRE uses public `Translator` arithmetic and the public plane-kernel API;
it does not construct, modify or force internal MIR instructions.

## Reproduce

From this directory on AArch64:

```sh
cargo run --quiet                      # first output: 81, expected 13
cargo run --quiet -- --no-compress     # passes: first output 13
```

Inputs on the first lane are `1, 2, ..., 9`. The first three outputs have
the form `x0^2*x1^2 + x2^2`; a fourth sums all nine squares. Reusing the
squares causes their ordinary materialization on the stack. With O3 and
compression enabled, SymJIT then selects its packed-stack-offset ("ultra")
calling convention. The MRE checks that it did so, rather than injecting
an ultra call itself. All SIMD lanes and the fourth output are checked.

Unmodified 2.25.4 returns success status zero but computes the first result
as **81 instead of 13**. An isolated 2.25.4 copy with **only** the attached
one-line patch passes the original scalar-expression/SIMD-output reproducer.
The uncompressed 2.25.4 control also passes.

## Cause and principled fix

The names `load_args` and `save_args` cover both sides of the call, which
can obscure the required direction of this instruction:

1. The caller packs the source stack offsets into integer registers
   (`arm/mod.rs:472–484`, `:510–511`). It does not load the corresponding
   floating-point argument registers in the ultra case.
2. The callee decodes each offset and invokes the ultra callback
   (`arm/mod.rs:533–539`). This callback must **load the value from the
   caller's stack slot into an argument register**.
3. Execution falls through to the normal entry, which stores those argument
   registers in the helper's argument slots (`topology.rs:109–114`).

The real-vector callback instead emits a store (`arm/vector.rs:312`), so
stale register values overwrite the caller's inputs. The equivalent scalar
and complex callbacks already load (`arm/scalar.rs:255`, `arm/complex.rs:236`).

`fix.patch` changes only that `str` to `ldr`; it restores the existing
calling convention rather than introducing a new one. Apply it to a clean
extracted 2.25.4 source tree, then run the same MRE:

```sh
patch -d /path/to/symjit-2.25.4 -p1 < /absolute/path/to/fix.patch
cargo run --quiet \
  --config 'patch.crates-io.symjit.path="/path/to/symjit-2.25.4"'
```

## Relevance to pyAmpliCol

This is specifically the **real-valued, AArch64 SIMD, O3, compression-enabled**
path. pyAmpliCol's native plane configuration supports scalar/SIMD kernels
and passes the requested optimization/compression settings to SymJIT
(`rust/crates/rusticol-core/src/evaluator/symjit_plane.rs:789–824`).
However, its normal generated matrix-element plane entry explicitly selects
**complex** arithmetic (same file, lines 744–759), and compression currently
defaults to off (`src/pyamplicol/evaluators/symbolica_settings.py:44`).
We therefore do **not** attribute the default matrix-element benchmarks, or
all complex evaluations, to this real-only failure. It was uncovered while
isolating compression issues in the shared compiler and is a reproducible
correctness bug in its supported real SIMD interface.
