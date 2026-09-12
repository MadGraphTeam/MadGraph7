# Compressed argument counts still overflow in SymJIT 2.25.4

Tested against the **published, unmodified `symjit = "=2.25.4"` crate**,
whose source revision is `23caa835c355080560b3d7fdca5c02bcbc93dcde`.
No Symbolica installation, pyAmpliCol, private MIR construction or other patch
is needed. The reproducer uses public `Translator` arithmetic and
`evaluate_matrix`.

## Reproduce

From this directory:

```sh
cargo run --quiet -- 63                  # passes: all outputs are 32
cargo run --quiet -- 64                  # fails: outputs remain NaN
cargo run --quiet -- 65                  # fails: outputs remain NaN
cargo run --quiet -- 64 --no-compress    # passes: all outputs are 32
cargo run --quiet --no-default-features -- 64
```

The last command disables the **Cargo** `symbolica` feature; it is distinct
from the runtime `config.set_symbolica(true)` setting. With the Cargo feature
disabled, `SLICE_CAP` is 32, so a 64-leaf expression is not compressed. With
it enabled, as in this manifest's default configuration, the capacity is 256.
That difference explains why testing only the 32-argument configuration
cannot expose this boundary failure.

The arithmetic builds three repeated, left-skewed expressions on disjoint
inputs, alternating multiplication and addition. At unit inputs the expected
answer is `floor((arity + 1) / 2)` for each expression.

## Cause and principled fix

In 2.25.4:

- `config.rs:50–54` sets `SLICE_CAP` to 256 or 32 by Cargo feature.
- `topology.rs:70–79` accepts compressed expressions up to that capacity.
- `serializer.rs:191–214` stores argument counts in the same byte as flags
  `0x80` and `0x40`; the reader masks the count with `0x3f` at lines 687–703.
- `compiler.rs:1033–1043` combines single-use intermediate expressions;
  `block.rs:243–252` limits register pressure, not the number of leaves.
  A skew tree needs only two temporary registers even with 64 leaves.

There is consequently a supported path to a count that cannot fit in the MIR
encoding. The MRE does not forge an oversized MIR instruction.

`fix.patch` limits **compressed helper arity** to `min(SLICE_CAP, 63)`.
Oversized expressions remain inline, while external-function argument
capacity and the MIR format remain unchanged. Alternatively, upstream could
deliberately widen the MIR count encoding; silently truncating it is not valid.

To test the patch on a clean extracted 2.25.4 source tree:

```sh
patch -d /path/to/symjit-2.25.4 -p1 < /absolute/path/to/fix.patch
cargo run --quiet \
  --config 'patch.crates-io.symjit.path="/path/to/symjit-2.25.4"' -- 64
```

## Relevance to pyAmpliCol

pyAmpliCol exposes compression through `jit_compress` and passes it to
Symbolica's JIT (`src/pyamplicol/evaluators/symbolica_compile.py:548`) and the
native plane compiler (`rust/crates/rusticol-core/src/evaluator/symjit_plane.rs:744`,
configuration at lines 789–824). Large repeated current expressions can
therefore reach this compressed path when the Cargo feature is enabled.
The original investigation included a generated evaluator whose compilation
failed after MIR decoding lost subsequent helper definitions.

The current default is compression **off**
(`src/pyamplicol/evaluators/symbolica_settings.py:44`), which avoids this
specific issue. Compression-enabled evaluations still require correct
encoding. This does not imply every large process, or every default benchmark,
is affected.
