# Real-input markers address the wrong variables

Affected revision: SymJIT 2.25.0,
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/commit/f1c193d301897149de6609f706297b0c97a4f018).

## Reproducer and result

`mre.rs` compiles two outputs: `x1*x1`, declaring `x1` real, and `x2*x2`,
with `x2` complex. At `x1=3`, `x2=5+7i`, the expected outputs are
`9` and `-24+70i`. Indirect translation instead returns `9` and `25`;
direct translation is correct before this fix. No compression is required.

Copy `mre.rs` into a SymJIT checkout as `tests/real_input_mre.rs`, then run:

```sh
cargo test --no-default-features --test real_input_mre
```

## Cause and suggested fix

The indirect translator records actual scalar locations, `Param(2*i)` or
arena `Mem(2*i)`, for complex input `i`. `Complexifier::new` doubles parameter
indices again and discards memory markers. It therefore treats another input
as real in ordinary translation, while arena translation loses real-arithmetic
specialization. Direct translation still records logical indices.

`fix.patch` consistently uses physical scalar-address locations: update the
direct producer, preserve locations in the consumer, and honor marked arena
loads. Writable arena locations are excluded before lowering, including loads
revisited by backward branches; stores also clear the local declaration.
The unreachable indirect function-result marker insertion is removed.

The patch also corrects an independently erroneous real-`abs` lowering:
its registers must be `re(dst)` and `re(src)`, not their original complex-register
numbers. The precise MIR regression test fails with the old line; simple calls
using the return register happen not to expose it.

Current indirect saved programs already contain physical markers and are fixed
when reloaded. Older direct saved programs with logical markers require
regeneration; the patch does not guess which convention a file contains.

## Focused validation

```sh
cargo test --no-default-features --jobs 2 --test real_input_locations --lib real_
```

Passes locally on Apple ARM64: three MIR checks plus two integration tests,
covering direct/indirect translation, parameters/arenas, O0/O2/O3,
compression on/off, native scalar/SIMD, both complex configurations and
save/load. The arithmetic test checks 36 configurations both fresh and reloaded.
The exact wrong-neighbor reproducer fails before the fix and passes afterwards.
Only this issue and its tests are included in the patch.
