# Real-input declarations use inconsistent coordinates

SymJIT **2.25.4**, published source commit
`23caa835c355080560b3d7fdca5c02bcbc93dcde` (from the crate's `.cargo_vcs_info.json`).
This is independent of compression and of the ARM-specific reports.

## Numerical reproducer

From this directory:

```sh
cargo run --bin mre
```

The program uses only the public `Translator` API. It evaluates two independent
squares, declaring only `x1` real. With `x1 = 3` and `x2 = 5 + 7i`:

| Translation | Expected | Unmodified 2.25.4 |
|---|---|---|
| Direct, control | `9`, `-24 + 70i` | `9`, `-24 + 70i` |
| Indirect | `9`, `-24 + 70i` | `9`, `25` |

The final argument of `append_mul` counts the real arguments. Thus `2` for
`x1*x1` and `0` for `x2*x2` accurately describe the inputs. No input is mutated,
and no saved program or private API is involved.

## Cause and proposed fix

Indirect translation already records physical scalar addresses: complex input
`i` is `Param(2*i)`. `Complexifier::new` multiplies the address by two again.
Consequently the declaration for `x1` addresses `x2` and discards its imaginary
part. Direct translation currently records logical indices, which is why that
control passes.

[`fix.patch`](fix.patch) establishes one invariant: real-input declarations and
the corresponding MIR loads use **the same scalar addresses**. The consumer
does not rescale them; the direct producer records the physical addresses too.
This fixes both producers together rather than special-casing either mode.
The public MRE passes both direct and indirect cases with this correction.

The earlier report also proposed specializing real arena loads. That is not
required to correct this bug: 2.25.4 conservatively loads arena values as complex.
We therefore omit that optimization and its mutable-slot invalidation machinery
from this follow-up. The patch does not change arena-load behavior.

## Independent real-`abs` register error

`Complexifier::abs` has another unchanged one-line error: its real branch emits
`abs(dst, src)` in logical complex-register coordinates, whereas the scalar MIR
requires `abs(re(dst), re(src))`. This is hidden when the chosen registers happen
to have identical mappings, such as the return register.

[`abs_mre.rs`](abs_mre.rs) is a short lowering test inside SymJIT (the lowering
types are private). A real constant is loaded into
`Gen(2)`, then `abs` targets `Gen(3)`. The scalar operation must read `Gen(8)` and
write `Gen(10)`, not read `Gen(2)` and write `Gen(3)`. This test does not depend on
the real-input declaration fix or on native machine code.

Apply that test in a clean 2.25.4 source checkout and run:

```sh
cp /path/to/abs_mre.rs src/symjit/followup_abs_mre.rs
patch -p1 < /path/to/abs-register-test.patch
cargo test --lib abs_register_regression
patch -p1 < /path/to/abs-register-fix.patch
cargo test --lib abs_register_regression
```

The one-line [`abs-register-fix.patch`](abs-register-fix.patch) is separate to
make the two diagnoses and their controls independently reviewable.
The lowering test fails on unmodified 2.25.4 and passes with this correction.

## How this arises in pyAmpliCol

pyAmpliCol lowers complex matrix-element expressions containing both complex
currents/polarizations and known-real momenta/model parameters through Symbolica
and SymJIT. Its [`symbols.py`](../../src/pyamplicol/_internal/physics/symbols.py)
constructs real symbols with Symbolica's `is_real=True` property. Its
[`symbolica_adapters.py`](../../src/pyamplicol/evaluators/symbolica_adapters.py),
`_export_symjit_application`, calls `export_symjit(complex=True)` and records
`translation_mode="indirect"`. This ordinary parameter-based evaluator combines
the real declarations with complex arithmetic and exposes precisely the
coordinate mismatch above.
Arena-based kernels use a separate input path and are not claimed to reproduce
this wrong-neighbour failure.

The `abs` issue is relevant when real absolute values from model expressions
reach complex lowering. We have isolated the invalid lowering, but have not
attributed a particular pyAmpliCol process-level discrepancy to that separate
line. It is included because it remains a demonstrable compiler error, not as
an explanation for every observed numerical discrepancy.
