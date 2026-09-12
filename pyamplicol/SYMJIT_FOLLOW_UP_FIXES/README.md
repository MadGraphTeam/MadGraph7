# SymJIT 2.25.4: remaining correctness fixes

Tested on Apple Silicon (AArch64), 12 September 2026, against the **published
2.25.4 crate**, not our previously patched 2.25.0 checkout. The published
`.cargo_vcs_info.json` identifies source commit
`23caa835c355080560b3d7fdca5c02bcbc93dcde`. The authoritative source is the
[2.25.4 crate archive](https://static.crates.io/crates/symjit/symjit-2.25.4.crate).

This follow-up contains **five reports covering seven remaining defects**.
Each defect reproduces in 2.25.4; the proposed corrections pass their focused
checks. The changes are ordinary compiler corrections, not pyAmpliCol-specific
exceptions. Our production dependency pins and installed libraries have not
been changed by this verification.

| Report / defect | Unmodified 2.25.4 | Proposed correction |
|---|---|---|
| [Compressed argument count](argument-count-overflow/README.md) | Helper arity 63 passes; 64/65 leave NaN outputs | Bound compressed arity by the six-bit encoding, not the larger external-argument capacity |
| [ARM real SIMD ultra arguments](arm-simd-real-ultra-arguments/README.md) | Computes `81` instead of `13` | Load the caller's stack value instead of overwriting it |
| [ARM normal-return path](arm-compressed-register-restore/README.md) | Correct numerical output, but caller's `d8` changes | Land before register restoration when skipping helper definitions |
| [ARM fast-complex register saves](arm-compressed-register-restore/README.md) | Even identity evaluation replaces `d8` with `d9` | Preserve physical ABI registers in distinct 64-bit slots |
| [Real-input coordinates](real-input-locations/README.md) | Declaring `x1` real changes `x2²` from `−24+70i` to `25` | Use the same physical scalar addresses in declarations and loads |
| [Real absolute-value register coordinates](real-input-locations/README.md#independent-real-abs-register-error) | Emits the scalar operation on the wrong registers | Map both operands to their scalar real-component registers |
| [Compressed complex returns](compressed-complex-return/README.md) | Returns `3+0i` instead of `3+4i`, or `7+4i` instead of `7` | Give helpers a consistent complex return contract |

## Reproducing and reviewing

### Run every reported regression from pyAmpliCol

The branch `updated_dependencies_and_misc_optimizations` ships the opt-in suite
[`tests/integration/test_symjit_upstream_regressions.py`](../tests/integration/test_symjit_upstream_regressions.py).
From the repository root, with Python 3.11+, `pytest` and Rust/Cargo installed:

```sh
# Test unmodified published SymJIT 2.25.4: remaining bugs must FAIL normally.
PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 \
  python3 -m pytest tests/integration/test_symjit_upstream_regressions.py -q

# Test a SymJIT checkout containing proposed/upstream fixes instead.
PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 \
PYAMPLICOL_SYMJIT_SOURCE=/absolute/path/to/symjit \
  python3 -m pytest tests/integration/test_symjit_upstream_regressions.py -q
```

For our current locally patched dependency, use
`PYAMPLICOL_SYMJIT_SOURCE="$PWD/TMP_FIXED_SYMJIT"`. `just symjit-regressions`
is a shortcut for the first command and also honours that source variable;
set `PYTHON=.venv/bin/python` if needed.

The 15 tests cover the seven remaining defects, the four original checks
already repaired in 2.25.4, and extra numerical controls. They compile the small MREs, not pyAmpliCol or
Symbolica, and run each in a separate subprocess so one failure does not hide
the others. ARM-specific checks are explicitly skipped on non-AArch64 hosts.
These are compiler dependency regressions shipped with pyAmpliCol, not full
matrix-element generation or performance tests.

Only test modules are added to a temporary copy for the private lowering checks.
**No corrective patch is applied automatically**, and the selected checkout,
root Cargo pins, and installed Python/native libraries are left unchanged.
The SymJIT `symbolica` Cargo feature is enabled deliberately so the arity test
is not masked by the smaller argument capacity. Cargo caches are reused;
initial dependency compilation takes longer than the tiny evaluations.
Without the opt-in variable, ordinary pytest/CI runs skip this dedicated suite.

Verified on Apple Silicon: our locally patched `TMP_FIXED_SYMJIT` passes all
15 tests (about 15 seconds with the Cargo cache available). Unmodified 2.25.4
gives 7 passes and 8 failures: seven distinct remaining defects, with the arity
defect exercised at both 64 and 65. These are real failures, not accepted xfails.

### Run an individual reproducer

The argument-count, real-SIMD and real-input folders contain standalone Cargo
projects, with `symjit = "=2.25.4"`. Run their `cargo run` examples as documented.
The ARM-register folder uses `cargo test --no-fail-fast -- --nocapture`.
No Python, Symbolica, pyAmpliCol installation or process generation is needed.
Numerical controls disable compression or use the alternative translation path.

The return and `abs` examples are deliberately small **lowering-unit tests**:
they exercise private compiler stages directly and are identified as such.
The return example uses readable MIR instructions, not an opaque saved payload.
Its README explains how those instructions correspond to compressed calls.

For patch testing, extract the published crate into a separate directory and
apply the relevant `fix.patch` there, for example:

```sh
patch -d /absolute/path/to/symjit-2.25.4 -p1 < /absolute/path/to/fix.patch
```

Point the standalone MRE at that copy without changing its manifest:

```sh
cargo run --config 'patch.crates-io.symjit.path="/absolute/path/to/symjit-2.25.4"'
```

Use `cargo test` instead for the ARM-register tests; the two private tests have
their short installation instructions alongside them. Individual fixes were
checked in isolated copies. The ARM correction also passed the broader existing
matrix of **72 generated applications / 144 scalar and SIMD calls**, spanning
O0/O2/O3, real/complex/fast-complex modes, and compression off/on.
Finally, all seven corrections were applied together to a clean 2.25.4 copy:
the public argument-count, real-SIMD and real-input checks, both reduced ARM
checks, and both private lowering tests all passed.

## How these arise in pyAmpliCol

pyAmpliCol passes mixed real/complex model and matrix-element expressions to
Symbolica/SymJIT, and runs generated native kernels both individually and in
SIMD batches. It can request compression of repeated expressions. These are
normal supported compiler inputs, not custom internal instructions.

- Mixed real momenta/parameters and complex currents expose the real-input
  coordinate error in the indirect evaluator.
- Large repeated expressions can exceed 63 leaves while having low register
  pressure, exposing the compressed-argument encoding mismatch.
- Compressed native kernels must preserve their caller's registers irrespective
  of whether their numerical outputs are correct.
- A compressed top-pair evaluator previously exposed the complex-return error;
  its observed discrepancy and the smaller stage-level reproducer are described
  separately in the return report.
- The real-only SIMD and fast-complex cases were isolated while investigating
  the same compiler. They are **not** claimed to affect pyAmpliCol's usual
  complex, non-fast-complex plane kernels. The `abs` lowering error is likewise
  not assigned to a particular measured process failure.

Each report points to the relevant pyAmpliCol settings/source. In particular,
`jit_compress` currently defaults to **false**: compression-specific failures
do not imply that every default evaluation or published timing is affected.

## What is no longer being requested

The output-reuse, unused-trailing-input and compressed-success-status tests now
pass. The previous ARM register-use accounting change is also present upstream;
the remaining ARM failures have different causes, isolated above.

For the two disputed reports: the arity failure requires the Cargo `symbolica`
feature (capacity 256 rather than 32); the real-SIMD failure passes when **only**
the one-line load/store correction is applied. Both are reproduced through the
public API. Downstream API migrations and optional arena specialization from
the older report are excluded.
