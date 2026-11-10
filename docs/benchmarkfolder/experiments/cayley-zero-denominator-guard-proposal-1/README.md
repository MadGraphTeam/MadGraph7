# Private zero-denominator guard proposal

This proposal changes one diagnostic quotient in pinned TIRust
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4`,
`crates/iregi-rust/src/direct/exec/gram.rs:787`. The baseline file SHA-256 is
`4af6ca30e8cd0a077b9b1217d2b256b3430c86c2a75c91a9cd5afbfd3e42e9a7`.
No installed library, reference repository, vendor tree, production source,
or active `invariant-symmetry-1` campaign was modified. This is a separate
private source proposal with a compiled **tiny extracted quotient fixture**.

The parent checked/trap investigation identified the original and adaptive
libraries both stopping in `cayley_cancellation` at its binary64 division.
This package does not reproduce that physical amplitude or establish that
this guard resolves every trap.

```rust
let denominator = size(&det_x);
if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {
    return None;
}
let ratio = terms / denominator;
ratio.is_finite().then_some(ratio)
```

`size()` is already `cnorm2(x).to_f64().sqrt()`. The denominator, numerator
`terms`, and ratio are binary64 even for generic MPFR or DD inputs. The
guard evaluates the existing divisor once, in its original position, and
recognizes both signed zeros after the existing conversion and square
root. The source copies and minimal patch are in [source](source).

Under masked exceptions, dividing any numerator by either zero always
produced a nonfinite ratio and the old `None` result. Returning `None`
before the division preserves that decision. Every nonzero denominator
retains the original quotient and filter. In particular, finite values
divided by infinity must continue to return `Some(signed zero)`; a blanket
nonfinite-denominator rejection would change the decision. Testing the
generic norm before its existing conversion would miss conversion
underflow to binary64 zero.

The guard intentionally changes exception behavior for zero divisors.
It avoids division invalid or divide-by-zero when applicable; it does
not clear preexisting flags. All determinant, term, norm, conversion,
and square-root arithmetic remains before the guard and may still raise
exceptions. This is not a general trap-safe Gram implementation.

The source bit mask also does **not** guarantee an integer-only machine
comparison. The actual optimized Rust compiler converted it to
`UCOMISD` with zero, followed by `JNE`/`JP`, and then `DIVSD` for nonzero
or unordered operands. A signaling-NaN denominator can therefore trap
at the comparison rather than at the old division. The tests qualify
masked results, final flags, rounding/halting state, and trap outcomes;
they do not assert identical faulting instruction addresses. See the
retained [optimized functions](evidence/tiny-1/optimized-quotient.s) and
complete assembly logs.

## Actual tiny-fixture results

Rust 1.97.1 on Linux x86_64/glibc compiled only the extracted final
quotient/filter at optimization levels 0 and 3. Its `size` closure is an
identity on an explicitly supplied binary64 divisor; the full generic
function and its dependencies were not built. No reducer or amplitude
was executed, and no numerical-test lock was acquired.

Across both builds:

- 6,400 paired masked cases, or 12,800 quotient calls: 20 numerator
  patterns × 20 denominator patterns × four rounding modes × two
  initial flag states × two builds.
- All 5,760 nonzero-denominator pairs have exact result bits and exception
  flags. Coverage includes normal/subnormal values, signed infinities,
  quiet and signaling NaNs, and retained overflow/underflow behavior.
- All 640 signed-zero-denominator pairs return `None`. There are 512
  intentional flag differences because the candidate skips the invalid
  or divide-by-zero arithmetic. The initial underflow/inexact flag seed
  remains set in both implementations.
- All 448 finite-over-infinity pairs retain `Some(signed zero)`.
- All 64 isolated trap workers were reaped. Twelve old zero-divisor
  SIGFPE outcomes become candidate `None` outcomes. Nonzero-denominator
  trap decisions remain unchanged. Deliberate invalid, divide-by-zero,
  and overflow controls trap; candidate nonzero overflow still traps.
- Rounding and halting controls remain unchanged around every returned
  call. Underflow and inexact are nonhalting in trap workers.

Raw text, assembly, command receipts, native artifact **hashes only**, and
the machine-readable [results](evidence/tiny-1/results.json) are retained.
Both tiny executables remain only in the ignored private working folder.
The run took approximately five seconds; this is fixture duration, with
no performance implication.

## Offline reproduction

From the repository root:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-zero-denominator-guard-proposal-1/verify.py
python3 -I -B docs/benchmarkfolder/experiments/cayley-zero-denominator-guard-proposal-1/test_analyze.py
```

The verifier checks every retained payload, the exact one-site source
replacement and patch, extraction, execution-source bindings, and raw-log
analysis. Seven parser tests include negative controls for changed
nonzero values, cleared flags, rejected infinity quotients, lost
negative-zero coverage, missing rows, and suppressed nonzero traps.
These commands do not compile or execute native code.

The already-executed tiny-run controller is retained as
`run_tiny_fixture.py`; it deliberately refuses to overwrite its existing
`attempt1`. Any future tiny rerun should use an additive attempt. A full
TIRust build and physical checked/trap replay require separate admission.
DP/QP reducer behavior, genuine MPFR arithmetic, independent stability,
physical accuracy, runtime, and overall integration acceptance remain
unqualified by this package. This proposal does not address the user’s
known COLLIER cache bug.

Upstream source is reproduced under its accompanying
[0BSD license](source/UPSTREAM-LICENSE).
