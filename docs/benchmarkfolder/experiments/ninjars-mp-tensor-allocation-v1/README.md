# Isolated MP tensor allocation experiment

This is a correctness-first prototype, not a production change or performance result.
It preserves the pinned generic recurrence and scalar expression order. A borrowed
complex product reproduces the exact pinned `Complex::mul` implementation while
removing the outer cloned complex value; retained scratch resets use `set_from`
instead of replacing the allocated values. No arithmetic operation, structural
zero, contraction, precision tracking or rescue threshold is omitted.

`prepare.py` snapshots the current bridge and the exact pinned complex/scalar
implementations, creates an isolated bridge inside `.benchmark-work`, and records
source hashes. `tensor.patch` is the complete numerical-source delta; only the
isolated tensor and its test modules differ from that original bridge.

The checks require identical stored rational values, tracked precision, signs
(including signed zero), and finite/nonfinite state. They cover evaluation and
all four expansion forms, rank 0–6, mu powers, changed coefficients and request
rows, mixed precision operands, empty/malformed batches, and exceptional values.
The actual-request replay compares every callback output against the current
original implementation and all eight returned ASCII fields against retained
successful binary128 requests. The physical golden fixture replay covers all 21
lanes with reused tensor/amplitude state. Existing bridge tests retain binary128
transport, context changes and 256/384-bit convergence controls.

Builds and all disposable outputs remain under the repository. The initial build
uses one worker on CPU 212 and an isolated copy of cached dependencies; it is
expected to take several minutes with thin LTO. Symbolica execution is serialized
under the explicitly assigned MP slot. Failed attempts and their logs are kept.
No timing experiment or performance claim is authorized by this preparation.

The initial build failed because a new synthetic request fixture omitted its
required `partition` field. That build log and source are retained. The first
numerical suite passed 19 tests but failed an overly broad direct test of
`set_from` with an arbitrary NaN source: Rug's `clone_from` can preserve the old
destination NaN sign, whereas `clone` initializes a fresh NaN. The prototype
never copies a NaN with `set_from`; its only sources are the stored finite zero
and one. The failed test and its original source remain under `unit-1`.
`scalar-diagnostic.rs` separates strict exceptional-value product comparisons
and actual zero/one resets from this unrelated arbitrary-NaN assignment. The
corrected direct test covers zero/one resets from all prior finite/nonfinite
states. This clarification does not change the numerical patch.

The corrected optimized suite passed all 20 tests. All 21 golden lanes matched
the original tensor callbacks exactly. The complete capture replay passed all
1,392 requests across nine topology/rank shapes: 86,436 complex callback outputs
matched stored values, tracked precisions and signs, and all 11,136 final ASCII
fields matched their retained originals. Both 256-bit and 384-bit convergence
results are also exactly unchanged. Actual-capture replay creates a fresh
tensor/amplitude per request; retained-state changes are checked separately by
the synthetic tests and the three golden lanes in each case.

`run-correctness.sh` records the exact sequential test commands. Build attempt 3
is the final test build; attempts 1/2, the failed arbitrary-NaN test, and a
premature scalar launch before its linker finished remain retained. No numerical
source or installed archive was changed, and there is no measured speedup.
The scalar diagnostic checks 245 exceptional products and 70 actual resets,
and preserves the 15 generalized NaN-copy counterexamples.

For an offline evidence check, run from the repository root:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python3 docs/benchmarkfolder/experiments/ninjars-mp-tensor-allocation-v1/verify.py
```

This revalidates the sealed files and recomputes `summary.json` from raw logs,
without requiring an executable or disposable build tree. `prepare.py` and
`build.sh` retain the original preparation/build recipe for a fresh unused
working namespace; they refuse to overwrite this completed experiment.
