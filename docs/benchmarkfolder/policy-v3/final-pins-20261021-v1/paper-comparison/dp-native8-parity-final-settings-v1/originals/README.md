# Held native-DP backend8 recorder-parity candidate

Prepared for review only. No compiler, build or numerical call has run. Actual launch requires the parent's post-review authorization. This private candidate adds no production or shared helper edits.

`run.py` adapts the audited legacy qcd1 parity runner. It accepts only selector8 and exact CPU12 affinity, preserves the same100 points and unchanged public Fortran driver `3d8d0632`, and runs the same forward/reverse/z90 transformations. It configures `MLStabThres=1e-9` and `COLLIERRequiredAccuracy=1e-12`, with all four thread settings1. Every compile/evaluation child is bounded to180 seconds and owned by the existing `pair.owned_preparation` registry. Raw files, guard logs, stdout, stderr, failure records and process-ledger observations are retained on failure. No qualification or timing claim occurs in this runner.

Both plain and observed executables link the same one guard object with identical linker flags; the only numerical-source difference is the source observer. The same driver and MadLoop card hashes, guard object hash, full native call/status sequence, Born/Laurent/accuracy bits, common FPE/rounding state and hardware flags must agree. All recorder rejection/nonfinite/error counters remain visible.

The exact `b0fa3802` DP module is retained privately as `ninjars_dp_stability_frozen_b0fa.py`. The runner imports that byte-frozen module for source transformation, unchanged driver rendering and raw parsing only, with its root reset to the repository root. It performs no qualification, so a later public Python reader change cannot alter the imported producer implementation. The public Fortran driver and ABI header remain separately hash-checked.

`guard.c` uses the exact types/argument order in `vendor/ninjars/ninjars.h`. It faithfully forwards context creation, native DP evaluation and context destruction, including native statuses and thread-local error behavior; it never reads or rewrites numeric buffers. Compensated-DP and MP entry increment their respective counter, log the forbidden event and exit92 before forwarding. This applies from process initialization onward. The current bridge's context creation leaves all DP/compensated/MP states lazy, and the MP worker starts only through the separately forbidden MP entry.

The guard grammar is one event per line, with no pointer values:

```text
EVENT [status=SIGNED_INT32] create=U64 dp=U64 compensated=U64 mp=U64 destroy=U64 nonnull=U64
```

`GUARD_BEGIN` and `GUARD_FINAL` have no status field. Successful forwarding produces `CREATE_RETURN`, `NATIVE_DP_RETURN` and `DESTROY_RETURN`, each with the exact native status. `dp` counts native DP calls; `nonnull` counts non-NULL destroy calls. Forbidden entries are `FORBIDDEN_COMPENSATED_DP_ENTRY` and `FORBIDDEN_MP_ENTRY`, without a return status. Log-open/write/close failures exit90/91. Guard/parser acceptance does not relabel native status1/2 as success: their complete per-call sequence and histogram remain in the evidence. Context lifecycle errors reject the parity control.

`static-validation.json` records25 ordinary Python source/signature and synthetic-parser checks. It compares all five wrapped signatures against the C header and confirms exact native argument forwarding. Synthetic native status0/1/2 inputs and seven invalid guard variants are retained under `synthetic-static-fixtures`; none is an actual scalar-integral or reduction result. This source check does not replace compilation or numerical qualification.
