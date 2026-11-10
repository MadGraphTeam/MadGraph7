# Private adaptive QP tier telemetry

This is an additive, unapplied Fortran diagnostic patch and mock qualification.
Production sources, installers, installed modules and active campaigns are
unchanged. No real TIRust/NinjaRS reduction or adaptive-library load is part of
the tests. The adaptive revision3 Rust test executable exists; an adaptive
cdylib/installation has not been qualified by this package.

`mtirust.patch` adds a capability query and an eleven-word snapshot to a private
copy of the existing module. `optional_getter.c` provides GNU/Linux weak-symbol
discovery: a released ABI1 bridge without the getter links successfully and
reports `TIRUST_UNSUPPORTED`, rather than creating a new mandatory dependency.
No loader API is used. Malformed metadata, short trailing replies and nonzero
getter status fail closed; output words are zeroed by Fortran. The underlying
ABI has no returned length, so arbitrary missing middle-word writes cannot be
proved absent. Normal Rust writes all eleven words atomically on success.

The getter is `ml_tirust_qp_precision_statistics(context, uint64_t*, size_t)`
with exact count11, as frozen in adaptive revision3 source-index
`f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213`.
Its fields are:

| Words (zero based) | Meaning |
|---|---|
| 0–4 | 256 attempts, eligible 256 refusals, 512 attempts, 512 successes, 512 failures |
| 5–8 | 256/512 cache hits, 256/512 check-pass evaluations |
| 9 | Last dispatched tensor output bits: 0, 256 or 512 |
| 10 | Exact binary64 bits of the last eligible powered condition; zero if absent |

Counters are cumulative for the context lifetime. Cache clearing and budget
changes do not reset them. Finalize/recreate starts a different generation.
There is no 256-success counter: ineligible failures and errors make attempts
minus eligible refusals invalid as a success estimate. Serialization explicitly
leaves that count unknown. Attempts include check passes; counters cannot
reconstruct which precision contributed to an accepted full amplitude.
Last-output bits may remain stale after a foreign call rejected before dispatch.
Working precision also does not establish QP numerator arithmetic when
`UseQPIntegrandForTIRust` is false. Return codes, including323, stay unchanged.

`qp_tier_records.f90` emits paired diagnostic records with exact hexadecimal
words and separately retained numerical/getter error text. Use
`write_qp_tier_snapshot(unit,point,'BEFORE')` and the corresponding `AFTER`
around one isolated diagnostic/benchmark point, outside the timed interval.
Consume numerical status, stability decisions and error text before querying:
the Rust getter clears native LAST_ERROR. The Fortran wrapper captures the
prior error first and preserves it in the module; it never creates a context,
clears caches or changes numerical return status. Query failures are separate.

`telemetry.py` parses the records and writes an additive per-point JSON object.
It computes deltas only for two successful snapshots in the same generation
with monotone counters. Missing capability, absent initial context, failed
snapshots, finalize/recreate or counter reset/wrap produce explicit attribution
gaps. A first point whose context does not yet exist is therefore unqualified
for per-point attribution; no artificial context creation or reset is inserted.
Per-worker/process isolation and correctly placed snapshots remain the caller's
responsibility. This package does not relink an active driver or emit real data.

The mock test runner uses CPU12, one command at a time, one120-second total cap
and the already reviewed owned-process supervisor. The C mocks implement only
transport/counter semantics and assert that no tensor evaluator was called.
The present/absent getter executables exercise Fortran context lifecycle,
unsigned words, native-error preservation, rejected short buffers/responses,
cache-clear persistence, and generated record parsing. Python tests cover exact
JSON integers, missing/duplicate/malformed records, delta/reset rules and the
absence of precision inference from historical return codes.

From the repository root, after sourcing `tools/tirust_env.sh`:

```
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/benchmarkfolder/experiments/adaptive-qp-tier-telemetry-private/run_tests.py
```

This command is one-shot; its private output directory must not already exist.
Real adaptive-runtime telemetry and installer capability metadata remain
unqualified until a separately reviewed integration uses this additive patch.
