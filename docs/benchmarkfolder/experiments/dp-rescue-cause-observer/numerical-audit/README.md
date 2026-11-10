# Supplemental native/Fortran cause joins

This read-only audit ran after all six authorized arms completed. It binds final
execution `c308d6bd…`, independently reparses both original/disabled/enabled public
triplets, reuses the unchanged frozen native parser, and adds source-derived
native/Fortran joins. It launches no worker and changes no frozen evidence.

| Case | Points accepted in DP / QP | Native DP / QP calls, including bootstrap | DP failures in points 1–100 | QP cache hits |
|---|---:|---:|---:|---:|
| QCD n3 | 87 / 13 | 23,432 / 3,016 | 134 UNSTABLE | 263 |
| EW n3 | 82 / 18 | 40,400 / 7,200 | 531 ERROR | 324 |

Both bootstrap calls return accepted DP status 219 without a recorded rejection.
Every native call has a unique Fortran diagnostic and matching returned status.
There are no wrapper rejections before the native call and no QP transport
rejections after native success. Native modes are full finite evaluation, both
poles, cache enabled; direction-check calls remain distinct and bypass native
cache use despite the enabled flag. QP cache-hit records occur only alone in
successful primary calls. Default options, one-lane routes, diagnostic indices,
execution-record ordering, and Fortran stage ordering all pass.

All 13 rescued QCD points include native Gram-conditioning refusals. All 18
rescued EW points include the exact native error `supplied invariant matrix must
be symmetric`. Every failure joins its matching Fortran returned status; each
rescued point also records group instability, the host rejection latch and QP
escalation. No observed rescue has only
successful DP calls together with a finite host-accuracy rejection. Three QCD
and two EW rescued points also exceed the host accuracy threshold before the
rejection override; these observations overlap and are not mutually exclusive
causes. No contraction rejection after a stable diagnostic was observed.

The retained upstream records add detail beyond immediate native error text.
For points 1–100, QCD records 454 expansion/method-choice rows with 320 successes;
EW records 25 with 25 successes. The native `last_diagnostics` strings, including
accepted unchecked choices, are retained verbatim with occurrence counts.
Engine counters and route counts are kept separately, with bootstrap separated.
Master-row counters aggregate internal precisions; a Direct route does not prove
exclusively binary64 arithmetic. No unique causal assignment, physics accuracy,
counterfactual improvement, or performance acceptance follows from this audit.

`audit_native_joins.py --case all` reproduces the read-only audit while the exact
completed workspace artifacts remain available. Run from the repository root:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/dp-rescue-cause-observer/numerical-audit/audit_native_joins.py --case all
```

The helper first requires completed/reaped arm metadata and exact raw-output
hashes. It then validates the frozen grammar and supplemental joins, checks
evidence again after reading, and prints JSON only. A single-case invocation
does not qualify the six-arm experiment. `results.json` is the retained successful
all-case stdout; the enclosing execution still leaves
`cause_attribution_qualified=false` for explicit scientific review. The separate
portable numerical package retains the original journals; its byte/inventory
verification is not represented as another semantic replay.
