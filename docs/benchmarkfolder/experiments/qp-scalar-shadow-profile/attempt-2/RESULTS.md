# QP scalar shadow observation: qualified recording, no cache reuse

The corrected private build succeeded in 709.09s within the 900s cap. One
authorized original/counters-off/counters-on replay triplet then completed:
36 requests per process,108 records total. All saved 113-bit input bytes,
returned tensor bytes, statuses, errors, cache counters and host IEEE flags
match exactly. The three complete tagged logs even have the same SHA256.
Private off/on MPFR flags match; the callback is available, and key/output
observation changed no MPFR flags. Historical 49af and private a4ed runtime
identities were checked through actual loader resolution and before/after
hashes. No production sources or installed libraries were changed.

The journal is complete: 202 scalar requests,202 completed,zero dropped
events,zero scalar errors and zero observed nonfinite scalar outputs.
Raw cap/availability safeguards and their tests remain in the frozen source
package. No scalar result was reused by this experiment.

## Physical subset

Only 24parents are physical: saved EW3 points 1 and 2, groups 1–12, CTMODE 4.
The 12-call lifecycle suffix is synthetic and excluded from this table.

| Saved point | Served parents | Scalar calls | New exact keys | Across-only repeats | Within-parent repeats | Prior eligible-parent key matches | All kernel work (ms) | Matched-request work (ms) | Share of observed scalar work |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary |12/12|104|42|52|10|62|13.725214|2.494449|18.17%|
| Threshold |10/12|74|34|34|6|40|10.107846|0.911915|9.02%|
| Total |22/24|178|76|86|16|102|23.833060|3.406364|14.29%|

Across-only repeats count the first occurrence in a parent after an earlier
parent already requested that key. Within-parent repetitions are counted
separately; all 16 also matched an earlier eligible parent's key in this
corpus. The 102 matches are 57.3% of calls but only 14.29% of the observed
scalar-kernel work. They are candidate input matches, not usable cache hits
or demonstrated saved time. Physical key construction took 0.166774ms.

| Scalar family | Physical calls | Observed kernel work (ms) | Work on prior eligible-parent key matches (ms) |
|---|---:|---:|---:|
| Tadpole |46|0.353691|0.309031|
| Bubble |82|1.708088|0.802544|
| Triangle |42|18.974398|2.294789|
| Box |8|2.796883|0|

Threshold groups 5 and 6 refused in geometry before requesting scalar kernels.
Both remain visible. Their synthetic repetitions also refused before any
scalar call. Consequently the native corpus does not exercise completed
scalar candidates discarded after a later parent failure; that behavior is
covered only by the offline accounting fixtures here.

## Synthetic controls and limits

The separately labeled 12 synthetic parents contribute 24 scalar calls. Two
whole-tensor cache hits make no scalar calls. Check/cache-off controls
exclude 12 candidate requests. Zero-budget controls record six insertion
refusals for three distinct keys attempted twice. The aggregate 82 modeled
committed keys accumulate across cleared epochs (76 physical + 6 synthetic);
they are not simultaneous resident occupancy. Estimated payload accounting
is not an allocated scalar store.

These are single instrumented kernel observations under substantial host
load. Parent wall times were not recorded. Process launch elapsed times
are not parent/amplitude attribution denominators, so no amplitude speedup
can be inferred. Scalar output values were not journaled: equal exact keys
do not independently establish equal scalar results. A future cache would
also need lookup/copy cost, MPFR sticky-flag behavior, budget/lifecycle and
independent-check isolation qualification. Observer neutrality does not
qualify a future cache-hit path. This small EW3 subset cannot explain all
QCD4/EW4 warmup costs. No cache implementation is authorized by these findings.

The root's production QP unpack-failure fix is intentionally absent from
the common historical Fortran archive used by all three variants. This is
a recording-neutrality comparison, not qualification of the later installer.

## Portable verification

Run the verifier with Python 3.10 or later from any location after preserving
this experiment directory and its attempt-2 child:

```sh
python attempt-2/verify.py
```

It checks the delivery inventory, both frozen source inventories, all 108
tagged records and expected input bits, MPFR availability/parity, and
recomputes the exact summary and physical-only analysis. Unlike invoking
`analyze.py` on raw JSON alone, it binds public statuses from `on.log` before
analysis. It never executes a reducer. To additionally hash available
ignored native artifacts without executing them, pass
`--native-root /absolute/path/to/madgraph7_madloop_tirust`.

`delivery-index.json` binds the unchanged failed attempt 1 and all corrected
source, callback review, build logs, raw replay, cap/error tests and analyses.
Native binaries/archives are hash-only entries; none are copied into docs.
