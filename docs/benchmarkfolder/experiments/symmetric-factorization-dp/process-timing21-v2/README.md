# Actual-shape DP factorization: fixed21 process timing

The isolated DP factorization did **not demonstrate a process speedup** in this
preregistered QCD3 comparison. The factor/original ratio of medians is
**1.01354625**, with paired-bootstrap 95% interval **[0.99659750, 1.03094910]**.
The point estimate is 1.35% slower; the interval includes equality. All 42
invocations and all 21 pairs are retained. This is a negative/unresolved
optimization result, not a final multi-process NinjaRS performance gate.

| Quantity | Original | Factored DP |
|---|---:|---:|
| Median seconds per point | 0.06128057031 | 0.06211069204 |
| Complete invocations | 21 | 21 |
| DP points per invocation | 98 | 98 |
| QP points per invocation | 44, 80 | 44, 80 |

Both variants use original Ninja6 and the same previously validated private QP
implementation. Only the selected DP FFV rank2/3 updates differ; the actual
generated layout has 70 coefficient slots. The factored cohort retains the
fresh IEEE flag query, admission guards and original fallback. Instrumentation
counters were removed before compilation. There is no added A-zero pruning,
new precision strategy or change to stability thresholds.

All 8,400 output rows (100 points × two orders × 42 invocations) match the
respective retained original/factor correctness controls, including their
accuracy estimates and return codes. The 16,800 pairwise Born/Laurent component
comparisons pass the prescribed mixed tolerance. The largest fraction of that
tolerance is 0.1876900707. The factor's changed rounding is therefore retained
honestly; it is not claimed to be bit-identical to the original. Each cohort is
bit-identical at the printed-output level to its own previous control. The
hypothesis that DP factorization eliminates either common QP rescue is negative.

Selection and execution were fixed before collecting samples:

* CPU5 failed the unchanged five-second preflight at 0% idle (sibling261 at
  100%). Root identified an unrelated busy process; it was not disturbed.
* A separately authorized CPU212 attempt failed at 42.4% idle. Neither rejected
  attempt launched an amplitude or produced a timing sample.
* A single retained five-second survey of all allowed CPUs identified CPU185
  as 100% idle. Root selected it before any measurements. Its separate unchanged
  preflight also passed at 100%; topology is package1/core105, NUMA node5, with
  only CPU185 exposed in its sibling list.
* The completed run is `quiet-cpu185-20261020-v1`. It alternates order for all
  21 pairs, using a fresh process and complete 100-point warm, numerical-output
  and timed passes in every arm. No sample was dropped, replaced or added.
  Other task-owned compilation/evaluation was held; the existing reference
  collector remained on CPU207. This coordinates local work on a shared host,
  not exclusive host ownership.

The activity monitor retains 165 observations over 819.814 seconds, including
selected/reference CPU counters, frequencies and host load. Raw samples and
pair ratios remain available regardless of transient host activity. The
unchanged plan uses 10,000 paired bootstrap resamples and seed410083.

The read-only verifier reparses every raw stream, checks full point/order
coverage, timed status histograms, exact control parity, card hashes/settings,
point/reference hashes and the paired statistics. Its optional post-run asset
check also rehashes both prepared executables, generated source manifests,
libraries, modules and link metadata. That check passed after all numerical
children exited. It is a post-run check, not a claim that the collector rehashed
all assets at every invocation. The raw verification output is retained in
`quiet-cpu185-20261020-v1/post-run-verification.json` (stderr is empty).

From the repository root:

```sh
source .ninjars-build/dev-env.sh
OPENBLAS_NUM_THREADS=1 python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/process-timing21-v2/verify_result.py quiet-cpu185-20261020-v1
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/process-timing21-v2/verify_seal.py
```

Add `--post-run-assets` to the first command while the original private build
trees are still present and unchanged. The enclosing evidence seal includes
the earlier source plan, v1 preparation without the pre-copy provenance check,
v2 corrected preparation, both rejected preflights, the host-only survey and
the full successful series. Older exact/safety/kernel/process evidence stays
unchanged. Production DP remains unchanged, and no promotion is recommended
from this timing result.
