# Schema-2 native trace qualification

`pilot-qcd-n1-tirust-fpe` qualifies the additional host FPE observations on the
original frozen n1 process and historical TIRust library. All 100 original
points classify as DP. The complete warmup and classification traces report
both failure flags false. Both calibration traversals and all seven scheduled
batches have zero observed NinjaRS retry counters, zero FPE flags, unchanged
numerical coefficients, and no cohort migration. The seven batches contain
79,800 public amplitude calls; the two calibrations contain another 11,500.

The recorded timing is **ineligible**, even as a cohort diagnostic median.
Calibration observed 100 calls in 0.010619244 s and then 11,400 calls in
1.227601806 s. The registered 11,400-call batches subsequently took only
0.799900021–0.809035609 s, below the required 0.9 s minimum. Changing shared
load is recorded in the host snapshots. The runner preserved every batch,
set `eligible_diagnostic=false`, and emitted a null median. There is no rerun
or replacement of these data.

Native trace qualification and timing eligibility are separate conclusions.
The earlier schema-1 diagnostic remains a different run and did not observe
FPE flags. Neither run qualifies the full-amplitude acceptance gate. The first
milestone inventory and its files remain unchanged.

Reparse from the MadGraph root:

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark_cohorts.py validate \
  --label pilot-qcd-n1-tirust-fpe
```
