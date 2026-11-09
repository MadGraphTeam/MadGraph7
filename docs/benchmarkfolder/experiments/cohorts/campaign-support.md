# Cohorts for an explicitly registered campaign

`tools/tirust_benchmark_cohorts.py --campaign ID` uses the same registered
campaign as `tools/tirust_benchmark.py --campaign ID`. It does not create a
campaign or select one implicitly. Generate and fully build each candidate
process through the main campaign runner before running these diagnostics.
The original generated process and the acceptance executable are unchanged:
only the diagnostic driver is compiled in a private copy.

From the MadGraph repository root, after selecting the registered campaign ID
and an available CPU, the initial qualification sequence is:

```sh
source tools/tirust_env.sh
cohort_campaign=REGISTERED_CAMPAIGN_ID
cohort_cpu=AVAILABLE_CPU

python3 tools/tirust_benchmark_cohorts.py run \
  --campaign "$cohort_campaign" --source candidate \
  --case qcd_n1_uux_z --backend 9 --label qcd-n1-tirust-01 \
  --cpu "$cohort_cpu" --timeout 1800
python3 tools/tirust_benchmark_cohorts.py validate \
  --campaign "$cohort_campaign" --label qcd-n1-tirust-01

python3 tools/tirust_benchmark_cohorts.py run \
  --campaign "$cohort_campaign" --source candidate \
  --case qcd_n3_uux_zgg --backend 9 --label qcd-n3-tirust-01 \
  --cpu "$cohort_cpu" --timeout 1800
python3 tools/tirust_benchmark_cohorts.py validate \
  --campaign "$cohort_campaign" --label qcd-n3-tirust-01
```

Replace the two uppercase values before executing these commands. Confirm the
case identifiers in the campaign's frozen `cases.json`. Schedule compilation
and measurements outside the formal acceptance timing windows. Backends 8 and
9 additionally acquire the shared numerical lock. The timeout bounds each
worker protocol wait, not the total campaign duration.

Repeat with explicit backends 1, 6, 7, and 8 and distinct labels when their
processes are ready. A campaign-bound baseline diagnostic requires a baseline
export prepared in that campaign by the main runner; an old unbound export is
rejected. Every run still warms and classifies all 100 original physical points
under the frozen automatic policy before timing its subsets. No forced-DP or
forced-QP mode is introduced. Failed or empty cohorts remain recorded; a
different candidate may have different membership than the historical n3
sample.

The three useful labels are `dp`, `compensated_dp`, and `qp`. DP means binary64
reduction arithmetic with ordinary stability checks and internal method
selection included. The NinjaRS retry counters distinguish compensated DP
from an accepted DP result without that retry. QP timing includes the preceding
DP work and automatic rescue. Each timed public call retains its accepted
status, retry counters, validity check, and both host reduction-failure flags.
Migrations and failures invalidate the pure-cohort timing without changing the
frozen point indices. False failure flags alone do not establish accuracy or
describe every internal bootstrap call. Comparisons between reducers require
identical point membership; comparing unrelated DP and QP subsets does not
measure incremental rescue cost.

Campaign reports are append-only under
`docs/benchmarkfolder/campaigns/ID/experiments/cohorts/LABEL/`; ignored private
builds live under
`.benchmark-work/tirust/campaigns/ID/cohort-diagnostics/LABEL/`. Each report
retains the campaign manifest, frozen configuration, exact point files,
protocol sources, driver-only build proof, runtime identities, classification,
per-call traces, calibration, timing batches, and any failure. Reusing a label
fails rather than replacing evidence.

The runner requires matching campaign bindings in the original generation and
compilation records, and in its own specification, inputs, compilation, and
results. Before native execution and after worker closure it checks the live
registered source/native identities. Reparse instead verifies the retained
source/native identities against the saved manifest, so a later installation
does not invalidate correctly retained historical results. Manifest, policy,
points, archive, DSO, metadata, and cross-campaign substitutions are rejected.

All cohort timings remain **diagnostic and ineligible for performance
acceptance**, including timings from a final candidate. Per-call tracing and
finite checks add overhead, and shared-host activity is recorded. Formal paired
acceptance timings remain in the main harness. The earlier qualified n1
performance failure and both historical cohort pilots remain unchanged.

Omitting `--campaign` preserves the legacy CLI, paths, and explicit schema-1
and schema-2 reparsing. This support change does not rewrite their copied
protocols, results, or milestone indices. Python-only validation of the new
campaign selection and binding checks passed 16 focused tests, and both
historical n1 results reparse unchanged. No new physical run is claimed here.
