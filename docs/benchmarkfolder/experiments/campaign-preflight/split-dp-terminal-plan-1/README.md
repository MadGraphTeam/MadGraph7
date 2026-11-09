# Proposed optimized campaign preflight

This is a planning snapshot, not a registered or executed campaign. Registration
is held until the coordinator freezes the production sources and protocol. The
proposed ID is `split-dp-terminal-1`, with predecessor `legacy`. A changed
production implementation before registration requires an updated planning
snapshot; this document does not authorize importing private prototypes.

The intended implementation combines the separate primary/stability DP
workspaces with the guarded DP terminal generator from commit
`8ae62e25d85b1c6c81f37ccdb86f71bc550ed16a`. Administrative timeout bindings are
committed in `f2cdf78566`. The actual installed package was built by the CLI from
`606603e720aff7e96df0c9c7981e7bafd09b5d1c`; later generator changes require fresh
Fortran exports but do not alter that bridge. Its prefix is
`.tirust-build/current-install/mg/HEPTools/tirust`, and its shared-library SHA256
is `095e957990c2e7ed87ed9ea73fee71eb132f00b330e0d38fd423163c173e358c`.
The package passed its actual installer DP/QP/poles/cache smoke and the retained
90-call, 110-event parity replay. These qualify the package, not the physical
matrix or its performance.

The private Born-cancellation fallback, 512-bit bridge and higher-rank kernel
proposals are excluded. The original `49af` installation, `c91cb85f3a` generated
exports, failed QCD/EW four-body preparations and qualified QCD 2→1 runtime
failure remain preserved under their original identities.

## Readiness at this snapshot

`readiness.json` binds the observed exporter inventory, installed package,
installer evidence, model pin, policy, all eleven baseline build records and
all point files. The collector performs no compilation or numerical evaluation.
All eleven baseline exports have source-built numerical provenance, matching
current generated source inventories and executable hashes, recorded effective
compiler policies and equal before/after link snapshots. No worker FIFO is
active. Current dependency and compiler-policy equality must still be checked
when reuse actually starts; recorded equality alone is not a fresh dependency
qualification.

The baseline remains
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. The existing QCD 2→1 deterministic
relink qualification proves that an untouched old driver and numerical objects
can reproduce their recorded executable byte for byte. Campaign reuse repeats
that proof separately for all eleven cases before copying their source-built
objects into private output trees. It checks exact dependencies and effective
flags, copies independent cards and relinks only the benchmark driver. A failed
relink proof requires a full numerical source rebuild; it is not waived.

The new candidate needs **eleven fresh actual MG CLI exports and full generated
source builds**. All configuration `set` commands use `--no_save`. Builds must
retain the installed backend availability, exact native artifacts, physical
cards, compiler flags, generated sources and new driver identity. No previously
generated candidate numerical objects are reused.

The frozen policy and points are unchanged: 100 physical points for each of
eight main processes and three controls, `MLStabThres=1e-6`, automatic
`CTModeInit=1`/`CTModeRun=-1`, forced host stability checks, full finite terms and
UV/IR poles, QP integrands and registered cache settings. The model archive is
`cffc0916c8edda35e78e3d45156b5356e84459dc086cf4c71bff9c9b6bbf5955`.
Baseline chains are materialized as `8|6|7|1` for main/mixed cases and `7|6|1`
for the two loop-induced controls. Candidate chains prepend 9; registration and
generation retain the observed chains rather than replacing them with a
hardcoded default.

## Proposed execution after source freeze

The commands below are a proposal only. Run from the repository root after the
coordinator's source freeze, with no further source or package mutation through
the campaign. Registration records the actual checkout revision, source bytes,
native bytes and timeout map, even if documentation commits followed `8ae`.

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark.py register-campaign \
  --campaign split-dp-terminal-1 --predecessor legacy \
  --tirust-prefix "$PWD/.tirust-build/current-install/mg/HEPTools/tirust" \
  --reason 'Separate primary/stability DP workspaces and guarded DP terminal generation' \
  --worker-timeout qcd_n4_uux_zzgg=14400 \
  --worker-timeout ew_n4_epem_mupmumaa=14400
python3 tools/tirust_benchmark.py reuse-baseline --campaign split-dp-terminal-1
python3 tools/tirust_benchmark.py build --campaign split-dp-terminal-1 --source baseline
python3 tools/tirust_benchmark.py build --campaign split-dp-terminal-1 --source candidate
```

The final command invokes actual generation before building each candidate.
All other cases retain 3,600-second administrative worker deadlines. Both
four-body cases use 14,400 seconds. These limits apply per worker startup or
protocol wait, not to the full campaign wall time, timing batch target or point
count. Specifications and worker inputs bind them and resume verifies them.

Next run the numerical preparation once per case, always baseline CutTools plus
explicit candidate backends 1, 6, 7, 8 and 9, for example:

```sh
python3 tools/tirust_benchmark_numerical.py \
  --campaign split-dp-terminal-1 --case qcd_n4_uux_zzgg \
  --label numerical-qcd-n4-steady-1 --warmup-only
```

Use unique labels for all eleven cases. The final complete steady 100-point
warmup traversal is sufficient for this untimed preparation and avoids a second
expensive traversal. Every point, answer column, split order, accepted status,
finite check and input binding must pass; partial/time-limited streams remain
failed evidence. Adjudicate every `3e-6` disagreement and record ordinary `1e-8`
target misses. This preparation supplies at least 6,600 steady point evaluations
across the six lanes, plus initialization calls. It never supplies timing
acceptance. Numerical qualification must be reviewed separately before timing.

After all owned compilation and diagnostics finish, sample allowed CPU and
sibling activity, select a quiet allowed CPU, retain shared-host conditions and
freeze that selection for the paired campaign. Keep all backend-8/9 executions
serialized with the common numerical lock. Do not assume exclusive use of the
host or qualify timing affected by a known resource failure.

```sh
python3 tools/tirust_benchmark.py campaign \
  --campaign split-dp-terminal-1 --cpu SELECTED_ALLOWED_CPU
python3 tools/tirust_benchmark.py summarize --campaign split-dp-terminal-1
```

Replace the symbolic CPU argument only after sampling the actual environment.
The controller first attempts all eight main TIRust-first comparisons. Every
one retains seven alternating pairs, complete 100-point traversals, sustained
calibration toward two-second batches and the upper-95%-bound limit of 1.05.
A failed or ineligible gate remains authoritative, returns nonzero and defers
the subsequent expensive phase. No eligible failed result may be replaced by
selecting a better rerun.

If all eight gates pass, the controller completes the explicit-backend
comparisons and controls. Main existing-backend 1/6/7/8 regressions retain the
same speed gate. Standalone 9 and control speed comparisons are descriptive;
their numerical and duration validity remain required. The complete timing
matrix contains 66 paired comparisons, or at least 92,400 timed point
evaluations before longer calibrated batches, initialization and warmup.

## Scheduling limits

The retained historical cost model gives approximately **17.0 hours of known
main-matrix work**, plus 10.8 minutes for known control work. It is an incomplete
planning subtotal, not a lower bound or a prediction for this candidate. It
excludes compilation, queues, startup/SETPARA, numerical preparation, retries,
stress qualification, candidate-chain costs and several expensive n4 lanes.
Its assumptions and per-case values are retained in
`../../scheduling/historical-complete-pilots-2/`.

The original explicit-9 n4 preparations reached only 28 complete public calls
for QCD and 66 for EW within their reported 1,800-second intervals; neither
reached READY. These partial streams do not establish steady full-traversal
costs. Moreover the full rescue chain tries the remaining DP backends before
TIRust QP, so explicit-9 QP-heavy cost cannot be substituted for chain cost.
Budget hours for numerical preparation and potentially multiple days for a
complete successful regression campaign; there is no measured finite upper
bound for the full schedule. Administrative deadlines retain actionable
failures without shortening the agreed physics or timing protocol.

Complete source preparation, independent evidence audits, report rendering
preparation and failure triage before quiet timing. During long numerical
preparation, independent source-only work may continue; no acceptance timing
runs concurrently with owned compilation. A failed main chain should guide the
next implementation step before spending the remaining regression budget.

To reproduce this read-only inventory, write to a fresh output file:

```sh
python3 docs/benchmarkfolder/experiments/campaign-preflight/split-dp-terminal-plan-1/inventory.py > /tmp/tirust-readiness-new.json
```

The observed timestamp or documentation revision may differ in a later read;
the retained snapshot is never overwritten or presented as final preregistration.
