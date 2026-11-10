# Current-source report supplement proposal

This source-only proposal retains the preceding `invariant-symmetry-1` report
and plots, then adds separately bound progress for `entry-latch-cache-guard-1`.
The frozen current-source snapshot contains **nine cases, 54 lanes and 100 saved
points per lane**. Both n4 cases are pending in this selection; that says nothing
about later or active work. No compiler, reducer or Typst render was launched.

The main campaign remains the preceding source `2c13246ffa63`. The supplement
binds production `0fcd2cd873d6`, exporter `afb6e0be…`, numerical identity
`3c13ad88…` and unchanged native DSO `eb3d67fd…`. It cannot import the preceding
66 lanes as qualification for the changed Fortran. All 66 fresh local lanes and
precise local timing remain Hua-Sheng's TODO.

QCD n3 passes `3e-6` and retains the `1e-8` target misses: candidate tools
6/7/8/9 each have 56 finite-column scalar misses across 28 saved points, with
both order rows retained. TIRust accepts 87 DP and 13 QP results. The EW n3
baseline receipt limitation is explicit: 375 inherited objects, 426 generated
source pins and donor compiler flags match, but the original full-build receipt
was not located. This does not claim a fresh baseline full rebuild.

The cache table uses the existing `ratio-fill` palette and includes all four
measured scopes, including the active-wrapper overhead. It measures the guard
alone; the entry-latch reset was not part of that paired experiment. Shared-host
observations do not establish the formal local performance acceptance.

## Narrow implementation

The five report source paths are in `proposed/`:

- `build_report.py`: optional `--supplement`, independent validation and a new
  explicitly named context file. The original context is preserved.
- `benchmark.typ`: a conditional upfront two-source panel and appended section.
- `campaign_report_refresh.py`: one optional historical documentation role.
- `current_source_supplement.py`: exact admitted index pairs, indexed temporary
  relocation and existing pure numerical/cache verifiers. No active discovery.
- `current-source-supplement.typ`: the existing typography, navy headings,
  table functions and timing colours, without changing `style.typ`.

The original NinjaRS report is outside this patch. Both existing-file and README
patches remain unapplied.

## Historical document provenance

The published 31-page PDF remains reproducible at Git commit
`913a746201eb6ef806ccb219c890a7f3486d6322`, SHA256
`91ceeae99968df742e7e141467e3bab321394a0512328be7df9be0f5b86d559a`.
Its original selection and `b03e1b93…` dossier are unchanged.

That dossier pins the old global precision TODO. The new selection instead
names the exact archived document at
`history/report-before-entry-latch-cache-guard-1/PRECISION_STREAM_TODO.md`,
10,893 bytes, SHA256
`b48071ea8b8c4a1176f88ab78eb5678792748deaecc30c2d4845b37316501c03`.
Only this documentation role and its exact path/hash are allowed. No logical
path has its bytes substituted, and no scientific validator is weakened.

The fresh proposed historical dossier is
`fa88e07bd94c08dbaafd8e8ad642d82dbbacf1220dea8462145281040543c558`.
`historical-comparison.json` proves that every scientific field, criterion,
numerical checkpoint and both plot packages are unchanged. All 10,141 common
input identities match; only the selection and documentation paths are replaced.

## Source-only validation

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-report-supplement-proposal-1/test_supplement.py -v
python3 -B docs/benchmarkfolder/experiments/current-source-report-supplement-proposal-1/run_existing_report_tests.py
```

The focused tests include actual retained n1, seven continuation and cache
replays with process/foreign calls blocked, plus clearly labelled synthetic
negative cases. Python `-O` is explicitly refused before retained verifier code
executes; the optimized test run checks that refusal. Existing report tests run
against the proposed build/refresh modules with their Typst calls mocked.

`prepared-receipt.json` records successful complete report preparation with
11,858 bound inputs, no context write and no render. Its source-only style update
records the subsequent root-requested ratio-cell fill; values and scientific
context are unchanged. Private context/input previews are review artifacts,
not a replacement for fresh canonical collection/build.

## Promotion recipe after root review

Run from the repository root. These commands have **not** been executed against
canonical files by the proposal author.

```sh
report_supplement_proposal=docs/benchmarkfolder/experiments/current-source-report-supplement-proposal-1
git apply --check --whitespace=error "$report_supplement_proposal/existing-files.patch"
git apply "$report_supplement_proposal/existing-files.patch"
cp "$report_supplement_proposal/proposed/current_source_supplement.py" docs/benchmarkfolder/current_source_supplement.py
cp "$report_supplement_proposal/proposed/current-source-supplement.typ" docs/benchmarkfolder/current-source-supplement.typ
mkdir -p docs/benchmarkfolder/history/report-before-entry-latch-cache-guard-1
cp "$report_supplement_proposal/historical-PRECISION_STREAM_TODO.md" docs/benchmarkfolder/history/report-before-entry-latch-cache-guard-1/PRECISION_STREAM_TODO.md
cp "$report_supplement_proposal/historical-selection.json" docs/benchmarkfolder/campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-selection-1.json
mkdir -p docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/report-supplements/progress-1
cp "$report_supplement_proposal/selection.json" docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/selection.json
python3 -B docs/benchmarkfolder/campaign_report.py \
  --campaign invariant-symmetry-1 \
  --selection campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-selection-1.json \
  --output campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json
```

Check the fresh dossier against `fa88e07b…`; do not overwrite an existing dossier.
Then root can render and verify the explicit two-source view:

```sh
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json
current_report_supplement=campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/selection.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
```

Review layout and every retained plot before publication. Apply the separate
`README-reproduction.patch` with the actual report release, updating any final
page count/hash after rendering. Future progress requires a new explicit
selection/context and reviewed checkpoint pins; this snapshot is immutable.
