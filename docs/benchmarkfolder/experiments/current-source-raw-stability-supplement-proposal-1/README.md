# Optional current-source raw-stability report page

This source-only proposal adds one optional, source-labelled QCD n3 raw-stability
page to the current-source supplement. It changes only
`current_source_supplement.py` and `current-source-supplement.typ`, and proposes
a new `progress-2` selection/context. The published 33-page report at Git
`48b664a029` and its immutable `progress-1` selection/context remain historical
reproduction anchors. Reproduce that exact report from that Git revision; its
ledger continues to pin its original helper bytes. The preceding-source plots
and all original NinjaRS reporting remain untouched.

The optional role accepts exactly four admitted index pins: current-source raw
data `b2589cfc…`, figure `cac6a230…`, pure figure verifier delivery2 `dcef210d…`
and plotter6 `a0521f17…`. It binds every indexed payload through `ReportInputs`
and invokes the existing raw and figure verifiers in indexed temporary copies.
It never discovers a latest path or follows documentary host paths. No raw
payload is duplicated in this proposal.

The retained manifest and production revision join the current campaign exactly.
The exporter revision `0fcd2cd…`, generated-receipt revision `6c77fe…` and pilot
controller seal `c071ec1c…` remain separate provenance fields. The nine-case
numerical whitelist is unchanged; both n4 cases stay pending in this selection.

All five raw workers completed: 1,500 primary and 25 bootstrap calls. Availability
is 100/100/100/100/95; unchecked return 110 identifies neither the actual reducer
nor internal precision. The five TIRust unknown points retain 15 rejected probes
and 27 failed group observations. Five later healthy calls clear incoming
software latches under qualified entry-reset semantics, without qualifying the
failed calls. All 100 points remain denominators. At `1e-3` TIRust's fraction is
unknown within `[0,0.05]`; dotted upper bounds are not measured failure fractions.
The study is host forced-DP/card-selected finite/Born internal consistency,
with 1% empirical resolution and no independent true-error, hardware IEEE,
ordinary numerical-acceptance or performance claim.

## Validation

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-source-raw-stability-supplement-proposal-1/test_raw_slot.py -v
```

Ten focused tests passed, including actual full raw and figure replay with
process/foreign calls blocked; synthetic negatives test wrong/unadmitted pins,
source mismatches, lost populations and false measured-fraction/performance
claims. The absent optional role preserves the previous supplement view and
does not probe raw data. `baseline-current-source-supplement.typ` is the exact
33-page milestone's supplement source for the additive layout assertion, so
that assertion does not depend on an unpromoted canonical file.

`preview.json` and `preview-inputs.json` came from a private complete report
preparation using explicit proposed-file mappings, not canonical replacement.
The actual preparation verified 11,988 inputs, wrote no canonical context and
made no process/foreign/native/compiler calls or Typst render. Removing only
the new `raw_stability` key yields exactly the existing `progress-1` context.
`comparison.json` records 11,857 common input paths: only the two authored
supplement sources change; all other 11,855 common identities remain exact.
The new view adds 131 explicitly bound paths and omits the old progress-1
selection/context from its own ledger; those old files are preserved.

Independent bounded source review found no blocker. Root must still render and
visually check the intended single added page, including label legibility at
125 mm image height. No final page count is asserted before that check.

## Promotion and explicit reproduction

These canonical commands are for root after review; the proposal author has not
executed them. The historical dossier is unchanged at
`fa88e07bd94c08dbaafd8e8ad642d82dbbacf1220dea8462145281040543c558`;
no new historical collection or numerical run is needed.

```sh
source tools/tirust_env.sh
raw_report_proposal=docs/benchmarkfolder/experiments/current-source-raw-stability-supplement-proposal-1
git apply --check --whitespace=error "$raw_report_proposal/existing-files.patch"
git apply "$raw_report_proposal/existing-files.patch"
mkdir -p docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/report-supplements/progress-2
cp "$raw_report_proposal/selection.json" docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/selection.json
current_report_dossier=campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json
current_report_supplement=campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/selection.json
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report "$current_report_dossier" --supplement "$current_report_supplement"
```

The build creates the new explicit `progress-2/report-context.json` and fresh
ledger. Do not copy a private preview into that path. Preserve progress-1 and
use the commands above in the canonical README only with the actual reviewed
PDF release and its measured page count/hash. Delivery stays incomplete and
precise local timing remains deferred to Hua-Sheng.
