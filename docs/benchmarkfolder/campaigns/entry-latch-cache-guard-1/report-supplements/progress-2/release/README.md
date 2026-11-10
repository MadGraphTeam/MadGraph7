# Reviewed 34-page report release

The current-source raw QCD Zgg plot is added on page 34. The first 33 pages have exactly the preceding report's extracted text after replacing only their 33 footer totals `/33` with `/34`. The source proposal and data verifier preserve every existing context field and numerical gate. Root visually reviewed the new page; no layout change beyond the sealed proposal was needed.

Root's ten focused tests, canonical render and subsequent serial `--verify-only` all passed. The final ledger binds 11,989 tracked/staged inputs, including the generated new context. The source-only preparation bound 11,988 inputs before that context was created. `verification.json` records exact PDF, context, selection and ledger hashes and the original tool-session results.

Reproduce from the MadGraph checkout:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/build_report.py --campaign invariant-symmetry-1 \
  --campaign-report campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json \
  --supplement campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/selection.json
python3 -B docs/benchmarkfolder/build_report.py --verify-only --campaign invariant-symmetry-1 \
  --campaign-report campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json \
  --supplement campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/selection.json
```

The published 33-page milestone and progress-1 inputs remain immutable at Git `48b664a029d35acb2646b8820f9c654cfed6e852`; use that revision for its exact reproduction. Both old-source and current-source evidence remain distinct. This release retains the current nine-case/54-lane snapshot: both n4 cases are pending in the selection. The added host forced-DP/card-selected consistency study has 95 available TIRust diagnostics, with five unknown outcomes retained; it establishes no independent true-error or performance acceptance. No amplitude was evaluated for this report release, and no production code or reduction-library binary changed. The goal remains incomplete and precise local timings remain Hua-Sheng's TODO.
