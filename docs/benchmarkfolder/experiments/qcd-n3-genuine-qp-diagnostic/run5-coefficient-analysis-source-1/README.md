# Bounded closed-run5 coefficient analysis

This additive pure-Python helper consumes an explicit closed JSON and its SHA256. It never discovers or executes native workers, compilers, loaders, or reducers. Inputs are either the unchanged adjudicator's 16-worker combined analysis or 16 complete parsed workers with public parity records. Source, build and run evidence remain immutable.

`ANS(0:3,0:1)` maps to eight column-major slots: Born, finite, single pole, double pole for total, then split order 1. The helper independently recomputes all 384 permitted scalar deltas, verifies the closed adjudicator values, and reports 48 pair/point comparisons. Defined and undefined zero/opposite metrics remain separate. Any common-input mismatch suppresses every numerical summary. Threshold counts are strict `>` discrepancy counts, with defined denominators, not accuracy tests. No raw native integral truth voting is performed.

All actual workers are unchecked status 120, accuracy [-1,-1]. The common-input gate establishes consistency only. There is no independent oracle, true-error estimate, stability acceptance, timing qualification, or general QP accuracy claim. Native precision/transport caveats are embedded in the JSON.

From the repository working directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/run5-coefficient-analysis-source-1/run_tests.py
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/run5-coefficient-analysis-source-1/summarize.py --input .tirust-build/qcd-n3-genuine-qp-diagnostic/run5/analysis.json --sha256 241ed7c23c6eb52728b7c2a79a7d17ba7558652eade794947c14a0223ff0c08a > /tmp/run5-coefficient-summary.json
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/run5-coefficient-analysis-source-1/render.py /tmp/run5-coefficient-summary.json > /tmp/run5-coefficient-summary.md
```

If the ignored run5 namespace is unavailable, the immutable `run5-retention-1/actual-run5/analysis.json.gz` decompresses to the same pinned input; the result packet references the retention index instead of duplicating raw data. `source-proof.json` pins that compressed and decompressed relationship, source5, driver slots, adjudicator formula and generated amplitude semantics. Eleven synthetic tests run with process/native invocation functions blocked and temporary synthetic fixtures only. Overflow of metric intermediates fails explicitly. JSON constants NaN/Infinity, malformed matrices, unchecked-status changes, duplicate arms, bad shared hashes, tampered supplied metrics and partial/mismatched public parity fail closed.

`render.py` is only a compact view of the bounded summary; JSON is the complete eight-slot numerical record.
