# Production compiled DP terminal contracts

These checks exercise the actual production generator and exporter for rank
pairs `(1,1)`, `(2,1)` and `(3,1)`. They contain no performance measurements,
backend reductions, or process-amplitude acceptance claims. The earlier private
experiment remains unchanged.

Recorded result: **PASS**, seven tests, 290 compiler/native subprocesses,
164.602 seconds on CPU 10. All recorded production/test/fixture source hashes
were unchanged between the start and end of the run.

The rank-one reference is the independently frozen expanded oracle. For higher
ranks, the reference is the frozen public compact body from the private
experiment, renamed only at its entry point. Its SHA256 is checked, and the
currently generated public routine must match that frozen body after fixed-form
rendering and removal of outer whitespace. This preserves the compact zero-B skip and its
original operation order. The actual generated polynomial constant module is
shared; independent COMB-table/consumer proof is a separate qualification.

The full suite covers:

- 4,608,000 comparisons across 24 numerical fixtures, 125 runtime shapes, four
  rounding modes, all incoming IEEE flag combinations, three rank pairs, and
  four compiler modes (`-O0 -frounding-math`, `-O3`, `-O3 -frounding-math`, and
  `-O3 -frounding-math -fno-automatic`). It checks consumed bits, complete
  fallback output, branch admission, input preservation, and all IEEE flags.
- 104 paired trap probes. Higher-rank probes include zero cut size with
  signaling-NaN B: compact traversal still compares B before the empty I loop.
  INEXACT probes that arm a trap with an already-set flag may signal during trap
  setup; the separate clear-entry off-diagonal witness tests arithmetic's
  INEXACT requirement directly.
- Actual module strides at minimal and rank-six storage sizes, non-four matrix
  dimensions, exceptional inactive A tails, highest active A/B endpoints,
  signaling-NaN A with signed-zero B, positive-zero full output initialization,
  and exact highest-coefficient impulses (14→34 and 34→69).
- The production exporter transforms the actual fixed-form caller fixture.
  The final vertex changes INEXACT in both directions; actual CREATE and MERGE
  consume its output at ranks two, three and four. Uninstrumented production
  helper signatures, immediate and final flags, final coefficient bits, and
  ordinary/static allocation are checked. Callers compile at O0 and polynomial
  routines at O3, reflecting their separate build rules.

Reproduce from the repository root, with the compiler on PATH:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. taskset -c 10 python3 \
  docs/benchmarkfolder/experiments/dp-higher-rank-production/compiled/run.py qualification
```

`qualification.log` is the unittest output. `qualification-commands.jsonl`
records each compiler/native invocation with its stdout, stderr, return code,
and compile-input hashes. `qualification-summary.json` records the toolchain,
CPU affinity, source hashes before and after execution, and overall outcome.
Durations are diagnostic wall times, not benchmark samples. No binaries are
retained here.

`attempt1-smoke.log` records a Python syntax error in the initial test-helper
edit, before any compiler invocation. `attempt2-smoke.log` records the corrected
three-test smoke run. The final qualification uses the complete suite,
including the added minimal-storage cases.
