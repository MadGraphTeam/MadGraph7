# Fortran call-boundary and cache-clear promotion

Two generated Fortran procedures change. The optimized amplitude entry resets
its DP/QP software reduction-failure flags before a new independent call. A
failure remains latched within that call across groups, helicities, split
orders, direction checks, rotations and existing precision rescues. This fixes
inherited state contamination in forced modes and after terminal failures.

The generated TIRust event-clear wrapper checks the shared module's context
generation before clearing. An absent context remains absent; an existing
context is cleared even if another subprocess created it. The public creating
clear API, native library, dependency pins and COLLIER code are unchanged.

The new canonical behavior tests first failed on the old live templates:
seven failures reproduce call contamination and inactive-context side effects.
After the two patches, all 89 tests in `green.json` pass. They include compiled
Fortran/C fixtures, adapter and NinjaRS wrapper checks, precision-stream checks,
and campaign/local provenance guards. `red.log` is an expected failed regression
run, not a successful numerical run. `green.log` is the complete passing output.

Independent private evidence:

* [Entry-latch proposal](../reduction-latch-lifecycle/source-proposal1/README.md)
  and [closed physical parity](../reduction-latch-lifecycle/qualification1-retention-1/evidence-index.json):
  four original/candidate QCD n3 DP/QP workers, 20 public calls, three points,
  exact output/status/IEEE parity. Fixed-mode status 120 remains unchecked.
* [Cache-clear measurement](../fortran-lazy-tirust-clear-measurement-1/README.md):
  7,200 retained numerical rows match exactly. Seven shared-host paired full
  amplitude repetitions give CutTools ratio 0.996826 (95% interval
  [0.994922, 0.998734]) and TIRust ratio 0.996882 ([0.990792, 1.003011]).
  The active-clear microbenchmark has a measured 0.76% overhead. These are
  descriptive observations for one QCD n1 process, not final acceptance.

The two changes were qualified separately with their exact candidate generated
sources; the combined live-template regression suite passes. The old eleven-
process campaign and its 31-page PDF remain evidence for their recorded source.
Fresh exports, builds and all 66 numerical lanes are required for this successor.
Precise process-wide performance remains Hua-Sheng's local TODO.

Reproduce the passing regression suite from the repository root:

```sh
source tools/tirust_env.sh
python3 -B -m unittest tests.unit_tests.loop.test_tirust_reduction_latch tests.unit_tests.loop.test_tirust_transport_cache tests.unit_tests.loop.test_tirust_fortran tests.unit_tests.loop.test_ninjars_fortran tests.unit_tests.loop.test_tirust_precision_stream tests.unit_tests.loop.test_tirust_benchmark_campaign tests.unit_tests.loop.test_tirust_benchmark_local -v
```
