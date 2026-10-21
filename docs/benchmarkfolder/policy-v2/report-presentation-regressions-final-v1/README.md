# Report presentation software validation

The actual report unittest module passes all 43 tests with no skips, including
the three added synthetic PDF presentation tests. The tests verify ggH/ggHH
backend DP/QP counts, median/95% interval/decision display, retained optimization
attribution, deterministic output and strict data-reader boundaries. Synthetic
layout fixtures are explicitly incomplete and do not certify physics or timing.

The reviewed patch changes only the Typst presentation template and its test
module. All three stress specs, nine frozen collection sources, generator
identity, unchanged HEAD and historical top-level PDF/Typst are preserved.
No stage or commit occurred while the numerical workers were active.

Reproduce from the repository root with the pinned environment:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export RAYON_NUM_THREADS=1
taskset -c 4 python -B -m unittest tests.unit_tests.loop.test_ninjars_report_v2 -v
```

Raw command/result/stdout/stderr and the exact application/patch are retained.
The generator and library runtime are unchanged; this proof adds no physics,
stability, rescue or final performance acceptance claim. The actual final
combined benchmark rendering and visual/reproducibility checks remain pending.
