# Independent final report and precision-reader review

The frozen report, supplement and precision-reader suites pass **55/55 tests**:
36 report tests, nine installed-precision proof tests and ten supplement tests.
This independent run took 6.853 seconds in unittest (7.037 seconds wall time) on
CPU 5. All thirteen recorded source/evidence hashes stayed unchanged between
the start and end. The unittest process and all synthetic Typst compilation
children exited; its process group was empty at handoff. No numerical library
was built and no reduction engine or MP worker was executed.

These tests certify report calculations, evidence boundaries, negative controls
and synthetic layout/reproducibility. They do **not** certify an actual final
eight-process dataset, loop-induced timing result, scalar integral, physical
stability rate, or final benchmark PDF. Those campaigns are still separate gates.
The exact command, source hashes, detailed successful test names and logs are
retained in `validation.json` and `unittest.stderr`.

## Findings addressed during independent review

1. **Precision proof could be relabelled with a final runtime identity.** The
   first reader compared the supplied identity string and a self-bound installation
   archive but did not prove they described the same measured runtime. The final
   reader verifies the full sealed reducer-runtime signature, compares the actual
   archive and all four NinjaRS interface/link metadata hashes, and accepts the
   exact signature independently derived from the final core target. Negative
   controls reject stale archives, stale modules, changed identities and forged
   descriptors.
2. **A large basis-dependent rational diagnostic could enlarge the physical
   convergence scale.** The first convergence reader normalized all eight fields,
   including R1. It now uses only the six complete finite/single/double Laurent
   components for the physical scale and reports R1 separately under that same
   stricter scale. A negative control with enormous R1 and a wrong finite result
   fails. Existing historical Rust expectations were not rewritten.
3. **The transitive precision reader was missing from renderer source provenance.**
   Rendering a supplemental report now hashes the actual precision-reader source
   alongside the report, supplement, profiler, template, style and regression
   verifier sources. The precision C driver also reports actual loaded-object
   paths, and its execution evidence retains/rechecks their hashes.

The owners made these changes; this reviewer changed no report or numerical
implementation. The independent test run followed their explicit source freeze.

## Gate and numerical-report checks

- Missing, failed and unresolved core/supplement/provenance/regression evidence
  cannot become a complete campaign pass. Failed or partial precision executions
  and rejected supplemental collections retain structured failed observations.
  Earlier original-model discrepancies stay visible in separate diagnostics.
- Corrected-cohort comparisons use the common point's largest Born/Laurent
  magnitude over all corrected targets and split orders. A known wrong
  uncorrected original value cannot inflate this scale. Actual discrepancies are
  compared with every other corrected target and are explicitly described as
  cross-backend discrepancies, rather than analytic truth errors.
- MadLoop's estimated accuracies have their own distributions over all order
  slots and the total-order slot. Physical-point DP/QP counts use one total-order
  row per point, with consistent status required across split orders. Timed
  histograms retain every actual traversal. Zero estimates imply no measured
  accuracy guarantee or infinite number of significant digits.
- Actual measured candidate bindings supply ABI 3 and runtime identity. A
  successful installation, archive/config equality, installed Cargo manifest and
  lockfile, full Git revisions and matching overrides are required for final
  pinned provenance. Historical setup ABI/pins and hidden local overrides cannot
  certify a later linked archive.
- Both loop-induced ggH/ggHH streams require zero Born slots, all generated
  squared orders, seven candidate/baseline timing repetitions and the three
  corrected-original references. The 2-to-1 fixed invariant is disclosed despite
  the distinct boosted serialized points. The fixed twelve-point top-threshold
  grid and source-bound installed 256/private 384 precision proof remain explicit.
- The MG7 regression adapter independently checks the 90-case campaign (89
  passes and one existing GoSam Python-3 bypass), its raw logs and narrowly
  reviewed fixtures, then requires all tested production hashes to match the
  actual measured candidate generator. The skip remains visible. This campaign
  excludes the Rust reduction/master engines, which need their own final suite.
- The report template keeps the attributed style and preserves historical top
  reports. Synthetic fixtures test loop-induced, threshold, precision and test
  pages, plus byte-identical PDF creation under different ambient
  SOURCE_DATE_EPOCH values. The final physical report still requires compilation
  and visual inspection after the real accepted measurements exist.

No remaining source blocker was found within this bounded reader/report review.
Any subsequent source, numerical dependency or evidence change needs appropriate
new bindings and regression validation. This review does not justify marking the
overall integration goal complete.

Verify the retained audit without executing tests, compilers or reducers:

```sh
source .ninjars-build/dev-env.sh
taskset -c 5 python -B docs/benchmarkfolder/final-report-audit-v1/verify.py --sources
```

`run_tests.py` is the exact retained one-run controller. It refuses to overwrite
its logs; reproduce in a new evidence namespace after coordinating any benchmark
quiet window. Its only compiler executions come from synthetic Typst tests.
