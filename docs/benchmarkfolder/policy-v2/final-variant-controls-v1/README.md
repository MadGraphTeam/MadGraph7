# Final rotation and parameter-rebinding refresh

This is source-only preparation. No variant has been compiled or evaluated.
The exact corrected `controls_driver.py` is copied byte for byte from the
completed `final-variant-controls-both-precision` evidence. Its original hash
and the operational `execute.py` retarget patch are retained in
`source-provenance.json` and `retarget.patch`. Numerical mathematics,
thresholds, ten-point streams and state schedules are unchanged.

The two actual final CLI parents replace the older thirteen-case generation
assumption:

- `control-gg-hh` uses the fresh `supplement-gg-hh` export under
  `.benchmark-work/policy-v2/final-supplement-generation-v1`.
- `control-forced-qp` uses the fresh export under
  `.benchmark-work/policy-v2/final-representative-generation-v1`.

`parent-bindings.json` retains their actual generation, compiled source,
physical card and complete reducer descriptors. Both belong to generator
revision `b0ea105a`, numerical source `d05b59f3`, installed archive
`800a941f...f5d42` and runtime descriptor `8b454519...26cff`. The complex-mass
control is a separate actual CLI build in `../final-complex-generation-v1`;
its completion does not imply a completed thirteen-case campaign.

The existing preparation clones those two parents into private writable
outputs. It preserves four ten-point controls: the fixed spatial rotation,
changing top mass and scale, the same state sequence after finalizing NinjaRS
contexts at each call, and the retained near-singular angles. The corrected
driver assigns both independent DP and MP top masses and scales before the
generated `COUP()` update. All four backends run rotation and rebinding;
NinjaRS runs the fresh-context comparison; CutTools, Ninja and NinjaRS run the
near-singular control. Raw failures remain failures. These diagnostics do not
enter performance acceptance, and DP rebinding results do not establish QP
rebinding coverage.

The fresh HH stream contains 100 points with the same seed. Its first ten
serialized momentum configurations exactly equal the retained ten-point
rotation parent; `prefix-equivalence.json` records that input fact without
claiming numerical results. The later rotation comparison consumes the
existing supplement binding validator, requires the fresh candidate's source,
runtime and card identities, and compares only those exact ten input points.
The full fresh numerical stream remains retained; selected records and counts
are recorded separately. Old candidate output rows are not used.

After root schedules compilation, then the sole numerical slot, the existing
operations are:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-v1/execute.py prepare
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-v1/execute.py run \
  --cpu "$NINJARS_FINAL_CPU"
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-v1/execute.py \
  compare-rotation --binding "$NINJARS_FRESH_HH_BINDING"
```

The first operation compiles variants and has not run during this source-only
phase. Root's process/timeout ledger surrounds later operations; previous
attempts are never overwritten. The fresh candidate HH binding must exist
before the last command. No additional numerical algorithm or acceptance
criterion is introduced here.
