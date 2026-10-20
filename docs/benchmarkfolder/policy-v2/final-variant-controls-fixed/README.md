# Final-source rotation and context controls

This first attempt is a retained failure. Rotation passed; the changing-state
driver assigned only the DP top mass before `COUP()`, leaving the independent
MP external mass unchanged. CutTools rescued three points through QP and
exposed 18 coefficient disagreements. Fresh-context and near-singular variants
were not started. See `execution-failure.json` and all eight raw streams.
The separate `../final-variant-controls-both-precision/` driver corrects both
precision assignments and passes; these failed outputs remain unchanged.
`SHA256SUMS.json` seals this namespace, including its original preparation.

This separate namespace reuses the existing control transformations and Fortran
driver from `tools/ninjars_controls.py`. Preparation clones only the final
`control-gg-hh` and `control-forced-qp` CLI exports, checks writable aliases before
building, and requires every transformed point stream to match the retained
historical stream byte for byte. Model/reduction sources and libraries must
remain identical to their parents; only the state-update driver may differ.

The four complete ten-point controls exercise:

- A fixed spatial rotation, with two DP/QP stability rotations.
- Repeated identical external momenta while top mass and scale change.
- The same state sequence with NinjaRS contexts destroyed before every call.
- The retained near-singular angles, with strict `1e-20` checks and actual QP.

All four backends run for rotation/rebinding. The fresh-context comparison uses
NinjaRS; the near-singular test uses CutTools, Ninja and NinjaRS. The latter
requires a successful NinjaRS rescue. Every raw output is retained, failed
statuses remain failures, and repeated evidence is never overwritten.
These fresh-process diagnostics do not enter performance acceptance.

From the repository root, after final CLI generation completes:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-fixed/execute.py prepare
```

After reserving the single Symbolica slot:

```sh
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-fixed/execute.py run --cpu 4
```

Finally compare rotation invariance to the independently collected, unrotated
final ggHH controls:

```sh
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-fixed/execute.py \
  compare-rotation --binding \
  docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-terminal-fixed/controls-candidate-abi3-terminal-fixed-hh-complex-20261020.json
```

The comparison verifies the reference's role, exact generated source inventory,
generator identity, libraries and physical card before comparing all coefficients
and orders. Ordinary forced-Ninja failures remain in their own evidence; this
control does not relabel those inputs or claim that strict checks always succeed.
