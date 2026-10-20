# Final-source rotation and context controls

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
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-both-precision/execute.py prepare
```

After reserving the single Symbolica slot:

```sh
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-both-precision/execute.py run --cpu 4
```

Finally compare rotation invariance to the independently collected, unrotated
final ggHH controls:

```sh
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-both-precision/execute.py \
  compare-rotation --binding \
  docs/benchmarkfolder/policy-v2/runtime-bindings-controls-abi3-terminal-fixed/controls-candidate-abi3-terminal-fixed-hh-complex-20261020.json
```

The comparison verifies the reference's role, exact generated source inventory,
generator identity, libraries and physical card before comparing all coefficients
and orders. Ordinary forced-Ninja failures remain in their own evidence; this
control does not relabel those inputs or claim that strict checks always succeed.

This retry corrects the control driver: both independent DP and MP top-mass
and scale variables are assigned before the generated full `COUP()` update.
The earlier driver left the MP top mass unchanged, exposed by CutTools QP
at points 3, 6 and 9. Its failed comparison and all raw outputs remain in
`../final-variant-controls-fixed/`; no production code or point bytes change.
`controls_driver.py` is the retained control helper with only these explicit
state assignments and their provenance description changed.

The completed run passes all eight comparisons: four-backend rotation and
rebinding, retained/fresh NinjaRS equality, three-backend near-singular rescue,
and four rotated-versus-unrotated comparisons. All ten near-singular points
use CutTools 329, Ninja 328 and NinjaRS 322. Rebinding includes three CutTools
QP points, while NinjaRS stays DP218. No generated QP NinjaRS mass-rebinding
coverage is inferred from this DP test.

`independent-audit.log` retains a separate read-only audit of all twelve raw
streams (240 point/order rows), source/runtime/card identities and comparisons.
`prepare.log` and `execution.log` retain the outer commands. The initial audit
script assertion failure is preserved in `independent-audit-initial.log`; it
compared a descriptor to a hash string and did not change numerical evidence.
`SHA256SUMS.json` seals this completed namespace; the neighboring failed
namespace has its own seal. Recheck both inventories and their raw rows with:

```sh
python -B docs/benchmarkfolder/policy-v2/final-variant-controls-both-precision/verify.py
```
