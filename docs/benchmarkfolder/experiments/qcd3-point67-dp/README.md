# QCD3 point 67 DP accuracy diagnosis

The extra NinjaRS rescue is an aggregate loop-direction accuracy failure.
Neither engine rejects any of the 116 reductions in either direction, and the
normal MadLoop DP decision reports `FPE_IN_DP_REDUCTION=false`.

With the unchanged `MLStabThres=1e-9`, the normal DP accuracy is
`1.74797848921705276e-9` for NinjaRS and `4.58530986091147102e-11` for original
Ninja. The diagnostic stops immediately after recording the normal DP decision,
before any possible rescue call. No threshold or stability guard was weakened.

| Engine | Forward finite | Reversed finite |
|---|---:|---:|
|NinjaRS|2.15825542034988968e-8|2.15825473632367783e-8|
|Ninja|2.15825535334004682e-8|2.15825534816969286e-8|

The isolated NinjaRS wrapper also calls original Ninja with the identical
tensor, thresholded S matrix, momenta, complex masses and squared scale. It
retains all inputs, statuses and Laurent coefficients for each call. The
largest same-input finite discrepancies occur in loop 52 (box, rank 3), loop 3
(triangle, rank 2), and loop 71 (pentagon, rank 4). Loop 52 differs by 7.34e-12 forward
and 2.19e-11 reversed before the common `2/(16*pi²)` normalization and `/72`
spin/color/symmetry average. Original Ninja is a comparator, not assumed exact.

`direction-1/` and `direction-2/` contain forced DP controls with status 110.
Those controls do not measure normal stability. `normal/` records the actual
normal DP decision and 232 paired reductions; `normal-ninja/` records the
original-Ninja decision using the same terminating driver. The compressed JSON
preserves binary64 values with round-trip JSON decimals. `points.dat`, cards,
source copies, build identities and commands are retained alongside them.

The following commands operate only on a copied generated output:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/qcd3-point67-dp/probe.py prepare
python docs/benchmarkfolder/experiments/qcd3-point67-dp/probe.py run --cpu 4
python docs/benchmarkfolder/experiments/qcd3-point67-dp/probe.py normal --cpu 4
python docs/benchmarkfolder/experiments/qcd3-point67-dp/probe.py reference --cpu 4
```

The first discarded forced-mode setup used `FORCE_STABILITY_CHECK(true)`,
which continually reports an initialization-style status with an explicitly
forced CT mode. It was stopped, retained under the ignored experiment's
`qcd3-initial-forcecheck`, and corrected to false for the forced controls.
Normal-mode accuracy still uses the existing MadLoop decision logic.

`oracle.py` built `replay.c` against the installed production ABI2 archive with
GNU runtime ordering and successfully evaluated loops 3/52/71 in both
directions. DP replay exactly reproduces all six recorded Laurent components.
The same frozen binary64 inputs are widened through 40-digit binary128
transport and reduced with genuine Float(256) arithmetic. The finite oracle
values agree between directions in all 40 recorded digits for each loop.

Signed finite real errors relative to that oracle, before common normalization:

| Loop | Direction | NinjaRS DP error | Original Ninja DP error |
|---|---:|---:|---:|
|3|1|-2.755e-12|-9.977e-12|
|3|2|-2.458e-11|-1.362e-11|
|52|1|+4.101e-12|+1.144e-11|
|52|2|-1.048e-11|+1.141e-11|
|71|1|-4.481e-13|-2.280e-12|
|71|2|-3.629e-12|+1.816e-12|

Original Ninja is not uniformly more accurate per reduction. Its box error is
nearly independent of direction, explaining part of its smaller aggregate
direction difference. The target remains actual DP accuracy; canonicalizing
directions or weakening the stability test would not establish that.

`oracle-inputs.json`, `oracle-output.txt`, `oracle-results.json` and
`oracle-build.json` preserve the complete inputs, outputs, errors and provenance.
Reproduce with `oracle.py prepare`, then `oracle.py run --cpu 207`, using the
shared Symbolica bookkeeping directory and no concurrent MP process. This
oracle tests reduction accuracy for frozen DP coefficients; actual QP numerator
construction is a separate control.
