# Final-integration compensation: negative diagnostic

This isolated copy of the pinned fast engine changes only final contraction of
its already computed reduction coefficients with captured native master values.
No production dependency or installed library is changed. Master requests are
checked against the captured arguments bit for bit, and unchanged mode reproduces
all six frozen Laurent components exactly for loops 3/52/71 in both directions.

Mode 1 uses Neumaier sums of rounded complex products. Mode 2 also retains the
FMA residual of each real product in a complex multiplication. Both keep the
original reduction, cut geometry, numerator callbacks, thresholds and failure
checks. Only finite successful frozen inputs have been tested; this is not a
validated replacement for the general production integrator.

| Finite real error vs full Float256 | Original | Compensated sums | Sums + product residuals |
|---|---:|---:|---:|
|loop 3 forward|−2.75e-12|−2.73e-12|−5.34e-12|
|loop 3 reverse|−2.46e-11|−2.46e-11|−2.31e-11|
|loop 52 forward|+4.10e-12|+4.05e-12|+2.17e-12|
|loop 52 reverse|−1.05e-11|−1.05e-11|−9.69e-12|
|loop 71 forward|−4.48e-13|−4.47e-13|−4.38e-13|
|loop 71 reverse|−3.63e-12|−3.63e-12|−3.59e-12|

Summation compensation alone has negligible effect. Product residuals reproduce
the exact-weight diagnostic from `../qcd3-point67-masters`, but do not remove
earlier numerator/cut/reduction errors; one forward component becomes less
accurate. No change is promoted and no performance claim is made. `summary.json`
and `mode-*.json` retain all real/imaginary Laurent components and rational terms.

`engine.patch` and `accumulator.rs` contain the complete isolated change, with
source and executable identities in `source.json`, `build.json`, and `run.json`.
The source is NinjaRS revision `0d0d7e7a8033431b60a8580731084742e0655cda`,
GPL-3.0-or-later. Reproduction reads the reference clone and writes only to a new
ignored experiment directory:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/qcd3-point67-compensated/prototype.py prepare
python docs/benchmarkfolder/experiments/qcd3-point67-compensated/prototype.py build
python docs/benchmarkfolder/experiments/qcd3-point67-compensated/prototype.py run
```

Preparation refuses an existing destination. The DP replay needs the retained
scalar capture in `../qcd3-point67-masters/results.json`; it does not invoke
Symbolica or a native integral provider during the reduction runs.
