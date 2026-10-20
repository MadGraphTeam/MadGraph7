# Warmed Zgg rescue cost partition

This diagnostic profiles the production ABI2 bridge in the original generated
qcd3 output, using normal stability mode and the common `MLStabThres=1e-9`.
No cache or tensor-plan prototype is linked. The retained process warms selected
points 1 and 44 before three recorded commands, each evaluating those same two
points. All printed coefficients, accuracy estimates and statuses are identical
between commands: point 1 is DP218 and point 44 uses genuine QP322. Each command
takes about 6.4 seconds. These are diagnostic profile times, not runtime gates.

The 1,911 user CPU samples partition as follows:

| Work | Samples |
|---|---:|
|Fortran QP numerator construction|44.1%|
|Generic tensor expansion and contraction|18.3%|
|Generic reduction and bookkeeping|15.2%|
|Scalar masters|7.7%|
|Generic cut geometry|6.2%|
|Other MP worker work|5.9%|
|Other main-thread work|2.3%|
|Bridge numerical transport|0.3%|

Scalar, tensor and geometry ownership take precedence over enclosing reducer
frames. The reduction category includes allocation and bookkeeping; it is not
all arithmetic. Unidentified work remains explicit. The profile contains the
small stable point as well as the rescued point, and samples are estimates.
Even eliminating every sampled scalar-master operation would cap the overall
speedup at about 1.083. The previously measured high cache hit rate therefore
does not imply a large total benefit. No cache or tensor-plan change is promoted.

`build.json` records executable, library, driver, source and input identities.
`run.json` and `block-*` retain all numerical rows and statuses. `perf-partition.json`
retains categorization and leaf frames; `perf-exemplars.txt` and `perf-flat.txt`
allow independent inspection. The large raw capture remains under the ignored
experiment directory.

From the repository root, with no other Symbolica process running:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/mp-qcd3-point44/profile.py prepare
SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks" \
  python docs/benchmarkfolder/experiments/mp-qcd3-point44/profile.py run --cpu 207
```

Preparation reuses the verified production worker output from the tensor-plan
comparison; it refuses an existing destination. The script retains the exact
perf command (99 Hz, user CPU-clock events, 65,528-byte DWARF stacks). Compilation
is not included in the measured commands.
