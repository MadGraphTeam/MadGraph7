# Isolated split-workspace full-amplitude comparison

This diagnostic compares the original baseline NinjaRS-first chain, the
original TIRust-first chain, and an isolated TIRust bridge with separate DP
workspaces for primary and stability-check calls. It does not modify installed
libraries, production source, cards or acceptance exports.

The main comparison loads the candidate DSO into a clone of the original
uninstrumented n1 executable. The bridge's temporary instrumentation defaults
to off. Seven triads rotate/reverse the three variants; the first six cover all
six orderings. Each variant starts one native worker, runs a retained sustained
calibration batch, then measures a second batch. Only one native worker is
alive at once. Three additional runs use a fresh instrumented clone with
Fortran and bridge/upstream timers enabled to attribute the changed geometry
cost. These are separate from the full-amplitude timings.

The runner holds the shared numerical lock. Original executable/driver,
physical point and parameter-card hashes must match qualified attempt 3.
All runtime library hashes must match that record, except for the explicitly
selected TIRust DSO; its build metadata must identify its exact hash. The
original settings and stability checks remain active. Every calibration and
measured batch must use the expected backend in native DP. Full coefficients
are compared at `1e-8`, and original/split equality is also tested by packing
the parsed coefficients as IEEE binary64 bytes. All raw output and observed
statuses are retained. Profiling counts must match the expected two N3/R2
calls per amplitude, with no failed bridge calls.

Wait for the bridge's native cache/failure equivalence checks and for the
campaign coordinator to release the timing window. Then, from the repository:

```sh
source tools/tirust_env.sh
python docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/run.py \
  --name split-workspace-1 --cpu 1 --loops 200 --pairs 7 \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so \
  --build-metadata .tirust-build/n1-profile/split-bridge/build-metadata.json
```

These comparisons are diagnostic and cannot qualify the runtime gate. The
isolated DSO contains disabled profiling branches and has its own build hash;
a production change requires separate numerical qualification and a new
qualified performance campaign. Original attribution and control observations
are retained separately under `../fortran/`.

The retained `split-workspace-1` run used CPU 9 while ordinary CutTools work
continued on other CPUs. All original/split coefficients are bit identical,
all measured and calibration batches use the expected native DP backend, and
all stage counts match. Median baseline/original/split times are
91.291/103.910/101.982 microseconds per amplitude. The paired split/original
ratio is 0.978537 with descriptive 95% interval [0.963998, 0.993296]; the paired
split/baseline ratio remains 1.113072 [1.099917, 1.126384]. The isolated change
therefore shows a small whole-amplitude improvement while leaving a residual
slowdown. This CPU differs from the earlier attribution, so absolute stage
times must not be subtracted across experiments.

Recompute the compact summary from all retained observations with:

```sh
python docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/analyse.py \
  docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/split-workspace-1
```

The separately authorized follow-up changes only compilation of the new
`TIRUST_interface.f` routine to `-O3`. `prepare_o3.py` records the original and
candidate compiler commands, pins the compiler, every other flag and static
archive to the frozen build, and first reconstructs the original executable
hash by relinking untouched objects. It then verifies that every common
object remains byte-identical. It does not alter the common
`TIR_interface.f` or baseline compiler recipes. The runner can then load the
same split DSO for both TIRust variants, so their paired difference isolates
the Fortran build change. This is a diagnostic compiler-policy experiment;
any eventual qualification must bind the resulting compiler recipes explicitly.

`tirust-adapter-o3-build` successfully reconstructed the exact frozen executable
before modifying the adapter. Only `TIRUST_interface.o` changed. The subsequent
`split-adapter-o3-1` run again used CPU 9, with concurrent installer/bridge
compilation documented as a diagnostic limitation. All coefficients remain
bit identical and all measured/calibration batches use the expected native DP
backend. The paired O3/original-adapter ratio is 0.994421 with descriptive 95%
interval [0.983175, 1.005795]; this does **not** establish an improvement.
The optimized-adapter/baseline ratio remains 1.088112 [1.072086, 1.104377].
No compiler policy or production source was changed on this evidence.

To reproduce in new output directories after the coordinator releases
compilation and numerical windows:

```sh
python docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/prepare_o3.py \
  --name tirust-adapter-o3-build --build
python docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/run.py \
  --name split-adapter-o3-1 --cpu 9 --loops 200 --pairs 7 \
  --profile-repetitions 0 --original-use-candidate-library \
  --fortran-build-metadata docs/benchmarkfolder/experiments/n1-profile/whole-amplitude/tirust-adapter-o3-build/build-metadata.json \
  --library .tirust-build/n1-profile/split-bridge/libmadloop_tirust.so \
  --build-metadata .tirust-build/n1-profile/split-bridge/build-metadata.json
```
