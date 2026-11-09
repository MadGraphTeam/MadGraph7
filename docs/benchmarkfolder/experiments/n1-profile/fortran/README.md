# Isolated n1 Fortran attribution

This diagnostic copies the generated `qcd_n1_uux_z` process. It does not modify
the installed bridge, templates, campaign exports or frozen baseline. The
process contains one rank-two massless triangle and sums helicities before
reduction. A checked ordinary point is expected to call it in CTMODE 1 and 2;
the profiler records the actual count rather than treating that inference as
measurement.

`prepare.py` copies the exact generated files and adds seven stage timers:
initialization/budget, ordered momentum preparation, on-shell-aware kinematic
matrix, invariant conversion/options, Fortran wrapper plus Rust call,
mapping/validation, and contraction/normalization/final checks. It retains the
diffs, source identities and original compilation metadata under the ignored
work directory. `ml_profile.f90` uses 64-bit `SYSTEM_CLOCK` for stages and
`CPU_TIME` for the full timed batch. It calibrates 10,000 empty timer pairs
outside each batch. Raw times are retained; adjusted stage estimates are
explicitly estimates. The temporary Rust profiling ABI belongs only to the
isolated profiling bridge prepared alongside this diagnostic.

Wait for the paired campaign controller to release its native-work window,
then run from the MadGraph repository after sourcing `tools/tirust_env.sh`:

```sh
python docs/benchmarkfolder/experiments/n1-profile/fortran/prepare.py \
  --bridge-library /absolute/path/to/isolated/libmadloop_tirust.so
make -C .tirust-build/n1-profile/fortran/instrumented/SubProcesses/P0_uux_z \
  SHELL="$(command -v bash)" FC="$FC" check
python docs/benchmarkfolder/experiments/n1-profile/fortran/run.py \
  --cpu CPU --name attribution-1 --loops 200 --repetitions 3
```

The runner holds `.tirust-build/numerical-tests.lock` exclusively and runs one
worker at a time. It copies the actual frozen n1 candidate settings, preserves
the 100-point stream, forces the same host stability checks through the original
worker, and records return codes and numerical parity. It compares a pristine
installed-library control, instrumentation disabled, Fortran timers alone,
bridge timers alone, bridge/upstream timers, and combined timers. Profile output
and timer calibration occur outside the timed batch. These measurements are
attribution evidence and cannot qualify the 1.05 runtime requirement.

The retained `attribution-1` matrix contains three repetitions of six levels,
each with 20,000 full amplitudes. All observed outputs are bit identical and all
timed statuses are `219` (backend 9, native DP). Instrumented counts confirm one
N=3/rank=2 call in each of CTMODE 1 and 2 per point. There were no bridge failures.

Fortran-only timer medians, summing both calls per amplitude:

| Stage | Raw time (microseconds) |
|---|---:|
| Initialization and budget | 0.750 |
| Ordered momenta and shifts | 0.785 |
| On-shell kinematic matrix | 0.701 |
| Invariant conversion and options | 0.120 |
| Fortran wrapper and Rust | 8.196 |
| Mapping and validation | 0.878 |
| Contraction, normalization and final checks | 0.586 |

The nonbridge stages total 3.819 microseconds inside these brackets. This does
not cover all Fortran overhead: a read-only object audit finds seven automatic
array allocations before the first timer and seven frees after the last
adapter timer, plus IEEE procedure entry/exit outside the brackets. Their costs
have not been measured separately. The stage observations do not establish
that their costs can all be removed.
The empty interval measured about 30 ns, while the complete empty timer pair
cost about 67 ns. `summary.json` retains both raw and adjusted estimates, all
three observations, ranges and CPU timings. The nested bridge timings must not
be added to the wrapper stage. The first pristine run was 116.919 microseconds
per amplitude, followed by 70.773 and 70.616; the wide first-run variation makes
cross-level timing differences unsuitable for causal speedup claims.

`original-controls-1` therefore repeats the original, uninstrumented baseline
and candidate binaries in seven alternating paired batches, after a sustained
calibration batch per worker. Their hashes match the qualified attempt-3
records. Median NinjaRS-first baseline and TIRust-first candidate times are
63.242 and 71.102 microseconds; the paired ratio is 1.121455, with a descriptive
95% interval [1.101518, 1.141752]. All numerical comparisons pass at `1e-8`.
These controls reproduce the slowdown under the current conditions. They are
diagnostic evidence; the separate qualified attempt-3 failure remains the
acceptance result. The raw statistics helper's `acceptance_eligible` field only
checks the pair count; the enclosing diagnostic record explicitly sets
`performance_eligible: false`.

```sh
python docs/benchmarkfolder/experiments/n1-profile/fortran/control_pair.py \
  --name original-controls-NEW --cpu 1 --loops 200 --pairs 7
python docs/benchmarkfolder/experiments/n1-profile/fortran/analyse.py
```

The `protocol` directory freezes the original worker Python/Fortran sources and
compiler metadata helper. The attribution matrix originally imported campaign
helpers from the working tree; its exact captured source is retained and
`analyse.py` verifies that the four used helper function ASTs are identical to
the frozen campaign. Future runs import that frozen protocol explicitly.
`profile-run-used.py` and `control-pair-used.py` preserve the exact historical
controllers; later runner maintenance does not alter the recorded experiment.

The next candidates are separate persistent primary/check workspaces in the
bridge, and measured reuse or cheaper construction of Fortran mapping and
kinematic inputs or automatic-array storage. Any check workspace must retain an independent evaluation
and must not attach primary-pass tensor results. No optimization is part of
these attribution results. The n1 adapter output slice is already contiguous
(`NC=LOOPMAXCOEFS=15`), so a general wrapper temporary warning does not establish
an n1 copy bottleneck.

The [evidence index](evidence-index.md) lists sources, raw measurements and
hash manifests. All native binaries, shared libraries and cloned processes
remain under ignored working directories.
