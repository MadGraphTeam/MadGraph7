# TIRust in MadLoop

TIRust is the explicitly selected reduction backend **9**. Its Fortran adapter
uses oneloopmaster scalar integrals and supports double precision, genuine
quadruple-precision inputs, integral caching and separate UV/IR diagnostics.
Installation preserves MadLoop's automatic backend defaults.

## Install and select

Start `./bin/madgraph` from the MadGraph checkout and enter:

```text
install tirust --jobs=4
import model loop_sm
generate u u~ > z [virt=QCD]
output standalone tirust_z
```

The qualified development environment uses GNU Fortran 15.2 on Linux and Rust
1.97.1. The installer requires Cargo/Rust, GNU make, Git, patch and a Fortran
compiler supporting `REAL(16)`. It fetches pinned dependencies, stages its own
source copies, builds the Fortran module and shared Rust library, and runs DP,
QP and pole-only smoke tests before publishing the installation and saving its
configuration. A failed build or configuration update preserves the previous
installation. Failure logs remain in the staging directory named by the error.

Optional installation forms are:

```text
install tirust --prefix=/path/to/HEPTools --jobs=4
install tirust --update --source=/path/to/TIRust --oneloop-source=/path/to/oneloopmaster
install tirust --update
```

`--update` rebuilds using the pins and patch shipped with the current MadGraph
checkout; it does not follow an upstream development branch. `--force` is an
alias. Local overrides are copied before patching; the TIRust files touched by
the patch must match its reviewed inputs. Rebuilding an existing installation
with source overrides requires `--update`. Installed metadata records source, lockfile, patch and
native-artifact hashes. See [the bridge contract](../vendor/tirust/README.md)
for ABI and dependency details.

In the exported process's `Cards/MadLoopParams.dat`, replace the value after
`#MLReductionLib` with:

```text
#MLReductionLib
9
```

Alternatively, `9|8|6|7|1` tries TIRust first with the named available rescue
backends. Backend numbers are CutTools 1, Ninja 6, COLLIER 7 and NinjaRS 8.
Install libraries before exporting a process so its interface and link rules
include them. Re-export an older process to add a newly installed backend.

## Controls

Values appear on the line immediately following each `#Parameter` name.

| Parameter | Default | Meaning |
|---|---:|---|
| `LoopPoleMode` | `-1` | Legacy pole behavior; TIRust includes UV and IR. `0` explicitly includes both, `1` selects UV, `2` selects IR, and `3` omits poles. The finite term remains evaluated. |
| `TIRustComputeFinite` | `.TRUE.` | Complete tensor evaluation. `.FALSE.` is reserved for the separate partial-result diagnostic. |
| `UseQPIntegrandForTIRust` | `.TRUE.` | Recompute the numerator in QP when the reduction uses QP. Turning this off retains a DP numerator even though reduction arithmetic is QP. |
| `TIRustGlobalCache` | `-1` | Cache every supported topology. `0` disables caching; a positive value limits the number of propagators cached. |
| `TIRustCacheMaxBytes` | `268435456` | Numerical-cache budget per worker. Once full, caches stop inserting entries; numerical results are unchanged. Plans and workspaces are outside this budget. |

UV-only, IR-only and no-pole modes require every selected rescue backend to
support pole separation: use only 9, 8 and/or 7. These controls select residues
of the bare loop integral; counterterms and R2 contributions keep their existing
meaning. Explicit pole mode `0` includes both UV and IR poles; its profiling
filter policy is described below.

Numerical caches clear at MadLoop's phase-space boundary and reuse integrals
across helicities and split orders. Full and partial results and DP/QP stores
remain separate. Independent stability checks never read the primary pass's
cached results: they share sub-integrals only with each other. Exact repeats of
a tensor request within a point (one per helicity while MadLoop learns its
filters) return the stored tensor.
Numerical failures propagate to MadLoop's existing rescue logic.

## Profile a backend

After installing the comparison backends, use MadGraph's actual `check profile`
command. For an optimized GNU Fortran build, launch it from the checkout root
with the flags in the environment **before exporting or compiling**:

```bash
FFLAGS='-w -fPIC -O3' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B bin/madgraph
```

Generated `Source/make_opts` assigns its default `FFLAGS` only when that variable
is undefined. The environment therefore supplies `-O3` to generated MODEL,
DHELAS, process routines and profiling drivers, while makefile include and
fixed-line-length flags are still appended. Preserve the actual compiler
commands and verify those four groups used `-O3`; the polynomial routine's
separate `-O3` rule alone does not establish an optimized full build.

These examples generate fresh profiling outputs and use the standard learned
filters with legacy `LoopPoleMode=-1`:

```text
set nb_core 1
import model loop_sm
check profile 10 u u~ > z g g g [virt=QCD] --reduction=8 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112
check profile 10 u u~ > z g g g [virt=QCD] --reduction=9 --loop_pole_mode=-1 --split_orders=-1 --energy=1000 --tirust_qp_integrand=true --tirust_cache=-1 --seed=20261112
```

Run one command per timed backend; a reduction list times only its first backend,
while the separate stability analysis can compare the list. `set nb_core 1`
keeps compilation serial. Changing `FFLAGS` does not invalidate existing object
files: use a fresh output as above, or rebuild MODEL, DHELAS, every process object
and the driver before adding `-reuse`. Reusing an old output can otherwise keep
unoptimized objects even when the current environment says `-O3`.

Here `10` controls the separate stability sample, not the number of timing
points. Legacy mode retains `HelicityFilterLevel=2` and learns the normal
helicity and loop filters. Read the **full helicity-sum time**, which includes
the physical sum with learned zero/equivalent configurations skipped; the
single-helicity time is a separate quantity. `FULL−SKIP` estimates reduction
time by subtracting a separate numerator-only run; it is not a directly timed
reducer call. Reused-export initialization may be unavailable rather than zero.

`check profile` preserves COLLIER's default UV/IR pole flags, both true. The
separate `check timing` wrapper disables those flags by default unless supplied;
retain the effective cards rather than assuming the two wrappers agree.
Explicit `--loop_pole_mode=0` remains supported for including both poles, but it
disables learned filters and evaluates every helicity and loop group. Its
workload differs from the legacy filtered profile; a configured single helicity
can contribute zero and is labeled unverified. No timing result is implied by
this reproduction example.

The stock profiler uses fixed-DP MadLoop timing mode and a separate stability
probe. TIRust can still retry individual DP-input tensors internally at
106-bit DD, 256 or 512 bits. Accepted code `219` describes DP transport, not
entirely native-DP arithmetic. Genuine QP-input rescue remains distinct and
uses code `323`; a QP numerator also requires the effective
`UseQPIntegrandForTIRust` setting. Optional bridge route counters do not observe
Fortran coefficient-cache hits. The stock profile driver does not emit those
counters, so its timed internal routes remain unknown.

Use the [current local validation checklist](benchmarkfolder/LOCAL_FINAL_VALIDATION_TODO.md)
for paired warmed measurements with natural stability/rescue behavior and the
precise per-process acceptance gates. The [benchmark report](benchmarkfolder/benchmark.pdf)
keeps those measurements separate from stock profile timings.

## Partial pole diagnostics

The partial evaluator skips finite scalar kernels and tensor arithmetic while
retaining UV single, IR single and IR double poles, including scaleless
residues. Supported requests have one through six propagators, ranks zero
through six and no dotted propagators. Unsupported requests fail explicitly.

Use the [standalone pole diagnostic](benchmarkfolder/POLES.md) with
`TIRustComputeFinite=.FALSE.`, backend 9 and at least one pole selected. It
marks the finite coefficient as unevaluated (`NaN`). Ordinary amplitude and
Monte Carlo entrypoints reject partial requests; their outputs require a
complete finite coefficient. Skipping finite work does not imply a speedup
for every topology.

## Qualification and reproduction

The [benchmark guide](benchmarkfolder/README.md) provides the pinned models,
saved points, build commands and acceptance protocol. The [report](benchmarkfolder/benchmark.pdf)
distinguishes completed numerical checks, failed or unqualified stress cases,
and pending runtime measurements. Acceptance remains in progress; the preserved
initial candidate has a qualified runtime regression in the smallest process.

The common benchmark uses `MLStabThres=1e-6` for every reducer and both revisions;
the production default remains `1e-3`. Independent numerical comparisons and QP
adjudication supplement each backend's own error estimate. TIRust QP transports
Fortran binary128 inputs without a DP conversion, evaluates at 256-bit MPFR
precision and contracts with QP numerator coefficients when enabled. Higher
working precision alone is not a guarantee of accuracy at singular kinematics.
