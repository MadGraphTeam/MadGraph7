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
meaning. `LoopPoleMode=0` also avoids profiling defaults that can disable
COLLIER poles when comparing all five backends.

Numerical caches clear at MadLoop's phase-space boundary and reuse integrals
across helicities and split orders. Full and partial results and DP/QP stores
remain separate. Independent stability checks bypass cached numerical results.
Numerical failures propagate to MadLoop's existing rescue logic.

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
