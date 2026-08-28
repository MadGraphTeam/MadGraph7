---
title: "FullColor FFT Profiling"
nav_order: 3
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# FullColor FFT Profiling

The source checkout contains a thin, resumable orchestrator for comparing
exact full-colour matrix-element implementations. It delegates physics
generation, profiling, validation, plotting, and PDF assembly to the existing
pyAmpliCol developer tools; the driver owns only selection, scheduling,
per-child resource guards, and persistent progress.

This workflow is distinct from selecting
`color.contraction = "symmetric-group-fft"` in an ordinary generated artifact:

- **symmetric-group FFT contraction** is a pyAmpliCol runtime algorithm for
  exact NLC/full colour in recurrence and on-the-fly execution;
- **Reference FFT** is the optional external `AllGluonsMultipletFFT` comparison
  checkout;
- **the profiling orchestrator** measures those implementations alongside
  direct pyAmpliCol, original AmpliCol, and MadGraph where applicable.

The orchestrator is source-checkout-only. It is intended for a workstation or
cluster node with explicit CPU, memory, and wall-time budgets.

## Prepare a profiling checkout

Enter the repository's Nix environment when available, then request only the
optional references needed by the scan:

```console
nix develop
just dev-install --with-reference-fft --with-legacy-amplicol
```

`--with-reference-fft` clones the public
`rikkert-frederix/AllGluonsMultipletFFT` revision pinned by pyAmpliCol.
`--with-legacy-amplicol` clones the pinned original-AmpliCol comparison.
Both are omitted by a default `just dev-install`; either opt-in also installs
the PDF/plotting dependencies into `.venv`. If a managed Reference FFT checkout
predates the public repository, add `--update` once so its `origin` is migrated.

MadGraph is not installed by `dev-install`. Pass a complete installation root
containing `bin/mg5_aMC` and `VERSION` with `--madgraph-root`.

## Select process families and curves

The scan has two independent process families:

| Selector | Process family |
| --- | --- |
| `gg` | `g g > g g + gluons` |
| `ddbar` | `d d~ > d d~ + gluons` |

`--lines` accepts grouped comparison series:

| Selector | Curves requested |
| --- | --- |
| `reference-fft` | External pure-gluon Reference FFT |
| `amplicol` | Original AmpliCol |
| `pyamplicol-recurrence` | pyAmpliCol recurrence direct and FFT |
| `pyamplicol-otf` | pyAmpliCol on-the-fly direct and FFT |
| `madgraph` | Same-host MadGraph standalone |

For a targeted top-up, use a concrete pyAmpliCol line:
`recurrence-direct`, `recurrence-fft`, `otf-direct`, or `otf-fft`.
Required authority/baseline shards are added automatically. For example, a
pyAmpliCol pure-gluon comparison may require Reference FFT, while a `ddbar`
comparison may require original AmpliCol.

## Small Reference FFT example

This fills only Reference FFT for the pure-gluon family at final-state
multiplicities 2, 3, and 4:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --output /shared/path/fft-reference-gg \
  --families gg --lines reference-fft \
  --multiplicities 2 3 4 \
  --cores 10 \
  --memory-limit-gib 100 --time-limit-seconds 36000 \
  --reference-fft-root "$PWD/dependencies/checkouts/reference-fft"
```

Use `--dry-run` first to print the deterministic plan without creating the
output or starting workers.

## Complete cluster scan

The following command requests both process families, every grouped series,
ten scheduler cores, a 100 GiB per-child memory guard, and a 10-hour per-cell
time limit in one explicit output directory:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --output /shared/path/pyamplicol-fft-profile \
  --families gg ddbar \
  --lines reference-fft amplicol pyamplicol-recurrence pyamplicol-otf madgraph \
  --multiplicities 2 3 4 5 6 7 8 9 \
  --cores 10 --candidate-cores 1 \
  --memory-limit-gib 100 --time-limit-seconds 36000 \
  --amplicol-root "$PWD/dependencies/checkouts/legacy-amplicol" \
  --reference-fft-root "$PWD/dependencies/checkouts/reference-fft" \
  --madgraph-root /shared/path/MG5_aMC \
  --build-amplicol
```

`--build-amplicol` permits the first fixed-helicity AmpliCol shard to build its
probe. It can remain on subsequent resumptions; an already valid probe is
reused.

Resource options have deliberately narrow meanings:

- `--cores` is the total scheduler slot budget across active children;
- `--candidate-cores` is one pyAmpliCol child's slot claim and exact
  `evaluator.optimization.cores` setting;
- `--memory-limit-gib` and `--time-limit-seconds` are strict per-child/per-cell
  limits, not aggregate campaign limits;
- the time limit covers generation/setup and runtime measurement;
- `--target-seconds` controls calibrated runtime sampling after setup and is
  not the cell cutoff.

The progress headline reports the active cell, aggregate live RSS, and occupied
core slots. A child crossing its own guard is terminated and retained as a
cutoff result; other cells continue.

## Fixed helicity and complete helicity sums

The default workload evaluates one shared known-nonzero helicity from artifacts
that retain general helicity coverage. Add `--compare-helicity-sums` for the
independent complete physical-helicity sum:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --output /shared/path/pyamplicol-fft-profile-helicity-sum \
  --compare-helicity-sums \
  --families gg ddbar \
  --lines reference-fft amplicol pyamplicol-recurrence pyamplicol-otf madgraph \
  --multiplicities 2 3 4 5 6 \
  --cores 10 --candidate-cores 1 \
  --memory-limit-gib 100 --time-limit-seconds 36000 \
  --amplicol-root "$PWD/dependencies/checkouts/legacy-amplicol" \
  --reference-fft-root "$PWD/dependencies/checkouts/reference-fft" \
  --madgraph-root /shared/path/MG5_aMC
```

Fixed-helicity and helicity-sum measurements have different workload
identities and must use separate output directories. MadGraph's summed lane
uses its generated `SMATRIX(P,ANS)` entry point with `USERHEL=-1`; it does not
reuse fixed-helicity measurements.

## Resume, retry, overwrite, or refresh

The default behavior is resumable and additive. Repeating a command with the
same `--output`:

- recovers unfinished prior requests;
- skips completed cells;
- adds newly selected families, lines, and multiplicities to the persistent
  history;
- refreshes plots as new completed evidence becomes available.

`--resume` is an explicit alias for that default. More forceful choices are
scoped to the current `--families`, `--lines`, and `--multiplicities`:

| Option | Effect |
| --- | --- |
| `--retry` | Rerun selected failed or skipped cells. |
| `--overwrite` | Rerun every selected cell; keep its old result until that worker actually starts. |
| `--refresh` | Delete the exact recognized output workspace and restart it from scratch. |

Use targeted overwrite to refine a measured subset without discarding the
rest of the campaign:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --output /shared/path/pyamplicol-fft-profile \
  --families ddbar --lines otf-fft \
  --multiplicities 6 --overwrite \
  --cores 10 --memory-limit-gib 100 --time-limit-seconds 36000
```

`--refresh` is intentionally conservative: it refuses an unrecognized path and
will not remove a workspace while a coordinated MadGraph writer is active.

Inspect machine-readable progress without launching work:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --status --output /shared/path/pyamplicol-fft-profile
```

## Render what is currently available

The driver publishes immutable snapshots during a scan. Rendering freezes the
latest already-published snapshot immediately and never waits for active
workers:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --render --output /shared/path/pyamplicol-fft-profile
```

For a custom output, the saved manifest determines whether the workload is
fixed-helicity or helicity-summed; repeating `--compare-helicity-sums` is not
required. The current PDF is written below:

```text
RUN/render/current/summary_plots_final.pdf
RUN/render/current/summary_plots_final_helicity_sum.pdf
```

Only the filename matching the run's workload is produced. The same render
contains six PNGs: setup time, warmed runtime, and peak RSS for each process
family.

Without a custom output, select the two canonical workspaces explicitly:

```console
.venv/bin/python tools/fft_profiling/fft_profiling.py --render
.venv/bin/python tools/fft_profiling/fft_profiling.py \
  --render --compare-helicity-sums
```

Generation selectors (`--lines`) decide what work enters the campaign. The
render-only `--main-include-lines`, `--main-veto-lines`,
`--ratio-include-lines`, and `--ratio-veto-lines` options only control which
retained series appear in the main and ratio panels.

## Interpret setup and warmed runtime

The plotted *setup time* is method-specific:

- pyAmpliCol includes artifact generation, fresh load, first requested
  evaluation, and OTF family construction where applicable;
- Reference FFT includes its build, initialization, and first pass;
- AmpliCol includes its process/colour-object or raw-library preparation for
  the selected workload.

Warmed runtime is measured separately after those setup boundaries. The exact
symmetric-group FFT is not guaranteed to beat direct contraction for every
process: small certified permutation subgroups or large direct residuals can
produce similar scaling.

Published rolling snapshots:

- [selected-helicity FullColor FFT PDF](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final.pdf)
- [complete-helicity-sum FullColor FFT PDF](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/summary_plots_final_helicity_sum.pdf)

These are measurement-host results rather than release-CI benchmarks. Raw
campaign workspaces remain local.

## See also

- [Configuration](configuration.md#color-accuracy-and-lc-layout) — enable exact FFT contraction in a generated artifact.
- [Command-Line Interface](command-line-interface.md#generate) — direct generation example.
- [Performance Reports](../performance_reports/README.md) — published frontiers and measurement boundaries.
- [Installation](installation.md#contributor-installation) — optional profiling references.
