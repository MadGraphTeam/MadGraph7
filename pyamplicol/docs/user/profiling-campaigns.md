---
title: "Profiling Campaigns"
nav_order: 1
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Profiling Campaigns

The profiling-campaign command creates a complete, movable measurement
workspace from an installed pyAmpliCol wheel. It is intended for controlled
generation/runtime comparisons and for reproducing the report format used by
the published MacBook M3 and AMD EPYC PDFs.

For a physicist-oriented end-to-end explanation of steering, interruption,
retained evidence, PDF rendering, and every entry in a concrete “Best vs
AmpliCol” matrix cell, follow
[Profiling campaign: from measurements to the PDF](profiling-campaign-walkthrough.md).

> A campaign can create very large artifacts and long-running workers. Start
> with the single-cell smoke below. Broader multiplicities belong on a
> dedicated measurement host.

## Create a fresh campaign

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
cd ./pyamplicol-profiling-campaign
./steer_performance_campaign.py --help
```

The copy is self-contained apart from the installed pyAmpliCol runtime. It
contains the controller, report sources, blank/reset data, generated-table
templates, and one visible state root:

```text
pyamplicol-profiling-campaign/
  steer_performance_campaign.py
  README.md
  TABLE_FILLING.md
  report-workspace.json
  campaign_artifacts/
    cells/
    coordination/
    manual-reproductions/
    report-snapshot-builds/
  results/
  result_*.tex
  pyAmpliCol.tex
```

All attempts, prepared artifacts, current pointers, logs, locks, leases, and
reproduction state live below `campaign_artifacts/`. The controller does not
consult an old checkout-level `.artifacts` directory.

## A deliberately small real run

This command measures only the final-state-multiplicity-one `d d~ > Z`
recurrence cell:

```console
./steer_performance_campaign.py run \
  --workers 1 \
  --table matrix \
  --process-id 1 \
  --multiplicity 1 \
  --color-approximation lc \
  --generation-mode non-union-flow \
  --generation-engine recurrence \
  --model built_in
```

It is the recommended installed-wheel campaign smoke. Release wheels omit the
optional Ratatui bindings, so the controller continues headlessly. A
contributor checkout prepared with `just dev-install` can show the interactive
dashboard.

Preview a broader selection without launching workers:

```console
./steer_performance_campaign.py run \
  --dry-run \
  --table z_table \
  --multiplicity 3 4 \
  --model built_in sm_ufo
```

The dry run separates directly selected cells from dependency work and prints
a public-CLI reproduction recipe for each runnable cell.

## Selection and numerical-authority closure

The campaign catalog has stable canonical cell IDs. Ordinary selectors choose
direct work by table, process family, multiplicity, color approximation,
generation mode, engine, model, and variant.

By default, the controller recursively adds each selected cell's available
active-source numerical-authority chain:

```text
original AmpliCol  →  recurrence  →  compiled / eager
```

Added cells are dependency-only work. Independent process/multiplicity cohorts
remain parallel. Hard construction and selector/provider dependencies always
block until available. A terminal optional authority releases its candidate:
the candidate can still run, but a compiled/eager result without independent
authority is retained as an explicit unverified timing diagnostic rather than
published as a successful current.

Use `--no-dependencies-added` only when you intentionally want to suppress
this optional numerical-authority expansion. It never removes hard
dependencies.

## Original AmpliCol is optional

All pyAmpliCol backends work without an original-AmpliCol checkout. The legacy
backend is enabled only when selected as a reference.

The supported input is a **clean, complete checkout** exposing the profiling
probe sources and Make targets from
[rikkert-frederix/AmpliCol PR #12](https://github.com/rikkert-frederix/AmpliCol/pull/12).
It is not a single Fortran file and is not bundled in pyAmpliCol.

Record a destination-local default while copying:

```console
pyamplicol profiling-campaign copy ./campaign \
  --local-amplicol /path/to/clean/complete/AmpliCol-checkout
```

Or provide/override it for one run:

```console
./campaign/steer_performance_campaign.py run \
  --original-amplicol /path/to/clean/complete/AmpliCol-checkout \
  --generation-engine amplicol recurrence \
  --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --model built_in
```

The copy stores the chosen path in `.pyamplicol-original-amplicol`. A later
`--force` reset preserves that pointer unless `--local-amplicol` explicitly
replaces it. Without a configured checkout, a selection that does not include
`amplicol` remains fully supported.

## Reuse and source revisions

Successful compatible currents are reused by default. The campaign's stored
attempts are not erased merely because pyAmpliCol's source revision changes.

Without `--continue-across-revisions`, planning and report operations use the
active source cohort. With it, valid historical currents may remain in the
campaign while new cells and dependencies are measured at the active revision:

```console
./steer_performance_campaign.py run \
  --continue-across-revisions \
  --workers 4 \
  --table matrix \
  --multiplicity 4 5 \
  --generation-engine recurrence compiled eager \
  --model built_in
```

This option reconciles compatible prior results; it does not reconstruct a
deleted campaign. Historical numerical authorities needed by current-revision
work are replanned at the active revision rather than silently reused across an
unsafe authority boundary.

The operations that actually reset managed state are
`pyamplicol profiling-campaign copy DEST --force` and deliberate manual
deletion. Stop every active controller before using `--force`.

## Move or rename a campaign

Move the complete directory, not selected result files:

```console
mv ./campaign-mac ./archive/macbook-M3-2026
cd ./archive/macbook-M3-2026
./steer_performance_campaign.py inspect
```

Portable current locators are interpreted relative to the campaign's new
`campaign_artifacts/` root. Two campaign directories with the same basename in
different parents do not share state.

If a directory was renamed while a shell remained inside its old inode, start
a new shell or `cd` through the new absolute path before launching the copied
controller. The launcher checks logical and physical working directories and
fails instead of reading an unintended campaign.

## Dashboard and progress

Contributor environments with Ratatui bindings display active worker lanes,
direct/dependency work, completed/capped/error/unverified partitions, elapsed
time, and compact terminal reasons.

Capture a read-only view from another terminal:

```console
./steer_performance_campaign.py dashboard-snapshot --live \
  --width 160 --height 48
```

The live snapshot reads only compact coordination leases. Use
`--instance ID_OR_PREFIX` when more than one controller is active. Omit
`--live` for the deterministic layout fixture.

`Ctrl-C` or `Esc` stops dispatch, terminates supervised process trees,
preserves completed currents, writes the summary-ID files, and restores the
terminal.

## Retry exactly what needs attention

Every completed or orderly interrupted run replaces
`campaign_summary_ids/` with one sorted text file per non-success status, for
example:

```text
campaign_summary_ids/
  blocked_dependency.txt
  error.txt
  unverified.txt
  validation_failed.txt
  worker_timeout.txt
```

Replay exact IDs:

```console
./steer_performance_campaign.py run \
  --cell-id-file campaign_summary_ids/error.txt \
                 campaign_summary_ids/validation_failed.txt \
  --force-refresh
```

Or filter an ordinary selection by structured state:

```console
./steer_performance_campaign.py run \
  --table z_table \
  --generation-engine recurrence compiled \
  --rerun-failed --rerun-capped \
  --continue-across-revisions
```

- `--rerun-failed` selects latest terminal non-successes that are not
  authenticated policy caps; unverified diagnostics are retryable here.
- `--rerun-capped` selects authenticated memory/time/generation policy caps.
- Combining them takes the union.
- Successful, structural-static, and never-attempted cells are not direct
  retry targets.
- Normal hard and optional dependency closure still applies.
- The selective flags conflict with `--force-refresh`.
- No matches is a successful no-op and preserves the previous summary IDs.

Replaying `blocked_dependency.txt` includes its prerequisites automatically.
An `unverified.txt` replay does not require `--force-refresh`: unverified is
not an authoritative successful current.

## Attempt retention and cleanup

Compact retention is the default. It keeps sealed results and current pointers,
provenance, commands, progress/timeline evidence, bounded log tails, and outputs
needed by currents or dependencies. Disposable copied AmpliCol, MadGraph, and
build workspaces are removed after publication. Cross-revision reuse reads the
sealed results rather than those workspaces.

Pass `--retain-workspaces` only for a bounded debugging run that needs the full
workspaces:

```console
./steer_performance_campaign.py run \
  --retain-workspaces \
  --table matrix --process-id 1 --multiplicity 1 \
  --generation-engine recurrence --model built_in
```

`--cleanup-artifacts` remains accepted as a compatibility spelling for the
default compact policy. Independently, `--attempt-output-limit` defaults to
67,108,864 bytes (64 MiB) per watched output/log file and
`--minimum-free-disk` reserves 5,368,709,120 bytes (5 GiB) on the artifact
volume. These storage guards complement, but do not replace, process-tree RAM
supervision.

## Inspect and render

Inspect coverage and AmpliCol-relative comparisons:

```console
./steer_performance_campaign.py inspect \
  --color-approximation lc \
  --generation-engine recurrence compiled
```

Refresh all tables and atomically replace the PDF from one coherent current
snapshot:

```console
./steer_performance_campaign.py refresh-pdf
```

Interactive refresh displays a colored progress bar while current records are
read and checked. It does not rescan every historical heavy artifact.

List stable section IDs or omit sections from one PDF build:

```console
./steer_performance_campaign.py refresh-pdf --list-sections
./steer_performance_campaign.py refresh-pdf \
  --remove-sections worked-zgg shared-current-dag
```

Section omission changes only the staged PDF. Measurements, JSON caches, table
TeX, and source remain; the next plain refresh restores the complete report.
`--quiet` suppresses scan progress and live LaTeX output.

The repository retains only two rendered report snapshots:

- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)

Raw campaign results are intentionally not published in the repository. A
fresh copied campaign contains everything needed to reproduce the report
format directly.

## What `--force` resets

```console
pyamplicol profiling-campaign copy DEST --force
```

replaces only managed template files and resets:

- `DEST/campaign_artifacts/`;
- the managed `pyAmpliCol.pdf`;
- `campaign_summary_ids/`;
- measurement lineage;
- known pyAmpliCol LaTeX build byproducts.

It preserves unrelated destination files, a legacy `DEST/.artifacts`
directory (which is ignored), and the recorded original-AmpliCol pointer unless
explicitly replaced. An active campaign makes reset fail without removing
state.

## Recommended operating habits

1. Use one copied directory per host and campaign policy.
2. Run `--dry-run` before a broad selection.
3. Set worker and RAM limits for the actual host.
4. Use `--continue-across-revisions` when intentionally extending a campaign
   across source updates.
5. Prefer selective retry flags over a broad force refresh.
6. Keep compact attempts by default; use `--retain-workspaces` only for a
   bounded debugging run.
7. Never commit `campaign_artifacts/`, raw results, logs, locks, attempts, or
   LaTeX auxiliary files.

## Related pages

- [Examples Gallery](examples-gallery.md) — ordinary generation/runtime examples.
- [Profiling campaign walkthrough](profiling-campaign-walkthrough.md) — the campaign idea, retained evidence, PDF lifecycle, and matrix interpretation.
- [Artifacts and Portability](artifacts-and-portability.md) — process-artifact portability and trust.
- [Troubleshooting](troubleshooting.md) — campaign state, source revision, and dashboard issues.
- [Packaged campaign guide](https://github.com/mg5amcnlo/pyamplicol/blob/main/src/pyamplicol/_profiling_campaign/TABLE_FILLING.md).
- [Published report index](../performance_reports/README.md).
