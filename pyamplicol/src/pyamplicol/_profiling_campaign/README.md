<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol profiling campaign template

This packaged directory is the authoritative reset template for a
self-contained profiling campaign. Create it in a destination whose name
identifies its measurement environment:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign \
  --local-madgraph /path/to/MG5_aMC
cd ./my-profiling-campaign
```

The copy contains the report's LaTeX sources, reset raw JSON measurements, and
generated table TeX. Its visible `campaign_artifacts/` directory contains all
evaluator attempts, prepared artifacts, worker logs, locks, leases, and other
runtime state. Moving or renaming the whole campaign therefore moves its state;
no repository-level or legacy `.artifacts` directory is consulted.

Use the executable manual controller described in `TABLE_FILLING.md`. Its
`--help` output is the authoritative selector, resource-limit, reuse, and
keyboard-control reference:

```console
./steer_performance_campaign.py --help
```

An installed pyAmpliCol wheel can create the portable reset copy without a
source checkout:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
./pyamplicol-profiling-campaign/steer_performance_campaign.py --help
./pyamplicol-profiling-campaign/steer_performance_campaign.py run \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
```

The final command is a real, deliberately small installation smoke test. It
measures only `d d~ > Z`; broader multiplicities belong on a dedicated
profiling host.

`--force` overwrites the managed template files and resets only this copy's
`campaign_artifacts/`, `pyAmpliCol.pdf`, `campaign_summary_ids/`, and known
measurement-lineage/LaTeX build byproducts. It keeps unrelated files and a
previously recorded `.pyamplicol-original-amplicol` default unless a new
`--local-amplicol PATH` is supplied. It likewise preserves the recorded
`.pyamplicol-madgraph` default unless `--local-madgraph PATH` is supplied.
Stop an active campaign before resetting its destination.

Release wheels do not include the optional `ratatui` and `ratatui_py`
dashboard bindings. Installed campaign runs therefore continue headlessly as
if `--no-dashboard` were supplied. `dashboard-snapshot` is available when
those bindings are present, including in a contributor checkout prepared with
`just dev-install`; otherwise it exits with the corresponding instruction.

By default, a run adds the available numerical-authority closure for every
selected cell at the active source revision. MadGraph standalone is the
authoritative full-colour boundary for the four UFO-SM comparison views;
recurrence is the independent authority for compiled/eager comparisons on the
other surfaces. Original AmpliCol is retained only as a legacy performance and
numerical diagnostic and never certifies or rejects a candidate. Independent
processes remain parallel, while a terminal required authority releases its
candidate to run unverified. Use
`--no-dependencies-added` to suppress optional authority expansion; hard
construction and selector/provider dependencies are always retained. A
selection that directly or automatically includes `amplicol` requires a clean,
complete checkout exposing the color-probe sources and Make targets from PR
#12. Supply it with `run --original-amplicol PATH`, or add
`--local-amplicol PATH` to the copy command to record that checkout as this
campaign's default; the run option overrides the saved default. The
`amplicol_with_patches` branch works now; a compatible upstream revision will
work unchanged after the PR is merged.

The full-colour MadGraph views require an installation with executable
`bin/mg5_aMC` and its standard `models/sm` UFO model. Supply it once with
`profiling-campaign copy --local-madgraph PATH`, or override the saved value
with `run --madgraph PATH`. Each authority cell streams `generate`, `output
standalone`, and `launch -f` through a MadGraph command card, counts the
standalone compilation in generation time, then compiles a small campaign-owned
Fortran driver to evaluate and benchmark the generated matrix element. The
same deterministic point is evaluated by pyAmpliCol at precision 200 for
recurrence, compiled, and eager, and at the supported precision 16 for OTF; all
four candidates must agree within relative tolerance `1e-10`.

For on-the-fly LC cells in the n<=4 matrix, the matching recurrence cell is the
single numerical correctness authority. No compiled artifact is generated or
loaded for that gate. Original AmpliCol remains the legacy performance
baseline used by the OTF-versus-AmpliCol tables. OTF NLC and full colour use a
selector-free contracted workload and require the authenticated contracted-
colour runtime capability; the UFO-SM full-colour cells compare directly with
MadGraph. This contract is report-cache schema v5.

With `run --fail-fast`, dispatch proceeds in multiplicity waves (`n=1`, then
`n=2`, and so on) and stops at the first required mismatch. A failed legacy
AmpliCol comparison is deliberately non-terminal: the existing table keeps all
other entries unchanged and renders only the AmpliCol generation time in red.

Rebuild every table and the PDF from one stable current-result snapshot with:

```console
./steer_performance_campaign.py refresh-pdf
```

The refresh shows a coloured progress bar while it reads and confirms the
campaign's current records. List the stable top-level PDF section IDs, or omit
selected sections from one generated PDF, with:

```console
./steer_performance_campaign.py refresh-pdf --list-sections
./steer_performance_campaign.py refresh-pdf \
  --remove-sections worked-zgg shared-current-dag
```

Section removal changes only that staged PDF build. It does not remove
measurements, result JSON, generated table TeX, or source material; the next
plain `refresh-pdf` restores the complete report. `--quiet` suppresses both the
scan progress display and live LaTeX output.

Review the deterministic dashboard fixture, or capture the newest running
campaign's read-only informational lease, with:

```console
./steer_performance_campaign.py dashboard-snapshot
./steer_performance_campaign.py dashboard-snapshot --live
```

The live form reads only compact lease JSON in the ignored coordination
directory. Use `--instance ID_OR_PREFIX` to choose among concurrent runs; see
`dashboard-snapshot --help` for staleness and styled-cell capture controls.

Every completed or orderly interrupted `run` atomically replaces
`campaign_summary_ids/` with one text file per non-success status. Each file
contains exact canonical cell IDs and can be fed straight back to `run` or
`inspect`:

```console
./steer_performance_campaign.py run \
  --cell-id-file campaign_summary_ids/error.txt \
                 campaign_summary_ids/validation_failed.txt \
  --force-refresh
```

`campaign_summary_ids/unverified.txt` is directly replayable without
`--force-refresh`: those timing diagnostics are not successful currents, and
they are rerun and validated automatically once the required recurrence or
MadGraph authority is available.

For state-based retries, keep the ordinary table/process/engine selectors and
add `--rerun-failed`, `--rerun-capped`, or both. The flags filter only that
direct selection: failed means its latest terminal non-success is not an
authenticated policy cap, while capped means an authenticated policy cap. The
combined flags take the union; successful, static, and never-attempted cells
are excluded. Unverified results remain automatically retryable and can be
selected as failed. Normal dependency closure and reuse of successful currents
still apply, and `--continue-across-revisions` controls whether historical
terminal state participates. The retry flags conflict with `--force-refresh`.
A retry filter with no matches exits successfully without replacing the
existing campaign summary.

The default lifecycle is compact. Each completed attempt retains its sealed
result and current pointer, provenance, commands, progress/timeline evidence,
bounded log tails, and reusable outputs needed by currents or dependencies.
Disposable copied AmpliCol, MadGraph, and build workspaces are removed after
publication; cross-revision reuse relies on the sealed results, not those
workspaces. Pass `run --retain-workspaces` only when full debug workspaces are
needed. `--cleanup-artifacts` remains a compatibility spelling for the default
compact policy.

Two storage guards apply independently of process-tree memory supervision:
`--attempt-output-limit` defaults to 67,108,864 bytes (64 MiB) per watched
output/log file, and `--minimum-free-disk` defaults to 5,368,709,120 bytes
(5 GiB) on the campaign artifact volume. Both options accept byte counts.

Create independent campaigns with separate `profiling-campaign copy` commands.
Every result artifact, worker lease, lock, and reproduction file remains below
that copy's `campaign_artifacts/`, even when two destinations share a basename.
A copy made from an installed wheel uses that wheel's recorded source revision
and installed runtime. Never commit `campaign_artifacts/`, evaluator artifacts,
candidate wheels, prepared models, attempts, logs, locks, coordination state,
page images, or LaTeX auxiliary files.

When the launcher runs from a source checkout, it checks the checkout index
and worktree before measuring. It reads ordinary Git metadata directly and,
if HEAD is packed, uses one read-only `git rev-parse` query for the committed
tree. Installed-wheel campaigns do not need a Git checkout.
