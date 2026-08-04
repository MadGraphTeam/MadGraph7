<!-- SPDX-License-Identifier: 0BSD -->
# Profiling campaign table filling

First create and enter a reset campaign. The executable controller in that
copy is the authoritative campaign interface:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign
cd ./my-profiling-campaign
./steer_performance_campaign.py --help
```

For an installed release wheel, first copy the reset campaign into a new or
empty destination and invoke the controller there:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
./pyamplicol-profiling-campaign/steer_performance_campaign.py \
  run --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
```

This real smoke campaign selects only the final-state-multiplicity-one
`d d~ > Z` recurrence cell and is suitable for checking an installation.

Release wheels omit the optional `ratatui` and `ratatui_py` dashboard
bindings. Installed runs automatically continue headlessly as though
`--no-dashboard` had been supplied. `dashboard-snapshot` remains available
when those bindings are present, notably in a contributor checkout prepared
with `just dev-install`; without them it exits with an actionable instruction.

By default, selected cells gain their available active-source numerical
authority closure, ordered original AmpliCol, recurrence, then compiled/eager.
Added cells are dependency-only work; independent processes remain parallel,
and a terminal authority releases its candidate to run unverified. Without a
configured legacy checkout, recurrence remains available as the authority for
compiled/eager work. Pass `--no-dependencies-added` to suppress optional
authority expansion while retaining every hard construction and
selector/provider dependency. A selection that directly or automatically
includes `amplicol` requires a clean, complete checkout exposing the PR #12
probe sources and Make targets. Supply it with
`run --original-amplicol PATH`, or pass `--local-amplicol PATH` to the initial
`profiling-campaign copy` command to save it as the campaign default; the run
option overrides the saved default. The `amplicol_with_patches` branch works
now, and a compatible upstream revision will work unchanged after the PR is
merged.

Inside a contributor checkout it re-executes with the repository `.venv`; an
installed copy uses the wheel's Python runtime. Both modes reuse compatible
same-source currents by default, supervise each process tree, and keep attempts
and large artifacts in this campaign's visible `campaign_artifacts/` directory.
That state moves with the entire campaign and never falls back to an old
repository-level `.artifacts` directory. Use separate copied directories for
independent campaigns, even when their basenames happen to match.

Preview selections before launching workers:

```console
./steer_performance_campaign.py run \
  --dry-run --table z_table --multiplicity 3 4 \
  --model built_in sm_ufo
```

The preview lists direct cells and dependency work separately and prints a
wrapped public-CLI recipe for every selected runnable cell, even for broad
selections. Each recipe uses the repository `.venv/bin/pyamplicol` and states
whether it is exact, a template, or a diagnostic protocol exception. UFO
recurrence/eager recipes include their public `model compile` prerequisite.

The same small installation campaign, when run from inside its directory, is:

```console
./steer_performance_campaign.py run \
  --workers 1 --table matrix --process-id 1 --multiplicity 1 \
  --color-approximation lc --generation-mode non-union-flow \
  --generation-engine recurrence --model built_in \
  --no-dependencies-added --no-dashboard
```

Broader selections may raise `--workers` and `--cores-per-worker` on a dedicated
profiling host. Defaults are one worker, one core, a one-hour
generation/preparation limit, and a decimal 30 GB process-tree RAM limit.
Press `Ctrl-C` or `Esc` to stop dispatch, terminate supervised process trees,
preserve completed currents, and restore the terminal.

After normal completion or an orderly interrupt, the controller prints the
absolute path of `campaign_summary_ids/`. It replaces that directory with one
sorted text file per non-success status, for example `error.txt`,
`blocked_dependency.txt`, `validation_failed.txt`, or `worker_timeout.txt`.
Replay any union of those exact IDs while retaining the normal dependency
closure and other selector intersections:

```console
./steer_performance_campaign.py run \
  --cell-id-file campaign_summary_ids/error.txt \
                 campaign_summary_ids/validation_failed.txt \
  --force-refresh
```

`--cell-id-file` is also accepted by `inspect`. Blank lines and full-line `#`
comments are ignored. For a structured retry within the ordinary direct
selection, add `--rerun-failed`, `--rerun-capped`, or both:

```console
./steer_performance_campaign.py run \
  --table z_table --generation-engine recurrence compiled \
  --rerun-failed --rerun-capped
```

`--rerun-failed` keeps cells whose latest terminal outcome is unsuccessful but
is not an authenticated policy cap; `--rerun-capped` keeps authenticated policy
caps, and combining them takes the union. Successful, static, and unseen cells
are not direct retry targets. Unverified results are automatically retryable
and may be selected as failed. Dependency closure is still planned normally,
including reuse of successful prerequisite currents. Historical outcomes are
recognized only with `--continue-across-revisions`. These retry flags cannot be
combined with `--force-refresh`; use that option only for an unconditional
refresh. If no selected cell matches the requested retry state, the command is
a successful no-op and preserves the existing campaign summary.

Replaying `blocked_dependency.txt` automatically includes required
prerequisites.
`unverified.txt` needs no `--force-refresh`: an unverified timing diagnostic is
not a successful current and is automatically rerun against a later available
recurrence or AmpliCol authority.

The default lifecycle retains every heavy attempt payload. Use
`--cleanup-artifacts` to move obsolete sealed attempts into compact history,
retaining metadata, results, logs, progress events, and phase timelines while
removing only their heavy evaluator payloads. Every current artifact and any
artifact borrowed by an equivalent current remains protected.

Running `pyamplicol profiling-campaign copy DEST --force` again resets only
`DEST/campaign_artifacts`, the managed PDF, the campaign summary-ID directory,
known lineage/LaTeX build byproducts, and the packaged template files. It
preserves unrelated files and the recorded original-AmpliCol checkout unless a
new `--local-amplicol` value is supplied. Stop active campaign processes before
resetting their destination.

Capture a running dashboard from another terminal without attaching to or
changing the campaign:

```console
./steer_performance_campaign.py \
  dashboard-snapshot --live --width 160 --height 48
```

This reads only the compact non-stale coordination lease. The newest active
invocation is the default; pass `--instance ID_OR_PREFIX` to choose one
concurrent run. Omit `--live` for the deterministic synthetic layout fixture.

Inspect coverage and AmpliCol-relative statistics:

```console
./steer_performance_campaign.py inspect \
  --color-approximation lc --generation-engine recurrence compiled
```

Rebuild all current JSON/TeX tables and atomically replace the PDF:

```console
./steer_performance_campaign.py refresh-pdf
```

Interactive refreshes show a coloured progress bar while the controller reads
and confirms the complete current-record snapshot. To discover stable
top-level section IDs or build a shorter PDF without changing campaign data:

```console
./steer_performance_campaign.py refresh-pdf --list-sections
./steer_performance_campaign.py refresh-pdf \
  --remove-sections worked-zgg shared-current-dag
```

The omission applies only to that PDF build. Measurements, JSON caches, table
TeX, and the canonical source remain present; a later plain refresh restores
the full report. Use `--quiet` to suppress the scan progress display and live
LaTeX output.

Human output is coloured by default. Use `--no-color` or `NO_COLOR`; JSON is
always uncoloured. Consult each subcommand’s `--help` for all selectors,
aliases, resource controls, profiling parameters, reuse rules, and examples.
