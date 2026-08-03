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
  run --dry-run --table scalar_contact --multiplicity 2 \
  --generation-engine compiled --no-dashboard
```

Release wheels omit the optional `ratatui` and `ratatui_py` dashboard
bindings. Installed runs automatically continue headlessly as though
`--no-dashboard` had been supplied. `dashboard-snapshot` remains available
when those bindings are present, notably in a contributor checkout prepared
with `just dev-install`; without them it exits with an actionable instruction.

PyAmpliCol-only dry runs and measurements do not need a legacy checkout.
Recurrence, compiled, and eager cells run independently when their original-
AmpliCol comparison is absent or terminal; the tables show their absolute
timings without an unavailable multiplier. Select that mode explicitly with
`--generation-engine recurrence compiled eager`. Omitted engine selection and
quoted `*` mean every engine, so a broad/default selection includes original
AmpliCol unless another selector excludes it. A selection that includes
`amplicol` requires `run --original-amplicol PATH`; `PATH` must be a clean,
complete checkout exposing the PR #12 probe sources and Make targets.
The `amplicol_with_patches` branch works now, and a compatible upstream
revision will work unchanged after the PR is merged.
Pass `--local-amplicol PATH` to the initial `profiling-campaign copy` command
to store that checkout as the copied campaign's default; an explicit
`run --original-amplicol PATH` still overrides it.

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

Run a bounded multicore selection:

```console
./steer_performance_campaign.py run \
  --workers 4 --cores-per-worker 2 --table z_table --multiplicity 3
```

Defaults are one worker, one core, a one-hour generation/preparation limit, and
a decimal 30 GB process-tree RAM limit. Press `Ctrl-C` or `Esc` to stop
dispatch, terminate supervised process trees, preserve completed currents, and
restore the terminal.

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
comments are ignored. Capped and failed currents are otherwise reused under
the ordinary rules, so use `--force-refresh` when the intent is a new attempt;
replaying `blocked_dependency.txt` automatically includes required
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

Human output is coloured by default. Use `--no-color` or `NO_COLOR`; JSON is
always uncoloured. Consult each subcommand’s `--help` for all selectors,
aliases, resource controls, profiling parameters, reuse rules, and examples.
