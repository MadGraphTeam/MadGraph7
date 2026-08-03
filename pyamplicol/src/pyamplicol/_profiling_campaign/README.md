<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol profiling campaign template

This packaged directory is the authoritative reset template for a
self-contained profiling campaign. Create it in a destination whose name
identifies its measurement environment:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign
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
  --dry-run --table scalar_contact --multiplicity 2 \
  --generation-engine compiled --no-dashboard
```

`--force` overwrites the managed template files and resets only this copy's
`campaign_artifacts/`, `pyAmpliCol.pdf`, `campaign_summary_ids/`, and known
measurement-lineage/LaTeX build byproducts. It keeps unrelated files and a
previously recorded `.pyamplicol-original-amplicol` default unless a new
`--local-amplicol PATH` is supplied. Stop an active campaign before resetting
its destination.

Release wheels do not include the optional `ratatui` and `ratatui_py`
dashboard bindings. Installed campaign runs therefore continue headlessly as
if `--no-dashboard` were supplied. `dashboard-snapshot` is available when
those bindings are present, including in a contributor checkout prepared with
`just dev-install`; otherwise it exits with the corresponding instruction.

Dry runs and explicit pyAmpliCol-only engine selections need no legacy source.
Recurrence, compiled, and eager cells remain runnable when an original-AmpliCol
comparison is absent or terminal; their report cells show absolute timings and
omit the unavailable multiplier. Use, for example,
`run --generation-engine recurrence compiled eager`. Omitted engine selection
and quoted `*` mean every engine, so a broad/default selection includes
original AmpliCol unless another selector excludes it. When `amplicol` is
selected, pass `run --original-amplicol PATH`, where `PATH` is a clean,
complete checkout exposing the color-probe sources and Make targets from PR
#12. The `amplicol_with_patches` branch works now; a compatible upstream
revision will work unchanged after the PR is merged.
Alternatively, add `--local-amplicol PATH` to the copy command to record that
checkout as this campaign's default; a later `run --original-amplicol PATH`
overrides it.

Rebuild every table and the PDF from one stable current-result snapshot with:

```console
./steer_performance_campaign.py refresh-pdf
```

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
they are rerun and validated automatically once recurrence or AmpliCol
authority is available.

By default, every heavy attempt payload is retained. Pass
`run --cleanup-artifacts` to archive obsolete sealed attempts, retain their
compact result, log, progress, and timeline diagnostics, and remove only their
heavy evaluator payloads. Current artifacts and artifacts borrowed by an
equivalent current remain protected.

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
