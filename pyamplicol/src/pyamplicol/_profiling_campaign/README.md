<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol profiling campaign template

This packaged directory is the authoritative reset template for a
self-contained profiling campaign. Create a working copy with a name that
identifies its measurement environment:

```console
pyamplicol profiling-campaign copy ./my-profiling-campaign
cd ./my-profiling-campaign
```

The copy contains the report's LaTeX sources, reset raw JSON measurements, and
generated table TeX. Large evaluator artifacts, worker logs, locks, and
coordination state are deliberately stored outside the publication directory.

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

Release wheels do not include the optional `ratatui` and `ratatui_py`
dashboard bindings. Installed campaign runs therefore continue headlessly as
if `--no-dashboard` were supplied. `dashboard-snapshot` is available when
those bindings are present, including in a contributor checkout prepared with
`just dev-install`; otherwise it exits with the corresponding instruction.

Dry runs and measurement selections whose planned closure contains only
pyAmpliCol need no legacy source. If the planned closure includes an original
AmpliCol comparison, pass `run --original-amplicol PATH`, where `PATH` is a
clean, complete checkout exposing the color-probe sources and Make targets
from PR #12. The `amplicol_with_patches` branch works now; a compatible
upstream revision will work unchanged after the PR is merged.
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

Create separately named campaigns with separate `profiling-campaign copy`
commands. The steering entry point derives the profile name from its containing
directory, so result artifacts, worker leases, locks, reproduction files, and
PDFs use independent roots. A copy made from an installed wheel uses that
wheel's recorded source revision and installed runtime. Never commit evaluator
artifacts, candidate wheels, prepared models, attempts, logs, locks,
coordination state, page images, or LaTeX auxiliary files.
