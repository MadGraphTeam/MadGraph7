<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol performance report: `macbook_M3_manual`

This directory is the self-contained publication workspace for the
`macbook_M3_manual` measurement environment. It contains the report's LaTeX sources,
canonical raw JSON measurements, generated table TeX, and the PDF when it has
been compiled. Large evaluator artifacts, worker logs, locks, and coordination
state are deliberately stored outside this tracked directory.

Use the executable manual controller described in `TABLE_FILLING.md`. Its
`--help` output is the authoritative selector, resource-limit, reuse, and
keyboard-control reference:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py --help
```

An installed pyAmpliCol wheel can create a portable reset copy without a
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

Rebuild every table and the PDF from one stable current-result snapshot with:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py refresh-pdf
```

Review the deterministic dashboard fixture, or capture the newest running
campaign's read-only informational lease, with:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py dashboard-snapshot
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py dashboard-snapshot --live
```

The live form reads only compact lease JSON in the ignored coordination
directory. Use `--instance ID_OR_PREFIX` to choose among concurrent runs; see
`dashboard-snapshot --help` for staleness and styled-cell capture controls.

To create a separately named manual campaign inside the same checkout, copy
this directory to another single directory under `docs/performance_reports/`.
The steering entry point derives the profile name from its containing
directory, so result artifacts, worker leases, locks, and reproduction files
use independent roots for the copied name. Commit the copy and rebuild the
repository environment before recording measurements so its source identity is
clean.

Create a portable copy, including raw data, TeX, and the reviewed PDF, from a
source checkout with:

```bash
python3 docs/arxiv/result_tables.py   export-profile macbook_M3_manual /absolute/output/path
```

The copied entry point selects this profile automatically. A copy made from an
installed wheel uses that wheel's recorded source revision and installed
runtime; a copy inside a contributor checkout retains the exact-source
workflow. Never commit evaluator artifacts, candidate wheels, prepared models,
attempts, logs, locks, coordination state, page images, or LaTeX auxiliary
files.
