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

Create a portable copy, including raw data, TeX, and the reviewed PDF, from a
source checkout with:

```bash
python3 docs/arxiv/result_tables.py   export-profile macbook_M3_manual /absolute/output/path
```

The copied entry point selects this profile automatically. Measurements still
require the exact pyAmpliCol source checkout and authenticated native runtime.
Never commit evaluator artifacts, candidate wheels, prepared models, attempts,
logs, locks, coordination state, page images, or LaTeX auxiliary files.
