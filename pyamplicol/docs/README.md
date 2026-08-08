<!-- SPDX-License-Identifier: 0BSD -->
# Documentation

Start with:

- [Get started: a gentle walkthrough](user/gentle-walkthrough.md) for a
  MadGraph-familiar, copy-paste first calculation;
- the [user guide](user/index.md) for installation, configuration, generation,
  runtime use, the native SDK, licensing, and release status;
- the [packaged examples](../examples/README.md) for complete cards and API
  drivers;
- the [performance reports](performance_reports/README.md) for the two
  retained rendered snapshots;
- the [development contracts](development/README.md) for stable API,
  configuration, physics-extraction, packaging, and architecture decisions.

Only two rendered report PDFs are retained in this repository. Raw measurements,
generated report source, campaign workspaces, process artifacts, wheels, logs,
locks, page images, and LaTeX auxiliary files are intentionally excluded. An
installed package can create a new empty workspace with
`pyamplicol profiling-campaign copy DEST --force`.
