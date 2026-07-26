<!-- SPDX-License-Identifier: 0BSD -->
# Architecture Performance Reports

Each child directory is an independent, tracked publication workspace for one
measurement environment:

- [`macbook_M3`](macbook_M3/README.md), with its authoritative
  [`TABLE_FILLING.md`](macbook_M3/TABLE_FILLING.md).
- [`x86_EPYC`](x86_EPYC/README.md), with its authoritative
  [`TABLE_FILLING.md`](x86_EPYC/TABLE_FILLING.md).

The workspaces intentionally duplicate the report source, raw JSON schema and
caches, generated TeX, build entry point, and PDF. This makes the two campaigns
portable and prevents one machine from depending on or modifying the other
machine's publication tree. Do not replace these files with links into the
canonical [arXiv source](../arxiv/README.md).

The runbook inside each profile is the sole authority for campaign ordering,
resource limits, approval pauses, numerical validation, visual review,
coordination, and publication. Process artifacts, prepared models, evaluator
attempts, logs, locks, coordination state, page images, and LaTeX auxiliary
files remain outside the tracked profile directories.
