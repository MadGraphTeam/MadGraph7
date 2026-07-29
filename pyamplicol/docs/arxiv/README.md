<!-- SPDX-License-Identifier: 0BSD -->
# arXiv Report Source

`pyAmpliCol.tex` is the public methodology and performance report for
pyAmpliCol 0.1.0. Its tables are generated from canonical JSON caches in
`results/`; measured values must never be edited into TeX directly.

This directory is a self-contained publication bundle suitable for
building and reviewing the canonical report. The architecture-specific
measurement workspaces and their authoritative campaign procedures live under
[`../performance_reports/`](../performance_reports/README.md).

The canonical document is intentionally a reset scaffold. Its 1,666 required
measurement cells currently have status `not_available`. Structural
process/multiplicity positions are shown as `not applicable`, and reference
execution fields without a compatible public timing boundary are shown as
`not exposed`. This keeps the complete declared coverage visible before
architecture-specific measurements are incorporated.

## Canonical commands

Run from the repository root:

```bash
python3 docs/arxiv/result_tables.py validate
python3 docs/arxiv/result_tables.py audit
python3 docs/arxiv/result_tables.py render --compile
python3 docs/arxiv/result_tables.py recover --compile
python3 docs/arxiv/result_tables.py populate --dry-run --missing-only
```

`render` preserves validated cache contents and regenerates the table views.
`recover` incorporates completed immutable worker attempts before rendering.
`populate` selects cells by dataset, mode, model, colour accuracy, process,
multiplicity, variant, workload, or exact cell ID. A non-dry population
requires an exact authenticated source and native runtime.

## Data contract

`results/report-cache.schema.json` is the formal cache schema. The report
service also verifies constraints that JSON Schema alone does not express:

- every required process-family, ladder, and multiplicity cell exists exactly
  once;
- every successful result carries the required numerical comparisons;
- unavailable observations contain no invented numerical value;
- multiplicities are positive, sorted, and unique; and
- checked-in table text exactly matches the checked-in caches.

Workers invoke the public `Generator`, `Runtime`, and `BenchmarkRunner` APIs.
Original AmpliCol is evaluated through the maintained independent reference
adapter. Prepared-model construction is recorded separately and excluded from
process-generation measurements.

## Generated inputs

The report contains 20 generated TeX inputs: 15 process matrices (including
the three best-mode summaries), two Z-plus-jets evaluator ladders, two
colour-singlet model ladders, and one numerical-validation summary. Recurrence
is the default pyAmpliCol mode. The overview matrices select the fastest
validated mode separately for each workload and label recurrence, compiled JIT
O3, and eager-DAG JIT O2 as `(A)`, `(B)`, and `(C)`.

Generated files carry an SPDX 0BSD header and a warning not to edit them
directly.
