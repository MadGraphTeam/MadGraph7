<!-- SPDX-License-Identifier: 0BSD -->
# Performance Report

`pyAmpliCol.tex` is the public methodology and performance report for
pyAmpliCol 0.1.0. Its tables are generated from canonical JSON caches in
`results/`; measured values must never be edited into TeX directly.

The canonical document and both architecture profiles are intentionally reset
scaffolds. Their 1,646 required measurement cells currently have status
`not_available`. Structural process/multiplicity positions are shown as
`not applicable`, and reference execution fields without a compatible public
timing boundary are shown as `not exposed`. This keeps the full declared
coverage visible before either measurement campaign begins.

## Canonical commands

Run from the repository root:

```bash
python3 docs/result_tables.py validate
python3 docs/result_tables.py audit
python3 docs/result_tables.py render --compile
python3 docs/result_tables.py recover --compile
python3 docs/result_tables.py populate --dry-run --missing-only
```

`render` preserves validated cache contents and regenerates the table views.
`recover` incorporates completed immutable worker attempts before rendering.
`populate` selects cells by dataset, mode, model, colour accuracy, process,
multiplicity, variant, workload, or exact cell ID. A non-dry population
requires an exact authenticated source and native runtime.

## Architecture profiles

The two campaigns use independent tracked workspaces:

- [`macbook_M3/TABLE_FILLING.md`](performance_reports/macbook_M3/TABLE_FILLING.md)
  is the authoritative sequential Apple-M3 procedure.
- [`x86_EPYC/TABLE_FILLING.md`](performance_reports/x86_EPYC/TABLE_FILLING.md)
  is the authoritative ten-worker AMD-EPYC procedure.

Both cover the complete declared range from `n=1` through `n=9`; neither has an
`n<=4` campaign cap. Both use five-second timing targets and require numerical
agreement before a timing can be published. Their hard resource policies,
phase ordering, resource-frontier rules, frequent PDF review, two mandatory
user-approval pauses, branch isolation, cluster-to-local support lane, hourly
cluster PDF review, and final publication protocol are specified only in those
runbooks.

Each profile contains raw JSON, the JSON schema, complete TeX sources,
generated table TeX, environment metadata, a workspace manifest, standalone
build and report entry points, and a reviewed PDF. Process/evaluator artifacts,
candidate wheels, prepared models, worker attempts, logs, locks, page images,
and coordination state stay in ignored machine-local roots.

Compile either checked-in profile without a pyAmpliCol runtime:

```bash
cd docs/performance_reports/macbook_M3
python3 build_pdf.py
```

Create a portable publication-only copy:

```bash
python3 docs/result_tables.py export-profile \
  macbook_M3 /absolute/output/path
```

The export rebuilds its tables and PDF and excludes all process artifacts and
machine-local campaign state. Generated runtime identities are preserved
byte-for-byte; only explicit unhashed locator fields may be made portable.

Each completed profile is audited at its own report-only publication revision.
When both independently audited profiles are assembled on `main`,
`tools.performance_report.aggregate_audit` accepts only raw result JSON,
generated environment/table TeX, the validation summary, and the PDF, and
verifies that each profile subtree exactly matches its announced audited
revision.

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
