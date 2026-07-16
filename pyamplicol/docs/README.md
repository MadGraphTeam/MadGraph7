<!-- SPDX-License-Identifier: 0BSD -->
# Performance Report

`pyAmpliCol.tex` is the public methodology and performance report for
standalone pyamplicol 0.1.0. Its result tables are generated from the JSON
caches in `results/`; measured values must never be edited into TeX directly.

The checked-in release state is deliberately empty. Every cache entry has
status `not_available`, every measurement field is `null`, and every rendered
cell is `N/A`. The process families and multiplicity grids remain present so a
future campaign cannot silently change coverage.

## Commands

Run these commands from the repository root:

```bash
python3 docs/result_tables.py validate
python3 docs/result_tables.py reset --compile
python3 docs/result_tables.py render --compile
```

`reset` reconstructs all canonical N/A caches. `render` preserves validated
cache contents. With `--compile`, JSON, generated TeX, and `pyAmpliCol.pdf` are
staged and published in one rollback-capable transaction. Neither command
generates a process artifact or runs a benchmark.

## Data Contract

`results/report-cache.schema.json` is the formal schema. The Python service
also performs cross-entry checks that JSON Schema alone does not express:

- every process-family/multiplicity cell exists exactly once;
- every ladder variant/multiplicity cell exists exactly once;
- N/A observations contain no numeric value, configuration, or environment;
- multiplicities are positive, sorted, and unique; and
- checked-in table text is exactly the rendering of the checked-in caches.

The `BenchmarkObservation.from_result()` adapter accepts the public typed
`BenchmarkResult` fields, including requested/effective `BenchmarkConfig`,
uncertainty, and environment. A measurement orchestrator must still invoke
`Generator` and `BenchmarkRunner` explicitly, obtain an independent reference
where required, and update one cache entry before calling `render --compile`.
Generation duration and independent-reference results are not currently part
of `BenchmarkResult`, so they must come from the explicit campaign
orchestrator.

## Generated Inputs

The main document inputs ten generated `result_*_table.tex` files: six
built-in/external SM LC/NLC/full matrices, two Z-plus-jets ladders, one scalar
contact ladder, and one scalar-gravity ladder. Generated files carry an SPDX
0BSD header and a warning not to edit them directly.
