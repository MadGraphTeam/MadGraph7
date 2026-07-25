<!-- SPDX-License-Identifier: 0BSD -->
# Performance Report

`pyAmpliCol.tex` is the public methodology and performance report for
standalone pyamplicol 0.1.0. Its result tables are generated from canonical
mode-owned JSON caches in `results/`; measured values must never be edited into
TeX directly.

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
python3 docs/result_tables.py recover --compile
python3 docs/result_tables.py populate --dry-run --missing-only
```

`reset` reconstructs all canonical N/A caches. `render` preserves validated
cache contents, joins baseline measurements dynamically, and renders the table
views. `recover` reconstructs caches from validated immutable worker attempts.
With `--compile`, JSON, generated TeX, and `pyAmpliCol.pdf` are staged and
published together. These three commands do not generate a process artifact or
run a benchmark.

`populate` selects cells through repeatable dataset, mode, model, colour,
process, multiplicity, variant, workload, or exact-cell filters. It runs one
isolated worker process per cell and supports parallel campaigns, explicit
artifact reuse/retiming/regeneration, per-cell time and RAM limits, and
`--missing-only` or deliberate `--rerun` operation. Use `--dry-run` to inspect
the dependency-ordered schedule before starting work. LC defaults to both
runtime-selected workloads; `--workload selected-flow` and
`--workload all-flow` restrict it to one.

## Data Contract

`results/report-cache.schema.json` is the formal schema. The report service
also performs cross-entry checks that JSON Schema alone does not express:

- every process-family/multiplicity cell exists exactly once;
- every ladder variant/multiplicity cell exists exactly once;
- N/A observations contain no numeric value, configuration, or environment;
- multiplicities are positive, sorted, and unique; and
- checked-in table text is exactly the rendering of the checked-in caches.

Workers invoke the public `Generator`, `Runtime`, and `BenchmarkRunner` Python
APIs directly. Process generation and runtime profiling therefore use the same
implementation as the public CLI without parsing terminal output. Original
AmpliCol is steered through the maintained independent oracle adapter.
Prepared-model construction is lock-protected, recorded separately, and
excluded from process-generation cells.

## Generated Inputs

The main document inputs sixteen generated `result_*_table.tex` files: twelve
three-mode process matrices, two Z-plus-jets ladders, one scalar-contact ladder,
and one scalar-gravity ladder. The recurrence matrices compare built-in and
UFO-SM with original AmpliCol. Built-in compiled JIT O3 and eager-DAG JIT O2
matrices use recurrence JIT O2 as their dynamically joined baseline. Generated
files carry an SPDX 0BSD header and a warning not to edit them directly.
