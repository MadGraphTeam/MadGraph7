# Persistent benchmark protocol

`tools/ninjars_benchmark_workers.py` retains one generated MadLoop process for
each source/backend combination. Every worker is pinned to the same requested
CPU, and only one worker receives a command at a time. The others block on
standard input. This amortizes initialization and removes the separate untimed
numerical traversal from each repetition.

The timer includes the full MadLoop call, scale update, numerator construction,
scalar integrals, stability checks, actual rescue, and a status counter for every
call. During the final traversal of a timed command, the driver copies every
Born/Laurent coefficient, split-order column, accuracy estimate, and return code
into arrays. All backends pay this copy overhead. File output happens after the
timer. Initialization and a complete warm traversal occur once per worker;
calibration and every measured command then replay the identical retained stream.

The original Ninja and candidate NinjaRS measurements are adjacent. Odd
repetitions reverse the entire backend order, preserving that adjacency. Seven
in-process blocks estimate warmed block variability, not fresh-process
initialization variability. Initialization timing remains a separate result.
Core physics and speed gates retain the agreed thresholds and settings.

## Representative validation

Before promotion, the exact controller, worker driver, and main-harness source
hashes must have representative fresh-process parity evidence. Validation uses
the complete configured point stream for each representative case: 100 points
for the core QCD example and ten for focused controls. Each representative
target runs its retained fresh-process executable, then a persistent worker
calibration and at least two subsequent commands. All physical rows and each
point's backend/rescue outcome must agree.

Coverage must include double precision for CutTools, Ninja, Collier, and NinjaRS;
actual higher-precision rescue for CutTools, Ninja, and NinjaRS; a loop-induced
process; and multiple physical split orders. Representative parity validates
the execution protocol. Every promoted core series independently retains and
compares all 100 points across all four backends and the original legacy
implementations. Expensive fresh qcd-4 duplication is therefore unnecessary.

Run commands from the repository root with the pinned environment:

```sh
source .ninjars-build/dev-env.sh
python tools/ninjars_benchmark_workers.py measure \
  --source both --backends 1,6,7 --cases qcd-1 \
  --cpu 3 --repetitions 2 --fresh-parity --run-id parity-qcd-legacy
python tools/ninjars_benchmark_workers.py measure \
  --source candidate --backends 1,6,7,8 \
  --config docs/benchmarkfolder/controls.json \
  --cases control-gg-h,control-mixed --cpu 3 --repetitions 2 \
  --fresh-parity --run-id parity-controls
python tools/ninjars_benchmark_workers.py measure \
  --source candidate --backends 8 --cases qcd-1 \
  --cpu 3 --repetitions 2 --fresh-parity --run-id parity-qcd-ninjars
python tools/ninjars_benchmark_workers.py measure \
  --source candidate --backends 1,6,8 \
  --config docs/benchmarkfolder/control-variants.json \
  --cases control-near-singular --cpu 3 --repetitions 2 \
  --fresh-parity --run-id parity-rescue
python tools/ninjars_benchmark_workers.py coverage --runs \
  docs/benchmarkfolder/experiments/persistent-workers/parity-qcd-legacy \
  docs/benchmarkfolder/experiments/persistent-workers/parity-controls \
  docs/benchmarkfolder/experiments/persistent-workers/parity-qcd-ninjars \
  docs/benchmarkfolder/experiments/persistent-workers/parity-rescue
```

Run each command serially. NinjaRS higher-precision evaluation requires the
shared Symbolica runtime slot; coordinate this separately from CPU affinity.
Changing any of the three protocol sources invalidates prior protocol evidence
and requires new representative runs.

## Core series and promotion

After coverage validation, collect at least seven repetitions for all seven
source/backend targets. Each core process may have its own run directory.

```sh
python tools/ninjars_benchmark_workers.py measure \
  --source both --backends 1,6,7,8 --cases qcd-1 --cpu 3 \
  --repetitions 7 --clean-build --run-id core-qcd-1
python tools/ninjars_benchmark_workers.py promote --runs \
  docs/benchmarkfolder/experiments/persistent-workers/core-qcd-1
```

`--clean-build` removes object files and the executable only from the isolated
worker subprocess copy, then recompiles its generated Fortran. Source/model,
DHELAS, and dependency libraries are copied from existing builds. This timing
is explicitly a clean subprocess compilation, not a cold dependency build.
Without the flag, compilation is an incremental worker-driver relink.

Measurement never overwrites core data. Promotion rejects incomplete series,
missing validation, mixed protocols, changed retained evidence, changed
executables, noncontiguous repetitions, inconsistent settings, and missing
physical rows/statuses. It first preserves superseded records under `rejected/`,
then writes canonical `data/<source>-<case>-<backend>-<rep>.json` records.
Numerical or performance failures remain published with outstanding gate
results, so the report can describe failures honestly. Publication does not
mean that the project's correctness or performance goals have passed.

## Retained evidence and failure handling

Each experiment contains:

- `protocol.json`: exact source hashes, configuration, CPU, initialization and
  timing policies, backend selection, repetition count, and uncertainty scope.
- `sources/`: snapshots of the controller, worker driver, and main harness.
- `run-status.json`: requested/completed cases and incomplete, failed, or
  complete status.
- `builds/<case>-<source>-<backend>/`: rendered Fortran, source manifest,
  executable and card hashes, compiler/build scope, cards, and linking inputs.
- `data/`: calibration, all measured rows, status histograms, and, for
  representative runs, fresh-process results and parity checks.
- `logs/` and `rejected/`: compilation/diagnostic logs and failed-command
  snapshots, including sequence, stage, backend/source, error, and raw output.

A dedicated FIFO carries bounded `READY`, `DONE`, and `BYE` messages; diagnostic
stdout is drained continuously. Commands use increasing sequence numbers.
Unexpected exits, timeouts, malformed replies, and nonzero shutdown fail the
run. Cleanup always stops and reaps the controller's worker process groups.

`worker-protocol-validation.json` binds coverage to protocol identity and the
SHA256 of every representative result and its fresh/build/parity evidence.
Promotion copies this proof to `inputs/worker-protocols/<evidence hash>.json`.
It creates `inputs/persistent-series/<manifest hash>.json` containing
`complete`, `benchmark_protocol`, `protocol_identity_sha256`,
`protocol_validation_sha256`, `repetitions`, `source_run`, and `measurements`.
The last field maps canonical data filenames to original experimental JSON
hashes. Promoted records add `protocol_series_complete`,
`protocol_repetitions`, `protocol_run_manifest_sha256`, and
`protocol_source_measurement_sha256`; the report verifies this chain before
treating a series as eligible.
