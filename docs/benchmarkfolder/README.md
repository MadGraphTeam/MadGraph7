# Reproducing the NinjaRS benchmark

Run all commands from the MadGraph7 repository on branch `madloop_ninjars`.
The baseline is revision `27147b3300f556a75661a886e92bdbe00558ee25`.
The baseline's tracked numerical sources must remain unchanged.

## Environment and installation

Enter the pinned Linux x86_64 environment with `nix develop path:.`. The flake
pins the compiler, Rust, Python, Typst, native arithmetic libraries, and report
fonts. `provenance.json` records the versions and host used for the measurements.

```sh
python tools/ninjars_setup_references.py
python bin/madgraph --plain docs/benchmarkfolder/commands/installer-install.mg5
```

The setup script archives the original revision under `.benchmark-work/original`,
checks downloaded reference-library archives against `reference-downloads.json`,
builds Ninja with its quadruple-precision library and Collier, and imports the
Gmu EW model through the original CLI. See `provenance.json` for the model source
and checksum. The repository-owned `install ninjars` command installs pinned
NinjaRS/oneloopmaster sources and runs a compiled Fortran DP/QP smoke test.
It does not need the sibling reference clones.

All generated outputs, logs, profiles, and temporary builds stay under
`.benchmark-work` or `.ninjars-build`. The benchmark uses a shared
`.ninjars-build/symbolica-locks` directory for Symbolica's normal license
coordination. Serialize MP benchmark processes when the license permits only one
active worker; a license with adequate worker capacity is required for parallel
MP production use.

## Core comparison

`cases.json` defines the eight processes, seeds, point counts, tolerances, and
performance gate. Command cards in `commands/` record the actual CLI inputs.
The harness regenerates their absolute paths for the current checkout.

```sh
python tools/ninjars_benchmark.py generate --source both
python tools/ninjars_benchmark.py build --source both
```

Validate the retained worker protocol using the commands in
[`persistent-workers.md`](persistent-workers.md), then collect and promote a
complete seven-target series for each core process:

```sh
python tools/ninjars_benchmark_workers.py measure --source both \
  --cases qcd-1 --cpu 0 --repetitions 7 --clean-build --run-id core-qcd-1
python tools/ninjars_benchmark_workers.py promote --runs \
  docs/benchmarkfolder/experiments/persistent-workers/core-qcd-1
```

Choose an allowed, otherwise idle CPU and use it for every measurement. Stop
concurrent compilation, profiling, and other task evaluations before acceptance
timing. The single evaluation thread is pinned with `taskset`. There are at least
seven repetitions; backend order alternates. If the confidence interval leaves
the gate unresolved, collect a longer complete series in a new run directory.

Rebuilds require a fresh full series: the acceptance checker rejects mixed
executables, numerical settings, phase-space streams, parameter cards, CPUs,
or unpaired baseline/candidate measurements. An unresolved confidence interval
does not pass the gate.

The retained Fortran drivers initialize helicity filters, warm the full point
stream, and time the complete MadLoop call with normal stability checks and
actual rescues. Persistent workers also copy numerical results during the last
timed traversal; each backend pays the same copy overhead. File I/O is outside
the timed loop. Collier UV and IR poles are enabled. Both drivers retain Born,
finite, single-pole, double-pole, and physical split-order results. Neither
helicity nor color sums are replaced by sampling. The original fresh-process
driver remains available through `ninjars_benchmark.py measure` and provides the
independent protocol-parity reference.

The common core settings use `MLStabThres=1e-9` and
`COLLIERRequiredAccuracy=1e-10`. Default settings allowed discrepancies beyond
the requested comparison tolerance at high multiplicity; the stricter settings
apply equally to every backend and to the untouched original baseline. Exact
physical inputs and tolerances remain unchanged, and actual rescues stay in the
timed evaluation. Diagnostic evidence is retained under
`experiments/default-stability-qcd-3` and `experiments/strict-stability-qcd-3`.

## Retained evidence

- `points/`: exact momenta, masses, seeds, and checksums.
- `inputs/`: parameter cards, backend-specific MadLoop cards, generation times,
  compilation settings, compiler version, and executable/driver checksums.
- `data/`: numerical rows, actual return-code histograms, raw repetitions,
  agreement decisions, and paired-bootstrap performance intervals.
- `experiments/`: independent Fortran profiles and isolated optimization tests,
  including rejected or statistically unresolved candidates.
- `controls.json`: focused loop-induced, mixed-order, complex-mass, and rescue
  process definitions. Additional focused controls are retained by the control
  harness.
- [`controls.md`](controls.md): focused-control commands, transformed inputs,
  genuine own-QP rescue results, parameter/context checks, and the retained
  strict-threshold original-Ninja diagnostic failure.
- `audit.md`: independent numerical and performance review.

The core comparison uses all six backend pairs at each point, with
`abs(a-b) <= 1e-7 max(abs(a),abs(b)) + 1e-10 S`; `S` is the largest absolute
Born/Laurent value over all backends and split orders at that point. Legacy
backends are also compared with their original implementations.

## Report

```sh
python tools/ninjars_report.py --compile
```

This regenerates `report-data.json` and compiles `benchmark.typ` to
`benchmark.pdf`. Missing or unresolved evidence remains explicitly pending.
The styling is attributed in the source to the
[pyamplicol reference report](https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ).
The final PDF must also be visually inspected.
