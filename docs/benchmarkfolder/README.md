# Reproducing the NinjaRS benchmark

Run all commands from the MadGraph7 repository on branch `madloop_ninjars`.
The baseline is revision `27147b3300f556a75661a886e92bdbe00558ee25`.
The baseline's tracked numerical sources must remain unchanged.

Final validation uses the [corrected-model policy v2](policy-v2/README.md).
It compares all four corrected candidate backends with three independently
bound corrected-original references, while keeping the literal original Ninja
as the timing denominator. Its common Collier accuracy is `1e-12`.
The commands and report described below retain the earlier v1 experiment;
its four accepted cases do not certify the current implementation. The final
eight-process validation and performance gates remain outstanding.

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

## Retained v1 core comparison

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

The retained v1 core settings use `MLStabThres=1e-9` and
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

## Retained v1 report

```sh
python tools/ninjars_report.py --compile
```

This regenerates `report-data.json` and compiles `benchmark.typ` to
`benchmark.pdf`. Missing or unresolved evidence remains explicitly pending.
The styling is attributed in the source to the
[pyamplicol reference report](https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ).
The final PDF must also be visually inspected.

The current report is an intermediate snapshot: four of eight core cases passed
both correctness and performance for their recorded implementation. These are
historical results pending recertification after the model-precision correction
described below. The seven-target series for `qcd-1`, `ew-1`,
`qcd-2` and `ew-2` contains 196 accepted measurements. The subsequent `ew-3`
calibration failed on original Collier and remains outstanding, together with
the two four-body cases and QCD three-body performance.

The complete failed run is retained under
`experiments/persistent-workers/runtime-v1-core-small-20261020`. Its completed
prefix is derived, without numerical changes, under
`runtime-v1-core-small-completed-20261020` in the same directory. The derivation
preserves the entire original failed run, rejects incomplete cases or omitted
completed cases, and uses the unchanged numerical protocol validator before
promotion:

```sh
python tools/ninjars_benchmark_extract.py verify --destination \
  docs/benchmarkfolder/experiments/persistent-workers/runtime-v1-core-small-completed-20261020
```

## Generated model precision correction

The EW3 investigation found numerical defects before loop reduction: generated
quadruple-precision helpers used default-kind complex constructors, mistyped pi
constants, and default-kind real casts. These are corrected in `export_v4.py` and
`ufo_expression_parsers.py`; the UFO input and parameter card are unchanged.
Compiled analytic tests exercise branch continuation, threshold sides, and
binary128 inputs distinguishable only beyond double precision. The same helpers
also occur in QCD counterterms.

The original archive and its timings remain unchanged. Original higher-precision
answers affected by these defects must remain visible as numerical discrepancies.
Final correctness needs separately identified, minimally corrected original
exports alongside all four corrected candidate backends. Performance continues
to use the unmodified original Ninja denominator. The frozen seven-target
collection protocol remains unchanged; the separate
[v2 validation and report tools](policy-v2/README.md) now encode this distinction,
check all 21 corrected-cohort pairs, and bind representative precision coverage
to the exact linked runtimes. Earlier accepted results do not certify the
repaired implementation. Final publication requires a complete validated v2
dataset and refreshed candidate evidence.

See `experiments/ew3-collier-stability/` for the failed original runs, precision
diagnostics, independent audit, fresh CLI export, and benchmark policy review.

## Guarded symmetric-tensor optimization

The production QP contraction-sharing change is committed as `f4bffc61e`.
Its fixed 21-pair QCD3 study resolves a 3.19% full-process runtime reduction
against the retained pre-factor candidate, with a 95% interval of 1.79–4.15%.
All compared coefficients, accuracy estimates and rescue counts are unchanged.
This is a process-specific optimization result, not the final NinjaRS gate.
The source proof, exact arithmetic controls and timing evidence are retained
under `experiments/symmetric-factorization*`.

Thirteen clean CLI exports/builds, refreshed representative precision coverage,
and a relocated internal-copy DP/QP/stability check now pass for that numerical
revision. See `policy-v2/generation-retention-abi3-tensor-factor-v1/`,
`policy-v2/representative-retention-abi3-tensor-factor-v1/` and
`policy-v2/install-export-abi3-tensor-factor-v1/` for offline verifiers.
The refreshed generated controls and rotation, rebinding, fresh-context and
near-singular variants also pass; see
`policy-v2/controls-retention-abi3-tensor-factor-v1/` and
`policy-v2/final-variant-controls-abi3-tensor-factor-v1/`.
Complete eight-process validation remains outstanding.
The current NinjaRS QCD3 diagnostic is about twice the candidate Ninja time;
the final performance requirement is unresolved. DP tensor factorization and
MP tensor allocation experiments remain isolated from production.
The separate DP factorization's fixed 21-pair process study finds no gain:
runtime ratio 1.013546, with a 95% interval of [0.996598, 1.030949]. All
16,800 coefficient comparisons pass and the two rescue points are unchanged.
Its complete negative result and rejected preflights are retained under
`experiments/symmetric-factorization-dp/process-timing21-v2/`.

A separate MP transport correction preserves signed zero while widening the
recovered binary128 value. Eighteen bridge tests, fourteen GNU REAL16 boundary
fixtures and all 1,392 stored requests pass; all 11,136 replayed output fields
are unchanged. Source and evidence are in
`experiments/ninjars-mp-transport-audit-v1/`. The installed-runtime refresh is
still pending, so earlier runtime proofs retain their original archive identity.

The complete stored-input eight-cell replay and independent audit are retained
under `experiments/ninjars-qp-numerator-dp-probe-v1/replay/`. All 3,712 calls
and 1,392 reproduction controls pass. Improved coefficients alone do not remove
the dominant native/generic discrepancy; distinct scalar thresholds and the
historical parser prevent interpreting it as a pure precision experiment.
No numerator-only retry is introduced. The source-only guard/fusion census in
`experiments/symmetric-factorization-dp/guard-sharing-study/` also remains an
unimplemented research lead.
