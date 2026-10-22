# Final measured benchmark report

The published document is [`../benchmark.pdf`](../benchmark.pdf), generated from
[`../benchmark.typ`](../benchmark.typ). It contains actual Ninja6/NinjaRS8
measurements and the separately scoped correctness/precision evidence.

* Main MG7 CLI profile: all 12 processes including both SM 2→4 cases and
  gg→H/HH/HHH/HHHH, three sessions/tool (72 accepted records). Large table values
  are NinjaRS; the comparison line is Ninja. Estimated reduction is FULL−SKIP;
  SKIP also contains non-reduction driver/Born/kinematic overhead. Actual phase
  counts and signed raw estimates are retained.
* Ten cases have matching scopes across all six sessions. The QCD1/EW1 timing
  helicity/filter changes after session 0, so their aggregate ratios remain NA
  under the unchanged conservative reader. Every measured median is retained.
  All actual selected weights are positive; optional zero-weight polarized
  ratios would be suppressed in presentation, without changing raw statistics.
* Ancillary original-code comparison: six cases, ten fixed prefix points and
  seven adjacent pairs, full sums/poles/normal rescue. The denominator is literal
  original 27147 Ninja. All six upper 95% ratio intervals are below 1.05, including
  the slower EW2 median. The 0.5 s setting is a calibration target; actual commands
  last 0.326–0.519s. Only final-traversal status vectors were saved.
* Normal correctness: eight SM cases with 100 unique points each, calibration and
  two exact repeated observations. NinjaRS returns 768 DP/32 automatic QP outcomes
  per 800-point cohort. Extra HHH/HHHH controls use two fixed points each.
* Native-DP random stability: exactly eight affordable cases ×100 points ×2 tools.
  Born cases use full sums; H/HH use all-minus. One-point streams have one
  constrained on-shell invariant and random boosts. Hardware IEEE-invalid flags
  are retained separately from algorithmic rejection; see the report.

## Offline PDF reproduction

Activate the documented environment or provide **Typst 0.15.1 (9dfd3a08)**, then
run from the repository root:

```sh
python -B doc/benchmarkfolder/final-report-v1/reproduce.py --output .benchmark-work/reproduced-benchmark.pdf
```

The script checks every finite manifest-listed input, uses the bundled
Libertinus 7.051 fonts with system/embedded fonts disabled, sets
`SOURCE_DATE_EPOCH=1792627200` and the same explicit creation timestamp, then
requires byte-identical PDF output. It performs no network, native-library read,
physics compilation or amplitude evaluation. `--verify-only` checks inputs
without invoking Typst. `plot.py` recreates the timing SVG from the copied
validated statistics; it recomputes no timers/bootstrap/scientific acceptance.
The 12 DP SVGs come from the separately audited vector capsule in the immutable
scientific archive below. Its retained `plot.py` and tail arrays document their
construction; the frozen vectors here suffice for offline PDF reproduction.

`report-data.json` is the compact Typst/data view. `cli-report.json.gz` retains
the exact concrete-consumer materialization (decoded hash in provenance),
including raw session values, intervals, per-stage counts, selected helicities,
runtime metadata and source hashes. `measurements.csv` provides the main full-sum
statistics in a flat format. `preparation-costs.csv` retains the exact captured
first-session output/export, HELAS/model compilation, initial process compilation,
initialization and boot costs, with 24 source-metadata hashes. These are neither
an additive total nor standalone symbolic-generation measurements. `provenance.json` links the concrete authorities.
No failed pilot fragment enters those statistics; all three earlier Ninja-only
failed pilots are retained separately in the archived publication inventory.

## Scientific/source custody and regeneration

The [immutable scientific archive](https://github.com/MadGraphTeam/MadGraph7/tree/19d67c8605337c176680878a521b3a014176a7fb/docs/benchmarkfolder) is commit
`19d67c8605337c176680878a521b3a014176a7fb`. It preserves the full raw observations,
command cards, authority readers, generated-source closure, audits and historical
partial report at their original paths. Those experiment trees are intentionally
absent from this lean report folder. `provenance.json` gives an immutable blob URL
and original byte hash for each concrete authority. Original paths inside the
copied JSON/gzip/CSV are archival identifiers, not current-checkout dependencies.

For raw scientific revalidation, open the archive's documented campaign recipe
or extract that exact commit into a separate directory; do not run the old raw
paths against the cleaned checkout. With that archive and its retained native
prerequisites restored, the original command was:

```sh
python -B tools/ninjars_cli_report_v3.py \
  --results docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/profile-cli-expanded-v4/run-v1/result-list.json \
  --expected docs/benchmarkfolder/final-report-v1/expected-cli-pins.json \
  --output revalidated-cli-report.json
```

This is an archived-checkout command, not an offline-PDF prerequisite. It used
consumer `6095252a` and concrete reader `b0681842`; native full-custody validation
also needs the exact retained b0d installation. The CLI observer commit 464b
differs from numerical source 856e; the shared Fortran/Rust numerical
implementation remains unchanged. Fresh scientific regeneration uses the
archive's command cards, installation pins, compiler and campaign recipes.
A different compiler can produce fresh executable/measurement identities;
that is distinct from reproducing this PDF from already validated measurements.

The source-bound numerical proof retains 1392-request 256/384 convergence, the
historical bit-identity failure, adjacent binary128 rounding and the unresolved
direct-provider exceptional cases. The report does not convert those failures
into passes. Standard Rust tests are 24 pass/7 ignored; the two failing diagnostics
are explicit ignored controls, not a failing ordinary suite.

Styling is attributed to [NinjaRS ca1009e](https://github.com/alphal00p/ninjars/blob/ca1009e89ecb03d3ff477f57b170241f7a9f8669/docs/benchmark/benchmark.typ)
and [pyamplicol](https://github.com/MadGraphTeam/pyamplicol/blob/loop/docs/performance_reports/loop/benchmarks.typ).
Method inspiration is [Hirschi–Peraro](https://arxiv.org/pdf/1604.01363).
Bundled unmodified Libertinus fonts use the included OFL 1.1 license. The
[historical partial report](https://github.com/MadGraphTeam/MadGraph7/tree/19d67c8605337c176680878a521b3a014176a7fb/docs/benchmarkfolder/historical/20261021-partial)
is retained in the immutable scientific archive.
