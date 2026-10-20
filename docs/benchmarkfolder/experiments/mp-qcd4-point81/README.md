# Normal-rescue diagnostic for qcd4 point 81

The same normal rescue remains expensive on a second call in the retained
process. This identifies repeat reduction work, rather than only lazy evaluator
construction, as a useful optimization target. These are diagnostic profiles,
not acceptance timings: the generator predates the final Fortran optimization,
profiling adds overhead, and other task builds were active during some runs.

Each retained driver first initializes MadLoop with DP point 1, then evaluates
point 81 with `CTModeRun=-1`, `MLStabThres=1e-9`, and the recorded parameter card.
The two-call runs use identical point, mass, scale, and numerator inputs.
NinjaRS uses its own 256-bit rescue. Original Ninja uses its original quad
rescue. Both returned successful rescued statuses and agreeing Laurent
coefficients.

| Engine and link runtime | First call (s) | Second call (s) | Status |
| --- | ---: | ---: | --- |
| Original Ninja | 86.110584821 | 85.946258089 | 328, 328 |
| NinjaRS, GNU runtime before Rust archive | 146.870983058 | 161.008765706 | 322, 322 |

The exact per-call coefficients, accuracies and times are in
[`../repeat-qcd4-point81-ninja/per-call.dat`](../repeat-qcd4-point81-ninja/per-call.dat)
and
[`../repeat-qcd4-point81-ninjars/per-call.dat`](../repeat-qcd4-point81-ninjars/per-call.dat).
Both calls within each engine produced identical printed coefficients. The
finite terms are respectively `6.15354295643188145e-11` and
`6.15354295643192022e-11`.

Sampling used `cpu-clock:u` at 49 Hz and 65,528-byte DWARF stacks, without
ptrace. The mutually exclusive classification tests the actual owning method
in stack frames, so a type parameter naming `SymbolicaMasters` does not count
as an invocation of that provider.

| Stack category | Original Ninja, 8,416 samples | NinjaRS, 14,966 samples |
| --- | ---: | ---: |
| Fortran QP numerator construction | 93.20% | 52.43% |
| Scalar masters | 0.25% | 15.54% |
| Generic tensor expansion/contraction | — | 14.28% |
| Generic cut geometry | — | 3.17% |
| Other MP worker | — | 13.31% |
| Ninja QP reduction | 4.36% | — |
| Other main-thread work | 2.19% | 1.27% |

The complete classification, representative stacks and flat profile are
retained beside each run. Per-call sample splits are approximate because the
first sample occurs after attaching perf; per-call elapsed times come from
Fortran counters. Scalar-master cost persists on the second NinjaRS call.

Earlier one-call runs retained under `normal-qcd4-point81-*` measured 86.6396 s
for Ninja and 131.8302 s for NinjaRS. The latter predates GNU runtime isolation:
Rust archive helpers interposed Fortran binary128 arithmetic. It is historical
diagnostic evidence only. See the independent
[runtime-link evidence](../quad-runtime-linkage/README.md) for the directed
rounding defect and its correction; symbol attribution alone does not establish
which implementation is faster.

The initial forced-QP (`CTModeRun=4`) profile in this directory was stopped
before evaluation completed. Its provenance records termination. It does not
provide a comparable normal-rescue time or successful physics result.

Reproduction uses the retained isolated generated outputs:

```sh
source .ninjars-build/dev-env.sh
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python docs/benchmarkfolder/experiments/mp-qcd4-point81/normal-profile.py --backend ninja
python docs/benchmarkfolder/experiments/mp-qcd4-point81/normal-profile.py --backend ninjars \
  --parent .benchmark-work/outputs/pre-final-candidate/qcd-4
python docs/benchmarkfolder/experiments/mp-qcd4-point81/repeat-profile.py --backend ninja
python docs/benchmarkfolder/experiments/mp-qcd4-point81/repeat-profile.py --backend ninjars
python docs/benchmarkfolder/experiments/mp-qcd4-point81/partition.py repeat-qcd4-point81-ninja
python docs/benchmarkfolder/experiments/mp-qcd4-point81/partition.py repeat-qcd4-point81-ninjars
```

The scripts refuse to overwrite existing experiments. They require the
recorded generated source trees and perf tool, and only one Symbolica process
may run at a time under the available license. Input cards and points, generated
source manifests, executable hashes, exact child commands and exit codes are
retained with each run. Raw perf data remains under `.benchmark-work/logs`;
the small derived evidence is versioned.

An isolated [master-cache prototype](../master-cache/README.md) tested reuse
within one full amplitude. No values survived into the next phase-space point
or a benchmark repetition. Its qcd3 diagnostic showed about 87% MP row hits but
only modest MP savings and a DP regression. The follow-up
[allocation-free lookup prototype](../master-cache-v2/README.md) also establishes
no performance benefit. Neither is promoted.

These qcd4 sample percentages must not be presented as a measured qcd3 MP
partition. Cache hit counts are not weighted by integral cost, and removing
the complete measured tensor bucket would cap this profile's speedup around
1.17. Another optimization needs a direct profile of the affected qcd3 rescue.
