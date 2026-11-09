# Production DP terminal qualification

This directory records source and numerical qualification of the generated
binary64 terminal updater. It is separate from the frozen isolated experiment
in `../dp-terminal/` and from any acceptance timing campaign.

All five fresh CLI exports passed the complete source proof. The selected
terminal updates were applied as follows; no selected update was declined:

| Process | Applied updates |
| --- | ---: |
| QCD `u u~ > z` | 1 |
| QCD `g g > t t~` | 10 |
| QCD `u u~ > z g g` | 26 |
| EW `u d~ > w+` | 6 |
| EW `e+ e- > mu+ mu-` | 35 |

The freshly compiled QCD n1 output passed the source proof again after the
benchmark driver was installed. Its 100-point CutTools replay, with
`MLStabThres=1e-6`, matches every retained reference coefficient bit for bit;
all 100 statuses are DP. `validation-summary.json` links the raw records and
binds the exact production and test sources. The n1 compiled textual source
closure is copied in `n1-compiled-source/`; native artifacts are represented by
their recorded hashes, not included in this evidence directory.

The final proof/exporter tests passed 46 tests and 117 subtests. The final
compiled IEEE run passed five tests and 42 subtests, including 1,536,000
admission/output/flag checks, 32 paired trap modes, and the actual injected
caller followed by CREATE/MERGE. `compiled-final-tests.json` binds the source
hashes before and after that run to its raw log. The earlier combined
regression log (68 tests, 80 subtests) also covers the existing QP terminal
and polynomial behavior; its earlier scope is distinguished in the summary.

The initial production scope is Born-interference output from the ordinary
standalone optimized Fortran exporter, rank 1 × rank 1, and four-component
wavefunctions. Custom exporter subclasses, MadEvent/FKS and loop-induced
outputs, vectorized outputs and other shapes retain
the existing dense calls. The HELAS graph selection is followed by an
independent completed-source consumer and alias proof. Failed or unrecognized
proofs restore the original batch bytes. The private routine still performs
the full output initialization and calls the unchanged dense updater whenever
its IEEE admission guard refuses.

The qualification exports each process separately. A combined multi-subprocess
export can conservatively retain dense calls when the source closure contains
another polynomial coefficient table; this initial proof does not attempt to
resolve arbitrary table bindings across multiple process modules.

`capture_cli.py` invokes the actual `bin/madgraph` CLI and observes the existing
finalizer before and after its normal call. It does not change graph selection,
proof decisions or generated statements. On a refusal it repeats the read-only
parser to record the reason. `run_qualification.py` records fresh exports for
five main processes, and can build and replay the n1 CutTools process against
the retained strict-policy 100-point baseline. The benchmark driver is also
subject to the completed-source consumer proof after its source build.

Commands, from the MadGraph checkout after sourcing `tools/tirust_env.sh`:

```sh
PYTHONDONTWRITEBYTECODE=1 taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-terminal-production/run_qualification.py
PYTHONDONTWRITEBYTECODE=1 taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-terminal-production/run_qualification.py --case qcd_n1_uux_z --build-n1 --observe-n1
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-terminal-production/verify.py --live-sources
```

After staging the evidence, add `--tracked` to verify the staged Git bytes as
well as filesystem contents. The verifier checks portable file hashes and
source/test bindings; it does not rerun numerical or performance tests.

These runs are numerical/source qualification only. Their elapsed times do not
establish a performance result. A new formal campaign must use newly generated
candidate sources; the original c91 n1 qualified performance failure and both
isolated guard variants remain preserved in their original evidence folders.

`initial-proof-hardening/` preserves an initial CLI attempt rejected when the
validator source changed during export. No numerical result or final-source
qualification is claimed from that attempt. The rejected working output was
moved to the location recorded in its `attempt.json`; recorded original paths
remain unchanged for provenance.

That initial attempt also inherited configuration-saving CLI commands. The
actual log attribution and observed configuration bytes remain separately in
`configuration-attribution/`; no missing previous configuration was
reconstructed or restored. The final five exports use `--no_save`, each with
an unchanged before/after configuration hash. These final guards are linked
by `configuration-attribution/final-no-save-followup.json`.

The build/replay invocation captured its exact runner in `protocol/`.
`benchmark-helper-packaging-observation.py` is explicitly a packaging-time
copy, not a claim of a snapshot taken at the start of CLI generation. The
generation phase retains its actual CLI commands, generated-source identities
and decisions. No acceptance timing result is inferred from any elapsed time
in this folder.
