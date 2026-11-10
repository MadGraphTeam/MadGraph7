# Local precise timing handoff — implementation draft

Hua-Sheng will run the precise final performance campaign locally, potentially
with an agent. This handoff records the required work and the existing protocol;
it does not claim that local portability or local measurements are complete.
The current shared-host provisional proposal never qualifies the 1.05 bound.

## Preserve the inputs and previous results

Keep the whole Git history, frozen baseline commit
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`, all 100-point files, UFO model pins,
cards policy and retained failures. In particular the historical `c91cb85f3a`
QCD 2→1 comparison remains a qualified failure for that implementation:
ratio 1.1210500947672746, paired interval [1.1034925280600616,
1.138887018281026]. A different host does not erase or replace it.

The current candidate manifest is
`docs/benchmarkfolder/campaigns/split-dp-terminal-1/manifest.json`, SHA256
`1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31`.
It records the shared-host installation and absolute artifact paths. Do not edit
those paths, relabel its old observations, or try to resume its compiled outputs
on a different machine. The final portable handoff must bind the eventual frozen
source revision separately from locally rebuilt library and executable bytes.

## Required local preparation

1. Check out the delivered source revision in a clean local clone, with all
   historical evidence retained. Rebuild the pinned comparison libraries and
   TIRust/oneloopmaster with the supported GNU Fortran environment. Preserve
   complete compiler versions, effective flags, lockfiles, runtime dependencies,
   installed ABI metadata and native hashes. Reuse neither shared-host `.o`
   files nor a guessed relocated installation.
2. Create a new isolated local measurement registry and ignored build directory.
   Import the preserved source campaign and failure ancestry as read-only
   evidence. This needs a reviewed additive launcher: the existing registry
   intentionally rejects duplicate numerical identities and its paths are not
   portable. Do not weaken that protection or delete historical manifests to
   make registration pass.
3. Freeze local source/native identity, exactly the existing point/model/card
   policy, `PYTHONHASHSEED=0` for new CLI exports, and MG `set --no_save`
   configuration handling. Preregister worker deadlines of 14,400 seconds for
   the two n4 cases and 3,600 seconds for every other case. Record one allowed
   local CPU, thread limits and the local timing conditions before starting.
4. Generate every baseline and candidate process through the real MG CLI and
   build all numerical objects locally. The original baseline source is fixed;
   baseline and candidate effective numerical compiler flags must match. A
   local pre-existing baseline may be reused only with the existing exact
   source/object/dependency and byte-identical old-driver relink proofs.
5. Run the eleven complete numerical checks, baseline CutTools plus candidate
   1/6/7/8/9, with the existing `--warmup-only` route. It requires the final
   complete steady 100-point traversal and does not manufacture timing data.
   Preserve refusals, 3e-6 failures, 1e-8 target misses, raw statuses and every
   unsuccessful initialization. No timing phase may imply that these checks
   passed if they are missing.

## Required precise measurements

Use the existing seven-pair protocol with every timed batch traversing all 100
saved points an integer number of times. Sustain calibration, target two
seconds, retain the registered 0.95-second duration floor, alternate paired
execution order, and report initialization separately. Preserve the independent
MadLoop checks, full helicity/color sums, UV/IR poles, rescue chain and
`MLStabThres=1e-6` for both sources. No finite-only COLLIER profiling default.

First run all eight main TIRust-first chain gates against each process's
materialized baseline default chain. Each upper 95% paired confidence bound
must be <=1.05. Retain every failed gate and return nonzero before expensive
existing-backend regressions. Once all eight gates pass, run existing-backend
1/6/7/8 regressions and descriptive standalone backend9 and control timings.
The same speed bound is required for main existing-backend regressions;
backend9-vs-CutTools and control ratios add no new speed-acceptance criterion.

Keep actual final accepted backend/precision histograms; these are not counts
of every internal attempt. Native-DP/rescued diagnostic cohorts, when reported,
need their own frozen full-stream classification and remain distinct from the
whole-stream acceptance timing. Do not select a favorable retry after an
eligible failure; preserve the existing preregistered retry rules and ancestry.

## Existing command interfaces and remaining launcher TODO

The following are existing, supported core interfaces. `LOCAL_ID` must first
be registered by the reviewed local-registry launcher; running these directly
against the old shared-host manifest is not a portable bootstrap procedure.

```sh
source tools/tirust_env.sh
python3 -B tools/tirust_benchmark.py build --campaign LOCAL_ID --source both --jobs 4
python3 -B tools/tirust_benchmark_numerical.py \
  --campaign LOCAL_ID --case qcd_n1_uux_z \
  --label full100-warmup-qcd_n1_uux_z-1 --warmup-only \
  --backends 1 6 7 8 9 --cpu LOCAL_CPU
# Repeat the numerical command once for each of the eleven frozen case IDs.
python3 -B tools/tirust_benchmark.py campaign --campaign LOCAL_ID --cpu LOCAL_CPU
python3 -B tools/tirust_benchmark.py summarize --campaign LOCAL_ID --cpu LOCAL_CPU
```

Before final delivery, replace this draft command section with the tested
launcher invocation for registry creation, baseline/model acquisition, local
library configuration, all-eleven numerical supervision, precise timing and
report reproduction. The adapter must map every command to the local registry;
merely printing the core commands above would not do that. Its tests must reject
changed bindings, omission of historical failures, favorable same-host reruns
and any promotion of provisional shared-host results into a local gate.

The final benchmark PDF must distinguish measured shared-host diagnostics,
completed ordinary numerical qualification, unresolved stress limitations and
pending or completed local precise performance gates. Local timings are an
explicit TODO until their retained raw observations have been independently
validated.
