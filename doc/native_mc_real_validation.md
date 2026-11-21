# Validating the native MC@NLO real provider

The subprocess-local `madevent_mintMC` executable owns each process's sampling, physical FKS histories and event output. A proof of real batching must observe multiple independent MINT draw IDs and different momenta in the same real provider call. SIMD padding and a group of calls with one physical point each do not satisfy that requirement.

Generated NLO real dispatchers select `NLO_REAL_STRICT=1` when building MadMatrix.
For GNU-compatible CPU compilers this uses `-O3 -fno-fast-math` for scalar and
SIMD code, including the shared model library. Ordinary LO output keeps its
existing fast-math default; GPU device flags are unchanged. The strict profile
has a separate build tag, so existing fast-math objects must be cleaned before
rebuilding that output. Use the generated `nlo_real.mk` dispatcher: building an
individual MadMatrix directory without its NLO settings, overriding `CXXFLAGS`
or adding `-Ofast`/`-ffast-math` can invalidate numerical qualification.

A bounded scalar-double smoke of `u u~ > e+ e- [real=QCD]`, with built-in CTEQ6L1
PDFs, 650 GeV proton beams, seed 314159 and integration/generation capacities of
three, generated 12 events and passed the ownership trace gate in all three MINT
modes. At the component tolerance stated below, sampled native comparisons gave
111 failures among 12,671 checks with the former fast-math build. Rebuilding all
scalar/common C++ code with `-fno-fast-math` gave 12,679 passing checks and no
failures. These used the same cards and seed, with adaptive trajectories allowed
to differ. This establishes one real-only correctness smoke, not general NLO
numerical coverage or a performance result.

Set `MG7_MC_BATCH_TRACE=1` in the environment of a generated native launch to write a separate `mc_batch_<pid>.trace` file in each job directory. An explicit path is also accepted for a single native job. The writer requires a new file for each stream. The following command checks the generated jobs and their total written event count.

```bash
python tests/acceptance_tests/mc_batch_trace.py --expected-writes 12 \
    /path/to/output/SubProcesses/P*/GF*/mc_batch_*.trace
```

For a scalar control, include `--width-one-control`. In both modes the checker verifies the physical provider's input momentum, flavour and couplings against the submitted request, projects the returned local coupling orders onto their native global slots, and checks the actual consuming point's total and components. Every owner must commit in order after consuming all its requests. In generation, every classified accepted candidate must appear once among the writes and explicitly discarded terminal surplus. A rejected candidate cannot be written. Actual provider calls, native cache hits and model/diagram fallbacks follow different code paths. Only successful multi-owner UMAMI calls contribute to the multi-point threshold in each of modes 0, 1 and 2.

`MG7_MC_DECISIONS=record` and `MG7_MC_DECISION_FILE=<new path>` record the random decisions of a single native job. Replay the same immutable native inputs with `MG7_MC_DECISIONS=replay` and the original ledger path. Use one file per subprocess, channel and MINT phase. Regeneration needs the grid and envelope from the corresponding previous phase and the numeric run-card seed for the separate LHE collection shuffle. The frozen width-one fixtures retain both signed per-channel integrals and ordered event-body checksums. [Width-one controls](../tests/input_files/nlo_mc_batch_reference/README.md)

For low-volume job summaries, set `MG7_MC_BATCH_PROFILE=1`. The `MG7_PROFILE_UMAMI` records give successful physical call counts by MINT mode and logical width. The `MG7_PROFILE_FORTRAN` record counts physical native calls by configured Fortran, required diagram weights, and unavailable acceleration. The `MG7_PROFILE_CLOCK` record contains the Fortran system clock rate and accumulated packing, provider and scatter ticks. The provider clock surrounds the actual UMAMI call and coupling reweighting. Packing and scatter clocks cover the cohort dispatcher. Use separate runs for timing and full momentum traces.

Set `MG7_MC_REAL_ORACLE_PERIOD=N` for an independent native Fortran matrix check at the first resumed offloaded request and every N such requests thereafter. The oracle runs the original local Fortran evaluator on the identical canonical momentum and local FKS row under the reactivated owner model. It compares the physical total and every global coupling-order component. Each job reports PASS or FAIL against an absolute tolerance of 10⁻³⁰ plus 10⁻¹² times the absolute native value, separately for the total and for every component. The reported maximum ratio is the largest absolute difference divided by its own tolerance. The default validation stops on the first nonfinite or out-of-tolerance result. Set `MG7_MC_REAL_ORACLE_RECORD_ONLY=1` to record all failures with the provider and native components while the workflow continues. Such failures still fail numerical qualification. Keep these diagnostics disabled in timing runs.
