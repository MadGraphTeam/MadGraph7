# Standalone pole diagnostics

The ordinary amplitude benchmark always evaluates finite and pole coefficients.
Use the separate tensor diagnostic to exercise `TIRustComputeFinite = .FALSE.`:

```sh
source tools/tirust_env.sh
python3 tools/tirust_benchmark_poles.py \
  .benchmark-work/tirust/outputs/candidate/ew_n4_epem_mupmumaa \
  --label ew_n4_poles --precision both --loops 1
```

The input is a standalone process built with `tirust_benchmark.py build`, with
its numerical objects, compiler recipes and link dependencies recorded. The runner clones that
export into `.benchmark-work/tirust/pole-diagnostics`, adds a diagnostic driver,
and calls the generated DP/QP tensor interfaces directly. It never calls an
amplitude routine with an incomplete finite term.

Fixtures come from the retained tensor-validation inputs in
`vendor/ninjars/tests/poles/tensor_cases.json`. For each supported fixture, the
driver evaluates both UV+IR poles, UV only, and IR only, first with finite
evaluation enabled and then disabled. Every call receives fresh kinematics;
numerical caching is disabled and cleared between calls. Partial finite
coefficients must explicitly be NaN; all evaluated pole components must be
finite and agree with full evaluation. The comparison uses decimal arithmetic
on the complete Fortran output, including QP digits.

Results and raw output are retained under
`docs/benchmarkfolder/experiments/poles-diagnostics/<label>`. Fixtures exceeding the generated
process's tensor-rank capacity are listed as omitted, and complete qualification
is false whenever an omission or failed fixture remains. A high-rank export is
therefore needed to qualify the entire fixture set.

The retained compilation record binds the diagnostic executable, copied generated
sources, effective compiler recipes, static and dynamic libraries, ABI metadata,
and the physical parameter card. Source and library bytes are checked again
after evaluation. Exact driver templates and fixture streams accompany the raw
output, so these diagnostics do not rely on a later checkout's protocol code.

The retained per-call times are diagnostics in a fixed full-then-poles order.
They do not enter the amplitude performance acceptance gate. Evidence that
finite kernels/arithmetic are skipped comes from the separate pole-kernel
instrumentation and implementation tests; a NaN mask alone is not that proof.

The recorded final-release runs and their explicit DP unavailable cases are
documented in `experiments/poles-diagnostics/README.md`. A separate cache runner
compares cached and uncached generated-adapter tensors, validates UV/IR and
DP/QP isolation, and exercises byte limits and context lifecycle:

```sh
python3 tools/tirust_benchmark_cache.py \
  .benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z \
  --label qcd_n1_cache
```

Both runners serialize native evaluations using the shared numerical-test lock.
