# Final candidate MG7 regression binding and genericity refresh

This read-only proof binds the existing broad MG7 regression campaign to exact
candidate commit `c2b48e988fe6e3c6ba9403b67ec4d26fa5fa9ccb`. It reruns no test,
starts no compiler and evaluates no numerical engine. Its result is reuse of
unchanged tested coverage, not fresh Rust or amplitude validation.

`result.json` records matching Git-object content for all 102 production paths in
`policy-v2/final-regressions-v1/validation.json`, including exact resolution of
its three source symlinks. The original seal, fixture review and report adapter
independently pass without changing old evidence. All 1,097 checked test, fixture,
source, log and verifier files also match the immutable candidate commit.

The reused campaign has **90 cases: 89 passed and one explicit existing GoSam
Python 3 bypass**. It covers polynomial generation, compiled Fortran updates and
IEEE guards, HELAS calls, loop exporter IO including loop-induced generation and
eight split-order variants, FKS IO and all five supported serial-versus-multicore
NLO generation comparisons. It excludes the Rust reduction and scalar-integral
engines. The new final Rust executable, actual installation, relocated numerical
exports, amplitude agreement/stability and performance campaigns are separate
gates. These 90 cases are not added to overlapping historical suite counts.

## Bounded genericity refresh

`genericity.json` preserves the earlier sealed 42-source census and extends its
exact baseline-to-candidate scope to 47 paths. Thirty-nine of the old paths are
unchanged. The three changed old paths are:

- `ninjars_install.py`: the recoverable filesystem publication/cleanup fix, whose
  final production and test hashes match the passed 37-test transaction proof.
  This code does not choose amplitude or tensor paths.
- `lib.rs`: two new module declarations, both explicitly `cfg(test)`.
- `tensor.rs`: the measured generic allocation changes and three explicitly
  `cfg(test)` oracle/capture modules.

The five new paths are the precision/panic/allocation/capture test modules and
frozen dense test oracle. Their only inclusion routes are checked `cfg(test)`
module declarations. They are excluded from production dispatch scans.

The tensor production change expands the pinned complex product into the same
four scalar products and real subtraction/imaginary addition, borrows its
operands, lazily grows scratch, and resets existing finite zero/one storage via
`set_from`. Its 25 existing conditional/shape-selector source lines are unchanged
in order. No file, environment, process, stored fixture or expected-output source
is consulted by the production tensor. Rank, Laurent shape, lane and dimensions
remain its general inputs. The source review finds no new process/model/fixed
point/benchmark dispatch; the inherited pattern census has zero new production
matches. This structural check does not prove numerical equivalence, arbitrary
model correctness or runtime performance. Those require the separately retained
numerical and measured controls.

## Verify

From the repository root with the established Python environment:

```sh
source .ninjars-build/dev-env.sh
taskset -c 4 python -B \
  docs/benchmarkfolder/policy-v2/final-regression-binding-v1/verify.py
```

The verifier checks this package's seal, recomputes the 102-path/1,097-file Git
binding through the old read-only report adapter, and recomputes the bounded
47-source census. It writes no evidence or bytecode. The candidate revision is
fixed even if a later documentation commit exists. Changing any tested source,
fixture, log or reviewed live source makes verification fail rather than silently
retargeting the old proof. The old genericity and regression seals are unchanged.
