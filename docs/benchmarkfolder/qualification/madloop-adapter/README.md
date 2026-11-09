# Generated MadLoop adapter qualification

These are correctness and integration checks, not timing measurements.
`release/manifest.json` records the completed final release checks using DSO
SHA-256 `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
Seventeen combinations pass: eight external-backend/precision runs, six
relocated internal-output runs, and three NLO runs, followed by the NLO
incomplete-amplitude rejection. Both copied installations retain validated
TIRust and NinjaRS dependency notice/provenance bundles. All builds use the
normal generated makefiles; the final subprocess builds also pass with `-j2`.

The earlier `debug/` records identify their temporary debug DSO separately.
Neither folder establishes runtime performance.

All subprocesses are generated `u u~ > z` in `loop_sm`, with QCD virtual
corrections. The standalone tests force CutTools, Ninja, COLLIER, NinjaRS and
TIRust separately; TIRust also runs with reversed propagator order and genuine
quadruple precision (CTModeRun 1, 2, 4 and 5). Each driver performs four evaluations of its fixed two-to-one kinematics.
The finite, single-pole and double-pole results agree to a maximum symmetric
relative difference of **1.29e-15**, far below the comparison threshold 3e-6.
The explicit positive CT modes intentionally disable the numerical stability
estimator; return codes 110/120 describe that choice and do not establish
stability statistics. The larger benchmark campaign covers stability checks.

The internal-output test copies TIRust through the real exporter, moves the
whole generated directory, removes its isolated source installation, builds
at the new location, and executes CutTools, NinjaRS and all four TIRust modes.
The runtime finds the copied DSO with the generated `$ORIGIN` RPATH. Optional
Ninja and COLLIER remain excluded by the existing internal-output policy.

The FKS test uses the real CLI `u u~ > z [QCD]` export with internal dependencies.
It builds the model/helicity libraries and the virtual subprocess's normal
`libMadLoop.a` target, then links its generated `check_sa.f` using that
subprocess's actual `LINKLIBS` and the usual citation routine. CutTools DP and
TIRust DP/QP agree. A second call with sole backend 9 and
`TIRustComputeFinite=.false.` exits with status 1 and explains that the
standalone tensor diagnostic is required. This exercises the ordinary physical
amplitude guard in an NLO-generated entry point. The test edits the runtime
`SubProcesses/MadLoopParams.dat`; its `Cards` copy is not read by this direct
virtual-subprocess driver. This is not a Monte Carlo integration test. Full
Source compilation was unnecessary and requires a configured PDF library.

The ordered-prefix optimization is present in all release outputs and in the
internal and NLO debug outputs; the external debug output records the earlier
triangular-prefix version.
Their results agree. Separate bitwise/IEEE and timing evidence for this small
Fortran optimization is in `../../experiments/fortran-prefix/`.

## Commands and evidence

Run from the MadGraph checkout after `source tools/tirust_env.sh`. Exact CLI
input files, generated-process compilation logs, per-run output, raw result
records and test scripts are retained under `release/` and `debug/`. Absolute
paths identify this qualification workspace. The external release and both NLO
exports use the real CLI. The internal exports and external debug dispatcher
run use an empty `_iregi_dir` to isolate an existing bundled IREGI wrapper that
addresses `$(PWD)/vendor/IREGI/src` during an internal build. The internal
exports otherwise exercise the same command dispatcher and actual exporter.
The generation input files record those commands. The debug source prefix was copied to
`.tirust-build/adapter-export/relocation-prefix` before the internal export.
The release source prefix was copied to `release-prefix` in that same parent;
its export was moved from `release-internal` to `release-relocated`, and
`release-prefix` was removed before compilation and execution. The installed
`HEPTools/tirust` prefix was never modified by these checks.

Representative build commands were:

```sh
make -C .tirust-build/adapter-export/native/SubProcesses/P0_uux_z SHELL="$(command -v bash)" check
make -C .tirust-build/adapter-export/relocated/native-internal/SubProcesses/P0_uux_z SHELL="$(command -v bash)" check
make -C .tirust-build/adapter-export/nlo-internal/Source SHELL="$(command -v bash)" libmodel libdhelas
make -C .tirust-build/adapter-export/nlo-internal/SubProcesses/P0_uux_z/V0_uux_z SHELL="$(command -v bash)"
```

The FKS driver adds this temporary target after including its generated makefile:

```make
adapter-check: check_sa.o
	$(FC) $(FFLAGS) -o check check_sa.o ../../../Source/mg5_citation.f ../libMadLoop.a $(LINKLIBS) -lstdc++
```

`debug/final-unit.log` records 92 passing focused adapter, reader, permutation,
installer, process-check and integration-boundary tests after the prefix change.
`debug/io-final.log` records both optimized `gg_ttx` and `ddx_ttx` exporter IO
cases passing against the reviewed references. `debug/io-legacy.log` adds three
unchanged legacy cases: default `gg_ttx`, default `ddx_ttx`, and loop-induced
`gg_hh`. The separate squared-order IO
fixture is explicitly skipped because `loop_qcd_qed_sm-full` is unavailable
locally. The physical mixed-correction benchmark is a separate validation.
