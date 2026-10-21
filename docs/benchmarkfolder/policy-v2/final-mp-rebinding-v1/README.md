# Direct QP mass and scale rebinding in generated Fortran

This separate diagnostic exercises the public `CTModeRun=4` path with
`UseQPIntegrandForNinjaRS=true`. It retains the exact ten fixed momentum
configurations and changing top-mass/scale schedule from the passed ordinary
`final-variant-controls-v1` retained-context and finalized-context controls.
The corrected updater assigns both independent DP and MP model variables
before the generated `COUP()` call.

The only Fortran source change in each private clone is the public test-driver
call `FORCE_STABILITY_CHECK(.TRUE.)` becoming `.FALSE.`. The actual installed
standalone QP proof in `../install-export-abi3-final-v1/runtime_validator.py`
already uses this behavior for explicit modes1/4; it enables forced stability
only for its separate stable-mode argument. Keeping forced stability enabled
in explicit mode would make the THRES wrapper label every call as
initialization140, preventing the existing warm-up loop from terminating.
The retained ordinary sources and measurements remain unchanged.

The generated HH `loop_matrix.f` selects the card's mode after initialization.
The expected return120 establishes that full QP was performed. The accuracy
value -1 is the documented absence of an automated stability estimate in
this direct mode. Neither value is treated as automatic rescue, a stability
acceptance result, or a performance measurement. The focused raw parser
requires exactly120 and -1, complete ordered point/split-order rows and finite
coefficients; it does not change the ordinary benchmark validator.

Two private drivers are relinked from the passed ordinary outputs. Their
one-line diffs, original and modified sources, compiled manifests,
executables, parameter cards and complete reduction descriptors are retained.
Scalar-integral thresholds and every normal benchmark setting remain fixed.
The context-retained/context-finalized QP values are compared with each other
and their already passed ordinary DP values using the unchanged
`controls_driver.compare_records` tolerance and common point scale.

```sh
source .ninjars-build/dev-env.sh
taskset -c 190 python -B docs/benchmarkfolder/policy-v2/final-mp-rebinding-v1/launch.py prepare \
  --authorization 'Root approved separate public CTModeRun4 HH state rebinding 2026-10-21'
taskset -c 190 python -B docs/benchmarkfolder/policy-v2/final-mp-rebinding-v1/launch.py run \
  --authorization 'Root approved separate public CTModeRun4 HH state rebinding 2026-10-21'
```

Each operation has its own six-hour bound and records PID, start ticks,
process group, child commands, result, failures and child-exit checks under
`ledgers/`. These are additional focused diagnostics on the final runtime;
they do not alter normal inputs or enter the runtime acceptance gate.
