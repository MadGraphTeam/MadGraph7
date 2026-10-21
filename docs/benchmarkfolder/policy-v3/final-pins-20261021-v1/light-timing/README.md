# Light Ninja/NinjaRS timing

The latest user request replaces the unexecuted 13-hour campaign with a small
study. This ancillary full-sum sample has exactly six cases: QCD1, EW1, QCD2,
EW2, loop-induced ggH and HH. Each uses the first ten fixed events of its original
100-event stream, identically for literal original Ninja6 and final NinjaRS8.
Seven adjacent alternating pairs retain full pole, helicity/color sum, stability
and rescue settings. There is no optional 2→3 full-sum gate. Actual MG7 CLI
profiles are the main benchmark.

The unchanged Fortran worker and `pair.run_pair_case` perform evaluation.
Only selection, provenance and actual-count validation are new. Generic answer
dimensions and numerator source ordering remain unchanged; the driver is the
only allowed generated-source difference. Point prefixes are bound back to their
original byte streams. The actual ten-point scope, parent seeds, runtime, raw
outputs and backend/rescue counts remain explicit.

The minimum measurement command is 0.5 seconds, approximately 42 seconds total
across twelve targets and seven pairs, plus calibration, initialization and clean
compilation. A 120-second phase timeout and 600-second evaluation/build budget
retain a failed partial attempt instead of allowing an hour-long initialization.
The budget uses SIGTERM through the existing deferred ownership path; it does
not weaken numerical thresholds or omit rescue. Pure provenance checks occur
outside that evaluation budget.

`scope-amendment-v1.json` preserves the previous plans as history and withdraws
them from future execution. No old full-eight-process 1.05 runtime gate is a
completion requirement. The new result reports medians, dispersion, confidence
intervals and the 1.05 reference comparison honestly; it cannot establish passage
of the withdrawn campaign. The completed eight-case 100-point correctness
capsule remains valid and separate.

After source review, the root orchestrator must retain actual release evidence
for every other numerical, profiling and compilation job and create a matching
quiet CPU207 handoff. The new launcher accepts only the light specification; it
reuses the reviewed ownership function without invoking either retired stage.
All four evaluation thread limits are one. No automatic retry, append or restart
of an existing attempt is supported.

```sh
source .ninjars-build/dev-env.sh
taskset -c 6,7,207 python3 -B \
  docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/launch.py \
  --spec docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/specification-held-v1.json \
  --handoff /absolute/path/to/actual-light-root-quiet-handoff.json \
  --attempt docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/light-timing/launch-v1 \
  --execute-numerical
```

This command is a reproduction recipe, not authorization to launch. The handoff
must bind the exact sealed specification, CPU207, the new light phase, an explicit
quiet declaration, and nonempty bound release evidence.

The required protocol's ordinary HH calibration supplies the ten-point normal
anchor for retained focused controls. It is an actual ten-point evaluation, not a
relabeled 100-point result. The later comparison may read those records without
extra backend6/8 calibration calls or numerical work during quiet measurements.
