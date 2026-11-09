# EW4 initialization timeout audit

The original `strict-policy-ew-n4-warmup-v2` attempt remains unchanged. Its
baseline CutTools and candidate CutTools, Ninja, COLLIER and NinjaRS lanes each
retain a complete 100-point observation. All pass the `3e-6` comparator against
baseline CutTools. Ninja, COLLIER and NinjaRS miss the stricter ordinary `1e-8`
target against that reference; this audit does not adjudicate those differences.
Both CutTools lanes use 93 DP and 7 QP evaluations. The other completed lanes
use 100 DP evaluations each. Their timings remain diagnostic and ineligible
for the full-amplitude runtime gate.

TIRust times out before READY and has no completed observation stream. Its
132 retained coefficient rows represent **66 completed public initialization
calls**: bootstrap call 0 on point 1, followed by calls 1 through 65 on points
1 through 65. There are 24 calls with status `219` (TIRust DP) and 42 with
status `323` (TIRust QP). The bootstrap is separate from the 65 subsequent
calls; point 1 appears twice. These partial
initialization values are not a qualified 100-point accuracy result, and the
trace does not establish completion of the next call.

The coordinating launch supplied `--timeout 1800`. The original specification
omits this argument and its failure records encode only a protocol timeout,
not a duration. The index therefore identifies 1800 seconds as session-reported
command provenance, rather than inferring it from file timestamps or pretending
it was recorded in the frozen specification.

The frozen worker template reconstructs the recorded native driver hash. The
five Python/Fortran protocol copies, all attempt files, physical points/cards,
generation commands, this audit and its parser helper are hashed in
`ew-n4-warmup-v2-index.json`. Each lane records a worker rebuild using earlier
compiled numerical objects; this audit preserves that distinction. Historical
binary and library identities are read from retained manifests, not from the
mutable installation. No original files are modified and no native code runs.

Reproduce the audit from the repository root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/numerical/audit_ew_n4_warmup_v2.py
```
