# Focused tensor replay — source/protocol preparation

No native compilation or evaluation has run. Input selection remains pending
the parent trace's `trace.json`, summary and successful historical-value check.
`prepare.py` performs only filesystem reads, hashing and input preparation; it
contains no compiler or subprocess calls.

The proposed replay isolates individual tensor coefficients. It does not
reconstruct a whole amplitude, rerun phase-space repair, alter scalar-library
thresholds or change working precision. Captured shifts, invariants, complex
squared masses and scale are passed directly to both backends.

The default bounded selection is the first failed stress group, the same group
at the ordinary point, and one successful stress control (prefer the same
propagator count and rank). An explicit list of unique `point:group` values can
replace it, with at most three cases. Each case evaluates one TIRust tensor and
`binomial(rank+4,4)` NinjaRS unit numerators, at most 105 NinjaRS calls for this
generated rank-three process. Selection records all group identities, native
statuses/errors and exact captured decimal strings. Repeated calls with
inconsistent statuses for the same captured group cause an explicit rejection.

The Fortran driver reads the strings directly into binary128 and echoes every
input with the same `ES45.35E4` format used by the parent trace. The analyzer
requires exact canonical token equality, including signed zeros, without
conversion through Python floats. Both bridges retain binary128 inputs and
evaluate at their pinned 256-bit MPFR precision.

TIRust uses full evaluation with both pole types, `check_pass=.FALSE.`, cache
disabled and a zero cache budget. It clears the context cache before each case
and records QP cache counters. Failed native results remain errors and are
excluded from numerical comparison. Successful tensors retain all four complex
slots `[UV1, IR1, IR2, finite]`.

For each zero-based MadLoop polynomial index `i`, the direct NinjaRS API receives
one nonzero coefficient `(-1)^degree(i)`. The TIRust raw result already includes
that sign through `Convention::MadLoop`; `TensorWeights::Components` adds no
multinomial factor. The independently checked 35-entry generated permutation
maps MadLoop indices to raw IREGI rows. Compare the complete complex values

```
[RAW(map(i),4), RAW(map(i),1)+RAW(map(i),2), RAW(map(i),3)]
```

with `NRS_RES(0:2)` before any real projection or `2/(16*pi**2)` normalization.
NinjaRS's finite result already includes R1; its separately recorded R1 must not
be added again. The scalar and degree-one/two/three blocks contain 1, 4, 10 and
20 coefficients respectively.

Both engines use the exact captured TIRust scale for this comparison. Production
TIRust computes the QP square of promoted `MU_R`, whereas production NinjaRS
promotes the DP square; the geometry report retains that original difference.
Using a common scale here is an intentional controlled-input diagnostic.

The historical runtime was frozen before publication of the newer split-pole
wrapper. Its private location is
`.tirust-build/ew-threshold-tensor-replay-runtime`. `runtime-snapshot.json`
records the old `mtirust.mod`, wrapper archive, Fortran source, NinjaRS module
and Fortran source, and bridge DSO. The wrapper hash is
`04c2b1ac3b8275dd2d0060416c4147bebd5e9bd62b71509e4cb29ce3d5c47d08`, and the DSO
hash is `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
Future compilation must use these private module and wrapper files, not the
live TIRust installation. The unchanged NinjaRS static archive remains at
`HEPTools/ninjars/lib/libninjars.a`, hash
`0edfa89d7d83161a6ce3fdf0bb1fbca928b73d2b5ec44211a3b2efa186a155d6`.

Source/protocol preparation from the repository root:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/prepare.py sources
# Only after the parent trace is complete:
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/prepare.py select
```

For a later authorized native phase, compile `replay.f90` in the private work
directory with the historical runtime as the first and only module include
directory. Link the private `libtirust.a`, the pinned NinjaRS static archive,
the historical `libmadloop_tirust.so`, and the recorded NinjaRS native link
requirements. Record the compiler command, object/executable and actual loaded
library hashes. Hold the shared numerical/Symbolica lock, use one designated
core, run the bounded selected stream with a hard wall-clock timeout, and retain
every failure. No compile/run helper is supplied at this preparation stage.

The future native launcher must write `execution.json` with `complete`,
`returncode`, `output_sha256`, `input_stream_sha256`,
`source_manifest_sha256`, `artifacts_sha256` and `loaded_libraries_sha256`.
The artifact map must include the private `replay.o` and `replay`; the loaded
library map must include the historical private DSO, using repository-relative
paths for internal files and absolute paths for external runtime libraries.
Retain the compiler command, execution command, timeout, CPU and lock record
alongside those bindings. The analyzer rejects incomplete or nonzero-exit runs,
changed source/input/runtime hashes, missing unit/slot pairs and any mapped
comparison inconsistent with the separately printed raw tensor. Successful
`END` records alone cannot hide a later finalizer failure.

Interpretation limits: TIRust and NinjaRS share oneloopmaster ancestry with
different pinned revisions, so agreement is not independent scalar validation.
Unit tensors contain neither the original numerator coefficients nor their
amplitude sum; the current parent trace does not capture those coefficients.
Absolute and complex relative differences must both be retained, especially
near zero. Runtime and numerical acceptance remain outside this diagnostic.
