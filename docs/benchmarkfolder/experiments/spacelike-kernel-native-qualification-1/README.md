# Synthetic qualification of the integer spacelike certificate

The GNU Fortran 15.2.0 build produced the expected decision for all 4,287
retained DP/QP and invalid-input bit cases: 1,572 accepted and 2,715 rejected.
Every accepted case satisfies `raw < 0` and `|raw| > exact_error_enclosure`.
The retained run reports no reference mismatch or sticky-flag comparison
failure. This packet contains source, synthetic inputs, command/output
receipts and a subsequent independent pure review. It contains no executable,
object, module, compiler or scientific library binary.

This is a conditional algebraic certificate test. In randomized cases `raw`
is generated independently of `q`; the harness does not establish that a
physical vector is spacelike. A production caller must prove the relationship
between raw arithmetic and its vector inputs, valid component/source-DAG error
bounds, pair classification and environment contracts. The source-DAG bound
factory, amplitude caller, tensor coefficients and full amplitudes remain
unqualified. No tensor or amplitude was evaluated here, and the elapsed times
are command accountability, not performance measurements.

## Source identity and provenance limits

The preserved source proposal index is
`1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964`.
Its earlier `Fortran_compiled: false` and source-only labels are copied
verbatim and describe that proposal's creation stage. This later qualification
compiled its exact `certificate.f90`, SHA256
`0cb41b0c10233027d65c85f63625556eefb819ce2570390447bdceff4d8f7088`.
The exact Python reference is
`7b52f707d16b5e248160b8fb7df0a162be41285d02a4cea4d5ae1fe2bac22de1`.
The original source manifest is retained for these two selected source pins;
this packet does not duplicate its other unrelated proposal payloads.

`independent-review.json` binds the qualifier, Fortran driver, kernel,
reference, synthetic inputs and all command/result text. The original
`qualify.py` imports the reference before authenticating a source seal, and
its original summary does not hash the qualifier, driver or oracle. The later
review verifies their retained bytes and source-manifest match. It does not
claim that an oracle seal was enforced before the native run. Recorded
compiler/executable hashes remain historical identities; portable verification
does not reopen those binary paths.

## IEEE and coverage scope

The driver seeds all 32 combinations for `IEEE_ALL`, the five standard sticky
exception flags: invalid, divide-by-zero, overflow, underflow and inexact.
Both widths have accepted controls for every seed, and all retained cases
report that the flags after the kernel call equal the requested initial flags.
The transcript does not separately emit the flag count/support metadata or
an immediate pre-call flag readback. It tests no rounding-mode, halting/trap,
FTZ/DAZ, denormal-status extension or whole-environment preservation claim.

The kernel itself contains only integer arithmetic and bit inspection. The
cases include nonfinite encodings, signed zero, subnormals, maximum finite
encodings, unavailable-contract branches, invalid widths and invalid DP upper
bits/sign extension. Random normal values use powers of two; the sample is
not exhaustive bit-pattern or compiler qualification. The qualifier checks
all 4,287 output lines and sequential IDs, so an early end of the Fortran
input loop could not silently pass this completed attempt.

## Portable verification

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/spacelike-kernel-native-qualification-1/verify.py
```

The verifier needs only Python's standard library. It reads only this packet,
authenticates the selected source pins and all retained text, and independently
decodes the integer128 carriers into exact binary64/binary128 rational values.
It reconstructs the conservative power-of-two threshold and the exact
conditional enclosure, checks every native decision and output ID, and
compares with five selected pure arithmetic functions from the sealed Python
reference. It executes no reference module initialization, qualifier,
compiler, loader, native executable, tensor library or amplitude. No original
external pathname or live binary is opened. `inputs.txt` and `run.stdout`
use deterministic gzip; their original hashes and byte counts remain in
`evidence-index.json`.

`capture.py` documents the one-time selection of explicit reviewed text
artifacts. It is not part of portable verification. All original private
records, source proposal and production code remain unchanged.
