# Installed Fortran transport success parity

This protocol compares the public `mtirust` Fortran module from two separately
installed archives and shared libraries. It is a bounded correctness diagnostic,
not a generated MadLoop amplitude test or a performance measurement. No native
code has run merely because the input files or this document exist.

The 65 calls per installation cover:

- A rank-2 infrared triangle in DP and QP, including full and direct-pole paths,
  uncached BOTH/UV/IR selection and full-evaluation NONE, real BOTH-mode cache hits,
  independent checks, disabled caching,
  zero-byte insertion refusal, budget restoration, clearing, and finalization.
  Two interleaved DP/QP hits verify isolation while both caches hold tensors.
- A rank-2 massive bubble in both precisions and finite modes.
- A rank-2 massive triangle in both precisions and finite modes. Its known DP
  unstable status is retained explicitly and does not count as a successful
  numerical evaluation. An unexpected error or unsupported status fails.
- A tadpole with an exact last binary128 mass bit, through cold, hit, independent,
  disabled, and clear/recovery calls. The executable checks that bit analytically.

The frozen rank-2 fixture source is identified by SHA-256 in `specification.json`.
Ordinary fixture coordinates enter as binary64 and are promoted for QP, matching
the earlier generated-adapter diagnostic convention; the last-bit tadpole uses
an explicit binary128 expression. The invariant matrix uses the `+---` metric
and is computed in the precision of the current call.

`driver.f90` emits native-order 64-bit words in hexadecimal for every real and
imaginary coefficient, including zeroed refusal output, plus all nine aggregate
and six QP cache counters before and after every call. The runner requires exact
old/new equality of the complete frame files, as well as independently checking
statuses, cache transitions, masks, and full/direct-pole agreement. Semantic
within-installation comparisons use the established tolerances of `1e-10` in DP
and `1e-24` in QP. Exact cross-installation parity does not use a tolerance.
The unsupported poles-only/NONE combination is excluded. The recorded scope is
coefficient bits, statuses, and cache/lifecycle counters; caller-input mutation
and floating-point exception/rounding environments are not measured here.

Preparation (Python only):

```sh
python3 .tirust-build/qp-transport-install/success-parity/prepare.py
```

After the coordinating agent authorizes native execution, run from the main
MadGraph repository root with the qualified toolchain environment:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 .tirust-build/qp-transport-install/success-parity/run.py \
  --old-prefix .tirust-build/current-install/mg/HEPTools/tirust \
  --new-prefix .tirust-build/qp-transport-install/mg/HEPTools/tirust \
  --output .tirust-build/qp-transport-install/success-parity/run-1 \
  --report docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/success-parity \
  --timeout 300
```

Both output directories must be new. The launcher links each installed archive and
DSO explicitly, verifies DSO resolution with `ldd`, hashes all resolved runtime
files and the external fixture source before and after, hashes both installations and
the protocol before and after execution, and records compiler and worker
identities. It holds `.tirust-build/numerical-tests.lock` across the two native
workers. Each has a 300-second default hard deadline; timeout is a retained
failure, never a qualification pass. It never rebuilds or writes either installed
prefix. Binaries remain in the ignored work directory. Compilation output, worker
output, parsed qualification, exact comparison, frozen protocol sources, and
execution metadata are written in the new report directory.
