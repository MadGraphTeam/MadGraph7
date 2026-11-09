# EW threshold adjudication

This independent diagnostic retains the exact two-point stream from
`../physical-stress/points/ew_n3_epem_ttxa/selected.dat`: an ordinary control
and the previously rejected threshold/soft-photon point for
`e+ e- > t t~ a [virt=QED]`. Its parent SHA-256 is
`f1663e764a65d5b9ca89691177f3607ba85f3daff8b52af9360b1a4f09e26de3`.
The earlier campaign and its accuracy failures remain unchanged. None of the
work here is a performance measurement or a checked numerical acceptance.

The historical native bridge is copied privately with SHA-256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
This permits a diagnostic replay independently of later installer publication.
`runtime-audit.json` records the pre-run comparison with every historical
static archive, shared library, and installation metadata identity. The private
library location is verified by the executable's actual dynamic-loader output.

## Geometry and backend inputs

The completed `geometry/` probe links a private driver to unchanged generated
numerical objects and calls the actual `SET_MP_PS` and kinematic-matrix helpers.
It performs no amplitude, tensor, or scalar reduction. The printed decimal
point is retained; source hashes and exact physical/numerical cards accompany
the probe.

- Repair reduces the largest top mass-shell residual from `2.88e-12` to
  `1.58e-29 GeV²`; four-momentum conservation reaches `2.07e-32 GeV`.
- The generated QP mass reader promotes a DP parsed mass. Its QP square exceeds
  the promoted DP squared mass passed to the reducers by `8.84e-13 GeV²`.
  Four times this difference is `0.543%` of the repaired pair threshold excess
  `s_tt − 4m_t² = 6.5113e-10 GeV²`.
- Representative repaired top invariants are snapped to the DP squared mass
  for `OSThres=1e-8` and `1e-14`, but retain the QP square for `1e-24`.
  Lowering the threshold can therefore change the mass-shell convention; it
  cannot be treated as a neutral precision improvement.
- The scale differs between a promoted DP square (Ninja/NinjaRS) and a QP
  square of the promoted scale (TIRust) by `2.78e-17` relatively. This is
  recorded separately from the much more sensitive threshold invariant.

`geometry/source-audit.md` documents the generated CutTools denominator
downcast: a genuine QP numerator is paired with repaired momenta rounded back
to DP before CutTools promotes them internally. Ninja, NinjaRS, and TIRust
receive QP momenta. The audit also records scalar on-shell classification
differences; it does not claim that an untraced scalar row triggers them.

The completed `ct-qp-transport/` replay retains the original scalar cutoff
`1e-6`, QP numerator, point stream, and numerical policy. It preserves the
original DP entry and uses a private QP denominator entry only in QP mode.
The ordinary control agrees within `1.12e-14`. At the threshold, the double
pole moves into agreement with Ninja/NinjaRS (`3.62e-16` relative), but the
finite coefficient shifts by `0.298%` and the single pole becomes
`−1.42959e8`. Fourteen instability warnings are retained. This experiment
demonstrates sensitivity to momentum transport; it does not identify a correct
finite result or resolve the original threshold discrepancy.

## Native tensor trace

`run.py prepare`, `build`, and `run` create an isolated historical export and
add logging around the QP TIRust native call. Only the driver and generated
TIRust adapter objects change; the logger is a new private object. Tensor
arithmetic, contraction, physical parameters, and the original numerical card
remain unchanged (`CTModeInit=1`, `CTModeRun=4`, forced checks disabled).
The ordinary fixed-QP status is unchecked `120`.

Each native call records its status. The first call for each group records
shifts, squared masses, supplied invariants, scale, and successful tensor
outputs. Failed calls never read undefined tensor output. End-of-point output
records both FPE flags and actual repaired QP momenta. FPE flags may persist
from warmup, so native per-call status supplies the direct failure evidence.
Early adapter returns before the native call are outside this hook.

The completed trace reproduces the historical final values **byte for byte**.
All 200 ordinary-control groups succeed. At the threshold, 163 groups succeed
and **37 return native `UNSTABLE`**, each reporting an ill-conditioned Gram
matrix. The generated adapter returns zero contributions for these rejected
groups and sets the QP FPE flag. Thus the historical forced-QP result with
double pole `−9839.38785` is an incomplete amplitude. It is invalid as a
coefficient reference; its closeness to another backend's finite coefficient
does not establish agreement. The automatic host rejection was appropriate.
The ordinary control agrees with the universal double-pole relation to
`1.95e-15` relative. The threshold's incomplete double pole differs by 51.3%.

The measured FPE flag is also true on the ordinary control because it persists
from stress-point warmup. The ordinary point's 200 successful native statuses
distinguish this inherited flag from a new tensor failure.

The native small-Gram guard compares a conditioning estimate raised to the
subset coefficient rank against the workspace threshold. The historical
bridge fixes that threshold at `2^−136`. Increasing `MP_BITS` alone would
therefore leave this particular rejection criterion unchanged. No higher
precision build or guard modification has been performed. The prepared
`tensor-replay/` diagnostic will compare a bounded selection of captured
tensors with NinjaRS using the exact same QP inputs, without treating failed
TIRust output as numerical zero. The original complete-amplitude threshold
accuracy issue remains unqualified.

The existing adapter returns a zero group and sets the QP FPE flag after a
native failure. Positive fixed-QP mode bypasses the automatic stability path,
so `120` alone does not prove a complete amplitude. `analyze.py` requires the
traced final values to reproduce the historical raw values exactly before
summarizing failures and checking the universal double pole.

The initial attempt in `failed-setup-attempt/` stopped before evaluation due to
an extra diagnostic card in the working directory. It selected the wrong
resource directory. The corrected attempt uses the historical resource-card
location. Explicitly labeled stale historical output copied during failure
cleanup is not new numerical evidence.

## Reproduction

From the repository root, with an empty ignored diagnostic work directory:

```sh
source tools/tirust_env.sh
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/geometry/run.py prepare
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/geometry/run.py execute
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/geometry/analyze.py
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/run.py prepare
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/run.py build
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/audit_runtime.py
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/run.py run --cpu 9
python docs/benchmarkfolder/experiments/ew-threshold-adjudication/analyze.py
```

Preparation requires the frozen historical generated exports and library
versions identified by the manifests. It refuses to overwrite existing
evidence. Native TIRust execution uses the shared numerical lock and has a
180-second deadline for each worker phase. Lock waiting itself is not a
numerical timeout. The geometry-only probe does not need that lock.
