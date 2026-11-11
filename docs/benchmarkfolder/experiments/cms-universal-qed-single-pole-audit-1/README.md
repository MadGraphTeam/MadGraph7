# CMS universal QED single-IR-pole consistency audit

The private candidate TIRust QP single pole at the retained forward point20
agrees with the source-derived universal QED IR formula to `5.23e-16` relative
residual. Original CutTools QP differs by approximately `52.985%`. Both selected
workers' point1 bootstrap results agree with the same formula. This supports
single-pole consistency; finite and full-amplitude accuracy, checked stability,
performance and production promotion remain unqualified. The parent candidate
comparison gate remains false.

For four massless charged external fermions in
`e+ e- > mu+ mu- [virt=QED]`, the retained FKS source gives the collinear
factor `-6`. Crossing the two incoming legs gives all-outgoing charges
`Q = [-1,+1,+1,-1]`. Charge-linked Born terms supply the soft factor:

```
A[-1] = Born * alpha/(2*pi) *
        [-6 - 2*sum(i<j, Q_i*Q_j*log(abs(2*k_i.k_j)/mu^2))]
```

The FKS source's `fksprefact=.FALSE.` convention adds
`double*log(mu^2/QES^2)`. Since the pair-charge sum is `-2`, this removes the
auxiliary QES scale. The retained calculation checks that cancellation and
the equivalent product of six pair invariants. Independent review recomputed
the factors and all eight residuals at 90 working digits and checked three
auxiliary scales. These are pure algebraic checks, not new scale-varied
amplitude evaluations.

The prediction applies to the intended UV-renormalized full `ANS` interference,
with UV+IR loop selection. Retained generated-source excerpts connect UV
`1/epsilon` counterterm amplitudes to their Born interference, the counterterm
buffer and the full amplitude sum. They establish that counterterms are
included in the path being compared. UV cancellation and the correctness of
every model counterterm were not separately measured. This result must not
be interpreted as a test of a bare tensor's `IR1` slot alone.

## Inputs and their limits

The original 28 files and index are preserved unchanged under `source/`, seal
`501302116affa5358b566282580d1f172a51499d93684625f27e73190094858e`.
Its private-path README and optional live-source commands are historical text.
The public wrapper uses only retained files and the declared siblings below.

Amplitude rows come from the current closed parent comparison: the eligible
original CutTools QP and candidate TIRust QP point20 workers, each with its
point1 bootstrap and two order columns. The wrapper joins all eight rows
word for word to both the parent comparison and the raw driver-bit files.
It does not interpret coefficients from failed workers or the old trace run.

The four complex binary128 external leaves and native `mu^2 = 1000000` come
from the **earlier** closed group3 QP traces for points1/20. Exact rational dot
products of those words feed 70-digit logarithms. The wrapper authenticates
the copied traces, source specifications, closure receipts and recorded input
words against their public parent. The old stress failure remains a failure.
Shell and momentum-conservation residuals of the rounded inputs are retained.

Physical cards, numerical cards and point-stream bytes match the saved trial
and historical capture. Reconstructed binary64 point-stream words are checked
as input identities; they are not fresh native measurements. MadLoop restores
the momentum kinematics before the QP leaves are formed, so the saved input
components must not be equated directly with those leaf words. For example,
the point20 outgoing longitudinal component has extra binary128 bits after
restoration. The current candidate trial made no new external-word capture.
Reusing the historical leaves relies on the documented unchanged momentum
source path and matched inputs, not a newly observed equality of candidate
leaf bits.

The existing Gmu complex-mass alpha reconstruction is copied byte for byte
from the published double-pole audit. Parsing the physical card through
binary64 is an explicit reconstruction assumption: these packets do not
retain the card-reader/MP-widening implementation or runtime coupling bits.
Working digits do not imply measured coupling precision or complete Fortran
rounding emulation. Source excerpts and full-file hashes are historical
provenance; no fresh parsing of every complete source file is claimed.

Neither pole audit diagnoses a particular CutTools scalar call, establishes
the finite coefficient, validates a general backend policy change, nor
requalifies the failed DP and stress gates. Fixed-mode return120 remains
unchecked. No native execution, library loading, timing run or production
source/model/card change belongs to this audit.

## Portable reproduction

Keep these sibling packets beside this directory:

- `../kinematic-invariant-amplitude-qualification-1`, seal
  `c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4`.
- `../current-kinematic-trace-runtime-qualification-1`, seal
  `730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4`.
- `../cms-universal-qed-pole-audit-1`, seal
  `1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77`.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/cms-universal-qed-single-pole-audit-1/verify.py
```

The wrapper authenticates all three public packets before using their records,
checks the ancestry joins and replays the retained calculation and independent
factor/QES algebra. The siblings' own verifiers provide their broader build,
protocol and source checks and need their separately declared dependencies.
Only Python's standard library is used; optional live-source entry points and
recorded executable paths are never invoked.
