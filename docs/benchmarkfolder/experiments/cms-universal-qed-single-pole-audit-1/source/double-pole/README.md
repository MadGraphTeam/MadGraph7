# CMS forward-point universal QED pole audit

Pure source/card reconstruction supports the private candidate9 QP **double
pole** at point20. It does not adjudicate its finite or single-pole result.
The four external leptons are massless and have unit squared charge. The
MadGraph FKS pole implementation gives

```
A[-2] = -4 * Born * alpha / (2*pi).
```

The selected Gmu complex-mass model has GFRedefinitionChoice=1 and CMSParam=1.
Writing A=muW^2*(1-muW^2/muZ^2), its existing definitions simplify to
alpha=sqrt(2)*Gf*abs(A)/pi, muV^2=MV^2-i*MV*GammaV. This is the existing model,
not a switch to a different alpha scheme. The actual parameter card is copied;
source excerpts and full-source/card hashes are retained. Values are parsed
through binary64 and widened, matching the external model-input boundary.
The calculation uses 70-digit Decimal arithmetic on the mathematical model
expression. It does not emulate all Fortran complex rounding, capture runtime
coupling bits or infer extra precision for the input parameters.

The reconstructed alpha is
0.007550411468681858048317751575476993451734746224223126999454643546653719.
For Born=0.033864286950528170450080978071127901785075664520263671875,
the expected pole is
-0.0001627768643257003036111856006445099428957082284571128306347909847886557.
Candidate9 QP differs by 2.679e-16 relatively. Original CutTools QP differs
by 0.4999905864, corresponding to an effective sum of squared charges
2.00003765 instead of four. Both ordinary bootstrap results obey the same
normalization. `result.json` retains all digits and both order columns.
This is a universal double-pole consistency result, not complete-amplitude
truth or checked stability. Fixed return120 is unchecked. Finite/single-pole
accuracy, the rejected candidate DP result and all other failed gates remain
open. No failed tensor buffers are used.

## Two distinct zeroing policies

CutTools initialization requests an absolute OneLOop on-shell cutoff1e-6.
The vendor wrapper forwards that value to DP, QP and MP. In the QP triangle
routine it zeros small classification magnitudes, while retaining the
original numeric invariant arguments. The QP box routine initially does so
for four external invariants, and attempts a permutation when a channel is
small. This is not blanket mutation of every supplied invariant. The resolved
spacelike transfer near -2.5e-7 GeV^2 is below that cutoff, making changed
scalar IR classification a plausible explanation to investigate. This audit
does not identify a particular actual scalar call or establish that it causes
the observed CutTools answer. Increasing QP arithmetic alone does not change
this explicit scalar cutoff.

Separately, the unchanged MadLoop shared DP/MP kinematic helper zeroes an
invariant when abs(DIFFSQ/REF_NORMALIZATION)<OSThres, where normalization uses
the two chord energies. Source excerpts show that **Ninja6 and NinjaRS8 both
call that same unchanged helper**, then reconstruct and pass its invariant
matrix. Their QP shift vectors retain their own component precision. Thus
their first legacy-policy comparisons are valuable counterchecks, but are
not automatically references for the candidate9-only spacelike policy:
the same resolved transfer can still be projected away in their supplied
scalar matrices. No native6/8 input or amplitude is captured by this audit.

## Future consistent-geometry comparison

Keep the current candidate9-only experiment and legacy helpers unchanged.
First retain the planned original-policy6/8 reference results and actual
input matrices, shifts, masses and scales. If they disagree, distinguish
source-policy differences from numerical reduction differences.

A separate, explicitly labeled private follow-up could supply6/8 with the
same structurally classified, certified spacelike entries used by9, while
preserving all protected on-shell assignments, physical point, masses,
numerators, scale policy and precision settings. Use direct invariant values
where needed to avoid mass-subtraction roundtrip loss. Capture exact native
inputs in every backend and state scalar-provider cutoff differences. This
is a matched-input experiment, not approval to change production templates,
global OSThres or scalar cutoffs. Any separate CutTools cutoff experiment
must retain its default-policy control and report the policy change.

Compare all Born/finite/single/double coefficients and the universal pole,
then run MadLoop's independent stability checks. Neither backend agreement,
absence of warnings nor a universal double-pole pass establishes finite or
single-pole truth. No native experiment or code/model/card change is made here.

## Reproduction

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/cms-universal-qed-pole-audit-1/verify.py
```

Portable verification reads only this packet and replays the pure calculation.
`calculate.py --verify-live-sources` additionally rehashes the explicitly pinned
closed source/card/result files. Neither command compiles, loads a scientific
library, starts an amplitude or changes an existing artifact.
