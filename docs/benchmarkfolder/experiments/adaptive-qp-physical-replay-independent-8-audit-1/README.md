# Independent backend8 comparison of the completed adaptive diagnostic

This additive audit compares the already completed genuine-MP-mass
fixed-QP NinjaRS (backend8) diagnostic with the single completed adaptive
TIRust (backend9) diagnostic. It executes **only frozen Python text parsing**.
It does not rerun either native process or modify earlier evidence.

Both points have identical physical point/model/MP-coordinate inputs,
helper masses, repaired coordinates, numerator coefficients, ordered
momenta and masses. The seven nonselection input/resource files agree
exactly. The complete MadLoop parameter cards differ only by explicit
backend selection (8 versus 9). Born flat indices 1 and 5 agree exactly at
both points. All 400 native calls in each journal were served; neither console
contains W03 omissions.

Final finite/single-pole/double-pole sums agree to a maximum symmetric relative
residual of `2.6841120946502901e-16` at point1 and
`2.0282415803300141e-16` at point2. The exact binary64 words, decoded values
and residuals for all 16 output entries are retained.

The native integral input encodings are **not identical at point2**: all 200
groups have a different `NATIVE_SCALE`. The frozen backend8 adapter computes
`REAL(MU_R**2,QP)`, squaring in double precision before promotion. Backend9
computes `REAL(MU_R,QP)**2`, squaring after promotion. Shifts, kinematic matrix,
invariants and squared masses match bitwise at every group of both points;
native scale also matches at point1. All 200 scale discrepancies and both
exact word sequences remain explicit in the result. The frozen adapter
source is included for the scale expressions. Backend8's native numerator
also has its documented odd-rank convention; physical numerator coefficient
parity is checked before that interface transformation.

This is independent diagnostic agreement with the same physical inputs and
an explicit native scale-encoding difference. It is **not** a claim of
identical native integral input data at point2, independent numerical truth,
checked MadLoop stability, per-call adaptive precision, or runtime
qualification. Both public fixed-mode status120 results remain unchecked.
The earlier same-input 363-call preservation and 37-call recovery result
retains its separate scope and unchanged original controller failure.

The two antecedent evidence indices are pinned in this package's seal.
All text required to reproduce this audit is stored in
`text-inputs.json.gz`, with original path and byte identities in
`text-inventory.json`. Verification replays only offline parsing:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-physical-replay-independent-8-audit-1/audit.py
```
