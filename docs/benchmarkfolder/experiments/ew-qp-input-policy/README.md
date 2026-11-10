The narrow QP correction worth qualifying is to construct squared propagator
masses from the existing MP model parameters and carry those values through a
QP-only reduction boundary. Keep the existing physical-input interpretation,
phase-space repair, on-shell-aware matrix helper, thresholds and refusal policy.
This removes an avoidable binary64 intermediate; it does not certify the
threshold amplitude or make finite-precision momenta exactly on shell.

This is a research proposal, with source review and arithmetic on already saved
data only. No compiler, reducer, new amplitude, altered-input integral, or
production change is included. The earlier failures remain qualification gaps.

Three separate effects must not be conflated:

| Effect | Fixed evidence and policy |
|---|---|
| Parameter input rounding | For this found card entry, the MP reader reads a `REAL*8` temporary and promotes it. Thus the frozen mass is `173.300000000000011368683772161602974`, not the exact rational `1733/10`. Keep that value. Reading the decimal afresh at higher precision would change the frozen model input. Missing-entry defaults follow a separate path. |
| Existing QP momentum repair | `SET_MP_PS` promotes the original momenta and calls `MP_IMPROVE_PS_POINT_PRECISION`. Its saved threshold output reduces the largest shell residual from `2.8838e-12` to `1.5777e-29 GeV²`, but changes a component by as much as `2.2692e-7 GeV` and the pair excess by `0.07125%`. Preserve this actual transformation and its input/output records; do not claim the original rounded vectors and repaired vectors are identical physical data. |
| Additional squared-mass rounding | QP reduction currently receives the already binary64-squared mass, `30032.8900000000030559021979570388794 GeV²`. Squaring the unchanged promoted mass in QP gives `30032.8900000000039403857954312117200 GeV²`. The extra difference, `8.8448359747e-13 GeV²`, is avoidable and is separate from momentum repair. |

The input mass's binary64 rounding is unavoidable **for this frozen input
contract**, not a fundamental limitation of model readers. This mass's exact
dyadic square fits in the 113-bit significand, so the QP square itself is exact
for the represented mass. It still differs from the square of the originally
spelled decimal. `audit.py` records all these quantities separately.

The contradiction is local and explicit. For a repaired external top momentum
appearing as `q_i-q_j`, the shifts give

```text
(q_i-q_j)^2 = square(promote(m_DP)) + O(binary128 shell roundoff),
```

while the current on-shell clamp supplies

```text
S_ij = promote(square_DP(m_DP)).
```

These differ by about `8.8448e-13 GeV²`, rather than the observed `1e-29` shell
residual. Both equalities cannot simultaneously describe the exact same
denominator geometry. In saved threshold group 5 this changes the 3×3 Gram
determinant by approximately `1.01847%`. Small absolute input discrepancies can
therefore have visible tensor consequences without any scalar or tensor
arithmetic bug. The native normalized-determinant guard is a heuristic, not a
certified error estimate.

A coherent QP policy is:

1. Preserve the parameter card, its existing binary64 external-parameter values,
   the original momentum stream, and the existing repair result. QP numerator
   construction uses that same repaired point and the existing MP model.
2. Bind each squared-mass source to the existing model and repair semantics
   before choosing the QP expression. For this card-supplied stable top,
   `MP__MDL_MT` equals promotion of `MDL_MT`, and repair explicitly uses
   `MP__MDL_MT`; squaring that promoted mass in QP is the narrow compatible
   correction. This equality is not general: a derived real mass can already
   be recomputed through MP model relations. Such a case must use the same
   existing MP particle-mass symbol as QP repair/numerator, or be explicitly
   excluded from this correction; silently substituting either a newly parsed
   decimal or a promoted DP-derived value is not justified. In a complex-mass
   model, the internal complex pole-mass expression is separate from the real
   external mass used for repair. Preserve that distinction and the existing
   MP width/model convention; never repair real momenta onto a complex mass
   shell or infer a mass from an already rounded squared mass.
3. Carry that QP squared mass through every QP formal argument, ordering helper,
   matrix construction and reducer transport. Preserve the DP entrypoints.
   Merely changing a generated call expression is unsafe while the callee still
   declares `M2L` as `COMPLEX*16`.
4. Keep `MP_BUILD_KINEMATIC_MATRIX` and its on-shell rules. It already accepts
   `COMPLEX*32` masses. For the stable real on-shell case, its snap target then
   agrees with the mass used by the MP model and repair, up to the unavoidable
   QP momentum representation error. A genuine complex pole mass is not that
   real external-shell target; retain the helper's existing complex comparison.
   Preserve the massless/on-shell classification rather than feeding raw tiny
   shell residuals into scalar integrals as artificial off-shell regulators.
5. Record residual consistency between the supplied invariant matrix and the
   fixed shifts; keep independent host stability tests and native refusals.
   If a consistency residual is amplified beyond the requested tolerance, a
   successful library status alone cannot qualify the amplitude. This proposal
   does not define or relax a new acceptance threshold.

The separate scale mismatch is much smaller: the historical QP wrappers do not
all agree on whether `MU_R**2` is squared before or after promotion. A shared
documented QP scale expression would remove that inconsistency, but its measured
relative difference `2.7826e-17` is not the top-pair threshold excess and does not
explain the observed Gram defect. It must remain separately tested.

Thus two implementation scopes must stay explicit. Squaring a promoted DP mass
is the smallest compatible change for the proven external-input mass class in
this fixture. A general generated QP mass path must bind the actual existing MP
mass source, including derived and complex parameters; it is a broader
consistency change and needs separate model coverage. It introduces no new card
parser or parameter relation, but it changes which already-existing model
representation reduction consumes. `source-routing.md` records that boundary.

There is no universal one-line correction that makes all represented identities
exact. Binary128 repair leaves finite shell and conservation residuals; forming
prefix sums and invariant differences adds rounding. Exact on-shell invariants
and exact distances of those finite represented shifts will generally differ.
The smallest identified correction removes the much larger DP squared-mass
intermediate. It cannot establish that remaining discrepancies are harmless for
every severely conditioned tensor. Changing shifts separately in each loop to
fit `S` would also change the numerator's denominator geometry and requires a
coordinated process-level construction. Replacing `S` with raw shift distances,
or repairing to `sqrt(promoted_DP_mass_square)`, would respectively discard the
requested on-shell treatment or disagree with the MP model mass.

The saved-data arithmetic includes a deliberately limited sensitivity check:
replace only invariant entries exactly equal to the old squared-top target by
the QP target, retaining all shifts and other invariant entries. It measures the
remaining distance mismatch and Gram determinants in `summary.json`. This is
**not** a bitwise replay of the rounded matrix helper, and no integral with
these hypothetical inputs was evaluated. A production change would also alter
the propagator masses and consequently the scalar integrals.

The rank-one mechanism is algebraically explicit. Let `p_j=q_j-q_1`,
`Zq_ij=2*p_i.p_j`, `Zs_ij=S_1i+S_1j-S_ij`,
`fq_j=p_j^2+m_1^2-m_j^2`, and `fs_j=S_1j+m_1^2-m_j^2`.
With common scalar box `I0` and common pinched scalars `C_j`,

```text
Zs*cs = C_j-C_1-fs*I0
Zq*cq = C_j-C_1-fq*I0
cq-cs = inverse(Zq)*[(Zs-Zq)*cs + (fs-fq)*I0]
M_q-M_s = -P*(cq-cs),        M = q_1*I0-P*c
```

For compatible operator/vector norms this gives

```text
||cq-cs|| <= ||inverse(Zq)|| *
            (||Zs-Zq||*||cs|| + ||fs-fq||*|I0|).
```

The independently frozen geometry supplement applies this exact identity to
saved coefficients and reduces the threshold rank-one finite/single differences
from approximately `0.58%/0.56%` to `1.70e-34/1.39e-34`. Separate saved scalar
pinches verify the two corresponding denominator identities. This explains a
representation difference with the **same held-fixed scalars**. Correcting the
actual masses and invariants changes those scalars too; their variation is an
additional right-hand-side term. Higher ranks additionally involve metric
coefficients, tensor pinches and dimensional/Laurent terms. Consequently neither
the identity nor its saved-data agreement is a proof for rank two/three, the
finite amplitude, numerator contraction, or host acceptance.

This also leaves the existing DP Born versus repaired-QP loop normalization
question separate. A consistent loop-input representation does not retroactively
evaluate the saved Born term at the repaired point; the earlier independent
pole-ratio check was useful precisely because the Born normalization cancels.

Qualification of a coordinated source correction would require generated
real/complex-mass ABI checks, all QP paths and numerator options, unchanged DP
results, exact saved input/repair bindings, regenerated per-group matrices and
all tensor ranks, independent analytic poles, direction/rotation tests, genuine
automatic stability acceptance, and ordinary/stress regression evidence. The
original threshold point must remain in the record, and all new results would
be a separately labeled candidate. No majority vote is an accuracy certificate.

Reproduce the research arithmetic and verify its inventory from the repository
root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-qp-input-policy/audit.py
python3 -B docs/benchmarkfolder/experiments/ew-qp-input-policy/freeze.py
# Optional: reconstruct all35 reviewed sources without installed libraries.
python3 -B docs/benchmarkfolder/experiments/ew-qp-input-policy/freeze.py --sources
```

The arithmetic rerun requires the unchanged historical generated files bound
by the original trace manifest. The inventory check requires only these saved
files. `source-excerpts.txt` preserves numbered excerpts and full source hashes;
`summary.json` binds the original evidence and distinguishes the hypothetical
target substitution from measured native results.
