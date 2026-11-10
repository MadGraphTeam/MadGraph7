# Saved rank-two EW threshold contraction audit

The retained private512-bit TIRust rank-two tensor is consistent with its supplied-invariant recurrence. This extends the earlier rank-one consistency check. It does not resolve the full tensor disagreement, certify an amplitude, or identify the cause/group of the current EW status420 refusal. Production guards and all original evidence remain unchanged.

The audit reads only closed historical bytes. It covers ordinary point1/group5 and threshold point2/group5. Private TIRust512 and NinjaRS256 box values were returned at113bits; pinches come from the original TIRust256 trace. All three merged Laurent slots are tested. No reducer, compiler, loader or native library was invoked.

For `D_i=(ell+q_i)^2-m_i^2` with metric `(+---)`, the saved masses already are squared. Let `M1=-integral(ell)`, `M2=integral(ell ell)`, `I0` the scalar, `p_j=q_j-q_1`, and `k=ell+q_1`. Then:

```text
J = -M1 + q1*I0
K = M2 - q1*M1 - M1*q1 + q1*q1*I0
fq_j = p_j^2 + m1^2 - mj^2
2*p_j.K + fq_j*J = J_pinch_j - J_pinch_1
```

The ten symmetric Cartesian rank-two components are `tt,tx,ty,tz,xx,xy,xz,yy,yz,zz`. Component weights are unity; offdiagonal entries receive no extra factor2. Finite is raw4, merged single pole raw1+raw2, and double pole raw3. No real projection or amplitude-level `2/(16*pi^2)` is applied, and NinjaRS finite already includesR1.

To evaluate the formal supplied-S contraction, decompose `K = g*A + P*B*P^T` using a cofactor normal covector `n` orthogonal to all three `p` vectors: `A=(n*K*n)/(n^2)`. Recover `B` using exact Gram duals and retain reconstruction residuals. This is component decomposition, not a D-dimensional trace identity. With `Zs_ij=S_1i+S_1j-S_ij` and `fs_j=S_1j+m1^2-mj^2`, the formal first contraction is `2*A*p_j + sum_a P_a*(Zs*B)_ja`; only the momentum term changes geometry. The metric term stays `2*A*p_j`.

Pinch vectors are rigorously regathered from the retained triangle invariant coefficients. Recover each triangle coefficient vector `c_i` from `-M1_triangle+qtri_1*C_i=P_tri*c_i`, retaining unused-coordinate residuals. In box order `pi`, use `J_pinch_i=sum_{r=2,3}(qbox_pi(r)-qbox_pi(1))*c_i,r + (qbox_1-qbox_pi(1))*C_i`. This explicitly substitutes box coordinates; it never treats tiny mismatched snapshots as an exact common translation.

| Omitted denominator | Triangle group | Box order | Routing eta |
|---:|---:|---|---:|
|1|8|(2,3,4)|+1|
|2|196|(3,4,1)|+1|
|3|4|(2,1,4)|-1|
|4|16|(1,2,3)|+1|

All eight signed affine routes have nonzero retained defects. The triangle S matrices, squared masses and scale match exactly after permutation. Reversal is handled through the recovered coefficient basis and new box vectors; a second odd sign would be wrong.

Both contraction geometries below use the SAME retained TIRust-S triangle coefficients after regather. Regather does not change the triangle recurrence to q geometry. Therefore the q rows are conditional residuals, not an independently q-consistent physical identity or a NinjaRS error measurement. No bubble RHS is inferred or substituted.

| Point | Backend | Box contraction | Slot | Max absolute residual L2 | Max residual / largest uncancelled term |
|---:|---|---|---|---:|---:|
| 1 | TIRust private512 | Cartesian q | finite | 3.301449e-19 | 1.913089e-17 |
| 1 | TIRust private512 | Cartesian q | single | 4.308321e-20 | 1.472525e-17 |
| 1 | TIRust private512 | Cartesian q | double | 0 | undefined (all termszero) |
| 1 | TIRust private512 | formal S | finite | 8.932849e-36 | 1.379591e-34 |
| 1 | TIRust private512 | formal S | single | 2.504705e-36 | 1.406546e-34 |
| 1 | TIRust private512 | formal S | double | 0 | undefined (all termszero) |
| 1 | NinjaRS256 | Cartesian q | finite | 8.770912e-19 | 3.111242e-18 |
| 1 | NinjaRS256 | Cartesian q | single | 1.117952e-19 | 2.295116e-18 |
| 1 | NinjaRS256 | Cartesian q | double | 0 | undefined (all termszero) |
| 1 | NinjaRS256 | formal S | finite | 1.124547e-18 | 1.878168e-17 |
| 1 | NinjaRS256 | formal S | single | 1.487059e-19 | 1.472525e-17 |
| 1 | NinjaRS256 | formal S | double | 0 | undefined (all termszero) |
| 2 | TIRust private512 | Cartesian q | finite | 0.0001671377 | 1.650298e-17 |
| 2 | TIRust private512 | Cartesian q | single | 4.520039e-06 | 1.472525e-17 |
| 2 | TIRust private512 | Cartesian q | double | 0 | undefined (all termszero) |
| 2 | TIRust private512 | formal S | finite | 1.568226e-21 | 7.742238e-35 |
| 2 | TIRust private512 | formal S | single | 2.952444e-24 | 4.809191e-36 |
| 2 | TIRust private512 | formal S | double | 0 | undefined (all termszero) |
| 2 | NinjaRS256 | Cartesian q | finite | 0.000454044 | 2.241589e-17 |
| 2 | NinjaRS256 | Cartesian q | single | 1.227911e-05 | 2.000126e-17 |
| 2 | NinjaRS256 | Cartesian q | double | 0 | undefined (all termszero) |
| 2 | NinjaRS256 | formal S | finite | 0.0004804463 | 2.371935e-17 |
| 2 | NinjaRS256 | formal S | single | 1.307545e-05 | 2.129841e-17 |
| 2 | NinjaRS256 | formal S | double | 0 | undefined (all termszero) |

The threshold TIRust formal-S finite/single normalized maxima are about `7.74e-35` / `4.81e-36`, versus `1.65e-17` / `1.47e-17` for Cartesian-q contraction against the same S-based pinches. Absolute q residuals are `1.67e-4` / `4.52e-6` amid cancellations. This supports supplied-S recurrence consistency for the saved degree-two components. NinjaRS q residuals cannot isolate a box error because the RHS has not been independently converted/evaluated in q algebra.

`summary.json` retains all72 vector residual rows, exact-zero RHS flags, term/RHS scales, undefined ratios, triangle reconstruction residuals, and full box-decomposition residuals. No tolerance is used to declare an accuracy pass. Exact rational arithmetic preserves rounded binary128 snapshots; it cannot recover their unrounded native values. Norms use80decimal digits; displayed rational components use55significant digits.

Massless point2/group24 is not audited: exact S/mass/scale matching over all successful rank>=1 trace triangles finds no pinch for omitted1 or3. Omitted2 matches191; omitted4 has25/26 and permutation ambiguity. These missing vectors are not fabricated.

A cheap metadata-only bubble availability check also leaves the triangle-vector S-to-q follow-up incomplete: all four triangles lack an exact saved bubble match for omitted3, and triangles8/16 also lack omitted1. Available matches are triangle8 omit2→7;196 omit1→197/198 andomit2→199/200;4 omit1→197/198 andomit2→7;16 omit2→199/200. This agrees with the parent independent scan. The present numerical audit does not use those bubbles.

Three external contraction equations leave transverse tensor freedom. Full rank-two agreement, an independent scalar/pinch oracle, finite-amplitude accuracy, current-source stability acceptance and performance remain unqualified. Increasing precision alone does not reconcile suppliedS and shift geometry. The next justified step is a separately scoped provenance-complete pinch/geometry diagnosis; this packet provides no basis for relaxing the production Gram guard.

Reproduce from the required repository checkout:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-threshold-rank2-geometry-audit-1/run_pure.py --check
```

This blocks native/process APIs, runs ten synthetic tests, recomputes the historical summary and compares it byte for byte. All required existing input paths, byte counts and SHA256 values are in `summary.json`; source excerpts and independent review receipts are included. No large raw files are duplicated.
