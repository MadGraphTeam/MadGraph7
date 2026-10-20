# Constant-term reuse: source-only follow-up

This is a separate source study. It implements no updater, changes no production
code, and supplies no numerical or timing claim. The existing derivative-row
prototype and its oracle evidence are unchanged.

Let `H(j,k)=COEFF(j,1,k)` and `U=H*A`. Both retained FFV sources give the identity
`B0=R*H`, with

```
R(s,p,m) =
 [ p0+s*p3    s*p1+i*p2       s*m              0       ]
 [ s*p1-i*p2  p0-s*p3           0              s*m      ]
 [ s*m           0         p0-s*p3        -s*p1-i*p2   ]
 [ 0             s*m      -s*p1+i*p2        p0+s*p3    ]
```

The routing and signs are source-specific:

| Source | `s` | Momentum passed to R | Mass |
|---|---:|---|---|
| `MP_FFV1L1_2` | −1 | returned `P2=P1+V3%P` | `M2` |
| `MP_FFV1L2_1` | +1 | returned `P1=P2+V3%P` | `M1` |

`analyze_b0.py` uses the sealed exact Gaussian-rational parser and retained source
snapshots to check each of the 16 matrix entries per orientation, the four
momentum-routing assignments, and the absence of the width argument from these
coefficient expressions. It also enumerates all 16 input-component support masks
under six explicit mass/momentum restrictions. Its JSON output is the
machine-checkable source proof; no floating-point arithmetic is used.
The retained run passes all 32 identities. Its output SHA256 is
`cc0b83ba3192003c38836eaac94c29801aca819170ab9aa1f59aae0ae33168b6`.
A read-only reproduction from the repository root is:

```sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization/b0-source-study/analyze_b0.py |
  cmp - docs/benchmarkfolder/experiments/symmetric-factorization/b0-source-study/analysis.json
```

The existing updater arguments do not contain `p` and `m`. A future private
helper would need the already returned `COMPLEX*32 P_out(0:3)` and the original
`REAL*16 M_out`, or a separately considered fused vertex/updater interface.
Recomputing `P_out` from incoming momentum and the wavefunction would introduce
another rounding step. Recovering `p` or `m` by inverting `H` is inappropriate:
`H` can be singular or have sparse/cancelled entries. The width is not used by
these particular polynomial numerator sources; this does not generalize to
other vertices or complex-mass conventions.

The generated QCD3 call sites already retain these arguments. For example,
`mp_coef_construction_1.f` passes `PL(0,5)` as the returned momentum before its
updater at lines 105–106; the L1 counterpart passes `PL(0,33)` at line 213.
Massive bottom/top examples pass `MDL_MB` and `MDL_MT`. These observations refer
to the retained b44-generated source, not a new generated interface.

Counting product slots after literal zeros gives a mixed result. The counts
below include the current derivative `U` work and the constant term, per
coefficient. Full counts are per cut component; terminal counts sum all four
diagonals. The same support counts apply to both orientations.

| Assumptions | Full: existing → `R*U` | Terminal: existing → `R*U` |
|---|---:|---:|
| Dense A, generic momentum, nonzero mass | 24 → 20 | 32 → 36 |
| Dense A, generic momentum, zero mass | 16 → 16 | 24 → 24 |
| Dense A, transverse momentum zero, nonzero mass | 24 → 16 | 32 → 32 |
| Only `A1` nonzero, generic momentum/mass | 6 → 8 | 8 → 12 |
| Only `A1,A3` nonzero, generic momentum/mass | 12 → 16 | 16 → 24 |

The terminal penalty has a concrete cause: each diagonal currently needs two
rows of `U` for its derivative terms. Its mass term needs a third row in the
opposite block. Computing that row adds two H×A products for dense input, then
the constant row uses three products instead of four. The extra U work exceeds
that saving. Full helpers already compute all four U rows.

These slot counts assign equal cost to complex and real scalar products.
Exploiting the real mass would reduce the cost of the four mass scalings.
If the returned momentum is additionally real, diagonal weights are real and
transverse weights have a special form. Such specializations are separate
implementation questions; neither the `COMPLEX*32` declaration nor these counts
establish that they apply to every generated call. Existing A/B zero pruning
can remove much of the apparent dense saving. Conversely, numerical cancellation
in U can create additional zeros. The prior physical operation trace does not
separate constant-term support, derivative-row support and A masks, so it cannot
resolve the actual savings of this follow-up.

There is also a new numerical issue. Exact source algebra does not imply
`stored B0 == R(stored p,m) * stored H`: the original ALOHA expressions construct
and round B0 independently. Replacing B0*A by R*U changes that coefficient
formation error as well as accumulation order. Original zero B0 entries may
result from cancellation that a reassociated computation does not reproduce.
The current derivative-row runtime guard proves nothing about this new
constant-term discrepancy. Any later experiment would need its own admission
and exact higher-precision reference design, unchanged physical accuracy and
stability gates, and complete source/parameter rebinding controls.

The source identity is useful, but these counts do not support a blanket B0
extension. In particular, terminal and sparse/massless cases can erase or reverse
the prospective gain. The current derivative-only implementation should remain
the separate evaluated candidate.
