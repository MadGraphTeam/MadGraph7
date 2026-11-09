The saved rank-one threshold discrepancy is explained by two different
geometric representations of the same supplied request. TIRust's private
512-bit result satisfies the recurrence built from the authoritative invariant
matrix `S`. NinjaRS satisfies the denominator identity built from the exact
binary128 momentum shifts. Transforming the former recurrence into the latter
reproduces the recorded NinjaRS rank-one coefficients to approximately `1e-34`,
including the ordinary and massless controls. This is a diagnosis of rank-one
representation consistency, not a full-amplitude accuracy result.

This supplement is separate from the frozen `ew-threshold-adjudication`
package. It uses only saved coefficients and pure Python rational/decimal
arithmetic. No compiler or reducer is called, no integral is evaluated with
altered inputs, and no original evidence or production source is changed.

For denominator `D_i=(ell+q_i)^2-m_i^2`, use metric `(+---)` and define
`p_j=q_j-q_1`, `I0` as the scalar box, `C_i` as the scalar triangle omitting
denominator `i`, and the recorded MadLoop rank-one moment `M=-integral(ell)`.
The shifted vector is `J=-M+q_1*I0=P*c`; the tiny nonzero captured `q_1` is
retained. The two systems are

```text
Zq_ij = 2*p_i.p_j                 fq_j = p_j^2 + m_1^2 - m_j^2
Zs_ij = S_1i + S_1j - S_ij        fs_j = S_1j + m_1^2 - m_j^2

Physical denominator identity:   2*p_j.J + fq_j*I0 = C_j - C_1
Supplied-invariant recurrence:   Zs*c + fs*I0 = C_j - C_1
```

The source routing is explicit. In the materialized TIRust
`crates/iregi-rust/src/direct/exec.rs:1535–1547`, `f` comes from supplied
distances while Cartesian vectors come from the shifts. At `1624–1642`, the
Gram solve uses supplied distances. Its inverse and `Z^-1*f` enter the
rank-one recurrence in `direct/plan.rs:1116–1152`. Cartesian gathering and
origin translation use the shift vectors, and the final MadLoop convention
changes odd-rank signs. `source-excerpts.txt` preserves the relevant numbered
excerpts and complete source-file hashes; the replay's Fortran numerator sign
is included too.

The frozen input audit already measured a maximum `S` versus exact-shift
distance mismatch of `8.8448359747e-13`. For threshold group 5, the
supplied-invariant Gram determinant is `1.2226858389e-13`; deriving it from
the fixed shifts changes it by approximately `-1.018469%`. Both systems are
severely conditioned. These are properties of the recorded inputs, not new
native requests or an accuracy bound derived from a condition heuristic.

`audit.py` recovers `c_s` from the saved TIRust Cartesian vector using the
invertible coordinate minor with largest absolute determinant, preserving the
unused-coordinate residual. It then computes, entirely as exact rational
arithmetic,

```text
c_q = inverse(Zq) * [Zs*c_s + (fs-fq)*I0]
M_q = q_1*I0 - P*c_q
```

The inferred pinched-scalar right-hand side is unchanged. The following
maximum is over the four Cartesian components, using the complex metric
`abs(a-b)/max(abs(a),abs(b))` with zero–zero set to zero.

| Threshold group 5 slot | Recorded TIRust versus NinjaRS | Geometry-transformed TIRust versus NinjaRS |
|---|---:|---:|
| Finite | `5.7966283e-3` | `1.6972959e-34` |
| Single pole | `5.5559986e-3` | `1.3860105e-34` |
| Double pole | `0` | `0` |

The same transformation matches the ordinary group-5 and threshold massless
group-24 controls at at most `2.09e-34` by this metric. Component scales and
absolute differences are retained in `summary.json`; a small component's
relative difference is not an amplitude-error estimate. The unused-coordinate
reconstruction residual for threshold group 5 is at most `1.38e-33` absolute.
Exact arithmetic preserves the recorded binary128 coefficients, not their
unrounded 512-bit values.

`pinch.py` supplies a separate check using scalar triangles already recorded
in the original trace, without inferring their right-hand sides from the box
vector. It verifies their native success and exact supplied-invariant,
squared-mass and scale correspondence, including propagator permutations:

| Omitted box denominator | Saved triangle group | Triangle order in box indices |
|---:|---:|---|
| 1 | 8 | `(2,3,4)` |
| 2 | 196 | `(3,4,1)` |
| 3 | 4 | `(2,1,4)` |
| 4 | 16 | `(1,2,3)` |

The maximum threshold finite/single residual divided by the largest uncancelled
term is:

| Recorded box vector | Exact-shift denominator identity | Supplied-invariant identity |
|---|---:|---:|
| TIRust private 512-bit | `1.65030e-17` | `6.54641e-35` |
| NinjaRS 256-bit | `6.55563e-35` | `1.66954e-17` |

For TIRust in the exact-shift identity, the maximum absolute finite and single
residuals are `9.64441e-7` and `2.60822e-8`; for NinjaRS in the supplied-invariant
identity they are `9.75685e-7` and `2.63897e-8`. These equations have large
cancellations, and some right-hand sides are exactly zero. `pinch-summary.json`
therefore retains the absolute residual, right-hand-side scale, uncancelled-term
scale, and explicit zero flags rather than treating a ratio to zero as accuracy.

Together the two checks explain the rank-one disagreement through the input
geometry used by the algebra. They do not establish an independent scalar
oracle: the saved triangles use TIRust's scalar machinery, and the two bridges
share oneloopmaster ancestry. They do not adjudicate rank-two/three tensors,
the finite amplitude, numerator contraction, host stability acceptance,
performance, or a production policy for reconciling `S` with shifts. The
original historical-256 TIRust refusals and all original failed points remain
unchanged. More internal precision evaluates the supplied representations
more accurately; it does not make their geometries identical.

Reproduce from the repository root with the materialized pinned source tree
listed in the source manifests:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-threshold-geometry-supplement/audit.py
python3 -B docs/benchmarkfolder/experiments/ew-threshold-geometry-supplement/pinch.py
python3 -B docs/benchmarkfolder/experiments/ew-threshold-geometry-supplement/freeze.py --check
```

The source/identity audit was reviewed independently by `complex_stress`; the
saved-pinch calculation was developed separately by `geometry_probe_audit`.
`evidence-index.json` binds every file in this supplement. It can be checked
without installed reducers, compilers or the ignored source tree.
