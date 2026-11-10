This offline audit confirms that passing the existing MP model masses through
the QP path removes the previously observed mass-square inconsistency in the
threshold group-5 Gram matrix. It does **not** remove the native conditioning
refusal: the same 37 threshold parent groups are refused in both arms.

The analysis uses only the four frozen `execution-1/raw/*/mass-observer.hex`
files. It decodes their IEEE binary64/binary128 storage words as exact rational
numbers, checks the generated model-to-propagator mapping, and computes matrix
differences and determinants without invoking a reducer. The native captures
remain fixed-CTMode4, status120, accuracy−1 diagnostics. No build, native call,
input alteration, or timing was performed for this audit.

| Threshold group5 quantity | Original mass path | QP mass path |
|---|---:|---:|
| Maximum supplied-invariant/Cartesian discrepancy, GeV² | 8.8448359747417284164e−13 | 7.6777812283169873354e−30 |
| Supplied Gram determinant | 1.2226858389020917935e−13 | 1.2102331616577179280e−13 |
| Exact Cartesian Gram determinant | 1.2102331616577179281e−13 | 1.2102331616577179281e−13 |
| (Cartesian−supplied)/supplied determinant | −0.010184690824223269899 | +1.5968096077240173181e−20 |
| Captured native conditioning estimate | 5.2240781136634144476e−30 | 5.1708724932341610239e−30 |

Across all 37 refused threshold parents, the candidate's maximum invariant
discrepancy is `9.9192541427825330563e−30 GeV²`. For the actual subtopology
named in each refusal, the largest relative Gram-determinant discrepancy is
`1.6221811416180583385e−20`. The original maximum is approximately 1.018%.
Every original/candidate refusal estimate is reproduced from its captured
supplied matrix to better than `1e−50` relative. These estimates are the
source-defined guard heuristics, not certified condition numbers or error
bounds.

`analysis.json` retains 800 geometry records: two arms × two points × 200
groups. Matrix entries, determinant fractions, invariant residuals, exact
mass-square changes, input comparisons, and refusal-subtopology membership
are retained rather than only the group-5 summary. The four raw logs also
match their original ignored working copies by SHA-256.

The actual MP model values and unchanged numerators are bound directly.
For every group, the candidate's MP mass arguments equal the captured
corresponding model symbol; the cyclic ordering and native arrays match.
The old squares are exactly the 53-bit-rounded squares promoted to QP, while
the new squares are the 113-bit-rounded squares. This fixture's present
external real card entries initialize the MP masses by DP promotion; that
observed property is not assumed for derived or complex model parameters.
There are 688 changed mass-square records across the two points: 278 top,
200 Z, and 210 W. Higgs and zero squares are unchanged.

The repaired/consumed QP point, model values, numerator coefficients, helper
and adapter momenta, and native shifts match bit for bit in all four scoped
comparisons. Native masses and invariants match between backends within each
arm. The comparison deliberately preserves the existing native-scale
difference: TIRust squares the promoted DP scale, whereas NinjaRS promotes
the DP square. Consequently, “nonmass input parity” does not mean that all
native API arguments are bit-identical across backends. Historical control
parity covers only retained fields; historical NinjaRS host flags were not
available.

`rounding.py` and `rounding.json` separately decompose the remaining input
differences. Exact integer nearest-even emulation follows the frozen real
Fortran arithmetic and reproduces every off-diagonal S matrix and recovered
native invariant in all 800 groups. An additional read-only check confirmed
exact S symmetry and `S_ii=−2m_i²` for all groups. The selected records cover
group5 at both points and every refused threshold group, with three additive
contributions:

1. Exact Cartesian distance minus sequential binary128 momentum arithmetic.
2. Sequential binary128 distance minus the distance assigned by on-shell
   thresholds.
3. Assigned distance minus the invariant recovered from the mass-subtracted S
   matrix.

For threshold group5, the original massive snaps each displace the rounded
distance by exactly `8.84483597474172840619346199691824e−13 GeV²`, the change
in the top mass square. With the new QP squares, both snap displacements are
exactly zero. The same massive and massless threshold assignments still
apply. The residuals in the table are then entirely in the first contribution;
mass subtraction and recovery contribute zero for this group. The actual
repaired top momenta have exact mass-shell residuals of approximately
`−1.59248e−30` and `+1.01932e−30 GeV²` relative to the captured MP top mass
squared. These are consistent with the surviving binary128 arithmetic
residuals. The source-emulated threshold sequence is not a separately
instrumented branch trace, and the emulation does not model exception flags
or complex masses.

The kinematic rank-one records in `analysis.json` are differences in
`f_i=d_oi−m_i²+m_o²` and Gram entries. They are **not numerical tensor identity
residuals**. In particular, no valid RAW tensor exists for any of the 37
refused parents, including threshold group5. Captured scalar pinches cannot
make those missing tensors available. The count 37 must not be confused with
group ID37, which succeeds.

The separately appended `rank-one.py`, `rank-one.json`, and `rank-one.md`
cover numerical identities only where successful current-arm parent and
input-matched scalar-pinch outputs were actually captured. They do not alter
the first geometry analysis. The independent source review confirmed the
same normalization across topology and the MadLoop odd-rank minus. With
`D_i=(ell+q_i)²−m_i²`, `I0=RAW(1)`, `R=RAW(2:5)=−integral(ell/product D)`,
and `P_i` denoting the scalar with denominator i removed, the pair identity is

```text
2(q_i−q_j)·R = P_j−P_i + (q_i²−q_j²−m_i²+m_j²) I0 .
```

The metric is (+−−−), and actual captured shifts must be used: the first shift
is generally nonzero. Matching only scalar masses/invariants/scale is labeled
as such; exact Cartesian denominator matching additionally requires a common
translation of the retained shifts. Original-to-candidate comparisons cannot
hold the pinched scalars or I0 fixed when the masses change. Old original-mass
NinjaRS unit replays are not inputs to this appendix.

Source anchors are the preserved candidate `CT_interface.f:248–328`
(sequential arithmetic and threshold policy), `TIRUST_interface.f:392–450`
(ordered masses, prepared geometry, and successful RAW capture),
`direct/exec.rs:1620–1691` (guard), `direct/exec.rs:3622–3632` and
`convention.rs:70–83` (MadLoop conversion), and `basis.rs:13–29` (raw order).
The original input/Gram audits and current inspected source bytes are hashed
in `analysis.json`. The inspected `exec.rs` hash
`255b250f6997cfd08f017a9962ac79e6a4a87421f1f2d9f7ddb6a8bc1129eace` and
`numeric.rs` hash
`c4189f1e59f0746ea4413319b9a85df926789c3b1c95f9e1461bf7088879ebee`
match the retained earlier QP lifecycle/guard source audits. These are source
formula bindings plus direct reproduction of captured conditioning values;
they are not an independent proof that a mutable staging tree is identical
to every source byte used to build the loaded historical DSO.

For a five-propagator whole-tensor matrix, the displayed determinant and
standard scale-relative root are geometric diagnostics. Only the explicitly
named refusal subtopology is compared to the captured native guard. No claim
is made that the engine evaluates a whole pentagon by that displayed Gram
diagnostic.

The original and candidate threshold finite amplitudes each change by about
0.27057% under the mass correction. Cross-backend finite values are close
under the historical 3e−6 criterion, while their single and double poles
disagree by approximately 92.41% and 51.30%. Those comparisons establish no
accuracy: both threshold TIRust aggregates omit 37 failed group contributions,
and fixed4 bypasses host stability acceptance. NinjaRS success also supplies
no independent threshold accuracy certificate. The ordinary two-arm/backend
comparisons agree at approximately 1e−15. This remains a two-point real-model
transport/geometry diagnostic; checked rescue, CT5, complex/derived masses,
runtime performance, and the full process campaign remain unqualified.

Reproduce the offline files from the repository root with Python3.12. The
scripts create the requested output exclusively; choose new output paths:

```sh
python3 -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/execution-1/geometry-audit/audit.py --output /tmp/ew-mass-geometry-audit.json
python3 -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/execution-1/geometry-audit/rounding.py --output /tmp/ew-mass-rounding-audit.json
```

The frozen first `analysis.json` SHA-256 is
`c8851a036ecfd12865e1747230555968d73d4fe8f73cfcea51c8edcf3184cdac`;
`rounding.json` is
`f6f236777c348f257e6daf0f1013a5842a34e898a4ebe638333569d344bc96fa`.
