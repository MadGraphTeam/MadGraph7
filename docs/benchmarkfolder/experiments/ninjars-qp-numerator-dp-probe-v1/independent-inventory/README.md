# Independent stored-input audit

This audit makes no engine calls. `check.py` independently parses the integer
capture grammar, reconstructs exact binary64/binary128 dyadic values, and rounds
binary128 inputs to binary64 using integer division with ties to even. Decoder
self-tests include signed zero, normal/subnormal boundaries, maximum finite
binary64 and halfway cases. The resulting records must equal the hardened
capture parser, and all retained output rows must equal the bound reference.

The 1,392 successful records form 464 complete semantic keys: point, direction,
loop ID, split order, propagator count and rank. Each of the three stages has IDs
1–116 at both directions of points 44 and 80. The native and compensated DP
inputs, including every coefficient and geometry word, are identical. Their
outputs differ in 1,039 Laurent component words and 371 rational component words;
those differences alone do not identify an accuracy improvement.

| Input field | Real components | Exact DP/QP value differences | Differences after QP→DP rounding | Records with rounded differences |
|---|---:|---:|---:|---:|
| Numerator coefficients C | 29,240 | 29,210 | 28,838 | 464 |
| Invariant matrix S | 5,332 | 2,392 | 1,490 | 362 |
| Propagator shifts P | 6,064 | 2,621 | 1,467 | 451 |
| Squared masses M | 3,032 | 0 | 0 | 0 |
| Squared scale U | 464 | 0 | 0 | 0 |

The independent integer lattice-distance distributions agree with every
retained owner distribution. Among changed, same-sign, nonzero components, the
median binary64 distances are C=28, S=1 and P=2; the largest S distance is 56.
Zero and sign transitions are separate categories. C has 22 zero→nonzero,
44 nonzero→zero and 1,388 opposite-sign transitions; P has 87, 300 and 17.
These often involve cancellation residuals. Coefficients have rank-dependent
units, and neither ULP distances nor near-zero relative maxima are physical
amplitude error bounds.

The capture-side effects are constrained by the observed complete-process
check: all 100 points, both output orders, coefficients, accuracy estimates and
return codes match the bound original exactly. The two rescued points remain
44 and 80. Integer-only `TRANSFER`/hex writes introduce no floating formatting
or new IEEE imports. This observed invariance is not a universal claim that
Fortran I/O cannot affect the floating environment. The first startup attempt
failed to resolve a shared library before amplitude evaluation; it remains a
separate retained failure. The successful attempt binds its actual loader
paths/hashes and uses the benchmark library-search policy.

`source-evidence.json` retains hashed excerpts explaining the replay limits:

* The QP path first widens the DP external point and then runs
  `MP_IMPROVE_PS_POINT_PRECISION`. It recomputes MP wavefunctions and loop
  coefficients, including the Born interference. Thus C and S/P changes are a
  changed input bundle. Substituting captured QP coefficients into the original
  DP geometry would combine quantities formed at different kinematics; it would
  not isolate numerator arithmetic or define a validated physical amplitude.
* The two directions are matched separately. Their independent coefficient
  construction and momentum routing must remain intact in any subsequent
  diagnostic. No averaging or direction canonicalization is justified here.
* The native DP and generic 256-bit paths also use different reduction/scalar
  arithmetic. The native provider's on-shell threshold is 1e-9; the MP bridge
  uses `1e15 / 2^112`, separately from MadLoop's thresholding of the supplied
  S matrix. A cross-domain output comparison therefore also includes provider
  and threshold conventions, even if supplied geometry is held fixed. This
  audit changes none of them and does not assert they caused an observed error.
* The actual wrapper forms `NRS_SCALE` by promoting `MU_R**2`; masses and scale
  in this capture are exactly promoted DP values. They do not supply additional
  high-precision input information.

There is no remaining capture completeness or input-matching blocker in the
reviewed successful stream. Replay remains a separately authorized diagnostic;
this audit establishes neither a new rescue strategy nor a performance result.

From the repository root, reproduce the read-only check with:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/ninjars-qp-numerator-dp-probe-v1/independent-inventory/check.py
```

`result.json` binds the raw data, parsers, inventories, capture metadata and
checker source. The enclosing capture seal also binds this note and the source
excerpts. No production or previously sealed file was modified.
