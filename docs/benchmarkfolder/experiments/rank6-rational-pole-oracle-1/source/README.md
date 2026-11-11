# Exact rational rank-six pole oracle — source proposal and retained-row audit

This private packet adds no native evaluation, compilation, library, production
change or performance result. It independently derives UV residues for the
already retained massive N=2/rank-six fixture109 and supplies a source-only
massless-triangle fixture with nonzero IR double poles in all210 Cartesian
components. It does not replace any historical failed IEEE/stress gate.

`oracle.py` uses only integer/Fraction polynomial arithmetic. It neither imports
TIRust/oneloopmaster nor fits a stored reducer answer. `DERIVATION.md` gives the
loop-moment formula, Wick multiplicities, component ordering, MadLoop odd-rank
conversion and exact simplex integration. The convention/scalar-contract
excerpts are separately pinned in `source-evidence.json`; they establish the
requested output definition, not an independent implementation oracle.

For fixture109, `analyze.py` reads exact copied original binary128 words and
checks all15 selected rows, all210 components and both pole slots (6300 complex
coefficients). There are ten rows with nonzero UV content and five IR-only rows.
All210 exact UV complex components are nonzero. Every row passes the explicit
complex symmetric residual1e-8 check, computed by comparing rational squares;
whole expected-zero complex residues must be exactlyzero. The original amplitude
comparator's finite-scaled small-pole rule is not used for this separate oracle.
Individual expected-zero real/imaginary parts are reported separately, avoiding
a misleading relative error againstzero. Their signed-zero words remain in the
original archive; this is numerical coefficient corroboration, not signed-zero
transport parity.

Maximum nonzero-part symmetric residuals: TIRust DP1.954e-14, TIRust QP4.430e-35,
COLLIER2.184e-16. Expected-zero imaginary leakage is at most2.498e-16 DP and
1.166e-76 QP. Exact per-row rational residuals and all210 expected fractions are
retained in `retrospective-analysis.json`. Full and accelerated partial rows,
UV+IR/UV-only/IR-only and DP/QP are kept distinct. The N=2 fixture is massive,
so all independent IR residues vanish; it does not qualify nonzero IR behavior.

The original rows inherit IEEE mask37 from a preceding COLLIER call. That
historical failure is unchanged. The original effective-input and generated
source/ABI correspondence belongs to the original closed runtime replay,
whose exact index and finite-audit index are copied here. This packet verifies
its copied input hashes and new algebra; it does not re-run that ancestor parser
or requalify a new/current export, stability rescue, cache, hardware exception
state, full amplitude or performance. Root can separately replay the original
portable capsule to join the new conclusion to its retained provenance.

`future-triangle-fixture.json` is only an exact mathematical specification:
shifts0,(3,1,2,2),(0,2,0,0), all masseszero, mu²4, cumulative rank6. Its parameter
polynomial is4*x0*x2. All210 IR2 values are nonzero; IR1 and UV1 are also supplied
exactly. Nine pure test methods include all70 possible rank≤4 trace identities
and252 external-momentum contraction identities through rank6, plus equal-mass
radial, odd-sign, component-count and binary128 controls. These are algebra
checks, not numerical execution of the proposed fixture.

A bounded future run should reuse the existing generated rank-six diagnostic
harness, change only its private driver, and authenticate the actual ABI inputs.
Predeclare12 TIRust rows: DP/QP × full/partial × UV+IR/UV-only/IR-only, with cache
disabled. Compare all210 pole components to the fractions here; require finite
slots marked unevaluated for partial requests and full/partial pole equality.
Observe fresh pre/post IEEE state and native errors before any comparison.
Use one private compiler/link step (120s/4GiB including cleanup) and one serialized
worker (120s/4GiB including cleanup); these are proposed caps, not an authorized
or existing controller. No new renderer, amplitude generation or broad campaign
is needed. The mathematical fixture does not guarantee native success or input
identity. A separately reviewed bounded controller is still required.

Pure reproduction from the MadGraph checkout (pinned Python shown literally):

```sh
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B -m unittest discover -s .tirust-build/rank6-rational-pole-oracle-proposal-1 -p 'test_*.py' -v
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/rank6-rational-pole-oracle-proposal-1/analyze.py --verify-analysis
```

`preparation-history.json` retains the first source-preparation path error. The
corrected preparation changes no historical bytes and required no numerical
rerun. Precise performance acceptance remains Hua-Sheng's local TODO.
