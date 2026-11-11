# Independent rank-six UV pole corroboration

An exact rational Feynman-parameter oracle corroborates all 210 Cartesian UV
components of the retained massive two-propagator, rank-six fixture 109.
All 15 full/partial and pole-selection rows pass the separately declared complex
symmetric `1e-8` coefficient check, covering 6,300 complex Laurent coefficients.
The oracle uses integer and Fraction polynomial arithmetic, without importing
TIRust or oneloopmaster or fitting a reducer result. Its derivation and nine
algebra tests are retained under [source/](source/README.md).

Maximum nonzero-part symmetric residuals are `1.954e-14` for TIRust DP,
`4.430e-35` for TIRust QP, and `2.184e-16` for COLLIER. Expected-zero component
parts have separate absolute-leakage diagnostics. The original binary128 words
and signed zeros are retained; this numerical comparison does not claim
signed-zero transport parity.

This is a retrospective coefficient audit of an existing synthetic fixture.
It adds no numerical execution or compiler call. The original inherited IEEE
mask 37 and every historical failed stress/runtime gate remain unchanged.
The massive bubble has zero IR poles. The included all-component massless
triangle IR specification remains uncompiled and unrun. Physical amplitudes,
new/current export correspondence, stability/rescue, cache behavior and precise
performance are outside this audit.

Reproduce using Python's standard library from the repository root:

```sh
python3 -B docs/benchmarkfolder/experiments/rank6-rational-pole-oracle-1/source/verify.py --index-sha256 9f866cc001a47cda221d6be4183ed135cb61af2ad6b48536012bce00b6e7587d
```

The `source/` directory is the unchanged sealed proposal and is independently
relocatable. Its historical `.tirust-build` command strings are provenance;
use the public verification command above. The original effective-input and
ABI correspondence is separately replayable from the retained
[rank-six runtime capsule](../generated-rank6-poles-proposal/attempt1/common-input-proposal/runtime1/README.md).
The 41-page benchmark PDF remains its separately frozen report snapshot.
Hua-Sheng's precise local timing campaign remains a TODO.
