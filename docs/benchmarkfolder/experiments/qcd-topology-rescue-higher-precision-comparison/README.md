# Provisional QCD512 contraction comparison

This package analyzes the raw output of the **incomplete runtime-capture attempt**. It launches no native process and does not repair that attempt's missing actual mapping evidence. All numerical results remain provisional. The corrected one-shot runtime replay is a separate task; no production, general precision policy, campaign or performance acceptance follows here.

The source-bound revision2 compares all232 group/direction calls and all696 complex contracted Laurent slots (1392 real components), without dropping small values. All common original/repaired coordinates, numerator coefficients, internal masses, matrix entries, shift components and model contexts match bit for bit across private512 TIRust, historical256 TIRust and the retained topology-C NinjaRS trace. Both observed native scale records are already `mu²=1,000,000`: TIRust squares the promoted DP scale in QP, while NinjaRS promotes its DP square; their values agree exactly at this point. This does not establish equivalence for arbitrary scales.

The raw host512/NinjaRS relative differences are finite7.243e-16, single4.129e-16 and double2.920e-16. The universal QCD double-pole normalization using the same C-Born has relative error1.0375e-15. This is an independent double-pole check; the finite/single comparison uses NinjaRS, not an analytic truth reference.

Individual contracted components do not all agree to3e-6. Forty-six of696 complex slots exceed that symmetric diagnostic in at least one component or complex norm: two finite,30 single and14 double slots. Every value, exact rational real/imaginary difference, squared complex norm and absolute error is retained. Large relative errors in small components are neither discarded nor treated as a new per-group acceptance criterion.

For example, group44/CT5 has finite raw real about−1.3996341e11 and imaginary about−4.9729773e10. Their absolute discrepancies are1.520653e6 and9.349471e6. Its actual projected DP real discrepancy is19,259.30, or4.4344e-18 of the host finite amplitude and2.3431e-20 of the recorded adapter group L1 sum. The observed adapters project imaginary contractions to zero, but the imaginary discrepancies remain in this comparison.

Across each116-group direction/slot cohort, the **sum of absolute projected discrepancies**, before any host averaging, is2.86e-14 to1.617e-13 of the corresponding host magnitude and3.26e-16 to1.13e-15 of the adapter group L1 magnitude. Signed sums and L1/absolute-sum cancellation factors are also reported. These descriptive ratios do not assume unrecorded assembly weights or certify every integral; they show that the small host difference is not concealing a large sum of the observed projected discrepancies.

The initial `compare.py`/`provisional1.json` are preserved as an analysis attempt. That draft incorrectly treated TIRust's logged native scale as `mu` and squared it again, so its scale-consistency conclusion is invalid. Revision2 checks the frozen generated source and compares both already-squared scale records directly. The raw host/coefficient comparisons did not change. Use **compare2.py, provisional2.json and provisional2-summary.json** for conclusions.

The NinjaRS observer records contracted coefficients; it does not expose an uncontracted unit-numerator tensor basis. This package therefore makes no monomial-by-monomial tensor truth claim. Working precision and the admission guard changed jointly. The effective C coordinates, private topology rules and promoted-DP mass-square convention remain explicit limitations; the untouched production pipeline is not qualified by this experiment.

Read-only verification:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-comparison/verify.py
```

A later separately captured process may be analyzed with the same frozen revision2 code into a new evidence namespace. Its original runtime provenance and its own acceptance limits must remain visible.
