# QP input-bundle capture: completed diagnostic, replay held

The corrected capture passed on CPU 2. All 100 points reproduce the bound
original's Born/Laurent coefficients, total/physical orders, accuracies and
statuses exactly: 98 DP `218` and 2 genuine QP `322`, at points 44 and 80. Only the copied
driver's integer point tags and copied NinjaRS wrappers' integer/hex logging
were added. The production retry controller, CT modes, thresholds, contributions
and genuine 256-bit path were unchanged. No mixed-precision attempt was implemented.

The clean f4b export's original executable is byte-identical to the validated
fresh dev-QCD3 executable. `bindings.json` retains its complete compiled-source,
runtime, recipe, physical card and point identities. `prepared.json` binds the
capture executable and proves that only `CT_interface.f` and `check_sa.f`
changed. Both exact before/after sources and the patch are retained.

Attempt 1 failed at the dynamic loader (`libninja.so.0` missing from the search
path), before any amplitude call or tensor capture. Its result, raw stderr and
original runner remain intact. Root authorized attempt 2 using the same executable
and numerical inputs with the validated benchmark's full library search policy.
The resolved Ninja and Collier shared-library hashes match the bound original
runtime. Nine strengthened parser tests pass. `launch-correction.json` records
the environment correction and parser checks; it does not replace the failure.

Attempt 2 retains 1,392 complete records, 464 per native DP/compensated DP/genuine QP
stage. Full matching uses point, direction, loop ID, squared order, denominator
count and rank; each stage/point/direction contains exactly IDs 1–116 with globally
consistent shapes. Native and compensated inputs are bit-identical. Every raw
reducer status is 0; points 44/80 still fail the DP **direction consistency test**.
A native success status is not a declaration of point-level numerical accuracy.
Full 100 parity is checked before any records are interpreted.

The normal QP path supplies a changed QP input bundle, including its improved
phase-space construction. It is not a pure numerator-arithmetic intervention:

| Field | Real components | Exact QP value differs from promoted DP | Rounded QP word differs from DP |
|---|---:|---:|---:|
| Tensor coefficients C |29,240|29,210|28,838|
| S matrix |5,332|2,392|1,490|
| Denominator momenta P |6,064|2,621|1,467|
| Complex squared masses M |3,032|0|0|
| Squared scale U |464|0|0|

All 464 records have rounded coefficient differences; 362 have changed S entries
and 451 have changed momentum entries after rounding. `input-inventory.json`
retains exact dyadic differences and every matched identity. The independent
audit uses a separate decoder under `independent-inventory/`.

`ulp-inventory.json` measures rounded-QP minus DP changes as binary64
representable-value steps, with zero and opposite-sign cases separated. Among
changed same-sign nonzero components, median distances are 28 for C, 1 for S and 2
for P. The S maximum is 56 steps. C has 22 zero-to-nonzero, 44 nonzero-to-zero and 1,388
opposite-sign transitions; P has 87, 300 and 17 respectively. There are no
signed-zero-only changes. Very large step counts occur when cancellation
residuals approach zero and are not physical amplitude-error estimates.

The maximum rounded-bundle absolute changes are 0.05078125 for C,
2.9103830456733704e-10 for S and 1.7053025658242404e-13 for P. Coefficient units
depend on rank. Maximum component-scaled relative changes approach 1 for C/P
and are 1.0937669399845483e-14 for S. Exact fractions, words, locations and
nearest-rank integer quantiles are retained; these numbers do not establish
accuracy or performance of any mixed reduction.

No replay was executed. The 8-cell matrix in `PLAN.md` remains separately held
for root review. Its DP-geometry inputs must remain exact; the full-QP geometry
reproduction control and the improved-point confound remain explicit. A future
replay belongs in `replay/`, outside this completed capture seal.

Offline checks (no compiler, amplitude, ABI/context or reducer execution):

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/ninjars-qp-numerator-dp-probe-v1/verify_capture.py
```

The seal includes the failed attempt, corrected raw 100-point stream, all captured
words, source/runtime identities, scripts, build/parser logs and independent
audit. It excludes Python bytecode caches and future `replay/` contents.
