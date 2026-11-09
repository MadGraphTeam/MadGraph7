# Generated-code threshold geometry probe

This probe calls the unchanged generated `ML5_0_SET_MP_PS` and
`ML5_0_MP_BUILD_KINEMATIC_MATRIX` routines for the exact two-point
`physical-stress/points/ew_n3_epem_ttxa/selected.dat` stream. It evaluates no
amplitude, tensor reduction or scalar integral. It does not promote the original
unchecked numerical results to accepted results.

The private clone is `.tirust-build/ew-threshold-geometry`, copied from the
historical `candidate-9-qp-init-dp` output. All existing numerical objects and
archives retain their hashes; the only new object is `geometry_probe.o`.
The driver links a private snapshot of the historical TIRust bridge,
SHA-256 `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`,
to avoid an installation-publication race. The bridge is not called.

The measured threshold point is point 2. Residuals below use the actual promoted
model mass, rather than the exact decimal value 173.3.

| Quantity | Before QP repair | After actual QP repair |
|---|---:|---:|
| Maximum external shell residual, GeV² | 2.8837820 × 10⁻¹² | 1.5777218 × 10⁻²⁹ |
| Maximum conservation residual, GeV | 2.8421703 × 10⁻¹⁴ | 2.0685073 × 10⁻³² |
| Top-pair (s-4m_t^2), GeV² | 6.5066591 × 10⁻¹⁰ | 6.5112951 × 10⁻¹⁰ |
| Top-pair beta | 7.3595366 × 10⁻⁸ | 7.3621579 × 10⁻⁸ |

The largest momentum component changes by `2.26916046577e-7` GeV. The pair
excess changes by 0.07125%, and beta changes by 0.03562%. These observations
describe the common QP input preparation; they do not explain a difference
between reducers that receive the same repaired input.

The generated MP card reader first reads a `REAL*8` temporary and promotes it
to `REAL*16` (`Source/MODEL/lha_read.f:374–375,403–404`). Consequently the actual
MP top mass is `173.300000000000011368683772161602974`, also used by QP external
shell repair. Squaring that value in QP gives
`30032.8900000000039403857954312117200`. The existing adapters receive the
promoted DP squared mass, `30032.8900000000030559021979570388794`. Their difference
is `8.8448359747417284e-13` GeV²; four times this difference is 0.54335% of the
repaired pair excess. This is a measured input-consistency sensitivity, not a
certificate that a specific library result is correct.

The matrix probes use explicitly representative canonical shifts from
`(-p1,-p2,p3,p4,p5)`, with the first shift zero and subsequent shifts given by
ordered prefix sums. They test all-zero, all-top and alternating zero/top
squared-mass patterns at `OSThres=1e-8,1e-14,1e-24`. These synthetic patterns
are not claimed to be captured physical diagrams. For repaired point 2,
top-shell entries are rounded to the promoted DP squared mass at the first two
thresholds and retain the QP shell value at `1e-24`. The repaired canonical
photon entries do not change across these thresholds. Use the separate actual
per-loop trace to identify affected physical topologies.

The raw log also records all raw-promoted and repaired momenta, both QP common
block arrays, real DP/MP electroweak couplings, all 900 matrix records, and the
scale arithmetic. At point 2, promoted `MU_R**2` is
`120131.584026313241338357329368591309`, while `REAL(MU_R,16)**2` is
`120131.584026313244681182911151971623`.

| Scale arithmetic diagnostic | Ordinary point 1 | Threshold point 2 |
|---|---:|---:|
| QP square minus promoted DP square, GeV² | 0 | 3.3428256 × 10⁻¹² |
| Difference divided by QP scale square | 0 | 2.7826367 × 10⁻¹⁷ |
| Difference divided by repaired pair (s-4m_t^2) | 0 | 0.0051338874 |

The final row compares numerical magnitudes only. The renormalization scale is
not the top-pair invariant or internal mass; this ratio does not assert that
scale rounding changes the threshold excess. The Ninja/NinjaRS QP interfaces
promote the DP square, while the TIRust QP interface squares promoted `MU_R`.

From the repository root, validate the preserved evidence with:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/ew-threshold-adjudication/geometry/analyze.py
```

To reproduce in a fresh matching checkout after coordinating a quiet native-job
window, run `run.py prepare`, then `run.py execute`, then `analyze.py`. The helper
deliberately refuses to overwrite an existing clone or evidence. `prepare`
verifies generated objects and sources against the frozen physical-stress
compilation manifest. `execute` adds the driver object and links directly to the
existing process objects; it never rebuilds them.

`prepared.json` and `execution.json` bind the source, input, object, private DSO,
card and raw-log hashes. `summary.json` records the validator result and measured
geometry. An independent source audit checked the driver ABI, absence of reducer
calls, exact input transport, matrix dimensions and thresholds, and object
preservation before native execution.
