# Final focused-control evidence audit

This is an evidence audit of numerical source revision
`b44d20e725f1ec68589df20284e4ce9116e9e2ab`, ABI3, and installed archive
`1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32`.
The audit script runs no compiler, reducer, amplitude, or performance measurement.
The separately authorized installed-ABI reversal controls retain their own
build and numerical execution evidence below. This audit does not certify a
subsequently changed generator or archive.

The requirements come from the retained `inputs/PLAN.md` snapshot of
`references/PLAN.md`, its original-path/hash provenance in
`inputs/plan-provenance.json`, and `policy-v2/final-controls-checklist.md`.
The live plan may subsequently change; this evidence binds the snapshot.
`capture.py` records current source
hashes, compares them with the committed revision and the previously tested
ABI3 source, and verifies the referenced evidence. Its output is
`audit.json`. It independently recomputes the eight completed variant
comparisons from their retained full rows. Root reviewed the source-scoped
evidence; the complete eight-core acceptance gates remain outside this audit.

| Requirement | Evidence valid for this source | Remaining scope |
| --- | --- | --- |
| Final CLI generation | `generation-abi3-terminal-fixed/summary.json`: all eight core and five control exports rebuilt at b44; exact inputs and dependencies retained. `final-runtime-audit-fixed/inventory.json` verifies all thirteen runtime groups. | A later accepted generator change would require renewed generation and binding. |
| Fresh/persistent execution parity | `representative-proof-abi3-terminal-fixed.json` and its retention validation cover final candidate DP backends 1/6/7/8, QP 1/6/8, loop-induced amplitudes and split orders. Original and helper-corrected references have explicit bindings. | Parity alone does not establish physical cross-backend agreement or timing acceptance. |
| ggH, ggHH, mixed orders, complex masses | `final-controls-plan-fixed/comparison.json` passes all four cases, each at ten points, two retained repetitions, and all 21 pairs among seven corrected targets. It compares every required coefficient/order. | No remaining requirement for these four final generated controls at the recorded source/runtime. |
| Genuine rescue | Final representative candidate runs retain ten CutTools 329, ten Ninja 328, and ten NinjaRS 322 statuses with fresh/persistent parity. The completed near-singular variant independently compares all three backends at ten points with those same successful rescue statuses. Installed smoke and relocated standalone also execute actual MP. | Deliberately strict ordinary Ninja failures remain failed diagnostics. The successful near-singular inputs are identified separately. |
| Rotations and near-singular inputs | `final-variant-controls-both-precision` passes all four-backend rotated-input comparisons and four separate rotated-versus-unrotated comparisons. The near-singular control at `MLStabThres=1e-20`, two DP/QP rotations, passes all three QP backends and full Born/finite/pole rows; maximum tolerance fraction `8.80e-6`. Independent NLO analytic pole checks also pass in the installation proof. | No remaining requirement for these recorded controls. Numerical controls are not performance acceptance. |
| Rebinding and fresh contexts | The final corrected generated mass/scale sequence agrees across all four backends; reused/fresh NinjaRS outputs are exactly equal for all ten points. CutTools genuinely rescues three changed-state points, while NinjaRS stays DP218. Unchanged Rust tests compare retained/fresh DP, compensated DP and MP after changing coefficients, masses and scale; native/compensated interleaving is exact. The genuine MP bubble sequence is `(m0,m1,mu²,c)=(0,4,1,1)→(4,0,9,2)→(0,0,4,−3)`. | Generated NinjaRS rebinding is DP coverage. Genuine MP rebinding comes from the bridge bubble test, whose Laurent values are decoded to binary64 for comparison; it does not check every binary128 output bit. |
| Direction reversal | The final installed-ABI3 physical triangle/box control passes all 30 rows in native DP, compensated DP and MP. It negates shifts and odd tensor coefficients, reverses propagator/mass order, permutes the supplied invariant matrix, checks retained/fresh outputs bitwise, and resolves deliberately unflipped-numerator negative controls. Complete Laurent outputs satisfy the common DP bound and `1e-31` MP bound. | This is a numerical bridge invariant, distinct from recording-wrapper transport mocks and historical ABI2 oracles. The first synthetic input remains a retained UNSTABLE result; it establishes no invariant. Separate R1 output remains an engine-basis diagnostic. |
| Binary128 beyond DP resolution | Unchanged Rust `mp_transport_preserves_binary128_information` recovers the exact 113-bit value `1+2^-100`; `complete_mp_tadpole_retains_input_precision` tests actual reduction. Fresh ABI3 Fortran installation smoke verifies the perturbation survives both transport directions and a genuine MP pole. | This is a precise tested input, not an exhaustive decimal-conversion proof for every binary128 value. |
| 256 → 384 convergence | Unchanged actual complex-mass rank-five pentagon test retains the `1+2^-100` tensor perturbation, all eight serialized output fields, finite-split identity, and a `1e-31` largest-component-scaled bound. | The observed `2.01e-76` discrepancy is computed from 40-significant-digit serialized fields and tiny residual poles; it is not a claim of 77-digit finite-part accuracy. This establishes the requested 31-digit convergence for one difficult fixture. |
| Errors and no stale outputs | Unchanged ABI3 tests cover invalid/nonfinite input, malformed/overlapping outputs, every short buffer before access, diagnostic text, failure output sentinels, and context rebinding. Fresh wrapper tests map explicit failures/nonfinite results to FPE/unstable and zero output. | A caught DP panic safely fails but can leave that DP mutex poisoned; current tests do not claim recovery from an injected production DP panic. Experimental cache recovery tests were not promoted. |
| Threshold preservation | Source hashes bind the unchanged MadLoop dimensionless `OSThres` kinematic helper, native scalar provider defaults, and explicit MP scalar threshold `10^15·2^-112`. Actual generated comparisons use unchanged common thresholds; the 24-case production state-flow proof checks the same acceptance threshold on native and compensated retries. | There is no dedicated final installed scalar/kinematic threshold boundary scan in this evidence. The recording wrapper calls the real helper but does not vary `OSThres` across a boundary. Do not describe it as a numerical boundary oracle. |
| Installer, absence, copied dependencies, NLO | Sealed `install-export-abi3-terminal-fixed`: actual pinned and dual-local-source ABI3 installation/smoke, hidden-install relocated standalone 110/120/218, full NLO 20/20 poles and zero soft/collinear failures, GNU quad runtime resolution, disabled-NinjaRS legacy execution, actionable exclusive-8 rejection. | Optional StdHEP warning remains; PY6/Herwig6 showers were not validated. |
| Return codes, statistics, retry | Same sealed run freshly passes all 16 Fortran/installer/statistics tests: legacy codes, contextual new counters, old/mixed XML, aggregation, NLO log parsing, failed rebuild preservation, idempotence and ABI rejection. The unchanged production-template 24-case mock proof covers rotations 0/2, backend 8/8→6, finite retry, hard/FPE bypass, failed retry→QP and consecutive-point cleanup. | Mock QP establishes control flow only. Actual rescue evidence is identified separately above. |
| Eight core streams and performance | Governed by the final policy and root's ongoing complete-cohort collection. | Still pending. Earlier timings and historical source cohorts are not final acceptance; every process requires its own original-Ninja gate and confidence bound. |

The source identity is the reason the 16-test Rust proof can be reused: every
recorded bridge input, test source and implementation file still matches, and
the fresh pinned installation produces the same archive as the tested ABI3
installation. This does not turn unchanged bridge tests into tests of later
generated Fortran sources. The final compiled wrapper/installer/statistics suite
and final CLI exports supply that separate coverage.

The first final variant run rejected its changed-mass comparison: CutTools
rescued points 3/6/9 through QP with the original MP top mass. The test driver
updated `MDL_MT` only, while `COUP()` reads the independent `MP__MDL_MT` when
recomputing MP internals. `MP_UPDATE_AS_PARAM()` synchronizes only scale, AS and
G, not masses. That result is a retained driver-setup failure, not evidence of a
NinjaRS stale-context defect. The completed separate
`final-variant-controls-both-precision` namespace assigns both precisions before
full `COUP()` and passes the complete comparison, including those three
CutTools QP points. The former successful DP-only rebinding evidence did not
test this QP initialization.

No scalar-cache, generic-tensor-plan, unconditional T3, or symmetric-sharing
prototype is part of this runtime. Their controls remain experiment evidence.
The historical external Cargo build-cache location is retained in its original
proof; this audit does not rewrite that provenance.

`reversal/` preserves the failed direct-ABI synthetic test. Its purely spatial
independent momenta produce a negative discriminant in the fast engine's real
cut-basis square root, consistent with its `UNSTABLE` result and invalid flag.
The source-bound diagnosis is in `reversal-initial-diagnosis.json`; neither
invariance nor scalar-master failure is claimed from this attempt.
`reversal-physical/` is the successful separate test using retained physical
triangle/box kinematics and the same predeclared negative control and tolerances.
Its numerator has nonzero odd terms, and its complex squared masses have
distinct exactly representable imaginary components. The two negative
controls exceed their predeclared bounds by factors of about `7.83e5` and
`9.01e5`; they show that the sign transformation is numerically meaningful.

From the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/final-focused-audit-fixed/capture.py
```

This command only reads sources/results and updates this audit's JSON before
sealing. To verify existing evidence without changing it, including after
sealing:

```sh
python -B docs/benchmarkfolder/policy-v2/final-focused-audit-fixed/verify.py
```

The verifier checks the recorded source revision, bound files and prior seals,
recomputes the variant comparisons, verifies reversal rows and checks this
directory's seal when present. It disables bytecode writing even without `-B`;
recursive evidence inventories exclude Python caches and compiler products.
The reversal README retains its independent transformation and reproduction
command. Completed attempts refuse overwriting.
