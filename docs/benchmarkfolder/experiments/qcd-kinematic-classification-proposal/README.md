# Topology-aware invariant classification: source-only proposal

The next private experiment should prevent a physical composite invariant from being treated as an external mass shell merely because it is small. The eligibility decision can be generated from the existing HELAS graph, with no per-point vector matching and no change to `OSThres`. This is a proposed input-construction policy, not a demonstrated amplitude repair. No production file, generated export, native library or physical input was changed for this proposal.

The frozen observer package is `../qcd-loop-kinematics-observer-native`, inventory SHA256 `6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac`. It records 19 nonzero composite channels set to zero per loop direction, alongside 178 known massless single-leg/complement shell clamps. Both backends reject the final stress amplitude. All 22 reported singular TIRust Gram matrices are exactly singular with supplied distances and nonsingular with distances derived from fixed Cartesian shifts. That establishes an input discrepancy and a correlation with actual refusals; it does not prove that a different matrix yields an accurate amplitude.

## Generate identities from the graph

`HelasWavefunction.get_struct_external_leg_ids()` returns the outer non-loop legs attached to one loop interaction. `LoopHelasAmplitude.get_denominators()` walks the loop ancestry and reverses the result, producing ordered corner sets paired with loop masses. Despite the denominator-shift wording in that method's docstring, its entries must not be used directly as cumulative shift vectors.

For each final loop group, use exact integer vectors over an all-incoming external-leg basis. Derive each corner vector from the actual HELAS momentum convention and leg state. Validate the corner partition, its sign, the external-leg numbering, emitted `PAIRING`/mother order, and agreement among every amplitude sharing that group. Sort only the new set-based metadata deterministically; do not change existing grouping keys. Map the final group IDs after `identify_loop_groups()` has sorted and renumbered them. Ungrouped and loop-induced exports assign IDs through different paths in `helas_call_writers.py`; bind the table to the actual emitted reduction-reference sequence and `LOOPNUM`, not unconditionally to an enumeration of `loop_groups`.

Then apply the actual adapter construction symbolically: its first shift is the sum of all corners, and the remaining shifts are zero-seeded prefixes. Keep that first shift's literal coefficient vector; do not replace it by zero. For CT2/CT5 with more than two denominators, reverse and negate the corners and apply the corresponding mass permutation before deriving shifts. N=1/2 follows the existing no-reversal branch. Generate a table for each direction rather than assuming that a denominator permutation preserves the literal closure residual.

For each unordered pair, retain its literal difference vector `v`. Let `u=(1,...,1)` encode exact momentum conservation in the chosen all-incoming basis. A direct single-leg identity has `v=+e_l` or `v=-e_l`. A complement has an exact integer witness `v=sign*e_l+k*u` with nonzero `k`. Everything else is composite, closure-only, or unknown. Do not reduce the literal vector in the numerical code. A conservation witness is a physical topological identity, not proof that the finite-precision supplied coordinates close exactly.

The generated metadata should retain both identity and policy:

| Field | Purpose |
|---|---|
| Pair class | Direct single, complement, composite, closure-only or unknown |
| External leg and integer witness | Auditable correspondence to a physical external channel |
| External mass symbol | Distinguishes the external shell from internal propagator masses |
| External `offshell` flag | Disallows imposing a model shell on a requested off-shell leg |
| Direction and group offsets | Selects the correct static table without mutable global context |

A compact packed triangular integer table per group and direction is sufficient at runtime. Full integer vectors can remain in a generation audit artifact. Table creation occurs once during export. Runtime work is an integer lookup and eligibility branch; it requires no allocation, topology reconstruction or floating-point comparison to identify a leg. Missing or inconsistent metadata should make the experimental path unavailable rather than inventing a shell identity.

`write_TIR_interface()` already emits group-specific static data. A shared generated include or canonical fixed-form helper can carry this table. Pass the group/direction or the selected table explicitly into the matrix-building path. The current standalone TIRust tensor diagnostic has no group ID, so it needs a distinct explicit classified entry point or a compatible wrapper; it must not infer a stale ID from a previous numerical call. Group IDs already present in some reducer COMMON blocks are not a justification for adding a new implicit global dependency.

## Separate classification from the shell value

The computed physical distance is `D_ij=(q_i-q_j)^2`, and the matrix entry is `S_ij=D_ij-M_i^2-M_j^2`. The proposed rule applies to the identity of `D_ij`, before constructing S.

| Channel identity | Proposed eligibility |
|---|---|
| Composite, closure-only or unknown | Keep the computed distance; small magnitude alone is never a shell certificate |
| Direct/complement external leg explicitly off shell | Keep its signed virtuality; do not impose the model mass |
| Known massless on-shell leg | Eligible for the existing massless shell treatment, subject to the chosen representation contract |
| Known massive on-shell leg | Eligible only for that external leg's declared real shell, with an explicit precision convention |

A first **private QCD-only** variant can gate the current three threshold branches on certified direct/complement on-shell eligibility. It would leave `OSThres`, the computed `DIFFSQ`, summation order, and the retained branches unchanged, while bypassing all three branches for composite channels. It must retain both backends' original failures as controls. This narrow variant does not qualify the general massive or complex-mass policy, and no build or replay of it is authorized by this document.

For a general implementation, numerical proximity to an internal propagator mass is insufficient. The present common helper tests `M2L(i)` and `M2L(j)`; COLLIER tests every internal mass. Neither establishes which external shell a pair represents. The new metadata must select the external mass identity first. A real external invariant must not acquire an internal complex pole mass or lose a width through a real-part projection. Internal complex masses remain complex in S and in the reducer input. Model-equal and numerically coincident but physically distinct masses need separate tests.

External `*` handling already exists. `write_improve_ps()` emits a model MP mass for an ordinary leg, or `SQRT(ABS(p^2))` for a starred leg, into the repair routines. The `KEEP_OFFSHELL_MASS` array is passed through but is not the runtime branch controlling those generated assignments. A future classifier should use the process leg's `offshell` metadata and validate its mapping, not infer policy from whether the array is read. In particular, the new invariant policy must preserve signed spacelike virtuality; `SQRT(ABS(p^2))` is not a signed virtuality representation.

## Cover every MadLoop-owned construction

| Path | Required source integration or limitation |
|---|---|
| Common `BUILD_KINEMATIC_MATRIX` and `MP_BUILD_KINEMATIC_MATRIX` | Add an explicitly classified variant and update all generated callers that use supplied matrices |
| Ninja and Samurai adapters in `CT_interface.inc` | Audit their existing common-helper calls and native convention separately |
| NinjaRS and TIRust DP/QP adapters | Both use the common helper and then reconstruct native distances from S plus masses; update both identically |
| COLLIER | Its separate cyclic `MomInv` enumeration and all-mass/massless thresholds require a topology table matching that precise enumeration |
| CutTools and other shift-based APIs | Inventory MadLoop-owned preprocessing and distinguish any library-internal thresholding; a common-helper edit alone does not cover them |
| Direct diagnostic callers and ungrouped exports | Require explicit identity metadata, or retain a separately documented legacy interface; do not claim classified coverage for an unknown context |

The COLLIER cyclic enumeration cannot simply reuse the common helper's packed pair index. Its supplied vectors also use a different construction. The current interface accepts only CT1/2, and its internal-stability mode can return cached tensors without reversing corners; the explicit `SWITCH_ORDER` call is in the other branch. Prove the map from each cyclic corner interval to the pair identity in the branch actually taken, including reversal and closure terms, before sharing data. Backend-private cache entries remain separated by actual inputs, precision, pole/finite mode and check pass. A runtime policy switch would need cache invalidation; preferably the experimental policy is fixed in the generated source so it cannot change mid-event.

Any new generated helper/include must be admitted explicitly by the existing DP terminal-consumer proof. It should not access the numerator wavefunction COMMON. Preserve independent check stores, error propagation, caller exception traps, entry flags and final finiteness/stability checks. No global IEEE save/restore or extra per-point scanning is proposed.

## Momentum repair is a separate unresolved contract

The earlier Born B/C replay used the same original physical record but two effective coordinate sets: B was exact binary64-to-binary128 promotion; C was the actual `SET_MP_PS` repair. The stress `p1-p4` invariant changed by about `-1.76473e-4` fractionally, and the Born value changed by `+3.53039e-4`. The private fallback evaluates its rescued Born on B while the loop numerator is evaluated on C. That mismatch is larger than the requested `3e-6` physics gate. The measured Born-normalization change alone does not explain the much larger rejected double-pole discrepancy, but it does not bound the loop's sensitivity to the repair either.

A successful future checked rescue must define one consistent effective-kinematics contract for Born aggregation, loop numerator, supplied distances, UV/IR normalization and independent references. Choosing C requires explicitly retaining the changed coordinates; it is not a literal unchanged-input acceptance. Choosing B requires establishing the reducer and wavefunction treatment of shell residuals. The ordinary path must retain its current results. The topology classifier alone settles neither choice.

Mass precision is another part of that contract. The separate EW audit found promoted DP internal mass squares and repaired MP external mass squares differing by about `8.84e-13 GeV^2`. Neither lowering `OSThres` nor identifying the correct leg resolves that representation difference. Real and complex masses, massless shell residuals, conservation complements and requested off-shell legs need an explicit, common precision rule before general promotion.

## Required qualification before promotion

1. Independently validate generated integer identities for all grouped and ungrouped 2-to-1 through 2-to-4 processes, both models, crossing, N=1/2, and CT1/2/4/5. Reproduce the retained 116-group/477-pair/19-target routing in both stress directions without parsing generated source in the production implementation.
2. Check that every physical composite, however small, retains its computed nonzero distance. Test certified massless, massive, complex-internal, equal-mass, off-shell and closure-residual cases separately. Preserve original source/data and exact before/after inputs.
3. Establish ordinary coefficient/status bit identity where the old thresholds did not alter a composite, preserve caller traps and entry flags, and record any IEEE differences caused by avoiding predicate arithmetic. Require finite, genuinely checked current-call QP results; retain code 420 on unresolved failures.
4. Resolve B/C and mass-precision consistency before comparing checked amplitudes to same-effective-input independent references. Full, UV/IR-selected and finite-skipping modes must agree on computed coefficients; cache and reversal checks remain independent.
5. Only after numerical qualification, measure the actual ordinary-path overhead under the frozen benchmark protocol. The existing per-process 5% runtime gate remains unchanged. Static metadata is intended to keep overhead small; no performance benefit is claimed without measurement.

`source-bindings.json` records the inspected source bytes, current inspection revision, historical evidence indices and the production-policy anchor. The portable index and `verify.py` bind this proposal; `verify.py --sources` additionally verifies its retained source paths. This package is a design artifact only.
