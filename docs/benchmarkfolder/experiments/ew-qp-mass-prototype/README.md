This is an uncompiled private prototype of consistent QP model-mass transport
for the retained EW3 ordinary and threshold fixture. No compiler, reducer or
new amplitude has run, and production/campaign sources are unchanged.

The existing MP model is sufficient for this source change. It already exposes
the real mass symbols, derived real masses and complex pole-mass symbols used
by the MP numerator. For the card-supplied top, the MP reader intentionally
preserves the promoted binary64 input; this prototype does not reinterpret the
decimal card. It removes the later binary64 squaring intermediate by squaring
that existing MP value in QP. For a derived mass, it uses its existing MP model
expression's result. For a complex pole mass, it uses the already selected
`MP__CMASS_*` value. Real external momentum repair remains separate.

`prepared.json` binds the original historical export and the input-policy
proposal whose index SHA256 begins `2f90c77b`. Full before/after generated
sources, patches and model/repair source references are retained here.

## Exact source scope

Only three generated files change in the isolated candidate:

| File | Change |
|---|---|
| `loop_CT_calls_1.f` | All 200 groups call private `_QPMASS` helpers with paired original DP and existing MP mass arguments. |
| `CT_interface.f` | Append seven paired helpers and a private NinjaRS QP wrapper. Every original routine and signature remains byte-identical. |
| `TIRUST_interface.f` | Append private QP diagnostic and contraction wrappers. Every original routine and signature remains byte-identical. |

The seven helper families have 34, 67, 58, 19, 16, 4 and 2 callers for
`LOOP_2`, `LOOP_3`, `LOOP_4`, `LOOP_5`, `LOOP_1_2`, `LOOP_2_3` and
`LOOP_3_4`, respectively. `group-mapping.json` records every actual/formal
mapping and cyclic square order, including grouped wavefunctions and pairings.

Each private helper immediately delegates DP execution to the original helper,
before computing any QP square. It also delegates requested backends other than
8/9. Inside the QP path it forms `COMPLEX*32 MP_M2L` in the same cyclic order
as the original DP array and supplies it to the private QP adapters. Existing
DP actuals, old DP mass squares and reducer entrypoints remain intact. Paired
argument construction and delegation add unmeasured overhead: preserved DP
inputs and ABI are not a DP performance claim.

The private adapters change only their versioned name, QP mass formal, and the
TIRust contraction-to-diagnostic target. Their private `MASSES` copies remain
because the direction-switch helper mutates order. Prefix sums, on-shell
matrix rules, thresholds, contractions, defensive finite checks, failure
propagation and native APIs remain unchanged. The native TIRust library is the
historical `49af12f0…` 256-bit implementation with its original guard, not the
separate higher-precision experiment. Both backends retain their existing
scale expressions: TIRust squares after promotion; NinjaRS promotes the DP
square. That pre-existing difference is explicitly outside this mass change.

The guard inspects requested `MLReductionLib(LIBINDEX)` before actual backend
selection. Therefore this prototype qualifies only explicit `[8]` or `[9]`
source routing. Mixed rescue chains remain unqualified. Ninja, CutTools and
COLLIER QP mass boundaries are unchanged; no all-five-backend ABI claim follows.
The retained fixture uses optimized interference reduction. Other generated
amplitude-reduction branches require separate coverage before generalization.

## Model policy and unchanged semantics

The mapping is based on the original model-selected identifier, never on a
numerical value or top-specific condition:

```fortran
DCMPLX(MDL_MT), CMPLX(MP__MDL_MT,KIND=16)
DCMPLX(CMASS_MDL_MT), CMPLX(MP__CMASS_MDL_MT,KIND=16)
```

The transformer rejects a missing genuine QP declaration or an arbitrary mass
expression rather than inventing a reconstruction. The fixture has 626 mass
arguments covering zero, top, Z, Higgs and W. The separate preserved CMS source
exercise checks 26 generated calls and 30 complex pole-mass arguments through
the same mapper. The derived-real MW example binds the existing MP expression
in the QCD model. These are source-routing checks, not CMS or derived-mass
numerical qualification.

For threshold group 5, the ordered array is
`[MP__MDL_MT**2, MP__ZERO**2, MP__MDL_MT**2, MP__MDL_MT**2]`.
The existing repaired external top also uses `MP__MDL_MT`. Model initialization
and `MP_UPDATE_AS_PARAM` are unchanged. This fixture's update synchronizes
MU_R, AS and G; direct later edits to DP masses or widths still require the
existing model reinitialization contract. No static mass table is introduced.

Input `P`, the existing repair to `C`, numerator generation, Born normalization,
cards and pole/cache policy are unchanged. `inputs/` retains both physical
points and the earlier repaired-geometry record. Consistent mass squares do
not remove finite-precision shell residuals or physical conditioning. Previous
threshold failures remain evidence, and new refusals must also remain visible.

## Bounded comparison protocol, not launch authorization

`protocol.json` freezes the proposed limits and required records. Source review
must precede any compiler/native authorization. The copied historical candidate
executable is renamed `check.historical-unmodified`; it is not a built prototype.

The first proposed numerical phase is four separately bounded processes:
original/candidate × explicit NinjaRS/TIRust, each on the same ordinary and
threshold points, with a 120-second cap including startup (480 seconds maximum
execution), no retry and no new warmup point set. It uses fixed QP only to
inspect transport and coefficients. The generated path must capture the actual
binary128 repaired `C` and assert it is identical across both reducers and both
source variants before comparing outputs. Per-group MP arguments, old/new mass
squares, ordering, adapter input geometry and native failures must be retained.
The bounded observer/driver implementing that capture still needs to be frozen
and reviewed before compilation; this source package does not silently reuse
the old unchecked driver as a stability qualification.

A subsequent separately reviewed phase must use `CTModeRun=-1` and the original
host direction/FPE/accuracy machinery. Positive fixed mode 4 is not checked QP,
even with `FORCE_STABILITY_CHECK(.TRUE.)`: the checked host block requires the
automatic mode. If automatic evaluation succeeds in DP and never reaches QP,
record that QP acceptance remains unqualified. Any diagnostic forced-DP-rejection
trigger would need its own explicit source review; none is implemented here.

The comparison must retain coefficients, returned status, estimated accuracy,
host flags, actual backend/precision use and every failed point. Ordinary
agreement targets `1e-8` with the existing `3e-6` regression threshold. At the
threshold, agreement alone is not an accuracy certificate: independent analytic
poles, stability acceptance and geometry consistency remain necessary. Both
`UseQPIntegrand` settings and reversal/permutation coverage are later explicit
qualification items. Nothing here changes a guard to make a result pass.

No runtime or performance gain is claimed. The shared numerical lock, one
assigned CPU and single-thread library controls remain mandatory for any later
native phase. The current production campaign remains independent.

## Portable source verification

```sh
python verify.py
# Optional: also verify the ignored candidate source files in this repository.
python verify.py --working-root /absolute/path/to/madgraph7_madloop_tirust
```

The verifier checks the inventory, reproduces all three candidate files and
the 200 mappings from retained source snapshots, confirms preserved CT/TIR
routines and fixed-form widths, and rejects unsupported model expressions.
It never invokes a compiler or reducer. Native archives/modules are recorded
by hash only in `native-identities.json`; they are not bundled into docs.
