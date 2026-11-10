# n1 overhead: retained evidence and source-only follow-up

The qualified historical candidate/baseline ratio 1.121 (upper95% bound1.139)
remains a retained failure. It is not a measurement of the new registered
implementation. The current generated n1 `coef_construction_1.f` calls the
previously promoted `DP_TERMINAL_WL_1_1`; the later 2x1/3x1 extension does not
add an n1 call. The earlier isolated 1x1 elemental experiment measured
terminal/original9 = 0.88866 [0.87559,0.90193], and terminal/old8 = 0.97852
[0.96888,0.98825], with identical coefficients. Those diagnostics are not
formal acceptance, but make it incorrect to assume the entire old gap remains.
The current formal result must come first.

The historical n1 attribution observed exactly two N3/rank2 DP evaluations,
one in each independent direction. All calls succeeded directly. Its relevant
non-overlapping Fortran stage brackets, in microseconds per amplitude, were:

| Region | Historical descriptive time |
|---|---:|
| Initialization/budget | 0.750 |
| Ordered momentum/shift preparation | 0.785 |
| On-shell kinematic matrix | 0.701 |
| Invariants/options | 0.120 |
| Wrapper plus Rust (inclusive) | 8.196 |
| Mapping/validation | 0.878 |
| Contraction/normalization/final checks | 0.586 |

The adapter regions outside the wrapper sum to3.819us, while historical
unprofiled medians were roughly71us candidate and63us baseline. Even deleting
all timed adapter-only work would not close that old8us difference. Deleting
it is neither possible nor proposed. Automatic-array/IEEE entry overhead and
common host/numerator work lie outside those brackets, so this is not a strict
global upper bound on all Fortran optimizations. The independently measured
1x1 terminal saving demonstrates that common numerator work was material.

The original nested native level-2 profile had geometry/arguments2.856us,
scalar masters1.148us and recurrence1.961us, with upstream total7.083us. These
timers are inclusive and overlap other reported regions; do not add them to
the Fortran wrapper or infer a saving by subtraction. After the directional
workspace fix, a separate profile on a different measurement setup reported
geometry1.330us and FFI10.704us. The split-workspace whole-amplitude diagnostic
measured an approximately2.15% improvement. That reuse is already implemented;
its previous cost is not a new optimization opportunity.

The current n1 process has one loop group and one squared order. Its ordinary
Born export has no generic host tensor cache. The two observed tensor calls
are primary and independent reversal, not two primary groups sharing identical
outputs. Reusing one result for the other would invalidate the check. A new
Fortran cache before preparation would add key construction/copy/lookup with
no observed repeated primary top-level result here. This does not prove zero
within-call subset sharing inside TIRust; existing subset/master hit counters
would be needed before making that more specific claim. Cross-event exact
reuse cannot be assumed, and default event clearing is part of the contract.

Additional plan registration is also not a hot n1 cost: generated code checks
the context epoch, registers all plans once, and historical register timers
were zero. QP decimal transport does not occur in this all-DP workload.
Changing it cannot explain the measured n1 slowdown.

Remaining small hypotheses have limited measured scope:

* Memoize the last successful budget in the shared Fortran context, including
  epoch/finalization, invalid updates and cross-subprocess calls. The original
  native budget-call bracket was only about0.10us per amplitude. Skipping a
  successful setter must also preserve public error-reset behavior; simply
  remembering the number is insufficient.
* Reserve the DP input vector's exact capacity, retaining every validation and
  original order. The complete packing bracket was about0.49us; allocation is
  only a fraction. This is a possible private Rust change, not a Fortran gain.
* Avoid an unchanged workspace-store reattachment after proving identity and
  failure/recreation behavior. Its old bracket was about0.16us. Check workspaces
  must remain detached; no numerical result may leak into independent checks.

None is a credible standalone fix for the old9–12% difference. Repeated
allocation/O3 experiments already failed to establish a whole-amplitude gain:
pointer scratch0.99748 [0.98826,1.00679], exact-shape sequence1.00073
[0.99014,1.01143], common-allocation1.01120 [0.99373,1.02898]. The public DP
finite scan removal was rejected because it weakened defensive behavior.
These results should not be rerun under a new label without a new cause.

If current formal n1 still fails, first obtain current uninstrumented controls
and a short source-bound attribution of the existing split-workspace/terminal
implementation. The upstream structural-variant selection and recurrence
scratch may then warrant a narrow reusable-plan experiment; numerical
geometry must still be recomputed when input changes. Keep the Fortran
numerator/contraction, compiler recipes, exact operation order and independent
checks. No such new patch or experiment is implemented or authorized here.

Inputs: `../n1-profile/{fortran/README.md,fortran/summary.json,
bridge/README.md,bridge/analysis.json,bridge/followup-candidates.md,
fortran-scratch/timing-summary.json,fortran-sequence/timing-summary.json,
fortran-common-allocation/README.md,fortran-dp-finite/README.md}` and
`../dp-terminal/{README.md,analysis-summary.json}`. Their identities are bound
in this proposal's source inventory, alongside the current n1 generated
numerical sources. No new timing or numerical evaluation supports this note.
