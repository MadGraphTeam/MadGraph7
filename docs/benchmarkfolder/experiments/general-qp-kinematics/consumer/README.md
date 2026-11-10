# Private QP identity consumer, source preparation

This candidate composes the frozen classifier97f with the corrected paired-MP
mass exporter and guarded caller. It changes only generated QP backend8/9 paths;
DP and other reducers retain their inherited sources. Production and the frozen
classifier package are untouched. No Born, coordinate-repair or normalization
policy is implemented here. Source review is required before any native test.

The paired helper already has the emitted LOOPNUM and current CTMODE. Its
selected QP8/9 calls pass those integers and policy1 explicitly to new
`MP_{NINJARS,TIRUST}_LOOP_QPMASS_KIN` entries. These loop entries use the explicit
direction for ordering and checks, not a CT COMMON read. TIRust's new tensor
diagnostic also receives group/policy/direction explicitly. The old diagnostic
signatures and their inherited arithmetic remain available unchanged. Numerator
and split-order storage retain the existing generated loop-helper contract.

The third adapter variants derive from the same scientific template as the
legacy and paired entries. Only formal metadata and matrix-builder dispatch
change; cache/check policy, tensor mapping, contraction, pole behavior, mass
transport, normalization and failure handling use the original template. No
second independent adapter algorithm is maintained.

The shared new matrix builder has two explicit policies:

* Policy0 delegates the original matrix builder immediately, including its
  historical internal-mass threshold guesses. It is the compatibility path.
* Policy1 computes the invariant using the original arithmetic order. A
  replacement requires a known direct-single or conservation-complement
  identity, the process's declared on-shell flag, and a supported real physical
  external mass. Its target is the square of the existing MP model mass symbol
  associated with that external particle. It never guesses from internal
  denominator masses or projects a complex pole mass onto a real shell.
  The relative massive residual must be below OSThres. For massless legs, the
  squared maximum absolute component of the actual pair-difference four-vector
  normalizes the residual. Complements additionally require the literal
  first-shift closure norm divided by that pair scale to be at most OSThres
  (zero-scale closure must be exactly zero). Invalid tolerance, unsupported mass, composite,
  unknown, closure-only, off-shell descriptor or failed residual leaves the
  computed invariant. Other policy values also leave it computed.

Policy1 requires every shift component to be finite before any shell adjustment;
MAXVAL alone is not treated as a NaN check. Massless pair scales outside the
conservative square-safe range are declined before normalization, which must
also be finite and positive. A closure norm larger than the pair scale is
declined before division. These guards do not mask exception flags or change
traps. Computation of an invalid input's ordinary invariant retains the existing
arithmetic and its caller-visible exception behavior.

This pair-local normalization is deliberately different from the historical
average shift energy. A hard common shift must not make a large relative defect
of a soft external leg pass. Both the shell and complement checks therefore use
that leg's actual pair difference. Policy0 retains the historical normalization
without this change. Soft-leg and extreme-scale native tests remain required;
the source formula alone is not a numerical qualification.

These tests are explicit numerical heuristics, not a repair-success or amplitude
certificate. The inherited adapter momentum-conservation gate also remains.
A caller of the new explicit diagnostic must supply the group/direction that
actually generated its corners; the routine cannot prove an arbitrary caller's
ancestry from floating-point numbers. It does not infer effective C, model epoch,
or integrand ownership from MP_DONE or a stale global group. Coordination with
the separate Born/coordinate owner remains necessary before adoption.

External targets use existing generated MP_masses declarations, including real
physical masses when the internal scheme uses CMASS. No card is reparsed. Zero
targets need no model read; nonzero targets reject nonfinite or negative masses.
Source model type and identifier checks conservatively decline unsupported
symbols. Existing model initialization and reinitialization remain required.
Scale prescriptions are intentionally inherited separately for each reducer.

The supported preparation scope is ordinary scalar optimized export. Vector
export is explicitly rejected by this private composition rather than silently
generating an unqualified scalar consumer. General off-shell currents, custom
propagators, CMS numerical behavior and loop-induced amplitude acceptance have
not been qualified. A classified external identity alone is insufficient to
claim physical correctness, and the prior private QCD observations are not
evidence for a general fix.

Run from the repository root with `PYTHONDONTWRITEBYTECODE=1 python -B` after
`source tools/tirust_env.sh`: `prepare.py` composes the sources and records hashes;
`test_generation.py` performs only Python/source-generation checks. No native
qualification is implied by those tests.

Eleven source checks pass: unchanged legacy and paired public adapter text;
unchanged DP helper text with QP dispatch isolated to selected8/9 branches;
actual SM and CMS physical-mass symbol selection; unsupported complex model
mass rejection; explicit compatibility/identity/residual gates; fixed-form
rendering; scalar-only guard; all19 retained small composite channels in both
directions; actual ungrouped224/loop-induced16 reduction identities; and exact
rational examples distinguishing soft-leg validity from hard-shift dilution.
The separate classifier's compiled integer test passed8,325 cases, which does
not qualify this floating-point consumer. `freeze.py` records this package;
`verify.py` checks its complete inventory and protected source inputs without
regenerating or executing anything.
