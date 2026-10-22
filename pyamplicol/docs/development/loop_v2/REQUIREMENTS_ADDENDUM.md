# Loop engine v2 — binding user requirements addendum (2026-10-20)

These requirements come directly from the user after reviewing the v2 design.
They override the design document (`loop-engine-v2-design.md`) where they differ.

## A. Freedom in structure and algorithms; performance gates unchanged

The structure and algorithms anticipated in MASTER_PLAN and in the design may be
changed freely (data structures, schedules, reduction strategies, rescue tiers,
anything internal), provided the public API below is honoured:

- colour correlations and spin correlations;
- per-coupling-order partial evaluations (every squared coupling order separately
  accessible) — this must ALSO be provided at tree level (extend the tree API);
- leading colour (LC), next-to-leading colour (NLC) and full colour;
- stability tests (section C).

This freedom concerns HOW the results are computed. It does NOT relax any
acceptance requirement. In particular the performance acceptance gates of
MASTER_PLAN §5.8 still hold in full, for a normal evaluation that can be compared
with other tools (MadLoop, Recola, OpenLoops2, ...): faster than MadLoop, and at
most 1.30x each available matched Recola/OpenLoops reference (section D gives
the comparison protocol). Correctness requirements are likewise unchanged.

Performance priorities (user, 2026-10-21):
- What matters is 2->3 and higher multiplicities. The gates apply to processes
  with at least five external legs (2->3 and beyond); 2->2 timings are reported
  but carry no gate, and being faster than MadLoop at 2->2 is not a goal.
- The aspiration at 2->3 and beyond is to be the fastest of all tools (MadLoop,
  Recola, OpenLoops2, ...). The formal acceptance threshold remains: faster than
  MadLoop and at most 1.30x each matched Recola/OpenLoops reference.
- Optimisation effort should therefore target the asymptotic, high-multiplicity
  regime first (the required 2->5 set and the 2->3/2->4 ladder below it).

Consequences: lifting/integrand-sum reduction, hybrid rescue tiers, schedule
choices, etc. may be adopted whenever measured faster at equal correctness.

## B. Poles are optional at runtime

- The Born-virtual result exposes the finite part and the 1/eps and 1/eps^2
  coefficients, but pole computation must be switchable OFF at runtime
  (finite-only mode) and ON on request, for the same generated artifact.
- Exploit every saving available in finite-only mode (e.g. skip pole-only
  epsilon sectors, pole-only counterterm columns, pole parts of masters, the
  IR-pole consistency check). If an efficient hybrid is too complicated, two
  independent interpretation/execution lanes (finite-only vs with-poles programs)
  inside one artifact are acceptable.

## C. Stability estimates are returned by the API; escalation is the user's job

- Evaluations return, alongside the values, an estimate of numerical stability
  (relative accuracy) measured with MadLoop-style checks:
  1. **Rotation checks**: re-evaluate with Lorentz-rotated external kinematics.
     The user chooses which rotations (axes/angles or rotation specifications)
     and how many, including 0.
  2. **Reduction-direction check**: re-reduce the same numerators with the loop
     direction reversed (or an equivalent relabelled reduction frame), reusing
     numerator coefficients and cached masters — the cheap check.
  Both modes must be supported, individually configurable and combinable. "No
  stability check" must always be possible; then no precision estimate is
  returned (explicitly absent, not a fake value).
- The library does not need to escalate precision automatically: the user decides
  whether to re-evaluate in DoubleFloat/arbitrary precision based on the returned
  estimate. (Existing automatic rescue may remain as an optional convenience but
  is not required and must not be on by default for timing.)

## D. Speed-gate protocol

- The performance gate (faster than MadLoop; <= 1.30x each available Recola /
  OpenLoops) is mandatory and is evaluated on the FINITE PART ONLY, for a normal
  evaluation comparable across tools, for processes with 2->3 and higher
  multiplicity (2->2 timings are reported without a gate).
- Reference codes run in their production mode with pole computation disabled
  whenever they allow it (OpenLoops truepoles 0; Recola without IR-pole
  evaluation; MadLoop at its production setting), Collier caches as in production.
- pyAmpliCol is timed in finite-only mode with stability checks disabled.
- Correctness validation still covers all three Laurent coefficients.

## E. Tree level

- Per squared-coupling-order partial results must be accessible at tree level too,
  through the same public API conventions (Python, CLI, SDKs where applicable).

## F. 45-minute generation rule (user, 2026-10-22)

Do not try, verify or compare against any process whose generation time exceeds
45 minutes, whether the generation is pyAmpliCol's own or a reference provider's.
Under this rule g g > g g g g g and u u~ > g g g g g are excluded (provider
generation: MadLoop > 17 h / 12.2 h; Recola > 57 h / 2.8 h), and the MadLoop
reference for g g > g g g g (2.4 h) is excluded (Recola, 34 min, and OpenLoops
remain). If pyAmpliCol's generation of a process would exceed 45 minutes, report it
as a scaling blocker instead of running it. Exclusions are stated explicitly in
MASTER_PLAN and the live report; substitute targets need the user's approval.
