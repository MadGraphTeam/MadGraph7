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

## F. Heavy processes and slow references (user, 2026-10-22, refined)

- All MASTER_PLAN §5.7 2->5 processes, including g g > g g g g g and
  u u~ > g g g g g, remain pyAmpliCol targets. Push pyAmpliCol's own generation
  and validation for them whenever it has a realistic shot at completing
  generation (estimate the cost first; otherwise report a precise scaling blocker).
- Do not launch or wait for a reference-provider generation that cannot complete
  in about 45 minutes or in a time comparable to pyAmpliCol's own generation of
  the process. Validate by other means then (Ward/gauge identities, Catani pole
  structure, LC/NLC/FC and DD/AP consistency, stability checks, lower-multiplicity
  oracles) and use references that already exist.
- Measured provider generation on the development host: g g > g g g g g MadLoop
  > 17 h (colour factors ~89 days by MG5's estimate), Recola > 57 h unfinished;
  u u~ > g g g g g MadLoop 12.2 h, Recola 2.8 h (results exist); g g > g g g g
  MadLoop 2.4 h, Recola 34 min (results exist).
- Where no matched provider timing is obtainable, the speed gate is reported as
  such (never extrapolated) and the user decides how it is judged.

## G. Portable prepared artifacts (user, 2026-10-22)

Artifacts prepared and compiled with SymJIT at -O2 (the default
jit_optimization_level) must be portable so that prepared artifacts for heavy
processes can be distributed: relocatable to another machine, user and path; no
absolute paths or workspace locations needed at load; no load-time regeneration
of the process (the full program, including trees, Born and counterterm scalars,
is in the artifact); no dependence on the build host's CPU features (SymJIT
applications are stored as IR and recompiled at load, as the tree engine's
storage-v4 plane kernels already are); bitwise-identical binary64 results after
relocation with the same native build. A dedicated acceptance test covers this.

## H. Generation budget at high multiplicity; checks without a provider timing (user, 2026-10-23)

- For one-loop matrix-element generation at high multiplicity, generation may
  take up to O(hours) and must stay below 150 GB of memory. This replaces the
  earlier "g g > g g g g g full colour < 10 min and < 8 GB" target. Generation
  and runtime must still be optimised as far as possible; the per-point speed
  gates of section D are unchanged wherever a matched provider timing exists.
- Order: first complete the planned 2->5 set in full, then push the limit.
  g g > g g g g g must be pushed through. g g > 6 g (2->6, the stretch target)
  is to be attempted if judged doable, leading colour first and full colour
  next.
- Where no matched provider timing exists (g g > g g g g g), the speed result
  is reported only, without a gate. It must come with sanity checks:
  rotational invariance, and Ward identities probed by replacing one external
  gluon's polarisation vector by its momentum.

## I. UV and IR pole selection (user, 2026-10-23)

- The user's words: "you can now pin/rebase the latest main of ninjars and
  oneloopmasters as those extend now to the possibility of specifying to compute
  just the IR or UV poles of the reduction, which is an option that should be
  broadcasted to the user-provided settings in pyAmpliCol for the one-loop matrix
  elements so that the user can also choose to compute / access only the IR or UV
  poles (not just no poles) if they so desire".
- Requirement (extends B): the one-loop matrix elements take a runtime pole
  option `poles = "all" | "none" | "uv" | "ir"` for the same generated artifact.
  `"all"` stays the default and its results are unchanged; `"none"` is the
  finite-only mode of B with all its savings; `"uv"` and `"ir"` return only the
  ultraviolet or only the infrared pole parts. The finite part is identical in
  all four modes. The option reaches TOML, CLI, the Python API and the result
  accessors (and the SDKs where they evaluate loop virtuals).
- Semantics, decided by the user (~19:20 UTC, relayed by the coordinator's
  DECISIONS 2026-10-23; binding): "poles='uv' should return the UV poles of
  the bare (unrenormalised) one-loop amplitude, with the counterterms' UV
  poles also accessible separately so I can check renormalisation;
  poles='ir' returns the IR (Catani) poles of the renormalised virtual."
  - `"uv"`: the selected values (`VirtualEvaluation.values` and the
    per-order, correlated and resolved outputs) are the UV poles of the bare
    loop amplitude interfered with the Born, the double pole zero. The
    counterterms' UV poles (`uv_local`, `uv_external_fields`, their sum `uv`,
    and `r2`) are in `pole_parts`; bare + counterterm UV poles = 0 per
    Laurent coefficient where renormalisation is complete.
  - `"ir"`: the IR poles of the renormalised virtual, equal to Catani's
    operator.
  - The finite part is identical in all four modes; `"all"` stays the default
    and `"none"` the finite-only mode.
  - Design §12.4 states what is a definition and what is evidence: a UFO
    model writes the external-field factors with one eps, so their UV part is
    fixed by the UV finiteness of the renormalised virtual and the bare +
    counterterm cancellation in `pole_parts` holds by construction. The
    independent evidence is the bare-loop UV residue against the generic
    engine's UV-selected masters, bare UV + UV_local + the textbook
    Feynman-gauge field renormalisation of every leg = 0, and the `"ir"`
    poles against Catani's operator.
- Companion plan (coordinator decision under the user's relaxed generation
  budget, DECISIONS 2026-10-23): kept as built, always generated and stored,
  so one artifact serves every mode. Its generation cost (about 11 % of the
  slice generation at five legs and 15 % at six, plus export) is paid by
  every process; its evaluation cost only by `"uv"` and `"ir"`: +1 %
  (u u~ > e- e+ g g, six legs with a photon), +7 % (g g > g g g, five legs)
  and +13–18 % (g g > g g g g, six legs; idle and loaded host) over `"all"` (design §12.4).
- As built (WP2.10 after WP2.6, design §12.2 and §12.4): TOML
  `evaluation.poles`, CLI `--poles` and the API `poles=` take all four values
  (`PoleSelection.UV`, `.IR`); one artifact serves them all (it stores the
  companion plan of the certified-scaleless families that `"uv"`/`"ir"` run;
  artifacts written before the split are refused at load with a regeneration
  message, no backward compatibility). `VirtualEvaluation.pole_parts` holds
  both origins by component (virtual, bare, uv, r2, uv_local,
  uv_external_fields), also for correlated and resolved outputs. The SDKs
  evaluate legacy plan-v3 artifacts only; v4 SDK evaluation (WP3.1) takes
  the same option.

## J. Counterterms as plain tree-level insertions; no loop matching (user, 2026-10-23)

- R2 and UV counterterm vertices, two-point ones included, are ordinary
  tree-level vertices of the recursion, with the constraint that each
  contribution contains exactly one counterterm insertion. They are not mapped
  onto their loop counterparts. MadLoop-style matching of R2/UV to the loop
  (witness tables of divergent loop routings, admitted alternatives per
  routing, leg-mask admission, fallbacks to site closures) is not used in
  production: it is complicated and costly. Site closure and similar machinery
  may remain only as test oracles.
- Admission of counterterm terms = the requested coupling orders, plus the
  exactly-once constraint, plus a `particle_veto` filter: a coupling term whose
  `loop_particles` contain a vetoed particle is dropped.
- `particle_veto` is a public generation option (TOML, CLI, Python API; stored
  in the artifact). A vetoed particle is also removed from every tree-level
  vertex in all generation steps (Born, blocks, loop content and propagators),
  so loops and counterterms stay consistent.
