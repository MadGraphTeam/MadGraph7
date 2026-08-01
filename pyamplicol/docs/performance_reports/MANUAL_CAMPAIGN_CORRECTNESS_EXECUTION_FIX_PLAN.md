# Manual Campaign Correctness and Execution Repair

## Goal

Eliminate the correctness, scalability, scheduling, dependency, and watchdog
defects exposed by the manual performance campaign while keeping the repair as
small and maintainable as possible. Numerical-current reuse must never change
physics, oversized optional evidence must never prevent an otherwise valid
recurrence measurement, workers must remain work-conserving and resource
bounded, legacy selected-flow measurements must not construct irrelevant
all-flow colour bases, and only compatible historical results may be recycled.
After focused verification, integrate the fixes and push them to `main`.

## Implementation constraints

- This document is a plan only. Do not begin implementation until its goal is
  explicitly assigned.
- At implementation start, inspect and safely update to the authoritative
  `origin/main` without resetting, stashing, deleting, or overwriting user
  results. Never request command escalation.
- Preserve the existing manual campaign outputs and attempt history. Do not
  clear the user's report data merely to simplify testing.
- Keep validation lightweight and metadata-based. Add no artifact hashing,
  recursive history verification, or parallel provenance system.
- Use focused unit tests, small deterministic integration cases, and at most
  one final development install. Do not run a broad performance campaign,
  release matrix, or long n7/n9 acceptance job without separate approval.
- Keep failed numerical validation fail-closed. Do not loosen tolerances to
  make discrepancies disappear.
- Use existing pyAmpliCol and report APIs and CLI paths. Do not duplicate the
  generation or profiling engines in the campaign controller.

## Confirmed diagnoses

| Area | Confirmed cause | Repair stance |
|---|---|---|
| Recurrence evidence envelope | The optional numerical-current evidence producer materializes a complete multi-point `Decimal` graph before its post-hoc spool and rejects it against a fixed 1 GiB safety envelope. | Keep the safety envelope; make oversized optional reuse fall back to the correct reuse-disabled plan instead of failing generation. |
| Compiled plane-Arena identity | Canonical evaluator serialization intentionally omits redundant `backend`, while two report consumers incorrectly require `backend == "jit"`. | Authenticate canonical `compiler_type`, capability, ABI, layout, optimization, path, and digest fields. |
| LC four-quark selector agreement | Without an AmpliCol baseline, selected-flow and all-flow artifacts derive selectors independently and are rejected only after both measurements. | Supply the selected-flow peer's selector to all-flow before measurement. |
| False zeros and top-pair discrepancies | Numerical relations were certified on sampled helicities and promoted to broader selector domains. All five failed cells applied numerical-current reuse. | Contain unsafe reuse first, then diagnose reuse-on/off at higher precision and permit only member-safe/stable relations. |
| Eighty-nine skips | Every prerequisite was scheduled; 81 were blocked by the compiled identity defect and eight by recurrence root failures. | Fix the roots; retain fail-closed dependency behavior. |
| Legacy n9 selected-flow failure | Post-generation common-component validation unnecessarily launches the direct all-flow probe. | Obtain the fixed-helicity component from the selected-flow generated library. |
| Worker under-utilization | Global rank waves wait for the slowest cell before admitting any later-rank independent work. | Replace waves with a dependency-aware, work-conserving ready queue. |
| Thirteen-hour worker | The one-hour guard closes before unbounded linking, profiling, and common-component validation; legacy subprocesses have no individual deadline. | Enforce a hard default worker wall limit and bounded stage/subprocess budgets. |

## A. Minimal handling of oversized recurrence evidence

The 1 GiB boundary exists to prevent unbounded Python `Decimal` graphs,
canonical JSON copies, and decompression bombs. It should not be raised or
removed merely to admit a larger process. Numerical-current reuse is an
optional optimization; the unmodified recurrence plan remains the correctness
baseline.

Implement the following narrow fallback:

1. Introduce a typed, geometry-specific outcome for an evidence capture that
   cannot fit its existing safety envelope. Do not catch arbitrary
   `ValueError` or suppress malformed-evidence, non-finite, ABI, or physics
   failures.
2. When certified numerical reuse encounters only that geometry outcome,
   return the existing no-applied-relations path and continue generation with
   the original recurrence plan.
3. Emit an explicit progress warning and provenance record containing:
   - requested relation mode;
   - effective reuse state;
   - a stable `evidence-envelope-fallback` reason code;
   - current/component/probe/scalar/row geometry;
   - zero certified and zero applied relations.
4. Never claim that reuse was active when the fallback was taken. The
   effective configuration, artifact inspection, report metadata, and manual
   CLI reproduction must all expose the fallback.
5. Preserve the fixed envelope and every existing evidence authentication
   check for shapes that continue through the evidence path.
6. Do not change the Rust transport, introduce a new evidence file format, or
   add a streaming subsystem as part of this repair. A genuine streaming
   optimizer can be considered later only if large-shape relation reuse is
   shown to be worth its complexity.

This makes the failing n7 recurrence cell measurable under the ordinary
campaign RAM/time caps without allocating the unsafe evidence graph. It may be
slower because the optional optimization was unavailable, which is acceptable
and must be visible in provenance.

Focused tests:

- Feed the exact failing geometry (`116319` currents, `561426` components,
  four candidate probes, four verification probes, ten runtime parameters)
  into the planner and assert a typed fallback rather than an error.
- Assert that fallback occurs before current capture or large allocation.
- Force the fallback with a tiny injected test envelope and compare the
  resulting plan/runtime output with explicit reuse-off output.
- Assert that small supported shapes retain the current authenticated evidence
  path unchanged.
- Assert that malformed evidence, non-finite observations, ABI drift, and
  unrelated failures still propagate as errors.
- Assert exact requested/effective mode, reason, geometry, warning, and zero
  relation counts in metadata.

## B. Canonical compiled evaluator identity

Update the report runner and final audit to use the fields actually retained by
canonical artifact serialization:

- require `compiler_type == "native"` for SymJIT application leaves;
- retain all runtime-capability, storage/application ABI, plane ABI, element
  layout, batch layout, translation, optimization, target, path, and digest
  checks;
- reject an explicitly contradictory legacy `backend` value if present, but
  do not require redundant `backend` metadata;
- leave C++ and ASM native-direct contracts unchanged.

Add a serializer-to-report round-trip test using a real canonical evaluator
payload, plus chunked/un-chunked O1/O3 fixtures and corruption cases. This
single repair should resolve all 179 observed compiled errors.

## C. Canonical LC selector propagation

Give candidate measurement an explicit selector provider:

1. Use the validation baseline selector when one exists.
2. Otherwise, for an LC all-flow cell, use the already-required selected-flow
   direct peer's selector.
3. Derive locally only when the cell has no selector-bearing comparison edge.
4. Validate the supplied contract against the candidate runtime before
   profiling.
5. Retain strict selector equality in direct-agreement attachment.

Add a no-AmpliCol-baseline selected/all-flow unit case that would choose
different maxima independently, and prove that the all-flow worker receives
the selected-flow contract before doing expensive work.

## D. Numerical-current correctness

Treat the five validation failures as a correctness defect, not a tolerance or
rounding problem.

### Immediate containment

- Do not apply a numerically discovered relation beyond the exact physical
  member set on which it was certified.
- Until member-safe application is implemented, disable numerical relation
  application for multi-helicity all-flow selector domains.
- Treat contracted `opposite` relations as unsafe for the binary64 headline
  path until the focused precision comparison below establishes whether their
  problem is semantic or numerical stability.
- Preserve structural/exact reuse only when its independent proof already
  covers the affected execution domain.

### Focused diagnosis

Regenerate only these small cases with reuse off and on, using the same stored
report point and selector:

1. n3 LC `u d~ > e+ ve g`, all-flow, at p16/p32/p80;
2. n4 LC `d d~ > t t~ g g`, all-flow, at p16/p32/p80;
3. one n4 Full or NLC contracted top-pair cell at p16/p32/p80;
4. n6 LC four-quark selected/all-flow after selector propagation.

Compare total values, resolved values, and the LC common component against the
reuse-off recurrence plan and original AmpliCol. Evaluating only the already
pruned reuse-on artifact is insufficient because removed currents cannot be
restored by higher precision.

### Final rule

- If a relation is false for any affected member, reject it.
- If it is correct at high precision but destabilizes binary64 output beyond
  report tolerance, do not apply it to the binary64 headline evaluator.
- Do not broaden evidence collection or add a large streaming architecture in
  this repair. Shapes that cannot cheaply establish the required scope take
  the explicit reuse-off fallback from section A.

Add synthetic two-helicity zero/opposite counterexamples and the n3 charged-
current canary so the former domain-wide promotion cannot regress.

## E. Dependency outcomes

Do not change dependency semantics merely to eliminate the word `skip`.
Required numerical baselines and direct-agreement peers remain fail-closed.
After B, C, and D are fixed, replan focused affected cells and assert that the
89 descendants become runnable or receive only a genuine terminal resource
censor.

For clarity, represent an unavailable prerequisite internally and in the
dashboard as `blocked by dependency`, retaining the exact prerequisite ID.
No new artifact-history inspection command is part of this plan.

## F. Legacy selected-flow and high-count LC probing

### Selected-flow validation

- Extend the generated-library selected-flow probe to accept the requested
  physical helicity and return the fixed-flow/fixed-helicity component.
- Match the requested source helicity through the generated spin aliases.
- Compute the component without applying the summed-helicity multiplicity.
- Use this result for selected-flow LC common-component validation.
- Assert that selected-flow measurement never launches the direct all-flow
  probe.

### Actual three-line LC all-flow

- Add a leading-colour sparse diagonal builder for the three-open-line matrix,
  using linear rather than quadratic storage.
- Verify it against the existing generic builder on zero-, one-, and, if still
  fast, two-gluon cases before using it for 15,120-flow LC input.
- Make the Python scope check accuracy-aware: allow the scalable LC path while
  retaining the existing high-count guard for NLC/full.
- Keep four open quark lines as original-AmpliCol static `N/A`.

Implement the legacy change on the pinned `amplicol_with_patches` branch, then
update only its revision in `dependencies/contributor-lock.toml`. Do not add a
local unpinned patch mechanism.

## G. Work-conserving scheduling

Replace the global rank-wave loop with a deterministic DAG ready queue:

- store explicit scheduled prerequisite edges for baselines, direct peers, and
  resource-lane predecessors;
- use rank only as a ready-queue priority;
- submit another ready cell immediately whenever a worker finishes;
- release dependents when their prerequisite finishes, then let existing
  current-policy checks decide success, censor, or blocked outcome;
- acquire cell and resource-lane locks non-blockingly and defer lock-busy work
  instead of consuming a worker slot;
- replace campaign-wide UFO/compiled serialization with a capacity-one token
  applying only to affected Symbolica work;
- stop new submissions promptly on keyboard interrupt while supervising and
  cleaning all active workers.

The dashboard must expose `Ready`, `Waiting dependency`, and
`Waiting coordination lock`. The scheduler invariant is: no configured worker
slot remains idle while a ready, lockable, resource-compatible cell exists.

Use deterministic sub-second tests with controlled futures and locks; do not
test this by launching a real campaign.

## H. Resource and stage watchdogs

- Change the default total worker wall limit from disabled to 3600 seconds.
- Keep the separate one-hour generation limit and generation elapsed evidence.
- Cover preparation, prewarming/building, per-process generation, compilation,
  and linking with the generation guard even when published generation timing
  intentionally excludes one-time preparation.
- Give runtime profiling and post-profile validation explicit stage budgets,
  each bounded by the remaining total worker budget.
- Start legacy profiling calibration with one point, not a fixed 100-point
  invocation.
- Never request a profiling chunk whose estimated work exceeds its remaining
  stage budget.
- Apply the remaining deadline to every child process and terminate the whole
  process tree on expiry.
- Emit and flush the phase, command, and intended point count before launching
  each subprocess.
- Publish distinct terminal reasons for generation, profiling, validation, and
  total-worker limits; all count as addressed without fabricating timing data.

Test phase boundaries, per-command deadlines, process-tree cleanup, timeout
classification, keyboard interruption, and the one-point calibration rule with
fake subprocesses and clocks.

## I. Cross-revision compatibility

The existing opt-in cross-revision continuation remains, but it must not retain
results produced by the unsafe relation policy.

- Add one narrow numerical-relation correctness ABI/state to result metadata.
- Recycle historical original-AmpliCol, reuse-off, and zero-applied-relation
  currents when their existing metadata is otherwise compatible.
- Reject only historical pyAmpliCol currents that actually applied the unsafe
  relation policy or cannot establish that they did not.
- Perform no content hashing or history scan beyond the already selected
  lightweight current metadata.

## Parallel implementation ownership

Use three dedicated subagents in the first implementation wave, with exclusive
file ownership to avoid overlapping edits:

1. **Numerical correctness lead** — sections A and D, recurrence warm-up/service
   code, numerical metadata, and focused tests.
2. **Scheduler/watchdog lead** — sections G and H, scheduler/resources/manual
   campaign wiring, dashboard counters, and deterministic concurrency tests.
3. **Legacy-oracle lead** — section F, pinned AmpliCol changes, adapter/API
   wiring, revision pin, and focused native/mock tests.

The primary agent owns B, C, E, I, integration, documentation, final focused
verification, commits, and the push to `main`.

After integration, use two fresh read-only audit passes:

- a numerical auditor checks that no relation can escape its certified scope
  and that unsafe historical currents cannot be recycled;
- a concurrency auditor checks worker utilization, locks, cancellation, and
  every watchdog boundary.

## Verification budget

Run only:

- focused unit suites for recurrence warm-up, runner/final-audit identity,
  measurement/agreements, scheduler/resources, legacy adapter/oracle, and
  manual campaign defaults;
- tiny deterministic integration cases for the n3 zero canary, one n4 opposite
  canary, one compiled n1 round trip, and small legacy LC equivalence;
- formatting, lint, and diff checks for touched files;
- one final `just dev-install` after the implementation is committed and the
  worktree is eligible.

Do not run the full campaign. The real n7 recurrence and n9 legacy cases remain
separate, explicitly approved acceptance runs under hard caps.

## Completion criteria

- Oversized optional recurrence evidence produces a visible reuse-off fallback,
  not a failed cell or unsafe allocation.
- The five known numerical failures pass against reuse-off and AmpliCol, or are
  still rejected with a localized unresolved cause; none may be accepted by
  tolerance relaxation.
- Canonically serialized compiled JIT artifacts pass runner and final-audit
  identity checks.
- LC all-flow consumes its canonical selected-flow selector when no AmpliCol
  baseline exists.
- The 89 former dependency skips are resolved by their root fixes or remain
  explicitly blocked by a genuine terminal prerequisite.
- Legacy selected-flow never constructs an all-flow colour basis for common-
  component validation.
- Ten workers remain busy whenever at least ten runnable compatible cells are
  available.
- No worker, profiling stage, validation stage, or child command can run beyond
  its effective deadline.
- Keyboard interruption leaves no active child process or incomplete published
  current.
- Compatible historical data remains available; unsafe relation-bearing
  currents are selectively regenerated.
- Focused verification passes, the final changes are committed, and `main` is
  pushed without starting a broad campaign or release workflow.
