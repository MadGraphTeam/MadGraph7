# Recurrence Native Runtime Pre-Canary Audit

This note records the independent review performed after the first recurrence
artifact/runtime lane was wired, and before accepting a public end-to-end
canary. It complements `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` and is a
checklist for the next independent re-audit.

## Confirmed Foundations

- The recurrence path forks before `GenericDAG` construction.
- Recurrence generation does not construct process-specific evaluators or
  SymJIT applications.
- The recurrence PACBIN member is authenticated consistently by the writer and
  loader.
- For complete `d d~ > z g g`, crossing, normalization, closure accumulation,
  prepared-kernel addressing, and default model-parameter projection showed no
  static blocker.
- The private built-in/UFO recurrence checks reproduce the two legacy LC flow
  values at the deterministic validation point.

## Required Corrections

1. Public physical flow identifiers must map explicitly to construction-sector
   and replay-target identifiers. Numeric equality between these domains is not
   a valid contract, especially for selected, reflected, or permuted flows.
2. A selected flow must execute only the replay targets needed for that flow.
   Computing every replay target and filtering during reduction is correct but
   defeats the topology-replay performance contract.
3. Recurrence loading must install the model-parameter derivation evaluator so
   derived parameters are recomputed after runtime parameter updates.
4. The low-level native `evaluate_into` path must reuse selector, momentum, and
   output storage after warm-up. Python convenience APIs may allocate returned
   arrays.
5. The current replay-factor reduction uses the squared norm of a complex
   factor. This is valid for resolved squared LC components, but a future
   complex-amplitude or exact execution path must retain the full phase.
6. The all-flow-union schedule is not yet implemented and remains a separate
   milestone after topology-replay correctness.

## Acceptance Checks

- Generate, publish, load, and evaluate built-in and UFO-SM `d d~ > z g g`
  through the public API without diagnostic bypasses.
- Compare every `(flow, helicity)` component with compiled mode and the legacy
  deterministic-point oracle.
- Exercise selected non-representative and reflected flows to prove the public
  mapping is explicit.
- Confirm selected-flow execution counters contain no unrelated replay targets.
- Change independent model parameters and compare all derived parameters and
  matrix elements with compiled mode.
- Measure warmed allocations in the native recurrence loop.
- Re-run an independent architecture audit against this checklist and the
  original AmpliCol audit before claiming parity.
