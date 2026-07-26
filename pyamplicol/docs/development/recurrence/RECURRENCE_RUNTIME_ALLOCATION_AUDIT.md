# Recurrence Runtime Allocation Audit

This audit freezes the allocation constraints for the native recurrence lane.

## Current State

The numeric tile loop itself contains no successful-path collections, but the
private Python diagnostic entry points rebuild the program, backend, execution
plan, workspace, inputs, outputs, and Python result objects on every call.
Those entry points are not a warmed runtime and must not be used for headline
timing or allocation claims.

Backend fallback and SymJIT tail buffers can grow on first use. They become
stable only after their maximum packet and tail capacities have been reached;
recurrence currently has no test proving that warm-up contract.

Selected-helicity evaluation also still sizes and clears output storage for all
live destinations. When most helicity-sector combinations are live, this is
effectively dense even though the semantic destination table is sparse.
Momentum storage is similarly duplicated per current rather than interned by
unique momentum form.

## Required Persistent Runtime

- A loaded recurrence object owns the authenticated program, backend handles,
  execution plan, and fixed workspace.
- A separately prepared `RecurrenceSelectorPlan` owns active replay targets,
  packet ranges, destination-to-local-slot maps, stable point permutations, and
  reusable sorting buffers.
- Selector plans allocate during creation, never during repeated evaluation.
- Resolved storage is sized from active destinations. Helicity sums reduce
  directly into caller-owned sector totals instead of materializing the full
  helicity by sector array.
- Unique momentum forms are interned; currents store momentum-slot IDs.
- Backend packet, fallback, and SIMD-tail capacity is reserved from plan maxima
  before the first measured call.
- Native `evaluate_f64_into` accepts caller-owned inputs and outputs.
- Python and NumPy convenience-result allocation remains outside native timing.

## Hard Tests

Using the existing counting allocator test infrastructure, warm the exact
maximum shape once and then require zero allocations across repeated:

- selected-helicity, helicity-sum, resolved, and topology-replay calls;
- batches 1, one SIMD tail, 128, and tile-plus-tail;
- homogeneous, pre-grouped, alternating, and random prepared selector plans;
- mock, JIT, C++, and ASM backend smokes.

Tests also assert backend capacities do not change after warm-up. Public
recurrence dispatch remains gated until these tests pass.
