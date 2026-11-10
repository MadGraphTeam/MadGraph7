# Private adaptive QP bridge prototype

This package is source only. It is not compiled, executed, installed, promoted,
or a physics qualification. Production and every frozen DSO remain unchanged.
Its basis is adaptive design94f82173 and the current production bridge source;
the scalar/reduction dependencies remain the existing pinned pole-development
stage. The private Cargo package and library names cannot overwrite the
installed `libmadloop_tirust` artifact.

The existing256-bit path remains first. Only a current positive finite powered
condition and an exact known `LaneStatus::Unstable` Gram message permit one
512-bit attempt. The256 gate stays2^-136; the512 gate2^-392 preserves the120-bit
headroom heuristic, not a proved error bound. The source does not change input
precision113, lightlike/affine policies, scalar formulas, Fortran inputs, model
symbols, geometry, Born normalization or MadLoop's stability checks. A promoted
DP integrand remains a DP integrand even with genuine512-bit tensor arithmetic.

Full and pole-only workspaces are distinct types, keyed separately by plan,
direction and working bits. Immutable plans are shared. The whole-result cache
uses explicit tier/mode/policy keys under one existing QP byte budget; no MPFR
subintegral store is attached. Check passes bypass all cache reads/writes.
A512 cache hit requires a fresh eligible256 refusal, so there is no negative
cache. Failed-tier workspaces in both directions are dropped; final failure
also invalidates both matching precision entries. Poisoned contexts do not
retry. Existing event-clear/budget/finalize and Fortran transport invalidation
continue to clear the same numerical-cache owner.

The additive private C query `ml_tirust_qp_precision_statistics` exposes11 words:
256 evaluations, eligible256 refusals,512 evaluations/successes/failures,
256/512 cache hits,256/512 check evaluations, last dispatched output precision,
and the f64 bit pattern of its last eligible powered condition. Counts survive
event clears until context destruction. Foreign argument failures before
dispatch do not reset last-dispatch fields. No timers are introduced.
The original cache-counter ABI layout remains unchanged; the private header
explicitly calls its hits/misses tier probes, since one adaptive request can
probe both tiers. Ordinary non-retried counter behavior stays the same.

Nine new Rust tests are written but **not run**. Six use injected native
outcomes to exercise the actual dispatch/cache/failure path without claiming
numerical truth; three use genuine scalar/tensor evaluations, including full
and pole-only modes of the frozen N4R3 group10/CT4 conditioning refusal. The
fixture comes from the fully captured historical256 topology-C stream, not the
incomplete512 capture. Its decimal records were generated offline and checked
to round exactly to every original binary128 component, including signed zero.
It is a trigger fixture, not an external physics reference. The existing Rust
tests are retained unchanged. Full Fortran IEEE/trap/transport and actual mixed
amplitude qualification remain later gates even if these unit tests pass.

`verify_source.py` checks the exact source package, live input pins, unchanged
DP/transport/lifecycle functions, generated patch, fixture identity and planned
commands. It does not parse Rust types, compile, import native libraries, or run
tests. The `mp_records_with` callback exists solely to inject outcomes into the
same internal dispatcher in unit tests; production dispatch always passes
`State::mp_at`. There is no second numerical engine or test-only production
branch.

Proposed commands, requiring separate approval before execution, are recorded
in `protocol.json`. The initial build uses the existing cached debug dependency
artifacts, one CPU11 job, offline/locked mode, and a900-second cap. A later single
test-binary process uses the shared numerical lock, CPU11, one test thread and
a180-second cap. No test process is launched by Cargo's `--no-run` build.
The original authorized runner2 diagnostic has priority at a free native slot.
No automatic retry is included after build/test failure; preserve the attempt
and review any correction before another build.
