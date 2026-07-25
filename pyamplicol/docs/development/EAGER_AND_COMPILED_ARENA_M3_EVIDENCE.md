# Eager and Compiled Direct-Arena M3 Evidence

This document records the risk-first milestone that precedes production
scheduler integration. The measurements below isolate real callable and
scheduling work; they are not end-to-end runtime claims.

## Exact feature history

- Plane-native amplitude reducers: `95e4f6e`.
- Manifest-driven compiled engine prototype: `a03b986`.
- Native compiled DirectApplication ABI prototype: `685cba2`.
- Real eager plan-v3 Direct-Arena prototype and selector corrections:
  `56feb0c`.
- SymJIT candidate: `2.21.1` at
  `48197f32536c894b51ef25b2cf05ddd05c22675f`, projected into candidate
  build overlays from `dependencies/contributor-lock.toml`.

The canonical release manifest continues to describe the published,
fail-closed release dependency set. Candidate builds use the repository's
existing overlay projection and patched local SymJIT checkout; release
metadata is not rewritten to pretend that this candidate is verified.

## Compiled fused-leaf prototype

The retained real fixture is
`.agent-work/artifacts/ddbar_z3g_compiled_o3`, artifact SHA-256
`7c0afee1...` and PACBIN SHA-256 `78712035...`. The prototype lowers the
canonical recursive manifest leaf layout directly to arena bindings, owns one
shared aligned zero plane, validates selector coverage and producer closure,
and preserves the compressed-O3 fused application.

For one real selected-LC late leaf over 129 points, seven interleaved samples
of 200 repetitions gave:

| path | median per tile |
|---|---:|
| Direct-Arena fused leaf | 23,443 ns |
| legacy gather/call/scatter | 47,385 ns |

The direct/legacy ratio is `0.494735` (`2.021286x` throughput). Parity passed
at 7, 127, 128, and 129 points with `rtol=1e-12`, `atol=1e-15`. The warmed
loop allocated zero times and reported zero forbidden leaf gather, scatter,
or remap traffic. Source filling, boundary transpose, amplitude reduction,
and whole-runtime dispatch are outside this timing boundary.

## Native compiled callable prototype

The backend-neutral native ABI uses fixed split-plane and scalar descriptors,
factor-free overwrite semantics, checked `u32` dimensions, explicit
no-input/output-alias and no-output/output-alias requirements, and an
explicit no-foreign-unwind status contract. The synthetic C++ validation
kernel is plane-native rather than a dense-row wrapper, but it is scalar and
is not evidence for a production C++ or assembly emitter.

Four focused tests passed and one manual microbenchmark remained ignored.
The synthetic adapter loop measured 55 ns versus 343 ns for a
gather-plus-129-scalar-calls-plus-scatter oracle. This is only ABI plumbing
evidence; it is not a production-kernel or end-to-end result.

## Eager table-aware prototype

The retained real fixture is
`.agent-work/artifacts/ddbar_z3g_eager_o2`. It contains four real stages,
96 invocations and attachments, and 24 distinct destinations. The selected
stage is position 3 / stage index 4 / kernel 7. A real selector group retains
four of four relevant rows.

At 129 points, the full-stage oracle compared 6,192 complex values: 6,159
were bitwise identical and the maximum relative difference was
`1.158e-15`. The selected oracle compared 258 values: 257 were bitwise
identical and the maximum relative difference was `1.503e-16`.

Seven release samples gave:

| invocation slice | direct median (MAD) | packet median (MAD) | reduction |
|---|---:|---:|---:|
| full stage | 62.589 us (0.388) | 108.707 us (0.390) | 42.4% |
| selected group | 2.366 us (0.020) | 4.543 us (0.068) | 47.9% |

The direct path reported zero forbidden packet, gather, and scatter traffic;
one-time arena/factor initialization bytes were reported separately. These
figures time invocation and fanout only. Whole-plan source fill,
finalization, closure, and amplitude reduction remain production-migration
work.

Selector hardening recomputes the first retained destination write after
pruning, treats an explicitly empty selector domain as pruning a domainless
row, and authenticates callable role plus exact ordered semantic inputs and
output width.

## Plane-native reductions

The shared reducers operate on borrowed split-complex amplitude planes for
resolved, materialized-helicity resolved, and materialized-helicity
add-into paths. They preserve the existing loop and summation order.
LC/NLC/full-color parity passed at the 129-point tail, and the warmed
add-into path allocated zero times.

## Production gates still open

- Wire complete eager and compiled plans into their independent public
  execution lanes.
- Fill sources, crossed momenta, parameters, factors, and canonical amplitude
  planes directly without a full row-major state.
- Implement schedule-safe clearing, deterministic shape-derived outer
  tiling, per-point selector grouping, all finalization and closure roles,
  and plane-native totals/resolved execution.
- Prove LC topology-replay and union-flow selectors plus NLC/full-color
  parity and unprofiled end-to-end wall-time gains.
- Replace real C++ and assembly dense-row callables with plane-native
  emitters where those public backend configurations remain supported.
- Complete malformed-artifact, native-language API, packaging, x86-64, and
  final release-policy evidence.

