# DP terminal wavefunction source/proof audit

Scope: read-only inspection of the existing QP proof, its tests, and the saved
QCD n=1 generated source. No builds or numerical executions were performed.
This is a recommendation for the isolated fixed-shape experiment, not evidence
that a production DP optimization has been qualified.

## Recommendation

The inspected QCD n=1 graph permits a private diagonal-only **arithmetic**
update for its final wavefunction, subject to the runtime guards below. Start
with rank `(1,1)` and `LCUT_SIZE=IN_SIZE=OUT_SIZE=MAXLWFSIZE=4`. Preserve the
complete original initialization, every consumed expression and its order,
and the existing `CREATE_LOOP_COEFS` / `MERGE_WL` calls. Do not replace the
public dense updater globally or infer terminal status from a routine name.

## Actual operation consumption and provenance

The inspected source directory is
`.tirust-build/n1-profile/fortran/pristine/SubProcesses/P0_uux_z`.

- `coef_construction_1.f:102-108` constructs the unique chain
  `WL0 -> WL1 -> WL2 -> WL3`. The last call is
  `UPDATE_WL_1_1(WL(...,2),4,COEFS,4,4,WL(...,3))` at line 107, followed
  by `CREATE_LOOP_COEFS(WL(...,3),2,4,...)` at line 108. No later updater
  reads WL3 in the inspected generated source.
- `polynomial.f:621-646` initializes all 16 matrix entries for coefficients
  0 through 14, then evaluates the same 15 statements for each `I,K,J`.
  Holding `J=I` only during arithmetic retains 4 of the 16 matrix entries.
  There are 25 complex products per `(I,K,J)` and four values of `K`, so
  the dense helper performs 1600 complex products and the diagonal helper
  400. This is an operation count, not a measured runtime estimate.
- `polynomial.f:127-130` passes `LOOP_WF` unchanged to `MERGE_WL`.
  `polynomial.f:346-350` consumes only `WL(I,J,I)` in increasing `I`, then
  increasing polynomial index `J`, and evaluates
  `OUT(J)=OUT(J)+WL(I,J,I)*CONST` separately for every `I`.
  Summing the trace first, moving `CONST`, fusing the updater with this merge,
  or reassociating the sums would change the numerical proof.

SHA-256 identities at inspection:

| Source | SHA-256 |
| --- | --- |
| generated `coef_construction_1.f` | `e2dbc5804c7f21139b46aa3cb6a49b97806f51e2d489e94e3e9f0f88760f3c1d` |
| generated `polynomial.f` | `0ebb4350ad0b7e58458405671e785b0476d9df8d8a94083ec7773247135a9275` |
| generated `mp_compute_loop_coefs.f` | `1d4347179afa33dbc29266fc17d66901c961a0385b12bae1c7fc447ca69b62b3` |
| `madgraph/iolibs/helas_call_writers.py` | `7a267fa78b75fa5b8efc4fc2f595769f938ee59919f55c6fffc3ab86ec5707ea` |
| `madgraph/loop/loop_exporters.py` | `0541caac8650b6ff0225f998bbbe10a3aecebf21c08150d174c441251364d426` |
| `madgraph/various/q_polynomial.py` | `607023aa6881f3e5b6b6baa06a29db9469885f3d539e8073f4f14eae67455b84` |

## Whole-graph requirements, including the DP alias

`helas_call_writers.py:1800-1852` proves terminal roots over the complete
optimized object graph: unique producers; consistent `number`/`me_id`;
known earlier loop mother or seed -1/0; no self-alias; known final roots;
no non-loop consumer of a loop wavefunction; and roots absent from the set
of all later update inputs. The one-loop-mother invariant is enforced by
`madgraph/core/helas_objects.py:2460-2471`. Multiple amplitudes may consume
one terminal root only when every consumer remains the diagonal merger.

The object graph is insufficient by itself. The rendered proof at
`loop_exporters.py:2773-2869` independently checks all generated update and
create calls, COMMON users, bare-array escapes, aliases, and the merger
implementation. Its current batch names, routine prefix and COMMON name
are **MP-specific** (`mp_coef_construction_*`, `MP_`, `<prefix>MP_WL`). A
successful QP proof does not certify the DP source. Current fail-closed
regeneration at `loop_exporters.py:2416-2424` also applies only to MP.

In particular, the generated `mp_compute_loop_coefs.f:128-131` contains
the legitimate DP alias:

```fortran
COMPLEX*16 DP_WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,-1:NLOOPWAVEFUNCS)
COMPLEX*16 DP_PL(0:3,-1:NLOOPWAVEFUNCS)
COMMON/ML5_0_WL/DP_WL,DP_PL
```

Every executable `DP_WL` use in that inspected file is a seed write at
fourth index -1 or 0 (`:210`, `:218-220`, `:234-237`). The corresponding
template is `madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc:139-141,242-268`.
The ordinary DP `loop_matrix.f` likewise touches only seed slots outside
coefficient-construction batches (`:688-709`).

A future DP rendered validator must either conservatively reject this
alias and retain dense updates, or recognize its exact type, dimensions,
COMMON member order and source identity, then validate **every** alias use
as a seed-only access. Do not merely allow any declaration named `DP_WL`:
reject terminal-slot reads, bare-array calls, EQUIVALENCE, alternate COMMON
layout/member names, additional alias chains, or unknown consumers. A
simple MP-to-DP prefix substitution in the current validator will reject
this legitimate alias, which is safe but disables the optimization.

For a complete production claim, the emitted-consumer inventory must include
all compiled source units and relevant INCLUDE contents. The present scanner
enumerates top-level `*.f`; arbitrary custom `.f90` files, included executable
code or out-of-directory COMMON consumers are not proved by that scan.
Unknown/custom export structures must retain the dense path.

Also retain dense updates for shared intermediate roots; duplicate or unknown
producers; unsupported shape/rank; argument/storage overlap; unexplained
COMMON users; changed merger semantics; and effective-theory compact kernels
with `left < right` when both ranks are nonzero. The latter are explicitly
outside the existing graph selection at `helas_call_writers.py:1841-1848`.

## IEEE guard proof for the fixed 4x4, rank-1x1 experiment

The existing QP guard and classifier are at `q_polynomial.py:520-589`.
Finite operands alone are insufficient: omitted operations can be the sole
source of INVALID, OVERFLOW, UNDERFLOW or INEXACT. QP's exponent range
[-1000,1000] must not be reused for binary64.

Admit the private DP helper only if all of these conditions hold:

1. The shape is exactly the supported 4x4 rank-1x1 case; arrays are distinct
   in the proved generated call; binary64 storage/representation is confirmed
   (`STORAGE_SIZE=64`, `DIGITS=53`, `RADIX=2`, `MAXEXPONENT=1024`, with the
   expected integer bit pattern for 1.0).
2. Rounding is IEEE nearest. The caller captures INEXACT immediately before
   the private helper, after the final vertex call, and that captured flag
   is already true. Do not manufacture sticky INEXACT by setting it.
   The existing MP boundary capture is `loop_exporters.py:2951-2957`.
3. INEXACT halting is disabled. Requiring **all** halting modes disabled is a
   conservative admissible experiment guard. If support or mode cannot be
   established, use the original dense helper. Guard queries and admission
   must not clear/set flags or perform new potentially signaling arithmetic.
4. Every real and imaginary component in active `A(1:4,0:4,1:4)` and
   `B(1:4,0:4,1:4)`, including components used only by omitted off-diagonal
   operations, is signed zero or a normal binary64 value with unbiased
   exponent in [-400,400]. Use integer TRANSFER/sign/exponent inspection;
   do not use floating ABS, arithmetic thresholds or NaN comparisons that
   could signal before the dense fallback. Nonzero exponent bits are
   623 through 1423 inclusive. Reject NaN, infinity and subnormal operands.

For these operands each component is a multiple of `2^-452` and has
magnitude less than `2^401`. Each exact real product is a multiple of
`2^-904`. Rounded products and subsequent sums retain that lattice, so
any nonzero cancellation result remains at least `2^-904`, well above
the minimum normal `2^-1022`. A rank-1x1 coefficient receives at most
eight complex products over four K values; a deliberately loose bound of
`2^810` covers every rounded partial sum, far below overflow. Thus omitted
arithmetic cannot raise invalid, overflow, underflow or divide-by-zero;
only inexact remains possible. Already-sticky, nonhalting INEXACT makes
those omissions unobservable through IEEE flags. Pre-existing flags remain
untouched.

Keep the original diagonal complex multiplication expressions, parenthesis
structure, ascending K accumulation, signed-zero initialization and merge
order. Do not introduce extra zero-product pruning or arithmetic
factorization. These restrictions establish the source-level argument;
bitwise executable equivalence still depends on compiler contraction,
vectorization and reassociation semantics. Qualification must use the actual
build flags and compiler, and may not rely on fast-math assumptions that
discard NaN, signed-zero or flag behavior. Preserving the source expression
alone is not evidence that different generated kernels lower identically.

## Existing tests and required experiment observations

`tests/unit_tests/loop/test_terminal_wavefunctions.py:43-105,170-236`
tests private graph selection, unknown/aliased/shared producers, hidden
COMMON/bare-array consumers, and modified merger rejection.
`tests/unit_tests/various/test_q_polynomial_compiled.py:513-601` exercises
QP terminal ranks (1,1)/(2,1), sizes 0..4, four rounding modes, clear/set
entry INEXACT, exceptional operands, exponent boundaries, consumed bits,
unchanged inputs, all IEEE flags and dense fallback. It does not qualify
a DP terminal implementation.

The isolated DP probe should demonstrate actual fast/fallback execution;
bitwise consumed output and unchanged A/B; exact dense output on fallback;
all IEEE flag states; signed zeros; cancellation at exponent boundaries;
off-diagonal-only NaN/Inf/overflow/underflow/inexact triggers; non-nearest
rounding; clear INEXACT; and trap-enabled fallback in a separate process.
It should also compare the updater-through-CREATE/MERGE output, and measure
guard-inclusive runtime on the actual coefficient stream. No speed claim
follows from the 75% product reduction alone: the guard scans 160 complex
operands and makes IEEE-mode queries on this small kernel.
