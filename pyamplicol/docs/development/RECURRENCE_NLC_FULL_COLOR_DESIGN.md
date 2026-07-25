# Recurrence NLC And Full-Color Design

## Status

This document defines the post-LC extension of the accepted Direct-Arena
recurrence implementation. The accepted LC source remains frozen on
`443f354a467cdda187996bef1a41fbd5a00ae28d`; development proceeds on
`codex/recurrence-nlc-full-color`.

## Workload

NLC and full color require coherent interference between physical color-basis
amplitudes at equal helicity. Neither accepted LC schedule is sufficient:

- topology replay shares local helicity ancestry, but evaluates one physical
  flow at a time;
- all-flow union shares dynamic color states, but fixes one runtime helicity
  per execution.

Running either accepted schedule over the missing axis would introduce a
flow-by-helicity replay factor. The contracted-color implementation therefore
uses a third internal strategy:

`contracted-color-union`

It is selected automatically for recurrence generation with
`color.accuracy = "nlc"` or `"full"`. It is not an LC flow layout and exposes
no color-flow selector.

## Builder Semantics

The contracted-color strategy combines:

- topology replay's fixed source templates and local source-state ancestry;
- all-flow union's dynamic ordered color forests and complete physical-sector
  materialization;
- no topology replay targets;
- one resolved amplitude destination for every retained
  `(helicity, physical color sector)` pair.

Current identity remains model-generic and includes the existing state,
momentum, coupling-order, quantum-flow, flavor-flow, dynamic-color, and local
source-ancestry contracts. No model, process, or particle-name special cases
are permitted.

Rust constructs and backward-prunes the schedule before serialization. It
must not construct `GenericDAG`, per-flow evaluator graphs, or a
flow-by-helicity edge table.

## Color Contraction

Python's existing generic color planner remains authoritative for the
requested NLC/full color matrix. Color payload v3 emits one sector-local
upper-triangular template:

- physical left and right sector IDs;
- one exact complex-rational factor-catalog reference and its independently
  checked binary64 value;
- optional proven elementary-Abelian Walsh factorization;
- color accuracy and sector-order digest.

It also emits one authenticated owner entry for every physical sector. Each
sector is classified as its own materialized owner, an exactly proved alias,
or an exact structural zero. The active-sector set must equal the fixed-owner
set. This prevents deterministic coherent-owner reduction from silently
dropping a physical sector and makes every alias/zero decision independently
checkable by both native and exact loaders.

The process columnar ABI carries this template once. Rust authenticates it
against the physical-sector catalog and lowers it into the recurrence PACBIN.
The runtime maps each sector-local entry across the contiguous helicity
component dimension.

Storage order is local-color-major and helicity-component-minor, matching the
validated compiled-DAG repeated contraction. Expanded sparse entries remain a
fail-closed fallback. Proven C2^k/Walsh transforms use the existing
unnormalized amplitude transform and exactly one inverse subgroup-order factor.

## Direct-Arena Runtime

One tile execution:

1. fills fixed source rows for all retained local source states;
2. executes each contribution and finalization row once;
3. accumulates every closure directly into the amplitude arena;
4. contracts equal-helicity color amplitudes into one real total;
5. applies helicity weights and process normalization.

The warmed native totals path allocates no heap memory. Persistent aligned
amplitude, contraction, selector, and output scratch is sized at load time.
Resolved output has the public shape `(point, helicity, color:contracted)`.
Helicity subsets remain supported; color-flow selectors are rejected.
The Rust API can resolve a batch-global helicity subset once into a
`NativeRecurrenceSelectorPlan`; repeated evaluation with that handle borrows
the retained selector set and caller-owned input/output buffers without heap
allocation. String-based Python convenience calls may allocate their returned
arrays.

The first runtime implementation may use the authenticated expanded sparse
template. The repeated compact reducer and Walsh path must land before broad
performance acceptance.

The runtime now decodes repeated storage directly. When the authenticated
payload carries a K4 or C2^k proof, it derives the transformed Hermitian
matrix once at load time and preallocates one local-color-by-point transform
workspace. The hot loop transforms one helicity component at a time, so its
scratch does not acquire an additional helicity factor. It falls back to the
canonical repeated or expanded rows when no factorization is proved.

## Exact Execution

The exact Python executor consumes the same sector-pair entries, owner map,
and signed-rational exact-factor catalog. It never reconstructs factors from
binary64 values, decimal strings, or an expanded JSON runtime schema. Exact
totals, resolved sums, parameter updates, and helicity subsets must agree with
native f64 semantics.

During this milestone, recurrence exact and recurrence native f64 agree for
contracted color. A separate pre-existing discrepancy was found in compiled
exact NLC/full evaluation: its helicity-quotient expansion does not reproduce
the compiled native total at a generic point, while compiled native,
recurrence native, and recurrence exact agree. Recurrence must not copy that
behavior; the compiled exact issue is tracked separately from this lane.

## First Vertical Slice

The first end-to-end milestone covers built-in SM and UFO-SM:

- `d d~ > z g g` in NLC and full color;
- the three-open-quark-line process `d d~ > u u~ s s~` in NLC and full color;
- the heavier factorization canary `g g > t t~ g g` in NLC and full color.

For each case:

- recurrence generation completes without `GenericDAG` or evaluator creation;
- every resolved helicity and contracted total agrees with compiled JIT O2 at
  `rtol=1e-12`, `atol=1e-15`;
- built-in and UFO-SM schedules agree after explicit model-state mapping;
- normal and exact evaluation agree;
- malformed contraction tables fail closed;
- repeated warmed `evaluate_f64_into` allocates zero heap bytes.

## Performance Sequence

1. Establish correctness with expanded sparse contraction.
2. Move the repeated matrix and group map into PACBIN.
3. Reuse the compiled reducer's four/eight independent Hermitian accumulators.
4. Enable proven K4/C2^k Walsh plans with independent Rust validation.
5. Profile direct arena, closure production, contraction, totals
   materialization, and selector handling separately.
6. Benchmark the mandatory mid-multiplicity matrix and `qq_Z6g` under the
   30 GiB watchdog.

Persistent whole-state AoSoA, random composed gathers, and fragmented
per-leaf slabs are explicitly rejected unless new native wall-time evidence
overturns the compiled-DAG measurements.

## Current Milestone Evidence

The built-in and UFO-SM `d d~ > z g g` and `d d~ > u u~ s s~` NLC/full
public generation, parameter-update, and exact tests pass. The three-open-line
case retains six physical endpoint pairings and removes the six permutations
of each disconnected open-string forest by assigning one deterministic
coherent-amplitude owner. Heavier built-in `g g > t t~ g g` artifacts validate
both optimized totals and every resolved-helicity sum against compiled
execution.

For `g g > t t~ g g`, the recurrence schedule has 1,708 currents, 6,400
compact direct contributions, 1,536 closures, and 130 dynamic color states.
Generation takes four to five seconds and peaks below 0.32 GiB on the current
macOS arm64 host. The compact v3 color payloads are:

- NLC: 132 repeated template entries, 8,448 logical rows, 29,960 bytes.
- Full: 300 repeated template entries, 19,200 logical rows, 36,264 bytes.
- Both payloads carry six independently validated K4 cosets and explicit
  sector/component coordinates for every active group, plus exact factors and
  an owner-or-zero record for all 24 physical sectors.

On the source-frozen LC base, five-second batch-128 profiles report
67.18 microseconds per point for recurrence NLC versus 61.42 for compiled,
and 68.87 for recurrence full color versus 82.08 for compiled. The recurrence
reducers take about 3.1--3.2 microseconds per point versus 3.9--4.7 for the
compiled artifacts. Thus the first v3 NLC canary is 1.094 times compiled and
the full-color canary is 0.839 times compiled. These are same-host development
comparisons and must be repeated after the feature branch incorporates the
newer compiled-runtime mainline.
The stored full-color validation point differs from compiled by
`2.2e-15` relatively, and its resolved sum reproduces the optimized total
within `2.4e-15`. A genuine factorized artifact passes the zero-allocation
warmed `evaluate_f64_into` test.

## Compatibility

Compiled and eager artifacts and runtimes are untouched. Accepted LC recurrence
semantics and performance remain regression-gated. Recurrence artifact ABI
backward compatibility is not required; any recurrence ABI bump fails old
recurrence artifacts with an exact regeneration message.
