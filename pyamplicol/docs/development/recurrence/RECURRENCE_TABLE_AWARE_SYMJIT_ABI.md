# Recurrence Table-Aware SymJIT Direct-Arena ABI

> **Superseded historical proposal.** This document records the investigation
> which motivated a generated row-table extension for the former SymJIT
> Direct-Arena interface. It is not an active ABI or implementation plan.
> The SymJIT 2.22.0 migration instead uses standard P-kernels and keeps row
> scheduling, factors, ordered overwrite/add operations, snapshots, and
> fanout in allocation-free Rusticol orchestration. See
> [`SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md`](../arena/SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md)
> and
> [`SYMJIT_UPSTREAM_SEPARATION.md`](../arena/SYMJIT_UPSTREAM_SEPARATION.md).
> DirectApplication, DirectTable, and proposed ABI names below are retained
> only to preserve the original measurements and design reasoning.

## Decision

The current SymJIT Direct-Arena ABI cannot execute a homogeneous recurrence
row table inside generated machine code. It executes one prepared kernel for
one resolved row and one scalar or SIMD point block. Rusticol resolves row
offsets into pointer descriptors, loops over rows, and SymJIT loops over point
blocks in Rust.

The narrowest production extension is a table-aware Direct-Arena callable
which:

1. is compiled from the existing portable complex O2 MIR;
2. receives one generic fixed-width row table plus arena views;
3. interprets model-fixed row-field bindings generated with the prepared
   template;
4. loops rows and point blocks inside generated machine code; and
5. preserves the exact row order and direct destination operations.

This requires a small but real SymJIT compiler and callable-ABI extension.
Rusticol alone cannot provide the same result by moving its current loop into
another Rust function: that would retain the millions of calls into the
generated one-block kernel.

This extension is worthwhile, but it cannot meet the AmpliCol runtime gate by
itself. It removes dispatch and row-projection overhead; it does not simplify
the prepared kernel algebra. The subsequent performance phase must introduce
recurrence-specific algebra reduction or exactly certified canonical
contract-class intrinsics.

## Evidence

This assessment uses the recurrence branch at `563a652` and the current
macOS arm64 `u u~ > Z+6g` topology-replay profile at batch 1024.

| Quantity | Value |
|---|---:|
| Wall time | 75.26 us/point |
| Profiled recurrence schedule | 74.63 us/point |
| Contribution kernels | 67.56 us/point |
| Finalization kernels | 5.42 us/point |
| All other recurrence work | 1.65 us/point |
| Contributions | 8,338 |
| Finalizations | 858 |
| Closures | 384 |
| Homogeneous row groups | 69 |
| AmpliCol wall time | 37.85 us/point |

The recurrence schedule has exact AmpliCol after-filter structural parity.
Missing current reuse, repeated finalization, selector planning, arena
clearing, closure reduction, and Rust orchestration are therefore not the
principal remaining problem.

At two f64 lanes on arm64, contribution evaluation alone performs at least:

```text
8,338 rows * (1,024 points / 2 lanes) = 4,269,056
```

generated machine-code calls per tile. Each contribution costs about
8.10 ns/point, while AmpliCol spends only 4.54 ns per contribution if its
entire runtime is divided by the same contribution count. The true AmpliCol
primitive cost is lower because its total also includes finalization, closures,
and reduction.

The hard report gate is `1.20 * 37.85 = 45.42 us/point`. Prior sampling and
phase accounting place the optimistic result of eliminating all
row-projection, wrapper, and call-boundary overhead around 63--67 us/point.
That remains 1.66--1.77 times AmpliCol and misses the hard gate by at least
39%.

## Current Call Path

The current hot path is:

```text
Rusticol row-group scheduler
  -> one typed executor call for the row group
    -> resolve/cache DirectPlane and DirectScalar descriptors for every row
      -> for each row
        -> DirectCallable::invoke_unchecked
          -> DirectApplet::evaluate_planes_unchecked
            -> for each scalar/SIMD point block
              -> CompiledFunc
```

The descriptor cache removes repeated row projection after warm-up, but the
two nested call loops remain.

The generated function type is currently:

```rust
fn(
    mem: *const f64,
    states: *const &mut [f64],
    index: usize,
    params: *const f64,
) -> i32
```

For Direct-Arena execution:

- `states` is a fixed array of plane descriptors for one recurrence row;
- `params` is a fixed array of scalar pointer descriptors for that row;
- `index` selects exactly one scalar point or SIMD block;
- every `Loc::Mem(i)` and `Loc::Param(i)` is a compile-time descriptor index;
- the generated body contains no row pointer, row count, row stride, point
  count, or outer loop.

`DirectApplet::evaluate_planes_unchecked` implements point alignment, SIMD
blocks, scalar tails, and fallback in Rust. The recurrence adapter implements
the row loop in Rust.

### Current feasibility

| Operation | Current SymJIT |
|---|---|
| Read/write persistent point-contiguous planes | Yes |
| Alias outputs to destination arena planes | Yes |
| Apply exact complex factors in generated code | Yes |
| Recompile portable O2 MIR for host SIMD | Yes |
| Loop points inside generated machine code | No |
| Loop homogeneous rows inside generated machine code | No |
| Accept a raw row table and row stride | No |
| Load dynamic arena offsets from row fields | No |
| Invoke one generated function per row group | No |

Although SymJIT MIR supports branches, adding a MIR loop alone is insufficient.
The callable entry point and code generators have no row-table arguments, and
Direct-Arena memory instructions currently dereference statically indexed
plane descriptors. The required change belongs primarily in the generated
function wrapper and Direct-Arena address binding, not in Symbolica expression
construction.

## Narrow Production ABI

The ABI should be generic to SymJIT and must not contain Rusticol row types,
particle identities, process names, or model-specific logic.

Suggested public identities are:

```text
pyamplicol-eager-plane-table-descriptor-v1
pyamplicol-eager-plane-table-binding-v1
```

The existing Direct-Arena API remains available to accepted recurrence
artifacts. New eager table descriptors are a separate lane-specific contract;
they do not reinterpret recurrence rows or callable roles.

### Portable binding metadata

Each canonical prepared template supplies a fixed binding for its logical
planes and scalars:

```rust
enum DirectTablePlaneBase {
    CurrentReal,
    CurrentImag,
    AmplitudeReal,
    AmplitudeImag,
    Momentum,
}

struct DirectTablePlaneBinding {
    base: DirectTablePlaneBase,
    row_u32_offset: u16,
    constant_index_or_sentinel: u32,
    component_delta: u16,
    point_stride: DirectTablePointStride,
}

enum DirectTableScalarBase {
    ParameterReal,
    ParameterImag,
    ExactFactorReal,
    ExactFactorImag,
    Literal,
}

struct DirectTableScalarBinding {
    base: DirectTableScalarBase,
    row_u32_offset_or_sentinel: u16,
    constant_index_or_sentinel: u32,
    literal_bits: u64,
}

struct DirectTableApplicationMetadata {
    row_stride: u32,
    destination_operation: DirectDestinationOperation,
    plane_bindings: Vec<DirectTablePlaneBinding>,
    scalar_bindings: Vec<DirectTableScalarBinding>,
    output_alias_inputs: Vec<u32>,
}
```

The names above are illustrative; the important contract is:

- row fields are addressed by byte offset and fixed width;
- each logical MIR input has one authenticated base and index source;
- the row table remains opaque bytes to SymJIT;
- constants and fixed parameter indices do not consume row fields;
- destination aliases retain the existing initialize, add,
  finalize-in-place, and closure-add semantics.

For example, a contribution template can map:

```text
logical parent-0 component
  = current_real
    + row.parent0_component_base
    + fixed_component_delta

logical momentum component
  = momentum
    + row.parent0_momentum_form_id
    + fixed_lorentz_component

logical exact factor
  = factor_real/factor_imag[row.exact_factor_id]

logical destination component
  = current_real/current_imag
    + row.destination_component_base
    + fixed_component_delta
```

The same mechanism covers finalization and closure rows without teaching
SymJIT their concrete layouts.

### Host call view

Use one host-only `repr(C)` call structure to avoid register-pressure and
calling-convention differences:

```rust
struct DirectTableCallViewV1 {
    rows: *const u8,
    row_count: u32,
    row_stride: u32,

    current_re: *mut f64,
    current_im: *mut f64,
    amplitude_re: *mut f64,
    amplitude_im: *mut f64,
    current_point_stride: u32,

    momenta: *const f64,
    momentum_form_count: u32,
    momentum_component_count: u16,
    momentum_point_stride: u32,

    parameter_re: *const f64,
    parameter_im: *const f64,
    factor_re: *const f64,
    factor_im: *const f64,

    point_start: u32,
    point_count: u32,
}

type DirectTableCompiledFunction =
    unsafe extern "C" fn(*const DirectTableCallViewV1) -> i32;
```

Lengths needed for safe validation may be present in the checked public view or
owned by the loaded callable context. They should not cause per-row checks in
the hot function.

The portable payload stores no native pointers and no machine code. The call
view exists only after loading on the receiving host.

### Generated loop

For the SIMD middle range, generated code should execute:

```text
for row in rows, preserving stored order:
    load row-dependent offsets and exact-factor pointers once
    derive logical plane/scalar pointers for this canonical template
    for SIMD block in aligned point range:
        execute the existing remapped MIR body inline
```

A scalar table function handles an unaligned head and tail. The owning
`DirectTableCallable` may dispatch at most three generated calls per row group:
scalar head, SIMD middle, and scalar tail. Normal aligned tiles need one call.

The row loop must be outside the point loop so row-derived pointers are
calculated once per row. Each point still observes contributions in the
existing deterministic row order.

The least invasive code-generation implementation can build a small
stack-resident plane/scalar descriptor array once per row, then execute the
current Direct-Arena MIR body against it. This changes no numerical storage and
does not pack kernel inputs. A later code generator may fold the descriptor
indirection into direct base-plus-offset addressing if profiling justifies it.

### Why a Rust wrapper is not enough

An additive Rust API that accepts a descriptor slab and loops rows would remove
one Rusticol-to-SymJIT method boundary, but it would still call the generated
`CompiledFunc` once for every row and SIMD block. It is useful only as a
correctness prototype.

Likewise, a generated wrapper that calls the existing one-block function
pointer from inside a loop removes the external FFI boundary but retains
millions of machine-function calls. The MIR body must be emitted inline inside
the generated row/point loops to obtain the intended benefit.

## Minimal SymJIT Changes

### Public Direct-Arena layer

Add alongside the existing API:

- `DirectTableApplicationMetadata`;
- generic plane/scalar row bindings;
- `DirectTableApplication::from_source_storage`;
- `DirectTableApplet`;
- `DirectTableCallable` and its unchecked table entry point;
- a separate portable storage magic/version.

The source application requirements remain:

- native Symbolica complex application;
- optimization level 2;
- no external functions;
- exact output-alias contract.

### MIR lowering

Reuse the current complex-to-split-plane remapping and destination-factor
transformation. No general-purpose dynamic-load MIR instruction is required if
table bindings are consumed by the Direct-Arena code generator.

Finalization must retain the current snapshot-before-alias behavior. The
snapshot lives inside one row iteration and is reused for each point block.

### Machine-code generators

Add table-call prologue, row loop, point loop, and epilogue support to:

- arm64 scalar;
- arm64 SIMD;
- x86-64 scalar;
- each enabled x86-64 SIMD width.

The compiler must expose a distinct function type rather than coercing the
current four-argument `CompiledFunc`.

The generated table wrapper should reuse the existing register allocation and
MIR body emitter. Loop registers, call-view bases, the current row pointer, and
point index must be reserved explicitly. Spill-stack allocation happens once
per table call, not once per row or point block.

### Loading and sealing

Loading should:

1. decode portable O2 MIR and table-binding metadata;
2. validate row offsets, widths, aliases, and scalar slots once;
3. compile scalar and host-SIMD table callables;
4. seal immutable contexts; and
5. expose checked and unchecked table calls.

No digest or bounds checks belong in the warmed row/point loops. Existing
artifact and prepared-model authentication remains sufficient at load.

## Rusticol Integration

Rusticol already provides the necessary ingredients:

- fixed-width `repr(C)` contribution, finalization, and closure rows;
- contiguous homogeneous row groups;
- aligned split-complex current and amplitude arenas;
- point-contiguous momentum planes;
- immutable parameter and exact-factor arrays.

Integration should:

1. derive generic table-binding metadata from the existing model-fixed
   `SymjitDirectPlaneProjection` and `SymjitDirectScalarProjection`;
2. record and validate the concrete row-layout digest at artifact load;
3. replace the per-row `DirectCallable::invoke_unchecked` loop with one
   `DirectTableCallable::invoke_table_unchecked` per homogeneous group;
4. preserve stage clearing and row-group order in Rusticol;
5. remove the warmed descriptor cache once raw-row table execution is
   validated; and
6. keep source filling and lightweight closure implementations unchanged
   unless profiles show a material benefit.

Contribution and finalization are the first implementation scope because they
account for about 97% of qq_Z6g wall time. The ABI can represent closure-add,
but closure migration is not a prerequisite for measuring the main gain.

The warmed `evaluate_f64_into` path must remain allocation-free. Row binding,
layout checks, and any fallback descriptor construction occur at load or
warm-up only.

## Portability

The table-aware payload remains portable under the same policy as current
Direct-Arena applications:

- only O2 MIR and fixed-width binding metadata are stored;
- native code is generated for the receiving CPU;
- no host pointer, `usize`, or machine instruction is serialized;
- little-endian integer fields use explicit widths;
- host-only call structures use `repr(C)` plus compile-time layout assertions;
- arm64 and x86-64 select their own scalar/SIMD implementations at load.

Optimization level 2 is mandatory. O3 applications must fail recurrence
prepared-pack validation rather than being accepted as portable input.

The table wrapper changes control flow and addressing, not the stored algebra.
Its scalar and SIMD results must therefore follow the same portability and
numerical policy as the current O2 Direct-Arena callable.

C++ and ASM prepared backends do not consume this SymJIT ABI. Their recurrence
implementations should expose the same Rusticol row-group contract through
their own target-native callables.

## Expected Gain And Limitation

### What table-aware execution removes

- one Rust row-loop iteration per recurrence row;
- one DirectCallable/Applet entry per row;
- one generated-function entry, prologue, epilogue, and status return per
  scalar/SIMD point block;
- repeated descriptor-array indexing at the Rust/SymJIT boundary;
- most row-projection work if raw table bindings replace the descriptor cache.

For qq_Z6g this changes approximately 4.27 million contribution machine calls
per 1,024-point tile into one to three calls per homogeneous row group.

### What it does not remove

- any arithmetic operation in the prepared MIR;
- unused or static-zero current components retained by that MIR;
- complex temporaries and register pressure;
- broad 20--40-plane generic template contracts;
- propagator algebra;
- the difference between generic oriented-kernel expressions and AmpliCol's
  compact QCD primitives.

The expected table-only range is approximately 63--67 us/point. Even the low
end is 1.66 times AmpliCol and above the 45.42 us/point hard gate.

Therefore:

> A table-aware SymJIT callable is necessary to remove pathological call
> granularity, but it is not sufficient for AmpliCol parity.

The follow-up must reduce primitive work through model-generic,
exactly-certified mechanisms:

- recurrence-specific live-component MIR;
- folding of fixed permutations and factors;
- static-zero elimination;
- fewer complex temporaries; or
- canonical contract-class Direct-Arena intrinsics shared by equivalent
  built-in and UFO-SM kernels.

## Staged Implementation And Test Plan

### Stage 1: SymJIT table-loop MRE

Implement only in SymJIT:

- one synthetic complex O2 Direct-Arena kernel;
- a fixed-width generic row table;
- generated scalar and SIMD row/point loops;
- initialize, add, and finalize-in-place destinations.

Compare against the existing per-row callable for row counts 1, 2, 64, and
1,024 and point counts:

```text
1, lane_width - 1, lane_width, lane_width + 1, 128, 1,024
```

Require:

- identical row order;
- numerical parity;
- valid scalar heads/tails;
- correct status returns;
- no machine call inside the generated row/point loop;
- portable save/load on x86-64 and arm64.

### Stage 2: Generic row bindings

Add plane/scalar base kinds and row-field offsets. Test:

- current, amplitude, and momentum bases;
- fixed and row-selected parameters/factors;
- output aliases;
- null pointers, overflow, bad row stride, and out-of-range offsets;
- multi-output in-place finalization snapshots;
- malformed portable metadata rejection.

Use checked calls for tests and loading. Benchmark only the unchecked sealed
call.

### Stage 3: Rusticol contribution integration

Integrate contribution row groups first:

- built-in and UFO-SM `d d~ > Zg` and `d d~ > Zgg`;
- both physical runtime flow selectors;
- exact structural-count parity;
- component-wise parity with the existing direct callable;
- zero warmed allocations;
- table-call counter bounded by row groups rather than rows or point blocks.

Then validate `u u~ > Z+6g` topology replay at batches 1, 128, and 1,024.

Go/no-go criterion:

- at least 8% wall reduction on qq_Z6g; and
- contribution time no worse than 62 us/point.

If this is not achieved, inspect generated instruction profiles before
extending the ABI further.

### Stage 4: Finalization and all-flow union

Move finalization groups to the table callable and validate:

- in-place alias snapshots;
- massive and massless propagators;
- topology replay and all-flow union;
- homogeneous, alternating, random, and pre-grouped selectors;
- built-in/UFO-SM parity.

Closure migration remains profile-gated.

### Stage 5: Cross-platform and release tests

Run:

- macOS arm64;
- Linux x86-64;
- macOS x86-64 where available;
- save on one supported architecture and load/recompile on another;
- clean wheel/sdist install;
- malformed storage and panic-containment tests.

Record loaded SymJIT version, O2 source digest, table-binding digest, target
architecture, SIMD lane width, row groups, rows, generated table calls, and
rows per call.

### Stage 6: Kernel-algebra phase

Treat the table-call result as the new dispatch baseline. Profile canonical
contract classes by total wall ownership and implement recurrence-specific
algebra or exact contract-class intrinsics in descending order.

The production recurrence merge remains blocked until:

- qq_Z6g reaches at most 45.42 us/point for the AmpliCol gate;
- built-in and UFO-SM use identical canonical eligibility rules;
- unsupported contracts use the generic table callable without incorrect
  merging; and
- compiled and eager lanes remain unchanged.

## Recommended Upstream Request

Request an additive SymJIT table-call facility with this concise scope:

> Given portable complex O2 MIR, fixed logical plane/scalar bindings, an opaque
> fixed-width row table, split arena bases, and a point tile, generate a sealed
> callable that preserves row order and evaluates the MIR inline for every
> row and scalar/SIMD point block. Row fields select dynamic arena offsets and
> exact factors. Store MIR plus fixed-width binding metadata only; recompile
> native code on load.

Do not request a general recurrence engine, Rusticol row definitions, particle
semantics, process-specific compilation, or a new general-purpose loop
language in MIR. Those would make the upstream change broader than necessary.
