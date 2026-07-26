# Recurrence And AmpliCol Architecture Audit Follow-Up

This read-only follow-up reviews the first executable topology-replay recurrence
against `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` and original AmpliCol's
`--library=create` implementation. It was performed after the dynamic LC color
state, layout-specific helicity identity, and exact transition-factor alias
changes.

## Conclusion

The two largest design gaps from the original audit are closed for topology
replay:

- recurrence construction uses executable dynamic LC color-state transitions;
- topology-replay current identity uses local source-state ancestry instead of
  materializing one complete graph for every global helicity;
- transition, color, and witness coefficients are combined as one exact
  contribution factor;
- production recurrence construction contains no process-name or model-name
  optimization branches.

This establishes the intended AmpliCol-like compact architecture, but does not
complete the feature. Full component replay, closure semantics, all-flow-union
helicity placeholders, exact execution, process-set sharing, and real-process
allocation/performance gates remain open.

## Findings

### Canonical `concatenate-keep` Parent Order

The audit found that transition alias canonicalization discarded parent order
for `concatenate-keep`. Unlike an empty color result, concatenating component
forests is order-sensitive. This could incorrectly alias transitions in an
external model even though current built-in and UFO-SM catalogs do not emit
that operation.

The catalog now retains the canonical witness input permutation for
`concatenate-keep`, with an adversarial mirrored-parent regression test.

### Topology-Replay Coverage

The successful `g g > g g` and `u u~ > Z g g g` diagnostics establish exact
aggregate agreement for one selected flow. They do not yet certify every
physical flow, non-representative replay mapping, reflection phase, helicity
component, or UFO-SM execution.

Required numerical coverage:

- built-in and UFO-SM `g g > g g`, every live flow/helicity component;
- built-in and UFO-SM `u u~ > Z g g g`, all six flows and all helicities;
- reflected pure-gluon closures;
- same-flavour reconstruction;
- three-open-quark-line exchange signs.

### All-Flow-Union Identity

The current all-flow-union constructor still instantiates retained source spin
states and includes numerical spin state in source-current identity. It has not
yet demonstrated graph-size independence from retained helicity count. The
union builder must use runtime-helicity source placeholders before that layout
can be considered AmpliCol mode-2-like.

### Runtime Packetization

The runtime groups calls by prepared kernel and executes one backend packet per
homogeneous range. It does not invoke one backend call per recurrence edge.
However, real-process rows per call, SIMD occupancy, gather/scatter cost, and
warmed allocation counts remain unmeasured. The existing zero-allocation test
uses a synthetic plan only.

### Process-Set Sharing

Processes are still constructed independently. Semantic schedule interning
across process bindings is required before claiming the planned
`p p > j j j j` scaling advantage.

## Exact Alias Assessment

Combining transition, color-coefficient, and witness factors is mathematically
consistent with both the current built-in and UFO catalogs and with Rust
runtime multiplication. Removing raw prepared-kernel ID from the alias key is
also valid because the callable signature contains the ABI, contract kind,
inputs, exact expressions, and output layout. The input-exchange factor must
remain separate because it is applied conditionally from runtime parent order.

## Remaining Acceptance Evidence

Before feature acceptance, record:

- complete built-in/UFO component comparisons for representative and replayed
  flows;
- numerical closure canaries for pure-gluon, same-flavour, and three-line
  processes;
- all-flow-union state-count invariance under retained-helicity changes;
- real-process packet sizes, lane occupancy, gather/scatter timings, and warmed
  allocations at batches 1, 128, and 1024;
- fewer semantic schedules than concrete subprocesses for
  `p p > j j j j`.
