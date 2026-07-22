# Recurrence Post-Checkpoint Architecture Audit

This source-only audit was performed independently at commit `c54ec95` after
the first topology-replay helicity-sum execution checkpoint. It rechecks the
implementation against
[`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`](RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md)
and the original AmpliCol recurrence implementation. The conclusion is a
**no-go for a general public recurrence artifact/runtime lane** until the
findings below are closed.

## P1 Findings

### Ordered multi-line leading-colour state is incomplete

The recurrence ABI requires ordered forest components unless a permutation
equivalence is proven. The current color-state implementation sorts passive
components during joins, inheritance, and closure construction without
retaining a permutation phase or partner multiplicity. Original AmpliCol keeps
distinct three-line partners even when they map to the same public physical
flow.

Evidence:

- `rust/crates/rusticol-core/src/recurrence/color.rs` around the passive
  component join and inheritance code.
- `rust/crates/rusticol-core/src/recurrence/construct.rs` around closure color
  construction.
- `docs/development/RECURRENCE_EXECUTION_ABI.md`, ordered-forest contract.
- `dependencies/checkouts/legacy-amplicol/amplitude_QCD.f03`, three-line
  current and closure partner handling.

Required disposition: preserve ordered open-line forest identity, or apply an
explicit exact permutation/reconstruction certificate carrying phase and
multiplicity. Unconditional canonical sorting is not acceptable.

### Closure parity is incomplete outside one-quark-line canaries

The builder currently forms generic anchor/complement closures and aggregates
compatible closure templates. It does not yet model all original-AmpliCol
closure semantics:

- reflected pure-gluon closure terms and phases;
- three-open-line partner additions;
- same-flavour reconstruction and exchange terms.

Required disposition: encode these as model-generic exact closure term lists
derived from canonical process/color contracts. Until then, unsupported
topologies must fail before schedule construction. A broad public capability
must not be advertised by one-quark-line canaries.

### Current private helper is not a persistent runtime benchmark

Each private Python entry point validates and decodes the inputs, constructs
the program, loads prepared evaluators, builds the execution plan, and creates
runtime scratch. The all-target helper then executes the complete representative
recurrence once per replay target. It is therefore a correctness convenience,
not evidence for amortized public runtime performance.

Required disposition:

- separate process generation, artifact loading, selector planning, and hot
  execution;
- execute only the requested target in single-flow mode;
- report backend calls and rows-per-call;
- benchmark a persistent loaded runtime through the public API.

## P2 Findings

### Physical-flow replay proof remains partial

The current proof certifies source contracts, sector remapping, model digest,
fermion sign, and exact amplitude factor. Rust does not yet derive or validate
the inductive current, contribution, and closure bijections required by the
execution ABI. The runtime keeps only the squared replay factor for the
helicity-sum path, so it does not establish a general complex-amplitude replay
contract.

Required disposition: construct a replay certificate over every live current,
contribution, finalization, and closure term, and retain the exact complex
factor through resolved execution.

### Stored compactness does not prove builder scaling

The final schedule correctly stores local source ancestry, sparse live
destinations, and one finalization per propagated current. Construction still
enumerates parent pairs and transitions before backward liveness, and lacks the
required candidate/rejection/peak-live-state counters.

Required disposition: add construction counters and peak-RSS/state telemetry,
then demonstrate that peak work does not acquire a flow-by-helicity-by-edge
factor. If candidate enumeration is a measured scaling owner, move reachability
earlier rather than relying only on final pruning.

### Runtime-helicity contracts are not yet all-flow execution proofs

Full-state embeddings and contract digests validate schema inventory, but do
not yet prove full-state kernel-expression equivalence or absence of spurious
transitions. The Rust runtime correctly rejects `all-flow-union` today.

Required disposition: complete exact full-state source, vertex, propagator,
and closure contracts before enabling all-flow union. Do not helicity-expand as
a fallback.

## Confirmed Foundations

- The recurrence fork is before `GenericDAG` construction.
- Local source ancestry and sparse live amplitude destinations are represented.
- Each live propagated current is finalized once.
- Backend operations are grouped into homogeneous kernel packets and submitted
  as batched lanes.
- Prepared recurrence templates are model-generic and fail closed when a
  contract is absent.

## Public-Lane Gate

Do not wire a generally available recurrence artifact/runtime until all of the
following are true:

1. Ordered multi-open-line LC state is exact.
2. Reflected gluon, multi-line partner, and same-flavour closure reconstruction
   are implemented or the unsupported topology is rejected before building.
3. Replay is certified inductively over the live recurrence and exact complex
   factors are retained.
4. A persistent loaded runtime executes only requested selector targets and
   exposes packet/call counters.
5. Built-in and UFO-SM differential tests cover pure gluon, one-, two-, and
   three-open-quark-line processes.
6. Builder peak-work counters demonstrate compact scaling, not merely compact
   final storage.

After these corrections, run a second independent audit against both this file
and `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` before enabling public dispatch.
