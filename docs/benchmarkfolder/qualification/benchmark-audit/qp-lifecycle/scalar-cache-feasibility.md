# Full-QP scalar-cache feasibility

This is an unimplemented proposal. No hit rate or performance benefit has
been measured. The relevant boundary is TIRust's `native_masters`, after it
assembles a scalar's arguments and before it calls oneloopmaster.

## API boundary

The current API cannot inject a bridge-owned MPFR cache. `MasterProvider` at
`crates/iregi-rust/src/store.rs:158–160` accepts/returns `Complex<f64>` only.
`direct/exec.rs:1046–1048` drops shared stores for `Float`; the public
`DirectWorkspace::set_tensor_store` documentation explicitly says MPFR
workspaces share nothing (`direct.rs:314–320`). `DirectMasters` selects
built-in evaluators rather than an arbitrary callback. oneloopmaster's
`NativeEvaluator<Float>` has evaluation/configuration APIs, but no cache
callback.

A narrow hook in the pinned installation copy around
`direct/exec.rs:939–940` could accept a context-owned MPFR scalar store shared
by the retained `(N,rank)` workspaces. It does not require changing
oneloopmaster or calling a different scalar implementation. The current
binary64 provider must not be used as an approximation to this hook.

## Exact key and stored result

Use the actual `native_input[..arity+1]` after assembly, not reconstructed
four-vectors. Family fixes its ordered argument sequence:

| Family | Arguments before the final μ² |
| --- | --- |
| A0 | m0² |
| B0 | s01, m0², m1² |
| C0 | s01, s12, s02, m0², m1², m2² |
| D0 | s01, s12, s23, s03, s02, s13, m0², m1², m2², m3² |

Every complex component needs its exact value, actual precision, and sign,
including signed zero. The audited ordinary `Float` is a precision-bearing
MPFR wrapper, not the separate error-tracking type
(`float/multiprecision.rs:25–34`); if the numeric type changes, its extra
error/rounding metadata would also require identity. Include working
precision, full/partial mode, scalar
normalization/pole convention, and evaluator configuration or bind those to
an immutable cache namespace. Preserve ordering; do not canonicalize
propagators or round keys through binary64. Parent rank and Cartesian shifts
are useful attribution, but the scalar key does not need them once its actual
ordered invariant/mass/scale arguments and configuration are fixed.

Symbolica's `Float::serialize_into` is insufficient by itself: it omits
precision and merges signed zeros (`float/serialization.rs:199`, with the
encoding contract at the file start). Add explicit precision and sign.
NativeEvaluator settings that belong to the namespace include working bits,
pole selection, handwritten/generated-sector switches, Gram reduction,
negative-Källén C0 handling, and small-momentum handling/tolerance. The
audited QP route uses 256 working bits, BOTH poles, and its existing kernel
choices; none would change. Bind the pinned scalar source/kernel hashes
and the immutable rounding policy to this namespace as well.

Store the exact three complex outputs `[finite, total single pole, double
pole]`, preserving each component's returned precision and signed zero.
TIRust derives separate UV/IR residues afterward. Nominal 256-bit evaluation
does not imply every intermediate/output object reports exactly that
precision; Symbolica operations can change it. `NativeEvaluator::evaluate`
returning `Ok` alone does not guarantee finite output.

## Lifetime, failure, and budget

Use per-worker/context ownership and the existing event-clear boundary.
Keep primary/check state disjoint; initially bypass both lookup and insertion
on independent check passes, matching current bridge policy. A future
check-only store would need its own evaluation epoch and explicit
qualification. Do not use `TensorStore::check_view` as an isolation guarantee.

Only finite scalar outputs are candidates. Keep new entries pending until
the enclosing tensor reduction and bridge packing succeed; discard pending
entries on scalar errors, nonfinite values, tensor failure, or panic. The
Fortran transport and amplitude safeguards still run. Existing prior
successful entries must never be relabeled as evidence that a failed tensor
succeeded. Charge owned keys, MPFR payloads, and metadata to the same
`TIRustCacheMaxBytes` budget as the other numerical stores; cache-off,
topology limits, and budget refusal must leave computation unchanged.

`TIRUST_interface.inc:171–172` computes eligibility from the parent tensor's
propagator count (`-1` enables all, `0` disables, a positive value caps N)
and marks CT modes 2/5 as independent checks. Preserve that parent eligibility;
do not enable a cache for an ineligible N=6 parent merely because its scalar
subtopology has at most four propagators. Full and partial stores and their
failure state must remain separate. A budget change and event/context clear
must invalidate this added store along with the present stores; budget
exhaustion stops insertion while all scalar computations continue normally.

No audited production code consumes MPFR sticky flags, but removing kernel
executions can affect them. This remains a qualification question, not a
claim of flag equivalence. The shadow-only phase still executes every kernel.
A later reuse experiment must compare MPFR sticky flags and host IEEE flags
under identical initial states; if caller-observable flags differ, reuse
needs an explicitly reviewed preservation strategy or must be rejected. Do
not silently clear flags or assume replaying stored flags covers ambient
exponent/rounding changes.

## Bounded private profile before implementation

First add shadow accounting only, with no reuse, in an isolated copy.
Instrument after argument assembly at `exec.rs:903` and around the scalar
evaluation at lines 939–940. At the bridge, label calls by event epoch,
primary/check, tensor-call ID, and `(N,rank)`; a private Fortran trace is
needed to attach a real generated loop-group ID. The present ABI does not
supply one.

Count requests, exact unique keys, within-call repetitions, repetitions
across distinct parent tensor calls/inputs, scalar errors/nonfinite outputs,
locally finite successes, parent-success committed candidates, discarded
pending candidates, and potential hits against earlier committed candidates.
Record family/configuration, key/output bytes, scalar kernel time, key-build
time, and the observation cap. Separate theoretical repeats from eligible
repeats: retain the actual parent cache-enabled flag, full/partial mode,
primary/check mode, current budget, and existing store usage. Never count
cache-off, topology-ineligible, check-pass, failed-parent, or budget-refused
requests as potential usable hits. Shadow bookkeeping has its own small
explicit diagnostic memory cap and cannot consume or modify the production
cache; any simulated shared-budget occupancy must be labeled an estimate
until the exact owned allocation accounting is implemented. Reset shadow
eligibility epochs on the same clear/budget/context events while retaining
aggregate observation totals. Use full keys for equality rather than trusting
a hash alone. Once capped, continue totals but label uniqueness/hit counts
incomplete. Existing `last_master_rows` counts requests, not avoided kernels.

Also split Fortran pack/unpack, Rust key/parse/pack, and native reduction
times. Current cache-clear resets QP statistics at amplitude exit, so a query
after the public call can return zero even if reuse occurred. Snapshot
`tirust_qp_cache_statistics` before clearing or use a separate diagnostic
accumulator. A bounded replay of saved actual tensor calls can attribute
scalar opportunities; only a separately qualified whole-amplitude campaign
can establish a runtime gain.
