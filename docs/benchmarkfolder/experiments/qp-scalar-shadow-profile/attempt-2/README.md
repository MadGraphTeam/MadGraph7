# Private QP scalar shadow profile — corrected attempt 2

Attempt 1 remains immutable in the parent evidence directory. Its compiler
rejected unsafe MPFR FFI inside the core's `forbid(unsafe_code)` crate. This
revision preserves that lint: the existing unsafe-capable bridge performs
the FFI and installs a safe `fn() -> u32` callback in the private observer.
An absent callback is reported by `mpfr_reader_available=false` and a sentinel
outside the MPFR flag mask; both the launcher and analyzer refuse comparison.
Two matching unavailable observations can never establish flag equivalence.

This experiment observes scalar requests and runs every original kernel. It
does not implement a scalar cache, change arithmetic precision, relax a
stability check, or modify production sources/installed libraries. There is
no performance claim. Compiler and native launch remain deferred pending
root review of this source package.

## Frozen numerical scope

The source-only preparer copies the supplied patched TIRust tree into
`.tirust-build/qp-scalar-shadow-profile-attempt-2/tirust` and the current bridge into
the sibling `bridge` directory. The source hashes and complete private-tree
inventory are in `prepared.json`. The new Rust module and three small patches
are retained here for review.

- The unchanged upstream scalar evaluation is bracketed in
  `crates/iregi-rust/src/direct/exec.rs::native_masters`, separately for its
  full and partial branches. The same input and output slices reach the
  same evaluator exactly once; its original `?` error propagation remains.
- The private bridge brackets `ml_tirust_evaluate_mp` outside its existing
  panic guard. Its final return status determines whether scalar candidates
  may be considered committed by the offline analyzer. The original
  `mp_records` path, complete-tensor cache, decimal transport, output copy,
  and all failures still execute. A note identifies whole-tensor cache hits.
- Clear, changed-budget, and context-destruction boundaries are observed.
  A store snapshot captures the existing QP, DP, and partial store usage.
  No numerical entry is read from or written to the observation journal.
- The existing 113-bit parse then 256-bit evaluation stays unchanged. The
  private Cargo library name differs so it cannot replace the cached normal
  bridge artifact. All dependency revisions remain pinned.

## Dataset and transport

`selected-inputs.dat` contains exactly EW3 trace points 1 (ordinary) and 2
(threshold), groups 1–12 in their original per-point order: 24 physical QP
parents. The threshold groups 5 and 6 were refused and remain in the dataset.
The source trace is
`experiments/ew-threshold-adjudication/trace.json`, SHA256
`03fbfd3795a468661162c0e574a7de6878065efbab79ca9d0f80e743f29e787f`.
These records cover CTMODE4 only. They do not represent a complete event or
an independently captured reversed direction.

Extraction imports the previously reviewed `tensor-replay/prepare.py::extract`
and preserves its decimal strings. The trace prints 36 significant decimal
digits (`ES45.35E4`), sufficient to round-trip binary128. `replay.F90` reads
those strings into REAL16 and calls the frozen `mtirust.f90`, which packs
the reconstructed values as `ES64.39E4` records. No Python floating-point
conversion or direct padding of the shorter trace records is used. Every
input's binary128 memory bytes are emitted for all three replay variants;
the existing exact-rational `input-consistency/audit.py::binary128` helper
will independently verify those bits before any numerical conclusion.

A separately labeled 12-call **synthetic lifecycle suffix** uses captured
inputs but deliberately changes cache/mode controls. It is excluded from
physical repeat-rate accounting:

1. Two identical full primary calls (the second exercises the existing
   whole-tensor cache).
2. Two identical independent-check calls, then one cache-disabled primary.
3. Two identical partial primary calls and one partial independent check.
4. Set budget zero and repeat two full primary calls.
5. Restore the original budget and repeat the captured refused group twice,
   with the final N=4 request excluded by an explicit positive topology cap 3.

The check suffix uses the same input on purpose to test store separation;
it is not an actual MadLoop direction-stability test. No n4 warmup is run.

## Key, failure, budget, and exception guard design

Each scalar observation records the actual ordered native arguments after
assembly, family, partial/full branch, exact serialized MPFR values, each
component's actual precision, and a separate sign bit including signed zero.
The immutable configuration namespace is bound to the source and dependency
hashes. Binary64 conversion and symmetry-based argument canonicalization are
absent. Parent rank, group, context, event epoch, external pole mode, primary
or check flag, and the actual Fortran cache-eligibility boolean are retained
separately. Parent topology limits therefore cannot be bypassed by the
smaller scalar family.

The offline analyzer distinguishes all requests, exact unique keys, repeats
within one parent, repeats across parents, local scalar errors/nonfinites,
and candidates discarded because the enclosing tensor/packing failed.
The replay analyzer also binds the public Fortran return status to each
bridge parent; successful bridge output cannot authorize a candidate if the
Fortran decimal parse/transport subsequently refused it.
Successful primary candidates never cross check, full/partial, pole-mode,
event, context, or synthetic/physical namespaces. Check and cache-off
requests remain visible but are excluded from candidate insertion. A failed
parent may have zero scalar calls if geometry rejected it first; that is an
observation, not a missing failure to hide.

The journal is capped at 4096 detailed scalar observations, 16384 event
records, and 8 MiB of retained JSON payload. Fixed global request/error/time
counters continue after the cap. Dropped events force
`uniqueness_complete=false`; derived uniqueness and candidate counts then
are suppressed, since a missing lifecycle/status event can invalidate epochs
and eligibility, not merely lower a count. The cap is diagnostic storage,
separate from the unchanged production cache budget. Budget modeling uses
recorded occupancy and explicit estimated MPFR payload/metadata sizes; it
cannot establish actual usable cache hits or precise allocated bytes.
The journal's 8 MiB limit covers retained JSON payload, not peak allocator
usage: event-vector metadata, transient key buffers, and JSON snapshot copies
need additional diagnostic memory. None is charged to the numerical cache.

The pinned MPFR header declares `mpfr_flags_save(void)` returning unsigned
int. The bridge exposes its read-only call through a safe callback. The
private core module uses that callback before and after key/output
observation, and records parent/kernel flags. It never clears or restores
MPFR flags. A key/output-observation flag change invalidates equivalence.
The replay also captures all five host IEEE flags immediately around the
Fortran API call from identical cleared host-flag states. The old production
DSO hides its MPFR symbols, so its internal sticky flags cannot be observed
through its public ABI. Counter-off versus counter-on private runs must
match MPFR flags exactly; original versus private must match outputs,
statuses, cache counters, and host flags. This limitation will remain
explicit rather than inventing access to the original hidden state.

## Deferred build and replay limits

The guarded `build.py` recipe uses two Cargo jobs, locked/offline dependencies,
and a 900-second build cap with no automatic retry. It reuses fingerprinted
objects in the ignored Cargo cache; the private library name and copied DSO
live under the isolated experiment. The root must first review source and
schedule a compiler window. A source review is not native launch approval.
The historical `49af12f0` and reconstructed `92da` lockfiles use path-resolved
oneloopmaster, so the private manifest binds an isolated unchanged
`bd703d09` source copy explicitly. The actual current CLI installation
`095e` uses the Git source at the same revision; its lockfile is not claimed
byte-identical. The retained parent `lockfile-equivalence.json` records this
source-kind difference with no package-version or dependency-list drift.
Changed path fingerprints may rebuild oneloopmaster as well as iregi and the
bridge; cached dependencies do not guarantee completion within 15 minutes.
Attempt 2 reuses the immutable attempt-1 oneloopmaster path to retain its
Cargo fingerprint. A complete source-hash comparison binds it to an identical
attempt-2 copy; the builder verifies the actual dependency path before use.
The private crate name is `madloop_tirust_qp_shadow_attempt2`.
The authorized build lane is CPUs 2 and 3; the guarded builder verifies and
records the actual inherited affinity. Native replay is separately deferred.

After a successful authorized build, the original DSO, private counters-off,
and private counters-on replay processes must each finish within 180 seconds
(maximum 540 seconds total), while holding
`.tirust-build/numerical-tests.lock`. Each process serves the same 36 requests
in the same order. No slow retry or additional point set is automatic.
Exact output/input bytes, returned statuses, error text, and host flags must
agree before interpreting observations. The instrumented on/off MPFR flag
files must also agree. Per-kernel and key-construction times are attribution
diagnostics with instrumentation overhead; they are not amplitude timings.

The separate `run.py` launcher requires its own explicit native-window flag,
verifies the frozen source index and executable/DSO hashes, takes the shared
numerical lock, and enforces these three subprocess limits without retry.
It suppresses non-protocol banners when comparing the complete tagged
36-call records. Raw logs remain retained. A cap suppresses opportunity and
budget interpretation; a parity failure prevents the profile summary.

At this source-only stage neither a compiler nor a native replay has run.
