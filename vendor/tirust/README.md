# MadLoop TIRust bridge

`install tirust` builds this package in an isolated installation directory and
applies the checked patch series to its own copy of TIRust. Reference checkouts
are never edited. The supported build target is GNU Fortran on Linux, with Rust
1.97.1 and the exact Git dependencies in `Cargo.toml`/`Cargo.lock`.

The Fortran module `mtirust` owns the MadLoop-facing interface. Its small
`libtirust.a` contains only Fortran code; the Rust implementation is isolated in
`libmadloop_tirust.so`. This keeps the bridge's numerical dependencies and Rust
runtime separate from an existing NinjaRS static installation. `make exports`
checks that the shared library exposes only its prefixed C API. Applications
must link both libraries and supply a runtime search path for the shared library.

## Symbolica OEM activation

The signed TIRust revision activates its private Symbolica library key for the
Rust crate `iregi_rust` at the library's symbolic construction and evaluation
boundaries. Calls to its oneloopmaster dependencies made within those operations
share the scoped authorization. Worker operations establish their own guards;
the key does not globally license unrelated Symbolica use in the embedding
application or a separate NinjaRS installation. No personal
`SYMBOLICA_LICENSE` is required for these covered TIRust operations.

For an existing MadGraph installation, run `install tirust --update` to build
and select the signed dependency. Omitting `--update` reuses an already valid
installation, which may predate the OEM activation. Installation still copies
sources to an isolated stage and smoke-tests the resulting DP/QP and poles-only
interface before transactional publication. Its source, lock and artifact
provenance identify the actual build; historical unsigned benchmark artifacts
remain unchanged and do not qualify the new source automatically.

## Numerical contract

The qualified first version accepts one through six propagators and cumulative
tensor ranks zero through six, ordinary undotted integrals in D=4-2 epsilon.
Shifts are time-first, the metric is (+,-,-,-), and denominators are
`(q+p_i)^2-m_i^2`. Exact supplied invariants, squared masses with nonpositive
imaginary parts, and positive real mu squared are required.

Results have IREGI Cartesian ordering and four independent slots:
`[UV single, IR single, IR double, finite]`. `Convention::MadLoop` applies the
odd-rank sign once. The generated MadLoop adapter performs its coefficient
permutation, contraction and overall normalization. No multinomial weight is
included. Pole selection preserves the complete finite coefficient; the separate
`compute_finite=false` request uses the patched truncated evaluator and reports
an unsupported status when that feature has not been built.
The low-level ABI pads the unevaluated finite slot with zero in this mode; that
placeholder must be ignored. MadLoop marks it NaN in partial-result diagnostics.

Double precision uses the embedded numerical reduction with the host's stability
checks. Genuine Fortran binary128 inputs use 64-byte, 40-significant-digit ASCII
records, parsed at 113 bits and then widened to 256-bit MPFR. Signed zero is kept.
Fortran contracts the returned moments in binary128 before the final amplitude
is rounded to the existing MadLoop result type. Double-double is not called
quadruple precision. Numerical working precision does not by itself certify the
accuracy of an ill-conditioned result; MadLoop still performs its checks.

## Ownership and caches

The C interface uses opaque caller-owned contexts and caller-owned result
buffers. All buffers have explicit lengths, errors leave C output unchanged,
and Rust panics are contained at the boundary. The Fortran wrapper initializes
failed results to zero and returns a non-success status. The module owns one
serial context; separate concurrent callers must use separate C contexts.
If a panic poisons the context, subsequent calls report an error and cannot read
its caches. Call `tirust_finalize` before retrying; the next request creates a
fresh context. Full-DP primary and independent-check workspaces share one
immutable reduction plan per `(N,rank)`, while retaining separate structural
variants and scratch buffers. Check passes still recompute the numerical
results. A failed full-DP reduction clears its store and discards both
directional workspaces while retaining the immutable plan; a failed pole-only
reduction discards its separate store and affected workspace.

Register each `(N,rank)` plan before reducing a point. Clear numerical caches at
the end of each phase-space point while retaining plans/workspaces. Full and
pole-only DP workspaces have separate subintegral stores. QP calls have a bounded
cache of complete tensor results keyed by the exact input records, `(N,rank)`,
pole selection, finite-term mode, and input/working precision. Full and pole-only
QP results cannot share an entry. Failed evaluations are never retained.

The default total numerical cache budget is 256 MiB. Half is reserved for QP
tensor results; the remainder is divided equally between the full and pole-only
DP stores when poles are supported, or assigned to the full DP store otherwise.
Setting the same budget is a no-op; changing it clears numerical entries. A full
QP cache refuses further insertions without evicting existing entries. Accounted
QP bytes include retained payloads and estimated entry metadata, excluding
plans/workspaces; this counter does not measure process resident memory.

Independent stability passes and calls with caching disabled bypass numerical
caches completely. `tirust_cache_statistics` returns nine aggregate counters;
its final five include QP tensor hits, misses, entries, bytes and refused
insertions. `tirust_qp_cache_statistics` exposes the six QP counters separately:
`[hits, misses, entries, bytes, not_stored, bypassed]`. A bypass increments only
the final QP counter, leaving cache hits and misses unchanged. Hits and misses
count complete tensor requests, including misses that later fail evaluation.
The C header documents both interfaces. QP reuse is of complete tensors; upstream MPFR
subintegral caching is not implied.

## Validation

In an installer-prepared source stage, `make check` runs bridge tests, an export
audit and the Fortran smoke. Installed
validation runs `smoke --require-poles`, so a build missing the accelerated
feature cannot be published as a complete installation. Tests cover binary128's
last mantissa bit, directed rounding, split scaleless poles, rank-six tensor
ordering/signs, selection/cache changes and error recovery. Physics and timing
qualification is performed by the repository's benchmark campaign, separately
from these interface checks.

`python check_fixtures.py PATH/TO/libmadloop_tirust.so
PATH/TO/TIRust/crates/iregi-rust/tests/fixtures/method_choice_lanes.json --qp`
checks the actual C ABI against the pinned upstream NinjaRS reference values.
It accounts for the MadLoop odd-rank sign and contracts all tensor moments.
DP refusals are reported explicitly; every QP reference lane must succeed.
The `gate_passed_lanes.json` stress fixture uses its documented `--dp-limit 1e-5`.
