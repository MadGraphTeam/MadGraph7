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

Full finite-term DP requests use the embedded reduction with local tensor
precision tiers: 106-bit DD, then 256-bit and 512-bit MPFR when conditioning
requires them. The tiers lift the binary64 inputs exactly and return DP tensor
coefficients for contraction with the DP numerator. They preserve the existing
conditioning gates and MadLoop's independent stability checks. Refused tensor
groups can therefore recover before MadLoop constructs an entire QP numerator.
These local tiers do not provide a binary128-input amplitude. Select backend 9
explicitly through `MLReductionLib`; use `install tirust --update` to rebuild an
existing installation with this policy.

Genuine Fortran binary128 inputs use 64-byte, 40-significant-digit ASCII
records, parsed at 113 bits and then widened to 256-bit MPFR. Signed zero is kept.
A 256-bit QP workspace accepts conditioning down to 2^-136, which keeps its
rounding near 2^-120 of the result. A lane that this gate refuses on
conditioning (small Gram determinants: loop-induced g g > h h at small pT) is
evaluated once more from the same exact inputs at 512 bits, with gate 2^-392.
Exactly singular pivots and every other failure are not retried.
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
results. A failed full-DP reduction clears its numerical stores and its
repeated-tensor cache. Ordinary numerical refusals retain warmed structural
workspaces; a Rust reduction error or invalid lane discards both directional
owners while retaining the immutable plan. Entries computed on any accepted
route (direct, method choice or a precision tier) stay in the store: every
entry is a pure function of its exact key (method, settings and input bits),
so TIRust returns the same bits with or without it, and a later integral or
retry receives exactly what it would compute itself.
The separate pole-only evaluator retains its existing failure cleanup.

DP precision tiers start a refused lane at the lowest tier whose own gate
admits its binary64 conditioning estimate (`tier_margin_bits = 0`): a lane
the double-double gate accepts is not sent to 256 bits on a pessimistic
margin. The double-double tier accepts conditioning down to 2^-80 (26 bits
of headroom, `double_double_headroom_bits`; its rounding amplified by at most
2^-26); the MPFR tiers keep 40 bits. On MadLoop's QCD Zggg the double-double
values of the lanes this admits round to the 256-bit binary64 values (all but
2 of 53,200 recorded tensors bit-identical, those within 3e-28).

Register each `(N,rank)` plan before reducing a point. Clear numerical caches at
the end of each phase-space point while retaining plans/workspaces. Full and
pole-only DP workspaces have separate subintegral stores. MadLoop requests each
loop group's tensor once per helicity while it learns its helicity filters;
an exact repeat of a primary full-DP request (identical input bits, `(N,rank)`
and pole selection) within the phase-space point returns the stored complete
tensor. QP calls have a bounded
cache of complete tensor results keyed by the exact input records, `(N,rank)`,
pole selection, finite-term mode, and input/working precision. Full and pole-only
QP results cannot share an entry. Failed evaluations are never retained.

The default total numerical cache budget is 256 MiB. Half is reserved for QP
tensor results and a quarter for the primary full-DP subintegral store. The last
quarter is split between the pole-only DP store (half of it, when poles are
supported), the independent-check subintegral store and the repeated-tensor
cache.
Setting the same budget is a no-op; changing it clears numerical entries. A full
QP cache refuses further insertions without evicting existing entries. Accounted
QP bytes include retained payloads and estimated entry metadata, excluding
plans/workspaces; this counter does not measure process resident memory.

Independent stability passes never read entries of the primary pass: they
share subintegrals only with each other, in their own store (bit-identical to
unshared evaluation), and never use the repeated-tensor cache. Calls with
caching disabled bypass numerical caches completely. `tirust_cache_statistics`
returns nine aggregate counters over the primary and check stores; its final
five also include the complete-tensor hits, misses, entries, bytes and refused
insertions of the QP and repeated-DP-tensor caches. A repeated-tensor hit is not
a bridge execution and does not change the route counters. `tirust_qp_cache_statistics` exposes the six QP counters separately:
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

## Optional reduction-route diagnostics

New installations advertise `dp_route_statistics: true` in their existing ABI-1
metadata. `tirust_dp_route_statistics(counts,status)` returns twenty cumulative
**bridge execution** counters, with a primary block followed by an independent
check block. Each block contains `[Direct, MethodChoice, Tier106, Tier256,
Tier512, Refused, unreported/reduction-error, attempted106, attempted256,
attempted512]`. Event-cache clearing preserves these counters. Fortran coefficient
cache hits bypass the bridge and are outside their scope. A nonzero unreported
bucket makes attempted-tier totals lower bounds.

Tier106 is DD. Tier256/512 process binary64 inputs with a DP numerator. None of
these routes is the genuine binary128-input QP interface (256 bits, with one
512-bit retry). The `TIRust DP` amplitude counter and return code `213`
(MadLoop return-code format 2: TIRust reports `U=3`, shared with IREGI, with
`T` for precision; format 1 used `219`) identify the DP-input interface; they
do not prove native-only tensor evaluation. Genuine QP keeps its own counter
and return code `323`.

The generated `GET_TIRUST_DP_ROUTE_STATISTICS` routine returns zero counters and
status `UNSUPPORTED` when an older ABI-1 installation lacks this optional
capability. Existing installations and exports remain supported; unavailable
route data must be reported as unavailable. Use `install tirust --update` to
obtain the new diagnostics. A statistics query does not initialize an unused
context or clear a prior local diagnostic.
