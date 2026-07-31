# SymJIT 2.22.0 Arena Migration Report

## Outcome

pyAmpliCol now builds its compiled, recurrence, and eager JIT arena lanes on
the standard P-kernel implementation from `siravan/symjit-crate` 2.22.0. The
old private SymJIT fork and its fork-specific SymJIT direct-application and
direct-table implementations are no longer active dependencies. Native C++ and
assembly direct-table lanes remain supported and unchanged.

One narrowly scoped, generic SymJIT change was needed: a sound raw
plane-descriptor callable for P-kernels. It was submitted as
[`siravan/symjit-crate#1`](https://github.com/siravan/symjit-crate/pull/1),
accepted upstream, and is part of immutable upstream revision
`d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`. pyAmpliCol therefore carries no
SymJIT patch and has no private-fork dependency. No pyAmpliCol schedule,
factor, operation, or table concept was added to SymJIT, and the generated
kernel body is unchanged. No second mixed plane/scalar or output-operation
change was needed: pyAmpliCol owns persistent broadcast planes, safe
direct-output classification, persistent scratch, and allocation-free complex
epilogues. The migration implementation originally closed through a
user-directed short finalization. A combined publication-readiness pass is now
in progress; regenerated prepared packs and fresh/repeat installation evidence
are recorded below, while unfinished local, cross-platform, and performance
gates remain explicitly pending.

The implementation preserves the Python and CLI source contracts and the public
C ABI v1 plus generated C++, Fortran, and Rust SDK source interfaces. Final
multilanguage verification remains part of the clean acceptance gates.
The exported Rust `NativeRuntimeProfile` field set and Python profile-dictionary
shape are unchanged; migration-only scratch and broadcast counters remain
private diagnostics. The public `DirectArenaTrafficKind` and
`DirectArenaTrafficCounters` shapes likewise remain unchanged. The cold
instruction-to-P-kernel compiler is exposed to the private PyO3 crate through
a non-default, documentation-hidden `__private` feature bridge. That opt-in
symbol is explicitly unstable; the default rusticol-core API and the generated
public Rust SDK remain unchanged.
Internal generated-artifact ABIs were bumped and predecessor artifacts fail
closed with an instruction to regenerate.

The acceptance commands below preserve the originally planned release gate for
reference. Focused contract checks, fresh active runtime assets, release assets,
and local installation checks are complete. The full-gate and
paired-performance tables remain open until their current-source runs finish.

## Reproducible identities

The pre-change baseline source is:

- pyAmpliCol revision:
  `172e58fd33a3c65563866c50cfbb5e1ddcd7b302`

The most recent pre-pack clean candidate snapshot is retained only as
supporting evidence because the prepared-pack provenance and benchmark-evidence
hardening landed afterward:

- synthetic commit:
  `3b6ca4b6a44b35e47dd599c10f40d1848eb0f1e6`
- Git tree:
  `4a89fe690484d2dc63ca4d43ac9b2a31126e9944`
- snapshot directory:
  `.artifacts/symjit-2.22-migration/clean-snapshots/candidate-prepack-final-20260731T051631Z`

The current candidate and release prepared packs were produced by GitHub
workflows at source revision
`906a660b9b87c2f36827180031d566cb060f6b49` and integrated by
`d25907546d89154f90cb281d0811b48657d2d90a`. They use storage-v3 MIR and
`pyamplicol-symjit-plane-application-v2`; their exact bundle and metadata
hashes are recorded in the Prepared artifacts section. Dependency source
identity remains in the compact build locks rather than being duplicated in
prepared-pack metadata.

The tracked `portable-64le` self-test is a retained v2 fixture produced at
source revision `7c1f17c00c297352563228a0816324768cbc14bc`. It contains only
the v2 plane-application ABI and records the default of two post-build
validation samples, but predates the current
`pyamplicol-runtime-payload-identity` marker. Its fail-closed release-test
rejection is the reason for the current generated-fixture refresh; it is not
presented as the producer identity of the current prepared packs.

The saved migration plan originally pinned SymJIT 2.22.0 revision
`4e288ce5f3132b05e2a81eb6452c011b9e2bb936`. After that plan was approved,
the SymJIT author published a superseding 2.22.0 revision which implements
direct-arena coefficient outputs and changes P2 SIMD indices from lane-block
numbers to actual row numbers. The saved plan remains verbatim as the
historical directive; the implementation, dependency contracts, and this
report use the superseding immutable revision below.

The active SymJIT source is:

- repository: `https://github.com/siravan/symjit-crate`
- version: `2.22.0`
- immutable revision:
  `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`
- upstream integration:
  [`siravan/symjit-crate#1`](https://github.com/siravan/symjit-crate/pull/1)

The compact dependency identities after upstream integration are:

- canonical `Cargo.lock` SHA-256:
  `c726ae93f4508de45631b81ff0cf5f269c28a906622ae469aaec839e6dc57403`
- contributor lock SHA-256:
  `9955fc97521e1eb7780c078877a433d73b84db64892720e99fe98131bdd72653`
- release policy lock SHA-256:
  `4e9554bf54911e5f25e27214564f10ac58ec3a526e999712fc6dc2a194f4868e`
- saved migration plan SHA-256:
  `bd94f5df5f58fc7a98b90ce6ff4febd0c7fbf8b1acbce247986fef57d2ceb40e`

All dependency caches, checkout materialization, build outputs, probes, and
benchmark captures used for this migration live under the workspace.

## Generic SymJIT upstream change

The unpatched P-kernel callable exposes a plane table through
`*const &mut [T]`. That representation cannot soundly describe duplicate input
planes or exact input/output aliases because constructing the table would
create overlapping mutable Rust references. Those layouts are required by a
generic arena consumer and are valid for the generated P-kernel contract.

The upstream change adds:

- a `#[repr(C)] PlaneDescriptor<T>` containing a raw plane pointer and length;
- an unsafe generic compiled-plane callable preserving SymJIT's existing
  compiled-function Rust ABI;
- scalar and SIMD P-kernel accessors on `Applet`;
- a standalone test which invokes both scalar and SIMD raw accessors with
  duplicate planes and an exact input/output alias.

The returned callable remains unsafe: the caller must retain the `Applet`,
validate that the application is an indirect P-kernel, keep every descriptor
and backing plane alive, and satisfy the scalar/SIMD range contract.
pyAmpliCol enforces those requirements before obtaining or invoking a
callable. The function pointer follows SymJIT's existing native
`CompiledFunc` Rust calling convention; the change stabilizes the `#[repr(C)]`
descriptor layout without transmuting across calling conventions and does not
create a new public C ABI.

The change has no pyAmpliCol names, schedules, factors, operation catalogs,
`DirectApplication`, or `DirectTable` concepts. It does not alter the generated
prologue, body, or epilogue.

An earlier revision of the patched upstream `kernels` executable passed the
ordinary real/complex P- and B-kernel cases, scalar and SIMD P-kernels, and the
added raw duplicate plane plus input/output alias case. Its retained execution
log has SHA-256
`39f398a8f90d337a39d9e60eb800fde3cc587bdf38fc6a9537268c5db9f542c9`.
The accepted upstream implementation makes that test configuration explicit,
enables identity outputs, and directly executes the raw SIMD accessor with
lane-aligned actual row indices. Before submission, `cargo test --lib` passed
all four library tests and `cargo run --bin kernels` passed the kernel example
suite. These are upstream-change checks; the current pyAmpliCol release gates
are tracked separately below.

No second upstream change was needed. Point-independent literals and couplings use
persistent broadcast planes, mutable model-parameter planes refresh only when
their source bits change, structural zeros share the existing zero plane, and
nontrivial output policies are handled in Rust-owned persistent scratch and
epilogues. The superseding upstream revision now implements
`set_direct_arena_identity_output(false)`: that mode multiplies kernel outputs
by coefficients passed through `params`, and upstream includes real and
complex SIMD examples. pyAmpliCol explicitly uses identity output, so this
coefficient path remains dormant and is not required for correctness. The
feature can only implement a disjoint nonidentity overwrite; it cannot read
and merge a destination, fan out one result, or provide an alias snapshot. If
the measured campaign attributes a miss to scratch traffic, the first
patchless experiment will use this upstream coefficient mode for the safe
nonidentity-overwrite subset. Only evidence remaining after that experiment
could warrant the saved plan's narrowly allowed mixed plane/scalar extension.

Contributor installation now treats SymJIT like the other exact Git sources:
it clones the official repository, checks out the locked revision, verifies
the package name/version and rlib-only manifest, and path-patches the local
Cargo build to that checkout. Release builds use the ordinary immutable Git
entry in `Cargo.lock`. There is no SymJIT archive extractor, local patch
inventory, source-tree hash, patch-closure hash, or release-only Cargo-lock
projection. Candidate install state records only source URL, revision, and an
optional branch. This smaller contract keeps dependency failures actionable
without coupling prepared artifacts to checkout formatting or installer
ceremony.

## Runtime architecture

### Shared P-kernel adapter

The shared adapter translates Symbolica's structured
`Evaluator.get_instructions()` program into `symjit::Compiler`, enables
`set_direct_arena(true)`, and produces a complex P-kernel. Configuration is
constructed explicitly, including target, scalar/SIMD mode, optimization,
compression, threading, fast-math, and direct-arena settings, so an ambient
`symjit.toml` cannot alter generated code. SIMD preparation happens before the
application is sealed and serialized.

The persisted plane binding records split-complex plane order, logical inputs
and outputs, scalar sources, optimization settings, target requirements, and
the structured source digest. At load time the adapter validates SymJIT
storage and binding shape, while the enclosing artifact manifest integrity
checks cover the recorded binding bytes. The inner validator
requires the serialized compiler target and every option bit to match one of
the canonical explicit configurations. All three runtime lanes also compare
the serialized application's actual compression bit with the recorded
plane manifest, so metadata cannot misattribute payload size or load/runtime
performance. The validator rejects malformed word-size, endianness,
target-triple, and CPU-feature records. It shape-checks the source digest but
does not independently recompute the Symbolica source semantics.
The adapter keeps the `Applet` alive, cold-binds stable raw descriptors, and
invokes the scalar or SIMD callable directly. Owner-managed descriptor caches
use an explicit unsafe lifetime-erasure constructor: its `'static` storage tag
does not extend an allocation lifetime, and each runtime must invalidate the
cache before reuse and prevent any invocation or descriptor dereference after
backing storage moves, reallocates, or drops.

Scalar and SIMD callable indices are actual row (point) indices. Unaligned
heads and tails use the scalar kernel. The adapter supports the AArch64
two-lane and x86 four-lane paths plus scalar fallback without hot allocation.
Divergent disjoint SIMD blocks replay each lane with the scalar kernel.
Input/output-aliased descriptor tables disable SIMD and rely on the scalar
P-kernel prologue snapshot. Only exact aliases are accepted; shifted or
otherwise partial input/output overlaps fail closed because sequential scalar
writes could clobber later input points. Output planes must be mutually
disjoint.

### Compiled JIT

Compiled O0 through O3 fused arena stages use P-kernel plane bindings.
Disjoint identity-overwrite outputs bind directly to their final arena planes.
Persistent, change-detected broadcast planes support both literals and
parameters; current production compiled lowering emits parameter bindings.
The old fork-specific SymJIT `DirectApplication` lowering is not used.

### Recurrence JIT

The existing arena, topology-replay and all-flow-union schedules, role order,
factors, and selector behavior are retained. A cold classifier permits direct
output only for a proven disjoint identity finalization overwrite without a
before-write alias hazard.

Accumulation, non-identity factors, and alias-sensitive rows execute once into
persistent scratch planes. Allocation-free vectorized Rust epilogues then
apply the complex factor and overwrite/add policy in schedule order.

### Eager JIT

Rusticol owns eager row and attachment validation and preserves row order,
cross-row dependencies, fanout order, factors, overwrite/add semantics, and
hazard rejection. A single safe identity-overwrite attachment may bind
directly; other invocations use persistent scratch and an ordered vectorized
fanout epilogue. Coupling and model-parameter broadcast planes are filled only
when their source values change. SymJIT rows reuse one event-pinned imaginary
momentum or coupling plane as their read-only structural zero, so duplicate
real-only inputs need neither an extra arena plane nor repeated zero fills.
Cold schedule validation rejects every output or direct-copy destination that
could overlap that plane. Mixed native/SymJIT plans retain the native row plane
IDs and their ordinary structural-zero initialization; fill elision is enabled
only when every eager callable is SymJIT.

The native C++ and assembly eager table lanes remain unchanged.

### Traffic accounting

The covered direct lanes retain zero pack, gather, scatter, or remap traffic at
the arena-kernel boundary. Private runtime counters track internal scratch and
broadcast work separately, without extending the existing public Rust profile
struct or Python profile dictionary. Focused core tests use those counters so
internal work cannot be confused with boundary traffic. Outer
public-evaluation routes are verified separately by the integration gates.
The paired timing campaign does not itself enforce allocation or traffic
counters: final evidence composes the focused warmed-allocation and boundary
counter tests, genuine generated-artifact profile tests, and the separate
content-addressed timing/resource campaign.

## Artifact ABI and compatibility

The active internal identities are:

- shared plane application:
  `pyamplicol-symjit-plane-application-v2`
- compiled plane binding:
  `pyamplicol-compiled-plane-kernel-v2`
- recurrence plane binding:
  `pyamplicol-recurrence-plane-binding-v2`
- eager plane-table binding:
  `pyamplicol-eager-plane-table-binding-v2`
- prepared kernel variant:
  `pyamplicol-prepared-kernel-variant-v2`
- prepared pack identity:
  `pyamplicol-prepared-kernel-pack-identity-v2`

Loaders reject predecessor private-fork artifacts as compatibility errors and
tell users to regenerate with `pyamplicol generate`. They also reject
`pyamplicol-symjit-plane-application-v1`: those applications encode the
superseded SIMD-block index contract and cannot safely run through the
actual-row dispatcher. Focused tests cover the exact predecessor shared
application, compiled, recurrence, eager, prepared-variant, and prepared-pack
identities.

## Correctness and resource coverage

Focused P-kernel and lane tests cover lengths:

`1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025`.

They exercise:

- real and split-complex scalar and SIMD execution;
- duplicate input planes and exact input/output aliases;
- fail-closed shifted input/output and overlapping-output descriptors;
- disjoint overwrite and accumulation;
- exact complex factors and fanout;
- literals, coupling broadcasts, and model-parameter refresh;
- scalar heads and odd tails;
- sentinel preservation;
- status propagation and panic containment;
- recurrence direct-output versus scratch classification;
- eager row order and cross-row dependencies;
- zero warmed allocations for repeated calls after the relevant recurrence
  row-table identity and lane state are bound;
- zero boundary pack/gather/scatter/remap bytes.

An earlier complete Python validation run reported 3,238 passed, 2 skipped,
and 1 expected failure. The benchmark-contract suite reported 204 passed, the
dependency-policy suite reported 31 passed, and focused Rust predecessor tests
reported 2 passed. These development results are supporting evidence; the
final clean-snapshot gate results below supersede them.

### Historical short implementation finalization

The user-directed short finalization ran five focused Python contract tests
(artifact identity, required prepared-record fields, and the tracked portable
self-test) successfully under a 30 GiB guard with 0.050 GiB peak RSS. The
focused Rust fail-closed identity test also passed under the guard with
4.136 GiB peak RSS. That Rust invocation exposed and fixed two test-only calls
which treated the stored scalar function pointer as a method. `cargo fmt
--all --check`, Ruff lint, and `git diff --check` also passed. No broad test,
performance, installer, or release campaign was run.

### Current release-readiness gates

The combined publication-readiness pass resumed these commands. The first
current-source `just dev-test` invocation established a clean 3,663-test unit
result, then reached the release suite and failed 34 tests for one shared
fail-closed reason: the tracked portable self-test predated the current
artifact-identity marker. That generated fixture is being refreshed before the
clean rerun; the failure did not implicate arena arithmetic or runtime code.

| Gate | Result | Peak guard | Retained log SHA-256 |
|---|---:|---:|---|
| `just rust-test` | pending | pending | pending |
| `just source-gate` | unit phase passed; release phase stopped on stale self-test fixture | 3.335 GiB | `c7a5292a182fd19cd0dad563f47915743ba346f3afa68cd14924722c54ad7a43` |
| `just dev-test` | clean rerun pending after self-test refresh | 3.335 GiB | `c7a5292a182fd19cd0dad563f47915743ba346f3afa68cd14924722c54ad7a43` |
| local x86/Rosetta core and C API | pending | pending | pending |

The deployment gate is run with native-language tests required so C, C++,
Fortran, Rust, and Python coverage cannot silently skip because a compiler is
missing. Genuine compiled, eager, and recurrence generated-artifact allocation
tests are run with their fixture environment variables set and retained in the
gate log, rather than relying on their fixture-absent skip path. The migration
`symjit_2_22_generated_fixture_gate.py` runner first validates five distinct,
workspace-local artifact directories and fail-closed JSON manifests. Every
manifest must bind the exact candidate source revision and native-build
digest to one exact candidate-version pyAmpliCol producer and the sole exact
process `d d~ > z g g g g g g` with external PDGs
`[1,-1,23,21,21,21,21,21,21]`. Before Cargo starts, the runner stages one
candidate overlay and requires its build information to match that fixture
source revision, native-build digest, full candidate version, and candidate
fingerprint. The runner maps all ten artifact environment variables, isolates
six cache/build/temp roots, runs eight Cargo commands with native tests
required, and requires all nineteen named test-success markers.
Four of those markers exercise exact point counts `127`, `129`, `1023`, and
`1025` in each execution lane. The gate resets outputs to NaN before every
call, surrounds caller buffers with canaries, checks every point against the
one-point resolved-total reference, and exercises an alternating per-point
helicity selector to expose lane reorder or stale-tail errors. Caller-output
evaluation must remain allocation-free after warmup. Compiled/eager profiles
must report zero exposed boundary traffic, while a crate-private genuine
recurrence profile separately requires all three legacy and all five
lane-neutral packet/gather/scatter/remap counters to remain zero for
topology-replay and all-flow-union at every odd-tail size. Eager,
topology-replay, and all-flow-union profiles must also demonstrate activity in
the requested lane. The terminal watchdog report supplies the exit and memory
record; the captured Cargo log supplies the named markers.

## Prepared artifacts

The current tracked candidate packs were produced by the two producer jobs in
[workflow 30631372557](https://github.com/mg5amcnlo/pyamplicol/actions/runs/30631372557)
at exact source revision `906a660b9b87c2f36827180031d566cb060f6b49`:

| Target | Bundle SHA-256 | Bytes | Metadata SHA-256 |
|---|---|---:|---|
| AArch64 | `8857181f6877984d0df954c856e5165fc20d9d2be3d29b4f7544099f1ef26d1c` | 7,536,998 | `4f993aa9a7b73807ecd5d44e22e58c958902548c9a9a790a76cfd3384f19ab42` |
| x86_64 | `4c647c6252c406281d3ab9aa98e74f77b38a9d65881beccacf560f7b2a0fa740` | 7,533,092 | `5e4f6af76607d84d6db2705ad981c6b829fe3fc6ee88103671186024805f7dc0` |

The publication-mode packs were produced successfully by both jobs in
[workflow 30631374216](https://github.com/mg5amcnlo/pyamplicol/actions/runs/30631374216)
at the same exact source revision:

| Target | Bundle SHA-256 | Bytes | Metadata SHA-256 |
|---|---|---:|---|
| AArch64 | `6461fb4931e7da0a9fae3e51c227dc93f080d569b911d5ba451649b7dd8256b4` | 7,537,253 | `a212de9b2ce8f230cf12fcb1183ecc0340f61045c334b317054c654e5aba72b1` |
| x86_64 | `ba97c30576a11cfbf7045ee934abc00405b8cb4b5d3e024f9cae7e085c862fb0` | 7,533,401 | `87f6f81c57104074b81830029404522a1693d941bf35d8645b265a92c3e121c6` |

Each pack contains 59 base kernels, 36 variants, and 95 storage-v3 P-kernel
plane applications. All metadata declares SymJIT 2.22.0,
`pyamplicol-symjit-plane-application-v2`, model source digest
`7f6fc84e6d7c4eda6c531b9901ab891abd67c3c0e2ad9d492b30ece760352ba8`,
model-compiler digest
`24e1af18b5bae66bf00be47b7a6bbd2b8c26a66e0748aa82035de4e856e5d209`,
and prepared-pack compiler digest
`4539e9b34af5924354f0372ecbab897a355a7b6128a06093af4513e0593f54f5`.
The candidate producer version is
`0.1.0.dev0+candidate.72aea6cff06d`; the release producer version is `0.1.0`.

The producer transfer records validate the complete 95-application bundle and
the `d d~ > z` numerical check at `rtol=1e-12`, `atol=1e-15`. The complete
five-job portability workflow finished successfully: both pack producers and
the AArch64, macOS x86_64, and Linux x86_64 consumers passed. Earlier pack-size
and numerical checks remain historical development evidence, but these current
tracked hashes supersede their binary identities.

## Installation and dependency provenance

The current official-upstream dependency configuration passed both contributor
installation modes under the 30 GiB watchdog:

| Install | Exit | Peak guard | Log SHA-256 |
|---|---:|---:|---|
| fresh cache | 0 | 19.049 GiB | `b668565a9c55654c86a3bddbc92dfb700f10bc526f39cba8d8484ebc92bdeb41` |
| repeat | 0 | 3.395 GiB | `9e785fc91ea9cb6d5f165612ab8c429f6e936bb2a27e24e2da85f597ba713327` |

Both runs cloned the official SymJIT repository at `d8abfeeb…`, validated its
package name, version, and rlib manifest, built the contributor environment,
and installed candidate version `0.1.0.dev0+candidate.72aea6cff06d`. No local
SymJIT patch was applied. A separate dependency-postcondition capture exited
zero with 0.099 GiB peak guard and log SHA-256
`6da77bf143cacaea2bd900a420768c3577a68caa8117af7c1cfb7881787b456a`;
its Cargo tree contains one `symjit v2.22.0` source, the official immutable Git
revision `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`.

For historical comparison, a superseded pre-upstream-integration candidate
also passed fresh/repeat installation with peaks 18.946/3.376 GiB and log
hashes `4d778a372e9896e17c8eb0acf84cbfa0ce12d1c5e9a40fa3101f4938628ad8d6`
and `80b914f8df0e5b9eaa0b52ea3639c794196aa4086b354334587642636da71d34`.
Those runs exercised the former local-patch machinery and fingerprint
`a38cc68d0520`; they are retained only as migration history and are not the
active dependency contract.

Current tracked-tree scans contain no private-fork URL, branch, or revision and
no retired ABI in active source, policy, or artifact roots. Immutable JSON
captures under
`docs/performance_reports/**/results/` retain their historical runtime ABI
metadata. The x86 performance-runtime reconstruction helper likewise embeds an
exact SymJIT 2.21.1 baseline identity so the candidate workflow can
reproduce the pre-migration comparison. Both are frozen baseline evidence, not
active dependency inputs or loadable generated artifacts.

## Original AmpliCol comparison

The original Fortran AmpliCol implementation was inspected at clean revision
`79c96cecf2a722e50c3d2030b6894d755f96518a`, together with its historical
`vectorisation` and `parallel` branches and the older Python/Symbolica
implementation. The primary clean-reference evidence is in
`amplitude_QCD.f03`, `feynmanrules.f03`, `amplicol_generate.f03`,
`amplicol_library_benchmark.f03`, and `readme.txt` in that checkout.

The dynamic evaluator lazily allocates its current/interaction work buffers on
the first evaluation and then retains them. The generated-library path is
different: each one-point call uses exact-sized automatic dense scratch while
the caller retains momentum and amplitude buffers outside its timing loop.
Thus the transferable invariant is zero repeated heap allocation after warmup,
not that every generated workspace must literally be allocated during cold
setup.

Its recurrence organization is more directly useful. For each subset size it
evaluates vertices into disjoint interaction scratch, clears each destination
current, accumulates ordered signed contributions, and applies one
propagator/finalization. Closures overwrite base amplitudes before ordered
partner and same-flavour additions. Child inputs are structurally disjoint, and
the only exact in-place operation is post-reduction propagation. This supports
pyAmpliCol's safe identity-finalization direct-output versus
scratch-and-epilogue boundary, but supplies no precedent for duplicate planes,
shifted aliases, or SIMD-safe aliases.

Generated AmpliCol modules group static rows by primitive type and chirality,
embed index/coupling/mass/width tables, and loop over those groups inside a
one-point routine. pyAmpliCol transposes those loops: each recurrence row or
eager invocation calls a P-kernel across points. That favors split-plane SIMD
at larger batches but makes per-row calls, descriptors, and broadcast planes a
specific batch-1 risk that the paired campaign must measure. AmpliCol's
component-contiguous one-point layout likewise warns about batch-1 locality; it
does not provide a reason to undo pyAmpliCol's point-contiguous planes.

The historical vector branch is not a safe tail model: it fills a fixed
eight-point batch before evaluation, has no partial-lane path, and some
wave-function branches choose a control path from lane zero for the whole
batch. It therefore reinforces the migrated adapter's scalar heads/tails and
divergent-block scalar replay rather than replacing them. The historical
parallel branch gives each concurrently evaluated point independent mutable
state; any future threaded pyAmpliCol evaluator must likewise retain
worker-owned arena, scratch, and descriptor storage.

AmpliCol embeds stationary couplings and requires regeneration after a model
change. Refresh-on-change broadcast planes and the destructive-write safety
classifier are therefore pyAmpliCol extensions needed to preserve its mutable
public API. If measurements isolate broadcast overhead, the saved plan's
generic mixed plane/scalar parameter path is the closest transferable
improvement.

Original AmpliCol evaluates one phase-space point per call and has no explicit
SIMD-block or odd-tail contract. Its numerical current/helicity equivalence
probes are also not structural proofs and are deliberately not copied.
Finally, its generated-library mode is per group/integral rather than one
universal all-flow-union kernel, so performance conclusions keep those models
distinct.

## Performance campaign

The exact-bound alternating campaign uses the process
`d d~ > Z g g g g g g`, with two warmups and seven independent alternating
subprocess pairs per cell. It covers topology replay with selected runtime flow
and helicity sum, all-flow union with all flows and a nonzero alternating
runtime helicity, recurrence JIT O2 and compiled JIT O3, and batches
`1`, `128`, and `1024`. The separate eager diagnostic consumes the immutable
prepared O2 applications. Odd-tail numerical/allocation cases `127/129` and
`1023/1025`, the retained built-in/UFO `u u~` portability route, and the eager
diagnostic are separate required evidence.

The orchestrator and comparison gate both require the exact final candidate
source revision, the retained baseline and candidate native-build-input
digests, and the retained baseline and candidate prepared-model SHA-256 values
as external inputs. They reject self-consistent captures from any other clean
candidate, native build, or loadable pack. The candidate capture driver passes
the expected source/native identity into the underlying harness so a mismatch
fails before useful generation work. Because the immutable nine-leg baseline
union generation already exceeded one hour, the authoritative driver enforces
minimum generation and coordination timeouts of 10,800 and 43,200 seconds.

Each retained outer subprocess sample contains at least seven raw warmed blocks
and continues until both its summed native timing and enclosing caller timing
account for at least five seconds. The producer and acceptance gate
independently recompute repetition calibration, raw counts, native and caller
durations, worker enclosure, medians, and raw MADs. Exactly two warmups are
required. Eager runtime, generation, and resource results are required evidence
under the same regression rules, not advisory output.

The campaign is intentionally not run while another task is collecting
authoritative measurements on the same host. The entire command is enclosed by
the 30-GiB watchdog. Its schema-2, content-addressed report binds the exact
watchdog bytes, command arguments, session, result-file digest, timestamps,
exit status, memory limit, peak observations, and probe counters. The report
permits recovered platform probe failures only within the watchdog's existing
retry policy and records the total and maximum consecutive failures; three
consecutive failures still fail closed. The migration gate independently
validates that watchdog report.

The campaign's content-addressed manifest, four capture hashes, watchdog
report, medians, raw MADs, paired compiled/recurrence ratios, eager results,
generation results, and payload/load/RSS results will be inserted here only
after that campaign completes. The current bundle/metadata hashes and
fresh/repeat install results are already recorded above. Remaining final
evidence includes the generated-fixture log and all nineteen required markers;
the manifest/overlay preflight binds all five generated fixtures and the staged
candidate build to one exact source revision, native-build-input digest,
candidate version/fingerprint, and the sole
`d d~ > z g g g g g g` process with external PDGs
`[1,-1,23,21,21,21,21,21,21]`. It also includes the compact one-source Cargo
tree, private-fork scans, and schema-7 built-in/UFO portability captures with
schema-6 capture-acceptance metadata. There is no active SymJIT patch or
source-tree attestation to reproduce.

### Runtime acceptance

Pending.

### Generation and resource acceptance

Pending.

## Final conclusion

The arena implementation migration and upstream dependency cutover are
complete: all active compiled, recurrence, and eager JIT arena paths use
standard SymJIT 2.22.0 P-kernels from official immutable revision
`d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`, and the required generic raw-plane
interface is upstream. No private fork or local SymJIT patch remains.

Candidate and publication-mode prepared packs have now been regenerated for
AArch64 and x86_64, the three cross-architecture consumers pass, and
fresh/repeat contributor installation plus the compact official-source
dependency check pass. The current-source unit suite passes all 3,663 tests;
portable self-test regeneration, the clean full multilanguage rerun, the
multi-hour alternating performance campaign, and final wheel/sdist publication
workflows are still pending in the combined release-readiness pass. They remain
explicit open evidence rather than implied passes.
