# SymJIT 2.22.0 Arena Migration Report

## Outcome

pyAmpliCol now builds its compiled, recurrence, and eager JIT arena lanes on
the standard P-kernel implementation from `siravan/symjit-crate` 2.22.0. The
old private SymJIT fork and its fork-specific SymJIT direct-application and
direct-table implementations are no longer active dependencies. Native C++ and
assembly direct-table lanes remain supported and unchanged.

One narrowly scoped, generic SymJIT patch is required. It exposes a sound raw
plane-descriptor callable for P-kernels. No pyAmpliCol schedule, factor,
operation, or table concept is added to SymJIT, and the generated kernel body
is unchanged. No need for a second mixed plane/scalar or output-operation patch
has been identified so far: pyAmpliCol owns persistent broadcast planes, safe
direct-output classification, persistent scratch, and allocation-free complex
epilogues. The wider performance and release-readiness campaigns were
explicitly deferred by the user to a later combined release pass; they are not
part of this short implementation finalization.

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
reference. The implementation round closed with focused contract checks and
fresh active runtime assets; the full-gate and paired-performance tables are
intentionally left for the later release-readiness pass.

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

The integrated source through `7c1f17c` produced candidate fingerprint
`34f4013080a1`. Fresh active AArch64 and x86_64 portable prepared-model bundles
were generated from the same storage-v3 MIR and have identical SHA-256
`86509430038002698d39eea9f0c4b104e3390194cbd58469de7e3c161862080a`.
Their metadata pins SymJIT
`77789ff0f78232b1ea4608aceb397058df50b06d` and declares
`pyamplicol-symjit-plane-application-v2`.

The tracked `portable-64le` self-test was also regenerated from the integrated
runtime. It contains only the v2 plane-application ABI, requires the current
`pyamplicol-runtime-payload-identity` marker, and records the new default of two
post-build validation samples. Publication-only prepared packs under
`release_assets/` were deliberately not regenerated in this implementation
round; release staging rejects them fail-closed until the later release pass.

The saved migration plan originally pinned SymJIT 2.22.0 revision
`4e288ce5f3132b05e2a81eb6452c011b9e2bb936`. After that plan was approved,
the SymJIT author published a superseding 2.22.0 revision which implements
direct-arena coefficient outputs and changes P2 SIMD indices from lane-block
numbers to actual row numbers. The saved plan remains verbatim as the
historical directive; the implementation, dependency contracts, and this
report use the superseding immutable revision below.

The authenticated SymJIT source is:

- repository: `https://github.com/siravan/symjit-crate`
- version: `2.22.0`
- immutable revision:
  `77789ff0f78232b1ea4608aceb397058df50b06d`
- archive SHA-256:
  `b3cb6451eff299b27709115053caed579bc266bbd46923a70066b5ac554dd0ac`
- pristine tree SHA-256:
  `88aa6a50ec7ad120d3d832f4d98e3efe89ea259e925c9ed139904b8dd7607453`
- configured tree SHA-256:
  `4b4b791b0f2bbef33a7dbd2936d20dc722f7301e2e9e986b65b2a8b94d220b31`
- ordered patch-closure SHA-256:
  `01b486a472d14f89d43be3c658407166b2793190fa3f79fa15175e49d4788474`

The authenticated dependency identities are:

- canonical `Cargo.lock` SHA-256:
  `1d368c76fd51aa5ab30aaf59fb31c89eb9ae3aaa8ac3e1768934566c6265476c`
- authenticated release-local Cargo lock projection SHA-256:
  `acc74268440a3913efcbfdff031d23abc2fc956b522b054996543a4576f9edd3`
- contributor lock SHA-256:
  `1a7ed087f2263b066619818df60cf40f14e2db4200d74b00623a256495ded9ff`
- release policy lock SHA-256:
  `a590b7b88e48789b01ac34ee96d93dd9d9c8c9456ae13137cf9bde7eb9481bcd`
- saved migration plan SHA-256:
  `bd94f5df5f58fc7a98b90ce6ff4febd0c7fbf8b1acbce247986fef57d2ceb40e`

All dependency caches, checkout materialization, build outputs, probes, and
benchmark captures used for this migration live under the workspace.

## Generic SymJIT patch

The only upstream patch is:

- identity: `symjit-raw-plane-descriptor-v1`
- file:
  `dependencies/patches/symjit/upstream/0001-Expose-a-stable-raw-P-kernel-plane-descriptor.patch`
- SHA-256:
  `70012117436d77265b349c013b0df8c4fe72a04ced972090ba3ac069721b436d`

The unpatched P-kernel callable exposes a plane table through
`*const &mut [T]`. That representation cannot soundly describe duplicate input
planes or exact input/output aliases because constructing the table would
create overlapping mutable Rust references. Those layouts are required by a
generic arena consumer and are valid for the generated P-kernel contract.

The patch adds:

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
`CompiledFunc` Rust calling convention; the patch stabilizes the `#[repr(C)]`
descriptor layout without transmuting across calling conventions and does not
create a new public C ABI.

The patch has no pyAmpliCol names, schedules, factors, operation catalogs,
`DirectApplication`, or `DirectTable` concepts. It does not alter the generated
prologue, body, or epilogue.

An earlier revision of the patched upstream `kernels` executable passed the
ordinary real/complex P- and B-kernel cases, scalar and SIMD P-kernels, and the
added raw duplicate plane plus input/output alias case. Its retained execution
log has SHA-256
`39f398a8f90d337a39d9e60eb800fde3cc587bdf38fc6a9537268c5db9f542c9`.
The rebased patch makes that test configuration explicit, enables identity
outputs, and directly executes the raw SIMD accessor with lane-aligned actual
row indices. Its replacement full-gate execution log is deferred to the
release-readiness pass.

No second patch has been added. Point-independent literals and couplings use
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

Contributor and release locks authenticate the final patch bytes and configured
source tree. Both candidate and release prepared-pack metadata now bind the
configured SymJIT tree and the exact ordered patch records directly under
`build_contract.symjit_source`; the producer additionally binds the canonical
native source closure through `native_build_inputs_sha256`. Candidate and
release validation rehash the patch files and reject any tree, order, path,
revision, or byte drift. Release builds canonicalize only the two authenticated
forms of `Cargo.lock`—the immutable Git source form and the exact installer
path projection—so those representations have one release native-build
identity without accepting a general path dependency. Before submitting
upstream, the mechanically generated `git format-patch` should be replayed onto
a named upstream branch so the proposed change has a stable reviewer-visible
commit identity. An accepted upstream revision would then require an
intentional dependency-lock refresh.

Source-runtime staging records whether its native identity uses the candidate
or release projection. Verification recomputes that exact projection, requires
the top-level and source-runtime digests to agree, and checks the staged
extension's exported build ID even for a publishable release build. Class-C
bridges retain the historical absent-mode candidate interpretation, while an
explicit release marker avoids materializing contributor-only checkouts and
state in the ancestor worktree.

Candidate native identity is likewise semantic rather than installation-path
dependent. It validates the installer-recorded hashes for the candidate Cargo,
contributor, Python-runtime, and release locks plus the raw generated Cargo
config, then hashes an exact canonical Cargo patch table and a stable installer
projection. That projection retains the ordered generic patch contract and the
four build-relevant source revisions and tree digests, while excluding
`created_utc`, absolute checkout paths, and the optional legacy AmpliCol
checkout. Consequently relocation, JSON/TOML formatting, and an otherwise
identical repeat `just dev-install` do not invalidate packs, while lock, source
tree, patch, or Cargo-mapping drift still fails closed. Release/sdist patch
inventory is derived from the authenticated ordered patch list rather than a
hard-coded filename. Wheel builds from an unpacked sdist do not require a
system Git executable: the release backend applies the authenticated
existing-text-file unified-diff subset itself, requires exact hunk matches,
rejects file creation/deletion/rename, binary changes, unsafe paths, and
symlinks, and verifies reverse applicability before writing. Its final
configured-tree hash remains authoritative. Contributor installation likewise
rejects a symlinked managed SymJIT checkout root or target before reading,
patching, hashing, configuring, or recording that tree.
Both secure archive extractors preserve only the authenticated executable bits
which participate in the source-tree digest; this is required by the pinned
GitHub archive and avoids importing unrelated archive permissions. A managed
checkout whose tree matches neither the pristine nor configured lock identity
is moved recoverably into the workspace-local `.trash` store and replaced.
Thus an ordinary `just dev-install` can repin two immutable revisions carrying
the same crate version without requiring a broad reset.

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
the structured source digest. At load time the adapter authenticates SymJIT
storage and validates binding shape, while the enclosing artifact
authentication protects the recorded binding bytes. The inner validator
requires the serialized compiler target and every option bit to match one of
the canonical explicit configurations. All three runtime lanes also compare
the serialized application's actual compression bit with the authenticated
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

### Short implementation finalization

The user-directed short finalization ran five focused Python contract tests
(artifact identity, required prepared-record fields, and the tracked portable
self-test) successfully under a 30 GiB guard with 0.050 GiB peak RSS. The
focused Rust fail-closed identity test also passed under the guard with
4.136 GiB peak RSS. That Rust invocation exposed and fixed two test-only calls
which treated the stored scalar function pointer as a method. `cargo fmt
--all --check`, Ruff lint, and `git diff --check` also passed. No broad test,
performance, installer, or release campaign was run.

### Deferred release-readiness gates

These commands are intentionally deferred by the user's short-finalization
directive:

| Gate | Result | Peak guard | Authenticated log |
|---|---:|---:|---|
| `just rust-test` | pending | pending | pending |
| `just source-gate` | pending | pending | pending |
| `just dev-test` | pending | pending | pending |
| local x86/Rosetta core and C API | pending | pending | pending |

The deployment gate is run with native-language tests required so C, C++,
Fortran, Rust, and Python coverage cannot silently skip because a compiler is
missing. Genuine compiled, eager, and recurrence generated-artifact allocation
tests are run with their fixture environment variables set and retained in the
gate log, rather than relying on their fixture-absent skip path. The migration
`symjit_2_22_generated_fixture_gate.py` runner first validates five distinct,
workspace-local artifact directories and fail-closed JSON manifests. Every
manifest must bind the authenticated candidate source revision and native-build
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
topology-replay, and all-flow-union profiles must also authenticate activity in
the requested lane. The terminal watchdog report supplies the exit and memory
attestation; the captured Cargo log supplies the named markers.

## Prepared artifacts

The following candidate prepared packs predate the final patch-provenance
refresh and are retained as supporting payload-size evidence only:

| Target | SHA-256 | Bytes |
|---|---|---:|
| AArch64 | `33c9f762362d9c0136db154e57dc666dc221386b5000cbdae8b34f848c59d5ef` | 7,536,825 |
| x86_64 | `781fd7fee8d6cc3d63ed4a59b5468065b8aa143edecb4edb5e81d672a3ebf3c5` | 7,533,028 |

Their corresponding release copies are:

| Target | SHA-256 | Bytes |
|---|---|---:|
| AArch64 | `32c492690c02c59c166fc4c6c5849965f36b0df85647575a826fb964d2c2439c` | 7,537,351 |
| x86_64 | `339d88e7c058337093ee5393e002f9079b29ccfcb0e7ef918891fe9040173c0c` | 7,533,318 |

Each supporting pack contains 59 base kernels, 36 variants, and 95 P-kernel
plane applications. Candidate payload growth against the private-fork pack is
2.649137% for AArch64 and 2.694938% for x86_64, inside the 3% threshold. Final
candidate packs carrying the final post-repin fingerprint and refreshed
release copies will replace these before the acceptance campaign. All four
candidate/release and AArch64/x86_64 bundle/metadata pairs must be regenerated:
the current binaries predate the kernel-pack
`native_build_inputs_sha256` field, while the final metadata requires exact
producer compiler/source/native identities and the direct
`build_contract.symjit_source` tree/patch closure. A temporary unit-test shim
which projects those future values onto the stale binary fixtures will be
removed immediately after the real regenerated assets are copied into the
tracked stores.

An earlier supporting AArch64 pack check found an absolute numerical difference
of `1.7053025658242404e-13` for a result near 142.1, within
`rtol=1e-12`, `atol=1e-15`. That check consumed an earlier candidate bundle
rather than either exact tracked bundle in the tables above. Earlier
three-platform portability and release workflows also passed at synthetic
revision `a344ef55…`; both results are supporting rather than final-pack or
final-tree evidence.

## Installation and dependency provenance

The superseded pre-hardening clean candidate passed both contributor
installation modes under the 30 GiB watchdog:

| Install | Exit | Peak guard | Log SHA-256 |
|---|---:|---:|---|
| fresh cache | 0 | 18.946 GiB | `4d778a372e9896e17c8eb0acf84cbfa0ce12d1c5e9a40fa3101f4938628ad8d6` |
| repeat | 0 | 3.376 GiB | `80b914f8df0e5b9eaa0b52ea3639c794196aa4086b354334587642636da71d34` |

Both supporting runs authenticated the then-current release and candidate
dependency projections, materialized the immutable SymJIT archive, verified
pristine and post-patch trees, applied the single ordered patch automatically,
built Symbolica against the same checkout, and built/installed candidate
pyAmpliCol with the superseded fingerprint `a38cc68d0520`. Fresh and repeat
installs for the final patch digest, configured tree, and fingerprint are
deferred to the combined release-readiness pass.

The final authenticated `cargo tree` capture and tracked-tree scans will be
recorded after the final snapshot. Current scans contain no private-fork URL,
branch, or revision in the tracked tree and no retired ABI in active
source/policy/artifact roots. Immutable JSON captures under
`docs/performance_reports/**/results/` retain their historical runtime ABI
metadata. The x86 performance-runtime reconstruction helper likewise embeds an
authenticated SymJIT 2.21.1 baseline identity so the candidate workflow can
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

The authenticated alternating campaign uses the exact process
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
generation results, and payload/load/RSS results will be inserted here after
the explicit host-free handoff. Final evidence also records all eight
regenerated bundle/metadata hashes, the canonical native-build identity, the
direct SymJIT tree/patch contracts, fresh/repeat install watchdog reports,
patched upstream-kernel and generated-fixture logs, all nineteen required
fixture markers, and a fail-closed manifest/overlay preflight binding all five
generated fixtures and the staged candidate build to the exact candidate source
revision, native-build-input digest, full candidate version/fingerprint, and
sole `d d~ > z g g g g g g` process with external PDGs
`[1,-1,23,21,21,21,21,21,21]`. It will also include authenticated
`cargo tree`, private-fork scans, and schema-7 built-in/UFO portability captures
with schema-6 capture-acceptance metadata.

### Runtime acceptance

Pending.

### Generation and resource acceptance

Pending.

## Final conclusion

Pending completion of the clean full gates and authenticated performance
campaign.
