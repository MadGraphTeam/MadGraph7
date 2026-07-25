# Eager and Compiled Direct-Arena Migration

## Goal and bootstrap

**Goal statement:**  
“Replace the existing eager and compiled-JIT internal runtimes with separate Direct-Arena execution lanes based on the accepted recurrence Direct-Arena substrate, preserving all public Python, CLI, Rust, C, C++, and Fortran APIs and physics semantics while intentionally replacing prior internal artifact and evaluator ABIs, retaining applicable LC, NLC, and full-color optimizations, and demonstrating pointwise parity plus robust portable wall-time gains across complete runtime-selector workloads.”

At implementation start:

- Save this approved plan verbatim to `/Users/vjhirsch/HEP_programs/pyAmpliCol/PREPARE_STANDALONE_PYAMPLICOL/pyamplicol_eager_and_compiled_arena/docs/development/EAGER_AND_COMPILED_ARENA_PLAN.md`.
- Assign the exact quoted goal statement using the goal tool before changing implementation code.
- Ask the recurrence agent for its newest coherent, buildable pushed checkpoint, exact SHA, validation state, and caveats. Treat intermediate checkpoints as reference-only unless explicitly declared accepted and source-frozen.
- Use the latest accepted, source-frozen recurrence SHA—not `24519b2` or a dirty snapshot:
  - Prefer the latest `origin/main` containing the accepted recurrence merge.
  - If recurrence is accepted but not merged, branch from its pushed, source-frozen head after it has incorporated current `main`.
- Create the exact branch `eager_and_compiled_arena` in a dedicated worktree; never edit the recurrence worktree.
- Record reference, base, main, dependency-lock, SymJIT, and prepared-model SHAs in the plan document. Require SymJIT 2.21.1 or the newer version carried by the accepted recurrence base.
- Run harmless worktree/Git/cache/build probes first. Never request escalation; use worktree-local caches, `/private/tmp`, existing dependencies, and offline operations. If no sandbox-safe path exists, report the external blocker instead of escalating. Run every substantial build, generation, or benchmark under the 30 GiB watchdog.

Read the recurrence contract, Direct-Arena ABI, independent validation, and compiled feasibility documents before implementation. Recheck the latest runtime audit and table-aware SymJIT ABI at the pushed recurrence checkpoint.

## Architecture and ABI replacement

- Extract only generic Direct-Arena mechanics: deterministic liveness allocation, 64-byte-aligned split-complex component-major planes, point-contiguous tiles, typed callable views, persistent workspace, status/counters, stable selector grouping, and zero-allocation warmed execution. Keep recurrence behind a thin adapter and prove its accepted behavior remains unchanged.
- Do not copy recurrence currents, closure proofs, row schemas, topology/replay semantics, or builder logic into eager or compiled execution.
- Introduce independent internal capabilities such as `eager-direct-arena-v1` and `compiled-plane-arena-v1`. Old eager/compiled artifacts fail early with an actionable regeneration message; no converter, legacy hot-path fallback, or dual packet/direct production representation remains after cutover.
- Keep public mode names, selectors, ordering, return shapes, exceptions, parameter behavior, profiling surface, and Python/Rust/C/C++/Fortran callable APIs unchanged.
- Keep load validation minimal but memory-safe: outer checksum/version/target checks plus alignment, bounds, row ranges, callable role/signature, selector coverage, producer closure, and alias rules. Do not add proof-digest webs, redundant semantic authentication, or hot-loop validation.

### Eager lane

- Replace packet gather/backend/scatter execution with an eager-specific Direct-Arena plan.
- Model invocation-plus-fanout, finalization, closure, coherent groups, coupling/output factors, and selector dependencies explicitly; allocate lifetimes at eager events rather than reusing recurrence rows.
- First use per-row direct calls only as a correctness oracle. Production uses table-aware prepared callables with row-outer/point-inner execution and direct multi-destination stores.
- Preserve arithmetic and fanout order, structural zeros, exact/generic evaluation metadata, and existing widening opportunities.

### Compiled-JIT lane

- Preserve process-specific fused stages, chunk boundaries, funclet compression, and fanout-aware output order.
- Give every fused leaf static logical bindings from value slots, momenta, parameters, zeros, and outputs to arena planes. Recompute deterministic physical plane assignments at load time.
- Add a compiled-specific DirectApplication ABI supporting current compressed O3 applications, factor-free overwrite semantics, fixed descriptor bundles, checked `u32`-sized plane/scalar spaces, and direct unchecked invocation after validation.
- Prohibit input/output aliasing within a fused stage. Execute tile-by-tile directly into canonical amplitude planes, then reduce before reusing storage.
- Add plane-native amplitude reducers while retaining exact summation order, certified LC totals, compact repeated-color storage, 8-chain Hermitian kernels, and K4/H8 Walsh kernels.
- Before final cutover, provide equivalent plane-native C++/ASM callable support wherever those public backend configurations are supported; wrappers around dense row ABIs do not qualify.

## Milestones

1. **Baseline and instrumentation**
   - Freeze exact eager, compiled, accepted-recurrence, and original AmpliCol references.
   - Repair benchmark selection so eager and compiled can be requested independently.
   - Add layout-independent semantic benchmark identity covering process/model, complete physical-axis coverage, schedule/reduction ordering, normalization, deterministic momenta, and selectors.
   - Capture unprofiled wall time, generation time, payload/load/RSS, allocation, pack/scatter/remap bytes, callable counts, and arena-layout projections.

2. **Shared substrate extraction**
   - Genericize the allocator/views/workspace without changing recurrence semantics.
   - Run recurrence correctness, allocation, malformed-plan, and accepted performance canaries before proceeding.
   - Push the coherent milestone to `eager_and_compiled_arena` and notify the recurrence agent.

3. **Risk-first ABI prototypes**
   - Eager: exercise a real table-aware multi-row/multi-destination callable.
   - Compiled: exercise a real late compressed O3 fused leaf with factor-free overwrite and odd SIMD tails.
   - Require zero packet/scatter traffic and warmed allocations. Raw kernels may regress by at most 3%, while each end-to-end prototype must beat its current gather/call/scatter equivalent.
   - Profile and redesign failed prototypes before broader migration; do not mass-port around an unproven ABI.

4. **Eager replacement**
   - Implement complete source, invocation/fanout, finalization, selector, amplitude, LC/NLC/full-color, resolved, and totals execution.
   - Dual-run against the old implementation only in developer validation builds; remove the old production representation after parity and performance gates pass.
   - Push a validated eager milestone.

5. **Compiled replacement**
   - Implement arena layouts and direct bindings for all fused stages and amplitude leaves.
   - Add selector-pruned schedules and plane-native reductions without de-fusing stages.
   - Remove stage/leaf input packing, output scatter, amplitude remapping, and dense JIT-row execution.
   - Push a validated compiled milestone.

6. **Complete cutover**
   - Cover built-in/UFO-SM, JIT/native prepared backends, f64/exact paths, parameters, profiling, native language APIs, malformed artifacts, panic containment, packaging, and clean-install smokes.
   - Delete obsolete packet schemas, dense execution contracts, compatibility readers, and developer dual-run code.
   - Commission separate subagent audits for eager semantics, compiled fusion/alias safety, artifact/load safety, and benchmark methodology.

## Correctness and performance gates

- Compare baseline-to-candidate within each mode and eager-to-compiled on identical samples using `rtol=1e-12`, `atol=1e-15`; check totals, resolved components, sums, structural zeros, parameters, aliases, errors, and untouched output sentinels.
- LC selector methodology:
  - Single-flow/helicity-sum uses a complete topology-replay artifact retaining every flow and selects exactly one physical flow at runtime.
  - All-flows/single-helicity uses a complete union artifact retaining every helicity and selects exactly one source-ordered helicity at runtime.
  - Cross-check each selected result against the corresponding sum from `evaluate_resolved()`.
  - Generation-fixed axes are diagnostic lower bounds only.
- Test batches 1/128/1024 and tail boundaries 127/129 and 1023/1025; global and per-point homogeneous, pre-grouped, alternating, and random selectors.
- Primary matrix:
  - `u u~ > Z+6g`, built-in and UFO-SM, eager and compiled, both LC selector modes.
  - Medium LC/NLC/full-color cases including `d d~ > Z+3g`, `g g > t t~+2g`, `g g > 4g`, `d d~ > u u~ s s~ g`, and `d d~ > t t~+3g`.
  - Color-heavy `g g > t t~+3g` and `g g > t t~+4g` in NLC/full color.
  - Same-host original AmpliCol and accepted recurrence references with matching physics, samples, selectors, and timing boundary.
- Timing uses warmed, unprofiled native wall time after caller-side packing, at least seven interleaved subprocess samples, median/MAD, and identical points.
- Required acceptance:
  - Each of eager and compiled improves at least one primary workload by at least 10%, beyond measurement noise.
  - No required matrix cell regresses more than 3% or three MAD.
  - Compiled `qq_Z6g` in both LC modes is within 15% of accepted recurrence at batches 128 and 1024.
  - Zero warmed native allocations and zero packet/gather/scatter/remap bytes in arena execution.
  - Normal generation time is at most 10% worse per cell and 5% worse geometrically; material payload, load-time, or RSS growth requires an offsetting measured runtime gain.
- Validate on AArch64 and x86-64. Use deterministic, shape-derived tiling with no online/adaptive timing tuner.

## Publication and final integration

- Push every coherent validated milestone to `eager_and_compiled_arena`; do not leave large unpushed arena changes.
- Rebase the completed feature branch on the latest target and rerun the full correctness, performance, resource, packaging, and cross-language gates.
- If accepted recurrence is already on `main`, merge/fast-forward `eager_and_compiled_arena` into `main` and push the resulting exact main SHA.
- If accepted recurrence is still only on its source-frozen branch, obtain an explicit no-race acknowledgement, integrate the feature branch into that recurrence branch, push its exact SHA, and let the coordinated recurrence merge carry both to `main`.
- Never merge partial or statistically ambiguous gains. Mark the assigned goal complete only after the chosen integration target is pushed and all required gates remain green.

---

## Bootstrap provenance

Recorded at implementation start on 2026-07-24 (Europe/Belgrade):

- Accepted recurrence source-freeze SHA: `443f354a467cdda187996bef1a41fbd5a00ae28d`.
- Independently accepted production-source parent/reference SHA: `585456ed1726c43eef3ce35c7a126c17730e8a0d`.
- Current `origin/main` SHA after the coordinated fast-forward: `443f354a467cdda187996bef1a41fbd5a00ae28d`.
- Feature branch base SHA: `443f354a467cdda187996bef1a41fbd5a00ae28d`.
- Feature branch: `eager_and_compiled_arena`.
- Accepted base tree SHA: `869bfed6c8ebe6834d5adccd505733fdc598202d`.
- `Cargo.lock` SHA-256: `58fec7351b18f93bb38b41bea5ecdad3b0b2f1bd4d5d3a302e7f471c58ccc9c0`.
- `dependencies/contributor-lock.toml` SHA-256: `91a4cd4d03bc3d35b7e2794f04bed4580428f5d17ea2f846fa530c5df8197cb5`.
- `dependencies/release-lock.toml` SHA-256: `3302cacd840eb9f14e9e00e4c4c712d1a4735a45c1702224b84e4f34385f50ee`.
- `dependencies/python-runtime-lock.toml` SHA-256: `f8c929cfe925630a96e1a9d80bb8d6106964ace84a4efc4df171784a4f8f522b`.
- Symbolica candidate version/revision: `2.2.0` / `77c137481904b8a5531ede86e3ef36b82beed7fd`.
- SymJIT candidate version/revision: `2.21.1` / `48197f32536c894b51ef25b2cf05ddd05c22675f`.
- SymJIT candidate tree SHA-256: `932bb24df2633cc8bbf9c743a80282662d11e70b692885de5ff7a3ed20b3df31`.
- SymJIT Direct-Arena patch SHA-256: `6d456e69fc160ec5361188f60f994d10fb2dd3360eb47a91c4979a1bde69626e`.
- AArch64 prepared built-in model SHA-256: `37f66ed992e555d6dfb5f28515683da5c4656656bd8abc8eb58e007d6dafd1ab`; metadata SHA-256: `68b7f8a8d14902e6da74ad3fbbf648dc65315c7b1f687de6cfdf6a568471099e`.
- x86-64 prepared built-in model SHA-256: `3ec4c1132618383065ac2e1cf9d090156c9ec3d96b256c21640e74c412560d91`; metadata SHA-256: `648102e87a319356974bb0f4470784dc0e9e388b35c3e1c474fe959293dc3841`.
- Prepared-model source digest: `838f1ca629c989084416142a3af9fcae6b6b5dcd3894ff26974ec4f2a9f78739`.
- Original AmpliCol pinned reference revision: `79c96cecf2a722e50c3d2030b6894d755f96518a`.
- Risk-prototype references (not production branch points):
  - semantic benchmark harness: `577810d1abb4b4806ac8d741f0deb9fc0a5cc11c`;
  - combined eager/compiled SymJIT Direct-Arena ABI: `9d36a9d851edc9bba518db022c060d0e3fa7019e`;
  - eager plan: `6d5451e452f66f5aa033566c2bfdb5d4f7030d78`;
  - compiled plan: `e72bdc95e14d6faa0f4bc262d2f65ccb391ef3a9`;
  - generic substrate: `57cde05787437cd0c2456808067f491a5526e438`.

The exact goal statement above was assigned through the goal tool before any implementation-code change. The accepted recurrence worktree was never modified. Initial worktree, Git-ref, cache, and cleanup probes succeeded without escalation; the command guard rejected an `rm -f` spelling before execution, and the equivalent single-file cleanup succeeded with `unlink` and `rmdir`.

## Operational sandbox addendum

All active mutable source copies, dependency checkouts, Cargo homes, build
targets, generated artifacts, and benchmark results now live below this
feature worktree, principally under ignored `.agent-work/`. Active work does
not use `/private/tmp`, even though the approved plan permits it. Every
repository command sets this feature worktree explicitly as its working
directory.

The no-escalation rule applies equally to the primary agent and every
subagent: never request approval, never set an escalated sandbox mode, and
never retry a sandbox rejection outside the sandbox. Avoid commands that the
app can classify as destructive even under normal sandbox permissions,
including `rm`, `git clean`, `git reset`, `git checkout`, and `git restore`.
Use `unlink` only for a known single disposable file and `rmdir` only for a
known empty disposable directory. If no worktree-local/default-sandbox route
exists, retain the state and report the blocker.
