# Migrate pyAmpliCol’s Arena Runtime to SymJIT 2.22.0

## Summary

- Replace the private SymJIT fork with [`siravan/symjit-crate`](https://github.com/siravan/symjit-crate) 2.22.0, pinned to the current immutable revision [`4e288ce5f3132b05e2a81eb6452c011b9e2bb936`](https://github.com/siravan/symjit-crate/commit/4e288ce5f3132b05e2a81eb6452c011b9e2bb936), not crates.io or a moving branch.
- Reimplement the compiled, recurrence, and eager JIT arena adapters around standard SymJIT P-kernels while retaining pyAmpliCol’s arena allocation, scheduling, factors, fanout, and operation policies.
- Preserve every Python/CLI/public C, C++, Fortran, and Rust interface and all numerical/selector behavior. Internal artifact ABIs may change and old generated artifacts will fail closed with a regeneration message.
- Start patchless. Add only objectively necessary, generic SymJIT patches, limited to a sound raw plane-descriptor ABI and/or mixed plane/scalar translation.

## Implementation

1. **Record the plan, goal, and baseline**
   - Before code changes, write the text inside this plan block verbatim to `docs/development/arena/SYMJIT_CRATE_2_22_ARENA_MIGRATION_PLAN.md`.
   - Immediately create the active goal: “Execute the saved SymJIT 2.22.0 arena-migration plan while preserving pyAmpliCol behavior and meeting all correctness, allocation, and performance gates.”
   - Keep all temporary files, dependency caches, build products, and benchmark captures inside the workspace using workspace-local `TMPDIR`, `CARGO_HOME`, Python/pip caches, and `.artifacts/symjit-2.22-migration/`. Never request command escalation.
   - Capture a reproducible pre-change build and benchmark baseline from current source revision `172e58fd33a3c65563866c50cfbb5e1ddcd7b302`, retaining hashes, wheels, prepared packs, dependency metadata, environment, and raw samples.

2. **Cut over dependency management**
   - Change Cargo, contributor/release locks, installer, release checks, licenses, and provenance to SymJIT 2.22.0 at the pinned Git revision. `just dev-install` must download the checksummed archive into `dependencies/checkouts/symjit`, verify its pristine tree and `rlib` manifest, and path-patch both pyAmpliCol and Symbolica to that checkout.
   - Initially retain `patches = []`. If a permitted upstream patch becomes necessary, restore the repository’s ordered patch machinery with revision matching, per-patch SHA-256, forward/reverse applicability checks, pristine-source and post-patch tree hashes, and automatic application by `just dev-install`.
   - Require `cargo tree` and repository scans to show only SymJIT 2.22.0 from `siravan/symjit-crate`; remove every private-fork URL, revision, ABI identity, and obsolete fork-patch reference.

3. **Build a shared standard P-kernel adapter**
   - Add an internal bridge from Symbolica’s structured `Evaluator.get_instructions()` representation to `symjit::Compiler`, compiling a complex P-kernel with `set_direct_arena(true)`. Generate and persist this plane-oriented application alongside any ordinary B-kernel still required by non-arena paths.
   - Construct `Config` explicitly rather than using `Config::default()`, preventing ambient `symjit.toml` influence. Set compiler target, complex/SIMD flags, optimization, compression, threading, fast-math, and arena mode deterministically; ensure SIMD preparation occurs before sealing.
   - Introduce a pyAmpliCol-owned P-kernel binding ABI recording split-complex plane order, inputs, outputs, scalar sources, optimization settings, target requirements, and source digest.
   - Keep the `Applet` alive, cold-bind stable descriptors, and invoke `scalar_kernel()`/`simd_kernel()` directly. SIMD indices are block indices; handle unaligned heads and tails with the scalar kernel. Support AArch64 two-lane, x86 AVX four-lane, and scalar fallback without hot allocation.
   - In the patchless candidate, represent point-independent couplings, literals, and model parameters through shared persistent broadcast planes: constants fill once, model-parameter planes refresh only when parameters change, and structural zeros reuse the existing zero plane.

4. **Replace the three fork-specific execution lanes**
   - **Compiled JIT:** replace `DirectApplication` lowering with direct P-kernel bindings to the fused O0–O3 arena. Outputs remain direct, disjoint overwrites; scalar bindings use shared broadcast planes or, if enabled later, scalar P-kernel parameters.
   - **Recurrence JIT:** preserve the current arena, schedules, topology-replay/all-flow-union behavior, factors, and role ordering. Use direct outputs only for proven disjoint identity overwrites. For accumulation, non-identity factors, or before-write aliasing, evaluate once into persistent scratch output planes and apply the complex scale plus overwrite/add policy in allocation-free SIMD Rust loops.
   - **Eager JIT:** move DirectTable row/attachment validation and orchestration into Rusticol. Preserve invocation order, cross-row dependencies, fanout, factors, overwrite/add semantics, and hazard rejection. Use direct output for a single safe identity-overwrite attachment and persistent scratch plus vectorized fanout otherwise. Native C++/ASM table paths remain unchanged.
   - Preserve zero warmed allocation and zero boundary pack/gather/scatter/remap traffic. Track internal scratch traffic separately so it cannot be mistaken for boundary traffic.

5. **Apply narrowly gated upstream patches only if required**
   - First run a P-kernel contract probe covering duplicate input planes, output/input aliasing, overwrite, accumulation, complex scaling, scalar/SIMD tails, and sentinels. If SymJIT’s `Vec<&mut [f64]>` descriptor contract cannot express the required aliases soundly, add one generic patch exposing a stable raw-pointer or `#[repr(C)]` plane descriptor and corresponding unsafe callable. Keep the kernel body unchanged.
   - Run the compiled/recurrence performance and resource gates with the patchless broadcast/scratch implementation. Only if a failure is attributed to plane-broadcast or separate-epilogue overhead, add a second generic patch exposing partitioned Symbolica-instruction translation: selected inputs become P-kernel state planes and the remainder use the existing scalar `params` argument. Then generate pyAmpliCol recurrence variants that algebraically encode overwrite or `destination + factor × result`, relying on the normal P-kernel prologue snapshot and epilogue.
   - Do not patch SymJIT with pyAmpliCol operations, schedules, factor catalogs, DirectApplication, or DirectTable concepts. Each upstream patch must have standalone SymJIT tests, generic documentation, no pyAmpliCol naming, and a recorded rationale suitable for submission upstream.
   - Do not relax acceptance thresholds to avoid a patch. Once all lanes pass, remove the old fork-only adapters and regenerate built-in prepared bundles.

## Interfaces and Compatibility

- Keep `generate`, `load`, `evaluate`, `evaluate_resolved`, selectors, model-parameter updates, prepared-model workflows, evaluator modes, LC-flow layouts, warnings, cards, and CLIs unchanged.
- Keep the public C ABI v1 and generated C++, Fortran, and Rust SDK interfaces unchanged.
- Preserve component ordering, crossing, normalization, exact fallbacks, built-in/UFO parity, complete flow/helicity axes, and numerical results.
- Bump the internal compiled-plane, recurrence-direct, eager-plane-table, and prepared-pack binding ABIs. Reject old private-fork artifacts deterministically and instruct users to regenerate them.

## Verification and Acceptance

- Add focused contract tests for real/complex scalar and SIMD P-kernels at lengths `1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025`, including duplicate inputs, aliases, factors, fanout, parameter refresh, sentinel preservation, status propagation, and panic containment.
- Run the relevant Rust allocation/arena/recurrence/eager/compiled tests, Python integration and selector tests, prepared-model and exact-fallback tests, multilanguage API tests, dependency-policy tests, then `just rust-test`, `just source-gate`, `just dev-test`, candidate smoke tests, and the existing x86 workflow.
- Verify numerical agreement for totals and resolved components with `rtol=1e-12`, `atol=1e-15`; require `evaluate()` to equal the selected sum of `evaluate_resolved()`.
- Verify a fresh-cache `just dev-install`, a repeat install, authenticated dependency trees, regenerated prepared packs, zero warmed heap allocations, and zero boundary pack/gather/scatter bytes.
- Extend the benchmark harness to accept the exact process `d d~ > z + 6*g` (canonical expanded form `d d~ > Z g g g g g g`) without weakening the existing `u u~` acceptance route.
- Compare baseline and candidate on the same host with two warmups and seven alternating independent subprocess pairs, retaining at least seven five-second warmed samples per cell under the 30-GiB watchdog:

  | Layout | Modes | Batches |
  |---|---|---|
  | Topology replay, selected runtime flow, helicity sum | recurrence JIT O2; compiled JIT O3 | 1, 128, 1024 |
  | All-flow union, all flows, nonzero alternating runtime helicity | recurrence JIT O2; compiled JIT O3 | 1, 128, 1024 |

- A runtime cell passes only when the candidate median is at most `1.03 ×` baseline and at most `baseline median + 3 × baseline raw MAD`. At batches 128/1024, compiled/recurrence paired ratio median plus three raw MAD must remain `≤1.15`.
- Process generation may regress by at most 10% per cell and 5% geometrically. Payload, cold-load, or RSS growth above 3% is accepted only with a runtime gain of at least 10% beyond noise.
- Exercise odd tails `127/129` and `1023/1025` for numerical/allocation parity, retain the existing built-in/UFO three-lane `u u~` portability campaign, and run an eager diagnostic campaign under the same regression rule.
- Publish a final migration report containing source/dependency revisions, whether either generic patch was needed, correctness/allocation evidence, raw benchmark files, baseline/candidate statistics, generated-artifact changes, and confirmation that no private-fork dependency remains.
