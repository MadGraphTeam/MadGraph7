# Standalone Release Milestones

This ledger is maintained by the main orchestrator. Agents report changes and
evidence; they do not commit, push, or update shared release metadata.

| Milestone | Owner | State | Acceptance evidence | Commit |
|---|---|---|---|---|
| Bootstrap and clean baselines | Main orchestrator | Complete | Plan preserved outside the nested repository; independent branch; 0BSD and fail-closed dependency/legal gates; canonical published Cargo lock separated from the ignored candidate lock; deterministic model assets and compact reference fixtures present; the complete source gate and isolated candidate deployment pass under the 30 GB watchdog | This commit |
| Mixed Maturin build and SDK | Main + Rust workers | In progress | Split core/Python/C API workspace, portable target contract, and SDK resources implemented; the fresh macOS arm64 candidate wheel passes SDK/archive audits, deterministic Python and C-API CycloneDX SBOM validation, a Symbolica-import-blocked direct-SymJIT physics self-test, and Python/C++17/Fortran deployment; the other release targets remain | Pending |
| Typed Python API and configuration | Turing + Franklin + Epicurus + Main | In progress | Immutable schema-v1 config, requested/effective provenance, quiet lazy root import, truthful public typing gate, centralized programmatic license clamps, complete LC generation coverage, and named process-set runtime identity implemented; final API review remains | Pending |
| Models, DAG, and schema v3 | Planck + Darwin + Maxwell + Main | In progress | Model assets, generic compilation, schema-v3 manifests, physics metadata, resolved reductions, full compiler-source cache fingerprinting, truthful validation samples/tolerances, and selector-free LC expressions implemented; seven installed-wheel built-in/external/scalar schema-v3 physics cases pass, while broader multiplicity coverage remains | Pending |
| Post-parity model independence | Main + future physics/model workers | Deferred until parity | After numerical, coverage, and performance parity is frozen, replace PDG/name/family branches in generic code with structural particle predicates; derive lowering from canonical UFO color/Lorentz tensors; validate relabeled-PDG and tensor-reordering invariance; retain any optimized built-in specialization only behind structural equivalence proofs | Pending |
| Examples and documentation | Python workers + Main | In progress | Installed examples, root API bundle, native runners, and clean N/A performance report implemented and visually reviewed; all 20 installed-wheel example tests pass, while final prose review remains | Pending |
| Deployment and release gates | Main + Ohm + Herschel + Leibniz + reviewers | In progress | Clean-overlay release/candidate Cargo execution, strict wheel/sdist inventory audits, immutable CI pins, guarded publishing, diagnostics, and self-test implemented; the exact macOS arm64 candidate passes isolated deployment and all 31 installed integration tests. Candidate sdist `ce662c926262...69f18d41` contains one portable saved-SymJIT-MIR source template; cross-target execution and release-mode artifact parity remain | Pending |

## Audit Ledger

| Audit | Agent | State | Integrated outcome |
|---|---|---|---|
| Requirements sentinel | Bacon | Complete | Canonical lowercase path, candidate/release split, API/config contracts, and source-port discipline recorded |
| Physics/model extraction | Huygens | Complete | Dual-SM preservation, exact port boundary, model determinism risks, schema-v2 rejection, physics invariants, and expanded oracle ladder recorded |
| Packaging/release | Aristotle | Complete | Read-only overlay build, pinned Maturin, abi3 wheel/SDK contract, candidate identity, sdist parity, native-link allowlists, and protected publishing recorded |
| Python/API implementation | Turing | Complete | Immutable schema-v1 config, precedence/clamps, typed services, CLI dispatch, scoped logging/progress, and root public exports reviewed and tested by Main |
| Release tooling implementation | Ohm | Complete | 44 focused release/backend compatibility tests; normalized sdist parity, isolated deployment, ABI/path audits, dry-run publishing, and protected Trusted Publishing workflow integrated | Pending |
| Model asset extraction | Planck | Complete | 66 payloads plus provenance manifest; deterministic scalar defaults; 16 loader-reproduced JSON files; six focused tests | Pending |
| Requirements re-audit | Bacon | Complete | No-go findings recorded; candidate gate, overlay isolation, metadata hooks, SDK validation, fixture provenance, schema constraints, and license coverage under main-orchestrator remediation | Pending |
| Rust license inventory | Curie | Complete | All 156 locked third-party crates inventoried by exact identity, source, checksum, and SPDX expression; Symbolica, SymJIT, and model notices covered by a deterministic release gate | Pending |
| Configuration provenance | Franklin | Complete | Requested and effective configurations, adjustment reasons, resolved TOML, and append validation flow through typed API and CLI paths | Pending |
| Sdist inventory and nested assets | Herschel | Complete | Recursive Maturin globs, dependency patches, required nested inventory, and real-sdist audit integrated | Pending |
| Canonical/candidate Cargo lock separation | Leibniz + Main | Complete | Root release lock contains registry packages only; ignored candidate lock and clean overlays carry local patched dependencies without mutating the canonical lock | Pending |
| Cross-language integration slice | Anscombe + Main | Complete | The installed macOS arm64 candidate wheel generated mixed LC, single-process LC, NLC, and full-colour artifacts; Python, C++17, and Fortran 2008 metadata, resolved components, parameter updates, explicit sums, and compatibility totals agree in all four integration cases | Pending |
| Target portability | Kepler + Main | Complete | Portable artifacts carry no native CPU requirements; native C++ artifacts record detected features; Rust and Python loaders reject incompatible targets before evaluator load; 43 focused tests passed | Pending |
| Static-LGPL release review | Galileo + Main | Complete | Shipped Python and C artifacts now use the direct-SymJIT f64 closure without Symbolica/Rug/GMP/Malachite; target Cargo closures and native wheel scans enforce the separation, while Symbolica remains lazy for generation and high precision | Pending |
| Requirements re-audit | Hume | Complete | Critical release, licensing, validation, cache, CI, physics, and documentation gaps recorded and assigned to the critical path | Pending |
| LC colour-coverage semantics | Lovelace + Wegener | Complete | Removed inert generation-time flow selectors and numerical current filters; generation always materializes complete LC coverage while evaluation and benchmarking retain runtime flow selectors; 42 focused tests passed in main review | Pending |
| Public typing gate | Epicurus + Main | Complete | Strict public-contract mypy, installed-style consumers, stub/export parity, and `just typing` pass | Pending |
| Direct SymJIT artifact contract | Nash | Complete | Identified self-contained SymJIT `Application::save` payload, empty-defun requirement, native/AArch64 layouts, capability dispatch, and exact Symbolica export API | Pending |
| Symbolica SymJIT exporter | Volta + Main | Complete | Managed patch exports a validated, empty-defun SymJIT application and architecture-specific f64 layout without compiling implicitly; patches 0001--0004 apply to the pinned candidate and 25 focused installer/release tests pass | Pending |
| Symbolica-free Rusticol f64 core | Rawls + Main | Complete | Rusticol uses Symbolica's own trusted `Application::load(...).seal()` mechanism through SymJIT, has no Symbolica dependency in the f64 extension/static SDK, rejects unsupported capabilities, and passes an installed physics self-test with Symbolica imports blocked | Pending |
| Python runtime feature boundary | Averroes + Main | Complete | The single f64-only `_rusticol` extension directly runs trusted SymJIT applications; non-f64 Python calls lazily use Symbolica's own persisted `Evaluator.load` path and replay the schema-v3 stage plan, avoiding a second Symbolica-linked extension | Pending |
| Runtime capability artifact metadata | Avicenna + Main | Complete | Schema-v3 records direct SymJIT, legacy Symbolica JIT, C++, and ASM capabilities from evaluator chunks through the root manifest; direct JIT retains optional precision fallback state, portable indirect translation is enforced, and 40 focused evaluator/schema tests pass | Pending |
| Generation concurrency and monitoring | Ptolemy + Main | Complete | One bounded executor now spans deterministic per-process DAG, warmup, schema, and evaluator phases; writes remain serialized, failures stop new submissions, progress is thread-safe, nested chunk workers stay at one, and 11 focused concurrency/progress tests pass | Pending |
| Concrete-process resource partitioning | Kuhn + Main | Complete | Restricted mode clamps to one worker and one Symbolica core before model work; licensed mode partitions one affinity-aware CPU budget only after multiparticle expansion reveals the concrete-process count, while preserving requested/effective provenance; 22 focused main-review tests pass | Pending |
| Release workflow completion | Copernicus + Main | Complete | Signed-tag chain now requires full typing, Python/Rust/physics/native/example/deployment gates before retained artifacts; publisher verifies every required job and uploads unchanged hashes through protected manual Trusted Publishing | Pending |
| Alias selector and permutation audit | Socrates + Locke + Singer + Main | Complete | Fixed non-self-inverse validation-point scattering, canonicalized contracted/singlet IDs, removed inert generation-time alias selectors, serialized alias expressions and external-PDG order, and remapped public particles, helicities, LC flows, representative IDs, reductions, selectors, and momenta through tested non-self-inverse three-cycles | Pending |
| SymJIT application trust boundary | Euler + Linnaeus + Main | Complete | Direct application bytes are trusted executable/compiler inputs. By product decision Rusticol follows Symbolica's existing SymJIT `Application::load` path; patch 0005 and the duplicate bounded decoder were rejected. Manifest hashing checks integrity, not authenticity, and documentation requires artifacts from a trusted source | Pending |
| SDK SBOM and release inventory | Lagrange + Main | Complete | The wheel carries Maturin's distribution SBOM plus a deterministic C-API CycloneDX SBOM exposed by `rusticol-config --sbom`; SDK metadata binds its hash, and source and isolated-deployment audits reject undeclared roots, legal files, schemas, resources, local paths, and unresolved Python dependencies | This commit |
| Selector capability semantics | Hooke + Main | Complete | LC advertises helicity and physical color-flow selectors; NLC/full advertise helicity only while retaining the singleton contracted-color output axis. Python, Rust metadata validation, schemas, high-precision fallbacks, and integration expectations agree | Pending |
| Trusted-loader requirements re-audit | Meitner + Main | Complete | Shipped direct f64 loading is the same `Config::default()` plus empty external-function map and `Application::load(...).seal()` sequence used by Symbolica. The installed-wheel deployment removes the license, blocks all Symbolica imports, and still passes the physics self-test | Pending |

## Agent Handoff

Every handoff records:

- Objective and exclusive ownership.
- Files changed and public behavior affected.
- Commands and tests run.
- Peak resource use for substantial work.
- Remaining risks and assumptions.
- Recommended integration order.

## Review Gates

Requirements, packaging, and physics reviewers independently inspect each
milestone. Critical findings must be resolved before the main orchestrator
commits that milestone.

## Current Critical Path

1. Broaden current-source model/physics/example/performance gates beyond the
   low-multiplicity installed LC/NLC/full cross-language slice.
2. Exercise the lazy Symbolica precision fallback and non-JIT evaluator
   artifacts without changing the direct SymJIT f64 loader or introducing a
   second SymJIT decoder.
3. Validate the same installed-wheel and static-SDK contract on macOS x86_64
   and manylinux x86_64, including host execution of the retargeted portable
   saved-SymJIT-MIR self-test fixture.
4. Resolve remaining alias/model metadata cases and
   complete the full release workflow test contract.
5. Freeze validated parity fixtures, then perform the dedicated model-independence
   hardening pass: remove unexplained PDG/name assumptions from generic modules,
   derive lowering from canonical model structure, and rerun the complete model
   ladder including relabeled-PDG and reordered-tensor invariance checks.
6. Complete release-mode sdist/wheel parity once the exact upstream dependency
   releases exist; candidate-mode source, wheel, SDK, legal, SBOM, and isolated
   deployment gates are validated in the bootstrap milestone.
