# UMAMI SDK for generated pyAmpliCol artifacts

## 1. Goal and agreed scope

Implement a standalone UMAMI provider in generated process artifacts, with model-independent grouping, MadSpace-compatible metadata, a shared library, a linked demonstration executable, documentation, and an end-to-end integration example.

Upon implementation approval:

1. Create and switch to the feature branch **`umami`**.
2. Save this approved plan verbatim as **`UMAMI_PLAN.md`**, including the original request below.
3. Set the implementation objective as the active goal.
4. Commit and push at the milestones below, preserving unrelated files. Do not merge or release.

Agreed choices:

- Support **LC**, **NLC**, and **full-colour** artifacts, with their distinct physical meanings.
- Produce separate providers where processes cannot share UMAMI’s fixed metadata.
- Use **`metadata.json`**, correcting the original filename.
- Include optimized grouping now, but derive every amplitude equivalence from the generated computation and model data. No Standard-Model-specific rules, particle-category shortcuts, or numerical coincidences.
- Run a bounded, genuine **MadSpace partonic integration** by default; document optional larger hadronic runs.
- The examples must **closely adapt the supplied AmpliCol Python scripts**, not replace them with newly designed demonstrations having merely similar outcomes.

## 2. Generated SDK and runtime interface

### Files and build

Extend the existing artifact API-bundle machinery to emit `API/umami`, containing:

- The current UMAMI public header and a C implementation.
- Generated provider tables.
- A Makefile producing shared libraries and a standalone C driver.
- `metadata.json` and concise build/use instructions.

Use the [current UMAMI contract](http://docs.madgraph.org/latest/madspace/umami-api.html) and authoritative MadGraph7 header, not AmpliCol’s older enum definitions. Preserve upstream licensing notices.

For a single compatible provider, produce `libumami.so` and a directly consumable `metadata.json`. For multiple providers, produce separately named libraries and per-provider metadata; the root `metadata.json` identifies the available providers. The example must select a provider explicitly rather than combine incompatible libraries.

Group providers by compatible external multiplicity, incoming-particle count, colour accuracy and external-mass semantics. Keep overlapping alternative process selections separate rather than implicitly summing them.

Follow the existing SDK convention of placing compiled products outside immutable artifact payloads, with an overridable `BUILD_DIR`.

### Linking

Build the provider with the headers and native library reported by:

```sh
rusticol-config --cflags
rusticol-config --libs
```

Support Linux and macOS, retaining the requested `.so` filename on macOS. Export only UMAMI entry points, preventing statically linked Rusticol symbols from interfering with another provider.

Link the demonstration executable against the generated UMAMI library—not a second copy of Rusticol—and configure the appropriate relative runtime-library search path. Verify the actual shared-library build before making any native-build changes.

### API behaviour

- `umami_initialize` interprets `param_card_path` as an **absolute artifact-directory path**, loads its saved default parameters, and validates compatibility with the generated provider.
- Metadata needed before initialization is compiled into the provider; it must not depend on a previously initialized handle.
- Handles own independent mutable runtimes, parameter state and scratch storage. Distinct handles may be used concurrently.
- Translate UMAMI’s column-major, strided momenta to Rusticol’s layout, respecting `count`, `stride` and `offset`.
- Support the reference workflow’s momenta, channel/flavour selection, event-wise αs, matrix-element output and informational random-helicity selection.
- Default matrix-element output remains the exact helicity sum over the artifact’s available coverage. Drawing a helicity label does not turn this into a sampled estimator.
- Advertise capabilities accurately. Unsupported diagram, GPU and other optional operations return the appropriate status, rather than fabricated results.
- Preserve restricted-helicity coverage explicitly; never advertise a restricted artifact as an unpolarized result.

Forward model-parameter updates through Rusticol’s existing validated setter. Add one small native/C getter for **current** parameter values, including refreshed derived values, instead of introducing a C JSON parser or duplicating parameter formulas.

Identify an optional αs input from explicit model metadata or its standardized UFO parameter declaration; do not guess from particle names or coupling powers. If no unambiguous mapping exists, report that capability as unsupported. Event-wise overrides must refresh dependent parameters through the existing setter and leave persistent handle parameters unchanged after the call.

## 3. First-principles grouping and metadata

### Generation option

Add:

```text
--umami-grouping exact|flavour_blind_observables|none
```

with the equivalent configuration/Python setting:

```text
generation.umami_grouping
```

| Mode | Meaning |
|---|---|
| `exact` — default | Reuse proven representative computations while retaining explicitly mapped physical contributions. No assumptions about cuts, observables or PDF symmetry. |
| `flavour_blind_observables` | Additionally compress integration orbits under an explicit assumption of compatible, final-state permutation/flavour-blind cuts and observables. |
| `none` | Export every available physical process, colour and helicity configuration without UMAMI grouping. Existing internal evaluator optimizations remain unchanged. |

These are SDK/export settings. They must not invalidate native builds or alter numerical evaluator identity.

### Equivalence construction

Implement a generation-side structural comparison pass, separate from existing runtime reuse rules:

- Compare actual source, kernel, propagator and coherent-root expressions, including momentum mappings, helicities, fermion signs and normalization.
- Resolve internal coupling definitions while retaining symbolic dependence on independently mutable parameters. Equal defaults do not establish equivalence.
- Reuse existing certified kernel symmetries, colour-orbit maps and genuine helicity-value relations without broadening their claims.
- Recognize exact equality and already-certified phase relations initially. Leave unproved candidates separate.
- Avoid a general graph-automorphism search or expansion of complete amplitudes. Reuse canonical local expressions and shared DAG structure.
- For FC/NLC, require equivalence of both the amplitude computation and its colour contraction. Compare structured FFT plans without constructing a dense Gram matrix.
- Never use the historical built-in SM process enumerator or numerical amplitude-ratio tests as the grouping authority.

Numerical comparisons validate this implementation; they do not establish its equivalences.

### Physical and integration semantics

LC exposes additive physical colour contributions. FC/NLC exposes contracted colour results, not invented independent colour flows.

In `exact` mode, preserve the momentum/selector mapping for each contribution and apply it when evaluating the representative. Different physical configurations remain identifiable so consumers can apply their own flavour-dependent cuts and weights.

In `flavour_blind_observables` mode:

- Collapse only algebraically established orbits with compatible phase-space measures.
- Record the covered members, permutations and multiplicities.
- Keep incoming legs oriented and retain the distinct incoming PDGs needed for PDF weights.
- Never assume equal PDFs, beam-swap symmetry, equal masses, or equal amplitudes merely because the user selected this mode.
- Put integration multiplicities in metadata; do not silently multiply the UMAMI matrix-element output.
- Document that compressed representatives need not reproduce the uncompressed pointwise sum, although their integrals agree under the declared assumptions.

Do not introduce multiple maps for the same contribution and then sum them without a partition of unity. Initially select one deterministic complete phase-space ordering for each contribution/orbit; share compatible maps without duplicating their integrands.

### Metadata construction

Derive metadata and the C lookup tables from one internal representation so indices cannot drift.

Retain the reference format’s channel, process, PDG, helicity, colour-order and colour-flow tables. Add only the information required for provider selection, grouping mode and explicit member mappings.

- Use physical incoming/outgoing PDGs and complete external-leg permutations.
- Derive LC colour tags from model colour structure and generation sectors, including quark chains and colour singlets.
- For contracted colour, use an explicitly absent flow reference; do not invent LHE colour tags.
- Obtain masses and parameter defaults from the compiled model.
- Account for normalization exactly once: Rusticol already includes incoming averaging and identical-final-state factors.
- Derive any additional grouping weights analytically.
- Treat aliases as mappings, not automatically as extra integrable contributions.
- Use neutral pyAmpliCol producer information rather than copying AmpliCol-specific XML bookkeeping.

The supplied local JSON is a shape reference only: its `pdg_ids` table is inconsistent and must not become a golden fixture.

## 4. Documentation, faithful examples, validation and milestones

### Documentation

Add a dedicated UMAMI documentation page and links from the native-API documentation. Explain:

- Building the `.so` using `rusticol-config`.
- Linking an executable and loading the library from MadSpace.
- The artifact-path interpretation of `param_card_path`.
- Momentum layout, indices, parameter updates and normalization.
- LC versus contracted-colour semantics.
- All three grouping modes and their integration assumptions.
- Provider selection, unavailable capabilities and restricted coverage.

### Required contents of `umami_example/`

Commit these files:

- **`README.md`**: prerequisites, commands, expected outputs, upstream source links and an explanation of adaptations.
- **`run.py`**: generate the pyAmpliCol artifacts, build the UMAMI provider and standalone driver, then invoke both adapted example scripts.
- **`test_umami.py`**: a close adaptation of the supplied [AmpliCol `test_umami.py`](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_umami.py).
- **`test_integrate.py`**: a close adaptation of the supplied [AmpliCol `test_integrate.py`](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_integrate.py).

Start from those upstream scripts and preserve their recognizable structure and workflow wherever compatible. Retain attribution and applicable licensing. Do not substitute a ctypes-only example, a custom toy integrator, or a script that merely generates phase-space points.

`test_umami.py` must retain the reference’s UMAMI/MadSpace loading and batched evaluation approach, including its relevant input/output buffers. Add deterministic checks against native pyAmpliCol values rather than merely printing results.

**`test_integrate.py` must actually perform numerical integration through the generated `.so`.** Preserve the reference’s essential sequence:

1. Read channel/flavour metadata.
2. Construct the MadSpace phase-space mappings and cuts.
3. Load the UMAMI provider through MadSpace’s matrix-element interface.
4. Build the differential cross section, including the appropriate flux and weights.
5. Run the reference-style integration machinery, using a bounded CPU VEGAS configuration by default.
6. Combine independent contributions correctly and report an integral with its statistical uncertainty.

Retain relevant controls for larger runs and the optional PDF/hadronic and MadNIS workflows. Smaller default sample counts are a workload adjustment, not a replacement of the original integration method.

Limit departures from upstream to those needed for:

- pyAmpliCol artifact paths, provider selection and generated metadata.
- Current MadSpace API signatures.
- Correct normalization, grouping and channel weights.
- Model-derived masses and parameters instead of hard-coded particle tables.
- A working no-PDF default and manageable test statistics.
- Numerical assertions, error handling and reproducible execution.

Document these differences in the README. In particular, correct the old script’s no-PDF αs handling and unsafe assumptions about summing alternative channels.

Use `g g > g g g` as the main demonstration, covering LC and full colour. Keep integration dependencies optional and outside pyAmpliCol’s core dependencies. Default to no PDF downloads.

Acceptance requires running **both adapted scripts**, with `test_integrate.py` completing a genuine integration through UMAMI. A successful library-loading test alone is insufficient.

### Focused tests

Add coverage for:

- Artifact layout, metadata references, complete permutations and packaged templates.
- Shared-library loading, standalone-driver linkage and relocation to paths containing spaces.
- Native/UMAMI numerical parity, normalization and LC channel reconstruction.
- Nonzero offsets, oversized strides and untouched output sentinels.
- Independent handles, parameter mutation, derived values and event-wise αs.
- Invalid inputs, unsupported capabilities and initialization/cleanup failures.
- Quark chains, colour singlets, mixed process sets and restricted helicities.
- All three grouping modes.
- A small non-SM test model with exactly equivalent flavour expressions.
- Rejection of false equivalences caused by equal default couplings, different propagators, chiral/Yukawa structure, fermion-exchange contributions or different colour contractions.
- Reconstruction of accepted groups after parameter changes.
- Symmetric integration agreement for the aggressive mode, and asymmetric-cut correctness for `exact`/`none`.

Require useful compression on a proven positive case, but do not require matching AmpliCol’s grouping counts or guarantee speedups.

### Implementation milestones

1. **Export foundation:** save the plan, add configuration, provider/metadata generation and focused exporter tests; commit and push.
2. **Working provider:** implement structural grouping, runtime adapter, parameter getter, shared-library build and numerical tests; commit and push.
3. **Complete demonstration:** finish documentation and the closely adapted example scripts, run both demonstrations, run the existing aggregate local gate `just source-gate` once after focused checks, resolve failures, and commit/push the completed feature.

Reuse existing builds and validation results. Extend the existing native-SDK CI coverage for Linux shared-library loading rather than adding a new platform matrix. Inspect the resulting CI status without duplicating completed local campaigns.

Completion requires passing local gates, successful standalone and MadSpace demonstrations—including an actual completed integration—correct grouping/normalization tests, and all intended changes pushed to `umami`.

## 5. Original request — verbatim

```text
Ok I would like you work off a new feature branch called `umami` which aims to implement a new sort of SDK in the artifacts output by `pyAmpliCol`, in the folder `./API/umami` of the process output (so alongside the other C++, fortran, c, rust, python etc...).

The difference is that contrary to the other SDK, the makefile contain therein should be creating a `.so` file which implements the matrix element C API described here:

[http://docs.madgraph.org/latest/madspace/umami-api.html](http://docs.madgraph.org/latest/madspace/umami-api.html)

with a concrete example of how it was implemented for the old `amplicol` code in:
[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/umami.c](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/umami.c)
(and
[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/umami_impl.f03](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/umami_impl.f03)
)

You have to make sure to add a page in the docs that coves this new umami api implementation, and explains how this .so file should be linked, into an executable. The folder `API/umami` should also include a standalone driver that can be compiled against the `.so` produced and do a basic call to the umami API also to illustrate how it should be properly linked using the `rusticol-config` binary.

Note that you'll need to have the artifact path available when calling `umami_initialize` and for this you are allowed to abuse the API and instead abuse the argument `char const *param_card_path` to instead point to the absolute path of the artifact to load, and use the default parameter values specified therein to get the default model values.

Moreover, you should also produce a special json metada file in `API/umami/metada.json` where you can find here in `./tmp/madspace_converted.json` an example of what it should look like, and more in general in:

[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/amplitude_library.f03#L146](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/amplitude_library.f03#L146 "https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/amplitude_library.f03#L146") 

[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/madspace_convert.py](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/madspace_convert.py "https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/madspace_convert.py")

how it was systematically built for amplicol. Of course here we should be able to easily derive it from our existing pyAmpliCol artifact metadata.

Once all of this is in place, we should be able to test that it works by using this new umami api output within:

[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_umami.py](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_umami.py)

(only tests proper loading of the `.so`, can be slightly adapted of course for current environment and pyAmpliCol umami impl, but should remain the same in spirit)

and

[https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_integrate.py](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_integrate.py)

(also can be slightly amended but must remain the same in spirit).

Also add for now a `./umami_example` local folder into this `umami` feature branch, with a README.md and a python script that generates the appropriate pyamplicol output and then use the new umami api to run within it both the equivalent of `test_umami.py` and `test_integrate.py` to demonstrate that this umami api works well.
It should be pushed for now in this `umami` feature branch (which you should create and switch into as soon as the implementation of the plan start).

Add a goal statement to the plan, as well as this prompt VERBATIM, and the rest of what you'll specify further for the plan now.
Upon starting implementation of the plan, write it verbatim in `UMAMI_PLAN.md` and set it as your goal.
You should also commit+push to the remote at periodic implementation milestone, add a couple corresponding tests, and at the end verify that all local test gates pass.


Do a deep investigation of all the material above, and ask question for anything unclear, and I'll then give you the green light to implement it.
```

