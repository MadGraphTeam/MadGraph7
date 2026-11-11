---
title: "Release and Support"
nav_order: 5
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Release and Support

pyAmpliCol 1.0.0 is the software release accompanying the official arXiv
publication, *pyAmpliCol: fast tree-level matrix elements*. Binary wheels and a
source distribution are available on [PyPI](https://pypi.org/project/pyamplicol/).
This page records the supported release boundary and explains how to report a
problem.

## Current release boundary

Version `1.0.1` is represented by the immutable
[`v1.0.1` source snapshot](https://github.com/MadGraphTeam/pyamplicol/tree/v1.0.1),
the [GitHub release](https://github.com/MadGraphTeam/pyamplicol/releases/tag/v1.0.1),
and [PyPI release](https://pypi.org/project/pyamplicol/1.0.1/). Its validated
inventory is one source distribution and three `cp311-abi3` wheels:

- macOS 11 or newer on Apple silicon;
- macOS 11 or newer on x86-64;
- manylinux 2.28 on x86-64.

The [release-artifacts workflow](https://github.com/MadGraphTeam/pyamplicol/actions/workflows/release-artifacts.yml)
installs each wheel into a clean CPython 3.11 environment and exercises the
Python, C11, C++17, Fortran 2008, and Rust 2021 APIs. It also runs a CPython
3.14 abi3 smoke test, source preflight, and independent Fortran physics oracle.
Publication uploads these already validated files without rebuilding them.

Version 1.0.1 uses published Symbolica 3.0.0, SymJIT 2.26.4 and
ufo-model-loader 1.0.0, as recorded in the dependency lockfiles. No local
dependency source patches or unpublished wheels are needed.

## 1.0.3 release

**Changed default: a refused point no longer interrupts a run.** A one-loop
evaluation now returns, by default (`on_failure="best-effort"`; CLI
`--on-failure best-effort`; SDK `set_on_failure("best-effort")`), every point
of the batch: a point that no configured alternative accepts (the reduction
backend's own acceptance checks, `"auto"`'s per-point fallback, the last
rescue level) is evaluated again without the backends' own acceptance checks
(the `reduction_unchecked` machinery) and, where that yields a finite value,
returned with lane status `"best-effort"` (SDK `RUSTICOL_VIRTUAL_BEST_EFFORT`
= 5): never a success, with its refusal and an uncertainty (the call's own
stability estimate, else a rotation re-test of that point; if none can be
formed the record says so). A point with no value at all is a failed lane.
Nothing is printed, logged or raised for such points: the result carries it
as data (`result.statuses`, `result.warnings`, `result.has_warnings`; the
SDKs' statuses, `uncertainties` and the diagnostics' `"best_effort"` and
`"warnings"`), so Monte-Carlo logs cannot be swamped. Points the configured
alternatives accept are evaluated exactly as before (bitwise, same cost).
`on_failure="return"` keeps such points as failed lanes without computing a
best-effort value; `on_failure="raise"` is the previous behaviour (raise at
the first such point; the SDKs return `RUSTICOL_VIRTUAL_RUNTIME_ERROR`);
`result.emit_warnings()` turns a result's warnings into one filterable
`NumericalAccuracyWarning`. Failure reasons are bounded to one message per
reason class with the count of the others. Known, benign dependency log
events of `generate` and `check` (Symbolica's "Created infinity by raising 0
to the power of -1" from loop_qcd_qed_sm's unused `aEW0` conditional, the UFO
loader's exact float-to-rational notice) are returned in the result's
`warnings` instead of printed. See
[Python API](python-api.md#best-effort-values-of-refused-points),
[Command-line interface](command-line-interface.md) and
[Native APIs](native-apis.md#runtime-and-precision-boundary).

Version `1.0.3` adds loop-induced one-loop processes: processes without a
Born amplitude within their coupling-order limits (`g g > h`, `g g > z z`,
`h > g g`), served as |A_loop|^2 at full colour and evaluated as
`VirtualEvaluation` results with `contribution = "loop-induced"` (empty Born
fields, poles `"all"` or `"none"`, which vanish; zeros are reported within
their absolute estimate), see
[Configuration](configuration.md#processes-and-multiparticles) and the
[Python API](python-api.md#one-loop-virtuals-and-loop-induced-processes).
`process.loop_induced` routes them (`"include"` serves both kinds of a mixed
multiparticle request as separate contributions). Leading and next-to-leading
colour are refused for loop-induced processes. The native SDKs (C, C++,
Fortran, Rust, UMAMI) open loop-induced processes directly with empty Born
counts and the Python API's values, including the helicity-resolved and the
colour- and spin-correlated outputs (the correlated history `"born"` is
|A|^2 itself).

The one-loop numerators are reduced by one of two backends, selected by
`[loop] reduction` (the generation default) and `evaluation.reduction` (CLI
`--loop-reduction`, Python `reduction=`, the SDKs' `set_reduction`):
NinjaRS (integrand reduction) and TIRust (tensor-integral reduction, built
into the native extension by default). `"auto"`, the default, takes TIRust
for loop-induced and NinjaRS for Born-interfered processes; on TIRust a point
it refuses or fails is evaluated again on NinjaRS and recorded per point. In
double-double and arbitrary precision each point this fallback serves is
guarded by a rotation re-test of the promoted, improved point. The guard's
estimate is an estimate, not a bound: strongly suppressed helicity rows near
exactly forward kinematics can be served beyond their own 1e-5 (see
[Python API](python-api.md), "Near-forward accuracy"). An explicitly
requested backend that cannot serve a call is refused, never switched; every
lane records the requested and effective backend and the reason. TIRust's
target accuracy is `evaluation.reduction_target` (default 1e-5 relative per
point). `evaluation.reduction_unchecked = true` (Python
`reduction_unchecked=`, CLI `--reduction-unchecked`) is a timing mode
without the backends' acceptance and refusal checks and without an accuracy
certificate; it is off by default, so the backends' own checks stay on.

The input kinematics of double-double and arbitrary-precision evaluations are
improved to exact energy-momentum conservation and exact on-shell external
momenta in the working arithmetic with MadLoop's algorithms
(`evaluation.kinematics_improvement`: `"z-shift"`, the default, MadLoop's
variant 2 with its fallback to variant 1, finished by a frame-independent
projection only when the produced point is not exact; `"rescaling"`,
`"projection"` or `"off"`). Binary64 evaluations are improved only with
`kinematics_improvement_binary64 = true` (default false). Points whose beams
lie along the z axis keep MadLoop's values bit for bit. Leave the
improvement on above binary64: switched off there, near-forward points can be
served far off without a flag. See
[Configuration](configuration.md#kinematics-improvement-of-the-input-points).

Binary64 stability checks (rotations, the reduction-direction and rescaled
checks, the cross-backend check) are available in the Python API and every
native SDK (`rusticol_virtual_set_stability_checks`), off by default; the
recommended checks are one rotation with the reduction-direction and
rescaled checks. The Python API's stability rescue is a stack of
(precision, backend) levels; for loop-induced processes its default runs
TIRust then NinjaRS in binary64 before double-double and arbitrary precision.
Full-colour and next-to-leading-colour Born-interfered virtuals take a
run-time helicity selection (`helicities=`) in the Python API and the native
SDKs.

`pyamplicol check` runs the pole (Catani; finiteness for loop-induced
processes), Ward and Goldstone checks of a one-loop run card or artifact,
regenerated in memory, escalating undecided rows to double-double and
arbitrary precision. Ward rows of a request with electroweak loop
corrections and an on-shell zero-width external massive quark read "not
applicable": the identity of that check is not guaranteed there. See
[Command-line interface](command-line-interface.md#check). Four one-loop
examples ship with the package (`loop_uux_ttxgg`, `loop_uux_ddxepem`,
`loop_gg_zgg` and `loop_gg_epemmpmm`, the last two loop-induced), see
[Examples](../../examples/README.md#one-loop-examples) and
[Build a wheel from a checkout and run the loop examples](installation.md#build-a-wheel-from-a-checkout-and-run-the-loop-examples).

In 1.0.2, storing a one-loop request whose Born couplings, as its scaleless
companion evaluates them, read a model helper function (e.g. `u u~ > d d~ g`
with QCD corrections and explicit coupling-order caps that keep both Born
orders, QCD^3 and QCD QED^2) failed with `Could not create evaluator: function
value_N(..) is not in a parameter map`, so nothing was stored and no value was
wrong; such requests now store.

Version `1.0.3` uses the Python dependencies of 1.0.2 (Symbolica 3.0.3
from PyPI, SymJIT 2.28.3, ufo-model-loader 1.0.1). The native code builds
Symbolica at the head of symbolica-dev/symbolica#163 (`dfe6bd97`), NinjaRS
and oneloopmaster at newer commits of their `main` branches, and TIRust
([huashengshao/TIRust](https://github.com/huashengshao/TIRust)) at a pinned
commit, linked by the default cargo feature `tir`; see
[dependencies/README.md](../../dependencies/README.md). With Python Symbolica
3.0.3 from PyPI, results of the Python API can depend on what else was
generated in the same Python process (about 1e-30 relative normally, amplified
at nearly degenerate 2 -> 2 kinematics): generate each artifact in a fresh
process where that matters. A Symbolica Community build with
symbolica-dev/symbolica#163, or a Symbolica release that contains it, is not
affected. Process artifacts load only in the version that wrote them:
artifacts written by 1.0.2 are refused by 1.0.3, regenerate them. The model
compiler version is unchanged (17), so prepared `.pyamplicol-model` bundles
written by 1.0.2 still load.

Version `1.0.2` was not published on PyPI and has no tag; its content is part
of 1.0.3.

## 1.0.2 release

Version `1.0.2` adds one-loop virtual matrix elements on the v2 engine:
QCD corrections on MG5_aMC's `loop_sm`, and QCD, electroweak and joint
corrections on the shipped `loop_qcd_qed_sm` and `loop_qcd_qed_sm_Gmu`
models, with finite parts and poles per squared coupling order, in binary64,
double-double or arbitrary precision. They load through `Runtime.load` and the
C, C++, Fortran and Rust SDKs. See
[One-loop QCD on loop_sm](loop-sm-and-madloop.md),
[One-loop QCD+QED](electroweak-and-madloop.md) and
[virtual artifacts](runtime-and-selectors.md#virtual-one-loop-artifacts).

The built-in Standard Model now equals the UFO `sm` (`no_masses`
restriction) at tree level. Processes without coloured legs (`e+ e- > a a`,
`e+ e- > mu+ mu-`) were too small by 1/(8 π α)² ≈ 27.8; massive vector and
Higgs initial states are averaged over their own spin states; the recurrence
mode counted every W W photon and W W Z vertex twice; and the lepton and
neutrino currents, the electroweak exchange between quark lines and the
H H V V and four-Higgs contacts were incomplete. Artifacts written by 1.0.1
are refused by 1.0.2, whose runtime admits only artifacts of its own package
version: regenerate them. A 1.0.2 artifact generated with a built-in SM
source from before these fixes (a development build) loads with a warning
and keeps the old values. See
[Built-in Standard Model](models-and-processes.md#built-in-standard-model).

Four-quark HEFT trees on `built-in-sm-heft` and on UFO HEFT models were too
large, because the H g g vertex did not subtract the U(1) part of its
internal gluons: 9/8 times for `q q~ > H q' q'~` and its crossings, other
factors with identical quark flavours or more gluons (1.36 for
`u u~ > H d d~ g`); fixed in
[#40](https://github.com/MadGraphTeam/pyamplicol/pull/40).

Processes with one initial particle (1 -> n decays, `t > b w+`) are served
at tree level on UFO models and on the built-in SM (`z > e+ e-`), and at one
loop on the loop models; decay chains and loop-induced decays are refused.
Mechanical hard-coded limits were lifted: the loop rank is the reduction
backend's; the colour-FFT degree, grouping factors and flavour mappings,
charges (now exact rationals) and full-colour N_c powers are uncapped; and
the native C++ compile has no default timeout. The remaining tuning bounds are
settings
(`color.adjoint_dense_max_gluons`, `color.fused_selector_max_sectors`,
`generation.validation.max_resolved_components`).

Version `1.0.2` uses Python Symbolica 3.0.3 from PyPI (or a Symbolica
Community build of symbolica-dev's `community` branch with
symbolica-dev/symbolica#163), and the native code builds Symbolica from that
branch and pull request at a pinned revision. It also uses SymJIT 2.28.3,
ufo-model-loader 1.0.1, and NinjaRS and oneloopmaster from their `main`
branches for the one-loop reduction. Tree and one-loop process artifacts
load only in the version that wrote them, and the model compiler version is
now 17: regenerate artifacts, compiled models and prepared
`.pyamplicol-model` bundles written by an earlier version.

## 1.0.1 release

Version `1.0.1` adds process grouping for multiprocess outputs. Subprocesses
with equal matrix elements are recorded in a new `process-groups.json`
artifact payload, with their leg permutation, rational factor, and charge
conjugation. `--flavor-blind` generates one representative per flavour
class, optionally including charge conjugation, and every other flavour
remains selectable from Python, the CLI, and the native SDKs, which route it
to its representative. `--analyze-process-groups numerical` and
`pyamplicol group-processes` find further relations numerically.
`pyamplicol inspect` and the generation summary display the groups, and UMAMI
exports flavour-blind members through their representative. UFO multiparticle
expansion now enumerates multisets and drops candidates that violate a charge
conserved by every vertex, which renumbers the process IDs of compiled-model
multiprocess requests. Artifacts written by 1.0.1 declare the new payload and
require a 1.0.1 runtime. See [Process Grouping](process-grouping.md).

## 1.0.0 release

Version `1.0.0` preserves exact model constants, colour weights, and
normalization factors through higher-precision evaluation. It also corrects
native evaluation and retains the exact next-to-leading-colour generation
shortcut. The release keeps the same supported platforms and Python versions.
Tensor algebra and prepared models have been updated for the published
Symbolica 3.0.0 API, including exact complex constants in the SymJIT bridge.
Equivalent exact and floating-point coefficients are again recognized when
selecting optimized recurrence kernels, including in the bundled SM and HEFT models.
Batched recurrence evaluations now fill the already allocated SIMD-aligned
workspace, avoiding unnecessarily small batches at higher multiplicities.
Completed on-the-fly caches can be saved and restored through Python and every
native SDK, without reconstructing the retained currents. See
[saving an OTF warm cache](runtime-and-selectors.md#saving-an-otf-warm-cache).
Symbolica's package license removes the need for a personal key for pyAmpliCol
use; personal-license and restricted-mode fallbacks remain available.
Compiled JIT O2 generation selects compressed code by default.

## 0.2.0 release

Version `0.2.0` adds exact symmetric-group FFT full-colour contraction,
helicity-parametric recurrence and on-the-fly scheduling, built-in SM+HEFT and
trusted UFO HEFT support, and the resumable fixed-helicity/helicity-sum FFT
profiling workflow with its final comparison reports.
See the runnable [scalar HEFT workflow](models-and-processes.md#built-in-scalar-heft-model)
and the [FullColor FFT profiling guide](fullcolor-fft-profiling.md).

## 0.1.4 release

Version `0.1.4` adds on-the-fly generation and its explicit one-point
`warm_up(...)` lifecycle, while retaining compiled, eager, and recurrence
generation. It also ships progress-aware warm-up entry points for the Python,
C, C++, Fortran, and Rust APIs and the packaged `p p > Z j j` OTF example.

## Install a release

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
```

Binary wheels include the Rust runtime and native SDK. Wheel users do not need
a Rust compiler. A C, C++, Fortran, or Rust compiler is required only when
compiling a consumer in that language against the included SDK.

Verify the environment with:

```console
pyamplicol doctor
pyamplicol self-test
```

Then run the primary example from an editable copy:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
pyamplicol evaluate_total.toml
```

See [Installation](installation.md) and [Quick Start](quick-start.md).

## What a release contains

The release set consists of:

- one source distribution;
- supported-platform `cp311-abi3` wheels;
- the Python extension and target-specific static Rusticol SDK in each wheel;
- packaged model resources, examples, profiling-campaign templates, and user
  documentation required by installed workflows.

Source installation may work elsewhere, but a platform is not advertised as
supported until its installed Python and native API matrix has passed.

Rendered performance reports are not PyPI wheel payloads or release-CI
benchmarks; the one-loop benchmark report lives under
[docs/performance_reports/loop](../performance_reports/loop/README.md).

## How artifacts are validated and published

The authoritative release workflow builds and tests the source distribution
and platform wheels. Deployment tests install each wheel into a clean
environment and exercise the installed Python runtime and native SDK consumers.

Publication then selects that already validated artifact run and uploads its
exact files through PyPI Trusted Publishing. Upload does not rebuild packages.
This keeps the tested and published bytes identical without adding a second
release build.

GitHub workflow history is available under
[Actions](https://github.com/MadGraphTeam/pyamplicol/actions), and tagged source
snapshots under [Releases](https://github.com/MadGraphTeam/pyamplicol/releases).

## Compatibility promises

- The public Python API is versioned and documented in the repository's API
  contract.
- The native public boundary is C ABI version 1; C++, Fortran, and safe Rust
  wrappers sit on that ABI.
- Run cards use schema version 1 and reject unknown fields.
- Current process artifacts use schema version 3 and the current runtime
  identity/ABI contract.
- Internal generated-artifact formats are not automatically migrated. An old
  artifact that lacks the current contract fails with regeneration guidance.

Regenerate process artifacts with the installed version when upgrading across
an artifact-contract change. Do not edit a manifest or executable payload by
hand.

## Supported and intentionally separate components

- pyAmpliCol has no LHAPDF dependency.
- Symbolica is required for model compilation, generation, and Python exact
  paths; the default compatible f64 JIT runtime executes through SymJIT without
  importing Symbolica.
- Original AmpliCol is optional campaign/developer comparison infrastructure,
  not an installed-package runtime dependency.
- Source contributor builds and release builds have separate dependency
  boundaries. Contributor candidate wheels are deliberately non-publishable.

See [Symbolica and Licensing](symbolica-and-licensing.md),
[Profiling Campaigns](profiling-campaigns.md), and
[Artifacts and Portability](artifacts-and-portability.md).

## Report a problem

Use [GitHub Issues](https://github.com/MadGraphTeam/pyamplicol/issues) for a
reproducible bug, compatibility failure, or documentation gap. Search existing
issues first.

Include the smallest information that distinguishes the failure:

1. exact pyAmpliCol version and installation method;
2. operating system, architecture, and Python version;
3. the command or short Python snippet;
4. the complete error message;
5. whether the model is built-in, JSON, compiled, prepared, or trusted UFO;
6. execution mode, backend, color accuracy, and process expression;
7. output from `pyamplicol doctor` when installation or SDK discovery is
   involved.

Useful read-only captures are:

```console
pyamplicol doctor --json > doctor.json
pyamplicol inspect ARTIFACT --json > inspect.json
pyamplicol config resolve RUN_CARD.toml --json > config.json
```

Review these files before attaching them. Do not publish private model data,
license keys, cluster paths, credentials, or an artifact you are not permitted
to redistribute.

For a generated native-driver failure, also include:

```console
rusticol-config --version
rusticol-config --target
rusticol-config --json
```

For a numerical report, provide the smallest kinematic point and parameter card
that reproduces it, the selected stable ID or process expression, precision,
and selectors. State whether the optimized total and explicit resolved sum
disagree.

## Triage checklist

Before filing, try the focused check matching the symptom:

| Symptom | First check |
| --- | --- |
| CLI/import failure | `pyamplicol doctor` |
| Installation/runtime smoke | `pyamplicol self-test` |
| Native headers or linker flags missing | Activate the environment, then `rusticol-config --json` |
| Artifact target/ABI rejection | `pyamplicol inspect ARTIFACT` and regenerate with the current release |
| Process expression not found | Inspect stable IDs; see [Process Selection and Permutations](process-selection-and-permutations.md) |
| Selector rejected | Inspect the selected process's helicity/color capabilities |
| Raw UFO/JSON fails in recurrence/eager/OTF | Prepare a compatible `.pyamplicol-model` bundle, or choose compiled mode |
| High-precision Python failure | Confirm Symbolica availability and license; f64 is `--precision 16` |
| Profiling result is surprising | Keep process, layout, selectors, batch, and precision fixed; see [Profiling and Benchmarking](profiling-and-benchmarking.md) |

More symptom-specific help is in [Troubleshooting](troubleshooting.md).

## Feature requests

A useful feature request explains:

- the physics or deployment workflow;
- why existing model, generation, runtime, or selector APIs are insufficient;
- the smallest representative process/model;
- expected public behavior, including error behavior;
- platform or performance constraints.

Avoid attaching a complete private project when a small public model/process
can demonstrate the need.

## Contributing a fix

Contributor setup is documented in [Installation](installation.md).
The repository policy favors focused tests for a demonstrated failure plus one
authoritative CI path. Changes should not add redundant manifests, repeated
hashing, dependency revalidation, or unrelated release ceremony.

Public API changes require corresponding documentation and typing coverage.
Runtime or artifact changes require proportionate native tests; documentation-
only changes do not justify rebuilding the platform release matrix.

## Security and artifact trust

Process artifacts are executable inputs. JIT applications are lowered to
native code at load time, and C++/ASM artifacts may contain native libraries.
Manifest validation establishes internal consistency and path confinement, not
publisher identity. Generate artifacts yourself or obtain them through a
trusted channel.

Do not post a suspected security vulnerability publicly before maintainers can
assess it. Use the repository's available private security-reporting channel if
enabled; otherwise contact the maintainers through the organization channels
without including exploit details in a public issue.

## Links

- [PyPI](https://pypi.org/project/pyamplicol/)
- [GitHub releases](https://github.com/MadGraphTeam/pyamplicol/releases)
- [Issue tracker](https://github.com/MadGraphTeam/pyamplicol/issues)
- [License](https://github.com/MadGraphTeam/pyamplicol/blob/main/LICENSE)
- [Third-party notices](https://github.com/MadGraphTeam/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md)
