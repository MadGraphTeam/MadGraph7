---
title: "Release and Support"
nav_order: 4
parent: "Profiling and Benchmarking"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Release and Support

pyAmpliCol publishes binary wheels and a source distribution on
[PyPI](https://pypi.org/project/pyamplicol/). This page is the authoritative
record of the supported release boundary and explains how to report a problem.

## Current release boundary

Version `0.1.3` is represented by the immutable
[`v0.1.3` source snapshot](https://github.com/mg5amcnlo/pyamplicol/tree/v0.1.3)
and [PyPI release](https://pypi.org/project/pyamplicol/0.1.3/). Its validated
inventory is one source distribution and three `cp311-abi3` wheels:

- macOS 11 or newer on Apple silicon;
- macOS 11 or newer on x86-64;
- manylinux 2.28 on x86-64.

The [release-artifacts workflow](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/release-artifacts.yml)
installs each wheel into a clean CPython 3.11 environment and exercises the
Python, C11, C++17, Fortran 2008, and Rust 2021 APIs. It also runs a CPython
3.14 abi3 smoke test, source preflight, and independent Fortran physics oracle.
Publication uploads these already validated files without rebuilding them.

Release dependencies include Symbolica 2.2.0 and the official
[`siravan/symjit-crate`](https://github.com/siravan/symjit-crate) 2.22.0 at
immutable revision `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`.

## 0.1.4 release candidate

The current source tree is preparing version `0.1.4`. It is not yet a tagged
or published release. Its source distribution and platform wheels must first
complete the validated release-artifacts workflow; publication will then reuse
those exact files without rebuilding them. Until both validation and
publication complete, version `0.1.3` remains the current PyPI release and the
supported release boundary described above.

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

Rendered performance PDFs are repository documents, not PyPI wheel payloads or
release-CI benchmarks. Their links intentionally point to the current `main`
branch:

- [MacBook M3 report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/macbook_M3_pyAmpliCol.pdf)
- [AMD EPYC report](https://github.com/mg5amcnlo/pyamplicol/blob/main/docs/performance_reports/EPYC_pyAmpliCol.pdf)

## How artifacts are validated and published

The authoritative release workflow builds and tests the source distribution
and platform wheels. Deployment tests install each wheel into a clean
environment and exercise the installed Python runtime and native SDK consumers.

Publication then selects that already validated artifact run and uploads its
exact files through PyPI Trusted Publishing. Upload does not rebuild packages.
This keeps the tested and published bytes identical without adding a second
release build.

GitHub workflow history is available under
[Actions](https://github.com/mg5amcnlo/pyamplicol/actions), and tagged source
snapshots under [Releases](https://github.com/mg5amcnlo/pyamplicol/releases).

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

Use [GitHub Issues](https://github.com/mg5amcnlo/pyamplicol/issues) for a
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
pyamplicol doctor --format json > doctor.json
pyamplicol inspect ARTIFACT --format json > inspect.json
pyamplicol config resolve RUN_CARD.toml --format json > config.json
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
- [GitHub releases](https://github.com/mg5amcnlo/pyamplicol/releases)
- [Issue tracker](https://github.com/mg5amcnlo/pyamplicol/issues)
- [License](https://github.com/mg5amcnlo/pyamplicol/blob/main/LICENSE)
- [Third-party notices](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md)
