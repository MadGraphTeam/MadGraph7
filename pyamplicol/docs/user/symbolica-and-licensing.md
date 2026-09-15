---
title: "Symbolica and Licensing"
nav_order: 1
parent: "Architecture Overview"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Symbolica and Licensing

pyAmpliCol uses Symbolica for symbolic model compilation and process
generation, but its default f64 deployment path is deliberately separated from
Symbolica's runtime. This page explains the technical boundary and summarizes
where to read the authoritative license terms.

> This page is a technical guide, not legal advice. Upstream license terms and
> eligibility rules control. The complete notices distributed with pyAmpliCol
> are in
> [`THIRD_PARTY_NOTICES.md`](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md)
> and the repository's `licenses/` directory.

## Package license at startup

Importing pyAmpliCol imports Symbolica and calls `symbolica.set_library_key()`
with the key issued for the `pyamplicol` package, before any symbolic work.
Users do not need a personal Symbolica license for calls made by pyAmpliCol,
including parallel generation and evaluator optimization. The key also
registers when a spawned Python worker imports pyAmpliCol.

```python
import sys
import pyamplicol

print("symbolica" in sys.modules)  # True

from pyamplicol.licensing import detect_symbolica_license
print(detect_symbolica_license(suggest=False).licensed)  # True
```

The key licenses calls within `pyamplicol`; it does not license unrelated
Symbolica code in the user's program. Personal license environment variables
are left unchanged. Model compilation and generation tooling remain lazy.

The installed Symbolica must provide `set_library_key()`. Startup reports an
error if the installed version cannot register the package key.

## Which operations need Symbolica?

| Operation | Uses Symbolica? | Notes |
| --- | --- | --- |
| Import `pyamplicol` | Yes | Registers the package key; public exports remain lazy. |
| Inspect an artifact | Startup only | Reads metadata and indexes only. |
| Ordinary Python f64 evaluation (`precision=16`) | Startup only | Runs through Rusticol and the artifact's native evaluator. |
| C11/C++17/Fortran 2008/Rust 2021 runtime | No | Ordinary and opt-in correlated APIs are f64-only. |
| Direct JIT f64 load | Startup only in Python | Uses the separate MIT-licensed SymJIT runtime. |
| Compatible C++/ASM evaluator load | Startup only in Python | Uses the artifact's target-native library. |
| Compile a JSON/UFO model | Yes | Symbolic model construction. |
| Generate a process artifact | Yes | Symbolic DAG/recurrence construction and evaluator production. |
| Ordinary Python precision other than 16 | Yes | Lazily loads retained Symbolica evaluator state when supported. |
| Python `evaluate_correlated(...)` | Yes | Uses retained Symbolica evaluator state at every precision, including 16. |

The absence of a Symbolica runtime dependency for f64 evaluation does not
change the terms governing Symbolica use during generation.

## Direct-JIT f64 runtime

The default JIT backend embeds a direct SymJIT application in the schema-v3
artifact. Rusticol loads and lowers that application to native code without:

- performing Symbolica computations (Python package startup registers the key);
- reading `SYMBOLICA_LICENSE`;
- applying Symbolica's generation-time worker clamp;
- linking the arbitrary-precision Symbolica/Rug/Malachite closure into the
  wheel's f64 native SDK.

This is the deployment path shared by ordinary Python evaluation at precision
16 and the C11, C++17, Fortran 2008, and Rust 2021 APIs.

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
values = runtime.evaluate(momenta, precision=16)
```

Independent runtime handles can execute concurrently even when their artifact
was generated under restricted Symbolica conditions. Do not call one mutable
handle concurrently because its parameter and warning state is handle-local.

See [Runtime and Selectors](runtime-and-selectors.md) and [Native APIs](native-apis.md).

## Generation with a valid license

pyAmpliCol calls `symbolica.is_licensed()` from inside the package to select
its generation resources. This detects the registered package key even when
`SYMBOLICA_LICENSE` is absent. A call from unrelated user code can still return
`False`, because the package key is scoped to pyAmpliCol.

With a valid license, automatic resource settings share one affinity-aware CPU
budget:

```toml
[generation]
workers = "auto"

[evaluator.optimization]
cores = "auto"
```

Concurrent process builds receive disjoint evaluator budgets. Explicit
requests are clamped when their product exceeds the available budget, and the
requested/effective difference is recorded in generation provenance.

## Requesting a personal license for other Symbolica work

For personal licenses covering Symbolica use outside pyAmpliCol, visit
[symbolica.io/license](https://symbolica.io/license). No personal key is
needed for work covered by pyAmpliCol's embedded package key.

## Python exact precision

Python may request positive decimal precision other than 16:

```python
exact = runtime.evaluate(momenta, precision=80)
```

This loads retained Symbolica evaluator state and requires the applicable
Symbolica package/runtime authorization. Decimal-string inputs can preserve
their source digits through the generated Python standalone driver:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --kinematics my_sample_point.json \
  --precision 80
```

A binary64 value converted to higher precision gains trailing arithmetic
digits, not additional physical input information. Use decimal strings in the
kinematics JSON when input precision matters.

Native C, C++, Fortran, and Rust callers reject precision other than 16.

The Python [Born Correlations API](../correlators.md) uses the same retained
Symbolica-state machinery through a separate exact executor at **all**
requested precisions, including 16. The C/C++/Fortran/Rust correlated SDK calls
instead use a separate RustiCol binary64 executor without importing or
licensing Symbolica at runtime. Both contract the selected LC/NLC/full colour
matrices prepared during generation, which still requires Symbolica.

## SymJIT is a separate dependency

SymJIT is the native JIT runtime used by Symbolica-generated applications. It
is distributed under the MIT License, not under the Symbolica proprietary
license. Release dependency metadata pins the official
[`siravan/symjit-crate`](https://github.com/siravan/symjit-crate) source and an
immutable revision.

SymJIT compression is opt-in. It shares repeated arithmetic sequences to
reduce generated code size. The extra calls can increase evaluation time, but
the smaller instruction footprint can make larger evaluators faster.
Neither setting is universally faster; use `profile` to compare them for your
workload. The default is `compress = false`. To enable it explicitly:

```toml
[evaluator.jit]
compress = true
```

The equivalent CLI option is:

```console
pyamplicol generate --card run.toml --jit-compress
```

Prepared model bundles bake their compression/backend choice into the kernel
pack, so an incompatible request is adjusted to the prepared pack and reported
in the effective configuration. To change that choice, regenerate the prepared
model with `--jit-compress` or `--no-jit-compress`.

## License and provenance map

| Component | Role | License/provenance boundary |
| --- | --- | --- |
| pyAmpliCol and project Rusticol source | Public package/runtime implementation | BSD Zero Clause License (`0BSD`). |
| Symbolica | Symbolic generation and Python exact runtime | Proprietary Symbolica Software License Agreement; project redistribution permission is documented separately. |
| SymJIT | Native JIT application runtime | MIT License. |
| UFO model loader | External UFO loading | MIT License. |
| Bundled SM/scalar/scalar-gravity assets | Example/model data | Asset-specific provenance and reproduced terms in `PROVENANCE.toml` and `licenses/`. |
| Optional original Fortran AmpliCol checkout | Independent campaign oracle | Not bundled in wheels or sdists; its upstream metadata/terms remain separate. |

The pyAmpliCol project has express authorization from the Symbolica licensor to
redistribute the Symbolica components required by pyAmpliCol's binary runtime.
That project-specific permission is not a general grant to redistribute
Symbolica separately. The embedded library key covers Symbolica calls within
pyAmpliCol; other Symbolica use requires its own applicable authorization.

Exact dependency versions and notices are included in release metadata and
[`THIRD_PARTY_NOTICES.md`](https://github.com/mg5amcnlo/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).

## Original AmpliCol is not a pyAmpliCol runtime dependency

The optional original Fortran AmpliCol checkout is used only as an independent
numerical/performance reference in [Profiling Campaigns](profiling-campaigns.md). It is not shipped
in pyAmpliCol wheels or source distributions and is unnecessary for ordinary
generation or evaluation.

pyAmpliCol has no LHAPDF dependency. The supported campaign comparison checkout
contains the profiling interface described in
[rikkert-frederix/AmpliCol PR #12](https://github.com/rikkert-frederix/AmpliCol/pull/12).

## Diagnose the active boundary

```console
pyamplicol doctor
```

The human output is colored and names Python, model assets, Rusticol extension,
native SDK, Symbolica license status, and available compiler tools separately.
Use JSON when attaching diagnostics to an issue:

```console
pyamplicol doctor --json
```

If ordinary or native correlated f64 evaluation fails, do not assume it is a Symbolica-license
problem: first run `pyamplicol self-test` and inspect the artifact target. If
generation, precision-80 evaluation, or Python correlated evaluation at any precision
fails, then inspect the Symbolica check. See
[Troubleshooting](troubleshooting.md).

## Related pages

- [Installation](installation.md) — wheel and source requirements.
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md) — where symbolic work occurs.
- [Artifacts and Portability](artifacts-and-portability.md) — what the resulting f64 artifact carries.
- [Born Correlations](../correlators.md) — Python exact and native binary64 correlation paths.
- [Release and Support](release-and-support.md) — published dependency and validation boundary.
