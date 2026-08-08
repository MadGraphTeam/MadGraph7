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

## Lazy import boundary

Importing pyAmpliCol does not import Symbolica:

```python
import sys
import pyamplicol

print("symbolica" in sys.modules)  # False

from pyamplicol import Runtime
print("symbolica" in sys.modules)  # False
```

The public package, configuration types, and runtime class remain lightweight.
Symbolica is loaded only when a requested operation needs symbolic model
compilation, process generation, or retained exact evaluator state.

## Which operations need Symbolica?

| Operation | Uses Symbolica? | Notes |
| --- | --- | --- |
| Import `pyamplicol` | No | Public exports are lazy. |
| Inspect an artifact | No | Reads metadata and indexes only. |
| Python f64 runtime (`precision=16`) | No | Runs through Rusticol and the artifact's native evaluator. |
| C11/C++17/Fortran 2008/Rust 2021 runtime | No | Native APIs are f64-only. |
| Direct JIT f64 load | No | Uses the separate MIT-licensed SymJIT runtime. |
| Compatible C++/ASM evaluator load | No | Uses the artifact's target-native library. |
| Compile a JSON/UFO model | Yes | Symbolic model construction. |
| Generate a process artifact | Yes | Symbolic DAG/recurrence construction and evaluator production. |
| Python precision other than 16 | Yes | Lazily loads retained Symbolica evaluator state. |

The absence of a Symbolica runtime dependency for f64 evaluation does not
change the terms governing Symbolica use during generation.

## Direct-JIT f64 runtime

The default JIT backend embeds a direct SymJIT application in the schema-v3
artifact. Rusticol loads and lowers that application to native code without:

- importing the Symbolica Python package;
- reading `SYMBOLICA_LICENSE`;
- applying Symbolica's generation-time worker clamp;
- linking the arbitrary-precision Symbolica/Rug/Malachite closure into the
  wheel's f64 native SDK.

This is the deployment path shared by Python at precision 16 and the C11,
C++17, Fortran 2008, and Rust 2021 APIs.

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

At the first operation that needs Symbolica, pyAmpliCol calls
`symbolica.is_licensed()`. Merely defining a `SYMBOLICA_LICENSE` environment
variable is not treated as proof that it is valid.

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

## Restricted generation

Without a valid license, pyAmpliCol offers a license-request reminder. Users
whose work is eligible under Symbolica's current terms can continue in
restricted mode. Symbolica describes that mode as non-commercial, one instance,
and one core per device; commercial work requires the applicable professional
license path.

pyAmpliCol enforces a technical clamp of one process worker and one Symbolica
core and records why the requested configuration changed. That clamp does not
grant eligibility or replace upstream terms.

Suppress the request reminder and Symbolica startup banner when appropriate:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --no-symbolica-suggestion
```

or in a run card:

```toml
[symbolica]
suggest_license = false
```

JSON CLI output suppresses the banner automatically so stdout remains valid
machine-readable JSON.

## Requesting a license

Interactive helpers collect the fields, show a confirmation, and submit
through Symbolica's Python API:

```console
pyamplicol request-symbolica-trial-license
pyamplicol request-symbolica-hobbyist-license
```

Complete noninteractive requests require all fields and `--yes`:

```console
pyamplicol request-symbolica-trial-license \
  --name "Ada Lovelace" \
  --email ada@example.org \
  --organization "Example Institute" \
  --yes

pyamplicol request-symbolica-hobbyist-license \
  --name "Ada Lovelace" \
  --email ada@example.org \
  --yes
```

pyAmpliCol does not retain the submitted identity fields and does not print the
returned key. Symbolica emails the issued key. Export it before generation:

```console
export SYMBOLICA_LICENSE='your-issued-key'
```

Consult the
[official Symbolica installation and licensing guide](https://symbolica.io/docs/get_started.html)
before choosing a request type.

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

## SymJIT is a separate dependency

SymJIT is the native JIT runtime used by Symbolica-generated applications. It
is distributed under the MIT License, not under the Symbolica proprietary
license. Release dependency metadata pins the official
[`siravan/symjit-crate`](https://github.com/siravan/symjit-crate) source and an
immutable revision.

SymJIT compression is enabled by default during generation. It factors
repeated complex instruction sequences into internal applets without changing
the evaluator ABI or numerical contract:

```toml
[evaluator.jit]
compress = true
```

For an intentional diagnostic comparison:

```console
pyamplicol generate --card run.toml --no-jit-compress
```

Prepared model bundles bake their compression/backend choice into the kernel
pack, so an incompatible request is adjusted to the prepared pack and reported
in the effective configuration.

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
Symbolica separately. Users remain responsible for an appropriate use license.

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

If f64 evaluation fails, do not assume it is a Symbolica-license problem: first
run `pyamplicol self-test` and inspect the artifact target. If generation or
precision-80 evaluation fails, then inspect the Symbolica check. See
[Troubleshooting](troubleshooting.md).

## Related pages

- [Installation](installation.md) — wheel and source requirements.
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md) — where symbolic work occurs.
- [Artifacts and Portability](artifacts-and-portability.md) — what the resulting f64 artifact carries.
- [Release and Support](release-and-support.md) — published dependency and validation boundary.
