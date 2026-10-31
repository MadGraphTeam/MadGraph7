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
> [`THIRD_PARTY_NOTICES.md`](https://github.com/MadGraphTeam/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md)
> and the repository's `licenses/` directory.

## Package license at startup

Importing pyAmpliCol imports Symbolica and calls `symbolica.set_library_key()`
with the key issued for the `pyamplicol` package, before any symbolic work.
When that key is accepted and valid, users do not need a personal Symbolica
license for calls made by pyAmpliCol, including parallel generation and
evaluator optimization. Registration is also attempted when a spawned Python
worker imports pyAmpliCol.

```python
import sys
import pyamplicol

print("symbolica" in sys.modules)  # True

from pyamplicol.licensing import detect_symbolica_license
print(detect_symbolica_license(suggest=False).licensed)  # Actual license state
```

If Symbolica is not installed, lightweight configuration and source tooling
remain available; operations requiring Symbolica still report the missing
dependency. Broken native extensions or other unexpected import errors are
not suppressed.

The key licenses calls within `pyamplicol`; it does not license unrelated
Symbolica code in the user's program. Personal license environment variables
are left unchanged. Model compilation and generation tooling remain lazy.

If registration raises `ValueError` or `PermissionError`, or an older
Symbolica lacks `set_library_key()`, import continues. Symbolica's public
`is_licensed()` check then decides whether a valid personal license permits
normal operation or restricted resources are needed; a nonempty
`SYMBOLICA_LICENSE` alone is not proof of validity. Personal-license
configuration and the existing license-request helpers remain available.

This fallback covers synchronous registration failures, not all later
revocations. Symbolica may terminate the process during asynchronous license
verification; pyAmpliCol cannot catch that upstream termination and switch
licenses in the same process.

The startup import and registration temporarily suppress Symbolica's banner
before CLI output policy is known, then restore the caller's banner setting.
Normal reminder suppression and JSON output behavior remain unchanged.

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
| Virtual (one-loop) evaluation at f64 | No (native code), except loops at light masses | Runs the artifact's stored SymJIT applications at any model parameters, scale and pole selection; see [Concurrent processes without a license](#concurrent-processes-without-a-license). A loop whose squared masses or invariants lie strictly between 0 and 1e-9 in the model's units is reduced by the exact engine natively under the package key, as double-double is. |
| Virtual evaluation in double-double or arbitrary precision | Yes, natively, under the package key | The loop runtime decodes saved Symbolica evaluator state in its Rust copy of Symbolica, which pyAmpliCol's key unlocks. |

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

## Concurrent processes without a license

The package key licenses the Python `symbolica` module. The native extension
(and the native loop SDK) contains its own Rust copy of Symbolica. pyAmpliCol
registers the same package key with that copy too, and the native operations
that need Symbolica state (decoding saved evaluators of loop processes,
double-double and arbitrary-precision loop evaluation) run under it. Without the key, that copy would run restricted: the
first Symbolica operation of a process takes the user's single restricted
thread, and any other process of the same user that needs one then aborts
with "Cannot start another restricted Symbolica thread while this user's
thread allowance is in use". A personal `SYMBOLICA_LICENSE` is used as
before. If the package key is expired or rejected, the native code falls back
to the personal license or to the restricted mode, as the Python side does.

- Importing pyAmpliCol and f64 tree evaluation perform no Symbolica operation
  in the native code, so any number of unlicensed processes can load and
  evaluate tree artifacts concurrently. This holds for the Python API and for
  the C, C++ and Fortran API (the static archive links Symbolica through the
  loop runtime, but tree evaluation never calls it). Tree evaluation at other
  precisions uses the Python `symbolica` module, which the package key
  unlocks, so concurrent unlicensed processes work there too.
- Virtual (one-loop) evaluation of a v2 artifact in f64 (the default
  `precision=16`) runs the SymJIT applications stored in the artifact and
  performs no Symbolica operation in the native code, so any number of
  unlicensed processes (for example a Monte Carlo farm) evaluate virtuals
  concurrently; it does not depend on the package key either. On an x86-64
  CPU without AVX and FMA3, where SymJIT cannot compile, f64 evaluates the
  saved Symbolica states instead, under the package key. One more exception:
  a loop whose squared masses or invariants at a point lie strictly between
  0 and 1e-9 in the model's units (a keV-scale mass in GeV; see "Light
  masses in binary64" in the configuration guide) is reduced by the exact
  generic engine, which uses Symbolica in the native code under the package
  key. Such evaluations also run in concurrent unlicensed processes, but
  they need the package key (or a personal licence) as double-double does.
  Model parameters, the renormalisation scale and the pole selection can
  change freely without Symbolica. The binary64 parameter program
  (couplings, counterterm couplings and masses as functions of the model
  parameters) is never checked at run time. The zero imaginary part of a
  branch-cut argument (`log`, `sqrt`, a power) is made positive, so an
  argument on its cut takes the principal value, `log(-x) = log(x) + iπ`;
  under the complex-mass scheme an argument that is real when some widths
  are zero reads its real part when they are, in every precision (its
  imaginary part could otherwise come out as a rounding residue of either
  sign); and every branch of a model helper's conditional (`recms`,
  `reglog`, `grreglog`, ...) is evaluated and the one the parameters pick is
  selected. Checked against MadLoop and double-double/arbitrary precision
  on the complex-mass control `e- e+ > mu- mu+` at zero, partial-zero
  (54 random rows) and physical widths.
- Double-double and arbitrary-precision virtual evaluation decode saved
  Symbolica evaluator states natively under the package key, so they also
  run in concurrent unlicensed processes.

The wheel's native SDK archive contains this loop runtime and therefore links
Symbolica. `PYAMPLICOL_NATIVE_LIBRARY_KEY=off` disables the package key in
the native code (a diagnostic; the portability tests use it to show that f64
virtual evaluation performs no Symbolica operation).

See [Runtime and Selectors](runtime-and-selectors.md) and [Native APIs](native-apis.md).

## Generation with a valid license

pyAmpliCol calls `symbolica.is_licensed()` from inside the package to select
its generation resources. This detects a valid registered package key even
when `SYMBOLICA_LICENSE` is absent, or a valid personal license. A call from
unrelated user code can still return `False`, because the package key is scoped
to pyAmpliCol.

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

Without a valid package or personal license, pyAmpliCol offers a
license-request reminder. Users whose work is eligible under Symbolica's
current terms can continue in restricted mode. Symbolica describes that mode
as non-commercial, one instance, and one core per device; commercial work
requires the applicable professional license path.

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

## Requesting a personal license

Personal licenses remain useful if the package key is unavailable or for
Symbolica work outside pyAmpliCol. Interactive helpers collect the fields,
show a confirmation, and submit through Symbolica's Python API:

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

SymJIT compression shares repeated arithmetic sequences to reduce generated
code size. The extra calls can increase evaluation time, but the smaller
instruction footprint can make larger evaluators faster. Neither setting is
universally faster; use `profile` to compare them for your workload.

The default is `compress = "auto"`: compression is enabled for compiled JIT
evaluators at optimization level O2 and disabled for other execution modes or
optimization levels. Explicit `true` or `false` settings override that policy.
To enable it explicitly:

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
Symbolica separately. While valid, the embedded library key covers Symbolica
calls within pyAmpliCol; other Symbolica use requires its own applicable
authorization.

Exact dependency versions and notices are included in release metadata and
[`THIRD_PARTY_NOTICES.md`](https://github.com/MadGraphTeam/pyamplicol/blob/main/THIRD_PARTY_NOTICES.md).

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
