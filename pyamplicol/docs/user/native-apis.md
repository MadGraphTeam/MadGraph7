---
title: "Native APIs"
nav_order: 2
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Native APIs

Every binary pyAmpliCol wheel includes a target-specific Rusticol SDK and every
generated process artifact can include a ready-to-build API bundle. Python,
C11, C++17, Fortran 2008, and Rust 2021 all select and evaluate the same
artifact through the same Rusticol core. C, C++, Fortran, and the standalone
Rust interface share the public C ABI v1; Python uses the wheel's PyO3 binding.

> **Prerequisites:** install a binary wheel as described in [Installation](installation.md),
> activate that environment, and generate the primary artifact from
> [Quick Start](quick-start.md). Native consumers need the corresponding language compiler;
> they do **not** need a Rust compiler unless the consumer itself is Rust.

## What the wheel provides

The installed SDK is owned by the wheel:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  rust/rusticol.rs
  fortran/rusticol.f90
  lib/librusticol_capi.a
  config.py
  metadata.json
  link.json
```

`rusticol-config` validates this SDK and emits correctly quoted paths and
target-specific link arguments:

```console
rusticol-config --abi-version
rusticol-config --version
rusticol-config --target
rusticol-config --include-dir
rusticol-config --library
rusticol-config --fortran-source
rusticol-config --rust-source
rusticol-config --cflags
rusticol-config --libs
rusticol-config --rustflags
rusticol-config --cargo-rustflags
rusticol-config --json
```

For a published 0.1.3 macOS arm64 wheel, the first three commands print:

```text
1
0.1.3
aarch64-apple-darwin
```

The target naturally differs on Intel macOS and manylinux. The ABI remains 1.

Run these commands from the environment that contains pyAmpliCol:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install pyamplicol
rusticol-config --json
```

If pyAmpliCol was invoked through an explicit path, either activate that
environment or put its `bin` directory on `PATH`. An explicit
`RUSTICOL_CONFIG=/path/to/rusticol-config` also works.

In the source workflow, generated drivers below a checkout can find the nearest
`.venv/bin/rusticol-config` created by `just dev-install`. This is a
development convenience, not an embedded build-machine path.

`--cflags`, `--libs`, and `--rustflags` emit shell-escaped argument streams.
By contrast, `--cargo-rustflags` emits the same linker arguments using Cargo's
unit-separator encoding for `CARGO_ENCODED_RUSTFLAGS`; it is not a shell
argument stream. `--json` exposes typed arrays when a program should avoid
parsing either textual form.

## The generated API bundle

With `generation.emit_api_bundle = true` (the default for ordinary
generation), an artifact contains:

```text
artifacts/pp_zjj/API/
  validation_points.dat
  python/check_standalone.py
  c/check_standalone.c
  rust/check_standalone.rs
  cpp/check_standalone.cpp
  fortran/check_standalone.f90
```

The per-language Makefiles place binaries, objects, and Fortran modules in the
sibling `artifacts/.pyamplicol-api-build/` directory. The integrity-checked
artifact itself is not modified.

Each driver:

- selects a process by stable ID or expression;
- accepts an optional JSON kinematic point;
- accepts a UFO-style JSON model-parameter card and direct overrides;
- evaluates all resolved components and sums them explicitly;
- compares the explicit sum with the optimized total;
- prints either a human result or JSON.

## One process expression in all five languages

The following commands deliberately request `d d~ > g z g`, even though its
stored representative uses another outgoing order. Rusticol resolves the
unique representative and permutes momenta, particles, helicities, color
flows, reductions, and resolved output metadata consistently.

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --set-parameter aS 0.117 0 \
  --json

make -C artifacts/pp_zjj/API/c run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'

make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --precision 16 --json'

make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'

make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process "d d~ > g z g" --set-parameter aS 0.117 0 --json'
```

No generated alias is required. Case and whitespace are normalized. Incoming
legs may permute only among incoming legs; outgoing legs may permute only among
outgoing legs. An ambiguous expression fails with the candidate stable IDs.

The drivers also accept the representative stable ID:

```console
--process p_p_to_z_j_j_4
```

After loading, the API exposes the representative stable key and active
representative-to-public permutation. The public process expression remains
the ordering requested by the caller.

## Custom kinematics

All five drivers accept `--kinematics PATH`. The file contains exactly one
point, either directly as `[external][4]` or inside a singleton batch
`[[external][4]]`. Its leg order is the order written in `--process`.

For `d d~ > g z g`, `my_sample_point.json` may be:

```json
[
  ["250.0", "0", "0", "250.0"],
  ["250.0", "0", "0", "-250.0"],
  ["204.406", "204.406", "0", "0"],
  ["91.188", "0", "0", "0"],
  ["204.406", "-204.406", "0", "0"]
]
```

Each vector is `[E, px, py, pz]`. Components may be JSON numbers or decimal
strings. Multiple points, booleans, non-finite values, incorrect rank, or an
incorrect particle count are rejected.

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --kinematics my_sample_point.json \
  --precision 80
```

When an artifact retains an exact evaluator, the Python driver keeps decimal
strings as `Decimal` values at non-f64 precision without an intermediate f64
round trip. Native drivers convert the same input to f64 and accept only
precision 16. OTF artifacts are binary64-only through the Python driver too.

When `--kinematics` is omitted, the bundled representative validation point is
reordered into the public process order automatically.

## Model-parameter cards and overrides

`--model-parameters PATH` reads one flat JSON object. A real external value is
a finite number; a complex external value is `[real, imaginary]`:

```json
{
  "aS": 0.117,
  "MZ": 91.188,
  "complex_external_parameter": [1.0, -0.25]
}
```

Apply the card, then override one entry:

```console
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process "d d~ > g z g" \
        --model-parameters data/model_parameters.json \
        --set-parameter aS 0.1165 0 --json'
```

Direct overrides are applied after the card and win atomically. Unknown,
immutable, non-finite, or otherwise invalid entries reject the complete update.

Compiled, eager, recurrence, and OTF artifacts use this same API surface when
the bundle is emitted. Eager artifacts carry compact invocation tables,
recurrence artifacts carry current schedules, and OTF artifacts carry the
referenced prepared kernels plus a compact process seed. A native caller never
needs the source `.pyamplicol-model` bundle used during generation.

## OTF warm-up from native APIs

OTF callers can make cold-path work explicit with
`rusticol_runtime_warm_up_f64` in C, `rusticol::Runtime::warm_up` in C++,
`runtime%warm_up` in Fortran, or `Runtime::warm_up`/`warm_up_f64` in the safe
Rust wrapper. Each call accepts exactly one flattened binary64 point and
optional global helicity/color ID subsets, constructs and retains that family,
and performs its first evaluation.

In C the entry point is:

```c
const char *one_flow[] = {flow_id};
RusticolWarmUpResult result = {0};
int status = rusticol_runtime_warm_up_f64(
    handle, point, momentum_count,
    NULL, 0, one_flow, 1,
    report_progress, user_data, &result);
```

The optional fixed-layout callback reports stage, completed and total query
counts, elapsed time, construction workers, and current/peak RSS when the
platform exposes it. Updates are throttled and callbacks run only on the
coordinating caller thread. Returning zero from the C/Fortran callback, or
`false` from the C++/Rust callback, cancels at the next pre-commit boundary;
the final post-commit first-evaluation notification cannot be cancelled. With no
observer, ordinary evaluation and warm-up perform no progress-callback or
memory-sampling work.

The cache belongs to one runtime handle and retains only the last selected
family. Native wrappers do not expose Python's clear-without-unload
convenience: close/free/drop and reload to start fully cold. See the
[OTF lifecycle walkthrough](lc-workloads-and-execution-modes.md#the-otf-warm-state-lifecycle)
for the corresponding C++, Fortran, and Rust call fragments.

## Selector support

The C, C++, Fortran, and Rust total-evaluation entry points accept optional
zero-based selector arrays with one entry per point. They map to the physical
helicity and LC-flow order reported by runtime metadata.

- Batch-global string-ID subsets and per-point selectors are mutually
  exclusive on the same axis.
- LC supports physical helicity and color-flow selection.
- NLC/full supports helicity selection; color is contracted.
- Rusticol groups equal per-point selectors for contiguous execution and
  restores the original point order on output.
- Rectangular resolved evaluation remains batch-global.

The selector contract is consistent across compiled, eager, recurrence, and
OTF artifacts; each lane applies its own internal planner. For OTF, LC exposes
selectable flow families. NLC and full colour expose one contracted component
and reject LC-flow selectors. Those contracted families are correctness
capabilities without a high-multiplicity runtime-practicality promise.

See [Runtime and Selectors](runtime-and-selectors.md) for the same behavior in Python.

## C11

The C header exposes C ABI v1 as an opaque runtime handle, typed status codes,
metadata getters, parameter updates, warning access, explicit OTF warm-up,
total evaluation, and resolved evaluation. Use `rusticol-config` rather than
hard-coding include or library paths:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
cc -std=c11 my_runtime.c "$@" -o my_runtime
```

The shell `eval` converts the command's shell-escaped output into an argument
vector and preserves SDK paths containing spaces.

## C++17

`rusticol.hpp` is a header-only RAII wrapper over C ABI v1. It exposes metadata,
parameters, selectors, warnings, one-point OTF warm-up, and total/resolved f64
evaluation:

```cpp
#include <rusticol.hpp>

#include <iostream>
#include <vector>

int main() {
    rusticol::Runtime runtime("artifacts/pp_zjj", "d d~ > g z g");
    runtime.set_model_parameter("aS", 0.117);

    std::vector<double> momenta = /* point-major [point][particle][4] */;
    auto total = runtime.evaluate(momenta, 1);
    auto resolved = runtime.evaluate_resolved(momenta, 1);

    std::cout << total.at(0) << "\n";
    std::cout << resolved.total().at(0) << "\n";
}
```

Compile with:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
c++ -std=c++17 my_runtime.cpp "$@" -o my_runtime
```

## Fortran 2008

The wheel ships portable module source rather than a compiler-specific `.mod`
file:

```console
RUSTICOL_FORTRAN="$(rusticol-config --fortran-source)"
eval "set -- $(rusticol-config --libs)"
gfortran -std=f2008 "$RUSTICOL_FORTRAN" my_runtime.f90 "$@" -o my_runtime
```

`type(rusticol_runtime)` owns load/close, metadata, parameter, warning,
one-point OTF warm-up, and total/resolved f64 methods. Resolved Fortran storage
is `(color, helicity, point)`, the column-major view of the C ABI sequence
`(point, helicity, color)`.

## Rust 2021

`rusticol.rs` is a dependency-free safe wrapper over C ABI v1. It owns the
handle, frees it on drop, exposes typed metadata, selectors, and explicit
one-point OTF warm-up, and remains bound to its creating thread.

The generated driver is compiled directly with `rustc`:

```console
make -C artifacts/pp_zjj/API/rust check_standalone
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > g z g" --precision 16 --json'
```

For a hand-written source:

```console
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)"
eval "set -- $(rusticol-config --rustflags)"
RUSTICOL_RUST_SOURCE="$RUSTICOL_RUST_SOURCE" \
  rustc --edition=2021 my_runtime.rs -o my_runtime "$@"
```

`rust-script` is optional and is not a pyAmpliCol runtime requirement. The
generated source includes its minimal Cargo header, so an installed
`rust-script` can run it directly:

```console
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)" \
  CARGO_ENCODED_RUSTFLAGS="$(rusticol-config --cargo-rustflags)" \
  rust-script artifacts/pp_zjj/API/rust/check_standalone.rs -- \
  --process 'd d~ > g z g' --precision 16 --json
```

The generated Makefile exposes the same path as:

```console
make -C artifacts/pp_zjj/API/rust run-script \
  ARGS='--process "d d~ > g z g" --precision 16 --json'
```

## Runtime and precision boundary

All native APIs use the Symbolica-independent f64 runtime:

- direct JIT artifacts load the separate MIT-licensed SymJIT application;
- eager and recurrence artifacts execute their prepared native schedules;
- OTF artifacts construct a selected recurrence family from their compact seed
  and then execute the artifact-local prepared kernels;
- C++/ASM evaluator artifacts load only on a compatible target and CPU;
- no native call imports Symbolica or performs a Symbolica license check.

Python is the only standalone driver that can request Symbolica-backed exact
precision when the artifact retains an exact evaluator. OTF does not retain
that path and rejects non-f64 precision. See
[Symbolica and Licensing](symbolica-and-licensing.md).

## Common setup failures

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: pyamplicol` | Run the Python driver with the environment's Python or activate that environment. |
| `rusticol-config: command not found` | Activate the same environment, set `RUSTICOL_CONFIG`, or use the supported checkout-local `.venv` workflow. |
| `rusticol.h` / `rusticol.hpp` not found | Do not invoke the compiler without the flags emitted by `rusticol-config`. |
| Empty `RUSTICOL_RUST_SOURCE` or `RUSTICOL_FORTRAN` | Verify `rusticol-config --rust-source` / `--fortran-source` in the active environment. |
| Target incompatibility | Use a portable JIT artifact (compiled O1/O2, or eager/recurrence/OTF from a prepared JIT O2 pack), or regenerate C++/ASM/O0/O3 on the destination target. |

See [Troubleshooting](troubleshooting.md) for a fuller decision tree.

## Related pages

- [Examples Gallery](examples-gallery.md) — complete copied examples and generated API commands.
- [Artifacts and Portability](artifacts-and-portability.md) — target rules and trusted-input boundary.
- [Runtime and Selectors](runtime-and-selectors.md) — process ordering and selector semantics.
