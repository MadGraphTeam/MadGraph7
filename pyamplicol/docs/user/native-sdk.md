<!-- SPDX-License-Identifier: 0BSD -->

# Native SDK

Each binary wheel owns a target-specific static Rusticol SDK:

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

`rusticol-config` validates the SDK schema, package version, target, and
confined resource paths before returning target-specific linker arguments:

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

`--cflags`, `--libs`, and `--rustflags` return shell-escaped argument streams.
`--rust-source` returns the safe Rust wrapper path. `--cargo-rustflags` returns
the same Rust linker arguments in Cargo's unit-separator encoding for
`CARGO_ENCODED_RUSTFLAGS`; it is not a shell argument stream. `--json` exposes
the paths and typed C/C++ and Rust linker arrays together.

The command is expected to fail in an unstaged source tree. Install a binary
wheel or a wheel built with `just wheel` before compiling native consumers.

## Generated Artifact APIs

Every generated artifact has one API bundle at its root:

```text
artifacts/pp_zjj/API/
  validation_points.dat
  python/check_standalone.py
  rust/check_standalone.rs
  cpp/check_standalone.cpp
  fortran/check_standalone.f90
```

The drivers load metadata, select a concrete process, accept a JSON
model-parameter card and direct overrides, evaluate all resolved components,
sum them explicitly, and compare with the optimized total. `--process` accepts
either the exact concrete expression stored in the artifact or its stable
process/alias ID. These selectors are equivalent:

```console
--process 'd d~ > z g g'
--process p_p_to_z_j_j_4
```

Case and whitespace are normalized for expression selectors, while particle
labels and ordering remain significant. Ambiguous expressions are rejected
with the matching stable IDs. Run the primary subprocess with:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > z g g' \
  --set-parameter aS 0.117 0 \
  --json

make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > z g g" --set-parameter aS 0.117 0 --precision 16 --json'
make -C artifacts/pp_zjj/API/cpp run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
make -C artifacts/pp_zjj/API/fortran run \
  ARGS='--process p_p_to_z_j_j_4 --set-parameter aS 0.117 0 --json'
```

Each Makefile writes binaries, objects, and Fortran modules to a sibling
`.pyamplicol-api-build/` directory, leaving the integrity-checked process
artifact unchanged.

## Rust 2021 f64

The wheel's `rust/rusticol.rs` is a dependency-free safe Rust 2021 wrapper over
C ABI v1. It provides an owning `Runtime`, typed physics metadata, atomic model
parameter updates, warning access, `Selectors`, compatibility-total
`evaluate_f64`, and resolved `evaluate_resolved_f64`. The handle is freed on
drop and remains bound to its creating thread.

The generated `API/rust/check_standalone.rs` includes that wrapper through
`RUSTICOL_RUST_SOURCE`; it does not duplicate FFI declarations or depend on a
published Rusticol crate. Its primary Makefile target invokes `rustc` directly:

```console
make -C artifacts/pp_zjj/API/rust check_standalone
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > z g g" --precision 16 --json'
```

The equivalent direct compilation is reproducible from the artifact root:

```console
cd artifacts/pp_zjj
build=../.pyamplicol-api-build/pp_zjj/rust/check_standalone
mkdir -p "$(dirname "$build")"
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)"
eval "set -- $(rusticol-config --rustflags)"
RUSTICOL_RUST_SOURCE="$RUSTICOL_RUST_SOURCE" \
  rustc --edition=2021 API/rust/check_standalone.rs -o "$build" "$@"
"$build" --process p_p_to_z_j_j_4 --precision 16 --json
```

`rust-script` is an optional separately installed convenience, not a pyAmpliCol
or runtime requirement. The generated source contains its minimal Cargo header,
so it can be run without changing the artifact:

```console
cd artifacts/pp_zjj
RUSTICOL_RUST_SOURCE="$(rusticol-config --rust-source)" \
  CARGO_ENCODED_RUSTFLAGS="$(rusticol-config --cargo-rustflags)" \
  rust-script API/rust/check_standalone.rs -- \
  --process p_p_to_z_j_j_4 --precision 16 --json
```

The Makefile exposes this as `make -C artifacts/pp_zjj/API/rust run-script`.
`--precision 16` is the only native precision; use the generated Python driver
for precision-controlled Symbolica evaluation.

## C++17

The header-only wrapper provides metadata, model parameters, warnings, and
total/resolved f64 evaluation. Momentum storage is row-major
`[point][external particle][E, px, py, pz]`; the external-particle axis follows
the selected process metadata. This complete example evaluates one
`d d~ > Z g g` point:

```cpp
#include <rusticol.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

int main() {
    try {
        rusticol::Runtime runtime("artifacts/pp_zjj", "d d~ > z g g");
        runtime.set_model_parameter("aS", 0.117);

        const std::vector<double> momenta{
            500.0, 0.0, 0.0, 500.0,
            500.0, 0.0, 0.0, -500.0,
            462.6501613061637, 14.340107538562991,
            155.76435943335707, -425.7484539710246,
            369.7738416261408, -17.479290785282917,
            2.0064955613504103, 369.3550355960509,
            167.57599706769557, 3.1391832467199254,
            -157.77085499470743, 56.3934183749737,
        };
        constexpr std::size_t point_count = 1;

        const auto totals = runtime.evaluate(momenta, point_count);
        const auto resolved = runtime.evaluate_resolved(momenta, point_count);
        const auto resolved_totals = resolved.total();
        const double scale = std::max(1.0, std::abs(totals.at(0)));
        if (std::abs(totals.at(0) - resolved_totals.at(0)) > 1.0e-12 * scale) {
            std::cerr << "resolved components do not reproduce the total\n";
            return 1;
        }
        std::cout << totals.at(0) << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Rusticol error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
```

Compile using installed-wheel discovery:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
c++ -std=c++17 runtime.cpp "$@" -o runtime_cpp
```

`rusticol-config` is part of the trusted local installation. The `eval` step
turns its shell-escaped flag stream into an argument vector and therefore also
preserves SDK paths containing spaces. Generated API bundles provide the same
logic in their Makefiles.

## Fortran 2008

The wheel ships module source rather than a compiler-specific `.mod` file:

```console
RUSTICOL_FORTRAN="$(rusticol-config --fortran-source)"
eval "set -- $(rusticol-config --libs)"
gfortran -std=f2008 "$RUSTICOL_FORTRAN" runtime.f90 "$@" -o runtime_fortran
```

`type(rusticol_runtime)` provides load/close, metadata, model-parameter and
warning methods, and total/resolved f64 evaluation. Resolved Fortran storage is
`(color, helicity, point)`, the column-major view of the C ABI sequence
`(point, helicity, color)`.

Complete hand-written C++ and Fortran examples are in
[`examples/native`](../../examples/native):

```console
make -C examples/native
examples/native/runtime_cpp artifacts/pp_zjj p_p_to_z_j_j_4 \
  examples/data/model_parameters.json
examples/native/runtime_fortran artifacts/pp_zjj 'd d~ > z g g' \
  examples/data/model_parameters.json
```

The C ABI, C++ wrapper, Fortran module, and generated Rust wrapper use the same
Rusticol resolver. After loading, `process_key()` returns the stable ID even
when the caller selected the process by expression.

## Runtime Capability

The static SDK supports direct SymJIT f64 artifacts identified by
`symjit.application.complex-f64.v1` and compiled ASM/C++ f64 artifacts
identified by their compiled runtime capabilities. None of these f64 paths
imports the Symbolica Python module or performs a Symbolica runtime-license
check. Direct applications use the separate MIT-licensed SymJIT runtime;
compiled artifacts dynamically load their evaluator library.

ASM/C++ libraries are target-specific. Rusticol requires an exact artifact and
runtime target-triple match and verifies every recorded CPU feature before
loading executable state. Higher-precision retained evaluator state is not a
native SDK capability and remains available only through the Symbolica-backed
Python path.

Process artifacts are trusted executable inputs. Their hashes verify payload
consistency but do not establish origin.
