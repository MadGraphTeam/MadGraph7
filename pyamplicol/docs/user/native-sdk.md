<!-- SPDX-License-Identifier: 0BSD -->

# Native SDK

Each binary wheel owns a target-specific static Rusticol SDK:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
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
rusticol-config --cflags
rusticol-config --libs
rusticol-config --rustflags
rusticol-config --json
```

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

Whitespace is normalized for expression selectors, while particle names and
ordering remain significant. Ambiguous expressions are rejected with the
matching stable IDs. Run the primary subprocess with:

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

## Rust f64

The generated Rust 2021 source directly wraps C ABI v1 for its artifact. It is
compiled with `rustc` and links the wheel-owned static library through
`rusticol-config --rustflags`; Cargo and a separately published Rusticol crate
are not required:

```console
make -C artifacts/pp_zjj/API/rust
make -C artifacts/pp_zjj/API/rust run \
  ARGS='--process "d d~ > z g g" --precision 16'
```

The generated wrapper owns the opaque runtime handle, frees it on drop, exposes
metadata and parameter updates, and checks both total and resolved f64
evaluation. `--precision 16` is the only native precision. Use the generated
Python driver for precision-controlled Symbolica evaluation.

## C++17

The header-only wrapper provides metadata, model parameters, warnings, and
total/resolved f64 evaluation:

```cpp
#include <rusticol.hpp>

rusticol::Runtime runtime("artifacts/pp_zjj", "d d~ > z g g");
runtime.set_model_parameter("aS", 0.117);
auto total = runtime.evaluate(flat_momenta, point_count);
auto resolved = runtime.evaluate_resolved(flat_momenta, point_count);
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
`symjit.application.complex-f64.v1`. Once generated, these payloads require
neither the Symbolica Python module nor a Symbolica runtime-license check. This
Symbolica-independent path uses the separate MIT-licensed SymJIT runtime. ASM
and C++ evaluator artifacts retain a Symbolica runtime capability and are
rejected by the lightweight native SDK before partial loading.

Process artifacts are trusted executable inputs. Their hashes verify payload
consistency but do not establish origin.
