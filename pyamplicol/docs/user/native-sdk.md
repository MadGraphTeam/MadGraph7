<!-- SPDX-License-Identifier: 0BSD -->

# Native SDK

Each binary wheel owns a target-specific static Rusticol SDK:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  fortran/rusticol.f90
  lib/librusticol_capi.a
  sboms/rusticol-capi.cyclonedx.json
  metadata.json
  link.json
```

`rusticol-config` verifies the SDK metadata, static-archive hash, and SBOM hash
before returning paths or typed linker flags:

```console
rusticol-config --abi-version
rusticol-config --target
rusticol-config --include-dir
rusticol-config --library
rusticol-config --fortran-source
rusticol-config --sbom
rusticol-config --cflags
rusticol-config --libs
rusticol-config --json
```

It is expected to fail in an unstaged source checkout. Install a binary wheel
or a wheel built with `just wheel` before compiling native examples.

## C++17

```cpp
#include <rusticol.hpp>

rusticol::Runtime runtime("artifacts/ddbar_zg", "ddbar_zg");
auto total = runtime.evaluate(flat_momenta, point_count);
auto resolved = runtime.evaluate_resolved(flat_momenta, point_count);
```

Compile using only installed-wheel discovery:

```console
c++ -std=c++17 runtime.cpp \
  $(rusticol-config --cflags) \
  $(rusticol-config --library) \
  $(rusticol-config --libs) \
  -o runtime_cpp
```

The header-only wrapper exposes metadata, physical helicities and colors,
total/resolved f64 evaluation, JSON/direct parameter updates, and warning
control. Native evaluation is f64 only.

## Fortran 2008

The wheel ships module source rather than a compiler-specific `.mod` file:

```console
gfortran -std=f2008 \
  $(rusticol-config --fortran-source) runtime.f90 \
  $(rusticol-config --library) \
  $(rusticol-config --libs) \
  -o runtime_fortran
```

`type(rusticol_runtime)` provides `load`, `evaluate`, `evaluate_resolved`,
parameter updates, metadata, and warning methods. Resolved Fortran storage is
`(color, helicity, point)`, the column-major view of the C ABI's
`(point, helicity, color)` sequence.

The complete source examples and Makefile are in
[`examples/native`](../../examples/native). Schema-v3 generation also emits
equivalent standalone drivers once at the artifact root:

```console
python artifacts/ddbar_zg/API/python/check_standalone.py \
  --process ddbar_zg --json

make -C artifacts/ddbar_zg/API/cpp run ARGS='--process ddbar_zg --json'
make -C artifacts/ddbar_zg/API/fortran run ARGS='--process ddbar_zg --json'
```

Both Makefiles keep binaries, objects, and Fortran modules in the artifact's
sibling `.pyamplicol-api-build/` directory. The signed artifact tree therefore
remains unchanged and can still pass strict payload verification at runtime.

Each driver loads typed metadata, applies optional model-parameter cards and
direct overrides, evaluates every resolved component represented by the
artifact, explicitly sums them, and compares that sum with the compatibility
total. The native drivers reject precision requests other than f64.

The wheel-owned static SDK intentionally supports direct SymJIT f64 artifacts,
identified by `symjit.application.complex-f64.v1`. These payloads contain the
compiled application and require neither the Symbolica Python module nor a
Symbolica license after generation. ASM and C++ evaluator artifacts retain a
Symbolica runtime capability and are rejected with a capability error rather
than being loaded partially.

The SDK treats process artifacts as trusted executable inputs. Manifest hashes
verify payload consistency but are not signatures or provenance evidence.
