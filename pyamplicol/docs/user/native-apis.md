---
title: "Native APIs"
nav_order: 2
parent: "Python API"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Native APIs

Every binary pyAmpliCol wheel includes a target-specific Rusticol SDK and every
generated process artifact can include a ready-to-build API bundle. Python,
C11, C++17, Fortran 2008, and Rust 2021 select and perform ordinary total/resolved
evaluation of the same artifact through the same Rusticol core. C, C++,
Fortran, and the standalone Rust interface share the public C ABI v1; Python
uses the wheel's PyO3 binding for these native operations.

For a shared matrix-element provider that can be loaded by MadSpace or linked
into another executable, see [UMAMI and MadSpace](umami-api.md). Its generated
`API/umami` Makefile builds `libumami.so` and a linked standalone driver using
the same installed Rusticol SDK.

`rusticol-config` (also available as `python -m rusticol_config`) only reads SDK
paths, version and linker metadata. It does not initialize pyAmpliCol or
Symbolica, so build-system queries can run alongside numerical evaluations.

FFT basis is a generation-time choice, not a native runtime flag. Generation
with `--color-accuracy full` or Python's `ColorConfig(accuracy="full")`
automatically tries adaptive adjoint FFT for recurrence/on-the-fly execution,
falling back to direct for unsupported FFT plans. Certified gluon trees and
single-insertion HEFT use DDM; quark processes use fundamental chains, while
uncertified interactions retain trace.
The saved `fft_basis_selection` reports the actual representation and reason;
the defaults are automatic contraction and adaptive adjoint selection.
C, C++, Fortran and Rust load either generated artifact and use their existing
total/resolved evaluation calls unchanged. Both FFT choices require NLC/full
colour in recurrence or on-the-fly mode. Automatic LC, compiled/eager, and
correlated generation use direct contraction and trace basis. Select
`--color-contraction direct` to opt out; `--fft trace` or `--fft adjoint` forces
FFT and retains unsupported-plan errors. Forced adjoint FFT rejects correlations.

> **Prerequisites:** install a binary wheel as described in [Installation](installation.md),
> activate that environment, and generate the primary artifact from
> [Quick Start](quick-start.md). Native consumers need the corresponding language compiler;
> they do **not** need a Rust compiler unless the consumer itself is Rust.

## Graded (formal 1/N_c) tree output

Artifacts generated with `--color-accuracy strict-lc` or `strict-nlc`, or with
full colour and `--color-graded`, report their colour sum split by N_c degree.
In every language the ordinary resolved evaluation exposes the split: the
contracted colour axis has one component per reported degree, with IDs
`color:nc-degree:<d>` in decreasing degree, and its components sum to the
ordinary total. The Rust core (`rusticol-core`, as used by Rust consumers that
link it directly) adds:

| Method | Result |
| --- | --- |
| `NativeRuntime::nc_expansion()` | `Option<NativeNcExpansion>`: accuracy (`strict-lc`, `strict-nlc` or `full`), reference degree P, degrees and their colour IDs |
| `evaluate_graded_f64(momenta, points, helicity_ids)` | `NativeGradedEvaluation`: `values[point * degrees + d]`, summed over the selected helicities, and `totals()` |
| `evaluate_graded_with_precision(momenta, points, digits, helicity_ids)` | `NativeDecimalGradedEvaluation` in double-double (`digits = 32`) or arbitrary precision (feature `symbolica-runtime`, compiled artifacts) |
| `evaluate_correlated_graded_f64(momenta, points, request, helicity_ids)` | `NativeGradedCorrelatedEvaluation`: one correlation, `values[point * degrees + d]` as `[re, im]` |

Each degree is the N_c^degree term of the expansion evaluated at N_c = 3. The
C ABI and the generated drivers use the ordinary resolved calls.

## Correlated Born evaluations

The C, C++, Fortran and standalone Rust SDKs expose the same generation-time
colour catalogue, literal spin replacements and grouped correlated evaluations.
These additions are on the `correlators` development branch; use its matching
SDK and library, not an older installed release. Declare the operators with
`Generator.generate(..., correlators=CorrelatorConfig(...))` or
`generate ... --correlators FILE.json` as in [Born Correlations](../correlators.md).

Native SDK correlated evaluation is **binary64 only**, like the ordinary
native SDK. It runs in RustiCol without Python or Symbolica. The separate Python
correlated executor retains Symbolica-backed double-double and arbitrary
precision; its `precision=16` path also uses that executor.

A Rust program that links `rusticol-core` with its `symbolica-runtime` feature
(as for `NativeRuntime::evaluate_with_precision`) can evaluate the same
requests natively in double-double and arbitrary precision:
`NativeRuntime::evaluate_correlated_many_with_precision(momenta, point_count,
decimal_digits, requests, helicity_ids)` takes decimal-string momenta
(`[point][external particle][E,px,py,pz]`), the decimal precision of
`evaluate_with_precision` (32 selects double-double, 16 binary64, otherwise
`ceil(digits * log2(10))` bits) and `NativeDecimalCorrelatedRequest`s whose spin
vectors are `[real, imaginary]` decimal strings per component. It returns
request-major `[real, imaginary]` decimal strings, `values[request *
point_count + point]`. Sources, stages, the exact colour weights, the
contraction and derived model parameters are evaluated at the working
precision; the normalization constant and the process-group factor are the
artifact's, exactly as in `evaluate_with_precision`. On a graded or strict
artifact (see above) each value sums the kept N_c degrees, as
`evaluate_correlated_many_f64` does.

LC, NLC, full colour and the strict `strict-lc`/`strict-nlc` are selected at
generation for each inserted colour matrix, independently of connection
order. The underlying amplitudes remain
complete and full-colour: ordinary `evaluate`/`evaluate_resolved` still return
that underlying process, while the correlated `"born"` request returns the
Born result at the requested correlated accuracy. Spin defaults affect only
correlated calls, never ordinary totals or resolved components.

### Catalogue, grouped requests and result indices

| Language | Catalogue | Grouped evaluation and component access |
| --- | --- | --- |
| C | `rusticol_runtime_color_correlation_count`, `_id`, `_catalogue_json` | `rusticol_runtime_evaluate_correlated_many_f64`; output offset `2*(request*point_count + point)` gives real then imaginary parts |
| C++ | `runtime.color_correlation_ids()`, `.color_correlation_catalogue_json()` | `runtime.evaluate_correlated_many(...)`; `result(request, point)` is `std::complex<double>` |
| Fortran | `runtime%color_correlation_ids()`, `%color_correlation_catalogue_json()` | `runtime%evaluate_correlated_many(...)`; `values(point, request)` is `complex(c_double_complex)` |
| Rust | `runtime.color_correlation_ids()`, `.color_correlation_catalogue_json()` | `runtime.evaluate_correlated_many_f64(...)`; `result.get(request, point)` returns `Option<Complex64>` |

The ID list and full JSON histories follow generation order, including
`"born"`. C/C++/Rust indices are zero-based; Fortran array indices are
one-based. Requests follow caller order and points follow input order.
Different requests share amplitudes and unchanged spin-dependent stages in
one call. All momenta use `[point][external particle][E,px,py,pz]`.

For example, using the registered `"T13"` dipole from the correlation guide:

```cpp
// runtime and flattened momenta have already been prepared.
std::vector<rusticol::SpinCorrelationVector> vectors{
    {3, {{{0., 0., 1., 0.}}}}};
std::vector<rusticol::SpinCorrelationVector> physical;
auto values = runtime.evaluate_correlated_many(momenta, point_count, {
    {"born", physical}, {"T13", physical}, {"T13", vectors},
});
std::complex<double> first_spin_dipole = values(2, 0);
runtime.set_spin_correlation_vectors(vectors);
auto inherited = runtime.evaluate_correlated(momenta, point_count, "T13");
runtime.set_spin_correlation_vectors();  // Clear defaults.
```

```fortran
type(rusticol_correlated_request) :: requests(2)
complex(c_double_complex), allocatable :: values(:, :)
! Unallocated spin_vectors inherits defaults; allocated size zero is physical.
requests(1)%color_correlation = "born"
allocate(requests(1)%spin_vectors(0), requests(2)%spin_vectors(1))
requests(2)%color_correlation = "T13"
requests(2)%spin_vectors(1)%leg = 3_c_size_t
allocate(requests(2)%spin_vectors(1)%components(4, 1))
requests(2)%spin_vectors(1)%components(:, 1) = cmplx([0,0,1,0], kind=c_double)
call runtime%evaluate_correlated_many(momenta, point_count, requests, values)
print *, values(1, 2)  ! First point, spin-correlated dipole.
```

```rust
let mut physical = rusticol::CorrelatedRequest::new("born");
physical.spin_vectors = Some(vec![]);
let mut spin = rusticol::CorrelatedRequest::new("T13");
spin.spin_vectors = Some(vec![rusticol::SpinCorrelationVector {
    leg: 3,
    components: vec![[rusticol::Complex64::new(0., 0.),
                      rusticol::Complex64::new(0., 0.),
                      rusticol::Complex64::new(1., 0.),
                      rusticol::Complex64::new(0., 0.)]],
}]);
let values = runtime.evaluate_correlated_many_f64(
    &momenta, point_count, &[physical, spin], &[])?;
let first_spin_dipole = values.get(1, 0).unwrap();
```

Spin labels are **one-based public external legs** in every language. Supply
one complex four-vector to broadcast, or one per momentum point. Fortran uses
`components(4,points)`; C++ and Rust use a vector of complex four-vectors. The
C representation explicitly interleaves real/imaginary doubles, avoiding any
assumption about a language's complex-number memory layout. The nonempty set
of replaced legs must match a declared spin class.

A C request sets `use_default_spin_vectors=1` with zero explicit vectors to
inherit the setter, or `0` to use its own vectors (zero means physical
helicities). C++ uses `std::nullopt` versus an explicit vector; Rust uses
`None` versus `Some(...)`; Fortran uses unallocated versus allocated
`spin_vectors`. Setters copy their input, and empty setter input clears it.
Optional helicity IDs select physical helicities of the unreplaced legs.

Complete examples for the mixed adjoint/fundamental N3LO interference are in
[`examples/native/correlated.c`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.c),
[`correlated.cpp`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.cpp),
[`correlated.f90`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.f90), and
[`correlated.rs`](https://github.com/MadGraphTeam/pyamplicol/blob/correlators/examples/native/correlated.rs).
The [generation example](../correlators.md#native-correlated-evaluations)
prepares their input. Build all four with `make -C examples/native correlated`;
each executable accepts `ARTIFACT PROCESS`. The generated ordinary standalone
drivers still demonstrate total/resolved evaluation, not correlations.

The one-loop example client
[`examples/native/loop_client.c`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/native/loop_client.c)
(`make -C examples/native loop`; `loop_client ARTIFACT PROCESS MOMENTA.txt
[HELICITY]`) opens a loop artifact through `rusticol_virtual.h`, prints its
squared-order axes and evaluates it with `reduction = "auto"` and the other
backend, in the unchecked setting, the declared colour and spin correlations
and, for a loop-induced process, one helicity against the summed call; it
needs no Symbolica licence. `examples/python/loop_walkthrough.py` writes its
momenta file and runs it.

## What the wheel provides

The installed SDK is owned by the wheel:

```text
pyamplicol/_sdk/
  include/rusticol.h
  include/rusticol.hpp
  include/rusticol_virtual.h
  include/rusticol_virtual.hpp
  include/umami.h
  include/pyamplicol_umami_virtual_v1.h
  rust/rusticol.rs
  fortran/rusticol.f90
  fortran/rusticol_virtual.f90
  umami/pyamplicol_umami_virtual_v1.c
  umami/UMAMI_LICENSE
  lib/librusticol_capi.a.xz         (distribution wheels; a source build ships lib/librusticol_capi.a)
  lib/libmpc-pyamplicol.so.3        (Linux distribution wheels)
  lib/libmpfr-pyamplicol.so.6
  lib/libgmp-pyamplicol.so.10
  config.py
  metadata.json
  link.json
```

The archive holds the tree and one-loop C ABIs (`rusticol_virtual.h`
included). Distribution wheels link GMP, MPFR and MPC dynamically: the three
shared libraries ship next to the archive (on macOS `libmpc.3.dylib`,
`libmpfr.6.dylib` and `libgmp.10.dylib` with `@rpath` install names), and
`link.json` lists them in link order, so `rusticol-config --libs` names them
by path with a run path to `lib/` (see
[GMP, MPFR and MPC in the wheels](installation.md#gmp-mpfr-and-mpc-in-the-wheels)).
The Python extension's own copies, which the wheel repair bundles next to
`_rusticol` (`pyamplicol.libs/` on Linux, `pyamplicol.dylibs/` on macOS),
carry the repair tool's file names (on Linux and macOS possibly with a content
hash in the name) and are not part of the SDK's link line: link only what
`rusticol-config --libs` names.
A source build links them statically and ships only the archive.

### The SDK archive cache

Distribution wheels ship the archive xz-compressed
(`lib/librusticol_capi.a.xz`, about 29 MB instead of 220 MB unpacked).
`metadata.json` records the SHA-256 and size of the unpacked archive. The
first `rusticol-config` query that names the archive (`--library`, `--libs`,
`--rustflags`, `--cargo-rustflags`, `--json`) unpacks it, in a second or two,
into a per-user cache:

```text
$PYAMPLICOL_CACHE_DIR/sdk/<version>-<sha256>/librusticol_capi.a   (if PYAMPLICOL_CACHE_DIR is set)
~/.cache/pyamplicol/sdk/<version>-<sha256>/librusticol_capi.a      (Linux; $XDG_CACHE_HOME/pyamplicol if set)
~/Library/Caches/pyamplicol/sdk/<version>-<sha256>/librusticol_capi.a   (macOS)
```

`PYAMPLICOL_CACHE_DIR` is the same override as for the compiled-model cache.
Later queries reuse the cached copy after checking its size and SHA-256; a copy
that fails the check is unpacked again. Each process unpacks into its own
temporary file in that directory and renames it into place only after the
check, so concurrent first uses (parallel builds) are safe and never see a
partial archive. The installation itself is only read, so a read-only
`site-packages` works; the cache directory must be writable (set
`PYAMPLICOL_CACHE_DIR` otherwise). Metadata-only queries (`--cflags`,
`--include-dir`, `--version`, ...) never unpack. Each pyAmpliCol version
unpacks its own copy (about 50 MB), and nothing deletes the copies of
earlier versions automatically: `rusticol-config --prune-other-versions`
removes every `<cache>/sdk/<version>-<sha256>/` directory other than this
installation's own and prints what it removed (other files under the cache
root are never touched; another installed version unpacks its copy again on
its next use). A source build ships the plain archive, which is used in
place.

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

For a 0.2.0 macOS arm64 release-candidate wheel, the first three commands print:

```text
1
0.2.0
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
  umami/umami_driver.c
  umami/metadata.json
```

The per-language Makefiles place binaries, objects, and Fortran modules in the
sibling `artifacts/.pyamplicol-api-build/` directory. The integrity-checked
artifact itself is not modified.

The ordinary language drivers:

- selects a process by stable ID or expression;
- accepts an optional JSON kinematic point;
- accepts a UFO-style JSON model-parameter card and direct overrides;
- evaluates all resolved components and sums them explicitly;
- compares the explicit sum with the optimized total;
- prints either a human result or JSON.

The [UMAMI driver](umami-api.md#generate-and-build) instead demonstrates the
shared-library ABI and its channel/flavour selectors. Its Makefile places
libraries and executables outside the artifact in the same way.

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
--process p_p_to_z_j_j_3
```

After loading, the API exposes the representative stable key and active
representative-to-public permutation. The public process expression remains
the ordering requested by the caller.

The same selectors reach processes that a `--flavor-blind` generation listed
in `process-groups.json` without generating them. Rusticol evaluates the
representative, multiplies every matrix element by the member's exact factor,
and, for a charge-conjugated member, reverses the LC flow words (and maps flow
selectors back). Runtime metadata JSON reports `process_group_member_of`,
`process_group_factor` (`{"numerator", "denominator"}`), and
`charge_conjugated` in every language. The C ABI adds one getter:

```c
uint64_t numerator, denominator;
int32_t conjugated;
rusticol_runtime_process_group_relation(runtime, &numerator, &denominator, &conjugated);
```

Selections other than non-generated members report `1/1` and `0`. See
[Process-group members](process-selection-and-permutations.md#process-group-members).

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
`rusticol_runtime_warm_up_f64_with_cores` in C, `rusticol::Runtime::warm_up` in C++,
`runtime%warm_up` in Fortran, or `Runtime::warm_up`/`warm_up_f64` in the safe
Rust wrapper. Each call accepts exactly one flattened binary64 point and
optional global helicity/color ID subsets, constructs and retains that family,
and performs its first evaluation.

In C the entry point is:

```c
const char *one_flow[] = {flow_id};
RusticolWarmUpResult result = {0};
int status = rusticol_runtime_warm_up_f64_with_cores(
    handle, point, momentum_count,
    NULL, 0, one_flow, 1,
    2, report_progress, user_data, &result);
```

The construction-core override is per call: C takes `size_t n_cores` before
the callback, with zero selecting the process-output default. The original
`rusticol_runtime_warm_up_f64` keeps its signature without `n_cores` and uses
that default. C++ takes a
trailing `std::size_t n_cores = 0`; Fortran accepts optional integer `n_cores`
and rejects an explicitly supplied value below one; Rust takes
`Option<usize>` before the callback, with `None` selecting the default and
`Some(0)` rejected. Positive values bound independent query-trace construction;
shared-cache merging and family finalization remain serial. The override does
not change later evaluation defaults or structural cache identity. Generated
API bundles use these same SDKs; their standalone drivers retain ordinary lazy
evaluation unless the caller explicitly invokes warm-up.

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

## Saving and restoring an OTF cache

All native APIs can save the completed structural cache of an OTF handle and
restore it into a handle loaded from the matching process output. The following
fragments use an OTF output for `g g > g g` at `artifacts/otf_gg_gg`; the
[Python example](runtime-and-selectors.md#saving-an-otf-warm-cache) gives the
generation command and explicit four-vector input.
Here `point` and `next_point` are two flattened binary64 phase-space points,
each containing 16 components in `[particle][E,px,py,pz]` order. They are arrays
in C/Fortran and vectors in C++/Rust. The first evaluation constructs the
recursion; after reopening, the second reuses it at different momenta.

In C, `check(status)` denotes the application's usual error handler: stop on
a nonzero status and retrieve the message with `rusticol_last_error_message`.

```c
RusticolRuntimeHandle *runtime = NULL;
double value;
check(rusticol_runtime_load("artifacts/otf_gg_gg", NULL, NULL, &runtime));
check(rusticol_runtime_evaluate_f64(runtime, point, 16, 1, &value, 1));
check(rusticol_runtime_save(runtime, "gg.otf-cache"));
check(rusticol_runtime_free(runtime));

check(rusticol_runtime_load("artifacts/otf_gg_gg", NULL, NULL, &runtime));
check(rusticol_runtime_load_cache(runtime, "gg.otf-cache"));
check(rusticol_runtime_evaluate_f64(runtime, next_point, 16, 1, &value, 1));
check(rusticol_runtime_free(runtime));
```

```cpp
{
    rusticol::Runtime runtime("artifacts/otf_gg_gg");
    runtime.evaluate(point, 1);
    runtime.save("gg.otf-cache");
} // The original handle is closed.
rusticol::Runtime restored("artifacts/otf_gg_gg");
restored.load_cache("gg.otf-cache");
auto values = restored.evaluate(next_point, 1);
```

```fortran
! Within a procedure using iso_c_binding and the rusticol module:
type(rusticol_runtime) :: runtime
real(c_double), allocatable :: values(:)
call runtime%load("artifacts/otf_gg_gg")
call runtime%evaluate(point, 1_c_size_t, values)
call runtime%save("gg.otf-cache")
call runtime%close()

call runtime%load("artifacts/otf_gg_gg")
call runtime%load_cache("gg.otf-cache")
call runtime%evaluate(next_point, 1_c_size_t, values)
call runtime%close()
```

```rust
{
    let mut runtime = rusticol::Runtime::load("artifacts/otf_gg_gg", None, None)?;
    runtime.evaluate_f64(&point, 1)?;
    runtime.save("gg.otf-cache")?;
} // The original handle is dropped.
let mut restored = rusticol::Runtime::load("artifacts/otf_gg_gg", None, None)?;
restored.load_cache("gg.otf-cache")?;
let values = restored.evaluate_f64(&next_point, 1)?;
```

The reopening block can run in a later invocation of the application. Unlike
the one-flow Python example, these examples use the default helicity/colour
sum; selected evaluations and explicit
`warm_up(...)` calls can be saved in exactly the same way. Reuse the same
selectors after loading to benefit from the saved family. Fortran reports
errors by stopping unless an optional `ierr=status` argument is supplied.

The Rust core's `NativeRuntime` exposes the same `save` and `load_cache`
methods as the dependency-free Rust SDK. C, C++, Fortran, Rust and Python
use the same snapshot format, so a compatible runtime can restore a cache
written through another API. Usual SDK status/exception handling applies.

The snapshot covers the currently retained selection, including LC sums and
contracted NLC/full-colour direct or FFT families. It stores structural rows
and reduction mappings, not machine addresses or evaluated currents. Loading
rebinds the original prepared kernels and allocates numeric workspaces;
the receiving handle's model parameters remain unchanged. The same saved
selection can then run at new phase-space points and batch sizes without
reconstructing its currents. Selecting a different family still follows OTF's
normal cache-replacement behavior. Non-OTF handles reject these operations.
See [Runtime and Selectors](runtime-and-selectors.md#saving-an-otf-warm-cache)
for the Python example and snapshot scope.

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
OTF cache save/restore, total evaluation, and resolved evaluation. Use
`rusticol-config` rather than hard-coding include or library paths:

```console
eval "set -- $(rusticol-config --cflags) $(rusticol-config --libs)"
cc -std=c11 my_runtime.c "$@" -o my_runtime
```

The shell `eval` converts the command's shell-escaped output into an argument
vector and preserves SDK paths containing spaces.

## C++17

`rusticol.hpp` is a header-only RAII wrapper over C ABI v1. It exposes metadata,
parameters, selectors, warnings, one-point OTF warm-up, cache save/restore,
and total/resolved f64 evaluation:

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
one-point OTF warm-up, cache save/restore, and total/resolved f64 methods.
Resolved Fortran storage is `(color, helicity, point)`, the column-major view
of the C ABI sequence `(point, helicity, color)`.

### Squared coupling-order partials

Recurrence artifacts split totals by squared coupling order in one pass.
`rusticol_runtime_coupling_orders_json` describes the axes
(`rusticol-amplitude-orders-v1`: coupling-order `names`, amplitude
`components` and `squared` orders with their ordered component `pairs`),
`rusticol_runtime_squared_order_count` sizes the output and
`rusticol_runtime_evaluate_orders_f64` fills `totals[point]` (bitwise the
selected total) and `partials[point][squared order]`. Squared orders, here and
in every binding, follow the canonical order of Python's `TreeOrderEvaluation`
(nonzero `(name, power)` pairs over the sorted names, compared
lexicographically: `QCD2 QED2`, `QCD4`, `QED4`); map columns through `squared`
rather than assuming a layout shared with other result types. C++ exposes
`Runtime::coupling_orders_json()` and `Runtime::evaluate_orders()`, Fortran
`coupling_orders_json` and `evaluate_orders` (`partials(order, point)`), and
Rust `coupling_orders_json()` and `evaluate_orders_f64()`.

## Rust 2021

`rusticol.rs` is a dependency-free safe wrapper over C ABI v1. It owns the
handle, frees it on drop, exposes typed metadata, selectors, explicit
one-point OTF warm-up and cache save/restore, and remains bound to its creating
thread.

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

Every binary64 tree call of the C, C++, Fortran and Rust APIs and of the UMAMI
provider (ordinary values, helicity- and colour-resolved values, global and
per-point helicity / colour-flow selections, squared-order partials, colour-
and spin-correlated values, the OTF warm-up, model-parameter updates) needs no
Symbolica licensing at all: neither a personal licence nor pyAmpliCol's
package key. Concurrent unlicensed processes with the key disabled
(`PYAMPLICOL_NATIVE_LIBRARY_KEY=off`), while another process holds the
restricted Symbolica permit, return the licensed run's values bit for bit
(`tests/integration/test_native_sdk_tree_licence_free.py`). On an x86-64 CPU
without AVX and FMA3, SymJIT tree artifacts are refused with the reason; they
are never evaluated through Symbolica.

Python is the only standalone driver that can request Symbolica-backed exact
precision when the artifact retains an exact evaluator. OTF does not retain
that path and rejects non-f64 precision. See
[Symbolica and Licensing](symbolica-and-licensing.md).
The separate Python correlated API is also Symbolica-backed at precision 16;
the native correlated entry points instead use the independent RustiCol
binary64 executor described above.

The one-loop SDK (`rusticol_virtual.h`) monitors binary64 stability like the
Python API's `stability_checks=`: `rusticol_virtual_set_stability_checks`
(C++ and Fortran `set_stability_checks`, the packaged Rust wrapper's
`set_stability_checks`) takes a count of default rotations or rotation
specifications, the reduction-direction check, its rescaled variant and the
cross-backend check. Without checks (the default, or a rotation count of 0
with the other checks off) nothing else runs and no estimate is formed. With
checks, every successful point of a binary64 ordinary evaluation carries its
estimate in the result's `diagnostics_json` (`"stability"`), equal bit for bit
to the Python API's `VirtualStabilityEstimate`; the precision rescue stays the
Python API's. The SDK reads every binary64 input as its exact value, so its
counterpart is the Python API given those exact values (for example
`str(decimal.Decimal(x))`). The Python API reads a float by its shortest
decimal (`33.6`, not the double's exact value): where the two differ, its
point lies half an ulp from the SDK's, and values and estimates may differ in
their last digits. Binary64 evaluation, with or without the monitoring, runs
the artifact's stored SymJIT applications and needs no Symbolica licence: the
shipped loop-induced self-test artifact evaluates through the C ABI without
`SYMBOLICA_LICENSE` while another process holds the restricted Symbolica
permit. Two binary64 cases evaluate saved Symbolica states natively, under
pyAmpliCol's own package library key (no user licence either; see
[Symbolica and licensing](symbolica-and-licensing.md)): a loop whose squared
masses or invariants lie strictly between 0 and 1e-9 in the model's units
(or an invariant within 1e-9 of a real squared mass), reduced by the generic
engine, and an x86-64 CPU without AVX and FMA3, where SymJIT cannot compile.
Double-double and arbitrary precision run under the package key too.

The SDK serves the loop-induced selections of the Python API
(`helicities=`, `color_resolution="amplitude-basis"`) through
`rusticol_virtual_evaluate_selected` and its wrappers (see the SDK README,
"Selections"): the amplitude-basis anchors are an explicitly basis-dependent
split, a helicity selection is |A(h)|²/avg with the total's average, only the
selected rows are evaluated, and colour flows are refused (full colour).

The reduction backend (`rusticol_virtual_set_reduction`; C++ and Fortran
`set_reduction`, the packaged Rust wrapper's `set_reduction`) is the Python
API's `reduction=`: `"auto"` takes TIRust for a loop-induced process and
NinjaRS for a Born-interfered one. The SDKs have no rescue stack; instead,
under `"auto"` on TIRust, a point TIRust refuses, or fails in the enabled
binary64 monitoring's own evaluations (for example a rotated evaluation), is
evaluated again on NinjaRS from its original exact inputs, in the call's own
arithmetic, after the TIRust call (the per-point fallback). The result's
`diagnostics_json` records it: `"reduction"` holds the call's `requested`,
`effective` and `reason` with its `"fallback"` backend and
`"fallback_points"` count, and `"fallbacks"` maps the id of every point the
fallback evaluated again to `{"effective", "reason", "guard_estimate"}`
(`effective` null: the point failed on NinjaRS too, with both backends'
reasons in `"failures"`, or its guard rejected it; such a point's
cross-backend check is in `not_run`). In double-double and arbitrary
precision every point the fallback serves is guarded, as in the Python API:
its NinjaRS value runs the reduction-direction check (the reversed re-test
frame, cached masters) in the call's own arithmetic, within the fallback's
own evaluation of the point, and the relative estimate of the outputs the
call returns (each sector against its largest Laurent coefficient: the
summed |A|²; on the resolved, correlated and selected calls each output of
the call's rows only, against its own magnitude, an output that vanishes
against the largest output of the point, as in the Python API) is the
entry's `"guard_estimate"`. A point whose estimate exceeds the guard's
limit, the call's target (below) never below 1e-5, has status `Failed`, not
`AccuracyNotEstablished`, with one reason naming the fallback, TIRust's
failure, the arithmetic, the estimate and the limit; a point within it keeps
the fallback's value unchanged. On the same exact inputs the guard's
estimates are the Python API's, bitwise or a few units in the last place
apart (the SDK forms the colour-metric norms of the inclusive |A|² in
binary64). In binary64 the fallback's value is
served unguarded (`"guard_estimate"` null), as before the guard. An explicit `"ninjars"`
or `"tirust"` has no fallback and no guard, nor has a TIR-only artifact
(NinjaRS refused typed); a library built without TIRust takes NinjaRS under
`"auto"`, with the reason, and refuses `"tirust"`.

NinjaRS before 71eddb15 (pin 5b6a8f57) returned finite, badly wrong values
with status `AccuracyNotEstablished` at near-planar 2 -> 3 points of pentagon families
whose rank equals their propagator count (all-vector quark loops: measured
on loop-induced g g > z z z, z z g, z g g, a a g, a g g and W+ W- g), in
binary64 and in double-double; the measured windows, the records and the
estimates of each check are in the [Python API](python-api.md),
"Near-planar NinjaRS hazard". The SDKs' binary64 monitoring is the Python
API's binary64 estimate, bitwise, and flags them as the Python API's checks
do (the recommended checks gave 3.0 to 236 at the measured wrong binary64
points, a Python API record: `workspace/tmp/ninjars-dd-zzz/C/out/tilt.json`
'nj_b64_chk'); in double-double and arbitrary precision the SDKs run no
monitoring (`AccuracyNotEstablished`, no estimate), so an explicit
`"ninjars"` call there, or any NinjaRS value in binary64 without monitoring
(the `"auto"` fallback's included), was exposed until NinjaRS's conditioning
bound (71eddb15, M-LIT lit-ninja-planar), which refuses such lanes typed
(status Failed). The target accuracy
`rusticol_virtual_set_reduction_target(handle, target)` (C++ and Fortran
`set_reduction_target`, the packaged Rust wrapper's `set_reduction_target`;
default 1e-5, the relative accuracy of each point's result) sets the
calibrated acceptance of TIRust's checked method choice, as the Python API's
`reduction_target=`, and in double-double and arbitrary precision the
fallback guard's threshold, never below 1e-5; a value that is not a finite
positive number is an
invalid argument. The unchecked setting
The kinematics improvement
`rusticol_virtual_set_kinematics_improvement(handle, setting)` (C++ / Fortran /
packaged Rust `set_kinematics_improvement`; `"z-shift"`, MadLoop's default and
the default here, `"rescaling"`, `"projection"` or `"off"`, as the Python
API's `kinematics_improvement=`) and
`rusticol_virtual_set_kinematics_improvement_binary64(handle, on)` (0 by
default, 1 on; `set_kinematics_improvement_binary64`) improve each point of a
double-double or arbitrary-precision call (a binary64 one only with the
switch) to exact conservation and on-shell externals in that carrier, from
the caller's inputs in the caller's leg order, before anything is evaluated
(the per-point fallback and the stability rotations act on that point);
`diagnostics_json` records it per point (`"kinematics"`: the Python API's
`VirtualKinematics` fields); another value is an invalid argument
([Configuration](configuration.md#kinematics-improvement-of-the-input-points)).
Additive symbols, ABI version 2 kept. The unchecked setting
`rusticol_virtual_set_reduction_unchecked(handle, unchecked)` (C++
`set_reduction_unchecked(bool)`, Fortran `set_reduction_unchecked(logical)`,
the packaged Rust wrapper's `set_reduction_unchecked(bool)`; 1 or 0, default
0, as the Python API's `reduction_unchecked=`) reduces without the backends'
own acceptance and refusal checks (TIRust: its gates and method choice as
method selection, no acceptance pass, plain PV at gate 0 as the last resort;
NinjaRS: zero stability thresholds, its box mu-expansion's conditioning
bound active; only points without a finite value or refused by that bound
fail); `"auto"` keeps its backend and its per-point fallback, which then
acts only on those points, each served one guarded as above in double-double
and arbitrary precision; the stability
monitoring stays its own setting;
`"reduction"` in `diagnostics_json` records `"unchecked"`, and any other
value is an invalid argument. It is off by default, a timing and benchmark
mode with no accuracy certificate (the measured accuracy per process:
[Python API](python-api.md), "Unchecked reductions"). The fallback, its
guard, the target and the unchecked setting need no Symbolica licence: a
binary64 call on NinjaRS or TIRust touches no Symbolica state, and a
double-double or arbitrary-precision fallback and its guard run under the
package key as any SDK call in that arithmetic.

## Common setup failures

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: pyamplicol` | Run the Python driver with the environment's Python or activate that environment. |
| `rusticol-config: command not found` | Activate the same environment, set `RUSTICOL_CONFIG`, or use the supported checkout-local `.venv` workflow. |
| `rusticol.h` / `rusticol.hpp` not found | Do not invoke the compiler without the flags emitted by `rusticol-config`. |
| Empty `RUSTICOL_RUST_SOURCE` or `RUSTICOL_FORTRAN` | Verify `rusticol-config --rust-source` / `--fortran-source` in the active environment. |
| ``librusticol_loop_sdk.so: undefined reference to `__isoc23_strtol@GLIBC_2.38'`` at link time, or ``version `GLIBC_2.38' not found (required by .../librusticol_loop_sdk.so)`` at run time | The loop SDK's shared release library needs glibc 2.38 or newer: the link fails with a toolchain on an older glibc, and the dynamic loader stops a client built on a newer glibc and run on an older one. Build and run the client on glibc 2.38 or newer, or build the SDK from source on this host (see [Native libraries and glibc](installation.md#native-libraries-and-glibc)). |
| Target incompatibility | Use a portable JIT artifact (compiled O1/O2, or eager/recurrence/OTF from a prepared JIT O2 pack), or regenerate C++/ASM/O0/O3 on the destination target. |

See [Troubleshooting](troubleshooting.md) for a fuller decision tree.

## Related pages

- [Examples Gallery](examples-gallery.md) — complete copied examples and generated API commands.
- [Artifacts and Portability](artifacts-and-portability.md) — target rules and trusted-input boundary.
- [Runtime and Selectors](runtime-and-selectors.md) — process ordering and selector semantics.
- [Born Correlations](../correlators.md) — opt-in LC/NLC/full-colour correlations in Python and native SDKs.
