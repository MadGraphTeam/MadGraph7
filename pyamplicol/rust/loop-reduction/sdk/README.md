# Standalone virtual SDK

> **Plan v4 only.** The per-diagram legacy one-loop engine was removed
> (2026-10-29, user ruling M-QCD Q3): the SDK opens plan v4 artifacts of the
> v2 engine; a plan v1-v3 artifact (`rusticol.loop-runtime.v3`) is refused
> with a compatibility error ("the legacy one-loop engine was removed ...;
> regenerate the artifact"). Resolved, directed-colour (correlated),
> spin-source and verified/rescued evaluations were served by the legacy
> runtime only: their entry points remain in the ABI and refuse with the
> plan v4 reason (use the Python API). Where the text below describes
> verification, rescue or resolved axes, it describes those refusing entry
> points; their tests and example clients were removed.

These additive Rust/C/C++/Fortran interfaces call the same `VirtualRuntime` used by the Python
facade. It reads ordinary verified process artifacts and needs no Python process,
model importer, graph generator or alternate reducer. The existing tree SDK is
unchanged. All interfaces return physical Born–loop interference, all generated
squared coupling orders, and finite/single/double poles. Loop amplitudes squared
and loop-induced production remain unsupported.

The ordinary package builds the tree and virtual APIs into one
`librusticol_capi.a`. Its existing `_sdk` bundle carries the virtual C/C++ headers,
Fortran module, UMAMI extension sources and the dependency-free safe Rust wrapper.
Use `rusticol-config` to obtain that installation's include path, archive and
platform link flags. There is one native build identity and one link manifest.
The Python facade uses the loop module registered inside `pyamplicol._rusticol`;
it does not load a separately installed companion extension.

A source build of the combined archive is:

```sh
cargo build --locked -p rusticol-capi
```

C declarations are in `include/rusticol_virtual.h`. The packaged single-file Rust
wrapper exposes `rusticol::virtual_amplitudes::Runtime` and owned `Evaluation`
values without a Cargo or Python dependency. Its native mixed-order control is
`examples/client.rs`. Direct source Cargo consumers can instead depend on
`rusticol-loop-sdk`, which provides the same shared runtime in its safe Rust API;
its standalone archive remains available for loop-only developer clients.
Do not link both archives into one client.

For example, on the tested Linux host:

```sh
cc -std=c11 examples/client.c -Iinclude /path/to/librusticol_capi.a \
  -ldl -lpthread -lm -o client
./client /path/to/resolved-gg-source-artifact
```

Those system libraries describe this tested host; use the installed SDK link
metadata on another platform. The Rust clients use the single-file wrapper, so
their Cargo example targets link the same combined archive:

```sh
cargo build --locked -p rusticol-capi
cargo rustc --locked -p rusticol-loop-sdk --example client -- \
  -L native=target/debug -l static=rusticol_capi
```

Plain `cargo build --examples` compiles them but cannot resolve the C ABI,
because the wrapper deliberately has no Cargo dependency on this crate.

The archive includes the required GMP/MPFR code and the existing
Symbolica/NinjaRS/master providers. It does not link PyO3 or require a Python
interpreter. Saved full-colour directed histories are available through the
additive grouped complex interface below; mapped-process calls remain
unavailable. A prepared, explicitly continued source-colour plan can be loaded
with LC/NLC metadata; generation requires its declared model continuation.

Plan v4 artifacts (the v2 loop engine, `point_inputs = "native-v2"`) open
through the same `rusticol_virtual_open`: the stored native programs (tree,
counterterm and parameter programs with their binary64 SymJIT plane
applications) evaluate every point without Python, generation or a Symbolica
licence, bitwise equal to the Python API in binary64. They serve ordinary
(unresolved) binary64 virtuals with model R2: `rusticol_virtual_set_poles`
selects every pole (`"all"`, the default), the finite part alone (`"none"`:
the pole-only work is skipped natively and the single and double pole entries
are quiet NaNs) or a split selection (`"uv"`: the poles are the bare loop's UV
poles; `"ir"`: the renormalised virtual's IR poles), which also runs the
stored companion of the certified-scaleless families; the finite part is the
same in every selection, and every selection is bitwise the Python API's.
Correlated, resolved, verified and extended-precision evaluations of a plan v4
process go through the Python API; plan v1-v3 handles accept `"all"` only.
The R2 strategy of a plan v4 process is the model's: `"auto"` (the default,
`NULL`) and `"model"` select it, `"direct"` is refused.

Model parameters of a plan v4 process (`parameters_json` at open,
`rusticol_virtual_update_parameters`) behave exactly as the Python API's
`Runtime.load(..., model_parameters=...)` and `set_model_parameters`: the
whole mapping is validated before anything is applied, and an unknown name
(not an external parameter of the artifact's model), the renormalisation
scale's parameter (set per point through `mu_r2`; generation resolves which
parameter it is, as the Python API does) and a parameter fixed at generation
(`[loop.preparation] fixed_parameters`, e.g. a coloured width fixed to zero)
set to another value are refused with `RUSTICOL_VIRTUAL_INVALID_ARGUMENT` and
the Python API's message; a refused update changes nothing (an open refuses
the handle). A known parameter the process does not read is accepted with no
effect, as in Python. Names are nonempty (an empty name is refused with
`RUSTICOL_VIRTUAL_INVALID_ARGUMENT` and the Python API's `TypeError` message).
Values are ASCII decimal strings read as Python's `Decimal` (surrounding ASCII
whitespace and underscores ignored; no fractions, no non-ASCII digits or
whitespace, which the Python API refuses too; an infinity or NaN is refused
as not finite) and real: a
nonzero imaginary part is the Python API's `CompatibilityError`, here
`RUSTICOL_VIRTUAL_RUNTIME_ERROR` (as `"direct"` R2). The checks run in the
Python API's order (every value, then every name, by name), so a mapping with
several refused entries is refused with the same message in every client. The
fixed parameters include those the model preparation fixes (e.g. a colour
declaration's external scheme constants); their refusal says so (they are not
run-time parameters), while one the request fixed points to
`[loop.preparation] fixed_parameters`.

A model whose Born couplings, kernel constants or masses depend on the
renormalisation scale (a running-coupling or MSbar-mass UFO; generation
records the parameter program's scale-dependent outputs) is evaluated per
exact `mu_r2`: a batch of mixed scales is partitioned by scale, so every point
equals its evaluation in a batch of its scale alone. The usual NLO models,
whose scale enters the counterterms only, run one batch.

`Runtime::evaluate` accepts binary64 inputs and returns binary64 numerical rows.
It preserves every finite input's exact value for optional higher-precision
verification. Standard fixed decimal formatting uses 1074 fractional places,
the maximum denominator exponent of IEEE binary64, then removes redundant
fractional zeros. This ingress cost belongs outside numerator/reducer/master
timings. `evaluate_decimal` accepts original exact decimal strings directly.
Neither method silently requests verification or rescue. `Runtime::open_with_policy`
retains an explicit default before exact source parameters are evaluated.
An evaluation-level policy replaces that default only for that call; omission
inherits it, including after `clear()`.

Results store compact successful IDs alongside statuses for **all** requested
IDs. Failed points have no numerical row. An ordinary success reports accuracy
not established; explicit verification delegates the existing native policy,
including exact source snapshots, increasing actual precision, subset retries,
comparison of all components/axes and output-conversion checks. Detailed JSON
records retain failures and attempt history. With a configured policy, a numerical
global source-binding failure retains the original exact overrides/defaults and
its failed primary stage when a configured higher carrier can realize them.
That admission is not an accuracy claim: rescue still requires agreement of
complete results at two successful levels and checked conversion to binary64.
Malformed parameters and unrecoverable formula errors remain atomic function
errors. Without a load-time policy, source binding keeps its eager behavior.

The additive C entry point is `rusticol_virtual_open_with_policy`, with the same
verification JSON accepted by batch requests. The installed dependency-free Rust
wrapper has `open_with_policy(..., verification_json)`. C++ accepts the optional
policy JSON as the final constructor argument; Fortran accepts it as the optional
trailing `verification_json=` argument of `load`. Existing open signatures and
positional Fortran calls remain valid. Capability JSON reports the retained
policy; all ordinary, resolved, directed and spin-source evaluations preserve
their full verification diagnostics.

Inclusive output axes are `[point,order,epsilon]`. Resolved output adds helicity
and colour-component axes before epsilon. Resolved LC components are signed
coherent interferences anchored to canonical groups of Born tensors, potentially
overcomplete. Inclusive projected identity correlations and NLC/full retain one
contracted component. The saved plan supplies component IDs and, for resolved LC,
Born-column memberships. Summing helicity and component axes
reconstructs the inclusive result. Helicity rows retain the inclusive initial spin
average. Failures and precision rescue operate on whole original physical points,
including every component and squared order in the agreement check.
The epsilon axis is always `(0,-1,-2)`, with physical couplings included.
Catalogue JSON supplies the exact contributing Born/loop order pairs. These
signed interference values are not probabilities.

The C result handle owns its buffers independently of the runtime. Query shapes,
then copy arrays, or query JSON byte sizes before copying strings. Every short
array buffer is rejected before writing any element. Runtime handles require
exclusive access, and parameter updates use the existing atomic source owner.
There is no per-event model-parameter interface. `mu_r2` is explicit, positive,
scalar or per-point; it is unrelated to the numerator's extra-dimensional mu².

C++17 callers include `include/rusticol_virtual.hpp` and use the move-only
`rusticol::VirtualRuntime` and `VirtualEvaluation` classes. Function errors throw
`VirtualError`; per-point numerical failures remain statuses in a successfully
returned result. Each evaluation makes one C bulk call. Result copies to standard
vectors retain the C axis order, and result ownership survives runtime moves,
`clear()` and destruction. Exact parameter and verification JSON pass unchanged.
For example, from this directory:

```sh
c++ -std=c++17 examples/client.cpp -Iinclude /path/to/librusticol_capi.a \
  -ldl -lpthread -lm -o client-cpp
./client-cpp /path/to/resolved-gg-source-artifact gg
```

Fortran callers compile `fortran/rusticol_virtual.f90` and `use rusticol_virtual`.
The tested compiler is gfortran with Fortran 2018 mode. A `virtual_runtime` owns a
reference-counted C handle: assignment shares it, and finalization or `close()`
releases that reference. Copies therefore share parameter updates and workspace
clears and must not be called concurrently. Direct, array, self, containing-type
assignment and function-result lifetimes have native controls. The allocatable
arrays in `virtual_evaluation` have independent value ownership, including after
runtime destruction. `load()` replaces only its own reference after a successful
open; a failed open preserves the existing handle.

Fortran momentum arrays have shape `(4, external_leg, point)`. Returned arrays
reverse the C axis order without numerical reordering:

| Array | Fortran dimensions |
|---|---|
| `born` | `(colour, helicity, order, successful_point)` |
| `virtual_values` | `(epsilon, colour, helicity, order, successful_point)` |
| `statuses`, `requested_ids` | `(requested_point)` |
| `successful_ids` | `(successful_point)` |

Inclusive evaluation retains singleton colour/helicity axes. Epsilon indices
1, 2, 3 mean finite, single pole, double pole. `mu_r2` takes a rank-one array of
length one or the number of points. Empty momentum and scale arrays are valid.
An optional `ierr` returns function status; when omitted, a function error stops
with its diagnostic. `evaluate(..., error_message=text)` supplies accurate local
shape or native error text, while `virtual_last_error()` exposes the last C
error. A failed function leaves output numerical arrays unallocated. A completed
partial batch allocates only successful numerical rows and retains all statuses.
Fortran `integer(c_int64_t)` IDs transport the C uint64 bit patterns; normal
nonnegative Fortran IDs have the same numeric interpretation in both languages.

```sh
gfortran -std=f2018 -c fortran/rusticol_virtual.f90
gfortran -std=f2018 examples/client.f90 rusticol_virtual.o \
  /path/to/librusticol_capi.a -ldl -lpthread -lm -o client-fortran
./client-fortran /path/to/resolved-gg-source-artifact gg
```

Both wrapper clients additionally accept the retained full-CKM source fixture
with process `uudd`. Their three-order inclusive values check the committed
MadLoop control, and each order's resolved helicity sum reconstructs all poles.
Both exercise sparse IDs, real numerical failures with compact rows, explicit
DoubleFloat verification, empty batches, exact parameter JSON and result
lifetimes. They call no Python interpreter. The ordinary SDK bundle includes
these wrappers alongside the same combined archive.

The [versioned pyAmpliCol UMAMI virtual extension](umami/README.md) transports
these same outputs using UMAMI's bulk offset/count/stride layout. Its separate
keys expose every squared order and Laurent coefficient, with mandatory lane
statuses and no probability or finite-only tree fallback.

## Correlated and spin-source entry points

`rusticol_virtual_evaluate_correlated` (grouped complex colour histories),
its joint spin-source form and the matching C++, Fortran, Rust and UMAMI
wrappers remain in the ABI. They were served by the removed legacy runtime
and refuse a plan v4 process with the plan v4 reason: evaluate correlated,
resolved and spin-source virtuals of plan v4 through the Python API
(`evaluate_correlated`, `evaluate_correlated_many`, `evaluate_resolved`).

## glibc floor (Linux)

The shared library `librusticol_loop_sdk.so` of a release native build
(`tools/developer/build_loop_native.py`) needs the glibc symbol versions it
links: `readelf -V librusticol_loop_sdk.so` lists them, and the build's
`build-info.json` records them (`glibc_version_needs`). A build with the
repository's toolchain needs at most `GLIBC_2.38` and no glibc ABI marker: the
build compiles its C dependencies (GMP, MPFR, MPC) with the traditional TLS
dialect, so the library does not need `GLIBC_ABI_GNU2_TLS`, which only glibc
builds carrying that marker define. The floor is therefore glibc 2.38 (the
same holds for the Python extension `_rusticol.so`). Its symbols bind numbered
versions (`readelf --dyn-syms -W librusticol_loop_sdk.so` shows
`__isoc23_strtol@GLIBC_2.38` and `hypot@GLIBC_2.35`, for instance), so an older
glibc is caught at the first step that sees it.

Linking a client executable with a toolchain whose glibc is older than 2.38
fails at link time: GNU ld checks the shared library's undefined symbols for
an executable (`--no-allow-shlib-undefined`, its default there), and the older
glibc's `libc.so.6` defines no symbol of version `GLIBC_2.38`:

```text
ld: .../librusticol_loop_sdk.so: undefined reference to `__isoc23_strtol@GLIBC_2.38'
```

A client built on glibc 2.38 or newer and run on an older glibc (or a shared
library or plugin linked against the SDK on an older toolchain, which GNU ld
does not check) is stopped by the dynamic loader before it runs:

```text
./client: .../libc.so.6: version `GLIBC_2.38' not found (required by .../librusticol_loop_sdk.so)
```

Both messages mean this floor: build and run the client on glibc 2.38 or newer
(`ldd --version` shows the release), or build the SDK from source on the older
host so that it links against that host's glibc.

C and C++ clients of the release build's shared library can stop at compile
time, before either: define `RUSTICOL_VIRTUAL_SHARED`, and
`rusticol_virtual.h` refuses a toolchain whose glibc release is older than
`RUSTICOL_VIRTUAL_GLIBC_FLOOR_MAJOR.MINOR` (2.38) with an `#error` that names
the floor (every glibc release defines the numbered versions of the releases
before it, so the release comparison is exact). The macro states the release
build's floor, not that of an SDK built from source on an older host: omit it
for such a build.

```sh
cc -std=c11 -DRUSTICOL_VIRTUAL_SHARED examples/client.c -Iinclude \
  -L/path/to/build -lrusticol_loop_sdk -Wl,-rpath,/path/to/build -lm -o client
```

Fortran has no such hook (the module is not preprocessed): the link-time error
or the loader message above names the floor. From Python, `import
pyamplicol._rusticol` first looks up every glibc version the extension needs in
the running glibc and raises `pyamplicol.NativePlatformError` naming the missing
versions, the running glibc and the remedy; a refusal by the dynamic loader
becomes the same error.
pyAmpliCol does not open `librusticol_loop_sdk.so` from Python; code that opens
it with `ctypes` gets the dynamic loader's `OSError`.
