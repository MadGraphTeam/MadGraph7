# Standalone virtual SDK

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
metadata on another platform. The archive includes the required GMP/MPFR code
and the existing Symbolica/NinjaRS/master providers. It does not link PyO3 or
require a Python interpreter. Correlation and mapped-process calls remain
unavailable. A prepared, explicitly continued source-colour plan can be loaded
with LC/NLC metadata; generation requires its declared model continuation.

`Runtime::evaluate` accepts binary64 inputs and returns binary64 numerical rows.
It preserves every finite input's exact value for optional higher-precision
verification. Standard fixed decimal formatting uses 1074 fractional places,
the maximum denominator exponent of IEEE binary64, then removes redundant
fractional zeros. This ingress cost belongs outside numerator/reducer/master
timings. `evaluate_decimal` accepts original exact decimal strings directly.
Neither method silently requests verification or rescue.

Results store compact successful IDs alongside statuses for **all** requested
IDs. Failed points have no numerical row. An ordinary success reports accuracy
not established; explicit verification delegates the existing native policy,
including exact source snapshots, increasing actual precision, subset retries,
comparison of all components/axes and output-conversion checks. Detailed JSON
records retain failures and attempt history. Global source-parameter binding
failures can still precede this subset rescue and remain function errors.

Inclusive output axes are `[point,order,epsilon]`. Resolved output adds helicity
and colour-component axes before epsilon. LC components are signed coherent
interferences anchored to canonical groups of Born tensors, potentially
overcomplete; NLC/full retain one contracted component. The saved plan supplies
component IDs and Born-column memberships. Summing helicity and component axes
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

Focused Rust controls accept `PYAMPLICOL_VIRTUAL_ARTIFACT` for the resolved gg
fixture and `PYAMPLICOL_VIRTUAL_MIXED_ARTIFACT` for the full-CKM uudd source fixture.
`PYAMPLICOL_VIRTUAL_LC_ARTIFACT` exercises genuine source LC anchors, including
analytic gg checks, all component axes and native subset precision rescue.
Run the opt-in debug controls with `RUST_MIN_STACK=16777216`: the Rust test
harness's default 2 MiB thread stack is smaller than these dependency-backed
debug evaluators need. The standalone C controls pass on the ordinary process
stack. This test configuration is not a process-size or multiplicity limit.

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
