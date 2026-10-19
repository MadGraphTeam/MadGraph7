# Standalone virtual SDK (source delivery)

These additive Rust/C/C++/Fortran interfaces call the same `VirtualRuntime` used by the Python
facade. It reads ordinary verified process artifacts and needs no Python process,
model importer, graph generator or alternate reducer. The existing tree SDK is
unchanged. All interfaces return physical Born–loop interference, all generated
squared coupling orders, and finite/single/double poles. Loop amplitudes squared
and loop-induced production remain unsupported.

Build inside the already configured loop consumer workspace:

```sh
cargo build --manifest-path rust/loop-reduction/Cargo.toml -p rusticol-loop-sdk
```

The target produces `librusticol_loop_sdk.a`, `librusticol_loop_sdk.so` (platform
equivalent elsewhere) and a safe Rust `rlib`. C declarations are in
`include/rusticol_virtual.h`; Rust callers use `rusticol_loop_sdk::Runtime` and
owned `Evaluation` results. `examples/client.c` demonstrates batched C use with a
resolved gluon-scattering control. On the tested Linux host its static link is:

```sh
cc -std=c11 examples/client.c -Iinclude /path/to/librusticol_loop_sdk.a \
  -ldl -lpthread -lm -o client
./client /path/to/resolved-gg-source-artifact
```

Those system libraries describe this tested host, not a portable release link
manifest. The dependency archives include their required GMP/MPFR code. A shared
library client can instead link `-lrusticol_loop_sdk` with its runtime search path.

This source target intentionally remains in the loop consumer workspace. The
selected validated NinjaRS revision uses local reference paths; the newer
upstream revision has installable Git dependencies but still requires a coherent
consumer update and validation before ordinary wheels can include this target. No optional
path dependency has been added to the ordinary tree workspace. Current build
links the existing Symbolica/NinjaRS/master providers and their normal native
libraries; it does not link the PyO3 crate. Correlation and mapped-process calls
remain unavailable, as do public source LC/NLC artifacts at this checkpoint.

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
and a singleton contracted full-colour component before epsilon. Helicity rows
retain the inclusive initial spin average and sum to the inclusive result.
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
c++ -std=c++17 examples/client.cpp -Iinclude /path/to/librusticol_loop_sdk.a \
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
  /path/to/librusticol_loop_sdk.a -ldl -lpthread -lm -o client-fortran
./client-fortran /path/to/resolved-gg-source-artifact gg
```

Both wrapper clients additionally accept the retained full-CKM source fixture
with process `uudd`. Their three-order inclusive values check the committed
MadLoop control, and each order's resolved helicity sum reconstructs all poles.
Both exercise sparse IDs, real numerical failures with compact rows, explicit
DoubleFloat verification, empty batches, exact parameter JSON and result
lifetimes. They call no Python interpreter. These wrappers are source delivery;
this does not yet add them to ordinary wheel/SDK release bundles.

The [versioned pyAmpliCol UMAMI virtual extension](umami/README.md) transports
these same outputs using UMAMI's bulk offset/count/stride layout. Its separate
keys expose every squared order and Laurent coefficient, with mandatory lane
statuses and no probability or finite-only tree fallback.
