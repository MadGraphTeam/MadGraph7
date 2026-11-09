# Standalone virtual SDK

> **Plan v4 only.** The per-diagram legacy one-loop engine was removed
> (2026-10-29, user ruling M-QCD Q3): the SDK opens plan v4 artifacts of the
> v2 engine; a plan v1-v3 artifact (`rusticol.loop-runtime.v3`) is refused
> with a compatibility error ("the legacy one-loop engine was removed ...;
> regenerate the artifact"). A plan v4 process serves the ordinary,
> helicity-resolved, colour-correlated and spin-correlated virtuals in
> binary64, double-double and arbitrary precision (M-FINAL MF-SDK, below);
> the verification policy was removed (ABI version 2) and the stability
> rescue is the Python API's alone (user ruling Q9).

These additive Rust/C/C++/Fortran interfaces call the same `VirtualRuntime` used by the Python
facade. It reads ordinary verified process artifacts and needs no Python process,
model importer, graph generator or alternate reducer. The existing tree SDK is
unchanged. All interfaces return physical Born–loop interference, all generated
squared coupling orders, and finite/single/double poles.

**Loop-induced processes** (|A_loop|², no Born; m-lit LIT-SDK) are opened by the
same `rusticol_virtual_open`, with no opt-in, in artifacts of their own or mixed
with Born-interfered processes: `metadata_json` names `"contribution_kind":
"loop-induced"` and `"interference": "conj(loop)*loop"`; the shape has zero Born
orders and Born colour components and every result zero Born values; the
virtual orders are the loop × loop sectors, each value |A_loop|² per sector and
epsilon power, full colour (one contracted component), whose computed poles
vanish up to rounding (a finiteness cross-check). The values are the Python
API's: binary64 bitwise (the executor's amplitude sum plus the counterterm kets,
squared by the one native square the Python API calls), double-double and
arbitrary precision as `Decimal` values (squared in the canonical order). A
loop-induced process serves the pole selections `"all"` and `"none"`; `"uv"`
and `"ir"` are refused (`RUSTICOL_VIRTUAL_RUNTIME_ERROR`, the Python API's
message). Its resolved and correlated outputs are served as the Python API's
(`evaluate_resolved`, `evaluate_correlated_many`): quadratic forms on the
point's one amplitude vector, one executor run per call. A resolved call gives
|A(h)|²/avg per helicity row with one colour-contracted component (`"full"`;
the rows sum to the ordinary value); the correlated history `"born"` is the
colour metric itself (|A|², the identity-metric history: there is no Born),
each declared history its stored matrix M_X on the amplitude basis (real for a
dipole T_i·T_j, complex otherwise), with zero Born values; a spin source of a
declared vector leg enters through its projection on the leg's polarisation
vectors. Binary64 values are bitwise the Python API's (the same Rust code:
`rusticol_loop_reduction::v2_runtime::induced_outputs`, which the Python API's
binary64 outputs call), double-double and arbitrary precision equal `Decimal`
values.

**Selections (loop-induced, F1).** `rusticol_virtual_evaluate_selected(handle,
request, selection, &result)` (C++ `evaluate_selected` with a
`VirtualSelection`, Fortran `evaluate_selected`, the packaged Rust wrapper's
`evaluate_selected` with a `Selection`, the safe Rust `Runtime::
evaluate_selected` with a `Selection`; UMAMI: `UMAMI_IN_HELICITY_INDEX` and
`PYAMPLICOL_UMAMI_V1_IN_COLOR_RESOLUTION`) serves the Python API's
`helicities=` and `color_resolution=` with the tree C ABI's selector
conventions (global string IDs, per-point zero-based indices, mutually
exclusive per axis, validated before evaluation, output only on success):

- an ordinary call (`request.resolved = 0`) with helicity IDs (`h<i>` or the
  tree engine's `h:<helicities in leg order>`) returns per point the sum over
  the selected rows of |A(h)|²/avg with the total's average (RULED Q39: a
  complete selection sums to the ordinary value; no parity or Bose row
  mapping), `helicity_by_point` each point's own row;
- a resolved call (`request.resolved = 1`) returns the selected rows (every
  row without IDs) with one colour-contracted component (`"contracted"`, the
  default) or with the amplitude-basis anchors (`"amplitude-basis"`: per basis
  element t, `amplitude-<t>`, Re(conj(a_t) (G a)_t)/avg, an explicitly
  basis-dependent split summing to the row's value); its axes are counted by
  the result shape and named by `rusticol_virtual_result_axes_json`;
- only the selected rows are dressed and reduced (the executor's row liveness,
  the Python API's path), binary64 bitwise the Python API's, double-double and
  arbitrary precision its `Decimal` values; the handle's poles, arithmetic and
  reduction backend apply as to every call (the stability monitoring returns
  `{"estimates": null, "reason": ...}`: the Python API estimates selections);
- colour flows are refused (full colour only, RULED Q8), and so are an unknown
  colour resolution or helicity ID, in the Python API's words. A
  Born-interfered process's leading-colour selection stays the Python API's
  (refused typed here; its NLC and full-colour refusals in the Python API's
  words).

The capabilities list the selection (`"helicity_selection"`) and the colour
axes (`"color_resolutions"`: the contracted ID and the anchors' IDs).

| Tree-level SDK feature (`rusticol.h`, tree UMAMI) | Loop-induced virtual SDK |
| --- | --- |
| Ordinary value | served (`rusticol_virtual_evaluate`) |
| Per-order partials | served (one value per loop x loop sector) |
| Helicity-resolved | served (`resolved = 1`) |
| Colour-resolved | served: the amplitude-basis anchors (`"amplitude-basis"`); colour flows not applicable (full colour, Q8) |
| Global helicity selection | served (`evaluate_selected`, helicity IDs) |
| Global colour-flow selection | not applicable (full colour, Q8; refused typed) |
| Per-point helicity selection | served (`helicity_by_point`; UMAMI `UMAMI_IN_HELICITY_INDEX`) |
| Per-point colour-flow selection | not applicable (full colour, Q8; refused typed) |
| Colour-correlated | served (`born` = \|A\|², declared histories) |
| Colour-correlated with helicity IDs | not served: the Python API refuses helicity selection on correlated evaluations |
| Spin-correlated | served (declared spin classes) |
| Carriers (binary64, double-double, arbitrary) | served |
| UMAMI random helicity / LC colour index | not served (the virtual UMAMI extension samples nothing; full colour) |

**Reduction backend.** `rusticol_virtual_set_reduction(handle, "auto" |
"ninjars" | "tirust")` (C++ `set_reduction`, Fortran `set_reduction`) selects
the backend of later evaluations, as the Python API's `reduction=`. `"auto"`
(the default, or the artifact's `[loop] reduction`) follows the one policy the
Python API reads (`rusticol_loop_reduction::reduction_policy`): TIRust for a
loop-induced process (TIRust for both types on a TIR-only artifact), NinjaRS
for a Born-interfered one, with the reason recorded. The SDKs have no rescue
stack; under `"auto"` on TIRust a point TIRust refuses, or fails in the
enabled binary64 monitoring's own evaluations, is evaluated again on NinjaRS
from its original exact inputs, in the call's own arithmetic, after the TIRust
call (the per-point fallback; none on a TIR-only artifact); a point NinjaRS
fails too fails with both reasons. Every point the fallback serves is guarded
(m-lit lit-fallback-guard): the reduction-direction re-test (reversed frame,
cached masters) of the point's summed |A|^2 in the call's own arithmetic
(binary64, double-double or arbitrary precision; reused from a binary64
monitoring that runs that check), and a point whose relative estimate exceeds
the target below has status Failed with one reason
(`reduction_policy::fallback_rejected`: the fallback, TIRust's failure, the
arithmetic, the estimate and the target). An explicit `"ninjars"` or
`"tirust"` has no fallback and no guard. NinjaRS (pin 5b6a8f57) can return
finite, badly wrong values with status AccuracyNotEstablished at near-planar
2 -> 3 points of rank-n pentagon families (all-vector quark loops), in
binary64 and double-double; an explicit `"ninjars"` call without binary64
monitoring (and every double-double or arbitrary-precision NinjaRS call,
which the SDKs do not monitor) stays exposed until the NinjaRS fix is pinned
(windows, records and the estimates of each check: `docs/user/python-api.md`,
"Near-planar NinjaRS hazard"). An explicit backend the library is built without (TIRust needs
the default cargo feature `tir`; without it `"auto"` takes NinjaRS) is refused
by the setter, one that cannot serve a call's carrier (TIRust serves every
carrier) by that call, never switched silently. Every result's
`diagnostics_json` records `"reduction": {"requested", "effective", "reason",
"fallback", "fallback_points"}` and `"fallbacks"`: per point id the fallback
evaluated again, the backend that served it (null: it failed on both, or its
guard rejected it), the reason and `"guard_estimate"` (the guard's relative
estimate; null where the guard did not run); the capabilities list the backends this library serves, the
`auto_default` and the target. `rusticol_virtual_set_reduction_target(handle,
target)` (C++ / Fortran / Rust `set_reduction_target`; default 1e-5, the
relative accuracy of each point's result) sets the calibrated acceptance of
TIRust's checked method choice, as the Python API's `reduction_target=`, and
is the fallback guard's only threshold.
`rusticol_virtual_set_reduction_unchecked(handle, unchecked)` (C++ / Fortran
/ Rust `set_reduction_unchecked`; 0 by default, 1 on) reduces without the
backends' own acceptance and refusal checks, as the Python API's
`reduction_unchecked=` (`reduction_policy::UNCHECKED_RECORD`: TIRust's gates
and method choice as method selection without the acceptance pass, plain PV
at gate 0 as its last resort; NinjaRS's zero stability thresholds); `"auto"`
keeps its backend and per-point fallback; each result records `"unchecked"`
in its `"reduction"` diagnostics; any other value is an invalid argument. It
is a timing and benchmark mode with no accuracy certificate (measured
accuracy per process: `docs/user/python-api.md`, "Unchecked reductions").

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
cargo rustc --locked -p rusticol-loop-sdk --example client --features c-abi-examples -- \
  -L native=target/debug -l static=rusticol_capi
```

The example sits behind the `c-abi-examples` feature: a plain `cargo build
--examples` or `cargo test --workspace` cannot resolve the C ABI, because the
wrapper deliberately has no Cargo dependency on this crate, so it skips it.

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
licence, bitwise equal to the Python API in binary64. They serve the
ordinary virtual with model R2 and, from the process's stored observable
columns, the helicity-resolved, colour-correlated and spin-correlated ones, in
binary64, double-double and arbitrary precision (the extended carriers equal
the Python API's `Decimal` results; "Plan v4 extras" below).
`rusticol_virtual_set_poles` selects every pole (`"all"`, the default), the finite part alone (`"none"`:
the pole-only work is skipped natively and the single and double pole entries
are quiet NaNs) or a split selection (`"uv"`: the poles are the bare loop's UV
poles; `"ir"`: the renormalised virtual's IR poles), which also runs the
stored companion of the certified-scaleless families; the finite part is the
same in every selection, and every selection is bitwise the Python API's
in binary64 (exactly its `Decimal` values in the extended carriers).
`rusticol_virtual_set_arithmetic` selects the arithmetic of later
evaluations (below); resolved and correlated evaluations read the process's
stored observable columns (below).
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
It reads every finite input as its exact value: standard fixed decimal
formatting uses 1074 fractional places, the maximum denominator exponent of
IEEE binary64, then removes redundant fractional zeros. This ingress cost
belongs outside numerator/reducer/master timings. `evaluate_decimal` accepts
original exact decimal strings directly.

Results store compact successful IDs alongside statuses for **all** requested
IDs. Failed points have no numerical row. An ordinary success reports accuracy
not established. The SDK has no verification policy (ABI version 2 dropped
version 1's `rusticol_virtual_open_with_policy` and the batch request's
`verification_json`, which plan v4 refused); the precision rescue is the
Python API's.

**Stability monitoring** (m-lit LIT-SDK, binary64): `Runtime::set_stability_checks`
(C `rusticol_virtual_set_stability_checks`, C++ `set_stability_checks`,
Fortran `set_stability_checks`) takes every field of the Python API's
`VirtualStabilityChecks`: a count of default rotations (or rotation
specifications), the reduction-direction check, its rescaled variant and the
cross-backend check. The default runs none: no estimate and no extra cost. With
checks, a binary64 ordinary evaluation (of a Born-interfered process with poles
"all" or "none", of a loop-induced process) returns each successful point's
estimate in the result's diagnostics (`"stability": {"estimates": [...]}`, one
record per successful point: the Python API's `VirtualStabilityEstimate`
fields; a loop-induced call with poles also its `"finiteness"` ratios),
computed by the same Rust implementation the Python API's binary64 estimates
call (`rusticol_loop_reduction::v2_stability`), so bitwise equal. The other
evaluations (double-double, arbitrary precision, resolved and correlated
outputs) return `"estimates": null` with the reason.
Point statuses keep their meaning; comparing an estimate with a target is the
caller's.

Inclusive output axes are `[point,order,epsilon]`. Resolved output adds helicity
and colour-component axes before epsilon. Resolved LC components are signed
coherent interferences anchored to canonical groups of Born tensors, potentially
overcomplete. Inclusive projected identity correlations and NLC/full retain one
contracted component. The saved plan supplies component IDs and, for resolved LC,
Born-column memberships. Summing helicity and component axes
reconstructs the inclusive result. Helicity rows retain the inclusive initial spin
average. Failures operate on whole original physical points.
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
`clear()` and destruction. Exact parameter JSON passes unchanged.
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
Both exercise sparse IDs, real numerical failures with compact rows, empty
batches, exact parameter JSON and result
lifetimes. They call no Python interpreter. The ordinary SDK bundle includes
these wrappers alongside the same combined archive.

The [versioned pyAmpliCol UMAMI virtual extension](umami/README.md) transports
these same outputs using UMAMI's bulk offset/count/stride layout. Its separate
keys expose every squared order and Laurent coefficient, with mandatory lane
statuses and no probability or finite-only tree fallback.

## Plan v4 extras: resolved, correlated, spin-source and extended precision

A plan v4 process generated with its observable columns (every v2 artifact;
`loops/v2/native_observables`) serves, from the stored columns and without
Python, generation or a Symbolica licence in binary64, exactly what the Python
API's `Runtime` returns:

- **Helicity-resolved** (`request.resolved = 1`; Rust `evaluate(..., true,
  ...)`; C++ `evaluate(..., resolved = true)`; Fortran `resolved=.true.`;
  UMAMI `options.resolved = 1`): the Python API's `evaluate_resolved`, per
  Born helicity row (the capabilities' `helicity_rows`, 1-based legs, ids
  `h<i>`) and, at leading colour, per Born basis element (`basis-<b>`; one
  contracted component at NLC and full colour), in the layout
  point/order/helicity/colour/(epsilon). The shape's `helicities` and colour
  component counts are those axes.
- **Colour-correlated** (`rusticol_virtual_evaluate_correlated`): every stored
  history in one batch, `born` (the inclusive Born metric) first, then the
  declared histories (`rusticol_virtual_correlation_histories_json`), each
  bitwise the Python API's `evaluate_correlated_many` with one request per
  history; complex values, point/history/order/(epsilon).
- **Spin-correlated** (`rusticol_virtual_evaluate_correlated_with_sources`):
  the same with literal complex four-vectors on the legs of one declared spin
  class (`spin_correlation_classes_json`), one row broadcast or one per
  point, as the Python API's `CorrelatedRequest(history, spin_vectors)` with
  the same vectors in every history (each source leg keeps its lowest
  helicity row).
- **Double-double and arbitrary precision**
  (`rusticol_virtual_set_arithmetic(handle, "double-double", 0)` or
  `(handle, "arbitrary", digits)`; `"double"` is the default): the Python API's
  `arithmetic=` / `precision=`. The native executor runs in the carrier (the
  generic reduction engine; MPFR at `max(digits + 20, 40)` working digits,
  double-double at 52) and its outputs are assembled in Python's `Decimal`
  arithmetic at the working digits, operation for operation, so the values
  equal the Python API's `Decimal` results exactly:
  `rusticol_virtual_result_decimal_json` (and
  `rusticol_virtual_correlated_result_decimal_json`) return them as decimal
  literals (`{"arithmetic", "digits", "born", "virtual"}`, the copy layouts,
  absent poles `null`), and the double copies hold their nearest binary64
  values. Ordinary, resolved and correlated evaluations take it; each
  extended evaluation runs the exact tree states (Symbolica's evaluators),
  like the Python API's.

Every extra takes the handle's pole selection (`"all"`, `"none"`, `"uv"`,
`"ir"`). Correlated and resolved calls of a batch of mixed `mu_r2` on a model
whose Born couplings run with the scale are refused, as in the Python API
(their batch is taken as given). A process stored without observable columns
(an older build's artifact) or generated with a forced seed FFT
(`colour contraction = "fft"`) refuses them with the reason; the capabilities
JSON then names it (`observables_unavailable`). The C++, Fortran, packaged
Rust (`rusticol::virtual_amplitudes`) and UMAMI wrappers expose the same
calls (`set_arithmetic`, `decimal_json`). The controls are
`examples/extras_v4.c`, `extras_v4.cpp`, `extras_v4.f90`, `extras_v4_umami.c`
and `tests/v4_extras.rs`, which `tests/unit/test_loop_v2_sdk_extras.py`
compares with the Python API on g g > t t~ and u u~ > d d~ g (binary64
bitwise, the extended carriers' decimals exactly).

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
