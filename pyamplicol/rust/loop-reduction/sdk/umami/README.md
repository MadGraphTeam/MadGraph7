# pyAmpliCol UMAMI virtual extension, version 1

> **Plan v4 only.** The legacy one-loop engine was removed (2026-10-29, user
> ruling M-QCD Q3): as in the shared SDK ([README](../README.md)), a plan v4
> process serves ordinary (unresolved) binary64 virtuals here; resolved,
> correlated and verified calls refuse with the plan v4 reason. The native
> control client below (`client_umami_virtual.c`) was written for the removed
> legacy artifacts.

This source adapter extends UMAMI's keyed bulk-memory conventions for the
pyAmpliCol virtual SDK. It is explicitly a **pyAmpliCol extension**, not a
standard UMAMI virtual interface. The upstream `umami.h`, existing tree provider
and their exported `umami_*` symbols are unchanged. This adapter exports only
`pyamplicol_umami_virtual_v1_*` symbols and shares the existing virtual runtime.
It has no Python dependency and adds no reducer or numerical engine.

Open one process with `rusticol_virtual_open`. Reuse the SDK's exact parameter
JSON, atomic updates, strategy selection, clear/free, shape and metadata queries.
The metadata gives separate Born and virtual squared-order catalogues, their
contributing order pairs, declared scheme, normalization, external states and
actual capabilities. Each call evaluates one saved process; channels, flavour
mappings and per-event coupling overrides are not exposed here. Saved directed
colour histories use the separate complex operation below.
No model or process names are used to infer support.

The extension accepts standard `UMAMI_IN_MOMENTA` and its own namespaced keys:

| Input | Type and shape | Requirement |
|---|---|---|
| `UMAMI_IN_MOMENTA` | `double[stride * external_count * 4]` | Required for nonempty calls |
| `PYAMPLICOL_UMAMI_V1_IN_MU_R2` | `double[stride]` | Exactly one scale input |
| `PYAMPLICOL_UMAMI_V1_IN_MU_R2_SCALAR` | `double[1]` | Alternative broadcast scale |
| `PYAMPLICOL_UMAMI_V1_IN_POINT_ID` | `uint64_t[stride]` | Optional; defaults to event indices |

Scales are positive real renormalization scales **squared**, distinct from loop
momentum mu squared. Binary64 values retain their exact value during native
higher-precision verification/rescue. Input capacities count typed elements in
the full stride-based layout. Only the requested slice is read.

| Output | Type and logical non-batch axes |
|---|---|
| `PYAMPLICOL_UMAMI_V1_OUT_BORN` | `double`, Born order; optionally helicity and colour |
| `PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL` | `double`, virtual order; optionally helicity and colour; epsilon |
| `PYAMPLICOL_UMAMI_V1_OUT_STATUS` | `uint32_t[stride]`, mandatory for nonempty calls |

Every output array has event as its contiguous dimension, following UMAMI's
column-major convention. For example, inclusive virtual values are addressed by
`event + stride * (order + virtual_orders * epsilon)`. Resolved values use
`event + stride * (order + virtual_orders * (helicity + H * (colour + C * epsilon)))`.
The header specifies all layouts. Epsilon indices `0,1,2` mean finite,
single pole and double pole. Born and virtual catalogue orderings are independent.
Resolved helicities retain the incoming spin average and sum to the inclusive
result. Resolved LC colour components are signed Born-anchored interference
contributions; inclusive projected identity correlations and NLC/full have one
contracted component. Shape and catalogue metadata determine
their counts and identities, never a positive-probability interpretation.

Pass `count`, `stride` and `offset`, with `offset + count <= stride`, plus the
parallel key/pointer/capacity arrays. Optional options select resolved output,
R2 strategy and the existing explicit verification JSON policy. There is no
hidden verification or rescue. Packing the requested columns, evaluating the
entire batch, and scattering results each happen natively; there is one SDK
batch call, with no event-by-event evaluator loop.

A successful function call writes every requested point's status. Numerical
outputs are written only for successful, verified or rescued points. Failed
numerical slots and every slot outside the slice stay untouched. Status values
are those in `rusticol_virtual.h`: initial success without established accuracy,
verified, rescued, failed, or accuracy not established. This mandatory status
output prevents stale caller-buffer values from masquerading as valid results.
Do not treat signed virtual values or resolved components as probabilities.
Random selection keys and `UMAMI_OUT_MATRIX_ELEMENT` are deliberately unsupported;
no finite-only scalar result is substituted.

All supplied layouts are checked before writes. Unknown or duplicate keys,
missing mandatory inputs/statuses, invalid capacities, and overflowing index
products fail without changing any output array. Zero-count calls accept absent
buffers but still validate supplied keys and native strategy/policy choices.
`UmamiStatus` reports function errors, while the output status array reports
per-point numerical outcomes. The extension's error getter provides its local
or forwarded SDK diagnostic. A handle opened with an explicit verification
policy can retain numerical global source-binding failures for complete-result
rescue; malformed or unrecoverable parameters remain function errors. A NULL
batch policy inherits that handle policy, as in the shared SDK.

An optional `rusticol_virtual_result **diagnostic_result` receives the ordinary
owned SDK result. It contains compact successful rows, original point IDs,
all-input statuses and complete diagnostics/verification history. Its numerical
buffers use the SDK's layout, not UMAMI's; its shape queries describe that
layout. Free it with `rusticol_virtual_result_free`. It remains valid after the
runtime is cleared or destroyed. Pass NULL to discard it. The output pointer is
set to NULL on error; it must not already own an unfreed result.

## Source build and native control

The standard UMAMI header already lives in the repository and retains its
upstream license. From the repository root, with existing SDK shared libraries
in `$SDK_LIB`, build the extension and control into a workspace directory:

```sh
cc -std=c11 -fPIC -shared \
  rust/loop-reduction/sdk/umami/pyamplicol_umami_virtual_v1.c \
  -Irust/loop-reduction/sdk/include -Isrc/pyamplicol/assets/api_templates/umami \
  -L"$SDK_LIB" -lrusticol_loop_sdk -Wl,-rpath,"$SDK_LIB" \
  -o "$BUILD_DIR/libpyamplicol_umami_virtual_v1.so"
cc -std=c11 rust/loop-reduction/sdk/examples/client_umami_virtual.c \
  -Irust/loop-reduction/sdk/include -Irust/loop-reduction/sdk/umami \
  -Isrc/pyamplicol/assets/api_templates/umami \
  -L"$BUILD_DIR" -lpyamplicol_umami_virtual_v1 \
  -L"$SDK_LIB" -lrusticol_loop_sdk -lm \
  -Wl,-rpath,"$BUILD_DIR" -Wl,-rpath,"$SDK_LIB" -o "$BUILD_DIR/client-umami"
"$BUILD_DIR/client-umami" /path/to/mixed-source-artifact uudd
```

These commands describe the tested Linux source delivery, not a release bundle.
The adapter can also be compiled directly alongside the client and linked to
`librusticol_loop_sdk.a` with the SDK's documented native link flags. Runtime
and adapter must use the same SDK build. The ordinary generated `API/umami`
bundle does not yet publish this virtual extension.

The native control covers the independent three-order MadLoop uudd fixture,
resolved sums, irregular offset/stride slices, poisoned unused inputs, untouched
padding and failed rows, actual numerical failures, optional/default IDs, scalar
and per-event scales, explicit DoubleFloat verification, empty batches, invalid
keys/capacities and overflow rejection. It also accepts the retained `gg` source
artifact for a separate one-order control. These are correctness tests, not new
benchmark timings. The same control accepts a saved LC gg plan and checks every
signed component through the strided layout, inclusive reconstruction and
whole-point failure/verification statuses.

## Correlated operation

`pyamplicol_umami_virtual_v1_evaluate_correlated` (grouped complex colour
histories and joint spin-vector inputs) remains exported with its keys and
`pyamplicol_umami_virtual_v1_supported_correlated_inputs`. It was served by the
removed legacy runtime and refuses a plan v4 process with the plan v4 reason;
evaluate correlated and spin-source virtuals of plan v4 through the Python API.

